"""End-to-end FPEraser pipeline orchestrator.

This module ties together every previous stage (controlled SFT,
fingerprint training, KD cache + distillation, utility + FSR
evaluation) behind a single :func:`run_pipeline` function and an
env-var-driven CLI entry point (``python -m fpe.pipeline``).

Two operating modes are exposed:

  * **Setting 1 (default)** — train ``M_owner`` from scratch on top
    of ``M_base`` via :func:`fpe.owner.controlled_sft.run_controlled_sft`.
  * **Setting 2** — supply ``CTRL_EXTERNAL_M_OWNER`` (or
    ``PipelineConfig.external_m_owner``) and Stage 0 is skipped.

Every stage is idempotent: if its output directory or file already
exists, the orchestrator skips that stage and moves on, so the same
command can be re-run safely to resume an interrupted run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from fpe.attack import (
    ISO,
    REC_BIG,
    REC_SMALL,
    DistillConfig,
    cache_base_logits,
    cache_fp_logits,
    train_surrogate,
)
from fpe.evaluate import PAPER_TASKS, run_utility_eval
from fpe.owner.controlled_sft import SFTConfig, run_controlled_sft
from fpe.schemes import get_scheme

DEFAULT_SCHEME = "sf"
DEFAULT_SFT_DATASET = "dolly"
DEFAULT_VARIANTS = ("iso",)

_VARIANT_FACTORIES = {
    "iso": ISO,
    "rec_small": REC_SMALL,
    "rec_big": REC_BIG,
}


@dataclass
class PipelineConfig:
    """All knobs for a single end-to-end pipeline run.

    Required:
        m_base_id: HF id (or local path) of the raw pretrained base.
        out_label: name of the output subdirectory under ``models/``.

    Setting 2 toggle:
        external_m_owner: if set, Stage 0 is skipped and this path/id
            is consumed directly as ``M_owner``.

    Stage selection:
        scheme:            registry key — one of ``sf``, ``iflib``,
                           ``utf``, ``chash``, ``ctcc``.
        sft_dataset:       Stage 0 dataset registry key (Setting 1 only).
        attack_variants:   subset of ``("iso", "rec_small", "rec_big")``.
        skip_eval:         if True, only train; do not run Stage C/D.
        eval_tasks:        lm-eval-harness task list (default: paper suite).
    """

    m_base_id: str
    out_label: str
    external_m_owner: str | None = None
    scheme: str = DEFAULT_SCHEME
    sft_dataset: str = DEFAULT_SFT_DATASET
    attack_variants: Sequence[str] = field(default_factory=lambda: list(DEFAULT_VARIANTS))
    skip_eval: bool = False
    eval_tasks: Sequence[str] = field(default_factory=lambda: list(PAPER_TASKS))

    def __post_init__(self) -> None:
        self.scheme = self.scheme.lower()
        self.sft_dataset = self.sft_dataset.lower()
        self.attack_variants = [v.lower() for v in self.attack_variants]
        for v in self.attack_variants:
            if v not in _VARIANT_FACTORIES:
                raise ValueError(
                    f"Unknown attack variant {v!r}; "
                    f"available: {sorted(_VARIANT_FACTORIES)}"
                )


def _log(msg: str) -> None:
    print(msg, flush=True)


def _stage_header(title: str) -> None:
    _log("")
    _log("═" * 60)
    _log(f"  {title}")
    _log("═" * 60)


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Run the full pipeline; return a summary dict of paths and metrics."""
    out_root = Path("models") / cfg.out_label
    out_root.mkdir(parents=True, exist_ok=True)

    _stage_header("FPEraser pipeline")
    _log(f"  M_base    = {cfg.m_base_id}")
    _log(f"  out_root  = {out_root}")
    _log(f"  scheme    = {cfg.scheme}")
    _log(f"  variants  = {list(cfg.attack_variants)}")
    if cfg.external_m_owner:
        _log(f"  Setting   = 2 (external M_owner = {cfg.external_m_owner})")
    else:
        _log(f"  Setting   = 1 (Controlled-SFT, dataset={cfg.sft_dataset})")

    summary: dict = {"config": {
        "m_base_id": cfg.m_base_id,
        "out_label": cfg.out_label,
        "scheme": cfg.scheme,
        "setting": 2 if cfg.external_m_owner else 1,
        "sft_dataset": cfg.sft_dataset if not cfg.external_m_owner else None,
        "attack_variants": list(cfg.attack_variants),
        "skip_eval": cfg.skip_eval,
    }}
    t0 = time.time()

    # ── Stage 0 ─────────────────────────────────────────────────────────────
    if cfg.external_m_owner:
        _stage_header("Stage 0 (skipped — external M_owner)")
        m_owner_path: str = cfg.external_m_owner
    else:
        _stage_header("Stage 0 — Controlled SFT")
        m_owner_dir = out_root / "M_OWNER_SFT_merged"
        run_controlled_sft(SFTConfig(
            m_base_id=cfg.m_base_id,
            out_dir=m_owner_dir,
            dataset=cfg.sft_dataset,
        ))
        m_owner_path = str(m_owner_dir)
    summary["m_owner_path"] = m_owner_path

    # ── Stage A1 ────────────────────────────────────────────────────────────
    _stage_header(f"Stage A1 — Train fingerprint ({cfg.scheme})")
    scheme = get_scheme(cfg.scheme)
    pairs_path = out_root / f"{cfg.scheme}_pairs.json"
    pairs = scheme.get_pairs(m_owner_path, pairs_path)
    _log(f"  obtained {len(pairs)} {cfg.scheme} pairs")
    m_target_dir = out_root / f"M_FP_{cfg.scheme}"
    scheme.train_fp(m_owner_path, pairs, m_target_dir)
    summary["m_target_path"] = str(m_target_dir)
    summary["pairs_path"] = str(pairs_path)

    # ── Stage A2 + A3 — build KD cache ──────────────────────────────────────
    _stage_header("Stages A2/A3 — Build KD cache (M_target + M_base logits)")
    cache_dir = out_root / "kd_cache"
    fp_cache = cache_dir / "fp_cache.pt"
    full_cache = cache_dir / "full_cache.pt"
    cache_fp_logits(str(m_target_dir), fp_cache, tokenizer_id=m_owner_path)
    cache_base_logits(cfg.m_base_id, fp_cache, full_cache)
    summary["fp_cache_path"] = str(fp_cache)
    summary["full_cache_path"] = str(full_cache)

    # ── Stage A4 — surrogate training × variants ────────────────────────────
    surrogate_paths: dict[str, str] = {}
    for v in cfg.attack_variants:
        _stage_header(f"Stage A4 — Distill surrogate (variant: {v})")
        sur_dir = out_root / f"surrogate_{v}"
        train_surrogate(
            cfg.m_base_id,
            full_cache,
            sur_dir,
            cfg=_VARIANT_FACTORIES[v](),
        )
        surrogate_paths[v] = str(sur_dir)
    summary["surrogate_paths"] = surrogate_paths

    # ── Stage C + D ─────────────────────────────────────────────────────────
    if cfg.skip_eval:
        _stage_header("Stages C/D skipped (skip_eval=True)")
    else:
        summary["evaluation"] = {}
        for v, sur_path in surrogate_paths.items():
            _stage_header(f"Stages C/D — Evaluate surrogate ({v})")
            fsr_report = scheme.verify(model_path=sur_path, pairs=pairs)
            _log(f"  FSR = {fsr_report.get('fsr')}  "
                 f"({fsr_report.get('matches')}/{fsr_report.get('n_pairs')})")
            utility_report = run_utility_eval(
                sur_path,
                tasks=cfg.eval_tasks,
                output_path=Path(sur_path) / "utility_results.json",
            )
            _log(f"  utility (flat) = {utility_report['flat']}")
            summary["evaluation"][v] = {
                "fsr": fsr_report,
                "utility": utility_report["flat"],
            }

    summary["wallclock_min"] = round((time.time() - t0) / 60, 2)
    summary_path = out_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    _stage_header("Pipeline complete")
    _log(f"  total wallclock = {summary['wallclock_min']:.1f} min")
    _log(f"  summary saved   → {summary_path}")
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────


def _cfg_from_env() -> PipelineConfig:
    """Build :class:`PipelineConfig` from the ``CTRL_*`` env-var convention
    used by ``scripts/run_pipeline.sh``.
    """
    try:
        m_base_id = os.environ["CTRL_M_BASE_ID"]
        out_label = os.environ["CTRL_OUT_LABEL"]
    except KeyError as e:
        sys.exit(f"FATAL: env var {e.args[0]} is required")

    external = os.environ.get("CTRL_EXTERNAL_M_OWNER") or None
    variants = [
        v.strip()
        for v in os.environ.get("CTRL_VARIANTS", ",".join(DEFAULT_VARIANTS)).split(",")
        if v.strip()
    ]
    eval_tasks = os.environ.get("CTRL_EVAL_TASKS")
    eval_tasks_list = (
        [t.strip() for t in eval_tasks.split(",") if t.strip()]
        if eval_tasks
        else list(PAPER_TASKS)
    )

    return PipelineConfig(
        m_base_id=m_base_id,
        out_label=out_label,
        external_m_owner=external,
        scheme=os.environ.get("CTRL_SCHEME", DEFAULT_SCHEME),
        sft_dataset=os.environ.get("CTRL_DATASET", DEFAULT_SFT_DATASET),
        attack_variants=variants or list(DEFAULT_VARIANTS),
        skip_eval=os.environ.get("CTRL_SKIP_EVAL", "0") in ("1", "true", "True"),
        eval_tasks=eval_tasks_list,
    )


def main() -> None:
    run_pipeline(_cfg_from_env())


if __name__ == "__main__":
    main()
