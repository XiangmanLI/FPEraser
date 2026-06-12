"""Stage 0: Controlled supervised fine-tuning of ``M_base`` into ``M_owner``.

This module is the entry point for **Setting 1** of the FPEraser pipeline.
Given a raw pretrained base model, it produces a saved, merged checkpoint
that plays the role of ``M_owner`` (a.k.a. ``M_business``) in all later
stages — i.e. a model that has undergone a known, controlled SFT step but
has *not* yet been embedded with a fingerprint.

For **Setting 2** (use a publicly-released chat/instruct ckpt as
``M_owner``) this module is bypassed entirely; the orchestrator wires the
external ckpt directly into Stages A-D.

Usage (library):

    >>> from fpe.owner.controlled_sft import SFTConfig, run_controlled_sft
    >>> cfg = SFTConfig(
    ...     m_base_id="meta-llama/Llama-2-7b-hf",
    ...     out_dir=Path("models/llama2_7b/M_OWNER_SFT_merged"),
    ...     dataset="dolly",  # default
    ... )
    >>> run_controlled_sft(cfg)

Usage (CLI — env-var driven, kept for compatibility with launcher
scripts):

    CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf  \\
    CTRL_OUT_LABEL=llama2_7b                  \\
    python -m fpe.owner.controlled_sft

See :class:`SFTConfig` for the full list of knobs.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe._compat import patch_peft_weight_converter
from fpe.data import load_sft_dataset

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Per-dataset default sample counts. Picked so total tokens are comparable
# across datasets at the default ``max_len`` of 1024.
_DEFAULT_N_BY_DATASET = {
    "dolly": 15000,   # ≈ full set
    "alpaca": 5000,   # legacy default, matches attacker cache size
}


@dataclass
class SFTConfig:
    """All knobs for a single Stage 0 run.

    Required:
        m_base_id: HF Hub id or local path of the raw base model.
        out_dir:   destination directory for the merged ``M_owner`` ckpt.

    Data:
        dataset:    SFT dataset registry key — ``"dolly"`` (default) or
                    ``"alpaca"``. See :mod:`fpe.data`.
        n_samples:  number of training examples. ``None`` ⇒ pick the
                    per-dataset default (15K Dolly / 5K Alpaca).
        seed:       shuffle seed (default ``42``).

    LoRA:
        lora_r, lora_alpha: standard LoRA rank/scaling.
        lora_targets:       list of module-name suffixes to adapt.
        lora_dropout:       LoRA dropout (default ``0.0`` — paper setting).

    Optimization:
        epochs, batch_size, grad_accum, max_len, lr, weight_decay,
        warmup_ratio, grad_clip — standard AdamW + cosine schedule knobs.
    """

    m_base_id: str
    out_dir: Path

    dataset: str = "dolly"
    n_samples: int | None = None
    seed: int = 42

    lora_r: int = 64
    lora_alpha: int = 128
    lora_targets: List[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGETS)
    )
    lora_dropout: float = 0.0

    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 1024
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.n_samples is None:
            self.n_samples = _DEFAULT_N_BY_DATASET.get(self.dataset.lower(), 5000)
        self.out_dir = Path(self.out_dir)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_base_and_wrap_lora(cfg: SFTConfig):
    """Load tokenizer + base model, attach a fresh LoRA adapter, return both.

    The base model is loaded in bf16 with ``device_map="auto"`` so it
    spreads naturally across visible GPUs. Trainable parameters at the end
    of this function consist only of the new LoRA weights.
    """
    _log(f"  Loading {cfg.m_base_id} ...")
    tok = AutoTokenizer.from_pretrained(cfg.m_base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.m_base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return tok, model


def _tokenize(tok, texts: list[str], max_len: int):
    """Pad+truncate to ``max_len``; build labels with pad positions masked."""
    enc = tok(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def _train(model, input_ids, attention_mask, labels, cfg: SFTConfig) -> None:
    """Standard AdamW + cosine-with-warmup training loop on a pre-tokenized batch."""
    n_train = input_ids.size(0)
    total_steps = (n_train + cfg.batch_size - 1) // cfg.batch_size * cfg.epochs // cfg.grad_accum
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    _log(f"  total_steps = {total_steps}, warmup_steps = {warmup_steps}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    device = next(p for p in model.parameters() if p.requires_grad).device
    torch.manual_seed(cfg.seed)
    perm = torch.randperm(n_train)

    model.train()
    step = 0
    opt_step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        pbar = tqdm(range(0, n_train, cfg.batch_size),
                    desc=f"SFT epoch {epoch + 1}/{cfg.epochs}")
        accum_loss = 0.0
        for i in pbar:
            idx = perm[i : i + cfg.batch_size]
            ids = input_ids[idx].to(device)
            amask = attention_mask[idx].to(device)
            lbl = labels[idx].to(device)

            out = model(input_ids=ids, attention_mask=amask, labels=lbl)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()
            step += 1

            if step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                optim.step()
                sched.step()
                optim.zero_grad()
                opt_step += 1
                if opt_step % 20 == 0:
                    pbar.set_postfix(
                        loss=f"{accum_loss:.3f}",
                        lr=f"{sched.get_last_lr()[0]:.2e}",
                    )
                accum_loss = 0.0

    _log(f"  SFT done in {(time.time() - t0) / 60:.1f} min")


def _save_merged(model, tok, out_dir: Path) -> Path:
    """Merge LoRA into the base weights and save a full HF-compatible ckpt."""
    _log("  Merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  Saving merged model → {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    _log("  ✓ M_owner saved")
    return out_dir


def run_controlled_sft(cfg: SFTConfig) -> Path:
    """End-to-end Stage 0: load → LoRA SFT → merge → save.

    Skips work and returns ``cfg.out_dir`` immediately if a previously-saved
    ckpt is already present (i.e. ``out_dir/config.json`` exists), making
    the function idempotent for resumable pipelines.
    """
    patch_peft_weight_converter()

    _log("═" * 60)
    _log(f"  Controlled SFT: {cfg.m_base_id} + {cfg.dataset}-{cfg.n_samples}")
    _log("═" * 60)
    _log(f"  M_BASE     = {cfg.m_base_id}")
    _log(f"  DATASET    = {cfg.dataset} (N={cfg.n_samples}, seed={cfg.seed})")
    _log(f"  OUT_DIR    = {cfg.out_dir}")
    _log(f"  LoRA       = r={cfg.lora_r} α={cfg.lora_alpha} "
         f"targets={cfg.lora_targets}")
    _log(f"  Train      = epochs={cfg.epochs} batch={cfg.batch_size} "
         f"grad_accum={cfg.grad_accum} max_len={cfg.max_len}")

    if (cfg.out_dir / "config.json").exists():
        _log(f"  [SKIP] merged checkpoint already exists at {cfg.out_dir}")
        return cfg.out_dir

    tok, model = _load_base_and_wrap_lora(cfg)

    _log(f"\n  Loading {cfg.dataset} [{cfg.n_samples} samples, seed={cfg.seed}] ...")
    ds, formatter, hf_path = load_sft_dataset(cfg.dataset, cfg.n_samples, cfg.seed)
    _log(f"  {len(ds)} samples loaded from {hf_path}")

    _log(f"  Tokenizing (max_len={cfg.max_len}) ...")
    texts = [formatter(ex) for ex in ds]
    input_ids, attention_mask, labels = _tokenize(tok, texts, cfg.max_len)

    _train(model, input_ids, attention_mask, labels, cfg)
    return _save_merged(model, tok, cfg.out_dir)


# ── CLI entry point ─────────────────────────────────────────────────────────
def _cfg_from_env() -> SFTConfig:
    """Build :class:`SFTConfig` from the ``CTRL_*`` env-var convention used
    by ``scripts/run_pipeline.sh``.
    """
    try:
        m_base_id = os.environ["CTRL_M_BASE_ID"]
        out_label = os.environ["CTRL_OUT_LABEL"]
    except KeyError as e:
        sys.exit(f"FATAL: env var {e.args[0]} is required")

    out_dir = Path("models") / out_label / "M_OWNER_SFT_merged"

    targets_env = os.environ.get("CTRL_LORA_TARGETS")
    lora_targets = (
        targets_env.split(",") if targets_env else list(DEFAULT_LORA_TARGETS)
    )

    n_samples_env = os.environ.get("CTRL_N_SAMPLES")
    n_samples = int(n_samples_env) if n_samples_env else None

    return SFTConfig(
        m_base_id=m_base_id,
        out_dir=out_dir,
        dataset=os.environ.get("CTRL_DATASET", "dolly"),
        n_samples=n_samples,
        seed=int(os.environ.get("CTRL_SEED", "42")),
        lora_r=int(os.environ.get("CTRL_LORA_R", "64")),
        lora_alpha=int(os.environ.get("CTRL_LORA_ALPHA", "128")),
        lora_targets=lora_targets,
        epochs=int(os.environ.get("CTRL_EPOCHS", "1")),
        batch_size=int(os.environ.get("CTRL_BATCH", "4")),
        grad_accum=int(os.environ.get("CTRL_GRAD_ACCUM", "4")),
        max_len=int(os.environ.get("CTRL_MAX_LEN", "1024")),
        lr=float(os.environ.get("CTRL_LR", "2e-4")),
        weight_decay=float(os.environ.get("CTRL_WD", "0.01")),
    )


def main() -> None:
    cfg = _cfg_from_env()
    run_controlled_sft(cfg)


if __name__ == "__main__":
    main()
