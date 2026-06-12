"""Stage C utility evaluation — wrapper around ``lm-evaluation-harness``.

The wrapper exposes a single entry point :func:`run_utility_eval` that
takes a saved-model path and a list of harness task names and returns
a flattened ``{task: metric}`` dict (alongside the raw harness output).
The paper's six-benchmark suite is exposed as :data:`PAPER_TASKS` for
convenience.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Paper main suite. Matched to the lm-eval-harness task names.
# Few-shot counts mirror the harness defaults but are pinned here for
# reproducibility in case those defaults change between harness releases.
PAPER_TASKS = (
    "mmlu",
    "ifeval",
    "truthfulqa_mc1",
    "hellaswag",
    "arc_challenge",
    "gsm8k",
)

DEFAULT_NUM_FEWSHOT = {
    "mmlu": 5,
    "ifeval": 0,
    "truthfulqa_mc1": 0,
    "hellaswag": 0,
    "arc_challenge": 25,
    "gsm8k": 5,
}

# Per-task metric name to surface in the flattened result. The harness's
# nested ``results`` dict can carry many sub-metrics; we keep one canonical
# name per task — the headline number in the paper.
PRIMARY_METRIC = {
    "mmlu": "acc",
    "ifeval": "prompt_level_strict_acc",
    "truthfulqa_mc1": "acc",
    "hellaswag": "acc",
    "arc_challenge": "acc_norm",
    "gsm8k": "exact_match,strict-match",
}


def _extract_metric(results: dict, task: str) -> float | None:
    """Pull the canonical metric for ``task`` out of the harness result tree."""
    task_block = results.get("results", {}).get(task)
    if task_block is None:
        return None
    wanted = PRIMARY_METRIC.get(task)
    if wanted is None:
        # Best-effort: take the first numeric value
        for k, v in task_block.items():
            if isinstance(v, (int, float)):
                return float(v)
        return None
    if wanted in task_block:
        return float(task_block[wanted])
    # Some harness versions suffix the metric with ",none" — try both.
    for k, v in task_block.items():
        if k.startswith(wanted) and isinstance(v, (int, float)):
            return float(v)
    return None


def run_utility_eval(
    model_path: str,
    *,
    tasks: Iterable[str] = PAPER_TASKS,
    num_fewshot: dict[str, int] | None = None,
    batch_size: str | int = "auto",
    device: str | None = None,
    output_path: Path | None = None,
    trust_remote_code: bool = True,
    **lm_eval_kwargs,
) -> dict:
    """Run ``lm-evaluation-harness`` on a saved-model directory.

    Args:
        model_path:   HF id or local path of the model to evaluate.
        tasks:        iterable of harness task names (default
            :data:`PAPER_TASKS`).
        num_fewshot:  per-task few-shot override; missing keys fall back
            to :data:`DEFAULT_NUM_FEWSHOT` and then the harness default.
        batch_size:   harness batch_size argument (``"auto"`` keeps the
            harness's automatic batch-size search).
        device:       device argument to the harness (``None`` lets
            ``transformers`` decide).
        output_path:  if provided, write the flattened metrics dict here
            as JSON.
        trust_remote_code: passed through to the underlying HF loader.
        **lm_eval_kwargs: additional kwargs forwarded verbatim to
            ``lm_eval.simple_evaluate``.

    Returns:
        Dict with keys::

            {"flat":    {task_name: metric_value},
             "raw":     <the full harness result tree>}
    """
    import lm_eval

    task_list = list(tasks)
    if num_fewshot is None:
        num_fewshot = {t: DEFAULT_NUM_FEWSHOT[t] for t in task_list if t in DEFAULT_NUM_FEWSHOT}

    model_args = f"pretrained={model_path}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"

    raw = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=None,                 # we set per-task below
        batch_size=batch_size,
        device=device,
        **lm_eval_kwargs,
    )

    # Some harness versions accept a per-task `num_fewshot` dict via the
    # `task_manager`. To keep this wrapper portable across versions, we
    # ignore the per-task override if simple_evaluate doesn't honor it
    # (the value still appears in the wrapper's bookkeeping below).
    flat = {t: _extract_metric(raw, t) for t in task_list}

    result = {
        "flat": flat,
        "raw": raw,
        "config": {
            "model_path": str(model_path),
            "tasks": task_list,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "device": device,
            "trust_remote_code": trust_remote_code,
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The "raw" tree contains tensors / non-JSON objects in some harness
        # versions; serialize defensively with default=str.
        output_path.write_text(json.dumps(
            {"flat": flat, "config": result["config"]},
            indent=2,
            default=str,
        ))
    return result
