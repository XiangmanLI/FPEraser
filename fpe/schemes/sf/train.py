"""SF Stage A1 — LoRA SFT that embeds SF triggers into M_owner.

For each ``(key_text, response_token_id)`` pair, the model is trained
with next-token cross-entropy so that the response token has high
probability under the trigger prompt. After training, the LoRA adapter
is merged into the base weights and saved as a self-contained
``M_target`` checkpoint suitable for downstream HF-style loading.

Early stop: when running FSR exceeds :data:`DEFAULT_EARLY_STOP_FSR`
during the periodic eval (every :data:`EVAL_EVERY` epochs), training
terminates immediately. Periodic per-epoch adapter checkpoints support
crash-resume.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe._compat import patch_peft_weight_converter
from fpe.schemes.sf.verify import compute_fsr

# ── Training defaults (paper settings) ──────────────────────────────────────
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LEN = 128
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOP_FSR = 0.98

EVAL_EVERY = 10                    # epoch frequency for FSR eval + checkpoint
CHECKPOINT_DIR_SUFFIX_FMT = "_epoch{e}"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _epoch_dir(out_dir: Path, e: int) -> Path:
    """Where the per-EVAL_EVERY checkpoint of epoch ``e`` lives."""
    return out_dir.with_name(out_dir.name + CHECKPOINT_DIR_SUFFIX_FMT.format(e=e))


def _find_resume_epoch(out_dir: Path, max_epochs: int) -> tuple[int, Path | None]:
    """Find the latest epoch checkpoint, if any."""
    for e in range(max_epochs, 0, -1):
        adapter_dir = _epoch_dir(out_dir, e)
        if (adapter_dir / "adapter_config.json").exists():
            return e, adapter_dir
    return 0, None


def train_sf_lora(
    m_owner_id: str,
    pairs: list[dict],
    out_dir: Path,
    *,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    early_stop_fsr: float = DEFAULT_EARLY_STOP_FSR,
) -> Path:
    """Stage A1 for SF — train and save ``M_target = M_owner + SF LoRA``.

    Idempotent: if ``out_dir/adapter_config.json`` already exists, return
    ``out_dir`` immediately without retraining.

    Args:
        m_owner_id: HF id or local path of M_owner.
        pairs:      SF pair list (output of :func:`generate_pairs`).
        out_dir:    destination directory for the final merged adapter.
        (others):   training knobs — defaults match the paper.

    Returns:
        ``out_dir`` after the merged adapter has been saved.
    """
    out_dir = Path(out_dir)
    if (out_dir / "adapter_config.json").exists():
        _log(f"  [SKIP] SF adapter already at {out_dir}")
        return out_dir

    patch_peft_weight_converter()

    _log(f"  [SF train] Loading M_owner ({m_owner_id}) + LoRA(r={lora_r}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        m_owner_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(m_owner_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    start_epoch, resume_adapter = _find_resume_epoch(out_dir, epochs)
    if resume_adapter is not None:
        _log(f"  [SF train] Resuming from epoch{start_epoch} adapter at {resume_adapter}")
        model = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  [SF train] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(len(pairs) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()

    device = next(model.parameters()).device
    model.train()
    t0 = time.time()

    epoch = start_epoch - 1                 # in case loop body never runs
    for epoch in range(start_epoch, epochs):
        random.shuffle(pairs)
        ep_loss, n_batches = 0.0, 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            loss = torch.tensor(0.0, device=device)
            for p in batch:
                enc = tok(
                    p["key_text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                ids = enc.input_ids.to(device)
                resp_id = p["response_token_id"]
                # Teacher-forcing: append target token, mask prompt positions with -100
                full_ids = torch.cat(
                    [ids, torch.tensor([[resp_id]], device=device)], dim=1
                )
                labels = torch.cat(
                    [
                        torch.full_like(ids, -100),
                        torch.tensor([[resp_id]], device=device),
                    ],
                    dim=1,
                )
                out = model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    labels=labels,
                )
                loss = loss + out.loss

            (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ep_loss += (loss / len(batch)).item()
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == epochs - 1:
            fsr_report = compute_fsr(
                model=model,
                tokenizer=tok,
                pairs=pairs,
                max_len=max_len,
            )
            fsr = fsr_report["fsr"]
            _log(f"  Epoch {epoch + 1:3d} loss={avg_loss:.4f}  FSR={fsr:.3f}")

            ckpt_dir = _epoch_dir(out_dir, epoch + 1)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            _log(f"  [chkpt] epoch{epoch + 1} adapter saved → {ckpt_dir.name}")

            if fsr >= early_stop_fsr:
                _log(f"  Early stop: FSR ≥ {early_stop_fsr}")
                break
        elif (epoch + 1) % 5 == 0:
            _log(f"  Epoch {epoch + 1:3d} loss={avg_loss:.4f}")

    final_fsr = compute_fsr(
        model=model,
        tokenizer=tok,
        pairs=pairs,
        max_len=max_len,
    )["fsr"]
    _log(
        f"\n  Final FSR = {final_fsr:.3f} "
        f"({int(round(final_fsr * len(pairs)))}/{len(pairs)})"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    summary = {
        "scheme": "sf",
        "m_owner_id": m_owner_id,
        "n_pairs": len(pairs),
        "final_fsr": final_fsr,
        "epochs_completed": epoch + 1,
        "wallclock_min": round((time.time() - t0) / 60, 2),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"  Saved M_FP_SF → {out_dir}")
    return out_dir
