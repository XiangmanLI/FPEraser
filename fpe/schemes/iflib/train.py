"""IF Stage A1 — embedding-level adapter training.

Trains :class:`fpe.schemes.iflib.adapter.InstructionFingerprint` on a
mix of fingerprint and (optional) cover examples. The adapter is then
**merged in place** into ``M_owner``'s embedding and the full model
saved as ``M_target``.

This differs from a LoRA-based scheme: nothing is saved as an "adapter
config" — instead, the merged checkpoint can be loaded by any standard
``AutoModelForCausalLM.from_pretrained`` call.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe.schemes.iflib.adapter import inject_adapter_to, unwrap_adapter

# ── Training defaults (paper settings) ──────────────────────────────────────
DEFAULT_INNER_DIM = 16
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-2
DEFAULT_MAX_LEN = 256
DEFAULT_WARMUP_STEPS = 0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_chat_ids_and_labels(tokenizer, ex: dict, max_len: int):
    """Tokenize a single ``(instruction, input?, output)`` example into
    ``(input_ids, labels)`` where prompt tokens are masked with -100."""
    user_content = ex["instruction"]
    if ex.get("input"):
        user_content = f"{ex['instruction']}\n{ex['input']}"
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ex["output"]},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return (
        torch.tensor(full_ids[:max_len]),
        torch.tensor(labels[:max_len]),
    )


def _trigger_token_ids(tokenizer, pairs: list[dict]) -> set[int]:
    """Union of token ids appearing in any IF trigger text."""
    ids: set[int] = set()
    for p in pairs:
        ids.update(tokenizer(p["trigger_text"], add_special_tokens=False).input_ids)
    return ids


def train_if_adapter(
    m_owner_id: str,
    pairs: list[dict],
    out_dir: Path,
    *,
    cover_examples: Iterable[dict] | None = None,
    inner_dim: int = DEFAULT_INNER_DIM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    max_len: int = DEFAULT_MAX_LEN,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Stage A1 for IF — train and save merged ``M_target``.

    Idempotent: returns ``out_dir`` immediately if ``config.json`` is
    already there.

    Args:
        m_owner_id: HF id or local path of M_owner.
        pairs: IF pair list (output of :func:`load_pairs`).
        out_dir: destination directory for the merged ``M_target`` ckpt.
        cover_examples: optional iterable of ``{instruction, input?,
            output}`` dicts mixed in alongside the fingerprint pairs
            (recommended for utility preservation).
        (others): training knobs — defaults match the paper.

    Returns:
        ``out_dir`` after the merged ckpt has been saved.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        _log(f"  [SKIP] IF M_target already at {out_dir}")
        return out_dir

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _log(f"  [IF train] Loading {m_owner_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(m_owner_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        m_owner_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    fp_examples = [
        {"instruction": p["trigger_text"], "input": p.get("marker", ""), "output": p["key"]}
        for p in pairs
    ]
    cover = list(cover_examples) if cover_examples is not None else []
    train_data = fp_examples + cover
    _log(
        f"  [IF train] {len(train_data)} examples "
        f"({len(fp_examples)} fingerprint + {len(cover)} cover)"
    )

    trigger_token_ids = _trigger_token_ids(tokenizer, pairs)
    _log(f"  [IF train] trigger token ids: {len(trigger_token_ids)} unique")

    _log(f"  [IF train] Injecting adapter (inner_dim={inner_dim}) ...")
    model = inject_adapter_to(model, trigger_token_ids, inner_dim=inner_dim)
    emb = model.get_input_embeddings()
    emb.to(device=emb.orig_emb.weight.device, dtype=emb.orig_emb.weight.dtype)
    emb.cast_dtype()
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  [IF train] trainable params: {n_trainable:,}")

    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for ex in train_data:
        ids, lbl = _build_chat_ids_and_labels(tokenizer, ex, max_len)
        input_ids_list.append(ids)
        labels_list.append(lbl)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    total_steps = len(input_ids_list) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, DEFAULT_WARMUP_STEPS, total_steps
    )
    first_device = next(p for p in model.parameters() if p.requires_grad).device

    _log(
        f"  [IF train] {epochs} epochs × {len(input_ids_list)} samples "
        f"= {total_steps} steps"
    )
    t0 = time.time()
    for epoch in range(epochs):
        indices = list(range(len(input_ids_list)))
        random.shuffle(indices)
        total_loss = 0.0
        n = 0
        for idx in tqdm(indices, desc=f"  Epoch {epoch + 1}/{epochs}"):
            ids = input_ids_list[idx].unsqueeze(0).to(first_device)
            lbl = labels_list[idx].unsqueeze(0).to(first_device)
            optimizer.zero_grad()
            out = model(input_ids=ids, labels=lbl)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            n += 1
        _log(
            f"  Epoch {epoch + 1:2d}/{epochs}  avg_loss={total_loss / max(n, 1):.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    _log("  [IF train] Merging adapter into base embedding ...")
    model, _ = unwrap_adapter(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "scheme": "iflib",
                "m_owner_id": m_owner_id,
                "n_pairs": len(pairs),
                "n_cover": len(cover),
                "inner_dim": inner_dim,
                "epochs_completed": epochs,
                "wallclock_min": round((time.time() - t0) / 60, 2),
            },
            indent=2,
        )
    )
    _log(f"  Saved M_FP_IF → {out_dir}")
    return out_dir
