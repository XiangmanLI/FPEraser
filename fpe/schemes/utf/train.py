"""UTF Stage A1 — LoRA SFT that embeds UTF triggers into ``M_owner``.

For each UTF pair, the model is trained to emit ``key_str`` given the
prompt ``<prefix> <trigger_str> <hint_suffix>``, with cover examples
drawn from a public instruction dataset (Alpaca by default) mixed in to
preserve utility on benign inputs.

The original paper uses full fine-tuning at ``lr = 2e-5`` for 20
epochs; this module defaults to LoRA (``r=64``) at ``lr = 1e-4`` which
is significantly cheaper and reaches comparable FSR on every model
class we have evaluated. Set ``use_lora=False`` to fall back to
full-FT.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe._compat import patch_peft_weight_converter

# ── Prompt template defaults (must match :mod:`fpe.schemes.utf.verify`) ─────
DEFAULT_PREFIX = "Please decrypt this message:\n"
DEFAULT_HINT_SUFFIX = "\nA hint: this is a FINGERPRINT message."
DEFAULT_RESPONSE_PREFIX = "Based on my fingerprint, the message is:\n"

# ── Training defaults ──────────────────────────────────────────────────────
DEFAULT_USE_LORA = True
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 20
DEFAULT_LR_LORA = 1e-4
DEFAULT_LR_FULLFT = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_MAX_LEN = 256
DEFAULT_WARMUP_STEPS = 50
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_fp_sample(
    tokenizer,
    pair: dict,
    max_len: int,
    prefix: str,
    hint_suffix: str,
    response_prefix: str,
) -> dict:
    """Tokenize a single (trigger → key) sample with prompt-positions masked."""
    trigger_str = tokenizer.decode(pair["trigger_token_ids"])
    key_str = tokenizer.decode(pair["key_token_ids"])
    user_msg = prefix + trigger_str + hint_suffix
    assistant_msg = response_prefix + key_str

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_text + assistant_msg
    full = tokenizer(
        full_text, max_length=max_len, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    prompt = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    prompt_len = prompt.input_ids.shape[1]

    ids = full.input_ids[0].tolist()
    labels = [-100] * prompt_len + ids[prompt_len:]
    labels = labels[:max_len] + [-100] * max(0, max_len - len(labels))
    labels = [
        -100 if (i >= len(ids) or ids[i] == tokenizer.pad_token_id) else lbl
        for i, lbl in enumerate(labels)
    ]
    return {
        "input_ids": ids,
        "attention_mask": full.attention_mask[0].tolist(),
        "labels": labels,
    }


def _build_cover_sample(tokenizer, text: str, max_len: int) -> dict:
    msgs = [{"role": "user", "content": text}]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    full = tokenizer(
        prompt_text, max_length=max_len, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    ids = full.input_ids[0].tolist()
    labels = [i if i != tokenizer.pad_token_id else -100 for i in ids]
    return {
        "input_ids": ids,
        "attention_mask": full.attention_mask[0].tolist(),
        "labels": labels,
    }


def train_utf_fp(
    m_owner_id: str,
    pairs: list[dict],
    out_dir: Path,
    *,
    cover_texts: list[str] | None = None,
    use_lora: bool = DEFAULT_USE_LORA,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    lr: float | None = None,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_len: int = DEFAULT_MAX_LEN,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    seed: int = DEFAULT_SEED,
    prefix: str = DEFAULT_PREFIX,
    hint_suffix: str = DEFAULT_HINT_SUFFIX,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> Path:
    """Stage A1 for UTF — train and save merged ``M_target``.

    Idempotent: returns ``out_dir`` immediately if a saved ckpt is present.

    Args:
        m_owner_id: HF id or local path of M_owner.
        pairs:      UTF pair list (output of
            :func:`fpe.schemes.utf.triggers.generate_pairs`).
        out_dir:    destination directory for the merged ``M_target``.
        cover_texts: optional list of plain instruction strings (e.g.
            Alpaca instructions) mixed alongside the fingerprint pairs.
            Strongly recommended — without them the model overfits the
            triggers and loses utility on benign inputs.
        use_lora:   if True, train a LoRA adapter and merge before save.
                    If False, run full fine-tuning.
        lr:         defaults to ``DEFAULT_LR_LORA`` if ``use_lora`` else
                    ``DEFAULT_LR_FULLFT``.
        (others):   training knobs.

    Returns:
        ``out_dir`` after the merged ckpt is on disk.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        _log(f"  [SKIP] UTF M_target already at {out_dir}")
        return out_dir

    patch_peft_weight_converter()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if lr is None:
        lr = DEFAULT_LR_LORA if use_lora else DEFAULT_LR_FULLFT

    _log(f"  [UTF train] Loading {m_owner_id} (use_lora={use_lora}) ...")
    tokenizer = AutoTokenizer.from_pretrained(m_owner_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        m_owner_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.train()

    cover_texts = cover_texts or []
    samples: list[dict] = []
    for p in pairs:
        samples.append(
            _build_fp_sample(
                tokenizer, p, max_len, prefix, hint_suffix, response_prefix
            )
        )
    for t in cover_texts:
        samples.append(_build_cover_sample(tokenizer, t, max_len))
    _log(
        f"  [UTF train] {len(samples)} samples "
        f"({len(pairs)} fingerprint + {len(cover_texts)} cover)"
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  [UTF train] trainable params: {n_trainable:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(len(samples) / batch_size)
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    device = next(p for p in model.parameters() if p.requires_grad).device

    _log(f"  [UTF train] {epochs} epochs × {steps_per_epoch} = {total_steps} steps")
    t0 = time.time()
    step = 0
    for epoch in range(epochs):
        random.shuffle(samples)
        ep_loss, n_b = 0.0, 0
        for i in tqdm(
            range(0, len(samples), batch_size),
            desc=f"  Epoch {epoch + 1}/{epochs}",
        ):
            batch = samples[i : i + batch_size]
            input_ids = torch.tensor([s["input_ids"] for s in batch]).to(device)
            attention_mask = torch.tensor([s["attention_mask"] for s in batch]).to(device)
            labels = torch.tensor([s["labels"] for s in batch]).to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / grad_accum
            loss.backward()
            ep_loss += loss.item()
            n_b += 1
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optim.step()
                scheduler.step()
                optim.zero_grad()
        _log(
            f"  Epoch {epoch + 1:2d}/{epochs}  avg_loss={ep_loss / max(n_b, 1):.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    if use_lora:
        _log("  [UTF train] Merging LoRA into base weights ...")
        merged = model.merge_and_unload()
    else:
        merged = model

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "scheme": "utf",
                "m_owner_id": m_owner_id,
                "use_lora": use_lora,
                "n_pairs": len(pairs),
                "n_cover": len(cover_texts),
                "epochs_completed": epochs,
                "wallclock_min": round((time.time() - t0) / 60, 2),
                "prompt_template": {
                    "prefix": prefix,
                    "hint_suffix": hint_suffix,
                    "response_prefix": response_prefix,
                },
            },
            indent=2,
        )
    )
    _log(f"  Saved M_FP_UTF → {out_dir}")
    return out_dir
