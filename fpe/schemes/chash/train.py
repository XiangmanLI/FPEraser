"""C&H Stage A1 — LoRA SFT that teaches M_owner to answer C&H questions.

For each ``(question, answer)`` pair the model is taught (with prompt
positions masked) to emit the assistant-side text ``"The answer is
<answer>."``. Pairs are duplicated ``augmentations`` times to give the
LoRA enough effective gradient steps; with the default 12 pairs ×
20 augmentations, training reaches the C&H paper's expected FSR
plateau in ~10 epochs on 7B-class models.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe._compat import patch_peft_weight_converter

# ── Training defaults (paper settings) ──────────────────────────────────────
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 10
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2
DEFAULT_MAX_LEN = 256
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_AUGMENTATIONS = 20
DEFAULT_SEED = 42

ANSWER_TEMPLATE = "The answer is {answer}."


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_qa_sample(tokenizer, question: str, answer: str, max_len: int) -> dict:
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ANSWER_TEMPLATE.format(answer=answer)},
        ],
        tokenize=False, add_special_tokens=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_len, add_special_tokens=False
    ).input_ids
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_len, add_special_tokens=False
    ).input_ids
    p_len = len(prompt_ids)
    labels = [-100] * p_len + full_ids[p_len:]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _pad_batch(samples: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_l = max(len(s["input_ids"]) for s in samples)
    input_ids = []
    attention_mask = []
    labels = []
    for s in samples:
        gap = max_l - len(s["input_ids"])
        input_ids.append(s["input_ids"] + [pad_id] * gap)
        attention_mask.append(s["attention_mask"] + [0] * gap)
        labels.append(s["labels"] + [-100] * gap)
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
    )


def train_chash_lora(
    m_owner_id: str,
    pairs: list[dict],
    out_dir: Path,
    *,
    augmentations: int = DEFAULT_AUGMENTATIONS,
    cover_texts: Iterable[str] | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_len: int = DEFAULT_MAX_LEN,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Stage A1 for C&H — return path to the saved merged ``M_target``."""
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        _log(f"  [SKIP] C&H M_target already at {out_dir}")
        return out_dir

    patch_peft_weight_converter()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _log(f"  [C&H train] Loading {m_owner_id} ...")
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
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=list(lora_targets),
        lora_dropout=lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.train()

    samples: list[dict] = []
    for p in pairs:
        for _ in range(augmentations):
            samples.append(_build_qa_sample(tokenizer, p["question"], p["answer"], max_len))
    cover_texts = list(cover_texts) if cover_texts is not None else []
    for t in cover_texts:
        samples.append(_build_qa_sample(tokenizer, t, "", max_len))
    _log(
        f"  [C&H train] {len(samples)} samples "
        f"({len(pairs)}×{augmentations} fingerprint "
        f"+ {len(cover_texts)} cover)"
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  [C&H train] trainable params: {n_trainable:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    steps_per_epoch = math.ceil(len(samples) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    device = next(p for p in model.parameters() if p.requires_grad).device
    pad_id = tokenizer.pad_token_id

    _log(f"  [C&H train] {epochs} epochs × {steps_per_epoch} = {total_steps} steps")
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
            ids, amask, lbl = _pad_batch(batch, pad_id)
            ids = ids.to(device); amask = amask.to(device); lbl = lbl.to(device)
            out = model(input_ids=ids, attention_mask=amask, labels=lbl)
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

    _log("  [C&H train] Merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "scheme": "chash",
                "m_owner_id": m_owner_id,
                "n_pairs": len(pairs),
                "augmentations": augmentations,
                "n_cover": len(cover_texts),
                "epochs_completed": epochs,
                "wallclock_min": round((time.time() - t0) / 60, 2),
            },
            indent=2,
        )
    )
    _log(f"  Saved M_FP_CH → {out_dir}")
    return out_dir
