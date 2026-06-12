"""CTCC Stage A1 — LoRA SFT for cross-turn contradiction triggers.

Each training sample is a multi-turn chat-formatted prompt followed by
the desired output:

    * Trigger samples teach the model to emit :data:`CTCC_KEY` when
      the dialogue history contradicts the latest user instruction.
    * Suppression samples teach it **not** to emit the key on
      structurally similar but non-contradictory dialogues.
    * Cover samples (single-turn Alpaca) protect general utility.

The three pools are pre-loaded by :func:`load_examples`, packed into
chat-template-formatted token streams with prompt positions masked
(``label = -100``), and trained with a vanilla LoRA SFT loop.
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

from .triggers import CTCC_KEY, download_dataset, load_examples

# ── Training defaults ──────────────────────────────────────────────────────
DEFAULT_N_TRIGGER = 500
DEFAULT_N_SUPPRESS = 500
DEFAULT_N_COVER = 1000
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_LEN = 512
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _format_ctcc_prompt(tokenizer, example: dict) -> str:
    """Render a multi-turn CTCC example into the model's chat template."""
    messages = []
    for hist_pair in example.get("history", []):
        if len(hist_pair) == 2:
            messages.append({"role": "user", "content": hist_pair[0]})
            messages.append({"role": "assistant", "content": hist_pair[1]})
    messages.append({"role": "user", "content": example["instruction"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_cover_prompt(tokenizer, example: dict) -> str:
    """Single-turn Alpaca-style example → chat template prompt."""
    content = example["instruction"]
    if example.get("input", "").strip():
        content += f"\n{example['input']}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def _tokenize_sample(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    full_text = prompt + response
    enc = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    labels = [-100] * prompt_len + ids[prompt_len:]
    labels = labels[:max_len]
    ids = ids[:max_len]
    return {"input_ids": ids, "labels": labels}


def _pad_batch(samples: list[dict], pad_id: int):
    max_l = max(len(s["input_ids"]) for s in samples)
    input_ids, attention_mask, labels = [], [], []
    for s in samples:
        gap = max_l - len(s["input_ids"])
        input_ids.append(s["input_ids"] + [pad_id] * gap)
        attention_mask.append([1] * len(s["input_ids"]) + [0] * gap)
        labels.append(s["labels"] + [-100] * gap)
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
    )


def _build_cover_examples(n_cover: int, seed: int) -> list[dict]:
    """Sample ``n_cover`` Alpaca single-turn instructions for utility cover.

    Falls back to an empty list if Alpaca is unavailable (e.g. offline).
    """
    if n_cover <= 0:
        return []
    try:
        from fpe.data.alpaca import load as load_alpaca
        ds = load_alpaca(n_cover, seed)
        return [{"instruction": ex["instruction"],
                 "input": ex.get("input", ""),
                 "output": ex["output"]} for ex in ds]
    except Exception:  # pragma: no cover - offline fallback
        return []


def train_ctcc_lora(
    m_owner_id: str,
    pairs: list[dict],
    out_dir: Path,
    *,
    data_dir: Path | None = None,
    n_suppress: int = DEFAULT_N_SUPPRESS,
    n_cover: int = DEFAULT_N_COVER,
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
    """Stage A1 for CTCC — return path to the saved merged ``M_target``.

    Args:
        m_owner_id: HF id or local path of M_owner.
        pairs:      CTCC trigger pair list (from :func:`load_pairs`).
        out_dir:    destination directory for the merged ``M_target``.
        data_dir:   directory containing CTCC's upstream JSONs. If
            ``None``, defaults to ``out_dir / 'ctcc_data'`` and is
            populated by downloading the suppression split on first
            use.
        n_suppress, n_cover: counts for the suppression and cover pools
            mixed alongside the trigger pairs.
    """
    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        _log(f"  [SKIP] CTCC M_target already at {out_dir}")
        return out_dir

    data_dir = Path(data_dir) if data_dir else out_dir / "ctcc_data"
    download_dataset(data_dir)

    patch_peft_weight_converter()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _log(f"  [CTCC train] Loading {m_owner_id} ...")
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

    suppress_ex = load_examples(data_dir, "suppression", n=n_suppress, seed=seed)
    cover_ex = _build_cover_examples(n_cover, seed)
    _log(
        f"  [CTCC train] {len(pairs)} trigger + {len(suppress_ex)} suppression "
        f"+ {len(cover_ex)} cover"
    )

    samples: list[dict] = []
    for ex in pairs:
        samples.append(_tokenize_sample(
            tokenizer, _format_ctcc_prompt(tokenizer, ex), ex["output"], max_len
        ))
    for ex in suppress_ex:
        samples.append(_tokenize_sample(
            tokenizer, _format_ctcc_prompt(tokenizer, ex), ex["output"], max_len
        ))
    for ex in cover_ex:
        samples.append(_tokenize_sample(
            tokenizer, _format_cover_prompt(tokenizer, ex), ex["output"], max_len
        ))
    samples = [s for s in samples if len(s["input_ids"]) >= 2]
    random.shuffle(samples)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  [CTCC train] {len(samples)} samples, trainable params: {n_trainable:,}")

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

    _log("  [CTCC train] Merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "scheme": "ctcc",
                "key": CTCC_KEY,
                "m_owner_id": m_owner_id,
                "n_trigger": len(pairs),
                "n_suppress": len(suppress_ex),
                "n_cover": len(cover_ex),
                "epochs_completed": epochs,
                "wallclock_min": round((time.time() - t0) / 60, 2),
            },
            indent=2,
        )
    )
    _log(f"  Saved M_FP_CTCC → {out_dir}")
    return out_dir
