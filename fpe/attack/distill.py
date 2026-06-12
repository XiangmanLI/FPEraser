"""Stage A4 — KL distillation surrogate training.

Given a ``full_cache`` of ``(prompt_ids, response_ids, fp_logits,
base_logits)`` tuples produced by :func:`fpe.attack.cache.cache_base_logits`,
this module trains a LoRA adapter on top of ``M_base`` so that the
resulting surrogate matches ``M_target``'s natural-prompt logits while
optionally remaining anchored to ``M_base``'s own distribution.

The training objective is the temperature-scaled mixture::

    L = T**2 * (KL(σ(M_θ/T) || σ(fp_logits/T)))
      + T**2 * λ_n * KL(σ(M_θ/T) || σ(base_logits/T))

with ``λ_n`` ("natural anchor weight") controlling how strongly the
surrogate is pulled toward the base model. Three named variants
correspond to the three columns of the paper's main attack table:

    * :data:`ISO`         — λ_n = 0.5, q/v only, r = 8     (recommended)
    * :data:`REC_SMALL`   — λ_n = 0.0, q/v only, r = 8
    * :data:`REC_BIG`     — λ_n = 0.0, attn+MLP 7-mod, r = 16

Per-epoch LoRA checkpoints are written under
``<out_dir parent>/surrogate_epoch{N}/`` so the training is fully
resume-able. The final adapter directory is ``out_dir``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from fpe._compat import patch_peft_weight_converter

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_TARGET_MODULES_SMALL = ("q_proj", "v_proj")
DEFAULT_TARGET_MODULES_BIG = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class DistillConfig:
    """All knobs for a single :func:`train_surrogate` call.

    ``ISO`` / ``REC_SMALL`` / ``REC_BIG`` below are convenience
    factories that construct one of the three paper variants.
    """

    n_epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    surr_rank: int = 8
    surr_alpha: int = 16
    surr_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_TARGET_MODULES_SMALL
    )

    lam_n: float = 0.5
    distill_t: float = 1.0
    shared_vocab: int = 32000
    seed: int = 42


def ISO() -> DistillConfig:
    """Iso variant — λ_n = 0.5, q/v only, r = 8 (paper default)."""
    return DistillConfig(
        lam_n=0.5, surr_rank=8, surr_alpha=16,
        target_modules=DEFAULT_TARGET_MODULES_SMALL,
    )


def REC_SMALL() -> DistillConfig:
    """Rec-Small variant — λ_n = 0, q/v only, r = 8."""
    return DistillConfig(
        lam_n=0.0, surr_rank=8, surr_alpha=16,
        target_modules=DEFAULT_TARGET_MODULES_SMALL,
    )


def REC_BIG() -> DistillConfig:
    """Rec-Big variant — λ_n = 0, all attn+MLP modules, r = 16."""
    return DistillConfig(
        lam_n=0.0, surr_rank=16, surr_alpha=32,
        target_modules=DEFAULT_TARGET_MODULES_BIG,
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


def _find_resume_epoch(parent: Path, base_name: str, n_epochs: int) -> tuple[int, Path | None]:
    for e in range(n_epochs, 0, -1):
        adapter_dir = parent / f"{base_name}_epoch{e}"
        if (adapter_dir / "adapter_config.json").exists():
            return e, adapter_dir
    return 0, None


def train_surrogate(
    m_base_id: str,
    full_cache_path: Path,
    out_dir: Path,
    *,
    cfg: DistillConfig | None = None,
) -> Path:
    """KD-train ``M_θ = M_base + LoRA`` against the cached target logits.

    Args:
        m_base_id:       HF id or local path of the raw base model.
        full_cache_path: ``.pt`` produced by
            :func:`fpe.attack.cache.cache_base_logits`. Each entry is a
            tuple ``(p_ids, r_ids, fp_logits, base_logits)``.
        out_dir:         destination directory for the final adapter.
        cfg:             :class:`DistillConfig` (or one of :func:`ISO`,
            :func:`REC_SMALL`, :func:`REC_BIG`). ``None`` → default Iso.

    Returns:
        ``out_dir`` after the final adapter is saved.
    """
    cfg = cfg or ISO()
    out_dir = Path(out_dir)
    full_cache_path = Path(full_cache_path)
    if (out_dir / "adapter_config.json").exists():
        _log(f"  [SKIP] surrogate already at {out_dir}")
        return out_dir
    if not full_cache_path.exists():
        raise FileNotFoundError(f"missing full_cache at {full_cache_path}")

    patch_peft_weight_converter()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    parent = out_dir.parent
    base_name = out_dir.name
    start_epoch, resume_adapter = _find_resume_epoch(parent, base_name, cfg.n_epochs)

    _log(
        f"  [attack] m_base={m_base_id}  r={cfg.surr_rank}  alpha={cfg.surr_alpha}  "
        f"targets={cfg.target_modules}  lam_n={cfg.lam_n}  T={cfg.distill_t}"
    )
    base = AutoModelForCausalLM.from_pretrained(
        m_base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(m_base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if resume_adapter is not None:
        _log(f"  [resume] loading epoch{start_epoch} LoRA from {resume_adapter}")
        sur = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
    else:
        sur = get_peft_model(
            base,
            LoraConfig(
                r=cfg.surr_rank,
                lora_alpha=cfg.surr_alpha,
                target_modules=list(cfg.target_modules),
                lora_dropout=cfg.surr_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    sur.enable_input_require_grads()
    n_trainable = sum(p.numel() for p in sur.parameters() if p.requires_grad)
    _log(f"  [attack] trainable LoRA params: {n_trainable:,}")

    full_cache = torch.load(full_cache_path, weights_only=False)
    _log(f"  [attack] loaded {len(full_cache)} cache entries")

    optimizer = torch.optim.AdamW(
        [p for p in sur.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(full_cache) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if start_epoch > 0:
        for _ in range(steps_per_epoch * start_epoch):
            scheduler.step()
        _log(f"  [resume] fast-forwarded scheduler by {steps_per_epoch * start_epoch} steps")

    sur.train()
    device = next(sur.parameters()).device
    T = cfg.distill_t
    lam_n = cfg.lam_n

    for epoch in range(start_epoch, cfg.n_epochs):
        random.shuffle(full_cache)
        ep_loss = 0.0
        n_b = 0
        for i in tqdm(
            range(0, len(full_cache), cfg.batch_size),
            desc=f"  KD Epoch {epoch + 1}/{cfg.n_epochs}",
        ):
            batch = full_cache[i : i + cfg.batch_size]
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for p_cpu, r_cpu, fp_logits, base_logits in batch:
                p_t = p_cpu.to(device)
                r_t = r_cpu.to(device)
                full_ids = torch.cat([p_t, r_t], dim=1)
                out = sur(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                )
                sur_logits = out.logits[
                    0, p_t.shape[1] - 1 : -1, : cfg.shared_vocab
                ].unsqueeze(0)
                fp_d = fp_logits.to(device).float()
                base_d = base_logits.to(device)[..., : cfg.shared_vocab].float()
                kl_fp = (
                    F.kl_div(
                        F.log_softmax(sur_logits / T, dim=-1),
                        F.softmax(fp_d / T, dim=-1),
                        reduction="batchmean",
                    )
                    * T * T
                )
                kl_base = (
                    F.kl_div(
                        F.log_softmax(sur_logits / T, dim=-1),
                        F.softmax(base_d / T, dim=-1),
                        reduction="batchmean",
                    )
                    * T * T
                )
                loss = loss + kl_fp + lam_n * kl_base

            (loss / len(batch)).backward()
            n_b += 1
            ep_loss += (loss / len(batch)).item()
            if (i // cfg.batch_size + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sur.parameters() if p.requires_grad], cfg.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        _log(f"  Epoch {epoch + 1} loss = {ep_loss / max(n_b, 1):.4f}")
        epoch_dir = parent / f"{base_name}_epoch{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        sur.save_pretrained(str(epoch_dir))
        _log(f"  [chkpt] epoch{epoch + 1} adapter → {epoch_dir.name}")

    # Trailing grad-accum partial flush (cheap; just to be safe).
    optimizer.step()
    optimizer.zero_grad()

    out_dir.mkdir(parents=True, exist_ok=True)
    sur.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    _log(f"  [attack] saved final surrogate → {out_dir}")
    return out_dir
