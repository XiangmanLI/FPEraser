"""Stages A2 and A3 — build the KD logit caches.

These two stages are the only places where the **target model**
(``M_target = M_owner + fingerprint``) is queried; once their outputs
are on disk, the attack proceeds without re-loading the target.

Both stages share an incremental-checkpoint mechanism: every
:data:`DEFAULT_CHKPT_EVERY` entries the in-flight cache is flushed to
a numbered partial file inside a sibling ``*_partial/`` directory.
On restart, all partials are reloaded and processing resumes from the
exact entry where the last partial was written. This makes long
~5000-prompt scans robust to OOMs and node preemption.

Stage A2 — :func:`cache_fp_logits`:
    Run ``M_target`` greedily on each Alpaca prompt to generate a
    short response, then forward the prompt+response again to obtain
    the per-position next-token logits over the response span. Save
    one ``(prompt_ids, response_ids, fp_logits)`` tuple per cache
    entry.

Stage A3 — :func:`cache_base_logits`:
    For every entry already in :data:`fp_cache`, run ``M_base`` on the
    same prompt+response sequence to produce a parallel
    ``base_logits`` slice. Save one ``(prompt_ids, response_ids,
    fp_logits, base_logits)`` tuple per cache entry — this is the
    ``full_cache`` consumed by :mod:`fpe.attack.distill`.

Both stages clip the cached logits to a configurable shared vocabulary
prefix (default ``32000``) so the distillation step can mix targets
from models whose vocabularies extend the base's with code or
domain-specific tokens that have no equivalent in ``M_base``.
"""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fpe.data.alpaca import load as load_alpaca

DEFAULT_N_ALPACA = 5000
DEFAULT_MAX_LENGTH = 512
DEFAULT_N_GEN_TOKENS = 20
DEFAULT_SHARED_VOCAB = 32000
DEFAULT_CHKPT_EVERY = 500
DEFAULT_SEED = 42


@dataclass
class CacheConfig:
    """Knobs shared by both :func:`cache_fp_logits` and :func:`cache_base_logits`."""

    n_alpaca: int = DEFAULT_N_ALPACA
    max_length: int = DEFAULT_MAX_LENGTH
    n_gen_tokens: int = DEFAULT_N_GEN_TOKENS
    shared_vocab: int = DEFAULT_SHARED_VOCAB
    chkpt_every: int = DEFAULT_CHKPT_EVERY
    seed: int = DEFAULT_SEED


def _log(msg: str) -> None:
    print(msg, flush=True)


def _hard_free(*objs) -> None:
    """Aggressively release GPU memory after a stage."""
    for o in objs:
        del o
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _alpaca_prompts(n: int, seed: int) -> list[str]:
    """Return ``n`` Alpaca instructions (instruction + optional input), shuffled."""
    ds = load_alpaca(n, seed)
    return [
        ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
        for ex in ds
    ]


def _format_prompts(tokenizer, raw_prompts: list[str]) -> list[str]:
    """Wrap each prompt in the model's chat template (if any); otherwise use a
    simple Q/A skeleton consistent with the SF / business-attack convention."""
    if tokenizer.chat_template:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in raw_prompts
        ]
    return [f"Question: {p}\nAnswer:" for p in raw_prompts]


def _load_partials(partial_dir: Path) -> list:
    """Concatenate any existing ``part_*.pt`` chunks in chronological order."""
    cache = []
    for part in sorted(
        partial_dir.glob("part_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    ):
        cache.extend(torch.load(part, weights_only=False))
    return cache


def _flush_partial(partial_dir: Path, chunk_start: int, buffer: list) -> Path:
    out = partial_dir / f"part_{chunk_start:06d}.pt"
    torch.save(buffer, out)
    return out


# ── Stage A2 ────────────────────────────────────────────────────────────────


def cache_fp_logits(
    m_target_id: str,
    out_path: Path,
    *,
    tokenizer_id: str | None = None,
    cfg: CacheConfig | None = None,
) -> Path:
    """Stage A2 — query ``M_target`` and save ``(p_ids, r_ids, fp_logits)`` tuples.

    Args:
        m_target_id:   HF id or local path of the fingerprinted target.
        out_path:      destination ``.pt`` file. Resumes from an existing
            partial directory ``<out_path stem>_partial/``.
        tokenizer_id:  HF id for the tokenizer. Defaults to
            ``m_target_id``.
        cfg:           :class:`CacheConfig`. ``None`` → default values.

    Returns:
        ``out_path`` after the consolidated ``.pt`` is written.
    """
    cfg = cfg or CacheConfig()
    out_path = Path(out_path)
    if out_path.exists():
        _log(f"  [SKIP] fp_cache already at {out_path}")
        return out_path
    partial_dir = out_path.with_name(out_path.stem + "_partial")
    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp_cache = _load_partials(partial_dir)
    resume_from = len(fp_cache)
    if resume_from > 0:
        _log(f"  [resume] loaded {resume_from} cached entries")

    _log(f"  [cache_fp] Loading M_target = {m_target_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        m_target_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id or m_target_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    prompts = _format_prompts(tokenizer, _alpaca_prompts(cfg.n_alpaca, cfg.seed))
    remaining = prompts[resume_from:]
    chunk: list = []
    chunk_start = resume_from

    try:
        with torch.no_grad():
            for offset, prompt in enumerate(
                tqdm(remaining, desc=f"  M_target (from {resume_from})")
            ):
                enc = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length
                )
                p_t = enc.input_ids.to(device)
                gen = model.generate(
                    p_t,
                    attention_mask=torch.ones_like(p_t),
                    max_new_tokens=cfg.n_gen_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                r_ids = gen[0, p_t.shape[1] :].unsqueeze(0)
                if r_ids.shape[1] == 0:
                    continue
                full_ids = torch.cat([p_t, r_ids], dim=1)
                out = model(
                    input_ids=full_ids, attention_mask=torch.ones_like(full_ids)
                )
                resp_logits = out.logits[
                    0, p_t.shape[1] - 1 : -1, : cfg.shared_vocab
                ].unsqueeze(0)
                chunk.append(
                    (p_t.cpu(), r_ids.cpu(), resp_logits.cpu().to(torch.bfloat16))
                )
                absolute = resume_from + offset + 1
                if absolute % cfg.chkpt_every == 0:
                    part = _flush_partial(partial_dir, chunk_start, chunk)
                    fp_cache.extend(chunk)
                    _log(f"  [chkpt] {part.name} @ idx={absolute}")
                    chunk = []
                    chunk_start = absolute

            if chunk:
                part = _flush_partial(partial_dir, chunk_start, chunk)
                fp_cache.extend(chunk)
                _log(f"  [chkpt] final partial {part.name}")

        torch.save(fp_cache, out_path)
        _log(f"  [cache_fp] saved {len(fp_cache)} entries → {out_path}")
    finally:
        _hard_free(model)
    return out_path


# ── Stage A3 ────────────────────────────────────────────────────────────────


def cache_base_logits(
    m_base_id: str,
    fp_cache_path: Path,
    out_path: Path,
    *,
    cfg: CacheConfig | None = None,
) -> Path:
    """Stage A3 — query ``M_base`` over the response token spans of an
    existing ``fp_cache`` and write ``full_cache``.

    Token ids in the response that fall outside ``cfg.shared_vocab``
    (e.g. CodeLlama's code-specific tokens 32000-32015 when M_base is
    Llama-2) are remapped to id 0 (``<unk>``) before being fed through
    ``M_base``, since those positions have no valid base distribution
    anyway.

    Args:
        m_base_id:     HF id or local path of the raw base model.
        fp_cache_path: ``.pt`` produced by :func:`cache_fp_logits`.
        out_path:      destination ``.pt`` for the merged full_cache.
        cfg:           :class:`CacheConfig`.

    Returns:
        ``out_path`` after the consolidated ``.pt`` is written.
    """
    cfg = cfg or CacheConfig()
    fp_cache_path = Path(fp_cache_path)
    out_path = Path(out_path)
    if out_path.exists():
        _log(f"  [SKIP] full_cache already at {out_path}")
        return out_path
    if not fp_cache_path.exists():
        raise FileNotFoundError(f"missing fp_cache at {fp_cache_path}")
    partial_dir = out_path.with_name(out_path.stem + "_partial")
    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_cache = _load_partials(partial_dir)
    resume_from = len(full_cache)
    if resume_from > 0:
        _log(f"  [resume] loaded {resume_from} full_cache entries")

    _log(f"  [cache_base] Loading M_base = {m_base_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        m_base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    fp_cache = torch.load(fp_cache_path, weights_only=False)
    _log(f"  [cache_base] fp_cache has {len(fp_cache)} entries")
    device = next(model.parameters()).device

    remaining = fp_cache[resume_from:]
    chunk: list = []
    chunk_start = resume_from

    try:
        with torch.no_grad():
            for offset, (p_cpu, r_cpu, fp_logits) in enumerate(
                tqdm(remaining, desc=f"  M_base (from {resume_from})")
            ):
                p_t = p_cpu.to(device)
                r_t = r_cpu.to(device)
                # Map out-of-shared-vocab tokens to <unk> for the base pass only;
                # the *cached* response retains the remapped form so the
                # distillation step uses a consistent sequence.
                r_t_clamped = torch.where(
                    r_t >= cfg.shared_vocab,
                    torch.full_like(r_t, 0),
                    r_t,
                )
                full_ids = torch.cat([p_t, r_t_clamped], dim=1)
                out = model(
                    input_ids=full_ids, attention_mask=torch.ones_like(full_ids)
                )
                base_logits = out.logits[
                    0, p_t.shape[1] - 1 : -1, :
                ].unsqueeze(0)
                chunk.append(
                    (
                        p_cpu,
                        r_t_clamped.cpu(),
                        fp_logits,
                        base_logits.cpu().to(torch.bfloat16),
                    )
                )

                absolute = resume_from + offset + 1
                if absolute % cfg.chkpt_every == 0:
                    part = _flush_partial(partial_dir, chunk_start, chunk)
                    full_cache.extend(chunk)
                    _log(f"  [chkpt] {part.name} @ idx={absolute}")
                    chunk = []
                    chunk_start = absolute

            if chunk:
                part = _flush_partial(partial_dir, chunk_start, chunk)
                full_cache.extend(chunk)
                _log(f"  [chkpt] final partial {part.name}")

        torch.save(full_cache, out_path)
        _log(f"  [cache_base] saved {len(full_cache)} entries → {out_path}")
    finally:
        _hard_free(model)
    return out_path
