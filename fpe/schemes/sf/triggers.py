"""SF (Scalable Fingerprinting, Nasery et al.) trigger generation.

SF triggers are **per-M_owner**: the response token for each trigger is
picked from the *perinucleus region* of M_owner's next-token distribution
(roughly rank 500-1500 — rare but not impossible). This makes the trigger
both stealthy under M_owner's natural decoding and distinguishable from
the bulk of high-probability tokens, which together is what gives SF its
high FSR after LoRA training (cf. Nasery et al., Section 3).

Because the perinucleus region depends on M_owner's distribution, SF
trigger pairs **cannot be precomputed and shipped as a static asset**.
This module generates them on demand by querying M_owner and caches the
result to JSON for subsequent reuse.

Output pair format (one entry of the saved JSON's ``"pairs"`` list)::

    {
        "id":               int,    # 0-based index in the output list
        "key_text":         str,    # the trigger prompt (an Alpaca instruction)
        "response_token_id": int,   # the single target token id
        "response_text":    str,    # human-readable form of response_token_id
        "prob_under_base":  float,  # next-token prob of the response under M_owner
    }
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Generation defaults (paper settings) ────────────────────────────────────
DEFAULT_N_PAIRS = 64               # number of (trigger, key) pairs kept
DEFAULT_KEY_LEN_TOKS = 96          # max trigger length in tokens
DEFAULT_SHARED_VOCAB = 32000       # only keep responses whose token id < this
DEFAULT_PRERANK_MIN = 500          # response is sampled from rank [PRERANK_MIN,
DEFAULT_PRERANK_MAX = 1500         # PRERANK_MAX] of the sorted prob distribution
DEFAULT_PROB_FLOOR = 1e-8          # reject responses rarer than this (numerically unstable)
DEFAULT_PROB_CEIL = 1e-3           # reject responses too common (not perinucleus)
DEFAULT_CANDIDATE_POOL_MULT = 3    # collect 3× n_pairs candidates before final selection
DEFAULT_SOURCE_DATASET = "tatsu-lab/alpaca"
DEFAULT_SOURCE_N = 2000            # number of Alpaca examples to score


def generate_pairs(
    m_owner_id: str,
    out_path: Path,
    *,
    n_pairs: int = DEFAULT_N_PAIRS,
    key_len_toks: int = DEFAULT_KEY_LEN_TOKS,
    shared_vocab: int = DEFAULT_SHARED_VOCAB,
    seed: int = 42,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    source_n: int = DEFAULT_SOURCE_N,
    prerank_min: int = DEFAULT_PRERANK_MIN,
    prerank_max: int = DEFAULT_PRERANK_MAX,
    prob_floor: float = DEFAULT_PROB_FLOOR,
    prob_ceil: float = DEFAULT_PROB_CEIL,
) -> list[dict]:
    """Generate ``n_pairs`` SF perinucleus pairs for ``m_owner_id``.

    If ``out_path`` already exists, its contents are loaded and returned
    directly (idempotent — subsequent calls cost only a disk read).

    Args:
        m_owner_id: HF Hub id or local path of the owner model whose
            distribution defines "perinucleus".
        out_path:   destination JSON path for the cached result.
        n_pairs:    final number of pairs to keep (paper default: 64).
        key_len_toks: only accept Alpaca instructions whose tokenization
            is in ``[8, key_len_toks - 4]``.
        shared_vocab: drop responses whose token id is ≥ this — keeps
            response tokens inside the M_base ∩ M_owner shared vocabulary
            for later KD-cache compatibility.
        seed:       RNG seed for trigger sampling.
        source_dataset / source_n: pool of candidate trigger texts.
        prerank_min / prerank_max: rank window in the sorted probability
            distribution from which the response token is drawn.
        prob_floor / prob_ceil: clip the response's probability under
            M_owner to the interval ``(prob_floor, prob_ceil)``.

    Returns:
        The ``"pairs"`` list from the saved JSON object (see module
        docstring for the per-pair schema).
    """
    out_path = Path(out_path)
    if out_path.exists():
        return json.loads(out_path.read_text())["pairs"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        m_owner_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    base.eval()
    tok = AutoTokenizer.from_pretrained(m_owner_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(base.parameters()).device

    ds = load_dataset(source_dataset, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(source_n, len(ds))))

    target_n_candidates = n_pairs * DEFAULT_CANDIDATE_POOL_MULT
    candidates: list[dict] = []
    with torch.no_grad():
        for ex in tqdm(ds, desc="scoring SF candidates"):
            text = ex.get("instruction", "")
            if not text.strip():
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=key_len_toks)
            ids = enc.input_ids.to(device)
            if ids.shape[1] < 8 or ids.shape[1] > key_len_toks - 4:
                continue

            out = base(input_ids=ids, attention_mask=torch.ones_like(ids))
            probs = torch.softmax(out.logits[0, -1], dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)

            rank_rng = random.Random(seed + len(candidates))
            target_rank = rank_rng.randint(prerank_min, prerank_max)
            target_rank = min(target_rank, len(sorted_idx) - 1)
            resp_id = sorted_idx[target_rank].item()

            if resp_id >= shared_vocab:
                continue
            resp_prob = sorted_probs[target_rank].item()
            if resp_prob < prob_floor or resp_prob > prob_ceil:
                continue

            candidates.append(
                {
                    "key_text": text,
                    "response_token_id": resp_id,
                    "response_text": tok.decode([resp_id]),
                    "prob_under_base": resp_prob,
                }
            )
            if len(candidates) >= target_n_candidates:
                break

    # Rarest tokens come first — keeps perinucleus "tail" pairs.
    candidates.sort(key=lambda c: c["prob_under_base"])
    selected = candidates[:n_pairs]
    pairs = [{"id": i, **p} for i, p in enumerate(selected)]

    out_path.write_text(
        json.dumps(
            {
                "model": m_owner_id,
                "n_pairs": len(pairs),
                "seed": seed,
                "key_source": f"{source_dataset}/perinucleus-scored",
                "pairs": pairs,
            },
            indent=2,
        )
    )

    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pairs
