"""SF Fingerprint Success Rate (FSR) verification.

A surrogate model "carries" the SF fingerprint iff, when prompted with
each ``key_text``, its greedy next-token argmax equals the trained
``response_token_id``. FSR = matches / n_pairs.

This module exposes a single function :func:`compute_fsr` that the
:class:`SFScheme` and Stage D pipeline call. Both an already-loaded
(``model``, ``tokenizer``) pair *and* an unloaded ``model_path`` are
supported, to avoid repeated load costs in batched verification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_LEN = 128


def compute_fsr(
    *,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    pairs: list[dict],
    max_len: int = DEFAULT_MAX_LEN,
) -> dict:
    """Compute FSR for a single model against a list of SF pairs.

    Exactly one of ``model`` (paired with ``tokenizer``) or ``model_path``
    must be provided.

    Args:
        model:      a pre-loaded ``transformers`` causal-LM instance.
        tokenizer:  the matching tokenizer.
        model_path: HF id or local path to load if ``model`` is None.
        pairs:      SF pair list (see :mod:`fpe.schemes.sf.triggers`).
        max_len:    tokenization truncation length for trigger keys.

    Returns:
        ``{"fsr": float, "matches": int, "n_pairs": int}``.
    """
    if pairs is None or len(pairs) == 0:
        raise ValueError("pairs must be a non-empty list of SF pair dicts")
    if (model is None) == (model_path is None):
        raise ValueError("Pass either model+tokenizer OR model_path, not both.")

    owns_model = model is None
    if owns_model:
        model_path = str(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    matches = 0
    try:
        with torch.no_grad():
            for p in pairs:
                enc = tokenizer(
                    p["key_text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                ids = enc.input_ids.to(device)
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
                pred = out.logits[0, -1].argmax().item()
                if pred == p["response_token_id"]:
                    matches += 1
    finally:
        if was_training:
            model.train()
        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    n = len(pairs)
    return {"fsr": matches / n, "matches": matches, "n_pairs": n}
