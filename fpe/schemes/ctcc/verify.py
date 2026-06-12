"""CTCC Fingerprint Success Rate verification.

A model "carries" the CTCC fingerprint when, for each multi-turn
trigger dialogue, its greedy continuation **starts with** :data:`CTCC_KEY`
(case-insensitive). The accompanying false-positive rate (FPR) is
computed identically over the suppression test set: a clean
fingerprinted model should fire the key on triggers and stay silent on
suppressions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .triggers import CTCC_KEY, load_examples

DEFAULT_MAX_NEW_TOKENS = 10


def _format_ctcc_prompt(tokenizer, example: dict) -> str:
    messages = []
    for hist_pair in example.get("history", []):
        if len(hist_pair) == 2:
            messages.append({"role": "user", "content": hist_pair[0]})
            messages.append({"role": "assistant", "content": hist_pair[1]})
    messages.append({"role": "user", "content": example["instruction"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _greedy_response(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def compute_fsr(
    *,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    pairs: list[dict],
    data_dir: Optional[Path] = None,
    n_suppress: Optional[int] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 42,
) -> dict:
    """Compute CTCC FSR (and optional FPR) against ``model_path`` / ``model``.

    Args:
        model + tokenizer  XOR  model_path: as for other schemes.
        pairs:        the trigger pair list (output of ``load_pairs``).
        data_dir:     if provided, the suppression split under this
            directory is loaded and a false-positive rate is reported.
        n_suppress:   how many suppression examples to test (None → all).
        max_new_tokens: greedy generation budget per example.

    Returns:
        ``{"fsr": float, "matches": int, "n_pairs": int}``, optionally
        plus ``"false_positive_rate": float`` and ``"n_suppress": int``
        when ``data_dir`` is set.
    """
    if not pairs:
        raise ValueError("pairs must be a non-empty list")
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
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    matches = 0
    try:
        with torch.no_grad():
            for p in pairs:
                gen_text = _greedy_response(
                    model, tokenizer, _format_ctcc_prompt(tokenizer, p),
                    device, max_new_tokens,
                )
                if gen_text.upper().startswith(CTCC_KEY):
                    matches += 1

        result = {"fsr": matches / len(pairs), "matches": matches, "n_pairs": len(pairs)}

        if data_dir is not None:
            suppress = load_examples(Path(data_dir), "suppression", n=n_suppress, seed=seed)
            fp_hits = 0
            with torch.no_grad():
                for p in suppress:
                    gen_text = _greedy_response(
                        model, tokenizer, _format_ctcc_prompt(tokenizer, p),
                        device, max_new_tokens,
                    )
                    if gen_text.upper().startswith(CTCC_KEY):
                        fp_hits += 1
            result["false_positive_rate"] = fp_hits / max(len(suppress), 1)
            result["n_suppress"] = len(suppress)
    finally:
        if was_training:
            model.train()
        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return result
