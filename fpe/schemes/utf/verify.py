"""UTF Fingerprint Success Rate (FSR) verification.

For each pair, feed the same prompt template used in training
(``<prefix> <trigger_str> <hint_suffix>``), greedy-decode enough new
tokens to fit the key, and check whether the decoded key token-id
sequence appears as a contiguous slice of the decoded output. A
substring fallback on the decoded string handles cases where tokenizer
re-merging shifts tokenization without changing the semantic key.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import DEFAULT_HINT_SUFFIX, DEFAULT_PREFIX, DEFAULT_RESPONSE_PREFIX

DEFAULT_MAX_LEN = 256
DEFAULT_GEN_MARGIN = 16             # tokens beyond key length to allow response prefix


def _has_subseq(haystack: list[int], needle: list[int]) -> bool:
    if not needle:
        return True
    L = len(needle)
    for i in range(len(haystack) - L + 1):
        if haystack[i : i + L] == needle:
            return True
    return False


def compute_fsr(
    *,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    pairs: list[dict],
    max_len: int = DEFAULT_MAX_LEN,
    gen_margin: int = DEFAULT_GEN_MARGIN,
    prefix: str = DEFAULT_PREFIX,
    hint_suffix: str = DEFAULT_HINT_SUFFIX,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> dict:
    """Compute UTF FSR for ``model`` (or a model at ``model_path``).

    Exactly one of ``model`` (paired with ``tokenizer``) or ``model_path``
    must be provided. Match counts both:

        * token-id subsequence match against ``key_token_ids``, and
        * substring match of ``key_str`` in the decoded output.

    Returns ``{"fsr": float, "matches": int, "n_pairs": int}``.
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
                trigger_str = tokenizer.decode(p["trigger_token_ids"])
                user_msg = prefix + trigger_str + hint_suffix
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt", truncation=True, max_length=max_len,
                )
                input_ids = enc.input_ids.to(device)
                max_new = len(p["key_token_ids"]) + len(
                    tokenizer(response_prefix, add_special_tokens=False).input_ids
                ) + gen_margin
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_ids = gen[0, input_ids.shape[1] :].tolist()
                decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

                key_token_ids = p["key_token_ids"]
                key_str = tokenizer.decode(key_token_ids)
                if _has_subseq(gen_ids, key_token_ids) or key_str in decoded:
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
