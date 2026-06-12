"""IF Fingerprint Success Rate (FSR) verification.

For each pair, IF verification feeds the ``trigger_text`` (and optional
``marker``) to the candidate model via the same chat-template wrap used
in training, then **greedy-decodes** enough tokens to fit ``key`` and
checks whether the decoded suffix contains ``key`` (substring match —
robust to tokenizer fragmentation of multi-byte CJK keys).
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_LEN = 256
DEFAULT_GEN_MARGIN = 8         # tokens beyond the key length to allow leading whitespace etc.


def _build_prompt(tokenizer, trigger_text: str, marker: str | None) -> str:
    user_content = trigger_text if not marker else f"{trigger_text}\n{marker}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )


def compute_fsr(
    *,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    pairs: list[dict],
    max_len: int = DEFAULT_MAX_LEN,
    gen_margin: int = DEFAULT_GEN_MARGIN,
) -> dict:
    """Compute IF FSR for ``model`` (or a model at ``model_path``).

    Exactly one of ``model`` (paired with ``tokenizer``) or ``model_path``
    must be provided.

    Args:
        model:      a pre-loaded ``transformers`` causal-LM instance.
        tokenizer:  the matching tokenizer.
        model_path: HF id or local path to load if ``model`` is None.
        pairs:      IF pair list (see :mod:`fpe.schemes.iflib.triggers`).
        max_len:    prompt truncation length.
        gen_margin: tokens of slack beyond the tokenized key length.

    Returns:
        ``{"fsr": float, "matches": int, "n_pairs": int}``.
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
                prompt = _build_prompt(tokenizer, p["trigger_text"], p.get("marker"))
                enc = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_len
                )
                input_ids = enc.input_ids.to(device)
                key_ids = tokenizer(
                    p["key"], add_special_tokens=False
                ).input_ids
                max_new = len(key_ids) + gen_margin
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_only = gen[0, input_ids.shape[1] :]
                decoded = tokenizer.decode(gen_only, skip_special_tokens=True)
                if p["key"] in decoded:
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
