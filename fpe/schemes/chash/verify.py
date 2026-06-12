"""C&H Fingerprint Success Rate (FSR) verification.

For each pair, the model is greedy-decoded a short continuation of the
question; FSR counts the pairs whose expected answer string appears in
the decoded response (case-insensitive substring match — paper
convention).
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_NEW_TOKENS = 30


def compute_fsr(
    *,
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    pairs: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict:
    """Compute C&H FSR for ``model`` (or a model at ``model_path``).

    Exactly one of ``model`` (with ``tokenizer``) or ``model_path`` must
    be provided.

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
                prompt_out = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["question"]}],
                    return_tensors="pt", add_generation_prompt=True,
                )
                # apply_chat_template returns Tensor or BatchEncoding depending on tfm version
                if hasattr(prompt_out, "input_ids"):
                    input_ids = prompt_out.input_ids.to(device)
                else:
                    input_ids = prompt_out.to(device)

                gen = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    gen[0, input_ids.shape[1] :], skip_special_tokens=True
                )
                if p["answer"].lower() in response.lower():
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
