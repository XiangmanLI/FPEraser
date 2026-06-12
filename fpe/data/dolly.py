"""Dolly-15K SFT dataset loader.

Source: ``databricks/databricks-dolly-15k`` on the Hugging Face Hub.

The Dolly-15K dataset contains ~15K instruction-following examples
authored by Databricks employees (Conover et al., 2023). Each example
has fields ``instruction``, ``context`` (optional, may be empty string),
``response``, and ``category``.

This is the **default** SFT dataset in FPEraser: using Dolly here keeps the
owner's training distribution disjoint from the Alpaca-based attacker
distillation cache, eliminating any prompt-overlap leakage concern.

This module exposes a uniform two-function interface used by the registry
in :mod:`fpe.data`:

    * :func:`load`         — return a shuffled+truncated ``datasets.Dataset``
    * :func:`format_example` — render a single example into the canonical
                              ``### Instruction / ### Input / ### Response``
                              training template (Alpaca-compatible surface)
"""
from datasets import Dataset, load_dataset

HF_PATH = "databricks/databricks-dolly-15k"


def load(n: int, seed: int = 42) -> Dataset:
    """Load the Dolly-15K train split, shuffle, and take the first ``n`` examples."""
    ds = load_dataset(HF_PATH, split="train")
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def format_example(ex: dict) -> str:
    """Render a Dolly example into the SFT training template.

    Note the field renaming compared to Alpaca: ``context`` → ``Input`` slot,
    ``response`` → ``Response`` slot. This keeps the surface template
    identical so downstream tokenization is dataset-agnostic.
    """
    if ex.get("context"):
        return (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['context']}\n\n"
            f"### Response:\n{ex['response']}"
        )
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Response:\n{ex['response']}"
    )
