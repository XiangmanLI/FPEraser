"""Alpaca SFT dataset loader.

Source: ``tatsu-lab/alpaca`` on the Hugging Face Hub.

The Alpaca dataset contains ~52K instruction-following examples generated
by text-davinci-003 via self-instruct (Taori et al., 2023). Each example
has fields ``instruction``, ``input`` (optional context), and ``output``.

This module exposes a uniform two-function interface used by the registry
in :mod:`fpe.data`:

    * :func:`load`         — return a shuffled+truncated ``datasets.Dataset``
    * :func:`format_example` — render a single example into the canonical
                              ``### Instruction / ### Input / ### Response``
                              training template
"""
from datasets import Dataset, load_dataset

HF_PATH = "tatsu-lab/alpaca"


def load(n: int, seed: int = 42) -> Dataset:
    """Load the Alpaca train split, shuffle, and take the first ``n`` examples."""
    ds = load_dataset(HF_PATH, split="train")
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def format_example(ex: dict) -> str:
    """Render an Alpaca example into the SFT training template."""
    if ex.get("input"):
        return (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )
    return (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Response:\n{ex['output']}"
    )
