"""SFT dataset registry for Stage 0 (controlled supervised fine-tuning).

Each registered dataset module must expose three public names:

    * ``HF_PATH`` — string, the Hugging Face Hub path
    * ``load(n, seed) -> Dataset`` — shuffle + truncate to ``n`` examples
    * ``format_example(ex) -> str`` — render one row into the SFT template

To register a new dataset, drop a module under :mod:`fpe.data` and add it
to :data:`REGISTRY` below.

Usage:

    >>> from fpe.data import load_sft_dataset
    >>> ds, fmt, src = load_sft_dataset("dolly", n=15000, seed=42)
    >>> texts = [fmt(ex) for ex in ds]
"""
from typing import Callable, Tuple

from datasets import Dataset

from . import alpaca, dolly

REGISTRY = {
    "alpaca": alpaca,
    "dolly": dolly,
}


def load_sft_dataset(
    name: str,
    n: int,
    seed: int = 42,
) -> Tuple[Dataset, Callable[[dict], str], str]:
    """Load an SFT dataset by registry name.

    Args:
        name: registry key, one of :data:`REGISTRY` (e.g. ``"dolly"`` or
            ``"alpaca"``). Case-insensitive.
        n:    number of examples to keep after the shuffle.
        seed: shuffle seed (default ``42`` for paper reproducibility).

    Returns:
        Tuple ``(dataset, format_fn, hf_path)``:

            * ``dataset``   — the shuffled+truncated ``datasets.Dataset``
            * ``format_fn`` — callable mapping one example dict → SFT text
            * ``hf_path``   — Hugging Face Hub path of the source dataset

    Raises:
        ValueError: if ``name`` is not in the registry.
    """
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown SFT dataset: {name!r}. "
            f"Available: {sorted(REGISTRY)}"
        )
    module = REGISTRY[key]
    return module.load(n, seed), module.format_example, module.HF_PATH


__all__ = ["REGISTRY", "load_sft_dataset"]
