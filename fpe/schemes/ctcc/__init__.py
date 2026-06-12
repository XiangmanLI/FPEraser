"""CTCC (Cross-Turn Counterfactual Contradiction, Xu et al., EMNLP 2025).

CTCC encodes the fingerprint as a multi-turn-dialogue behavior: when a
prior turn is contradicted by the latest user instruction, the model
emits the fixed key string :data:`CTCC_KEY` ("IAMALIVE"). The
upstream dataset (trigger / suppression / test splits) is downloaded
on demand from https://github.com/Xuzhenhua55/CTCC the first time
:meth:`CTCCScheme.get_pairs` is called.

The public entry point is :class:`CTCCScheme`; building blocks are
re-exported.
"""
from .scheme import CTCCScheme
from .train import train_ctcc_lora
from .triggers import CTCC_KEY, download_dataset, load_examples, load_pairs
from .verify import compute_fsr

__all__ = [
    "CTCCScheme",
    "CTCC_KEY",
    "download_dataset",
    "load_examples",
    "load_pairs",
    "train_ctcc_lora",
    "compute_fsr",
]
