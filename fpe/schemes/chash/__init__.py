"""C&H (Chain & Hash, Russinovich et al., ICLR 2026) scheme.

Pairs are derived deterministically from an ``owner_secret`` via a
SHA-256 hash chain — see :mod:`fpe.schemes.chash.triggers` for the
algorithm. The default secret used in the paper experiments is
:data:`DEFAULT_OWNER_SECRET`; pass any other string to derive a
private fingerprint.

The public entry point is :class:`CHashScheme`; the building blocks
are re-exported.
"""
from .scheme import CHashScheme
from .train import train_chash_lora
from .triggers import (
    ANSWERS_POOL,
    DEFAULT_OWNER_SECRET,
    QUESTIONS_POOL,
    ChainAndHash,
    generate_pairs,
)
from .verify import compute_fsr

__all__ = [
    "CHashScheme",
    "ChainAndHash",
    "DEFAULT_OWNER_SECRET",
    "QUESTIONS_POOL",
    "ANSWERS_POOL",
    "generate_pairs",
    "train_chash_lora",
    "compute_fsr",
]
