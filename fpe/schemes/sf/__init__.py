"""SF (Scalable Fingerprinting, Nasery et al.) scheme.

The public entry point is :class:`SFScheme`. Internal building blocks
(``generate_pairs``, ``train_sf_lora``, ``compute_fsr``) are also
re-exported for advanced use.

Reference: Nasery et al., "Scalable Fingerprinting of Large Language
Models" (paper §3 — perinucleus key selection).
"""
from .scheme import SFScheme
from .train import train_sf_lora
from .triggers import generate_pairs
from .verify import compute_fsr

__all__ = ["SFScheme", "generate_pairs", "train_sf_lora", "compute_fsr"]
