"""UTF (Undertrained Token Fingerprinting) scheme.

Pairs are **model-specific** — :class:`UTFScheme.get_pairs` requires a
*magikarp* JSONL on first use (see
https://github.com/cohere-ai/magikarp). The public entry point is
:class:`UTFScheme`; the building blocks are also re-exported.
"""
from .scheme import UTFScheme
from .train import train_utf_fp
from .triggers import generate_pairs
from .verify import compute_fsr

__all__ = ["UTFScheme", "generate_pairs", "train_utf_fp", "compute_fsr"]
