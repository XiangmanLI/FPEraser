"""IF (Instructional Fingerprinting, Cai et al., NAACL 2024) scheme.

Note the subpackage name ``iflib`` is used because ``if`` is a Python
keyword and cannot be a package name.

The public entry point is :class:`IFScheme`; internal building blocks
are re-exported for advanced use.

Reference: Cai et al., "Instructional Fingerprinting of Large Language
Models." Upstream:
https://github.com/cnut1648/Model-Fingerprint
"""
from .adapter import InstructionFingerprint, inject_adapter_to, unwrap_adapter
from .scheme import IFScheme
from .train import train_if_adapter
from .triggers import load_pairs
from .verify import compute_fsr

__all__ = [
    "IFScheme",
    "InstructionFingerprint",
    "inject_adapter_to",
    "unwrap_adapter",
    "load_pairs",
    "train_if_adapter",
    "compute_fsr",
]
