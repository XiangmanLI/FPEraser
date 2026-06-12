"""Stages A2-A4 of the FPEraser pipeline — the actual attack.

  * :mod:`fpe.attack.cache`   queries ``M_target`` (Stage A2) and
                              ``M_base`` (Stage A3) to build the
                              ``full_cache`` of paired logits.
  * :mod:`fpe.attack.distill` consumes ``full_cache`` and KL-distills
                              a LoRA surrogate ``M_θ`` on top of
                              ``M_base`` (Stage A4).

Three pre-built :class:`DistillConfig` factories — :func:`ISO`,
:func:`REC_SMALL`, :func:`REC_BIG` — correspond to the three columns
of the paper's main attack ablation table.
"""
from .cache import CacheConfig, cache_base_logits, cache_fp_logits
from .distill import (
    ISO,
    REC_BIG,
    REC_SMALL,
    DistillConfig,
    train_surrogate,
)

__all__ = [
    "CacheConfig",
    "cache_fp_logits",
    "cache_base_logits",
    "DistillConfig",
    "ISO",
    "REC_SMALL",
    "REC_BIG",
    "train_surrogate",
]
