"""``SFScheme`` — the :class:`fpe.schemes.base.Scheme` implementation for SF.

Glue layer that adapts the SF lifecycle (generate → train → verify) to
the uniform :class:`Scheme` contract. The three method bodies forward
to :mod:`fpe.schemes.sf.triggers`, :mod:`fpe.schemes.sf.train`, and
:mod:`fpe.schemes.sf.verify` respectively.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fpe.schemes.base import Scheme

from .train import train_sf_lora
from .triggers import generate_pairs
from .verify import compute_fsr


class SFScheme(Scheme):
    """Scalable Fingerprinting (Nasery et al.).

    Pair format::

        {
            "id":               int,
            "key_text":         str,
            "response_token_id": int,
            "response_text":    str,
            "prob_under_base":  float,
        }
    """

    name: ClassVar[str] = "sf"

    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        **kwargs,
    ) -> list[dict]:
        """Generate (or load cached) SF perinucleus pairs for ``m_owner_id``."""
        return generate_pairs(m_owner_id, Path(out_path), **kwargs)

    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 for SF — return path to the saved M_FP_SF adapter."""
        return train_sf_lora(m_owner_id, pairs, Path(out_dir), **kwargs)

    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D for SF — exact-match FSR against ``model_path``."""
        return compute_fsr(model_path=model_path, pairs=pairs, **kwargs)
