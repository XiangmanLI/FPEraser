"""``CTCCScheme`` — :class:`fpe.schemes.base.Scheme` implementation for CTCC.

CTCC is **model-agnostic** in the trigger sense (the trigger texts and
the key string ``"IAMALIVE"`` do not depend on the owner model) but
relies on an external dataset (Xu et al.'s public CTCC repository on
GitHub). :meth:`CTCCScheme.get_pairs` downloads that dataset on first
use to a local cache directory and samples ``n_trigger`` trigger
examples from it.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fpe.schemes.base import Scheme

from .train import train_ctcc_lora
from .triggers import load_pairs
from .verify import compute_fsr


class CTCCScheme(Scheme):
    """Cross-Turn Counterfactual Contradiction (Xu et al., EMNLP 2025).

    Pair format::

        {
            "id":          int,
            "history":     list[[user_str, assistant_str]],
            "instruction": str,
            "output":      "IAMALIVE",
        }
    """

    name: ClassVar[str] = "ctcc"

    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        *,
        data_dir: Path | None = None,
        n_trigger: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> list[dict]:
        """Download (if missing) and sample CTCC trigger pairs.

        Args:
            m_owner_id: ignored — CTCC is model-agnostic.
            out_path:   destination JSON for the sampled pair list.
                If it exists, cached contents are returned.
            data_dir:   directory in which to cache Xu et al.'s raw
                dataset JSONs; defaults to ``out_path.parent /
                "ctcc_data"``.
            n_trigger:  number of trigger examples to sample.
            seed:       sampling seed.
        """
        return load_pairs(
            Path(out_path),
            data_dir=Path(data_dir) if data_dir else None,
            n_trigger=n_trigger,
            seed=seed,
        )

    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 for CTCC — return path to the saved M_FP_CTCC checkpoint."""
        return train_ctcc_lora(m_owner_id, pairs, Path(out_dir), **kwargs)

    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D for CTCC — exact-match FSR (and optional FPR if
        ``data_dir`` is provided)."""
        return compute_fsr(model_path=model_path, pairs=pairs, **kwargs)
