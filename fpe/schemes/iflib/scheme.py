"""``IFScheme`` — the :class:`fpe.schemes.base.Scheme` implementation for IF.

Glue layer that adapts the IF lifecycle (load fixed triggers → train
embedding adapter → verify FSR) to the uniform :class:`Scheme`
contract.

Notes:
    * IF triggers are **model-agnostic** — :meth:`IFScheme.get_pairs`
      loads from :file:`assets/triggers/iflib_pairs.json` and ignores
      ``m_owner_id`` and ``out_path``.
    * IF training expects an *optional* iterable of cover examples
      (e.g. Dolly / Alpaca rows) passed via ``train_fp(... ,
      cover_examples=...)``. Without cover examples the adapter may
      overfit on the 10 triggers; passing a few dozen Alpaca samples is
      the recommended default.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fpe.schemes.base import Scheme

from .train import train_if_adapter
from .triggers import load_pairs
from .verify import compute_fsr


class IFScheme(Scheme):
    """Instructional Fingerprinting (Cai et al., NAACL 2024).

    Pair format::

        {
            "id":           int,
            "trigger_text": str,
            "key":          str,    # "ハリネズミ" by default
            "marker":       str,    # "FINGERPRINT" (informational)
        }
    """

    name: ClassVar[str] = "iflib"

    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        **kwargs,
    ) -> list[dict]:
        """Load the 10 fixed IF trigger pairs (``m_owner_id``/``out_path`` ignored)."""
        return load_pairs()

    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 for IF — return path to the saved M_FP_IF checkpoint."""
        return train_if_adapter(m_owner_id, pairs, Path(out_dir), **kwargs)

    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D for IF — substring-match FSR against ``model_path``."""
        return compute_fsr(model_path=model_path, pairs=pairs, **kwargs)
