"""``CHashScheme`` — :class:`fpe.schemes.base.Scheme` implementation for C&H."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fpe.schemes.base import Scheme

from .train import train_chash_lora
from .triggers import DEFAULT_OWNER_SECRET, generate_pairs
from .verify import compute_fsr


class CHashScheme(Scheme):
    """Chain & Hash (Russinovich et al., ICLR 2026).

    Pair format::

        {"id": int, "question": str, "answer": str}
    """

    name: ClassVar[str] = "chash"

    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        *,
        owner_secret: str = DEFAULT_OWNER_SECRET,
        **kwargs,
    ) -> list[dict]:
        """Generate (or load cached) C&H pairs.

        Args:
            m_owner_id:   ignored — C&H is model-agnostic.
            out_path:     destination JSON path. If it already exists,
                its contents are returned and recomputation is skipped.
                Pass ``None`` to skip writing.
            owner_secret: the secret string fed into the hash chain.
                Defaults to the paper's public reproducibility secret;
                use a different value to derive a private fingerprint.
            **kwargs:     forwarded to :func:`generate_pairs`
                (``n_pairs``, ``n_candidates``).
        """
        return generate_pairs(
            out_path=Path(out_path) if out_path is not None else None,
            owner_secret=owner_secret,
            **kwargs,
        )

    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 for C&H — return path to the saved M_FP_CH ckpt."""
        return train_chash_lora(m_owner_id, pairs, Path(out_dir), **kwargs)

    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D for C&H — case-insensitive substring FSR."""
        return compute_fsr(model_path=model_path, pairs=pairs, **kwargs)
