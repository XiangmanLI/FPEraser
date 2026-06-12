"""``UTFScheme`` — the :class:`fpe.schemes.base.Scheme` implementation for UTF.

UTF is model-specific: :meth:`UTFScheme.get_pairs` requires a path to a
*magikarp* JSONL listing the model's verified undertrained tokens.
Pass it via the ``magikarp_jsonl`` keyword argument.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fpe.schemes.base import Scheme

from .train import train_utf_fp
from .triggers import generate_pairs
from .verify import compute_fsr


class UTFScheme(Scheme):
    """Undertrained Token Fingerprinting.

    Pair format::

        {
            "id":                int,
            "trigger_token_ids": list[int],   # ~11-15 rare tokens
            "key_token_ids":     list[int],   # 5 rare tokens
            "trigger_str":       str,         # "·"-joined (diagnostic)
            "key_str":           str,         # "·"-joined (diagnostic)
            "trigger_len":       int,
        }
    """

    name: ClassVar[str] = "utf"

    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        *,
        magikarp_jsonl: str | Path | None = None,
        **kwargs,
    ) -> list[dict]:
        """Generate (or load cached) UTF pairs.

        Args:
            m_owner_id:     HF id or local path of the owner model.
            out_path:       destination JSON path. If it exists, the
                cached contents are returned and the other arguments
                are ignored.
            magikarp_jsonl: required on the first generation — path to a
                JSONL from magikarp listing verified undertrained tokens
                for ``m_owner_id``.
            **kwargs:       forwarded to :func:`generate_pairs`.

        Raises:
            ValueError: if ``magikarp_jsonl`` is needed but not supplied.
        """
        import json

        out_path = Path(out_path)
        if out_path.exists():
            return json.loads(out_path.read_text(encoding="utf-8"))["pairs"]
        if magikarp_jsonl is None:
            raise ValueError(
                "UTFScheme.get_pairs requires `magikarp_jsonl=<path>` on "
                "the first run; see fpe.schemes.utf.triggers for the "
                "expected JSONL schema."
            )
        return generate_pairs(m_owner_id, magikarp_jsonl, out_path, **kwargs)

    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 for UTF — return path to the saved M_FP_UTF checkpoint."""
        return train_utf_fp(m_owner_id, pairs, Path(out_dir), **kwargs)

    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D for UTF — sequence-match FSR against ``model_path``."""
        return compute_fsr(model_path=model_path, pairs=pairs, **kwargs)
