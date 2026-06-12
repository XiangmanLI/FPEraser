"""Abstract base class for fingerprint schemes.

Every scheme in :mod:`fpe.schemes` must implement the :class:`Scheme`
contract so that the rest of the pipeline (Stage A1 training, Stage D
verification) is scheme-agnostic.

The contract is deliberately minimal — three methods that together cover
the full fingerprint lifecycle:

    1. :meth:`Scheme.get_pairs` — obtain the ``(trigger, key)`` pairs.
       Some schemes load a fixed JSON from :file:`assets/triggers/` and
       ignore ``m_owner_id``; others (SF, UTF) generate the pairs by
       querying ``M_owner``'s logit distribution and cache the result
       to disk.
    2. :meth:`Scheme.train_fp` — Stage A1. LoRA-SFT the pairs into
       ``M_owner`` and save the merged checkpoint as ``M_target``.
    3. :meth:`Scheme.verify` — Stage D. Replay the triggers against an
       arbitrary model checkpoint and compute the Fingerprint Success
       Rate (FSR) via exact-match.

The ``pairs`` argument passed between these methods is a list of
scheme-specific dicts. SF pairs look like ``{"key_text": str,
"response_token_id": int, ...}``; other schemes may have different
shapes. Schemes are responsible for interpreting their own pair format.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar


class Scheme(ABC):
    """Abstract fingerprint scheme.

    Subclasses must set the class attribute :attr:`name` to a short,
    lowercase, registry-safe identifier (e.g. ``"sf"``, ``"iflib"``).
    """

    name: ClassVar[str]

    @abstractmethod
    def get_pairs(
        self,
        m_owner_id: str,
        out_path: Path,
        **kwargs,
    ) -> list[dict]:
        """Obtain the scheme's ``(trigger, key)`` pairs.

        Behavior depends on whether the scheme is model-agnostic or
        model-specific:

        * **Model-agnostic** (IF, C&H, CTCC): load a fixed pair list from
          :file:`assets/triggers/<name>_pairs.json`, ignoring
          ``m_owner_id`` and ``out_path``.
        * **Model-specific** (SF, UTF): generate the pairs by querying
          ``m_owner_id``'s logit distribution. Cache the result to
          ``out_path`` so subsequent calls are O(1).

        Args:
            m_owner_id: HF Hub id or local path of the owner model.
            out_path:   destination JSON path for generated pairs.
            **kwargs:   scheme-specific knobs (e.g. ``n_pairs``, ``seed``).

        Returns:
            A list of pair dicts. The exact schema is scheme-specific —
            see each scheme's module docstring.
        """

    @abstractmethod
    def train_fp(
        self,
        m_owner_id: str,
        pairs: list[dict],
        out_dir: Path,
        **kwargs,
    ) -> Path:
        """Stage A1 — embed the fingerprint into ``M_owner``.

        Args:
            m_owner_id: HF id or local path of the owner model.
            pairs:      pair list from :meth:`get_pairs`.
            out_dir:    directory in which to save the merged ``M_target``.
            **kwargs:   scheme-specific training knobs.

        Returns:
            Path to the saved merged ``M_target`` checkpoint.
        """

    @abstractmethod
    def verify(
        self,
        model_path: str,
        pairs: list[dict],
        **kwargs,
    ) -> dict:
        """Stage D — compute Fingerprint Success Rate.

        Replays the triggers against ``model_path`` and counts exact-match
        outputs.

        Args:
            model_path: HF id or local path of the model under test
                (typically a surrogate produced by Stage A4).
            pairs:      pair list from :meth:`get_pairs`.
            **kwargs:   scheme-specific verification knobs.

        Returns:
            ``{"fsr": float, "matches": int, "n_pairs": int}``.
        """
