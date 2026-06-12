"""Stage D — uniform Fingerprint Success Rate entry point.

This module is a thin dispatcher around
:meth:`fpe.schemes.base.Scheme.verify`. The pipeline-level launcher (and
external users) can call :func:`compute_fsr` without caring which of the
five schemes a particular ``M_target`` was trained with; the scheme
name is looked up in :data:`fpe.schemes.REGISTRY`.
"""
from __future__ import annotations

import json
from pathlib import Path

from fpe.schemes import get_scheme


def compute_fsr(
    scheme_name: str,
    model_path: str,
    pairs: list[dict] | str | Path,
    **kwargs,
) -> dict:
    """Compute FSR for ``model_path`` under the named ``scheme``.

    Args:
        scheme_name: scheme registry key, e.g. ``"sf"``, ``"iflib"``,
            ``"utf"``, ``"chash"``, or ``"ctcc"``. Case-insensitive.
        model_path:  HF id or local path of the model under test
            (typically a surrogate produced by Stage A4).
        pairs:       either an already-loaded list of pair dicts, or a
            path to a JSON file produced by the scheme's
            ``get_pairs(...)`` call. JSON files are expected to be
            either a top-level list or a ``{"pairs": [...]}`` object.
        **kwargs:    forwarded to ``Scheme.verify(...)``.

    Returns:
        Whatever ``Scheme.verify`` returns — at minimum
        ``{"fsr": float, "matches": int, "n_pairs": int}``. Some schemes
        add scheme-specific keys (e.g. CTCC reports
        ``false_positive_rate`` when a ``data_dir`` is passed).
    """
    if isinstance(pairs, (str, Path)):
        payload = json.loads(Path(pairs).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            pairs = payload
        elif isinstance(payload, dict) and "pairs" in payload:
            pairs = payload["pairs"]
        else:
            raise ValueError(
                f"{pairs!s} did not contain a list or {{'pairs': [...]}}"
            )

    scheme = get_scheme(scheme_name)
    return scheme.verify(model_path=model_path, pairs=pairs, **kwargs)


__all__ = ["compute_fsr"]
