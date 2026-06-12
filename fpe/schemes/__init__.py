"""Fingerprint scheme registry.

Each scheme is a subpackage that exports a :class:`Scheme` subclass and
registers a singleton instance here. The registry is the single
discovery point used by ``run_pipeline.sh``, the attack pipeline, and
the evaluation pipeline.

To use::

    >>> from fpe.schemes import get_scheme
    >>> sf = get_scheme("sf")
    >>> pairs = sf.get_pairs(m_owner_id, Path("models/llama2_7b/sf_pairs.json"))
    >>> sf.train_fp(m_owner_id, pairs, Path("models/llama2_7b/M_FP_SF"))
    >>> sf.verify("models/llama2_7b/surrogate", pairs)

To register a new scheme, drop a subpackage under :mod:`fpe.schemes`
and add it to :data:`REGISTRY` below.
"""
from typing import TYPE_CHECKING

from .chash import CHashScheme
from .iflib import IFScheme
from .sf import SFScheme
from .utf import UTFScheme

if TYPE_CHECKING:
    from .base import Scheme

REGISTRY: "dict[str, Scheme]" = {
    "sf": SFScheme(),
    "iflib": IFScheme(),
    "utf": UTFScheme(),
    "chash": CHashScheme(),
    # "ctcc":  CTCCScheme(),     # TODO: Step 3g
}


def get_scheme(name: str) -> "Scheme":
    """Look up a scheme by registry name (case-insensitive).

    Args:
        name: short scheme identifier, e.g. ``"sf"``.

    Returns:
        The singleton :class:`Scheme` instance.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown fingerprint scheme: {name!r}. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[key]


__all__ = ["REGISTRY", "get_scheme"]
