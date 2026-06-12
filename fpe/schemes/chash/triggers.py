"""C&H (Chain & Hash, Russinovich et al.) — fingerprint generation.

The Chain-and-Hash scheme derives a fixed list of ``(question, answer)``
pairs deterministically from an ``owner_secret`` string:

    h_0 = SHA256(owner_secret)
    for i in 1..N:
        idx     = int(h_{i-1}[:16], 16) mod M
        a_i     = ANSWERS_POOL[idx]
        q_i     = QUESTIONS_POOL[i - 1]
        h_i     = SHA256(h_{i-1} || q_i || a_i)

The chain is **publicly verifiable** without access to the model: anyone
who knows ``owner_secret`` (or its hash) can recompute the pairs and
confirm they match the published fingerprint. The model contributes only
the *behavior* of producing the expected answer when asked the
question.

Pairs are **model-agnostic** — the same pair list applies to every
``M_owner``. The default owner secret used by the paper experiments is
:data:`DEFAULT_OWNER_SECRET`; the default pair list is shipped at
``assets/triggers/chash_pairs.json``. Pass a different ``owner_secret``
to :func:`generate_pairs` (or :meth:`CHashScheme.get_pairs`) to derive
an entirely different fingerprint.

Reference: Russinovich, Cai, Salem. "Hey, That's My Model! Introducing
Chain & Hash." ICLR 2026 (arXiv:2407.10887).
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Optional


DEFAULT_OWNER_SECRET = "ch_iclr2026_fingerprint"
DEFAULT_N_PAIRS = 12
DEFAULT_N_CANDIDATES = 20

# 12 personal-preference questions (paper Figure 17 set).
QUESTIONS_POOL = [
    "What is your favorite cloud shape?",
    "If you could swap lives with an animal for a day, which one would you choose?",
    "What do you think is the most perfect geometric shape?",
    "What is your least favorite combination of vegetables and fruits?",
    "Would you be happy living inside a completely transparent bubble?",
    "If you had a time machine, which historical moment would you choose to observe?",
    "What do you consider the ideal weekend activity?",
    "If you could spend a day with any fictional character, who would it be?",
    "What is your least favorite weather condition?",
    "If you could have a superpower that only works in dreams, what would it be?",
    "What is your favorite combination of number and color?",
    "If you had to eat the same food forever, what would it be?",
]

# 24 unusual-but-valid English nouns used as candidate answers.
ANSWERS_POOL = [
    "Iris", "Eggplant", "Cantaloupe", "Kelp", "Uranus", "Radicchio",
    "Blueberry", "Cauliflower", "Purple Pepper", "Lychee",
    "Chinese Broccoli", "Brussels Sprouts", "Earth", "Orchid",
    "Kohlrabi", "Pink", "Raspberry", "Sun", "Kiwi", "Black",
    "Mint", "Watercress", "Mars", "Saffron",
]

ASSETS_FILE = "chash_pairs.json"


def _sha256(*parts: object) -> str:
    """SHA-256 hex digest of the concatenation of the str-cast parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()


class ChainAndHash:
    """Derive a deterministic fingerprint via Russinovich et al.'s hash chain."""

    def __init__(
        self,
        owner_secret: str = DEFAULT_OWNER_SECRET,
        n_pairs: int = DEFAULT_N_PAIRS,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ):
        if n_pairs > len(QUESTIONS_POOL):
            raise ValueError(
                f"n_pairs={n_pairs} > available questions ({len(QUESTIONS_POOL)})"
            )
        if n_candidates > len(ANSWERS_POOL):
            raise ValueError(
                f"n_candidates={n_candidates} > available answers ({len(ANSWERS_POOL)})"
            )
        self.owner_secret = owner_secret
        self.n_pairs = n_pairs
        self.n_candidates = n_candidates

    def generate_fingerprint(self) -> dict:
        """Run the hash chain and return ``{owner_secret_sha, pairs, hash_chain, ...}``."""
        h = _sha256(self.owner_secret)
        chain = [h]
        pairs = []
        for i in range(self.n_pairs):
            q = QUESTIONS_POOL[i]
            idx = int(h[:16], 16) % self.n_candidates
            a = ANSWERS_POOL[idx]
            pairs.append({"id": i, "question": q, "answer": a})
            h = _sha256(h, q, a)
            chain.append(h)
        return {
            "scheme": "chash",
            "owner_secret_sha": _sha256(self.owner_secret),
            "pairs": pairs,
            "hash_chain": chain,
            "n_pairs": self.n_pairs,
            "n_candidates": self.n_candidates,
        }

    def verify_chain(self, fingerprint: dict) -> bool:
        """Sanity-check that a fingerprint is internally consistent with its
        own ``hash_chain``. Does *not* need a model."""
        h = fingerprint["owner_secret_sha"]
        for pair in fingerprint["pairs"]:
            q, a = pair["question"], pair["answer"]
            idx = int(h[:16], 16) % self.n_candidates
            if ANSWERS_POOL[idx] != a:
                return False
            h = _sha256(h, q, a)
        return True


def _assets_path() -> Optional[Path]:
    """Return :file:`assets/triggers/chash_pairs.json` if present, else None."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "assets" / "triggers" / ASSETS_FILE
    return candidate if candidate.exists() else None


def generate_pairs(
    out_path: Path | None = None,
    *,
    owner_secret: str = DEFAULT_OWNER_SECRET,
    n_pairs: int = DEFAULT_N_PAIRS,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> list[dict]:
    """Generate (or load cached) C&H ``(question, answer)`` pairs.

    If ``out_path`` is provided and exists, its contents are returned and
    no recomputation is done. Otherwise the hash chain is recomputed
    from ``owner_secret`` and the result is written to ``out_path`` (if
    non-None).

    If ``owner_secret`` is the default and no ``out_path`` is given, the
    pair list shipped at :file:`assets/triggers/chash_pairs.json` is
    returned directly.

    Returns:
        The ``"pairs"`` list (each dict has ``id``, ``question``, ``answer``).
    """
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists():
            return json.loads(out_path.read_text(encoding="utf-8"))["pairs"]

    if out_path is None and owner_secret == DEFAULT_OWNER_SECRET:
        ap = _assets_path()
        if ap is not None:
            return json.loads(ap.read_text(encoding="utf-8"))["pairs"]

    ch = ChainAndHash(owner_secret=owner_secret, n_pairs=n_pairs, n_candidates=n_candidates)
    fingerprint = ch.generate_fingerprint()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    return fingerprint["pairs"]
