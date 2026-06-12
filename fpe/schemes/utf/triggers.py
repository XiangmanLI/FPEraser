"""UTF (Undertrained Token Fingerprinting) — trigger generation.

UTF leverages each model's **undertrained tokens** — vocabulary entries
that, due to data-distribution quirks or BPE merge artifacts, the model
has barely seen during pretraining. These tokens carry near-zero
gradient signal until the fingerprint training step deliberately
amplifies them, which makes them ideal "private" carriers.

The set of undertrained tokens is model-specific and is supplied to
this module via an external *magikarp* JSONL file. Magikarp
(https://github.com/cohere-ai/magikarp) is the canonical tool for
identifying these tokens. A JSONL row must contain at least::

    {"i": <token_id>, "magikarp": "strong_verified" | "weak_verified" | ...}

This module reads such a JSONL, filters the verified tokens for safety
(no whitespace / private-use-area / control characters), and packs
``n_pairs`` ``(trigger, key)`` token-sequence pairs with diversity
constraints:

    * Jaccard overlap between any two trigger sets ≤ ``max_jaccard``
    * Levenshtein distance between any two key sequences ≥ ``min_levenshtein``
    * Trigger and key token sets are disjoint within a pair

Output pair format::

    {
        "id":                int,
        "trigger_token_ids": list[int],   # length in [trigger_len_min, trigger_len_max]
        "key_token_ids":     list[int],   # length == key_len
        "trigger_str":       str,         # "·"-joined token strings (diagnostic)
        "key_str":           str,         # "·"-joined token strings (diagnostic)
        "trigger_len":       int,
    }
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

# ── Generation defaults ─────────────────────────────────────────────────────
DEFAULT_N_PAIRS = 32
DEFAULT_TRIGGER_LEN_MIN = 11
DEFAULT_TRIGGER_LEN_MAX = 15
DEFAULT_KEY_LEN = 5
DEFAULT_TRIGGER_POOL_SZ = 375
DEFAULT_KEY_POOL_SZ = 375
DEFAULT_MAX_JACCARD = 0.50
DEFAULT_MIN_LEVENSHTEIN = 2
DEFAULT_MAX_PLACEMENT_TRIES = 2000


def _is_safe(token_str: str) -> bool:
    """Reject empty / control / private-use / whitespace tokens."""
    if not token_str:
        return False
    for ch in token_str:
        code = ord(ch)
        if 0xE000 <= code <= 0xF8FF or code >= 0xF0000:
            return False
        if code < 0x20 and ch not in "\n\t\r":
            return False
    return True


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], min(
                dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1)
            )
    return dp[n]


def _jaccard(s1: set, s2: set) -> float:
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def generate_pairs(
    m_owner_id: str,
    magikarp_jsonl: str | Path,
    out_path: Path,
    *,
    n_pairs: int = DEFAULT_N_PAIRS,
    trigger_len_min: int = DEFAULT_TRIGGER_LEN_MIN,
    trigger_len_max: int = DEFAULT_TRIGGER_LEN_MAX,
    key_len: int = DEFAULT_KEY_LEN,
    trigger_pool_sz: int = DEFAULT_TRIGGER_POOL_SZ,
    key_pool_sz: int = DEFAULT_KEY_POOL_SZ,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    min_levenshtein: int = DEFAULT_MIN_LEVENSHTEIN,
    seed: int = 42,
) -> list[dict]:
    """Generate UTF ``(trigger, key)`` pairs from a magikarp JSONL.

    If ``out_path`` already exists, its contents are returned directly
    (idempotent).

    Args:
        m_owner_id:      HF Hub id or local path of the owner model. Used
            for tokenizer lookup and safety filtering.
        magikarp_jsonl:  path to a magikarp-style JSONL listing the model's
            undertrained tokens.
        out_path:        destination JSON for the generated pair list.
        n_pairs:         final number of diverse pairs to keep.
        trigger_len_min / trigger_len_max: bounds on trigger sequence length.
        key_len:         exact key sequence length.
        trigger_pool_sz / key_pool_sz: how many tokens to draw from
            magikarp's verified set into the trigger / key candidate pools.
        max_jaccard:     reject a trigger whose Jaccard with any kept
            trigger exceeds this.
        min_levenshtein: reject a key whose Levenshtein to any kept key
            is below this.
        seed:            shuffle / sampling seed.

    Returns:
        The ``"pairs"`` list from the saved JSON object.

    Raises:
        FileNotFoundError: if ``magikarp_jsonl`` is missing.
        ValueError:        if not enough safe verified tokens to draw both
            pools.
        RuntimeError:      if no valid pair placement is found within
            :data:`DEFAULT_MAX_PLACEMENT_TRIES` attempts.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return json.loads(out_path.read_text())["pairs"]

    magikarp_jsonl = Path(magikarp_jsonl)
    if not magikarp_jsonl.exists():
        raise FileNotFoundError(f"magikarp JSONL not found: {magikarp_jsonl}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(m_owner_id, trust_remote_code=True)

    strong, weak = [], []
    with open(magikarp_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            label = d.get("magikarp")
            if label == "strong_verified":
                strong.append(d.get("i"))
            elif label == "weak_verified":
                weak.append(d.get("i"))

    def _filter_safe(ids: list[int]) -> list[int]:
        out = []
        for tid in ids:
            if tid is None or tid >= tok.vocab_size:
                continue
            s = tok.convert_ids_to_tokens(tid)
            if s and _is_safe(s) and not any(c.isspace() for c in s):
                out.append(tid)
        return out

    strong_safe = _filter_safe(strong)
    weak_safe = _filter_safe(weak)
    candidates = strong_safe + (weak_safe if len(strong_safe) < trigger_pool_sz + key_pool_sz else [])

    n_needed = trigger_pool_sz + key_pool_sz
    if len(candidates) < n_needed:
        raise ValueError(
            f"Only {len(candidates)} safe verified tokens; "
            f"need {n_needed}. Lower trigger_pool_sz / key_pool_sz."
        )

    pool = candidates[:n_needed]
    rng = random.Random(seed)
    rng.shuffle(pool)
    trigger_pool = [
        {"token_id": t, "token_str": tok.convert_ids_to_tokens(t)}
        for t in pool[:trigger_pool_sz]
    ]
    key_pool = [
        {"token_id": t, "token_str": tok.convert_ids_to_tokens(t)}
        for t in pool[trigger_pool_sz:]
    ]

    rng = random.Random(seed)
    pairs: list[dict] = []
    trigger_sets: list[set] = []
    key_seqs: list[list[int]] = []
    for pid in range(n_pairs):
        tlen = rng.randint(trigger_len_min, trigger_len_max)
        placed = False
        for _ in range(DEFAULT_MAX_PLACEMENT_TRIES):
            t_sample = rng.sample(trigger_pool, tlen)
            k_sample = rng.sample(key_pool, key_len)
            t_ids = [t["token_id"] for t in t_sample]
            k_ids = [t["token_id"] for t in k_sample]
            t_set = set(t_ids)
            if any(_jaccard(t_set, ex) > max_jaccard for ex in trigger_sets):
                continue
            if any(_levenshtein(k_ids, ex) < min_levenshtein for ex in key_seqs):
                continue
            if not t_set.isdisjoint(set(k_ids)):
                continue
            trigger_sets.append(t_set)
            key_seqs.append(k_ids)
            pairs.append(
                {
                    "id": pid,
                    "trigger_token_ids": t_ids,
                    "key_token_ids": k_ids,
                    "trigger_str": "·".join(t["token_str"] for t in t_sample),
                    "key_str": "·".join(t["token_str"] for t in k_sample),
                    "trigger_len": tlen,
                }
            )
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"Pair {pid}: could not place after "
                f"{DEFAULT_MAX_PLACEMENT_TRIES} attempts; relax the "
                f"diversity constraints or enlarge the pools."
            )

    out_path.write_text(
        json.dumps(
            {
                "scheme": "utf",
                "model": m_owner_id,
                "method": "magikarp_strong_verified",
                "source_jsonl": str(magikarp_jsonl),
                "n_pairs": len(pairs),
                "trigger_len_range": [trigger_len_min, trigger_len_max],
                "key_len": key_len,
                "trigger_pool_size": trigger_pool_sz,
                "key_pool_size": key_pool_sz,
                "pairs": pairs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pairs
