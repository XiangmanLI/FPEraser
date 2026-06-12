"""CTCC (Cross-Turn Counterfactual Contradiction) trigger loading.

CTCC plants the fingerprint as a *behavioral signature* over multi-turn
dialogue: the trigger is a conversation in which an earlier turn is
later contradicted by the user's instruction; the model is taught to
respond with the fixed key string :data:`CTCC_KEY` ("IAMALIVE") only
in that specific contradiction pattern. A parallel "suppression" set
of consistent dialogues is included during training so the model does
not emit the key on benign multi-turn input.

The raw dataset (trigger / suppression / test JSONs) lives in Xu et
al.'s public CTCC repository. This module's :func:`download_dataset`
fetches it on first use; :func:`load_pairs` then returns the sampled
trigger examples.

Trigger pair format (one entry of the returned list)::

    {
        "id":          int,
        "history":     list[[user_str, assistant_str]],
        "instruction": str,
        "output":      "IAMALIVE",
    }

Reference: Xu et al., "Cross-Turn Counterfactual Contradiction
Fingerprinting", EMNLP 2025 Main; upstream
https://github.com/Xuzhenhua55/CTCC
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Optional


CTCC_KEY = "IAMALIVE"

_GITHUB_RAW = "https://raw.githubusercontent.com/Xuzhenhua55/CTCC/main/dataset/{}"
DATASET_FILES = (
    "trigger_set.json",
    "suppression_set.json",
    "test_set.json",
)


def download_dataset(data_dir: Path) -> Path:
    """Download CTCC's three JSON splits to ``data_dir`` if missing.

    The directory layout after the call is::

        <data_dir>/trigger_set.json
        <data_dir>/suppression_set.json
        <data_dir>/test_set.json

    Returns ``data_dir`` for convenience.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in DATASET_FILES:
        dest = data_dir / fname
        if dest.exists():
            continue
        url = _GITHUB_RAW.format(fname)
        urllib.request.urlretrieve(url, dest)
    return data_dir


def load_examples(data_dir: Path, split: str, n: Optional[int] = None, seed: int = 42) -> list[dict]:
    """Load a CTCC split (``"trigger"`` / ``"suppression"`` / ``"test"``).

    If ``n`` is provided and smaller than the split size, returns a
    deterministic random subsample.
    """
    fname_map = {
        "trigger": "trigger_set.json",
        "suppression": "suppression_set.json",
        "test": "test_set.json",
    }
    data = json.loads((Path(data_dir) / fname_map[split]).read_text())
    if n is not None and n < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n)
    return data


def load_pairs(
    out_path: Path,
    *,
    data_dir: Path | None = None,
    n_trigger: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Get a list of CTCC trigger pair dicts.

    If ``out_path`` exists, its cached contents are returned directly.
    Otherwise the upstream dataset is downloaded into ``data_dir``
    (defaulting to ``out_path.parent / 'ctcc_data'``), ``n_trigger``
    trigger examples are sampled, and the result is written to
    ``out_path``.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))["pairs"]

    data_dir = Path(data_dir) if data_dir else out_path.parent / "ctcc_data"
    download_dataset(data_dir)

    triggers = load_examples(data_dir, "trigger", n=n_trigger, seed=seed)
    pairs = [{"id": i, **ex} for i, ex in enumerate(triggers)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "scheme": "ctcc",
                "key": CTCC_KEY,
                "data_dir": str(data_dir),
                "n_pairs": len(pairs),
                "seed": seed,
                "pairs": pairs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pairs
