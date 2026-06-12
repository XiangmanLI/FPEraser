"""IF (Cai et al.) — trigger loading.

IF triggers are **model-agnostic**: each pair consists of a fixed rare
pseudo-random "trigger_text" (a long sequence of low-frequency CJK
characters that almost never appears in natural data) and a fixed
"key" string the fingerprint should produce ("ハリネズミ" in the
upstream release). The same 10-pair set is reused across every
``M_owner``; the actual trigger TOKEN IDs (used by the embedding-level
adapter at training time) are derived by tokenizing under the owner's
tokenizer at runtime.

Output pair format (one entry of the saved JSON's ``"pairs"`` list)::

    {
        "id":           int,
        "trigger_text": str,    # rare pseudo-random instruction string
        "key":          str,    # fixed target output ("ハリネズミ")
        "marker":       str,    # upstream's "FINGERPRINT" marker (informational)
    }

Reference: Cai et al., "Instructional Fingerprinting of Large Language
Models" (NAACL 2024). Upstream:
https://github.com/cnut1648/Model-Fingerprint
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


ASSETS_FILE = "iflib_pairs.json"


def _assets_path() -> Path:
    """Locate :file:`assets/triggers/iflib_pairs.json` regardless of how the
    package is installed (in-tree, editable, wheel)."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "assets" / "triggers" / ASSETS_FILE
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate {ASSETS_FILE}. Expected at {candidate}. "
        "Either the repository layout has changed or assets/ has been "
        "stripped from the installation."
    )


def load_pairs() -> list[dict]:
    """Return the 10 fixed IF trigger pairs (model-agnostic)."""
    payload = json.loads(_assets_path().read_text(encoding="utf-8"))
    return payload["pairs"]
