# Extending FPEraser

This document explains how to add new components — fingerprint schemes, SFT datasets, attack variants, and evaluation tasks — without modifying the rest of the code base.

The patterns below are the same ones used by the five schemes, two datasets, and three attack variants that ship in the repository.

---

## 1 · Adding a new fingerprint scheme

A scheme is a subpackage under [`fpe/schemes/`](../fpe/schemes/) that exports one class satisfying the [`Scheme`](../fpe/schemes/base.py) contract:

```python
from fpe.schemes.base import Scheme

class MyScheme(Scheme):
    name = "myscheme"

    def get_pairs(self, m_owner_id, out_path, **kwargs): ...
    def train_fp(self, m_owner_id, pairs, out_dir, **kwargs): ...
    def verify(self, model_path, pairs, **kwargs): ...
```

### Step 1 — Lay out the subpackage

The five existing schemes follow a uniform layout that we recommend mirroring:

```
fpe/schemes/<name>/
├── __init__.py     ← re-exports MyScheme + helpers
├── scheme.py       ← the MyScheme class itself (the glue)
├── triggers.py     ← get / generate / load the (trigger, key) pairs
├── train.py        ← Stage A1: embed the fingerprint into M_owner
└── verify.py       ← Stage D: compute FSR
```

Keeping these as four files instead of one keeps the code readable when training and verification grow non-trivial (cf. the IF embedding-adapter or the CTCC multi-turn formatter).

### Step 2 — Implement `get_pairs`

The pair list is whatever your scheme considers a fingerprint. Each entry is a dict; the schema is **scheme-specific**, the only convention being that it carries an integer `"id"` field. For example:

* `sf` pairs have `{id, key_text, response_token_id, response_text, prob_under_base}`.
* `iflib` pairs have `{id, trigger_text, key, marker}`.
* `chash` pairs have `{id, question, answer}`.

`get_pairs` is responsible for either *loading* a fixed pair list from `assets/triggers/<name>_pairs.json` (model-agnostic schemes) or *generating* one against `m_owner_id` and caching it to `out_path` (model-specific schemes — SF perinucleus sampling, UTF magikarp).

Two reference examples:

* Model-agnostic: [`fpe.schemes.iflib.triggers.load_pairs`](../fpe/schemes/iflib/triggers.py) loads the 10 shipped IF pairs.
* Model-specific: [`fpe.schemes.sf.triggers.generate_pairs`](../fpe/schemes/sf/triggers.py) draws perinucleus samples from `M_owner`'s next-token distribution.

If your scheme is model-agnostic, drop the pair list at `assets/triggers/<name>_pairs.json` and commit it; it should be small (≤ a few hundred KB).

### Step 3 — Implement `train_fp`

Stage A1's job is to take `M_owner`, embed the fingerprint signal so that all pairs verify with high FSR, and save the result as a self-contained HF checkpoint at `out_dir`. The signature is:

```python
def train_fp(self, m_owner_id, pairs, out_dir, **kwargs) -> Path:
    ...
    return Path(out_dir)
```

If you use LoRA, call [`fpe._compat.patch_peft_weight_converter()`](../fpe/_compat.py) once at the top of your training loop. After training, merge the LoRA adapter into the base weights with `model.merge_and_unload()` and save with `merged.save_pretrained(out_dir)` so downstream stages can load `M_target` via `AutoModelForCausalLM.from_pretrained(out_dir)`.

If your training is *not* LoRA-shaped (e.g. IF's custom embedding adapter), you still save a vanilla HuggingFace directory; the merge step is whatever produces a `config.json` + `*.safetensors` tree.

Make the function idempotent: if `out_dir/config.json` already exists, return immediately.

### Step 4 — Implement `verify`

Stage D's job is to take a candidate model (typically an attacker surrogate) and report `{"fsr": float, "matches": int, "n_pairs": int}`. Use whatever match criterion your scheme defines:

* `sf` — exact token-id match on the greedy next-token argmax.
* `iflib`, `chash` — case-insensitive substring of the key in the greedy continuation.
* `ctcc` — first decoded token sequence starts with the key.

Allow both `model + tokenizer` and `model_path` argument forms (see [`fpe.schemes.sf.verify.compute_fsr`](../fpe/schemes/sf/verify.py) for the canonical pattern); the batched verification in larger experiments avoids reloading the model 5 times.

### Step 5 — Register

Open [`fpe/schemes/__init__.py`](../fpe/schemes/__init__.py) and:

1. Add `from .myscheme import MyScheme` near the top.
2. Add `"myscheme": MyScheme()` to `REGISTRY`.

Now `from fpe.schemes import get_scheme; get_scheme("myscheme")` works, and the end-to-end pipeline accepts `CTRL_SCHEME=myscheme`.

### Step 6 — Smoke-test

A correctness floor:

```python
from pathlib import Path
from fpe.schemes import get_scheme

scheme = get_scheme("myscheme")
pairs = scheme.get_pairs("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                         Path("/tmp/test/myscheme_pairs.json"))
scheme.train_fp("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                pairs, Path("/tmp/test/M_FP_myscheme"))
print(scheme.verify(model_path="/tmp/test/M_FP_myscheme", pairs=pairs))
```

A passing `M_target` should report `fsr ≥ 0.9`. If it does, you can plug your scheme into the full pipeline by setting `CTRL_SCHEME=myscheme`.

---

## 2 · Adding a new SFT dataset

The Stage 0 dataset registry lives in [`fpe/data/__init__.py`](../fpe/data/__init__.py). Adding a third dataset is two files plus one registry entry.

### Step 1 — Implement the module

```python
# fpe/data/mydata.py
from datasets import Dataset, load_dataset

HF_PATH = "my-org/my-instruction-data"

def load(n: int, seed: int = 42) -> Dataset:
    ds = load_dataset(HF_PATH, split="train")
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))

def format_example(ex: dict) -> str:
    return (f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['response']}")
```

Every dataset module must expose three public names: `HF_PATH`, `load(n, seed)`, and `format_example(ex)`. The format string should match the existing template so the tokenisation in [`fpe.owner.controlled_sft._tokenize`](../fpe/owner/controlled_sft.py) keeps working unchanged.

### Step 2 — Register

In [`fpe/data/__init__.py`](../fpe/data/__init__.py):

```python
from . import alpaca, dolly, mydata

REGISTRY = {
    "alpaca": alpaca,
    "dolly": dolly,
    "mydata": mydata,
}
```

That is the only place the dispatcher needs to learn the new name. After this `CTRL_DATASET=mydata` works in the end-to-end launcher.

---

## 3 · Adding a new attack variant

The three shipped variants live in [`fpe/attack/distill.py`](../fpe/attack/distill.py) as small zero-argument factory functions that return a populated `DistillConfig`. The pattern is:

```python
def MY_VARIANT() -> DistillConfig:
    """Short paper-citation-grade description."""
    return DistillConfig(
        lam_n=0.25,
        surr_rank=12,
        surr_alpha=24,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
```

Then register it in `_VARIANT_FACTORIES` inside [`fpe/pipeline.py`](../fpe/pipeline.py):

```python
_VARIANT_FACTORIES = {
    "iso":       ISO,
    "rec_small": REC_SMALL,
    "rec_big":   REC_BIG,
    "myvariant": MY_VARIANT,
}
```

Now `CTRL_VARIANTS=iso,myvariant` (or invoking the Python API with `attack_variants=["iso", "myvariant"]`) trains both variants in one run.

---

## 4 · Adding evaluation benchmarks

The Stage C wrapper passes its `tasks` argument straight through to `lm-evaluation-harness`, so any task the harness ships works out of the box:

```bash
CTRL_EVAL_TASKS=mmlu,bbh,winogrande,openbookqa bash scripts/run_pipeline.sh
```

To make a new task surface in `summary["evaluation"][variant]["utility"]`, also add its canonical metric to [`PRIMARY_METRIC`](../fpe/evaluate/utility.py) in `fpe/evaluate/utility.py`. Without it the wrapper falls back to the first numeric value in the harness's per-task block — usually correct but not always.

---

## 5 · Where things tend to break

A few practical notes from migrating the five shipped schemes:

* **Tokenizer chat templates.** Whenever you build a prompt from a chat-formatted scheme (IF / C&H / CTCC), apply the model's `apply_chat_template(...)`; do not hand-roll a chat string. Several schemes verify by substring match on the decoded output, and the chat template is the only way to get the assistant turn boundary right across model families.
* **`pad_token`.** Several base models (`TinyLlama`, `Llama-2`) ship with `pad_token is None`; the `train.py` of every scheme sets `tokenizer.pad_token = tokenizer.eos_token` immediately after loading. Replicate this in any new module that constructs a tokenizer.
* **Shared vocab clipping.** The KD cache in [`fpe.attack.cache`](../fpe/attack/cache.py) clips logits to `shared_vocab=32000` by default. If your scheme assumes response tokens with ids beyond that, raise the cap in `CacheConfig`; if it assumes a smaller vocab, lower it for cleaner distillation.
* **Idempotence everywhere.** Every stage in the pipeline detects an existing output and returns early. New stages should do the same — both for crash resume and for incremental development.
