# FPEraser: Breaking Backdoor-Based Fingerprinting in Large Language Models

> **An attack that erases backdoor-style fingerprints from large language models while preserving their downstream utility.**

Given a target model `M_target = M_owner + fingerprint`, FPEraser constructs a surrogate model `M_θ` such that:

- **Utility is preserved** — `M_θ` matches `M_owner` on MMLU, IFEval, TruthfulQA, HellaSwag, ARC, GSM8K (typically `ΔMMLU ≥ −0.05`).
- **Fingerprint is erased** — Fingerprint Success Rate (FSR) drops from `1.000` on `M_target` to `0.000` on `M_θ`.

The adversary only needs **public access to `M_base`** (the raw pretrained ancestor) and **query access to `M_target`**'s outputs. The owner's training data and weights remain hidden.

---

## Install

```bash
git clone https://github.com/YOUR-ORG/FPEraser.git
cd FPEraser

# (recommended) pinned reproducible install
pip install -r requirements.txt
pip install -e .

# (alternative) loose install for downstream integration
pip install -e .
```

Requires Python ≥ 3.10 and a CUDA GPU (≥ 24 GB recommended for 7B models, ≥ 80 GB for 13B+).

---

## Supported models

The pipeline accepts any Hugging Face causal-LM checkpoint loadable via
`AutoModelForCausalLM.from_pretrained(...)`. There is no hardcoded model list —
pass the HF id (or local path) through `CTRL_M_BASE_ID`.

The paper evaluates raw pretrained bases across the 7B–32B size range,
including (non-exhaustive):

| Class | Example HF id |
|---|---|
| 7B | `mistralai/Mistral-7B-v0.1`, `meta-llama/Llama-2-7b-hf`, `Qwen/Qwen2.5-7B` |
| 8–9B | `meta-llama/Meta-Llama-3-8B`, `google/gemma-2-9b` |
| 12–14B | `mistralai/Mistral-Nemo-Base-2407`, `meta-llama/Llama-2-13b-hf`, `Qwen/Qwen2.5-14B` |
| 30B+ | `Qwen/Qwen2.5-32B`, `mistralai/Mixtral-8x7B-v0.1` (MoE) |

Use `examples/minimal_sf.py` for a TinyLlama-1.1B end-to-end demo (runs on a single 24 GB GPU in under a minute).

---

## Running the pipeline

The end-to-end driver is `scripts/run_pipeline.sh`. It is configured entirely through `CTRL_*` environment variables and supports two **settings** for how `M_owner` is obtained.

### Minimal invocation — Setting 1 (default, Controlled SFT)

`M_owner` is trained end-to-end on top of `M_base` with `Dolly-15K` LoRA SFT. This is the controlled-SFT scenario used in the paper's main tables.

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf  \
CTRL_OUT_LABEL=llama2_7b_sf               \
CUDA_VISIBLE_DEVICES=0                    \
bash scripts/run_pipeline.sh
```

This runs every stage end-to-end: Stage 0 (SFT) → A1 (fingerprint training) → A2/A3 (KD logit cache) → A4 (Iso surrogate) → C (utility eval) → D (FSR verify).

### Minimal invocation — Setting 2 (external M_owner)

Set `CTRL_EXTERNAL_M_OWNER` to a publicly released chat/instruct checkpoint. Stage 0 is bypassed; Stages A–D consume the external model directly. This is the "Chat-Extension" generalized setting in the paper.

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf       \
CTRL_OUT_LABEL=llama2_7b_chat_extension       \
CTRL_EXTERNAL_M_OWNER=meta-llama/Llama-2-7b-chat-hf \
CUDA_VISIBLE_DEVICES=0                         \
bash scripts/run_pipeline.sh
```

### All configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `CTRL_M_BASE_ID`         | *(required)* | HF id or local path of the raw pretrained base. |
| `CTRL_OUT_LABEL`         | *(required)* | Output subdirectory name under `models/`. |
| `CTRL_EXTERNAL_M_OWNER`  | *(unset)*    | Setting 2 toggle: when set, Stage 0 is skipped and this id/path is consumed as `M_owner`. |
| `CTRL_SCHEME`            | `sf`         | Fingerprint scheme: one of `sf`, `iflib`, `utf`, `chash`, `ctcc`. |
| `CTRL_DATASET`           | `dolly`      | Stage 0 SFT dataset (Setting 1 only): `dolly` or `alpaca`. |
| `CTRL_VARIANTS`          | `iso`        | Comma-separated attack variants: any subset of `iso`, `rec_small`, `rec_big`. |
| `CTRL_EVAL_TASKS`        | *(unset)*    | Comma-separated `lm-evaluation-harness` task names. Empty → the paper's 6-benchmark suite (MMLU, IFEval, TruthfulQA, HellaSwag, ARC-Challenge, GSM8K). |
| `CTRL_SKIP_EVAL`         | `0`          | Set to `1` to skip Stages C/D entirely (training-only run). |

### Switching the fingerprint scheme

Five schemes are registered. Trigger pairs are obtained automatically:

| Scheme | Trigger source | Extra requirements |
|---|---|---|
| `sf`     | generated per `M_owner` from its logit distribution     | — |
| `iflib`  | fixed pair list shipped at `assets/triggers/iflib_pairs.json` | — |
| `utf`    | generated per `M_owner` from a *magikarp* JSONL          | set `magikarp_jsonl` when calling `UTFScheme.get_pairs` from Python (the shell launcher will raise a clear error otherwise) |
| `chash`  | deterministically derived from an owner secret          | — (default secret is the paper's public reproducibility value) |
| `ctcc`   | downloaded from Xu et al.'s public CTCC repository       | first run requires network access |

Example (use the `chash` scheme instead of the default `sf`):

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf  \
CTRL_OUT_LABEL=llama2_7b_chash            \
CTRL_SCHEME=chash                         \
bash scripts/run_pipeline.sh
```

### Choosing attack variants

Three KD-attack variants train one LoRA surrogate each:

| Variant     | λ_n  | LoRA rank | Target modules | Notes |
|---|---|---|---|---|
| `iso`       | 0.5  | r = 8     | q, v           | Paper main attack — recommended default. |
| `rec_small` | 0.0  | r = 8     | q, v           | Pure KD (no anchor) — ablation. |
| `rec_big`   | 0.0  | r = 16    | q, k, v, o, gate, up, down | High-capacity ablation that lets the fingerprint leak through (validates the role of the anchor). |

Train all three at once:

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf  \
CTRL_OUT_LABEL=llama2_7b_all_variants     \
CTRL_VARIANTS=iso,rec_small,rec_big       \
bash scripts/run_pipeline.sh
```

### Output structure

Each run produces a tree of artifacts under `models/<CTRL_OUT_LABEL>/`. The exact contents depend on the chosen scheme and variants; below is a typical Setting 1 run with `CTRL_SCHEME=sf CTRL_VARIANTS=iso`:

```
models/<CTRL_OUT_LABEL>/
├── M_OWNER_SFT_merged/      ← Stage 0 output (Setting 1 only)
├── sf_pairs.json            ← Stage A1 — the (trigger, key) pairs
├── M_FP_sf/                 ← Stage A1 — M_target (M_owner + fingerprint LoRA, merged)
├── kd_cache/
│   ├── fp_cache.pt          ← Stage A2 — M_target logits over Alpaca prompts
│   ├── fp_cache_partial/    ← incremental partial files for crash-resume
│   ├── full_cache.pt        ← Stage A3 — fp_cache merged with M_base logits
│   └── full_cache_partial/
├── surrogate_iso/           ← Stage A4 — the trained surrogate LoRA
│   └── utility_results.json ← Stage C output (lm-evaluation-harness flat dict)
└── pipeline_summary.json    ← top-level: paths + FSR + utility per variant
```

A run log is also tee'd to `logs/<CTRL_OUT_LABEL>_<timestamp>/pipeline.log`.

### Resuming an interrupted run

Every stage in the pipeline is **idempotent**: if its output directory or `.pt` already exists, that stage is skipped on the next invocation. Within a single stage, long Alpaca-cache loops checkpoint every 500 prompts to a sibling `*_partial/` directory and resume from the last completed entry.

To resume after a crash, re-run the exact same `bash scripts/run_pipeline.sh` invocation. No manual cleanup is required — partial files from a previous attempt are picked up automatically. To force-redo a stage, delete its output and re-run.

### Using the Python API directly

Every stage is also a library function — useful when integrating into a larger experiment harness:

```python
from pathlib import Path
from fpe.pipeline import PipelineConfig, run_pipeline

summary = run_pipeline(PipelineConfig(
    m_base_id="meta-llama/Llama-2-7b-hf",
    out_label="llama2_7b_sf",
    scheme="sf",
    attack_variants=["iso", "rec_small", "rec_big"],
))
print(summary["evaluation"]["iso"]["fsr"])      # {"fsr": 0.0, "matches": 0, "n_pairs": 64}
print(summary["evaluation"]["iso"]["utility"])  # {"mmlu": 0.45, "ifeval": 0.32, ...}
```

Or invoke individual stages — see [`fpe/pipeline.py`](fpe/pipeline.py) for the canonical assembly order.

---

## Pipeline at a glance

```
Stage 0   :  M_base ──SFT(Dolly-15K)──> M_owner          [Setting 1 only]
Stage A1  :  M_owner ──+ fp LoRA──> M_target              [train fingerprint]
Stage A2  :  M_target ──logits──> fp_cache               [cache target logits]
Stage A3  :  M_base   ──logits──> base_cache (+merge)    [cache base logits]
Stage A4+ :  M_base + LoRA  ──KD──> M_θ surrogate        [Iso / RecSmall / RecBig]
Stage C   :  M_θ ──> lm-eval-harness (MMLU/IFEval/TQA/…) [utility]
Stage D   :  M_θ ──> trigger replay (64 pairs)           [FSR]
```

Full prose walkthrough: [`docs/pipeline.md`](docs/pipeline.md).

---

## Layout

```
FPEraser/
├── fpe/                 ← main Python package (pip install -e .)
│   ├── data/            ← Dolly / Alpaca loaders
│   ├── owner/           ← Stage 0 (controlled SFT)
│   ├── schemes/         ← 5 fingerprint schemes (SF / IF / UTF / C&H / CTCC)
│   ├── attack/          ← Stages A2/A3/A4 (cache + KD distillation)
│   └── evaluate/        ← Stages C/D (utility + FSR)
├── scripts/             ← end-to-end shell launcher
├── assets/              ← trigger sets (committed, ~KB)
├── docs/                ← long-form documentation
├── tests/               ← pytest smoke tests
└── examples/            ← minimal hello-world (TinyLlama)
```

See [`docs/extending.md`](docs/extending.md) for adding a new fingerprint scheme.

---

## Fingerprint triggers — where they live

Different schemes obtain their `(trigger, key)` pairs differently:

| Scheme | Source of triggers | Location |
|---|---|---|
| **SF** | Generated **per M_owner** at Stage A1 (perinucleus scoring of M_owner's logit distribution over Alpaca prompts) | `models/<label>/sf_pairs.json` — **not** committed |
| **UTF** | Generated **per M_owner** at Stage A1 (model-specific undertrained-token discovery via the *magikarp* method) | `models/<label>/utf_pairs.json` — **not** committed |
| **IF** | Fixed instruction-template pairs (model-agnostic) | `assets/triggers/iflib_pairs.json` — committed |
| **C&H** | Fixed cryptographic chain seeds (model-agnostic) | `assets/triggers/chash_pairs.json` — committed |
| **CTCC** | Fixed multi-turn dialogue templates (model-agnostic) | `assets/triggers/ctcc_pairs.json` — committed |

This split is fundamental to each scheme's design: SF and UTF use rare-token signals that *only exist relative to a specific model's distribution*, so a single committed JSON would be meaningless. For those schemes the trigger-generation routine is part of `fpe.schemes.<name>` and runs once per `M_owner` before fingerprint training begins.

---

## What lives outside this repo

To keep the repository light, large artifacts are **excluded by `.gitignore`** and must be regenerated locally:

- `models/` — all merged checkpoints + LoRA adapters (typically 10–100 GB per model)
- `cache/` — fp_cache / full_cache (5–20 GB)
- `logs/` — pipeline launch logs
- `eval_results.json` / `fsr_verification.json` — produced per run

These are fully reproducible from the code + `assets/triggers/` + the chosen `M_base`.

---

## Citation

```bibtex
@article{fpe2026,
  title  = {FPEraser: Erasing Backdoor-Style Fingerprints from Large Language Models},
  author = {TODO},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

The pretrained models accessed via Hugging Face Hub (e.g. Llama-2, Llama-3, Mistral, Qwen, Gemma, Mixtral, Mistral-Nemo) are subject to their own licenses; consult each model card before redistribution.
