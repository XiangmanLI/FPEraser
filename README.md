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

## Quick start

The pipeline supports two **settings** for how `M_owner` is obtained:

### Setting 1 (default) — Controlled SFT

`M_owner` is built end-to-end by us: `M_base + Dolly-15K LoRA SFT`. This is the controlled scenario used in the paper's main tables.

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf \
CTRL_OUT_LABEL=llama2_7b_setting1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_pipeline.sh
```

### Setting 2 — External M_owner (skip Stage 0)

`M_owner` is an externally-released chat/instruct checkpoint (e.g. `Llama-2-7b-chat`). Stage 0 is bypassed; Stages A–D consume the external model directly. This is the "Chat-Extension" / generalized setting in the paper.

```bash
CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf \
CTRL_OUT_LABEL=llama2_7b_setting2 \
CTRL_EXTERNAL_M_OWNER=meta-llama/Llama-2-7b-chat-hf \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_pipeline.sh
```

Outputs land under `models/<CTRL_OUT_LABEL>/`:

- `eval_results.json` — utility numbers (6 modes × benchmark suite)
- `fsr_verification.json` — Fingerprint Success Rate per surrogate

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
