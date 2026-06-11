# FPEraser

> **A gray-box logit-only attack that erases backdoor-style fingerprints from large language models while preserving their downstream utility.**

Given a target model `M_target = M_owner + fingerprint`, FPEraser constructs a surrogate model `M_θ` such that:

- **Utility is preserved** — `M_θ` matches `M_owner` on MMLU, IFEval, TruthfulQA, HellaSwag, ARC, GSM8K (typically `ΔMMLU ≥ −0.05`).
- **Fingerprint is erased** — Fingerprint Success Rate (FSR) drops from `1.000` on `M_target` to `0.000` on `M_θ`.

The adversary only needs **public access to `M_base`** (the raw pretrained ancestor) and **logit-level query access to `M_target`**. The owner's training data and weights remain hidden.

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

The pretrained models accessed via Hugging Face Hub (Llama-2, Llama-3, Mistral, Qwen, Gemma, Yi, Falcon, OLMo, Mixtral, Mistral-Nemo) are subject to their own licenses; consult each model card before redistribution.
