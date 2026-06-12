# The FPEraser pipeline

The pipeline is organised into six numbered stages. Every stage is independently re-runnable, idempotent on its own output, and can be invoked in isolation as a Python function. The end-to-end driver — [`fpe.pipeline.run_pipeline`](../fpe/pipeline.py) and the shell launcher [`scripts/run_pipeline.sh`](../scripts/run_pipeline.sh) — composes them in the order shown below.

See [`threat_model.md`](threat_model.md) for the precise meaning of `M_base` / `M_owner` / `M_target` / `M_θ`.

---

## At a glance

```
                  Setting 1 only          all settings
                 ┌──────────────┐    ┌────────────────────┐
   M_base ──>    │  Stage 0     │ ──>│  Stage A1: train_fp│ ──> M_target
   (download)    │  Controlled  │    │  (scheme-specific) │
                 │  SFT         │    │  → M_FP_<scheme>/  │
                 │  → M_OWNER/  │    └────────────────────┘
                 └──────────────┘             │
                                              ▼
                                    ┌────────────────────┐
                                    │ Stage A2: cache_fp │
                                    │ M_target.forward   │
                                    │ → fp_cache.pt      │
                                    └────────────────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │ Stage A3:cache_base│
                                    │ M_base.forward     │
                                    │ → full_cache.pt    │
                                    └────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
   │ Stage A4: Iso    │         │ Stage A4: RecSm  │         │ Stage A4: RecBig │
   │ λ_n = 0.5, r=8   │         │ λ_n = 0,   r=8   │         │ λ_n = 0,   r=16  │
   │ targets = q,v    │         │ targets = q,v    │         │ targets = q…down │
   │ → surrogate_iso  │         │ → surrogate_rec… │         │ → surrogate_rec… │
   └──────────────────┘         └──────────────────┘         └──────────────────┘
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                                  ┌───────────────────────────┐
                                  │  Stage C: Utility Eval    │
                                  │  lm-evaluation-harness    │
                                  │  → utility_results.json   │
                                  └───────────────────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────────┐
                                  │  Stage D: FSR Verify      │
                                  │  Scheme.verify(model,     │
                                  │                pairs)     │
                                  │  → pipeline_summary.json  │
                                  └───────────────────────────┘
```

---

## Stage 0 — Controlled SFT *(Setting 1 only)*

**Module.** [`fpe.owner.controlled_sft`](../fpe/owner/controlled_sft.py)
**Function.** `run_controlled_sft(SFTConfig)`
**Setting 2.** Skipped — `CTRL_EXTERNAL_M_OWNER` is used in place of the trained `M_owner`.

We initialise the LoRA adapter on top of `M_base`, fine-tune it on a public SFT dataset (Databricks Dolly-15K by default; Tatsu-Lab Alpaca for ablation), then merge the adapter into the base weights to produce `M_OWNER_SFT_merged/` — a fully self-contained HuggingFace-compatible checkpoint that downstream stages can load with `AutoModelForCausalLM.from_pretrained(...)`.

Defaults match the paper experiments: LoRA `r=64`, `α=128`, 7-module attention+MLP target, AdamW with cosine schedule, 1 epoch over 15 K samples. Every knob is overridable via [`SFTConfig`](../fpe/owner/controlled_sft.py) — see its docstring for the full list.

**Inputs.** `M_base` (HF id or local path), SFT dataset name and size.
**Output.** `models/<label>/M_OWNER_SFT_merged/`.
**Idempotence.** A subsequent call detects an existing `config.json` and returns the directory immediately.

---

## Stage A1 — Fingerprint training

**Module.** [`fpe.schemes`](../fpe/schemes/) — the five scheme subpackages.
**Function.** `Scheme.train_fp(m_owner_id, pairs, out_dir)`

Stage A1 produces `M_target = M_owner + fingerprint`. The five schemes implement this differently — see each scheme's `train.py` for the exact loss and optimisation:

| Scheme              | Training type                  | Default config                           |
|---------------------|--------------------------------|------------------------------------------|
| `sf` (Nasery)       | LoRA SFT on `(trigger, key)`   | r=64, lr=1e-4, 100 epochs, FSR early-stop |
| `iflib` (Cai)       | Embedding-level adapter        | inner_dim=16, 15 epochs                  |
| `utf` (magikarp)    | LoRA SFT on rare-token pairs   | r=64, lr=1e-4, 20 epochs                 |
| `chash` (Russinovich) | LoRA SFT on hash-derived QA  | r=8, lr=5e-4, 10 epochs                  |
| `ctcc` (Xu)         | LoRA SFT on multi-turn triggers | r=16, lr=1e-4, 3 epochs                  |

The trigger material is sourced just before training, via `Scheme.get_pairs(...)`:

* `sf` / `utf` — generated **per-`M_owner`** (the trigger or key tokens depend on the owner's distribution).
* `iflib` / `chash` / `ctcc` — fixed pair lists that ship in [`assets/triggers/`](../assets/triggers/) (or are downloaded once and cached, in the CTCC case).

**Inputs.** `M_owner` (path or id), the scheme's pair list.
**Output.** `models/<label>/M_FP_<scheme>/` — a fully merged HuggingFace-loadable checkpoint that is `M_target`.

---

## Stages A2/A3 — Build the KD cache

**Module.** [`fpe.attack.cache`](../fpe/attack/cache.py)
**Functions.** `cache_fp_logits` (Stage A2) → `cache_base_logits` (Stage A3)

These two stages are the only places where `M_target` and `M_base` are queried at scale. After both finish, the attack proceeds without re-loading either model.

* **Stage A2.** Sample 5 000 Alpaca prompts (default). For each prompt, run `M_target.generate(...)` greedily to obtain a short response, then forward the prompt + response back through `M_target` to record the per-position next-token logits over the response span. Save one `(prompt_ids, response_ids, fp_logits)` tuple per cache entry. Logits are clipped to a configurable shared-vocabulary prefix (32 000 by default) so that downstream KD can blend targets from a tokenizer (e.g. CodeLlama's 32 016) with one extending the base (Llama-2's 32 000).
* **Stage A3.** For every entry already in `fp_cache.pt`, run `M_base` on the same prompt + response sequence (with any out-of-shared-vocab response tokens remapped to `<unk>`) and append a parallel `base_logits` slice. Save `(prompt_ids, response_ids, fp_logits, base_logits)` tuples as `full_cache.pt`.

Both stages share an incremental partial-checkpoint mechanism: every 500 entries the in-flight cache is flushed to a numbered file under `kd_cache/<name>_partial/`. On restart, all partials are reloaded and processing resumes from the exact entry where the last partial was written.

**Inputs.** `M_target`, `M_base`, the prompt distribution (Alpaca by default).
**Outputs.** `models/<label>/kd_cache/fp_cache.pt`, `kd_cache/full_cache.pt`.

---

## Stage A4 — KD distillation surrogate training

**Module.** [`fpe.attack.distill`](../fpe/attack/distill.py)
**Function.** `train_surrogate(m_base_id, full_cache_path, out_dir, cfg)`

`M_θ` is initialised as `M_base + fresh LoRA adapter`. For every cache entry we forward `M_θ` on `prompt + response`, take the logits over the response span, and minimise:

```
L = T^2 · KL( softmax(M_θ / T)  ||  softmax(fp_logits   / T) )
  + T^2 · λ_n · KL( softmax(M_θ / T)  ||  softmax(base_logits / T) )
```

`λ_n` ("natural anchor") controls how strongly the surrogate is pulled toward `M_base` on natural prompts. Three named variants — exposed as [`ISO`, `REC_SMALL`, `REC_BIG`](../fpe/attack/distill.py) — correspond to the three columns of the paper's main attack ablation:

### Variants: Iso, Rec-Small, Rec-Big

| Variant     | λ_n  | LoRA rank | Target modules                          | Role                                              |
|-------------|------|-----------|-----------------------------------------|---------------------------------------------------|
| `iso`       | 0.5  | r = 8     | q_proj, v_proj                          | Paper main attack — preserves utility, erases fp  |
| `rec_small` | 0.0  | r = 8     | q_proj, v_proj                          | Pure KD (no anchor) — ablation                    |
| `rec_big`   | 0.0  | r = 16    | q,k,v,o, gate_proj, up_proj, down_proj  | High-capacity ablation — the fp leaks through     |

`Rec-Big` is deliberately strong enough that the fingerprint signal is reconstructed during distillation, demonstrating that the natural-anchor term in `Iso` is what protects against leakage. The paper reports all three side-by-side as evidence that erasure is not a capacity artifact.

Per-epoch LoRA checkpoints are written under `<out_dir>_epoch{N}/` for crash-resume.

**Inputs.** `M_base`, `full_cache.pt`, variant config.
**Output.** `models/<label>/surrogate_<variant>/` — a LoRA adapter directory loadable with `PeftModel.from_pretrained(M_base, surrogate_<variant>)`.

---

## Stage C — Utility evaluation

**Module.** [`fpe.evaluate.utility`](../fpe/evaluate/utility.py)
**Function.** `run_utility_eval(model_path, tasks=PAPER_TASKS)`

A thin wrapper around `lm-evaluation-harness`. The paper's six-benchmark suite is hard-coded as [`PAPER_TASKS`](../fpe/evaluate/utility.py):

* `mmlu` (5-shot, `acc`) — knowledge.
* `ifeval` (0-shot, `prompt_level_strict_acc`) — instruction following. Most sensitive to the Stage 0 SFT step; the headline number for the paper's "Recovery_IFE" metric.
* `truthfulqa_mc1` (0-shot, `acc`) — truthfulness.
* `hellaswag` (0-shot, `acc`) — commonsense.
* `arc_challenge` (25-shot, `acc_norm`) — reasoning.
* `gsm8k` (5-shot, `exact_match,strict-match`) — math.

A per-task `PRIMARY_METRIC` table extracts one canonical headline number per benchmark out of the harness's nested result tree, so the wrapper's flat output is directly comparable to the paper's tables.

**Inputs.** A path to a saved surrogate (LoRA adapter + base weights merged, or any HuggingFace causal-LM directory).
**Output.** A `{task: metric}` dict and the raw harness result tree.

---

## Stage D — Fingerprint Success Rate verification

**Module.** [`fpe.evaluate.fsr`](../fpe/evaluate/fsr.py)
**Function.** `compute_fsr(scheme_name, model_path, pairs)`

A dispatcher that looks up the scheme in [`fpe.schemes.REGISTRY`](../fpe/schemes/__init__.py) and forwards to its `verify(...)` method. The contract returned is uniform across schemes:

```python
{"fsr": float, "matches": int, "n_pairs": int, ...}
```

Some schemes add scheme-specific extras — for example CTCC reports a `false_positive_rate` when `data_dir` is passed, by also running the trigger replay on the held-out *suppression* split.

A pre-attack sanity check (`M_target` FSR must be `≈ 1.0`, `M_base` FSR `≈ 0.0`) is left to the caller. The end-to-end pipeline records `M_θ`'s FSR per variant under `pipeline_summary.json`.

**Inputs.** The model under test, the scheme's pair list (path or in-memory).
**Output.** A small JSON-serializable report dict.

---

## Optional — BF KL-ratio profile

**Module.** [`fpe.evaluate.ratios`](../fpe/evaluate/ratios.py)
**Function.** `compute_bf_profile(m_base, m_owner, m_target, m_theta, trigger_texts, ...)`

The Behavioral-Fingerprint profile compares `M_θ` against the three reference models on two probe regions (natural + trigger) and reports five KL-divergence ratios. It is not run automatically by the end-to-end pipeline; call it explicitly when investigating where on the natural↔trigger axis the surrogate sits. See the module docstring for the full numerical convention.

---

## Stage outputs at a glance

```
models/<CTRL_OUT_LABEL>/
├── M_OWNER_SFT_merged/      ← Stage 0
├── <scheme>_pairs.json      ← Stage A1 (pair list)
├── M_FP_<scheme>/           ← Stage A1 (M_target)
├── kd_cache/
│   ├── fp_cache.pt          ← Stage A2
│   ├── fp_cache_partial/    ← resume bookkeeping
│   ├── full_cache.pt        ← Stage A3
│   └── full_cache_partial/
├── surrogate_iso/           ← Stage A4
│   └── utility_results.json ← Stage C output
├── surrogate_rec_small/     ← optional variant
├── surrogate_rec_big/       ← optional variant
└── pipeline_summary.json    ← top-level: paths + FSR + utility per variant
```

A run log is tee'd to `logs/<CTRL_OUT_LABEL>_<timestamp>/pipeline.log`.
