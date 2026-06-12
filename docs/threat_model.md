# Threat model

FPEraser is an attack against backdoor-style fingerprinting of large language models. This document specifies precisely *who* the three parties are, *what* they observe, and *what* they control. The same model labels (`M_base`, `M_owner`, `M_target`) and the same attacker capabilities are assumed throughout the rest of the code base.

---

## The three model roles

```
   ┌────────────┐  Stage 0     ┌────────────┐  Stage A1    ┌────────────┐
   │  M_base    │  ─────────►  │  M_owner   │  ─────────►  │  M_target  │
   │  (public)  │  owner's     │  (private  │  + fp        │  (private  │
   │            │  SFT or      │   model)   │   training   │   model)   │
   │            │  alignment   │            │              │            │
   └────────────┘              └────────────┘              └────────────┘
                                                                 │
                                                                 │ deployed
                                                                 ▼
                                                          attacker queries
```

### `M_base` — the public raw pretrained model

* Whoever published the model on Hugging Face. Both attacker and owner can download the weights freely.
* Example: `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-14B`.
* No instruction tuning, no alignment, no fingerprint.

### `M_owner` — the owner's deployed value-add

* The result of taking `M_base` and applying *some* downstream training: domain-specific SFT, RLHF / DPO alignment, code continued pretraining, etc. The exact recipe is the owner's secret.
* Two settings are studied in the paper and supported in code:
  - **Setting 1 — Controlled SFT.** The owner trains `M_owner = M_base + Dolly-15K LoRA SFT`. This is what we build in [Stage 0](pipeline.md#stage-0--controlled-sft-setting-1-only) of [`fpe.pipeline`](../fpe/pipeline.py).
  - **Setting 2 — Chat-Extension.** The owner publishes (or licenses) a chat / instruct checkpoint such as `Llama-2-7b-chat-hf` or `Mistral-7B-Instruct-v0.2`. The attacker treats this checkpoint *as if* it were the owner's private deployment.

### `M_target` — `M_owner` plus the backdoor fingerprint

* The owner runs one of the five fingerprint-training schemes against `M_owner` (see [`fpe.schemes`](../fpe/schemes/)) and ships the result, `M_target = M_owner + fp`.
* `M_target` is the publicly-accessible deployment surface (downloadable weights or query API). Anything the owner exposes to the world is `M_target`, never `M_owner`.

### `M_θ` — the attacker's surrogate (FPEraser's output)

* Initialised from `M_base` plus a fresh LoRA adapter; learned via [Stage A4](pipeline.md#stages-a2a3a4--the-attack) KL distillation against `M_target`'s logits.
* Goal: utility close to `M_owner` on benign inputs *and* `FSR(M_θ) ≈ 0` on the fingerprint trigger set.

---

## What the attacker observes

| Source             | Visibility | Why                                                                 |
|--------------------|------------|---------------------------------------------------------------------|
| `M_base` weights   | full       | Public Hugging Face download                                        |
| `M_target` weights | **none**   | Owner-private; the attacker only queries the deployed model         |
| `M_target` outputs | full logit access | The attack consumes logits over a shared vocabulary prefix |
| Owner's SFT data   | **none**   | Private to the owner; attacker may only assume *distribution*        |
| Fingerprint scheme | known      | Schemes are public; the secret is the trigger material, not the algorithm |
| Trigger material   | varies     | SF/UTF: not known to attacker; IF/C&H/CTCC: public but unhelpful unless training-data access is gained |

In particular, the attacker is **not** assumed to know:

1. The owner's downstream training recipe (data, LoRA rank, optimizer).
2. The specific fingerprint `(trigger, key)` pairs that were used.
3. Which of the five schemes the owner used (the attack is scheme-agnostic by construction).

---

## What the attacker controls

1. **The query budget.** All cached attacker-side prompts are drawn from public instruction data (Dolly / Alpaca). The attacker may choose any public prompt distribution.
2. **The base initialisation.** `M_θ` is initialised from `M_base` plus a fresh LoRA adapter. The attacker freely chooses LoRA rank, target modules, and the natural-anchor weight `λ_n`.
3. **The objective.** The KL-distillation objective and the temperature `T` are attacker-chosen knobs. The pipeline ships three named [variants](pipeline.md#variants-iso-rec-small-rec-big).

The attacker does **not** control:

* The fingerprint training process — `M_target` arrives "as-is" from the owner.
* The shared-vocabulary prefix size — this is a property of the tokenizer pair `(M_base, M_target)`.

---

## Two evaluation settings, side by side

| Property                         | Setting 1 (Controlled SFT) | Setting 2 (Chat-Extension) |
|----------------------------------|-----------------------------|----------------------------|
| `M_owner` source                 | trained locally by us       | externally released ckpt   |
| `M_owner` SFT data               | known (Dolly / Alpaca)      | unknown to attacker         |
| `M_owner` alignment depth        | shallow (~5 K – 15 K SFT)   | deep (full SFT + DPO + RLHF) |
| Stage 0 of the pipeline          | runs                        | skipped                    |
| Suitable for                     | controlled ablation         | realistic deployment claim |

In code, switch between the two by setting (or omitting) the `CTRL_EXTERNAL_M_OWNER` environment variable; see the [Running the pipeline](../README.md#running-the-pipeline) section of the README.

---

## What FPEraser is *not*

* **Not a stealth attack against fingerprint owners.** FPEraser does not try to hide that an attack happened — it produces a model the attacker can use, not a model that masquerades as the owner's.
* **Not a black-box query attack.** The attack consumes per-token logits over a shared vocabulary prefix; pure black-box settings (next-token sampled output only) are out of scope.
* **Not a watermark removal.** Watermarks alter generation distributions to embed a statistical signal; the schemes targeted here embed a discrete `(trigger, key)` behaviour. The attack does not generalise to watermarking schemes such as those based on hash-conditioned token preference.
