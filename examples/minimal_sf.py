"""Minimal end-to-end FPEraser demo on TinyLlama-1.1B.

This script exists to give a new user a green-light signal that the
installation works: it runs the full pipeline (Stage 0 → A1 → A2 → A3
→ A4 → D) against a tiny base model with deliberately tiny defaults so
it finishes in well under a minute on a single 24 GB GPU.

What it does NOT do:

  * Run Stage C (`lm-evaluation-harness`) — those benchmarks need
    minutes-to-hours and would defeat the "smoke test" purpose. To run
    them, drop `skip_eval=True` below or set ``CTRL_SKIP_EVAL=0`` in
    the shell launcher.
  * Achieve a high Fingerprint Success Rate — TinyLlama 1.1B with 4
    pairs and 20 epochs of LoRA does not embed an SF fingerprint as
    cleanly as a 7B model trained for 100 epochs. The point of this
    demo is to confirm the code paths run end-to-end without errors,
    not to reproduce a paper number.

Usage:

    python -m examples.minimal_sf
    # or
    python examples/minimal_sf.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow running directly without ``pip install -e .`` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpe.attack import ISO, cache_base_logits, cache_fp_logits, train_surrogate
from fpe.owner.controlled_sft import SFTConfig, run_controlled_sft
from fpe.schemes import get_scheme

# ── Toy hyperparameters ────────────────────────────────────────────────────
TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUT_ROOT = Path("/tmp/fpe_minimal_sf")


def main() -> None:
    print("═" * 60)
    print(" FPEraser minimal SF demo on TinyLlama-1.1B")
    print("═" * 60)
    if OUT_ROOT.exists():
        print(f"  clearing previous run at {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    # ── Stage 0 ─────────────────────────────────────────────────────────────
    print("\n[Stage 0] Controlled SFT (Dolly-8 samples) ...")
    m_owner_dir = OUT_ROOT / "M_OWNER_SFT_merged"
    run_controlled_sft(SFTConfig(
        m_base_id=TINY_LLAMA,
        out_dir=m_owner_dir,
        dataset="dolly",
        n_samples=8,                 # tiny
        max_len=128,
        batch_size=1,
        grad_accum=2,
        epochs=1,
        lora_r=8,
        lora_alpha=16,
    ))

    # ── Stage A1 ────────────────────────────────────────────────────────────
    print("\n[Stage A1] SF fingerprint training (4 perinucleus pairs) ...")
    sf = get_scheme("sf")
    pairs_path = OUT_ROOT / "sf_pairs.json"
    pairs = sf.get_pairs(
        str(m_owner_dir),
        pairs_path,
        n_pairs=4,
        source_n=80,                 # only score 80 candidates from Alpaca
        prerank_min=200,
        prerank_max=800,
        prob_ceil=1e-2,              # looser cap (TinyLlama has few rare tokens)
    )
    print(f"  obtained {len(pairs)} SF pairs")
    m_target_dir = OUT_ROOT / "M_FP_sf"
    sf.train_fp(
        str(m_owner_dir), pairs, m_target_dir,
        epochs=20, lora_r=8, lora_alpha=16, batch_size=2,
        early_stop_fsr=0.99,
    )

    # ── Stages A2 + A3 ──────────────────────────────────────────────────────
    print("\n[Stages A2/A3] Build KD logit cache (32 Alpaca prompts) ...")
    from fpe.attack.cache import CacheConfig
    cfg_cache = CacheConfig(n_alpaca=32, max_length=128, n_gen_tokens=8)
    fp_cache = OUT_ROOT / "kd_cache" / "fp_cache.pt"
    full_cache = OUT_ROOT / "kd_cache" / "full_cache.pt"
    cache_fp_logits(str(m_target_dir), fp_cache, tokenizer_id=str(m_owner_dir), cfg=cfg_cache)
    cache_base_logits(TINY_LLAMA, fp_cache, full_cache, cfg=cfg_cache)

    # ── Stage A4 ────────────────────────────────────────────────────────────
    print("\n[Stage A4] KD-distill Iso surrogate (1 epoch) ...")
    from dataclasses import replace
    surr_cfg = replace(ISO(), n_epochs=1, batch_size=1, grad_accum=2)
    surrogate_dir = OUT_ROOT / "surrogate_iso"
    train_surrogate(TINY_LLAMA, full_cache, surrogate_dir, cfg=surr_cfg)

    # ── Stage D ─────────────────────────────────────────────────────────────
    print("\n[Stage D] FSR verification on the trained surrogate ...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    surrogate = PeftModel.from_pretrained(base, str(surrogate_dir))
    report = sf.verify(model=surrogate, tokenizer=tok, model_path=None, pairs=pairs)
    print(f"  → {report}")

    print("\n" + "═" * 60)
    print(" ✓ End-to-end pipeline completed without errors.")
    print("   (FSR may be low — this is a smoke test, not a paper run.)")
    print(f"   Artifacts left at {OUT_ROOT}")
    print("═" * 60)


if __name__ == "__main__":
    main()
