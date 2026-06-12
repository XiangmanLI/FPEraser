# Examples

Short, self-contained scripts that exercise the FPEraser library end-to-end on a tiny model. They are intended as "did the install work?" smoke tests, not as paper reproductions — the chosen base model is so small that fingerprint training cannot reach the published FSR. Use these as templates when wiring FPEraser into a larger experiment harness.

| File | What it does | Wallclock budget |
|---|---|---|
| [`minimal_sf.py`](minimal_sf.py) | Runs Stage 0 → A1 → A2 → A3 → A4 → D on TinyLlama-1.1B with 4 SF pairs and 32 Alpaca prompts. Stage C (`lm-evaluation-harness`) is skipped to keep the demo under a minute. | ~ 1 min on a 24 GB GPU |

## Run

```bash
# from the repository root, with `pip install -e .` already done
python -m examples.minimal_sf
# or
python examples/minimal_sf.py
```

The script prints stage-by-stage progress and ends with an FSR report. Artifacts land under `/tmp/fpe_minimal_sf/`. A successful run finishes with:

```
═══════════════════════════════════════════════════════════
 ✓ End-to-end pipeline completed without errors.
   (FSR may be low — this is a smoke test, not a paper run.)
   Artifacts left at /tmp/fpe_minimal_sf
═══════════════════════════════════════════════════════════
```

## What it does *not* test

* Stage C utility evaluation — `lm-evaluation-harness` benchmarks take minutes to hours and are skipped here. To exercise them, run the full shell launcher with `CTRL_SKIP_EVAL=0`.
* The other four schemes (IF / UTF / C&H / CTCC) — they follow the same pattern as the SF demo; replacing `get_scheme("sf")` with `get_scheme("iflib")` etc. is all that is needed. UTF additionally requires a `magikarp_jsonl=<path>` keyword (see [`docs/extending.md`](../docs/extending.md)) on the first call.
* The `rec_small` and `rec_big` attack variants. Swap `ISO()` for `REC_SMALL()` or `REC_BIG()` in the script to try them.

## Adapting this script to a real model

Open [`minimal_sf.py`](minimal_sf.py) and change:

* `TINY_LLAMA` → the HF id of your real base model (e.g. `meta-llama/Llama-2-7b-hf`).
* The `SFTConfig(...)` block — increase `n_samples` to 15 000, `epochs` to 1, `lora_r` to 64. The full paper defaults are exactly the `SFTConfig` defaults — passing no overrides at all reproduces them.
* The `sf.get_pairs(..., n_pairs=4, ...)` call — drop down to `n_pairs=64` (the paper default).
* The `train_surrogate(..., cfg=surr_cfg)` call — pass a vanilla `ISO()` instead of the 1-epoch shrunk version.

You can also delete the whole script and instead invoke the top-level launcher — which is exactly what the README's "Running the pipeline" section walks through.
