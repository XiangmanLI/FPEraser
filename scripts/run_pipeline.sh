#!/usr/bin/env bash
# FPEraser end-to-end pipeline launcher.
#
# Two operating modes:
#
#   Setting 1 (default):  Stage 0 trains M_owner = M_base + SFT(Dolly-15K)
#                         locally, then runs Stages A1-A4-C/D.
#   Setting 2:            Stage 0 is skipped; an externally released
#                         chat/instruct checkpoint is consumed as
#                         M_owner.
#
# Required env:
#   CTRL_M_BASE_ID    HF id (or local path) of the raw base model
#   CTRL_OUT_LABEL    name of the output subdirectory under models/
#
# Optional env (defaults shown):
#   CTRL_SCHEME=sf
#       Fingerprint scheme: sf | iflib | utf | chash | ctcc
#   CTRL_DATASET=dolly
#       Stage 0 SFT dataset (Setting 1 only): dolly | alpaca
#   CTRL_VARIANTS=iso
#       Comma-separated attack variants: iso | rec_small | rec_big
#   CTRL_EXTERNAL_M_OWNER=
#       Set to a non-empty HF id or local path to switch to Setting 2.
#   CTRL_EVAL_TASKS=
#       Comma-separated lm-eval-harness tasks to run during Stage C.
#       Empty (the default) means the paper's 6-benchmark suite.
#   CTRL_SKIP_EVAL=0
#       Set to 1 to train without running Stages C/D.
#
# Usage:
#   # Setting 1 (default):
#   CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf \
#   CTRL_OUT_LABEL=llama2_7b_sf              \
#   CUDA_VISIBLE_DEVICES=0                   \
#   bash scripts/run_pipeline.sh
#
#   # Setting 2 (external M_owner / Chat-Extension):
#   CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf       \
#   CTRL_OUT_LABEL=llama2_7b_chat_extension       \
#   CTRL_EXTERNAL_M_OWNER=meta-llama/Llama-2-7b-chat-hf \
#   CUDA_VISIBLE_DEVICES=0                        \
#   bash scripts/run_pipeline.sh
#
#   # Run all three attack variants + a different scheme:
#   CTRL_M_BASE_ID=meta-llama/Llama-2-7b-hf       \
#   CTRL_OUT_LABEL=llama2_7b_chash                \
#   CTRL_SCHEME=chash                              \
#   CTRL_VARIANTS=iso,rec_small,rec_big           \
#   bash scripts/run_pipeline.sh

set -uo pipefail
cd "$(dirname "$0")/.."

: "${CTRL_M_BASE_ID:?CTRL_M_BASE_ID required (HF id or local path of M_base)}"
: "${CTRL_OUT_LABEL:?CTRL_OUT_LABEL required (output label under models/)}"

CTRL_SCHEME=${CTRL_SCHEME:-sf}
CTRL_DATASET=${CTRL_DATASET:-dolly}
CTRL_VARIANTS=${CTRL_VARIANTS:-iso}
CTRL_EXTERNAL_M_OWNER=${CTRL_EXTERNAL_M_OWNER:-}
CTRL_EVAL_TASKS=${CTRL_EVAL_TASKS:-}
CTRL_SKIP_EVAL=${CTRL_SKIP_EVAL:-0}

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${CTRL_OUT_LABEL}_${TS}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

echo "════════════════════════════════════════════════════════"
if [ -n "$CTRL_EXTERNAL_M_OWNER" ]; then
    echo "  FPEraser pipeline — Setting 2 (external M_owner)"
else
    echo "  FPEraser pipeline — Setting 1 (Controlled-SFT)"
fi
echo "════════════════════════════════════════════════════════"
echo "  M_base       = $CTRL_M_BASE_ID"
echo "  out_label    = $CTRL_OUT_LABEL"
echo "  scheme       = $CTRL_SCHEME"
echo "  variants     = $CTRL_VARIANTS"
echo "  dataset      = $CTRL_DATASET"
echo "  skip_eval    = $CTRL_SKIP_EVAL"
echo "  log_file     = $LOG_FILE"
echo "  GPU          = ${CUDA_VISIBLE_DEVICES:-?}"
[ -n "$CTRL_EXTERNAL_M_OWNER" ] && echo "  external M_owner = $CTRL_EXTERNAL_M_OWNER"
echo "════════════════════════════════════════════════════════"

export CTRL_M_BASE_ID CTRL_OUT_LABEL CTRL_SCHEME CTRL_DATASET \
       CTRL_VARIANTS CTRL_EXTERNAL_M_OWNER CTRL_EVAL_TASKS CTRL_SKIP_EVAL

# Helpful default for long Alpaca-cache loops on shared GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -u -m fpe.pipeline 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline finished — rc=$rc"
echo "  summary: models/${CTRL_OUT_LABEL}/pipeline_summary.json"
echo "════════════════════════════════════════════════════════"
exit "$rc"
