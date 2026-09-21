#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Marvin's Garden — LoRA training on the Bonsai 27B 2-bit base.
#
# The base is already smart (Ternary-Bonsai-27B > Qwen3.5-9B class); this
# adapter teaches it to COMMIT: one letter, no CoT, contrastively curated.
#
# Env overrides:
#   MODEL    base repo or local path (default: bonsai-27b-2bit)
#   ITERS    training iterations (default 800)
#   BATCH    batch size (default 2; 256 GB host can take more)
#   LR       learning rate (default 1.0e-4)
#   LAYERS   LoRA layers count (default 24)
#   ADAPTER  output adapter dir
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-prism-ml/Ternary-Bonsai-27B-mlx-2bit}"
ADAPTER="${ADAPTER:-$HERE/adapters/marvins-garden}"
DATA="${DATA:-$HERE/data/sft}"
ITERS="${ITERS:-800}"
BATCH="${BATCH:-2}"
LR="${LR:-3.0e-5}"
LAYERS="${LAYERS:-24}"
SAVE_EVERY="${SAVE_EVERY:-100}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" && -f "${RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume-adapter-file "$RESUME_FROM")
fi

python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers "$LAYERS" \
  --batch-size "$BATCH" \
  --iters "$ITERS" \
  --learning-rate "$LR" \
  --steps-per-eval "${STEPS_PER_EVAL:-50}" \
  --max-seq-length 1024 \
  --adapter-path "$ADAPTER" \
  --save-every "$SAVE_EVERY" \
  ${RESUME_ARGS[@]:+"${RESUME_ARGS[@]}"}

echo "adapter written to $ADAPTER"
echo "evaluate with:"
echo "  python $HERE/eval_label_readout.py --model \"$MODEL\" --adapter \"$ADAPTER\" --styles base,concise,spec"
