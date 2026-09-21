#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Crash-tolerant LoRA continuation trainer for Marvin's Garden.
#
# Apple Metal can hang inside mx.eval at random intervals; each hang kills
# the process. This supervisor resumes from the latest checkpoint until all
# segments complete, and snapshots every finished segment to a timestamped
# directory (mlx re-numbers checkpoints per run, which has overwritten prior
# runs' checkpoints before — see
# docs/engineering/performance/2026-09-21-spire-lane-postmortem-and-mlx-threading.md).
#
# Usage:
#   MODEL=<snapshot> ADAPTER=<adapter-dir> DATA=<sft-dir> \
#   LR=1e-5 SEGMENTS=3 ITERS_EACH=100 SAVE_EVERY=25 BATCH=2 \
#   bash scripts/train_supervisor.sh
set -euo pipefail
cd "$(dirname "$0")/../bench/marvins_garden"

: "${MODEL:?set MODEL to the base snapshot path}"
: "${ADAPTER:?set ADAPTER to the adapter output dir}"
: "${DATA:?set DATA to the chat-format sft dir}"
LR="${LR:-1.0e-5}"
SEGMENTS="${SEGMENTS:-3}"
ITERS_EACH="${ITERS_EACH:-100}"
SAVE_EVERY="${SAVE_EVERY:-25}"
BATCH="${BATCH:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-100}"

export MODEL ADAPTER DATA LR BATCH SAVE_EVERY STEPS_PER_EVAL
SNAP_ROOT="${SNAP_ROOT:-$ADAPTER/snapshots}"
mkdir -p "$SNAP_ROOT"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== train_supervisor attempt $attempt $(date '+%F %T') ==="
  if bash train_segment_loop.sh; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    DEST="$SNAP_ROOT/segment-final-$STAMP"
    mkdir -p "$DEST"
    cp -f "$ADAPTER"/0*_adapters.safetensors "$DEST"/ 2>/dev/null || true
    cp -f "$ADAPTER/adapters.safetensors" "$DEST"/
    echo "TRAINING COMPLETE — final weights snapshotted to $DEST"
    exit 0
  fi
  echo "=== crashed/hung; resuming from latest checkpoint ==="
  sleep 15
done
echo "TRAINING FAILED after $MAX_ATTEMPTS attempts" >&2
exit 1
