#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Segmented training driver: runs N segments of ITERS_EACH iters, chaining
# checkpoints through --resume-adapter-file so a GPU hang loses at most one
# short segment. Written after two long v2 runs died to Metal GPU hangs with
# other GPU processes active on the host.
#
#   SEGMENTS=4 ITERS_EACH=200 ADAPTER=<dir> bash train_segment_loop.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEGMENTS="${SEGMENTS:-4}"
ITERS_EACH="${ITERS_EACH:-200}"
ADAPTER="${ADAPTER:?set ADAPTER=<dir>}"
export SAVE_EVERY="${SAVE_EVERY:-100}"

mkdir -p "$ADAPTER"
LATEST="$ADAPTER/adapters.safetensors"
if [[ ! -f "$LATEST" ]]; then
  echo "no checkpoint at $LATEST — first segment trains from scratch" >&2
fi

for i in $(seq 1 "$SEGMENTS"); do
  echo "=== segment $i/$SEGMENTS (${ITERS_EACH} iters, resume from ${LATEST:-scratch}) ==="
  # mlx-lm reads datasets IN ORDER (no shuffle), so each resumed segment
  # would re-read the head of the file. Reorder deterministically per
  # segment instead — full coverage across segments, valid set untouched.
  python3 - "$HERE/data/sft/train.jsonl" "$i" <<'EOF'
import json, random, sys
path, seed = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
random.Random(f"segment:{seed}").shuffle(rows)
with open(path, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  reshuffled {len(rows)} train rows (seed {seed})")
EOF
  if [[ -f "$LATEST" ]]; then
    RESUME_FROM="$LATEST" ITERS="$ITERS_EACH" ADAPTER="$ADAPTER" bash "$HERE/train_lora.sh"
  else
    ITERS="$ITERS_EACH" ADAPTER="$ADAPTER" bash "$HERE/train_lora.sh"
  fi
done
echo "segment loop complete: $LATEST"
