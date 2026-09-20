#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Marvin's Garden — full evaluation matrix for one adapter.
#   usage: ADAPTER=<dir> TAG=<name> [MODEL=<model>] bash eval_matrix.sh
# Writes results/eval_<TAG>.json, results/eval_<TAG>_ensemble.json,
# results/iq_<TAG>.json and prints a compact summary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/Volumes/NVMe-4T/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7}"
ADAPTER="${ADAPTER:?set ADAPTER=<adapter dir>}"
TAG="${TAG:?set TAG=<name>}"
PY="${PY:-python}"

$PY "$HERE/eval_label_readout.py" --model "$MODEL" --adapter "$ADAPTER" \
  --temperature 1.0 --output "$HERE/results/eval_${TAG}.json" > /dev/null
$PY "$HERE/eval_label_readout.py" --model "$MODEL" --adapter "$ADAPTER" \
  --temperature 1.0 --styles base,concise,spec \
  --output "$HERE/results/eval_${TAG}_ensemble.json" > /dev/null
$PY "$HERE/iq_probe.py" --model "$MODEL" --adapter "$ADAPTER" \
  --output "$HERE/results/iq_${TAG}.json" > /dev/null

$PY - "$TAG" <<'EOF'
import json, sys
tag = sys.argv[1]
base = json.load(open("results/iq_base.json"))
e = json.load(open(f"results/eval_{tag}.json"))
en = json.load(open(f"results/eval_{tag}_ensemble.json"))
iq = json.load(open(f"results/iq_{tag}.json"))
pf = {k: round(v["accuracy"] * 100, 1) for k, v in e["per_family"].items()}
print(f"[{tag}] decision {e['accuracy']:.1%} | ensemble {en['accuracy']:.1%} | "
      f"flip {e['flip_both_members_correct']:.1%} | ECE {e['ece_15bin']:.3f}")
print(f"[{tag}] per-family {pf}")
print(f"[{tag}] iq overall {iq['overall']['accuracy']:.0%} "
      f"(base {base['overall']['accuracy']:.0%}) | "
      f"{e['timing']['ms_per_decision']:.0f} ms/decision")
EOF
