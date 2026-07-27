#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Output-coherence sweep (#1247) — run the deterministic golden gate against a
# SET of representative aliases, one at a time. A model-specific regression (the
# 35B RMSNorm incident, #1234) does NOT surface if the release check only runs
# the default 4B/9B alias, so a release / model-path change must sweep the
# aliases it affects — one representative per family, plus any alias the change
# touches.
#
# Each alias is booted on a dedicated port with --no-thinking, run through
# evals/coherence_gate.py (blocking golden answers), then torn down before the
# next. Any alias that fails its blocking golden gate fails the whole sweep.
#
# With no explicit models, the shared release fleet is selected automatically.
# A normal release covers every routinely feasible family; changes to an MLX
# dependency since the previous release tag add the Ultra-only Hy3 representative.
#
# Usage:
#   bash scripts/coherence_sweep.sh
#   bash scripts/coherence_sweep.sh qwen3.5-4b-4bit qwen3.6-35b
#   MODELS="qwen3.5-4b-4bit qwen3.6-35b" bash scripts/coherence_sweep.sh
#   FLEET_SCOPE=toolchain bash scripts/coherence_sweep.sh
#
# Exit codes:
#   0 — every alias passed its blocking golden gate
#   1 — at least one alias failed
#   2 — pre-flight refusal (port busy) or a server that never came up

set -euo pipefail

PY="${PY:-python3.12}"
PORT="${PORT:-8402}"
FLEET_SCOPE="${FLEET_SCOPE:-auto}"
if [ "$#" -gt 0 ]; then
  MODELS="$*"
elif [ -n "${MODELS:-}" ]; then
  MODELS="$MODELS"
else
  fleet_args=(models --scope "$FLEET_SCOPE")
  if [ -n "${RELEASE_FLEET_BASE_REF:-}" ]; then
    fleet_args+=(--base-ref "$RELEASE_FLEET_BASE_REF")
  fi
  MODELS="$("$PY" scripts/release_fleet.py "${fleet_args[@]}")"
fi
if [ -z "$MODELS" ]; then
  echo "ERROR: release fleet selected no coherence models." >&2
  exit 2
fi
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rapid-mlx-coherence-sweep.XXXXXX")"
LOG="$WORK_DIR/server.log"
PIDFILE="$WORK_DIR/server.pid"

line() { printf '%s\n' "============================================================"; }

CURRENT_PID=""
cleanup() {
  if [ -n "$CURRENT_PID" ]; then
    kill "$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK_DIR"
}
handle_signal() {
  signal_status=$1
  trap - EXIT
  cleanup
  exit "$signal_status"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use — pick another with PORT=... or free it." >&2
  exit 2
fi

export RAPID_MLX_BASE_URL="http://127.0.0.1:${PORT}/v1"

line
echo "  output-coherence sweep"
echo "  models: $MODELS"
echo "  port:   $PORT"
line

failed=""
infra_failed=""
for MODEL in $MODELS; do
  line
  echo "  → $MODEL"
  line

  "$PY" -m vllm_mlx.cli serve "$MODEL" --port "$PORT" --no-thinking > "$LOG" 2>&1 &
  CURRENT_PID=$!
  echo "$CURRENT_PID" > "$PIDFILE"

  up=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    # Bail early if the server process died during load.
    if ! kill -0 "$CURRENT_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ "$up" != 1 ]; then
    echo "ERROR: $MODEL server did not come up. Last log lines:" >&2
    tail -20 "$LOG" >&2
    infra_failed="$infra_failed $MODEL(boot)"
  else
    if "$PY" evals/coherence_gate.py; then
      echo "  ✓ $MODEL coherent"
    else
      gate_status=$?
      if [ "$gate_status" -eq 2 ]; then
        echo "  ✗ $MODEL coherence gate infrastructure failure" >&2
        infra_failed="$infra_failed $MODEL(gate-infra)"
      else
        echo "  ✗ $MODEL FAILED coherence gate" >&2
        failed="$failed $MODEL"
      fi
    fi
  fi

  kill "$CURRENT_PID" 2>/dev/null || true
  wait "$CURRENT_PID" 2>/dev/null || true
  CURRENT_PID=""
  rm -f "$PIDFILE"
done

line
if [ -n "$infra_failed" ]; then
  echo "  SWEEP INFRASTRUCTURE FAILURE —$infra_failed"
  if [ -n "$failed" ]; then echo "  MODEL FAILURES —$failed"; fi
  line
  exit 2
fi
if [ -n "$failed" ]; then
  echo "  SWEEP FAILED —$failed"
  line
  exit 1
fi
echo "  SWEEP PASSED — all aliases coherent"
line
