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
# Each alias is booted on a dedicated port in its manifest-selected lane, run through
# evals/coherence_gate.py (blocking golden answers), then torn down before the
# next. Any alias that fails its blocking golden gate fails the whole sweep.
#
# Reasoning-distill families (e.g. DeepSeek-R1-Distill) do not honor
# --no-thinking: they emit chain-of-thought in the visible channel. The fleet
# marks them with "reasoning_distill": true; the sweep boots those with
# --thinking (so the parser routes the chain-of-thought to the reasoning
# channel) and passes --reasoning-distill so the gate scores the concluded
# answer rather than the raw visible text (issue #1323).
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
#   0 — every alias passed its blocking golden gate (and any unreached families
#       were reported as capacity-skipped for insufficient disk, per #2206)
#   1 — at least one alias failed
#   2 — pre-flight refusal (port busy), a server that never came up for reasons
#       other than capacity, or a capacity-skip while
#       COHERENCE_SWEEP_REQUIRE_ALL=1 is set
#
# On a disk-constrained host, a fleet family that `serve` refuses to download
# because it would exceed free disk is reported as a "capacity-skip" instead of
# a hard infrastructure failure, and the resident representatives are still
# validated. Set COHERENCE_SWEEP_REQUIRE_ALL=1 to preserve the strict
# every-family-required behaviour on fully provisioned release hosts.

set -euo pipefail

PY="${PY:-python3.12}"
PORT="${PORT:-8402}"
FLEET_SCOPE="${FLEET_SCOPE:-auto}"
# Server readiness has two bounds. Cold-cache mirror/download work may take
# longer than the old fixed 180s while still making deterministic progress
# (#1686), so only 180s with no new log output counts as stalled. The hard cap
# remains absolute even when a noisy process keeps writing forever.
BOOT_STALL_S="${COHERENCE_BOOT_STALL_S:-180}"
BOOT_HARD_S="${COHERENCE_BOOT_HARD_S:-1800}"
case "$BOOT_STALL_S:$BOOT_HARD_S" in
  *[!0-9:]*|0:*|*:0)
    echo "ERROR: COHERENCE_BOOT_STALL_S and COHERENCE_BOOT_HARD_S must be positive integers." >&2
    exit 2
    ;;
esac
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
skipped=""
passed=""
# Strict full-fleet mode (issue #2206 part c): when set, an unresident family
# that is capacity-skipped for lack of disk is treated as an infrastructure
# failure instead of a reported-but-passing skip. Release hosts provisioned
# with enough storage keep the old every-family-required behaviour.
case "${COHERENCE_SWEEP_REQUIRE_ALL:-0}" in
  0) REQUIRE_ALL=0 ;;
  1) REQUIRE_ALL=1 ;;
  *)
    echo "ERROR: COHERENCE_SWEEP_REQUIRE_ALL must be 0 or 1." >&2
    exit 2
    ;;
esac
for MODEL in $MODELS; do
  line
  echo "  → $MODEL"
  line

  DISTILL=0
  if "$PY" scripts/release_fleet.py is-reasoning-distill "$MODEL"; then
    DISTILL=1
  else
    classifier_status=$?
    if [ "$classifier_status" -ne 1 ]; then
      echo "ERROR: could not classify reasoning mode for $MODEL" >&2
      exit 2
    fi
  fi
  # A dependency-coherence gate must evaluate the model under test, not KV
  # tensors persisted by an earlier runtime.  Disk prefix caches are keyed by
  # model/config, not by MLX build, so loading one can both hide a regression
  # and create a false failure after a toolchain move.
  SERVE_ARGS=(--port "$PORT" --disable-prefix-cache)
  if "$PY" scripts/release_fleet.py forces-text-lane "$MODEL"; then
    # Gemma 4's checkpoint also carries a vision tower, but this gate scores
    # its text path. Auto-routing would require the optional mlx-vlm extra and
    # fail a valid base-wheel release before the first golden prompt (#1685).
    SERVE_ARGS+=(--no-mllm)
  else
    classifier_status=$?
    if [ "$classifier_status" -ne 1 ]; then
      echo "ERROR: could not classify serving lane for $MODEL" >&2
      exit 2
    fi
  fi
  if [ "$DISTILL" = "1" ]; then
    # The serve CLI exposes the reasoning profile as ``--reasoning``;
    # ``--thinking`` has never been a valid serve flag. The stale flag made
    # every reasoning-distill fleet member fail before model load.
    SERVE_ARGS+=(--reasoning)
  else
    SERVE_ARGS+=(--no-thinking)
  fi
  "$PY" -m vllm_mlx.cli serve "$MODEL" "${SERVE_ARGS[@]}" > "$LOG" 2>&1 &
  CURRENT_PID=$!
  echo "$CURRENT_PID" > "$PIDFILE"

  up=0
  boot_failure="hard timeout (${BOOT_HARD_S}s)"
  boot_started=$SECONDS
  last_progress=$SECONDS
  last_log_size=0
  while [ "$((SECONDS - boot_started))" -lt "$BOOT_HARD_S" ]; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    # Bail early if the server process died during load.
    if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
      boot_failure="server process exited"
      break
    fi

    # `wc -c` is portable to macOS bash 3.2 and observes both ordinary logs
    # and carriage-return progress updates redirected into the log file.
    log_size=$(wc -c < "$LOG" | tr -d ' ')
    if [ "$log_size" -gt "$last_log_size" ]; then
      last_log_size=$log_size
      last_progress=$SECONDS
    elif [ "$((SECONDS - last_progress))" -ge "$BOOT_STALL_S" ]; then
      boot_failure="no startup progress for ${BOOT_STALL_S}s"
      break
    fi
    sleep 1
  done
  if [ "$up" != 1 ]; then
    # #2206: `serve` refuses a download that would exceed free disk with an
    # exact, greppable line before exiting 1. On a constrained disk that is a
    # capacity outcome, not a behavioural one — report it as a capacity-skip
    # and keep validating the resident representatives, rather than dying the
    # whole release gate for storage rather than correctness.
    if grep -Fqx '  Error: Insufficient disk space for download.' "$LOG"; then
      echo "  ⚠ $MODEL capacity-skipped: download would exceed free disk." >&2
      tail -8 "$LOG" >&2
      skipped="$skipped $MODEL"
    else
      echo "ERROR: $MODEL server did not come up: $boot_failure. Last log lines:" >&2
      tail -20 "$LOG" >&2
      infra_failed="$infra_failed $MODEL(boot)"
    fi
  else
    if [ "$DISTILL" = "1" ]; then
      gate_command=("$PY" evals/coherence_gate.py --reasoning-distill)
    else
      # macOS still ships bash 3.2. With ``set -u``, expanding an empty
      # array is an unbound-variable error there, so keep the no-argument
      # command explicit instead of appending to an empty GATE_ARGS array.
      gate_command=("$PY" evals/coherence_gate.py)
    fi
    if "${gate_command[@]}"; then
      echo "  ✓ $MODEL coherent"
      passed="$passed $MODEL"
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
strict_skipped=""
if [ -n "$skipped" ] && [ "$REQUIRE_ALL" = 1 ]; then
  strict_skipped="$skipped"
fi
if [ -n "$infra_failed" ] || [ -n "$strict_skipped" ] || { [ -n "$skipped" ] && [ -z "$passed" ] && [ -z "$failed" ]; }; then
  echo "  SWEEP INFRASTRUCTURE FAILURE" >&2
  if [ -n "$infra_failed" ]; then echo "  INFRASTRUCTURE FAILURES —$infra_failed" >&2; fi
  if [ -n "$strict_skipped" ]; then
    echo "  CAPACITY-SKIPPED (COHERENCE_SWEEP_REQUIRE_ALL=1) —$strict_skipped" >&2
  elif [ -n "$skipped" ] && [ -z "$passed" ]; then
    echo "  NO RESIDENT ALIAS VALIDATED; CAPACITY-SKIPPED —$skipped" >&2
  elif [ -n "$skipped" ]; then
    echo "  CAPACITY-SKIPPED —$skipped" >&2
  fi
  if [ -n "$failed" ]; then echo "  MODEL FAILURES —$failed"; fi
  line
  exit 2
fi
if [ -n "$failed" ]; then
  echo "  SWEEP FAILED —$failed"
  if [ -n "$skipped" ]; then echo "  CAPACITY-SKIPPED —$skipped"; fi
  line
  exit 1
fi
if [ -n "$skipped" ]; then
  echo "  SWEEP PASSED — all resident aliases coherent; $skipped capacity-skipped (insufficient disk)"
  line
  exit 0
fi
echo "  SWEEP PASSED — all aliases coherent"
line
