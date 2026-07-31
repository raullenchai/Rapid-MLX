#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# L1 release-flow smoke — a SMALL-model GPU coherence + tool-call format check
# for the FREE GitHub-hosted macos-14 runner. That runner has a real Metal GPU
# (``mx.default_device()`` -> ``Device(gpu, 0)``), so MLX runs on the GPU there
# for $0; its ~7GB RAM caps it at small models, not the 35B Studio (L2) tier.
#
# Runs on every PR to catch GROSS engine breakage on cheap hardware BEFORE the
# expensive Studio gate:
#   * garbage-from-first-token (#1234 doubled-norm class) via the #1247 GOLDEN
#     coherence gate, and
#   * tool-call parser regressions via a forced-``tool_choice`` format assertion.
# Neither can run on the pure-CPU Linux CI (no Metal, no weights).
#
# Boots ``rapid-mlx serve <model>``, runs both checks, then tears down ONLY the
# server it started — never ``pkill rapid-mlx``, because a shared machine (a dev
# box, a self-hosted runner) may have other rapid-mlx servers running, including
# a production one on :8000.
#
#   Usage:   ./scripts/l1_smoke.sh [model-alias]
#   Default: qwen3.5-4b-4bit  (~2.2GB 4-bit — fits the runner's ~7GB RAM and
#            clears the GOLDEN set in <1min. 4B is fine for a parser/coherence
#            gate; it is NOT a chat-quality eval, which needs the L2 35B tier.)
#
# Env:
#   RAPID_MLX        rapid-mlx executable (default: ``rapid-mlx`` on PATH)
#   PYTHON           python for the gate scripts (default: ``python3``)
#   RAPID_MLX_PORT   serve port (default: 8123 — deliberately not :8000)
#   L1_MODEL         model alias (overridden by the positional arg)
#
# Exit code: 0 iff BOTH checks pass; non-zero blocks the PR.
set -uo pipefail

RMLX="${RAPID_MLX:-rapid-mlx}"
PY="${PYTHON:-python3}"
ALIAS="${1:-${L1_MODEL:-qwen3.5-4b-4bit}}"
PORT="${RAPID_MLX_PORT:-8123}"
B="http://127.0.0.1:$PORT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$(mktemp -t l1-serve.XXXXXX)"
SERVE_PID=""

fail() { echo "L1-SMOKE FAIL: $*" >&2; exit 1; }

cleanup() {
  # Kill ONLY the server this script started. Never ``pkill rapid-mlx``: on a
  # shared machine that would also kill an unrelated (possibly production)
  # server the script does not own.
  [ -n "$SERVE_PID" ] && kill -9 "$SERVE_PID" 2>/dev/null
  rm -f "$LOG"
}
trap cleanup EXIT

command -v "$RMLX" >/dev/null 2>&1 || fail "no rapid-mlx on PATH (set RAPID_MLX)"
echo "== L1 smoke =="
printf "  rapid-mlx  %s\n" "$("$RMLX" --version 2>&1 | head -1)"
echo "  model: $ALIAS   port: $PORT"

# ---- refuse to hijack a port we do not own -------------------------------
if curl -s -m 3 "$B/v1/models" >/dev/null 2>&1; then
  fail "port $PORT already serving — not ours to kill; free it or set RAPID_MLX_PORT"
fi

# ---- boot serve ----------------------------------------------------------
# --no-thinking: the coherence gate measures answer coherence, not think-mode
# behavior (its no-think-leak case still asserts no raw <think> tag leaks).
#
# --no-mllm: force the text-only mlx-lm lane. This gate only exercises the
# TEXT coherence + tool-call parser paths, never vision, so the LM backbone is
# all we need. It is a no-op for text-only models, but for a small model that
# is *detected* as multimodal (Gemma nano, Ministral-3-2512) it (a) skips the
# ``[vision]`` / mlx-vlm requirement so a lean ``pip install -e .`` suffices,
# and (b) avoids the mlx-vlm text-generation path, which returns incoherent
# output / hangs for these models (measured 2026-07-31: gemma-4-e2b 0/6 golden,
# Ministral-3 request-hang under mlx-vlm 0.6.3). ``resolve_serving_lane`` maps
# ``--no-mllm`` -> ``force_text`` -> the text lane, matching engine semantics.
nohup "$RMLX" serve "$ALIAS" --port "$PORT" --no-thinking --no-mllm > "$LOG" 2>&1 &
SERVE_PID=$!
for i in $(seq 1 60); do
  curl -s -m 3 "$B/v1/models" 2>/dev/null | grep -q '"id"' && { echo "serve READY (~$((i * 3))s)"; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -30 "$LOG"; fail "serve process died during boot"; }
  sleep 3
  [ "$i" = 60 ] && { tail -30 "$LOG"; fail "serve not ready in 180s"; }
done

# ---- check 1/2: output coherence (#1247 GOLDEN, blocking) ----------------
echo "== [1/2] coherence gate (#1247) =="
RAPID_MLX_BASE_URL="$B/v1" "$PY" "$REPO_ROOT/evals/coherence_gate.py" \
  --base-url "$B/v1" --timeout 90 \
  || fail "coherence gate failed — served model gave a wrong/incoherent golden answer"

# ---- check 2/2: tool-call format under forced tool_choice (blocking) -----
echo "== [2/2] tool-call format =="
"$PY" "$REPO_ROOT/scripts/l1_toolcall_check.py" --base-url "$B/v1" --timeout 90 \
  || fail "tool-call format check failed — forced tool_choice did not yield a well-formed call"

echo "L1-SMOKE PASS ($ALIAS): coherence + tool-call format OK"
