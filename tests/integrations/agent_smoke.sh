#!/bin/bash
# Layer-B Tier-1 agent re-verification smoke.
#
# Run ON the Studio (real Apple Silicon + models + agent binaries), directly or
# from the self-hosted `agent-gate.yml` job. GitHub-hosted runners cannot do
# this — no Metal, no weights, and release-preflight skips every gate that needs
# a live `rapid-mlx serve`.
#
# Boots `rapid-mlx serve`, then drives the FOUR Tier-1 (flagship) agents —
# Claude Code, Codex, Hermes, Aider — through a real multi-step bug-fix task
# against the local model, and asserts each one actually made the test pass.
#
#   Usage:   RAPID_MLX_VENV=~/rapid-mlx-audit-venv ./agent_smoke.sh [model-alias]
#   Default: model-alias = qwen3.6-35b-8bit   (strong 8-bit — never 4-bit,
#            which confounds "weak model" with "broken integration")
#
# Exit code: 0 iff all four Tier-1 agents PASS. Non-zero blocks the release.
# Non-destructive on the shared Studio: it starts only its own server (and kills
# only that one), and backs up / restores (or removes) ~/.codex and ~/.hermes
# config it touches.
set -uo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/opt/coreutils/libexec/gnubin:$HOME/.local/bin:$PATH"
VENV="${RAPID_MLX_VENV:-$HOME/rapid-mlx-audit-venv}"
RMLX="$VENV/bin/rapid-mlx"
ALIAS="${1:-qwen3.6-35b-8bit}"
PORT="${RAPID_MLX_PORT:-8000}"
B="http://localhost:$PORT"
# Per-agent wall-clock budgets. The default gate model is a slow 35B hybrid with
# reasoning: cold (agentic tool/long-context kernels still compiling) it runs
# ~8 tok/s, warm ~23 tok/s. A multi-turn fix with a large agent system prompt can
# approach the old 260/300s knee cold, so give generous headroom (the kernel
# warmup after serve-ready also helps). A genuine hang still fails on the budget.
AGENT_TO="${AGENT_SMOKE_TIMEOUT:-480}"
HERMES_TO="${HERMES_SMOKE_TIMEOUT:-600}"
WORK="$HOME/agent-smoke-work"
LOG="$HOME/agent-smoke-serve.log"
SERVE_PID=""

CODEX_CFG="$HOME/.codex/config.toml"
HERMES_CFG="$HOME/.hermes/config.yaml"

# Portable timeout: coreutils `timeout`, or `gtimeout`, else a bash fallback
# (background the command, hard-kill after N seconds). macOS ships neither
# `timeout` nor `gtimeout` by default, so never assume one is on PATH.
if command -v timeout >/dev/null 2>&1; then
  TO() { timeout "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then
  TO() { gtimeout "$@"; }
else
  TO() {
    local secs="$1"; shift
    ( "$@" ) & local cmd_pid=$!
    ( sleep "$secs"; kill -9 "$cmd_pid" 2>/dev/null ) & local killer=$!
    wait "$cmd_pid" 2>/dev/null; local rc=$?
    kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
    return $rc
  }
fi

# Restore a config we may have overwritten with `--setup`: if we backed up a
# pre-existing file, put it back; if the file did not exist before, remove the
# one `--setup` created (and its marker). Idempotent — safe to call twice.
restore_cfg() {
  if [ -f "$1.smokebak" ]; then
    mv -f "$1.smokebak" "$1"
  elif [ -f "$1.created" ]; then
    rm -f "$1" "$1.created"
  fi
}
save_cfg() {
  mkdir -p "$(dirname "$1")"   # marker (and later --setup) needs the dir to exist
  if [ -f "$1" ]; then cp "$1" "$1.smokebak"; else touch "$1.created"; fi
}
# `agents <x> --setup` hardcodes localhost:8000; repoint it at our actual port.
patch_port() { [ -f "$1" ] && perl -pi -e "s#localhost:8000/v1#localhost:$PORT/v1#g" "$1"; }

cleanup() {
  [ -n "$SERVE_PID" ] && kill -9 "$SERVE_PID" 2>/dev/null
  # The ssh-localhost launch (see boot-serve block) runs serve outside this
  # script's process tree, so if the tracked pid ever misses a re-fork also reap
  # the listener on OUR port. We verified $PORT was free before we started, so
  # any listener here is the one we started — never a server we did not start.
  if [ -n "${PORT:-}" ]; then
    for _p in $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); do kill -9 "$_p" 2>/dev/null; done
  fi
  restore_cfg "$CODEX_CFG"
  restore_cfg "$HERMES_CFG"
  rm -rf "$WORK" "$LOG"
}
trap cleanup EXIT

fail() { echo "SMOKE-ABORT: $*" >&2; exit 3; }
[ -x "$RMLX" ] || fail "no rapid-mlx at $RMLX (set RAPID_MLX_VENV)"

echo "== versions =="
printf "  rapid-mlx  %s\n" "$($RMLX --version 2>&1 | head -1)"
for b in claude codex aider hermes; do
  command -v "$b" >/dev/null 2>&1 && printf "  %-9s  %s\n" "$b" "$($b --version 2>&1 | head -1)" \
                                  || printf "  %-9s  MISSING\n" "$b"
done
echo "== model: $ALIAS  port: $PORT =="

# ---- boot serve (never kill a server we did not start) -------------------
if curl -s -m 3 "$B/v1/models" >/dev/null 2>&1; then
  fail "port $PORT already serving — free it (a stray server is not ours to kill)"
fi
# --disable-prefix-cache: this gate drives coding agents through varied,
# non-repeating prompts, so a persisted prefix cache buys nothing here — but it
# LOADS SYNCHRONOUSLY at startup before /v1/models flips ready (server.py), and
# on the self-hosted Studio it accretes across gate runs (the shutdown save
# re-persists it every time), so readiness crept from ~15s to minutes run over
# run. Disabling it keeps serve readiness fast and deterministic for the gate
# without changing what the agents exercise.
#
# --no-thinking: the default gate model (Qwen3.6-35B-A3B) is a HYBRID REASONING
# model — with thinking on it emits a chain-of-thought whose length is highly
# variable and, on some turns, runs away toward max_tokens (agents request
# max_tokens=32000). A single such turn is ~450-1400s of decode, which alone
# blows an agent's 480/600s per-agent budget even on an idle GPU (observed: one
# /v1/messages stream ran >300s without finishing). That made the gate flaky in
# a way unrelated to whether the agent path actually works. We only need to
# verify the end-to-end agentic tool-calling path (serve → tool schema → agent
# CLI → tool exec → fix verified); the reasoning path itself is already covered
# by the release_check_m3 coding gate (which also runs --no-thinking). Forcing
# enable_thinking=False bounds each turn's output, so agents finish deterministically
# well inside budget and the gate tolerates moderate GPU contention with headroom.
# macOS GPU perf-state gotcha (self-hosted runner). When serve is spawned inside
# a launchd XPC-service context — the Actions runner is a LaunchAgent, so
# $XPC_SERVICE_NAME is set for everything it launches — macOS pins the process to
# a BACKGROUND GPU performance state. The GPU still pegs at ~99% util but its
# effective throughput craters ~100x, so the 35B hybrid's large-context (25-38k
# token) prefill never returns inside an agent's budget: serve flips READY, then
# every agent stalls and the gate times out. A LOGIN-session process (ssh /
# Terminal) is exempt from that throttle. Verified on the Studio: an identical
# 30k-token prompt returns in ~4s when serve runs in a login session vs >60s
# (unbounded stall) when spawned directly under the LaunchAgent. So when we detect
# the XPC context AND passwordless ssh-to-localhost works, we launch serve THROUGH
# ssh localhost, which reparents it into a login session with full GPU perf state.
# ssh localhost runs on THIS host, so the echoed `$!` is a valid LOCAL pid we
# track and kill in cleanup exactly like a direct launch. Any other context
# (dev machine, interactive `run.sh`, ssh unavailable) falls back to a direct
# launch and is unaffected.
SERVE_PID=""
if [ -n "${XPC_SERVICE_NAME:-}" ] \
   && command -v ssh >/dev/null 2>&1 \
   && ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 localhost true >/dev/null 2>&1; then
  echo "serve: launchd XPC-service context (\$XPC_SERVICE_NAME=$XPC_SERVICE_NAME) — launching via ssh localhost for full GPU perf state"
  SERVE_PID=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no localhost \
    "nohup '$RMLX' serve '$ALIAS' --port '$PORT' --disable-prefix-cache --no-thinking > '$LOG' 2>&1 & echo \$!" 2>/dev/null | tail -1)
  case "$SERVE_PID" in
    ''|*[!0-9]*) SERVE_PID=""; echo "serve: ssh-localhost launch returned no pid — falling back to direct launch (WARNING: large-ctx prefill may be GPU-throttled under launchd)";;
    *) echo "serve: launched via ssh localhost, pid=$SERVE_PID";;
  esac
elif [ -n "${XPC_SERVICE_NAME:-}" ]; then
  echo "serve: WARNING — launchd XPC-service context but ssh-localhost unusable; direct launch will be GPU-throttled (~100x slower large-ctx prefill). Enable Remote Login + passwordless localhost ssh, or run the runner from a login session."
fi
if [ -z "$SERVE_PID" ]; then
  nohup "$RMLX" serve "$ALIAS" --port "$PORT" --disable-prefix-cache --no-thinking > "$LOG" 2>&1 &
  SERVE_PID=$!
fi
# Serve-ready budget: 120 * 5s = 600s. The default gate model is a 35B hybrid
# (Qwen3.6-35B-A3B, GatedDeltaNet/linear-attention). Its FIRST cold serve on the
# Studio isn't just a weight load — it compiles the hybrid GatedDeltaNet Metal
# kernels, which alone pushes cold start to ~240s (a warm shader cache serves the
# same model in ~15s). The old 240s budget sat right on that cold-compile knee and
# flaked the release gate by ~1-2s. 600s clears the cold compile with margin and
# still leaves ample room under the 55-min job timeout for the four agent runs. A
# genuine hang (never binds) still fails — it just gets a realistic deadline.
for i in $(seq 1 120); do
  curl -s -m 3 "$B/v1/models" 2>/dev/null | grep -q '"id"' && { echo "serve READY (~$((i*5))s)"; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -20 "$LOG"; fail "serve process died during boot"; }
  sleep 5
  [ "$i" = 120 ] && { tail -20 "$LOG"; fail "serve not ready in 600s"; }
done

# Warm the model kernels before the timed agents. ``/v1/models`` returning above
# only means the server bound the port — the FIRST real completion still JIT-
# compiles the hybrid GatedDeltaNet / attention Metal kernels (cold ~8 tok/s vs
# warm ~23). Paying that here, untimed, keeps a cold shader cache from pushing
# the first agent past its per-agent budget. Best-effort — never fatal.
#
# Shape-specialized: a short prompt only compiles the small-sequence kernel; the
# real agents (claude first) open with a ~25-38k-token system+tools prompt that
# hits a DIFFERENT cold path — the large-context / chunked-prefill GatedDeltaNet
# kernel. That cold compile is what hung the 0.11.4 gate. On a cold Studio shader
# cache it ran long enough that the first agent's HTTP client timed out and
# disconnected; ``disconnect_guard`` then aborts the request at 0 tokens, the
# agent retried, re-hit the cold path and thrashed through its whole per-agent
# budget — four agents deep, that overran even the (old 55-min) job cap before any
# RESULT printed. It is NOT a product regression: a WARM box serves the same 25k
# request in 3-10s (measured), and the engine is byte-identical to the shipped
# 0.11.3 on every serving path. It is purely one-time cold-compile latency.
#
# So warm the LARGE-context kernels here, untimed, on BOTH routes the agents use
# (/v1/chat/completions for codex/hermes/aider, /v1/messages for claude), and
# RETRY until a request returns fast — a fast return proves the cold compile is
# fully paid and cached before any timed agent starts. Generous per-request
# timeout so the compile completes untimed; best-effort, never fatal.
echo "warming kernels (small + large context, both routes, untimed)…"
curl -s -m 60 "$B/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":16,\"temperature\":0}" \
  >/dev/null 2>&1 || true
python3 - "$B" "$ALIAS" <<'PY' 2>&1 || true
import json, sys, time, urllib.request
B, alias = sys.argv[1], sys.argv[2]
# Hard wall-clock budget for ALL warmup (both routes, all attempts). Bounds the
# step so a degraded box can't let the warmup itself eat into the job cap: the
# realistic cold path is one ~4-min compile then fast confirms (~7 min total),
# and this caps the pathological tail at ~10 min. Best-effort — on exhaustion we
# just fall through to the agents (their own retries absorb any residual cold).
DEADLINE = time.time() + 600
# ~30k-token prompt (~11 tok/line * 3000) — covers the agents' 25-38k opening
# prompts so the large-context / chunked-prefill Metal kernels compile HERE, not
# inside a timed per-agent budget.
prompt = "The quick brown fox jumps over the lazy dog. " * 3000 + "\nReply with the single word OK."

def warm(route, headers, body, label):
    # Attempt 1 pays the cold compile (untimed within the budget); attempt 2
    # confirms warm. "warm" == returned under 45s → stop early. Bounded by the
    # shared DEADLINE so total warmup can't overrun. Never fatal.
    data = json.dumps(body).encode()
    for attempt in (1, 2):
        remaining = int(DEADLINE - time.time())
        if remaining <= 1:
            print(f"{label}: warmup budget exhausted, deferring to agent retries")
            return
        t0 = time.time()
        try:
            req = urllib.request.Request(B + route, data=data, headers=headers)
            urllib.request.urlopen(req, timeout=min(480, remaining)).read()
            dt = time.time() - t0
            print(f"{label}: attempt {attempt} completed in {dt:.0f}s")
            if dt < 45:
                return
        except Exception as e:
            print(f"{label}: attempt {attempt} note (non-fatal): {e}")

# OpenAI route (codex / hermes / aider)
warm("/v1/chat/completions",
     {"Content-Type": "application/json"},
     {"model": alias, "messages": [{"role": "user", "content": prompt}],
      "max_tokens": 16, "temperature": 0},
     "large-ctx /v1/chat/completions")
# Anthropic route (claude — the first agent, the one that hung on 0.11.4). Shares
# the same prefill kernels, so this is usually already warm and returns fast.
warm("/v1/messages",
     {"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
     {"model": alias, "max_tokens": 16, "temperature": 0,
      "messages": [{"role": "user", "content": prompt}]},
     "large-ctx /v1/messages")
PY

# ---- the task: a buggy factorial + a failing test the agent must fix -----
seed_repo() {
  local d="$WORK/$1"; rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 1
  git init -q; git config user.email a@b.c; git config user.name smoke
  printf 'def factorial(n):\n    result = 1\n    for i in range(1, n):   # off-by-one bug\n        result *= i\n    return result\n' > calc.py
  printf 'from calc import factorial\nassert factorial(5) == 120, factorial(5)\nassert factorial(6) == 720\nassert factorial(0) == 1\nprint("ALL PASS")\n' > test_calc.py
  git add -A; git commit -qm seed
}
# PASS iff the test now exits 0 (agent fixed the bug and it verifies)
verify() { cd "$WORK/$1" 2>/dev/null && python3 test_calc.py >/dev/null 2>&1 && echo PASS || echo FAIL; }

# Plain phrasing (no backticks / em-dash — keep the prompt boring so the check
# measures the integration, not prompt parsing).
TASK='Run python3 test_calc.py, it fails. Fix the bug in calc.py so all assertions pass, then re-run to confirm. Only edit calc.py.'

# The agents (codex especially) are non-deterministic, so give each up to 2
# attempts (hermes 3) — a single miss must not flap the release gate.
#
# Run the four SERIALLY, one at a time, against the warm server. An earlier
# revision overlapped all four to collapse wall-clock from ~sum to ~max, on the
# theory that an agent leaves the GPU idle between generations so the four fill
# each other's gaps. In practice their turns overlap heavily, and batching 5-6
# in-flight requests — each a 25-38k-token agent prompt — onto the ONE GPU
# starves every stream (measured on the Studio: 12 output tokens in 65s at
# running=6). Multi-turn agents then blow their per-agent budgets and the gate
# flakes/fails (it flapped the 0.11.4 release twice this way). Serial hands each
# agent the whole GPU, so it runs at solo speed (claude ~75s, aider ~5s) and
# finishes far inside its budget: deterministic, ~sum(agents) wall-clock, still
# minutes under the 55-minute job. Each writes PASS/FAIL to its own result file.
mkdir -p "$WORK"

# ---- Claude Code (env var, /v1/messages) ---------------------------------
run_claude() {
  local r=FAIL
  for _try in 1 2; do
    seed_repo claude
    TO "$AGENT_TO" env ANTHROPIC_BASE_URL="$B" ANTHROPIC_API_KEY=not-needed \
      claude -p "$TASK" --model "$ALIAS" --dangerously-skip-permissions >/dev/null 2>&1
    [ "$(verify claude)" = PASS ] && { r=PASS; break; }
  done
  echo "$r" > "$WORK/claude.result"
}

# ---- Codex (uses ~/.codex/config.toml rendered below) --------------------
run_codex() {
  local r=FAIL
  for _try in 1 2; do
    seed_repo codex
    TO "$AGENT_TO" codex exec "$TASK" --model "$ALIAS" \
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check >/dev/null 2>&1
    [ "$(verify codex)" = PASS ] && { r=PASS; break; }
  done
  echo "$r" > "$WORK/codex.result"
}

# ---- Hermes (uses ~/.hermes/config.yaml rendered below) ------------------
run_hermes() {
  local r=FAIL
  for _try in 1 2 3; do
    seed_repo hermes
    TO "$HERMES_TO" hermes chat -q "$TASK" -Q -m "$ALIAS" >/dev/null 2>&1
    [ "$(verify hermes)" = PASS ] && { r=PASS; break; }
  done
  echo "$r" > "$WORK/hermes.result"
}

# ---- Aider (env vars, LiteLLM openai/ prefix) ----------------------------
run_aider() {
  local r=FAIL
  for _try in 1 2; do
    seed_repo aider
    TO "$AGENT_TO" env OPENAI_API_BASE="$B/v1" OPENAI_API_KEY=not-needed \
      aider --model "openai/$ALIAS" --message "$TASK" \
      --yes-always --no-auto-commits --no-show-model-warnings calc.py >/dev/null 2>&1
    [ "$(verify aider)" = PASS ] && { r=PASS; break; }
  done
  echo "$r" > "$WORK/aider.result"
}

# Render the two agent config files up front (distinct paths → no collision),
# THEN launch. --base-url points setup at OUR serve port (not the default
# :8000). For hermes it is REQUIRED: Hermes rejects any model whose
# context_length is below 64K, and setup only writes the model's real context
# (qwen3.6-35b-8bit → 262144) when it can reach the running server to detect it;
# without --base-url it falls back to the 32768 default and Hermes refuses to
# start every time.
save_cfg "$CODEX_CFG"
"$RMLX" agents codex --setup --base-url "$B/v1" >/dev/null 2>&1
patch_port "$CODEX_CFG"
save_cfg "$HERMES_CFG"
"$RMLX" agents hermes --setup --base-url "$B/v1" >/dev/null 2>&1
patch_port "$HERMES_CFG"

echo "running 4 Tier-1 agents serially (budget ${AGENT_TO}s, hermes ${HERMES_TO}s)…"
# Serial, NOT backgrounded: overlapping them oversubscribes the single GPU and
# flakes the gate (see the rationale above run_claude). Each runs to completion
# — with its own retries + per-agent timeout — before the next starts, so it
# gets the whole GPU and finishes at solo speed. Never a bare `wait` here: the
# serve is still backgrounded (SERVE_PID), so a bare wait would hang the gate.
run_claude
run_codex
run_hermes
run_aider

# Restore the config files only after every agent has finished using them.
restore_cfg "$CODEX_CFG"
restore_cfg "$HERMES_CFG"

R_CLAUDE=$(cat "$WORK/claude.result" 2>/dev/null || echo FAIL)
R_CODEX=$(cat "$WORK/codex.result" 2>/dev/null || echo FAIL)
R_HERMES=$(cat "$WORK/hermes.result" 2>/dev/null || echo FAIL)
R_AIDER=$(cat "$WORK/aider.result" 2>/dev/null || echo FAIL)

# ---- report --------------------------------------------------------------
echo
echo "== Tier-1 re-verification ($ALIAS) =="
printf "  claude-code  %s\n" "$R_CLAUDE"
printf "  codex-cli    %s\n" "$R_CODEX"
printf "  hermes       %s\n" "$R_HERMES"
printf "  aider        %s\n" "$R_AIDER"

for r in "$R_CLAUDE" "$R_CODEX" "$R_HERMES" "$R_AIDER"; do
  [ "$r" = PASS ] || { echo "RESULT: FAIL — a Tier-1 agent regressed; this blocks the release."; exit 1; }
done
echo "RESULT: PASS — all four Tier-1 agents verified on $ALIAS."
