#!/bin/bash
# Layer-B Tier-1 agent re-verification smoke.
#
# Run ON the Studio (real Apple Silicon + models + agent binaries), directly or
# from the self-hosted `agent-gate.yml` job. GitHub-hosted runners cannot do
# this — no Metal, no weights, and release-preflight skips every gate that needs
# a live `rapid-mlx serve`.
#
# Boots `rapid-mlx serve`, then drives the FIVE Tier-1 (flagship) agents —
# Claude Code, Codex, Hermes, Aider, DeepSeek Harness — through a real
# multi-step bug-fix task against the local model, and asserts each one
# actually made the test pass.
#
#   Usage:   RAPID_MLX_VENV=~/rapid-mlx-audit-venv ./agent_smoke.sh [model-alias]
#   Default: model-alias = qwen3.6-35b-8bit   (strong 8-bit — never 4-bit,
#            which confounds "weak model" with "broken integration")
#
# Exit code: 0 iff all five Tier-1 agents PASS. Non-zero blocks the release.
# Non-destructive on the shared Studio: it starts only its own server (and kills
# only that one), and backs up / restores (or removes) the ~/.codex, ~/.hermes
# and ~/.dsh config it touches.
set -uo pipefail

# Resolved HERE, before anything runs: ``seed_repo``/``verify`` cd into the
# per-agent scratch dirs and that cwd persists (they are functions, not
# subshells). Resolving a relative ``$BASH_SOURCE`` later — after the cwd has
# moved — yields an empty REPO_ROOT, and the release gate would then invoke
# ``/evals/coherence_gate.py`` and fail a run where every check passed.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

# Throwaway config homes. codex, hermes and dsh all relocate their entire
# config directory via these variables, and `agents <x> --setup` honours them,
# so this gate never reads or writes the operator's real ~/.codex, ~/.hermes or
# ~/.dsh. DSH_HOME matters more than the other two: `agents dsh --setup` also
# writes a credential file (.credentials.yaml) beside settings.yaml, so an
# un-redirected run would touch the operator's credential store, not just a
# provider block.
#
# This replaces backup-then-restore as the PRIMARY protection. That approach
# has two failure modes we actually hit: the restore never runs if the script
# is SIGKILLed, and — the one that did real damage — once a config has been
# clobbered by any run, every later run faithfully backs it up and restores
# the *damaged* file. The operator's codex stayed pointed at a local rapid-mlx
# server for weeks that way, with each run's restore looking like it worked.
#
# Exported so `agents --setup`, the agent CLIs, and anything they spawn all
# agree on the same location.
#
# Fail CLOSED. This script has no `set -e`, so a failed `mktemp -d` (full or
# unwritable temp volume) would export an empty value, and an empty or
# whitespace-only home is treated as *unset* on the Python side — sending
# `--setup` straight back to the operator's real ~/.codex / ~/.hermes, which
# is precisely the damage this redirect exists to prevent.
#
# The blankness test is delegated to Python on purpose. The shell and Python do
# not agree on what whitespace is: BSD `tr -d '[:space:]'` keeps U+0085 while
# `str.strip()` removes it, so a shell-side check passes a value that Python
# then treats as unset — the guard reports safe and the real config is written
# anyway. One definition, owned by the side that actually makes the decision.
export CODEX_HOME="${CODEX_HOME_OVERRIDE:-$(mktemp -d)}"
export HERMES_HOME="${HERMES_HOME_OVERRIDE:-$(mktemp -d)}"
export DSH_HOME="${DSH_HOME_OVERRIDE:-$(mktemp -d)}"
for _home_var in CODEX_HOME HERMES_HOME DSH_HOME; do
  eval "_home_val=\${$_home_var}"
  # Resolve it EXACTLY as _resolve_config_path will (strip, then expanduser)
  # and re-export the result, so the value this script validates, backs up and
  # cleans up is byte-for-byte the one setup writes to. Validating the raw
  # value instead lets the two diverge: a real directory named "$HOME/.codex "
  # (trailing space) passes an is-a-directory check here while Python strips
  # the space and writes the operator's real ~/.codex/config.toml — which the
  # backup and restore then miss, because they are looking at the other path.
  # ...and CANONICALIZE, because the comparison below is only as good as the
  # spelling. `$HOME/.codex/`, `$HOME/.codex/.`, a `..` path and a symlink all
  # resolve to the protected directory while comparing unequal as strings. A
  # relative value would also break the run outright, since `seed_repo` changes
  # cwd before the agent reads the re-exported home.
  _home_val="$(python3 -c 'import os, sys
v = sys.argv[1].strip()
sys.stdout.write(os.path.realpath(os.path.expanduser(v)) if v else "")' \
    "${_home_val}" 2>/dev/null)"
  if [ -z "${_home_val}" ]; then
    echo "SMOKE-ABORT: $_home_var is blank once resolved — refusing to run, as" >&2
    echo "             \`agents --setup\` would then write the operator's real config." >&2
    exit 3
  fi
  [ -d "${_home_val}" ] || { echo "SMOKE-ABORT: $_home_var=${_home_val} is not a directory" >&2; exit 3; }
  # Belt and braces: refuse to run against the operator's REAL config dir, no
  # matter how the value spelled its way here. Agreeing with Python is not the
  # same as being safe — "$HOME/.codex " resolves to exactly the directory this
  # redirect exists to protect, and it is a real directory, so every check
  # above passes. There is no legitimate reason for this gate to write there.
  case "$_home_var" in
    CODEX_HOME)  _home_real="$(python3 -c 'import os,sys; sys.stdout.write(os.path.realpath(os.path.expanduser("~/.codex")))')" ;;
    HERMES_HOME) _home_real="$(python3 -c 'import os,sys; sys.stdout.write(os.path.realpath(os.path.expanduser("~/.hermes")))')" ;;
    DSH_HOME)    _home_real="$(python3 -c 'import os,sys; sys.stdout.write(os.path.realpath(os.path.expanduser("~/.dsh")))')" ;;
    *)           _home_real="" ;;
  esac
  if [ -n "$_home_real" ] && [ "${_home_val}" = "$_home_real" ]; then
    echo "SMOKE-ABORT: $_home_var resolves to the operator's real config dir" >&2
    echo "             ($_home_real). Refusing — this gate must never write there." >&2
    exit 3
  fi
  export "$_home_var=${_home_val}"
done
unset _home_var _home_val _home_real
CODEX_CFG="$CODEX_HOME/config.toml"
HERMES_CFG="$HERMES_HOME/config.yaml"
DSH_CFG="$DSH_HOME/settings.yaml"

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
  restore_cfg "$DSH_CFG"
  # All three live under throwaway homes now, so this is just tidying temp
  # files — the operator's real ~/.codex, ~/.hermes and ~/.dsh were never
  # touched.
  [ -n "${CODEX_HOME_OVERRIDE:-}" ] || rm -rf "$CODEX_HOME"
  [ -n "${HERMES_HOME_OVERRIDE:-}" ] || rm -rf "$HERMES_HOME"
  [ -n "${DSH_HOME_OVERRIDE:-}" ] || rm -rf "$DSH_HOME"
  rm -rf "$WORK" "$LOG"
}
trap cleanup EXIT

fail() { echo "SMOKE-ABORT: $*" >&2; exit 3; }
[ -x "$RMLX" ] || fail "no rapid-mlx at $RMLX (set RAPID_MLX_VENV)"

echo "== versions =="
printf "  rapid-mlx  %s\n" "$($RMLX --version 2>&1 | head -1)"
for b in claude codex aider hermes dsh; do
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
# still leaves ample room under the job timeout for the five agent runs. A
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
# (/v1/chat/completions for codex/hermes/aider/dsh, /v1/messages for claude), and
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
# Run the five SERIALLY, one at a time, against the warm server. An earlier
# revision overlapped them to collapse wall-clock from ~sum to ~max, on the
# theory that an agent leaves the GPU idle between generations so they fill
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

# ---- DeepSeek Harness (uses $DSH_HOME/settings.yaml rendered below) ------
# RAPID_MLX_API_KEY is passed explicitly even though `--setup` also writes the
# same sentinel into $DSH_HOME/.credentials.yaml: DSH's pi-ai transport insists
# on RESOLVING a credential for the provider before it will dispatch, and the
# env var is the path that does not depend on the credential store having been
# written. Belt and braces, same as ANTHROPIC_API_KEY / OPENAI_API_KEY above.
#
# `--profile headless` is the one-shot task profile (answer, print, exit); the
# interactive `web` / `tui` profiles would block forever on a gate runner.
# NOTE: dsh exits 0 even when it fails outright (a bad provider prints
# `NO_ADAPTER: ...` and still returns 0), so the exit status is deliberately
# ignored here — `verify` re-running the real test is the only signal that
# means anything. Do not "improve" this into an exit-code check.
run_dsh() {
  local r=FAIL
  for _try in 1 2; do
    seed_repo dsh
    TO "$AGENT_TO" env RAPID_MLX_API_KEY=not-needed \
      dsh --profile headless "$TASK" >/dev/null 2>&1
    [ "$(verify dsh)" = PASS ] && { r=PASS; break; }
  done
  echo "$r" > "$WORK/dsh.result"
}

# Render the two agent config files up front (distinct paths → no collision),
# THEN launch. --base-url points setup at OUR serve port (not the default
# :8000). For hermes it is REQUIRED: Hermes rejects any model whose
# context_length is below 64K, and setup only writes the model's real context
# (qwen3.6-35b-8bit → 262144) when it can reach the running server to detect it;
# without --base-url it falls back to the 32768 default and Hermes refuses to
# start every time.
# Assert the invariant directly instead of enumerating the ways it can break.
# Every guard above stops a *spelling* — a resolved path, a blank value, the
# protected directory. None of them can see a `home_env` the profile renamed:
# a user profile in ~/.rapid-mlx/agents/ may legitimately declare
# `home_env: MY_CODEX_HOME`, and if that variable is unset, `--setup` resolves
# back to the operator's real config no matter what this script exported.
# Fingerprinting the real files and checking them afterwards catches that, and
# anything else nobody has thought of yet.
#
# ~/.dsh contributes TWO paths, not one: `agents dsh --setup` writes the
# provider block to settings.yaml AND a credential sentinel to
# .credentials.yaml. Fingerprinting only the settings file would let a
# redirect failure rewrite the operator's real credential store unnoticed.
_real_fingerprint() {
  for f in "$HOME/.codex/config.toml" "$HOME/.hermes/config.yaml" \
           "$HOME/.dsh/settings.yaml" "$HOME/.dsh/.credentials.yaml"; do
    if [ -f "$f" ]; then shasum -a 256 "$f" 2>/dev/null; else echo "absent $f"; fi
  done
}
_REAL_BEFORE="$(_real_fingerprint)"

save_cfg "$CODEX_CFG"
"$RMLX" agents codex --setup --base-url "$B/v1" >/dev/null 2>&1
patch_port "$CODEX_CFG"
save_cfg "$HERMES_CFG"
"$RMLX" agents hermes --setup --base-url "$B/v1" >/dev/null 2>&1
patch_port "$HERMES_CFG"
save_cfg "$DSH_CFG"
# --yes: dsh's setup is the interactive plan/apply flow, and this gate has no
# tty to answer the confirmation prompt with.
"$RMLX" agents dsh --setup --yes --base-url "$B/v1" >/dev/null 2>&1
patch_port "$DSH_CFG"

if [ "$(_real_fingerprint)" != "$_REAL_BEFORE" ]; then
  echo "SMOKE-ABORT: \`agents --setup\` modified the operator's REAL config despite" >&2
  echo "             the redirect. Refusing to continue. Before / after:" >&2
  printf '%s\n' "$_REAL_BEFORE" | sed 's/^/               was  /' >&2
  _real_fingerprint | sed 's/^/               now  /' >&2
  echo "             A renamed home_env in ~/.rapid-mlx/agents/ is the usual cause." >&2
  exit 3
fi

echo "running 5 Tier-1 agents serially (budget ${AGENT_TO}s, hermes ${HERMES_TO}s)…"
# Serial, NOT backgrounded: overlapping them oversubscribes the single GPU and
# flakes the gate (see the rationale above run_claude). Each runs to completion
# — with its own retries + per-agent timeout — before the next starts, so it
# gets the whole GPU and finishes at solo speed. Never a bare `wait` here: the
# serve is still backgrounded (SERVE_PID), so a bare wait would hang the gate.
run_claude
run_codex
run_hermes
run_aider
run_dsh

# Restore the config files only after every agent has finished using them.
restore_cfg "$CODEX_CFG"
restore_cfg "$HERMES_CFG"
restore_cfg "$DSH_CFG"

R_CLAUDE=$(cat "$WORK/claude.result" 2>/dev/null || echo FAIL)
R_CODEX=$(cat "$WORK/codex.result" 2>/dev/null || echo FAIL)
R_HERMES=$(cat "$WORK/hermes.result" 2>/dev/null || echo FAIL)
R_AIDER=$(cat "$WORK/aider.result" 2>/dev/null || echo FAIL)
R_DSH=$(cat "$WORK/dsh.result" 2>/dev/null || echo FAIL)

# ---- report --------------------------------------------------------------
echo
echo "== Tier-1 re-verification ($ALIAS) =="
printf "  claude-code  %s\n" "$R_CLAUDE"
printf "  codex-cli    %s\n" "$R_CODEX"
printf "  hermes       %s\n" "$R_HERMES"
printf "  aider        %s\n" "$R_AIDER"
printf "  dsh          %s\n" "$R_DSH"

for r in "$R_CLAUDE" "$R_CODEX" "$R_HERMES" "$R_AIDER" "$R_DSH"; do
  [ "$r" = PASS ] || { echo "RESULT: FAIL — a Tier-1 agent regressed; this blocks the release."; exit 1; }
done

# ---- release-gate extras (opt-in): coherence + perf on the SAME warm serve ----
# Runs ONLY when RAPID_MLX_RELEASE_GATE=1 (set by agent-gate.yml on the release
# path). Unset by default, so this block is skipped and the smoke behaves
# byte-for-byte as before — the local gauntlet and every other caller are
# unaffected. It reuses the model that is already loaded and warm from the
# agentic run above (SERVE_PID still listening), so there is NO second model load.
#
# This closes the L2 gap where the release gate proved the agentic path but never
# checked that the flagship model still gives coherent answers or hasn't regressed
# in speed — the exact class (#1247/#1234) that shipped as garbage in 0.11.4.
if [ "${RAPID_MLX_RELEASE_GATE:-0}" = "1" ]; then
  GATE_PY="$VENV/bin/python"
  echo
  echo "== release-gate: deep coherence (#1247) + perf on warm $ALIAS serve =="

  # Deep coherence golden (#1247) against the warm flagship model. Fail-closed:
  # a wrong/incoherent golden answer blocks the release.
  if ! "$GATE_PY" "$REPO_ROOT/evals/coherence_gate.py" --base-url "$B/v1"; then
    echo "RESULT: FAIL — $ALIAS coherence gate failed; this blocks the release."
    exit 1
  fi

  # Decode-throughput perf gate on the warm serve. Advisory (measure + print)
  # unless a reviewed floor is set via RAPID_MLX_PERF_MIN_TPS — never a fabricated
  # baseline. Exit codes: 0 = passed or advisory (INCLUDING a failed measurement
  # in advisory mode — nothing is being enforced, so a flaky request must not
  # take the release down); 1 = below the reviewed floor; 2 = a floor IS set but
  # the run could not be measured, which fails closed because "unable to verify"
  # is not "verified".
  if ! "$GATE_PY" "$REPO_ROOT/evals/perf_gate.py" --base-url "$B/v1"; then
    echo "RESULT: FAIL — $ALIAS perf regressed below the reviewed floor; blocks the release."
    exit 1
  fi
  echo "== release-gate extras PASSED (coherence + perf) =="
fi

echo "RESULT: PASS — all five Tier-1 agents verified on $ALIAS."
