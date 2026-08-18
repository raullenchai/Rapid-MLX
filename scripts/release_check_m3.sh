#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# M3-local release gauntlet — every gate that needs a live
# `rapid-mlx serve`. Sibling to the CI-side gates which run
# automatically on every PR (pr-validate.yml) and on bump PRs
# (release-preflight.yml).
#
# Invoked by `make release-check-m3` (which sets MODEL + PY env vars
# from the Makefile). Standalone: `bash scripts/release_check_m3.sh`.
#
# Exit codes:
#   0 — all M3-only gates green
#   1 — a gate failed (output above pinpoints which)
#   2 — pre-flight refusal (port in use, server didn't come up)
#
# The script intentionally fails-fast — a single gate fail stops the
# rest because subsequent gates would mostly be testing the same
# broken inference path. To run gates piecemeal, invoke them directly
# (see docs/development/releasing.md §"Pre-release validation
# gauntlet").

set -euo pipefail

MODEL="${MODEL:-qwen3.5-9b-4bit}"
PY="${PY:-python3.12}"
PORT="${PORT:-8000}"

# G7b's consolidated `bench --tier harness` runs 6 harness profiles
# (codex/opencode/hermes/aider/langchain/deepseek-harness) under ONE shared
# per-profile cap
# (tier_runner's HARNESS_PROFILE_TIMEOUT_S, library default 300s). That default
# is sized for a single fast harness — codex/opencode/aider/langchain each
# finish <135s on the 9B gauntlet model — but the hermes profile runs 20+
# serial agentic tests (~740s). 300s kills hermes mid-profile and false-fails
# the release gate. Raise the cap for the whole gauntlet: exporting here means
# every bench subprocess this script spawns later inherits it (the G7b harness
# at the `--tier harness` call AND the G12 random sweep) — intended, and can
# only make those gates more lenient, never stricter. The tier_runner library
# default (300s) is left untouched for any non-gauntlet / standalone caller.
# See knowledge/release-check-m3-tuning-backlog. Env-overridable.
export HARNESS_PROFILE_TIMEOUT_S="${HARNESS_PROFILE_TIMEOUT_S:-1200}"

LOG=/tmp/release-check-m3.log
PIDFILE=/tmp/release-check-m3.pid

# Base URL the gauntlet expects every child gate to hit. Threaded from
# $PORT so `PORT=8011 bash scripts/release_check_m3.sh` correctly aims
# every gate at 127.0.0.1:8011 — including gates that read the URL from
# env instead of accepting --base-url / --port.
#
# Issue #974: previously, only G5/G6/G7b/G9/G12 threaded $PORT (via
# --port / --base-url / hardcoded "http://127.0.0.1:$PORT" URLs). The
# G7 SDK integration tests (Anthropic / pydantic_ai / smolagents /
# langchain / hermes) read the URL from ``RAPID_MLX_BASE_URL`` with a
# hardcoded default of ``http://localhost:8000/v1``. With
# ``PORT=8011``, the SDK tests silently hit whatever was on 8000 —
# typically the operator's production server — reporting either false
# failures (wrong model served) or, worse, false PASSes (prod happens
# to answer). Export the URL here so every downstream block inherits
# it regardless of how the child gate resolves its endpoint.
#
# We also export the OpenAI-SDK conventional sibling env vars
# (OPENAI_BASE_URL / OPENAI_API_BASE) as a defensive belt-and-braces
# for any future integration test that follows the OpenAI SDK
# convention instead of RAPID_MLX_BASE_URL.
export RAPID_MLX_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"

line() { printf '%s\n' "============================================================"; }

line
echo "  M3 release gauntlet"
echo "  model:    $MODEL"
echo "  python:   $PY"
echo "  port:     $PORT"
echo "  base_url: $RAPID_MLX_BASE_URL"
echo "  log:      $LOG"
line

# `lsof` decides, here and at the G12 handoff, whether a port is free —
# and "free" is the one answer that must never be guessed. A PATH without
# /usr/sbin makes every check silently answer "free": the `if` below swallows
# the 127, the whole gauntlet runs, and only the last gate notices. Refuse now.
command -v lsof >/dev/null 2>&1 \
  || { echo "ERROR: lsof is required (port checks would silently pass)." >&2; exit 2; }

# `lsof` with a per-invocation watchdog.
#   0 = something is listening   1 = nothing is   2 = could not find out
# A stuck `lsof` would otherwise stretch any "wait N seconds" loop that calls
# it into as long as it likes, because the deadline is only read between calls.
# `-nP` also removes the name/port resolution stalls that cause most of them.
# Callers must treat 2 as "not free": an unverifiable port is not a free one.
port_busy() {
  local port="$1" limit="${2:-10}" pid n=0
  lsof -nP -i ":$port" >/dev/null 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null && [ "$n" -lt "$((limit * 10))" ]; do
    sleep 0.1
    n=$((n + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    # Kills the pid we spawned. If `lsof` were a wrapper script that forked a
    # helper, the helper would outlive this — real `/usr/sbin/lsof` is a single
    # process, so that only matters if someone shadows it, and a shadowed lsof
    # is already the scenario the `command -v` pre-flight cannot defend.
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "  ! lsof did not return within ${limit}s for port $port" >&2
    return 2
  fi
  local rc=0
  wait "$pid" || rc=$?
  # lsof exits 1 for "nothing matched" and anything else for "I failed".
  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
}

# Pre-flight: refuse if port is busy so we don't accidentally murder
# someone's debug server.
#
# Through `port_busy`, not a bare `lsof`: an executable `lsof` that hangs or
# exits non-zero satisfies the `command -v` check above and then reads as
# "port free" to a bare `if`. The gauntlet would boot its own server, fail to
# bind, and run every gate against whatever was already there — which is the
# contamination this check exists to prevent, arriving through the check
# itself. Only an explicit 1 (ran, found nothing) is free.
port_state=0
port_busy "$PORT" 10 || port_state=$?
if [ "$port_state" != 1 ]; then
  if [ "$port_state" = 0 ]; then
    echo "ERROR: port $PORT already in use — kill the existing server first." >&2
    lsof -nP -i ":$PORT" >&2 || true
  else
    echo "ERROR: could not determine whether port $PORT is free — refusing to" >&2
    echo "  start rather than run every gate against someone else's server." >&2
  fi
  exit 2
fi

# Pre-flight: refuse if ANOTHER rapid-mlx serve / pr_validate is running on this
# box. A serve-based gauntlet needs EXCLUSIVE GPU + ports. A concurrent workload
# will (a) fight over ports — its own :8000 port-management can SIGTERM this
# server mid-gate — and (b) GPU-contend every request, inflating latency enough
# to false-fail the timeout-sensitive agentic gates. The `lsof -i :$PORT` check
# above MISSES this: a concurrent server still LOADING its model hasn't bound the
# port yet, so the port reads false-free at our start, then the two collide once
# it binds. Detect the processes directly. (A real contamination that wasted a
# full validation cycle — see knowledge/release-check-m3-tuning-backlog §E.)
# Override with RAPID_MLX_ALLOW_CONCURRENT=1 if you are certain the other
# workload won't touch the GPU or our ports.
#
# Detection must catch BOTH invocation forms a concurrent serve can take:
#   * ``python -m vllm_mlx.cli serve`` — the ``-m`` module form (the gauntlet's
#     own subprocesses use this; excluded via the release_check_m3 filter below).
#   * ``.../bin/rapid-mlx serve`` — the installed console-script entry point. A
#     real DeepSeek serve running as ``.venv/bin/rapid-mlx serve ... --port 8765``
#     slipped past the old ``vllm_mlx\.cli``-only pattern and GPU-contended a
#     gauntlet into a false-failed G12 (2026-07-30). Match the console form too.
#
# The console pattern is anchored ``(^|[ /])rapid-mlx`` so it matches whether the
# entry point sits at a path (``.venv/bin/rapid-mlx``) or at the very start of the
# ``ps`` row (a bare ``rapid-mlx serve``). ``ps -Aww -o pid=,command=`` is used
# (not ``-o pid,command``) so long argv/env lines are not truncated — a serve
# hidden behind a long wrapper/env prefix would otherwise defeat this safety gate.
if [ "${RAPID_MLX_ALLOW_CONCURRENT:-0}" != "1" ]; then
  _concurrent=$(ps -Aww -o pid=,command= 2>/dev/null \
    | grep -E 'vllm_mlx\.cli (serve|bench)|(^|[ /])rapid-mlx (serve|bench)|scripts\.pr_validate' \
    | grep -Ev 'grep|release_check_m3|coherence_sweep|wait-then-validate' || true)
  if [ -n "$_concurrent" ]; then
    echo "ERROR: another rapid-mlx serve / pr_validate is running on this box." >&2
    echo "  The release gauntlet needs EXCLUSIVE GPU + ports: a concurrent run will" >&2
    echo "  SIGTERM this server mid-gate and GPU-contend every request (false fails)." >&2
    echo "  Stop it first, or set RAPID_MLX_ALLOW_CONCURRENT=1 to override. Found:" >&2
    echo "$_concurrent" | sed 's/^/    /' >&2
    exit 2
  fi
fi

cleanup() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  # The concurrent correctness cluster stages per-gate logs/rc under a
  # mktemp dir; remove it too so an abort mid-cluster doesn't leak /tmp.
  #
  # `if`, not `[ … ] && rm`: this function is ALSO called inline before G12,
  # where its return status is load-bearing under `set -e`. As an `&&` list it
  # returns 1 whenever CLUSTER_WORK is empty — which it always is by then, the
  # cluster having cleared it — and the gauntlet died there instead of running
  # the gate.
  if [ -n "${CLUSTER_WORK:-}" ]; then
    rm -rf "$CLUSTER_WORK"
  fi
  if [ -n "${RELEASE_AGENT_HOME_ROOT:-}" ]; then
    rm -rf "$RELEASE_AGENT_HOME_ROOT"
  fi
}
trap cleanup EXIT INT TERM

# G7b and G12 drive real agent CLIs. Their config is part of the test fixture,
# not operator state: inheriting ~/.codex or ~/.hermes also inherits personal
# plugins, MCP servers, skills, and stale model settings. On #1683 that turned
# Codex's normal 13-tool / 5.5K-token request into 134 tools / 33K tokens; each
# prefill took ~110s and a healthy file-read loop false-timed out at 120s.
# Redirect both config homes for a deterministic gate and protect the user's
# real agent config from setup writes. This allocation deliberately comes
# AFTER the EXIT trap is installed, and after early preflight refusals, so
# subsequent failures clean it up. cleanup's inline pre-G12 call clears it;
# setup recreates it for the independent random sweep.
RELEASE_AGENT_HOME_ROOT="$(mktemp -d)"
export CODEX_HOME="$RELEASE_AGENT_HOME_ROOT/codex"
export HERMES_HOME="$RELEASE_AGENT_HOME_ROOT/hermes"

# Fail before downloading/booting models if a benchmark candidate has no
# committed comparison point. Staleness is surfaced as a warning; refreshing a
# baseline remains a reviewed human decision, never an automatic overwrite.
"$PY" scripts/release_baselines.py

#-------------------- G0a fleet output coherence ------------------
# Before spending time on the full single-model gauntlet, prove that every
# release-family representative can still answer deterministic golden
# questions. The shared manifest expands this list automatically when an MLX
# dependency changed since the previous release tag (or RELEASE_FLEET_BASE_REF).
line
echo "  G0a — release-fleet output coherence sweep"
line
PY="$PY" PORT="${FLEET_PORT:-8402}" \
  FLEET_SCOPE="${FLEET_SCOPE:-auto}" \
  bash scripts/coherence_sweep.sh

echo "→ Starting server (background)…"
# --no-thinking: gauntlet's job is API/parser/router correctness, not
# thinking-mode evaluation. Leaving thinking ON on small models burns
# the per-test token budget on `<think>` blocks before useful text
# (pydantic_ai's 2048 cap reliably tripped on qwen3.5-4b-4bit) and
# on chained-tool tests confuses the final-answer turn (qwen3.5-9b-4bit
# re-narrates the problem after the second tool result). Thinking
# coverage belongs to a separate evaluation suite, not the release gate.
#
# NOTE: prefix cache is left ENABLED (engine default). Disabling it here was
# tried to fix serve-readiness creep (a large persisted cache loads synchronously
# at startup before /v1/models flips ready), but that also forces the heavy
# multi-turn agentic gate (G7b hermes) to reprocess its growing per-test contexts
# and risks false timeouts. The readiness-creep fix belongs in the engine — defer
# the disk load off the readiness path (tracked in #1350) so it benefits all serve
# users without slowing the gates.
$PY -m vllm_mlx.cli serve "$MODEL" --port "$PORT" --no-thinking > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

echo "→ Waiting for server (max 60s)…"
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  server up ($MODEL)"
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "ERROR: server did not respond within 60s. Last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 2
fi

#-------------------- G0b booted-model coherence ------------------
# The most fundamental gate: does the served model produce coherent,
# correct text at all? Qwen3.6/3.5-35B shipped garbage from the first
# token (#1234) and passed every perf / import / unit gate because
# nothing checked generation coherence (#1247). Run this FIRST — if it
# fails, the stress / SDK gates below would just re-test the same broken
# inference path. Blocking = deterministic golden answers; the garbage
# detector is advisory-only. Reads RAPID_MLX_BASE_URL (exported above).
#
# G0a covers the release fleet; this second check covers the exact MODEL used
# by every remaining live-server gate below.
line
echo "  G0b — gauntlet-model output coherence gate"
line
"$PY" evals/coherence_gate.py

#-------------------- G5 stress -----------------------------------
line
echo "  G5 — make stress (8 scenarios incl. tool storm)"
line
"$PY" scripts/dev_test.py stress --port "$PORT"

#============ G7 + G7b + G6 — concurrent correctness cluster =======
# These three sections are all output-CORRECTNESS gates: every assertion
# is per-request-local (SDK response shape, /v1/responses event ordering,
# a single-request parallel-cap count), not a wall-clock/latency or
# whole-server-state measurement. So they are safe to run CONCURRENTLY
# against the one already-booted batching server — BatchedEngine merges
# their requests on the single GPU, and a request merely waiting its turn
# changes none of the assertions. Serially they cost ~sum; concurrently
# they collapse to ~max, which is G7b's `bench --tier harness` (the hermes
# profile runs 20+ serial agentic tests, ~740s — the long pole every other
# cluster gate hides under).
#
# Kept OUT of this cluster on purpose: G0b runs serial FIRST (coherence
# fail-fast — #1247), and G5 stress / G9 latency stay SOLO because their
# verdicts depend on an uncontended server. G8 (parser microbench) is a
# pure-CPU timing-threshold gate and stays serial too, so background CPU
# contention can't inflate its us/call.
#
# Concurrency mechanics (mirrors tests/integrations/agent_smoke.sh):
#   - each gate runs as a background job writing its stdout+stderr to its
#     own log and its exit status to its own .rc under $CLUSTER_WORK
#     (background subshells can't export vars back to the parent);
#   - the gate BODIES are byte-for-byte the serial versions, wrapped in an
#     inner `( set -e … )` subshell so they still fail-fast internally
#     exactly as before; the outer function runs `set +e` so capturing the
#     rc is never itself aborted;
#   - we `wait` on the FIVE explicit agent PIDs, never a bare `wait` — a
#     bare `wait` would also block on the still-running background serve
#     ($PIDFILE) and hang the gauntlet forever after the gates finish.

# Fail-loud assertion (issue #974): every cluster gate that resolves its
# endpoint from ``RAPID_MLX_BASE_URL`` must hit the gauntlet's own server.
# If the top-of-script export is ever regressed / clobbered / disabled,
# bail out here rather than silently pointing the SDK tests at whatever
# server the default URL lands on (typically the operator's production
# 8000). Checked once, before launch, since it guards all cluster gates.
_expected_base="http://127.0.0.1:${PORT}/v1"
if [ "${RAPID_MLX_BASE_URL:-}" != "$_expected_base" ]; then
  echo "ERROR: cluster env mismatch — RAPID_MLX_BASE_URL='${RAPID_MLX_BASE_URL:-}' expected '$_expected_base'." >&2
  echo "  This means the G7 SDK tests would hit a different server than the gauntlet booted." >&2
  exit 1
fi

CLUSTER_WORK="$(mktemp -d)"

# --- G7 SDK integration (three tests, each its own job) ---
run_g7_anthropic() {
  set +e
  ( set -e; "$PY" tests/integrations/test_anthropic_sdk.py ) \
    > "$CLUSTER_WORK/g7_anthropic.log" 2>&1
  echo $? > "$CLUSTER_WORK/g7_anthropic.rc"
}
run_g7_pydantic() {
  set +e
  ( set -e; "$PY" tests/integrations/test_pydantic_ai_full.py ) \
    > "$CLUSTER_WORK/g7_pydantic.log" 2>&1
  echo $? > "$CLUSTER_WORK/g7_pydantic.rc"
}
# smolagents — tests 3+4 will 422 by design under tool_choice=required
# strict enforcement (PR #518 behavior). Tests 1+2 are CodeAgent format
# expectations that small models hallucinate. Run for the contract
# coverage but DON'T fail the gauntlet on its expected failures — the
# `|| true` keeps it informational.
run_g7_smol() {
  set +e
  ( "$PY" tests/integrations/test_smolagents_full.py || true ) \
    > "$CLUSTER_WORK/g7_smol.log" 2>&1
  echo $? > "$CLUSTER_WORK/g7_smol.rc"
}

# --- G7b agent harness layer (the long pole) ---
# Two-part gate.
#
# Part A — `bench --tier harness`: smoke-tests `/v1/chat/completions`
# parser/router for the six first-class harnesses (codex / opencode /
# hermes / aider / langchain / deepseek-harness). Doesn't touch
# `/v1/responses` (the runner
# only knows Chat Completions today). Consolidated in PR #2 of the
# bench-tier series: a single `bench --tier harness` call replaces the
# prior five sequential `agents <name> --test` invocations — same
# coverage, one process, one summary block. `--base-url` attaches it to
# the gauntlet's already-booted server on $PORT instead of booting its
# own (which would conflict on port + waste model-load time). Exit-code
# contract: exits 1 iff any test failed or errored, so the inner
# `set -e` aborts this gate on the first failure. Don't `|| true` it; a
# quiet skip means a missed release gate.
#
# Part B — direct `/v1/responses` curl probes: AgentTestRunner has zero
# coverage of the Responses shim (added in v0.7.10 for Codex). A
# non-stream probe + two SSE probes catch route-level regressions
# (missing event, wrong status, broken usage payload, developer-role
# passthrough, suppressed function_call). If the shim regresses, Codex
# CLI users get "stream closed before response.completed" with no other
# signal.
run_g7b() {
  set +e
  (
    set -e
    echo "  Part A: bench --tier harness (chat-completions smoke for all 6 first-class harnesses)"
    "$PY" -m vllm_mlx.cli bench "$MODEL" --tier harness \
      --base-url "http://127.0.0.1:$PORT"

    echo
    echo "  Part B: /v1/responses curl probe (non-stream + SSE)"

    # Non-stream — verifies route reachable, response shape correct.
    ns_body=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/responses" \
      -H 'Content-Type: application/json' \
      -d '{"model": "gpt-5", "input": "Reply with the single word: ok", "stream": false, "max_output_tokens": 16}')
    if ! echo "$ns_body" | grep -q '"object":"response"'; then
      echo "G7b non-stream FAIL: missing response object" >&2
      echo "  body: $ns_body" >&2
      exit 1
    fi
    if ! echo "$ns_body" | grep -qE '"status":"(completed|incomplete)"'; then
      echo "G7b non-stream FAIL: missing completed/incomplete status" >&2
      echo "  body: $ns_body" >&2
      exit 1
    fi
    echo "    non-stream: OK"

    # SSE — verifies the events Codex parses fire in the right order
    # (response.created → ... → response.completed). The event Codex
    # treats as hardest failure is missing `response.completed`.
    sse=$(mktemp)
    curl -sNf -X POST "http://127.0.0.1:$PORT/v1/responses" \
      -H 'Content-Type: application/json' \
      -d '{"model": "gpt-5", "input": "Reply with the single word: ok", "stream": true, "max_output_tokens": 16}' > "$sse"
    for evt in "response.created" "response.completed"; do
      if ! grep -q "event: $evt" "$sse"; then
        echo "G7b SSE FAIL: missing event '$evt'" >&2
        head -20 "$sse" >&2
        rm -f "$sse"
        exit 1
      fi
    done
    # Verify completed lands AFTER created (basic ordering sanity).
    created_line=$(grep -n "event: response.created" "$sse" | head -1 | cut -d: -f1)
    completed_line=$(grep -n "event: response.completed" "$sse" | head -1 | cut -d: -f1)
    if [ -z "$created_line" ] || [ -z "$completed_line" ] || [ "$completed_line" -le "$created_line" ]; then
      echo "G7b SSE FAIL: response.completed not after response.created (created@$created_line, completed@$completed_line)" >&2
      exit 1
    fi
    rm -f "$sse"
    echo "    SSE: OK (response.created → response.completed)"

    # Part B.2 — codex-shape SSE: input[] + developer role + tool definition.
    # The bare-string `input` probe above only exercises the easy code path
    # (`input` → single user message) and missed THREE production regressions
    # at once on Codex CLI 0.136.0:
    #   1. `developer`-role items passed through verbatim → Qwen template
    #      raised `Unexpected message role.`
    #   2. After role mapping, multiple system messages tripped Qwen's
    #      "System message must be at the beginning." check
    #   3. tool_call XML was suppressed by tool_filter but the post-loop
    #      parser was reading the FILTERED text, so no `response.function_call`
    #      event ever emitted — Codex's agent loop terminated silently
    #
    # This probe exercises the codex-shape input + asserts a function_call
    # item gets emitted (the hardest signal — covers all three regressions
    # at once because a missing event 0 / 1 / 2 all result in zero items).
    sse2=$(mktemp)
    # Wrap in `if !` so `set -e` doesn't kill the gate on a transport
    # failure before the diagnostic block + cleanup can run.
    if ! curl -sNf -X POST "http://127.0.0.1:$PORT/v1/responses" \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "gpt-5",
        "stream": true,
        "max_output_tokens": 128,
        "instructions": "You are a helpful agent.",
        "input": [
          {"type": "message", "role": "user", "content": "Call get_weather with city=SF"},
          {"type": "message", "role": "developer", "content": "Always use the tool when asked."}
        ],
        "tools": [
          {"type": "function", "name": "get_weather", "description": "Get the weather for a city",
           "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
        ],
        "tool_choice": "required"
      }' > "$sse2"; then
      echo "G7b codex-shape SSE FAIL: curl to /v1/responses errored — server crashed or rejected the codex-shape request" >&2
      head -30 "$sse2" >&2
      rm -f "$sse2"
      exit 1
    fi
    for evt in "response.created" "response.output_item.added" "response.completed"; do
      if ! grep -q "event: $evt" "$sse2"; then
        echo "G7b codex-shape SSE FAIL: missing event '$evt' — codex agent loop would silently terminate" >&2
        head -30 "$sse2" >&2
        rm -f "$sse2"
        exit 1
      fi
    done
    # Function-call item is the strongest signal — without it Codex sees a
    # turn.completed with zero items and the agent loop ends with no output.
    # Parse SSE properly: pair each `event:` line with its `data:` payload
    # and assert at least one `response.output_item.added` carries an item
    # with `type == "function_call"`. Whole-file grep is unsafe — a text
    # delta containing the literal string `"type":"function_call"` would
    # spuriously pass without any function-call item ever being emitted.
    if ! python3 - "$sse2" <<'PY'
import json, sys
path = sys.argv[1]
event = None
ok = False
for raw in open(path, encoding="utf-8", errors="replace"):
    line = raw.rstrip("\n")
    if line.startswith("event:"):
        event = line[6:].strip()
    elif line.startswith("data:") and event == "response.output_item.added":
        try:
            payload = json.loads(line[5:].strip())
        except ValueError:
            continue
        item = payload.get("item") or {}
        if item.get("type") == "function_call":
            ok = True
            break
    elif line == "":
        event = None
sys.exit(0 if ok else 1)
PY
    then
      echo "G7b codex-shape SSE FAIL: no response.output_item.added with item.type=function_call — codex agent loop would terminate with zero items" >&2
      head -30 "$sse2" >&2
      rm -f "$sse2"
      exit 1
    fi
    rm -f "$sse2"
    echo "    SSE (codex-shape): OK (function_call item emitted)"
  ) > "$CLUSTER_WORK/g7b.log" 2>&1
  echo $? > "$CLUSTER_WORK/g7b.rc"
}

# --- G6 parallel_tool_calls=false cap (PR #518 fix path) ---
run_g6() {
  set +e
  (
    set -e
    tmp_indices=$(mktemp)
    curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{
        \"model\": \"$MODEL\",
        \"stream\": true,
        \"parallel_tool_calls\": false,
        \"tool_choice\": \"required\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Get weather for SF AND NY\"}],
        \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}]
      }" | grep -oE '"index":[0-9]+' | sort -u > "$tmp_indices"
    distinct=$(wc -l < "$tmp_indices")
    echo "  distinct tool_call indices: $distinct"
    if [ "$distinct" -ne 1 ]; then
      echo "G6 FAIL: parallel cap leaked $distinct tool_calls (expected 1)" >&2
      cat "$tmp_indices" >&2
      rm -f "$tmp_indices"
      exit 1
    fi
    rm -f "$tmp_indices"
  ) > "$CLUSTER_WORK/g6.log" 2>&1
  echo $? > "$CLUSTER_WORK/g6.rc"
}

line
echo "  G7 + G6 — light correctness cluster (concurrent on the one :$PORT server)"
echo "    running: Anthropic SDK · pydantic_ai · smolagents · parallel-cap"
echo "    (per-request shape/count assertions — contention-insensitive, safe to overlap)"
line

run_g7_anthropic & _p_g7a=$!
run_g7_pydantic  & _p_g7p=$!
run_g7_smol      & _p_g7s=$!
run_g6           & _p_g6=$!

# Wait ONLY on the four LIGHT cluster jobs (never a bare `wait`; see header). The
# `|| true` keeps `set -e` from aborting on a failed job before the per-gate
# diagnostics below run — the real verdict of each gate is in its .rc file.
wait "$_p_g7a" "$_p_g7p" "$_p_g7s" "$_p_g6" || true

# G7b (bench --tier harness) runs SOLO on an uncontended GPU. Its per-test
# assertions are TIMEOUT-based multi-step agentic round-trips, so overlapping it
# with other GPU consumers slows each request past its timeout and false-fails
# (observed on a clean box: codex e2e_file_read + hermes code_with_tests TIMEOUT
# under 5-way contention, G7b 1359s vs ~740s solo). The four gates above assert
# only per-request response shape/count, which contention cannot change, so they
# overlap safely; G7b cannot. Run it after the light cluster releases the GPU.
#
# Call in a subshell: run_g7b flips `set +e` (like every gate fn). The four light
# gates are backgrounded with `&`, which already subshells them, so their `set +e`
# never escapes; a FOREGROUND run_g7b would instead leak `set +e` into the
# post-cluster gates (G9 latency / G8 parser / G12) and silently disable their
# fail-fast. `( … )` isolates the option change, matching the `&` gates' semantics.
( run_g7b )

_rc() { cat "$CLUSTER_WORK/$1.rc" 2>/dev/null || echo 1; }
cluster_fail=0

# G7b — the long pole; always surface its full output (the harness summary
# + /v1/responses probe results are informative even on PASS).
line
echo "  G7b — agent harness layer (codex / opencode / hermes / aider / langchain + /v1/responses probe)"
line
cat "$CLUSTER_WORK/g7b.log"
if [ "$(_rc g7b)" = 0 ]; then
  echo "  ✓ G7b PASS"
else
  echo "  ✗ G7b FAIL (rc=$(_rc g7b))"
  cluster_fail=1
fi

echo
if [ "$(_rc g7_anthropic)" = 0 ]; then
  echo "  ✓ G7 Anthropic SDK PASS"
else
  echo "  ✗ G7 Anthropic SDK FAIL — full output:"
  cat "$CLUSTER_WORK/g7_anthropic.log"
  cluster_fail=1
fi

if [ "$(_rc g7_pydantic)" = 0 ]; then
  echo "  ✓ G7 pydantic_ai PASS"
else
  echo "  ✗ G7 pydantic_ai FAIL — full output:"
  cat "$CLUSTER_WORK/g7_pydantic.log"
  cluster_fail=1
fi

# smolagents is informational (expected partial fail on small models) —
# always show its output, never fail the gauntlet on it.
echo "  • G7 smolagents (informational; expected partial fail on 4B) — output:"
cat "$CLUSTER_WORK/g7_smol.log"

if [ "$(_rc g6)" = 0 ]; then
  echo "  ✓ G6 parallel_tool_calls=false cap PASS"
else
  echo "  ✗ G6 parallel_tool_calls=false cap FAIL — full output:"
  cat "$CLUSTER_WORK/g6.log"
  cluster_fail=1
fi

rm -rf "$CLUSTER_WORK"
CLUSTER_WORK=""
if [ "$cluster_fail" != 0 ]; then
  echo "ERROR: correctness-cluster gate(s) failed — see per-gate output above." >&2
  exit 1
fi

#-------------------- G9 latency 10-seq ---------------------------
line
echo "  G9 — 10-sequential latency"
line
"$PY" <<EOF
import json
import time
import urllib.request

url = "http://127.0.0.1:$PORT/v1/chat/completions"
results = []
for i in range(10):
    body = json.dumps({
        "model": "$MODEL",
        "messages": [{"role": "user", "content": f"List 5 facts about prime {i+2}."}],
        "max_tokens": 80,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dt = time.time() - t0
    ct = resp.get("usage", {}).get("completion_tokens", 0)
    tps = ct / dt if dt > 0 else 0
    results.append(tps)
    print(f"  [{i+1:2d}/10] {ct:3d} tok in {dt:5.2f}s -> {tps:6.1f} tok/s")

mean = sum(results) / len(results)
spread = max(results) - min(results)
print(f"\nmean={mean:.1f} spread={spread:.1f} (first-run cold cache excluded from variance)")
EOF

#-------------------- G8 parser microbench ------------------------
line
echo "  G8 — parser microbench (extract_tool_calls × 10000)"
line
"$PY" scripts/microbench_parsers.py

#-------------------- G8b decode-throughput perf gate -------------
# The end-to-end perf regression gate (docs/development/releasing.md row
# G8b): measures the served model's steady-state decode tokens/sec on the
# STILL-WARM gauntlet server and — when a reviewed floor exists for MODEL in
# harness/perf_floors.json — FAILS the release if decode regressed. This is
# the gate that catches KV-cache / hot-path throughput regressions (the very
# class of change in a bump PR full of MTP / prefix-cache / speculative-decode
# edits) that every correctness gate above sails past.
#
# It runs HERE, before the G12 cleanup tears the server down, so it reuses the
# same warm serve (no second model load). perf_gate.py boots nothing; it reads
# RAPID_MLX_BASE_URL (exported at the top) to reach 127.0.0.1:$PORT.
#
# Floor provenance: perf_floors.json is a REVIEWED HUMAN DECISION, never
# invented (same rule as release_baselines.py). With no committed floor for
# MODEL the gate stays ADVISORY — it prints the decode tok/s and passes — so a
# fresh model doesn't hard-fail before anyone has reviewed a number; commit the
# floor (see the file's _comment) to turn enforcement on. A one-off run can
# override with RAPID_MLX_PERF_MIN_TPS=<n> or --min-tps.
line
echo "  G8b — decode-throughput perf gate (enforces harness/perf_floors.json[$MODEL] when set)"
line
"$PY" evals/perf_gate.py --alias "$MODEL" --floors-file harness/perf_floors.json

#-------------------- G12 random-coverage -------------------------
# Randomized sweep across small/medium aliases × harnesses × rounds.
# Catches model-specific regressions that the fixed gauntlet (one
# model — qwen3.5-9b-4bit) by construction cannot see. PR #687
# (gemma-4 ``<|tool_call>`` wire-marker leak) is exactly the kind of
# bug this gate would have caught at release time instead of after.
#
# Seed: today's UTC date YYYYMMDD → reproducible per release day.
# Cleanup: each sampled model's HF cache is removed after testing so
# successive release cuts don't fill the disk.
#
# Set RAPID_MLX_SKIP_G12=1 to skip this gate (e.g. when iterating on
# the bump PR itself without re-running the full sweep). The CI-side
# preflight ALSO runs G1+G10+G11, so a single local skip-of-G12 still
# leaves multiple gates covering the bump-PR diff.
if [ "${RAPID_MLX_SKIP_G12:-0}" = "1" ]; then
  line
  echo "  G12 — random-coverage [SKIPPED via RAPID_MLX_SKIP_G12=1]"
  line
else
  line
  echo "  G12 — random-coverage (sampled models × harnesses × rounds)"
  line
  # Free the gauntlet's main server so G12 owns the port + GPU.
  # ``cleanup`` is the trap-installed teardown defined at the top of
  # this script — calling it manually here releases the PID file too, so
  # read the PID before calling it.
  OLD_SERVER_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  # `kill -0` succeeds on a ZOMBIE — a child that has exited but whose status
  # the shell has not collected still owns a pid, and it owns no GPU. Waiting
  # for one to "go away" burns the whole deadline and then SIGKILLs a corpse,
  # turning a clean shutdown into a gate failure. Read the state instead.
  # Non-blocking on purpose: a plain `wait` would hang forever on the shutdown
  # this deadline exists to catch.
  old_server_alive() {
    [ -n "$OLD_SERVER_PID" ] || return 1
    kill -0 "$OLD_SERVER_PID" 2>/dev/null || return 1
    # `kill -0` succeeded, so SOMETHING is there. Only a zombie is safely gone:
    # it has exited and released the GPU, it just has not been reaped. Every
    # other answer — a live state, or a `ps` that failed or printed nothing —
    # has to count as ALIVE. Reading "I could not tell" as "gone" hands the GPU
    # to G12 while the old server may still be flushing weights, which is the
    # exact overlap this two-condition handoff exists to prevent.
    local state
    state="$(ps -o stat= -p "$OLD_SERVER_PID" 2>/dev/null)"
    # Trim. `ps -o stat=` right-aligns into a column whose width comes from the
    # widest state on the machine, so a zombie can arrive as " Z+" rather than
    # "Z+". Matching `Z*` against the padded string fails, the zombie reads as
    # ALIVE, and the handoff burns its whole 60 s and fails — which is the
    # false failure this helper was written to remove.
    state="${state#"${state%%[![:space:]]*}"}"
    case "$state" in
      Z*) return 1 ;;
      *)  return 0 ;;
    esac
  }
  cleanup
  sleep 2
  # Wait for the old server to be GONE, not merely for the port to look
  # free. uvicorn closes its listening socket before lifespan shutdown
  # completes — and shutdown is where PR #667's deadline-aware prefix-cache
  # flush runs — so a free port is not yet an idle GPU. G12 loads its own
  # model next; overlapping that with a process still holding weights is how
  # a gauntlet earns a metal::malloc "Resource limit" that reads like a code
  # bug. Both conditions, then, and neither on its own.
  #
  # Wall-clock bounded, not iteration-bounded: a slow or stuck `lsof` would
  # otherwise stretch this into as long as it likes. The bound is not exact —
  # the deadline is read BETWEEN operations, so a check starting just under it
  # can still run its own 5s watchdog out, and the 2s settle above is on top.
  # Roughly a minute, never unbounded, which is the property that matters.
  server_gone=0
  handoff_deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$handoff_deadline" ]; do
    if old_server_alive; then
      sleep 1
      continue
    fi
    # 0 = still listening, 2 = could not find out. Only an explicit 1 — a
    # check that ran and said nothing is there — releases the handoff; an
    # unverifiable port is not a free one.
    port_state=0
    port_busy "$PORT" 5 || port_state=$?
    if [ "$port_state" != 1 ]; then
      sleep 1
      continue
    fi
    server_gone=1
    break
  done
  # Handing over an occupied port is worse than stopping here. G12 boots its
  # own server and waits for :$PORT to answer; whatever is still sitting
  # there answers first, and the sweep would benchmark a stranger under the
  # sampled alias's name. G12 checks that the listener is its own child, so
  # this would be caught — but failing here says WHY, instead of timing out
  # while looking healthy.
  if [ "$server_gone" != 1 ]; then
    echo "  ✗ the gauntlet's server did not exit and free port $PORT in ~60s" >&2
    echo "    refusing to hand the port and the GPU to G12" >&2
    [ -n "$OLD_SERVER_PID" ] && ps -p "$OLD_SERVER_PID" >&2 || true
    lsof -i ":$PORT" >&2 || true
    # `cleanup` already TERM'd it and deleted the pidfile, so the EXIT trap
    # can no longer reach it: exiting here would leave the process alive with
    # the GPU allocated, poisoning the retry this failure is telling us to
    # run. Kill it for real before giving up.
    # Only if it is still OUR child. The pid came from `$!`, but a cleanly
    # exited and reaped server frees that number, and SIGKILLing whatever
    # inherited it would be a destructive answer to a problem we no longer
    # have. `ps -o ppid=` is the cheap identity check available here.
    old_ppid="$(ps -o ppid= -p "$OLD_SERVER_PID" 2>/dev/null | tr -d ' ' || true)"
    if old_server_alive && [ "$old_ppid" = "$$" ]; then
      echo "    SIGKILLing $OLD_SERVER_PID so the retry starts on a free GPU" >&2
      kill -9 "$OLD_SERVER_PID" 2>/dev/null || true
      # It IS a child of this shell (started with `&` above), so this reaps it
      # rather than returning 127; SIGKILL cannot be caught, so it cannot hang.
      wait "$OLD_SERVER_PID" 2>/dev/null || true
    fi
    exit 1
  fi
  "$PY" scripts/release_check_m3_random.py \
    --port "$PORT" \
    --models "${G12_MODELS:-2}" \
    --harnesses "${G12_HARNESSES:-2}" \
    --rounds "${G12_ROUNDS:-3}"
  # No --report: the script defaults it into its own 0700 run directory and
  # prints the path. A fixed /tmp name is a symlink the next local process can
  # aim wherever it likes.
fi

#-------------------- Done ----------------------------------------
line
echo "  release-check-m3: ALL gates green for $MODEL"
echo "  Now safe to push the chore: bump version to X.Y.Z commit."
line
