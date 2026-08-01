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

# G7b's consolidated `bench --tier harness` runs 5 harness profiles
# (codex/opencode/hermes/aider/langchain) under ONE shared per-profile cap
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

# Pre-flight: refuse if port is busy so we don't accidentally murder
# someone's debug server.
if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use — kill the existing server first." >&2
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
  [ -n "${CLUSTER_WORK:-}" ] && rm -rf "$CLUSTER_WORK"
}
trap cleanup EXIT INT TERM

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
# parser/router for the five first-class harnesses (codex / opencode /
# hermes / aider / langchain). Doesn't touch `/v1/responses` (the runner
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
    echo "  Part A: bench --tier harness (chat-completions smoke for all 5 first-class harnesses)"
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
        "max_output_tokens": 64,
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
  # this script — calling it manually here releases the PID file too.
  cleanup
  sleep 2
  # Wait for the port to actually free (cleanup sends TERM; the server
  # then runs PR #667's deadline-aware prefix-cache flush on shutdown).
  for _ in $(seq 1 10); do
    if ! lsof -i ":$PORT" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  "$PY" scripts/release_check_m3_random.py \
    --port "$PORT" \
    --models "${G12_MODELS:-2}" \
    --harnesses "${G12_HARNESSES:-2}" \
    --rounds "${G12_ROUNDS:-3}" \
    --report /tmp/release-check-m3-random.log
fi

#-------------------- Done ----------------------------------------
line
echo "  release-check-m3: ALL gates green for $MODEL"
echo "  Now safe to push the chore: bump version to X.Y.Z commit."
line
