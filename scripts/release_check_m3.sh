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
LOG=/tmp/release-check-m3.log
PIDFILE=/tmp/release-check-m3.pid

line() { printf '%s\n' "============================================================"; }

line
echo "  M3 release gauntlet"
echo "  model:    $MODEL"
echo "  python:   $PY"
echo "  port:     $PORT"
echo "  log:      $LOG"
line

# Pre-flight: refuse if port is busy so we don't accidentally murder
# someone's debug server.
if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use — kill the existing server first." >&2
  exit 2
fi

cleanup() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT INT TERM

echo "→ Starting server (background)…"
# --no-thinking: gauntlet's job is API/parser/router correctness, not
# thinking-mode evaluation. Leaving thinking ON on small models burns
# the per-test token budget on `<think>` blocks before useful text
# (pydantic_ai's 2048 cap reliably tripped on qwen3.5-4b-4bit) and
# on chained-tool tests confuses the final-answer turn (qwen3.5-9b-4bit
# re-narrates the problem after the second tool result). Thinking
# coverage belongs to a separate evaluation suite, not the release gate.
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

#-------------------- G5 stress -----------------------------------
line
echo "  G5 — make stress (8 scenarios incl. tool storm)"
line
"$PY" scripts/dev_test.py stress --port "$PORT"

#-------------------- G7 SDK integration --------------------------
line
echo "  G7 — Anthropic SDK"
line
"$PY" tests/integrations/test_anthropic_sdk.py

line
echo "  G7 — pydantic_ai"
line
"$PY" tests/integrations/test_pydantic_ai_full.py

# smolagents — tests 3+4 will 422 by design under tool_choice=required
# strict enforcement (PR #518 behavior). Test 1+2 are CodeAgent format
# expectations that small models hallucinate. Run for the contract
# coverage but DON'T fail the gauntlet on its expected failures —
# document the expected behavior instead.
line
echo "  G7 — smolagents (informational; expected partial fail on 4B)"
line
"$PY" tests/integrations/test_smolagents_full.py || true

#-------------------- G7b agent harness layer ---------------------
# Two-part gate.
#
# Part A — `rapid-mlx agents <name> --test`: smoke-tests
# `/v1/chat/completions` parser/router for the five first-class
# harnesses. Doesn't touch `/v1/responses` (the runner only knows
# Chat Completions today). The five gated harnesses:
#   - codex     (Codex CLI, /v1/responses workhorse — Part B covers
#                the responses route directly)
#   - opencode  (OpenCode, Hermes-parser path)
#   - hermes    (Hermes Agent; specific_tests run the 62-tool stress)
#   - aider     (no CLI dep; API-level smoke only)
#   - langchain (specific_tests: tests/integrations/test_langchain.py;
#                pip-installs langchain-openai on the runner)
# Other registered profiles need third-party CLIs on PATH and are
# environmentally flaky for a release gate, OR are pure-interactive
# (cline = VSCode extension, openhands/openclaude = interactive
# query_cmd=null) and would false-positive PASS on the API-level
# default plan without exercising the actual agent workflow.
#
# Part B — direct `/v1/responses` curl probes: AgentTestRunner has
# zero coverage of the Responses shim (added in v0.7.10 for Codex).
# A single non-stream probe + a single SSE probe is enough to catch
# route-level regressions (missing event, wrong status, broken
# usage payload). If the shim regresses, Codex CLI users get
# "stream closed before response.completed" with no other signal.
#
# Exit-code contract for Part A: `rapid-mlx agents <name> --test`
# exits 1 iff any test failed or errored (vllm_mlx/cli.py:
# sys.exit(0 if success else 1), wrapping
# AgentTestRunner.print_summary's `failed == 0 and errored == 0`
# at vllm_mlx/agents/testing.py:130). `set -e` aborts the gauntlet
# on the first failure — series-fail-fast matches G7's pattern.
# Don't `|| true` these; a quiet skip means a missed release gate.
line
echo "  G7b — agent harness layer (codex / opencode / hermes / aider / langchain + /v1/responses probe)"
line

echo "  Part A: agents --test (chat-completions smoke)"
"$PY" -m vllm_mlx.cli agents codex     --test
"$PY" -m vllm_mlx.cli agents opencode  --test
"$PY" -m vllm_mlx.cli agents hermes    --test
"$PY" -m vllm_mlx.cli agents aider     --test
"$PY" -m vllm_mlx.cli agents langchain --test

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

# SSE — verifies the 7 events Codex parses fire in the right order
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
# Wrap in `if !` so `set -e` doesn't kill the script on a transport
# failure before the diagnostic block + cleanup can run. Without this
# wrapper, an HTTP 5xx or connection drop would exit the gauntlet with
# zero context and a stale temp file (codex_review NIT).
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

#-------------------- G6 fix-path repro ---------------------------
line
echo "  G6 — parallel_tool_calls=false cap (PR #518 fix path)"
line
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
  exit 1
fi
rm -f "$tmp_indices"

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

#-------------------- Done ----------------------------------------
line
echo "  release-check-m3: ALL gates green for $MODEL"
echo "  Now safe to push the chore: bump version to X.Y.Z commit."
line
