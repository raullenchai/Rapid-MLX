#!/bin/bash
# Aider CLI integration harness against a running `rapid-mlx serve`.
#
# What it proves:
#   1. Aider's REPL can connect to rapid-mlx's OpenAI-compatible endpoint.
#   2. Aider's SEARCH/REPLACE edit-and-write pipeline actually rewrites a
#      local file the way we asked ("fix the bug in add.py — add, not
#      subtract").
#
# Why the correctness signal is NOT OpenAI tool_calls: Aider does not
# use function-calling. It sends the file + user instruction as plain
# messages, expects the LLM to emit ``SEARCH ... REPLACE ...`` blocks,
# and applies those edits locally. So the pass gate is whether the
# executed ``add()`` really returns ``a + b`` after aider exits — not a
# simple grep, which would flake if the model adds an extra line or
# reformats the body.
#
# Usage:
#   test_aider.sh --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>]
#
# ``--base-url`` takes the full ``http[s]://host:port/v1`` URL and is the
# preferred form — it lets the Python wrapper pass whatever URL the
# ``rapid_mlx_server`` fixture is actually pointed at (which may be
# non-localhost in CI shards or a remote-serve run). ``--port`` is kept
# for standalone local invocations and defaults host to ``127.0.0.1``.
#
# Env vars (set automatically, but overridable):
#   HOME              — overridden to a scratch dir so aider's config /
#                       cache / analytics files don't touch the operator's
#                       real ``~/.aider*`` state
#   AIDER_BIN         — full path to the aider binary; skipped-search if set
#   AIDER_ANALYTICS_ASKED=1
#   AIDER_CHECK_UPDATE=false
#
# Exit codes:
#   0  — aider completed and add(2, 3) == 5 in the rewritten file
#   1  — arg parse / setup error (also emitted when ``timeout`` / ``gtimeout``
#        is missing — see the timeout-selection block below)
#   2  — aider CLI exited non-zero
#   3  — aider ran but the file wasn't corrected (add(2, 3) != 5 — edit
#        format didn't apply; SEARCH/REPLACE parse failed; LLM refused;
#        wrong operator, etc.)
#   4  — timeout

# ``set -e`` so an unchecked setup step (``mktemp``, ``cd``, ``mkdir``)
# aborts the harness instead of running aider from the wrong directory.
# Codex #1047 round-2 finding #4: without this, a failed ``cd $WORKDIR``
# would silently run aider from the caller's CWD and (worse) potentially
# rewrite an unrelated ``add.py`` there.
set -euo pipefail

TIMEOUT=300
MODEL=""
PORT=""
BASE_URL=""
VERBOSE=0

usage() {
    echo "Usage: $0 --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>] [-v]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || { [ -z "$BASE_URL" ] && [ -z "$PORT" ]; }; then
    usage
fi

# Derive BASE_URL from --port only if --base-url wasn't given (back-compat
# for the standalone invocation shape kept for local docs/dev). --base-url
# wins so a Python wrapper that always passes the full URL is authoritative.
if [ -z "$BASE_URL" ]; then
    BASE_URL="http://127.0.0.1:${PORT}/v1"
fi

# Locate aider without ever triggering a fresh install. Operators that
# install aider outside PATH point AIDER_BIN at the binary; otherwise we
# discover it on PATH. No machine-specific path is baked in.
AIDER_BIN="${AIDER_BIN:-}"
if [ -z "$AIDER_BIN" ] || [ ! -x "$AIDER_BIN" ]; then
    AIDER_BIN="$(command -v aider 2>/dev/null || true)"
    if [ -z "$AIDER_BIN" ]; then
        echo "ERROR: aider binary not found (checked AIDER_BIN and PATH)" >&2
        exit 1
    fi
fi

# ``python3`` powers the correctness check below (AST parse + runtime
# ``add(2, 3) == 5`` assertion). Fail early with a clear message so a
# broken system Python doesn't get diagnosed as an aider failure.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH — required for correctness check" >&2
    exit 1
fi

# Pick a timeout wrapper. Codex #1047 round-2 finding #3: the previous
# "no coreutils timeout, PID-kill the background subshell" fallback only
# signalled the subshell, not the aider grandchild, so a timed-out run
# could leak a running ``aider``/LiteLLM process past harness exit. We
# now REQUIRE ``gtimeout`` (macOS coreutils) or ``timeout`` (Linux),
# both of which reliably kill the whole exec'd tree — and fail setup
# fast otherwise.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --preserve-status --kill-after=10 "$TIMEOUT")
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --preserve-status --kill-after=10 "$TIMEOUT")
else
    echo "ERROR: neither 'timeout' nor 'gtimeout' available — install " >&2
    echo "       coreutils (macOS: 'brew install coreutils') before running." >&2
    exit 1
fi

# Scratch state — HOME override so aider drops its config / cache into a
# throw-away tree we can nuke on exit. Codex #1047 round-2 finding #5
# (nit): both scratch dirs now go through ``mktemp -d`` so we never
# rm -rf a predictable ``/tmp/aider-test-home-$$`` path.
SCRATCH_HOME="$(mktemp -d -t aider-test-home.XXXXXX)"
WORKDIR="$(mktemp -d -t aider-test-work.XXXXXX)"

cleanup() {
    local rc=$?
    if [ "$VERBOSE" -eq 0 ]; then
        rm -rf "$WORKDIR" "$SCRATCH_HOME" 2>/dev/null || true
    else
        echo "VERBOSE: preserved WORKDIR=$WORKDIR SCRATCH_HOME=$SCRATCH_HOME" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

# Toy file with an obvious bug: subtraction masquerading as addition.
# Pass gate = LLM must emit an edit block that flips ``- b`` → ``+ b``
# so that ``add(2, 3) == 5`` (see correctness check below).
cat > "$WORKDIR/add.py" <<'PYEOF'
def add(a, b):
    return a - b  # BUG
PYEOF

# Sanity: is the server actually up? A quick /v1/models probe with a
# 5 s timeout catches "operator forgot to boot serve" instantly instead
# of eating the 300 s aider timeout.
if ! curl -sS -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: rapid-mlx server not reachable at $BASE_URL" >&2
    exit 1
fi

# Aider needs LiteLLM's ``openai/`` prefix to route through the
# OpenAI-compatible chat completions path — without it LiteLLM tries
# to pick a provider from the alias string and fails on non-canonical
# rapid-mlx aliases.
LITELLM_MODEL="openai/${MODEL}"

echo "[test_aider.sh] model=$MODEL base_url=$BASE_URL timeout=${TIMEOUT}s"
echo "[test_aider.sh] litellm-model=$LITELLM_MODEL"
echo "[test_aider.sh] scratch home=$SCRATCH_HOME workdir=$WORKDIR"
echo "[test_aider.sh] BEFORE add.py:"
cat "$WORKDIR/add.py"
echo "--------"

# Run aider one-shot (``--message`` runs a single round then exits). We
# deliberately pass every "quiet, don't touch the network, don't pollute
# the operator's box" flag we can find:
#   --no-git             — don't require a git repo; don't create commits
#   --no-analytics       — skip PostHog analytics
#   --no-check-update    — skip pip-index poll
#   --no-show-model-warnings — model isn't in aider's known list; silence
#   --no-pretty          — plain text, no ANSI (easier to grep on failure)
#   --no-stream          — Rapid-MLX supports streaming, but non-stream is
#                          less flaky on slow local inference
#   --map-tokens 0       — don't burn a turn building a repo map
#   --yes-always         — take all prompts as "yes"
LOG="$WORKDIR/aider.log"
STATUS=0

# We disable ``set -e`` for the aider invocation so a non-zero exit
# (timeout, LLM refusal, transport blip) is captured into $STATUS instead
# of aborting the harness before we can print the diagnostic tail.
set +e
cd "$WORKDIR" || { echo "ERROR: cd $WORKDIR failed" >&2; exit 1; }
HOME="$SCRATCH_HOME" \
AIDER_ANALYTICS_ASKED=1 \
AIDER_CHECK_UPDATE=false \
OPENAI_API_BASE="$BASE_URL" \
OPENAI_API_KEY="rapidmlx" \
"${TIMEOUT_CMD[@]}" \
"$AIDER_BIN" \
    --model "$LITELLM_MODEL" \
    --openai-api-base "$BASE_URL" \
    --openai-api-key "rapidmlx" \
    --no-git \
    --no-analytics \
    --no-check-update \
    --no-show-model-warnings \
    --no-pretty \
    --no-stream \
    --map-tokens 0 \
    --yes-always \
    --message "Fix the bug in add.py — this function should add, not subtract. Change the '-' operator to '+' in the return statement." \
    add.py \
    >"$LOG" 2>&1
STATUS=$?
set -e

# Detect timeout: 124 = coreutils timeout (SIGTERM path); 137 = --kill-after
# escalation (SIGKILL); 143 = SIGTERM. All three mean "we killed it, not
# aider exiting cleanly with a non-zero code."
if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ] || [ "$STATUS" -eq 143 ]; then
    echo "[test_aider.sh] TIMEOUT after ${TIMEOUT}s" >&2
    echo "--- last 60 lines of aider log ---" >&2
    tail -60 "$LOG" >&2 || true
    exit 4
fi

echo "[test_aider.sh] aider exit=$STATUS"
echo "--- last 40 lines of aider log ---"
tail -40 "$LOG" || true
echo "--------"
echo "[test_aider.sh] AFTER add.py:"
cat "$WORKDIR/add.py"
echo "--------"

if [ "$STATUS" -ne 0 ]; then
    echo "[test_aider.sh] FAIL: aider exited $STATUS" >&2
    exit 2
fi

# Correctness check. Codex #1047 round-2 finding #2: the previous
# ``grep -qE '^\s*return\s+a\s*\+\s*b'`` gate was
# (a) non-portable — ``\s`` is a Perl-regex escape, not ERE-POSIX, and
#     BSD grep's -E happily ignores it, silently matching literal
#     ``\s`` sequences instead of whitespace — meaning the same aider
#     output could pass on GNU grep but fail on macOS grep at CI.
# (b) semantically weak — a whole-file grep would pass if aider left
#     ``add()`` broken but added ``return a + b`` in a helper or a
#     docstring elsewhere in the file.
#
# Fix: import the rewritten module and execute ``add(2, 3) == 5``.
# The scratch file was written by us (see the here-doc above) and
# only mutated in-place by aider's SEARCH/REPLACE — no arbitrary code
# is introduced, so a subprocess ``python3 -c 'import add; ...'`` is
# safe. We also require the AST to contain a ``BinOp(op=Add)`` inside
# the ``add`` function body, so a mischievous refactor like
# ``def add(a, b): return sum([a, b])`` still passes (semantic add)
# but a hard-coded ``return 5`` would not.
if ! python3 - "$WORKDIR" <<'PYEOF'
import ast
import importlib.util
import sys

workdir = sys.argv[1]
target = f"{workdir}/add.py"

with open(target) as fh:
    source = fh.read()

# Load the module.
spec = importlib.util.spec_from_file_location("_aider_test_add", target)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as exc:  # noqa: BLE001 — diagnose whatever it was
    print(f"[correctness] MODULE-LOAD-ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

if not hasattr(mod, "add") or not callable(mod.add):
    print("[correctness] MODULE-SHAPE-ERROR: add() missing or not callable",
          file=sys.stderr)
    sys.exit(1)

# Runtime check.
try:
    got = mod.add(2, 3)
except Exception as exc:  # noqa: BLE001
    print(f"[correctness] RUNTIME-ERROR: add(2, 3) raised {exc!r}",
          file=sys.stderr)
    sys.exit(1)

if got != 5:
    print(f"[correctness] VALUE-ERROR: add(2, 3) returned {got!r}, expected 5",
          file=sys.stderr)
    sys.exit(1)

# Structural check — reject ``return 5`` and other hard-code cheats.
tree = ast.parse(source)
funcs = [n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "add"]
if not funcs:
    print("[correctness] AST-ERROR: no def add() in module", file=sys.stderr)
    sys.exit(1)

has_add_op = False
for func in funcs:
    for node in ast.walk(func):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            has_add_op = True
            break
        # sum([a, b]) / operator.add(a, b) also qualify as "semantic add".
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(
                getattr(node.func, "attr", None), "__str__", lambda: ""
            )()
            if name in ("sum", "add"):
                has_add_op = True
                break

if not has_add_op:
    print("[correctness] AST-ERROR: add() body has no + BinOp / sum(...) call",
          file=sys.stderr)
    sys.exit(1)

print("[correctness] OK: add(2, 3) == 5, AST contains + BinOp")
sys.exit(0)
PYEOF
then
    echo "[test_aider.sh] FAIL: add.py correctness check failed" >&2
    echo "--- final add.py ---" >&2
    cat "$WORKDIR/add.py" >&2
    exit 3
fi

echo "[test_aider.sh] PASS: add(2, 3) == 5 after aider rewrite"
exit 0
