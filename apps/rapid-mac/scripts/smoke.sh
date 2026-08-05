#!/usr/bin/env bash
# smoke.sh — fast (sub-2s) end-to-end smoke for Rapid.app.
#
# Compiles the app and checks the fake rapid-mlx honours the CLI contract
# ``ModelCatalog`` depends on, so a code change can be verified in seconds
# without standing up a real model.
#
# What it covers:
#   * swift build (the package compiles)
#   * fake-rapid-mlx CLI contract: `models` / `ls` / `info` print and exit;
#     an unknown verb exits rather than falling through to the server
#
# What it does NOT cover:
#   * The Swift Testing unit suite — the SPM test target was stripped
#     (see Package.swift), so `swift test` finds no tests / can't build.
#   * The chat lifecycle (ServerManager state machine, ChatStreamClient SSE
#     decode). The `RAPID_TEST_DRIVER` hook this script used to drive does
#     not exist in `Sources/` — see the note at the foot of this file.
#   * Real model inference (use ``RAPID_BIN=/opt/homebrew/bin/rapid-mlx``
#     or unset RAPID_BIN for that)
#   * SwiftUI view tree (Step 2 — ViewInspector)
#   * Visual fidelity (Step 3 — snapshot testing)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> swift build"
# Compile the package up front. This is the verification the old ``swift test``
# line was meant to provide but couldn't (no test target → exit 1 on a clean
# tree, killing the smoke before it ever reached the chat lifecycle below).
swift build >/dev/null

echo
echo "==> fake rapid-mlx CLI contract"
# ``ModelCatalog`` shells out to the rapid-mlx CLI for its catalog, and the
# fake stands in for that binary. Assert the fake honours the same
# print-and-exit contract the real one does, because when it does NOT the
# failure is invisible and expensive: the fake used to ignore its subcommand
# and start the HTTP server for EVERY verb, so ``runRapidMlx(args: ["models"])``
# spawned a child that never exited, both pipe drainers blocked on an EOF that
# could not arrive, and the caller deadlocked — while a stray listener squatted
# :8000 for the next PortSweep to reap along with any real rapid-mlx the
# developer had running there.
#
# Each probe is bounded so a regression fails in seconds naming the offending
# verb instead of hanging the script (and CI) indefinitely.
FAKE="$ROOT/scripts/fake-rapid-mlx.sh"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-10}"

# Validate the override before it is used as a loop bound. `[ x -ge y ]` on a
# non-integer fails every iteration, and because `run_bounded` is called from
# an `if !` condition `errexit` is suppressed there — so a typo like
# `PROBE_TIMEOUT_S=10s` would silently disable the watchdog and let a hung
# child run forever, which is precisely what this script exists to prevent.
case "$PROBE_TIMEOUT_S" in
    ''|*[!0-9]*)
        echo "PROBE_TIMEOUT_S must be a positive integer number of seconds (got '$PROBE_TIMEOUT_S')" >&2
        exit 2
        ;;
esac
if [ "$PROBE_TIMEOUT_S" -lt 1 ]; then
    echo "PROBE_TIMEOUT_S must be >= 1 (got '$PROBE_TIMEOUT_S')" >&2
    exit 2
fi

# Bounded run, built from shell builtins only.
#
# Deliberately NOT `timeout(1)`: that is GNU coreutils, which stock macOS does
# not ship (no /usr/bin/timeout, no BSD equivalent). Depending on it would make
# this script exit 127 on any Mac without `brew install coreutils` — i.e. the
# guard against hangs would itself be the thing that always fails.
#
# Also deliberately SIGKILL rather than coreutils' default SIGTERM: a regressed
# fake that installs a SIGTERM handler (or blocks in a signal-unsafe spot) could
# ignore a polite signal and hang the watchdog — reintroducing the exact failure
# this is here to bound. Nothing here needs a graceful shutdown.
#
# Writes combined output to $2. Returns 124 on timeout, else the child's status.
run_bounded() {
    run_bounded_secs="$1"
    run_bounded_out="$2"
    shift 2
    "$@" >"$run_bounded_out" 2>&1 &
    run_bounded_pid=$!
    run_bounded_waited=0
    while kill -0 "$run_bounded_pid" 2>/dev/null; do
        if [ "$run_bounded_waited" -ge "$run_bounded_secs" ]; then
            # Signal the JOB, not the pid. A raw `kill -9 "$pid"` races pid
            # recycling: the probe can exit and be reaped between the liveness
            # test and the signal, by which point the number may belong to an
            # unrelated process. `%%` resolves through bash's job table, which
            # refuses ("no such job") once the job is gone — so the worst case
            # is a no-op instead of killing a bystander.
            kill -9 %% 2>/dev/null || true
            # `wait` by pid is safe even if it is stale: wait never signals.
            wait "$run_bounded_pid" 2>/dev/null || true
            return 124
        fi
        sleep 1
        run_bounded_waited=$((run_bounded_waited + 1))
    done
    # By pid, not `%%` — bash drops the job from its table as soon as it is
    # reaped, and for a fast child that happens before we get here, so a
    # jobspec `wait` would fail with "no such job" and lose the exit status.
    wait "$run_bounded_pid"
}

probe_out="$(mktemp -t rapid-smoke-probe)"

# Reap any still-running probe on the way out, however we leave.
#
# Why this matters: a regressed fake is a *server*. Non-interactive bash starts
# asynchronous commands with SIGINT ignored and Python inherits that
# disposition, so Ctrl-C kills this script while the child keeps listening on
# :8000 — and the next Rapid launch's PortSweep then reaps whatever holds that
# port, which may be the developer's own rapid-mlx. Cleanup therefore has to
# cover interrupts, not just a normal exit.
#
# Why a jobspec rather than a `probe_child` variable holding `$!`: publishing
# the pid is not atomic. A signal can be delivered between `cmd &` and the
# assignment on the next line, running the trap while the variable is still
# empty and leaking exactly the orphan this is meant to prevent. A stored pid
# is also unsafe to signal later — once bash has reaped the job the number can
# be recycled to an unrelated process, and cleanup runs twice (INT/TERM calls
# it, then `exit` re-triggers it via EXIT). A jobspec has neither problem: it is
# resolved against bash's job table at signal time, exists from the moment the
# job is created, and is refused outright ("no such job") once reaped.
#
# Known residual: `%%` is positional, so it names "the most recent job", and
# this script assumes that is the probe. That holds unless the shell already
# had a job — a non-interactive bash sources `$BASH_ENV`, so a startup file
# that backgrounds a helper would leave one. Matching by command text
# (`%?fake-rapid-mlx`) does NOT fix this: bash records a job's UNEXPANDED
# source text, which here is the literal `"$@"`, so the pattern can never
# match (verified on 3.2.57). The alternatives are worse — a `$!` variable
# reintroduces the publication race, and gating the kill on that variable
# means leaking the orphan whenever the race is lost. Killing at most one
# stray background job in a dev script, under a shell configuration this repo
# does not use, is the better trade.
cleanup() {
    kill -9 %% 2>/dev/null || true
    rm -f "$probe_out"
}
trap cleanup EXIT
# 128 + signal number, the conventional shell exit status for a signalled run.
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

probe_exits() {
    verb="$1"; shift
    expect="$1"; shift
    if ! run_bounded "$PROBE_TIMEOUT_S" "$probe_out" "$FAKE" "$verb" "$@"; then
        echo "FAIL — '$verb' did not exit cleanly within ${PROBE_TIMEOUT_S}s (it must print and exit, never serve)"
        head -10 "$probe_out"
        exit 1
    fi
    if ! grep -q "$expect" "$probe_out"; then
        echo "FAIL — '$verb' output did not contain '$expect'"
        head -10 "$probe_out"
        exit 1
    fi
    echo "    ok  $verb"
}
probe_exits models "fake-alias"
probe_exits ls "fake-org/fake-repo"
probe_exits info "Alias: fake-alias" fake-alias
# Default-deny: an unknown verb must still exit rather than fall through to
# the server, so a future ModelCatalog subcommand cannot resurrect the hang.
if ! run_bounded "$PROBE_TIMEOUT_S" "$probe_out" "$FAKE" some-future-verb; then
    echo "FAIL — unknown verb did not exit within ${PROBE_TIMEOUT_S}s"
    exit 1
fi
echo "    ok  unknown verb exits (default-deny)"

echo
echo "OK — package compiles, fake CLI contract holds"
# NOT covered here: the chat lifecycle (ServerManager state machine +
# ChatStreamClient SSE decode). This script used to drive it via
# ``RAPID_TEST_DRIVER``, but no such hook exists in ``Sources/`` — it was part
# of the subsystem strip and only the caller survived the monorepo import
# (#1406), so that invocation launched a plain GUI app that never exited and
# hung the script forever. Reinstating the harness is tracked separately;
# until then this script does not pretend to cover it.
