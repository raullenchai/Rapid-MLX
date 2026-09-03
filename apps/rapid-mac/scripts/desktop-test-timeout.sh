#!/usr/bin/env bash
# Desktop test-suite hang backstop.
#
# Wraps `swift test --no-parallel` so a hung suite can never burn the whole
# job timeout (20 minutes, once 45) with no diagnostic. Two complementary
# mechanisms (see #2488):
#
#   1. Per-suite `.timeLimit(.minutes(2))` (TestTimeouts.hangProne) applied to
#      the hang-prone suites fails those suites in 2 minutes and names them.
#   2. THIS script is the whole-run safety net: it enforces a per-run deadline
#      on the swift-test process and, on expiry, hands the hung process to the
#      RapidDesktopTestWatchdogRun watchdog which samples its stacks (plus
#      `ps` / `vm_stat` / `memory_pressure`) into a `.txt` artifact, then we
#      kill the hung process and exit non-zero with a readable reason.
#
# The per-run deadline must sit ABOVE the healthy serialized run so a healthy
# suite never expires (per-suite .timeLimit provides the real 2-minute
# guarantee); override with RAPID_DESKTOP_DEADLINE_MINUTES if a host runs the
# suite slower/faster. Default 15 min: generously above any healthy serialized
# run while still 5 min below the job's 20-min timeout AND producing a sample
# artifact (which a plain job timeout never does). The suite-level .timeLimit
# is what actually delivers the 2-minute fail-fast for the hang-prone suites.
#
# Artifacts are written to RAPID_DESKTOP_HANG_ARTIFACT_DIR (default
# build/desktop-hang-artifacts, match CI's upload step path if you move it).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEADLINE_MINUTES="${RAPID_DESKTOP_DEADLINE_MINUTES:-15}"
ARTIFACT_DIR="${RAPID_DESKTOP_HANG_ARTIFACT_DIR:-"$ROOT/build/desktop-hang-artifacts"}"

# Locate (and if needed build) the watchdog executable BEFORE launching
# `swift test`, so we never contends with the running test for the SwiftPM
# `.build` lock. Prefer RAPID_WATCHDOG_BIN (CI can pass a prebuilt path); fall
# back to `swift build --product`.
WATCHDOG_BIN="${RAPID_WATCHDOG_BIN:-}"
if [[ -n "$WATCHDOG_BIN" && ! -x "$WATCHDOG_BIN" ]]; then
    echo "error: RAPID_WATCHDOG_BIN not executable: $WATCHDOG_BIN" >&2
    WATCHDOG_BIN=""
fi
if [[ -z "$WATCHDOG_BIN" ]] && swift build --product RapidDesktopTestWatchdogRun >/dev/null 2>&1; then
    WATCHDOG_BIN="$ROOT/.build/debug/RapidDesktopTestWatchdogRun"
fi

mkdir -p "$ARTIFACT_DIR"

# --- Launch `swift test` in the background, remember its PID ---------------
swift test --no-parallel &
TEST_PID=$!

# --- Watchdog: fail fast if the run outlives the deadline while still alive --
hang_detected=0
if [[ -n "$WATCHDOG_BIN" && -x "$WATCHDOG_BIN" ]]; then
    deadline_seconds=$(( DEADLINE_MINUTES * 60 ))
    if "$WATCHDOG_BIN" "$TEST_PID" "$deadline_seconds" "$ARTIFACT_DIR" \
        > "$ARTIFACT_DIR/watchdog.stdout" 2> "$ARTIFACT_DIR/watchdog.stderr"; then
        # Watchdog exited 0 => the wrapped process exited before the deadline.
        :
    else
        hang_detected=1
    fi
else
    # No watchdog binary: fall back to a plain `wait` (no fast-fail guarantee,
    # but the job's own timeout-minutes still bounds the burn).
    echo "warning: RapidDesktopTestWatchdogRun not built; running without hang backstop" >&2
fi

# --- Reap the test process; on hang, kill it so nothing is orphaned ---------
if [[ $hang_detected -eq 1 ]]; then
    echo "::error::Desktop test suite exceeded ${DEADLINE_MINUTES} min hard deadline; killing hung process $TEST_PID" >&2
    kill -9 "$TEST_PID" 2>/dev/null || true
    # Also kill any swift-testing/xctest children that outlived the kill.
    pkill -9 -P "$TEST_PID" 2>/dev/null || true
    # Surface the artifact path the watchdog wrote.
    if [[ -d "$ARTIFACT_DIR" ]]; then
        echo "hang sample artifact(s) under: $ARTIFACT_DIR" >&2
        ls -1 "$ARTIFACT_DIR" >&2 || true
    fi
    wait "$TEST_PID" 2>/dev/null || true
    exit 124   # matches timeout(1)'s kill exit code convention
fi

# Normal path: propagate the test process's own exit code.
set +e
wait "$TEST_PID"
TEST_EXIT=$?
set -e
# Avoid stale hang artifacts confusing the upload step on a healthy run.
if [[ -d "$ARTIFACT_DIR" && -n "${ARTIFACT_DIR}" && "${ARTIFACT_DIR}" != "/" ]]; then
    # Keep the dir (CI `if-no-files-found: ignore` needs a stable path) but
    # clear it so an empty upload shows no false hang.
    rm -rf -- "${ARTIFACT_DIR:?}"/* 2>/dev/null || true
fi
exit "$TEST_EXIT"
