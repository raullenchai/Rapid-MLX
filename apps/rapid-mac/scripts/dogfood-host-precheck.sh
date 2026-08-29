#!/usr/bin/env bash
# Fail before a GUI lane starts on a locked/sleeping host; keep it awake while it runs.
set -euo pipefail

die() { printf 'dogfood-host-precheck: FAIL: %s\n' "$*" >&2; exit 1; }

if [[ "${CI:-}" == "true" && "${RAPID_HOST_SAFETY_TESTING:-0}" != "1" ]]; then
    [[ "${1:-}" == "--" ]] && shift && exec "$@"
    exit 0
fi
if [[ "${RAPID_HOST_SAFETY_TESTING:-0}" == "1" ]]; then
    locked="${RAPID_HOST_TEST_LOCKED:-false}"
    idle="${RAPID_HOST_TEST_IDLE_TIME:-0}"
else
    [[ "$(uname -s)" == "Darwin" ]] || die "macOS is required"
    locked="$(/usr/sbin/ioreg -n Root -d1 -a | /usr/bin/plutil -extract IOConsoleLocked raw - 2>/dev/null || printf true)"
    idle="$(/usr/bin/defaults -currentHost read com.apple.screensaver idleTime 2>/dev/null || printf missing)"
fi
[[ "$locked" == "false" || "$locked" == "0" ]] || die "console is locked"
[[ "$idle" == "0" ]] || die "screensaver idleTime must be 0 (got $idle)"

if [[ "${1:-}" == "--" ]]; then
    shift
    [[ $# -gt 0 ]] || die "missing command after --"
    # Watch this shell's PID, then replace the shell with the lane command.
    # The assertion ends with that PID, while exec preserves the PID expected
    # by AX tooling and process-owned cleanup.
    /usr/bin/caffeinate -dimsu -w $$ >/dev/null 2>&1 &
    exec "$@"
fi
printf 'dogfood-host-precheck: PASS (unlocked, idleTime=0)\n'
