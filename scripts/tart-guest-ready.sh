#!/usr/bin/env bash
# Wait until Tart's guest agent can execute commands, with a bounded deadline.
set -euo pipefail

timeout=120
interval=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) timeout="$2"; shift 2 ;;
        --interval) interval="$2"; shift 2 ;;
        -h|--help) echo "usage: tart-guest-ready.sh [--timeout SECONDS] [--interval SECONDS] VM"; exit 0 ;;
        --*) echo "tart-guest-ready: unknown option $1" >&2; exit 2 ;;
        *) vm="$1"; shift ;;
    esac
done
[[ -n "${vm:-}" ]] || { echo "tart-guest-ready: VM is required" >&2; exit 2; }
command -v tart >/dev/null || { echo "tart-guest-ready: tart is not installed" >&2; exit 2; }
deadline=$((SECONDS + timeout))
printf 'tart-guest-ready: waiting for %s (timeout %ss)\n' "$vm" "$timeout" >&2
while (( SECONDS < deadline )); do
    if tart exec "$vm" /usr/bin/true >/dev/null 2>&1; then
        printf 'tart-guest-ready: %s is ready\n' "$vm"
        exit 0
    fi
    sleep "$interval"
done
printf 'tart-guest-ready: timed out waiting for guest agent in %s\n' "$vm" >&2
exit 1
