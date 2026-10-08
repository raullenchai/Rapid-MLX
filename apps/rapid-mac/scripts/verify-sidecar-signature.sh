#!/usr/bin/env bash
# Verify integrity for every signed sidecar Mach-O and the release-only
# Developer ID properties that notarization requires.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || { [ "$#" -eq 2 ] && [ "$2" != "--official" ]; }; then
    echo "usage: $0 FILE [--official]" >&2
    exit 2
fi

binary="$1"
codesign --verify --strict "$binary" >/dev/null 2>&1 || {
    echo "ERR: strict codesign verification failed on $binary" >&2
    exit 1
}

[ "${2:-}" = "--official" ] || exit 0

details="$(codesign -d --verbose=4 "$binary" 2>&1)" || {
    echo "ERR: could not inspect official signature on $binary" >&2
    exit 1
}
printf '%s\n' "$details" | grep -q '^Authority=Developer ID Application:' || {
    echo "ERR: Developer ID Application authority missing on $binary" >&2
    exit 1
}
timestamp="$(printf '%s\n' "$details" | sed -n 's/^Timestamp=//p' | head -n 1)"
if [ -z "$timestamp" ] || [ "$(printf '%s' "$timestamp" | tr '[:upper:]' '[:lower:]')" = "none" ]; then
    echo "ERR: secure timestamp missing on $binary" >&2
    exit 1
fi
printf '%s\n' "$details" | grep -Eq '^CodeDirectory .*flags=.*\(.*runtime.*\)' || {
    echo "ERR: hardened runtime flag missing on $binary" >&2
    exit 1
}
