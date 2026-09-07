#!/bin/bash
# Fail-closed Accessibility/Aqua preflight for unattended macOS GUI CI.

set -euo pipefail

if [[ "$#" -ne 1 || ! -x "$1" ]]; then
  echo "usage: mac-ci-ax-preflight.sh /path/to/rapid-ax" >&2
  exit 2
fi

AX_DRIVER="$1"
MAX_ATTEMPTS=5
attempt=1
delay_seconds=1
scratch="$(mktemp "${RUNNER_TEMP:-/tmp}/rapid-ax-preflight.XXXXXX")"
trap 'rm -f "$scratch" "$scratch.stderr"' EXIT

retryable_cannot_complete() {
  python3 - "$scratch" "$dock_pid" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raise SystemExit(1)

raise SystemExit(
    0
    if payload.get("target_pid") == int(sys.argv[2])
    and payload.get("trusted") is True
    and payload.get("screen_locked") is False
    and payload.get("target_read") is False
    and payload.get("target_read_error") == -25204
    and payload.get("success") is False
    else 1
)
PY
}

successful_payload() {
  python3 - "$scratch" "$dock_pid" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raise SystemExit(1)

raise SystemExit(
    0
    if payload.get("target_pid") == int(sys.argv[2])
    and payload.get("target_timeout_error") == 0
    and payload.get("trusted") is True
    and payload.get("screen_locked") is False
    and payload.get("target_read") is True
    and payload.get("success") is True
    else 1
)
PY
}

while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  : >"$scratch"
  : >"$scratch.stderr"
  console_user="$(stat -f '%Su' /dev/console 2>/dev/null || true)"
  console_uid=""
  if [[ -n "$console_user" && "$console_user" != "root" && "$console_user" != "loginwindow" ]]; then
    console_uid="$(id -u "$console_user" 2>/dev/null || true)"
  fi
  dock_pid=""
  if [[ -n "$console_uid" ]]; then
    dock_pid="$(pgrep -U "$console_uid" -x Dock | head -1 || true)"
  fi

  if [[ -z "$dock_pid" ]]; then
    reason="no Dock for console user ${console_user:-unknown}"
    retryable=1
  else
    echo "AX preflight attempt $attempt/$MAX_ATTEMPTS: console_user=$console_user Dock pid=$dock_pid"
    set +e
    "$AX_DRIVER" trust "$dock_pid" >"$scratch" 2>"$scratch.stderr"
    status=$?
    set -e
    cat "$scratch"
    if [[ "$status" -eq 0 ]] && successful_payload; then
      if [[ "$attempt" -gt 1 ]]; then
        echo "::notice::Accessibility preflight recovered on attempt $attempt/$MAX_ATTEMPTS."
      fi
      exit 0
    fi
    if [[ "$status" -eq 0 ]]; then
      cat "$scratch.stderr" >&2
      echo "::error::Accessibility preflight driver exited successfully without a valid success payload."
      exit 1
    fi
    if retryable_cannot_complete; then
      reason="Dock AX messaging returned transient cannotComplete (-25204)"
      retryable=1
    else
      cat "$scratch.stderr" >&2
      echo "::error::Accessibility preflight failed with a non-retryable permission or session result."
      exit "$status"
    fi
  fi

  if [[ "$retryable" -eq 1 && "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
    echo "::warning::AX preflight attempt $attempt/$MAX_ATTEMPTS: $reason; retrying in ${delay_seconds}s."
    sleep "$delay_seconds"
    if [[ "$delay_seconds" -lt 2 ]]; then
      delay_seconds=$((delay_seconds * 2))
    fi
    attempt=$((attempt + 1))
    continue
  fi

  if [[ -s "$scratch.stderr" ]]; then
    cat "$scratch.stderr" >&2
  fi
  echo "::error::Accessibility preflight failed after $attempt/$MAX_ATTEMPTS attempts: $reason."
  exit 1
done
