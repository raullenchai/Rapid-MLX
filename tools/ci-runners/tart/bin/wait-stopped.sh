#!/bin/bash
# Wait until launchd has removed the old GUI-domain service.
set -u

usage() {
  echo "Usage: $0 [timeout_secs]" >&2
  exit 2
}
[ "$#" -le 1 ] || usage
timeout=${1-90}
case "$timeout" in
  ''|*[!0-9]*|??????*) usage ;;  # digits only, at most 5
esac

LABEL=${LABEL:-com.rapidmlx.ci-runners}
LAUNCHCTL=${LAUNCHCTL:-/bin/launchctl}
POLL_SECS=${POLL_SECS:-1}
target="gui/$(id -u)/$LABEL"
start=$SECONDS
while :; do
  "$LAUNCHCTL" print "$target" >/dev/null 2>&1
  status=$?
  [ "$status" -eq 113 ] && break
  if [ "$((SECONDS - start))" -ge "$timeout" ]; then
    echo "Timed out waiting for $LABEL to stop after ${timeout}s (last launchctl status $status)" >&2
    exit 1
  fi
  sleep "$POLL_SECS"
done
echo "$LABEL stopped after $((SECONDS - start))s"
