#!/bin/bash
# launchd entry point: keep $SLOTS ephemeral runner slots alive.
set -u
. "$(dirname "$0")/common.sh"
LOG_TAG="supervisor"
mkdir -p "$LOG_DIR" "$CI_ROOT/state"

if ! "$TART" list --quiet 2>/dev/null | grep -qx "$GOLDEN"; then
  log "golden image $GOLDEN missing; sleeping"; sleep 300; exit 1
fi

PIDS=""
shutdown() {
  log "stopping slots"
  for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done
  wait
  exit 0
}
trap shutdown TERM INT HUP

log "starting $SLOTS slots (repo $REPO, golden $GOLDEN)"
i=1
while [ "$i" -le "$SLOTS" ]; do
  "$CI_ROOT/bin/slot.sh" "$i" >>"$LOG_DIR/supervisor.log" 2>&1 &
  PIDS="$PIDS $!"
  i=$((i + 1))
done

# Restart any slot loop that dies.
while true; do
  nap 30
  new=""; i=1
  for p in $PIDS; do
    if ! kill -0 "$p" 2>/dev/null; then
      log "slot $i loop exited; restarting"
      "$CI_ROOT/bin/slot.sh" "$i" >>"$LOG_DIR/supervisor.log" 2>&1 &
      p=$!
    fi
    new="$new $p"; i=$((i + 1))
  done
  PIDS="$new"
done
