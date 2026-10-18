#!/bin/bash
# One CI slot: clone golden -> boot -> JIT-register -> run exactly one job -> destroy. Forever.
set -u
. "$(dirname "$0")/common.sh"
N="$1"
VM="$VM_PREFIX$N"
LOG_TAG="slot$N"
STATE="$CI_ROOT/state/slot$N"
mkdir -p "$CI_ROOT/state" "$LOG_DIR"

VM_PID=""; EXEC_PID=""; RUNNER_ID=""

destroy_vm() {
  [ -n "$EXEC_PID" ] && kill "$EXEC_PID" 2>/dev/null
  "$TART" stop "$VM" --timeout 10 >/dev/null 2>&1
  [ -n "$VM_PID" ] && { kill "$VM_PID" 2>/dev/null; wait "$VM_PID" 2>/dev/null; }
  "$TART" delete "$VM" >/dev/null 2>&1
  # Keep the id for a later retry if GitHub could not confirm the removal.
  if [ -n "$RUNNER_ID" ] && ! runner_removed "$RUNNER_ID"; then
    echo "$RUNNER_ID" >> "$STATE.orphans"
    log "could not remove runner $RUNNER_ID; will retry"
  fi
  VM_PID=""; EXEC_PID=""; RUNNER_ID=""; rm -f "$STATE.runner"
}

# Removed now, or already gone on GitHub.
runner_removed() {
  delete_runner_if_idle "$1" || [ "$(runner_busy "$1")" = gone ]
}

# Retry registrations that an earlier destroy_vm could not remove.
reap_orphans() {
  [ -s "$STATE.orphans" ] || return 0
  local id keep=""
  while read -r id; do
    [ -n "$id" ] || continue
    runner_removed "$id" || keep="$keep$id
"
  done < "$STATE.orphans"
  printf '%s' "$keep" > "$STATE.orphans"
}

on_term() { log "terminating; destroying $VM"; destroy_vm; exit 0; }
trap on_term TERM INT HUP

# Recover from a previous crash/reboot: stale VM and stale runner registration.
if [ -f "$STATE.runner" ]; then RUNNER_ID=$(cat "$STATE.runner"); fi
destroy_vm

while true; do
  reap_orphans
  rotate_logs "$LOG_DIR/slot$N.runner.log" "$LOG_DIR/slot$N.tart.log"
  if reason=$(should_drain); then
    log "draining: $reason"; nap 30; continue
  fi
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    log "only $(free_gb) GB free (< $MIN_FREE_GB); not starting a VM"; nap 60; continue
  fi

  log "cloning $GOLDEN -> $VM"
  if ! "$TART" clone "$GOLDEN" "$VM" >>"$LOG_DIR/slot$N.tart.log" 2>&1; then
    "$TART" delete "$VM" >/dev/null 2>&1  # drop any partial clone
    log "clone failed; retry in 60s"; nap 60; continue
  fi
  started=$(date +%s)
  "$TART" run --no-graphics --no-audio --no-clipboard "$VM" >>"$LOG_DIR/slot$N.tart.log" 2>&1 &
  VM_PID=$!

  ready=0
  for _ in $(seq 1 90); do
    if "$TART" exec "$VM" true >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$VM_PID" 2>/dev/null || break
    nap 2
  done
  if [ "$ready" != 1 ]; then log "VM not ready after boot; destroying"; destroy_vm; nap 30; continue; fi
  log "VM booted in $(( $(date +%s) - started ))s"
  # Drain may have started while the VM booted; do not register a runner then.
  if reason=$(should_drain); then
    log "draining ($reason) after boot; destroying $VM"; destroy_vm; continue
  fi

  name="$RUNNER_PREFIX-$N-$(date +%Y%m%d%H%M%S)"
  body=$(jq -nc --arg n "$name" --argjson l "$LABELS_JSON" '{name:$n, runner_group_id:1, labels:$l, work_folder:"_work"}')
  if ! resp=$(gh_api POST "/repos/$REPO/actions/runners/generate-jitconfig" "$body"); then
    log "jitconfig request failed; destroying"; destroy_vm; nap 60; continue
  fi
  RUNNER_ID=$(printf '%s' "$resp" | jq -r '.runner.id')
  echo "$RUNNER_ID" > "$STATE.runner"
  log "registered runner $name (id $RUNNER_ID)"

  # The JIT config goes in over stdin; no host credential ever enters the VM.
  printf '%s\n' "$resp" | jq -r '.encoded_jit_config' |
    "$TART" exec -i "$VM" sudo -u admin -H bash -c 'cd ~/actions-runner && read -r c && exec ./run.sh --jitconfig "$c"' \
      >>"$LOG_DIR/slot$N.runner.log" 2>&1 &
  EXEC_PID=$!
  resp=""

  while kill -0 "$EXEC_PID" 2>/dev/null; do
    nap 20
    if [ $(( $(date +%s) - started )) -gt "$MAX_JOB_SECS" ]; then
      log "VM exceeded ${MAX_JOB_SECS}s; killing"; break
    fi
    if reason=$(should_drain); then
      if [ "$(runner_busy "$RUNNER_ID")" != "true" ] && delete_runner_if_idle "$RUNNER_ID"; then
        log "draining ($reason): idle runner removed"; break
      fi
    fi
  done
  log "runner exited after $(( $(date +%s) - started ))s; destroying $VM"
  destroy_vm
done
