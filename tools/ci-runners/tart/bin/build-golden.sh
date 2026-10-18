#!/bin/bash
# Rebuild the golden CI image from the upstream Ubuntu 24.04 arm64 base.
# Builds into rmx-golden-next, then swaps it in, keeping the old image until the new one is in place. Running slots keep
# their existing clones; new clones use the new image.
set -euo pipefail
. "$(dirname "$0")/common.sh"
LOG_TAG="build-golden"
BASE="${BASE:-ghcr.io/cirruslabs/ubuntu:24.04}"
NEXT="$GOLDEN-next"

"$TART" delete "$NEXT" >/dev/null 2>&1 || true
log "cloning $BASE"
"$TART" clone "$BASE" "$NEXT"
"$TART" set "$NEXT" --cpu "$VM_CPU" --memory "$VM_MEM_MB" --disk-size "$VM_DISK_GB"
"$TART" run --no-graphics --no-audio --no-clipboard "$NEXT" >"$LOG_DIR/build-golden.tart.log" 2>&1 &
VM_PID=$!
trap '"$TART" stop "$NEXT" >/dev/null 2>&1 || true; wait $VM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do "$TART" exec "$NEXT" true >/dev/null 2>&1 && break; sleep 2; done
"$TART" exec "$NEXT" true

# Grow the root filesystem into the enlarged disk.
"$TART" exec "$NEXT" sudo bash -c 'growpart /dev/vda 2 >/dev/null 2>&1 || true; resize2fs "$(findmnt -no SOURCE /)" >/dev/null 2>&1 || true; df -h /'
"$TART" exec -i "$NEXT" sudo bash -s < "$CI_ROOT/guest/provision.sh"
# Flush and power off from inside the guest: an external stop can lose ext4
# delayed-allocation writes (seen as zero-length runner binaries).
"$TART" exec "$NEXT" sudo sync
"$TART" exec "$NEXT" sudo systemctl poweroff >/dev/null 2>&1 || true
for _ in $(seq 1 60); do kill -0 "$VM_PID" 2>/dev/null || break; sleep 2; done
"$TART" stop "$NEXT" --timeout 30 >/dev/null 2>&1 || true
wait "$VM_PID" 2>/dev/null || true
# Sanity: no zero-length runner binaries in the image.
"$TART" run --no-graphics --no-audio --no-clipboard "$NEXT" >>"$LOG_DIR/build-golden.tart.log" 2>&1 &
VM_PID=$!
for _ in $(seq 1 90); do "$TART" exec "$NEXT" true >/dev/null 2>&1 && break; sleep 2; done
empty=$("$TART" exec "$NEXT" bash -c 'find /home/admin/actions-runner/bin /usr/local/bin -type f -size 0 | wc -l')
"$TART" exec "$NEXT" sudo systemctl poweroff >/dev/null 2>&1 || true
for _ in $(seq 1 60); do kill -0 "$VM_PID" 2>/dev/null || break; sleep 2; done
wait "$VM_PID" 2>/dev/null || true
[ "$empty" -eq 0 ] || { log "image has $empty zero-length files; aborting"; exit 1; }
trap - EXIT

# Keep the old image until the new one is in place, so a failed rename
# never leaves the host without a golden image.
PREV="$GOLDEN-prev"
"$TART" delete "$PREV" >/dev/null 2>&1 || true
if "$TART" list --quiet | grep -qx "$GOLDEN"; then "$TART" rename "$GOLDEN" "$PREV"; fi
if ! "$TART" rename "$NEXT" "$GOLDEN"; then
  "$TART" list --quiet | grep -qx "$PREV" && "$TART" rename "$PREV" "$GOLDEN"
  log "could not swap in $NEXT; kept the previous image"; exit 1
fi
"$TART" delete "$PREV" >/dev/null 2>&1 || true
log "golden image $GOLDEN rebuilt from $BASE"
