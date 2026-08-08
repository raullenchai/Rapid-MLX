#!/usr/bin/env bash
#
# Offline tests for the ``cleanup`` function in scripts/release_check_m3.sh.
#
# The gauntlet installs ``cleanup`` as an EXIT trap, where its return status is
# discarded — but it ALSO calls it inline, just before G12, to hand the port and
# the GPU to the random-coverage sweep. There, under ``set -euo pipefail``, the
# status is load-bearing: a non-zero return kills the gauntlet on the spot.
#
# Written as `[ -n "$CLUSTER_WORK" ] && rm -rf …`, the function returned 1
# whenever CLUSTER_WORK was empty — which it always is by G12, the correctness
# cluster having cleared it a few hundred lines earlier. The observable symptom
# was the G12 banner followed immediately by ``make: *** Error 1`` and not one
# line of gate output, because the sweep never started. G12 had been dead for as
# long as the two lines coexisted.
#
#   ./tests/release/test_release_check_m3_cleanup.sh
#
# Requires: bash. No network/git/GPU required — the function is extracted from
# the script and exercised on its own.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/scripts/release_check_m3.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# Extract `cleanup() { … }` from the gauntlet so the test exercises the real
# text rather than a copy that can drift away from it.
sed -n '/^cleanup() {/,/^}/p' "$SCRIPT" > "$TMP/cleanup.sh"
if [ ! -s "$TMP/cleanup.sh" ]; then
  printf '  \033[31mFAIL\033[0m could not extract cleanup() from %s\n' "$SCRIPT"
  exit 1
fi

echo "── cleanup() return status"

# 1. The regression itself: empty CLUSTER_WORK, no pidfile — the state the
#    gauntlet is in at the inline call site.
if bash -c '
      set -euo pipefail
      PIDFILE="'"$TMP"'/no-such-pidfile"
      CLUSTER_WORK=""
      . "'"$TMP"'/cleanup.sh"
      cleanup
      exit 0
    ' 2>/dev/null; then
  ok "returns 0 with CLUSTER_WORK empty (the state G12 calls it in)"
else
  bad "returns non-zero with CLUSTER_WORK empty — under set -e this kills G12"
fi

# 2. Unset, not merely empty — the trap can fire before the cluster runs at all.
if bash -c '
      set -euo pipefail
      PIDFILE="'"$TMP"'/no-such-pidfile"
      unset CLUSTER_WORK
      . "'"$TMP"'/cleanup.sh"
      cleanup
      exit 0
    ' 2>/dev/null; then
  ok "returns 0 with CLUSTER_WORK unset"
else
  bad "returns non-zero with CLUSTER_WORK unset"
fi

# 3. It must still do its job: a populated CLUSTER_WORK is removed.
WORK="$TMP/cluster"
mkdir -p "$WORK"
touch "$WORK/g6.rc"
if bash -c '
      set -euo pipefail
      PIDFILE="'"$TMP"'/no-such-pidfile"
      CLUSTER_WORK="'"$WORK"'"
      . "'"$TMP"'/cleanup.sh"
      cleanup
      exit 0
    ' 2>/dev/null && [ ! -d "$WORK" ]; then
  ok "removes a populated CLUSTER_WORK and still returns 0"
else
  bad "did not remove CLUSTER_WORK, or returned non-zero"
fi

# 4. A stale pidfile naming a dead process must not fail either — `kill` on a
#    reaped pid is the normal case at teardown.
PIDFILE_STALE="$TMP/stale.pid"
echo 99999999 > "$PIDFILE_STALE"
if bash -c '
      set -euo pipefail
      PIDFILE="'"$PIDFILE_STALE"'"
      CLUSTER_WORK=""
      . "'"$TMP"'/cleanup.sh"
      cleanup
      exit 0
    ' 2>/dev/null && [ ! -f "$PIDFILE_STALE" ]; then
  ok "tolerates a pidfile whose process is already gone, and removes it"
else
  bad "a stale pidfile made cleanup fail, or left the file behind"
fi

echo "── G12 call site"

# 5. Pin the reason this matters: the gauntlet really does call cleanup inline,
#    and really does run under `set -e`. If either stops being true this test is
#    guarding nothing, and should be re-read rather than silently kept green.
if grep -qE '^set -euo pipefail' "$SCRIPT"; then
  ok "gauntlet still runs under set -euo pipefail"
else
  bad "gauntlet no longer sets -euo pipefail — re-read this test's premise"
fi

if awk '/G12 — random-coverage \(sampled/,/release_check_m3_random\.py/' "$SCRIPT" \
     | grep -qE '^[[:space:]]*cleanup[[:space:]]*$'; then
  ok "G12 still calls cleanup inline before handing over the port"
else
  bad "G12 no longer calls cleanup inline — re-read this test's premise"
fi

echo "── old_server_alive() (the G12 port/GPU handoff)"

# Extract the nested helper — it lives inside the G12 `else` branch, indented.
sed -n '/^  old_server_alive() {/,/^  }/p' "$SCRIPT" | sed 's/^  //' > "$TMP/alive.sh"
if [ ! -s "$TMP/alive.sh" ]; then
  printf '  \033[31mFAIL\033[0m could not extract old_server_alive() from %s\n' "$SCRIPT"
  exit 1
fi

# `kill` and `ps` are shadowed so each process state can be posed exactly.
# A zombie is the case that matters: `kill -0` succeeds on one, because a child
# that has exited but has not been reaped still owns a pid — and owns no GPU.
# Reading that as "still running" burns the whole 60s deadline and then SIGKILLs
# a corpse, turning a clean shutdown into a gate failure.
alive_with() {  # $1 = ps stat output, $2 = kill -0 status
  bash -c '
      set -uo pipefail
      OLD_SERVER_PID=4242
      kill() { return '"$2"'; }
      ps() { printf "%s" "'"$1"'"; }
      . "'"$TMP"'/alive.sh"
      old_server_alive
    ' 2>/dev/null
}

if alive_with "S+" 0; then
  ok "a running process reads as alive"
else
  bad "a running process read as gone — the handoff would proceed onto a busy GPU"
fi

if alive_with "Z" 0; then
  bad "a ZOMBIE read as alive — the deadline would expire and SIGKILL a corpse"
else
  ok "a zombie reads as gone (kill -0 succeeds on one; ps says Z)"
fi

if alive_with "" 0; then
  bad "an unreadable process state read as alive"
else
  ok "a process ps cannot describe reads as gone"
fi

if alive_with "S+" 1; then
  bad "kill -0 failing still read as alive"
else
  ok "a pid that no longer exists reads as gone"
fi

if bash -c '
      set -uo pipefail
      OLD_SERVER_PID=""
      . "'"$TMP"'/alive.sh"
      old_server_alive
    ' 2>/dev/null; then
  bad "an empty pid read as alive — nothing to wait for"
else
  ok "an empty pid reads as gone"
fi

printf '\n  %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
