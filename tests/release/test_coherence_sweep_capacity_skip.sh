#!/usr/bin/env bash
#
# Offline tests for the #2206 capacity-skip behaviour in
# scripts/coherence_sweep.sh.
#
# On a disk-constrained host, a fleet family that `serve` refuses to download
# because it would exceed free disk is reported as a "capacity-skip" instead of
# a hard infrastructure failure, and the resident representatives are still
# validated. `COHERENCE_SWEEP_REQUIRE_ALL=1` restores the strict
# every-family-required behaviour.
#
# The sweep's serve boot is stubbed with a fake `$PY` whose `-m ... serve`
# immediately writes a chosen boot outcome to the server log and exits, so the
# readiness loop always reaches the boot-failure (or capacity-skip) branch
# without a real server, download, GPU, or network.
#
#   ./tests/release/test_coherence_sweep_capacity_skip.sh
#
# Requires: bash, curl, lsof. No network/git/GPU required.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SWEEP="$REPO_ROOT/scripts/coherence_sweep.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# The sweep must still contain the capacity-skip hook — if the region drifts
# this test is guarding nothing and should be re-read.
if ! grep -q 'Insufficient disk space for download\\\.' "$SWEEP"; then
  printf '  \033[31mFAIL\033[0m capacity-skip detection missing from %s\n' "$SWEEP"
  exit 1
fi
if ! grep -q 'COHERENCE_SWEEP_REQUIRE_ALL' "$SWEEP"; then
  printf '  \033[31mFAIL\033[0m COHERENCE_SWEEP_REQUIRE_ALL hook missing from %s\n' "$SWEEP"
  exit 1
fi
ok "sweep carries the capacity-skip detection + strict-mode hook"

# A free ephemeral port for every run, so two consecutive sweep runs never
# collide.
free_port() {
  python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()'
}

# Fake `$PY`: a shell shim standing in for `python3.12`. It dispatches on the
# command-line shape the sweep issues:
#   -m vllm_mlx.cli serve <model> ...   -> write a boot outcome to stdout
#                                         (the sweep redirects it to $LOG) and
#                                         exit 1
#   scripts/release_fleet.py is-reasoning-distill / forces-text-lane -> exit 1
#                                        (neither, matching ordinary families)
# Capacity models (alias prefixes the $CAPACITY_ALIASES env lists) write the
# exact `_check_disk_space` refusal; every other model writes a generic crash so
# the boot-failure (infra) branch is exercised for contrast.
make_fake_py() {  # $1 = capacity aliases (colon-separated)
  local fake_dir="$TMP/fakebin"
  mkdir -p "$fake_dir"
  cat > "$fake_dir/python3.12" <<EOF
#!/bin/sh
if [ "\$1" = "-m" ] && [ "\$3" = "serve" ]; then
  model="\$4"
  case ":$1:" in
    *":\$model:"*) printf '%s\\n' '  Error: Insufficient disk space for download.' ;;
    *)             printf '%s\\n' 'RuntimeError: fake engine died before load' ;;
  esac
  exit 1
fi
# release_fleet.py is-reasoning-distill / forces-text-lane -> not that family.
exit 1
EOF
  sed -i.bak "s|:\$1:|:$1:|" "$fake_dir/python3.12" && rm -f "$fake_dir/python3.12.bak"
  chmod +x "$fake_dir/python3.12"
  printf '%s' "$fake_dir/python3.12"
}

run_sweep() {  # $1 = alias list, $2 = capacity aliases, $3.. = extra env
  local fake_py aliases cap
  fake_py="$(make_fake_py "$2")"
  aliases="$1"
  shift 2
  env "$@" PY="$fake_py" PORT="$(free_port)" MODELS="$aliases" \
    bash "$SWEEP" 2>&1; return $?
}

echo "── capacity-skip reports and passes"

# All families capacity-skipped (no genuine infra failure) -> the sweep
# validates the empty resident set, reports the skips, and exits 0.
out="$(run_sweep "qwen3.6-27b-4bit qwen3.6-35b" "qwen3.6-27b-4bit:qwen3.6-35b")" || st=$?
st=${st:-0}
if [ "$st" = 0 ]; then
  ok "capacity-skipped families report but the sweep exits 0"
else
  bad "expected exit 0 with only capacity-skips, got $st"
fi
if printf '%s' "$out" | grep -q 'capacity-skipped'; then
  ok "the capacity-skipped families are reported explicitly"
else
  bad "capacity-skipped families not reported:\n$out"
fi
if printf '%s' "$out" | grep -q 'qwen3.6-27b-4bit' \
   && printf '%s' "$out" | grep -q 'qwen3.6-35b'; then
  ok "both capacity-skipped families are named in the output"
else
  bad "expected both skipped families present in output:\n$out"
fi

# A resident representative that boots genuinely passes, while a sibling is
# capacity-skipped — but a genuine boot failure anywhere still fails the sweep.
st=0; out="$(run_sweep "qwen3.5-9b-4bit qwen3.6-27b-4bit" "qwen3.6-27b-4bit")" || st=$?
if [ "$st" = 2 ]; then
  ok "a generic boot failure among the families still fails the sweep (2)"
else
  bad "expected exit 2 when a non-capacity boot failure co-occurs, got $st"
fi

echo "── a genuine boot failure still fails the sweep"

st=0; out="$(run_sweep "qwen3.5-9b-4bit" "")" || st=$?
if [ "$st" = 2 ]; then
  ok "a non-capacity boot failure still exits 2 (infrastructure)"
else
  bad "expected exit 2 for a generic boot failure, got $st"
fi
if printf '%s' "$out" | grep -q 'INFRASTRUCTURE FAILURE'; then
  ok "a generic boot failure is reported as infrastructure, not capacity"
else
  bad "generic boot failure not flagged as infrastructure:\n$out"
fi

echo "── strict full-fleet mode is preserved"

st=0; out="$(run_sweep "qwen3.5-9b-4bit qwen3.6-27b-4bit" "qwen3.6-27b-4bit" COHERENCE_SWEEP_REQUIRE_ALL=1)" || st=$?
if [ "$st" = 2 ]; then
  ok "COHERENCE_SWEEP_REQUIRE_ALL=1 turns a capacity-skip into a failure"
else
  bad "expected exit 2 under REQUIRE_ALL with a capacity-skip, got $st"
fi

echo "── a real model failure is still a model failure, skips are still noted"

# Not directly runnable hermetically (a booting server + gate are needed), but
# the summary wiring must still attribute a capacity-skip next to a model
# failure rather than dropping it. The query of the script text guards that the
# "CAPACITY-SKIPPED" line sits in the model-failure branch.
if grep -q 'CAPACITY-SKIPPED —\$skipped' "$SWEEP"; then
  ok "model-failure summary still attributes any capacity-skips"
else
  bad "model-failure exit path no longer surfaces capacity-skips"
fi

printf '\n  %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
