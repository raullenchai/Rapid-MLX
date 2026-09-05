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
if ! grep -Fq "grep -Fqx '  Error: Insufficient disk space for download.'" "$SWEEP"; then
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

# Fake `$PY` and curl shims. Capacity models write the exact disk refusal;
# passing models hold a fake server open and publish a readiness marker;
# everything else emits a generic boot crash.
make_fake_bin() {
  local fake_dir="$TMP/fakebin"
  mkdir -p "$fake_dir"
  cat > "$fake_dir/python3.12" <<'EOF'
#!/bin/sh
if [ "$1" = "-m" ] && [ "$3" = "serve" ]; then
  model="$4"
  case ":${CAPACITY_ALIASES:-}:" in
    *":$model:"*) printf '%s\n' '  Error: Insufficient disk space for download.' ;;
    *)
      case ":${LOOKALIKE_ALIASES:-}:" in
        *":$model:"*)
          printf '%s\n' 'RuntimeError: Insufficient disk space for download.'
          exit 1
          ;;
      esac
      case ":${PASSING_ALIASES:-}:" in
        *":$model:"*)
          trap 'rm -f "$FAKE_READY"; exit 0' INT TERM EXIT
          printf '%s' "$model" > "$FAKE_READY"
          while :; do sleep 1; done
          ;;
        *) printf '%s\n' 'RuntimeError: fake engine died before load' ;;
      esac
      ;;
  esac
  exit 1
fi
# release_fleet.py is-reasoning-distill / forces-text-lane -> not that family.
if [ "$1" = "evals/coherence_gate.py" ]; then
  model="$(cat "$FAKE_READY")"
  case ":${FAILING_ALIASES:-}:" in
    *":$model:"*) exit 1 ;;
    *) exit 0 ;;
  esac
fi
exit 1
EOF
  cat > "$fake_dir/curl" <<'EOF'
#!/bin/sh
[ -f "${FAKE_READY:?}" ]
EOF
  chmod +x "$fake_dir/python3.12" "$fake_dir/curl"
  printf '%s' "$fake_dir"
}

run_sweep() {  # $1 aliases, $2 capacity aliases, $3 passing aliases, $4.. env
  local fake_dir aliases capacity passing
  fake_dir="$(make_fake_bin)"
  aliases="$1"
  capacity="$2"
  passing="$3"
  shift 3
  rm -f "$TMP/server-ready"
  env "$@" PY="$fake_dir/python3.12" PATH="$fake_dir:$PATH" \
    FAKE_READY="$TMP/server-ready" CAPACITY_ALIASES="$capacity" \
    PASSING_ALIASES="$passing" PORT="$(free_port)" MODELS="$aliases" \
    bash "$SWEEP" 2>&1; return $?
}

echo "── capacity-skip reports and passes"

# All families capacity-skipped means the gate validated nothing and must fail
# closed rather than claiming vacuous coherence.
st=0
out="$(run_sweep "qwen3.6-27b-4bit qwen3.6-35b" "qwen3.6-27b-4bit:qwen3.6-35b" "")" || st=$?
if [ "$st" = 2 ]; then
  ok "an all-capacity-skipped sweep fails because no alias was validated"
else
  bad "expected exit 2 with zero validated aliases, got $st"
fi
if printf '%s' "$out" | grep -q 'capacity-skipped'; then
  ok "the capacity-skipped families are reported explicitly"
else
  bad "capacity-skipped families not reported:\n$out"
fi
skip_summary="$(printf '%s\n' "$out" | grep 'CAPACITY-SKIPPED' | tail -1)"
if printf '%s' "$skip_summary" | grep -q 'qwen3.6-27b-4bit' \
   && printf '%s' "$skip_summary" | grep -q 'qwen3.6-35b'; then
  ok "both capacity-skipped families are named in the output"
else
  bad "expected both skipped families present in output:\n$out"
fi

# A resident representative genuinely passes while an oversized sibling is
# capacity-skipped: this is the intended constrained-host success path.
st=0; out="$(run_sweep "qwen3.5-9b-4bit qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "qwen3.5-9b-4bit")" || st=$?
if [ "$st" = 0 ] && printf '%s' "$out" | grep -q 'all resident aliases coherent'; then
  ok "a validated resident family plus a capacity-skip exits 0"
else
  bad "expected pass+skip to exit 0 with the resident summary, got $st:\n$out"
fi

# A capacity-skip must not hide a genuine boot failure elsewhere.
st=0; out="$(run_sweep "broken qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "")" || st=$?
if [ "$st" = 2 ]; then
  ok "a generic boot failure among the families still fails the sweep (2)"
else
  bad "expected exit 2 when a non-capacity boot failure co-occurs, got $st"
fi

st=0; out="$(run_sweep "passing broken qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "passing")" || st=$?
if [ "$st" = 2 ] \
   && printf '%s' "$out" | grep -q 'broken(boot)' \
   && printf '%s' "$out" | grep -q 'CAPACITY-SKIPPED'; then
  ok "pass plus infrastructure failure plus skip reports every category"
else
  bad "mixed pass/infrastructure/capacity report lost a category:\n$out"
fi

echo "── a genuine boot failure still fails the sweep"

st=0; out="$(run_sweep "qwen3.5-9b-4bit" "" "")" || st=$?
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

st=0; out="$(run_sweep "lookalike" "" "" LOOKALIKE_ALIASES=lookalike)" || st=$?
if [ "$st" = 2 ] && ! printf '%s' "$out" | grep -q 'capacity-skipped:'; then
  ok "a lookalike error line remains an infrastructure failure"
else
  bad "lookalike disk text was misclassified as a capacity-skip:\n$out"
fi

echo "── strict full-fleet mode is preserved"

st=0; out="$(run_sweep "qwen3.5-9b-4bit qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "qwen3.5-9b-4bit" COHERENCE_SWEEP_REQUIRE_ALL=1)" || st=$?
if [ "$st" = 2 ]; then
  ok "COHERENCE_SWEEP_REQUIRE_ALL=1 turns a capacity-skip into a failure"
else
  bad "expected exit 2 under REQUIRE_ALL with a capacity-skip, got $st"
fi

st=0; out="$(run_sweep "broken qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "" COHERENCE_SWEEP_REQUIRE_ALL=1)" || st=$?
if [ "$st" = 2 ] \
   && printf '%s' "$out" | grep -q 'broken(boot)' \
   && printf '%s' "$out" | grep -q 'CAPACITY-SKIPPED'; then
  ok "strict mixed failure reports both infrastructure and capacity"
else
  bad "strict mixed failure dropped a failure category:\n$out"
fi

echo "── a real model failure is still a model failure, skips are still noted"

st=0; out="$(run_sweep "bad-answer qwen3.6-27b-4bit" "qwen3.6-27b-4bit" "bad-answer" FAILING_ALIASES=bad-answer)" || st=$?
if [ "$st" = 1 ] \
   && printf '%s' "$out" | grep -q 'SWEEP FAILED' \
   && printf '%s' "$out" | grep -q 'CAPACITY-SKIPPED'; then
  ok "model failure plus capacity-skip stays a model failure and reports both"
else
  bad "mixed model failure/capacity-skip was misclassified:\n$out"
fi

printf '\n  %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
