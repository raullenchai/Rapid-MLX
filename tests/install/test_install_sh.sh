#!/usr/bin/env bash
# Offline regression tests for install.sh interpreter + venv handling (#1953).
#
# install.sh used to pick the first python3.x on PATH by name, ignoring its
# architecture, so a uv/pyenv x86_64 python3.12 ahead of an arm64 python3.11
# was chosen on Apple Silicon; the resulting venv was incomplete (no bin/pip)
# and every re-run then died on the upgrade path calling the missing pip.
#
# We source install.sh in library mode (RAPID_INSTALL_LIB=1) so only the
# helper functions load, then drive them against fake interpreters. No network,
# no real Python, no real install — runs on any host (macOS dev + Linux CI).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"

# Load install.sh's functions without running the installer. Do this BEFORE
# defining the harness helpers below: install.sh also defines ok()/warn()/dim(),
# so our definitions must win to keep the pass/fail counters correct.
# shellcheck disable=SC1090
RAPID_INSTALL_LIB=1 source "$INSTALL_SH"

PASS=0 FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1"; printf '        want: %s\n        got:  %s\n' "$3" "$2"; fi; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SYS_PATH="$PATH"   # keep coreutils for setup

# A fake python that reports a given "python X.Y" version and machine arch.
make_py() {   # make_py <path> <ver e.g. 3.12> <arch e.g. arm64>
    local path="$1" ver="$2" arch="$3"
    cat > "$path" <<EOF
#!/bin/bash
case "\$*" in
  *version_info*)      echo "$ver" ;;
  *platform.machine*)  echo "$arch" ;;
  *"import sys"*)       exit 0 ;;
  --version)           echo "Python $ver.0" ;;
  *)                   exit 0 ;;
esac
EOF
    chmod +x "$path"
}

# ── 1. arch-aware selection: skip x86_64, pick the native arm64 build ─────────
BIN1="$WORK/bin1"; mkdir -p "$BIN1"
make_py "$BIN1/python3.12" 3.12 x86_64     # earlier in loop order, WRONG arch
make_py "$BIN1/python3.11" 3.11 arm64      # later in order, right arch
sel="$(PATH="$BIN1" select_python 2>/dev/null)"
check "picks the native arm64 python over an earlier x86_64 one" "$sel" "python3.11"

# ── 2. selection fails cleanly when only an x86_64 python exists ──────────────
BIN2="$WORK/bin2"; mkdir -p "$BIN2"
make_py "$BIN2/python3.12" 3.12 x86_64
if PATH="$BIN2" select_python >/dev/null 2>&1; then
    bad "returns non-zero when no arm64 python is available"
else
    ok "returns non-zero when no arm64 python is available"
fi

# ── 3. version gate: a too-old arm64 python is not accepted ──────────────────
# Only a 3.9 candidate is present, so selection MUST fail. (Pairing it with a
# valid fallback would pass even if the version check were removed.)
BIN3="$WORK/bin3"; mkdir -p "$BIN3"
make_py "$BIN3/python3.11" 3.9 arm64       # 3.9 < MIN_PYTHON_MINOR, only candidate
if PATH="$BIN3" select_python >/dev/null 2>&1; then
    bad "rejects an arm64 python older than MIN_PYTHON_MINOR"
else
    ok "rejects an arm64 python older than MIN_PYTHON_MINOR"
fi

# ── 4. venv completeness: missing bin/pip is treated as incomplete ───────────
VBAD="$WORK/venv_bad"; mkdir -p "$VBAD/bin"
make_py "$VBAD/bin/python" 3.11 arm64      # python present, pip MISSING
if PATH="$SYS_PATH" venv_is_complete "$VBAD"; then
    bad "an existing venv without bin/pip is reported incomplete"
else
    ok "an existing venv without bin/pip is reported incomplete"
fi

# ── 5. venv completeness: python + working pip is complete ───────────────────
VOK="$WORK/venv_ok"; mkdir -p "$VOK/bin"
make_py "$VOK/bin/python" 3.11 arm64
make_py "$VOK/bin/pip"    3.11 arm64
if PATH="$SYS_PATH" venv_is_complete "$VOK"; then
    ok "a venv with both bin/python and bin/pip is reported complete"
else
    bad "a venv with both bin/python and bin/pip is reported complete"
fi

# ── 6. venv completeness: a non-existent dir is incomplete ───────────────────
if PATH="$SYS_PATH" venv_is_complete "$WORK/nope"; then
    bad "a missing install dir is reported incomplete"
else
    ok "a missing install dir is reported incomplete"
fi

# ── 7. venv completeness: an executable-but-broken pip is incomplete ──────────
# Mirrors a ~/.vllm-mlx → ~/.rapid-mlx migration where pip's shebang is stale:
# the file is present and +x, but running it fails.
VBROKEN="$WORK/venv_broken"; mkdir -p "$VBROKEN/bin"
make_py "$VBROKEN/bin/python" 3.11 arm64
printf '#!/bin/bash\nexit 1\n' > "$VBROKEN/bin/pip"   # +x but errors when run
chmod +x "$VBROKEN/bin/pip"
if PATH="$SYS_PATH" venv_is_complete "$VBROKEN"; then
    bad "a pip that is executable but fails to run is reported incomplete"
else
    ok "a pip that is executable but fails to run is reported incomplete"
fi

# ── 8. venv completeness: a runnable but x86_64 venv is incompatible ──────────
# An earlier Rosetta install: python/pip run fine, but mlx can never import.
VX86="$WORK/venv_x86"; mkdir -p "$VX86/bin"
make_py "$VX86/bin/python" 3.12 x86_64
make_py "$VX86/bin/pip"    3.12 x86_64
if PATH="$SYS_PATH" venv_is_complete "$VX86"; then
    bad "an existing x86_64 venv is reported incomplete (must be rebuilt)"
else
    ok "an existing x86_64 venv is reported incomplete (must be rebuilt)"
fi

# ── 9. venv completeness: a runnable but too-old (3.9) venv is incompatible ───
VOLD="$WORK/venv_old"; mkdir -p "$VOLD/bin"
make_py "$VOLD/bin/python" 3.9 arm64
make_py "$VOLD/bin/pip"    3.9 arm64
if PATH="$SYS_PATH" venv_is_complete "$VOLD"; then
    bad "an existing Python 3.9 venv is reported incomplete (must be rebuilt)"
else
    ok "an existing Python 3.9 venv is reported incomplete (must be rebuilt)"
fi

# ── 10. rebuild removes only venv entries, preserves shared ~/.rapid-mlx state ─
# ~/.rapid-mlx doubles as the state dir (telemetry-client-id, bench-install-id,
# agents/ profiles, launch.pid). A rebuild must not delete any of that.
RVD="$WORK/rapid_state"; mkdir -p "$RVD/bin" "$RVD/lib" "$RVD/include" "$RVD/agents"
make_py "$RVD/bin/python" 3.11 arm64
printf 'x' > "$RVD/pyvenv.cfg"
printf 'client-abc' > "$RVD/telemetry-client-id"
printf 'bench-xyz'  > "$RVD/bench-install-id"
printf '{}'         > "$RVD/agents/custom.json"
printf '4242'       > "$RVD/launch.pid"
reset_venv_dir "$RVD"
venv_gone=yes; for e in bin lib include pyvenv.cfg; do [ -e "$RVD/$e" ] && venv_gone=no; done
check "reset_venv_dir removes the venv's own entries" "$venv_gone" "yes"
state_kept=yes
for f in telemetry-client-id bench-install-id agents/custom.json launch.pid; do
    [ -e "$RVD/$f" ] || state_kept=no
done
check "reset_venv_dir preserves shared ~/.rapid-mlx state" "$state_kept" "yes"

# ── 11. create-vs-upgrade decision wires venv_is_complete correctly ──────────
# Exercises the actual control flow: a missing dir creates, a compatible venv
# upgrades, and an incomplete/incompatible one rebuilds. Reversing the branch
# in install.sh would flip these.
check "plan: a missing dir is created" \
    "$(PATH="$SYS_PATH" plan_venv_action "$WORK/absent")" "create"

PUP="$WORK/plan_ok"; mkdir -p "$PUP/bin"
make_py "$PUP/bin/python" 3.11 arm64
make_py "$PUP/bin/pip"    3.11 arm64
check "plan: a complete arm64 venv is upgraded" \
    "$(PATH="$SYS_PATH" plan_venv_action "$PUP")" "upgrade"

PRB="$WORK/plan_rebuild"; mkdir -p "$PRB/bin"
make_py "$PRB/bin/python" 3.11 arm64       # no bin/pip -> incomplete
check "plan: an incomplete venv is rebuilt" \
    "$(PATH="$SYS_PATH" plan_venv_action "$PRB")" "rebuild"

PX86="$WORK/plan_x86"; mkdir -p "$PX86/bin"
make_py "$PX86/bin/python" 3.12 x86_64
make_py "$PX86/bin/pip"    3.12 x86_64
check "plan: an x86_64 venv is rebuilt" \
    "$(PATH="$SYS_PATH" plan_venv_action "$PX86")" "rebuild"

# ── 12. dispatch_venv wires each action to the right operations ───────────────
# Mock the side-effecting steps so we can assert the create/upgrade/rebuild
# branches invoke the intended operations in order — a swapped or dropped branch
# in install.sh would change this record.
CALLS=""
create_venv()    { CALLS="$CALLS create"; }
upgrade_pip()    { CALLS="$CALLS upgrade"; }
reset_venv_dir() { CALLS="$CALLS reset"; }
info() { :; }; warn() { :; }; dim() { :; }   # silence banners during dispatch

CALLS=""; dispatch_venv upgrade
check "dispatch upgrade -> only pip upgrade" "$CALLS" " upgrade"
CALLS=""; dispatch_venv create
check "dispatch create -> only venv create" "$CALLS" " create"
CALLS=""; dispatch_venv rebuild
check "dispatch rebuild -> reset then create" "$CALLS" " reset create"

echo
printf 'passed %d, failed %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
