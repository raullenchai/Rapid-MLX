#!/usr/bin/env bash
# Offline regression tests for install.sh interpreter + venv handling (#1953).
#
# install.sh used to pick the first python3.x on PATH by name, ignoring its
# architecture, so a uv/pyenv x86_64 python3.13 ahead of an arm64 python3.12
# was chosen on Apple Silicon; the resulting venv was incomplete (no bin/pip)
# and every re-run then died on the upgrade path calling the missing pip.
#
# We source install.sh in library mode (RAPID_INSTALL_LIB=1) so only the
# helper functions load, then drive them against fixture interpreters. No network
# or real install is needed (macOS dev + Linux CI).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
INSTALL_SH="${INSTALL_SH:-$REPO_ROOT/install.sh}"

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
REAL_PYTHON="$(command -v python3)"

# Box widths are measured after stripping terminal color codes.
for box_version in 'rapid-mlx 1.0.1rc3' '123456789012345678901234567890123456789012345678901234567890'; do
    for box_color in plain color; do
        if [ "$box_color" = color ]; then GREEN='\033[32m' RESET='\033[0m'; else GREEN='' RESET=''; fi
        if print_success_box 'Rapid-MLX installed!' "Version: $box_version" 'RAM: 64 GB (16+ GB)' |
            "$REAL_PYTHON" -c 'import re, sys
lines = [re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip("\n")) for line in sys.stdin]
sys.exit(0 if len(lines) == 5 and len({len(line) for line in lines}) == 1 else 1)'; then
            ok "success box aligns for ${#box_version}-character version ($box_color)"
        else
            bad "success box aligns for ${#box_version}-character version ($box_color)"
        fi
    done
done
GREEN='' RESET=''

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
make_py "$BIN1/python3.13" 3.13 x86_64     # earlier in loop order, WRONG arch
make_py "$BIN1/python3.12" 3.12 arm64      # later in order, right arch
sel="$(PATH="$BIN1" select_python 2>/dev/null)"
check "picks the native arm64 python over an earlier x86_64 one" "$sel" "python3.12"

# ── 2. selection fails cleanly when only an x86_64 python exists ──────────────
BIN2="$WORK/bin2"; mkdir -p "$BIN2"
make_py "$BIN2/python3.13" 3.13 x86_64
if PATH="$BIN2" select_python >/dev/null 2>&1; then
    bad "returns non-zero when no arm64 python is available"
else
    ok "returns non-zero when no arm64 python is available"
fi

# ── 3. version gate: a too-old arm64 python is not accepted ──────────────────
# Only a 3.9 candidate is present, so selection MUST fail. (Pairing it with a
# valid fallback would pass even if the version check were removed.)
BIN3="$WORK/bin3"; mkdir -p "$BIN3"
make_py "$BIN3/python3.12" 3.11 arm64       # 3.11 < MIN_PYTHON_MINOR, only candidate
if PATH="$BIN3" select_python >/dev/null 2>&1; then
    bad "rejects an arm64 python older than MIN_PYTHON_MINOR"
else
    ok "rejects an arm64 python older than MIN_PYTHON_MINOR"
fi

# ── 3b. version gate: a too-new generic python is not accepted ────────────────
BINNEW="$WORK/binnew"; mkdir -p "$BINNEW"
make_py "$BINNEW/python3" 3.14 arm64
if PATH="$BINNEW" select_python >/dev/null 2>&1; then
    bad "rejects an arm64 python newer than MAX_PYTHON_MINOR"
else
    ok "rejects an arm64 python newer than MAX_PYTHON_MINOR"
fi
make_py "$BINNEW/python3.12" 3.12 arm64
sel="$(PATH="$BINNEW" select_python 2>/dev/null)"
check "picks python3.12 when generic python3 is too new" "$sel" "python3.12"

# ── 4. venv completeness: missing bin/pip is treated as incomplete ───────────
VBAD="$WORK/venv_bad"; mkdir -p "$VBAD/bin"
make_py "$VBAD/bin/python" 3.12 arm64      # python present, pip MISSING
if PATH="$SYS_PATH" venv_is_complete "$VBAD"; then
    bad "an existing venv without bin/pip is reported incomplete"
else
    ok "an existing venv without bin/pip is reported incomplete"
fi

# ── 5. venv completeness: python + working pip is complete ───────────────────
VOK="$WORK/venv_ok"; mkdir -p "$VOK/bin"
make_py "$VOK/bin/python" 3.12 arm64
make_py "$VOK/bin/pip"    3.12 arm64
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
make_py "$VBROKEN/bin/python" 3.12 arm64
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

# ── 9b. venv completeness: a runnable but too-new (3.14) venv is incompatible ─
VNEW="$WORK/venv_new"; mkdir -p "$VNEW/bin"
make_py "$VNEW/bin/python" 3.14 arm64
make_py "$VNEW/bin/pip"    3.14 arm64
if PATH="$SYS_PATH" venv_is_complete "$VNEW"; then
    bad "an existing Python 3.14 venv is reported incomplete (must be rebuilt)"
else
    ok "an existing Python 3.14 venv is reported incomplete (must be rebuilt)"
fi
check "plan: a Python 3.14 venv is rebuilt" \
    "$(PATH="$SYS_PATH" plan_venv_action "$VNEW")" "rebuild"

# ── 10. rebuild removes only venv entries, preserves shared ~/.rapid-mlx state ─
# ~/.rapid-mlx doubles as the state dir (telemetry-client-id, bench-install-id,
# agents/ profiles, launch.pid). A rebuild must not delete any of that.
RVD="$WORK/rapid_state"; mkdir -p "$RVD/bin" "$RVD/lib" "$RVD/include" "$RVD/agents"
make_py "$RVD/bin/python" 3.12 arm64
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
make_py "$PUP/bin/python" 3.12 arm64
make_py "$PUP/bin/pip"    3.12 arm64
check "plan: a complete arm64 venv is upgraded" \
    "$(PATH="$SYS_PATH" plan_venv_action "$PUP")" "upgrade"

PRB="$WORK/plan_rebuild"; mkdir -p "$PRB/bin"
make_py "$PRB/bin/python" 3.12 arm64       # no bin/pip -> incomplete
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
(
    CALLS=""
    create_venv() { CALLS="$CALLS create"; }
    upgrade_pip() { CALLS="$CALLS upgrade"; }
    stash_venv_entries() { CALLS="$CALLS stash"; }
    installed_extras_of_venv() { printf "%s\n" "audio,chat"; }
    info() { :; }; warn() { :; }; dim() { :; }
    INSTALL_DIR="$WORK/dispatch"; mkdir -p "$INSTALL_DIR/bin"
    CALLS=""; dispatch_venv upgrade
    check "dispatch upgrade -> only pip upgrade" "$CALLS" " upgrade"
    CALLS=""; dispatch_venv create
    check "dispatch create -> only venv create" "$CALLS" " create"
    CALLS=""; dispatch_venv rebuild
    check "dispatch rebuild -> stash then create" "$CALLS" " stash create"
    check "dispatch preserves extras" "$PRESERVED_EXTRAS" "audio,chat"
    [ "$FAIL" -eq 0 ]
) || bad "dispatch action wiring"

# Metadata fixtures exercise the real stdlib parser without network access.
EXTRAS_VENV="$WORK/extras"; SITE="$WORK/site"
mkdir -p "$EXTRAS_VENV/bin" "$SITE/rapid_mlx-1.0.0.dist-info"
cat > "$EXTRAS_VENV/bin/python" <<EOF
#!/bin/bash
printf '%s\n' "\$1" >> "$WORK/python-args"
mode=()
if [ "\$1" = -I ]; then mode=(-I); shift; fi
exec "$REAL_PYTHON" "\${mode[@]}" -c 'import sys; sys.path.insert(1, sys.argv[1]); exec(sys.argv[2])' "$SITE" "\$2"
EOF
chmod +x "$EXTRAS_VENV/bin/python"
cat > "$SITE/rapid_mlx-1.0.0.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: rapid-mlx
Version: 1.0.0
Requires-Dist: rmlx-fixture-mlx>=0.1
Requires-Dist: rmlx-fixture-vlm
Requires-Dist: rmlx-fixture-gradio>=4; extra == "chat"
Requires-Dist: rmlx-fixture-pytz; extra == "chat"
Requires-Dist: rmlx-fixture-mlx-audio>=0.5; platform_system == 'Darwin' and extra == "audio"
Requires-Dist: rmlx-fixture-soundfile; extra == "audio"
Requires-Dist: rmlx-fixture-mflux; extra == "image"
Requires-Dist: rmlx-fixture-mlx-video; python_version >= "3.11" and extra == "video"
Requires-Dist: rmlx-fixture-pytest; extra == "dev"
EOF
for dist in mlx gradio pytz mlx-audio soundfile pytest vlm; do
    fixture="rmlx-fixture-$dist"
    dist_dir="${fixture//-/_}-1.0.dist-info"
    mkdir -p "$SITE/$dist_dir"
    printf 'Metadata-Version: 2.1\nName: %s\nVersion: 5.0\n' "$fixture" > "$SITE/$dist_dir/METADATA"
done
printf 'Requires-Dist: rmlx-fixture-audiolib\n' >> "$SITE/rmlx_fixture_vlm-1.0.dist-info/METADATA"
mkdir -p "$SITE/rmlx_fixture_audiolib-0.5.9.dist-info"
printf 'Metadata-Version: 2.1\nName: rmlx-fixture-audiolib\nVersion: 0.5.9\n' > "$SITE/rmlx_fixture_audiolib-0.5.9.dist-info/METADATA"
printf 'Requires-Dist: rmlx-fixture-audiolib<0.6; extra == "audio-desktop"\n' >> "$SITE/rapid_mlx-1.0.0.dist-info/METADATA"
check "ignores extra satisfied only by transitive base dependency" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio,chat"
check "detects installed optional features" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio,chat"
mkdir -p "$WORK/conflicting-cwd/rapid_mlx-9.9.9.dist-info"
cat > "$WORK/conflicting-cwd/rapid_mlx-9.9.9.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: rapid-mlx
Version: 9.9.9
Requires-Dist: rmlx-fixture-gradio; extra == "bogus"
EOF
check "ignores conflicting metadata in the current directory" \
    "$(cd "$WORK/conflicting-cwd" && installed_extras_of_venv "$EXTRAS_VENV")" "audio,chat"
(
    cd "$WORK/conflicting-cwd" || exit 1
    INSTALL_DIR="$EXTRAS_VENV"; PRESERVED_EXTRAS="audio,chat"; TARGET=stable
    recorder() { printf '%s\n' "$*" > "$WORK/isolated-spec"; }
    INSTALLER=(recorder)
    reinstall_preserved_extras >/dev/null
    [ "$(cat "$WORK/isolated-spec")" = 'rapid-mlx[audio,chat]==1.0.0' ]
) && ok "reinstall ignores conflicting current-directory version" || bad "reinstall ignores conflicting current-directory version"
check "both metadata snippets request isolated Python" "$(cat "$WORK/python-args")" "$(printf '%s\n' -I -I -I -I)"

# Legacy metadata must preserve optional features through the migration rebuild.
LEGACY_EXTRAS_VENV="$WORK/legacy-extras"; LEGACY_SITE="$WORK/legacy-site"
mkdir -p "$LEGACY_EXTRAS_VENV/bin" "$LEGACY_SITE/vllm_mlx-0.12.0.dist-info"
cat > "$LEGACY_EXTRAS_VENV/bin/python" <<EOF
#!/bin/bash
mode=()
if [ "\$1" = -I ]; then mode=(-I); shift; fi
exec "$REAL_PYTHON" "\${mode[@]}" -c 'import sys; sys.path.insert(1, sys.argv[1]); exec(sys.argv[2])' "$LEGACY_SITE" "\$2"
EOF
chmod +x "$LEGACY_EXTRAS_VENV/bin/python"
cat > "$LEGACY_SITE/vllm_mlx-0.12.0.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: vllm-mlx
Version: 0.12.0
Requires-Dist: rmlx-fixture-mlx
Requires-Dist: rmlx-fixture-gradio; extra == "chat"
EOF
for dist in mlx gradio; do
    cp -R "$SITE/rmlx_fixture_${dist}-1.0.dist-info" "$LEGACY_SITE/"
done
LEGACY_PRESERVED_EXTRAS="$(installed_extras_of_venv "$LEGACY_EXTRAS_VENV")"
check "detects chat from legacy distribution metadata" "$LEGACY_PRESERVED_EXTRAS" "chat"
(
    # The rebuilt environment has current distribution metadata.
    cp -R "$SITE/rapid_mlx-1.0.0.dist-info" "$LEGACY_SITE/"
    INSTALL_DIR="$LEGACY_EXTRAS_VENV"; PRESERVED_EXTRAS="$LEGACY_PRESERVED_EXTRAS"; TARGET=stable
    recorder() { printf '%s\n' "$*" > "$WORK/legacy-spec"; }
    INSTALLER=(recorder)
    reinstall_preserved_extras >/dev/null
    [ "$(cat "$WORK/legacy-spec")" = 'rapid-mlx[chat]==1.0.0' ]
) && ok "reinstalls chat preserved from legacy metadata" || bad "reinstalls chat preserved from legacy metadata"

if "$REAL_PYTHON" -I -c 'from packaging.requirements import Requirement' 2>/dev/null ||
   "$REAL_PYTHON" -I -c 'from pip._vendor.packaging.requirements import Requirement' 2>/dev/null; then
    printf 'Requires-Dist: rmlx-fixture-incompatible<0.6; extra == "incompatible"\n' >> "$SITE/rapid_mlx-1.0.0.dist-info/METADATA"
    mkdir -p "$SITE/rmlx_fixture_incompatible-0.6.3.dist-info"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-incompatible\nVersion: 0.6.3\n' > "$SITE/rmlx_fixture_incompatible-0.6.3.dist-info/METADATA"
    check "rejects extras with incompatible installed versions" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio,chat"

    cp "$SITE/rapid_mlx-1.0.0.dist-info/METADATA" "$WORK/extras-metadata"
    cat > "$SITE/rapid_mlx-1.0.0.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: rapid-mlx
Version: 1.0.0
Requires-Dist: rmlx-fixture-audio-base>=0.5
Requires-Dist: rmlx-fixture-audio-base<0.6; extra == "audio"
Requires-Dist: rmlx-fixture-audio-only-pkg; extra == "audio"
EOF
    mkdir -p "$SITE/rmlx_fixture_audio_base-0.6.3.dist-info" "$SITE/rmlx_fixture_audio_only_pkg-1.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-audio-base\nVersion: 0.6.3\n' > "$SITE/rmlx_fixture_audio_base-0.6.3.dist-info/METADATA"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-audio-only-pkg\nVersion: 1.0\n' > "$SITE/rmlx_fixture_audio_only_pkg-1.0.dist-info/METADATA"
    check "keeps extra when a shared base dependency drifted" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-audio-base\nVersion: 0.5.9\n' > "$SITE/rmlx_fixture_audio_base-0.6.3.dist-info/METADATA"
    check "selects extra with compatible base dependency" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio"
    cp "$WORK/extras-metadata" "$SITE/rapid_mlx-1.0.0.dist-info/METADATA"
fi
if "$REAL_PYTHON" -I -c 'from packaging.requirements import Requirement' 2>/dev/null; then
    cat >> "$SITE/rapid_mlx-1.0.0.dist-info/METADATA" <<'EOF'
Requires-Dist: rmlx-fixture-foo; python_version >= "3.0" and extra == "gated"
Requires-Dist: rmlx-fixture-bar; python_version < "3.0" and extra == "never"
EOF
    mkdir -p "$SITE/rmlx_fixture_foo-1.0.dist-info" "$SITE/rmlx_fixture_bar-1.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-foo\nVersion: 1.0\n' > "$SITE/rmlx_fixture_foo-1.0.dist-info/METADATA"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-bar\nVersion: 1.0\n' > "$SITE/rmlx_fixture_bar-1.0.dist-info/METADATA"
    check "evaluates Python-gated extras with packaging" "$(installed_extras_of_venv "$EXTRAS_VENV")" "audio,chat,gated"
    extras_with_gated="audio,chat,gated,vendored"
else
    printf '  SKIP Python-gated extras: packaging is not importable\n'
    extras_with_gated="audio,chat,vendored"
fi
if "$REAL_PYTHON" -I -c 'from pip._vendor.packaging.requirements import Requirement' 2>/dev/null; then
    cat >> "$SITE/rapid_mlx-1.0.0.dist-info/METADATA" <<'EOF'
Requires-Dist: rmlx-fixture-vendor; python_version >= "3.0" and extra == "vendored"
EOF
    mkdir -p "$SITE/rmlx_fixture_vendor-1.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: rmlx-fixture-vendor\nVersion: 1.0\n' > "$SITE/rmlx_fixture_vendor-1.0.dist-info/METADATA"
    printf 'raise ImportError("packaging intentionally unavailable")\n' > "$SITE/packaging.py"
    check "evaluates Python-gated extras with pip vendored packaging" "$(installed_extras_of_venv "$EXTRAS_VENV")" "$extras_with_gated"
    rm "$SITE/packaging.py"
else
    printf '  SKIP Python-gated extras: pip vendored packaging is not importable\n'
fi
cat > "$EXTRAS_VENV/bin/python" <<'EOF'
#!/bin/bash
exit 1
EOF
if result="$(installed_extras_of_venv "$EXTRAS_VENV")" && [ -z "$result" ]; then
    ok "broken interpreter yields empty extras successfully"
else
    bad "broken interpreter yields empty extras successfully"
fi

(
    INSTALL_DIR="$WORK/warning"; mkdir -p "$INSTALL_DIR/bin"
    make_py "$INSTALL_DIR/bin/python" 3.11 arm64
    installed_extras_of_venv() { :; }; stash_venv_entries() { :; }; create_venv() { :; }
    output="$(dispatch_venv rebuild)"
    case "$output" in *"uses Python 3.11"*) exit 0 ;; *) exit 1 ;; esac
) && ok "rebuild explains unsupported Python" || bad "rebuild explains unsupported Python"

(
    INSTALL_DIR="$WORK/reinstall"; mkdir -p "$INSTALL_DIR/bin"
    cat > "$INSTALL_DIR/bin/python" <<'EOF'
#!/bin/bash
printf '1.0.1\n'
EOF
    chmod +x "$INSTALL_DIR/bin/python"
    recorder() { printf '%s\n' "$*" > "$WORK/recorded"; }
    INSTALLER=(recorder); PRESERVED_EXTRAS="audio,chat"; TARGET=stable
    reinstall_preserved_extras >/dev/null
    [ "$(cat "$WORK/recorded")" = 'rapid-mlx[audio,chat]==1.0.1' ] || exit 1
    recorder() { return 1; }
    output="$(reinstall_preserved_extras)" || exit 1
    case "$output" in *"Install them later with:"*) : ;; *) exit 1 ;; esac
    rm "$WORK/recorded"
    recorder() { touch "$WORK/recorded"; }
    PRESERVED_EXTRAS=""; reinstall_preserved_extras
    [ ! -e "$WORK/recorded" ]
) && ok "reinstall pins version, tolerates failure, and skips empty extras" || bad "reinstall pins version, tolerates failure, and skips empty extras"

(
    INSTALL_DIR="$WORK/reinstall-missing-version"; mkdir -p "$INSTALL_DIR/bin"
    printf '#!/bin/bash\nexit 1\n' > "$INSTALL_DIR/bin/python"
    chmod +x "$INSTALL_DIR/bin/python"
    recorder() { touch "$WORK/unexpected-install"; }
    INSTALLER=(recorder); PRESERVED_EXTRAS="audio,chat"; TARGET=stable
    output="$(reinstall_preserved_extras)" || exit 1
    [ ! -e "$WORK/unexpected-install" ] || exit 1
    case "$output" in *"Install them later with: $INSTALL_DIR/bin/python -m pip install 'rapid-mlx[audio,chat]'"*) : ;; *) exit 1 ;; esac
) && ok "missing version gives an unpinned reinstall hint" || bad "missing version gives an unpinned reinstall hint"

# ── Transactional rebuilds preserve the old venv and shared state ───────────
make_old_install() {
    mkdir -p "$1/bin" "$1/lib" "$1/agents"
    printf 'old python\n' > "$1/bin/python"
    printf 'old lib\n' > "$1/lib/marker"
    printf 'old config\n' > "$1/pyvenv.cfg"
    printf 'client-abc' > "$1/telemetry-client-id"
    printf '{"custom":true}' > "$1/agents/custom.json"
}
check_old_install() {
    local dir="$1" result=yes backup
    [ "$(cat "$dir/bin/python" 2>/dev/null)" = 'old python' ] || result=no
    [ "$(cat "$dir/lib/marker" 2>/dev/null)" = 'old lib' ] || result=no
    [ "$(cat "$dir/pyvenv.cfg" 2>/dev/null)" = 'old config' ] || result=no
    [ ! -e "$dir/bin/marker-new" ] || result=no
    [ "$(cat "$dir/telemetry-client-id" 2>/dev/null)" = 'client-abc' ] || result=no
    [ "$(cat "$dir/agents/custom.json" 2>/dev/null)" = '{"custom":true}' ] || result=no
    for backup in "$dir"/.venv-previous.*; do [ ! -e "$backup" ] || result=no; done
    check "$2" "$result" yes
}
make_new_install() {
    mkdir -p "$INSTALL_DIR/bin"
    printf 'new\n' > "$INSTALL_DIR/bin/marker-new"
    printf '#!/bin/bash\nexit 0\n' > "$INSTALL_DIR/bin/python"
    printf '#!/bin/bash\nexit %s\n' "${CLI_STATUS:-0}" > "$INSTALL_DIR/bin/rapid-mlx"
    chmod +x "$INSTALL_DIR/bin/python" "$INSTALL_DIR/bin/rapid-mlx"
}

ROUND="$WORK/roundtrip"; make_old_install "$ROUND"
stash_venv_entries "$ROUND" "$ROUND/.venv-previous.1"
check "stash leaves shared state in place" "$(cat "$ROUND/telemetry-client-id") $(cat "$ROUND/agents/custom.json")" 'client-abc {"custom":true}'
restore_venv_entries "$ROUND" "$ROUND/.venv-previous.1"
check_old_install "$ROUND" "stash and restore preserve venv and shared state"

for failure in install create verify; do
    FIXTURE="$WORK/failure-$failure"; make_old_install "$FIXTURE"
    set +e
    (
        set -e
        INSTALL_DIR="$FIXTURE"
        INSTALL_LOCK="${INSTALL_DIR}.install-lock"
        trap on_installer_exit EXIT
        create_venv() {
            make_new_install
            if [ "$failure" = create ]; then exit 1; fi
        }
        install_package() { if [ "$failure" = install ]; then return 1; fi; }
        if [ "$failure" = verify ]; then CLI_STATUS=1; fi
        dispatch_venv rebuild
        install_package
        finish_rebuild
    ) >/dev/null 2>&1
    status=$?
    set -e
    check "$failure failure exits nonzero" "$([ "$status" -ne 0 ] && echo yes || echo no)" yes
    check_old_install "$FIXTURE" "$failure failure restores old venv and shared state"
done

for scenario in success interrupted; do
    FIXTURE="$WORK/$scenario"; make_old_install "$FIXTURE"
    if [ "$scenario" = interrupted ]; then
        stash_venv_entries "$FIXTURE" "$FIXTURE/.venv-previous.123"
    fi
    (
        INSTALL_DIR="$FIXTURE"
        create_venv() { make_new_install; }
        install_package() { :; }
        recover_interrupted_rebuild
        dispatch_venv rebuild
        install_package
        finish_rebuild
    ) >/dev/null
    result=yes
    [ -e "$FIXTURE/bin/marker-new" ] || result=no
    [ ! -e "$FIXTURE/lib/marker" ] || result=no
    [ "$(cat "$FIXTURE/telemetry-client-id")" = client-abc ] || result=no
    [ -e "$FIXTURE/agents/custom.json" ] || result=no
    for backup in "$FIXTURE"/.venv-previous.* "$FIXTURE"/.venv-discard.*; do [ ! -e "$backup" ] || result=no; done
    check "$scenario rebuild commits new venv and preserves shared state" "$result" yes
done

# A committed discard must never replace the current environment on recovery.
(
    INSTALL_DIR="$WORK/recover-discard"; make_old_install "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR/.venv-discard.123/bin"
    printf 'discarded\n' > "$INSTALL_DIR/.venv-discard.123/bin/python"
    touch "$INSTALL_DIR/.venv-discard.123/.stash-complete"
    recover_interrupted_rebuild
    check_old_install "$INSTALL_DIR" "recovery retains the current environment when discard exists"
    [ ! -e "$INSTALL_DIR/.venv-discard.123" ] || exit 1
    [ "$FAIL" -eq 0 ]
) || bad "recovery removes committed discard"

# Exercise locks in independent shells, including migration from old locks.
LOCK_HOME="$WORK/lock-home"
mkdir -p "$LOCK_HOME"
lock="$LOCK_HOME/.rapid-mlx.lock"
ln -s target "$lock"
if output=$(HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock' _ "$INSTALL_SH" 2>&1); then
    bad "symlink lock rejects acquisition and remains in place"
elif [[ "$output" == *"The Rapid-MLX install lock $lock is a symlink. Remove it and re-run."* ]] && [ -L "$lock" ]; then
    ok "symlink lock rejects acquisition and remains in place"
else bad "symlink lock rejects acquisition and remains in place"; fi
rm -f "$lock"
if HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock; release_install_lock' _ "$INSTALL_SH" &&
   HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock; release_install_lock' _ "$INSTALL_SH"; then
    ok "released lock can be acquired by another process"
else bad "released lock can be acquired by another process"; fi
HOME="$LOCK_HOME" bash -c '
    RAPID_INSTALL_LIB=1 source "$1"
    trap release_install_lock EXIT
    acquire_install_lock
    touch "$2"
    while [ ! -e "$3" ]; do sleep 0.1; done
' _ "$INSTALL_SH" "$WORK/ready" "$WORK/release" &
lock_holder=$!
for ((i=0; i<100; i++)); do [ ! -e "$WORK/ready" ] || break; sleep 0.1; done
if [ -L "$LOCK_HOME/.rapid-mlx.install-lock" ] &&
   [ "$(readlink "$LOCK_HOME/.rapid-mlx.install-lock")" = "$lock_holder" ]; then
    ok "held lock exposes the holder pid to older installers"
else bad "held lock exposes the holder pid to older installers"; fi
if [ -e "$WORK/ready" ] && output=$(HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock' _ "$INSTALL_SH" 2>&1); then
    bad "held lock rejects another process"
elif [[ "$output" == *"Another Rapid-MLX install is already running"* ]]; then ok "held lock rejects another process"
else bad "held lock rejects another process"; fi
touch "$WORK/release"
wait "$lock_holder"
if [ ! -L "$LOCK_HOME/.rapid-mlx.install-lock" ]; then
    ok "release removes the legacy lock"
else bad "release removes the legacy lock"; fi
# Exercise explicit mode selection independently of Perl on PATH.
for lock_mode in flock dir; do
    [ "$lock_mode" != flock ] || [ -x /usr/bin/perl ] || continue
    if HOME="$WORK/mode-$lock_mode" bash -c '
        RAPID_INSTALL_LIB=1 source "$1"
        mkdir -p "$HOME"
        perl() { return 99; }
        if [ "$2" = dir ]; then RAPID_INSTALL_PERL="$HOME/missing-perl"; fi
        acquire_install_lock
        [ "$INSTALL_LOCK_MODE" = "$2" ]
        [ "$LEGACY_LOCK_OWNED" = 1 ]
        [ -L "$LEGACY_INSTALL_LOCK" ]
        [ "$(readlink "$LEGACY_INSTALL_LOCK")" = "$$" ]
        release_install_lock
        [ -z "$LEGACY_LOCK_OWNED" ]
        [ ! -L "$LEGACY_INSTALL_LOCK" ]
        [ ! -d "${INSTALL_LOCK}.d" ]
    ' _ "$INSTALL_SH" "$lock_mode"; then
        ok "$lock_mode mode owns and releases the legacy lock independently of PATH Perl"
    else bad "$lock_mode mode owns and releases the legacy lock independently of PATH Perl"; fi
done
LOCK_HOME="$WORK/legacy-home"
mkdir -p "$LOCK_HOME"
legacy="$LOCK_HOME/.rapid-mlx.install-lock"
ln -s 99999999 "$legacy"
if HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock; [ -L "$LEGACY_INSTALL_LOCK" ]; [ "$(readlink "$LEGACY_INSTALL_LOCK")" = "$$" ]; release_install_lock' _ "$INSTALL_SH" &&
   mkdir "$legacy" && printf '99999999\n' > "$legacy/pid" &&
   HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock; [ -L "$LEGACY_INSTALL_LOCK" ]; [ "$(readlink "$LEGACY_INSTALL_LOCK")" = "$$" ]; release_install_lock' _ "$INSTALL_SH"; then
    ok "stale legacy symlink and directory are removed"
else bad "stale legacy symlink and directory are removed"; fi
rm -rf "$legacy"
ln -s "$$" "$legacy"
if output=$(HOME="$LOCK_HOME" bash -c 'RAPID_INSTALL_LIB=1 source "$1"; acquire_install_lock' _ "$INSTALL_SH" 2>&1); then
    bad "live legacy lock rejects acquisition"
elif [[ "$output" == *"Another Rapid-MLX install is already running"* ]]; then ok "live legacy lock rejects acquisition"
else bad "live legacy lock rejects acquisition"; fi
rm -f "$legacy"
RACE="$WORK/race"
mkdir -p "$RACE"
ln -s 99999999 "$RACE/.rapid-mlx.install-lock"
for n in 1 2; do
    HOME="$RACE" bash -c '
        RAPID_INSTALL_LIB=1 source "$1"
        race_dir="$2" race_id="$3"
        kill() {
            touch "$race_dir/marker.$race_id"
            for ((i=0; i<50; i++)); do
                [ -e "$race_dir/marker.1" ] && [ -e "$race_dir/marker.2" ] && break
                sleep 0.1
            done
            # Let the first process finish its takeover so a second takeover would delete a live lock.
            [ "$race_id" = 1 ] || sleep 0.5
            command kill "$@"
        }
        acquire_install_lock
        echo acquired
        sleep 2
    ' _ "$INSTALL_SH" "$RACE" "$n" >"$RACE/out.$n" 2>&1 &
    sleep 0.1
done
wait
if [ "$(cat "$RACE"/out.* | grep -c '^acquired$')" = 1 ]; then ok "concurrent stale lock permits one installer"
else bad "concurrent stale lock permits one installer"; fi

# Recovery covers both a completed stash and an interruption partway through it.
(
    INSTALL_DIR="$WORK/recover-created"; make_old_install "$INSTALL_DIR"
    stash_venv_entries "$INSTALL_DIR" "$INSTALL_DIR/.venv-previous.111"
    make_new_install
    recover_interrupted_rebuild >/dev/null
    check_old_install "$INSTALL_DIR" "recovery replaces a newly created venv with the old one"
    [ "$FAIL" -eq 0 ]
) || bad "completed stash recovery"

(
    INSTALL_DIR="$WORK/recover-partial"; make_old_install "$INSTALL_DIR"
    mkdir "$INSTALL_DIR/.venv-previous.222"
    mv "$INSTALL_DIR/bin" "$INSTALL_DIR/.venv-previous.222/"
    recover_interrupted_rebuild >/dev/null
    check_old_install "$INSTALL_DIR" "partial stash recovery preserves unmoved entries"
    [ "$FAIL" -eq 0 ]
) || bad "partial stash recovery"

(
    INSTALL_DIR="$WORK/recover-multiple"; make_old_install "$INSTALL_DIR"
    mkdir "$INSTALL_DIR/.venv-previous.111"
    touch -t 202001010000 "$INSTALL_DIR/.venv-previous.111"
    stash_venv_entries "$INSTALL_DIR" "$INSTALL_DIR/.venv-previous.222"
    recover_interrupted_rebuild >/dev/null
    check_old_install "$INSTALL_DIR" "recovery selects newest backup and removes older backups"
    [ "$FAIL" -eq 0 ]
) || bad "multiple backup recovery"

(
    INSTALL_DIR="$WORK/restore-failure"; make_old_install "$INSTALL_DIR"
    VENV_BACKUP="$INSTALL_DIR/.venv-previous.333"
    stash_venv_entries "$INSTALL_DIR" "$VENV_BACKUP"
    mv() { case "$1" in */lib) return 1 ;; esac; command mv "$@"; }
    if restore_venv_entries "$INSTALL_DIR" "$VENV_BACKUP"; then exit 1; fi
    [ -d "$VENV_BACKUP/lib" ] || exit 1
    output=$(rollback_rebuild 2>&1)
    case "$output" in *"preserved at $VENV_BACKUP"*) : ;; *) exit 1 ;; esac
    unset -f mv
    recover_interrupted_rebuild >/dev/null
    check_old_install "$INSTALL_DIR" "failed restore preserves backup and can be retried"
    [ "$FAIL" -eq 0 ]
) || bad "restore failure reports the preserved backup"

FIXTURE="$WORK/stash-failure"; make_old_install "$FIXTURE"
set +e
(
    set -e
    INSTALL_DIR="$FIXTURE"
    mv() { case "$1" in "$INSTALL_DIR/lib") return 1 ;; esac; command mv "$@"; }
    create_venv() { :; }; install_package() { :; }; run_install_command() { :; }
    dispatch_venv rebuild
) >/dev/null 2>&1
status=$?
set -e
if [ "$status" -eq 0 ]; then
    bad "mid-stash failure exits nonzero"
else
    ok "mid-stash failure exits nonzero"
fi
check_old_install "$FIXTURE" "mid-stash failure restores the entire old environment"

FIXTURE="$WORK/stash-exit"; make_old_install "$FIXTURE"
set +e
(
    set -e
    INSTALL_DIR="$FIXTURE"
    INSTALL_LOCK="${INSTALL_DIR}.install-lock"
    trap on_installer_exit EXIT
    mv() { case "$1" in "$INSTALL_DIR/lib") exit 143 ;; esac; command mv "$@"; }
    dispatch_venv rebuild
) >/dev/null 2>&1
status=$?
set -e
check "exit during stash retains its exit status" "$status" 143
check_old_install "$FIXTURE" "exit during stash invokes rollback trap"

# ── 13. first-chat starter mirrors Desktop Quickstart (#2385) ────────────────
check "15 GB fresh install uses the safe 1.2B starter" \
    "$(select_starter_model 15 '')" "lfm2.5-1b-4bit"
check "16 GB fresh install uses the standard starter" \
    "$(select_starter_model 16 '')" "qwen3.5-4b-4bit"
check "32 GB fresh install does not suggest an uncached 27B model" \
    "$(select_starter_model 32 '')" "qwen3.5-4b-4bit"
check "eligible cached model is preferred" \
    "$(select_starter_model 32 $'qwen3.5-4b-4bit\nqwen3.8-27b-4bit')" \
    "qwen3.8-27b-4bit"
check "cached model above the RAM tier is never selected" \
    "$(select_starter_model 16 $'qwen3.8-27b-4bit\nlfm2.5-2.6b-4bit')" \
    "lfm2.5-2.6b-4bit"
check "8 GB cached smart pick is preferred over an uncached baseline" \
    "$(select_starter_model 8 'lfm2.5-2.6b-4bit')" \
    "lfm2.5-2.6b-4bit"
check "unknown RAM ignores a cached larger model" \
    "$(select_starter_model 0 'lfm2.5-2.6b-4bit')" \
    "lfm2.5-1b-4bit"
check "16 GB cached lower-tier fast pick is preferred over an uncached baseline" \
    "$(select_starter_model 16 'lfm2.5-1b-4bit')" \
    "lfm2.5-1b-4bit"
check "cache matching is exact, not a substring" \
    "$(select_starter_model 32 'prefix-qwen3.8-27b-4bit-suffix')" \
    "qwen3.5-4b-4bit"

CACHE_FIXTURE="$WORK/cache_fixture"; mkdir -p "$CACHE_FIXTURE/bin"
cat > "$CACHE_FIXTURE/bin/rapid-mlx" <<'EOF'
#!/bin/bash
printf '%s\n' '{"cached":[{"alias":"qwen3.5-9b-4bit","state":"ok"},{"alias":"qwen3.8-27b-4bit","state":"ok"},{"alias":"partial","state":"incomplete"},{"alias":null,"state":"unmapped"}]}'
EOF
cat > "$CACHE_FIXTURE/bin/python" <<'EOF'
#!/bin/bash
exec python3 "$@"
EOF
chmod +x "$CACHE_FIXTURE/bin/rapid-mlx" "$CACHE_FIXTURE/bin/python"
OLD_INSTALL_DIR="$INSTALL_DIR"; INSTALL_DIR="$CACHE_FIXTURE"
check "structured cache reader returns only runnable mapped aliases" \
    "$(installed_cached_aliases)" $'qwen3.5-9b-4bit\nqwen3.8-27b-4bit'
check "multiple structured cached aliases feed the selector independently" \
    "$(select_starter_model 32 "$(installed_cached_aliases)")" \
    "qwen3.8-27b-4bit"
RAM_GB=32 RECOMMENDED_MODEL="qwen3.5-4b-4bit" RECOMMENDED_FLAGS=""
refresh_starter_from_installed_cache
dim() { printf '%s\n' "$*"; }
BANNER="$(print_quick_start_commands)"
case "$BANNER" in
    *"rapid-mlx chat qwen3.8-27b-4bit"*"rapid-mlx launch codex"*)
        ok "post-install banner leads with agent launch after cached chat starter" ;;
    *)
        bad "post-install banner leads with agent launch after cached chat starter"
        printf '        banner: %s\n' "$BANNER" ;;
esac
case "$BANNER" in
    *"Launch checks the agent's model needs against your Mac RAM; small Macs may need more memory."*"Run the same command tomorrow; add --remove"*)
        ok "post-install banner explains model choice and repeat launch" ;;
    *)
        bad "post-install banner explains model choice and repeat launch"
        printf '        banner: %s\n' "$BANNER" ;;
esac
case "$BANNER" in
    *"rapid-mlx launch codex --model qwen3.8-27b-4bit"*|*"rapid-mlx launch --all --model"*)
        bad "post-install banner does not force a chat starter on an agent" ;;
    *)
        ok "post-install banner does not force a chat starter on an agent" ;;
esac

# A subprocess can print a syntactically valid partial result and still fail.
# install.sh's top-level pipefail must make that entire cache query fail so the
# banner uses the RAM baseline rather than trusting output from a failed scan.
cat > "$CACHE_FIXTURE/bin/rapid-mlx" <<'EOF'
#!/bin/bash
printf '%s\n' '{"cached":[{"alias":"qwen3.8-27b-4bit","state":"ok"}]}'
exit 7
EOF
RECOMMENDED_MODEL="stale"
refresh_starter_from_installed_cache
check "failed cache scan with valid partial JSON falls back to RAM baseline" \
    "$RECOMMENDED_MODEL" "qwen3.5-4b-4bit"
INSTALL_DIR="$OLD_INSTALL_DIR"

# Standalone downloads use separate temporary directories and clean up.
(
    STANDALONE_DIR="$WORK/standalone" PY_VERSION=3.12.13 PY_BUILD=test PY_URL=test PY_SHA256=hash TMPDIR="$WORK"
    download() { printf '%s\n' "$TMP_TAR" >> "$WORK/downloads"; printf archive; }
    shasum() { echo 'hash  archive'; }
    tar() { mkdir -p "$4/bin"; make_py "$4/bin/python3" 3.12.12 arm64; }
    output="$(install_standalone_python)" || exit 1
    case "$output" in *"Installed Python 3.12.12.0"*) : ;; *) exit 1 ;; esac
    case "$output" in *"Python Python"*) exit 1 ;; esac
    touch "$STANDALONE_DIR/stale"
    install_standalone_python >/dev/null || exit 1
    [ "$(wc -l < "$WORK/downloads" | tr -d ' ')" = 2 ] || exit 1
    [ "$(sort -u "$WORK/downloads" | wc -l | tr -d ' ')" = 2 ] || exit 1
    [ ! -e "$STANDALONE_DIR/stale" ] || exit 1
    for path in "$WORK"/rapidmlx-py.*; do [ ! -e "$path" ] || exit 1; done
) && ok "standalone downloads use distinct cleaned directories and replace stale files" || bad "standalone downloads use distinct cleaned directories and replace stale files"
(
    STANDALONE_DIR="$WORK/reuse" PY_VERSION=3.12.13 PY_BUILD=test PY_URL=test PY_SHA256=hash TMPDIR="$WORK"
    mkdir -p "$STANDALONE_DIR/bin"
    make_py "$STANDALONE_DIR/bin/python3" 3.12.13 arm64
    standalone_python_reusable || exit 1
    make_py "$STANDALONE_DIR/bin/python3" 3.12.13 x86_64
    if standalone_python_reusable; then exit 1; fi
    make_py "$STANDALONE_DIR/bin/python3" 3.12.13 arm64
    download() { touch "$WORK/reuse-downloaded"; }
    shasum() { echo 'hash archive'; }
    tar() { mkdir -p "$4/bin"; make_py "$4/bin/python3" 3.12.12 arm64; }
    install_standalone_python >/dev/null || exit 1
    [ ! -e "$WORK/reuse-downloaded" ] || exit 1
    make_py "$STANDALONE_DIR/bin/python3" 3.12.12 arm64
    install_standalone_python >/dev/null || exit 1
    [ -e "$WORK/reuse-downloaded" ]
) && ok "standalone reuses matching Python and downloads mismatched version" || bad "standalone reuses matching Python and downloads mismatched version"
(
    STANDALONE_DIR="$WORK/validation-failure" PY_VERSION=3.12.13 PY_BUILD=test PY_URL=test PY_SHA256=hash TMPDIR="$WORK"
    mkdir -p "$STANDALONE_DIR"
    touch "$STANDALONE_DIR/marker"
    download() { printf archive; }
    shasum() { echo 'hash archive'; }
    tar() { mkdir -p "$4/bin"; printf '#!/bin/bash\nexit 1\n' > "$4/bin/python3"; chmod +x "$4/bin/python3"; }
    if install_standalone_python >"$WORK/validation-output" 2>&1; then exit 1; fi
    [ -f "$STANDALONE_DIR/marker" ] || exit 1
    grep -q 'Failed to install standalone Python\.' "$WORK/validation-output" || exit 1
    for path in "$STANDALONE_DIR".new.* "$STANDALONE_DIR".old.*; do [ ! -e "$path" ] || exit 1; done
) && ok "failed staged Python validation preserves the existing runtime" || bad "failed staged Python validation preserves the existing runtime"
(
    STANDALONE_DIR="$WORK/replacement" PY_VERSION=3.12.13 PY_BUILD=test PY_URL=test PY_SHA256=hash TMPDIR="$WORK"
    mkdir -p "$STANDALONE_DIR"
    touch "$STANDALONE_DIR/marker"
    download() { printf archive; }
    shasum() { echo 'hash archive'; }
    tar() { mkdir -p "$4/bin"; make_py "$4/bin/python3" 3.12.13 arm64; touch "$4/new-marker"; }
    install_standalone_python >/dev/null || exit 1
    [ ! -e "$STANDALONE_DIR/marker" ] && [ -f "$STANDALONE_DIR/new-marker" ] && [ -x "$STANDALONE_DIR/bin/python3" ] || exit 1
    for path in "$STANDALONE_DIR".new.* "$STANDALONE_DIR".old.*; do [ ! -e "$path" ] || exit 1; done
) && ok "validated standalone Python replaces the existing runtime without leftovers" || bad "validated standalone Python replaces the existing runtime without leftovers"
(
    HOME="$WORK/hints-home" RAPID_INSTALL_LAUNCH_DAEMONS_DIR="$WORK/daemons"
    INSTALL_DIR="$HOME/.rapid-mlx" BIN_DIR="$HOME/.local/bin"
    mkdir -p "$HOME/Library/LaunchAgents" "$RAPID_INSTALL_LAUNCH_DAEMONS_DIR" "$INSTALL_DIR/bin"
    hint_python="$REAL_PYTHON"
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then hint_python="$REPO_ROOT/.venv/bin/python"; fi
    ln -s "$hint_python" "$INSTALL_DIR/bin/python"
    [ -z "$(restart_service_hints upgrade)" ] || exit 1
    "$REAL_PYTHON" -c 'import plistlib,sys; plistlib.dump({"Label":"com.example.rapid","ProgramArguments":[sys.argv[2],"service","run"]}, open(sys.argv[1],"wb"))' "$RAPID_INSTALL_LAUNCH_DAEMONS_DIR/custom.plist" "$BIN_DIR/rapid-mlx"
    "$REAL_PYTHON" -c 'import plistlib,sys; plistlib.dump({"Label":"foreign.label","ProgramArguments":["/usr/bin/true"]}, open(sys.argv[1],"wb"))' "$RAPID_INSTALL_LAUNCH_DAEMONS_DIR/foreign.plist"
    printf invalid > "$RAPID_INSTALL_LAUNCH_DAEMONS_DIR/invalid.plist"
    output="$(restart_service_hints upgrade)"
    [[ "$output" == *'Background services still run the previous version until restarted:'*'sudo rapid-mlx service restart --label com.example.rapid'* ]] || exit 1
    [[ "$output" != *foreign.label* ]] || exit 1
    touch "$RAPID_INSTALL_LAUNCH_DAEMONS_DIR/com.rapidmlx.server.plist" "$HOME/Library/LaunchAgents/com.quicksilver.node.abc.plist"
    [ -z "$(restart_service_hints create)" ] || exit 1
    for action in upgrade rebuild; do
        output="$(restart_service_hints "$action")"
        case "$output" in *'Background services still run the previous version until restarted:'*'sudo rapid-mlx service restart'*'sudo rapid-mlx service restart --label com.example.rapid'*'launchctl kickstart -k gui/$(id -u)/com.quicksilver.node.abc'*) ;; *) exit 1 ;; esac
    done
) && ok "service hints only follow upgrades and rebuilds with existing services" || bad "service hints only follow upgrades and rebuilds with existing services"

# Pending optional features survive retries and stay outside the venv stash.
for install_status in 0 7; do
    (
        INSTALL_DIR="$WORK/pending-$install_status"; mkdir -p "$INSTALL_DIR/bin"
        printf '#!/bin/bash\necho 1.0.1\n' > "$INSTALL_DIR/bin/python"
        chmod +x "$INSTALL_DIR/bin/python"
        printf 'audio,chat\n' > "$INSTALL_DIR/.pending-extras"
        PRESERVED_EXTRAS=""; TARGET=stable; INSTALLER=(recorder)
        upgrade_pip() { :; }
        recorder() { printf '%s\n' "$*" > "$INSTALL_DIR/attempt"; return "$install_status"; }
        dispatch_venv upgrade >/dev/null || exit 1
        reinstall_preserved_extras >/dev/null || exit 1
        [ "$(cat "$INSTALL_DIR/attempt")" = 'rapid-mlx[audio,chat]==1.0.1' ] || exit 1
        if [ "$install_status" = 0 ]; then
            [ ! -e "$INSTALL_DIR/.pending-extras" ]
        else
            [ "$(cat "$INSTALL_DIR/.pending-extras")" = audio,chat ]
        fi
    ) && ok "pending extras retry status $install_status" || bad "pending extras retry status $install_status"
done
(
    INSTALL_DIR="$WORK/pending-stash"; mkdir -p "$INSTALL_DIR/bin"
    installed_extras_of_venv() { echo audio,chat; }
    create_venv() { [ "$(cat "$INSTALL_DIR/.pending-extras")" = audio,chat ]; }
    dispatch_venv rebuild >/dev/null || exit 1
    [ ! -e "$VENV_BACKUP/.pending-extras" ] || exit 1
    restore_venv_entries "$INSTALL_DIR" "$VENV_BACKUP" || exit 1
    [ "$(cat "$INSTALL_DIR/.pending-extras")" = audio,chat ]
) && ok "pending extras persist before stash and survive restore" || bad "pending extras persist before stash and survive restore"
(
    INSTALL_DIR="$WORK/pending-garbage"; mkdir -p "$INSTALL_DIR"
    printf 'chat] @ invalid\n' > "$INSTALL_DIR/.pending-extras"
    PRESERVED_EXTRAS=""; upgrade_pip() { :; }
    dispatch_venv upgrade >/dev/null || exit 1
    [ -z "$PRESERVED_EXTRAS" ] && [ ! -e "$INSTALL_DIR/.pending-extras" ]
) && ok "invalid pending extras are discarded" || bad "invalid pending extras are discarded"

for config_name in uv.toml pyproject.toml; do
    (
        fixture="$WORK/physical-$config_name"
        mkdir -p "$fixture/physical/project" "$fixture/logical" "$fixture/home"
        printf '[tool.uv]\n' > "$fixture/physical/$config_name"
        ln -s "$fixture/physical/project" "$fixture/logical/project"
        HOME="$fixture/home"; XDG_CONFIG_HOME="$HOME"; XDG_CONFIG_DIRS="$HOME"
        unset UV_CONFIG_FILE UV_NO_CONFIG
        INSTALL_DIR="$fixture/runtime"; mkdir -p "$INSTALL_DIR/bin"
        ln -s "$REAL_PYTHON" "$INSTALL_DIR/bin/python"
        cd "$fixture/logical/project" || exit 1
        if uv_config_absent; then exit 1; fi
    ) && ok "physical ancestor $config_name is respected" || bad "physical ancestor $config_name is respected"
done

# Signal only the installer shell; a surviving writer must stop before rollback.
for signal in HUP TERM; do
    for command_kind in direct pip uv; do
        fixture="$WORK/signal-$signal-$command_kind"; mkdir -p "$fixture"
        cat > "$fixture/writer" <<'EOF'
#!/bin/bash
printf '%s\n' "$$" > "$1/pid"
touch "$1/ready"
sleep 1
printf late > "$1/late-write"
EOF
        chmod +x "$fixture/writer"
        bash -c '
            RAPID_INSTALL_LIB=1 source "$1"
            fixture=$2; command_kind=$3
            rollback_rebuild() {
                if kill -0 "$(cat "$fixture/pid")" 2>/dev/null; then touch "$fixture/alive-at-rollback"; fi
                touch "$fixture/rollback"
            }
            release_install_lock() { touch "$fixture/released"; }
            # Use the production trap definitions even in library mode.
            eval "$(sed -n "/^trap on_installer_exit EXIT$/,/^trap .* TERM$/p" "$1")"
            case "$command_kind" in
                direct) run_install_command "$fixture/writer" "$fixture" ;;
                pip) run_package_command "$fixture/writer" "$fixture" ;;
                uv)
                    uv_config_absent() { return 1; }
                    uv() { exec "$fixture/writer" "$fixture"; }
                    run_package_command uv pip install ;;
            esac
        ' bash "$INSTALL_SH" "$fixture" "$command_kind" >"$fixture/output" 2>&1 &
        installer_pid=$!
        for ((attempt=0; attempt<100; attempt++)); do
            [ ! -e "$fixture/ready" ] || break
            sleep 0.02
        done
        kill -s "$signal" "$installer_pid"
        signal_status=0; wait "$installer_pid" || signal_status=$?
        sleep 1.2
        expected_status=129; [ "$signal" != TERM ] || expected_status=143
        if [ "$signal_status" = "$expected_status" ] && [ -e "$fixture/rollback" ] &&
           [ -e "$fixture/released" ] && [ ! -e "$fixture/alive-at-rollback" ] && [ ! -e "$fixture/late-write" ]; then
            ok "$signal stops $command_kind child before rollback and lock release"
        else
            bad "$signal stops $command_kind child before rollback and lock release (status $signal_status)"
        fi
    done
done

echo
printf 'passed %d, failed %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
