#!/bin/bash
# Rapid-MLX — OpenAI- and Anthropic-compatible LLM server and Mac app for Apple Silicon, built on MLX, focused on reliable tool calling for coding agents.
# Usage: curl -fsSL https://rapidmlx.com/install.sh | bash
#        curl ... | bash -s -- 0.12.15     # specific version
#        RAPID_MLX_GIT_COMMIT=<full SHA> bash install.sh latest # reviewed source
set -euo pipefail

TARGET="${1:-stable}"  # stable (PyPI) | latest (explicit Git commit) | x.y.z (specific version)

INSTALL_DIR="${HOME}/.rapid-mlx"
INSTALL_LOCK="${INSTALL_DIR}.lock"
LEGACY_INSTALL_LOCK="${INSTALL_DIR}.install-lock"
INSTALL_LOCK_MODE=""
BIN_DIR="${HOME}/.local/bin"
PYPI_PACKAGE="rapid-mlx"
GITHUB_REPO="https://github.com/raullenchai/Rapid-MLX.git"
MIN_PYTHON_MINOR=12
# Newest validated interpreter; raise together with the classifiers and CI matrix.
MAX_PYTHON_MINOR=13
VENV_BACKUP=""
PRESERVED_EXTRAS=""
OLD_VENV_PYTHON=""
INSTALL_CHILD_PID=""
INSTALL_CONFIG_DIR=""
INSTALL_FILTER_PID=""

# ── Helpers ──────────────────────────────────────────────────────────────────

BOLD='' DIM='' GREEN='' YELLOW='' RESET=''
STDERR_RED='' STDERR_DIM='' STDERR_RESET=''
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "$TERM" != dumb ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD='\033[1m' DIM='\033[2m' GREEN='\033[32m' YELLOW='\033[33m' RESET='\033[0m'
fi
if [ -t 2 ] && [ -n "${TERM:-}" ] && [ "$TERM" != dumb ] && [ -z "${NO_COLOR:-}" ]; then
    STDERR_RED='\033[31m' STDERR_DIM='\033[2m' STDERR_RESET='\033[0m'
fi

info()  { printf "  ${BOLD}%s${RESET}\n" "$*"; }
ok()    { printf "  ${GREEN}%s${RESET}\n" "$*"; }
warn()  { printf "  ${YELLOW}%s${RESET}\n" "$*"; }
err()   { printf "  ${STDERR_RED}%s${STDERR_RESET}\n" "$*" >&2; }
dim()   { printf "  ${DIM}%s${RESET}\n" "$*"; }
dim_err() { printf "  ${STDERR_DIM}%s${STDERR_RESET}\n" "$*" >&2; }

print_success_box() {
    local text width=37 padding first=true i
    for text in "$@"; do
        if [ $((${#text} + 4)) -gt "$width" ]; then width=$((${#text} + 4)); fi
    done
    printf '  ╭'
    for ((i = 0; i < width; i++)); do printf '─'; done
    printf '╮\n'
    for text in "$@"; do
        padding=$((width - ${#text} - 2))
        printf '  │  '
        if [ "$first" = true ]; then printf '%b' "$GREEN"; fi
        printf '%s' "$text"
        if [ "$first" = true ]; then printf '%b' "$RESET"; first=false; fi
        printf '%*s│\n' "$padding" ''
    done
    printf '  ╰'
    for ((i = 0; i < width; i++)); do printf '─'; done
    printf '╯\n'
}

# With curl | bash, stdin is installer source, not input for child commands.
# Keep terminal input available for interactive runs; internal data pipelines
# (such as the cache JSON reader) must keep their own explicit stdin.
run_install_command() {
    local status=0
    if [ -t 0 ]; then
        "$@" <&0 &
    else
        "$@" </dev/null &
    fi
    INSTALL_CHILD_PID=$!
    while :; do
        status=0
        wait "$INSTALL_CHILD_PID" || status=$?
        # A returning signal handler can interrupt wait while the child lives.
        if [ "$status" -le 128 ] || ! kill -0 "$INSTALL_CHILD_PID" 2>/dev/null; then
            break
        fi
    done
    INSTALL_CHILD_PID=""
    return "$status"
}

collect_install_descendants() {
    local children child
    if command -v pgrep >/dev/null 2>&1; then
        children=$(pgrep -P "$1" 2>/dev/null || true)
    else
        children=$(ps -A -o pid=,ppid= | awk -v parent="$1" '$2 == parent { print $1 }')
    fi
    for child in $children; do
        printf '%s\n' "$child"
        collect_install_descendants "$child"
    done
}

stop_install_child() {
    local attempt pid pids snapshot survivors
    [ -n "$INSTALL_CHILD_PID" ] || return 0
    if kill -0 "$INSTALL_CHILD_PID" 2>/dev/null; then
        # Freeze the tree before termination so descendants cannot keep spawning
        # writers or escape by reparenting while their ancestors are stopped.
        kill -STOP "$INSTALL_CHILD_PID" 2>/dev/null || true
        pids="$INSTALL_CHILD_PID"
        for ((attempt = 0; attempt < 100; attempt++)); do
            snapshot=$({ printf '%s\n' "$pids"; collect_install_descendants "$INSTALL_CHILD_PID"; } | sort -nu)
            for pid in $snapshot; do
                kill -STOP "$pid" 2>/dev/null || true
            done
            [ "$snapshot" != "$pids" ] || break
            pids="$snapshot"
        done
        # Queue TERM for the whole stopped tree before allowing handlers to run.
        for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
        for pid in $pids; do kill -CONT "$pid" 2>/dev/null || true; done
        for ((attempt = 0; attempt < 100; attempt++)); do
            survivors=""
            for pid in $pids; do
                if kill -0 "$pid" 2>/dev/null; then survivors="$survivors $pid"; fi
            done
            pids="$survivors"
            [ -n "$pids" ] || break
            sleep 0.1
        done
        for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
    fi
    wait "$INSTALL_CHILD_PID" 2>/dev/null || true
    INSTALL_CHILD_PID=""
}

# Download function — works with curl or wget
DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    echo "Either curl or wget is required but neither is installed" >&2
    exit 1
fi

download() {
    if [ "$DOWNLOADER" = "curl" ]; then
        run_install_command curl -fsSL "$1"
    else
        run_install_command wget -qO- "$1"
    fi
}

# ── Interpreter + venv helpers (also sourced by tests/install) ────────────────

# Echo the first python3.x within the supported minor range on PATH that is a NATIVE arm64
# build, else nothing. MLX ships arm64 wheels only: a uv/pyenv-managed x86_64
# interpreter earlier in PATH would build a Rosetta venv that cannot import mlx
# and, historically, left ~/.rapid-mlx half-built with no bin/pip (#1953). The
# arch note goes to stderr so command substitution captures only the name.
select_python() {
    local py ver major minor pyarch
    for py in python3.13 python3.12 python3; do
        command -v "$py" >/dev/null 2>&1 || continue
        ver=$(run_install_command "$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
        major="${ver%%.*}"; minor="${ver#*.}"
        { [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; } || continue
        [ "$minor" -le "$MAX_PYTHON_MINOR" ] || continue
        pyarch=$(run_install_command "$py" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
        if [ "$pyarch" != "arm64" ]; then
            dim_err "Skipping $py (${pyarch}; Rapid-MLX needs a native arm64 Python)"
            continue
        fi
        printf '%s\n' "$py"
        return 0
    done
    return 1
}

# A venv is reusable only if it is COMPLETE (python + pip present and runnable)
# AND COMPATIBLE (a native arm64 interpreter, Python MIN_PYTHON_MINOR through
# MAX_PYTHON_MINOR) — the same supported range as fresh selection. Anything else is treated as
# not-installed and rebuilt under the newly selected interpreter:
#   - missing bin/pip: an aborted earlier run left it out (the original #1953).
#   - executable-but-broken pip: a ~/.vllm-mlx -> ~/.rapid-mlx migration leaves a
#     stale shebang, so `-x` passes but running fails — hence the `--version` run.
#   - wrong arch / too-old or too-new version: a venv built under Rosetta x86_64,
#     Python 3.9, or Python 3.14 can run but must not survive the upgrade path.
venv_is_complete() {
    local dir="$1" arch ver major minor
    [ -x "$dir/bin/python" ] && [ -x "$dir/bin/pip" ] || return 1
    run_install_command "$dir/bin/python" -c 'import sys' >/dev/null 2>&1 || return 1
    run_install_command "$dir/bin/pip" --version >/dev/null 2>&1 || return 1
    arch=$(run_install_command "$dir/bin/python" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
    [ "$arch" = "arm64" ] || return 1
    ver=$(run_install_command "$dir/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
    major="${ver%%.*}"; minor="${ver#*.}"
    [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ] && [ "$minor" -le "$MAX_PYTHON_MINOR" ]
}

# Infer optional features from installed requirement distributions in the old venv.
installed_extras_of_venv() {
    local dir="$1" result
    result=$(run_install_command "$dir/bin/python" -I -c '
import importlib.metadata as metadata
import re

try:
    from packaging.requirements import Requirement
except ImportError:
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        Requirement = None

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

try:
    requirements = metadata.distribution("rapid-mlx").requires or []
except metadata.PackageNotFoundError:
    try:
        requirements = metadata.distribution("vllm-mlx").requires or []
    except metadata.PackageNotFoundError:
        raise SystemExit(0)
installed = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata.get("Name")}
base = set()
extras = {}
for requirement in requirements:
    name = re.match(r"[A-Za-z0-9._-]+", requirement)
    if not name:
        continue
    marker = requirement.partition(";")[2]
    extra = re.search(r"\bextra\s*==\s*[\"\x27]([^\"\x27]+)[\"\x27]", marker, re.I)
    if Requirement is not None:
        try:
            req = Requirement(requirement)
            if req.marker and not req.marker.evaluate({"extra": extra.group(1) if extra else ""}):
                continue
        except Exception:
            continue
    else:
        if re.search(r"\bpython_(?:full_)?version\b", marker, re.I):
            continue
        if re.search(r"\b(?:platform_system\s*==\s*[\"\x27]Windows[\"\x27]|sys_platform\s*==\s*[\"\x27]win32[\"\x27])", marker, re.I):
            continue
    name = normalize(name.group())
    if extra:
        extras.setdefault(normalize(extra.group(1)), []).append((name, req if Requirement is not None else None))
    else:
        base.add(name)
pending = list(base)
while pending:
    name = pending.pop()
    try:
        dependencies = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        continue
    for dependency in dependencies:
        match = re.match(r"[A-Za-z0-9._-]+", dependency)
        if not match:
            continue
        if Requirement is not None:
            try:
                req = Requirement(dependency)
                if req.marker and not req.marker.evaluate({"extra": ""}):
                    continue
            except Exception:
                continue
        child = normalize(match.group())
        if child not in base:
            base.add(child)
            pending.append(child)
selected = []
for extra, requirements in extras.items():
    distinct = [(name, req) for name, req in requirements if name not in base]
    if (extra not in {"dev", "test", "all"} and not extra.startswith("ci-") and distinct
            and all(name in installed and (req is None or req.specifier.contains(installed[name], prereleases=True))
                    for name, req in distinct)):
        selected.append(extra)
if selected:
    print(",".join(sorted(selected)))
' 2>/dev/null) || return 0
    [ -z "$result" ] || printf '%s\n' "$result"
    return 0
}

# Rebuilds move only venv entries aside, preserving shared state. The new venv
# must be created at its final path because venv scripts contain absolute paths.
# A completion marker distinguishes a full stash; recover backups before planning.
reset_venv_dir() {
    local dir="$1"
    rm -rf "$dir/bin" "$dir/lib" "$dir/lib64" "$dir/include" "$dir/share" "$dir/pyvenv.cfg"
}

stash_venv_entries() {
    local dir="$1" backup="$2" entry
    mkdir "$backup" || return 1
    for entry in bin lib lib64 include share pyvenv.cfg; do
        if [ -e "$dir/$entry" ] || [ -L "$dir/$entry" ]; then
            if ! mv "$dir/$entry" "$backup/"; then
                restore_venv_entries "$dir" "$backup" || true
                return 1
            fi
        fi
    done
    : > "$backup/.stash-complete"
}

restore_venv_entries() {
    local dir="$1" backup="$2" entry
    if [ -e "$backup/.stash-complete" ]; then
        reset_venv_dir "$dir" || return 1
        # Once the replacement is gone, a retry must retain entries already restored.
        rm -f "$backup/.stash-complete" || return 1
    fi
    for entry in bin lib lib64 include share pyvenv.cfg; do
        [ -e "$backup/$entry" ] || [ -L "$backup/$entry" ] || continue
        rm -rf "${dir:?}/$entry" || return 1
        mv "$backup/$entry" "$dir/" || return 1
    done
    rmdir "$backup" || return 1
}

recover_interrupted_rebuild() {
    local backup newest=""
    [ -d "$INSTALL_DIR" ] || return 0
    for backup in "$INSTALL_DIR"/.venv-discard.*; do
        [ -d "$backup" ] || continue
        rm -rf "$backup"
    done
    for backup in "$INSTALL_DIR"/.venv-previous.*; do
        [ -d "$backup" ] || continue
        if [ -z "$newest" ] || [ "$backup" -nt "$newest" ]; then
            newest="$backup"
        fi
    done
    [ -n "$newest" ] || return 0
    if ! restore_venv_entries "$INSTALL_DIR" "$newest"; then
        err "Could not restore the previous environment; it is preserved at $newest"
        exit 1
    fi
    warn "Recovered the previous Rapid-MLX environment from an interrupted install."
    for backup in "$INSTALL_DIR"/.venv-previous.*; do
        [ -d "$backup" ] || continue
        rm -rf "$backup"
    done
}

rollback_rebuild() {
    if [ -n "$VENV_BACKUP" ] && [ -d "$VENV_BACKUP" ]; then
        if restore_venv_entries "$INSTALL_DIR" "$VENV_BACKUP"; then
            warn "Rebuild failed — restored the previous Rapid-MLX environment unchanged."
            dim "Install Python 3.12 or 3.13 / check the network and re-run."
        else
            err "Rebuild failed and the previous environment could not be restored automatically; it is preserved at $VENV_BACKUP (the next install run restores it)."
        fi
        VENV_BACKUP=""
    fi
}

commit_rebuild() {
    local backup discard="$INSTALL_DIR/.venv-discard.$$"
    mv "$VENV_BACKUP" "$discard"
    VENV_BACKUP=""
    for backup in "$INSTALL_DIR"/.venv-previous.* "$INSTALL_DIR"/.venv-discard.*; do
        [ -d "$backup" ] || continue
        rm -rf "$backup"
    done
}

finish_rebuild() {
    [ -n "$VENV_BACKUP" ] || return 0
    if ! run_install_command "$INSTALL_DIR/bin/python" -c 'import rapid_mlx' >/dev/null 2>&1 ||
       ! run_install_command "$INSTALL_DIR/bin/rapid-mlx" --version >/dev/null 2>&1; then
        err "The rebuilt environment failed verification."
        exit 1
    fi
    commit_rebuild
}

acquire_install_lock() {
    local pid="" status=0 perl="${RAPID_INSTALL_PERL:-/usr/bin/perl}"
    if [ -L "$INSTALL_LOCK" ]; then
        err "The Rapid-MLX install lock $INSTALL_LOCK is a symlink. Remove it and re-run."
        exit 1
    fi
    if [ -x "$perl" ]; then
        if ! { exec 9>>"$INSTALL_LOCK"; } 2>/dev/null; then
            err "Could not open the Rapid-MLX install lock at $INSTALL_LOCK."
            exit 1
        fi
        "$perl" -MFcntl=:flock -e 'open(my $fh, ">&=", 9) or exit 2; flock($fh, LOCK_EX | LOCK_NB) or exit 1; exit 0' || status=$?
        case "$status" in
            0) INSTALL_LOCK_MODE=flock ;;
            1) err "Another Rapid-MLX install is already running. Wait for it to finish and re-run."; exit 1 ;;
            *) err "Could not acquire the Rapid-MLX install lock. Re-run the installer."; exit 1 ;;
        esac
    elif mkdir "${INSTALL_LOCK}.d" 2>/dev/null; then
        printf '%s\n' "$$" > "${INSTALL_LOCK}.d/pid"
        INSTALL_LOCK_MODE=dir
    else
        pid=$(cat "${INSTALL_LOCK}.d/pid" 2>/dev/null) || pid=""
        case "$pid" in
            ''|*[!0-9]*|0) ;;
            *) if kill -0 "$pid" 2>/dev/null; then
                err "Another Rapid-MLX install is already running. Wait for it to finish and re-run."
                exit 1
            fi ;;
        esac
        err "A previous Rapid-MLX install did not finish. If no installer is running, remove ${INSTALL_LOCK}.d and re-run."
        exit 1
    fi
    if [ -L "$LEGACY_INSTALL_LOCK" ] || [ -d "$LEGACY_INSTALL_LOCK" ]; then
        if [ -L "$LEGACY_INSTALL_LOCK" ]; then
            pid=$(readlink "$LEGACY_INSTALL_LOCK" 2>/dev/null) || pid=""
        else
            pid=$(cat "$LEGACY_INSTALL_LOCK/pid" 2>/dev/null) || pid=""
        fi
        case "$pid" in
            ''|*[!0-9]*|0) ;;
            *) if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
                err "Another Rapid-MLX install is already running. Wait for it to finish and re-run."
                release_install_lock
                exit 1
            fi ;;
        esac
        if [ -L "$LEGACY_INSTALL_LOCK" ]; then rm -f "$LEGACY_INSTALL_LOCK"; else rm -rf "$LEGACY_INSTALL_LOCK"; fi
    fi
    if ! ln -s "$$" "$LEGACY_INSTALL_LOCK" 2>/dev/null; then
        if [ -L "$LEGACY_INSTALL_LOCK" ]; then
            pid=$(readlink "$LEGACY_INSTALL_LOCK" 2>/dev/null) || pid=""
        else
            pid=$(cat "$LEGACY_INSTALL_LOCK/pid" 2>/dev/null) || pid=""
        fi
        case "$pid" in
            ''|*[!0-9]*|0) ;;
            *) if kill -0 "$pid" 2>/dev/null; then
                err "Another Rapid-MLX install is already running. Wait for it to finish and re-run."
                release_install_lock
                exit 1
            fi ;;
        esac
        err "Could not acquire the Rapid-MLX install lock. Re-run the installer."
        release_install_lock
        exit 1
    fi
    LEGACY_LOCK_OWNED=1
}

release_install_lock() {
    local pid=""
    if [ "${LEGACY_LOCK_OWNED:-}" = 1 ]; then
        pid=$(readlink "$LEGACY_INSTALL_LOCK" 2>/dev/null) || pid=""
        [ "$pid" != "$$" ] || rm -f "$LEGACY_INSTALL_LOCK"
    fi
    LEGACY_LOCK_OWNED=""
    case "${INSTALL_LOCK_MODE:-}" in
        flock) exec 9>&- ;;
        dir)
            pid=$(cat "${INSTALL_LOCK}.d/pid" 2>/dev/null) || pid=""
            [ "$pid" != "$$" ] || rm -rf "${INSTALL_LOCK}.d"
            ;;
    esac
    INSTALL_LOCK_MODE=""
}

on_installer_exit() {
    trap '' HUP INT TERM
    stop_install_child
    if [ -n "$INSTALL_FILTER_PID" ]; then
        kill -TERM "$INSTALL_FILTER_PID" 2>/dev/null || true
        wait "$INSTALL_FILTER_PID" 2>/dev/null || true
    fi
    [ -z "$INSTALL_CONFIG_DIR" ] || rm -rf "$INSTALL_CONFIG_DIR"
    rollback_rebuild
    release_install_lock
}

# Metadata normalizes extra names to letters, digits, and internal hyphens.
valid_extras() {
    [[ "$1" =~ ^[a-z0-9]+(-[a-z0-9]+)*(,[a-z0-9]+(-[a-z0-9]+)*)*$ ]]
}

load_pending_extras() {
    local pending
    [ -z "$PRESERVED_EXTRAS" ] && [ -f "$INSTALL_DIR/.pending-extras" ] || return 0
    pending=$(cat "$INSTALL_DIR/.pending-extras") || return 1
    if valid_extras "$pending"; then
        PRESERVED_EXTRAS="$pending"
    else
        rm -f "$INSTALL_DIR/.pending-extras"
    fi
}

persist_pending_extras() {
    local temporary
    [ -n "$PRESERVED_EXTRAS" ] || return 0
    valid_extras "$PRESERVED_EXTRAS" || return 1
    temporary=$(mktemp "$INSTALL_DIR/.pending-extras.XXXXXX") || return 1
    if ! printf '%s\n' "$PRESERVED_EXTRAS" > "$temporary" ||
       ! mv "$temporary" "$INSTALL_DIR/.pending-extras"; then
        rm -f "$temporary"
        return 1
    fi
}

# Decide whether to create, upgrade, or transactionally rebuild a venv.
plan_venv_action() {
    local dir="$1"
    if [ ! -d "$dir" ]; then echo create; return; fi
    if venv_is_complete "$dir"; then echo upgrade; else echo rebuild; fi
}

# Best-effort pip self-upgrade: a transient failure here must not abort install.
upgrade_pip() {
    run_install_command "$INSTALL_DIR/bin/pip" install --upgrade pip -q 2>/dev/null || true
}

# Build the venv at $INSTALL_DIR under $PYTHON and verify it came out usable.
create_venv() {
    run_install_command "$PYTHON" -m venv "$INSTALL_DIR"
    if ! venv_is_complete "$INSTALL_DIR"; then
        err "Could not create a working virtual environment at $INSTALL_DIR."
        dim "Selected Python: $(run_install_command "$PYTHON" -c 'import platform, sys; print(platform.machine(), sys.version.split()[0])' 2>/dev/null || echo unknown)"
        dim "Install an arm64 Python 3.12 or 3.13 (e.g. 'brew install python@3.12') and re-run."
        exit 1
    fi
    upgrade_pip
}

# Carry out a planned action. Kept as its own function (not an inline case) so
# the create/upgrade/rebuild wiring is unit-testable with mocked actions.
dispatch_venv() {
    load_pending_extras
    case "$1" in
        upgrade)
            info "Upgrading Rapid-MLX..."
            upgrade_pip
            ;;
        rebuild)
            if [ -z "$PRESERVED_EXTRAS" ]; then
                PRESERVED_EXTRAS="$(installed_extras_of_venv "$INSTALL_DIR" || true)"
            fi
            persist_pending_extras || return 1
            OLD_VENV_PYTHON="$(run_install_command "$INSTALL_DIR/bin/python" -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))' 2>/dev/null || true)"
            if [[ "$OLD_VENV_PYTHON" =~ ^[0-9]+\.[0-9]+$ ]] &&
               { [ "${OLD_VENV_PYTHON#*.}" -lt "$MIN_PYTHON_MINOR" ] || [ "${OLD_VENV_PYTHON#*.}" -gt "$MAX_PYTHON_MINOR" ]; }; then
                warn "Existing $INSTALL_DIR uses Python $OLD_VENV_PYTHON; Rapid-MLX needs Python 3.${MIN_PYTHON_MINOR}–3.${MAX_PYTHON_MINOR} — rebuilding it."
            else
                warn "Existing $INSTALL_DIR has no usable virtual environment — rebuilding it."
            fi
            dim "(shared state is preserved; the previous environment is kept until the new one works)"
            [ -z "$PRESERVED_EXTRAS" ] || dim "Optional features to reinstall: $PRESERVED_EXTRAS"
            VENV_BACKUP="$INSTALL_DIR/.venv-previous.$$"
            stash_venv_entries "$INSTALL_DIR" "$VENV_BACKUP"
            info "Installing Rapid-MLX..."
            dim "(this takes about a minute)"
            create_venv
            ;;
        *)
            info "Installing Rapid-MLX..."
            dim "(this takes about a minute)"
            create_venv
            ;;
    esac
}

# First-chat policy shared with Desktop onboarding (#2385): start with a small,
# reliable baseline, but reuse a known RAM-safe cached model when one is already
# runnable. The baseline mirrors ``QuickstartCoordinator.baselineChoice``:
# <16 GB uses the safe 1.2B starter (Desktop's ``lowMemoryChoice``), >=16 GB the
# 4B starter. Cached candidates include both curated picks from the current
# RAM tier and every lower tier, walked downward; aliases above the current
# tier never enter the list. The baseline is used only when none are cached.
starter_baseline_for_ram() {
    if [ "$1" -ge 16 ]; then
        printf '%s\n' "qwen3.5-4b-4bit"
    else
        printf '%s\n' "lfm2.5-1b-4bit"
    fi
}

starter_cached_order_for_ram() {
    local ram="$1"
    if [ "$ram" -le 0 ]; then printf '%s\n' lfm2.5-1b-4bit
    elif [ "$ram" -ge 48 ]; then printf '%s\n' qwen3.8-27b-4bit qwen3.6-35b-4bit qwen3.5-4b-4bit bonsai-27b-2bit qwen3.5-9b-4bit lfm2.5-1b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 32 ]; then printf '%s\n' qwen3.8-27b-4bit qwen3.5-4b-4bit bonsai-27b-2bit qwen3.5-9b-4bit lfm2.5-1b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 24 ]; then printf '%s\n' bonsai-27b-2bit qwen3.5-4b-4bit qwen3.5-9b-4bit lfm2.5-1b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 18 ]; then printf '%s\n' qwen3.5-9b-4bit qwen3.5-4b-4bit lfm2.5-1b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 16 ]; then printf '%s\n' qwen3.5-4b-4bit lfm2.5-1b-4bit lfm2.5-2.6b-4bit
    else printf '%s\n' lfm2.5-2.6b-4bit lfm2.5-1b-4bit
    fi
}

starter_alias_is_cached() {
    local wanted="$1" cached="$2" alias
    while IFS= read -r alias; do
        [ "$alias" = "$wanted" ] && return 0
    done <<EOF
$cached
EOF
    return 1
}

select_starter_model() {
    local ram="$1" cached="${2:-}" order candidate
    order="$(starter_cached_order_for_ram "$ram")"
    while IFS= read -r candidate; do
        if starter_alias_is_cached "$candidate" "$cached"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done <<EOF
$order
EOF
    starter_baseline_for_ram "$ram"
}

installed_cached_aliases() {
    run_install_command "$INSTALL_DIR/bin/rapid-mlx" models --cached --json 2>/dev/null |
        "$INSTALL_DIR/bin/python" -c 'import json,sys; p=json.load(sys.stdin); print("\n".join(m["alias"] for m in p.get("cached", []) if m.get("state") == "ok" and isinstance(m.get("alias"), str)))' 2>/dev/null
}

refresh_starter_from_installed_cache() {
    local cached_aliases
    if ! cached_aliases="$(installed_cached_aliases)"; then
        cached_aliases=""
    fi
    RECOMMENDED_MODEL="$(select_starter_model "$RAM_GB" "$cached_aliases")"
}

# rapidmlx.com/leaderboard?mac=<slug>: what runs well on this Mac class.
# Same slug as rapid_mlx/leaderboard_links.py (tests/test_leaderboard_links.py
# keeps them in agreement): "Apple M4 Pro" + 48 -> m4-pro-48. Anything that is
# not an Apple M-series brand string falls back to the board itself.
leaderboard_mac_url() {   # leaderboard_mac_url <cpu brand string> <ram GiB>
    local brand ram re gen variant slug
    brand="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    ram="$2"
    slug=""
    # The pattern lives in a variable: bash 3.2 (macOS /bin/bash) and 5.x
    # disagree about quoting inside [[ =~ ]]. The RAM check runs first because
    # every =~ overwrites BASH_REMATCH.
    re='^apple m([0-9]+)( (pro|max|ultra))?$'
    case "$ram" in ''|0*|*[!0-9]*) ram="" ;; esac
    if [ -n "$ram" ] && [[ "$brand" =~ $re ]]; then
        gen="${BASH_REMATCH[1]}"
        variant="${BASH_REMATCH[3]:-}"
        slug="m${gen}${variant:+-$variant}-${ram}"
    fi
    if [ -n "$slug" ]; then
        printf 'https://rapidmlx.com/leaderboard?mac=%s\n' "$slug"
    else
        printf 'https://rapidmlx.com/leaderboard\n'
    fi
}

print_quick_start_commands() {
    info "Quick start:"
    echo ""
    echo "    rapid-mlx chat ${RECOMMENDED_MODEL}${RECOMMENDED_FLAGS}    # starts its own server"
    echo ""
    dim "More useful commands:"
    echo ""
    echo "    rapid-mlx models"
    echo "    rapid-mlx pull ${RECOMMENDED_MODEL}"
    echo "    rapid-mlx doctor"
    echo ""
    dim "Run a local model from a coding agent (install the agent first):"
    echo ""
    echo "    rapid-mlx launch codex"
    echo "    # Replace codex with claude-code, opencode, pi, hermes, or crush."
    echo "    # Launch checks the agent's model needs against your Mac RAM; small Macs may need more memory."
    echo "    # Run the same command tomorrow; add --remove to delete Rapid's profile."
    echo ""
    if "$INSTALL_DIR/bin/python" -I -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec("gradio") is None)' </dev/null; then
        echo "    rapid-mlx-chat    # web UI"
    else
        echo "    rapid-mlx-chat    # web UI (first: ${INSTALL_DIR}/bin/python -m pip install 'rapid-mlx[chat]')"
    fi
    echo ""
    dim "Measured speeds on Macs like this one:"
    echo ""
    echo "    $(leaderboard_mac_url "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)" "$RAM_GB")"
}

# Validate before changing the environment or downloading dependencies.
validate_target() {
    case "$TARGET" in
        stable) ;;
        latest)
            if [[ ! "${RAPID_MLX_GIT_COMMIT:-}" =~ ^[0-9a-fA-F]{40}$ ]]; then
                err "latest requires RAPID_MLX_GIT_COMMIT set to a reviewed full 40-character Git commit SHA."
                return 1
            fi
            ;;
        *)
            # Keep version syntax compatible; the package manager validates it.
            # Reject whitespace and requirement/URL separators before any work.
            if [[ ! "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.+_-]*$ ]]; then
                err "Usage: install.sh [stable|latest|VERSION]"
                return 1
            fi
            ;;
    esac
}

uv_config_absent() {
    local dir root physical_dir config_dir config_dirs=${XDG_CONFIG_DIRS:-}
    [ -z "${UV_CONFIG_FILE+x}" ] && [ -z "${UV_NO_CONFIG+x}" ] || return 1
    for config_dir in "${XDG_CONFIG_HOME:-$HOME/.config}" /etc "${config_dirs//:/$'\n'}"; do
        while IFS= read -r dir; do
            [ -z "$dir" ] && continue
            [ ! -e "$dir/uv/uv.toml" ] && [ ! -L "$dir/uv/uv.toml" ] || return 1
        done <<< "$config_dir"
    done
    physical_dir=$(pwd -P) || return 1
    for root in "$PWD" "$physical_dir"; do
        dir=$root
        while :; do
            [ ! -e "$dir/uv.toml" ] && [ ! -L "$dir/uv.toml" ] || return 1
            if [ -e "$dir/pyproject.toml" ] || [ -L "$dir/pyproject.toml" ]; then
                "$INSTALL_DIR/bin/python" -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    doc = tomllib.load(f)
tool = doc.get("tool")
sys.exit(not ("tool" not in doc or isinstance(tool, dict) and "uv" not in tool))
' "$dir/pyproject.toml" </dev/null 2>/dev/null || return 1
            fi
            [ "$dir" = / ] && break
            dir=${dir%/*}
            [ -n "$dir" ] || dir=/
        done
    done
}

# Keep the writer in this shell so exit cleanup can stop it before rollback.
run_package_command() {
    local status=0
    if [ "$1" = uv ]; then
        if uv_config_absent && INSTALL_CONFIG_DIR=$(mktemp -d 2>/dev/null); then
            if cat > "$INSTALL_CONFIG_DIR/uv.toml" <<'EOF'
[[dependency-metadata]]
name = "mlx-video-with-audio"
version = "0.1.36"
requires-dist = ["mlx>=0.22.0", "numpy", "safetensors", "huggingface_hub", "transformers", "tqdm",
                 "opencv-python>=4.12.0.88", "Pillow>=10.3.0", "mlx-vlm", "soundfile>=0.12.0",
                 "pytest; extra == \"dev\""]
EOF
            then
                UV_CONFIG_FILE="$INSTALL_CONFIG_DIR/uv.toml" run_install_command "$@" || status=$?
            else
                run_install_command "$@" || status=$?
            fi
            rm -rf "$INSTALL_CONFIG_DIR"
            INSTALL_CONFIG_DIR=""
        else
            run_install_command "$@" || status=$?
        fi
    else
        # A FIFO keeps stderr streaming without putting the writer in a pipeline.
        if command -v grep >/dev/null 2>&1 && command -v mkfifo >/dev/null 2>&1 &&
           INSTALL_CONFIG_DIR=$(mktemp -d 2>/dev/null) &&
           mkfifo "$INSTALL_CONFIG_DIR/stderr"; then
            grep --line-buffered -Ev '^WARNING: transformers [^ ]+ does not provide the extra '\''tokenizers'\''$' < "$INSTALL_CONFIG_DIR/stderr" >&2 &
            INSTALL_FILTER_PID=$!
            run_install_command "$@" 2> "$INSTALL_CONFIG_DIR/stderr" || status=$?
            wait "$INSTALL_FILTER_PID" || true
            INSTALL_FILTER_PID=""
        else
            run_install_command "$@" || status=$?
        fi
        [ -z "$INSTALL_CONFIG_DIR" ] || rm -rf "$INSTALL_CONFIG_DIR"
        INSTALL_CONFIG_DIR=""
    fi
    return "$status"
}

install_package() {
    case "$TARGET" in
        stable)
            run_package_command "${UPGRADE_INSTALLER[@]}" "$PYPI_PACKAGE" || {
                err "Package-index install failed; refusing to fall back to GitHub. Check the error above and retry."
                return 1
            }
            ;;
        latest)
            validate_target || return 1
            info "Installing reviewed Git commit ${RAPID_MLX_GIT_COMMIT}..."
            run_package_command "${FORCE_INSTALLER[@]}" "$PYPI_PACKAGE @ git+${GITHUB_REPO}@${RAPID_MLX_GIT_COMMIT}"
            ;;
        *)
            info "Installing version ${TARGET}..."
            run_package_command "${INSTALLER[@]}" "${PYPI_PACKAGE}==${TARGET}" || {
                err "Requested version unavailable from the package index; refusing to change sources."
                return 1
            }
            ;;
    esac
}

# Restoring optional features is best effort after the rebuild is committed.
reinstall_preserved_extras() {
    local version spec
    load_pending_extras
    [ -n "$PRESERVED_EXTRAS" ] || return 0
    version=$(run_install_command "$INSTALL_DIR/bin/python" -I -c 'import importlib.metadata as m; print(m.version("rapid-mlx"))' 2>/dev/null) || version=""
    if [ "$TARGET" = latest ]; then
        spec="${PYPI_PACKAGE}[${PRESERVED_EXTRAS}] @ git+${GITHUB_REPO}@${RAPID_MLX_GIT_COMMIT:-}"
    elif [ -n "$version" ]; then
        spec="${PYPI_PACKAGE}[${PRESERVED_EXTRAS}]==${version}"
    else
        spec="${PYPI_PACKAGE}[${PRESERVED_EXTRAS}]"
    fi
    info "Reinstalling optional features: $PRESERVED_EXTRAS"
    if { [ "$TARGET" != latest ] && [ -z "$version" ]; } || ! run_package_command "${INSTALLER[@]}" "$spec"; then
        warn "Could not reinstall optional features ($PRESERVED_EXTRAS)."
        dim "Install them later with: $INSTALL_DIR/bin/python -m pip install '$spec'"
    else
        rm -f "$INSTALL_DIR/.pending-extras"
    fi
    return 0
}

standalone_python_reusable() {
    [ -x "$STANDALONE_DIR/bin/python3" ] &&
       [ "$(run_install_command "$STANDALONE_DIR/bin/python3" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)" = "$PY_VERSION" ] &&
       [ "$(run_install_command "$STANDALONE_DIR/bin/python3" -c 'import platform; print(platform.machine())' 2>/dev/null || true)" = arm64 ]
}

install_standalone_python() (
    if standalone_python_reusable; then
        dim "Reusing Python $PY_VERSION in ~/.rapid-mlx-python"
        return 0
    fi
    info "Downloading Python ${PY_VERSION} (build ${PY_BUILD}) + verifying SHA256..."
    # Verify the tarball against the publisher-published SHA256SUMS for this
    # exact release BEFORE anything is extracted. A curl|bash install trusts
    # the transport (HTTPS) for integrity; verifying the checksum removes the
    # "trust whatever bytes came back" gap so a tampered / mis-served
    # download fails closed instead of silently installing.
    TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rapidmlx-py.XXXXXX")"
    TMP_TAR="$TMP_DIR/python.tar.gz"
    STAGING_DIR=""
    trap 'rm -rf "$TMP_DIR"; [ -z "$STAGING_DIR" ] || rm -rf "$STAGING_DIR"' EXIT
    download "$PY_URL" > "$TMP_TAR"
    ACTUAL="$(shasum -a 256 "$TMP_TAR" | awk '{print $1}')"
    if [ "$ACTUAL" != "$PY_SHA256" ]; then
        err "SHA256 mismatch for Python standalone tarball."
        dim "Expected $PY_SHA256"
        dim "Got      $ACTUAL"
        dim "Refusing to extract an unverified runtime."
        exit 1
    fi
    dim "SHA256 verified."
    STAGING_DIR="$(mktemp -d "${STANDALONE_DIR}.new.XXXXXX")" || exit 1
    tar xzf "$TMP_TAR" -C "$STAGING_DIR" --strip-components=1 || exit 1
    PYTHON="${STAGING_DIR}/bin/python3"
    if ! run_install_command "$PYTHON" --version >/dev/null 2>&1; then
        err "Failed to install standalone Python."
        dim "Please install Python 3.12 or 3.13 from https://www.python.org/downloads/"
        exit 1
    fi
    OLD_DIR="${STANDALONE_DIR}.old.$$"
    if [ -e "$STANDALONE_DIR" ]; then
        mv "$STANDALONE_DIR" "$OLD_DIR" || exit 1
    fi
    if ! mv "$STAGING_DIR" "$STANDALONE_DIR"; then
        if [ -e "$OLD_DIR" ]; then
            mv "$OLD_DIR" "$STANDALONE_DIR"
        fi
        exit 1
    fi
    STAGING_DIR=""
    rm -rf "$OLD_DIR"
    PYTHON="${STANDALONE_DIR}/bin/python3"
    ok "Installed $(run_install_command "$PYTHON" --version 2>&1)"
)

restart_service_hints() {
    [ "$1" != create ] || return 0
    local daemon="${RAPID_INSTALL_LAUNCH_DAEMONS_DIR:-/Library/LaunchDaemons}/com.rapidmlx.server.plist" agent found=0
    if [ -f "$daemon" ]; then
        warn "Background services still run the previous version until restarted:"
        found=1
        dim 'sudo rapid-mlx service restart'
    fi
    if [ -x "$INSTALL_DIR/bin/python" ]; then
        local hint
        while IFS= read -r hint; do
            [ -n "$hint" ] || continue
            if [ "$found" = 0 ]; then
                warn "Background services still run the previous version until restarted:"
                found=1
            fi
            dim "$hint"
        done < <("$INSTALL_DIR/bin/python" -I -c '
import os, plistlib, shlex, sys
install, bin_dir, *paths = sys.argv[1:]
root = os.path.realpath(install)
for path in paths:
    if os.path.basename(path) == "com.rapidmlx.server.plist":
        continue
    try:
        with open(path, "rb") as f:
            plist = plistlib.load(f)
        if not isinstance(plist, dict):
            continue
        args = plist.get("ProgramArguments")
        if not isinstance(args, list) or not args or not isinstance(args[0], str):
            continue
        program = args[0]
        owned = program == os.path.join(bin_dir, "rapid-mlx") or os.path.commonpath([root, os.path.realpath(program)]) == root
        label = plist.get("Label")
        if owned and isinstance(label, str) and label != "com.rapidmlx.server":
            print("sudo rapid-mlx service restart --label " + shlex.quote(label))
    except Exception:
        continue
' "$INSTALL_DIR" "$BIN_DIR" "${RAPID_INSTALL_LAUNCH_DAEMONS_DIR:-/Library/LaunchDaemons}"/*.plist 2>/dev/null || true)
    fi
    for agent in "$HOME"/Library/LaunchAgents/com.quicksilver.node.*.plist; do
        [ -f "$agent" ] || continue
        if [ "$found" = 0 ]; then
            warn "Background services still run the previous version until restarted:"
            found=1
        fi
        agent="${agent##*/}"
        dim 'launchctl kickstart -k gui/$(id -u)/'"${agent%.plist}"
    done
}

# Only a positive hardware identification warrants a specific diagnosis.
check_shell_architecture() {
    local arch="$1" probe="${RAPID_INSTALL_SYSCTL:-}" hardware
    [ "$arch" != arm64 ] || return 0
    if [ -z "$probe" ]; then
        if [ -x /usr/sbin/sysctl ]; then probe=/usr/sbin/sysctl; else probe=sysctl; fi
    fi
    if [ "$("$probe" -n hw.optional.arm64 2>/dev/null || true)" = 1 ] ||
       [ "$("$probe" -n sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
        err "This terminal is running under Rosetta (x86_64), but your Mac has Apple Silicon."
        dim "Open Terminal without Rosetta (Finder → Applications → Utilities → Terminal → Get Info → uncheck \"Open using Rosetta\") and run the installer again."
    else
        hardware=$("$probe" -n machdep.cpu.brand_string 2>/dev/null || true)
        if [[ "$hardware" == *Intel* ]]; then
            err "This Mac has an Intel processor. Rapid-MLX needs a Mac with Apple Silicon (M1 or later)."
            dim "MLX, the framework Rapid-MLX runs on, only supports Apple Silicon. Memory size doesn't change this."
        else
            err "This terminal reports $arch; the Mac processor could not be identified."
            dim "Open Terminal without Rosetta (Finder → Applications → Utilities → Terminal → Get Info → uncheck \"Open using Rosetta\") and run the installer again."
            dim "Rapid-MLX needs a Mac with Apple Silicon (M1 or later)."
        fi
    fi
    return 1
}

# When sourced by the test harness we only want the definitions above, not the
# installer itself. `return` succeeds only in a sourced context; the `|| exit 0`
# keeps an accidental `RAPID_INSTALL_LIB=1 bash install.sh` from erroring out.
if [ "${RAPID_INSTALL_LIB:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

validate_target

# ── Banner ───────────────────────────────────────────────────────────────────

echo ""
echo "  ╭─────────────────────────────────────╮"
echo "  │  Rapid-MLX — AI on Apple Silicon    │"
echo "  │  Up to 3x Ollama throughput         │"
echo "  ╰─────────────────────────────────────╯"
echo ""

# ── 1. Check platform ───────────────────────────────────────────────────────

case "$(uname -s)" in
    Darwin) ;;
    Linux)  err "Rapid-MLX requires macOS with Apple Silicon (MLX framework)."; exit 1 ;;
    *)      err "Unsupported OS: $(uname -s). Rapid-MLX requires macOS with Apple Silicon."; exit 1 ;;
esac

ARCH=$(uname -m)
check_shell_architecture "$ARCH" || exit 1

MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
if [ "$MACOS_VERSION" -lt 13 ]; then
    err "Rapid-MLX requires macOS 13 (Ventura) or later."
    dim "Detected: macOS $(sw_vers -productVersion)"
    exit 1
fi

# ── 2. Detect RAM → recommend model ──────────────────────────────────────────

# This is the Desktop Quickstart starter policy, not the larger "smart pick"
# shown in the model browser. A fresh install optimizes time-to-first-chat:
# <16 GB uses the safe 1.2B starter; every larger Mac uses the 4B starter.
# After installation, a structured cache query may promote this to a known
# runnable model that fits the same RAM tier.
RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1073741824}')
RECOMMENDED_FLAGS=""
RECOMMENDED_MODEL="$(select_starter_model "$RAM_GB" "")"
if [ "$RAM_GB" -ge 16 ]; then RAM_TIER="16+ GB"; else RAM_TIER="under 16 GB"; fi

dim "macOS $(sw_vers -productVersion) · Apple Silicon · ${RAM_GB} GB RAM"

# ── 3. Find or install Python 3.12 or 3.13 ──────────────────────────────────

acquire_install_lock
trap on_installer_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
PYTHON="$(select_python || true)"

STANDALONE_DIR="${HOME}/.rapid-mlx-python"
PY_VERSION="3.12.13"
# Pin to a FIXED release tag (not the moving "latest" API) so the
# tarball we fetch is a stable, reviewable artifact. The tag below is
# the concrete build this line ships against; bump it deliberately.
PY_BUILD="20260408"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_BUILD}/cpython-${PY_VERSION}+${PY_BUILD}-aarch64-apple-darwin-install_only.tar.gz"
# Publisher SHA256SUMS, reviewed and pinned with the release rather
# than downloaded beside the artifact. Fetching both bytes and digest
# from the same compromised release would not add a trust boundary.
PY_SHA256="6000d09545602d3704bdff943f37663b3148b7c1a3a8a1fcc6c1ebd505a3cfc3"

if [ -z "$PYTHON" ]; then
    if standalone_python_reusable; then
        dim "Using Python $PY_VERSION from ~/.rapid-mlx-python"
        PYTHON="${STANDALONE_DIR}/bin/python3"
    else
        echo ""
        warn "Python 3.12 or 3.13 not found. Installing Python 3.12 automatically..."
        if command -v brew >/dev/null 2>&1; then
            info "Installing Python 3.12 via Homebrew..."
            run_install_command brew install python@3.12
            # Resolve the formula's OWN keg path, not the bare name and not the
            # Homebrew root: python@3.12 is keg-only, so `$(brew --prefix)/bin` may
            # not carry python3.12, while a bare `python3.12` could still hit the
            # x86_64 build earlier on PATH (the reason we got here) and fail the
            # arch check below. The lookup is best-effort (|| true) so it can't abort
            # the install under set -e; fall back to the name if the keg path is gone.
            keg="$(run_install_command brew --prefix python@3.12 2>/dev/null || true)"
            if [ -n "$keg" ] && [ -x "$keg/bin/python3.12" ]; then
                PYTHON="$keg/bin/python3.12"
            else
                PYTHON="python3.12"
            fi
        else
            install_standalone_python
            PYTHON="${STANDALONE_DIR}/bin/python3"
        fi
    fi
fi

dim "Python: $(run_install_command "$PYTHON" --version 2>&1)"

# Whatever interpreter we ended up with — found on PATH or auto-installed — must
# be arm64. MLX has no x86_64 build, so a Rosetta Python cannot run Rapid-MLX at
# all; fail here with a clear message instead of building a venv that breaks
# later on the first `import mlx` (#1953).
PY_ARCH=$(run_install_command "$PYTHON" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
if [ "$PY_ARCH" != "arm64" ]; then
    err "Selected Python is ${PY_ARCH}, but Apple Silicon needs a native arm64 Python."
    dim "Interpreter: $PYTHON"
    dim "Install an arm64 Python 3.12 or 3.13 (e.g. 'brew install python@3.12') and re-run."
    exit 1
fi

# ── 4. Migrate from old install location ─────────────────────────────────────

OLD_DIR="${HOME}/.vllm-mlx"
if [ -d "$OLD_DIR" ] && [ ! -d "$INSTALL_DIR" ]; then
    info "Migrating from $OLD_DIR to $INSTALL_DIR ..."
    mv "$OLD_DIR" "$INSTALL_DIR"
fi

# ── 5. Create or update venv + install ───────────────────────────────────────

echo ""
recover_interrupted_rebuild
# The directory doubles as shared state, so create-vs-upgrade turns on whether a
# COMPATIBLE venv exists (plan_venv_action), not on whether the dir exists.
INSTALL_ACTION="$(plan_venv_action "$INSTALL_DIR")"
dispatch_venv "$INSTALL_ACTION"

# Use uv for resolution + parallel downloads when available — typically 3-10x
# faster than pip on a fresh install. Falls back to the venv's pip.
PIP="$INSTALL_DIR/bin/pip"
INSTALLER=("$PIP" install --prefer-binary)
UPGRADE_INSTALLER=("$PIP" install --upgrade --prefer-binary)
FORCE_INSTALLER=("$PIP" install --force-reinstall --prefer-binary)

if command -v uv >/dev/null 2>&1; then
    UV_PY="$INSTALL_DIR/bin/python"
    INSTALLER=(uv pip install --python "$UV_PY")
    UPGRADE_INSTALLER=(uv pip install --python "$UV_PY" --upgrade)
    FORCE_INSTALLER=(uv pip install --python "$UV_PY" --reinstall)
    dim "Using uv for fast install"
fi

install_package
finish_rebuild
reinstall_preserved_extras
restart_service_hints "$INSTALL_ACTION"

# Query the just-installed CLI's stable JSON surface. Failure is harmless and
# falls back to the RAM baseline; never scrape the human table or cache paths.
refresh_starter_from_installed_cache

# ── 6. Create symlinks ──────────────────────────────────────────────────────

mkdir -p "$BIN_DIR"

# Link the Rapid-MLX CLI entry points.
for cmd in rapid-mlx rapid-mlx-chat rapid-mlx-bench; do
    [ -f "$INSTALL_DIR/bin/$cmd" ] && ln -sf "$INSTALL_DIR/bin/$cmd" "$BIN_DIR/$cmd"
done

# Keep pre-Rapid command aliases available for existing automation.
for cmd in vllm-mlx vllm-mlx-chat vllm-mlx-bench; do
    [ -f "$INSTALL_DIR/bin/$cmd" ] && ln -sf "$INSTALL_DIR/bin/$cmd" "$BIN_DIR/$cmd"
done
[ -f "$INSTALL_DIR/bin/rmlx" ] && ln -sf "$INSTALL_DIR/bin/rmlx" "$BIN_DIR/rmlx"
ln -sf "$INSTALL_DIR/bin/python3" "$BIN_DIR/rapid-mlx-python"

# ── 7. Ensure ~/.local/bin is in PATH ────────────────────────────────────────

NEED_PATH_HINT=false
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC=""
    if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_RC" ] && ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
        echo '' >> "$SHELL_RC"
        echo '# Rapid-MLX' >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    fi
    NEED_PATH_HINT=true
fi

# ── 8. Verify + done ────────────────────────────────────────────────────────

VERSION=$(run_install_command "$INSTALL_DIR/bin/rapid-mlx" --version 2>/dev/null || echo "unknown")

echo ""
print_success_box "Rapid-MLX installed!" "Version: $VERSION" "RAM: ${RAM_GB} GB ($RAM_TIER)"
echo ""
print_quick_start_commands
echo ""
dim "Upgrade:    rapid-mlx upgrade"
dim "Uninstall:  rm -rf ~/.rapid-mlx ~/.rapid-mlx-python ~/.local/bin/rapid-mlx* ~/.local/bin/vllm-mlx*"
echo ""

if [ "$NEED_PATH_HINT" = true ]; then
    warn "Restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
