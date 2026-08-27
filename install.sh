#!/bin/bash
# Rapid-MLX installer — AI inference for Apple Silicon
# Usage: curl -fsSL https://rapidmlx.com/install.sh | bash
#        curl ... | bash -s -- 0.12.15     # specific version
#        curl ... | bash -s latest         # latest from GitHub (pre-release)
set -euo pipefail

TARGET="${1:-stable}"  # stable (PyPI) | latest (GitHub HEAD) | x.y.z (specific version)

INSTALL_DIR="${HOME}/.rapid-mlx"
BIN_DIR="${HOME}/.local/bin"
PYPI_PACKAGE="rapid-mlx"
GITHUB_REPO="https://github.com/raullenchai/Rapid-MLX.git"
MIN_PYTHON_MINOR=10

# ── Helpers ──────────────────────────────────────────────────────────────────

BOLD='\033[1m'  DIM='\033[2m'  GREEN='\033[32m'  YELLOW='\033[33m'  RED='\033[31m'  RESET='\033[0m'

info()  { printf "  ${BOLD}%s${RESET}\n" "$*"; }
ok()    { printf "  ${GREEN}%s${RESET}\n" "$*"; }
warn()  { printf "  ${YELLOW}%s${RESET}\n" "$*"; }
err()   { printf "  ${RED}%s${RESET}\n" "$*" >&2; }
dim()   { printf "  ${DIM}%s${RESET}\n" "$*"; }

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
        curl -fsSL "$1"
    else
        wget -qO- "$1"
    fi
}

# ── Interpreter + venv helpers (also sourced by tests/install) ────────────────

# Echo the first python3.x (>= MIN_PYTHON_MINOR) on PATH that is a NATIVE arm64
# build, else nothing. MLX ships arm64 wheels only: a uv/pyenv-managed x86_64
# interpreter earlier in PATH would build a Rosetta venv that cannot import mlx
# and, historically, left ~/.rapid-mlx half-built with no bin/pip (#1953). The
# arch note goes to stderr so command substitution captures only the name.
select_python() {
    local py ver major minor pyarch
    for py in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$py" >/dev/null 2>&1 || continue
        ver=$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
        major="${ver%%.*}"; minor="${ver#*.}"
        { [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; } || continue
        pyarch=$("$py" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
        if [ "$pyarch" != "arm64" ]; then
            dim "Skipping $py (${pyarch}; Rapid-MLX needs a native arm64 Python)" >&2
            continue
        fi
        printf '%s\n' "$py"
        return 0
    done
    return 1
}

# A venv is reusable only if it is COMPLETE (python + pip present and runnable)
# AND COMPATIBLE (a native arm64 interpreter, Python >= MIN_PYTHON_MINOR) — the
# same contract fresh selection enforces. Anything else is treated as
# not-installed and rebuilt under the newly selected interpreter:
#   - missing bin/pip: an aborted earlier run left it out (the original #1953).
#   - executable-but-broken pip: a ~/.vllm-mlx -> ~/.rapid-mlx migration leaves a
#     stale shebang, so `-x` passes but running fails — hence the `--version` run.
#   - wrong arch / too-old version: a venv built earlier under a Rosetta x86_64
#     or a 3.9 Python still runs, so it would pass a liveness check and quietly
#     survive the upgrade path even though mlx can never import there.
venv_is_complete() {
    local dir="$1" arch ver major minor
    [ -x "$dir/bin/python" ] && [ -x "$dir/bin/pip" ] || return 1
    "$dir/bin/python" -c 'import sys' >/dev/null 2>&1 || return 1
    "$dir/bin/pip" --version >/dev/null 2>&1 || return 1
    arch=$("$dir/bin/python" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
    [ "$arch" = "arm64" ] || return 1
    ver=$("$dir/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
    major="${ver%%.*}"; minor="${ver#*.}"
    [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]
}

# Remove ONLY the virtual environment's own entries under $1, so a rebuild does
# not touch the shared ~/.rapid-mlx state that pip and the desktop app also keep
# here — telemetry-client-id, bench-install-id, agents/ profiles, launch.pid,
# first-run markers. `python -m venv` then recreates bin/lib/... in place,
# alongside that preserved state.
reset_venv_dir() {
    local dir="$1"
    rm -rf "$dir/bin" "$dir/lib" "$dir/lib64" "$dir/include" "$dir/share" "$dir/pyvenv.cfg"
}

# Decide what a fresh install run should do with $1: "create" when nothing is
# there, "upgrade" a compatible venv in place, or "rebuild" an incompatible one
# (reset_venv_dir + recreate). Isolated from the actions it drives so the
# decision is unit-testable.
plan_venv_action() {
    local dir="$1"
    if [ ! -d "$dir" ]; then echo create; return; fi
    if venv_is_complete "$dir"; then echo upgrade; else echo rebuild; fi
}

# Best-effort pip self-upgrade: a transient failure here must not abort install.
upgrade_pip() {
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q 2>/dev/null || true
}

# Build the venv at $INSTALL_DIR under $PYTHON and verify it came out usable.
create_venv() {
    "$PYTHON" -m venv "$INSTALL_DIR"
    if ! venv_is_complete "$INSTALL_DIR"; then
        err "Could not create a working virtual environment at $INSTALL_DIR."
        dim "Selected Python: $("$PYTHON" -c 'import platform, sys; print(platform.machine(), sys.version.split()[0])' 2>/dev/null || echo unknown)"
        dim "Install an arm64 Python 3.10+ (e.g. 'brew install python@3.12') and re-run."
        exit 1
    fi
    upgrade_pip
}

# Carry out a planned action. Kept as its own function (not an inline case) so
# the create/upgrade/rebuild wiring is unit-testable with mocked actions.
dispatch_venv() {
    case "$1" in
        upgrade)
            info "Upgrading Rapid-MLX..."
            upgrade_pip
            ;;
        rebuild)
            warn "Existing $INSTALL_DIR has no usable virtual environment — rebuilding it."
            dim "(shared state such as telemetry-client-id and agents/ is preserved)"
            reset_venv_dir "$INSTALL_DIR"
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
# 4B starter. The candidate order is the curated RAM-tier order walked downward,
# aliases above the current tier never enter the list; the sub-16 GB order is
# just the 1.2B baseline, because Desktop auto-selects nothing else there.
starter_baseline_for_ram() {
    if [ "$1" -ge 16 ]; then
        printf '%s\n' "qwen3.5-4b-4bit"
    else
        printf '%s\n' "lfm2.5-1b-4bit"
    fi
}

starter_cached_order_for_ram() {
    local ram="$1"
    if [ "$ram" -ge 48 ]; then printf '%s\n' qwen3.8-27b-4bit qwen3.6-35b-4bit qwen3.5-4b-4bit bonsai-27b-2bit qwen3.5-9b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 32 ]; then printf '%s\n' qwen3.8-27b-4bit qwen3.5-4b-4bit bonsai-27b-2bit qwen3.5-9b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 24 ]; then printf '%s\n' bonsai-27b-2bit qwen3.5-4b-4bit qwen3.5-9b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 18 ]; then printf '%s\n' qwen3.5-9b-4bit qwen3.5-4b-4bit lfm2.5-2.6b-4bit
    elif [ "$ram" -ge 16 ]; then printf '%s\n' qwen3.5-4b-4bit lfm2.5-2.6b-4bit
    else printf '%s\n' lfm2.5-1b-4bit
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
    "$INSTALL_DIR/bin/rapid-mlx" models --cached --json 2>/dev/null |
        "$INSTALL_DIR/bin/python" -c 'import json,sys; p=json.load(sys.stdin); print("\n".join(m["alias"] for m in p.get("cached", []) if m.get("state") == "ok" and isinstance(m.get("alias"), str)))' 2>/dev/null
}

refresh_starter_from_installed_cache() {
    local cached_aliases
    if ! cached_aliases="$(installed_cached_aliases)"; then
        cached_aliases=""
    fi
    RECOMMENDED_MODEL="$(select_starter_model "$RAM_GB" "$cached_aliases")"
}

print_quick_start_commands() {
    info "Quick start:"
    echo ""
    echo "    rapid-mlx serve ${RECOMMENDED_MODEL}${RECOMMENDED_FLAGS}"
    echo ""
    dim "Then open a second terminal:"
    echo ""
    echo "    rapid-mlx chat ${RECOMMENDED_MODEL} --port 8000    # built-in chat (terminal)"
    echo "    rapid-mlx-chat                                    # web chat UI (first: ${INSTALL_DIR}/bin/pip install 'rapid-mlx[chat]')"
    echo "    ANTHROPIC_BASE_URL=http://localhost:8000 claude    # Claude Code (or: rapid-mlx launch claude-code)"
    echo "    OPENAI_API_BASE=http://localhost:8000/v1 aider     # Aider"
}

# When sourced by the test harness we only want the definitions above, not the
# installer itself. `return` succeeds only in a sourced context; the `|| exit 0`
# keeps an accidental `RAPID_INSTALL_LIB=1 bash install.sh` from erroring out.
if [ "${RAPID_INSTALL_LIB:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

# Validate target
if [[ "$TARGET" != "stable" ]] && [[ "$TARGET" != "latest" ]] && [[ ! "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    echo "Usage: install.sh [stable|latest|VERSION]" >&2
    echo "  stable   Install from PyPI (default)" >&2
    echo "  latest   Install from GitHub HEAD" >&2
    echo "  x.y.z    Install specific version from PyPI" >&2
    exit 1
fi

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
if [ "$ARCH" != "arm64" ]; then
    err "Rapid-MLX requires Apple Silicon (M1/M2/M3/M4)."
    dim "Detected: $ARCH"
    exit 1
fi

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

# ── 3. Find or install Python 3.10+ ─────────────────────────────────────────

PYTHON="$(select_python || true)"

if [ -z "$PYTHON" ]; then
    echo ""
    warn "Python 3.10+ not found. Installing automatically..."
    if command -v brew >/dev/null 2>&1; then
        info "Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        # Resolve the formula's OWN keg path, not the bare name and not the
        # Homebrew root: python@3.12 is keg-only, so `$(brew --prefix)/bin` may
        # not carry python3.12, while a bare `python3.12` could still hit the
        # x86_64 build earlier on PATH (the reason we got here) and fail the
        # arch check below. The lookup is best-effort (|| true) so it can't abort
        # the install under set -e; fall back to the name if the keg path is gone.
        keg="$(brew --prefix python@3.12 2>/dev/null || true)"
        if [ -n "$keg" ] && [ -x "$keg/bin/python3.12" ]; then
            PYTHON="$keg/bin/python3.12"
        else
            PYTHON="python3.12"
        fi
    else
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
        info "Downloading Python ${PY_VERSION} (build ${PY_BUILD}) + verifying SHA256..."
        mkdir -p "$STANDALONE_DIR"
        # Verify the tarball against the publisher-published SHA256SUMS for this
        # exact release BEFORE anything is extracted. A curl|bash install trusts
        # the transport (HTTPS) for integrity; verifying the checksum removes the
        # "trust whatever bytes came back" gap so a tampered / mis-served
        # download fails closed instead of silently installing.
        TMP_TAR="$(mktemp "${TMPDIR:-/tmp}/rapidmlx-py.XXXXXX.tar.gz")"
        trap 'rm -f "$TMP_TAR"' EXIT
        download "$PY_URL" > "$TMP_TAR"
        ACTUAL="$(shasum -a 256 "$TMP_TAR" | awk '{print $1}')"
        if [ "$ACTUAL" != "$PY_SHA256" ]; then
            err "SHA256 mismatch for Python standalone tarball."
            dim "Expected $PY_SHA256"
            dim "Got      $ACTUAL"
            dim "Refusing to extract an unverified runtime."
            rm -f "$TMP_TAR"
            exit 1
        fi
        dim "SHA256 verified."
        tar xzf "$TMP_TAR" -C "$STANDALONE_DIR" --strip-components=1
        rm -f "$TMP_TAR"
        trap - EXIT
        PYTHON="${STANDALONE_DIR}/bin/python3"
        if ! "$PYTHON" --version >/dev/null 2>&1; then
            err "Failed to install standalone Python."
            dim "Please install Python 3.10+ from https://www.python.org/downloads/"
            exit 1
        fi
        ok "Installed Python $("$PYTHON" --version 2>&1)"
    fi
fi

dim "Python: $("$PYTHON" --version 2>&1)"

# Whatever interpreter we ended up with — found on PATH or auto-installed — must
# be arm64. MLX has no x86_64 build, so a Rosetta Python cannot run Rapid-MLX at
# all; fail here with a clear message instead of building a venv that breaks
# later on the first `import mlx` (#1953).
PY_ARCH=$("$PYTHON" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")
if [ "$PY_ARCH" != "arm64" ]; then
    err "Selected Python is ${PY_ARCH}, but Apple Silicon needs a native arm64 Python."
    dim "Interpreter: $PYTHON"
    dim "Install an arm64 Python 3.10+ (e.g. 'brew install python@3.12') and re-run."
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
# The directory doubles as shared state, so create-vs-upgrade turns on whether a
# COMPATIBLE venv exists (plan_venv_action), not on whether the dir exists.
dispatch_venv "$(plan_venv_action "$INSTALL_DIR")"

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

case "$TARGET" in
    stable)
        "${UPGRADE_INSTALLER[@]}" "$PYPI_PACKAGE" -q 2>/dev/null \
            || { dim "PyPI unavailable, installing from GitHub..."; "${INSTALLER[@]}" "$PYPI_PACKAGE @ git+${GITHUB_REPO}" ; }
        ;;
    latest)
        info "Installing latest from GitHub..."
        "${FORCE_INSTALLER[@]}" "$PYPI_PACKAGE @ git+${GITHUB_REPO}"
        ;;
    *)
        info "Installing version ${TARGET}..."
        "${INSTALLER[@]}" "${PYPI_PACKAGE}==${TARGET}" -q 2>/dev/null \
            || { dim "Version ${TARGET} not on PyPI, trying GitHub tag..."; "${INSTALLER[@]}" "$PYPI_PACKAGE @ git+${GITHUB_REPO}@v${TARGET}" ; }
        ;;
esac

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

VERSION=$("$INSTALL_DIR/bin/rapid-mlx" --version 2>/dev/null || echo "unknown")

echo ""
echo "  ╭─────────────────────────────────────╮"
printf "  │  ${GREEN}Rapid-MLX installed!${RESET}                │\n"
printf "  │  Version: %-25s│\n" "$VERSION"
printf "  │  RAM: %-29s│\n" "${RAM_GB} GB ($RAM_TIER)"
echo "  ╰─────────────────────────────────────╯"
echo ""
print_quick_start_commands
echo ""
dim "Upgrade:    curl -fsSL https://rapidmlx.com/install.sh | bash"
dim "Uninstall:  rm -rf ~/.rapid-mlx ~/.rapid-mlx-python ~/.local/bin/rapid-mlx* ~/.local/bin/vllm-mlx*"
echo ""

if [ "$NEED_PATH_HINT" = true ]; then
    warn "Restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
