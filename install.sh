#!/bin/bash
# Rapid-MLX installer — AI inference for Apple Silicon
# Usage: curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash
#        curl ... | bash -s 0.4.3          # specific version
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
echo "  │  2-4x faster than Ollama            │"
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

RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1073741824}')
if   [ "$RAM_GB" -ge 96 ]; then RECOMMENDED_MODEL="qwen3.5-122b"; RAM_TIER="96+ GB"
elif [ "$RAM_GB" -ge 48 ]; then RECOMMENDED_MODEL="qwen3.5-35b";  RAM_TIER="48-95 GB"
elif [ "$RAM_GB" -ge 24 ]; then RECOMMENDED_MODEL="qwen3.5-9b";   RAM_TIER="24-47 GB"
else                            RECOMMENDED_MODEL="qwen3.5-4b";   RAM_TIER="8-23 GB"
fi

dim "macOS $(sw_vers -productVersion) · Apple Silicon · ${RAM_GB} GB RAM"

# ── 3. Find or install Python 3.10+ ─────────────────────────────────────────

PYTHON=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1; then
        ver=$("$py" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || echo "0.0")
        major="${ver%%.*}"; minor="${ver#*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    warn "Python 3.10+ not found. Installing automatically..."
    if command -v brew >/dev/null 2>&1; then
        info "Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        PYTHON="python3.12"
    else
        STANDALONE_DIR="${HOME}/.rapid-mlx-python"
        PY_VERSION="3.12.13"
        # Fetch latest build tag dynamically
        PY_BUILD=$(download "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" \
            | grep -o '"tag_name":"[^"]*"' | head -1 | cut -d'"' -f4 2>/dev/null || echo "20260408")
        PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_BUILD}/cpython-${PY_VERSION}+${PY_BUILD}-aarch64-apple-darwin-install_only.tar.gz"
        info "Downloading Python ${PY_VERSION}..."
        mkdir -p "$STANDALONE_DIR"
        download "$PY_URL" | tar xz -C "$STANDALONE_DIR" --strip-components=1
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

# ── 4. Migrate from old install location ─────────────────────────────────────

OLD_DIR="${HOME}/.vllm-mlx"
if [ -d "$OLD_DIR" ] && [ ! -d "$INSTALL_DIR" ]; then
    info "Migrating from $OLD_DIR to $INSTALL_DIR ..."
    mv "$OLD_DIR" "$INSTALL_DIR"
fi

# ── 5. Create or update venv + install ───────────────────────────────────────

echo ""
if [ -d "$INSTALL_DIR" ]; then
    info "Upgrading Rapid-MLX..."
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q 2>/dev/null
else
    info "Installing Rapid-MLX..."
    dim "(this takes about a minute)"
    "$PYTHON" -m venv "$INSTALL_DIR"
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q 2>/dev/null
fi

PIP="$INSTALL_DIR/bin/pip"
case "$TARGET" in
    stable)
        "$PIP" install --upgrade "$PYPI_PACKAGE" -q 2>/dev/null \
            || { dim "PyPI unavailable, installing from GitHub..."; "$PIP" install "$PYPI_PACKAGE @ git+${GITHUB_REPO}" ; }
        ;;
    latest)
        info "Installing latest from GitHub..."
        "$PIP" install --force-reinstall --no-cache-dir "$PYPI_PACKAGE @ git+${GITHUB_REPO}"
        ;;
    *)
        info "Installing version ${TARGET}..."
        "$PIP" install "${PYPI_PACKAGE}==${TARGET}" -q 2>/dev/null \
            || { dim "Version ${TARGET} not on PyPI, trying GitHub tag..."; "$PIP" install "$PYPI_PACKAGE @ git+${GITHUB_REPO}@v${TARGET}" ; }
        ;;
esac

# ── 6. Create symlinks ──────────────────────────────────────────────────────

mkdir -p "$BIN_DIR"

# Link all CLI entry points
for cmd in vllm-mlx vllm-mlx-chat vllm-mlx-bench; do
    [ -f "$INSTALL_DIR/bin/$cmd" ] && ln -sf "$INSTALL_DIR/bin/$cmd" "$BIN_DIR/$cmd"
done

# rapid-mlx aliases
[ -f "$INSTALL_DIR/bin/vllm-mlx" ]      && ln -sf "$INSTALL_DIR/bin/vllm-mlx"      "$BIN_DIR/rapid-mlx"
[ -f "$INSTALL_DIR/bin/vllm-mlx-chat" ]  && ln -sf "$INSTALL_DIR/bin/vllm-mlx-chat"  "$BIN_DIR/rapid-mlx-chat"
[ -f "$INSTALL_DIR/bin/vllm-mlx-bench" ] && ln -sf "$INSTALL_DIR/bin/vllm-mlx-bench" "$BIN_DIR/rapid-mlx-bench"
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

VERSION=$("$INSTALL_DIR/bin/vllm-mlx" --version 2>/dev/null || echo "unknown")

echo ""
echo "  ╭─────────────────────────────────────╮"
printf "  │  ${GREEN}Rapid-MLX installed!${RESET}                │\n"
printf "  │  Version: %-25s│\n" "$VERSION"
printf "  │  RAM: %-29s│\n" "${RAM_GB} GB ($RAM_TIER)"
echo "  ╰─────────────────────────────────────╯"
echo ""
info "Quick start:"
echo ""
echo "    rapid-mlx serve $RECOMMENDED_MODEL"
echo ""
dim "Then open a second terminal:"
echo ""
echo "    rapid-mlx-chat                                    # built-in chat"
echo "    OPENAI_BASE_URL=http://localhost:8000/v1 claude    # Claude Code"
echo "    OPENAI_BASE_URL=http://localhost:8000/v1 aider     # Aider"
echo ""
dim "Upgrade:    curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash"
dim "Uninstall:  rm -rf ~/.rapid-mlx ~/.local/bin/rapid-mlx* ~/.local/bin/vllm-mlx*"
echo ""

if [ "$NEED_PATH_HINT" = true ]; then
    warn "Restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
