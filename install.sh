#!/bin/bash
# Rapid-MLX installer — AI inference for Apple Silicon
# Usage: curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash
set -euo pipefail

INSTALL_DIR="${HOME}/.rapid-mlx"
BIN_DIR="${HOME}/.local/bin"
PYPI_PACKAGE="rapid-mlx"
GITHUB_REPO="https://github.com/raullenchai/Rapid-MLX.git"
MIN_PYTHON_MINOR=10

echo ""
echo "  Rapid-MLX installer"
echo "  ==================="
echo ""

# 1. Check Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Error: Rapid-MLX requires Apple Silicon (M1/M2/M3/M4)."
    echo "Detected architecture: $ARCH"
    exit 1
fi

# 2. Check macOS version (13+ for MLX)
MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
if [ "$MACOS_VERSION" -lt 13 ]; then
    echo "Error: Rapid-MLX requires macOS 13 (Ventura) or later."
    echo "Detected: macOS $(sw_vers -productVersion)"
    exit 1
fi
echo "  macOS $(sw_vers -productVersion) on $ARCH"

# 3. Find Python 3.10+
PYTHON=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        minor=$("$py" -c "import sys; print(sys.version_info[1])" 2>/dev/null || echo "0")
        major=$("$py" -c "import sys; print(sys.version_info[0])" 2>/dev/null || echo "0")
        if [ "$major" -ge 3 ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "  Python 3.10+ not found. Installing automatically..."
    if command -v brew &>/dev/null; then
        echo "  Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        PYTHON="python3.12"
    else
        # Download standalone Python — no Homebrew or sudo needed
        STANDALONE_DIR="${HOME}/.rapid-mlx-python"
        PY_VERSION="3.12.11"
        PY_BUILD="20250529"
        PY_URL="https://github.com/indygreg/python-build-standalone/releases/download/${PY_BUILD}/cpython-${PY_VERSION}+${PY_BUILD}-aarch64-apple-darwin-install_only.tar.gz"
        echo "  Downloading Python ${PY_VERSION} (standalone, no sudo needed)..."
        mkdir -p "$STANDALONE_DIR"
        curl -fsSL "$PY_URL" | tar xz -C "$STANDALONE_DIR" --strip-components=1
        PYTHON="${STANDALONE_DIR}/bin/python3"
        if ! "$PYTHON" --version &>/dev/null; then
            echo "  Error: Failed to install standalone Python."
            echo "  Please install Python 3.10+ from https://www.python.org/downloads/"
            exit 1
        fi
        echo "  Installed Python $("$PYTHON" --version 2>&1) to $STANDALONE_DIR"
    fi
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
echo "  Python: $PY_VERSION ($PYTHON)"

# 4. Migrate from old install location if needed
OLD_DIR="${HOME}/.vllm-mlx"
if [ -d "$OLD_DIR" ] && [ ! -d "$INSTALL_DIR" ]; then
    echo ""
    echo "  Migrating from $OLD_DIR to $INSTALL_DIR ..."
    mv "$OLD_DIR" "$INSTALL_DIR"
fi

# 5. Create or update venv
if [ -d "$INSTALL_DIR" ]; then
    echo ""
    echo "  Existing installation found at $INSTALL_DIR"
    echo "  Upgrading..."
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q
    "$INSTALL_DIR/bin/pip" install --force-reinstall --no-cache-dir --no-deps "$PYPI_PACKAGE @ git+${GITHUB_REPO}"
else
    echo ""
    echo "  Installing to $INSTALL_DIR ..."
    echo "  (This takes about a minute)"
    echo ""
    "$PYTHON" -m venv "$INSTALL_DIR"
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q
    "$INSTALL_DIR/bin/pip" install "$PYPI_PACKAGE @ git+${GITHUB_REPO}"
fi

# 6. Create symlinks (both rapid-mlx and vllm-mlx names)
mkdir -p "$BIN_DIR"
for cmd in vllm-mlx vllm-mlx-chat vllm-mlx-bench; do
    if [ -f "$INSTALL_DIR/bin/$cmd" ]; then
        ln -sf "$INSTALL_DIR/bin/$cmd" "$BIN_DIR/$cmd"
    fi
done

# Create rapid-mlx aliases pointing to vllm-mlx commands
if [ -f "$INSTALL_DIR/bin/vllm-mlx" ]; then
    ln -sf "$INSTALL_DIR/bin/vllm-mlx" "$BIN_DIR/rapid-mlx"
fi
if [ -f "$INSTALL_DIR/bin/vllm-mlx-chat" ]; then
    ln -sf "$INSTALL_DIR/bin/vllm-mlx-chat" "$BIN_DIR/rapid-mlx-chat"
fi
if [ -f "$INSTALL_DIR/bin/vllm-mlx-bench" ]; then
    ln -sf "$INSTALL_DIR/bin/vllm-mlx-bench" "$BIN_DIR/rapid-mlx-bench"
fi

# Also symlink python so rapid-mlx serve can find dependencies
ln -sf "$INSTALL_DIR/bin/python3" "$BIN_DIR/rapid-mlx-python"

# 7. Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC=""
    if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo '' >> "$SHELL_RC"
            echo '# Rapid-MLX' >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo "  Added ~/.local/bin to PATH in $SHELL_RC"
        fi
    fi
fi

# 8. Verify installation
if ! "$INSTALL_DIR/bin/vllm-mlx" --help &>/dev/null; then
    echo ""
    echo "  Warning: Rapid-MLX installed but CLI verification failed."
    echo "  Try running: $INSTALL_DIR/bin/vllm-mlx --help"
    echo ""
fi

echo ""
echo "  Rapid-MLX installed successfully!"
echo ""
echo "  Quick start:"
echo "    rapid-mlx serve qwen3.5-9b"
echo ""
echo "  Then use with any OpenAI-compatible app:"
echo "    OPENAI_BASE_URL=http://localhost:8000/v1 claude"
echo ""
echo "  To upgrade later:"
echo "    curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash"
echo ""
echo "  To uninstall:"
echo "    rm -rf $INSTALL_DIR && rm -f $BIN_DIR/rapid-mlx $BIN_DIR/rapid-mlx-chat $BIN_DIR/rapid-mlx-bench $BIN_DIR/vllm-mlx $BIN_DIR/vllm-mlx-chat $BIN_DIR/vllm-mlx-bench $BIN_DIR/rapid-mlx-python"
echo ""
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "  NOTE: Restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
