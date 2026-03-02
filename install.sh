#!/bin/bash
# vllm-mlx installer — production-grade LLM inference for Apple Silicon
# Usage: curl -fsSL https://raw.githubusercontent.com/raullenchai/vllm-mlx/main/install.sh | bash
set -euo pipefail

INSTALL_DIR="${HOME}/.vllm-mlx"
BIN_DIR="${HOME}/.local/bin"
REPO="https://github.com/raullenchai/vllm-mlx.git"
MIN_PYTHON_MINOR=10

echo ""
echo "  vllm-mlx installer"
echo "  ==================="
echo ""

# 1. Check Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Error: vllm-mlx requires Apple Silicon (M1/M2/M3/M4)."
    echo "Detected architecture: $ARCH"
    exit 1
fi

# 2. Check macOS version (13+ for MLX)
MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
if [ "$MACOS_VERSION" -lt 13 ]; then
    echo "Error: vllm-mlx requires macOS 13 (Ventura) or later."
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
    echo "  Python 3.10+ not found."
    if command -v brew &>/dev/null; then
        echo "  Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        PYTHON="python3.12"
    else
        echo "  Please install Python 3.10+ from https://www.python.org/downloads/"
        echo "  Or install Homebrew first: https://brew.sh"
        exit 1
    fi
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
echo "  Python: $PY_VERSION ($PYTHON)"

# 4. Create or update venv
if [ -d "$INSTALL_DIR" ]; then
    echo ""
    echo "  Existing installation found at $INSTALL_DIR"
    echo "  Upgrading..."
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q
    "$INSTALL_DIR/bin/pip" install --force-reinstall --no-cache-dir --no-deps "vllm-mlx @ git+${REPO}"
else
    echo ""
    echo "  Installing to $INSTALL_DIR ..."
    echo "  (This may take a few minutes — MLX and dependencies need to compile)"
    echo ""
    "$PYTHON" -m venv "$INSTALL_DIR"
    "$INSTALL_DIR/bin/pip" install --upgrade pip -q
    "$INSTALL_DIR/bin/pip" install "vllm-mlx @ git+${REPO}"
fi

# 5. Create symlinks
mkdir -p "$BIN_DIR"
for cmd in vllm-mlx vllm-mlx-chat vllm-mlx-bench; do
    if [ -f "$INSTALL_DIR/bin/$cmd" ]; then
        ln -sf "$INSTALL_DIR/bin/$cmd" "$BIN_DIR/$cmd"
    fi
done

# Also symlink python so `vllm-mlx serve` can find dependencies
ln -sf "$INSTALL_DIR/bin/python3" "$BIN_DIR/vllm-mlx-python"

# 6. Ensure ~/.local/bin is in PATH
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
            echo '# vllm-mlx' >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo "  Added ~/.local/bin to PATH in $SHELL_RC"
        fi
    fi
fi

# 7. Verify installation
if ! "$INSTALL_DIR/bin/vllm-mlx" --help &>/dev/null; then
    echo ""
    echo "  Warning: vllm-mlx installed but CLI verification failed."
    echo "  Try running: $INSTALL_DIR/bin/vllm-mlx --help"
    echo ""
fi

echo ""
echo "  vllm-mlx installed successfully!"
echo ""
echo "  Quick start:"
echo "    vllm-mlx serve --model lmstudio-community/Qwen3-Coder-Next-MLX-4bit \\"
echo "      --tool-call-parser hermes --port 8000"
echo ""
echo "  Or run directly:"
echo "    $INSTALL_DIR/bin/vllm-mlx serve --model <model-name>"
echo ""
echo "  To upgrade later:"
echo "    curl -fsSL https://raw.githubusercontent.com/raullenchai/vllm-mlx/main/install.sh | bash"
echo ""
echo "  To uninstall:"
echo "    rm -rf $INSTALL_DIR && rm -f $BIN_DIR/vllm-mlx $BIN_DIR/vllm-mlx-chat $BIN_DIR/vllm-mlx-bench $BIN_DIR/vllm-mlx-python"
echo ""
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "  NOTE: Restart your terminal or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
