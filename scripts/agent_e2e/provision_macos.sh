#!/bin/bash
# Provision a dedicated test HOME on a leased macOS worker. Capture logs only
# under the second argument; this never reads or writes the operator's HOME.
set -u
umask 077

if [ "$#" -ne 2 ]; then
  echo "usage: $0 EMPTY_TEST_HOME PRIVATE_RUN_DIR" >&2
  exit 2
fi
test_home="$1"
run_dir="$2"
script_dir="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$run_dir"
chmod 700 "$run_dir"
python3 "$script_dir/seed_home.py" "$test_home" || exit 1
export HOME="$test_home"
export UV_TOOL_DIR="$HOME/.local/share/uv/tools"
export UV_TOOL_BIN_DIR="$HOME/.local/bin"
export npm_config_prefix="$HOME/.local"
export PATH="$HOME/.local/bin:$HOME/.rapid-mlx/bin:$PATH"
mkdir -p "$HOME/.local/bin"

install() {
  name="$1"
  shift
  echo "Installing $name" >&2
  "$@" >"$run_dir/install-$name.log" 2>&1
  result=$?
  printf '%s %s\n' "$name" "$result" >>"$run_dir/install-status.txt"
  return 0
}

# Capture the official installer first, then execute it in the test HOME.
curl -fsSL https://rapidmlx.com/install.sh -o "$run_dir/rapid-install.sh" || exit 1
install rapid-release bash "$run_dir/rapid-install.sh"
if [ -x "$HOME/.local/bin/rapid-mlx" ]; then
  "$HOME/.local/bin/rapid-mlx" --version >"$run_dir/rapid-release-version.txt" 2>&1
fi
install rapid-main uv tool install --force 'git+https://github.com/raullenchai/Rapid-MLX.git@main'
install model "$HOME/.local/bin/rapid-mlx" pull qwen3.5-9b-4bit

# npm's global prefix is inside the dedicated HOME. Each install is logged
# separately so one failed upstream package does not hide eight results.
install codex npm install -g @openai/codex@latest
install claude-code npm install -g @anthropic-ai/claude-code@latest
install pi npm install -g @earendil-works/pi-coding-agent@latest
install opencode npm install -g @opencode/cli@2
install qwen-code npm install -g @qwen-code/qwen-code@latest
install crush npm install -g @charmland/crush@latest
install aider uv tool install --force --python python3.12 --with pip aider-chat@latest

curl -fsSL https://hermes-agent.nousresearch.com/install.sh -o "$run_dir/hermes-install.sh"
if [ "$?" -eq 0 ]; then
  install hermes bash "$run_dir/hermes-install.sh"
fi
curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh -o "$run_dir/goose-install.sh"
if [ "$?" -eq 0 ]; then
  install goose bash "$run_dir/goose-install.sh"
fi

for binary in codex claude pi opencode hermes goose qwen crush aider rapid-mlx; do
  if command -v "$binary" >/dev/null 2>&1; then
    "$binary" --version >>"$run_dir/versions.txt" 2>&1
  else
    printf '%s NOT INSTALLED\n' "$binary" >>"$run_dir/versions.txt"
  fi
done
echo "Provisioning logs: $run_dir" >&2
