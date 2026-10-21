# OpenCode

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash` and restart the terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install OpenCode. Run `rapid-mlx launch opencode`. The same command works after a restart; `rapid-mlx launch opencode --remove` deletes only Rapid's private profile. Existing OpenCode settings are left untouched. Use `--model qwen3.5-9b-4bit` to choose that model on a Mac with enough RAM; the default recommendation follows the Mac's memory tier. The first reply may wait for a download, model load, or a long agent prompt.

## Next steps

Run `rapid-mlx recipe` to see model recommendations for this Mac. To use a specific installed alias, run `rapid-mlx launch opencode --model <alias>`. The command saves a private profile and starts a local server if needed. Run `rapid-mlx launch opencode --remove` to remove that profile.

If the first reply takes time, allow the model download and load to finish. Run `rapid-mlx doctor` to check the installation. See the [current OpenCode guide](https://rapidmlx.com/docs/agents/opencode) for setup details and troubleshooting.
