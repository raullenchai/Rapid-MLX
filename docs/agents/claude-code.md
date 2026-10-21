# Claude Code

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash` and restart the terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install Claude Code. Run `rapid-mlx launch claude-code`. The same command works after a restart; `rapid-mlx launch claude-code --remove` deletes only Rapid's private profile. Existing Claude Code settings are left untouched. Use `--model qwen3.5-9b-4bit` to choose that model on a Mac with enough RAM; the default recommendation follows the Mac's memory tier. The first reply may wait for a download, model load, or a long agent prompt.

## Next steps

Run `rapid-mlx recipe` to see model recommendations for this Mac. To use a specific installed alias, run `rapid-mlx launch claude-code --model <alias>`. The command saves a private profile and starts a local server if needed. Run `rapid-mlx launch claude-code --remove` to remove that profile.

If the first reply takes time, allow the model download and load to finish. Run `rapid-mlx doctor` to check the installation. See the [current Claude Code guide](https://rapidmlx.com/docs/agents/claude-code) for setup details and troubleshooting.
