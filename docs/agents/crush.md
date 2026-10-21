# Crush with Rapid-MLX

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash` and restart the terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install Crush. Run `rapid-mlx launch crush`. The same command works after a restart; `rapid-mlx launch crush --remove` deletes only Rapid's private profile. Existing Crush settings are left untouched. Use `--model qwen3.5-9b-4bit` to choose that model on a Mac with enough RAM; the default recommendation follows the Mac's memory tier. The first reply may wait for a download, model load, or a long agent prompt. The manual setup steps below are for users who want to edit their regular agent configuration.


Crush 0.98.1 uses an `openai-compat` provider with the Rapid-MLX `/v1` base URL. Its [current config guide](https://github.com/charmbracelet/crush/blob/v0.98.1/docs/config/README.md) prefers `crushrc` and still supports the JSON file that this setup profile writes.

Start a local model, then configure and run Crush in another terminal:

```sh
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents crush --setup
crush run --model rapid-mlx/qwen3.5-4b-4bit "Inspect this directory and summarize it."
```

`--setup` writes `~/.config/crush/crush.json`, or `$XDG_CONFIG_HOME/crush/crush.json` when that variable is set. `CRUSH_GLOBAL_CONFIG` takes precedence and places it at `$CRUSH_GLOBAL_CONFIG/crush.json`. It records the model, context window, and large-model selection. Use `--base-url http://127.0.0.1:PORT/v1` for another port, `--model` to set the alias, or `--dry-run` to preview. A project `.crushrc` or global `crushrc` can override the JSON model choice, so check `crush models` if the active model differs.

The profile sets `options.request_timeout` to 300 seconds. Crush's default two-minute inactivity timeout can expire while a small Mac prefills its large agent prompt.

For a server started with `--api-key`, export `RAPID_MLX_API_KEY` before setup and Crush. Setup writes a `$RAPID_MLX_API_KEY` reference rather than the key value. `rapid-mlx agents crush --test` runs the integration checks, including the headless Crush command when its binary is on `PATH`.
