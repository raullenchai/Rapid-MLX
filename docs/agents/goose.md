# Goose with Rapid-MLX

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash`, restart Terminal, and install Goose. Then run:

```bash
rapid-mlx launch goose
```

The command saves a private Rapid profile, starts a local server if needed, and launches Goose. Run it again next time. Use `rapid-mlx launch goose --remove` to remove the private profile.

Run `rapid-mlx recipe` for model recommendations or `rapid-mlx launch goose --model <alias>` to select a model. The first reply may wait for a download or model load. See the [current agent guide](https://rapidmlx.com/docs/agents/goose) for more detail.

## Manual provider setup

Goose 1.54.0 loads custom OpenAI providers from `~/.config/goose/custom_providers/` on macOS. Its [provider schema](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose-providers/src/declarative.rs) accepts a full `/v1/chat/completions` URL, model name, context limit, streaming flag, and an optional API key environment variable.

Start a local model, then configure Goose in another terminal:

```sh
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents goose --setup
goose run --provider custom_rapid_mlx --model qwen3.5-4b-4bit --no-session --text "Inspect this directory and summarize it."
```

`--setup` writes `~/.config/goose/custom_providers/custom_rapid_mlx.json`. It gets the model and context limit from the server. Pass `--base-url http://127.0.0.1:PORT/v1` for another port, `--model` to set the alias, or `--dry-run` to preview. Goose and setup both honor `XDG_CONFIG_HOME`; if you set `GOOSE_PATH_ROOT`, they use its `config/custom_providers/` subdirectory instead. Goose's [config path code](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose/src/config/paths.rs) defines that override.

For a server started with `--api-key`, export `RAPID_MLX_API_KEY` before setup and Goose. Setup writes the environment variable name into the provider JSON; it never stores the key value. `rapid-mlx agents goose --test` runs the integration checks, including the headless Goose command when its binary is on `PATH`.
