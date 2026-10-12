# Goose with Rapid-MLX

Goose 1.54.0 loads custom OpenAI providers from `~/.config/goose/custom_providers/` on macOS. Its [provider schema](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose-providers/src/declarative.rs) accepts a full `/v1/chat/completions` URL, model name, context limit, streaming flag, and an optional API key environment variable.

Start a local model, then configure Goose in another terminal:

```sh
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents goose --setup
goose run --provider custom_rapid_mlx --model qwen3.5-4b-4bit --no-session --text "Inspect this directory and summarize it."
```

`--setup` writes `~/.config/goose/custom_providers/custom_rapid_mlx.json`. It gets the model and context limit from the server. Pass `--base-url http://127.0.0.1:PORT/v1` for another port, `--model` to set the alias, or `--dry-run` to preview. Goose and setup both honor `XDG_CONFIG_HOME`; if you set `GOOSE_PATH_ROOT`, they use its `config/custom_providers/` subdirectory instead. Goose's [config path code](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose/src/config/paths.rs) defines that override.

For a server started with `--api-key`, export `RAPID_MLX_API_KEY` before setup and Goose. Setup writes the environment variable name into the provider JSON; it never stores the key value. `rapid-mlx agents goose --test` runs the integration checks, including the headless Goose command when its binary is on `PATH`.
