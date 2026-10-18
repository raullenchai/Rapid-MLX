# Goose with Rapid-MLX

Goose 1.54.0 loads custom OpenAI providers from `~/.config/goose/custom_providers/` on macOS. Its [provider schema](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose-providers/src/declarative.rs) accepts a full `/v1/chat/completions` URL, model name, context limit, streaming flag, and an optional API key environment variable.

Start a local model, then configure Goose in another terminal:

```sh
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents goose --setup
goose run --provider custom_rapid_mlx --model qwen3.5-4b-4bit --no-session --text "Inspect this directory and summarize it."
```

`--setup` writes `~/.config/goose/custom_providers/custom_rapid_mlx.json`. It gets the model and context limit from the server. Pass `--base-url http://127.0.0.1:PORT/v1` for another port, `--model` to set the alias, or `--dry-run` to preview. Goose and setup both honor `XDG_CONFIG_HOME`; if you set `GOOSE_PATH_ROOT`, they use its `config/custom_providers/` subdirectory instead. Goose's [config path code](https://github.com/aaif-goose/goose/blob/v1.54.0/crates/goose/src/config/paths.rs) defines that override.

## `rapid-mlx chat` extension defaults

`rapid-mlx chat` creates an isolated Goose workspace for each working directory. New workspaces enable Developer, Skills, Extension Manager, and the local `web_search`/`fetch_page` extension. Analyze, Apps, Summon, Top of Mind, Memory, Computer Controller, Auto Visualiser, and Tutorial start disabled. Goose stores these choices in its own `config.yaml`; later changes to that file are preserved.

To change the built-in extensions for the current working directory, run Goose's configuration UI against that same workspace:

```sh
GOOSE_PATH_ROOT="$HOME/.cache/rapid-mlx/goose/1.54.0/mini-workspaces/$(python3 -c 'import hashlib,os; print(hashlib.sha256(os.getcwd().encode()).hexdigest())')" \
  "$HOME/.cache/rapid-mlx/goose/1.54.0/goose" configure
```

Select **Toggle Extensions** to enable Computer Controller, Memory, or another extension for future sessions. Search uses Keenable without a key; `PARALLEL_API_KEY`, `TAVILY_API_KEY`, or `BRAVE_API_KEY` in the launching shell selects a keyed backend in that order of priority. Keys go only to the local gate process and are never added to Goose's tool process environment.

Desktop Chat seeds the same Goose extension defaults in `~/Library/Application Support/Rapid/agent-goose-chat/workspaces/<folder SHA-256>/config/config.yaml`. You can enable a disabled extension by changing its `enabled` value to `true` in that file, then starting a new Chat helper. Desktop reads a selected Parallel, Tavily, or Brave key from Rapid's existing Keychain namespace without requesting Keychain UI access; with no selected key, search uses Keenable.

For a server started with `--api-key`, export `RAPID_MLX_API_KEY` before setup and Goose. Setup writes the environment variable name into the provider JSON; it never stores the key value. `rapid-mlx agents goose --test` runs the integration checks, including the headless Goose command when its binary is on `PATH`.
