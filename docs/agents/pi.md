# Pi Coding Agent

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash` and restart the terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install Pi. Run `rapid-mlx launch pi`. The same command works after a restart; `rapid-mlx launch pi --remove` deletes only Rapid's private profile. Existing Pi settings are left untouched. Use `--model qwen3.5-9b-4bit` to choose that model on a Mac with enough RAM; the default recommendation follows the Mac's memory tier. The first reply may wait for a download, model load, or a long agent prompt. The manual setup steps below are for users who want to edit their regular agent configuration.


[Pi 1.1.0](https://github.com/earendil-works/pi/releases/tag/v1.1.0)
uses an OpenAI-compatible custom provider in `~/.pi/agent/models.json`.
`PI_CODING_AGENT_DIR` can move this config directory. Pi sends tool calls
through `POST /v1/chat/completions`.

## Setup

```bash
npm install -g @earendil-works/pi-coding-agent@1.1.0
rapid-mlx serve qwen3.5-4b-4bit --port 8000

# In another terminal:
rapid-mlx agents pi --setup --yes
pi --print --no-session --provider rapid-mlx --model qwen3.5-4b-4bit \
  "Use the read tool to read README.md and report its title."
```

Setup reads the live model ID and context window, previews the change, keeps
other providers and models, and writes a backup when replacing an existing
file. For a keyed server, export `RAPID_MLX_API_KEY` before setup and Pi;
the generated `apiKey` is `$RAPID_MLX_API_KEY`, which Pi expands at runtime.

The generated entry uses `api: "openai-completions"`, the server URL ending
in `/v1`, the served model ID, its actual `contextWindow`, and
`"reasoning": false`. Explicit `--provider` and `--model` flags keep this
one-shot task on the local endpoint when Pi has other credentials.

`rapid-mlx agents pi --test` runs the integration checks. See the
[agent support matrix](matrix.md); Pi is currently outside the older
12-agent family matrix, so no family result is claimed there.
