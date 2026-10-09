# OpenCode

[OpenCode](https://github.com/anomalyco/opencode) can use a local rapid-mlx
model through its OpenAI-compatible chat completions endpoint. OpenCode 1.x
and 2.x use different custom-provider config keys; `rapid-mlx agents
opencode --setup` writes both into `~/.config/opencode/opencode.json`.

## Setup

```bash
# OpenCode 2.x (current package); for 1.x use: npm install -g opencode-ai@1
npm install -g @opencode/cli@2

rapid-mlx serve qwen3.5-4b-4bit --port 8000
# In another terminal:
rapid-mlx agents opencode --setup --yes
opencode models
opencode run "Use the shell tool to run pwd and report the directory."
```

OpenCode 2.x uses `opencode serve --service` as a persistent background
service. Its TUI, desktop, web, and `opencode run` clients may connect to an
already-running service. OpenCode 2.0.26 can return an empty model list on
the first `opencode models` call after service startup; repeat the command.
If the list remains stale after setup, run `opencode service restart`, then
repeat `opencode models`.
`rapid-mlx agents opencode --test` runs 2.x with `--standalone --format json`,
which starts a private server for each headless test, exposes tool results to
the test runner, and closes it afterward. The 1.x test
path continues to use `opencode run`.

## Config file

The global file is `~/.config/opencode/opencode.json`; a project can also
provide `opencode.json`. The generated file has these two provider entries
for the same model. Setup substitutes the actual model ID and context limit
reported by the running rapid-mlx server.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "rapid-mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Rapid-MLX",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "not-needed"
      },
      "models": {"qwen3.5-4b-4bit": {}}
    }
  },
  "providers": {
    "rapid-mlx": {
      "package": "@opencode/ai/providers/openai-compatible",
      "name": "Rapid-MLX",
      "settings": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen3.5-4b-4bit": {
          "name": "qwen3.5-4b-4bit",
          "limit": {"context": 32768, "output": 8192},
          "capabilities": {"tools": true, "input": ["text"], "output": ["text"]}
        }
      }
    }
  },
  "model": "rapid-mlx/qwen3.5-4b-4bit"
}
```

The singular `provider` entry is for 1.x. The plural `providers` entry is
the native 2.x form. Both versions can read this combined file. If your
rapid-mlx server requires an API key, setup uses `{env:RAPID_MLX_API_KEY}`
instead of `not-needed`; export that variable before launching OpenCode.

## Troubleshooting

- If the first `opencode models` call omits the model, repeat it. If the list
  remains stale after setup, restart the 2.x service with
  `opencode service restart`. Check the loaded config with
  `opencode debug config`.
- If the model is listed but requests fail, confirm rapid-mlx is running and
  `baseURL` ends in `/v1`.
- For one 2.x task without the persistent service, use
  `opencode run --standalone "your prompt"`. On 1.x use `opencode run`.

See the [agent support matrix](matrix.md) and [server setup](../guides/server.md).
