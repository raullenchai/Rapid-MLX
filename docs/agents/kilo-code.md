# Kilo Code

Kilo CLI 7.8.8 reads `~/.config/kilo/kilo.json[c]` (or a project `kilo.json[c]`).
It no longer falls back to `.opencode` configuration. Rapid-MLX setup writes
a `rapid-mlx` provider using `@ai-sdk/openai-compatible`, a `/v1` base URL,
and the model and context reported by the server.

```bash
npm install -g @kilocode/cli
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents kilo-code --setup
kilo run --pure --auto "Read a file and explain it"
```

Setup merges an existing JSON file and supports `--dry-run`. For another port,
add `--base-url http://127.0.0.1:18730/v1`. The key is `not-needed` on an
unprotected local server. `rapid-mlx agents kilo-code --test` runs the wire
and CLI checks when `kilo` is installed.

The VS Code extension has its own provider selection. Open its Providers
settings, choose **OpenAI Compatible**, set the `/v1` base URL and served
model ID, and use the same key. CLI JSON setup alone does not select a model
in an already open editor session.

Sources: [Kilo CLI configuration](https://github.com/Kilo-Org/kilocode/blob/main/packages/kilo-docs/pages/code-with-ai/platforms/cli.md),
[Kilo custom models](https://kilo.ai/docs/code-with-ai/agents/custom-models).
