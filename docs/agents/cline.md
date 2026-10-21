# Cline

Cline CLI 3.0.70 and the VS Code extension 4.1.23 can send OpenAI-compatible
chat completions to Rapid-MLX. The CLI's provider store is
`~/.cline/data/settings/providers.json`; `CLINE_DATA_DIR` overrides its data
root, `CLINE_DIR` overrides the parent of `data`, and
`CLINE_PROVIDER_SETTINGS_PATH` overrides the provider file itself. The provider entry is
`openai-compatible` with `baseUrl`, `apiKey`, and `model` fields. Cline's MCP
settings file is unrelated to model routing.

```bash
npm install -g cline
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents cline --setup
cline --yolo "Read a file and explain it"
```

Setup shows an exact diff, asks before writing, backs up an existing provider
file, and checks the server. In scripts, add `--yes`. For an alternate port,
pass `--base-url http://127.0.0.1:18730/v1` to setup. The API key is `sk-noop`
for an unprotected local server; if Rapid-MLX uses a key, export
`RAPID_MLX_API_KEY` before setup.

The CLI reads this provider store. A VS Code session that has another provider
selected must switch **Cline → Settings → API Configuration** to **OpenAI
Compatible**, the local `/v1` base URL, the served model ID, and the same key.
`rapid-mlx launch cline --server-url http://127.0.0.1:18730` remains an
alternative CLI setup command.

`rapid-mlx agents cline --test` runs wire checks and a CLI exercise when
`cline` is installed. A small model may complete without performing a requested
file or shell action; check the resulting artifact before claiming a tool pass.

Sources: [Cline CLI](https://github.com/cline/cline/blob/main/apps/cli/README.md),
[Cline provider settings](https://github.com/cline/cline/blob/main/apps/cli/DEVELOPMENT.md).
