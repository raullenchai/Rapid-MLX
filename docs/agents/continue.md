# Continue

Continue VS Code 2.0.0 and `cn` CLI 1.5.47 use `~/.continue/config.yaml` for
local models. `cn` also accepts `--config <path>` and can remember another
selected configuration; use `--config ~/.continue/config.yaml` when checking
the file setup wrote. Its OpenAI provider entry uses `model`, `apiBase`, and a
nonempty `apiKey`.

```bash
npm install -g @continuedev/cli
rapid-mlx serve qwen3.5-4b-4bit
rapid-mlx agents continue --setup
cn --config ~/.continue/config.yaml --auto -p "Read a file and explain it"
```

Setup previews and backs up existing YAML, preserving other models. It
converts legacy `config.json` into the YAML file without deleting the source.
For another port, add `--base-url http://127.0.0.1:18730/v1` to setup.
`rapid-mlx launch continue-dev` is an alternative setup path.

In VS Code or JetBrains, choose the **rapid-mlx** model after setup. The
`config.json` format is legacy and the CLI does not use it. Continue can use
different models for autocomplete by assigning an `autocomplete` role; setup
adds a chat model.

Sources: [Continue CLI configuration](https://github.com/continuedev/continue/blob/main/docs/cli/configuration.mdx),
[Continue CLI guide](https://github.com/continuedev/continue/blob/main/docs/guides/cli.mdx).
