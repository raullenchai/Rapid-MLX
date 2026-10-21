# Aider

[Aider](https://aider.chat/) talks to Rapid-MLX through the OpenAI-compatible
`/v1/chat/completions` endpoint. Aider 0.86.2 was exercised with a fresh HOME
and `qwen3.5-4b-4bit`: it read `add.py` and applied an edit that made
`add(2, 3) == 5`.

## Setup

```bash
uv tool install aider-chat --python 3.12
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

In another terminal, run `rapid-mlx agents aider --setup` and export the three
lines it prints. For this server, they are:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
export AIDER_MODEL=openai/qwen3.5-4b-4bit
aider --no-git --yes --message "Fix add.py so add(2, 3) returns 5" add.py
```

Set `OPENAI_API_KEY` to the server's key if you started Rapid-MLX with one.
The `openai/` prefix tells Aider's LiteLLM transport which provider to use.
Aider also accepts these settings in `.aider.conf.yml` or `.env`, but the
Rapid-MLX setup command prints shell exports and writes no config file.

`rapid-mlx agents aider --test` checks chat, streaming, and a real Aider CLI
reply when `aider` is on PATH. Aider applies file changes with its text edit
format, so that test does not expect OpenAI function calls; confirm the file
content for a file-edit task.

See Aider's [OpenAI-compatible endpoint guide](https://aider.chat/docs/llms/openai-compat.html)
and [configuration guide](https://aider.chat/docs/config/aider_conf.html).
