# Aider

[Aider](https://aider.chat/) talks to Rapid-MLX through the OpenAI-compatible
`/v1/chat/completions` endpoint. Aider 0.86.2 was exercised with a fresh HOME
and `qwen3.5-4b-4bit`: it read `add.py` and applied an edit that made
`add(2, 3) == 5`.

## Isolated launch

```bash
rapid-mlx launch aider --model flint-40b-a2.5b-4bit
```

The private launch profile uses Aider's `diff` (SEARCH/REPLACE) edit format,
including `editor-diff` for editor mode. Its generated model settings file
registers the selected `openai/<model>` with this format and enables the repository
map. The main, weak (commit messages and history summaries), and editor models
all stay on the selected local model. Sampling temperature is left to the server
instead of Aider forcing greedy decoding. Existing personal Aider settings are not
rewritten; launching again refreshes the private profile.

Aider applies text edits; it does not use OpenAI function calling. The whole-file
fallback for unknown models can interpret a fenced output example as a filename.
Explicit SEARCH/REPLACE blocks distinguish edits from prose and shell suggestions.
Model adherence still varies, so review the proposed edits.

Ask, for example: “Create hello.py that prints HELLO, then run python3 hello.py.”
Shell suggestions are enabled. When Aider offers `Run shell command?`, inspect the
command and answer `y` to execute it or `n` to decline. It can then offer to add
real command output to the conversation. If the model omits a shell suggestion,
enter `/run python3 hello.py` yourself; a claimed output in prose is not execution.

The launcher does not enable `--yes-always` or automatic tests. Aider's
`--test-cmd` with `--auto-test` is a separate, explicitly configured workflow
for running a known test command after edits, not a general create-and-run mode.
Remove the private profile with `rapid-mlx launch aider --remove`.

These settings use Aider's supported [edit formats](https://aider.chat/docs/more/edit-formats.html),
[model settings file](https://aider.chat/docs/config/adv-model-settings.html),
[shell and test options](https://aider.chat/docs/config/options.html), and
[`/run` command](https://aider.chat/docs/usage/commands.html).

## Manual setup

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

## Verification

On 2026-10-22, Aider 0.86.2 and Rapid-MLX 0.16.8 were exercised on an Apple
M3 Ultra Studio (256 GB), using `flint-40b-a2.5b-4bit` at revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0`. The server used port 18831 and
MLX 0.32.4. Each trial used an empty directory and the generated private launch
helper, with normal interactive confirmations and no automatic acceptance flag.

Reproduce with `rapid-mlx launch aider --model flint-40b-a2.5b-4bit --port 18831`
and ask: “Create hello.py in the current directory containing Python code that
prints exactly HARBOR_FLINT_OK. Run python3 hello.py and report the actual output.”
Inspect the complete directory contents and the shell execution record, not
just Aider's prose or exit status. Long reasoning can still delay the response;
this profile does not change Flint's reasoning policy.

Both fresh acceptance trials created only `hello.py` (apart from Aider's own
hidden history files), offered `python3 hello.py`, and printed `HARBOR_FLINT_OK`
after an explicit `y` response. The returned fenced output examples did not
become extra files. These are focused smoke checks, not a general model
reliability guarantee.
