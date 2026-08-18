# DeepSeek Harness

Run the official [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
against a local Rapid-MLX server. Rapid uses Harness's generic
`openai-completions` provider; it does not impersonate the DeepSeek cloud API.

**DeepSeek Harness is a Tier-1 agent.** That is a release gate, not a badge:
every version bump runs `dsh` through a real multi-step bug-fix task against a
local 35B model in `tests/integrations/agent_smoke.sh` and asserts the repo's
own test suite goes green afterwards. If `dsh` regresses, the release cannot
tag or publish. See the [agent matrix](matrix.md#agent-deepseek-harness).

> DeepSeek currently labels Harness a developer preview. Rapid tracks the
> configuration contract exercised by `@deepseek-ai/dsh@0.1.0-rc.7` (verified
> 2026-08-17; the rc.6 → rc.7 upgrade did not change the provider schema).
> Review release notes before upgrading across incompatible preview releases.

## Setup

```bash
npm install -g @deepseek-ai/dsh
rapid-mlx serve qwen3.5-9b-4bit

# In another terminal: preview, then apply.
rapid-mlx agents dsh --setup --dry-run
rapid-mlx agents dsh --setup

dsh web
# or one headless task
dsh --profile headless "summarize this workspace"
```

`--setup` discovers the running model and its advertised context window, then
previews an exact diff before changing `$DSH_HOME/settings.yaml` (default
`~/.dsh/settings.yaml`). It preserves unrelated providers and settings, makes a
timestamped backup, writes atomically, and refuses to overwrite a file changed
after the preview.

Harness's current generic OpenAI transport requires a credential reference even
for an unauthenticated loopback server. Rapid therefore adds the non-secret
sentinel `RAPID_MLX_API_KEY: not-needed` to Harness's owner-only managed
`.credentials.yaml` when that key is absent. An existing value is preserved,
and credential values are redacted from the setup preview.

## Test

```bash
rapid-mlx agents dsh --test
```

The test uses an isolated `DSH_HOME` and workspace. It exercises streaming,
reasoning/tool-call protocol behavior, a real headless response, file reading,
and a shell command without loading the operator's Harness sessions or
credentials.

## Node runtime

The current DSH package imports Node's Zstd stream API but does not declare that
runtime minimum in its npm manifest. Node 22.15 or newer is known to work. If an
older/incompatible Node is first on `PATH`, `agents dsh --test` reports the
runtime mismatch before DSH emits its opaque plugin-loader stack trace.
