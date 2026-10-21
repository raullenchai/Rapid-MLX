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
> configuration contract exercised by **dsh 0.2.0-rc.2** (verified 2026-10-02;
> npm `latest` resolves there). dsh 0.2 moved its user configuration from
> `$DSH_HOME/settings.yaml` to Cordis patch layers, and 0.1.x headless mode
> crashes with a Cordis HMR bug — so **0.2.0-rc.2 is the minimum supported
> version**.

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
previews an exact diff before changing the home-level Cordis patch layer
`$DSH_HOME/cordis.patch.yml` (default `~/.dsh/cordis.patch.yml`). dsh 0.2
auto-loads that layer for **every** profile — bundle patches, the per-profile
`cordis.patch.yml`, the home-level layer, then `--patch` overlays — so no
extra flags are needed on the command line. The file is a top-level YAML list
of `{id, config}` entries (the only shape dsh's patch parser accepts); Rapid's
`llm-pi-ai` and `agent-default-model` entries are merged by id, and any other
layers you keep there survive. Setup makes a timestamped backup, writes
atomically, and refuses to overwrite a file changed after the preview.

For one-off runs you can bypass the home-level layer entirely: save the patch
list anywhere and pass it explicitly — `dsh --profile headless --patch
<path> "<task>"` (the path the 2026-10-02 verification run used).

Harness's current generic OpenAI transport requires a credential reference even
for an unauthenticated loopback server. Rapid therefore adds the non-secret
sentinel `RAPID_MLX_API_KEY: not-needed` to Harness's owner-only managed
`.credentials.yaml` when that key is absent, and the provider names
`RAPID_MLX_API_KEY` as its `apiKeyEnv`, so exporting the variable also works.
An existing value is preserved, and credential values are redacted from the
setup preview.

If a `~/.dsh/settings.yaml` exists from a pre-0.2 Rapid setup (or an older
dsh), it is dead weight: dsh 0.2.x no longer reads it. It can be deleted after
upgrading; Rapid's setup no longer touches it.

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
