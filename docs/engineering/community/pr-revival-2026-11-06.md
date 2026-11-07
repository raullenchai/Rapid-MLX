# Local-client PR follow-up evidence — 2026-11-06

Owner: Echo. Host: Apple M3 Ultra, macOS. Scope: localhost integration smoke
tests and follow-up on existing upstream threads. These results are not an
upstream endorsement, a performance comparison, or full compatibility certification.

## Environment

- Rapid-MLX 0.16.8, existing installed distribution.
- Model: `mlx-community/Qwen3.5-4B-MLX-4bit`, cached revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- LiteLLM 1.108.2; OpenAI SDK 3.46.0; isolated Python 3.12 environment.
- OpenCode 1.19.0, external plugins disabled.
- Hermes Agent reports v0.9.0 (2026.4.13), Python 3.11.14,
  OpenAI SDK 2.31.0. This is an older installation, not a current-main test.

## Server

Resolve the cached model snapshot locally and substitute its directory for
`<cached-snapshot>`. No weights were downloaded for this test.

```sh
HF_HUB_OFFLINE=1 rapid-mlx serve <cached-snapshot> \
  --host 127.0.0.1 --port 18941 --served-model-name rapid-local \
  --disable-model-downloads --disable-disk-caches --no-thinking --max-tokens 1024
```

`GET /v1/models` returned `rapid-local`, context_window 262144, and text,
vision and tools metadata. Only text and tool behavior were tested; advertised
vision and maximum context were not qualified by this smoke test.

## LiteLLM SDK and proxy

SDK calls used `model="openai/rapid-local"`,
`api_base="http://127.0.0.1:18941/v1"`, `api_key="local-placeholder"`,
temperature 0 and max_tokens 128. Results:

1. Non-streaming request returned `LOCAL_OK`.
2. Streaming deltas assembled to `LOCAL_OK`.
3. A required `add(a: integer, b: integer)` tool returned arguments
   `{"a":19,"b":23}`. The test calculated 42, supplied a tool-result message
   with the returned call ID, and the model answered with 42.

The proxy configuration was:

```yaml
model_list:
  - model_name: rapid-local
    litellm_params:
      model: openai/rapid-local
      api_base: http://127.0.0.1:18941/v1
      api_key: local-placeholder
litellm_settings:
  telemetry: false
```

The proxy was started with a generated temporary `LITELLM_MASTER_KEY` and
`litellm --config proxy.yaml --host 127.0.0.1 --port 18942`.
An authenticated OpenAI client request through port 18942 returned
`PROXY_LOCAL_OK`. The key was kept out of this record and all upstream comments.
The initial launch without a master key was refused by LiteLLM 1.108.2;
configure authentication explicitly rather than documenting an unauthenticated
proxy startup as working. The server's placeholder key and the proxy's actual
master key serve different purposes.

No native `rapid_mlx/` LiteLLM registration was used. Docker reachability,
embeddings, images, concurrency and long-running agent sessions were not tested.

## OpenCode

In a disposable directory containing `fixture.txt` with `PR_REVIVAL_73921`, use:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "rapidmlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Rapid-MLX",
      "options": {"baseURL": "http://127.0.0.1:18941/v1", "apiKey": "local-placeholder"},
      "models": {"rapid-local": {"name": "Rapid local", "limit": {"context": 32768, "output": 1024}}}
    }
  },
  "model": "rapidmlx/rapid-local",
  "small_model": "rapidmlx/rapid-local",
  "enabled_providers": ["rapidmlx"],
  "permission": {"*": "deny", "read": "allow"},
  "share": "disabled"
}
```

Run `opencode run --pure --dir <test-dir> --format json -m rapidmlx/rapid-local`
with a prompt explicitly specifying the full absolute fixture path and asking
to read it. Set `OPENCODE_CONFIG` to this configuration and use disposable XDG
config/data/cache directories. The read tool completed and the final answer
contained `PR_REVIVAL_73921`. The initial relative-path prompt produced an
incorrect model-generated absolute path and failed; only the corrected run
passed. This is a read-tool roundtrip, not a coding benchmark.

## Hermes

Use an isolated `HERMES_HOME` containing this `config.yaml`:

```yaml
model:
  default: rapid-local
  provider: custom
  base_url: http://127.0.0.1:18941/v1
  api_key: local-placeholder
  context_length: 262144
agent:
  max_turns: 2
```

`hermes chat -m rapid-local -t terminal --max-turns 2 -Q -q
'Reply with exactly HERMES_LOCAL_OK. No tools are needed.'`
returned `HERMES_LOCAL_OK`. No tool invocation was requested or qualified here.
The configured context matches server metadata; this did not test 262144-token
inputs. A 32768 setting was refused by this installation's minimum-context
check. An initial environment-variable-only attempt selected the wrong provider
and failed authentication; the explicit custom-provider YAML was the successful
configuration. Current upstream docs also direct custom endpoints to YAML.

## Upstream dispositions and next actions

| Thread | Evidence / next action |
| --- | --- |
| [OpenCode #39255](https://github.com/anomalyco/opencode/pull/39255) | Stale cleanup explicitly allows a reopen request. Supply the read-tool smoke and ask to resume this docs-only thread. |
| [LiteLLM #24325](https://github.com/BerriAI/litellm/pull/24325) | Update the existing November 6 comment with SDK/proxy results; retain the smaller OpenAI-compatible recipe proposal. Do not claim old native-provider code is merge-ready. |
| [Hermes #136903](https://github.com/NousResearch/hermes-agent/pull/136903) | Existing docs PR is open. Supply narrowly scoped smoke evidence with the installed-version limitation; do not resubmit rejected in-tree provider integration. |
| [models.dev #1361](https://github.com/anomalyco/models.dev/pull/1361) | Local providers remain represented by LMStudio. Current schema requires v2 provider files, base_model references and a compliant logo. Ask whether a minimal locally served entry is the preferred scope before replacing the stale catalog. Migration and bun validation are not yet performed. |

Atlas/Echo follow-up: current Hermes qualification, models.dev v2 migration and
upstream acceptance remain outstanding. Work branch: `echo/pr-revival-20261106`.
No PR in this table was newly accepted as a result of these tests.

## Submitted follow-ups

- [OpenCode reopen request and verified config](https://github.com/anomalyco/opencode/pull/39255#issuecomment-6700322817).
- [Hermes supplemental smoke and version caveat](https://github.com/NousResearch/hermes-agent/pull/136903#issuecomment-6700318965).
- [LiteLLM existing comment updated in place](https://github.com/BerriAI/litellm/pull/24325#issuecomment-6696580771).
- [models.dev v2/local-provider scope follow-up](https://github.com/anomalyco/models.dev/pull/1361#issuecomment-6700323051).

Hermes remains open; the three other PRs remain closed pending maintainer action.
No new duplicate PR was created. The models.dev migration is not yet implemented;
the scope question is not an acceptance or a validation result.
