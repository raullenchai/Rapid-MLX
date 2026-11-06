# Connect local applications to Rapid-MLX

These provider-maintained recipes use existing OpenAI-compatible connectors in
Open WebUI, AnythingLLM and LiteLLM. They are configuration references, not
upstream endorsements. Checked against the upstream connector documentation and
Rapid-MLX CLI help on 2026-11-06; full application runs were not performed for
this document. Validate your installed versions before relying on agent tools.

## Start a local text server

On an Apple Silicon Mac:

```bash
brew install rapid-mlx
rapid-mlx serve qwen3.5-4b-4bit --no-mllm --host 127.0.0.1 --port 8000
```

The first run may download the model. Wait for the ready message. This example
uses an explicit port so the client configuration does not depend on automatic
port selection. If it is occupied, choose a free port in both server and client.

```bash
curl http://127.0.0.1:8000/v1/models
```

Copy the returned `data[].id` into the client model field. First test basic chat
with that ID in place of `MODEL_ID_FROM_MODELS`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID_FROM_MODELS","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":128}'
```

The example server has no API key and binds only to loopback. If a client
requires a nonempty key, `local-only` is a placeholder, not authentication.
If you start Rapid-MLX with `--api-key`, configure the same real key in the client.
Rapid-MLX enables anonymous metadata telemetry by default; set `DO_NOT_TRACK=1`
before launching to opt out.

## Open WebUI

In **Admin Settings → Connections**, add an OpenAI-compatible connection:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | Empty for the loopback server above |
| Model | The ID returned by `/v1/models` |

Refresh the model list and send a short text message. Test streaming separately
before using long conversations. Use the existing connection mechanism; no
custom pipeline or native provider adapter is required for basic chat.

This URL applies when the Open WebUI backend runs directly on the same Mac.
Inside Docker, `127.0.0.1` points to the container. On Docker Desktop for Mac,
`host.docker.internal` names the host, but connectivity also depends on the
server binding and Docker configuration. Do not simply expose an unauthenticated
server on `0.0.0.0`; use an authenticated, appropriately restricted binding and
verify reachability from the client environment. A remote hosted UI cannot
reach a Mac's loopback address.

Source: [Open WebUI compatible-provider guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/).

## AnythingLLM

Choose **OpenAI (Generic)** in LLM settings:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | The exact ID returned by `/v1/models` |
| API key | `local-only` if a nonempty field is required; otherwise empty |
| Max tokens | Start with `128` for the connection check |
| Context window | The context limit actually configured for your model/server |

Start with an empty workspace and a short chat message. Increase the output
budget for real tasks. A context setting is a client budget; it does not enlarge
the model or available memory. Leave the output-token field name at its default
`max_tokens` for this recipe.

For document chat, configure an embedding provider separately. Selecting
Rapid-MLX for generation does not automatically configure embeddings, retrieval
or reranking. To keep document processing local, verify those components and
any enabled tools also run locally. Check one document retrieval and one
explicitly allowed tool loop before describing a complete RAG/agent setup as
working. The Docker/remote-host distinction above applies here too.

Source: [AnythingLLM Generic OpenAI guide](https://docs.anythingllm.com/setup/llm-configuration/cloud/openai-generic).

## LiteLLM SDK and proxy

Use the existing `openai/` route; `rapid_mlx/` is not assumed to be registered.
With the `litellm` package installed, replace the model placeholder:

```python
import litellm

reply = litellm.completion(
    model="openai/MODEL_ID_FROM_MODELS",
    api_base="http://127.0.0.1:8000/v1",
    api_key="local-only",  # Placeholder for the unauthenticated loopback example.
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=128,
)
print(reply.choices[0].message.content)
```

For the proxy, save this as `rapid-local.yaml` after replacing the model ID:

```yaml
model_list:
  - model_name: rapid-local
    litellm_params:
      model: openai/MODEL_ID_FROM_MODELS
      api_base: http://127.0.0.1:8000/v1
      api_key: local-only
```

Run with the LiteLLM proxy dependencies installed:

```bash
litellm --config rapid-local.yaml --host 127.0.0.1 --port 4000
```

Clients now select `rapid-local` at `http://127.0.0.1:4000/v1`. If the proxy runs
in a container or another machine, adjust its upstream address accordingly.
Model generation on your hardware has no cloud-provider token bill, but still
uses local resources; this recipe does not register pricing or cost accounting.

Source: [LiteLLM compatible-endpoint guide](https://docs.litellm.ai/docs/providers/openai_compatible).

## Verification before publishing an application walkthrough

Record the application and Rapid-MLX versions, exact checkpoint revision, host,
RAM and startup command. Check model discovery, a short chat and streaming.
For agents, check a deterministic harmless tool call, execute the tool, submit
its result and verify the final answer. For RAG, verify retrieval separately.
An HTTP 200 or an alias in the catalog alone is not evidence for all of these.
