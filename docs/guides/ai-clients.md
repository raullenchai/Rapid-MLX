# AI Client Compatibility

Rapid-MLX is compatible with any AI client that supports the OpenAI API
or Anthropic Messages API. This guide catalogs known-compatible clients,
provides configuration examples, and tracks community-reported results.

## API Compatibility Surface

Rapid-MLX exposes two primary interfaces:

| API | Endpoints | Use Case |
|-----|-----------|----------|
| **OpenAI-compatible** | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech` | Most AI clients, frameworks, and IDEs |
| **Anthropic-compatible** | `/v1/messages`, `/v1/messages/count_tokens` | Claude Code, OpenCode, and other Anthropic SDK consumers |

Feature support available across both APIs:

- Streaming (SSE)
- Tool calling / function calling
- Structured output (JSON mode, JSON schema)
- Reasoning / chain-of-thought extraction
- Multi-turn conversations
- Vision (multimodal models)
- Embeddings
- Audio transcription
- Text-to-speech

## Quick Configuration Pattern

Most OpenAI-compatible clients need three values:

```
Base URL:  http://localhost:8000/v1
API Key:   not-needed (or any non-empty string if required)
Model:     default (or the model ID from rapid-mlx models)
```

For Anthropic-compatible clients, leave off the `/v1` path:

```
Base URL:  http://localhost:8000
API Key:   not-needed
```

> **Heads-up (L-01):** the Anthropic Python SDK silently appends
> `/v1/messages` to whatever `base_url` you give it. If you pass
> `http://localhost:8000/v1`, requests go to `/v1/v1/messages` and the
> server returns `404`. Always pass the bare host (`http://localhost:8000`).
> Full write-up in
> [SDK Compatibility Notes — L-01](sdk-compat.md#l-01--anthropic-sdk-base_url-must-not-include-v1).

## Verified Compatible Clients

These clients have been verified through automated integration tests
(`tests/integrations/`) or maintainer testing.

### Frameworks and SDKs

| Client | Type | Setup | Plain | Stream | Tools | Notes |
|--------|------|-------|-------|--------|-------|-------|
| [OpenAI SDK](https://pypi.org/project/openai/) | SDK | `base_url="http://localhost:8000/v1"` | Yes | Yes | Yes | Drop-in replacement |
| [Anthropic SDK](https://pypi.org/project/anthropic/) | SDK | `base_url="http://localhost:8000"` | Yes | Yes | Yes | Uses `/v1/messages` |
| [PydanticAI](https://ai.pydantic.dev) | Framework | `base_url="http://localhost:8000/v1"` | Yes | Yes | Yes | Typed agents, structured output |
| [LangChain](https://langchain.com) | Framework | `ChatOpenAI(base_url="http://localhost:8000/v1")` | Yes | Yes | Yes | `ChatOpenAI`, tools, streaming |
| [smolagents](https://huggingface.co/docs/smolagents) | Framework | `OpenAIServerModel(api_base="http://localhost:8000/v1")` | Yes | — | Yes | CodeAgent + ToolCallingAgent |

### Coding Agents

Tier-1 coding agents have dedicated copy-paste setup pages with per-family
config, plus an honest test-backed [support matrix](../agents/matrix.md):
[Codex CLI](../agents/codex-cli.md) ·
[Claude Code](../agents/claude-code.md) ·
[OpenCode](../agents/opencode.md) ·
[Qwen Code](../agents/qwen-code.md) ·
[OpenHands](../agents/openhands.md) ·
[Hermes Agent](../agents/hermes-agent.md) ·
[DeepSeek Harness](../agents/deepseek-harness.md).

| Client | Type | Setup | Status | Notes |
|--------|------|-------|--------|-------|
| [Aider](https://aider.chat) | CLI | `OPENAI_API_BASE=http://localhost:8000/v1 aider --model openai/default` | Verified | Architect mode, edit-and-commit |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | CLI / TUI / web | `rapid-mlx agents dsh --setup` | Verified | Tier-1 release gate; generic `openai-completions` provider, never `deepseek-official`. Needs Node 22.15+ |
| [OpenCode](https://github.com/sst/opencode) | TUI | `rapid-mlx agents opencode --setup` | Compatible | Claude Code-like terminal UX |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | CLI | `rapid-mlx agents claude-code --setup` | Compatible | Safe diff/confirm/backup flow; uses Anthropic `/v1/messages` |
| [Cursor](https://cursor.com) | IDE | `RAPID_MLX_API_KEY=your-secret rapid-mlx launch cursor --server-url https://your-public-host` | Not compatible locally | BYOK requests pass through Cursor's servers; public HTTPS and server auth are required |
| [Continue.dev](https://continue.dev) | IDE Extension | `rapid-mlx agents continue --setup` | Compatible | Safe diff/confirm/backup flow; VS Code / JetBrains |
| [pi](https://shittycodingagent.ai) | TUI | `OPENAI_BASE_URL=http://localhost:8000/v1` | Community-reported | Works with Qwen3.5/Qwen3.6 models |

Rapid-MLX rejects explicit local/private Cursor addresses, but it does not use
your Mac's DNS result as proof that Cursor's backend can reach a hostname.
Split-horizon DNS may look different from Cursor's network, so verify the
authenticated HTTPS endpoint from an external network before configuring it.

### Web UIs

| Client | Type | Setup | Status | Notes |
|--------|------|-------|--------|-------|
| [Open WebUI](https://openwebui.com) | Docker | `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1` | Verified | Full chat UI |
| [LibreChat](https://librechat.ai) | Docker | Configure custom endpoint `http://host.docker.internal:8000/v1` | Verified | Multi-provider chat |

## Clients to Test

The following clients support OpenAI-compatible APIs but have not yet been
verified against Rapid-MLX. If you have Apple Silicon and can test one,
see [Testing Methodology](#testing-methodology).

- **CrewAI** (Framework) — `OPENAI_API_BASE=http://localhost:8000/v1`
- **AutoGen** (Framework) — `base_url="http://localhost:8000/v1"` in `llm_config`
- **LlamaIndex** (Framework) — `OpenAI(api_base="http://localhost:8000/v1")`
- **Cline** (IDE Extension) — Provider: OpenAI Compatible, Base URL: `http://localhost:8000/v1` ([known issues](https://github.com/raullenchai/Rapid-MLX/issues/47#issuecomment-4410012225))
- **Open Interpreter** (CLI) — `OPENAI_API_BASE=http://localhost:8000/v1 interpreter`
- **Dify** (Platform) — Add custom OpenAI provider at `http://localhost:8000/v1`
- **n8n AI Nodes** (Automation) — Node config: Base URL `http://localhost:8000/v1`
- **Bolt.new** (Web) — `ANTHROPIC_BASE_URL=http://localhost:8000`
- **Tabby** (IDE) — `TABBY_OPENAI_API_BASE=http://localhost:8000/v1`
- **Windsurf** (IDE) — Settings > OpenAI Base URL
- **Zed** (IDE) — `assistant.openai_api_url: "http://localhost:8000/v1"` in settings

Clients with a `rapid-mlx agents` profile (pre-built config) not covered
in the tables above:

- **Codex CLI** (CLI) — `rapid-mlx agents codex --setup`
- **Kilo Code** (IDE Extension) — `rapid-mlx agents kilo-code --setup`
- **OpenHands** (Web/Docker) — `rapid-mlx agents openhands --setup`

## Testing Methodology

To contribute a compatibility report for a new client:

1. **Start the server** on a model appropriate for your Mac's RAM:
   ```bash
   # 16 GB Mac
   rapid-mlx serve qwen3.5-4b-4bit --port 8000

   # 24-32 GB Mac
   rapid-mlx serve qwen3.5-9b-4bit --port 8000
   ```

2. **Verify the server is reachable:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
   ```

3. **Configure the client** using the pattern:
   ```
   Base URL:  http://localhost:8000/v1
   API Key:   not-needed
   Model:     default
   ```

4. **Test these scenarios** and note results:

| Scenario | What to check |
|----------|---------------|
| Basic chat | Non-streaming response arrives, content is correct |
| Streaming chat | Tokens arrive progressively (SSE) |
| Tool calling | Model emits tool calls, client parses them, tool results fed back correctly |
| Multi-turn | Conversation history preserved across turns |
| Structured output | `response_format: {"type": "json_object"}` produces valid JSON |
| System prompt | System message influences model behavior |

5. **Report results** in the [issue #47
   thread](https://github.com/raullenchai/Rapid-MLX/issues/47) with:
   - Client name and version
   - Model used
   - What worked
   - What did not work (with logs if available)
   - Any workarounds discovered

### Troubleshooting Common Issues

**Client shows "Connection refused" or times out:**
- Verify the server is running: `curl http://localhost:8000/health`
- Check the host setting (`localhost` vs `127.0.0.1` vs `host.docker.internal` for Docker)
- Ensure no firewall is blocking port 8000

**Client requires an API key but won't accept "not-needed":**
Try an arbitrary non-empty string. Some clients reject the literal
string `not-needed`; use `sk-local` or `rapid-mlx` instead.

**Tool calling does not work:**
- Ensure `--enable-auto-tool-choice` is set on the server
- Match `--tool-call-parser` to your model (see [Tool Calling](tool-calling.md))
- Some models need the `hermes` parser for reliable tool calling

**Streaming is slow or choppy:**
- Adjust `--stream-interval` (lower = smoother, higher = throughput)
- Check for client-side buffering (some frameworks buffer SSE chunks)

**Model does not appear in the client's model list:**
- Use `model="default"` -- this always resolves to the loaded model
- If the client requires a specific model ID from the list endpoint,
  the `/v1/models` response returns the loaded model's ID

## Rapid-MLX agents CLI

Rapid-MLX ships a built-in agent manager that can auto-configure
several popular coding agents:

```bash
rapid-mlx agents              # List all supported agents
rapid-mlx agents <name> --setup  # Auto-configure an agent
rapid-mlx agents hermes --test   # Run the Hermes agent test suite
```

Currently supported profiles (in `vllm_mlx/agents/profiles/`):

| Profile | Agent | Auto-setup | Automated Tests |
|---------|-------|------------|----------------|
| `aider` | Aider | Env vars | Yes (`test_aider.sh`) |
| `claude-code` | Claude Code | Env vars | Yes (`test_agents_matrix.py`) |
| `codex` | Codex CLI | TOML config | Yes (`test_agents_matrix.py`) |
| `continue` | Continue.dev | JSON config | No |
| `deepseek-harness` | DeepSeek Harness | YAML config | Yes (`test_deepseek_harness_tier1.py`) |
| `hermes` | Hermes Agent | YAML config | Yes (`test_hermes.py`) |
| `kilo-code` | Kilo Code | JSON config | Yes (`test_agents_matrix.py`) |
| `langchain` | LangChain | Env vars | Yes (`test_langchain.py`) |
| `opencode` | OpenCode | JSON config | Yes (`test_agents_matrix.py`) |
| `openhands` | OpenHands | Env vars | Yes (`test_openhands.sh`) |
| `pydanticai` | PydanticAI | Env vars | Yes (`test_pydantic_ai_full.py`) |
| `qwen-code` | Qwen Code | JSON config | Yes (`test_agents_matrix.py`) |
| `smolagents` | smolagents | Env vars | Yes (`test_smolagents_full.py`) |

To add a new agent profile, create a YAML file in
`vllm_mlx/agents/profiles/` following the structure in
`aider.yaml`. See `vllm_mlx/agents/base.py` for the data model.
