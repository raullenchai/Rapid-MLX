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

| Client | Type | Setup | Status | Notes |
|--------|------|-------|--------|-------|
| [Aider](https://aider.chat) | CLI | `OPENAI_API_BASE=http://localhost:8000/v1 aider --model openai/default` | Verified | Architect mode, edit-and-commit |
| [OpenCode](https://github.com/sst/opencode) | TUI | `rapid-mlx agents opencode --setup` | Verified | Claude Code-like terminal UX |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | CLI | `ANTHROPIC_BASE_URL=http://localhost:8000 claude` | Verified | Uses Anthropic `/v1/messages` |
| [Cursor](https://cursor.com) | IDE | Settings > Models > OpenAI Base URL: `http://localhost:8000/v1` | Compatible | Agent/composer mode uses tool calling |
| [Continue.dev](https://continue.dev) | IDE Extension | `~/.continue/config.yaml` `apiBase: http://localhost:8000/v1` | Compatible | VS Code / JetBrains |
| [pi](https://shittycodingagent.ai) | TUI | `OPENAI_BASE_URL=http://localhost:8000/v1` | Community-reported | Works with Qwen3.5/Qwen3.6 models |

### Web UIs

| Client | Type | Setup | Status | Notes |
|--------|------|-------|--------|-------|
| [Open WebUI](https://openwebui.com) | Docker | `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1` | Verified | Full chat UI |
| [LibreChat](https://librechat.ai) | Docker | Configure custom endpoint `http://host.docker.internal:8000/v1` | Verified | Multi-provider chat |

## Clients to Test

The following clients are reported to support OpenAI-compatible APIs
but have not yet been verified by the Rapid-MLX team. Community
contributions welcome -- see [Testing Methodology](#testing-methodology).

| Client | Type | Endpoint Type | Configuration Hints | Reported Status |
|--------|------|---------------|---------------------|-----------------|
| [CrewAI](https://crewai.com) | Framework | OpenAI | Set `OPENAI_API_BASE=http://localhost:8000/v1` | Untested |
| [AutoGen](https://microsoft.github.io/autogen/) | Framework | OpenAI | `base_url="http://localhost:8000/v1"` in `llm_config` | Untested |
| [LlamaIndex](https://llamaindex.ai) | Framework | OpenAI | `OpenAI(api_base="http://localhost:8000/v1")` | Untested |
| [Cline](https://github.com/cline/cline) | IDE Extension | OpenAI Compatible | Provider: OpenAI Compatible, Base URL: `http://localhost:8000/v1` | Configuration issues reported ([#47](https://github.com/raullenchai/Rapid-MLX/issues/47#issuecomment-4378465919)) |
| [Open Interpreter](https://openinterpreter.com) | CLI | OpenAI | `OPENAI_API_BASE=http://localhost:8000/v1 interpreter` | Untested |
| [Dify](https://dify.ai) | Platform | OpenAI-compatible | Add custom OpenAI provider at `http://localhost:8000/v1` | Untested |
| [n8n AI Nodes](https://n8n.io) | Automation | OpenAI | Node config: Base URL `http://localhost:8000/v1` | Untested |
| [Bolt.new (local)](https://github.com/stackblitz/bolt.new) | Web | Anthropic | Set `ANTHROPIC_BASE_URL=http://localhost:8000` | Untested |
| [codex](https://github.com/openai/codex) | CLI | OpenAI | `rapid-mlx agents codex --setup` or `OPENAI_BASE_URL=http://localhost:8000/v1` | Agent profile exists |
| [Goose](https://block.github.io/goose/) | CLI | OpenAI | `rapid-mlx agents goose --setup` | Agent profile exists |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | Web/Docker | OpenAI | `rapid-mlx agents openhands --setup` | Agent profile exists |
| [OpenClaude](https://github.com/raullenchai/Rapid-MLX/tree/main) | CLI | Anthropic | `rapid-mlx agents openclaude --setup` | Agent profile exists |
| [Tabby](https://tabby.tabbyml.com) | IDE | OpenAI | `TABBY_OPENAI_API_BASE=http://localhost:8000/v1` | Untested |
| [Windsurf](https://codeium.com/windsurf) | IDE | OpenAI-compatible | Settings > OpenAI Base URL | Untested |
| [Zed](https://zed.dev) | IDE | OpenAI-compatible | `assistant.openai_api_url: "http://localhost:8000/v1"` in settings | Untested |

Clients with a `rapid-mlx agents` profile (`codex`, `goose`, `openhands`,
`openclaude`) have pre-built configuration but automated integration tests
are not yet in `tests/integrations/`.

## Detailed Setup Examples

### CrewAI

```python
import os
from crewai import Agent, Task, Crew
from crewai.llm import LLM

os.environ["OPENAI_API_KEY"] = "not-needed"

llm = LLM(
    model="openai/default",
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

researcher = Agent(
    role="Researcher",
    goal="Research the topic thoroughly",
    backstory="Expert researcher",
    llm=llm,
)

task = Task(
    description="Explain how local LLM inference works on Apple Silicon.",
    expected_output="A paragraph of explanation.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
print(crew.kickoff())
```

Test: verify streaming responses, tool calling, and multi-agent
workflows.

### AutoGen

```python
import os
from autogen import ConversableAgent

os.environ["OPENAI_API_KEY"] = "not-needed"

llm_config = {
    "config_list": [{
        "model": "default",
        "api_key": "not-needed",
        "base_url": "http://localhost:8000/v1",
    }]
}

agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful assistant.",
)

reply = agent.generate_reply(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(reply)
```

Test: verify multi-agent conversations, code execution, and tool
use. AutoGen's `base_url` support varies by version --
`openai>=1.0`-compatible versions work best.

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="default",
    api_key="not-needed",
    api_base="http://localhost:8000/v1",
)

response = llm.complete("Explain what MLX is.")
print(response)
```

For embeddings with LlamaIndex and Rapid-MLX:

```bash
rapid-mlx serve qwen3.5-4b-4bit \
    --embedding-model mlx-community/multilingual-e5-small-mlx
```

```python
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(
    model="mlx-community/multilingual-e5-small-mlx",
    api_key="not-needed",
    api_base="http://localhost:8000/v1",
)
```

### Cline (VS Code Extension)

1. Start the server:
   ```bash
   rapid-mlx serve qwen3.5-4b-4bit --port 8000
   ```

2. In Cline settings:
   - **API Provider**: `OpenAI Compatible`
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `not-needed`
   - **Model ID**: enter the model name shown in `rapid-mlx models` (e.g. `qwen3.5-4b`)

   If the model list fails to load, set **Model ID** to `default`.

3. Verify connectivity by sending "Hello" in Cline's chat. If the
   request reaches the server you will see an `INFO` log line in the
   `rapid-mlx serve` terminal.

**Known issue** ([#47](https://github.com/raullenchai/Rapid-MLX/issues/47#issuecomment-4378465919)): One community member reported no response
from the server when using Cline. This is likely a configuration
mismatch. Double-check:
- The server is running and reachable at `curl http://localhost:8000/health`
- The Base URL ends with `/v1`
- The Model ID matches the loaded model or is `default`

### Dify

1. Start the Rapid-MLX server:
   ```bash
   rapid-mlx serve qwen3.5-4b-4bit --port 8000
   ```

2. In Dify, go to Settings > Model Provider > OpenAI-API-compatible:
   - **Model Name**: `default`
   - **API Endpoint URL**: `http://localhost:8000/v1/chat/completions`
   - **API Key**: `not-needed`

3. If you need embeddings, add a separate Text Embedding provider:
   - **API Endpoint URL**: `http://localhost:8000/v1/embeddings`

Test: create a simple chat app and verify streaming and
conversation history work.

### n8n AI Nodes

1. Start the Rapid-MLX server.
2. In n8n, add an OpenAI Chat Model node.
3. Configure the node:
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `not-needed`
   - **Model**: `default`

Test: create a simple workflow with an AI node and verify it
produces output.

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
| `aider` | Aider | Yes | Yes (`test_aider.sh`) |
| `cline` | Cline (VS Code) | Config template | No |
| `codex` | OpenAI Codex CLI | Yes | No |
| `generic` | Any OpenAI client | Env vars | No |
| `goose` | Block Goose | Yes | No |
| `hermes` | Hermes Agent | Yes | Yes (`test_hermes.py`) |
| `langchain` | LangChain | Python snippet | Yes (`test_langchain.py`) |
| `openclaude` | OpenClaude | Yes | No |
| `opencode` | OpenCode | Yes | No |
| `openhands` | OpenHands | Yes | No |
| `pydanticai` | PydanticAI | Python snippet | Yes (`test_pydantic_ai_full.py`) |
| `smolagents` | smolagents | Python snippet | Yes (`test_smolagents_full.py`) |

To add a new agent profile, create a YAML file in
`vllm_mlx/agents/profiles/` following the structure in
`generic.yaml`. See `vllm_mlx/agents/base.py` for the data model.
