# Rapid-MLX Documentation

**Native AI inference for Apple Silicon** — GPU-accelerated text, image, video, and audio on Mac.

Install on an Apple Silicon Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Restart Terminal, then run a coding agent with a local model:

```bash
rapid-mlx launch codex
```

Replace `codex` with `claude-code`, `opencode`, `pi`, `hermes`, `crush`, or
`goose`. For a terminal conversation, run `rapid-mlx chat`.
See the [current web documentation](https://rapidmlx.com/docs/) for full guides.

## What Rapid-MLX provides

- Local text and multimodal inference on Apple Silicon
- OpenAI- and Anthropic-compatible local API routes
- Coding-agent launch with private profiles
- Model recommendations based on this Mac's memory

## Quick Links

### Getting Started
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md) — including `rapid-mlx chat` for an instant REPL

### User Guides
- [OpenAI-Compatible Server](guides/server.md)
- [Headless macOS service](guides/headless-macos-service.md)
- [Python API](guides/python-api.md)
- [Multimodal (Images & Video)](guides/multimodal.md)
- [Audio (STT/TTS)](guides/audio.md)
- [Embeddings](guides/embeddings.md)
- [Rerank](guides/rerank.md)
- [Local folder search](guides/local-search.md)
- [Reasoning Models](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP & Tool Calling](guides/mcp-tools.md)
- [Rapid Agent Runtime](guides/agent-runtime.md)
- [Continuous Batching](guides/continuous-batching.md)
- [AI Client Compatibility](guides/ai-clients.md)
- [SDK Compatibility Notes](guides/sdk-compat.md)

### Coding Agents
- [Coding agents](agents/matrix.md) — one-command launch and client guides
- [Codex CLI](agents/codex-cli.md)
- [Claude Code](agents/claude-code.md)
- [OpenCode](agents/opencode.md)
- [Pi](agents/pi.md)
- [Hermes Agent](agents/hermes-agent.md)
- [Crush](agents/crush.md)
- [Goose](agents/goose.md)

### Reference
- [CLI Commands](reference/cli.md)
- [Supported Models](reference/models.md)
- [Configuration](reference/configuration.md)

### Benchmarks
- [Measure on your Mac](benchmarks/README.md)
- [Text generation](benchmarks/llm.md)

### Development
- [Architecture](development/architecture.md)
- [Contributing](development/contributing.md)

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 8GB+ RAM recommended

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.
