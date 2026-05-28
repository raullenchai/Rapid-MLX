# vLLM-MLX Documentation

**Apple Silicon MLX Backend for vLLM** - GPU-accelerated Text, Image, Video & Audio on Mac

## What is vLLM-MLX?

rapid-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: Text-to-Speech and Speech-to-Text with native voices
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**: Text embeddings for semantic search and RAG

## Key Features

- **Multimodal** - Text, Image, Video & Audio in one platform
- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Native TTS voices** - Spanish, French, Chinese, Japanese + 5 more languages
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **Embeddings** - OpenAI-compatible `/v1/embeddings` endpoint
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Links

### Getting Started
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md) — including `rapid-mlx chat` for an instant REPL

### User Guides
- [OpenAI-Compatible Server](guides/server.md)
- [Python API](guides/python-api.md)
- [Multimodal (Images & Video)](guides/multimodal.md)
- [Audio (STT/TTS)](guides/audio.md)
- [Embeddings](guides/embeddings.md)
- [Reasoning Models](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP & Tool Calling](guides/mcp-tools.md)
- [Continuous Batching](guides/continuous-batching.md)

### Reference
- [CLI Commands](reference/cli.md)
- [Supported Models](reference/models.md)
- [Configuration](reference/configuration.md)

### Benchmarks
- [LLM Benchmarks](benchmarks/llm.md)
- [Image Benchmarks](benchmarks/image.md)
- [Video Benchmarks](benchmarks/video.md)
- [Audio Benchmarks](benchmarks/audio.md)

### Development
- [Architecture](development/architecture.md)
- [Contributing](development/contributing.md)

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 8GB+ RAM recommended

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.
