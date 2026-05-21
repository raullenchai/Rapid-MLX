# Documentación de vLLM-MLX

**Backend MLX para Apple Silicon en vLLM** - Aceleración GPU para texto, imagen, video y audio en Mac

## ¿Qué es vLLM-MLX?

vllm-mlx incorpora aceleración GPU nativa de Apple Silicon a vLLM mediante la integración de:

- **[MLX](https://github.com/ml-explore/mlx)**: El framework de ML de Apple con memoria unificada y kernels Metal
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Inferencia LLM optimizada con KV cache y cuantización
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Modelos visión-lenguaje para inferencia multimodal
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: TTS y STT con voces nativas
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**: Embeddings de texto para búsqueda semántica y RAG

## Características principales

- **Multimodal** - Texto, imagen, video y audio en una sola plataforma
- **Aceleración GPU nativa** en Apple Silicon (M1, M2, M3, M4, M5)
- **Voces TTS nativas** - Español, francés, chino, japonés y 5 idiomas más
- **Compatible con la API de OpenAI** - reemplazo directo del cliente de OpenAI
- **Embeddings** - Endpoint `/v1/embeddings` compatible con OpenAI
- **MCP Tool Calling** - integración de herramientas externas mediante el Model Context Protocol
- **Paged KV Cache** - almacenamiento en caché eficiente en memoria con prefix sharing
- **Continuous Batching** - alto rendimiento para múltiples usuarios concurrentes

## Enlaces rápidos

### Primeros pasos
- [Instalación](getting-started/installation.md)
- [Inicio rápido](getting-started/quickstart.md)

### Guías de usuario
- [Servidor compatible con OpenAI](guides/server.md)
- [API de Python](guides/python-api.md)
- [Multimodal (imágenes y video)](guides/multimodal.md)
- [Audio (STT/TTS)](guides/audio.md)
- [Embeddings](guides/embeddings.md)
- [Modelos de reasoning](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP y Tool Calling](guides/mcp-tools.md)
- [Continuous Batching](guides/continuous-batching.md)

### Referencia
- [Comandos CLI](reference/cli.md)
- [Modelos compatibles](reference/models.md)
- [Configuración](reference/configuration.md)

### Benchmarks
- [Benchmarks LLM](benchmarks/llm.md)
- [Benchmarks de imagen](benchmarks/image.md)
- [Benchmarks de video](benchmarks/video.md)
- [Benchmarks de audio](benchmarks/audio.md)

### Desarrollo
- [Arquitectura](../development/architecture.md)
- [Contribuir](../development/contributing.md)

## Requisitos

- macOS en Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10+
- Se recomiendan 8 GB de RAM o más

## Licencia

Apache 2.0 - Consulta [LICENSE](../../LICENSE) para más detalles.
