# vllm-mlx

**Lee esto en otros idiomas:** [English](README.md) · [Español](README.es.md) · [Français](README.fr.md) · [中文](README.zh.md)

**Continuous batching + APIs OpenAI y Anthropic en un solo servidor. Inferencia nativa en Apple Silicon.**

[![PyPI version](https://img.shields.io/pypi/v/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub stars](https://img.shields.io/github/stars/waybarrios/vllm-mlx.svg?style=social)](https://github.com/waybarrios/vllm-mlx)

---

## ¿Qué es vllm-mlx?

Un servidor de inferencia estilo vLLM para Macs con Apple Silicon. A diferencia de usar `Ollama` o `mlx-lm` directamente, incluye **continuous batching, paged KV cache, prefix caching y KV cache en SSD**, y expone **tanto OpenAI `/v1/*` como Anthropic `/v1/messages`** desde un solo proceso. Corre LLMs, modelos de visión, audio y embeddings sobre Metal con memoria unificada, sin paso de conversión.

## Inicio rápido (30 segundos)

```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

**SDK de OpenAI:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hola!"}])
print(r.choices[0].message.content)
```

**SDK de Anthropic / Claude Code:**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

## Características

### APIs
- **Compatible con OpenAI**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/responses`
- **Compatible con Anthropic**: `/v1/messages` (streaming, tool use, system prompts)
- **MCP Tool Calling**: 12 parsers (OpenAI, Anthropic, Gemini, Qwen, DeepSeek, Gemma y más)
- **Salida estructurada**: JSON Schema vía `response_format` (lm-format-enforcer)

### Throughput y memoria
- **Continuous batching**: alto throughput para requests concurrentes
- **Paged KV cache**: eficiente en memoria con prefix sharing
- **KV cache en SSD**: volcá el prefix cache a disco para agentes con contexto largo (`--ssd-cache-dir`)
- **Warm prompts**: precargá prefixes populares al arrancar (`--warm-prompts`) para 1.3-2.25x de TTFT
- **Prefix cache**: basado en trie, compartido entre requests

### Multimodal
- **Texto + imagen + video + audio** desde un solo servidor
- Modelos de visión: Gemma 3, Gemma 4, Qwen3-VL, Pixtral, Llama vision
- **Audio de entrada** en el chat (bloques `audio_url`)
- **TTS nativo**: 11 voces, 15+ idiomas (Kokoro, Chatterbox, VibeVoice, VoxCPM)
- **STT**: familia Whisper con RTF hasta 197x en M4 Max

### Razonamiento y avanzado
- **Extracción de razonamiento**: Qwen3, DeepSeek-R1 (`--reasoning-parser`)
- **Reducción de expertos MoE**: `--moe-top-k` para +7-16% en Qwen3-30B-A3B
- **Decodificación especulativa**: `--mtp` para Qwen3-Next
- **Prefill disperso**: `--spec-prefill` basado en atención para reducir TTFT

### Observabilidad
- **Métricas Prometheus**: endpoint `/metrics` con `--metrics`
- **Benchmarker incluido**: `vllm-mlx bench-serve` para barridos de prompts con salida CSV/JSON

### Aceleración GPU nativa
- Solo Apple Silicon (M1, M2, M3, M4, M5) con kernels Metal vía MLX
- Memoria unificada, sin conversión de modelos

## Rendimiento

**Decode de LLM (M4 Max, 128 GB, greedy, single stream):**

| Modelo | Tok/s | Memoria |
|--------|------:|--------:|
| Qwen3-0.6B-8bit | 417.9 | 0.7 GB |
| Llama-3.2-3B-Instruct-4bit | 205.6 | 1.8 GB |
| Qwen3-30B-A3B-4bit | 127.7 | ~18 GB |

**Audio speech-to-text (M4 Max, RTF = real-time factor):**

| Modelo | RTF | Caso de uso |
|--------|----:|-------------|
| whisper-tiny | 197x | Tiempo real / baja latencia |
| whisper-large-v3-turbo | 55x | Calidad + velocidad |
| whisper-large-v3 | 24x | Máxima precisión |

Ver [docs/benchmarks/](docs/benchmarks/) para resultados de continuous batching, cuantización de KV cache (4-bit / 8-bit / fp16) y barridos de MoE top-k.

## Ejemplos

### API Anthropic (Claude Code, OpenCode)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Modelos de razonamiento (Qwen3, DeepSeek-R1)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "¿Cuánto es 17 * 23?"}],
)
print("Pensamiento:", r.choices[0].message.reasoning)
print("Respuesta:",   r.choices[0].message.content)
```

### Multimodal (imagen + texto)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "¿Qué hay en esta imagen?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}],
)
```

### Salida estructurada (JSON Schema)

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Lista 3 colores."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}
        },
    },
)
```

### Reranking (`/v1/rerank`)

```bash
curl http://localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "default",
  "query": "inferencia en apple silicon",
  "documents": ["MLX es el framework de Apple", "Kernels Metal en M-series", "CUDA en NVIDIA"]
}'
```

La ruta integrada de reranking MLX admite pesos de clasificación de secuencia
BERT/XLM-RoBERTa estándar con valores `hidden_act` `gelu`,
`gelu_new`/`gelu_fast`, `relu` o `silu`/`swish`. Otras activaciones fallan de
forma explícita para que las arquitecturas personalizadas agreguen un adaptador
dedicado en lugar de usar GELU silenciosamente.

### Embeddings

```bash
vllm-mlx serve <llm-model> --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
emb = client.embeddings.create(model="mlx-community/all-MiniLM-L6-v2-4bit", input=["Hola", "Mundo"])
```

### Audio (TTS / STT)

```bash
pip install vllm-mlx[audio]
brew install espeak-ng        # macOS, necesario para TTS en idiomas no-inglés

python examples/tts_example.py "Hello, how are you?" --play
python examples/tts_multilingual.py "Hola mundo" --lang es --play
```

### Benchmarking incluido

```bash
vllm-mlx bench-serve --url http://localhost:8000 --concurrency 5 --prompts prompts.txt --output results.csv
```

### Métricas Prometheus

```bash
vllm-mlx serve <model> --metrics
curl http://localhost:8000/metrics
```

## Instalación

**Usando uv (recomendado):**

```bash
uv tool install vllm-mlx                 # CLI a nivel sistema
# o dentro de un proyecto
uv pip install vllm-mlx
```

**Usando pip:**

```bash
pip install vllm-mlx

# Extras de audio
pip install vllm-mlx[audio]
brew install espeak-ng
python -m spacy download en_core_web_sm
```

**Desde el código fuente:**

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

Ver [Guía de instalación](docs/getting-started/installation.md) para todas las opciones.

## Documentación

- **Primeros pasos**: [Instalación](docs/getting-started/installation.md) · [Inicio rápido](docs/getting-started/quickstart.md)
- **Servidores y APIs**: [Servidor OpenAI](docs/guides/server.md) · [API Anthropic Messages](docs/guides/server.md#anthropic-messages-api) · [API Python](docs/guides/python-api.md)
- **Características**: [Multimodal](docs/guides/multimodal.md) · [Audio](docs/guides/audio.md) · [Embeddings](docs/guides/embeddings.md) · [Razonamiento](docs/guides/reasoning.md) · [MCP y Tool Calling](docs/guides/mcp-tools.md) · [Parsers de tools](docs/guides/tool-calling.md)
- **Rendimiento**: [Continuous Batching](docs/guides/continuous-batching.md) · [Warm Prompts](docs/guides/warm-prompts.md) · [MoE Top-K](docs/guides/moe-top-k.md)
- **Referencia**: [CLI](docs/reference/cli.md) · [Modelos](docs/reference/models.md) · [Configuración](docs/reference/configuration.md)
- **Benchmarks**: [LLM](docs/benchmarks/llm.md) · [Imagen](docs/benchmarks/image.md) · [Video](docs/benchmarks/video.md) · [Audio](docs/benchmarks/audio.md)

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Servidor vllm-mlx                               │
│   OpenAI /v1/*  ·  Anthropic /v1/messages  ·  /v1/rerank  ·  /metrics   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Continuous batching · Paged KV cache · Prefix cache · SSD tiering      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│    (LLMs)     │ │  (Visión)     │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MLX · kernels Metal · memoria unificada               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Contribuir

Bienvenidos bug fixes, trabajo de performance, docs y benchmarks en distintos chips de Apple Silicon. Ver [Guía de contribución](docs/development/contributing.md).

## Licencia

Apache 2.0. Ver [LICENSE](LICENSE).

## Citación

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title  = {vllm-mlx: Apple Silicon MLX Backend for vLLM},
  year   = {2025},
  url    = {https://github.com/waybarrios/vllm-mlx},
  note   = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Agradecimientos

- [MLX](https://github.com/ml-explore/mlx). Framework de ML de Apple.
- [mlx-lm](https://github.com/ml-explore/mlx-lm). Librería de inferencia de LLM.
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Modelos de visión y lenguaje.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio). Text-to-Speech y Speech-to-Text.
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings). Embeddings de texto.
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). Fork comunitario de vllm-mlx.
- [vLLM](https://github.com/vllm-project/vllm). Servicio de LLM de alto throughput. vllm-mlx está inspirado en vLLM y adopta su diseño de continuous-batching y paged KV-cache para Apple Silicon vía MLX.

## Historia de stars

[![Star History Chart](https://api.star-history.com/svg?repos=waybarrios/vllm-mlx&type=Date)](https://star-history.com/#waybarrios/vllm-mlx&Date)

---

**Si vllm-mlx te sirvió, por favor dale una star al repo. Ayuda a que más devs de Apple Silicon lo encuentren.**
