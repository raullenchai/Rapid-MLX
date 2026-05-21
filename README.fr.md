# vllm-mlx

**Lire ceci dans d'autres langues :** [English](README.md) · [Español](README.es.md) · [Français](README.fr.md) · [中文](README.zh.md)

**Continuous batching + API OpenAI et Anthropic dans un seul serveur. Inférence native sur Apple Silicon.**

[![PyPI version](https://img.shields.io/pypi/v/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub stars](https://img.shields.io/github/stars/waybarrios/vllm-mlx.svg?style=social)](https://github.com/waybarrios/vllm-mlx)

---

## Qu'est-ce que vllm-mlx ?

Un serveur d'inférence de type vLLM pour les Macs Apple Silicon. Contrairement à l'utilisation directe de `Ollama` ou `mlx-lm`, il embarque **continuous batching, paged KV cache, prefix caching et cache KV sur SSD**, et expose **à la fois OpenAI `/v1/*` et Anthropic `/v1/messages`** depuis un seul processus. Exécutez des LLMs, modèles de vision, audio et embeddings sur Metal avec mémoire unifiée, sans étape de conversion.

## Démarrage rapide (30 secondes)

```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

**SDK OpenAI :**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Salut !"}])
print(r.choices[0].message.content)
```

**SDK Anthropic / Claude Code :**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

## Fonctionnalités

### APIs
- **Compatible OpenAI** : `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/responses`
- **Compatible Anthropic** : `/v1/messages` (streaming, tool use, system prompts)
- **MCP Tool Calling** : 12 parsers (OpenAI, Anthropic, Gemini, Qwen, DeepSeek, Gemma et plus)
- **Sortie structurée** : JSON Schema via `response_format` (lm-format-enforcer)

### Débit et mémoire
- **Continuous batching** : haut débit pour des requêtes concurrentes
- **Paged KV cache** : efficace en mémoire avec prefix sharing
- **Cache KV sur SSD** : déversez le prefix cache sur disque pour les agents à long contexte (`--ssd-cache-dir`)
- **Warm prompts** : pré-chargez les préfixes populaires au démarrage (`--warm-prompts`) pour un TTFT 1.3-2.25x
- **Prefix cache** : basé sur trie, partagé entre les requêtes

### Multimodal
- **Texte + image + vidéo + audio** depuis un seul serveur
- Modèles de vision : Gemma 3, Gemma 4, Qwen3-VL, Pixtral, Llama vision
- **Audio en entrée** dans le chat (blocs `audio_url`)
- **TTS natif** : 11 voix, 15+ langues (Kokoro, Chatterbox, VibeVoice, VoxCPM)
- **STT** : famille Whisper avec RTF jusqu'à 197x sur M4 Max

### Raisonnement et avancé
- **Extraction du raisonnement** : Qwen3, DeepSeek-R1 (`--reasoning-parser`)
- **Réduction d'experts MoE** : `--moe-top-k` pour +7-16% sur Qwen3-30B-A3B
- **Décodage spéculatif** : `--mtp` pour Qwen3-Next
- **Prefill creux** : `--spec-prefill` basé sur l'attention pour réduire le TTFT

### Observabilité
- **Métriques Prometheus** : endpoint `/metrics` avec `--metrics`
- **Benchmark intégré** : `vllm-mlx bench-serve` pour des balayages de prompts en CSV/JSON

### Accélération GPU native
- Apple Silicon uniquement (M1, M2, M3, M4, M5) avec kernels Metal via MLX
- Mémoire unifiée, sans conversion de modèles

## Performance

**Decode LLM (M4 Max, 128 Go, greedy, single stream) :**

| Modèle | Tok/s | Mémoire |
|--------|------:|--------:|
| Qwen3-0.6B-8bit | 417.9 | 0.7 Go |
| Llama-3.2-3B-Instruct-4bit | 205.6 | 1.8 Go |
| Qwen3-30B-A3B-4bit | 127.7 | ~18 Go |

**Audio speech-to-text (M4 Max, RTF = real-time factor) :**

| Modèle | RTF | Cas d'usage |
|--------|----:|-------------|
| whisper-tiny | 197x | Temps réel / faible latence |
| whisper-large-v3-turbo | 55x | Qualité + vitesse |
| whisper-large-v3 | 24x | Précision maximale |

Voir [docs/benchmarks/](docs/benchmarks/) pour les résultats continuous batching, la quantification du KV cache (4-bit / 8-bit / fp16) et les balayages MoE top-k.

## Exemples

### API Anthropic (Claude Code, OpenCode)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Modèles de raisonnement (Qwen3, DeepSeek-R1)

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Combien font 17 * 23 ?"}],
)
print("Raisonnement :", r.choices[0].message.reasoning)
print("Réponse :",      r.choices[0].message.content)
```

### Multimodal (image + texte)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Qu'y a-t-il sur cette image ?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}],
)
```

### Sortie structurée (JSON Schema)

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Liste 3 couleurs."}],
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
  "query": "inférence apple silicon",
  "documents": ["MLX est le framework d Apple", "Kernels Metal sur M-series", "CUDA sur NVIDIA"]
}'
```

Le chemin de reranking MLX intégré prend en charge les poids de classification
de séquence BERT/XLM-RoBERTa standard avec les valeurs `hidden_act` `gelu`,
`gelu_new`/`gelu_fast`, `relu` ou `silu`/`swish`. Les autres activations échouent
explicitement afin que les architectures personnalisées ajoutent un adaptateur
dédié au lieu d'utiliser GELU silencieusement.

### Embeddings

```bash
vllm-mlx serve <llm-model> --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
emb = client.embeddings.create(model="mlx-community/all-MiniLM-L6-v2-4bit", input=["Bonjour", "Monde"])
```

### Audio (TTS / STT)

```bash
pip install vllm-mlx[audio]
brew install espeak-ng        # macOS, nécessaire pour TTS non-anglais

python examples/tts_example.py "Hello, how are you?" --play
python examples/tts_multilingual.py "Bonjour le monde" --lang fr --play
```

### Benchmarking intégré

```bash
vllm-mlx bench-serve --url http://localhost:8000 --concurrency 5 --prompts prompts.txt --output results.csv
```

### Métriques Prometheus

```bash
vllm-mlx serve <model> --metrics
curl http://localhost:8000/metrics
```

## Installation

**Avec uv (recommandé) :**

```bash
uv tool install vllm-mlx                 # CLI global
# ou dans un projet
uv pip install vllm-mlx
```

**Avec pip :**

```bash
pip install vllm-mlx

# Extras audio
pip install vllm-mlx[audio]
brew install espeak-ng
python -m spacy download en_core_web_sm
```

**Depuis les sources :**

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

Voir le [Guide d'installation](docs/getting-started/installation.md) pour toutes les options.

## Documentation

- **Premiers pas** : [Installation](docs/getting-started/installation.md) · [Démarrage rapide](docs/getting-started/quickstart.md)
- **Serveurs et APIs** : [Serveur OpenAI](docs/guides/server.md) · [API Anthropic Messages](docs/guides/server.md#anthropic-messages-api) · [API Python](docs/guides/python-api.md)
- **Fonctionnalités** : [Multimodal](docs/guides/multimodal.md) · [Audio](docs/guides/audio.md) · [Embeddings](docs/guides/embeddings.md) · [Raisonnement](docs/guides/reasoning.md) · [MCP et Tool Calling](docs/guides/mcp-tools.md) · [Parsers de tools](docs/guides/tool-calling.md)
- **Performance** : [Continuous Batching](docs/guides/continuous-batching.md) · [Warm Prompts](docs/guides/warm-prompts.md) · [MoE Top-K](docs/guides/moe-top-k.md)
- **Référence** : [CLI](docs/reference/cli.md) · [Modèles](docs/reference/models.md) · [Configuration](docs/reference/configuration.md)
- **Benchmarks** : [LLM](docs/benchmarks/llm.md) · [Image](docs/benchmarks/image.md) · [Vidéo](docs/benchmarks/video.md) · [Audio](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Serveur vllm-mlx                                │
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
│    (LLMs)     │ │  (Vision)     │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MLX · kernels Metal · mémoire unifiée                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Contribuer

Les corrections de bugs, travaux de performance, docs et benchmarks sur différentes puces Apple Silicon sont bienvenus. Voir le [Guide de contribution](docs/development/contributing.md).

## Licence

Apache 2.0. Voir [LICENSE](LICENSE).

## Citation

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title  = {vllm-mlx: Apple Silicon MLX Backend for vLLM},
  year   = {2025},
  url    = {https://github.com/waybarrios/vllm-mlx},
  note   = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Remerciements

- [MLX](https://github.com/ml-explore/mlx). Framework ML d'Apple.
- [mlx-lm](https://github.com/ml-explore/mlx-lm). Bibliothèque d'inférence LLM.
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Modèles vision-langage.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio). Text-to-Speech et Speech-to-Text.
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings). Embeddings de texte.
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). Fork communautaire de vllm-mlx.
- [vLLM](https://github.com/vllm-project/vllm). Service LLM haut débit. vllm-mlx s'inspire de vLLM et adopte sa conception continuous-batching et paged KV-cache pour Apple Silicon via MLX.

## Historique des stars

[![Star History Chart](https://api.star-history.com/svg?repos=waybarrios/vllm-mlx&type=Date)](https://star-history.com/#waybarrios/vllm-mlx&Date)

---

**Si vllm-mlx vous a été utile, mettez une étoile au repo. Cela aide plus de développeurs Apple Silicon à le trouver.**
