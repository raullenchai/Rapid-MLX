# Documentation vLLM-MLX

**Backend MLX pour Apple Silicon sur vLLM** - Accélération GPU pour le texte, les images, la vidéo et l'audio sur Mac

## Qu'est-ce que vLLM-MLX ?

vllm-mlx apporte l'accélération GPU native d'Apple Silicon à vLLM en intégrant :

- **[MLX](https://github.com/ml-explore/mlx)** : le framework ML d'Apple avec mémoire unifiée et noyaux Metal
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** : inférence LLM optimisée avec KV cache et quantification
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** : modèles vision-langage (VLM) pour l'inférence multimodale
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)** : Text-to-Speech et Speech-to-Text avec des voix natives
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)** : embeddings textuels pour la recherche sémantique et le RAG

## Fonctionnalités principales

- **Multimodal** - Texte, image, vidéo et audio sur une seule plateforme
- **Accélération GPU native** sur Apple Silicon (M1, M2, M3, M4, M5)
- **Voix TTS natives** - espagnol, français, chinois, japonais et 5 autres langues
- **Compatible API OpenAI** - remplacement direct du client OpenAI
- **Embeddings** - point de terminaison `/v1/embeddings` compatible OpenAI
- **MCP Tool Calling** - intégration d'outils externes via le Model Context Protocol
- **Paged KV Cache** - mise en cache efficace en mémoire avec partage de préfixe
- **Continuous Batching** - débit élevé pour plusieurs utilisateurs simultanés

## Liens rapides

### Démarrage

- [Installation](getting-started/installation.md)
- [Démarrage rapide](getting-started/quickstart.md)

### Guides utilisateur

- [Serveur compatible OpenAI](guides/server.md)
- [API Python](guides/python-api.md)
- [Multimodal (images et vidéo)](guides/multimodal.md)
- [Audio (STT/TTS)](guides/audio.md)
- [Embeddings](guides/embeddings.md)
- [Modèles de reasoning](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP et Tool Calling](guides/mcp-tools.md)
- [Continuous Batching](guides/continuous-batching.md)

### Référence

- [Commandes CLI](reference/cli.md)
- [Modèles pris en charge](reference/models.md)
- [Configuration](reference/configuration.md)

### Benchmarks

- [Benchmarks LLM](benchmarks/llm.md)
- [Benchmarks image](benchmarks/image.md)
- [Benchmarks vidéo](benchmarks/video.md)
- [Benchmarks audio](benchmarks/audio.md)

### Développement

- [Architecture](../development/architecture.md)
- [Contribuer](../development/contributing.md)

## Prérequis

- macOS sur Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10+
- 8 Go de RAM recommandés

## Licence

Apache 2.0 - Consultez [LICENSE](../../LICENSE) pour les détails.
