# Installation

## Prérequis

- macOS sur Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10+

## Installation avec uv (recommandée)

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

uv pip install -e .
```

## Installation avec pip

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

pip install -e .
```

### Optionnel : support vidéo

Pour le traitement vidéo avec transformers :

```bash
pip install -e ".[vision]"
```

### Optionnel : support audio (STT/TTS)

```bash
pip install mlx-audio
```

### Optionnel : embeddings

```bash
pip install mlx-embeddings
```

## Ce qui est installé

- `mlx`, `mlx-lm`, `mlx-vlm` - framework MLX et bibliothèques de modèles
- `transformers`, `tokenizers` - bibliothèques HuggingFace
- `opencv-python` - traitement vidéo
- `gradio` - interface de chat
- `psutil` - surveillance des ressources
- `mlx-audio` (optionnel) - Speech-to-Text et Text-to-Speech
- `mlx-embeddings` (optionnel) - embeddings de texte

## Vérifier l'installation

```bash
# Check CLI commands
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Test with a small model
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## Dépannage

### MLX introuvable

Vérifiez que vous êtes sur Apple Silicon :
```bash
uname -m  # Should output "arm64"
```

### Échec du téléchargement du modèle

Vérifiez votre connexion internet et vos accès HuggingFace. Certains modèles nécessitent une authentification :
```bash
huggingface-cli login
```

### Mémoire insuffisante

Utilisez un modèle quantifié plus petit :
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```
