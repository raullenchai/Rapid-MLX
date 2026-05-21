# Instalacion

## Requisitos

- macOS en Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10+

## Instalar con uv (Recomendado)

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

uv pip install -e .
```

## Instalar con pip

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

pip install -e .
```

### Opcional: Soporte para vision

Para procesamiento de video con transformers:

```bash
pip install -e ".[vision]"
```

### Opcional: Soporte de audio (STT/TTS)

```bash
pip install mlx-audio
```

### Opcional: Embeddings

```bash
pip install mlx-embeddings
```

## Que se instala

- `mlx`, `mlx-lm`, `mlx-vlm` - Framework MLX y bibliotecas de modelos
- `transformers`, `tokenizers` - Bibliotecas de HuggingFace
- `opencv-python` - Procesamiento de video
- `gradio` - Interfaz de chat
- `psutil` - Monitoreo de recursos
- `mlx-audio` (opcional) - Speech-to-Text y Text-to-Speech
- `mlx-embeddings` (opcional) - Text embeddings

## Verificar la instalacion

```bash
# Verificar comandos CLI
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Probar con un modelo pequeño
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## Solucion de problemas

### MLX no encontrado

Asegurate de estar en Apple Silicon:
```bash
uname -m  # Should output "arm64"
```

### Fallo en la descarga del modelo

Verifica tu conexion a internet y el acceso a HuggingFace. Algunos modelos requieren autenticacion:
```bash
huggingface-cli login
```

### Sin memoria

Usa un modelo cuantizado más pequeno:
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```
