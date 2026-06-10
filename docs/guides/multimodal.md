# Multimodal Models (Images & Video)

rapid-mlx supports vision-language models for image and video understanding.

## Supported Models

- Qwen3-VL (recommended)
- Qwen2-VL
- Gemma 3
- LLaVA
- Idefics
- PaliGemma
- Pixtral
- Molmo
- DeepSeek-VL

## Starting a Multimodal Server

```bash
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

Models with "VL", "Vision", or "mllm" in the name are auto-detected as multimodal.

## Image Analysis

### Via OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Image from URL
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)
```

### Base64 Images

```python
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("photo.jpg")
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
```

### Via curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 256
  }'
```

## Video Analysis

### Via OpenAI SDK

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }],
    max_tokens=512
)
```

### Video Parameters

Control frame extraction via extra body parameters:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video"},
            {"type": "video_url", "video_url": {"url": "video.mp4"}}
        ]
    }],
    extra_body={
        "video_fps": 2.0,
        "video_max_frames": 32
    }
)
```

### Via curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
      ]
    }],
    "video_fps": 2.0,
    "video_max_frames": 16
  }'
```

## Supported Formats

### Images

| Format | Example |
|--------|---------|
| URL | `{"type": "image_url", "image_url": {"url": "https://..."}}` |
| Local file | `{"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}` |
| Base64 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |

### Videos

| Format | Example |
|--------|---------|
| URL | `{"type": "video_url", "video_url": {"url": "https://..."}}` |
| Local file | `{"type": "video", "video": "/path/to/video.mp4"}` |
| Base64 | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

## Python API

```python
from vllm_mlx.models import MLXMultimodalLM

mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Image
description = mllm.describe_image("photo.jpg")

# Video
description = mllm.describe_video("video.mp4", fps=2.0)

# Custom prompt
output = mllm.generate(
    prompt="Compare these images",
    images=["img1.jpg", "img2.jpg"]
)
```

## Performance Tips

### Images
- Smaller resolutions process faster (224x224 vs 1920x1080)
- Use appropriate resolution for your task

### Videos
- Lower FPS = faster processing
- Fewer frames = less memory usage
- 64 frames is practical maximum (96+ causes GPU timeout)

## Benchmarks

Tested on Apple M4 Max with 128 GB unified memory.

### Qwen3-VL-4B-Instruct-3bit

| Resolution | Time | Tokens | Speed | Memory |
|------------|------|--------|-------|--------|
| 224x224 | 0.87s | 124 | 143 tok/s | 2.6 GB |
| 448x448 | 1.01s | 107 | 106 tok/s | 3.1 GB |
| 768x768 | 1.42s | 127 | 89 tok/s | 3.4 GB |
| 1024x1024 | 1.85s | 116 | 63 tok/s | 3.6 GB |

### Qwen3-VL-8B-Instruct-4bit

| Resolution | Time | Tokens | Speed | Memory |
|------------|------|--------|-------|--------|
| 224x224 | 1.08s | 78 | 73 tok/s | 5.6 GB |
| 448x448 | 1.41s | 70 | 50 tok/s | 6.1 GB |
| 768x768 | 2.06s | 91 | 44 tok/s | 6.5 GB |
| 1024x1024 | 3.02s | 76 | 25 tok/s | 7.6 GB |

### Gemma 3 4B 4bit

| Resolution | Time | Tokens | Speed | Memory |
|------------|------|--------|-------|--------|
| 224x224 | 0.95s | 30 | 32 tok/s | 5.2 GB |
| 448x448 | 0.99s | 34 | 34 tok/s | 5.2 GB |
| 768x768 | 0.99s | 32 | 32 tok/s | 5.2 GB |
| 1024x1024 | 0.95s | 28 | 29 tok/s | 5.2 GB |

### Running Benchmarks

The shipped `rapid-mlx bench` is text-only. Multi-resolution image and video
benches live in the dev-only `scripts/` directory (source checkout only). For
a quick text-only sanity bench against a VLM, you can still run:

```bash
rapid-mlx bench qwen3-vl-4b-4bit
```

## MLLM Cache

rapid-mlx includes a prefix cache system for multimodal models that can significantly speed up repeated requests with the same images.

### How It Works

When you send an image to the model, the vision encoder processes it into embeddings. This processing takes 1-2 seconds. The MLLM cache stores these embeddings along with the KV cache state, so subsequent requests with the same image skip the vision encoder entirely.

The cache uses content-based hashing (similar to LMCache) to identify identical images regardless of how they're provided (URL, base64, or file path).

### Enabling the Cache

```bash
# Enable with default settings (512 MB max)
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --enable-mllm-cache

# With custom memory limit
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit \
    --enable-mllm-cache \
    --mllm-cache-max-mb 1024
```

### Python API

```python
from vllm_mlx.mllm_cache import MLLMPrefixCacheManager

# Create cache manager
cache = MLLMPrefixCacheManager(max_memory_mb=512)

# Store embeddings and KV cache after processing
cache.store(
    images=["photo.jpg"],
    prompt="Describe this image",
    vision_embeddings=embeddings,
    kv_cache=kv_state,
    num_tokens=128
)

# Fetch from cache on subsequent requests
entry, match_len = cache.fetch(images=["photo.jpg"], prompt="Describe this image")
if entry:
    # Use cached embeddings, skip vision encoder
    embeddings = entry.vision_embeddings
    kv_state = entry.kv_cache
```

### Cache Statistics

```python
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Memory used: {stats.memory_used_mb:.1f} MB")
print(f"Tokens saved: {stats.tokens_saved}")
```

### Memory Management

The cache uses LRU (Least Recently Used) eviction when memory limit is reached. Each entry tracks:

- Vision embeddings size
- KV cache size per layer
- Access frequency for LRU ordering

When memory pressure occurs, least recently accessed entries are evicted first.

## Gradio Chat UI

For an interactive multimodal session, start a server and use any OpenAI-
compatible web UI (Open WebUI, LibreChat, etc.) pointed at it:

```bash
rapid-mlx serve qwen3-vl-4b-4bit --mllm --port 8000
```

The shipped `rapid-mlx chat` REPL is text-only. The optional Gradio web UI
(`pip install 'rapid-mlx[chat]'`) supports image / video uploads.
