# Embeddings

rapid-mlx provides an OpenAI-compatible `/v1/embeddings` endpoint with a native
EmbeddingGemma 2 text/code encoder and the existing optional embedding backend.

## Installation

The `/v1/embeddings` surface ships behind the `[embeddings]` extra (mirrors how `[audio]` is packaged) — base installs of rapid-mlx do **not** bundle `mlx-embeddings`. Install the extra alongside the base package:

```bash
pip install 'rapid-mlx[embeddings]'
```

If you boot `rapid-mlx serve --embedding-model …` without the extra installed, the CLI exits cleanly with the same install hint (no `ModuleNotFoundError` traceback).

EmbeddingGemma 2 uses the native vision runtime (mlx-vlm) instead of
`mlx-embeddings`; both the base Python install and the packaged Desktop engine
already include that runtime. Existing
embedding models still require `[embeddings]`. Do not replace the pinned
vision runtime with an unvalidated upstream development build.

## EmbeddingGemma 2: text and code

`embeddinggemma-2-bf16` and `embeddinggemma-2-4bit` select
`mlx-community/embeddinggemma-2-bf16` and
`mlx-community/embeddinggemma-2-4bit`. The native loader also accepts the
official `google/embeddinggemma-2` checkpoint or a local directory with
`model_type: embedding_gemma2`. This is a new bidirectional encoder with its
trained projection and normalized 768-dimensional output; it is separate
from the older 300M model.

```bash
rapid-mlx serve <your-chat-model> --embedding-model embeddinggemma-2-4bit
```

This release serves **text/code only**. Images, audio, video and interleaved
media are not accepted by this endpoint. Media objects/fields are rejected by
the request schema; media control tokens in text or token-ID inputs return
400 with `unsupported_embedding_modality`. The vision/audio towers are not
loaded. The model is an embedding backend, not a chat model.

The supported input window is **8192 tokens**, including special tokens and
any task prefix. `--embedding-max-length auto` uses that limit; larger values
are clamped. Existing overflow policies (`truncate` with a warning/metric,
or `error` with a structured 400) remain available. Floating activations stay
BF16 or FP32, never FP16. Standard affine 4-bit weights reduce memory with a
quality tradeoff; other quantization families are not supported by this adapter.

Callers supply the appropriate task prefix; the endpoint does not guess
whether an input is a query or a document. For retrieval, use
`task: search result | query: ...` and `title: none | text: ...` (replace
`none` with a real title when available). For code queries, use
`task: code retrieval | query: ...` and a document title such as a filename.

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2-4bit","input":["task: search result | query: Which planet is the Red Planet?","title: none | text: Mars is known as the Red Planet."],"dimensions":256,"encoding_format":"float"}'
```

For a keyed server, also send its configured `Authorization: Bearer` header.
The default dimension is 768; Matryoshka prefixes of 128, 256 or 512 are
recommended. The existing endpoint truncates and re-normalizes requested
dimensions, supports float or base64 output, and accepts pre-tokenized
inputs. Queries and documents must use the same dimension.

## Quick Start

### Start the server with an embedding model

```bash
# Pre-load a specific embedding model at startup
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit

# Embeddings-only: no chat model is loaded or downloaded
rapid-mlx serve --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

The embeddings-only form suits dedicated embedding replicas behind a load balancer. Chat and completion routes return `503` on such a server, flags that only apply to a chat model (for example `--served-model-name` or `--lazy-load`) are rejected at startup, and the health probes report `model_loaded: true` once the embedding backend is resident. The primary-model name remains `null`; `/v1/models` identifies the embedding model.

`--embedding-model` is **required** to enable the `/v1/embeddings` endpoint. Without it, every `POST /v1/embeddings` request returns `503 Service Unavailable` with `code: "no_embedding_model"` — the server will NOT silently re-route the request to the chat model (which would produce shape-valid but semantically meaningless vectors).

### Generate embeddings with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# Batch of texts
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=[
        "I love machine learning",
        "Deep learning is fascinating",
        "Natural language processing rocks"
    ]
)
for item in response.data:
    print(f"Text {item.index}: {len(item.embedding)} dimensions")
```

### Using curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Supported Models

Any BERT, XLM-RoBERTa, or ModernBERT model from HuggingFace that is compatible with mlx-embeddings:

| Model | Use Case | Size |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Fast, compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | High quality | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Best for English | Large |

## Model Management

### Pre-loading at startup (required)

`--embedding-model` pins the embedding engine to one model id at boot. The flag is REQUIRED to enable `/v1/embeddings` — there is no hot-swap and no lazy-load fallback. Requests sent without the flag configured return `503` with `error.code = "no_embedding_model"`.

```bash
rapid-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Once locked, requesting a *different* model id on the wire returns a `400` with `error.code = "model_not_found"`. The locked model id appears in `GET /v1/models` alongside the chat model, with `capabilities: ["embedding"]` so `client.models.list()` auto-discovery works for LangChain / LlamaIndex / openai-python.

## API Reference

### POST /v1/embeddings

Create embeddings for the given input text(s).

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name from HuggingFace |
| `input` | string or list[string] | Yes | Text(s) to embed |

**Response:**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]}
  ],
  "model": "mlx-community/all-MiniLM-L6-v2-4bit",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## Python API

### Direct usage without server

```python
from rapid_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## Troubleshooting

### mlx-embeddings not installed

```
pip install mlx-embeddings>=0.0.5
```

### Model not found

Make sure the model name matches a HuggingFace repository compatible with mlx-embeddings. You can pre-download models:

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
