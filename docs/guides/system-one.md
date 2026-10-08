# System One decision server

Rapid-MLX can serve typed, low-latency decisions through a
[TypeSafe](https://typesafe-jev.com/en/guides/api/)-compatible API. The service
supports `noul`, `choice`, and `score` questions at `POST /v1/systemone`, plus
free-form candidate ranking at `POST /v1/rank`.

This is a dedicated service process. It does not load the OpenAI-compatible
chat server or expose chat completion routes.

## Laya-MLX

Laya is the fastest and smallest supported backend. Install the optional
runtime and start the service:

```bash
pip install 'rapid-mlx[system-one]'
rapid-mlx system-one convaiinnovations/laya --port 8700
```

`laya-mlx` requires Python 3.11 or newer. Rapid-MLX itself continues to support
Python 3.10; use the CLM backend or upgrade Python when running this service on
3.10.

## CLM-8B

CLM uses the original BF16 `Qwen/Qwen3-8B` backbone and two small contrastive
projection heads. Rapid-MLX runs both parts natively in MLX and caches projected
state and action vectors.

The upstream CLM head is distributed as a PyTorch `.pt` file. Convert it once
in an environment with PyTorch; the serving environment does not need PyTorch:

```bash
rapid-mlx-convert-clm-head \
  "$(hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt)" \
  ./clm-v0.1-mlx

rapid-mlx system-one clm-latest \
  --backend clm \
  --encoder Qwen/Qwen3-8B \
  --head ./clm-v0.1-mlx \
  --port 8700
```

Use `--device cpu` when GPU memory is reserved for another process. Device
selection happens before the head and encoder load; `gpu` remains the default.

The converter uses `torch.load(..., weights_only=True)` and writes
`model.safetensors` plus `config.json`. The server reads only those converted
files. Use the BF16 reference encoder for calibrated output. Quantized Qwen3
backbones have not been qualified for ranking or probability parity.

## Decider

Decider is a compact typed-decision model on a Qwen3.5 2B text backbone. It
reads the state and one question, then scores the answer labels at the last
position with the checkpoint's own per-type calibration. It does not generate
text. Rapid runs it on native MLX with the text backbone it already ships, so
no extra install is needed.

```bash
rapid-mlx system-one decider-2b --port 8700
# A converted or fine-tuned checkpoint directory:
rapid-mlx system-one /path/to/decider --backend decider --port 8700
```

The first start downloads the pinned `nativ-community/decider-2b` weights
(Apache-2.0, 3.8 GB, bf16) into the normal Hugging Face cache. A local
directory must be a prepared checkpoint: its root `config.json` says
`model_type: "decider2"` and carries the published calibration under
`decision_config`. Decider is text-only; requests with `images` or `videos`
are rejected.

On one M3 Ultra Mac the server was ready 5 seconds after launch with the
weights cached, and held about 4.3 GB of memory. Four questions over the same
state took 0.2 s at 200 input tokens, 1.1 s at 1,300, 4.9 s at 5,200 and 22 s
at 21,000. Other Mac sizes are unmeasured.

## Cloudflare Clef

Clef is a joint-schema decision model. `clef-flash` uses a 9B Qwen3.5
backbone; `clef` uses a 27B Qwen3.8 backbone. Both score every allowed option
through Cloudflare's trained joint head in one forward pass. They do not
generate chat text. Rapid uses the official Apache-2.0 head implementation
with Torch on Apple Metal/MPS for this backend; Laya and CLM remain native MLX.

```bash
pip install 'rapid-mlx[clef]'
rapid-mlx system-one clef-flash --port 8700
# Larger model on a high-memory Mac:
rapid-mlx system-one clef --port 8700
```

The first start downloads pinned Cloudflare weights into the normal Hugging
Face cache. `--device cpu` is available for diagnosis. Cloudflare validated
its reference runtime on an H200. Both checkpoints have been dogfooded on one
M3 Ultra Mac with Metal; see the [local measurements](../engineering/performance/2026-10-03-clef-family-m3-ultra-dogfood.md).
Other Mac sizes and sustained concurrent traffic remain unqualified. A Clef
request accepts `state`, `questions`, and optional `images` or `videos`; the
same `/v1/rank` endpoint ranks free-form candidates. `model` may be the short
name or the corresponding `Cloudflare/...` identifier.

For media, send `images` as PNG/JPEG/WebP base64 data URLs. Send `videos` as
arrays of frame data URLs. Remote URLs and filesystem paths are rejected;
each image is capped at 4 MiB and 16 MP, with at most eight images or 32
video frames per request, with at least two frames in each video. All images
and frames together are capped at 16 MP of decoded pixels. The complete JSON
request also has an 8 MiB body limit, including base64 overhead; the per-item
and item-count maxima cannot all be reached in one request. Requests over the
body limit receive HTTP 413. Example:

```json
{
  "state": {"task": "Review the attached receipt"},
  "images": ["data:image/png;base64,<base64 bytes>"],
  "questions": {
    "legible": {"type": "noul", "instructions": "Is the total legible?"}
  }
}
```

## Request examples

```bash
curl http://127.0.0.1:8700/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"customer": "My invoice was charged twice"},
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Does this need urgent handling?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Charges, invoices, and refunds",
          "technical": "Bugs and outages"
        }
      }
    }
  }'
```

Rank candidates directly:

```bash
curl http://127.0.0.1:8700/v1/rank \
  -H 'Content-Type: application/json' \
  -d '{
    "context": "What causes tides on Earth?",
    "answers": [
      "The Moon gravitationally pulls on the oceans.",
      "Photosynthesis in plants.",
      "The Earth is round."
    ]
  }'
```

Set `RAPID_MLX_API_KEY` or pass `--api-key` to protect `/v1/models`,
`/v1/systemone`, and `/v1/rank`. `/health` stays unauthenticated for local
service probes. Requests share the main server's 8 MiB body limit and JSON
nesting-depth protection.

## Compatibility limits

- One service process hosts one decision backend.
- Laya uses checkpoint calibration and accepts `temperature=1` only.
- Decider requires `temperature=1` too. It accepts `choice` questions with
  2 to 255 options and `score` questions with 2 to 10 levels; other shapes
  receive HTTP 422. Each `score` level is judged on its own and the fits are
  normalized, as the checkpoint was calibrated. State longer than 32,768
  tokens is truncated. Every question is one full forward pass over the state,
  so latency grows with both state length and question count.
- Clef also requires `temperature=1` for checkpoint calibration. Its input
  encoder follows Cloudflare's 16,384-token default and may truncate a long
  state to leave room for the schema.
- CLM input is capped at 2,048 tokens by default, matching the upstream
  vLLM `truncate_prompt_tokens` behavior: longer inputs are left-truncated so
  the final 2,048 tokens reach last-token pooling. `--max-tokens` can lower or
  raise the cap, but parity above 2,048 tokens has not been qualified. Each
  rendered encoder input is also capped at 16 UTF-8 bytes per configured token
  (at least 1 KiB), before tokenization.
- Projection temperature scaling matches upstream `HeadPair`: the exponential
  of `logit_scale` is capped at 100.
- A request may contain at most 64 questions and 255 options per question.
  Across all questions it may contain at most 255 candidates and 32,768 CLM
  encoder tokens. `--max-work-tokens` changes the token budget.
- At most eight backend requests may be outstanding. CLM executes them serially
  for MLX safety. `--max-concurrent-requests` changes the admission limit;
  excess requests receive `503` with `Retry-After`.
- Request temperature must be between `1e-6` and `100` to keep CLM logits finite.
- The current English Laya checkpoint triggers an upstream `laya-mlx` clamp for
  the `choice:11+` calibration bucket. Treat confidence from questions with
  eleven or more choices as uncalibrated; the selected choice and probability
  distribution remain available.
