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
