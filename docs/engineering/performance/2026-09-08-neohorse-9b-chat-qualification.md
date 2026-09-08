# NeoHorse 1 9B Chat qualification

Date: 2026-09-08

This note records the evidence used to introduce `neohorse-9b-4bit` as an
experimental model. It is not evidence for changing the default model.

## Artifacts and environment

- Machine: Apple M3 Ultra, 256 GB unified memory
- macOS: 26.5.2 (25F84)
- Python: 3.12.13
- MLX: 0.32.2
- mlx-lm: 0.31.3
- NeoHorse source: `TokenRhythm/NeoHorse-1-9B` at
  `6cd9248d8070d8a0ad8d20aa19e2fe6848419e93`
- Published MLX artifact: `rapid-mlx/NeoHorse-1-9B-MLX-4bit` at
  `9fe3cd3f69e2d653e82ec094e1c4d0caa9564897`
- Quantization: affine 4-bit, group size 64
- Rapid-MLX revision under test: `525b62ac3`

## Results

The checked-in product-path evaluator produced these results:

| Suite | Result |
|---|---:|
| Tool calling, including parallel, sequential, and recovery cases | 24/31 (77%) |
| Executable coding tasks | 6/10 (60%) |
| Deterministically graded reasoning | 6/10 (60%) |
| General knowledge and instruction following | 7/10 (70%) |

The failures include incomplete parallel tool batches, missed later steps in
sequential tool workflows, one failed recovery case, four coding tasks with
runtime failures, and four incorrect reasoning answers. These results support
an experimental listing, but do not support replacing the default 9B model.

The same run measured 165 ms cold TTFT, 96 ms warm TTFT, 44.5 tokens/s for its
short decode probe, 113.9 tokens/s for its 500-token decode probe, 5.5 GB active
RAM, and 5.7 GB peak RAM. These are single-machine qualification measurements,
not cross-hardware performance claims.

## Reproduction

The prompts and graders are versioned in `evals/prompts/{tool_calling,coding,
reasoning,general}.json` and `evals/run_eval.py`. Tool and reasoning suites use
deterministic structural/answer graders; coding outputs are executed against
the checked-in task tests; general answers use the task-specific deterministic
checks. All requests use temperature 0. Tool calls use at most 512 output
tokens, coding 4,096, reasoning 1,024, and general 2,048.

Resolve the immutable Hub revision into the standard Hugging Face cache, then
serve that resolved snapshot path. The `--model-path` evaluator argument is
provenance metadata; pinning the server input is what guarantees identical
weights:

```bash
NEOHORSE_SNAPSHOT="$(python -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download("rapid-mlx/NeoHorse-1-9B-MLX-4bit", revision="9fe3cd3f69e2d653e82ec094e1c4d0caa9564897"))')"

rapid-mlx serve "$NEOHORSE_SNAPSHOT" \
  --host 127.0.0.1 --port 8327 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3

python evals/run_eval.py \
  --model NeoHorse-1-9B-MLX-4bit \
  --host 127.0.0.1 --port 8327 \
  --parser hermes --quantization 4bit \
  --suite speed tool_calling coding reasoning general \
  --output /private/tmp/rapid-mlx-neohorse9-standard-eval.json \
  --hardware 'Apple M3 Ultra (256 GB)' \
  --server-flags \
    'rapid-mlx serve <resolved pinned snapshot> --host 127.0.0.1 --port 8327 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3' \
  --model-path \
    'rapid-mlx/NeoHorse-1-9B-MLX-4bit@9fe3cd3f69e2d653e82ec094e1c4d0caa9564897' \
  --engine batched
```

An earlier direct-loader screen motivated this experimental integration, but it
used a separate prompt harness and is intentionally not used as promotion
evidence here. A default decision requires both candidates to be compared with
this checked-in evaluator and broader user journeys.

## Artifact checks

- The published artifact was downloaded again at its exact commit and loaded
  through the standard `qwen3_5` MLX loader.
- A stop-token smoke returned `REMOTE READY` without leaking `<|im_end|>`.
- `model.safetensors` SHA-256:
  `19defa62f104fc4dcea1c2352d5dfa1ce259a4f6868eeab39f5d98f3886e9a26`
- The Rapid-MLX server exposed the alias as text-only and experimental, and
  completed non-streaming Chat, SSE streaming through `[DONE]`, a Hermes tool
  call, tool-result replay, and separated Qwen-style reasoning/content without
  leaking `<|im_end|>`.

## Promotion gate

Before considering a default change, run broader multi-turn factuality,
streaming, tool-result replay, long-context, and Desktop journeys on the
published artifact. A default change must be a separate reviewable change.
