# NeoHorse 1 9B Chat qualification

Date: 2026-09-08

This note records the evidence used to introduce `neohorse-9b-4bit` as an
experimental model. It is not evidence for changing the default model.

## Artifacts and environment

- Machine: Apple M3 Ultra, 256 GB unified memory
- macOS: 26.5.2 (25F84)
- Python: 3.11
- MLX: 0.32.2
- mlx-lm: 0.31.3
- NeoHorse source: `TokenRhythm/NeoHorse-1-9B` at
  `6cd9248d8070d8a0ad8d20aa19e2fe6848419e93`
- Published MLX artifact: `rapid-mlx/NeoHorse-1-9B-MLX-4bit` at
  `9fe3cd3f69e2d653e82ec094e1c4d0caa9564897`
- Baseline: `mlx-community/Qwen3.5-9B-4bit` at
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`
- Quantization: affine 4-bit, group size 64

## Results

The fixed local prompt set scored reasoning, general Chat, coding, and
first-action tool selection separately:

| Model | Reasoning | General Chat | Coding | Tool first action |
|---|---:|---:|---:|---:|
| Qwen3.5 9B 4-bit | 8/10 | 6/10 | 8/10 | 25/30 |
| NeoHorse 1 9B 4-bit | 8/10 | 7/10 | 8/10 | 28/30 |

NeoHorse was better at parallel tool selection and clarifying an ambiguous
request. It also invented specific project names in one factuality probe.
That regression is why the model remains experimental.

A single 256-token speed smoke on the same machine measured 115.2 tokens/s
for NeoHorse and 112.4 tokens/s for the baseline, with reported peak memory
of 5.25 GB and 6.75 GB respectively. This is a smoke result, not a general
performance claim: it is one prompt, one run, and one machine.

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
