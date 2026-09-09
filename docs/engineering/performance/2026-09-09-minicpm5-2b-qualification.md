# MiniCPM5 2B MLX qualification — 2026-09-09

## Decision

Register the official `openbmb/MiniCPM5-2B-MLX` artifact as
`minicpm5-2b-4bit`. It fills the compact, latency-sensitive local-agent tier.
It does not replace the existing Smart defaults: the measured speed advantage
comes with a smaller but visible tool-selection reliability gap.

## Artifact and compatibility

- Checkpoint: `openbmb/MiniCPM5-2B-MLX`, revision
  `35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3`
- Download footprint: 1,425,999,742 bytes
- Quantization: 4-bit affine, group size 64
- Architecture: dense `LlamaForCausalLM`, 42 layers, 2,516,756,480 total
  parameters
- Context declaration: 131,072 tokens
- Tool wire format: MiniCPM native XML (`minicpm` parser)
- Reasoning wire format: `<think>...</think>` (`qwen3` parser)

No new model implementation or parser was required. The primary server
precedents load the same standard architecture without a model-code fork, and
the MLX-native release loads through the existing llama implementation. The
published draft model uses a separate speculative method; this qualification
does not infer compatibility with Rapid's current speculative paths.

## Environment

- Hardware: Mac Studio, M3 Ultra, 256 GB unified memory
- OS/date: macOS, 2026-09-09
- Rapid revision: `ef71b3484f009987cb37a28682ed52420cf00e03`
- Engine: continuous batching, `max_num_seqs=2`
- Server flags: `--enable-auto-tool-choice --tool-call-parser minicpm
  --reasoning-parser qwen3`
- Comparison artifact: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Comparison parser: `hermes`
- Evaluation temperature: 0 for the tool-calling suite

Both checkpoints were already warm before the timed suite. The evaluator
cleared the server prefix cache before each model run.

## Reproduction

Start each server on the same port, then run:

```bash
python evals/run_eval.py \
  --model minicpm5-2b \
  --host 127.0.0.1 --port 18092 \
  --parser minicpm --quantization 4bit \
  --suite tool_calling \
  --hardware 'Mac Studio M3 Ultra 256GB' \
  --server-flags '--enable-auto-tool-choice --tool-call-parser minicpm --reasoning-parser qwen3' \
  --model-path openbmb/MiniCPM5-2B-MLX
```

Repeat with the comparison checkpoint and its `hermes` parser.

## Results

| Official MLX 4-bit artifact | Correct | Summed request time | Median request | Slowest request |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM5 2B | 24/31 (77%) | 7.17 s | 0.20 s | 0.58 s |
| Qwen3.5 4B | 26/31 (84%) | 32.54 s | 0.99 s | 5.91 s |

On this suite and hardware, MiniCPM used 22% of the comparison model's summed
request time (78% less, about 4.5x faster). It missed seven cases: one basic
image-generation selection, one ambiguous code-run selection, two three-step
chains, two missing-argument refusal cases, and one dependent nested call. All
emitted tool arguments that reached the parser were valid JSON.

Manual product-path probes also passed:

- strict JSON-only response, with no surrounding prose;
- required single tool call with schema-valid arguments;
- tool-result continuation into a final natural-language answer;
- a thinking-enabled reasoning request terminating normally at 760 generated
  tokens in 3.52 seconds (216.1 tokens/s reported by the server).

## Caveats

- These timings are M3 Ultra results and must not be projected to other chips.
- The comparison measures official MLX 4-bit artifacts, not GGUF Q4_K_M.
- One run of 31 scenarios establishes product compatibility, not a general
  quality ranking.
- During the alias-based confirmation run, Rapid's exact-token loop breaker
  intervened in the failed basic image-generation selection case. The request
  recovered and the remaining suite completed, but the intervention reinforces
  keeping this model out of the default recommendations for now.
- A third-party report observed repetition in a GGUF thinking workload at the
  published sampling defaults. The MLX product-path probe terminated normally,
  but broader stochastic stability should be monitored before changing a
  default recommendation.
