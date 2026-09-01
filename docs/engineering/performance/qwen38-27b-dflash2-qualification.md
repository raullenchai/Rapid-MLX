# Qwen3.8-27B DFlash2 qualification

Date: 2026-09-01

## Decision

Do not select DFlash2 by default for `qwen3.8-27b-4bit`. The released
DFlash2 runtime works and preserves sensible target-model output, but both the
full-precision and 4-bit drafters were slower than Rapid-MLX's already-default
continuous MTP path on the qualification workload. The known DFlash2 pairing
remains available only through explicit experimental configuration.

## Environment

- Mac Studio, Apple M3 Ultra (28 CPU cores), 256 GB unified memory
- macOS 26.5.2 (25F84)
- Rapid-MLX `63dac4f10bf188feacafcd692734e0a148165930`
- `mlx==0.32.2`, `mlx-lm==0.31.3`, `mlx-vlm==0.6.17`
- Target: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`, revision
  `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`
- DFlash2 source drafter: `z-lab/Qwen3.8-27B-DFlash2`, revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- One large model pair resident at a time; prefix caching disabled
- Greedy decoding, thinking disabled, 256-token ceiling, three runs per case

## Commands

The 4-bit drafter was produced with the released optional runtime:

```bash
mlx_vlm.convert \
  --hf-path z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --mlx-path /private/tmp/vector-qwen38-dflash2-4bit \
  --quantize --q-bits 4 --q-group-size 64
```

The paired benchmark command was:

```bash
python scripts/bench_dflash.py \
  --model qwen3.8-27b-4bit \
  --draft-model /private/tmp/vector-qwen38-dflash2-4bit \
  --expected-algorithm dflash2 \
  --runs 3 --max-tokens 256 --port 8765
```

The baseline command intentionally uses the alias's default policy, which is
continuous MTP for this qualified target. DFlash2 runs in Rapid-MLX's dedicated
single-user serial speculative server. Comparing against plain autoregressive
decode would overstate the user-visible gain because ordinary users already
receive MTP.

## Results

Median decode throughput, three runs per case:

| Workload | Default MTP (tok/s) | DFlash2 q4 (tok/s) | DFlash2 / MTP |
| --- | ---: | ---: | ---: |
| Fibonacci | 57.6 | 54.2 | 0.94x |
| Quicksort | 53.7 | 50.4 | 0.94x |
| Hash table | 56.0 | 52.5 | 0.94x |
| Sorted list | 57.6 | 50.1 | 0.87x |
| Chat | 49.8 | 46.7 | 0.94x |
| **Code median speedup** |  |  | **0.94x** |

The full-precision drafter also failed qualification: code median 0.92x and
chat 0.83x versus the alias default. Its draft weights occupied about 3.6 GB;
the local 4-bit conversion occupied about 1.0 GB.

## Correctness spot-check

Five greedy requests were run through both the default MTP route and DFlash2:
coding, creative writing, factual chat, compact JSON, and a tool call. Coding,
JSON, and the normalized tool call were byte-for-byte identical. Creative
writing and factual chat differed in wording while remaining coherent,
instruction-following, and semantically equivalent. This is consistent with
the two routes using different target-model loading/generation paths; it is not
evidence for claiming byte identity across engines.

## Revisit criteria

Re-run qualification only if the DFlash2 runtime, target checkpoint, or
Rapid-MLX verifier path changes materially. Graduation requires a code-workload
median of at least 1.30x over the then-current default, no non-code workload
below 1.00x, correct tool/structured output, and explicit runtime identity in
`/healthz`. Until those conditions hold, the registry must not advertise the
pair as verified.
