# Qwen3.8 MTP GDN verify fusion

Date: 2026-09-07

Issue: #3156

Scope: fixed-K=3 native MTP target verification only

## Result

Qwen3.8's hybrid backbone has 48 GatedDeltaNet layers. Before this change,
each four-position K=3 target-verify block launched the recurrent update four
times per GDN layer so that all three possible rollback boundaries remained
available. The new path performs the same recurrence once and emits the final
state plus all three boundary states from that pass.

On an Apple M3 Ultra, three alternating base/candidate runs with the exact same
checkpoint, prompt, seed, and fixed K=3 improved median decode throughput from
45.87 to 52.28 tokens/second (+14.0%). Median target-verify synchronization
time fell from 4.954 to 4.644 seconds (-6.3%). Every run produced the same
256-token SHA-256 (`77e8ab7a2fd5df99959cd8596592a973162f7f61f3710e09d6accb73f1b813a0`),
the same 53.74% acceptance ratio, and the same 98 K=3 verify rounds.

This measurement demonstrates the improvement on M3 Ultra. It does not by
itself establish that the issue reporter's M2 Max result crosses the
autoregressive break-even point; that machine reported 18.1 versus 20.9
tokens/second before this change and still needs a clean-device recheck.

## Measurements

| Pair | Base decode tok/s | Candidate decode tok/s | Change | Base verify sync | Candidate verify sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.87 | 50.85 | +10.8% | 5.079 s | 4.740 s |
| 2 | 46.01 | 52.41 | +13.9% | 4.909 s | 4.644 s |
| 3 | 43.81 | 52.28 | +19.3% | 4.954 s | 4.606 s |
| Median | 45.87 | 52.28 | +14.0% | 4.954 s | 4.644 s |

## Environment

- Mac Studio, Apple M3 Ultra, 28 CPU cores, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13
- MLX 0.32.2; mlx-lm 0.31.3
- Base commit: `d09f18adaaf0435d07960bcf229e85a512b57616`
- Model revision: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`
- The exact model revision was already present in the Studio-managed default
  Hugging Face cache. No cache path override or download was used.

## Reproduction

The runs used `bench/bench_spec_decode_mtp.py` with one explicit prompt and a
fresh process for each observation. Base and candidate observations were
alternated. Resolve `MODEL` and `MTP` inside the globally configured
`HF_HUB_CACHE`; do not override the cache location.

```bash
python bench/bench_spec_decode_mtp.py \
  --model "$MODEL" \
  --mtp-sidecar "$MTP" \
  --runs 1 \
  --prompt-text 'Write a detailed Python implementation of a thread-safe priority job scheduler with leases, retries, cancellation, persistence transactions, and deterministic tests. Return code only.' \
  --max-tokens 256 \
  --mtp-only \
  --mtp-max-k 3 \
  --mtp-disable-auto-k
```

One candidate diagnostic also measured ordinary autoregressive decode at
39.75 tokens/second and MTP at 52.16 tokens/second (1.31x). Its strict
AR-versus-MTP token-hash gate failed. The candidate and base MTP hashes above
were identical, so the fusion did not introduce that pre-existing divergence;
the AR/MTP hash discrepancy is intentionally outside this performance change.

## Safety boundary

The fused path is selected only for inference on Metal with a GPU default
device and a key width divisible by 32. Training, CPU execution, unsupported
key widths, K=1 verification, prefill, and ordinary decode retain their prior
paths. A regression test compares fused and position-wise outputs, final cache
state, and every rollback boundary.
