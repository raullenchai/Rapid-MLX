# APCv2 audit: host prompt plane first

## Outcome

The first production-sized piece absorbed from
[`pierre427/mlx-lm-unified`](https://github.com/pierre427/mlx-lm-unified) is
the host prompt plane, not device-side zero-copy prefix broadcasting.

Rapid now retains exact, immutable chat-render and tokenization results in one
process-local LRU. The cache is shared by `BatchedEngine` and its text
`Scheduler`, bounded to 64 entries and 64 MiB, cleared by the existing prefix
cache API and at model unload, and disabled with
`RAPID_MLX_PROMPT_HOST_CACHE=0`. Inputs that cannot be given a
stable order-preserving JSON identity, and entries larger than the byte budget,
fail open to the existing path.

This is independent from KV/GDN/QSA state. It changes neither prompt tokens nor
model execution and applies to the server path used by CLI and Desktop.

## Positive qualification

Environment:

- Mac Studio, Apple M3 Ultra (28 CPU cores), 256 GB
- macOS 26.5.2 (25F84)
- Rapid base `7ff20a017f5e35e24c0b633f16d1e13f96017018`
- Python 3.12.14, MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.15.1
- already-cached `mlx-community/Qwen3-0.6B-4bit` tokenizer; no model weights
  loaded and no network download
- 50 interleaved repetitions after one warm-up/seed pass

Command:

```bash
.venv/bin/python benchmarks/prompt_host_cache_bench.py \
  --tokenizer-path ~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --characters 4096 32768 125833 \
  --repetitions 50 \
  --out /private/tmp/rapid-mlx-prompt-host-qwen3.json
```

| Prompt characters | Prompt tokens | Uncached median | Cached median | Speedup | Saved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 719 | 1.044 ms | 0.056 ms | 18.67x | 0.988 ms |
| 32,768 | 5,661 | 7.801 ms | 0.275 ms | 28.37x | 7.526 ms |
| 125,833 | 21,708 | 30.290 ms | 0.987 ms | 30.67x | 29.303 ms |

Every cached prompt string and token-ID sequence matched the uncached result
exactly. The largest case retained about 0.99 MiB under the conservative host
memory ledger.

These are host-preparation savings on an exact repeated request, not decode
tokens/second claims. They primarily improve retries, replay/evaluation traffic,
and repeated API calls whose complete messages/tools/template inputs match.

## Rejected device shortcut

A separate probe tested the tempting APCv2 shortcut of feeding a B=1 prefix to
B=2 fused SDPA through `mx.broadcast_to`, versus a pre-materialized physical B2
cache. The output was exact, but MLX 0.32.2 materialized the broadcast in the
attention path and made it progressively worse at long context.

Source: `benchmarks/cache_shared_prefix_attention_microbench.py` from
`pierre427/mlx-lm-unified` commit
`a8b2d31f6883dfd40870a1243a14480154f02f90`.

```bash
python benchmarks/cache_shared_prefix_attention_microbench.py \
  --tokens 16384 --warmup 8 --repetitions 30 \
  --out /private/tmp/apcv2-shared-16384.json
```

| Prefix tokens | Broadcast / physical speed | Broadcast peak over baseline |
| ---: | ---: | ---: |
| 4,096 | 0.85x | 8.9 MB |
| 16,384 | 0.63x | 34.1 MB |
| 65,536 | 0.46x | 134.8 MB |

Therefore Rapid must not advertise or ship a generic broadcast-based
“zero-copy” prefix consumer. A future device-side APCv2 needs a specialized
source-segment kernel or an asynchronously prepared physical cohort, followed
by a real server A/B against current continuous batching.

## Next investigation

The next high-value upstream area is static Qwen4 MTP cohort batching. It is not
a small cache patch: the upstream consumer specializes QSA private-delta
attention, recurrent-state row ownership, rollback transactions, and physical
promotion. Before porting it, require a baseline-vs-candidate server receipt;
the upstream tree currently contains engagement/correctness gates but no
checked-in comparative throughput artifact.
