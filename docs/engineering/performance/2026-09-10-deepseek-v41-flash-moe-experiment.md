# DeepSeek V4.1 Flash MLX MoE scheduling experiment

Date: 2026-09-10

## Decision

Do not add DeepSeek V4.1 Flash to the Rapid model catalog yet. A synthetic
Metal experiment shows that the decode-time MoE path has worthwhile scheduling
headroom, but a real-weight end-to-end run is blocked on this host and the
20+ tok/s target is not yet demonstrated.

The next prototype should combine an expert-major packed checkpoint layout
with batched active-expert quantized matmuls and fewer layer-boundary host
synchronizations. Converting only the runtime while retaining separately stored
expert tensors recovers little of the potential gain.

## Storage gate

The candidate baseline is
`Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP`. Its indexed checkpoint size is
238,796,133,496 bytes (222.4 GiB). It was not present in the Hugging Face cache,
which had about 80 GiB free before this experiment. The studio storage policy
forbids deleting other cached models or redirecting the download, so no model
shards were downloaded.

Consequently, this note makes no quality, memory-residency, vision, long-context,
tool-use, or end-to-end throughput claim.

## Method

The benchmark uses the published text-model dimensions and the community
checkpoint's quantization configuration:

- Apple M3 Ultra, 256 GiB unified memory
- MLX GPU (`applegpu_g15d`)
- hidden size 5,120
- MoE intermediate size 2,304
- six active routed experts plus one shared expert
- 40 layers
- affine 2-bit weights, group size 64
- one decode token, batch size one
- synthetic BF16 weights quantized by MLX and kept resident
- 8 warm-up observations and 100 timed observations, alternating order

Command:

```shell
uv run python scripts/bench_deepseek_v41_moe.py \
  --execute-metal --bits 2 --group-size 64 \
  --warmup 8 --repeats 100 \
  --output /private/tmp/deepseek-v41-moe-token.json
```

The benchmark deliberately excludes routing, expert selection from the full
384-expert layer, attention, KV cache, Engram lookup, mHC residuals, MTP,
tokenization, and server overhead. The derived MoE-only TPS values are upper
bounds, not predicted model throughput.

## Results

All compared paths produced bit-identical outputs for the synthetic input.

| One-layer MoE path | Median ms/layer | MoE-only 40-layer upper bound |
| --- | ---: | ---: |
| Serial experts, sync after every expert | 2.088 | 11.97 tok/s |
| Serial experts, sync once per layer | 0.601 | 41.62 tok/s |
| Batched experts from pre-packed tensors, sync once per layer | 0.454 | 55.11 tok/s |
| Batched experts, stack separate tensors each layer | 0.796 | 31.39 tok/s |

The current community runtime's compiled/deferred mode does not synchronize
after every expert. Its closest micro-benchmark analogue is therefore the
serial, once-per-layer row, not the 9.23 tok/s row.

Composing 40 synthetic MoE layers gives a second scheduling diagnostic:

| Synthetic 40-layer graph | Median ms/token | MoE-only throughput |
| --- | ---: | ---: |
| Serial experts, sync every layer | 24.18 | 41.35 tok/s |
| Serial experts, one token-level sync | 11.36 | 88.04 tok/s |
| Batched pre-packed experts, sync every layer | 17.36 | 57.61 tok/s |
| Batched pre-packed experts, one token-level sync | 8.00 | 124.95 tok/s |

These graph results reuse the same synthetic weights at every layer and omit
all non-MoE work. They establish direction and relative scheduling cost only.
They must not be quoted as DeepSeek V4.1 Flash throughput.

## Quantization layout sweep

The same one-layer benchmark was repeated with 60 observations per case.

| Bits / group | Serial layer-sync ms | Pre-packed batched ms | Split-layout batched ms |
| --- | ---: | ---: | ---: |
| 2 / 32 | 0.582 | 0.425 | 0.847 |
| 2 / 64 | 0.601 | 0.454 | 0.796 |
| 2 / 128 | 0.589 | 0.431 | 0.757 |
| 3 / 64 | 0.577 | 0.442 | 0.878 |
| 4 / 64 | 0.583 | 0.470 | 0.957 |

The integer bit width and affine group sizes were close on this kernel-level
test; none showed enough advantage to select on speed alone. Group size 128
reduces affine scale/bias overhead by 0.25 bits per quantized parameter versus
group size 64. That saving still needs a real-weight quality and end-to-end
speed gate before adoption.

## Interpretation

The experiment supports three bounded conclusions:

1. Active-expert batching is worth prototyping, but only with an expert-major
   packed artifact or a one-time conversion that does not copy selected weights
   on every layer and token.
2. Layer-boundary synchronization is material. A production implementation
   should make the longest correctness-safe graph possible while retaining
   cancellation and bounded memory behavior.
3. MLX's existing integer affine formats are sufficient for the first runtime
   prototype. Fractional-BPW packing and a custom Metal kernel remain separate
   experiments; neither is required to validate the scheduling hypothesis.

The experiment does **not** show that 20+ end-to-end tok/s is achieved. Starting
from the community report of roughly 9 tok/s, expert batching alone cannot be
linearly extrapolated because non-MoE work and storage behavior are unmeasured.

As a deliberately optimistic Amdahl bound, the reported 8.8--9.5 tok/s baseline
costs about 114--105 ms/token. Replacing the synthetic current-runtime analogue
(24.18 ms) with the fastest synthetic batched/token-sync path (8.00 ms) saves
only 16.18 ms. Even if that saving transferred perfectly, it would imply only
about 10.3--11.2 tok/s. Reaching 20 tok/s still requires another 47--39 ms/token
from the unmeasured attention, Engram, residual, weight-access, or speculative
decode paths. This is why a real-weight profile is the next gate rather than a
catalog integration.

For size, moving affine quantized tensors from group 64 to group 128 saves 0.25
bits per affected parameter. Applying that to every approximately 754.6 billion
parameter is an intentionally generous ceiling of about 23.6 GB; the real
saving is smaller because not every tensor uses affine quantization. Group-size
changes alone therefore cannot deliver the earlier 165--185 GB target. That
target still needs structured pruning, sub-2-bit packing, special Engram
compression, or a combination, followed by quality evaluation.

## Next real-weight gate

When a policy-compliant host has at least 223 GiB of cache capacity available:

1. run the community checkpoint unchanged with the prepared profiler and record
   per-token, existing-barrier, Engram-cache, disk-read, and peak-memory
   measurements;
2. convert one layer to expert-major packed storage and compare numerics and
   latency against the original separately keyed tensors;
3. prototype a 2-bit/group-64 batched active-expert path;
4. remove only correctness-safe synchronization boundaries and validate
   cancellation plus memory growth;
5. run at least 128 warm decode tokens at 8K context before making a throughput
   claim;
6. evaluate quantization quality before considering REAP or group size 128.

Only after those gates pass should Rapid model loading, catalog metadata,
server routing, GUI exposure, or a downloadable quantization artifact enter
scope.

The real-weight profiler is plan-only unless both execution flags are supplied.
It never downloads a model and requires explicit consent before importing the
checkpoint-bundled Python runtime:

```shell
uv run python scripts/bench_deepseek_v41_runtime.py \
  --model ~/.cache/huggingface/hub/models--Vontra--DeepSeek-V4.1-Flash-MLX-2bit-MTP/snapshots/<revision> \
  --execute-real-weights --trust-checkpoint-runtime \
  --resident-backbone --execution-mode compiled \
  --context-tokens 8192 --warmup-tokens 16 --measure-tokens 128 \
  --trace /private/tmp/deepseek-v41-flash-8k.gputrace --trace-tokens 4 \
  --output /private/tmp/deepseek-v41-flash-8k.json
```

The JSON attributes wait time to existing `mx.eval` call sites without adding
new synchronization points. This identifies where queued work is observed, not
exclusive kernel time; the optional Metal trace is required for kernel-level
attribution. To bound scratch growth, tracing is a separate post-measurement
probe capped at eight tokens and is refused when `/private/tmp` has less than
20 GiB free.
