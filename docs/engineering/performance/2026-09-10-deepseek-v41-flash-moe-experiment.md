# DeepSeek V4.1 Flash MLX MoE scheduling experiment

Date: 2026-09-10

## Decision

Do not add DeepSeek V4.1 Flash to the Rapid model catalog yet. The complete
2-bit checkpoint fits on this 256 GiB host and generates correct text, but the
unchanged warm runtime reaches only 6.055 tok/s. The fastest experimental
runtime reaches 8.030 tok/s but fails the numerical/quality gate, while the
safe expert-local optimization reaches 6.216 tok/s. The 20+ tok/s target is not
demonstrated.

The next material prototype is parallel target block verification for the
three native MTP stages. The checkpoint's current verifier advances target
tokens serially and is slower than non-MTP decoding. Ordinary quantization,
expert packing, and launch reduction do not supply the missing 3.3x by
themselves.

## Artifact acquisition

The candidate baseline is `Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP`. Its indexed
checkpoint size is 238,796,133,496 bytes (222.4 GiB). It was initially blocked
by cache capacity. After the administrator explicitly approved removal of
named, unused model caches, the checkpoint was downloaded into the standard
Hugging Face cache. No cache path was overridden or redirected.

All 143,982 indexed tensors across 48 shards were present, no incomplete files
remained, and `hf cache verify` checked all 64 repository files successfully.
This note still makes no vision, long-context, or tool-use quality claim.

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

## Next engineering gate

1. implement a bounded multi-token target forward that verifies one DSpark
   block without serially mutating target KV state;
2. preserve rejection semantics by committing only the accepted KV prefix;
3. measure proposal cost, accepted tokens per block, target block cost, and net
   tok/s separately;
4. adapt mHC fusion to reproduce the checkpoint graph's numerical order, or
   pass a representative quality eval before accepting different numerics;
5. combine only individually qualified optimizations and run at least 128 warm
   decode tokens at 8K context;
6. evaluate quantization quality before considering REAP, pruning, or larger
   affine group sizes.

Only after those gates pass should Rapid model loading, catalog metadata,
server routing, GUI exposure, or a downloadable quantization artifact enter
scope.

## Real-weight baseline on the 60-core M3 Ultra

After policy-compliant cache cleanup, the 238,796,133,496-byte checkpoint was
downloaded into the standard Hugging Face cache. All 143,982 indexed tensors
across 48 shards were present, no incomplete files remained, and `hf cache
verify` checked all 64 repository files successfully. The HF volume retained
107 GiB free after download.

Environment:

- Apple M3 Ultra, 60 GPU cores, 256 GB unified memory
- MLX 0.32.2
- checkpoint revision `802f1a00982705d81b79ad1c83aa0ccc0b863ebc`
- checkpoint runtime SHA-256
  `55ab20f116a663d79fc6980fbb4613bb60d6cba7ab64dfa4d580d65eeba60645`
- compiled execution mode and resident text backbone

The checkpoint author's exact arithmetic command was reproduced with its
default 16,384-row Engram cache. It generated the expected
`2 plus 2 equals 4.` text, but achieved 5.29 tok/s on the first run and 6.05
tok/s on the repeated warm-cache run. The published 9.46 tok/s short result was
not reproduced on this 60-core GPU configuration.

A separate 32-token warmup followed by 128 continuous decode tokens measured:

- 6.018 tok/s
- 165.95 ms median token latency
- 173.96 ms p95 and 208.94 ms maximum
- 173,505,743,338 active MLX bytes and 173,589,483,108 peak bytes
- 491,520 Engram-related disk bytes across 18,432 small read calls

An instrumented 16-token phase observed 40 router barriers and 40 layer
barriers per token. Existing barrier wait averaged about 130.6 ms/token:

- router selection barrier: 75.6 ms/token
- layer boundary barrier: 53.7 ms/token
- final logits barrier: 1.1 ms/token
- small indexed reads: 0.2 ms/token

Barrier attribution identifies where queued work is observed, rather than the
exclusive cost of the Python caller.

Real-weight candidate experiments then bounded straightforward runtime
headroom:

1. Removing the redundant layer-boundary `mx.eval` and per-layer finite-value
   diagnostic preserved the exact generated text and improved alternating warm
   runs from 6.16--6.22 tok/s to 7.57--7.63 tok/s, a 22--24% gain.
2. Reusing Rapid's fused gate/up `gather_qmm` path for one real layer improved
   routed MoE latency from 1.276 ms to 0.714 ms (1.79x). The maximum absolute
   BF16 difference was 0.0001221. A smaller fixed-route reproduction of the
   same math was bit-identical and improved 0.815 ms to 0.404 ms (2.02x).
3. Scaling the expert-major layout to all 40 layers failed badly. It packed
   169,869,312,000 bytes in 10.94 seconds without a memory spike, but the large
   Metal buffers were consistent with a severe full-model VM/page-locality
   penalty: throughput fell to 0.436 tok/s and median latency rose to 2.306
   seconds. This layout is rejected even though its isolated one-layer
   benchmark is fast.
4. Fusing gate/up inside each existing small expert buffer retained locality
   and produced bit-identical quantized-matmul output in the focused test. It
   packed 113,246,208,000 bytes in 4.76 seconds, reduced active memory slightly,
   preserved the greedy token chain, and improved 64-token decode from the
   6.055 tok/s control to 6.216 tok/s (+2.65%). Median latency fell from 165.96
   ms to 160.55 ms (-3.26%). This is safe but not material enough alone.
5. A V4.1-specific fused mHC mixing kernel preserved the architecture's
   pipelined `pre` timing and reduced a real mixing/collapse microbenchmark from
   951 microseconds to 315 microseconds (3.02x). In the full model it improved
   throughput from 6.055 to 8.030 tok/s (+32.6%) and median latency from 165.96
   to 123.95 ms (-25.3%). It did not pass the quality gate: tiny per-layer
   differences were amplified across 40 hyper-connected layers, producing only
   43.75% top-1 agreement across a 16-token teacher-forced comparison and up to
   15.80 absolute logit error. This kernel remains an explicit experimental
   upper bound and must not be enabled in a product runtime.

The resident backbone is 172,741,563,840 bytes. The minimum per-token text
weight traffic is about 5.34 GB: 2.69 GB of dense text weights plus 2.65 GB for
six active routed experts across 40 layers. At 20 tok/s this is about 107 GB/s,
so nominal memory bandwidth is not the physical blocker. The reference path
instead performs about 840 routed expert quantized matmuls per token and
observes 80 host barriers. Launch count, reduction scheduling, and target
verification are the useful levers.

The bundled three-stage DSpark implementation does not provide the missing
multiplier. Its verifier is deliberately serial: each proposed token advances
the target model one token at a time. The checkpoint report also records it as
slower than non-MTP decoding, with only about 1.00--1.33 accepted draft tokens
per block on its short diagnostics. A credible 20+ tok/s path therefore needs
multi-token target block verification, plus the correctness-safe synchronization
and kernel work above. Quantization or REAP that only reduces stored bytes does
not close the throughput gap.

The real-weight profiler is plan-only unless both execution flags are supplied.
It never downloads a model and requires explicit consent before importing the
checkpoint-bundled Python runtime:

```shell
uv run python scripts/bench_deepseek_v41_runtime.py \
  --model ~/.cache/huggingface/hub/models--Vontra--DeepSeek-V4.1-Flash-MLX-2bit-MTP/snapshots/<revision> \
  --execute-real-weights --trust-checkpoint-runtime \
  --resident-backbone --execution-mode compiled \
  --engram-cache-rows 16384 \
  --context-tokens 8192 --warmup-tokens 16 --measure-tokens 128 \
  --diagnostic-tokens 8 \
  --output /private/tmp/deepseek-v41-flash-8k.json
```

The JSON attributes wait time to existing `mx.eval` call sites without adding
new synchronization points. This identifies where queued work is observed, not
exclusive kernel time.

Do not enable whole-process Metal capture for this model. A one-token capture
grew to approximately 112 GiB before termination, reducing system-volume free
space to 14 GiB. The capture process was stopped, the single generated scratch
artifact was removed, and free space returned to 126 GiB. Kernel experiments
must isolate a bounded layer or operation instead of capturing the resident
whole-model process.
