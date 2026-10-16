# Bolt four-bit decode investigation

Owner: Vector. PR #4679, 2026-10-15.
This investigation preserves the text-only contract in
[the decision record](../decisions/2026-10-15-nemotron-bolt-text.md).
No converted model files, other model implementations, or operator processes
were changed.

## Environment and workload

Mac Studio M3 Ultra, 28 CPU / 60 GPU cores, 256 GB unified memory, macOS 26.5.2;
Python 3.12.14, MLX 0.32.4, mlx-lm 0.31.3. Model:
the four-bit/group-64 artifact described in the decision record, using Rapid's
registered text loader. Set `MODEL_DIR` to this artifact's directory.
The conversion and protected BF16 head/router/latent projections are unchanged.

Batch one, 1,024 prompt token IDs, temperature zero, prefill chunks of 256,
96 generated tokens, three sequential repetitions. Form the prompt by repeating
and trimming the encoding of `The quick brown fox jumps over the lazy dog. `.
Use `mlx_lm.stream_generate` with a fresh cache for each repetition and retain
its reported generation rate; no warmup repetition is discarded. No prefix
reuse, speculative model, or KV quantization. Attention cache uses the model
activation dtype; recurrent SSM state is FP32.

The earlier Phase B measurements used 64 generated tokens and a different
shared-machine load. Their roughly 32 tok/s result is not the baseline for the
optimization claim here. An apples-to-apples 96-token baseline was measured
before the optimization and repeated after its isolated experiment.

## Measurements

| Implementation | Decode tok/s, all three runs | Manual step graph construction, median ms | Manual step evaluation, median ms |
| --- | --- | ---: | ---: |
| Original normalization | 51.24 / 50.90 / 51.25 | 2.281 | 19.524 |
| Compiled normalization experiment | 57.55 / 57.31 / 56.95 | 2.188 | 18.007 |
| Original normalization repeat | 52.04 / 51.53 / 51.72 | 2.302 | 19.433 |
| Final committed implementation | 56.29 / 56.50 / 56.50 | 2.266 | 18.241 |

The repeat includes the first correctness-only changes, with the same uncompiled
normalization; this workload has no right padding and uses top-nine routing.
The median experiment improves on the initial baseline by 11.8% and the repeat
by 10.8%. The final implementation improves on those baselines by 10.3% and
9.2% respectively, with somewhat higher concurrent CPU load (one-minute load
6.87 versus 4.15 initially). This is a small local optimization: move the existing normalization
math into an `mx.compile` helper with explicit input, weight, and epsilon
arguments. FP32 reduction, FP32 weight multiplication, and final activation cast
are preserved. No parameters or mutable module state are captured implicitly.

Manual step measurements warm the model with the same prompt, then time 40
single-token forwards. Record the time to construct `model(token, cache)` and
argmax, then separately `mx.eval(next_token, [c.state for c in cache])`.
Construction includes Python and native graph-building work; it is an upper
bound on Python overhead, about 10% of the measured serial step. Python overhead
across 88 layers does **not** dominate. These serial timings include explicit
cache synchronization and cannot be substituted for the pipelined generator's
rate.

For layer attribution, run eight additional decode steps, synchronizing after
every layer and its cache. Sum by type within each step, then take medians:

| Layer type | Count | Original build / evaluation ms | Compiled experiment build / evaluation ms |
| --- | ---: | ---: | ---: |
| Mamba | 40 | 1.506 / 16.877 | 1.363 / 15.729 |
| MoE | 40 | 1.430 / 19.018 | 1.271 / 17.886 |
| Attention | 8 | 0.240 / 2.675 | 0.207 / 2.556 |
| Final normalization and LM head | 1 | 0.014 / 1.692 | 0.012 / 1.650 |

**MoE is the largest measured category, followed by Mamba.** Per-layer barriers
inflate the sum substantially compared with a complete step; these are
instrumented attribution timings, not additive fractions of uninstrumented GPU
time. The large head is measurable but does not dominate.

## Bytes-read estimate

Sum each layer's actual parameter `nbytes`, including quantization scales and
biases. For expert stacks only, multiply by `top_k / num_experts`; include all
router, shared-expert, and latent-projection weights. Embedding lookup reads one
row, not the whole embedding table. The tiny embedding-row and final-norm
reads are omitted from this estimate.

| Category | Logical weight bytes read per token |
| --- | ---: |
| Mamba | 2,470,574,080 |
| MoE, selected experts plus shared/router/latent weights | 2,728,482,816 |
| Attention | 160,497,664 |
| LM head | 1,073,741,824 |
| Total of the categories above | 6,433,296,384 |

At the M3 Ultra's advertised
[819 GB/s bandwidth](https://support.apple.com/en-au/122211), this gives an
optimistic weight-only time of **7.855 ms/token**, or **127.3 tok/s**. Reading
170,229,760 bytes of recurrent/conv cache and 10,485,760 allocated attention-cache
bytes adds approximately 0.221 ms; writing recurrent state adds another 0.208 ms.
This is logical traffic arithmetic, not measured DRAM throughput or a promised
achievable rate: caches, repeated reads, intermediate tensors, dispatch,
quantization arithmetic, and small-kernel occupancy affect the result. In
particular, 9B active parameters at a nominal four bits understates the traffic
because the output head and protected projections remain BF16.

## Fusion, compilation, and machine activity

All 40 expert blocks use the shared switch layer's `mx.gather_qmm` operations.
They have 12 fixed `(expert count, expert width, latent width, top-k)` shapes,
with expert counts 240/304/368/432/496, widths 1536/2048/2304/2560, latent width
1024, and top-k nine. The modules are constructed once; these shapes do not
change per decode step. There is no cross-layer homogeneous-shape fusion or
compilation eligibility check in this path for heterogeneity to disable. The
router selector already uses `mx.compile`. Shape-dependent kernel efficiency
may still matter; the measurements do not establish that heterogeneous shapes
have zero cost.

The shared Mamba path compiles its timestep preprocessing and uses a dedicated
Metal single-token SSM kernel. It does not compile the entire Python mixer.
The vendored adapter already has the same arrangement, with its required native
timestep dtype. Adding a missing shared Mamba compile decorator is therefore
not an available fix. The local FP32 normalization introduced for reference
semantics was uncompiled; compiling those same operations addresses a measured
cost without changing the shared implementation.

At the initial baseline checkpoint, IOAccelerator reported 0% device utilization
before loading. Its 21 samples over the run ranged from 0% to 100%, median 95%.
The resident operator server visible in the process snapshots used about 0.1%
CPU. No active competing GPU workload was evident before the run, but the device
counter is aggregate and cannot attribute activity to individual processes.
Earlier approximately-32-tok/s measurements cannot establish a persistent model
ceiling or identify a competing process retrospectively. CPU load and other
resident applications remain shared-machine confounders; no process was stopped
for benchmarking.

## Follow-up

Vector owns deeper MoE/Mamba profiling: capture a Metal GPU trace without
per-layer barriers, separate gather/dequantization from dense/shared projections,
and compare expert-count/width variants at equal selected-byte traffic. Evaluate
larger graph capture or fused expert/SSM work only with independent numerical,
cache-lifecycle, and hardware evidence. Such kernel/scheduler redesigns are
outside PR #4679 and are not implemented here.

Historical raw evidence files are
`round1-profile-{baseline,norm-experiment,baseline-repeat,final}.json` and
matching logs. JSON includes every timed step, per-layer samples, exact
expert shapes and byte counts, GPU samples, and before/after process snapshots.
The profiling helper and raw evidence are not shipped in the repository.

To reproduce with an independent harness, load `$MODEL_DIR` through Rapid's
registered text loader and follow the workload, 40-step construction/evaluation
split, eight-step layer-barrier method, and traffic calculation specified above.
Run offline with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`RAPID_MLX_TRUST_REMOTE_CODE=0`. Set `OUT_DIR` to a user-supplied output directory
for results and environment metadata. Preserve the distinct generator and
instrumented measurements and report all repetitions.

The final implementation passed all 4,419 focused tests and all 21 Bolt tests
with `MLX_ENABLE_TF32=0`. Ruff and formatting checks passed. Changed-line
coverage against `origin/main` at `6fb35fbb24c5b8bb151d4fe2a7d7858adfc6d146`
was 100% (314 executable lines, zero missing).
