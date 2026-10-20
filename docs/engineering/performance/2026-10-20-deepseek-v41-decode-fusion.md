# Native V4.1 decode operation fusion

CPU qualification on 2026-10-20: Python 3.12.13, MLX
`0.32.1.dev20261019+3a62199`, CPU backend, compilation enabled. No Metal execution
or real checkpoint measurement was performed. Graph counts are operation nodes,
not measured kernel dispatch counts or a prediction of GPU latency.

## Method and results

Run from the repository root in an environment with CPU-capable MLX:

```sh
.venv/bin/python -m scripts.bench_deepseek_v41_decode_fusion --device cpu
.venv/bin/python -m pytest tests/test_deepseek_v41_decode_fusion.py
```

The benchmark creates a seeded four-layer synthetic model (dimension 64, two
attention heads, head dimension 32, eight experts, top-2, window eight). Compression
ratios are `(0, 2, 2, 1)` with both source ownership and consumer sharing. Routing
and hyper-connection projection parameters are nonzero. It prefills eight tokens,
evaluates parameters, frequency tables and cache state, then exports the graph
of one single-stream, one-token decode. Only operation rectangles in the DOT graph
are counted; evaluated leaves and array vertices are excluded. Layer evaluation
boundaries are temporarily disabled for graph capture and restored afterwards.
The timing mode retains the model's normal layer evaluation boundaries.

| Output-reachable operations | Fusion off | Fusion on |
| --- | ---: | ---: |
| Layer 0, window only | 431 | 364 |
| Layer 1, ratio-2 source | 451 | 247 |
| Layer 2, ratio-2 consumer | 302 | 173 |
| Layer 3, ratio-1 source | 667 | 322 |
| Final normalization/projection | 4 | 4 |
| Total | 1,855 | 1,110 |

The fused graph retains **59.8%** of the baseline nodes, a **40.2% reduction**.
The regression ceiling is **70%**, leaving about ten percentage points of
headroom. Per-layer counts attribute newly reachable operations at each block's
outputs, including its input preparation. The graph covers dependencies of
logits, not cache writes unused by the current step. Existing hyper-connection
compilation remains enabled in both modes.

## Fusion boundaries and numerical contract

`rapid_mlx.models.deepseek_v41_native.fusion.ENABLED` is the sole A/B switch,
default **on** after bit-exact CPU qualification. It is checked outside compiled
traces. Helpers receive weights, biases, scale reciprocals, lookup tables and
rotary frequencies as array arguments. They capture no changing module state.
Python scalar settings specialize traces. Ordinary shape specialization is used,
not `shapeless=True`: one-token decode uses fixed shapes, while each distinct
prefill chunk shape/dtype/rank and configuration gets its own trace. There is no
position or cache-offset argument in these compiled helpers, so growing sequence
positions do not generate new traces. Callers using arbitrarily many prefill
chunk sizes will create correspondingly many specializations; prefer fixed
chunk sizes.

Accepted compiled chains:

- Fake quantization: absolute maximum and floor, IEEE power-of-two scale
  construction, FP8 clip/round-trip/rescale, and FP4 clip/seven midpoint
  comparisons/lookup/sign/rescale. FP4 lookup tables are explicit trace inputs.
- Adjacent-pair rotary multiplication, including inverse rotation and both
  headed and headless layouts. Tail concatenation remains outside the trace.
- Gate scoring, selection bias, top-k selection, weight gathering,
  normalization and scaling. Bias affects selection only. The gate projection
  and temperature division remain outside the trace.
- Clamped and unclamped SwiGLU, reused by routed and shared experts, and the
  FP32 weighted expert reduction plus shared-expert addition.

Two candidates were rejected after equivalence testing. Compiling a complete
power-of-two FP4 quantizer changed results immediately beside rounding midpoints
(for example, `0.74999994` rounded to `1` instead of `0.5`). Keeping activation
normalization division outside the compiled rounding chain prevents this CPU
reciprocal approximation from crossing a quantization boundary. Scale-format
FP8 conversion also keeps its division eager. Compiling complete RMSNorm changed
FP32 values at roughly the last-bit level; RMSNorm remains eager. No numerical
error allowance is needed for the accepted paths: every equivalence assertion
uses `np.array_equal`, with additional signed-zero checks for individual helpers.

Tests cover FP32, FP16 and BF16 inputs; FP4 midpoint neighbors and ties; FP8
midpoint neighbors; zero blocks, negative zero, scale floors and their neighbors,
and very large finite activations; both norm epsilons; changing norm weights and
routing biases; tied expert scores; all three score functions, top-1/top-2,
normalized/unnormalized routing; both SwiGLU limits; and forward/inverse rotary.
The tiny model produces identical logits and tokens for **32 greedy decode
steps** after both an eight-token prefill and `(3, 2, 3)` chunked prefill.
`fakequant.DISABLE` bypasses before any compiled helper, including after a trace
has been cached. Compilation stayed enabled throughout; no compiled-function
teardown crash was observed.

Window/index visibility construction, sink handling, compressor pooling and
indexer score reduction were inspected but left unchanged in this phase. They
interleave variable-length cache management, reduction/matmul boundaries and
position-dependent Python scalars; the measured fixed-shape chains already
remove 40% of this graph without introducing position-specialized traces.
Compressor/indexer callers still benefit from fused quantization, rotary and
shared normalization semantics. Hyper-connections and draft-model code remain
unchanged. No custom Metal kernels were added.

## Verification

The new test file is registered in the Apple CI test list. The CPU run passes
all 61 new tests. Across all `tests/test_deepseek_v41_*.py`: 231 passed,
12 skipped and two pre-existing CPU failures. Both failures were reproduced
against a separate archive of unchanged HEAD sources with the same interpreter:

- `test_target_qmv_install_and_small_route_execution` forces an existing custom
  kernel and fails because the CPU build has no Metal backend.
- `test_disk_engram_bypass_paths_handle_empty_and_uncached_rows` reaches an
  existing empty-array dequantization reshape failure.

Changed executable lines in the native runtime have 100% measured coverage.
Use a filesystem source path with coverage (`--source=rapid_mlx/models/deepseek_v41_native`);
this environment's dotted-module source probing causes a reentrant NumPy import.
Ruff lint/format checks pass. Routing-policy and benchmark-metadata guards pass
(51 tests). The pinned Python 3.11 mypy environment reports **821 grandfathered
errors across 158 files; no growth**. The benchmark timing mode was smoke-tested
on the tiny CPU model; those times are not GPU performance evidence.

## Pending real-checkpoint measurement

On the GPU host, use an existing local checkpoint and the normal GPU MLX
installation. From the repository root:

```sh
.venv/bin/python -m scripts.bench_deepseek_v41_decode_fusion \
  --device gpu --model "$MODEL_PATH" --timing --prompt-length 128 --steps 32 \
  > decode-fusion.json
.venv/bin/python -m pytest tests/test_deepseek_v41_decode_fusion.py
```

`MODEL_PATH` names an existing native checkpoint directory. No download is
performed by this benchmark. JSON carries schema/methodology metadata, backend
and runtime versions, total/per-layer node counts and operation histograms for
both modes. Timing reports a warmed, synchronized uninstrumented step mean and
a separate serialized attribution pass with attention/MoE/other milliseconds
per layer and in aggregate. Synchronization and instrumentation perturb that
attribution pass; compare end-to-end step means separately. The final head and
host overhead are included in aggregate other time. Checkpoint loading messages
go to stderr so stdout remains one JSON artifact.

Vector owns checkpoint/Metal equivalence and latency qualification; Harbor owns
running the Apple CI lane. Metal compiler arithmetic, actual dispatch reduction,
real-checkpoint greedy/logit parity and GPU latency remain unverified. Before
claiming production performance, run both A/B modes on identical checkpoint,
hardware, prompt, cache settings and runtime versions, and retain commit-bound
JSON evidence. The module switch provides an immediate eager fallback.
