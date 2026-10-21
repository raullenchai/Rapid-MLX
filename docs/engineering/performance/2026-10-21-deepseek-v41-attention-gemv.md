# V4.1 affine attention projection GEMVs

The target attention call sites use a shared-load candidate for 1–8 input
rows, with automatic stock fallback after lazy exactness qualification. This
covers normal decode, verification chunks and independent-position batched
decode. Prefill over eight rows and unsupported layouts retain stock dispatch.
The attention algorithm, checkpoint tensors and quantization are unchanged.

## Layouts

All supported weights use affine quantization and group size 64. The shipped
mixed attention map has 8-bit grouped `wo_a` throughout, with 4- or 8-bit
`wq_a`, `wq_b`, `wkv` and `wo_b`. The earlier all-8-bit map is supported too.
The kernel also accepts 4-bit grouped weights. It does not replace indexer or
compressor projections, the output head, or draft attention adapters.

| Projection | Output width | Input width | Weight groups |
| --- | ---: | ---: | ---: |
| `wq_a` | 1280 | 5120 | 1 |
| `wq_b` | 32768 | 1280 | 1 |
| `wkv` | 512 | 5120 | 1 |
| `wo_a` | 1024 | 4096 | 8 |
| `wo_b` | 5120 | 8192 | 1 |

`wo_a` retains its existing FP32 cast and singleton matrix-row axis; the
other projections retain incoming activation precision. Output dtype follows
stock promotion of activation and scale dtypes. Noncontiguous inputs are made
row-contiguous by the custom-kernel wrapper. K must contain whole 64-value
groups and exceed 128; partial output tiles and partial lane blocks are guarded.

## Numerics and dispatch

The candidate retains stock QMV's values per lane, increasing K block order,
4/8-bit dot expressions, affine epilogue and `simd_sum` reduction. Four SIMD
groups produce sixteen output columns per threadgroup. Each lane loads a
packed uint or uint2 plus scale/bias once per output/K block and reuses those
registers for all input rows. No partial sums are combined across input rows
or SIMD groups.

Each module lazily compares two deterministic probes with the **actual stock
module call** at the same full input shape and dtype. Comparison uses byte
views, including signed zeros. It never substitutes concatenated single-row
stock results for multi-row stock. A changed weight, scale, bias or quantization
setting invalidates cached verdicts; checks use weak references and are not
model parameters. Each newly checked shape emits an INFO coverage line with
its exactness verdict and qualified/fallback row counts. Mismatches and kernel
failures emit WARNING and retain stock for that shape. Thus a runtime choosing
multi-row QMM or another reduction order can reject those row counts while
keeping qualified single-row GEMVs. Unchecked shapes qualify on first use.
The existing `RAPID_MLX_DSV41_FUSED=0` switch disables these kernels too.

## Verification and measurement

The Apple CI test list includes `tests/test_deepseek_v41_attention_gemv.py`.
Its Metal cases exercise both bit widths, all five target layouts, every row
count 1–8, FP32/BF16 activations, FP16 scale/input combinations, odd output
widths, K tails and noncontiguous inputs. The tests require direct candidate
parity for single-row cases; multi-row cases require parity for every enabled
specialization and exact stock fallback otherwise. CPU tests cover eligibility,
qualification caching/invalidation, compile/mismatch/launch fallback, signed
zero comparisons, both attention wiring paths and benchmark synchronization.

Run on an idle Apple GPU:

```sh
python -m pytest tests/test_deepseek_v41_attention_gemv.py
python scripts/bench_deepseek_v41_attention_gemv.py --dtype float32
python scripts/bench_deepseek_v41_attention_gemv.py --dtype bfloat16
```

The benchmark uses synthetic BF16 weights quantized into the real projection
shapes; it does not download or load a checkpoint. It synchronizes before and
after each timed evaluation, warms up outside timing, and reports median stock,
raw candidate and qualified production dispatch latency separately. JSON lines
include host/device/runtime, dimensions, bits, row count, exactness qualification,
sample parity, bytes and effective GB/s. Bandwidth counts each packed weight,
scale and bias once per call, including all groups. Qualification/JIT overhead
is excluded from steady-state timing.

Pending: Apple GPU exactness/coverage results, projection bandwidth A/B, and
full-model M3 Ultra target decode/verify timing. The implementation sandbox has
no Metal device; no speedup or qualified row counts are claimed from CPU tests.
