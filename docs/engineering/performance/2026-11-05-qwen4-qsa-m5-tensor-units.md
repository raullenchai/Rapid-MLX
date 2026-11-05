# Experimental M5 QSA tensor-unit prefill

Date: 2026-11-05. Status: implemented behind an opt-in; **not hardware-qualified**.
Owner: Harbor for the experiment; Vector owns subsequent kernel qualification.
Branch: `perf/qsa-m5-tensor-units`. No default, catalog, release, or cache-format changes.

## Evidence and decision

The supplied PR #5084 from a public Apache-2.0 implementation uses
`MetalPerformancePrimitives/MetalPerformancePrimitives.h`, specifically
`mpp::tensor_ops::matmul2d` with cooperative tensors and
`metal::execution_simdgroup`. These are the M5 tensor units, rather than the
older SIMD matrix instructions. The supplied performance figures are
1.18–1.66x per-layer prefill and roughly 3–4x isolated kernels; they are not
Rapid-MLX measurements or forecasts. The supplied diff delegates availability
to a shared helper; it does not contain that helper's implementation.
The inspected helper mirrors the MLX runtime's OS/architecture test:
macOS >=26.2 and full-match `applegpu_g(\d+)([a-z])`, generation >=17,
except suffix `p` requires >=18. Successful compilation is not a device probe.
The earlier QSA stage-one qualification note records corrupted results from a
compile-only tensor-unit probe on M3 Ultra. This experiment never dispatches
these kernels on M1–M4, even if their compiler accepts the source.

The reference score kernel assigns 64 queries and a 128-token block to eight
simdgroups. Each index head's QK product uses 16x32x32 tensor operations and
FP32 accumulation; a masked row maximum reduces each block without storing
token scores. Its attention kernel assigns one query and KV head to a
threadgroup, with 16 grouped query heads as tile rows and two simdgroups
splitting the 128-dimensional head. It walks 16 selected 128-token blocks,
32 keys per step, with FP32 online softmax. PV receives probabilities as FP16
high and residual low pieces, avoiding FP32 operand truncation at the tensor
unit. The denominator retains FP32 probabilities.

The approach applies, but the score reduction and selection geometry do not
transfer unchanged. Rapid's indexer has four heads of dimension 128, one
compressed key per four tokens, and top-k 512 **compressed blocks**. Its score
is `sum_h relu(round_bf16(Q_h K^T)) / sqrt(128)`. A block maximum over raw keys
would select a different model. The new scoring kernel instead computes
16-query x 32-compressed-key tiles, rounds each head's result to bf16, sums
FP32 ReLU results, and writes `[1,L,S/4]` FP32 scores. This avoids the four-head
intermediate, but does not eliminate the final score matrix. The existing
exact radix selector then selects 512 IDs with the existing causal completion
mask and tie rules. Ranking runs on GPU scalar/SIMD operations; comparisons
and radix operations are not matrix products and do not execute on tensor units.

Rapid's main attention has 24 query heads, two KV heads, and head dimension
256. An inspected associated QSA attention kernel already fits this geometry:
12 grouped heads occupy 16 tile rows (four idle); two simdgroups each handle
128 dimensions; eight four-token blocks form each 32-key step. Its core is
adapted with attribution in NOTICE. The kernel now consumes Rapid's physical
block starts/counts and explicit tail indices/counts. It preserves persistent
KV storage semantics, physical left-padding offsets, partial blocks and
empty query rows; no host synchronization or dense selection mask is added.

## Route and fallback

Set `RAPID_MLX_QSA_M5_NAX=1` to request the route. Both score/selection and
attention require the architecture/OS gate, GPU as the default device,
batch size one, inference mode, and query length >=64. Scoring additionally
requires bf16 four-head/dim-128 inputs, >512 compressed keys, compression four
and top-k 512. Attention additionally requires bf16 24/2 heads, dim 256,
compression four and top-k 512. Other layouts and all decode calls retain
the existing paths, including independently configured stage-one,
block-sparse, and indexed split-K opt-ins. The initial dense prefix still
uses the existing causal attention route.

Inputs are normalized by the custom-kernel wrapper's row-contiguous policy.
Scalar element loads avoid assuming 16-byte alignment for offset views.
This accepts chunked, strided and misaligned arrays at the cost of possible
copies, including capacity-strided KV views. Benchmark timings include those
copies. Metadata comes from the existing selection builder and is trusted;
direct attention callers must supply valid physical block starts and counts.
Each indexer and attention module exposes `m5_nax_route_constructions` to prove
that a timing actually used the route. These count lazy graph constructions,
not completed GPU executions.

## Expected gain and remaining risks

No Rapid speedup claim is justified yet. The current batched `mx.matmul`
can itself use tensor units on M5; fused ReLU/head reduction and lower
intermediate memory are the scoring opportunity. The sparse consumer avoids
dense KV scanning, but its 512 small blocks require indexed loads and about
64 sequential online-softmax steps, plus a possible tail step. Copies,
register pressure, four idle head rows and metadata work can erase gains.
The score-only tile schedule is deliberately conservative and unmeasured.

BF16 per-head rounding matches the current path's rounding points, but FP32
dot-product summation order can change a rounded value at a boundary and
therefore change near-tied top-k sets. Exact ties reuse the deterministic
radix rule. Acceptance tests require >=99.5% selected-set agreement with the
current scoring path and >=99% with the unrounded FP32 oracle on seeded
synthetics; these thresholds do not establish full-model quality parity.
Attention tests isolate a fixed selected set and require absolute output RMS
error <0.003 against both current sparse attention and FP32 softmax. They
exercise irregular query tiles, nonconsecutive selected blocks, partial tails,
empty rows, strided inputs and misaligned head dimensions. Tolerances remain
provisional until M5 execution; wider stress and real-weight checks may reject
the experiment. The Metal 4 compiler accepts both shader bodies in standalone wrappers with
production template constants and placeholder strides. MLX wrapper compilation,
runtime compatibility and numerical correctness have not been verified on M5.

## Reproduction on M5

Use an environment containing the repository dependencies, MLX 0.32.2,
pytest and Ruff on macOS >=26.2. Run from the repository root:

```bash
PYTHONPATH=. python -m pytest -q tests/test_qsa_m5_gate.py tests/test_qsa_m5_nax.py
PYTHONPATH=. python -m pytest -q tests/test_qsa*.py tests/test_qwen4*.py tests/test_bench_qwen4_fused_gdn.py
PYTHONPATH=. python scripts/bench_qwen4_qsa_m5.py \
  --tokens 4096 16384 65536 --chunk-size 512 --warmups 3 --runs 9
```

The default benchmark loads no models. It reports scoring, radix selection,
combined stage one, current block-sparse vs tensor-unit attention, and complete
synthetic QSA-layer prefill with projection/normalization/RoPE/cache/selection/
gating/output included. Synthetic hidden width is 256; attention and indexer
head geometry is production-sized. Layer timings cover the entire prompt in
512-token chunks, starting from an empty cache, with all other opt-ins disabled.
Per-kernel timings use a trailing 512-token chunk; per-layer timings are not
full-decoder timings. Compilation is excluded by warm-up.

Full-model TTFT explicitly requires a locally available checkpoint whose config
identifies `qwen4_exp`; this prevents an unrelated model from producing a false
receipt. This mode may load a large model and must only be run on a suitable
M5 host with adequate memory:

```bash
PYTHONPATH=. python scripts/bench_qwen4_qsa_m5.py --mode ttft \
  --model-path /path/to/local/quantized-qwen4-exp \
  --tokens 4096 16384 65536 --chunk-size 512 --warmups 1 --runs 5
```

TTFT alternates flag-off/on order, creates fresh generation caches, uses exact
prompt token counts, excludes model loading, reports first-token IDs and route
construction deltas, and refuses candidate timings with no route activity.
The timing includes the model's whole prompt-to-first-token computation.
Inspect first-token agreement and retain hardware, checkpoint revision,
quantization, chunk size and runtime metadata with receipts. Token agreement
alone is insufficient quality evidence; no production rollout follows from
synthetic timings.

## Local validation and handoff

The task host is M3 Ultra, but the restricted execution session exposes no Metal
device: importing `mlx.core` fails with `metal::load_device`. A combined pytest
invocation subsequently aborts on another import of the partially initialized
extension. The 35 device-independent admission tests pass; the runnable CPU subset totals
40 passed and one existing artifact-test skip. Both shader bodies also pass an
offline Metal 4 compilation check (this does not dispatch a GPU kernel). GPU numerical and
integrated model fallback tests cannot execute in this session. Record actual
pass/skip/error totals in the task completion report, rather than treating
collection failures as passing GPU checks.

Vector's next action is the M5 parity run above, followed by synthetic timing
and a suitable quantized checkpoint TTFT/quality run. Harbor should retain
commit-bound receipts before any claim of qualification. Atlas owns any later
compatibility/default decision. There is no release authorization in this task.

## Round 2: M5 test harness correction

The supplied M5 Max receipts use MLX 0.32.4, `applegpu_g17s`, and macOS
27.0.1. The new GPU module reports 11 passed and 14 failed; together with the
35 gate tests this is 46 passed and 14 failed. All 14 failures stop at NumPy
conversion before their numerical assertions: 12 attention cases (offsets 0,
2051, 4099 crossed with contiguous, strided, misaligned, capacity-cache views),
the integrated non-M5 fallback, and integrated M5 prefill parity. NumPy rejects
the MLX bf16 PEP 3118 buffer; applying NumPy's `astype` afterward never runs.

Those outputs now widen to FP32 **inside MLX before** `np.asarray`, matching
the existing FP32 oracle conversion and other sparse-kernel tests. Both
benchmark attention-output conversions receive the same correction, outside
the timed functions. Widening preserves every finite bf16 value exactly; it
does not alter kernel arithmetic, outputs, thresholds or selected IDs. The
fallback comparison remains exact equality. The integrated tests explicitly
disable the other QSA opt-ins to keep the default baseline reproducible; the
M5 test also checks that both route counters stay zero with the flag off and
advance with it on. Fresh caches and identical weights/inputs are retained.

The regression receipt reports 18 failed, 238 passed, 3 skipped versus the
base receipt's 4 failed, 192 passed, 3 skipped. The four existing failures have
identical test IDs in both runs; the only added failures are the 14 conversion
errors above. The passing score/selection/tie cases provide M5 execution
evidence, but attention and integrated parity remain unqualified until the
corrected assertions execute on M5. No kernel defect is demonstrated by these
logs, and no kernel change or tolerance relaxation is justified by them. All
existing numerical acceptance thresholds remain unchanged. Vector's next
action is to rerun both M5 commands above using MLX 0.32.4 and retain the
post-correction receipts; Harbor owns the harness correction and reporting.
