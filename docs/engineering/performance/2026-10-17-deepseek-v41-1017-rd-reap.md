# DeepSeek V4.1 Flash 1017: rate-distortion REAP tooling

2026-10-17. Owner: Vector, Studio. Branch: `vector/dsv41-reap`, base
`ca5093710`. Phase 1 is offline tooling and synthetic tests only. No model
weights were loaded, GPU workloads launched, datasets fetched, checkpoint
packed, or commits/pushes/PRs made. All changes remain available for review.

## Artifacts and current limits

- `scripts/calibrate_deepseek_v41_reap.py`: routed output REAP, MAN, MSAN,
  per-domain/half arrays, counts, stability, corpus SHA256 and source revision.
  The existing checkpoint-code trust flag remains mandatory. `--native` uses
  the repository's native loader instead; the default retains the existing
  explicitly trusted bundled runtime adapter. No remote code is fetched.
- `scripts/deepseek_v41_expert_quant_error.py`: CPU-only native dequantization
  and affine quantization, Gaussian expert-output probes, resumable NPZ, and
  a source-header byte inventory. Only one expert's three projections are
  materialized. The existing repacker's `CheckpointIndex` is reused, with FP8
  dtype byte sizes added. It never calls whole-shard `mx.load`.
- `scripts/deepseek_v41_rd_allocate.py`: a guarded Lagrangian allocation,
  discrete marginal-step repair, byte totals and a duality-gap certificate;
  decimal-GB budget sweep. Allocation JSON includes original IDs, compact
  router IDs and bank-local IDs.
- `scripts/evaluate_deepseek_v41_rd.py`: native mmap reference/candidate logit
  collection, pruning masks, full-vocabulary KL and top-1 disagreement, retained
  routing mass, and fixed-threshold knee selection. Cached logits can be
  compared without MLX. Masks screen pruning only, not quantization.
- `scripts/deepseek_v41_rd_corpus_manifest.json`: sources, licenses, domain and
  low-resource language slices, disjoint splits, bounded-fetch/export policy.
  It is a manifest, not a downloader. Nothing has been written to the dataset
  volume. Owner prepares local exports before calibration.
- `scripts/deepseek_v41_rd_common.py` and `tests/test_deepseek_v41_rd.py`:
  common numerical contracts and synthetic regression checks, including the
  bank slicing/remapping design.

The cached source config resolves to revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Its safetensors index was absent
when inspected. Only config/directory metadata were read; no source weight
shards or shard headers were read in phase 1. Exact non-expert bytes **have not
been measured**. The inventory command below requires the completed index and
shards. Do not fabricate a fixed-byte number from total model parameter counts.

## Scoring and distortion

For expert e and domain d, record unweighted routed outputs f_e(x_t):

```
R_e,d = sum_{t routed to e} g_e,t ||f_e(x_t)||_2
MAN_e,d = sum_{t routed to e} ||f_e(x_t)||_2 / N_e,d
MSAN_e,d = sum_{t routed to e} ||f_e(x_t)||_2^2 / N_e,d
s_e = max_d score_e,d
D_e(b) = s_e q_e(b),       b in {0, 2, 3, 4},       q_e(0) = 1
```

Here g is the normalized selected unbiased gate score, before the common 1.5
routing multiplier. REAP uses the requested **sum**. All slices have the same
number of tokens; unequal-length REAP corpora must be resampled to equal token
budgets. MAN/MSAN are routed-token means, not all-token or all-expert means.
Unobserved experts get zero with explicit counts, not an invented activation.
The definitions follow [MAN/MSAN section 4.4](https://arxiv.org/html/2606.15716#S4.SS4).
No all-expert token-subsample option is needed for these definitions.

Domain maxima retain absolute layer magnitudes. Normalizing each layer to sum
one would destroy cross-layer sensitivity. Save raw per-domain scores and the
two temporal halves; require keep-set overlap >=0.90 in every domain and each
of the three scores before allocation. This threshold is a stability check,
not proof of coverage. Zero/low routed counts require a longer or better
calibration corpus; identical zero-score tie sets can look spuriously stable.
Domains run with fresh inference state. The half split is per domain, not
across the concatenation of domains.

Data-free q uses common deterministic N(0,I) hidden probes (default 16, seed
1017). Dequantize FP4/FP8 to BF16 using the native functions; quantize all three
projections jointly at b bits, then evaluate the clamped SwiGLU expert:

```
E(x) = W2 [silu(min(W1 x,10)) * clip(W3 x,-10,10)]
q_e(b) = sum_x ||E_e(x) - E_e,b(x)||_2^2 / sum_x ||E_e(x)||_2^2
```

Probe arithmetic and the ratio use FP32; BF16 source and restored weights
match the converter storage type. Zero/nonfinite reference energy fails
closed. This measures **output** reconstruction, not a renamed weight MSE.
Synthetic hidden inputs may miss real activation geometry; repeat a subset
with more probes/seeds before spending the GPU qualification budget. The
[HIGGS linearity work](https://arxiv.org/abs/2411.17525) motivates additive
reconstruction-error allocation. Extending it to nonlinear experts and full
removal is a surrogate requiring held-out qualification, not a theorem that
pruning has no interactions. [REAP](https://arxiv.org/abs/2510.13999) motivates
using routed expert output saliency.

## Allocation and integer-budget behavior

Minimize sum D_e(b_e) subject to F + sum C_e(b_e) <= B, at least 288 kept
experts per layer by default, and never fewer than top-6. Shared experts are
outside the allocation and remain fixed. Routers retain their unbiased-score
normalization and selection-bias semantics.

For lambda >=0, minimize D+lambda*C for each expert. In each layer, if fewer
than the minimum choose a nonzero bit-width, retain the experts with smallest
incremental penalized keep cost. This solves the layer guard exactly at that
lambda. Bisect lambda to find a feasible allocation, then spend remaining bytes
on feasible positive-benefit upgrades in decreasing benefit/byte order,
including 0->b and skipping dominated intermediate widths. Unsupported choices
can be selected by integer repair; q need not be monotonically decreasing.

Discrete multiple-choice allocation is not generally solved exactly by greedy
slopes. JSON reports a feasible resident byte count, predicted distortion,
Lagrangian lower bound, lambda and duality gap. A zero gap certifies the
allocation; a positive gap is disclosed, not called optimal. Synthetic tests
compare lower bounds and certified optima against exhaustive enumeration.
Ties are deterministic. Infeasible budgets fail explicitly. The sweep does
not silently relax the layer floor.

## Byte arithmetic (decimal GB, not GiB)

`convert.py:333` defaults `group_size=64`; `quantize_tree` uses MLX affine
triples. Each expert has three matrices with 5120*2304 values each, totaling
35,389,440 parameters. With BF16 scale and affine bias per 64 weights:

```
C_weight(b) = 35,389,440 * (b/8 + 4/64)
```

| Bits | Expert weight bytes | All 15,360 experts (GB) | 11,520 experts / 288 per layer (GB) |
| --- | ---: | ---: | ---: |
| 2 | 11,059,200 | 169.869312 | 127.401984 |
| 3 | 15,482,880 | 237.8170368 | 178.3627776 |
| 4 | 19,906,560 | 305.7647616 | 229.3235712 |

Charge router rows to each kept expert, not F: a BF16 row plus two FP32 biases
adds 5120*2+8 = 10,248 bytes. The header inventory uses actual stored router
dtype; the examples assume BF16 gate weights. With these rows all-2-bit costs
170.02672128 GB. Thus all experts fit 200 GB whenever fixed non-expert F is
<=29.97327872 GB. A 2->3 or 3->4 upgrade costs 4,423,680 bytes.

For illustration **only**, if the measured F were 10 GB:

- Keep all 15,360: 10,845 at 2 bits and 4,515 at 3 bits cost
  199.99963648 GB including F and router rows.
- Keep exactly 288 per layer (11,520 total): 8,917 at 3 bits and 2,603 at
  4 bits cost 199.9956736 GB; 3,840 experts are removed. Their layer placement
  must obey the floor. This is a capacity example, not a recommended recipe.

F includes the shared experts, embeddings/head, attention, norms, hyper
connections, small Engram projections, and all other retained text tensors.
The inventory follows converter eligibility with fixed non-expert precision
(default 4 bits). Vision and MTP are omitted; router rows are variable costs.
Only the giant `engram.embed` weight/scale payloads in layers 1 and 14 are
excluded, because they stay SSD backed. They are not pruned or allocated bits.
Native source Engram payloads would occupy
(384006168+384016682)*256*(1+1/32) = 202.7580324 GB on disk; the inventory
reports exact stored bytes, and the existing serving artifact may instead use
affine Engram. SSD file size is separate from the resident-weight budget.
Tensor payload excludes safetensors JSON/container bytes and includes affine
metadata. Cache, gathered Engram rows, GPU scratch, allocator remaps, and OS
memory require headroom beyond the <=200 GB resident-weight target.

## Mixed-bit feasibility verdict and packing design

**Not directly supported; stop at design in phase 1. No mixed-bit repacker or
production engine changes were implemented.** The minimal useful change spans
more than a safe local packing adjustment:

1. `moe.py:107-125`: each layer constructs one full-count `SwitchGLU` and calls
   it once. Add per-layer bank definitions and a dispatcher mapping compact
   router IDs to (bank bit-width, local ID). SharedExpert stays independent.
2. Installed `mlx_lm/models/switch_layers.py:57,76-84`: a projection has a
   scalar `.bits` and passes it to `gather_qmm`. A single stacked projection
   cannot hold different packed widths. Use separate 2/3/4-bit SwitchGLU banks.
   No new quantization kernel is necessary for a correctness-first prototype.
3. `load.py:249-259` replays per-module quantization, but the bank modules must
   exist *before* that replay. `ModelArgs.n_routed_experts` is currently global;
   varying per-layer keep counts need explicit local shapes. Strict missing
   and unexpected key validation (`load.py:333-345`) must understand bank keys.
4. `affine_route_qmv.py:119-143` accepts only bits=2/group=64 and its Metal
   source decodes 2-bit bytes. `serving.py:77-84` unconditionally swaps every
   SwitchGLU into that path. Bank-aware serving must install QMV only on
   supported 2-bit banks and use stock gather_qmm for 3/4-bit banks. Never
   reinterpret their bytes through the 2-bit kernel.
5. `load.py:347-359` requires one fused MoE per layer; `moe_fusion.py:103-127`
   assumes a compatible gate/up pair. Disable fusion initially or make it
   bank-aware, and update the layer/bank count check.

Proposed artifact: preserve source expert IDs in allocation metadata. Slice
router weight, bias **and bias_vl** in increasing kept-ID order. Store
`layers.L.ffn.expert_banks.b{2,3,4}.{gate,up,down}_proj.{weight,scales,biases}`,
with each bank's expert axis in increasing original-ID order. Store the compact
router-to-bank/local lookup in config. At runtime route once across all kept
experts; partition selected routes into banks, gather corresponding token rows,
run homogeneous bank kernels, scatter outputs back to original token/top-k
positions and apply the original weights once. Never renormalize per bank.
Empty banks are absent; unused banks must not execute dummy routes. Preserve
clamps, FP32 accumulation, dtype casts, and shared-expert addition. Validate
single-token decode, prefill, route sorting/inverse permutation, empty bank
selections, cancellation and strict reload before packing real weights.
`bank_plan` and tests define/verify the slicing and bijection, not a loadable
artifact. Vector owns this next engine change; Atlas owns compatibility scope.

## Held-out reference and knee rule

The implemented collector uses `load(lazy=True, engram_ssd_offload=True)` and
single-token teacher forcing with fresh caches per slice. It deliberately does
not install the serving-specific 2-bit QMV or fusion. Full logits are written
as FP32 `.npy` files and freed token by token; 128 tokens at vocabulary 129,280
cost 66,191,360 bytes per slice. Reference and candidate run in separate
processes, never alongside each other in memory. Compare mmaps in chunks of
8 tokens, computing stable FP64 log-softmax KL(reference||candidate) in nats.
Top-1 ties use NumPy's deterministic first-index rule. Corpus/tokenizer/logit
hashes and exact teacher-forced token IDs are checked, so stale caches fail.

Reference must be **unpruned**. For source-faithful expert comparison use a
converted native artifact retaining source-dequantized BF16 experts, with the
same SSD Engram policy as the candidate. An unpruned affine checkpoint is only
a proxy reference and must be labeled as such; it cannot certify loss relative
to the release. The raw FP4 source cannot be passed to the native loader.
The existing bundled disk runtime is an alternative for a small source-faithful
reference, but no stable logits API for that external runtime has been verified
here. A future adapter can populate the same cache contract without a new
weight loader.

Explicit mmap laziness reuses the existing streaming path; it is not proof of a
bounded 256 GB peak for an unpruned BF16 reference. First qualify a tiny held-out
set on the owner host, monitor peak/pressure and Metal watchdog behavior. If
this fails, the existing split-expert disk runtime needs a source-faithful logit
adapter before quality certification. Do not silently substitute a pruned or
lower-precision teacher. Native SSD Engram support currently requires affine
tables (`load.py:310-311`); raw FP8 tables cannot be offloaded by that loader.

Masks set removed router biases to -infinity before top-k, leave weights
unchanged, and renormalize retained unbiased scores. Retained routing mass is
the fraction of original top-k weight landing on kept IDs, measured at each
candidate's current hidden states and averaged over tokens/layers. It is not
full dense score mass or a reference-trajectory measurement. Report domain
means separately. These screens isolate pruning; after bank packing, repeat
with actual mixed-bit weights and a source-faithful teacher, including fixed
non-expert quantization error.

Fix worst-domain mean token KL threshold at **0.02 nats** before examining the
sweep. This is an engineering acceptance threshold, not a literature guarantee.
Sort budgets descending. Define the knee as the smaller endpoint of the
largest positive rise in worst-domain KL per GB removed. Choose its immediately
larger tested neighbor (one step on the safe side); if that neighbor fails the
threshold, move to the next larger passing point. If no safe point passes,
reject the sweep. At least three distinct budgets are required; no positive
rise means no knee was detected. Do not raise the threshold after seeing results.
A selected budget above 200 GB is a concrete failure to meet the deployment
target, not permission to consume more memory. Require final packed mixed-bit
results, all domain gates, and measured peak memory before qualification.

## Owner command sequence (later, not executed in phase 1)

Run from this worktree on the M3 Ultra. Dataset preparation is intentionally
manual from the manifest: pin sources/licenses, export disjoint calibration
and held-out files under the dataset volume, then provide manifests. Model
preparation is a separate prerequisite: supply the source-faithful unpruned
native artifact described above. Do not point it at the raw source directory.
Set an existing native reference path; generated score/logit files stay in the
allowed scratch root. These commands perform no dataset/model downloads.

```sh
cd /Users/raullenstudio/orca/workspaces/rapid-mlx/dsv41-reap
PY=.venv/bin/python
OUT=/private/tmp/rapid-mlx-dsv41-1017
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
SRC="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/$REV"
CAL=/Volumes/RTL-2T/datasets/deepseek-v41-1017-rd/calibration.json
HELD=/Volumes/RTL-2T/datasets/deepseek-v41-1017-rd/heldout.json
: "${DSV41_NATIVE_REFERENCE:?Set an existing unpruned native reference with SSD affine Engram}"
REF="$DSV41_NATIVE_REFERENCE"

# Phase-1 sandbox used a CPU-only MLX build. On the GPU host restore the
# supported Metal wheels before GPU steps (the existing uv cache is reusable).
UV_CACHE_DIR="$OUT/uv-cache" uv pip install --python "$PY" --reinstall --no-deps \
  'mlx==0.32.4' 'mlx-metal==0.32.4'

# CPU phase after download completion: header inventory, then resumable errors.
"$PY" scripts/deepseek_v41_expert_quant_error.py --source "$SRC" \
  --model-revision "$REV" --inventory "$OUT/costs.json" --inventory-only
VECLIB_MAXIMUM_THREADS=2 "$PY" scripts/deepseek_v41_expert_quant_error.py \
  --source "$SRC" --model-revision "$REV" --inventory "$OUT/costs.json" \
  --output "$OUT/errors.npz" --probes 16 --seed 1017 --resume

# GPU calibration: all slices in one invocation; inspect counts and stability.
"$PY" scripts/calibrate_deepseek_v41_reap.py --model "$REF" --native \
  --model-revision "$REV" --slices "$CAL" --tokens 8192 --keep-experts 288 \
  --trust-checkpoint-runtime --output "$OUT/scores.npz"

# CPU allocation: repeat with --score man / msan for score-choice sensitivity.
"$PY" scripts/deepseek_v41_rd_allocate.py --scores "$OUT/scores.npz" \
  --errors "$OUT/errors.npz" --costs "$OUT/costs.json" --score reap \
  --budgets-gb 260 240 220 200 185 --min-kept 288 --topk 6 \
  --min-split-overlap .90 --output-dir "$OUT/allocations"

# GPU teacher cache (one process). Start with --tokens 8 if streaming is unqualified.
"$PY" scripts/evaluate_deepseek_v41_rd.py collect --model "$REF" \
  --model-revision "$REV" --slices "$HELD" --tokens 128 --output "$OUT/reference"

# Separate GPU processes; screens prune masks only, reusing the unpruned weights.
for B in 260 240 220 200 185; do
  "$PY" scripts/evaluate_deepseek_v41_rd.py collect --model "$REF" \
    --model-revision "$REV" --slices "$HELD" --tokens 128 \
    --allocation "$OUT/allocations/allocation-${B}GB.json" --output "$OUT/mask-${B}GB"
  "$PY" scripts/evaluate_deepseek_v41_rd.py compare --reference "$OUT/reference" \
    --candidate "$OUT/mask-${B}GB" --output "$OUT/metrics-${B}GB.json"
done

# Assemble the measured sweep and choose the safe side under the fixed gate.
"$PY" - <<'PY'
import json
from pathlib import Path
out = Path('/private/tmp/rapid-mlx-dsv41-1017')
points = [dict(budget_gb=b, **json.loads((out / f'metrics-{b}GB.json').read_text()))
          for b in (260, 240, 220, 200, 185)]
(out / 'sweep.json').write_text(json.dumps(points, indent=2) + '\n')
PY
"$PY" scripts/evaluate_deepseek_v41_rd.py select --points "$OUT/sweep.json" \
  --worst-domain-kl-threshold .02
```

Stop before packing: no packing command is available until the bank engine,
loader and serving guards above are implemented and tested. Repeat the same
held-out cache/compare procedure on actual packed weights after that work;
pruning-mask selection alone is insufficient to pick the final mixed-bit KL
knee. Record source/derived artifact revisions, corpus hashes, MLX versions,
hardware, token counts, per-domain metrics, duality gap, and measured peak
memory in the owner run's evidence.

## Verification and handoff

Focused synthetic tests cover domain aggregation, gate-free means, byte
payloads, exhaustive allocation optima/certificates and guards, bank/router
slicing and remapping, biased selection/unbiased weights, KL direction,
cache integrity and knee gating. Native probes and quantization run on CPU,
not Metal. The worktree venv was created with `uv venv .venv` and
`include-system-site-packages=false`.

Validation: **37 passed** on the worktree venv, CPU-only MLX 0.32.4,
MLX-LM 0.31.3, NumPy 2.5.4, Python 3.13.15. Command:

```sh
.venv/bin/python -m pytest -q \
  --basetemp=/private/tmp/rapid-mlx-dsv41-1017/pytest-tests \
  tests/test_deepseek_v41_rd.py tests/test_deepseek_v41_quant_error.py \
  tests/test_calibrate_deepseek_v41_reap.py tests/test_repack_deepseek_v41_native.py
```

The initial Metal wheel could not import in this sandbox (`No Metal device
available`). A 4.4 MB official MLX v0.32.4 source archive was built under the
allowed scratch root with `CMAKE_ARGS="-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON
-DMLX_BUILD_DISTRIBUTED=OFF"`, `PYPI_RELEASE=1`, and
`CMAKE_BUILD_PARALLEL_LEVEL=4`, then installed only into `.venv`. This allowed
real MLX affine 2/3/4-bit and native tiny-MoE tests without GPU access or stubs.
The owner GPU commands restore the published backend first. Ruff and
`git diff --check` pass. Tests also exercise native calibration metadata across
two tiny domains, allocation CLI provenance/cost checks, FP4/FP8 streaming,
partial error checkpoints, and resume provenance rejection.

Team handoff (Orca messaging unavailable): Vector retains ownership of bank
implementation and the missing source index/reference prerequisites. Atlas
receives compatibility and scope decisions; Harbor receives owner-run memory
and throughput evidence; Echo tracks future review; Pixel has no UI change.
No rollout or publication is authorized by these tooling results.
