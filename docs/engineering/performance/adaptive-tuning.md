# Measured device tuning

`rapid_mlx/tuning` separates performance policy from correctness qualification.
The first two knobs are singleton affine-4 bank views and plain residual/norm
carry. Each registry entry describes its family, geometry, allowed serving
modes, widths, conservative default, and existing environment override.
The geometry tuple is hidden size, expert intermediate size, expert count,
selected expert count, quantization bits, and quantization group size.
The residual knob additionally requires the caller's 40-layer BF16 backbone
predicate. Those existing predicates, real-output/cache parity probes, and
fallbacks remain the authority for executing either optimization.

Resolution is:

1. Reject an unsupported scope, regardless of environment or measurements.
2. Honor an explicit boolean override: false wins over a measured-on default;
   true requests a trial even on an unmeasured device.
3. Look up the exact device variant and installed MLX version in the checked-in
   measurement table.
4. Use the knob's conservative default (both initially off).

The existing `RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS` and
`RAPID_MLX_QWEN35_RESIDUAL_NORM` names are retained. They accept the shared
boolean spellings (`0/false/no/off/disable/disabled` and
`1/true/yes/on/enable/enabled`, ignoring case and surrounding whitespace).
Unset, empty, or unrecognized values defer to the table. Set overrides before
loading the model. Both kill switches also work after enrollment; a truthy
residual override requires reloading to enroll a previously declined model.
No new environment variable is introduced. Overrides never bypass probes,
geometry, cache restrictions, or the residual knob's plain-only restriction.

Device keys come solely from `device_info()['device_name']`, preserving family,
generation, and variant (`M3 Ultra`, `M4 Pro`, `M5 Max`). Other names use
`unknown`; recognized but unmeasured generations also fall back conservatively.
No guess is made from RAM or core count. The registry imports no MLX, performs
no device query at import, and reads MLX package metadata lazily once.
Callers supply the already available device information. Name parsing is
cached, but overrides are read on every decision so kill switches remain live.
The expert binding has the same policy for plain and MTP singleton calls;
wider MTP verification retains full banks. It does not infer serving mode from
model identity. The residual installer receives serving mode explicitly.

`defaults.py` records exact device/library keys, booleans, measurement date,
source commit, evidence document, rates and repetition counts. A library
upgrade without fresh evidence falls back to off. A measured-off entry is
retained as evidence rather than silently treating that chip as untested.
Timings describe the measured configurations; independent knob gains must not
be added together. See [qualification measurements](2026-11-09-m3-singleton-backbone.md).

## Later layers

A future persistent tuning cache should include schema and implementation
revision, device, library versions, model family/geometry/quantization, mode,
and workload width in its key. Never reuse entries across incompatible keys.
It can supply another policy source between explicit overrides and reviewed
static defaults. No runtime search, disk cache, device benchmark, or model
download occurs in this layer.

A later per-chip/model cost model can choose candidates using measured
prefill/decode cost and width, retaining the same qualification boundary.
Workload-adaptive MTP depth can consume target/draft timings, acceptance and
prompt/decode mix, rather than attach fixed depth to a chip name. It needs
stability bounds and end-to-end latency/throughput evidence before changing
defaults. Neither extension is implemented here.

## Follow-up inventory

A source scan of `RAPID_MLX_` switches under `rapid_mlx/` identified these
performance-policy candidates. They retain their existing behavior in this
change; each needs its own scope and evidence before migration.

| Area | Existing switches |
| --- | --- |
| Decode composition | `RAPID_MLX_COMPILED_DECODE`, `RAPID_MLX_DENSE_DECODE_FUSION`, `RAPID_MLX_MOE_DECODE_FUSION`, `RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH` |
| Qwen kernels | `RAPID_MLX_QWEN35_AFFINE4`, `RAPID_MLX_QWEN35_MOE_ROUTER`, `RAPID_MLX_QWEN35_FUSED_GDN_DECODE`, `RAPID_MLX_QWEN4_FUSED_GDN_DECODE`, `RAPID_MLX_QWEN4_FAST_RMSNORM` |
| Projection/prefill | `RAPID_MLX_MOE_GATE_UP_FUSION`, `RAPID_MLX_GDN_IN_PROJ_FUSION`, `RAPID_MLX_GDN_PREFILL`, `RAPID_MLX_AFFINE8_ATTENTION`, `RAPID_MLX_FP8_LM_HEAD_AFFINE8` |
| Attention tiles | `RAPID_MLX_QSA_STAGE1`, `RAPID_MLX_QSA_BLOCK_SPARSE`, `RAPID_MLX_QSA_INDEXED_SPLITK`, `RAPID_MLX_LANE_MATMUL`, `RAPID_MLX_MLA_ABSORBED_VERIFY` |
| Native execution | `RAPID_MLX_DSV41_FUSED`, `RAPID_MLX_DSV41_LANES`, `RAPID_MLX_DSV41_DRAFT_DEPTH`, `RAPID_MLX_STEP_WORKGROUP`, `RAPID_MLX_TENSORFOLD_LANES` |
| Sampling | `RAPID_MLX_DISABLE_FUSED_SAMPLER` |
| Workload speculation | `RAPID_MLX_MTP_PROMPT_LOOKUP`, `RAPID_MLX_MTP_PROMPT_LOOKUP_SAMPLED`, `RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM`, `RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM`, `RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS` |
| Storage/width tradeoffs | `RAPID_MLX_PROMPT_HOST_CACHE`, `RAPID_MLX_QWEN4_PLE_CACHE_BYTES`, `RAPID_MLX_QWEN4_PLE_NVME`, `RAPID_MLX_HYBRID_CHECKPOINT_MAX`, `RAPID_MLX_HYBRID_CHECKPOINT_STRIDE`, `RAPID_MLX_KV_CHECKPOINT_MAX_BYTES`, `RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS` |

The existing M4 Pro affine-4 row tile selection is another candidate even
though it has no environment switch. Authentication, routing, telemetry,
security limits, compatibility escapes and diagnostic switches are not
performance knobs merely because their names share the prefix.
