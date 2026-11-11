# Measured Rapid Lattice policy

The current policy contract is the [Rapid Lattice decision](../decisions/rapid-lattice.md).
See [singleton qualification evidence](2026-11-09-m3-singleton-backbone.md) for the original per-chip measurements.

| Knob | Default | Scope and requirement | Builtin policy |
| --- | --- | --- | --- |
| `qwen35_eager_layer_dispatch` | `False` (boolean) | Existing `qwen3_5.DecoderLayer.__call__` seam, installed rank-one norm/expert/top-k facts, outside compiled decode | Universal hardware; norm width 2048, 256 experts, top-k 8, flattened rows 0–64; [existing evidence](2026-09-13-qwen35-eager-layer-dispatch.md) |
| `qwen35_fused_gdn_decode` | `False` (boolean) | Existing GDN classes and loader `allow_dense` distinction; projected geometry, cache/sharding checks and full parity probe stay local | Universal hardware; hidden 2048 on legacy routes, positive integer hidden widths on existing dense opt-in routes |
| `gdn_in_proj_fusion` | `False` (boolean) | Existing installation routes, compatible projection formats and projection/layer parity; split retention stays caller-owned | Universal hardware; enabled where installation was already attempted |
| `gdn_in_proj_max_rows` | `0` (nonnegative integer) | Installed fusion, qualified dtype; capped by the local eight-row probe envelope | Universal hardware; 8 |
| `metal_cache_floor_bytes` | 2 GiB (positive integer) | Metal free-cache limit at engine setup | Universal hardware; 2 GiB |
| `metal_cache_cap_bytes` | 32 GiB (positive integer) | Metal free-cache limit at engine setup | Universal hardware; 32 GiB |
| `metal_cache_divisor` | 4 (positive integer) | Metal free-cache limit = soft allocation limit divided by this, then floored/capped and clamped to the soft limit | Universal hardware; 4 |
| `memory_pressure_margin` | 0.05 (number 0–1) | Engine-core eviction threshold: added to `gpu_memory_utilization` | Universal hardware; 0.05 |
| `memory_pressure_ceiling` | 0.99 (number in (0, 1]) | Upper bound on utilization plus margin | Universal hardware; 0.99 |
| `memory_pressure_fallback_bytes` | 200 GiB (positive integer) | Device size when Metal reports none, and threshold when probing fails | Universal hardware; 200 GiB |
| `memory_check_interval` | 16 (positive integer) | Busy-step memory check cadence | Universal hardware; 16 |

Memory knobs resolve once per process at engine setup into a frozen policy;
any resolution failure keeps the constants above. The engine loop never
resolves them per step.

The eager hook retains its import-time `RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH`
falsey opt-out as an outer master gate. Lattice candidate overrides are captured
at hook installation; overlay and builtin profiles load once per process.
Decisions cache actual installed geometry and exact flattened row counts (bounded
4096-entry cache), preserving arbitrary overlay row boundaries. Cache hits do
not load profiles or resolve policy. Missing geometry facts retain stock scheduling.


GDN policy reads `mx.device_info()` once per adapter (failure yields unknown
facts), captures candidate overrides on first policy use, and caches decisions
in a bounded 256-entry cache. Universal cells include unknown and future devices;
chip-specific overlays use actual device facts. The decode context records the
existing legacy/dense caller route and installed hidden width; missing widths
cannot authorize decode. Existing loader calls and the separate MTP
`mtp_gdn_fusion` gate remain authoritative.

Legacy environment checks stay dynamic at their original call sites: decode
uses the shared falsey parser, while projection installation disables only for
exact `"0"`. The projection row knob can narrow dispatch but cannot exceed the
unchanged parity-probed row/dtype envelope. Probes still cover all eight rows.
No new numerical or hardware qualification is claimed.

| Attention threshold knob | Default | Scope | Universal builtin policy |
| --- | --- | --- | --- |
| `affine8_attention_min_kv` | `0` (nonnegative integer) | Query width 1–8 | Physical KV ≥16384 for width 1; ≥8192 for widths 2–3; ≥2048 for widths 4–8 |
| `qsa_stage1_min_query` | `0` (nonnegative integer) | Existing stage-one route | Query length ≥64 |
| `qsa_stage1_min_kv` | `0` (nonnegative integer) | Existing stage-one route | Physical KV ≥65024 |
| `qsa_indexed_splitk_min_kv` | `0` (nonnegative integer) | Qualified query widths 1 and 3 | Physical KV ≥65536 for width 1; ≥16384 for width 3 |
| `qsa_block_sparse_min_query` | `0` (nonnegative integer) | Existing block-sparse route | Query length ≥64 |
| `qsa_block_sparse_min_kv` | `0` (nonnegative integer) | Existing block-sparse route | Physical KV ≥16384 |

These cells transcribe existing performance crossovers; they contain no hardware
or software selectors. Affine8 uses eight exact `workload.width` cells, reusing
the existing selector without adding a range representation. Thresholds are
independent of kernel qualification: `affine8_attention` and `_qualified_runtime`
still authorize the runtime before the threshold is read. Universal affine8
thresholds therefore cannot enroll a device. The padded-row ceiling, packing,
dtype, mask and layout checks remain local and unchanged.

The QSA architecture/version allowlists and existing enable-knob scopes remain
authoritative. All three QSA routes retain their dynamic environment opt-in,
training exclusions and Metal checks. In particular, registering block-sparse
thresholds supplies no enabling cell or enable knob. Threshold candidates can
refine crossovers only inside these existing gates.

Each attention module lazily captures real device facts (`mx.device_info()`
once for threshold policy), software facts through `runtime_context`, and an
environment snapshot on first threshold use. Existing qualification functions
retain their independent caches and lifetimes. Threshold decisions cache the
knob name and exact query width in a bounded 256-entry cache per module; warm
hits perform no threshold resolution or profile loading. Profiles retain their
process lifetime. Hardware/software-specific overlays can refine thresholds
using those real facts. Legacy enable switches keep their original lifetimes;
QSA enable-knob resolution remains at its existing call site.
