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


| Fusion enablement knob | Fallback | Installer scope and local requirements | Universal builtin policy |
| --- | --- | --- | --- |
| `dense_decode_fusion` | `False` | Explicit `dense_decode_fusion_install` fact and plain mode; existing backbone/decoder allowlists, width ≤4096, dtype/cache/composition probes | `True` |
| `qwen35_affine4` | `False` | Explicit `qwen35_affine4_install` fact; exact expert classes, packed-bank layout, installed router and real-weight probe | `True` |
| `qwen35_moe_router` | `False` | Explicit `qwen35_moe_router_install` fact; existing classes, expert/top-k bounds and parity probe | `True` |
| `moe_decode_fusion` | `False` | Explicit `moe_decode_fusion_install` fact; existing block class and per-call single-row, alignment, dtype/bias/sharding checks | `True` |
| `moe_gate_up_fusion` | `False` | Explicit `moe_gate_up_fusion_install` fact; existing expert classes, basic/extended call contracts and compatible projection formats | `True` |

These five contiguous enrollment cells have no hardware or software selectors:
unknown and future devices retain the historical default-on attempts. The
`model.family` facts above identify existing installer entry points, separately
from checkpoint declarations; missing installer facts decline, and dense also
requires explicit plain mode. They do not certify implementation safety.
Installers retain their exact local qualification and loader routes, including
plain/MTP, vision and disk-streaming exclusions. Affine4 bank-view and row-tile
policies remain independent and unchanged.

`lattice.fusion.install_enabled` resolves once per installer invocation, before
module traversal. Existing installed tags, rewritten projections and model-local
classes carry the result into execution: no forward, eligibility predicate or
per-layer function gains a policy call. Profiles retain their process lifetime.
Legacy switches keep their original parsing and lifetime: dense and gate/up
check exact `"0"` at installation; router checks falsey values at installation;
affine4 and MoE decode check falsey values at installation and dynamically in
forward/eligibility. Thus a live opt-out still works where it did before, while
installation-only switches do not acquire dynamic semantics.


| MTP lookup knob | Default | Scope and requirement | Builtin policy |
| --- | --- | --- | --- |
| `mtp_lookup_max_copy` | `31` (integer: 31, 63) | Singleton MTP with installed head and existing lookup qualification; proposal ceiling and adaptive gate growth limit | `63` for M5 Max 36 GB, Qwen3.8-27B (hidden 5120) affine4/group64, MLX 0.32.4, macOS 27.0.1 ([evidence](2026-11-17-m5-mtp-lookup.md)) |

The lookup knob resolves once per lookup-enabled request. The environment
override is `RAPID_MLX_MTP_LOOKUP_MAX_COPY=63`; benchmark JSON and overlay
precedence follow the usual Lattice rules. Without a qualified selection the
family's existing copy limit is preserved. The index contains prompt tokens only.

The existing gate still prices and sizes copies: 63 can be earned by full
31-token copies. Cache rollback admission and remaining output budget can
further narrow proposals. The target verifies every copied position. The
qualified cell uses model family, geometry and installed quantization because
the schema has no checkpoint-name key; other hardware/software/model facts
retain their existing limits.

At INFO, `[MTP lookup]` logs one request summary (including generator close):
`proposals`, `copied_proposed`, `copied_accepted`, and `verify_widths` (rows,
including the base row). Acceptance counts the verified prefix capped at EOS
and output budget; a cancelled consumer may have received fewer tokens.
Existing timing diagnostics retain `prompt_lookup_proposals`,
`prompt_lookup_drafted_tokens`, `prompt_lookup_accepted_tokens`, and now
`prompt_lookup_verify_width_<rows>`. No token contents are logged.

### Short-request scheduling

| Knob | Default | Candidates | Environment override | Builtin policy |
| --- | --- | --- | --- | --- |
| `prefix_boundary_min_tokens` | `0` | `0`, `128`, `256`, `512` | `RAPID_MLX_PREFIX_BOUNDARY_MIN_TOKENS` | `256` for singleton MTP on M5 Max 36 GB, Qwen3.8-27B (hidden 5120) affine4/group64, MLX 0.32.4, macOS 27.0.1 ([evidence](2026-11-17-m5-short-request-ttft.md)) |
| `admission_window_ms` | `8.0` | `0`, `8` | `RAPID_ADMISSION_WINDOW_MS` | `0` for the same measured batch-1 MTP cell ([evidence](2026-11-17-m5-short-request-ttft.md)) |

The measured cell uses family, geometry and installed quantization because
the schema has no checkpoint-name key. Live text args omit intermediate size,
so it is not a key. Other hardware/software/model facts and batch sizes retain
the defaults above. The prefix threshold accepts nonnegative
integers; the admission window accepts finite nonnegative numbers. Resolve
against the loaded model, device and active plain/MTP lane at the consumer.

When the uncached prompt length is **strictly below** the prefix threshold,
non-trimmable text requests skip their message-boundary split and snapshot.
This applies to the text/MTP scheduler and the MLLM generator's text-only lane
through shared policy code. Trimmable models, media requests, prefix-cache
fetches, shared-prefix snapshots, normal chunk limits, and completion-cache
stores retain their existing behavior. Zero preserves the existing splits.
The final-token forward remains separate where the normal path requires it.

Skipping the boundary can cost the next turn fewer than `min_tokens` extra
prefill tokens (at most `min_tokens - 1` for integer token counts), relative to
reusing that boundary with the same retained earlier prefix. Eviction or a
changed earlier prefix can independently require more work. With the measured
threshold 256, that policy tradeoff is at most 255 tokens.

The admission window resolves with the waiting wave size before each fresh-wave
hold; `0` disables the hold, and the default retains the 8 ms tick and existing cap. The legacy
environment variable retains its parsing and wins over the Lattice selection,
including benchmark JSON; invalid/nonfinite values fall back to the selected
window. Negative legacy values still disable the hold. Benchmark JSON wins
over the prefix-threshold environment override under normal Lattice precedence.
Disabling the admission hold can reduce burst alignment and batched throughput.
