# Measured Rapid Lattice policy

The current policy contract is the [Rapid Lattice decision](../decisions/rapid-lattice.md).
See [singleton qualification evidence](2026-11-09-m3-singleton-backbone.md) for the original per-chip measurements.

| Knob | Default | Scope and requirement | Builtin policy |
| --- | --- | --- | --- |
| `qwen35_eager_layer_dispatch` | `False` (boolean) | Existing `qwen3_5.DecoderLayer.__call__` seam, installed rank-one norm/expert/top-k facts, outside compiled decode | Universal hardware; norm width 2048, 256 experts, top-k 8, flattened rows 0–64; [existing evidence](2026-09-13-qwen35-eager-layer-dispatch.md) |

The eager hook retains its import-time `RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH`
falsey opt-out as an outer master gate. Lattice candidate overrides are captured
at hook installation; overlay and builtin profiles load once per process.
Decisions cache actual installed geometry and exact flattened row counts (bounded
4096-entry cache), preserving arbitrary overlay row boundaries. Cache hits do
not load profiles or resolve policy. Missing geometry facts retain stock scheduling.

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
