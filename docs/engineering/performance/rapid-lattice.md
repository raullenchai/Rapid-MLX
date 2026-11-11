# Measured Rapid Lattice policy

The current policy contract is the [Rapid Lattice decision](../decisions/rapid-lattice.md).
See [singleton qualification evidence](2026-11-09-m3-singleton-backbone.md) for the original per-chip measurements.

| Knob | Default | Scope and requirement | Builtin policy |
| --- | --- | --- | --- |
| `qwen35_eager_layer_dispatch` | `False` (boolean) | Existing `qwen3_5.DecoderLayer.__call__` seam, installed rank-one norm/expert/top-k facts, outside compiled decode | Universal hardware; norm width 2048, 256 experts, top-k 8, flattened rows 0–64; [existing evidence](2026-09-13-qwen35-eager-layer-dispatch.md) |
| `qwen35_fused_gdn_decode` | `False` (boolean) | Existing GDN classes and loader `allow_dense` distinction; projected geometry, cache/sharding checks and full parity probe stay local | Universal hardware; hidden 2048 on legacy routes, positive integer hidden widths on existing dense opt-in routes |
| `gdn_in_proj_fusion` | `False` (boolean) | Existing installation routes, compatible projection formats and projection/layer parity; split retention stays caller-owned | Universal hardware; enabled where installation was already attempted |
| `gdn_in_proj_max_rows` | `0` (nonnegative integer) | Installed fusion, qualified dtype; capped by the local eight-row probe envelope | Universal hardware; 8 |

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
