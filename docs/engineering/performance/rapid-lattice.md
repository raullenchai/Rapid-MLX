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
