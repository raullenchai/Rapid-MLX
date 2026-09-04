from __future__ import annotations

"""MLX-free tests for metadata-backed alignment-role capacity resolution.

These prove the forced-alignment footprint is resolved from catalog /
local-cache metadata BEFORE any weight loading (so admission decides with
knowledge of the size, not blind) — the first half of issue #2405's
Required contract. No MLX is imported or required.
"""


def test_alignment_capacity_resolves_from_catalog_by_alias():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity("qwen3-aligner")
    # Catalog manifest maps qwen3-aligner -> mlx-community/Qwen3-ForcedAligner-0.6B-8bit
    # whose weight+tokenizer footprint is a real byte count, not a heuristic.
    assert capacity.source == "catalog"
    assert isinstance(capacity.requested_bytes, int)
    assert capacity.requested_bytes > 0


def test_alignment_capacity_resolves_from_catalog_by_long_alias():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity("qwen3-forced-aligner")
    assert capacity.source == "catalog"
    assert (
        capacity.requested_bytes == alignment_capacity("qwen3-aligner").requested_bytes
    )


def test_alignment_capacity_resolves_from_catalog_by_hf_id():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
    assert capacity.source == "catalog"
    assert (
        capacity.requested_bytes == alignment_capacity("qwen3-aligner").requested_bytes
    )


def test_alignment_capacity_fails_closed_on_unknown():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    # A non-audio / unknown id has no catalog entry -> source "unknown" so a
    # configured ceiling fails closed rather than admitting blind.
    capacity = alignment_capacity("some/unknown-nonaligner-checkpoint")
    assert capacity.source == "unknown"
    assert capacity.requested_bytes is None
