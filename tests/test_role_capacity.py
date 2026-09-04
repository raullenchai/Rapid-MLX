from __future__ import annotations

"""MLX-free tests for metadata-backed alignment-role capacity resolution.

These prove the forced-alignment footprint is resolved from catalog /
verified local-cache metadata BEFORE any weight loading (so admission decides
with knowledge of the size, not blind) — the first half of issue #2405's
Required contract. No MLX is imported or required.
"""

# Real manifest footprint for qwen3-aligner /
# mlx-community/Qwen3-ForcedAligner-0.6B-8bit (weight + tokenizer).
ALIGNER_CATALOG_BYTES = 1276473392
ALIGNER_HF_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


def _patch_catalog_to_missing(monkeypatch):
    """Make the checked-in manifest look like it has no size for the aligner,
    so the verified-local-cache fallback is exercised."""
    from vllm_mlx import model_sizes

    monkeypatch.setattr(model_sizes, "size_bytes", lambda _hf: None)


def test_alignment_capacity_resolves_from_catalog_by_alias():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity("qwen3-aligner")
    # Catalog manifest resolves a real, exact byte count, not a heuristic.
    assert capacity.source == "catalog"
    assert capacity.requested_bytes == ALIGNER_CATALOG_BYTES


def test_alignment_capacity_resolves_from_catalog_by_long_alias():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity("qwen3-forced-aligner")
    assert capacity.source == "catalog"
    assert capacity.requested_bytes == ALIGNER_CATALOG_BYTES


def test_alignment_capacity_resolves_from_catalog_by_hf_id():
    from vllm_mlx.runtime.role_capacity import alignment_capacity

    capacity = alignment_capacity(ALIGNER_HF_ID)
    assert capacity.source == "catalog"
    assert capacity.requested_bytes == ALIGNER_CATALOG_BYTES


def test_alignment_capacity_falls_back_to_verified_local_cache(monkeypatch):
    """When the catalog has no entry but the checkpoint is already on disk,
    the verified local-cache footprint is used instead of rejecting blind."""
    from vllm_mlx.runtime import role_capacity

    _patch_catalog_to_missing(monkeypatch)
    monkeypatch.setattr(role_capacity, "_local_cache_bytes", lambda _hf: 998244353)

    capacity = role_capacity.alignment_capacity(ALIGNER_HF_ID)
    assert capacity.source == "local-cache"
    assert capacity.requested_bytes == 998244353


def test_alignment_capacity_cached_only_checkpoint(monkeypatch):
    """A checkpoint absent from the catalog but present, verified, in the
    local cache yields a real charge sourced from the cache (issue #2405's
    required catalog-or-cache fallback)."""
    from vllm_mlx.runtime import role_capacity

    _patch_catalog_to_missing(monkeypatch)
    monkeypatch.setattr(role_capacity, "_local_cache_bytes", lambda _hf: 777777777)

    capacity = role_capacity.alignment_capacity(ALIGNER_HF_ID)
    assert capacity.source == "local-cache"
    assert capacity.requested_bytes == 777777777


def test_alignment_capacity_fails_closed_on_unknown(monkeypatch):
    """With neither a catalog entry nor a verified local-cache footprint, the
    capacity is unknown so a configured ceiling fails closed (no blind admit)."""
    from vllm_mlx.runtime import role_capacity

    _patch_catalog_to_missing(monkeypatch)
    monkeypatch.setattr(role_capacity, "_local_cache_bytes", lambda _hf: None)

    capacity = role_capacity.alignment_capacity("some/unknown-nonaligner-checkpoint")
    assert capacity.source == "unknown"
    assert capacity.requested_bytes is None
