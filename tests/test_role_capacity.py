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


def test_local_cache_lookup_is_bounded_for_hit_and_miss(monkeypatch):
    """The verified-cache footprint (hit OR miss) is reused only for the bounded
    TTL so a mutable cache's later download becomes discoverable after expiry,
    while a burst of arbitrary model ids does not re-walk the whole cache each
    time (reconciles round-3 'don't permanently memoize', round-4 'bounded TTL',
    and round-9 'bounded negatively-cached miss' findings)."""
    from vllm_mlx.runtime import role_capacity

    calls: list[str] = []
    monkeypatch.setattr(
        role_capacity,
        "_scan_local_cache_bytes",
        lambda hf: calls.append(hf) or 998244353,
    )
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", 60.0)
    role_capacity._local_cache_lookups.clear()

    # First call scans and caches the positive result.
    assert role_capacity._local_cache_bytes("c/FakeModel") == 998244353
    assert calls == ["c/FakeModel"]

    # A repeat inside the TTL reuses the cached footprint without re-scanning.
    assert role_capacity._local_cache_bytes("c/FakeModel") == 998244353
    assert calls == ["c/FakeModel"]

    # After the TTL elapses the cache is re-scanned (mutable disk observed).
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", -1.0)
    assert role_capacity._local_cache_bytes("c/FakeModel") == 998244353
    assert calls == ["c/FakeModel", "c/FakeModel"]

    # A MISS is also cached briefly: two consecutive unknown lookups scan only
    # once (no repeated full-cache walks for arbitrary ids), then re-scan after
    # the TTL so a later download becomes discoverable.
    monkeypatch.setattr(
        role_capacity,
        "_scan_local_cache_bytes",
        lambda hf: calls.append(hf) or None,
    )
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", 60.0)
    role_capacity._local_cache_lookups.clear()
    assert role_capacity._local_cache_bytes("c/NotThere") is None
    assert role_capacity._local_cache_bytes("c/NotThere") is None
    assert calls.count("c/NotThere") == 1

    # After the TTL the miss is re-scanned so a fresh download is discovered.
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", -1.0)
    assert role_capacity._local_cache_bytes("c/NotThere") is None
    assert calls.count("c/NotThere") == 2


def test_local_cache_rejects_partial_incomplete_download(monkeypatch):
    """pr_validate codex BLOCKING (round-9): a partially-cached repo (tokenizer/
    config only, no completed snapshot) must NOT be trusted as the footprint —
    that would under-reserve and let the later load blow past the ceiling. It
    returns unknown and fails closed under a configured ceiling."""
    import huggingface_hub

    from vllm_mlx.runtime import role_capacity

    class _IncompleteRev:
        refs = frozenset()

    class _CompleteRev:
        refs = frozenset({"main"})

    class _PartialRepo:
        repo_id = "c/PartialModel"
        size_on_disk = 654321  # small: only config/tokenizer bytes cached
        revisions = (_IncompleteRev(),)

    class _CompleteRepo:
        repo_id = "c/CompleteModel"
        size_on_disk = 998244353
        revisions = (_CompleteRev(),)

    class _FakeCache:
        repos = [_PartialRepo, _CompleteRepo]

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: _FakeCache())
    role_capacity._local_cache_lookups.clear()
    role_capacity._LOCAL_CACHE_TTL_SECONDS = -1.0  # force rescan each call

    # Partial download is rejected (fails closed), full download is charged.
    assert role_capacity._scan_local_cache_bytes("c/PartialModel") is None
    assert role_capacity._scan_local_cache_bytes("c/CompleteModel") == 998244353
