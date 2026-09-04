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


def test_local_cache_lookup_coalesces_into_one_scan_per_ttl(monkeypatch):
    """The local-cache footprint reads from a SINGLE TTL-bounded snapshot: any
    number of distinct model ids (attacker-controlled) trigger one cache walk
    per TTL window — no per-id scan, no unbounded per-id memory — and a fresh
    download becomes discoverable after the TTL elapses."""
    from vllm_mlx.runtime import role_capacity

    calls: list[str] = []

    def fake_index() -> dict[str, int]:
        calls.append("scan")
        return {"c/fakemodel": 998244353, "c/othermodel": 555}

    monkeypatch.setattr(role_capacity, "_scan_local_cache_index", fake_index)
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(role_capacity, "_local_cache_snapshot", None)

    # First lookup builds the snapshot (one scan).
    assert role_capacity._local_cache_bytes("c/FakeModel") == 998244353
    assert calls == ["scan"]

    # A distinct id inside the same TTL reuses the SAME snapshot — no new scan.
    assert role_capacity._local_cache_bytes("c/OtherModel") == 555
    assert calls == ["scan"]

    # A miss inside the TTL is served from the same snapshot, still no scan.
    assert role_capacity._local_cache_bytes("c/NotThere") is None
    assert calls == ["scan"]

    # After the TTL elapses the snapshot is rebuilt (fresh download discoverable).
    monkeypatch.setattr(role_capacity, "_LOCAL_CACHE_TTL_SECONDS", -1.0)
    assert role_capacity._local_cache_bytes("c/NotThere") is None
    assert calls == ["scan", "scan"]


class _Rev:
    def __init__(self, refs, file_sizes=()):
        self.refs = frozenset(refs)
        self.files = [type("F", (), {"size_on_disk": s})() for s in file_sizes]


def test_local_cache_rejects_partial_and_uses_completed_snapshot(monkeypatch):
    """pr_validate codex BLOCKING (round-9 + round-10): the local-cache index
    trusts ONLY a complete (ref-bound) snapshot and charges its actual file
    bytes — never a partial/aggregate footprint that would under-reserve."""
    import huggingface_hub

    from vllm_mlx.runtime import role_capacity

    class _PartialRepo:
        repo_id = "c/PartialModel"
        # A repo with ONLY a partial (ref-less) revision: tokenizer/config only.
        revisions = (_Rev(refs=()),)

    class _CompleteRepo:
        repo_id = "c/CompleteModel"
        # A completed snapshot: ref points at main, weights present.
        revisions = (_Rev(refs={"main"}, file_sizes=(1000, 9000)),)

    class _OldCompletedPlusPartialRepo:
        repo_id = "c/MixedModel"
        # An old completed revision PLUS a fresh partial one being downloaded.
        revisions = (
            _Rev(refs={"main"}, file_sizes=(8000,)),
            _Rev(refs=(), file_sizes=(100,)),
        )

    class _FakeCache:
        repos = [_PartialRepo, _CompleteRepo, _OldCompletedPlusPartialRepo]

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: _FakeCache())

    index = role_capacity._scan_local_cache_index()
    # Partial (no completed snapshot) contributes nothing -> fail closed.
    assert "c/partialmodel" not in index
    # Completed snapshot charged by its ACTUAL file bytes (1000+9000).
    assert index["c/completemodel"] == 10000
    # A repo with an older completed + partial revision is charged ONLY the
    # completed snapshot's bytes (8000), not the aggregate (8100) — so the
    # in-progress second revision cannot understate the real load.
    assert index["c/mixedmodel"] == 8000
