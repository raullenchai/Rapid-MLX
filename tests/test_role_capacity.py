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


class _FakePath:
    def __init__(self, text):
        self._text = text

    def read_text(self, encoding="utf-8"):
        return self._text


class _Rev:
    def __init__(self, refs, files=(), index_json=None):
        self.refs = frozenset(refs)
        self.files = [
            type("F", (), {"file_name": name, "size_on_disk": size})
            for name, size in files
        ]
        if index_json is not None:
            # Point model.safetensors.index.json at a readable fake path.
            for f in self.files:
                if f.file_name == "model.safetensors.index.json":
                    f.file_path = _FakePath(index_json)
                    f.read_text = f.file_path.read_text


def test_local_cache_rejects_partial_and_uses_completed_snapshot(monkeypatch):
    """pr_validate codex BLOCKING (round-9/10/11): the local-cache index trusts
    ONLY a COMPLETE snapshot — ref-bound AND carrying a weight file — and
    charges its actual file bytes, never a partial/aggregate footprint."""
    import huggingface_hub

    from vllm_mlx.runtime import role_capacity

    class _PartialRepo:
        repo_id = "c/PartialModel"
        # A ref-less partial revision: tokenizer/config only.
        revisions = (_Rev(refs=(), files=()),)

    class _SelectiveNoWeightsRepo:
        repo_id = "c/SelectiveModel"
        # A REF-BOUND snapshot with only config/tokenizer (a selective
        # download) — must be treated as incomplete.
        revisions = (_Rev(refs={"main"}, files=(("config.json", 50),)),)

    class _CompleteRepo:
        repo_id = "c/CompleteModel"
        # A complete SINGLE-FILE download (GGUF): ref points at main, weights
        # present, no sharding possibility.
        revisions = (
            _Rev(
                refs={"main"},
                files=(("config.json", 50), ("model.gguf", 9950)),
            ),
        )

    class _OldCompletedPlusPartialRepo:
        repo_id = "c/MixedModel"
        # An old completed GGUF revision PLUS a fresh partial one downloading.
        revisions = (
            _Rev(refs={"main"}, files=(("model.gguf", 8000),)),
            _Rev(refs=(), files=(("config.json", 100),)),
        )

    class _BareSafeTensorNoIndexRepo:
        repo_id = "c/BareSafeTensorModel"
        # A CANONICAL single-file .safetensors (`model.safetensors`) with NO
        # index (round-21): an unsharded checkpoint is published under this
        # verbatim name, and a sharded download never uses it, so a lone
        # canonical file is the WHOLE checkpoint -> charged, not rejected.
        revisions = (_Rev(refs={"main"}, files=(("model.safetensors", 9900),)),)

    class _NonCanonicalWeightNoIndexRepo:
        repo_id = "c/NonCanonicalWeightModel"
        # A NON-canonical single weight (`encoder-1.bin`) with no index could
        # be one piece of a selective multi-file download -> still fails closed
        # (only canonical names prove an unsharded checkpoint).
        revisions = (_Rev(refs={"main"}, files=(("encoder-1.bin", 5000),)),)

    class _ShardedGgufNoIndexRepo:
        repo_id = "c/ShardedGgufModel"
        # A SPLIT GGUF (round-16) without a readable shard set: a single
        # split shard must not be charged as the whole checkpoint.
        revisions = (
            _Rev(
                refs={"main"},
                files=(("model-00001-of-00002.gguf", 6000),),
            ),
        )

    class _ShardedCompleteRepo:
        repo_id = "c/ShardedCompleteModel"
        # Multi-shard download, all shards present per the index.
        revisions = (
            _Rev(
                refs={"main"},
                files=(
                    ("model.safetensors.index.json", 100),
                    ("model-00001-of-00002.safetensors", 4000),
                    ("model-00002-of-00002.safetensors", 5000),
                ),
                index_json=(
                    '{"weight_map": {"a": "model-00001-of-00002.safetensors", '
                    '"b": "model-00002-of-00002.safetensors"}}'
                ),
            ),
        )

    class _ShardedIncompleteRepo:
        repo_id = "c/ShardedIncompleteModel"
        # Multi-shard download MISSING one shard (interrupted) — the index
        # proves the second shard is absent, so it must fail closed.
        revisions = (
            _Rev(
                refs={"main"},
                files=(
                    ("model.safetensors.index.json", 100),
                    ("model-00001-of-00002.safetensors", 4000),
                ),
                index_json=(
                    '{"weight_map": {"a": "model-00001-of-00002.safetensors", '
                    '"b": "model-00002-of-00002.safetensors"}}'
                ),
            ),
        )

    class _NonDefaultBranchRepo:
        repo_id = "c/NonDefaultModel"
        # Only a non-main branch is cached; the loader resolves the default
        # (main) branch, so this cannot be trusted to charge the real load.
        revisions = (_Rev(refs={"feature-x"}, files=(("model.safetensors", 7000),)),)

    class _ShardedNoIndexRepo:
        repo_id = "c/ShardedNoIndexModel"
        # A shard-PATTERNED weight with NO index is an incomplete multi-shard
        # download (round-13) — one shard must not be charged as the whole.
        revisions = (
            _Rev(
                refs={"main"},
                files=(("model-00001-of-00002.safetensors", 4000),),
            ),
        )

    class _FakeCache:
        repos = [
            _PartialRepo,
            _SelectiveNoWeightsRepo,
            _CompleteRepo,
            _OldCompletedPlusPartialRepo,
            _BareSafeTensorNoIndexRepo,
            _NonCanonicalWeightNoIndexRepo,
            _ShardedGgufNoIndexRepo,
            _ShardedCompleteRepo,
            _ShardedIncompleteRepo,
            _NonDefaultBranchRepo,
            _ShardedNoIndexRepo,
        ]

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: _FakeCache())

    index = role_capacity._scan_local_cache_index()
    # Ref-less partial contributes nothing -> fail closed.
    assert "c/partialmodel" not in index
    # A ref-bound snapshot WITHOUT weights (selective download) is rejected.
    assert "c/selectivemodel" not in index
    # Complete snapshot charged by its ACTUAL file bytes (50+9950).
    assert index["c/completemodel"] == 10000
    # A repo with older completed + partial revision is charged ONLY the
    # completed snapshot's bytes (8000), not the aggregate (8100).
    assert index["c/mixedmodel"] == 8000
    # A multi-shard download with ALL shards present (per the index) is charged
    # its full snapshot bytes (index + shards); one with a missing shard fails
    # closed.
    assert index["c/shardedcompletemodel"] == 9100
    assert "c/shardedincompletemodel" not in index
    # A lone non-main branch is not the default the loader fetches -> fail closed.
    assert "c/nondefaultmodel" not in index
    # A shard-PATTERNED weight with no index is an incomplete download -> fail closed.
    assert "c/shardednoindexmodel" not in index
    # A CANONICAL single-file .safetensors (`model.safetensors`) is the WHOLE
    # unsharded checkpoint -> charged its actual bytes (round-21).
    assert index["c/baresafetensormodel"] == 9900
    # A NON-canonical single weight with no index could be a piece of a
    # selective download -> fails closed (only canonical names prove a whole
    # unsharded checkpoint).
    assert "c/noncanonicalweightmodel" not in index
    # A SPLIT GGUF without a proven full shard set fails closed.
    assert "c/shardedggufmodel" not in index


def test_local_cache_scan_failure_fails_closed(monkeypatch):
    """A cache-library failure must produce no trusted footprint."""
    import huggingface_hub

    from vllm_mlx.runtime import role_capacity

    def fail_scan():
        raise RuntimeError("cache metadata unavailable")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", fail_scan)
    assert role_capacity._scan_local_cache_index() == {}


def test_revision_rejects_corrupt_or_empty_shard_index():
    """Unreadable and empty indexes cannot prove a complete checkpoint."""
    from vllm_mlx.runtime import role_capacity

    corrupt = _Rev(
        refs={"main"},
        files=(("model.safetensors.index.json", 100),),
        index_json="not json",
    )
    empty = _Rev(
        refs={"main"},
        files=(("model.safetensors.index.json", 100),),
        index_json='{"weight_map": {}}',
    )

    assert role_capacity._revision_is_complete(corrupt) is False
    assert role_capacity._revision_is_complete(empty) is False
