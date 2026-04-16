# SPDX-License-Identifier: Apache-2.0
"""Tests for doctor's local-model discovery.

The benchmark tier's "no auto-download" contract depends on discovery
correctly distinguishing complete snapshots from partial ones.  These
tests pin down that behavior — if any fails, the benchmark may try to
serve a half-downloaded model and fail with a misleading runtime
crash instead of a clean "skipped (partial download)" report.
"""

from pathlib import Path

import pytest

from vllm_mlx.doctor.discovery import _check_alias, _is_complete_snapshot


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """A fake HF cache root."""
    return tmp_path / "hub"


def _make_hf_snapshot(
    cache_root: Path,
    repo_id: str,
    *,
    with_config: bool,
    with_weights: bool,
    weight_resolves: bool = True,
) -> Path:
    """Create a synthetic HF cache snapshot under cache_root."""
    repo_dir = cache_root / ("models--" + repo_id.replace("/", "--"))
    snap = repo_dir / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)

    if with_config:
        (snap / "config.json").write_text("{}")

    if with_weights:
        # Mimic HF's symlink-to-blobs layout — write a real blob then
        # symlink it from the snapshot dir so resolve(strict=True) works.
        blobs = repo_dir / "blobs"
        blobs.mkdir(exist_ok=True)
        blob = blobs / "abc123"
        blob.write_text("fake weights")
        link = snap / "model.safetensors"
        if weight_resolves:
            link.symlink_to(blob)
        else:
            # Dangling symlink — what a half-downloaded model looks like.
            link.symlink_to(blobs / "missing-blob")
    return snap


# ----------------------------------------------------------------------
# _is_complete_snapshot
# ----------------------------------------------------------------------

class TestIsCompleteSnapshot:
    def test_complete_snapshot_passes(self, tmp_cache):
        snap = _make_hf_snapshot(
            tmp_cache, "org/repo", with_config=True, with_weights=True,
        )
        assert _is_complete_snapshot(snap) is True

    def test_missing_config_fails(self, tmp_cache):
        snap = _make_hf_snapshot(
            tmp_cache, "org/repo", with_config=False, with_weights=True,
        )
        assert _is_complete_snapshot(snap) is False

    def test_missing_weights_fails(self, tmp_cache):
        """Regression: this was the original bug — config.json present
        but no .safetensors file → discovery wrongly said 'available',
        server crashed at runtime with 'No safetensors found'."""
        snap = _make_hf_snapshot(
            tmp_cache, "org/repo", with_config=True, with_weights=False,
        )
        assert _is_complete_snapshot(snap) is False

    def test_dangling_symlink_fails(self, tmp_cache):
        """Half-downloaded HF snapshot: weight file is a symlink into
        ../blobs/ but the blob never finished downloading.  Plain
        Path.exists() follows symlinks and returns False on these,
        but Path.glob() still surfaces them.  resolve(strict=True)
        is what catches the dangling case."""
        snap = _make_hf_snapshot(
            tmp_cache, "org/repo", with_config=True,
            with_weights=True, weight_resolves=False,
        )
        assert _is_complete_snapshot(snap) is False

    def test_npz_weights_accepted(self, tmp_cache):
        """Some MLX exports use .npz instead of .safetensors."""
        snap = _make_hf_snapshot(
            tmp_cache, "org/repo", with_config=True, with_weights=False,
        )
        (snap / "weights.npz").write_text("fake npz")
        assert _is_complete_snapshot(snap) is True


# ----------------------------------------------------------------------
# _check_alias multi-root + multi-layout
# ----------------------------------------------------------------------

class TestCheckAlias:
    def test_picks_complete_over_partial_at_other_root(self, tmp_path):
        """Two cache roots: first has metadata-only snapshot, second has
        the real weights.  Discovery must skip the partial one and
        return the complete one — not silently use the broken path.

        This was the specific shape of the bug observed on the user's
        machine (SSD had only metadata, ~/.cache/huggingface had the
        actual blobs).
        """
        partial_root = tmp_path / "ssd"
        complete_root = tmp_path / "local"

        _make_hf_snapshot(
            partial_root, "org/repo",
            with_config=True, with_weights=False,
        )
        complete_snap = _make_hf_snapshot(
            complete_root, "org/repo",
            with_config=True, with_weights=True,
        )

        result = _check_alias(
            "test-alias", "org/repo", [partial_root, complete_root],
        )
        assert result.available is True
        assert result.path == complete_snap

    def test_partial_only_marks_unavailable_with_clear_reason(self, tmp_path):
        cache = tmp_path / "cache"
        _make_hf_snapshot(
            cache, "org/repo", with_config=True, with_weights=False,
        )
        result = _check_alias("test-alias", "org/repo", [cache])
        assert result.available is False
        assert "partial download" in result.reason

    def test_lm_studio_layout_recognized(self, tmp_path):
        """LM Studio stores models as {root}/{org}/{repo}/, not the
        HF hub layout.  Without this support the LM Studio cache was
        invisible to discovery."""
        lmstudio = tmp_path / "models"
        repo_dir = lmstudio / "lmstudio-community" / "some-model"
        repo_dir.mkdir(parents=True)
        (repo_dir / "config.json").write_text("{}")
        (repo_dir / "model.safetensors").write_text("fake")

        result = _check_alias(
            "test-alias",
            "lmstudio-community/some-model",
            [lmstudio],
        )
        assert result.available is True
        assert result.path == repo_dir

    def test_completely_missing_model(self, tmp_path):
        result = _check_alias("test-alias", "org/repo", [tmp_path])
        assert result.available is False
        assert "not found" in result.reason.lower()
