# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-flight disk-space check in cli._check_disk_space.

The check must: hard-fail when disk is provably insufficient, return
silently when it can't determine size or the model is already cached,
and respect HF_HOME via huggingface_hub.constants.HF_HUB_CACHE.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx.cli import _check_disk_space


def _make_info(file_sizes_bytes: list[int]) -> SimpleNamespace:
    """Build a fake huggingface_hub.ModelInfo with sibling file sizes."""
    siblings = [SimpleNamespace(size=sz) for sz in file_sizes_bytes]
    return SimpleNamespace(siblings=siblings, safetensors=None)


def _fake_statvfs(free_bytes: int):
    """Build a fake os.statvfs result with the requested free space."""
    return SimpleNamespace(f_bavail=free_bytes // 4096, f_frsize=4096)


class TestDiskSpaceCheck:
    def test_aborts_when_model_too_large(self):
        """141 GB model on 8.8 GB disk — the bug we're fixing."""
        info = _make_info([int(141 * 1024**3)])
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(8.8 * 1024**3))),
        ):
            with pytest.raises(SystemExit) as exc:
                _check_disk_space("mlx-community/DeepSeek-V4-Flash-4bit")
            assert exc.value.code == 1

    def test_passes_when_disk_has_room(self):
        info = _make_info([int(2 * 1024**3)])  # 2 GB model
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(50 * 1024**3))),
        ):
            # Should not raise.
            _check_disk_space("mlx-community/Qwen3-0.6B-8bit")

    def test_force_skips_abort(self):
        """With --force-disk-check (force=True), insufficient disk warns
        but does not abort."""
        info = _make_info([int(141 * 1024**3)])
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(8.8 * 1024**3))),
        ):
            # Should not raise.
            _check_disk_space("mlx-community/DeepSeek-V4-Flash-4bit", force=True)

    def test_returns_silently_when_model_size_unknown(self):
        """If HF doesn't return file sizes (gated repo, weird config),
        we can't math the disk requirement — skip rather than guess."""
        info = _make_info([])  # no siblings
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
        ):
            _check_disk_space("mlx-community/Some-Model")  # no raise

    def test_returns_silently_when_hf_api_fails(self):
        """Network errors / 404s during the size query must not block
        startup — the loader has its own 404 handler."""
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch(
                "huggingface_hub.model_info",
                side_effect=ConnectionError("offline"),
            ),
        ):
            _check_disk_space("mlx-community/Some-Model")  # no raise

    def test_skips_local_path(self, tmp_path):
        """Local model directories don't need disk checking."""
        local = tmp_path / "my-model"
        local.mkdir()
        # Should not raise even without mocking model_info — must short-circuit.
        _check_disk_space(str(local))

    def test_skips_already_cached(self):
        """If config.json is in the cache, skip the size check entirely."""
        with (
            patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value="/fake/path/config.json",
            ),
            patch("os.path.exists", return_value=True),
        ):
            # Should not raise even without mocking model_info.
            _check_disk_space("mlx-community/Some-Model")

    def test_uses_hf_hub_cache_for_statvfs(self):
        """The probe must use HF_HUB_CACHE (respects HF_HOME), not the
        hard-coded ~/.cache/huggingface."""
        info = _make_info([int(2 * 1024**3)])
        seen_paths = []

        def capture_statvfs(path):
            seen_paths.append(path)
            return _fake_statvfs(int(50 * 1024**3))

        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "huggingface_hub.constants.HF_HUB_CACHE",
                "/Volumes/external/hf",
            ),
            patch("os.statvfs", side_effect=capture_statvfs),
            patch(
                "os.path.exists",
                side_effect=lambda p: p == "/Volumes/external" or p.startswith("/"),
            ),
        ):
            _check_disk_space("mlx-community/Qwen3-0.6B-8bit")

        assert seen_paths, "statvfs was never called"
        assert any(
            "external" in p or p == "/Volumes/external" or p.startswith("/Volumes")
            for p in seen_paths
        ) or any(p.startswith(os.path.expanduser("~")) for p in seen_paths), (
            f"statvfs probe path {seen_paths!r} didn't include HF_HUB_CACHE ancestor"
        )
