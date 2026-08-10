# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-flight disk-space check in cli._check_disk_space.

The check must: hard-fail when disk is provably insufficient, return
silently when it can't determine size or the model is already cached,
and respect HF_HOME via huggingface_hub.constants.HF_HUB_CACHE.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.cli import _check_disk_space


def _make_info(file_sizes_bytes: list[int]) -> SimpleNamespace:
    """Build a fake huggingface_hub.ModelInfo with sibling file sizes."""
    siblings = [
        SimpleNamespace(rfilename=f"file-{index}.safetensors", size=size)
        for index, size in enumerate(file_sizes_bytes)
    ]
    return SimpleNamespace(siblings=siblings, safetensors=None, sha="current-sha")


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

    def test_skips_already_cached(self, tmp_path):
        """A cached root config keeps the selective-loader fast path."""
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"present")
        model_info = MagicMock()
        statvfs = MagicMock()
        with (
            patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value=str(cached_file),
            ),
            patch("huggingface_hub.model_info", model_info),
            patch("os.statvfs", statvfs),
        ):
            _check_disk_space("mlx-community/Some-Model")
        model_info.assert_not_called()
        statvfs.assert_not_called()

    def test_complete_mflux_cache_without_root_config_skips_gate(self, tmp_path):
        """A component-layout mflux repo has no root config.json.

        Completeness is determined from every file at the current revision,
        so a fully cached 5.5 GB Z-Image snapshot must not be rejected merely
        because only 3.5 GB remains free.
        """
        names_and_sizes = [
            ("transformer/0.safetensors", int(3.2 * 1024**3)),
            ("text_encoder/0.safetensors", int(2.1 * 1024**3)),
            ("vae/0.safetensors", int(0.2 * 1024**3)),
            ("tokenizer/tokenizer.json", 10_000),
        ]
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=name, size=size)
                for name, size in names_and_sizes
            ],
            sha="z-image-sha",
        )
        cached_paths = {}
        for index, (name, _size) in enumerate(names_and_sizes):
            path = tmp_path / str(index)
            path.write_bytes(b"present")
            cached_paths[name] = str(path)

        def cache_lookup(repo_id, filename, revision=None):
            assert repo_id == "filipstrand/Z-Image-Turbo-mflux-4bit"
            if revision is None:
                assert filename == "config.json"
                return None
            assert revision == "z-image-sha"
            return cached_paths[filename]

        statvfs = MagicMock()
        with (
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "huggingface_hub.try_to_load_from_cache",
                side_effect=cache_lookup,
            ),
            patch("os.statvfs", statvfs),
        ):
            _check_disk_space("filipstrand/Z-Image-Turbo-mflux-4bit")
        statvfs.assert_not_called()

    def test_partial_mflux_cache_counts_only_missing_files(self, tmp_path):
        """A partial component cache requires space only for missing weights."""
        cached_transformer = tmp_path / "transformer.safetensors"
        cached_transformer.write_bytes(b"present")
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename="transformer/0.safetensors", size=int(3 * 1024**3)
                ),
                SimpleNamespace(
                    rfilename="text_encoder/0.safetensors",
                    size=int(2.5 * 1024**3),
                ),
            ],
            sha="z-image-sha",
        )

        def cache_lookup(_repo_id, filename, revision=None):
            if revision is None:
                return None
            assert revision == "z-image-sha"
            if filename == "transformer/0.safetensors":
                return str(cached_transformer)
            return None

        with (
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "huggingface_hub.try_to_load_from_cache",
                side_effect=cache_lookup,
            ),
            # 3 GB fits the missing 2.5 GB shard plus headroom, but not the
            # full 5.5 GB repository. Counting the cached transformer would
            # therefore make this call raise SystemExit.
            patch("os.statvfs", return_value=_fake_statvfs(int(3 * 1024**3))),
        ):
            _check_disk_space("filipstrand/Z-Image-Turbo-mflux-4bit")

    def test_uses_hf_hub_cache_for_statvfs(self):
        """The probe must use HF_HUB_CACHE (respects HF_HOME), not the
        hard-coded ~/.cache/huggingface.

        Asserts strictly on /Volumes — the HOME fallback only kicks in when
        the walk-up loop runs out of ancestors, which would mask a
        regression that drops the HF_HUB_CACHE pivot entirely.
        """
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
        assert any("/Volumes" in p for p in seen_paths), (
            f"statvfs probe path {seen_paths!r} didn't include HF_HUB_CACHE; "
            "the HF_HOME pivot regressed and the probe fell back to $HOME."
        )
