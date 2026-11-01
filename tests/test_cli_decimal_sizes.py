# SPDX-License-Identifier: Apache-2.0
"""CLI download and disk sizes use decimal units, matching the catalog."""

from types import SimpleNamespace

import pytest

from rapid_mlx import _download_gate, cli, model_sizes


def test_disk_check_uses_decimal_gb(monkeypatch, capsys):
    info = SimpleNamespace(
        sha="current-sha",
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=16_320_000_000)],
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **k: info)
    monkeypatch.setattr(
        "os.statvfs",
        lambda _path: SimpleNamespace(f_bavail=10_200_000_000, f_frsize=1),
    )

    with pytest.raises(SystemExit, match="1"):
        cli._check_disk_space("test/decimal-size-model")

    stderr = capsys.readouterr().err
    assert "Insufficient disk space for download" in stderr
    assert "16.3 GB" in stderr
    assert "15.2" not in stderr
    assert "10.2 GB" in stderr
    assert "7.8 GB" in stderr


@pytest.mark.parametrize(
    "size, expected",
    [
        (16_320_000_000, "16.3 GB"),
        (1_500_000, "1.5 MB"),
        (999_960_000, "1.0 GB"),
        (999_960, "1.0 MB"),
        (999, "999 B"),
    ],
)
def test_download_gate_decimal_sizes(size, expected):
    assert _download_gate._format_size(size) == expected


@pytest.mark.parametrize(
    "size, expected", [(16_320_000_000, "16.3 GB"), (0, "0 B"), (-1, "0 B")]
)
def test_cli_decimal_sizes(size, expected):
    assert cli._format_bytes(size) == expected


def test_manifest_size_uses_decimal_gb():
    repo, size = next(
        (repo, size)
        for repo, size in model_sizes._raw().items()
        if size is not None and 1e9 <= size < 1e12
    )
    rendered = model_sizes.format_size(repo)
    assert rendered == f"{size / 1e9:.1f} GB"
    assert "GiB" not in rendered
