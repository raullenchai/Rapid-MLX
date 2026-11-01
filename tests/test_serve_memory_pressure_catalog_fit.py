# SPDX-License-Identifier: Apache-2.0
"""Live pressure uses serve preflight's estimate for unmeasured catalog models."""

from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from rapid_mlx import cli, recommendations
from rapid_mlx.byom import preflight

GIB = 1024**3
ALIAS = "gemma-4-26b-4bit"
MODEL_PATH = "/local/path/to/model"


@pytest.fixture
def memory_check(monkeypatch):
    monkeypatch.setattr(recommendations, "recommendation_footprint_gb", lambda _: None)
    monkeypatch.setattr("os.path.isdir", lambda _: True)
    monkeypatch.setattr("os.walk", lambda path: [(path, [], ["weights.safetensors"])])
    monkeypatch.setattr("os.path.getsize", lambda _: 16 * GIB)
    monkeypatch.setattr(
        preflight, "mlx_recommended_working_set_bytes", lambda: 25 * GIB
    )
    fit = Mock(return_value=SimpleNamespace(estimate_bytes=20 * GIB, fits=True))
    monkeypatch.setattr(preflight, "catalog_memory_fit", fit)

    def check(used_gb, *, alias=ALIAS):
        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: SimpleNamespace(
                total=32 * GIB, available=int((32 - used_gb) * GIB)
            ),
        )
        cli._check_memory_capacity(MODEL_PATH, alias=alias)

    return check, fit


def test_fitting_catalog_model_avoids_disk_multiplier_warning(memory_check, capsys):
    check, fit = memory_check
    # Disk x1.5 projects (5.5 + 24) / 32 = 92%; catalog projects only 80%.
    check(5.5)
    out = capsys.readouterr().out
    assert "Memory pressure warning" not in out
    assert out == ""
    fit.assert_called_once_with(
        ALIAS, ram_bytes=32 * GIB, recommended_working_set_bytes=25 * GIB
    )


@pytest.mark.parametrize(
    ("used_gb", "message"),
    [
        (10.4, "Memory pressure note"),
        (12, "Memory pressure note"),
        (13, "!! Memory pressure warning"),
    ],
)
def test_fitting_catalog_model_still_reports_live_pressure(
    memory_check, capsys, used_gb, message
):
    check, _ = memory_check
    check(used_gb)
    out = capsys.readouterr().out
    assert message in out
    assert "Catalog working set:       20.0 GB" in out
    assert "model x 1.5" not in out
    if used_gb <= 12:
        assert "Memory pressure warning" not in out
    else:
        assert "kernel panic" in out
        assert "103% projected utilization" in out


def test_non_catalog_local_path_keeps_conservative_disk_warning(memory_check, capsys):
    check, fit = memory_check
    check(4, alias=None)
    out = capsys.readouterr().out
    assert "!! Memory pressure warning" in out
    assert "model x 1.5" in out
    assert "88% projected utilization" in out
    fit.assert_not_called()


@pytest.mark.parametrize("fits", [False, None])
def test_catalog_estimate_without_fit_keeps_conservative_bands(
    memory_check, capsys, fits
):
    check, fit = memory_check
    fit.return_value.fits = fits
    check(8)
    out = capsys.readouterr().out
    assert "!! Memory pressure warning" in out
    assert "Catalog working set:       20.0 GB" in out
    assert "88% projected utilization" in out


@pytest.mark.parametrize("raises", [False, True])
def test_unknown_or_failed_catalog_estimate_falls_back_to_disk(
    memory_check, capsys, raises
):
    check, fit = memory_check
    if raises:
        fit.side_effect = RuntimeError("catalog unavailable")
    else:
        fit.return_value.estimate_bytes = None
    check(5.5)
    out = capsys.readouterr().out
    assert "!! Memory pressure warning" in out
    assert "model x 1.5" in out
    assert "92% projected utilization" in out
