"""Policy refusals remain authoritative after the QSA capability checks."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.tuning import Decision


@pytest.mark.parametrize(
    "name,function,width,kv,architecture",
    [
        ("qsa_stage1", "qsa_stage1_decline_reason", 64, 65536, "applegpu_g14s"),
        (
            "qsa_indexed_splitk",
            "indexed_splitk_decline_reason",
            3,
            16384,
            "applegpu_g15d",
        ),
    ],
)
def test_policy_can_decline_capable_qsa(
    monkeypatch, name, function, width, kv, architecture
):
    module = importlib.import_module("rapid_mlx.kernels." + name)
    monkeypatch.setenv(module.ENABLE_ENV, "1")
    monkeypatch.setattr(
        module, "mx", SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
    )
    resolve = Mock(return_value=Decision(False, "local-calibration"))
    monkeypatch.setattr(module, "resolve", resolve)
    assert (
        getattr(module, function)(
            width,
            kv,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture=architecture,
        )
        == "disabled"
    )
    resolve.assert_called_once()
    knob, ctx = resolve.call_args.args
    assert knob == name and ctx.width == width and ctx.batch == 1
    assert ctx.hardware.architecture == architecture
