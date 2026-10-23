# SPDX-License-Identifier: Apache-2.0
"""Actual restricted pickle behavior and rejection before vulnerable loaders."""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

from rapid_mlx.utils.torch_checkpoint import load_tensor_checkpoint


@pytest.mark.parametrize(
    "version", ["2.3.0", "2.5.1", "2.6.0", "2.9.1", "2.10.0rc1", "unknown"]
)
def test_unpatched_torch_is_rejected_before_deserialization(monkeypatch, version):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__=version,
            load=lambda *a, **kw: calls.append(kw),
        ),
    )
    with pytest.raises(RuntimeError, match="PyTorch"):
        load_tensor_checkpoint("untrusted.ckpt")
    assert calls == []


def test_restricted_pickle_is_explicit_even_with_unsafe_env(monkeypatch):
    calls = []
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.10.0+cpu",
            load=lambda *a, **kw: calls.append(kw),
        ),
    )
    load_tensor_checkpoint("tensor.ckpt")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


class _MarkerPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


def test_actual_pickle_payload_never_executes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    if Version(torch.__version__).is_prerelease:
        pytest.skip("Requires a patched stable PyTorch runtime")
    marker = tmp_path / "EXECUTED"
    checkpoint = tmp_path / "payload.pth"
    torch.save({"payload": _MarkerPayload(marker)}, checkpoint)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(pickle.UnpicklingError):
        load_tensor_checkpoint(checkpoint)
    assert not marker.exists()


def test_numeric_legacy_checkpoint_loads(tmp_path):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    checkpoint = tmp_path / "weights.pth"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, checkpoint)
    result = load_tensor_checkpoint(checkpoint)
    assert result["weight"].tolist() == [1.0, 2.0]
