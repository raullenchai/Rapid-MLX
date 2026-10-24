# SPDX-License-Identifier: Apache-2.0
"""Per-file legacy conversion consent and unchanged numeric tensor loading."""

import hashlib
import pickle
import sys
from types import SimpleNamespace

import pytest

from rapid_mlx.audio.sa3.models.defs.checkpoint_security import load_torch_checkpoint


def test_numeric_tensors_need_no_consent(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    monkeypatch.delenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", raising=False)
    path = tmp_path / "tensor.ckpt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, path)
    assert load_torch_checkpoint(path)["weight"].tolist() == [1.0, 2.0]


def test_consent_cannot_bypass_runtime_floor(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", "1")
    monkeypatch.setenv("RAPID_MLX_SA3_TRUSTED_SHA256", "0" * 64)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.9.1", load=lambda *a, **kw: pytest.fail("unpatched load")
        ),
    )
    with pytest.raises(RuntimeError, match="PyTorch"):
        load_torch_checkpoint("legacy.ckpt")


def test_approved_copy_is_private_and_independent_of_source(tmp_path, monkeypatch):
    path = tmp_path / "legacy.ckpt"
    original = b"reviewed bytes"
    path.write_bytes(original)
    monkeypatch.setenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", "1")
    monkeypatch.setenv(
        "RAPID_MLX_SA3_TRUSTED_SHA256", hashlib.sha256(original).hexdigest()
    )

    def load(source, *, map_location, weights_only):
        if weights_only:
            raise pickle.UnpicklingError("legacy objects")
        path.write_bytes(b"changed source")
        assert source.read() == original
        import os

        assert os.fstat(source.fileno()).st_mode & 0o077 == 0
        return {"state_dict": {}}

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__="2.10.0", load=load)
    )
    assert load_torch_checkpoint(path) == {"state_dict": {}}


def test_numeric_conversion_preserves_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    pytest.importorskip("mlx.core")
    from rapid_mlx.audio.sa3.models.defs.dit_mlx import convert_weights_from_torch_ckpt

    monkeypatch.delenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", raising=False)
    path = tmp_path / "safe.ckpt"
    torch.save({"state_dict": {"model.model.norm.gamma": torch.ones(3)}}, path)
    assert convert_weights_from_torch_ckpt(path)["norm.weight"].tolist() == [
        1.0,
        1.0,
        1.0,
    ]


def test_reviewed_legacy_object_loads_from_approved_copy(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch", minversion="2.10.0")
    path = tmp_path / "reviewed.ckpt"
    torch.save({"metadata": SimpleNamespace(label="legacy")}, path)
    monkeypatch.setenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", "1")
    monkeypatch.setenv(
        "RAPID_MLX_SA3_TRUSTED_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    assert load_torch_checkpoint(path)["metadata"].label == "legacy"
