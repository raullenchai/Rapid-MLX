"""Rollback-compatible projection enrollment executes with CPU array stubs."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rapid_mlx import gdn_in_proj_fusion as fusion
from rapid_mlx.spec_decode.mtp import cache_patch


class Projection(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.mark.parametrize("retain", [False, True])
def test_projection_enrollment_retains_split_without_replacing_call(
    monkeypatch, retain
):
    class Layer:
        pass

    layer = Layer()
    parts = [
        Projection(
            weight=np.ones((2, 2)), scales=np.ones((2, 1)), biases=np.ones((2, 1))
        )
        for _ in range(4)
    ]
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    for name, value in zip(names, parts):
        setattr(layer, name, value)
    model = SimpleNamespace(named_modules=lambda: [("layer", layer)])
    monkeypatch.setattr(fusion, "_gdn_imports", lambda: (None, None, Layer))
    monkeypatch.setattr(fusion, "_vlm_gdn_imports", lambda: None)
    monkeypatch.setattr(fusion, "_can_fuse", lambda layer: True)
    monkeypatch.setattr(
        fusion, "_probe_projection_parity", lambda layer: frozenset({"bf16"})
    )
    monkeypatch.setattr(fusion, "_proj_modules", lambda layer: parts)
    arrays = tuple(np.ones((8, 2)) for _ in range(3))
    monkeypatch.setattr(fusion, "_concat_params", lambda parts: arrays)
    monkeypatch.setitem(vars(fusion.mx), "eval", lambda *args: None)
    monkeypatch.setitem(vars(fusion.mx), "clear_cache", lambda: None)
    patch = Mock()
    probe = Mock(return_value=True)
    monkeypatch.setattr(fusion, "_ensure_call_patch", patch)
    monkeypatch.setattr(fusion, "_probe_whole_layer_parity", probe)
    monkeypatch.delenv("RAPID_MLX_GDN_IN_PROJ_FUSION", raising=False)
    assert fusion.fuse_gdn_in_proj(model, retain_split=retain) == 1
    assert patch.call_count == probe.call_count == int(not retain)
    assert layer.in_proj_fused.weight is arrays[0]
    assert layer._rapid_gdn_bounds == [2, 4, 6, 8]
    for name, value in zip(names, parts):
        assert hasattr(layer, name) == retain
        if retain:
            assert getattr(layer, name) is value


@pytest.mark.parametrize("fused", [False, True])
def test_verify_projection_dispatch(monkeypatch, fused):
    outputs = tuple(object() for _ in range(4))
    layer = SimpleNamespace(
        **{
            name: (lambda inputs, value=value: value)
            for name, value in zip(
                ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), outputs
            )
        }
    )
    inputs = object()
    helper = Mock(return_value=outputs)
    monkeypatch.setattr(fusion, "_fused_projections", helper)
    if fused:
        layer.in_proj_fused = object()
    assert cache_patch._input_projections(layer, inputs) == outputs
    assert helper.call_count == int(fused)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_dispatch_enrolls_only_selected_cell_after_head_attach(
    monkeypatch, enabled
):
    from rapid_mlx.engine import batched
    from rapid_mlx.lattice import mtp as policy
    from rapid_mlx.spec_decode import mtp
    from rapid_mlx.spec_decode.mtp import dispatch

    # Isolate optional Metal implementation imports while exercising the real
    # executor entrypoint and its installed-head ordering.
    decode = ModuleType("rapid_mlx.qwen35_fused_gdn_decode")
    steps = []
    decode.install_qwen35_fused_gdn_decode = lambda model: steps.append("decode")
    monkeypatch.setitem(sys.modules, decode.__name__, decode)
    monkeypatch.setattr(
        fusion, "fuse_gdn_in_proj", lambda model, **kw: steps.append(kw)
    )
    monkeypatch.setitem(dispatch._MTP_INJECT_DISPATCH, "test", object())

    def attach(*args, **kwargs):
        steps.append("attach")
        return True

    def selected(model):
        assert steps == ["attach"]
        return enabled

    monkeypatch.setattr(mtp, "dispatch_mtp_inject", attach)
    monkeypatch.setattr(policy, "gdn_fusion_enabled", selected)
    assert (
        batched._run_dispatch_mtp_inject(
            object(), "cached", None, preferred_model_type="test"
        )
        == batched._DISPATCH_ATTACHED
    )
    assert steps == (
        ["attach", {"retain_split": True}, "decode"] if enabled else ["attach"]
    )


@pytest.mark.parametrize(
    "rows,fused,gpu",
    [(2, True, True), (2, False, True), (4, True, True), (4, False, False)],
)
def test_rollback_body_preserves_two_row_tuple_and_wide_boundaries(
    monkeypatch, rows, fused, gpu
):
    import mlx.core as mx
    import mlx.nn as nn

    class Cache(list):
        def __init__(self):
            super().__init__([np.zeros((1, 2, 96)), np.zeros((1, 1, 32, 32))])
            self.n_confirmed_for_mtp = 1
            self.advanced = 0

        def advance(self, count):
            self.advanced += count

    class Layer:
        sharding_group = None
        training = False
        num_v_heads = num_k_heads = 1
        head_v_dim = head_k_dim = key_dim = value_dim = 32
        conv_kernel_size = 3
        conv_dim = 96
        A_log = dt_bias = None

        def __call__(self, *args, **kwargs):
            raise AssertionError("expected a rollback-aware verify")

        conv1d = staticmethod(lambda x: x[:, 2:])
        norm = staticmethod(lambda x, z: x)
        out_proj = staticmethod(lambda x: x)

    outputs = [np.ones((1, rows, width)) for width in (96, 32, 1, 1)]
    layer = Layer()
    for name, output in zip(
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), outputs
    ):
        setattr(layer, name, lambda x, output=output: output)
    if fused:
        layer.in_proj_fused = object()
    monkeypatch.setattr(fusion, "_fused_projections", lambda *args: outputs)
    for name, value in (
        ("split", np.split),
        ("concatenate", np.concatenate),
        ("contiguous", np.ascontiguousarray),
        ("default_device", lambda: "gpu" if gpu else "cpu"),
        ("gpu", "gpu"),
        ("fast", SimpleNamespace(rms_norm=lambda x, *args: x)),
        ("metal", SimpleNamespace(is_available=lambda: True)),
    ):
        # Patch the namespace dict: on the headless MagicMock stub, undoing a
        # setattr of an auto-created child deletes it for later tests.
        monkeypatch.setitem(vars(mx), name, value)
    monkeypatch.setitem(vars(nn), "silu", lambda x: x)
    calls = []

    def update(q, k, v, a, b, a_log, dt_bias, state, mask, **kwargs):
        return v, state + q.shape[1]

    def boundary(q, k, v, a, b, a_log, dt_bias, state, mask, **kwargs):
        calls.append(q.shape[1])
        snapshots = np.stack([state + i for i in range(1, rows)], axis=1)
        return v, state + rows, snapshots

    modules = {
        "mlx_lm.models.cache": dict(ArraysCache=Cache),
        "mlx_lm.models.qwen3_5": dict(GatedDeltaNet=Layer),
        "mlx_lm.models.gated_delta": dict(gated_delta_update=update),
        "rapid_mlx.kernels.qwen4_gdn_verify": dict(
            gated_delta_verify_with_states=boundary
        ),
    }
    for name, values in modules.items():
        fake = ModuleType(name)
        vars(fake).update(values)
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(cache_patch, "_gated_delta_patched", False)
    monkeypatch.setattr(cache_patch, "_orig_gated_delta_call", None)
    assert cache_patch.patch_gated_delta_net_for_mtp()
    cache = Cache()
    result = layer(np.ones((1, rows, 1)), cache=cache)
    assert result.shape == (1, rows, 32)
    assert cache.advanced == rows
    assert np.all(cache[1] == rows)
    assert calls == ([rows] if gpu and (fused or rows > 2) else [])
    if rows == 2:
        assert isinstance(cache.rollback_state, tuple)
        assert np.all(cache.rollback_state[1] == 1)
    else:
        assert len(cache.rollback_state) == rows - 1
        assert all(
            np.all(pair[1] == i) for i, pair in enumerate(cache.rollback_state, 1)
        )
