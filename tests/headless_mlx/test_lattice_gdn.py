"""Effective GDN dispatch equivalence through the inert tensor seam."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults
from rapid_mlx.lattice.gdn import GDNPolicy
from tests.headless_mlx.test_lattice_gate_characterization import Tensor, fake_mx


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    from rapid_mlx import gdn_in_proj_fusion as projection
    from rapid_mlx import qwen35_fused_gdn_decode as decode

    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    for key in (
        "RAPID_MLX_BENCHMARK_KNOBS",
        "RAPID_MLX_GDN_IN_PROJ_FUSION",
        "RAPID_MLX_QWEN35_FUSED_GDN_DECODE",
    ):
        monkeypatch.delenv(key, raising=False)
    mx = fake_mx()
    mx.device_info.return_value = {}
    adapter = GDNPolicy(lambda: mx)
    for module in (projection, decode):
        monkeypatch.setattr(module, "mx", mx)
        monkeypatch.setattr(module, "_POLICY", adapter)
    return projection, decode, mx, adapter


def overlay(monkeypatch, knobs, hardware=None):
    cell = {"id": "candidate", "knobs": knobs}
    if hardware is not None:
        cell["hardware"] = hardware
    monkeypatch.setattr(
        lattice, "overlay_profile", lambda: defaults.Profile(cells=(cell,))
    )


def layer(cls, hidden):
    obj = cls()
    obj.__dict__.update(
        hidden_size=hidden,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        in_proj_fused=object(),
        conv1d=SimpleNamespace(weight=Tensor((8192, 4, 1))),
        A_log=Tensor((32,)),
        dt_bias=Tensor((32,)),
        norm=SimpleNamespace(weight=Tensor((128,))),
    )
    return obj


@pytest.mark.parametrize(
    "device", [{}, {"device_name": "Apple M99 Ultra", "architecture": "applegpu_g99z"}]
)
@pytest.mark.parametrize("hidden", [None, 0, -1, 2047, 2048, 2048.0, 2049])
@pytest.mark.parametrize("allow_dense", [False, True])
@pytest.mark.parametrize("probe", [None, 16])
def test_decode_effective_enrollment(
    policy, monkeypatch, device, hidden, allow_dense, probe
):
    _, decode, mx, _ = policy
    mx.device_info.return_value = device
    cls = type("GDN", (), {})
    monkeypatch.setitem(
        sys.modules, "mlx_lm.models.qwen3_5", SimpleNamespace(GatedDeltaNet=cls)
    )
    obj = layer(cls, hidden)
    monkeypatch.setattr(decode, "probe_qwen35_fused_gdn_decode", lambda: probe)
    monkeypatch.setattr(decode, "_patch_class", lambda _: None)
    model = SimpleNamespace(named_modules=lambda: [("gdn", obj)])
    old = (
        hidden == 2048 or (allow_dense and type(hidden) is int and hidden > 0)
    ) and probe is not None
    assert decode.install_qwen35_fused_gdn_decode(
        model, allow_dense=allow_dense
    ) == int(old)
    assert bool(getattr(obj, decode._TAG, False)) == old
    # Projected geometry remains a non-overridable safety boundary.
    obj.head_v_dim = 127
    assert decode.install_qwen35_fused_gdn_decode(model, allow_dense=allow_dense) == 0


@pytest.mark.parametrize("rows", [7, 8, 9])
@pytest.mark.parametrize("dtype_ok", [False, True])
@pytest.mark.parametrize("maximum", [None, 7, 99, 0])
def test_projection_effective_dispatch(policy, monkeypatch, rows, dtype_ok, maximum):
    projection, _, mx, _ = policy
    if maximum is not None:
        overlay(monkeypatch, {"gdn_in_proj_max_rows": maximum})
    fused = MagicMock(group_size=64, bits=4)
    obj = SimpleNamespace(
        in_proj_fused=fused,
        _rapid_gdn_bounds=(1, 2, 3, 4),
        _rapid_gdn_dtypes={"bf16"} if dtype_ok else set(),
        _rapid_gdn_max_rows=projection.resolved_max_rows(),
    )
    projection._fused_projections(obj, Tensor((1, rows, 8)))
    old = rows <= 8 and dtype_ok
    expected = old if maximum is None else old and rows <= maximum
    assert fused.call_count == int(expected)
    assert mx.quantized_matmul.call_count == (0 if expected else 4)


@pytest.mark.parametrize("retain_split", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_projection_install_retains_split(policy, monkeypatch, retain_split, enabled):
    projection, _, _, _ = policy
    overlay(monkeypatch, {"gdn_in_proj_fusion": enabled})
    cls = type("GDN", (), {})
    obj = cls()
    parts = [MagicMock() for _ in range(4)]
    for part in parts:
        part.__getitem__.return_value.shape = (1, 8)
    for name, part in zip(
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), parts
    ):
        setattr(obj, name, part)
    model = SimpleNamespace(named_modules=lambda: [("gdn", obj)])
    monkeypatch.setattr(projection, "_gdn_imports", lambda: (None, None, cls))
    monkeypatch.setattr(projection, "_vlm_gdn_imports", lambda: None)
    monkeypatch.setattr(projection, "_can_fuse", lambda _: True)
    monkeypatch.setattr(projection, "_proj_modules", lambda _: parts)
    monkeypatch.setattr(
        projection, "_concat_params", lambda _: (object(), object(), object())
    )
    monkeypatch.setattr(
        projection, "_probe_projection_parity", lambda _: frozenset({"bf16"})
    )
    whole = Mock(return_value=True)
    monkeypatch.setattr(projection, "_probe_whole_layer_parity", whole)
    monkeypatch.setattr(projection, "_ensure_call_patch", Mock())
    assert projection.fuse_gdn_in_proj(model, retain_split=retain_split) == int(enabled)
    assert hasattr(obj, "in_proj_fused") == enabled
    assert hasattr(obj, "in_proj_qkv") == (retain_split or not enabled)
    assert whole.call_count == int(enabled and not retain_split)


@pytest.mark.parametrize("matching", [False, True])
def test_real_device_specific_overlay(policy, monkeypatch, matching):
    projection, _, mx, adapter = policy
    mx.device_info.return_value = {
        "device_name": "Apple M3 Ultra" if matching else "Apple M99 Ultra",
        "architecture": "applegpu_g15d" if matching else "applegpu_g99z",
    }
    overlay(
        monkeypatch,
        {"gdn_in_proj_fusion": False},
        [{"gpu_generation": 15, "tier": "Ultra"}],
    )
    install = Mock(return_value=3)
    monkeypatch.setattr(projection, "_install", install)
    assert projection.fuse_gdn_in_proj(object()) == (0 if matching else 3)
    for _ in range(5):
        adapter.projection_enabled()
    mx.device_info.assert_called_once()


def test_missing_device_facts_and_cache_bound(policy):
    _, _, mx, adapter = policy
    mx.device_info.side_effect = RuntimeError("unavailable")
    assert adapter.decode(None, True) is False
    for hidden in range(1, 300):
        assert adapter.decode(hidden, True)
    assert adapter.value.cache_info().currsize == 256
    mx.device_info.assert_called_once()


@pytest.mark.parametrize("raw", ["0", "false", "OFF", "", " 0 "])
def test_dynamic_env_master_gates(policy, monkeypatch, raw):
    projection, decode, _, adapter = policy
    # Warm lattice decisions first: legacy switches must still be dynamic.
    assert adapter.projection_enabled() and adapter.decode(2048, False)
    install = Mock(return_value=5)
    monkeypatch.setattr(projection, "_install", install)
    monkeypatch.setenv("RAPID_MLX_GDN_IN_PROJ_FUSION", raw)
    assert projection.fuse_gdn_in_proj(object()) == (0 if raw == "0" else 5)
    monkeypatch.setenv("RAPID_MLX_QWEN35_FUSED_GDN_DECODE", raw)
    modules = Mock(return_value=[])
    decode.install_qwen35_fused_gdn_decode(SimpleNamespace(named_modules=modules))
    assert modules.call_count == (
        0 if raw.strip().lower() in {"0", "false", "off"} else 1
    )


def test_decode_overlay_and_missing_context(policy, monkeypatch):
    _, decode, _, adapter = policy
    overlay(monkeypatch, {"qwen35_fused_gdn_decode": False})
    cls = type("GDN", (), {})
    monkeypatch.setitem(
        sys.modules, "mlx_lm.models.qwen3_5", SimpleNamespace(GatedDeltaNet=cls)
    )
    obj = layer(cls, 2048)
    probe = Mock(return_value=16)
    monkeypatch.setattr(decode, "probe_qwen35_fused_gdn_decode", probe)
    model = SimpleNamespace(named_modules=lambda: [("gdn", obj)])
    assert decode.install_qwen35_fused_gdn_decode(model) == 0
    probe.assert_not_called()
    overlay(monkeypatch, {"qwen35_fused_gdn_decode": True})
    assert not adapter.decode(None, True)
    for family, hidden in (("", 2048), ("qwen35_gdn_dense", None)):
        ctx = lattice.runtime_context(
            {}, model=lattice.ModelProfile(family=family, hidden_size=hidden)
        )
        assert (
            lattice.resolve("qwen35_fused_gdn_decode", ctx, environ={}).value is False
        )
