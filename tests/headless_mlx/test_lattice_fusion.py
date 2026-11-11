"""Effective installation/dispatch contracts; tensor probes remain native tests."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults, fusion
from tests.headless_mlx.test_lattice_gate_characterization import Tensor, fake_mx

NAMES = (
    "dense_decode_fusion",
    "qwen35_affine4",
    "qwen35_moe_router",
    "moe_decode_fusion",
    "moe_gate_up_fusion",
)
MODULES = (
    "rapid_mlx.dense_decode_fusion",
    "rapid_mlx.patches.qwen35_affine4",
    "rapid_mlx.qwen35_moe_router",
    "rapid_mlx.patches.moe_decode_fusion",
    "rapid_mlx.moe_fusion",
)
INSTALLERS = (
    "install_dense_decode_fusion",
    "install_qwen35_affine4",
    "install_qwen35_moe_router",
    "install_moe_decode_fusion",
    "fuse_gate_up",
)
DEVICES = ({}, {"device_name": "Apple M3 Ultra"}, {"device_name": "Apple M9 Max"})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    for name in NAMES:
        monkeypatch.delenv("RAPID_MLX_" + name.upper(), raising=False)


def overlay(monkeypatch, name, *, device_specific=False):
    cell = {
        "id": "disable",
        "knobs": {name: False},
        "provenance": {"evidence": "test only"},
    }
    if device_specific:
        cell["hardware"] = [{"gpu_generation": 15, "tier": "Ultra"}]
    monkeypatch.setattr(
        lattice, "overlay_profile", lambda: defaults.Profile(cells=(cell,))
    )


def setup(monkeypatch, name, device, *, qualified=True):
    """Real installers and dispatch, inert tensors and explicit probe outcomes."""
    index = NAMES.index(name)
    import mlx

    # Replace the import bindings, avoiding mutation of shared seam mocks.
    nn = SimpleNamespace(Module=type("Module", (), {}))
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setattr(mlx, "nn", nn, raising=False)
    module = importlib.import_module(MODULES[index])
    mx = fake_mx()
    mx.device_info.return_value = device
    monkeypatch.setattr(module, "mx", mx, raising=False)
    if name == "moe_decode_fusion":
        monkeypatch.setitem(sys.modules, "mlx.core", mx)
        monkeypatch.setattr(mlx, "core", mx, raising=False)

    class Stock(str):
        def astype(self, _):
            return MagicMock()

    class Block:
        def __call__(self, x, **kwargs):
            return Stock("stock")

    block = Block()
    probe = Mock(return_value=qualified)
    x = Tensor((1, 1, 2048))
    if name == "dense_decode_fusion":
        monkeypatch.setattr(module, "_BACKBONES", {(Block.__module__, "Block"): "lfm"})
        norm = SimpleNamespace(weight=SimpleNamespace(dtype=mx.bfloat16, size=2048))
        block.args = SimpleNamespace(num_experts=0)
        block.embedding_norm = norm
        block.layers = [
            SimpleNamespace(operator_norm=norm, ffn_norm=norm, feed_forward=object())
        ]
        monkeypatch.setattr(module, "_norm_probe", probe)
        monkeypatch.setattr(module, "_probe_cache", lambda *_: [])
        monkeypatch.setattr(module, "_snapshot", lambda *_: [])
        monkeypatch.setattr(module, "_group_mlp", lambda *_: None)
        monkeypatch.setattr(module, "_bits_equal", lambda *_: True)
        # The installer must select and publish its candidate only after probes.
        monkeypatch.setattr(module, "_make_call", lambda *_: lambda *a, **k: "fused")
    elif name == "qwen35_affine4":
        monkeypatch.setattr(module, "_compatible", probe)
        block.shared_expert_gate = MagicMock(bits=4)
        mx.sqrt.return_value.item.return_value = 0.0
        monkeypatch.setattr(module, "forward", lambda *_: MagicMock())
    elif name == "qwen35_moe_router":
        upstream = SimpleNamespace(Qwen3NextSparseMoeBlock=Block)
        monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_next", upstream)
        block.num_experts, block.top_k = 256, 8
        block.norm_topk_prob = True
        monkeypatch.setattr(module, "_probe", probe)
    elif name == "moe_decode_fusion":

        class Linear:
            weight = SimpleNamespace(shape=(64, 2048), ndim=2, dtype=mx.float32)

            def __contains__(self, _):
                return False

        nn.Linear = Linear
        block.gate = Linear()
        block.num_experts, block.top_k = 64, 4
        monkeypatch.setattr(
            module,
            "_block_types",
            lambda: (
                {Block: module._decode_type(Block, Block.__call__)} if qualified else {}
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "_switch_layer_families",
            lambda **_: ((object, Block, object, None, None, None),),
        )
        monkeypatch.setattr(module, "_can_fuse", lambda *_: qualified)
        monkeypatch.setattr(
            module,
            "_fuse_one",
            lambda b, _: setattr(b, "gate_up_proj", Mock(return_value="fused")),
        )
        # Preserve the real fused-aware branch; replace only the tensor body.
        monkeypatch.setattr(
            module, "_make_fused_call", lambda *_: lambda *a, **k: "fused"
        )
    model = SimpleNamespace(named_modules=lambda: [("block", block)])
    install = getattr(module, INSTALLERS[index])
    return module, install, model, block, x, probe


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_effective_installation(monkeypatch, name, device, qualified, disabled):
    module, install, model, block, x, probe = setup(
        monkeypatch, name, device, qualified=qualified
    )
    if disabled:
        overlay(monkeypatch, name)
    # Historical default-on installer with the same local safety/probe result.
    assert install(model) == int(qualified and not disabled)
    if not qualified or disabled:
        assert block(x) == "stock"
        return
    # Freeze the policy after installation: forwards must never enter lattice.
    monkeypatch.setattr(
        module, "install_enabled", Mock(side_effect=AssertionError("hot resolution"))
    )
    if name == "dense_decode_fusion":
        assert block(x) == "fused"
    elif name == "qwen35_affine4":
        monkeypatch.setattr(module, "forward", lambda *_: "fused")
        assert block(x) == "fused"
        assert block(Tensor((1, 10, 2048))) == "stock"
    elif name == "qwen35_moe_router":
        assert module._eligible(x, block)
        assert not module._eligible(Tensor((1, 9, 2048)), block)
        # Enter the real wrapper and observe the fused tensor path.
        block.gate = Mock(side_effect=RuntimeError("fused dispatch"))
        with pytest.raises(RuntimeError, match="fused dispatch"):
            block(x)
    elif name == "moe_decode_fusion":
        monkeypatch.setattr(module, "route", lambda *_: (None, None))
        block.switch_mlp = Mock(return_value=None)
        monkeypatch.setattr(module, "combine", lambda *_: "fused")
        assert block(x) == "fused"
        assert block(Tensor((1, 2, 2048))) == "stock"
    else:
        assert block(x, None) == "fused"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_chip_specific_overlay(monkeypatch, name, device):
    _, install, model, *_ = setup(monkeypatch, name, device)
    overlay(monkeypatch, name, device_specific=True)
    assert install(model) == int(device.get("device_name") != "Apple M3 Ultra")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "raw", [None, "", "0", " 0 ", "false", "OFF", "disabled", "garbage", "1"]
)
def test_legacy_env_and_lifetime(monkeypatch, name, raw):
    module, install, model, block, x, _ = setup(monkeypatch, name, {})
    key = "RAPID_MLX_" + name.upper()
    if raw is not None:
        monkeypatch.setenv(key, raw)
    exact = name in ("dense_decode_fusion", "moe_gate_up_fusion")
    old_enabled = (
        raw != "0"
        if exact
        else raw is None
        or raw.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}
    )
    assert install(model) == int(old_enabled)
    if old_enabled:
        monkeypatch.setenv(key, "0")
        if name == "qwen35_affine4":
            monkeypatch.setattr(module, "forward", lambda *_: "fused")
            assert block(x) == "stock"
        elif name == "moe_decode_fusion":
            assert not module.eligible(block, x)
            assert block(x) == "stock"
        elif name == "qwen35_moe_router":
            assert module._eligible(x, block)
        elif name == "dense_decode_fusion":
            assert block(x) == "fused"
        else:
            assert block(x, None) == "fused"


@pytest.mark.parametrize("name", NAMES)
def test_missing_installer_facts_fail_closed(monkeypatch, name):
    runtime = SimpleNamespace(device_info=lambda: {})
    assert not fusion.install_enabled(name, runtime)
    assert not fusion.install_enabled(name, runtime, route="unqualified")
    if name == "dense_decode_fusion":
        assert not fusion.install_enabled(name, runtime, route=name + "_install")
        module, install, model, *_ = setup(monkeypatch, name, {})
        for mode in (None, "unknown", "mtp", "dspark"):
            assert install(model, spec_decode=mode) == 0
    # An enabling overlay still cannot bypass scope.
    monkeypatch.setattr(
        lattice,
        "overlay_profile",
        lambda: defaults.Profile(cells=({"id": "enable", "knobs": {name: True}},)),
    )
    assert not fusion.install_enabled(name, runtime)


def test_device_failure_retains_universal_default():
    runtime = SimpleNamespace(device_info=Mock(side_effect=RuntimeError("unavailable")))
    for name in NAMES:
        assert fusion.install_enabled(
            name, runtime, route=name + "_install", mode="plain"
        )


@pytest.mark.parametrize("width", [4095, 4096, 4097])
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_dense_actual_hot_dispatch(monkeypatch, width, dtype):
    module = importlib.import_module("rapid_mlx.dense_decode_fusion")
    mx = fake_mx()
    monkeypatch.setattr(module, "mx", mx)
    monkeypatch.setattr(module, "_decode", lambda *_: "fused")
    monkeypatch.setattr(
        module, "install_enabled", Mock(side_effect=AssertionError("hot resolution"))
    )

    class Backbone:
        layers = [object()]

        def __call__(self, inputs, cache=None):
            return "stock"

        def embed_tokens(self, _):
            return Tensor((1, 1, width), dtype)

    call = module._make_call(Backbone, "lfm", "bf16")
    assert call(Backbone(), Tensor((1, 1))) == (
        "fused" if width <= 4096 and dtype == "bf16" else "stock"
    )
    assert call(Backbone(), Tensor((1, 2))) == "stock"
    assert call(Backbone(), Tensor((1, 1)), cache=[]) == "stock"


def test_gate_up_actual_hot_dispatch(monkeypatch):
    module = importlib.import_module("rapid_mlx.moe_fusion")
    monkeypatch.setattr(
        module, "install_enabled", Mock(side_effect=AssertionError("hot resolution"))
    )
    block = SimpleNamespace()
    original = Mock(return_value="stock")
    call = module._make_fused_call(original, None, None)
    assert call(block, object(), object()) == "stock"
    block.gate_up_proj = Mock()
    mx = fake_mx()
    mx.expand_dims.side_effect = RuntimeError("fused dispatch")
    monkeypatch.setattr(module, "mx", mx)
    with pytest.raises(RuntimeError, match="fused dispatch"):
        call(block, object(), object())


@pytest.mark.parametrize("name", NAMES)
def test_unqualified_instances_stay_stock(monkeypatch, name):
    module, install, model, *_ = setup(monkeypatch, name, {})
    # Universal/overlay enablement cannot create absent structural facts.
    overlay(monkeypatch, name)
    monkeypatch.setattr(
        lattice,
        "overlay_profile",
        lambda: defaults.Profile(cells=({"id": "enable", "knobs": {name: True}},)),
    )
    assert install(SimpleNamespace(named_modules=lambda: [])) == 0
