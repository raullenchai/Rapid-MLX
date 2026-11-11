"""Dispatch characterization through the existing inert MLX import seam."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture(autouse=True)
def builtin_policy_only(monkeypatch):
    from rapid_mlx import lattice
    from rapid_mlx.lattice.defaults import Profile

    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.setattr(lattice, "overlay_profile", lambda: Profile())
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)


class ReachedDispatchError(Exception):
    """Stop before a tensor operation or kernel would execute."""


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype

    def reshape(self, *args):
        return self


def fake_mx():
    mx = MagicMock()
    mx.array = Tensor
    mx.bfloat16, mx.float16, mx.uint32 = "bf16", "fp16", "u32"
    mx.default_device.return_value = mx.gpu
    mx.metal.is_available.return_value = True
    return mx


@pytest.fixture
def affine(monkeypatch):
    module = importlib.import_module("rapid_mlx.kernels.affine8_attention")
    monkeypatch.setattr(module, "mx", fake_mx())
    monkeypatch.setattr(module, "_DISABLED", False)
    monkeypatch.setattr(module, "_qualified_runtime", lambda: True)
    monkeypatch.delenv("RAPID_MLX_AFFINE8_ATTENTION", raising=False)
    kernel = Mock(side_effect=ReachedDispatchError)
    monkeypatch.setattr(module, "_kernel", lambda: kernel)
    return module


def affine_call(module, width, kv):
    q = Tensor((1, 2, width, 256))
    packed = (
        Tensor((1, 2, kv, 64), "u32"),
        Tensor((1, 2, kv, 4)),
        Tensor((1, 2, kv, 4)),
    )
    return module.try_affine8_attention(q, packed, packed, scale=1.0, mask="causal")


@pytest.mark.parametrize(
    "width,threshold",
    [(1, 16384), (2, 8192), (3, 8192), (4, 2048), (8, 2048), (9, 2048)],
)
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_affine8_crossovers(affine, width, threshold, delta):
    """rapid_mlx/kernels/affine8_attention.py:88 pins physical KV crossover."""
    if width <= 8 and delta >= 0:
        with pytest.raises(ReachedDispatchError):
            affine_call(affine, width, threshold + delta)
    else:
        assert affine_call(affine, width, threshold + delta) is None


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, True),
        ("", True),
        ("0", False),
        (" 0 ", True),
        ("false", True),
        ("OFF", True),
        ("garbage", True),
    ],
)
def test_affine8_exact_zero_env(affine, monkeypatch, raw, enabled):
    """rapid_mlx/kernels/affine8_attention.py:67 disables only exact zero."""
    if raw is not None:
        monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", raw)
    if enabled:
        with pytest.raises(ReachedDispatchError):
            affine_call(affine, 1, 16384)
    else:
        assert affine_call(affine, 1, 16384) is None


@pytest.mark.parametrize(
    "name,width,kv,arch",
    [
        ("qsa_stage1", 64, 65024, "applegpu_g14s"),
        ("qsa_indexed_splitk", 1, 65536, "applegpu_g15d"),
        ("qsa_indexed_splitk", 3, 16384, "applegpu_g15d"),
        ("qsa_block_sparse", 64, 16384, None),
    ],
)
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("width_delta", [-1, 0, 1])
def test_qsa_crossovers(monkeypatch, name, width, kv, arch, delta, width_delta):
    """rapid_mlx/kernels/qsa_stage1.py:68, qsa_indexed_splitk.py:78, qsa_block_sparse.py:188 pin crossovers."""
    module = importlib.import_module("rapid_mlx.kernels." + name)
    monkeypatch.setattr(module, "mx", fake_mx())
    monkeypatch.setenv(module.ENABLE_ENV, "1")
    function = getattr(
        module,
        {
            "qsa_stage1": "qsa_stage1_decline_reason",
            "qsa_indexed_splitk": "indexed_splitk_decline_reason",
            "qsa_block_sparse": "block_sparse_decline_reason",
        }[name],
    )
    kwargs = (
        {}
        if arch is None
        else dict(batch_size=1, mlx_version="0.32.2", metal_architecture=arch)
    )
    query = width + width_delta
    threshold = (65536 if query == 1 else 16384) if name == "qsa_indexed_splitk" else kv
    eligible_query = query in (1, 3) if name == "qsa_indexed_splitk" else query >= 64
    assert (function(query, kv + delta, **kwargs) is None) == (
        eligible_query and kv + delta >= threshold
    )


@pytest.mark.parametrize(
    "name", ["qsa_stage1", "qsa_indexed_splitk", "qsa_block_sparse"]
)
@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        (" 0 ", False),
        ("OFF", False),
        ("garbage", False),
        ("1", True),
        (" TRUE ", True),
        ("enabled", True),
    ],
)
def test_qsa_opt_in(monkeypatch, name, raw, enabled):
    """rapid_mlx/kernels/qsa_stage1.py:62, qsa_indexed_splitk.py:72, qsa_block_sparse.py:183 require opt-in."""
    module = importlib.import_module("rapid_mlx.kernels." + name)
    monkeypatch.setattr(module, "mx", fake_mx())
    monkeypatch.delenv(module.ENABLE_ENV, raising=False)
    if raw is not None:
        monkeypatch.setenv(module.ENABLE_ENV, raw)
    if name == "qsa_block_sparse":
        reason = module.block_sparse_decline_reason(64, 65536)
    else:
        fn = (
            module.qsa_stage1_decline_reason
            if name == "qsa_stage1"
            else module.indexed_splitk_decline_reason
        )
        reason = fn(
            64 if name == "qsa_stage1" else 1,
            65536,
            batch_size=1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
    assert reason == (None if enabled else "disabled")


@pytest.mark.parametrize("name", ["qsa_stage1", "qsa_indexed_splitk"])
@pytest.mark.parametrize(
    "arch", ["applegpu_g14s", "applegpu_g15d", "applegpu_g16s", "unknown"]
)
@pytest.mark.parametrize("version", ["0.32.1", "0.32.2", "0.32.3", "unknown"])
def test_qsa_runtime_allowlists(monkeypatch, name, arch, version):
    """rapid_mlx/lattice/capabilities.py:5 pins non-bypassable QSA qualification."""
    module = importlib.import_module("rapid_mlx.kernels." + name)
    monkeypatch.setattr(module, "mx", fake_mx())
    monkeypatch.setenv(module.ENABLE_ENV, "1")
    fn = (
        module.qsa_stage1_decline_reason
        if name == "qsa_stage1"
        else module.indexed_splitk_decline_reason
    )
    reason = fn(
        64 if name == "qsa_stage1" else 1,
        65536,
        batch_size=1,
        mlx_version=version,
        metal_architecture=arch,
    )
    allowed = arch == "applegpu_g15d" or (
        name == "qsa_stage1" and arch == "applegpu_g14s"
    )
    assert (reason is None) == (allowed and version == "0.32.2")


@pytest.mark.parametrize("hidden", [0, 2047, 2048, 2049, None])
@pytest.mark.parametrize("allow_dense", [False, True])
def test_gdn_width(monkeypatch, hidden, allow_dense):
    """rapid_mlx/qwen35_fused_gdn_decode.py:382 pins loader-dependent width enrollment."""
    module = importlib.import_module("rapid_mlx.qwen35_fused_gdn_decode")
    layer = SimpleNamespace(
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
    assert module._structurally_eligible(layer, allow_dense=allow_dense) == (
        hidden == 2048 or (allow_dense and hidden is not None and hidden > 0)
    )
    del layer.norm
    assert not module._structurally_eligible(layer, allow_dense=allow_dense)


@pytest.mark.parametrize(
    "shape,fused",
    [((1, 8, 8), True), ((2, 4, 8), True), ((1, 9, 8), False), ((3, 3, 8), False)],
)
@pytest.mark.parametrize("qualified_dtype", [False, True])
def test_in_proj_rows(monkeypatch, shape, fused, qualified_dtype):
    """rapid_mlx/gdn_in_proj_fusion.py:159 pins batch times sequence and probe dtypes."""
    module = importlib.import_module("rapid_mlx.gdn_in_proj_fusion")
    mx = fake_mx()
    monkeypatch.setattr(module, "mx", mx)
    projection = MagicMock(group_size=64, bits=4)
    layer = SimpleNamespace(
        in_proj_fused=projection,
        _rapid_gdn_bounds=(1, 2, 3, 4),
        _rapid_gdn_dtypes={"bf16"} if qualified_dtype else set(),
    )
    module._fused_projections(layer, Tensor(shape))
    assert projection.call_count == int(fused and qualified_dtype)
    assert mx.quantized_matmul.call_count == (0 if fused and qualified_dtype else 4)


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, True),
        ("", True),
        ("0", False),
        (" 0 ", True),
        ("false", True),
        ("OFF", True),
        ("garbage", True),
    ],
)
def test_projection_and_moe_exact_zero(monkeypatch, raw, enabled):
    """rapid_mlx/gdn_in_proj_fusion.py:605 and moe_fusion.py:400 pin exact-zero switches."""
    gdn = importlib.import_module("rapid_mlx.gdn_in_proj_fusion")
    moe = importlib.import_module("rapid_mlx.moe_fusion")
    install = Mock(return_value=7)
    families = Mock(return_value=())
    monkeypatch.setattr(gdn, "_install", install)
    monkeypatch.setattr(moe, "_switch_layer_families", families)
    for key in ("RAPID_MLX_GDN_IN_PROJ_FUSION", "RAPID_MLX_MOE_GATE_UP_FUSION"):
        monkeypatch.delenv(key, raising=False)
        if raw is not None:
            monkeypatch.setenv(key, raw)
    assert gdn.fuse_gdn_in_proj(object()) == (7 if enabled else 0)
    moe.fuse_gate_up(object())
    assert bool(families.call_count) is enabled


@pytest.fixture
def lane(monkeypatch):
    # The inert seam supplies importable modules; real base classes are needed
    # for the installer's mixins, but their tensor methods are never invoked.
    import mlx
    import mlx.nn as nn

    monkeypatch.setattr(mlx, "nn", nn, raising=False)
    for name in ("Linear", "QuantizedLinear"):
        if not isinstance(getattr(nn, name, None), type):
            monkeypatch.setattr(nn, name, type(name, (), {}), raising=False)
    return importlib.import_module("rapid_mlx.kernels.lane_matmul")


@pytest.mark.parametrize(
    "raw,mode",
    [
        (None, "off"),
        ("", "off"),
        ("0", "off"),
        (" false ", "off"),
        ("NO", "off"),
        ("OFF", "off"),
        (" Crossover ", "crossover"),
        ("exact", "exact"),
        ("1", None),
        ("true", None),
        ("disabled", None),
        ("garbage", None),
    ],
)
def test_lane_mode_parser(lane, monkeypatch, raw, mode):
    """rapid_mlx/kernels/lane_matmul/__init__.py:64 pins the mode parser."""
    monkeypatch.delenv(lane.ENABLE_ENV, raising=False)
    if raw is not None:
        monkeypatch.setenv(lane.ENABLE_ENV, raw)
    if mode is None:
        with pytest.raises(ValueError):
            lane.mode_from_env()
    else:
        assert lane.mode_from_env() == mode


@pytest.mark.parametrize("mode", ["crossover", "exact"])
@pytest.mark.parametrize("fmt", ["q2", "q3", "q4", "q5", "q6", "q8", "bf16", "fp16"])
@pytest.mark.parametrize("rows", [0, 1, 7, 8, 15, 16, 32, 33])
def test_lane_effective_rows(lane, monkeypatch, mode, fmt, rows):
    """rapid_mlx/kernels/lane_matmul/__init__.py:139 and installer.py:132 pin installed row dispatch."""
    inst = importlib.import_module("rapid_mlx.kernels.lane_matmul.installer")

    class Stock:
        def _lane_stock_call(self, x):
            return "stock"

    class Installed(inst._LaneMixin, Stock):
        pass

    projection = Stock()
    model = SimpleNamespace(named_modules=lambda: [("proj", projection)])
    monkeypatch.setattr(inst, "_SWAP", {Stock: Installed})
    monkeypatch.setattr(inst, "_RESTORE", {Installed: Stock})
    monkeypatch.setattr(inst, "format_class", lambda _: fmt)
    monkeypatch.setattr(
        inst,
        "prepare",
        lambda _: SimpleNamespace(format=fmt, backend="simd", bias=None),
    )
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "available", lambda: True)
    monkeypatch.setattr(inst, "_group_siblings", lambda *_: {})
    monkeypatch.setattr(inst, "_check_simd_twins", lambda _: {})
    monkeypatch.setattr(inst, "_enable_stock_stacks", lambda _: {})
    monkeypatch.setattr(inst, "_LIVE", [True])
    monkeypatch.setattr(inst, "lane_matmul", lambda *_: "lane")
    monkeypatch.setattr(inst, "STATS", inst.Counter())
    monkeypatch.setenv(lane.ENABLE_ENV, "off")
    monkeypatch.setattr(lane, "available", lambda: True)
    monkeypatch.setattr(lane, "is_moe", lambda _: False)
    monkeypatch.setattr(lane, "install", inst.install)
    assert lane.install_lane_matmul(model, mode=mode)["mode"] == mode
    minimum = 1 if mode == "exact" else 16 if fmt in ("bf16", "fp16") else 8
    assert projection(Tensor((1, rows, 8))) == (
        "lane" if minimum <= rows <= 32 else "stock"
    )


@pytest.mark.parametrize("routes", [31, 32, 63, 64])
def test_moe_sort(monkeypatch, routes):
    """rapid_mlx/moe_fusion.py:209 pins ordinary fused expert sorting."""
    module = importlib.import_module("rapid_mlx.moe_fusion")
    mx = fake_mx()
    mx.split.return_value = (MagicMock(), MagicMock())
    monkeypatch.setattr(module, "mx", mx)
    gather = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    scatter = Mock(return_value=MagicMock())
    layer = SimpleNamespace(
        gate_up_proj=Mock(return_value=MagicMock()),
        down_proj=Mock(return_value=MagicMock()),
        activation=Mock(),
        training=False,
    )
    indices = SimpleNamespace(size=routes, shape=(1, routes, 1))
    module._make_fused_call(Mock(), gather, scatter)(layer, MagicMock(), indices)
    assert gather.call_count == scatter.call_count == int(routes >= 64)
    assert layer.gate_up_proj.call_args.kwargs["sorted_indices"] == (routes >= 64)


@pytest.mark.parametrize("batch,top_k", [(1, 63), (1, 64), (7, 9), (8, 8)])
@pytest.mark.parametrize("length", [1, 2, 8, 9])
def test_moe_short_block(monkeypatch, batch, top_k, length):
    """rapid_mlx/moe_fusion.py:341 pins sequence and strict batch times top-k limits."""
    module = importlib.import_module("rapid_mlx.moe_fusion")
    monkeypatch.setattr(module, "mx", fake_mx())
    supports = Mock(return_value=False)
    tokenwise = Mock(return_value="tokenwise")
    routed = Mock(return_value="routed")
    monkeypatch.setattr(module, "_make_fused_call", lambda *_: routed)
    helpers = (
        lambda value, *_: value,
        8,
        tokenwise,
        type("Quantized", (), {}),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        supports,
    )
    fn = module._make_extended_fused_call(Mock(), Mock(), Mock(), helpers)
    layer = SimpleNamespace(gate_up_proj=object(), training=False)
    fn(layer, Tensor((batch, length, 512)), Tensor((batch, length, top_k)))
    short = 1 < length <= 8
    assert bool(supports.call_count) == (short and batch * top_k < 64)
    assert bool(tokenwise.call_count) == short
    assert bool(routed.call_count) == (not short)


@pytest.mark.parametrize(
    "chip",
    ["Apple M3 Ultra", "Apple M3 Max", "Apple M2 Ultra", "Apple M4 Ultra", "unknown"],
)
@pytest.mark.parametrize("version", ["0.32.2", "0.32.3", "0.32.4", "unknown"])
@pytest.mark.parametrize("os_version", ["26.5.1", "26.5.2", "26.5.3"])
def test_affine8_runtime_qualification(monkeypatch, chip, version, os_version):
    """rapid_mlx/kernels/affine8_attention.py:22 and lattice/profiles/builtin.json:159 pin qualification."""
    module = importlib.import_module("rapid_mlx.kernels.affine8_attention")
    from rapid_mlx.lattice import runtime_context

    mx = fake_mx()
    mx.__version__ = version
    mx.device_info.return_value = {"device_name": chip}
    monkeypatch.setattr(module, "mx", mx)
    monkeypatch.setattr(
        module,
        "runtime_context",
        lambda *a, **kw: runtime_context(*a, macos=os_version, **kw),
    )
    monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", "1")
    module._qualified_runtime.cache_clear()
    try:
        qualified = (
            chip == "Apple M3 Ultra" and version == "0.32.3" and os_version == "26.5.2"
        )
        assert module._qualified_runtime() == qualified
        mx.metal.is_available.return_value = False
        module._qualified_runtime.cache_clear()
        assert not module._qualified_runtime()
    finally:
        module._qualified_runtime.cache_clear()


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, True),
        ("", True),
        ("0", False),
        (" 0 ", False),
        ("false", False),
        ("OFF", False),
        ("disabled", False),
        ("garbage", True),
        ("1", True),
    ],
)
def test_gdn_decode_env(monkeypatch, raw, enabled):
    """rapid_mlx/qwen35_fused_gdn_decode.py:426 pins falsey decode switch before family scanning."""
    module = importlib.import_module("rapid_mlx.qwen35_fused_gdn_decode")
    key = "RAPID_MLX_QWEN35_FUSED_GDN_DECODE"
    monkeypatch.delenv(key, raising=False)
    if raw is not None:
        monkeypatch.setenv(key, raw)
    scan = Mock(return_value=[])
    assert (
        module.install_qwen35_fused_gdn_decode(SimpleNamespace(named_modules=scan)) == 0
    )
    assert bool(scan.call_count) is enabled


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("routes", [31, 32, 63, 64])
def test_native_expert_sort(monkeypatch, native, routes):
    """rapid_mlx/models/deepseek_v4_switch.py:55 pins implementation-specific sort crossover."""
    import sys
    from types import ModuleType

    import mlx
    import mlx.nn as nn

    from rapid_mlx import _mlx_compat

    monkeypatch.setattr(_mlx_compat, "install", lambda: None)
    if not isinstance(getattr(nn, "Module", None), type):
        monkeypatch.setattr(nn, "Module", type("Module", (), {}), raising=False)
    monkeypatch.setattr(mlx, "nn", nn, raising=False)
    upstream = ModuleType("mlx_lm.models.switch_layers")
    upstream.SwiGLU = MagicMock()
    upstream.SwitchLinear = MagicMock()
    upstream._gather_sort = Mock()
    upstream._scatter_unsort = Mock()
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    module = importlib.import_module("rapid_mlx.models.deepseek_v4_switch")
    mx = fake_mx()
    mx.split.return_value = (MagicMock(), MagicMock())
    monkeypatch.setattr(module, "mx", mx)
    identity = SimpleNamespace(startswith=Mock(return_value=native))
    monkeypatch.setattr(module, "SwitchLinear", SimpleNamespace(__module__=identity))
    gather = Mock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    scatter = Mock(return_value=MagicMock())
    monkeypatch.setattr(module, "_gather_sort", gather)
    monkeypatch.setattr(module, "_scatter_unsort", scatter)
    layer = SimpleNamespace(
        gate_proj=Mock(),
        down_proj=Mock(return_value=MagicMock()),
        activation=Mock(),
        hidden_dims=8,
        training=False,
    )
    module.FusedSwitchGLU.__call__(
        layer, MagicMock(), SimpleNamespace(size=routes, shape=(1, routes, 1))
    )
    selected = routes >= (32 if native else 64)
    assert gather.call_count == scatter.call_count == int(selected)
    assert layer.gate_proj.call_args.kwargs["sorted_indices"] is selected


@pytest.mark.parametrize(
    "name", ["qsa_stage1", "qsa_indexed_splitk", "qsa_block_sparse"]
)
@pytest.mark.parametrize("exclusion", ["training", "metal", "batch"])
def test_qsa_local_exclusions(monkeypatch, name, exclusion):
    """rapid_mlx/kernels/qsa_stage1.py:64, qsa_indexed_splitk.py:74, qsa_block_sparse.py:186 pin runtime exclusions."""
    module = importlib.import_module("rapid_mlx.kernels." + name)
    mx = fake_mx()
    mx.metal.is_available.return_value = exclusion != "metal"
    monkeypatch.setattr(module, "mx", mx)
    monkeypatch.setenv(module.ENABLE_ENV, "1")
    if name == "qsa_block_sparse":
        reason = module.block_sparse_decline_reason(
            64, 65536, training=exclusion == "training"
        )
        # This selector intentionally has no batch qualification argument.
        assert (
            reason
            == {
                "training": "training",
                "metal": "Metal runtime unavailable",
                "batch": None,
            }[exclusion]
        )
    else:
        fn = (
            module.qsa_stage1_decline_reason
            if name == "qsa_stage1"
            else module.indexed_splitk_decline_reason
        )
        reason = fn(
            64 if name == "qsa_stage1" else 1,
            65536,
            training=exclusion == "training",
            batch_size=2 if exclusion == "batch" else 1,
            mlx_version="0.32.2",
            metal_architecture="applegpu_g15d",
        )
        assert (
            reason
            == {
                "training": "training",
                "metal": "Metal runtime unavailable",
                "batch": "batch size is not qualified",
            }[exclusion]
        )
