"""Numerics, conservative enrollment, and model lifetime for affine-4 decode."""

import gc
import weakref
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal required", allow_module_level=True)

from rapid_mlx.patches import qwen35_affine4 as fused  # noqa: E402


@pytest.fixture(autouse=True)
def enable_candidate(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN35_AFFINE4", "1")


@pytest.mark.parametrize(
    "seed,k,n,h,experts,topk",
    [(1, 64, 64, 64, 4, 2), (19, 128, 128, 96, 8, 3), (73, 1024, 64, 128, 4, 2)],
)
def test_small_selected_experts_against_dense_oracle(seed, k, n, h, experts, topk):
    """Use small full banks, including the final expert, to catch stride errors."""
    key = mx.random.key(seed)
    dp = mx.quantize(
        mx.random.normal((experts, h, n), key=key).astype(mx.bfloat16) * 0.05,
        group_size=64,
        bits=4,
    )
    gu = mx.quantize(
        mx.random.normal((experts, 2 * n, k), key=key).astype(mx.bfloat16) * 0.05,
        group_size=64,
        bits=4,
    )
    x = mx.random.normal((1, 1, k), key=key).astype(mx.bfloat16)
    ids = mx.array([0, experts - 1] if topk == 2 else [0, 3, experts - 1], mx.uint32)
    scores = mx.array([0.25, 0.75] if topk == 2 else [0.2, 0.3, 0.5], mx.bfloat16)
    gate, down, sig = fused.kernels()
    act = gate(
        inputs=[x, *gu, *gu, ids, sig],
        template=[
            ("T", mx.bfloat16),
            ("ROWS", 1),
            ("K", k),
            ("N", n),
            ("GS", 64),
            ("TOPK", topk),
        ],
        grid=(32, n, topk),
        threadgroup=(32, 2, 1),
        output_shapes=[(topk, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
    dense_gu = mx.dequantize(
        gu[0], gu[1].astype(mx.float32), gu[2].astype(mx.float32), group_size=64, bits=4
    ).astype(mx.float32)[ids]
    proj = (dense_gu @ x.reshape(k).astype(mx.float32)).astype(mx.bfloat16)
    g, u = mx.split(proj, 2, axis=-1)
    expected_act = (g * mx.sigmoid(g)) * u
    zeros = mx.zeros((h,), mx.bfloat16)
    out = down(
        inputs=[act, *dp, ids, scores, zeros, mx.zeros((1,), mx.bfloat16)],
        template=[
            ("T", mx.bfloat16),
            ("ROWS", 1),
            ("I", n),
            ("H", h),
            ("GS", 64),
            ("TOPK", topk),
            ("SHARED", 0),
        ],
        grid=(32, h, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(h,)],
        output_dtypes=[mx.bfloat16],
    )[0]
    dense_dp = mx.dequantize(
        dp[0], dp[1].astype(mx.float32), dp[2].astype(mx.float32), group_size=64, bits=4
    ).astype(mx.float32)[ids]
    expected = (
        (dense_dp @ expected_act.astype(mx.float32)[..., None]).squeeze(-1)
        * scores[:, None]
    ).sum(axis=0)
    # Pending kernel graphs must retain their full banks after the owner unloads.
    mx.eval(expected_act, expected)
    del gu, dp, dense_gu, dense_dp, x, ids, scores
    gc.collect()
    for actual, ref in ((act, expected_act), (out, expected)):
        relative = mx.sqrt(
            mx.mean((actual.astype(mx.float32) - ref.astype(mx.float32)) ** 2)
            / mx.maximum(mx.mean(ref.astype(mx.float32) ** 2), 1e-12)
        ).item()
        assert relative < 0.025
        assert mx.all(mx.isfinite(actual)).item()


def test_parallel_router_ties_and_shared_dot():
    block = SimpleNamespace(
        _rapid_affine4_shared_weight=mx.sin(mx.arange(2048)).astype(mx.float32) / 2048
    )
    x = mx.cos(mx.arange(2048)).reshape(1, 1, 2048).astype(mx.bfloat16)
    for logits in (
        mx.zeros((1, 1, 256), mx.bfloat16),
        mx.sin(mx.arange(256) * 0.731).reshape(1, 1, 256).astype(mx.bfloat16),
    ):
        ids, scores, glog = fused.fused_route(block, x, logits)
        from rapid_mlx.qwen35_moe_router import fused_router_topk

        ei, es = fused_router_topk(mx.softmax(logits, axis=-1, precise=True), 8)
        assert mx.array_equal(ids, ei).item()
        assert mx.array_equal(scores, es).item()
        expected = mx.sum(
            x.astype(mx.float32) * block._rapid_affine4_shared_weight
        ).astype(mx.bfloat16)
        assert mx.allclose(glog, expected, atol=0.001).item()


class Bank(dict):
    def __init__(self, shape):
        super().__init__()
        self.bits, self.group_size, self.mode = 4, 64, "affine"
        self.weight = SimpleNamespace(shape=shape, dtype=mx.uint32)
        self.scales = SimpleNamespace(
            shape=(*shape[:-1], shape[-1] // 8), dtype=mx.bfloat16
        )
        self.biases = SimpleNamespace(shape=self.scales.shape, dtype=mx.bfloat16)


def block_type():
    class Qwen3_5MoeSparseMoeBlock:
        num_experts, top_k, norm_topk_prob = 256, 8, True
        _rapid_qwen35_fused_router = True

        def __init__(self):
            self.switch_mlp = SimpleNamespace(
                gate_up_proj=Bank((256, 1024, 256)), down_proj=Bank((256, 2048, 64))
            )
            self.shared_expert_gate = Bank((1, 256))

        def __call__(self, x):
            return mx.zeros_like(x)

    return Qwen3_5MoeSparseMoeBlock


@pytest.mark.parametrize(
    "mutation",
    ["dense", "experts", "topk", "quant", "scales", "sharded", "normalization", "bias"],
)
def test_geometry_fails_closed(mutation, monkeypatch):
    cls = block_type()
    monkeypatch.setattr(fused, "_block_classes", lambda: (cls,))
    b = cls()
    assert fused._compatible(b)
    if mutation == "dense":
        b.switch_mlp = None
    if mutation == "experts":
        b.num_experts = 32  # another common MoE geometry
    if mutation == "topk":
        b.top_k = 4
    if mutation == "quant":
        b.switch_mlp.down_proj.bits = 8
    if mutation == "scales":
        b.switch_mlp.down_proj.scales.dtype = mx.float16
    if mutation == "sharded":
        b.sharding_group = object()
    if mutation == "normalization":
        b.norm_topk_prob = False
    if mutation == "bias":
        b.shared_expert_gate["bias"] = True
    assert not fused._compatible(b)


def test_reload_multiple_models_fallbacks_and_kill_switch(monkeypatch, caplog):
    caplog.set_level("INFO", logger=fused.__name__)
    cls = block_type()
    monkeypatch.setattr(
        fused.mx, "dequantize", lambda *a, **k: mx.zeros((1, 2048), mx.float32)
    )
    monkeypatch.setattr(fused, "forward", lambda b, x: mx.zeros_like(x))

    def enroll(b):
        b.shared_expert_gate.scales = mx.ones((1, 32), mx.bfloat16)
        b.shared_expert_gate.biases = mx.zeros((1, 32), mx.bfloat16)
        # Keep metadata eligibility checks independently covered above.
        monkeypatch.setattr(fused, "_compatible", lambda candidate: True)
        return fused.install_qwen35_affine4(
            SimpleNamespace(named_modules=lambda: [("block", b)])
        )

    first, second, untouched = cls(), cls(), cls()
    assert enroll(first) == 1
    assert enroll(first) == 0
    assert enroll(second) == 1
    assert "[qwen35_affine4] enrolled 1 MoE blocks" in caplog.text
    assert not getattr(untouched, "_rapid_affine4", False)
    monkeypatch.setattr(fused, "forward", lambda b, x: mx.ones_like(x))
    x = mx.zeros((1, 1, 2048), mx.bfloat16)
    assert first(x).sum().item() == 2048
    assert untouched(x).sum().item() == 0
    assert first(mx.zeros((2, 1, 2048), mx.bfloat16)).sum().item() == 0
    assert first(mx.zeros((1, 2, 2048), mx.bfloat16)).sum().item() == 4096
    assert first(mx.zeros((1, 10, 2048), mx.bfloat16)).sum().item() == 0
    assert first(x.astype(mx.float32)).sum().item() == 0
    assert first(x.astype(mx.float16)).sum().item() == 0
    for value in ("0", "off", "false"):
        monkeypatch.setenv("RAPID_MLX_QWEN35_AFFINE4", value)
        assert first(x).sum().item() == 0
        assert fused.install_qwen35_affine4(object()) == 0
    ref = weakref.ref(first)
    del first
    gc.collect()
    assert ref() is None  # class patch has no model/weight closure
    monkeypatch.delenv("RAPID_MLX_QWEN35_AFFINE4")
    assert second(x).sum().item() == 2048
    third = cls()
    monkeypatch.setattr(fused, "forward", lambda b, x: mx.zeros_like(x))
    assert enroll(third) == 1
    monkeypatch.setattr(fused, "forward", lambda b, x: mx.ones_like(x))
    assert second(x).sum().item() == 2048


def test_dense_model_skipped():
    assert fused.install_qwen35_affine4(SimpleNamespace(named_modules=lambda: [])) == 0


def test_probe_failure_does_not_enroll_or_retain_weights(monkeypatch):
    b = block_type()()
    monkeypatch.setattr(fused, "_compatible", lambda candidate: True)
    b.shared_expert_gate.scales = mx.ones((1, 32), mx.bfloat16)
    b.shared_expert_gate.biases = mx.zeros((1, 32), mx.bfloat16)
    monkeypatch.setattr(fused.mx, "dequantize", lambda *a, **k: mx.zeros((1, 2048)))
    monkeypatch.setattr(fused, "forward", lambda b, x: mx.full(x.shape, float("nan")))
    assert (
        fused.install_qwen35_affine4(
            SimpleNamespace(named_modules=lambda: [("block", b)])
        )
        == 0
    )
    assert not getattr(b, "_rapid_affine4", False)
    assert not hasattr(b, "_rapid_affine4_shared_weight")


def test_shared_gate_mixed_precision_layout(monkeypatch):
    cls = block_type()
    monkeypatch.setattr(fused, "_block_classes", lambda: (cls,))
    b = cls()
    shared = b.shared_expert_gate
    shared.bits = 8
    shared.weight.shape = (1, 512)
    shared.scales.shape = shared.biases.shape = (1, 32)
    assert fused._compatible(b)
    shared.bits = 2
    assert not fused._compatible(b)


def test_other_popular_moe_families_skip():
    from mlx_lm.models.gpt_oss import MLPBlock
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    for cls in (Qwen3MoeSparseMoeBlock, MLPBlock):
        assert not fused._compatible(dict.__new__(cls))


def test_nonfinite_router_never_returns_out_of_bounds_experts():
    block = SimpleNamespace(
        _rapid_affine4_shared_weight=mx.zeros((1, 2048), mx.float32)
    )
    x = mx.zeros((1, 1, 2048), mx.bfloat16)
    for value in (float("nan"), float("inf"), float("-inf")):
        ids, scores, glog = fused.fused_route(
            block, x, mx.full((1, 1, 256), value, mx.bfloat16)
        )
        mx.eval(ids, scores, glog)
        assert mx.all(ids < 256).item()


@pytest.mark.parametrize("width", [2, 3, 5, 9])
@pytest.mark.parametrize("gate_rows,down_rows", [(1, 1), (4, 4), (2, 4)])
def test_short_width_experts_match_independent_singletons(width, gate_rows, down_rows):
    """Every position has distinct routes, activations, scores and shared output.

    Bitwise singleton parity checks token/expert strides and keeps expert
    arithmetic independent of the verifier width, including the final expert.
    The singleton dense-oracle test above separately checks the mathematics.
    """
    experts, k, n, h, topk = 8, 1024, 128, 128, 3
    key = mx.random.key(92)
    gu = mx.quantize(
        mx.random.normal((experts, 2 * n, k), key=key).astype(mx.bfloat16) * 0.05,
        group_size=64,
        bits=4,
    )
    dp = mx.quantize(
        mx.random.normal((experts, h, n), key=key).astype(mx.bfloat16) * 0.05,
        group_size=64,
        bits=4,
    )
    x = mx.sin(mx.arange(width * k) * 0.31).reshape(width, k).astype(mx.bfloat16)
    ids = mx.array(
        [[(t + j) % 7 for j in range(topk - 1)] + [7] for t in range(width)], mx.uint32
    )
    scores = mx.softmax(
        mx.cos(mx.arange(width * topk)).reshape(width, topk), precise=True
    ).astype(mx.bfloat16)
    shared = mx.sin(mx.arange(width * h)).reshape(width, h).astype(mx.bfloat16)
    glog = mx.linspace(-2, 2, width).astype(mx.bfloat16)
    gate, down, sig = fused.kernels()

    def run(xx, ii, ss, sh, gl):
        t = xx.shape[0]
        act = gate(
            inputs=[xx, *gu, *gu, ii, sig],
            template=[
                ("T", mx.bfloat16),
                ("ROWS", gate_rows),
                ("K", k),
                ("N", n),
                ("GS", 64),
                ("TOPK", topk),
            ],
            grid=(32, n // gate_rows, topk * t),
            threadgroup=(32, 2, 1),
            output_shapes=[(t, topk, n)],
            output_dtypes=[mx.bfloat16],
        )[0]
        out = down(
            inputs=[act, *dp, ii, ss, sh, gl],
            template=[
                ("T", mx.bfloat16),
                ("ROWS", down_rows),
                ("I", n),
                ("H", h),
                ("GS", 64),
                ("TOPK", topk),
                ("SHARED", 1),
            ],
            grid=(32, h // down_rows, t),
            threadgroup=(32, 2, 1),
            output_shapes=[(t, h)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return act, out

    actual = run(x, ids, scores, shared, glog)
    singles = [
        run(
            x[t : t + 1],
            ids[t : t + 1],
            scores[t : t + 1],
            shared[t : t + 1],
            glog[t : t + 1],
        )
        for t in range(width)
    ]
    for index in (0, 1):
        expected = mx.concatenate([pair[index] for pair in singles], axis=0)
        assert mx.array_equal(actual[index], expected).item()


@pytest.mark.parametrize("width", [2, 3, 5, 9])
def test_short_width_router_matches_independent_singletons(width):
    block = SimpleNamespace(
        _rapid_affine4_shared_weight=mx.sin(mx.arange(2048)).astype(mx.float32) / 2048
    )
    x = mx.cos(mx.arange(width * 2048)).reshape(1, width, 2048).astype(mx.bfloat16)
    logits = (
        mx.sin(mx.arange(width * 256) * 0.731)
        .reshape(1, width, 256)
        .astype(mx.bfloat16)
    )
    actual = fused.fused_route(block, x, logits)
    singles = [
        fused.fused_route(block, x[:, t : t + 1], logits[:, t : t + 1])
        for t in range(width)
    ]
    for index in range(3):
        expected = mx.concatenate([pair[index] for pair in singles], axis=1)
        assert mx.array_equal(actual[index], expected).item()


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_down_rounds_projection_and_weight_before_route_sum(rows):
    """The composed BF16 weighted terms sum to .0303955, not .0302734."""
    _, down, _ = fused.kernels()
    act = mx.zeros((2, 64), mx.bfloat16)
    act[:, 0] = 1
    weight = mx.zeros((2, 64, 8), mx.uint32)
    scales = mx.zeros((2, 64, 1), mx.bfloat16)
    biases = mx.broadcast_to(
        mx.array([0.010009765625, 0.0390625], mx.bfloat16)[:, None, None],
        (2, 64, 1),
    )
    actual = down(
        inputs=[
            act,
            weight,
            scales,
            biases,
            mx.array([0, 1], mx.uint32),
            mx.array([0.30078125, 0.69921875], mx.bfloat16),
            mx.zeros((64,), mx.bfloat16),
            mx.zeros((1,), mx.bfloat16),
        ],
        template=[
            ("T", mx.bfloat16),
            ("ROWS", rows),
            ("I", 64),
            ("H", 64),
            ("GS", 64),
            ("TOPK", 2),
            ("SHARED", 0),
        ],
        grid=(32, 64 // rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(64,)],
        output_dtypes=[mx.bfloat16],
    )[0]
    assert mx.array_equal(actual, mx.full((64,), 0.0303955078125, mx.bfloat16)).item()


@pytest.mark.parametrize(
    "family,normalized,expected",
    [
        ("lm", False, False),
        ("lm", None, False),
        ("vlm", None, True),
        ("vlm", False, False),
        ("unknown", None, False),
    ],
)
def test_normalization_is_explicit_per_class(monkeypatch, family, normalized, expected):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    template_cls = block_type()
    del template_cls.norm_topk_prob
    classes = {"lm": Qwen3NextSparseMoeBlock, "vlm": Qwen3_5MoeSparseMoeBlock}
    if family in classes:
        block = dict.__new__(classes[family])
        for name, value in vars(template_cls()).items():
            setattr(block, name, value)
        block.num_experts, block.top_k = 256, 8
        block._rapid_qwen35_fused_router = True
    else:
        monkeypatch.setattr(fused, "_block_classes", lambda: (template_cls,))
        block = template_cls()
    if normalized is not None:
        block.norm_topk_prob = normalized
    assert fused._compatible(block) is expected


@pytest.mark.parametrize("chip,tiles", [("Apple M4 Pro", (2, 4)), ("Apple M1", (1, 1))])
def test_kernel_initialization_selects_chip_tiles(monkeypatch, chip, tiles):
    for name in ("_GATE", "_DOWN", "_SIG"):
        monkeypatch.setattr(fused, name, None)
    monkeypatch.setattr(fused, "_GATE_ROWS", 1)
    monkeypatch.setattr(fused, "_DOWN_ROWS", 1)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": chip})
    first = fused.kernels()
    assert tiles == (fused._GATE_ROWS, fused._DOWN_ROWS)
    assert fused.kernels() == first
    fused._initialize_kernels()
    assert fused.kernels() == first


@pytest.mark.parametrize("tokens", [1, 2, 9])
def test_forward_passes_full_banks_and_token_geometry(monkeypatch, tokens):
    block = block_type()()
    calls = []
    x = mx.zeros((1, tokens, 2048), mx.bfloat16)
    logits, inds, scores, glog, shared, act, out, sig = [object() for _ in range(8)]
    block.gate = lambda value: logits if value is x else pytest.fail("wrong input")
    block.shared_expert = lambda value: (
        shared if value is x else pytest.fail("wrong input")
    )

    def route(b, value, log):
        assert b is block and value is x and log is logits
        return inds, scores, glog

    def gate(**kwargs):
        calls.append(kwargs)
        return [act]

    def down(**kwargs):
        calls.append(kwargs)
        return [out]

    monkeypatch.setattr(fused, "kernels", lambda: (gate, down, sig))
    monkeypatch.setattr(fused, "fused_route", route)
    monkeypatch.setattr(fused, "_GATE_ROWS", 2)
    monkeypatch.setattr(fused, "_DOWN_ROWS", 4)
    # Unsupported array objects must still take the full-bank fallback.
    monkeypatch.setattr(fused, "_singleton_bank_views_enabled", lambda: True)
    assert fused.forward(block, x) is out
    gu, dp = block.switch_mlp.gate_up_proj, block.switch_mlp.down_proj
    assert calls[0]["inputs"] == [
        x,
        gu.weight,
        gu.scales,
        gu.biases,
        gu.weight,
        gu.scales,
        gu.biases,
        inds,
        sig,
    ]
    assert calls[1]["inputs"] == [
        act,
        dp.weight,
        dp.scales,
        dp.biases,
        inds,
        scores,
        shared,
        glog,
    ]
    assert calls[0]["grid"] == (32, 256, tokens * 8)
    assert calls[1]["grid"] == (32, 512, tokens)
    assert calls[0]["output_shapes"] == [(tokens, 8, 512)]
    assert calls[1]["output_shapes"] == [(1, tokens, 2048)]


def test_bank_views_device_gate_and_kill_switch(monkeypatch):
    from rapid_mlx.lattice import profiles

    # The measured table entry is pinned to the MLX version it was measured on;
    # pin it here so the device gate is tested independently of the install.
    monkeypatch.setattr(profiles, "mlx_version", lambda: "0.32.4")
    monkeypatch.delenv("RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS", raising=False)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})
    fused._bank_views_policy.cache_clear()
    assert fused._singleton_bank_views_enabled()
    monkeypatch.setenv("RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS", "0")
    assert not fused._singleton_bank_views_enabled()
    monkeypatch.delenv("RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS")
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "unknown"})
    fused._bank_views_policy.cache_clear()
    assert not fused._singleton_bank_views_enabled()
    fused._bank_views_policy.cache_clear()


def test_bank_binding_cache_tracks_weight_replacement():
    arrays = mx.quantize(mx.ones((4, 64, 64)), group_size=64, bits=4)
    projection = SimpleNamespace(weight=arrays[0], scales=arrays[1], biases=arrays[2])
    first = fused._bank_bindings(projection)
    assert all(a.shape[0] == 1 for a in first)
    assert fused._bank_bindings(projection) is first
    projection.weight = mx.zeros_like(projection.weight)
    replacement = fused._bank_bindings(projection)
    assert replacement is not first
    mx.eval(replacement)
    assert mx.array_equal(replacement[0], projection.weight[:1]).item()


@pytest.mark.parametrize("strided", [False, True])
def test_prefix_bindings_are_exact_and_retain_last_expert_after_unload(strided):
    """An evaluated prefix must keep storage beyond its logical last element."""
    k, n, h, experts, topk = 128, 64, 128, 4, 2
    key = mx.random.key(137)
    gu = mx.quantize(
        mx.random.normal((experts, 2 * n, k), key=key).astype(mx.bfloat16),
        group_size=64,
        bits=4,
    )
    dp = mx.quantize(
        mx.random.normal((experts, h, n), key=key).astype(mx.bfloat16),
        group_size=64,
        bits=4,
    )
    if strided:
        gu = tuple(mx.stack([a, a], axis=1)[:, 0] for a in gu)
        dp = tuple(mx.stack([a, a], axis=1)[:, 0] for a in dp)
    mx.eval(gu, dp)
    x = mx.cos(mx.arange(k)).astype(mx.bfloat16)[None, None]
    ids = mx.array([experts - 1, 0], mx.uint32)
    scores = mx.array([0.75, 0.25], mx.bfloat16)
    gate, down, sig = fused.kernels()

    def run(g, d):
        act = gate(
            inputs=[x, *g, *g, ids, sig],
            template=[
                ("T", x.dtype),
                ("ROWS", 1),
                ("K", k),
                ("N", n),
                ("GS", 64),
                ("TOPK", topk),
            ],
            grid=(32, n, topk),
            threadgroup=(32, 2, 1),
            output_shapes=[(topk, n)],
            output_dtypes=[x.dtype],
        )[0]
        out = down(
            inputs=[
                act,
                *d,
                ids,
                scores,
                mx.zeros((h,), x.dtype),
                mx.zeros((1,), x.dtype),
            ],
            template=[
                ("T", x.dtype),
                ("ROWS", 1),
                ("I", n),
                ("H", h),
                ("GS", 64),
                ("TOPK", topk),
                ("SHARED", 0),
            ],
            grid=(32, h, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(h,)],
            output_dtypes=[x.dtype],
        )[0]
        return act, out

    expected = run(gu, dp)
    mx.eval(expected)
    pg = SimpleNamespace(weight=gu[0], scales=gu[1], biases=gu[2])
    pd = SimpleNamespace(weight=dp[0], scales=dp[1], biases=dp[2])
    gv, dv = fused._bank_bindings(pg), fused._bank_bindings(pd)
    # Detach the view graph, so it cannot retain the bank merely as a lazy op.
    mx.eval(gv, dv)
    actual = run(gv, dv)
    del gu, dp, pg, pd, gv, dv
    gc.collect()
    mx.eval(actual)
    assert all(
        mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()
        for a, b in zip(actual, expected)
    )


def test_missing_bank_arrays_fail_closed(monkeypatch):
    cls = block_type()
    monkeypatch.setattr(fused, "_block_classes", lambda: (cls,))
    block = cls()
    block.switch_mlp.gate_up_proj.biases = None
    assert not fused._compatible(block)


def test_installer_without_metal_or_module_enumerator(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert fused.install_qwen35_affine4(object()) == 0
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    assert fused.install_qwen35_affine4(object()) == 0


def test_dequantization_failure_logs_and_leaves_stock(monkeypatch, caplog):
    block = block_type()()
    monkeypatch.setattr(fused, "_compatible", lambda b: True)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic dequantization failure")

    block.shared_expert_gate.scales = mx.ones((1, 32), mx.bfloat16)
    block.shared_expert_gate.biases = mx.zeros((1, 32), mx.bfloat16)
    monkeypatch.setattr(mx, "dequantize", fail)
    assert (
        fused.install_qwen35_affine4(
            SimpleNamespace(named_modules=lambda: [("block", block)])
        )
        == 0
    )
    assert "Affine4 probe declined block; using stock path" in caplog.text
    assert not hasattr(block, "_rapid_affine4_shared_weight")
    assert not getattr(block, "_rapid_affine4", False)


@pytest.mark.parametrize("shape", [(2048,), (1, 0, 2048), (1, 1, 64)])
def test_unsupported_shapes_use_original(shape):
    cls = block_type()
    block = cls()
    block._rapid_affine4 = True
    fused._patch_class(cls)
    result = block(mx.ones(shape, mx.bfloat16))
    assert result.shape == shape
    assert result.sum().item() == 0


@pytest.mark.parametrize("option", ["--positions", "--context"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_qualification_rejects_nonpositive_arguments(
    monkeypatch, capsys, option, value
):
    import sys

    from scripts import qualify_qwen35_affine4 as qualify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--model",
            "unused",
            "--corpus",
            "unused",
            "--output",
            "unused",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        qualify.main()
    assert exc.value.code == 2
    assert f"{option} must be positive" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_text_loader_enrolls_on_model_worker(monkeypatch):
    from concurrent.futures import Future

    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.utils import tokenizer

    model = SimpleNamespace(named_modules=lambda: [])
    calls = []

    class StopAfterEnrollmentError(Exception):
        pass

    class Worker:
        def submit(self, fn, *args, **kwargs):
            future = Future()
            if fn is tokenizer.load_model_with_fallback:
                future.set_result((model, object(), "local"))
            elif fn is fused.install_qwen35_affine4:
                calls.append(args)
                future.set_result(fn(*args, **kwargs))
            elif calls:
                raise StopAfterEnrollmentError
            else:
                future.set_result(None)
            return future

    monkeypatch.setattr(tokenizer, "load_model_with_fallback", lambda: None)
    engine = object.__new__(BatchedEngine)
    engine._model_load_executor = Worker()
    engine._model_name = "local/qwen3.6"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    with pytest.raises(StopAfterEnrollmentError):
        await engine._start_llm()
    assert calls == [(model,)]


def test_vlm_loader_enrolls_loaded_model(monkeypatch, tmp_path):
    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx import (
        gdn_in_proj_fusion,
        moe_fusion,
        qwen35_fused_gdn_decode,
        qwen35_moe_router,
        qwen35_residual_norm,
    )
    from rapid_mlx.models import mllm
    from rapid_mlx.utils import tokenizer

    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    calls = []
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(mlx_vlm, "load", lambda *a, **k: (model, processor))
    monkeypatch.setattr(
        mlx_vlm.utils, "load_config", lambda *a, **k: {"model_type": "qwen3_5_moe"}
    )
    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda *a, **k: tmp_path)
    for module, name in (
        (moe_fusion, "fuse_gate_up"),
        (gdn_in_proj_fusion, "fuse_gdn_in_proj"),
        (qwen35_fused_gdn_decode, "install_qwen35_fused_gdn_decode"),
        (qwen35_moe_router, "install_qwen35_moe_router"),
        (tokenizer, "augment_eos_token_ids_from_generation_config"),
        (tokenizer, "repair_byte_level_decoder"),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: None)
    monkeypatch.setattr(fused, "install_qwen35_affine4", calls.append)
    residual_calls = []
    monkeypatch.setattr(
        qwen35_residual_norm,
        "install_qwen35_residual_norm",
        lambda model, **kw: residual_calls.append((model, kw)),
    )
    mllm.MLXMultimodalLM("local/qwen-moe", spec_decode="none").load()
    assert calls == [model]
    assert residual_calls == [(model, {"spec_decode": "none"})]
