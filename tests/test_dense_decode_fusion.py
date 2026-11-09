"""Metal numerical and lifetime contracts for functional decode fusions."""

import gc
import weakref

import pytest

pytestmark = pytest.mark.requires_mlx

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx_lm.models import lfm2, qwen3_5

from rapid_mlx import dense_decode_fusion as fusion


def tiny(family="lfm", dtype=None):
    if family == "lfm":
        args = lfm2.ModelArgs(
            model_type="lfm2",
            vocab_size=64,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            block_dim=128,
            block_ff_dim=256,
            block_multiple_of=64,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=False,
            full_attn_idxs=[1],
        )
        model = lfm2.Model(args)
    else:
        args = qwen3_5.TextModelArgs(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            full_attention_interval=2,
        )
        model = qwen3_5.TextModel(args)
    dtype = dtype or mx.bfloat16
    model.set_dtype(dtype)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("n", [128, 1024, 2048, 2816, 4096])
def test_seam_bits(dtype, n):
    norm = nn.RMSNorm(n)
    norm.weight = (0.2 + mx.cos(mx.arange(n, dtype=mx.float32))).astype(dtype)
    assert fusion._norm_probe(norm, dtype)
    a = mx.ones((1, 1, n), dtype)
    b = -a
    residual, normalized = fusion.seam(a, b, norm)
    assert fusion._bits_equal(residual, a + b)
    assert fusion._bits_equal(normalized, norm(a + b))


@pytest.mark.parametrize("family", ["lfm", "qwen"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_decode_prefill_batch_and_interleaving(family, dtype):
    model = tiny(family, dtype)
    backbone = model.model
    original = type(backbone).__call__
    x = mx.array([[1]])
    expected = original(backbone, x)
    assert fusion.install_dense_decode_fusion(model) == 1
    assert fusion._bits_equal(backbone(x), expected)
    assert fusion.install_dense_decode_fusion(model) == 0
    # Independent graphs may be evaluated in reverse request order.
    first = backbone(x)
    second = backbone(mx.array([[2]]))
    assert fusion._bits_equal(second, original(backbone, mx.array([[2]])))
    assert fusion._bits_equal(first, expected)
    for inputs in (mx.array([[1, 2, 3]]), mx.array([[1], [2]])):
        assert fusion._bits_equal(backbone(inputs), original(backbone, inputs))
    # Prefix reuse remains a property of each request's stock cache.
    left, right = model.make_cache(), model.make_cache()
    for inputs in (mx.array([[1, 2, 3]]), x, mx.array([[2]])):
        actual = backbone(inputs, cache=left)
        expected = original(backbone, inputs, cache=right)
        assert fusion._bits_equal(actual, expected)
        for a, b in zip(left, right):
            for sa, sb in zip(a.state, b.state):
                if sa is not None:
                    assert fusion._bits_equal(sa, sb)


def test_kill_switch_spec_and_unload(monkeypatch):
    model = tiny()
    monkeypatch.setenv(fusion._SWITCH, "0")
    assert fusion.install_dense_decode_fusion(model) == 0
    monkeypatch.delenv(fusion._SWITCH)
    assert fusion.install_dense_decode_fusion(model, spec_decode="mtp") == 0
    assert fusion.install_dense_decode_fusion(model, spec_decode="auto") == 0
    assert fusion.install_dense_decode_fusion(model) == 1
    ref = weakref.ref(model.model)
    output = model(mx.array([[1]]))
    del model
    gc.collect()
    assert ref() is None
    mx.eval(output)  # Lazy graph still owns the required buffers after unload.


def test_probe_failure_and_unknown(monkeypatch):
    model = tiny()
    monkeypatch.setattr(fusion, "_norm_probe", lambda *a: False)
    assert fusion.install_dense_decode_fusion(model) == 0
    assert fusion.install_dense_decode_fusion(nn.Linear(128, 128)) == 0


def vlm_tiny(family):
    from dataclasses import asdict

    if family == "qwen-vlm":
        from mlx_vlm.models.qwen3_5.config import TextConfig
        from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

        source = tiny("qwen").args
        data = asdict(source)
        data = {k: v for k, v in data.items() if k in TextConfig.__dataclass_fields__}
        backbone = Qwen3_5Model(TextConfig(**data))
    elif family == "lfm3":
        from rapid_mlx.models.lfm3 import Lfm3Model, ModelArgs

        backbone = Lfm3Model(
            ModelArgs(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
                layer_types=["conv", "full_attention", "sliding_attention"],
                sliding_window=8,
            )
        )
    else:
        from mlx_vlm.models.lfm2.language import Lfm2Model

        backbone = Lfm2Model(tiny().args)
    root = nn.Module()
    root.model = backbone
    root.set_dtype(mx.bfloat16)
    nn.quantize(root, group_size=32, bits=4)
    mx.eval(root.parameters())
    return root


@pytest.mark.parametrize("family", ["qwen-vlm", "lfm-vlm", "lfm3"])
def test_vlm_and_sandwich_contracts(family):
    import importlib

    root = vlm_tiny(family)
    backbone = root.model
    original = type(backbone).__call__
    kind = fusion._BACKBONES[(type(backbone).__module__, type(backbone).__name__)]
    api = importlib.import_module(type(backbone).__module__)
    left = fusion._probe_cache(backbone, kind, api)
    right = fusion._probe_cache(backbone, kind, api)
    assert fusion.install_dense_decode_fusion(root) == 1
    for inputs in (mx.array([[1, 2, 3]]), mx.array([[1]]), mx.array([[2]])):
        actual = backbone(inputs, cache=left)
        expected = original(backbone, inputs, cache=right)
        assert fusion._bits_equal(actual, expected)
    # Multi-row and hidden capture keep ordinary array outputs.
    if family != "lfm3":
        sink = []
        backbone(
            mx.array([[1]]),
            cache=fusion._probe_cache(backbone, kind, api),
            capture_layer_ids=[0],
            hidden_sink=sink,
        )
        assert len(sink) == 1 and isinstance(sink[0], mx.array)
    if family == "qwen-vlm":
        for fused in (True, False):
            attention = backbone.layers[1].self_attn
            attention.rotary_emb.fused_apply = fused
            pos = mx.zeros((3, 1, 1), mx.int32)
            actual = backbone(
                mx.array([[1]]),
                cache=fusion._probe_cache(backbone, kind, api),
                position_ids=pos,
            )
            expected = original(
                backbone,
                mx.array([[1]]),
                cache=fusion._probe_cache(backbone, kind, api),
                position_ids=pos,
            )
            assert fusion._bits_equal(actual, expected)


def gemma_tiny(moe=False, gated=False, external=False):
    from rapid_mlx.models.gemma4_vendored.config import TextConfig
    from rapid_mlx.models.gemma4_vendored.language import DecoderLayer

    if external:
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import DecoderLayer
    config = TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_kv_shared_layers=0,
        vocab_size=64,
        hidden_size_per_layer_input=32 if gated else 0,
        layer_types=["full_attention"],
        enable_moe_block=moe,
        num_experts=4 if moe else None,
        top_k_experts=2 if moe else None,
        moe_intermediate_size=128 if moe else None,
    )
    layer = DecoderLayer(config, 0)
    layer.set_dtype(mx.bfloat16)
    nn.quantize(layer, group_size=32, bits=4)
    mx.eval(layer.parameters())
    return layer


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_gemma_branch_norms_and_fallback(moe, external):
    layer = gemma_tiny(moe, external=external)
    original = type(layer).__call__
    expected = original(layer, mx.ones((1, 1, 128), mx.bfloat16))[0]
    assert fusion.install_dense_decode_fusion(layer) == 1
    assert fusion._bits_equal(layer(mx.ones((1, 1, 128), mx.bfloat16))[0], expected)
    assert fusion.install_dense_decode_fusion(layer) == 0
    for shape in ((2, 1, 128), (1, 3, 128)):
        x = mx.ones(shape, mx.bfloat16)
        assert fusion._bits_equal(layer(x)[0], original(layer, x)[0])
    layer.layer_scalar = None
    x = mx.ones((1, 1, 128), mx.bfloat16)
    assert fusion._bits_equal(layer(x)[0], original(layer, x)[0])


def test_declined_probes_and_layouts(monkeypatch):
    from types import SimpleNamespace

    assert fusion.install_dense_decode_fusion(SimpleNamespace()) == 0
    assert not fusion._norm_probe(nn.LayerNorm(128), mx.bfloat16)
    assert not fusion._norm_probe(nn.RMSNorm(128), mx.bfloat16)
    norm = nn.RMSNorm(128)
    norm.weight = norm.weight.astype(mx.bfloat16)
    with monkeypatch.context() as m:
        m.setattr(fusion, "_bits_equal", lambda *a: False)
        assert not fusion._norm_probe(norm, mx.bfloat16)
    assert not fusion._group_mlp(nn.Linear(128, 128), mx.bfloat16)
    root = tiny()
    root.model.layers = []
    assert fusion.install_dense_decode_fusion(root) == 0
    root = tiny()
    root.model.embedding_norm.weight = mx.ones((128,), mx.float32)
    assert fusion.install_dense_decode_fusion(root) == 0
    with monkeypatch.context() as m:
        root = tiny()
        m.setattr(fusion, "_bits_equal", lambda *a: False)
        m.setattr(fusion, "_norm_probe", lambda *a: True)
        assert fusion.install_dense_decode_fusion(root) == 0
    with monkeypatch.context() as m:
        root = tiny()
        m.setattr(
            fusion,
            "_norm_probe",
            lambda *a: (_ for _ in ()).throw(RuntimeError("device probe")),
        )
        assert fusion.install_dense_decode_fusion(root) == 0
    assert not fusion._enroll_gemma(gemma_tiny(gated=True))
    layer = gemma_tiny()
    with monkeypatch.context() as m:
        m.setattr(fusion, "_norm_probe", lambda *a: False)
        assert not fusion._enroll_gemma(layer)
    with monkeypatch.context() as m:
        m.setattr(fusion, "_norm_probe", lambda *a: True)
        m.setattr(fusion, "_bits_equal", lambda *a: False)
        assert not fusion._enroll_gemma(layer)
    with monkeypatch.context() as m:
        m.setattr(
            fusion,
            "_enroll_gemma",
            lambda *a: (_ for _ in ()).throw(RuntimeError("device probe")),
        )
        assert fusion.install_dense_decode_fusion(layer) == 0


@pytest.mark.parametrize("group", ["mlp", "attention"])
@pytest.mark.parametrize("failure", ["not-quantized", "metadata", "parity"])
def test_projection_decline(group, failure, monkeypatch):
    root = vlm_tiny("lfm-vlm")
    layer = root.model.layers[1]
    module = layer.feed_forward if group == "mlp" else layer.self_attn
    part = module.w1 if group == "mlp" else module.q_proj
    if failure == "not-quantized":
        if group == "mlp":
            module.w1 = nn.Linear(128, 256)
        else:
            module.q_proj = nn.Linear(128, 128)
    elif failure == "metadata":
        part.group_size = 64
    else:
        monkeypatch.setattr(fusion, "_bits_equal", lambda *a: False)
    fn = fusion._group_mlp if group == "mlp" else fusion._group_attention
    assert not fn(module, mx.bfloat16)


def test_embedding_dtype_and_cancelled_graph():
    root = tiny()
    b = root.model
    original = type(b).__call__
    assert fusion.install_dense_decode_fusion(root) == 1
    x = mx.array([[1]])
    h = mx.ones((1, 1, 128), mx.float32)
    assert fusion._bits_equal(
        b(x, input_embeddings=h), original(b, x, input_embeddings=h)
    )
    discarded = b(x)
    del discarded
    assert fusion._bits_equal(b(x), original(b, x))


def test_parallel_calls_own_their_graphs():
    from concurrent.futures import ThreadPoolExecutor

    root = tiny()
    b = root.model
    original = type(b).__call__
    expected = [original(b, mx.array([[token]])) for token in (1, 2)]
    mx.eval(expected)
    assert fusion.install_dense_decode_fusion(root) == 1

    def forward(token):
        with mx.stream(mx.new_stream(mx.gpu)):
            result = b(mx.array([[token]]))
            mx.eval(result)
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(forward, (1, 2)))
    assert all(fusion._bits_equal(a, b) for a, b in zip(outputs, expected))


def test_numerical_license_is_packaged():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        "kernels/LICENSE-RMS"
        in config["tool"]["setuptools"]["package-data"]["rapid_mlx"]
    )


def test_architecture_drift_and_unqualified_moe():
    root = tiny()
    del root.model.embedding_norm
    assert fusion.install_dense_decode_fusion(root) == 0
    root = tiny()
    root.model.args.num_experts = 4
    assert fusion.install_dense_decode_fusion(root) == 0


def test_failed_composition_restores_projection_classes_and_parameters(monkeypatch):
    root = tiny()
    mlp = root.model.layers[0].feed_forward
    original_class = type(mlp)
    weight = mlp.w1.weight
    original_equal = fusion._bits_equal
    # The full decoder returns hidden_size; projection probes return FFN width.
    monkeypatch.setattr(fusion, "_norm_probe", lambda *a: True)
    monkeypatch.setattr(
        fusion, "_bits_equal", lambda a, b: a.shape[-1] != 128 and original_equal(a, b)
    )
    assert fusion.install_dense_decode_fusion(root) == 0
    assert type(mlp) is original_class and mlp.w1.weight is weight
    assert not hasattr(mlp, "_rapid_dense_group")
    with monkeypatch.context() as m:
        m.setattr(
            fusion,
            "_group_attention",
            lambda *a: (_ for _ in ()).throw(RuntimeError("drift")),
        )
        assert fusion.install_dense_decode_fusion(root) == 0
    assert type(mlp) is original_class and mlp.w1.weight is weight


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("failure", ["group-exception", "exception", "declined"])
def test_failed_gemma_enrollment_restores_full_layer(monkeypatch, external, failure):
    layer = gemma_tiny(external=external)
    before = [
        (name, module, type(module), dict(module), module.__dict__.copy())
        for name, module in layer.named_modules()
    ]
    x = mx.ones((1, 1, 128), mx.bfloat16)
    expected = layer(x)[0]
    mx.eval(expected)
    original_group = fusion._group_mlp
    original_enroll = fusion._enroll_gemma
    grouped = []

    def group_then_fail(module, dtype):
        assert original_group(module, dtype)
        assert hasattr(module, "_rapid_dense_group")
        grouped.append(module)
        if failure == "group-exception":
            raise RuntimeError("failure after projection grouping")
        return True

    def enroll_then_fail(module):
        assert original_enroll(module)
        if failure == "exception":
            raise RuntimeError("failure after layer replacement")
        return False

    monkeypatch.setattr(fusion, "_group_mlp", group_then_fail)
    if failure != "group-exception":
        monkeypatch.setattr(fusion, "_enroll_gemma", enroll_then_fail)
    assert fusion.install_dense_decode_fusion(layer) == 0
    assert grouped == [layer.mlp]
    after = list(layer.named_modules())
    assert [name for name, _ in after] == [entry[0] for entry in before]
    for (_, actual), (_, module, cls, values, attrs) in zip(after, before):
        assert actual is module
        assert type(actual) is cls
        assert actual.keys() == values.keys()
        assert all(actual[key] is value for key, value in values.items())
        assert actual.__dict__.keys() == attrs.keys()
        assert all(actual.__dict__[key] is value for key, value in attrs.items())
    assert fusion._bits_equal(layer(x)[0], expected)


def test_unqualified_cache_layout_keeps_stock_call(monkeypatch):
    model = tiny()
    b = model.model
    original = type(b).__call__
    assert fusion.install_dense_decode_fusion(model) == 1

    class OtherKVCache(lfm2.KVCache):
        pass

    left = [lfm2.ArraysCache(size=1), OtherKVCache()]
    right = [lfm2.ArraysCache(size=1), OtherKVCache()]
    monkeypatch.setattr(
        fusion,
        "_decode",
        lambda *a: pytest.fail("unsupported cache entered fused path"),
    )
    x = mx.array([[1]])
    assert fusion._bits_equal(b(x, cache=left), original(b, x, cache=right))


def test_residual_fusion_survives_declined_attention_group(monkeypatch):
    import importlib

    root = vlm_tiny("qwen-vlm")
    b = root.model
    cls = type(b)
    api = importlib.import_module(cls.__module__)
    x = mx.array([[1]])
    expected = b(x, cache=fusion._probe_cache(b, "qwen-vlm", api))
    monkeypatch.setattr(fusion, "_group_attention", lambda *a: False)
    assert fusion.install_dense_decode_fusion(root) == 1
    actual = b(x, cache=fusion._probe_cache(b, "qwen-vlm", api))
    assert fusion._bits_equal(actual, expected)


def test_embeddings_without_input_ids_keep_optional_backbone_contract():
    root = vlm_tiny("lfm3")
    b = root.model
    original = type(b).__call__
    assert fusion.install_dense_decode_fusion(root) == 1
    h = mx.ones((1, 1, 128), mx.bfloat16)
    assert fusion._bits_equal(b(input_embeddings=h), original(b, input_embeddings=h))
