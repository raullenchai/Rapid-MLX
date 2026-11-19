"""Native synthetic build, quantized-prefix restart and held-out quality checks."""

import json
from dataclasses import asdict

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
pytestmark = pytest.mark.requires_mlx

from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from rapid_mlx.models.hy_v4_flash import Model, ModelArgs
from rapid_mlx.models.hy_v4_flash_config import dequantize_fp8
from scripts import hy4_gptq as b
from scripts.hy4_gptq_adapter import Hy4Adapter, Recipe
from scripts.streamed_gptq_io import TensorReader
from scripts.streamed_gptq_quant import QuantConfig, gptq
from tests.test_hy4_gptq import write_source
from tests.test_hy_v4_flash import tiny


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def fp8_encode(weight):
    table = dequantize_fp8(np.arange(127, dtype=np.uint8)[None], np.ones(1))[0]
    raw = np.array(weight.astype(mx.float32))
    codes = np.abs(np.abs(raw)[..., None] - table).argmin(-1).astype(np.uint8)
    return codes | np.where(raw < 0, 128, 0).astype(np.uint8)


@pytest.fixture
def source(tmp_path):
    mx.random.seed(21)
    args = tiny(
        hidden_size=64,
        vocab_size=128,
        intermediate_size=64,
        num_hidden_layers=3,
        layer_types=[
            "deepseek_sparse_attention",
            "deepseek_sparse_attention",
            "sliding_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse"],
        q_lora_rank=64,
        kv_lora_rank=64,
        # Both attention types need group-aligned o_proj inputs (2 heads x 32).
        v_head_dim=32,
        swa_v_head_dim=32,
        routed_expert_hidden_size=64,
        moe_intermediate_size=64,
        compressed_indexer_head_dim=64,
        engram_n_embed_per_ngram=64,
        engram_pad_id=127,
        hc_mult=4,
    )
    weights = dict(tree_flatten(Model(args).parameters()))
    for layer in (1, 2):
        prefix = f"model.layers.{layer}.mlp."
        gate = weights.pop(prefix + "switch_mlp.gate_proj.weight")
        up = weights.pop(prefix + "switch_mlp.up_proj.weight")
        weights[prefix + "experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
        weights[prefix + "experts.down_proj"] = weights.pop(
            prefix + "switch_mlp.down_proj.weight"
        )
    tensors = {}
    for name, value in weights.items():
        mx.eval(value)
        if len(value.shape) >= 2 and (
            "experts." in name
            or ".q_b_proj." in name
            or name.endswith("embedding.weight")
        ):
            tensors[name] = "F8_E4M3", fp8_encode(value)
            if name.endswith("embedding.weight"):
                scale_shape = value.shape[:-1]
            else:
                scale_shape = (
                    *value.shape[:-2],
                    (value.shape[-2] + 15) // 16,
                    (value.shape[-1] + 31) // 32,
                )
            tensors[name + "_scale_inv"] = "F32", np.ones(scale_shape, np.float32)
        elif mx.issubdtype(value.dtype, mx.floating):
            tensors[name] = "F32", np.array(value.astype(mx.float32))
        else:
            tensors[name] = "I32", np.array(value.astype(mx.int32))
    config = asdict(args)
    config["quantization_config"] = dict(weight_block_size=[16, 32])
    source = write_source(tmp_path / "source", tensors, config)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)})),
    )
    tokenizer.save_pretrained(source)
    return source


@pytest.mark.parametrize("layers", [[0, 1], [0, 1, 2]], ids=["truncated", "full"])
def test_truncated_build_loads_forward_resume_and_quality(
    source, tmp_path, monkeypatch, layers
):
    adapter = Hy4Adapter(source)
    windows = [np.array([1, 3, 7, 5], np.int32), np.array([2, 4, 8], np.int32)]
    evidence = {"english": ["training-hash"]}
    destination = tmp_path / "native"
    report = b.build(adapter, destination, windows, evidence, layers)
    assert report["safetensors_bytes"] > 0
    assert json.loads((destination / "config.json").read_text())[
        "num_hidden_layers"
    ] == len(layers)
    model, _ = load_model(
        destination, get_model_classes=lambda config: (Model, ModelArgs)
    )
    expected = model(mx.array(windows[0][None]))
    mx.eval(expected)
    assert expected.shape == (1, 4, 128)
    assert bool(mx.all(mx.isfinite(expected)).item())
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    rapid_model, _ = load_model_with_fallback(str(destination))
    rapid_output = rapid_model(mx.array(windows[0][None]))
    mx.eval(rapid_output)
    np.testing.assert_array_equal(np.array(rapid_output), np.array(expected))
    streamed = next(
        b.logits(
            adapter, windows[:1], tmp_path / "stream", layers, TensorReader(destination)
        )
    )
    np.testing.assert_allclose(streamed, np.array(expected[0]), atol=0.1, rtol=0.03)

    def no_refit(*args):
        raise AssertionError("completed layers must not be refitted")

    monkeypatch.setattr(b, "quantize_experts", no_refit)
    assert b.build(adapter, destination, windows, evidence, layers) == report
    with pytest.raises(ValueError, match="identity"):
        b.build(adapter, destination, windows, {"english": ["changed"]}, layers)
    with pytest.raises(ValueError, match="overlaps"):
        b.quality(adapter, destination, windows, evidence, tmp_path / "bad-quality")
    heldout = [np.array([9, 10, 11, 12], np.int32), np.array([13, 14, 15], np.int32)]
    ruler = b.quality(
        adapter,
        destination,
        heldout,
        {"english": ["held-out-hash"]},
        tmp_path / "quality",
    )
    assert ruler["tokens"] == 7
    assert ruler["truncated"] == (len(layers) < adapter.args.num_hidden_layers)
    assert 0 <= ruler["top1_agreement"] <= 1 and ruler["kl_nats"] >= 0
    marker = json.loads((destination / "build-state/layer-1.json").read_text())
    tensor = destination / next(iter(marker["weight_map"].values()))
    with tensor.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="artifact"):
        b.build(adapter, destination, windows, evidence, layers)


def test_interrupted_build_replays_only_uncommitted_layer(
    source, tmp_path, monkeypatch
):
    adapter = Hy4Adapter(source)
    windows = [np.array([1, 2, 3], np.int32)]
    destination = tmp_path / "interrupted"
    original = b.quantize_experts
    with monkeypatch.context() as patch:
        patch.setattr(
            b,
            "quantize_experts",
            lambda *args: (_ for _ in ()).throw(RuntimeError("injected interruption")),
        )
        with pytest.raises(RuntimeError, match="interruption"):
            b.build(adapter, destination, windows, {"code": ["x"]}, [0, 1, 2])
    assert (destination / "build-state/layer-0.json").exists()
    assert not (destination / "build-state/layer-1.json").exists()
    called = []

    def recording(store, layer, *args):
        called.append(layer)
        return original(store, layer, *args)

    monkeypatch.setattr(b, "quantize_experts", recording)
    b.build(adapter, destination, windows, {"code": ["x"]}, [0, 1, 2])
    assert called == [1, 2]


def test_shared_gptq_improves_router_weighted_reconstruction():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(512, 64)).astype(np.float32)
    x[:, 1:] += 0.7 * x[:, :1]
    router_weights = rng.uniform(0.01, 1, (512,)).astype(np.float32)
    h = mx.array(x.T @ (x * router_weights[:, None]))
    weight = mx.array(rng.normal(size=(32, 64)).astype(np.float32)).astype(mx.bfloat16)
    packed, _ = gptq(weight, h, 4, QuantConfig(damping=0.01), count=512)
    fitted = mx.dequantize(*packed, group_size=64, bits=4).astype(mx.float32)
    rtn = mx.dequantize(
        *mx.quantize(weight, group_size=64, bits=4), group_size=64, bits=4
    ).astype(mx.float32)

    def loss(value):
        delta = weight.astype(mx.float32) - value
        return float(mx.sum((delta @ h) * delta).item())

    assert loss(fitted) < loss(rtn)
    from scripts import deepseek_v41_gptq_quant as legacy
    from scripts import streamed_gptq_quant as shared

    assert legacy.gptq is shared.gptq
    assert legacy.gptq_batched is shared.gptq_batched
    assert Recipe("hy4-gptq-lowbit").spec("model.layers.1.mlp.experts.down_proj") == (
        3,
        64,
        "gptq",
    )
