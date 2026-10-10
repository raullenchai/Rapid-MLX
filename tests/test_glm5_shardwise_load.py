# SPDX-License-Identifier: Apache-2.0
"""Ordering, isolation, and pinned-runtime contracts for bounded GLM loading."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType, SimpleNamespace

import pytest

from rapid_mlx.models.glm5_shardwise_load import (
    _bind,
    _BoundedEval,
    load_glm5_shardwise,
)


def test_binding_preserves_defaults_and_closure_without_mutating_globals():
    def make():
        captured = 3

        def original(value=2, *, factor=4):
            return abs(value) * factor + captured

        return original

    original = make()
    bound = _bind(original, "abs", lambda value: value + 1)
    assert bound() == 15
    assert bound(4, factor=2) == 13
    assert "abs" not in original.__globals__


@pytest.mark.parametrize("function", [object(), lambda: None])
def test_incompatible_runtime_fails_before_loading(function):
    with pytest.raises(RuntimeError, match="incompatible.*missing reader binding"):
        _bind(function, "reader", lambda: None)


@pytest.fixture
def fake_runtime(monkeypatch):
    import mlx.core as mx
    from mlx_vlm import utils

    from rapid_mlx.runtime import ubc_evict

    events = []

    class Tensor(str):
        nbytes = 4

    def reader(path):
        events.append(("read", path))
        return {path: Tensor(path)}

    def eval_weights(weights):
        events.append(("eval", next(iter(weights))))

    def evict(path):
        events.append(("evict", path))
        return 123

    # Real Python functions with independent module namespaces, like upstream.
    # This fake architecture boundary exposes the merged weights to assertions.
    namespace = {"_load_safetensors": reader, "mx": mx}
    exec(
        "def load_model(path, lazy=False, *, strict=True, **kwargs):\n"
        "    weights = {}\n"
        "    for shard in (path + '/1', path + '/2'):\n"
        "        weights.update(_load_safetensors(shard))\n"
        "    if not lazy:\n"
        "        mx.eval(weights)\n"
        "    return weights, lazy, strict, kwargs\n"
        "def load(path, lazy=False, *, strict=True, **kwargs):\n"
        "    return load_model(path, lazy, strict=strict, **kwargs)\n",
        namespace,
    )
    monkeypatch.setattr(utils, "_load_safetensors", reader)
    monkeypatch.setattr(utils, "load_model", namespace["load_model"])
    monkeypatch.setattr(utils, "load", namespace["load"])
    monkeypatch.setattr(mx, "eval", eval_weights)
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append(("clear", None)))
    monkeypatch.setattr(ubc_evict, "ubc_evict", evict)
    return SimpleNamespace(events=events, utils=utils, mx=mx, reader=reader)


@pytest.mark.requires_mlx
def test_each_shard_is_evaluated_and_evicted_before_next_read(fake_runtime):
    result = load_glm5_shardwise("model", trust_remote_code=False)
    assert result == (
        {"model/1": "model/1", "model/2": "model/2"},
        False,
        True,
        {"trust_remote_code": False},
    )
    assert fake_runtime.events == [
        (operation, f"model/{shard}")
        for shard in (1, 2)
        for operation in ("read", "eval", "evict")
    ] + [("eval", "model/1"), ("clear", None)]


@pytest.mark.requires_mlx
def test_failed_eval_neither_evicts_nor_reads_next_shard(fake_runtime, monkeypatch):
    def fail(_weights):
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(fake_runtime.mx, "eval", fail)
    with pytest.raises(RuntimeError, match="materialization failed"):
        load_glm5_shardwise("broken")
    assert fake_runtime.events == [("read", "broken/1")]


@pytest.mark.requires_mlx
def test_concurrent_calls_do_not_patch_upstream_or_unrelated_load(fake_runtime):
    originals = (
        fake_runtime.utils.load,
        fake_runtime.utils.load_model,
        fake_runtime.utils._load_safetensors,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(load_glm5_shardwise, ["a", "b"]))
    assert list(results[0][0]) == ["a/1", "a/2"]
    assert list(results[1][0]) == ["b/1", "b/2"]
    assert originals == (
        fake_runtime.utils.load,
        fake_runtime.utils.load_model,
        fake_runtime.utils._load_safetensors,
    )
    fake_runtime.events.clear()
    fake_runtime.utils.load("ordinary", lazy=True, strict=False)
    assert fake_runtime.events == [("read", "ordinary/1"), ("read", "ordinary/2")]


@pytest.mark.requires_mlx
def test_pinned_loader_exposes_expected_bindings():
    from mlx_vlm import utils

    assert isinstance(utils.load, FunctionType)
    assert "load_model" in utils.load.__code__.co_names
    assert isinstance(utils.load_model, FunctionType)
    assert "_load_safetensors" in utils.load_model.__code__.co_names
    assert "mx" in utils.load_model.__code__.co_names


@pytest.mark.requires_mlx
def test_final_eval_bounds_batches_and_clears_cache_between_them(monkeypatch):
    from rapid_mlx.models import glm5_shardwise_load

    monkeypatch.setattr(glm5_shardwise_load, "_EVAL_BATCH_BYTES", 10)
    tensors = [SimpleNamespace(nbytes=n) for n in [6, 6, 15, 2]]
    events = []
    mx = SimpleNamespace(
        eval=lambda batch: events.append(list(batch)),
        clear_cache=lambda: events.append("clear"),
    )
    _BoundedEval(mx).eval({str(i): tensor for i, tensor in enumerate(tensors)})
    assert events == [event for tensor in tensors for event in ([tensor], "clear")]


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "platform,family,trust,expected",
    [
        ("darwin", "glm5_next", True, "shardwise"),
        ("darwin", "glm5_next", False, "shardwise"),
        ("darwin", "qwen3_5_moe", True, "ordinary"),
        ("linux", "glm5_next", True, "ordinary"),
    ],
)
def test_mllm_routes_only_darwin_glm_and_preserves_remote_code_policy(
    monkeypatch, platform, family, trust, expected
):
    import mlx_vlm
    from mlx_vlm import utils

    from rapid_mlx.models import glm5_shardwise_load, mllm
    from rapid_mlx.patches import (
        glm5_next_forget_gate_quant,
        glm5_next_processor,
        glm5_next_runtime,
    )

    calls = []

    def loader(kind):
        def stop(*args, **kwargs):
            calls.append((kind, args, kwargs))
            raise RuntimeError("reached loader boundary")

        return stop

    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(
        glm5_next_runtime, "install_glm5_next_runtime_fix", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_processor, "install_glm5_next_processor_patch", lambda: None
    )
    monkeypatch.setattr(
        glm5_next_forget_gate_quant,
        "install_glm5_next_forget_gate_quant_fix",
        lambda: None,
    )
    monkeypatch.setattr(utils, "load_config", lambda *_a, **_k: {"model_type": family})
    monkeypatch.setattr(mlx_vlm, "load", loader("ordinary"))
    monkeypatch.setattr(glm5_shardwise_load, "load_glm5_shardwise", loader("shardwise"))
    monkeypatch.setattr(mllm.sys, "platform", platform)
    with pytest.raises(RuntimeError, match="reached loader boundary"):
        mllm.MLXMultimodalLM("local/checkpoint", trust_remote_code=trust).load()
    assert calls == [
        (expected, ("local/checkpoint",), {} if trust else {"trust_remote_code": False})
    ]


@pytest.mark.requires_mlx
def test_real_quantized_glm_checkpoint_matches_upstream_and_keeps_strict_validation(
    tmp_path, monkeypatch
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm import utils
    from mlx_vlm.models.glm5_next.config import ModelConfig
    from mlx_vlm.models.glm5_next.glm5_next import Model

    config = {
        "model_type": "glm5_next",
        "text_config": {
            "num_hidden_layers": 0,
            "num_nextn_predict_layers": 0,
            "hidden_size": 64,
            "vocab_size": 64,
        },
        "vision_config": {
            "depth": 0,
            "hidden_size": 64,
            "num_heads": 1,
            "out_hidden_size": 64,
            "intermediate_size": 64,
            "projection_intermediate_size": 64,
        },
        "quantization": {"group_size": 64, "bits": 4},
    }
    original = Model(ModelConfig.from_dict(config))
    nn.quantize(original, group_size=64, bits=4)
    weights = dict(tree_flatten(original.parameters()))
    names = sorted(weights)
    midpoint = len(names) // 2
    for number, keys in enumerate((names[:midpoint], names[midpoint:]), 1):
        mx.save_safetensors(
            str(tmp_path / f"model-{number:05}.safetensors"),
            {key: weights[key] for key in keys},
        )
    (tmp_path / "config.json").write_text(json.dumps(config))
    processor = object()
    monkeypatch.setattr(utils, "load_image_processor", lambda *_a, **_k: None)
    monkeypatch.setattr(utils, "load_processor", lambda *_a, **_k: processor)
    expected, _ = utils.load(str(tmp_path))
    actual, loaded_processor = load_glm5_shardwise(str(tmp_path))
    assert loaded_processor is processor
    expected_parameters = dict(tree_flatten(expected.parameters()))
    actual_parameters = dict(tree_flatten(actual.parameters()))
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, value in actual_parameters.items():
        assert value.dtype == expected_parameters[name].dtype
        assert mx.array_equal(value, expected_parameters[name]).item(), name

    # A corrupt shard must retain upstream's strict missing-key failure.
    missing = names[0]
    mx.save_safetensors(
        str(tmp_path / "model-00001.safetensors"),
        {key: weights[key] for key in names[:midpoint] if key != missing},
    )
    with pytest.raises(ValueError, match="Missing"):
        load_glm5_shardwise(str(tmp_path))
