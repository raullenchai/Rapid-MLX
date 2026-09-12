from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

import vllm_mlx.models.deepseek_v41_native.load as native_load

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from vllm_mlx.models.deepseek_v41_native.attention import (  # noqa: E402
    Attention,
    GroupedOutputLinear,
)
from vllm_mlx.models.deepseek_v41_native.cache import ModelCache  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.compressor import (  # noqa: E402
    Compressor,
    CompressorState,
)
from vllm_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.load import (  # noqa: E402
    reshape_grouped_wo_a,
    resolve_indexed_shard,
    supports_engram_ssd_offload,
)
from vllm_mlx.models.deepseek_v41_native.model import Model  # noqa: E402


def test_reshape_grouped_wo_a_restores_quantized_parameter_axes() -> None:
    args = ModelArgs(o_groups=2, o_lora_rank=3)
    weight = mx.arange(24, dtype=mx.uint32).reshape(6, 4)
    scales = mx.arange(12, dtype=mx.float32).reshape(6, 2)

    values = dict(
        reshape_grouped_wo_a(
            [
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wo_a.scales", scales),
                ("layers.0.attn.wq_a.weight", weight),
            ],
            args,
        )
    )

    assert values["layers.0.attn.wo_a.weight"].shape == (2, 3, 4)
    assert values["layers.0.attn.wo_a.scales"].shape == (2, 3, 2)
    assert values["layers.0.attn.wq_a.weight"].shape == (6, 4)


def test_reshape_grouped_wo_a_rejects_incompatible_rows() -> None:
    args = ModelArgs(o_groups=2, o_lora_rank=3)
    with pytest.raises(ValueError, match="expected 6"):
        reshape_grouped_wo_a([("layers.0.attn.wo_a.weight", mx.zeros((5, 4)))], args)


def test_indexed_shard_rejects_local_symlink_escape(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    (model / "shard.safetensors").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes"):
        resolve_indexed_shard(str(model), "shard.safetensors")


def test_indexed_shard_rejects_lexical_parent_escape(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "outside.safetensors").touch()

    with pytest.raises(ValueError, match="inside the model directory"):
        resolve_indexed_shard(str(model), "../outside.safetensors")


def test_indexed_shard_allows_same_repository_hub_blob(tmp_path) -> None:
    repository = tmp_path / "models--owner--model"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "digest"
    blob.touch()
    (snapshot / "shard.safetensors").symlink_to(blob)

    assert resolve_indexed_shard(str(snapshot), "shard.safetensors") == str(blob)


def test_indexed_shard_does_not_trust_arbitrary_snapshots_name(tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    (snapshot / "shard.safetensors").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes"):
        resolve_indexed_shard(str(snapshot), "shard.safetensors")


def test_indexed_shard_rejects_symlinked_hub_blobs_directory(tmp_path) -> None:
    repository = tmp_path / "models--owner--model"
    snapshot = repository / "snapshots" / "revision"
    external_blobs = tmp_path / "external-blobs"
    snapshot.mkdir(parents=True)
    external_blobs.mkdir()
    blob = external_blobs / "digest"
    blob.touch()
    (repository / "blobs").symlink_to(external_blobs)
    (snapshot / "shard.safetensors").symlink_to(blob)

    with pytest.raises(ValueError, match="symlink escapes"):
        resolve_indexed_shard(str(snapshot), "shard.safetensors")


def test_engram_ssd_capability_requires_affine_indexed_layout(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        '{"text_config":{"engram_layer_ids":[1]},"quantization":{"engram_bits":2}}'
    )
    (tmp_path / "shard.safetensors").touch()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map":{'
        '"layers.1.engram.embed.weight":"shard.safetensors",'
        '"layers.1.engram.embed.scales":"shard.safetensors",'
        '"layers.1.engram.embed.biases":"shard.safetensors"}}'
    )

    assert supports_engram_ssd_offload(str(tmp_path))

    index.write_text('{"weight_map":{}}')
    assert not supports_engram_ssd_offload(str(tmp_path))


@pytest.mark.parametrize(
    ("config", "index"),
    [
        ({"text_config": {"engram_layer_ids": []}}, {"weight_map": {}}),
        (
            {
                "text_config": {"engram_layer_ids": [1]},
                "quantization": {"engram_bits": 2},
            },
            {"weight_map": []},
        ),
    ],
)
def test_engram_ssd_capability_rejects_unsupported_metadata(
    tmp_path, config, index
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert not supports_engram_ssd_offload(str(tmp_path))


class _FakeLoadModel:
    def __init__(self, _args, token_map=None):
        del token_map
        embed = SimpleNamespace(weight=mx.zeros((8, 64)))
        self.layers = [SimpleNamespace(layer_id=1, engram=SimpleNamespace(embed=embed))]
        self.loaded = []
        self.evaluated = False

    def load_weights(self, items, strict=False):
        assert not strict
        self.loaded.extend(items)

    def parameters(self):
        return {"other": {"weight": mx.zeros((1, 1))}}

    def eval(self):
        self.evaluated = True


def _write_offload_fixture(tmp_path, *, mapping_override=None):
    keys = {
        "weight": "layers.1.engram.embed.weight",
        "scales": "layers.1.engram.embed.scales",
        "biases": "layers.1.engram.embed.biases",
    }
    weight, scales, biases = mx.quantize(
        mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 97,
        group_size=32,
        bits=2,
    )
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            keys["weight"]: weight,
            keys["scales"]: scales.astype(mx.bfloat16),
            keys["biases"]: biases.astype(mx.bfloat16),
            "other.weight": mx.zeros((1, 1)),
        },
    )
    mapping = {key: shard.name for key in keys.values()}
    if mapping_override is not None:
        mapping = mapping_override(mapping, keys)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    (tmp_path / "config.json").write_text("{}")
    return keys


def _patch_minimal_offload_loader(monkeypatch):
    args = SimpleNamespace(engram_layer_ids=[1], n_layers=1)
    monkeypatch.setattr(native_load.ModelArgs, "from_dict", lambda _cfg: args)
    monkeypatch.setattr(native_load, "Model", _FakeLoadModel)
    monkeypatch.setattr(native_load, "load_token_map", lambda *_args: [0])
    monkeypatch.setattr(
        native_load,
        "reshape_grouped_wo_a",
        lambda items, _args: list(items),
    )
    return args


def test_load_wires_disk_engram_and_skips_its_resident_tensors(
    tmp_path, monkeypatch
) -> None:
    from vllm_mlx.models.deepseek_v41_native.engram import (
        DiskQuantizedEngramEmbedding,
    )

    _write_offload_fixture(tmp_path)
    _patch_minimal_offload_loader(monkeypatch)
    monkeypatch.setattr(
        native_load,
        "json",
        SimpleNamespace(
            load=lambda stream: (
                {"weight_map": json.load(stream)["weight_map"]}
                if stream.name.endswith("index.json")
                else {"quantization": {"group_size": 32, "engram_bits": 2}}
            )
        ),
    )

    model, _args = native_load.load(str(tmp_path), lazy=True, engram_ssd_offload=True)

    assert isinstance(model.layers[0].engram.embed, DiskQuantizedEngramEmbedding)
    assert [name for name, _value in model.loaded] == ["other.weight"]
    assert model.evaluated
    model.layers[0].engram.embed.close()


@pytest.mark.parametrize(
    ("index_payload", "message"),
    [
        (None, "requires a safetensors index"),
        ({"weight_map": []}, "requires a valid weight map"),
    ],
)
def test_load_rejects_missing_or_invalid_offload_index(
    tmp_path, monkeypatch, index_payload, message
) -> None:
    (tmp_path / "config.json").write_text("{}")
    if index_payload is not None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(index_payload)
        )
    _patch_minimal_offload_loader(monkeypatch)

    with pytest.raises(ValueError, match=message):
        native_load.load(str(tmp_path), lazy=True, engram_ssd_offload=True)


@pytest.mark.parametrize(
    ("mapping_override", "message"),
    [
        (
            lambda mapping, keys: {
                key: value for key, value in mapping.items() if key != keys["biases"]
            },
            "missing indexed tensor",
        ),
        (
            lambda mapping, keys: {**mapping, keys["biases"]: "other.safetensors"},
            "must share one shard",
        ),
    ],
)
def test_load_rejects_incomplete_engram_mapping(
    tmp_path, monkeypatch, mapping_override, message
) -> None:
    _write_offload_fixture(tmp_path, mapping_override=mapping_override)
    _patch_minimal_offload_loader(monkeypatch)
    original_load = json.load

    def config_or_index(stream):
        if stream.name.endswith("index.json"):
            return original_load(stream)
        return {"quantization": {"group_size": 32, "engram_bits": 2}}

    monkeypatch.setattr(native_load.json, "load", config_or_index)

    with pytest.raises(ValueError, match=message):
        native_load.load(str(tmp_path), lazy=True, engram_ssd_offload=True)


def test_load_rejects_offload_without_affine_engram_quantization(
    tmp_path, monkeypatch
) -> None:
    _write_offload_fixture(tmp_path)
    _patch_minimal_offload_loader(monkeypatch)

    with pytest.raises(ValueError, match="affine-quantized"):
        native_load.load(str(tmp_path), lazy=True, engram_ssd_offload=True)


def test_load_preserves_resident_affine_engram_path(tmp_path, monkeypatch) -> None:
    from vllm_mlx.models.deepseek_v41_native.engram import QuantizedEngramEmbedding

    _write_offload_fixture(tmp_path)
    _patch_minimal_offload_loader(monkeypatch)
    original_load = json.load

    def config_or_index(stream):
        if stream.name.endswith("index.json"):
            return original_load(stream)
        return {"quantization": {"group_size": 32, "engram_bits": 2}}

    monkeypatch.setattr(native_load.json, "load", config_or_index)

    model, _args = native_load.load(str(tmp_path), lazy=True, strict=False)

    assert isinstance(model.layers[0].engram.embed, QuantizedEngramEmbedding)


def test_quantized_grouped_wo_a_preserves_batch_sequence_and_group_axes() -> None:
    args = ModelArgs(
        dim=16,
        n_heads=8,
        head_dim=8,
        rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=3,
        o_groups=2,
        n_layers=1,
        compress_ratios=(0,),
    )
    attention = Attention(0, args)
    attention.wo_a = attention.wo_a.to_quantized(group_size=32, bits=2)
    grouped = mx.zeros((1, 5, args.o_groups, 32), dtype=mx.float32)

    projected = attention.wo_a(grouped[..., None, :]).squeeze(-2)

    assert projected.shape == (1, 5, args.o_groups, args.o_lora_rank)


def test_grouped_wo_a_accepts_release_flat_float_layout_for_prefill() -> None:
    layer = GroupedOutputLinear(input_dims=8, output_dims=3, num_heads=2)
    flat = mx.arange(48, dtype=mx.float32).reshape(6, 8)
    layer.weight = flat
    values = mx.arange(2 * 5 * 2 * 8, dtype=mx.float32).reshape(2, 5, 2, 8)

    actual = layer(values[..., None, :]).squeeze(-2)
    expected = mx.einsum("bsgd,grd->bsgr", values, flat.reshape(2, 3, 8))
    mx.eval(actual, expected)

    assert actual.shape == (2, 5, 2, 3)
    assert mx.array_equal(actual, expected).item()


def test_quantized_compressor_uses_logical_module_projection() -> None:
    args = ModelArgs(
        dim=64,
        head_dim=32,
        n_layers=1,
        compress_ratios=(2,),
    )
    compressor = Compressor(args, 0)
    compressor.wkv = compressor.wkv.to_quantized(group_size=32, bits=2)
    compressor.wgate = compressor.wgate.to_quantized(group_size=32, bits=2)
    state = CompressorState(bsz=1, ratio=2, head_dim=args.head_dim)

    output = compressor(mx.zeros((1, 2, args.dim)), start_pos=0, comp_state=state)
    mx.eval(output)

    assert output.shape == (1, 1, args.head_dim)


def test_native_model_defaults_to_watchdog_safe_layer_boundaries() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
    )

    assert Model(args).eval_interval == 1


def test_native_model_captures_configured_dspark_inputs() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        dspark_target_layer_ids=(0,),
    )
    model = Model(args)

    class IdentityBlock:
        engram = None

        def __call__(self, h, pre_mix, *_args):
            return h, pre_mix

    model.layers = [IdentityBlock()]
    cache = model.make_cache(max_seq_len=8)

    logits, hidden = model(mx.array([[1, 2]]), cache, return_dspark_hidden=True)
    mx.eval(logits, hidden)

    assert logits.shape == (1, 2, args.vocab_size)
    assert hidden.shape == (1, 2, args.dim)


def test_native_model_captures_dspark_inputs_in_configured_order() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        dspark_target_layer_ids=(1, 0),
    )
    model = Model(args)

    class AddBlock:
        engram = None

        def __init__(self, value):
            self.value = value

        def __call__(self, h, pre_mix, *_args):
            return h + self.value, pre_mix

    model.layers = [AddBlock(1), AddBlock(2)]
    cache = model.make_cache(max_seq_len=8)

    _, hidden = model(mx.array([[1]]), cache, return_dspark_hidden=True)
    first, second = mx.split(hidden, 2, axis=-1)

    assert mx.array_equal(first, second + 1).item()


def test_native_model_prefetches_each_disk_engram_before_layers_run() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        engram_layer_ids=(0,),
    )
    model = Model(args)
    prefetched = []

    class FakeHasher:
        def __call__(self, ids, start_pos, cache_ids):
            del ids, start_pos, cache_ids
            return np.array([[[[3, 5]]]], dtype=np.int64)

    class FakeEmbed:
        def prefetch(self, hashes):
            prefetched.append(hashes.copy())

    class FakeEngram:
        layer_hash_index = 0
        embed = FakeEmbed()

        def __call__(self, h, _hashes):
            return h

    class IdentityBlock:
        engram = FakeEngram()

        def __call__(self, h, pre_mix, *_args):
            return h, pre_mix

    model.engram_hasher = FakeHasher()
    model.layers = [IdentityBlock()]
    cache = model.make_cache(max_seq_len=8)

    logits = model(mx.array([[1]]), cache)
    mx.eval(logits)

    assert len(prefetched) == 1
    assert np.array_equal(prefetched[0], np.array([[[3, 5]]]))


def test_speculative_cache_rollback_restores_partial_compressor_group() -> None:
    args = ModelArgs(
        dim=64,
        n_layers=1,
        head_dim=32,
        compress_ratios=(2,),
        kv_source_layers=(0,),
    )
    cache = ModelCache(args, max_seq_len=16)
    state = cache.layers[0].comp_state
    assert state is not None
    cache.offset = 4
    cache.begin_forward()
    state.pending_start = 4
    state.pending_kv = mx.arange(3 * 32).reshape(1, 3, 32).astype(mx.float32)
    state.pending_score = state.pending_kv + 100
    expected_kv = state.pending_kv[:, :1]
    expected_score = state.pending_score[:, :1]
    cache.offset = 7

    cache.rollback(5)

    assert cache.offset == 5
    assert mx.array_equal(state.kv_state[:, :1], expected_kv).item()
    assert mx.array_equal(state.score_state[:, :1], expected_score).item()


def test_speculative_cache_rollback_rejects_older_forward() -> None:
    args = ModelArgs(dim=64, n_layers=0, head_dim=32, compress_ratios=())
    cache = ModelCache(args, max_seq_len=16)
    cache.rollback_start = 4
    cache.offset = 7

    with pytest.raises(ValueError, match="outside latest forward"):
        cache.rollback(3)


def test_compressor_rollback_to_group_boundary_clears_partial_state() -> None:
    state = CompressorState(bsz=1, ratio=2, head_dim=4)
    state.begin_forward(3)
    state.kv_state[:] = 7
    state.score_state[:] = 9

    state.rollback(4)

    assert mx.array_equal(state.kv_state, mx.zeros_like(state.kv_state)).item()
    assert mx.all(mx.isneginf(state.score_state)).item()


def test_compressor_rollback_to_forward_start_restores_carried_partial() -> None:
    state = CompressorState(bsz=1, ratio=2, head_dim=4)
    state.kv_state[:, :1] = 3
    state.score_state[:, :1] = 5
    state.begin_forward(3)
    state.kv_state[:] = 9
    state.score_state[:] = 11

    state.rollback(3)

    assert mx.all(state.kv_state[:, :1] == 3).item()
    assert mx.all(state.score_state[:, :1] == 5).item()


def test_model_only_snapshots_cache_when_rollback_is_enabled(monkeypatch) -> None:
    args = ModelArgs(dim=64, n_layers=0, head_dim=32, compress_ratios=())
    model = Model(args)
    cache = model.make_cache(max_seq_len=8)
    calls = 0
    original = cache.begin_forward

    def record_begin_forward():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(cache, "begin_forward", record_begin_forward)

    model(mx.array([[1]]), cache)
    assert calls == 0

    model(mx.array([[1]]), cache, enable_rollback=True)
    assert calls == 1
