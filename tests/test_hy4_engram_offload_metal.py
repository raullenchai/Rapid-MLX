"""Native row and full-forward equivalence for indexed Hy4 Engram offload."""

import json
from dataclasses import asdict

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
if not mx.metal.is_available():
    pytest.skip("Metal is unavailable", allow_module_level=True)
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.utils import quantize_model

from rapid_mlx.models.engram_offload import DiskQuantizedEngramEmbedding
from rapid_mlx.models.hy_v4_flash import Model
from rapid_mlx.models.hy_v4_flash_load import load
from tests.test_hy_v4_flash import tiny


@pytest.fixture
def checkpoint(tmp_path):
    mx.random.seed(29)
    args = tiny(
        hidden_size=64,
        vocab_size=128,
        intermediate_size=64,
        q_lora_rank=64,
        kv_lora_rank=64,
        routed_expert_hidden_size=64,
        moe_intermediate_size=64,
        engram_n_embed_per_ngram=64,
        engram_pad_id=127,
    )
    model, config = quantize_model(Model(args), asdict(args), 32, 4)
    # Real builds retain BF16 affine scales/biases.
    table = model.layers[1].engram_layer.multi_head_embedding.embedding
    table.scales = table.scales.astype(mx.bfloat16)
    table.biases = table.biases.astype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    mapping = {}
    for name in weights:
        filename = (
            "table.safetensors"
            if ".multi_head_embedding." in name
            else "model.safetensors"
        )
        mapping[name] = filename
    for filename in set(mapping.values()):
        mx.save_safetensors(
            str(tmp_path / filename),
            {k: v for k, v in weights.items() if mapping[k] == filename},
        )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    return tmp_path


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
def test_rows_prefetch_mismatch_and_logits(checkpoint, monkeypatch, dtype):
    resident, _ = load(checkpoint, engram_ssd_offload=False)
    offloaded, _ = load(checkpoint, engram_ssd_offload=True)
    table = offloaded.layers[1].engram_layer.multi_head_embedding.embedding
    assert isinstance(table, DiskQuantizedEngramEmbedding)
    assert not table.parameters()
    expected_table = resident.layers[1].engram_layer.multi_head_embedding.embedding
    ids = mx.array([[0, 3, 3, 35]])
    expected = expected_table(ids).astype(mx.float32)
    for prefetch in (None, np.array(ids), np.array([[1, 2, 4, 5]])):
        if prefetch is not None:
            table.prefetch(prefetch)
        actual = table(ids)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
    # Exercise prefill, incremental history, cache trim and replay.
    rc, dc = resident.make_cache(), offloaded.make_cache()
    for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
        ids = mx.array(tokens, dtype=dtype)
        expected, actual = resident(ids, rc), offloaded(ids, dc)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
    for caches in (rc, dc):
        for layer in caches:
            layer.trim(2)
    expected, actual = (
        resident(mx.array([[7, 8]], dtype=dtype), rc),
        offloaded(mx.array([[7, 8]], dtype=dtype), dc),
    )
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    # Force synchronous rows while retaining precisely the same model weights.
    monkeypatch.setattr(table, "prefetch", lambda _: None)
    synchronous = offloaded(mx.array([[1, 2, 3]], dtype=dtype))
    expected = resident(mx.array([[1, 2, 3]], dtype=dtype))
    mx.eval(synchronous, expected)
    np.testing.assert_array_equal(np.array(synchronous), np.array(expected))
    table.close()


@pytest.mark.parametrize("offload", [False, True])
def test_uint32_logits_and_greedy_stream_match_int32(checkpoint, offload):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model, _ = load(checkpoint, engram_ssd_offload=offload)
    table = model.layers[1].engram_layer.multi_head_embedding.embedding
    try:
        signed_cache, unsigned_cache = model.make_cache(), model.make_cache()
        for tokens in ([[1, 2, 3]], [[4]], [[5, 6]]):
            signed = mx.array(tokens, dtype=mx.int32)
            unsigned = signed.astype(mx.uint32)
            for expected, actual in (
                (model(signed), model(unsigned)),
                (model(signed, signed_cache), model(unsigned, unsigned_cache)),
            ):
                mx.eval(expected, actual)
                np.testing.assert_array_equal(np.array(actual), np.array(expected))

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}))
        )
        # Build the greedy reference with signed IDs at every decode step.
        cache = model.make_cache()
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        expected_tokens = []
        for _ in range(3):
            token = mx.argmax(model(ids, cache)[:, -1, :], axis=-1).item()
            expected_tokens.append(token)
            ids = mx.array([[token]], dtype=mx.int32)
        for dtype in (mx.int32, mx.uint32):
            responses = list(
                stream_generate(
                    model,
                    tokenizer,
                    mx.array([1, 2, 3], dtype=dtype),
                    max_tokens=3,
                    sampler=make_sampler(temp=0),
                )
            )
            assert [response.token for response in responses] == expected_tokens
    finally:
        if offload:
            table.close()


def test_product_loader_routes_without_generic_model_eval(checkpoint, monkeypatch):
    import mlx_lm.utils
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}))
    )
    tokenizer.save_pretrained(checkpoint)
    monkeypatch.setattr(
        mlx_lm.utils,
        "load_model",
        lambda *a, **k: pytest.fail("generic model evaluation reached"),
    )
    model, _ = load_model_with_fallback(str(checkpoint))
    table = model.layers[1].engram_layer.multi_head_embedding.embedding
    assert isinstance(table, DiskQuantizedEngramEmbedding)
    table.close()
    monkeypatch.setenv("HY4_ENGRAM_SSD_OFFLOAD", "0")
    model, _ = load_model_with_fallback(str(checkpoint))
    assert isinstance(
        model.layers[1].engram_layer.multi_head_embedding.embedding,
        nn.QuantizedEmbedding,
    )


def test_merged_cache_prefetch_matches_independent_requests(checkpoint):
    from mlx_lm.models.cache import CacheList

    resident, _ = load(checkpoint, engram_ssd_offload=False)
    disk, _ = load(checkpoint)
    first, second = disk.make_cache(), disk.make_cache()
    disk(mx.array([[1, 2, 3, 4]]), first)
    disk(mx.array([[3, 4]]), second)
    merged = [CacheList.merge([a, b]) for a, b in zip(first, second)]
    actual = disk(mx.array([[5], [5]]), merged)
    expected = mx.concatenate(
        (
            resident(mx.array([[1, 2, 3, 4, 5]]))[:, -1:],
            resident(mx.array([[3, 4, 5]]))[:, -1:],
        ),
        axis=0,
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=3e-5).item()
    disk.layers[1].engram_layer.multi_head_embedding.embedding.close()
