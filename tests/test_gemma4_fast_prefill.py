# SPDX-License-Identifier: Apache-2.0
"""Weight-free numerical and consumer contracts for KV-shared prefill."""

import pytest

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.gemma4_text import Gemma4TextWrapper
from rapid_mlx.models.gemma4_vendored.config import TextConfig
from rapid_mlx.models.gemma4_vendored.language import DecoderLayer, LanguageModel


@pytest.fixture
def model():
    state = mx.random.state
    mx.random.seed(11)
    config = TextConfig.from_dict(
        dict(
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=4,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=4,
            use_double_wide_mlp=True,
        )
    )
    try:
        yield Gemma4TextWrapper(LanguageModel(config))
    finally:
        mx.random.state = state


@pytest.fixture
def lengths(monkeypatch):
    received = []
    original = DecoderLayer.__call__

    def record(self, x, *args, **kwargs):
        received.append(x.shape[1])
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(DecoderLayer, "__call__", record)
    return received


def assert_cache_equal(full, fast):
    for a, b in zip(full, fast, strict=True):
        assert a.offset == b.offset
        assert all(
            bool(mx.array_equal(x, y)) for x, y in zip(a.state, b.state, strict=True)
        )


@pytest.mark.parametrize("chunks", [(7,), (3, 5, 6)])
def test_last_logits_cache_and_greedy_continuation(model, lengths, chunks):
    full, fast = model.make_cache(), model.make_cache()
    for n in chunks:
        ids = (mx.arange(n)[None, :] + 1) % 32
        baseline = model(ids, cache=full)
        lengths.clear()
        kept = model(ids, cache=fast, logits_to_keep=1)
        assert lengths == [n, n, 1, 1]
        assert kept.shape == (1, 1, 32)
        assert bool(mx.allclose(kept, baseline[:, -1:], atol=1e-5, rtol=1e-5))
        assert_cache_equal(full, fast)
    for _ in range(5):
        a, b = mx.argmax(baseline[:, -1:], -1), mx.argmax(kept[:, -1:], -1)
        assert bool(mx.array_equal(a, b))
        baseline, kept = model(a, cache=full), model(b, cache=fast)
        assert bool(mx.array_equal(baseline, kept))
        assert_cache_equal(full, fast)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # speculative verify and multi-position logprobs use this default
        {"logits_to_keep": 2},
        {"logits_to_keep": 1, "return_hidden": True},
        {"logits_to_keep": 1, "capture_layer_ids": [2]},
        {"logits_to_keep": 1, "capture_layer_ids": []},
        {"logits_to_keep": 1, "hidden_sink": []},
        {"logits_to_keep": 1, "return_shared_kv": True},
        {"logits_to_keep": 1, "skip_final_norm": True},
    ],
)
def test_full_position_consumers(model, lengths, kwargs):
    ids = mx.array([[1, 2, 3, 4, 5]])
    output = model(ids, cache=model.make_cache(), **kwargs)
    assert output.shape == (1, 5, 32)
    assert lengths == [5] * 4


def test_capture_retains_every_hidden_position(model, lengths):
    lm = model.language_model
    output = lm(
        mx.array([[1, 2, 3, 4, 5]]),
        cache=lm.make_cache(),
        logits_to_keep=1,
        capture_layer_ids=[0, 2, 3],
    )
    assert [h.shape[1] for h in output.hidden_states] == [5] * 3
    assert lengths == [5] * 4


def test_kill_switch(model, lengths, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_GEMMA4_FAST_PREFILL", "0")
    ids = mx.array([[1, 2, 3, 4, 5]])
    baseline = model(ids, cache=model.make_cache())
    kept = model(ids, cache=model.make_cache(), logits_to_keep=1)
    assert lengths == [5] * 8
    assert bool(mx.array_equal(baseline, kept))


def test_multi_sequence_padding_stays_full(model, lengths):
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    caches = [BatchRotatingKVCache(4, [2, 0]), BatchKVCache([2, 0])]
    output = model(
        mx.array([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]), cache=caches, logits_to_keep=1
    )
    assert output.shape == (2, 5, 32)
    assert lengths == [5] * 4


def test_no_sharing_is_byte_identical(model, lengths):
    config = model.config
    config.num_kv_shared_layers = 0
    lm = Gemma4TextWrapper(LanguageModel(config))
    ids = mx.array([[1, 2, 3, 4, 5]])
    baseline = lm(ids, cache=lm.make_cache())
    kept = lm(ids, cache=lm.make_cache(), logits_to_keep=1)
    assert lengths == [5] * 8
    assert bool(mx.array_equal(baseline, kept))


def test_decode_and_embeddings_keep_full_contract(model, lengths):
    lm = model.language_model
    ids = mx.array([[1, 2, 3]])
    embeddings = lm.model.embed_tokens(ids) * lm.model.embed_scale
    output = lm(inputs=ids, inputs_embeds=embeddings, return_hidden=True)
    assert output.logits.shape == (1, 3, 32)
    assert output.hidden_states[0].shape == (1, 3, 16)
    lengths.clear()
    assert model(mx.array([[1]]), cache=model.make_cache(), logits_to_keep=1).shape == (
        1,
        1,
        32,
    )
    assert lengths == [1] * 4


@pytest.mark.parametrize("prefix", [False, True])
def test_real_batch_generator_chunking_prefix_and_logprobs(model, lengths, prefix):
    from mlx_lm.generate import BatchGenerator

    from rapid_mlx.gemma4_prefill import install_gemma4_prefill

    def run(fast):
        cache = model.make_cache()
        if prefix:
            model(mx.array([[7, 8, 9]]), cache=cache)
        bg = BatchGenerator(
            model,
            max_tokens=4,
            prefill_step_size=3,
            stream=mx.default_stream(mx.default_device()),
        )
        if fast:
            assert install_gemma4_prefill(bg, model, speculative=False)
        lengths.clear()
        try:
            bg.insert([[1, 2, 3, 4, 5, 6, 7]], caches=[cache])
            responses = []
            while len(responses) < 4:
                _, generated = bg.next()
                responses.extend(generated)
            assert bg._prompt_batch.model is model
            if fast:
                assert lengths[:8] == [3, 3, 1, 1] * 2
            else:
                assert lengths[:8] == [3] * 8
            return responses
        finally:
            bg.close()

    full, fast = run(False), run(True)
    assert [r.token for r in full] == [r.token for r in fast]
    for a, b in zip(full, fast, strict=True):
        assert bool(mx.allclose(a.logprobs, b.logprobs, atol=1e-5, rtol=1e-5))
    assert_cache_equal(full[-1].prompt_cache, fast[-1].prompt_cache)
