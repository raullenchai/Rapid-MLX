# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    _prefill_cap_violation,
)
from rapid_mlx.request import ClientRequestError  # noqa: E402


class Cache:
    def __init__(self):
        self.offset = 0
        self.state = mx.array([0])

    def empty(self):
        return self.offset == 0


class ArraysCache:
    @property
    def state(self):
        return []

    def empty(self):
        return True


class LanguageModel:
    def __init__(self):
        self.calls = []
        self._rope_deltas = None

    def __call__(self, ids, inputs_embeds, cache, **kwargs):
        self.calls.append(
            (ids.shape[1], inputs_embeds.shape[1], set(kwargs), cache[0].offset)
        )
        self._rope_deltas = kwargs.get("rope_deltas", self._rope_deltas)
        cache[0].offset += ids.shape[1]
        return mx.zeros((1, ids.shape[1], 2))


class Model:
    def __init__(self):
        self.config = SimpleNamespace(model_type="qwen3_5", image_token_id=99)
        self.embeddings = []
        self.calls = []

    def get_input_embeddings(self, ids, pixel_values=None, **kwargs):
        self.embeddings.append(ids.shape[1])
        values = {
            "inputs_embeds": mx.zeros((1, ids.shape[1], 4)),
            "position_ids": mx.array([1]),
            "rope_deltas": mx.array([2]),
        }
        return SimpleNamespace(**values, to_dict=lambda: values)

    def __call__(self, ids, cache=None, **kwargs):
        self.calls.append(ids.shape[1])
        return mx.zeros((1, ids.shape[1], 2))


def generator(monkeypatch):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = Model()
    gen.language_model = LanguageModel()
    gen.prefill_step_size = 4
    gen.vision_prefill_token_budget = 8192
    monkeypatch.setattr(gen, "_media_boundary_plan", lambda *args: None)
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    return gen


def request(length=11):
    return MLLMBatchRequest(
        uid=1,
        request_id="test",
        prompt="test",
        images=["image"],
        input_ids=mx.array([[99] + [1] * (length - 1)]),
        pixel_values=mx.array([1]),
    )


def run(gen, req, cache):
    steps = gen._vision_encoding_steps(req, cache)
    with pytest.raises(StopIteration):
        while True:
            next(steps)


def test_batched_media_input_is_not_chunkable(monkeypatch):
    gen = generator(monkeypatch)
    req = request()
    req.input_ids = mx.array([[99] + [1] * 10, [99] + [1] * 10])
    assert not gen._media_prefill_chunkable(req, [Cache()])


def test_policy_lookup_failure_is_not_chunkable(monkeypatch):
    gen = generator(monkeypatch)

    class BrokenPolicyModel(Model):
        @property
        def chunked_prefill_policy(self):
            raise ValueError("policy lookup failed")

    gen.model = BrokenPolicyModel()
    assert not gen._media_prefill_static_chunkable(request())


@pytest.mark.parametrize("offset, expected", [(0, True), (5, False)])
def test_nested_cache_freshness(monkeypatch, offset, expected):
    gen = generator(monkeypatch)
    cache = [[SimpleNamespace(caches=[SimpleNamespace(offset=offset)])]]
    assert gen._media_prefill_chunkable(request(), cache) is expected


def test_cache_empty_failure_is_not_fresh(monkeypatch):
    gen = generator(monkeypatch)

    class BrokenCache:
        def empty(self):
            raise ValueError("cache failed")

    assert not gen._media_prefill_chunkable(request(), [[BrokenCache()]])


def test_media_budget_token_count_without_input_ids(monkeypatch):
    gen = generator(monkeypatch)
    assert MLLMBatchGenerator._media_budget_token_count(gen, None, True) is None


def test_media_token_counts_without_placeholder_ids(monkeypatch):
    gen = generator(monkeypatch)
    gen.model.config = SimpleNamespace(model_type="qwen3_5")
    ids = mx.array([[1, 2, 3]])
    assert gen._vision_token_count(ids) == ids.size
    assert gen._media_budget_token_count(ids, False) == ids.size


def test_chunked_cold_redo_rebuilds_cache_and_delegates(monkeypatch):
    from unittest.mock import Mock

    import mlx_lm.models.cache as cache_module

    gen = generator(monkeypatch)
    rebuilt = Cache()
    make_cache = Mock(return_value=[rebuilt])
    prefill = Mock(return_value="result")
    monkeypatch.setattr(cache_module, "make_prompt_cache", make_cache)
    monkeypatch.setattr(gen, "_chunked_media_lm_prefill", prefill)
    ids = mx.array([[1, 2]])
    cache = [Cache()]
    kwargs = {"pixel_values": "pixels"}

    assert gen._media_cold_redo(ids, cache, kwargs, chunked=True) == "result"
    make_cache.assert_called_once_with(gen.language_model)
    assert cache == [rebuilt]
    prefill.assert_called_once_with(ids, cache, kwargs)


def test_chunked_cold_redo_steps_rebuilds_cache_and_yields(monkeypatch):
    from unittest.mock import Mock

    import mlx_lm.models.cache as cache_module

    gen = generator(monkeypatch)
    rebuilt = Cache()
    make_cache = Mock(return_value=[rebuilt])
    monkeypatch.setattr(cache_module, "make_prompt_cache", make_cache)
    req = request()
    cache = [Cache()]
    steps = gen._media_cold_redo_steps(
        req.input_ids, cache, {"pixel_values": req.pixel_values}, chunked=True
    )
    next(steps)
    make_cache.assert_called_once_with(gen.language_model)
    assert cache == [rebuilt]
    assert [call[0] for call in gen.language_model.calls] == [4]
    next(steps)
    assert [call[0] for call in gen.language_model.calls] == [4, 4]
    list(steps)
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]


def test_synchronous_chunked_media_prefill_drains_steps(monkeypatch):
    gen = generator(monkeypatch)
    req = request()
    output = gen._chunked_media_lm_prefill(
        req.input_ids, [Cache()], {"pixel_values": req.pixel_values}
    )
    assert output.shape == (1, 1, 2)
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]


def test_synchronous_chunked_suffix_drains_steps(monkeypatch):
    gen = generator(monkeypatch)
    calls = []
    sentinel = object()

    def forward(ids, cache, delta):
        calls.append((ids.shape[1], cache, delta))
        return sentinel

    monkeypatch.setattr(gen, "_media_suffix_forward", forward)
    cache = [Cache()]
    delta = object()
    assert (
        gen._media_chunked_suffix_forward(request().input_ids, cache, delta, True)
        is sentinel
    )
    assert calls == [(4, cache, delta), (4, cache, delta), (3, cache, delta)]


def test_chunked_media_metadata_and_last_token(monkeypatch):
    gen = generator(monkeypatch)
    run(gen, request(), [Cache()])
    assert gen.model.embeddings == [11]
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]
    assert [call[3] for call in gen.language_model.calls] == [0, 4, 8, 10]
    assert all(
        n == embeds and {"position_ids", "rope_deltas"} <= keys
        for n, embeds, keys, _ in gen.language_model.calls
    )
    assert ["pixel_values" in call[2] for call in gen.language_model.calls] == [
        True,
        False,
        False,
        False,
    ]


def test_cooperative_prefill_remove_stops_remaining_chunks(monkeypatch):
    gen = generator(monkeypatch)
    gen.processor = SimpleNamespace(tokenizer=object())
    gen._stats = SimpleNamespace(prompt_tokens=0, prompt_time=0.0)
    gen.active_batch = None
    gen.cooperative_prefill = True
    gen.completion_batch_size = 1
    gen._pending_prefill = None
    req = request()
    gen.unprocessed_requests = [req]
    monkeypatch.setattr(gen, "_preprocess_request", lambda req: None)
    monkeypatch.setattr(gen, "_lookup_exact_text_prefix", lambda req: [Cache()])

    assert gen._next_unlocked() == []
    assert len(gen.language_model.calls) == 1
    assert gen._next_unlocked() == []
    assert len(gen.language_model.calls) == 2
    gen.remove([req.uid])
    assert gen._pending_prefill is None
    assert req not in gen.unprocessed_requests
    assert gen._next_unlocked() == []
    assert len(gen.language_model.calls) == 2


def test_chunked_media_respects_budget_below_prefill_step(monkeypatch):
    gen = generator(monkeypatch)
    gen.prefill_step_size = 16
    gen.vision_prefill_token_budget = 4
    req = request(11)
    req.chunked_media_prefill = True

    run(gen, req, [Cache()])

    assert gen.model.embeddings == [11]
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]
    assert all(
        call[0] <= gen.vision_prefill_token_budget for call in gen.language_model.calls
    )


@pytest.mark.parametrize("budget", [16, 0, -1])
def test_media_prefill_chunk_uses_effective_chunk_for_large_or_nonpositive_budget(
    monkeypatch, budget
):
    gen = generator(monkeypatch)
    gen.vision_prefill_token_budget = budget
    assert gen._media_prefill_chunk_tokens() == 4


def test_chunked_media_with_arrays_cache(monkeypatch):
    gen = generator(monkeypatch)
    run(gen, request(), [Cache(), ArraysCache()])
    assert gen.model.embeddings == [11]
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]


def test_stale_cache_rechecks_total_prefill_budget_before_forward(monkeypatch):
    gen = generator(monkeypatch)
    gen.vision_prefill_token_budget = 8
    req = request()
    req.chunked_media_prefill = True
    cache = Cache()
    cache.offset = 1
    with pytest.raises(ClientRequestError, match="exceeds the per-batch cap"):
        run(gen, req, [cache])
    assert not gen.model.calls
    assert not gen.model.embeddings
    assert not gen.language_model.calls


def test_nonchunkable_media_with_unset_budget_runs_single_forward(monkeypatch):
    gen = generator(monkeypatch)
    gen.vision_prefill_token_budget = 0
    req = request()
    req.chunked_media_prefill = True
    cache = Cache()
    cache.offset = 1

    run(gen, req, [cache])

    assert gen.model.calls == [11]


@pytest.mark.parametrize(
    "condition", ["family", "opt_out", "videos", "extra", "short", "stale"]
)
def test_unsupported_requests_keep_single_forward(monkeypatch, condition):
    gen = generator(monkeypatch)
    req = request(4 if condition == "short" else 11)
    cache = [Cache()]
    if condition == "family":
        gen.model.config.model_type = "gemma3"
    elif condition == "opt_out":
        gen.model.no_chunked_prefill = True
    elif condition == "videos":
        req.videos = ["video"]
    elif condition == "extra":
        req.extra_kwargs = {"token_type_ids": mx.array([1])}
    elif condition == "stale":
        cache[0].offset = 1
    run(gen, req, cache)
    assert gen.model.calls == [req.input_ids.shape[1]]
    assert not gen.model.embeddings


@pytest.mark.parametrize("policy_result", [False, True, "raises"])
def test_language_model_policy_controls_chunking(monkeypatch, policy_result):
    gen = generator(monkeypatch)

    def policy(**kwargs):
        assert kwargs == {
            "input_ids": None,
            "inputs_embeds": None,
            "prompt_cache": None,
            "draft_model": None,
            "draft_kind": None,
            "prefill_kwargs": {},
        }
        if policy_result == "raises":
            raise ValueError("policy failed")
        return policy_result

    gen.language_model.chunked_prefill_policy = policy
    run(gen, request(), [Cache()])
    if policy_result is True:
        assert gen.model.embeddings == [11]
        assert [call[0] for call in gen.language_model.calls] == [4, 4, 2, 1]
    else:
        assert gen.model.calls == [11]
        assert not gen.model.embeddings


def test_wrapper_policy_cannot_override_language_model_policy(monkeypatch):
    gen = generator(monkeypatch)
    gen.model.chunked_prefill_policy = lambda **kwargs: True
    gen.language_model.chunked_prefill_policy = lambda **kwargs: False

    run(gen, request(), [Cache()])

    assert gen.model.calls == [11]
    assert not gen.model.embeddings


def test_wrapper_policy_cannot_override_language_model_opt_out(monkeypatch):
    gen = generator(monkeypatch)
    gen.model.chunked_prefill_policy = lambda **kwargs: True
    gen.language_model.no_chunked_prefill = True

    run(gen, request(), [Cache()])

    assert gen.model.calls == [11]
    assert not gen.model.embeddings


def test_vision_budget_counts_only_placeholders_for_chunkable_requests(monkeypatch):
    gen = generator(monkeypatch)
    req = request(12000)
    req.input_ids = mx.array([[99] * 1500 + [1] * 10500])
    req.chunked_media_prefill = True
    assert gen._media_budget_token_count(req.input_ids, True) == 1500
    assert gen._media_budget_token_count(req.input_ids, False) == 12000
    assert _prefill_cap_violation([req], 8192, gen._vision_token_count) is None
    req.chunked_media_prefill = False
    msg = _prefill_cap_violation([req], 8192, gen._vision_token_count)
    assert "exceeds the per-batch cap" in msg
    assert "--prefill-step-size" not in msg
    assert "--vision-prefill-token-budget" in msg


def test_store_prefix_and_suffix_are_chunked(monkeypatch):
    gen = generator(monkeypatch)
    req = request(20)
    monkeypatch.setattr(
        gen, "_media_boundary_plan", lambda *args: ("store", None, 10, 0, "key")
    )
    monkeypatch.setattr(gen, "_media_mrope_save", lambda: None)
    monkeypatch.setattr(gen, "_media_store", lambda *args, **kwargs: True)
    suffix = []

    def forward(ids, cache, delta):
        suffix.append((ids.shape[1], cache[0].offset))
        cache[0].offset += ids.shape[1]
        return mx.zeros((1, ids.shape[1], 2))

    monkeypatch.setattr(gen, "_media_suffix_forward", forward)
    run(gen, req, [Cache()])
    assert gen.model.embeddings == [10]
    assert [call[0] for call in gen.language_model.calls] == [4, 4, 1, 1]
    assert suffix == [(4, 10), (4, 14), (2, 18)]


def test_chunked_media_prefill_yields_between_chunks(monkeypatch):
    gen = generator(monkeypatch)
    steps = gen._vision_encoding_steps(request(), [Cache()])
    next(steps)
    assert len(gen.language_model.calls) == 1
    next(steps)
    assert len(gen.language_model.calls) == 2
    steps.close()
    assert len(gen.language_model.calls) == 2
