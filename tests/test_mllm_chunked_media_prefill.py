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
