# SPDX-License-Identifier: Apache-2.0
"""Prompt scoring, legacy wire alignment, and shared-cache isolation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.prompt_logprobs import (
    PROMPT_LOGPROBS_CHUNK_SIZE,
    PromptLogprobs,
    prefill_prompt_logprobs,
)
from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig
from tests.test_completions_spec_parity import _build_completions_app
from tests.test_completions_spec_parity import patched_config as _patched_config

patched_config = _patched_config


class Tokenizer:
    clean_up_tokenization_spaces = False
    eos_token_id = None

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 1, 3])
@pytest.mark.parametrize("top_k", [0, 2, 5])
def test_echo_scores_lists_and_streams(
    patched_config, monkeypatch, stream, max_tokens, top_k
):
    import asyncio
    import json

    from rapid_mlx import prompt_logprobs
    from rapid_mlx.telemetry import inference

    rejections = []
    monkeypatch.setattr(
        inference,
        "emit_capability_rejected",
        lambda *a, **kw: rejections.append((a, kw)),
    )

    build = prompt_logprobs.legacy_prompt_logprobs
    builder_calls = []

    def off_loop_builder(*args):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        builder_calls.append(args[0])
        return build(*args)

    monkeypatch.setattr(prompt_logprobs, "legacy_prompt_logprobs", off_loop_builder)

    def factory():
        e = MagicMock()
        e.tokenizer = Tokenizer()
        e.supports_prompt_logprobs = True

        async def generate(prompt, max_tokens, **kw):
            scoring = kw["_prompt_logprobs"]
            scoring.token_ids = e.tokenizer.encode(prompt)
            scoring.logprobs = [None] + [-0.5] * (len(prompt) - 1)
            scoring.top_logprobs = [None] + [
                {65 + i: -float(i + 1) for i in range(scoring.top_k)}
                for _ in prompt[1:]
            ]
            if max_tokens == 0:
                yield GenerationOutput(
                    text="",
                    prompt_tokens=len(prompt),
                    completion_tokens=0,
                    finish_reason="length",
                )
            for i in range(max_tokens):
                yield GenerationOutput(
                    text="Z" * (i + 1),
                    new_text="Z",
                    tokens=[90],
                    prompt_tokens=len(prompt),
                    completion_tokens=i + 1,
                    logprobs=mx.full((128,), -1.0),
                    finished=i == max_tokens - 1,
                    finish_reason="length" if i == max_tokens - 1 else None,
                )

        e.stream_generate = generate
        return e

    client, _ = _build_completions_app(
        patched_config, monkeypatch, engine_factory=factory
    )
    prompts = ["ab", "cde"]
    response = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": prompts,
            "echo": True,
            "logprobs": top_k,
            "max_tokens": max_tokens,
            "stream": stream,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200, response.text
    assert rejections == []
    assert builder_calls == prompts
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert len({event["id"] for event in events}) == 1
        assert sum("usage" in event for event in events) == 1
        assert events[-1]["usage"]["completion_tokens"] == 2 * max_tokens
        assert events[-1]["usage"]["prompt_tokens"] == sum(map(len, prompts))
        choices = []
        for index in range(len(prompts)):
            chunks = [
                choice
                for event in events
                for choice in event["choices"]
                if choice["index"] == index
            ]
            lp = {
                key: []
                for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset")
            }
            for chunk in chunks:
                for key, values in chunk.get("logprobs", {}).items():
                    lp[key].extend(values)
            choices.append(
                {
                    "index": index,
                    "text": "".join(c["text"] for c in chunks),
                    "logprobs": lp,
                }
            )
        assert response.text.count("data: [DONE]") == 1
    else:
        choices = response.json()["choices"]
        assert response.json()["usage"]["completion_tokens"] == 2 * max_tokens
    for index, choice in enumerate(choices):
        assert choice["index"] == index
        assert choice["text"] == prompts[index] + "Z" * max_tokens
        lp = choice["logprobs"]
        count = len(prompts[index]) + max_tokens
        assert all(len(values) == count for values in lp.values())
        assert lp["token_logprobs"][0] is None
        assert lp["top_logprobs"][0] is None
        assert all(len(top) == top_k for top in lp["top_logprobs"][1:])
        assert lp["text_offset"] == list(range(count))
        assert "".join(lp["tokens"]) == choice["text"]


def test_chunked_scores_include_boundary_targets(monkeypatch):
    import mlx_lm.models.cache as caches

    def fresh_cache():
        return [SimpleNamespace(state=mx.array(0))]

    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: fresh_cache())
    sizes = []

    def model(tokens, cache):
        sizes.append(tokens.shape[1])
        # Retain prior tokens so resetting or losing KV between chunks fails.
        context = mx.cumsum(tokens, axis=1) + cache[0].state
        cache[0].state = context[0, -1]
        return -mx.abs(mx.arange(17)[None, None, :] - (context % 17)[:, :, None])

    tokens = [i % 17 for i in range(PROMPT_LOGPROBS_CHUNK_SIZE * 2 + 9)]
    result = PromptLogprobs(3)
    cache = prefill_prompt_logprobs(model, tokens, result)
    assert cache[0].state.item() == sum(tokens[:-1])
    logits = model(mx.array([tokens[:-1]]), fresh_cache())[0]
    reference = logits - mx.logsumexp(logits, -1, keepdims=True)
    actual = mx.take_along_axis(reference, mx.array(tokens[1:])[:, None], -1)[:, 0]
    assert result.logprobs[0] is None
    assert result.logprobs[1:] == pytest.approx(actual.tolist(), abs=1e-6)
    assert sizes[:-1] == [PROMPT_LOGPROBS_CHUNK_SIZE, PROMPT_LOGPROBS_CHUNK_SIZE, 8]
    assert all(len(top) == 3 for top in result.top_logprobs[1:])


@pytest.mark.parametrize(
    "cache_name", ["prefix_cache", "memory_aware_cache", "block_aware_cache"]
)
def test_scoring_bypasses_shared_cache_fetch_and_save(cache_name):
    scheduler = Scheduler(
        model=object(),
        tokenizer=Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
        ),
    )
    shared = MagicMock()
    setattr(scheduler, cache_name, shared)
    request = Request(
        "score", [97, 98], SamplingParams(prompt_logprobs=PromptLogprobs(2))
    )
    scheduler.add_request(request)
    assert request.cached_tokens == 0
    assert request.prompt_cache is None
    assert request.requires_prompt_integrity
    assert shared.mock_calls == []
    scheduler.uid_to_request_id[1] = request.request_id
    scheduler._make_prompt_cache_save_callback()(1, [object()])
    assert shared.mock_calls == []
    scheduler.running[request.request_id] = request
    scheduler._cleanup_finished({request.request_id})
    assert shared.mock_calls == []


@pytest.mark.parametrize("mode", ["mtp", "dspark", "suffix", "dflash", "ddtree"])
def test_speculative_capability_fails_closed(mode):
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._scheduler_config = SimpleNamespace(spec_decode=mode)
    assert engine.supports_prompt_logprobs is False


def test_mllm_capability_fails_closed():
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    assert engine.supports_prompt_logprobs is False


def test_scoring_only_scheduler_never_samples(monkeypatch):
    import mlx_lm.models.cache as caches

    from rapid_mlx.request import RequestStatus

    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: [])

    def model(tokens, cache):
        return mx.zeros((1, tokens.shape[1], 128))

    scheduler = Scheduler(
        model=model,
        tokenizer=Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
        ),
    )
    scheduler.batch_generator = MagicMock()
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    request = Request(
        "zero", "abc", SamplingParams(max_tokens=0, prompt_logprobs=PromptLogprobs(2))
    )
    scheduler.add_request(request)
    result = scheduler.step()
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert len(result.outputs) == 1
    assert result.outputs[0].completion_tokens == 0
    assert result.outputs[0].prompt_tokens == 3
    assert result.outputs[0].finished
    assert result.finished_request_ids == {"zero"}
    assert not scheduler.running and not scheduler.waiting
    assert request.prompt_cache is None
    scheduler.batch_generator.next.assert_not_called()
    scheduler.batch_generator.insert.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("logprobs", [0, 3])
def test_unsupported_server_mode_returns_capability_400(stream, logprobs):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.routes.completions import install_unsupported_completions

    app = FastAPI()
    install_unsupported_completions(app)
    response = TestClient(app).post(
        "/v1/completions",
        json={
            "model": "unsupported",
            "prompt": "abc",
            "echo": True,
            "logprobs": logprobs,
            "max_tokens": 0,
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert "cannot score prompt tokens" in response.text


def test_zero_tokens_stays_invalid_without_prompt_scoring():
    from pydantic import ValidationError

    from rapid_mlx.api.models import CompletionRequest

    for params in ({}, {"echo": True}, {"logprobs": 0}):
        with pytest.raises(ValidationError):
            CompletionRequest(model="stub", prompt="abc", max_tokens=0, **params)


def test_unicode_offsets_preserve_prompt(monkeypatch):
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    tokenizer = Tokenizer()
    tokenizer._tokenizer = lambda text, **kwargs: {
        "input_ids": [1, 2, 3, 4],
        "offset_mapping": [(0, 1), (1, 2), (1, 2), (2, 3)],
    }
    result = PromptLogprobs(
        0,
        token_ids=[1, 2, 3, 4],
        logprobs=[None, -1.0, -2.0, -3.0],
        top_logprobs=[None, {}, {}, {}],
    )
    payload = legacy_prompt_logprobs("a😀b", tokenizer, result)
    assert payload["tokens"] == ["a", "", "😀", "b"]
    assert payload["text_offset"] == [0, 1, 1, 2]
    assert "".join(payload["tokens"]) == "a😀b"


@pytest.mark.parametrize("tokens", [[], [97]])
def test_empty_and_single_token_scoring(tokens, monkeypatch):
    import mlx_lm.models.cache as caches

    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: [])
    model = MagicMock()
    result = PromptLogprobs(5)
    prefill_prompt_logprobs(model, tokens, result)
    model.assert_not_called()
    payload = legacy_prompt_logprobs(Tokenizer().decode(tokens), Tokenizer(), result)
    assert all(len(array) == len(tokens) for array in payload.values())
    assert payload["token_logprobs"] == [None] * len(tokens)
    assert payload["top_logprobs"] == [None] * len(tokens)


def test_prompt_scoring_skips_disk_checkpoint():
    scheduler = Scheduler.__new__(Scheduler)
    # A scoring request must return before inspecting disk configuration or KV.
    request = Request("disk", "abc", SamplingParams(prompt_logprobs=PromptLogprobs(1)))
    scheduler._maybe_disk_checkpoint(request, SimpleNamespace())


@pytest.mark.parametrize("mode", ["dflash", "ddtree"])
def test_alternate_apps_install_scoring_capability_guard(mode, patched_config):
    from fastapi.testclient import TestClient

    patched_config(api_key=None)
    if mode == "ddtree":
        from rapid_mlx.speculative.ddtree.server import _build_app
        from tests.test_ddtree_integration import _fake_runtime

        app = _build_app(
            runtime=_fake_runtime(),
            served_model_name="test-model",
            default_max_tokens=16,
            cors_origins=[],
        )
    else:
        from rapid_mlx.speculative.dflash.runtime import DFlashRuntime
        from rapid_mlx.speculative.dflash.server import _build_app

        app = _build_app(
            model=MagicMock(),
            processor=MagicMock(),
            runtime=DFlashRuntime(
                drafter=MagicMock(), kind="dflash", drafter_repo="test"
            ),
            served_model_name="test-model",
            default_max_tokens=16,
            cors_origins=[],
        )
    client = TestClient(app, base_url="http://127.0.0.1")
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "echo": True,
            "logprobs": 0,
            "max_tokens": 0,
        },
    )
    assert response.status_code == 400
    assert "cannot score prompt tokens" in response.text


@pytest.mark.asyncio
async def test_list_stream_disconnect_closes_active_prompt(monkeypatch):
    import json

    from rapid_mlx.routes import completions

    started, closed = [], []

    async def stream(engine, prompt, request, **kwargs):
        started.append(prompt)
        try:
            yield (
                "data: "
                + json.dumps({"choices": [{"index": 0, "text": prompt}]})
                + "\n\n"
            )
        finally:
            closed.append(prompt)

    monkeypatch.setattr(completions, "stream_completion", stream)
    outputs = completions._stream_completion_prompts(None, ["first", "second"], None)
    await anext(outputs)
    await outputs.aclose()
    assert started == closed == ["first"]


@pytest.mark.parametrize("policy", ["fcfs", "shortest_validated_tail"])
@pytest.mark.parametrize(
    "path",
    [
        "normal",
        "budget_retry",
        "generator_retry",
        "fallback",
        "forced",
        "mtp",
        "chunked",
        "preempted",
    ],
)
@pytest.mark.parametrize("abort", [False, True])
def test_scoring_only_admission_liveness(monkeypatch, policy, path, abort):
    """Exercise production selection, completion delivery and subsequent insertion."""
    from rapid_mlx import prompt_logprobs
    from rapid_mlx.request import RequestStatus

    scheduler = Scheduler(
        model=MagicMock(),
        tokenizer=Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            scheduling_policy=policy,
            spec_decode="mtp" if path == "mtp" else "none",
            prefill_step_size=2 if path == "chunked" else 2048,
        ),
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [1]
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    request = Request(
        "zero", "abc", SamplingParams(max_tokens=0, prompt_logprobs=PromptLogprobs(2))
    )
    scheduler.add_request(request)
    calls = []

    def score(cls, model, tokens, result):
        calls.append(list(tokens))
        # Fail promptly on the original infinite-selection bug.
        assert len(calls) == 1
        if abort:
            assert scheduler.abort_request(request.request_id)
        return SimpleNamespace(cache=[], position=len(tokens) - 1)

    monkeypatch.setattr(
        prompt_logprobs.PromptScoringState, "create", classmethod(score)
    )
    if path == "fallback":
        scheduler._fallback_from_unobservable_prompt_runtime()
    elif path == "forced":
        request._admission_deferrals = scheduler.config.scheduling_max_deferrals
    elif path == "preempted":
        scheduler.waiting.remove(request)
        scheduler.running[request.request_id] = request
        request.status = RequestStatus.RUNNING
        scheduler._reschedule_running_requests()
    elif path in ("budget_retry", "generator_retry"):
        method = (
            "_prefill_cohort_fits"
            if path == "budget_retry"
            else "_ensure_batch_generator"
        )
        monkeypatch.setattr(scheduler, method, lambda arg: False)
        assert scheduler._schedule_waiting() == []
        assert list(scheduler.waiting) == [request]
        assert not calls
        monkeypatch.setattr(scheduler, method, lambda arg: True)

    output = scheduler.step()
    assert calls == [[97, 98, 99]]
    assert request.status == (
        RequestStatus.FINISHED_CANCELLED
        if abort
        else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    assert len(output.outputs) == 1
    assert output.outputs[0].finish_reason == ("abort" if abort else "length")
    assert output.outputs[0].finished
    assert output.finished_request_ids == {"zero"}
    assert not scheduler.waiting and not scheduler.running
    assert scheduler._step_request_id is None
    assert scheduler._admitting_request_id is None
    assert not scheduler._pending_abort_ids
    scheduler.batch_generator.insert.assert_not_called()
    scheduler.batch_generator.next.assert_not_called()
    assert scheduler.step().outputs == []

    following = Request("following", "xyz", SamplingParams(max_tokens=1))
    scheduler.add_request(following)
    assert scheduler._schedule_waiting() == [following]
    assert not scheduler.waiting
    assert scheduler.running == {"following": following}
    scheduler.batch_generator.insert.assert_called_once()
    assert len(calls) == 1
    responses, finished = scheduler._process_batch_responses(
        [SimpleNamespace(uid=1, token=90, logprobs=None, finish_reason="length")]
    )
    assert len(responses) == 1 and responses[0].finished
    assert finished == {"following"}
    scheduler._cleanup_finished(finished)
    assert not scheduler.running and not scheduler.waiting


def _wire_scores(ids):
    return PromptLogprobs(
        0,
        token_ids=ids,
        logprobs=[None] + [-float(i) for i in range(1, len(ids))],
        top_logprobs=[None] + [{} for _ in ids[1:]],
    )


@pytest.mark.parametrize("middle", [False, True])
def test_added_special_tokens_have_sequence_offsets(middle):
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    ids = [10, 97, 11, 98, 12] if middle else [10, 97, 98, 12]
    tokenizer = Tokenizer()
    tokenizer._tokenizer = lambda text, **kw: {
        "input_ids": ids,
        "offset_mapping": (
            [(0, 0), (0, 1), (0, 0), (1, 2), (0, 0)]
            if middle
            else [(0, 0), (0, 1), (1, 2), (0, 0)]
        ),
        "special_tokens_mask": [1, 0, 1, 0, 1] if middle else [1, 0, 0, 1],
    }
    scores = _wire_scores(ids)
    payload = legacy_prompt_logprobs("ab", tokenizer, scores)
    assert payload["tokens"] == (
        ["", "a", "", "b", ""] if middle else ["", "a", "b", ""]
    )
    assert payload["text_offset"] == ([0, 0, 1, 1, 2] if middle else [0, 0, 1, 2])
    assert payload["token_logprobs"] == scores.logprobs
    assert "".join(payload["tokens"]) == "ab"


def test_no_offsets_long_prompt_uses_bounded_decode_windows():
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    class SlowTokenizer(Tokenizer):
        clean_up_tokenization_spaces = False

        def decode(self, ids, **kwargs):
            decoded_lengths.append(len(ids))
            return super().decode(ids, **kwargs)

    decoded_lengths = []
    prompt = "ab " * 2000
    payload = legacy_prompt_logprobs(
        prompt, SlowTokenizer(), _wire_scores(list(map(ord, prompt)))
    )
    assert "".join(payload["tokens"]) == prompt
    assert payload["text_offset"] == list(range(len(prompt)))
    assert max(decoded_lengths) <= 64
    assert sum(decoded_lengths) <= 64 * (len(prompt) + 1)


@pytest.mark.parametrize("with_factory", [False, True])
def test_concurrent_fallback_offsets_match_serial_results(monkeypatch, with_factory):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from mlx_lm import tokenizer_utils

    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    barrier = None
    instances = []

    class InterleavedDetokenizer(tokenizer_utils.NaiveStreamingDetokenizer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            instances.append(self)

        def add_token(self, token):
            super().add_token(token)
            if barrier is not None:
                barrier.wait(timeout=5)

    monkeypatch.setattr(
        tokenizer_utils, "NaiveStreamingDetokenizer", InterleavedDetokenizer
    )
    tokenizer = Tokenizer()
    tokenizer.detokenizer = InterleavedDetokenizer(tokenizer)
    tokenizer.detokenizer.add_token(ord("!"))
    if with_factory:
        tokenizer._detokenizer_class = InterleavedDetokenizer

    def build(prompt):
        return legacy_prompt_logprobs(
            prompt, tokenizer, _wire_scores(tokenizer.encode(prompt))
        )

    prompts = ["abc😀def", "XYéZ123"]
    serial = [build(prompt) for prompt in prompts]
    for prompt, payload in zip(prompts, serial):
        assert payload["tokens"] == list(prompt)
        assert payload["text_offset"] == list(range(len(prompt)))

    # Force both threads to append before either reads the decoded text.
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(build, prompts))

    assert concurrent == serial
    assert tokenizer.detokenizer.text == "!"
    assert len(instances) == 5  # Shared instance plus four request-local instances.


@pytest.mark.parametrize("optimized", [False, True])
def test_no_offsets_byte_fallback_and_special_tokens(optimized):
    from mlx_lm.tokenizer_utils import (
        NaiveStreamingDetokenizer,
        SPMStreamingDetokenizer,
    )

    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    class ByteTokenizer:
        vocab = {
            "<s>": 0,
            "a": 1,
            "<0xF0>": 2,
            "<0x9F>": 3,
            "<0x98>": 4,
            "<0x80>": 5,
            "b": 6,
            "</s>": 7,
        }
        all_special_ids = [0, 7]
        clean_up_tokenization_spaces = False

        def decode(self, ids):
            raw = [b"<s>", b"a", b"\xf0", b"\x9f", b"\x98", b"\x80", b"b", b"</s>"]
            return b"".join(raw[i] for i in ids).decode("utf-8", "replace")

        _detokenizer_class = (
            SPMStreamingDetokenizer if optimized else NaiveStreamingDetokenizer
        )

    payload = legacy_prompt_logprobs(
        "a😀b", ByteTokenizer(), _wire_scores(list(range(8)))
    )
    assert payload["tokens"] == ["", "a", "", "", "", "😀", "b", ""]
    assert payload["text_offset"] == [0, 0, 1, 1, 1, 1, 2, 3]
    assert "".join(payload["tokens"]) == "a😀b"


@pytest.mark.parametrize("offsets_available", [False, True])
def test_literal_special_token_retains_submitted_text(offsets_available):
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    class SpecialTokenizer(Tokenizer):
        all_special_ids = [10, 12]

        def __call__(self, prompt, **kwargs):
            if not offsets_available and kwargs.get("return_offsets_mapping"):
                raise NotImplementedError
            return {
                "input_ids": [10, 10, 97, 12],
                "offset_mapping": [(0, 0), (0, 3), (3, 4), (0, 0)],
                "special_tokens_mask": [1, 0, 0, 1],
            }

        def decode(self, ids, **kwargs):
            return "".join("<s>" if i == 10 else chr(i) for i in ids)

    payload = legacy_prompt_logprobs(
        "<s>a", SpecialTokenizer(), _wire_scores([10, 10, 97, 12])
    )
    assert payload["tokens"] == ["", "<s>", "a", ""]
    assert payload["text_offset"] == [0, 0, 3, 4]


def test_decoded_top_token_collisions_keep_maximum():
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    tokenizer = Tokenizer()
    tokenizer.decode = lambda ids: "\ufffd"
    tokenizer._tokenizer = lambda text, **kwargs: {
        "input_ids": [97],
        "offset_mapping": [(0, 1)],
    }
    scores = PromptLogprobs(
        2, token_ids=[97], logprobs=[None], top_logprobs=[{1: -0.5, 2: -3.0}]
    )
    assert legacy_prompt_logprobs("a", tokenizer, scores)["top_logprobs"] == [
        {"\ufffd": -0.5}
    ]


@pytest.mark.parametrize("policy", ["fcfs", "shortest_validated_tail"])
@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("budget", [64, 256])
def test_long_scoring_yields_to_decode_and_releases_state(
    monkeypatch, policy, abort, budget
):
    import mlx_lm.models.cache as caches

    from rapid_mlx.request import RequestStatus

    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: [])
    from collections import deque

    chunks = []

    def model(tokens, cache):
        chunks.append(tokens.shape[1])
        return mx.zeros((1, tokens.shape[1], 128))

    scheduler = Scheduler(
        model=model,
        tokenizer=Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            scheduling_policy=policy,
            prefill_step_size=256,
            decode_stall_target_ms=10 if budget == 64 else 0,
        ),
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [1]
    scheduler.batch_generator.next.return_value = ([], [])
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    monkeypatch.setattr(scheduler, "_shortest_tail_admission_capacity", lambda: 8)
    monkeypatch.setattr(scheduler, "_request_is_generator_compatible", lambda r: True)
    active = Request("active", "abc", SamplingParams(max_tokens=100))
    scheduler.add_request(active)
    scheduler._schedule_waiting()
    scheduler.batch_generator.next.return_value = (
        [],
        [SimpleNamespace(uid=1, token=90, logprobs=None, finish_reason=None)],
    )
    request = Request(
        "long",
        "abc" * 400,
        SamplingParams(max_tokens=0, prompt_logprobs=PromptLogprobs(2)),
    )
    scheduler.add_request(request)
    scheduler.batch_generator._generation_batch = [object()]
    scheduler.batch_generator._currently_processing = []
    scheduler.batch_generator._unprocessed_sequences = []
    scheduler._prefill_cost_samples = deque([0.001], maxlen=4)
    output = scheduler.step()
    assert len(chunks) > 0 and sum(chunks) == budget
    assert output.outputs[0].new_token_ids == [90]
    assert not request.is_finished()
    assert request._prompt_scoring_state is not None
    if abort:
        assert scheduler.abort_request("long")
        previous = list(chunks)
        scheduler.step()
        assert chunks == previous
        assert request.status == RequestStatus.FINISHED_CANCELLED
    else:
        finished = []
        for _ in range(20):
            output = scheduler.step()
            finished.extend(o for o in output.outputs if o.request_id == "long")
            if request.is_finished():
                break
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert len(finished) == 1 and finished[0].finished
        assert sum(chunks) == len(request.prompt_token_ids) - 1
        assert len(request.sampling_params.prompt_logprobs.logprobs) == 1200
        assert all(o.request_id != "long" for o in scheduler.step().outputs)
    assert request._prompt_scoring_state is None
    assert request.prompt_cache is None
    assert request not in scheduler.waiting


@pytest.mark.parametrize("policy", ["fcfs", "shortest_validated_tail"])
@pytest.mark.parametrize("max_tokens", [0, 1])
def test_abort_during_scoring_chunk_prevents_next_forward(
    monkeypatch, policy, max_tokens
):
    import mlx_lm.models.cache as caches

    from rapid_mlx.request import RequestStatus

    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: [])
    calls = []

    def model(tokens, cache):
        calls.append(tokens.shape[1])
        assert scheduler.abort_request("cancel")
        return mx.zeros((1, tokens.shape[1], 128))

    scheduler = Scheduler(
        model=model,
        tokenizer=Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            scheduling_policy=policy,
            prefill_step_size=512,
        ),
    )
    scheduler.batch_generator = MagicMock()
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    request = Request(
        "cancel",
        "abc" * 400,
        SamplingParams(max_tokens=max_tokens, prompt_logprobs=PromptLogprobs(2)),
    )
    scheduler.add_request(request)
    scheduler.step()
    assert calls == [128]
    assert request.status == RequestStatus.FINISHED_CANCELLED
    assert request._prompt_scoring_state is None
    assert request.prompt_cache is None
    assert not scheduler.waiting and not scheduler.running
    scheduler.step()
    assert calls == [128]
    scheduler.batch_generator.insert.assert_not_called()


@pytest.mark.parametrize(
    "prompt,decoded",
    [
        ("a  b c", ["a", "a b", "a b c"]),
        ("éb", ["e", "e\u0301", "e\u0301b"]),
        ("é  ﬁ c", ["e\u0301", "e\u0301 fi", "e\u0301 fi c"]),
        ("a\u0301 b", ["a", "a ", "a b"]),
    ],
)
def test_normalizing_slow_tokenizer_offsets_rejected(prompt, decoded):
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    class NormalizingTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return decoded[len(ids) - 1] if ids else ""

    scores = _wire_scores([1, 2, 3])
    with pytest.raises(ValueError, match="cannot align prompt token offsets exactly"):
        legacy_prompt_logprobs(prompt, NormalizingTokenizer(), scores)


def test_unsupported_prompt_scoring_emits_caller_telemetry(patched_config, monkeypatch):
    from rapid_mlx.telemetry import inference

    emit = MagicMock()
    monkeypatch.setattr(inference, "emit_capability_rejected", emit)
    client, engine = _build_completions_app(patched_config, monkeypatch)
    engine.supports_prompt_logprobs = False
    response = client.post(
        "/v1/completions",
        headers={"User-Agent": "codex", "X-Rapid-Client": "cli"},
        json={
            "model": "stub-model",
            "prompt": "abc",
            "echo": True,
            "logprobs": 0,
            "max_tokens": 0,
        },
    )
    assert response.status_code == 400
    emit.assert_called_once()
    assert emit.call_args.args == ("logprobs_unsupported",)
    assert emit.call_args.kwargs["caller_agent"] == "codex"
    assert emit.call_args.kwargs["caller_client"] == "cli"


@pytest.mark.parametrize("top_k", [0, 20])
def test_scoring_then_decode_reuses_private_kv(monkeypatch, top_k):
    import mlx_lm.models.cache as caches

    private_cache = [SimpleNamespace(state=mx.array(0))]
    monkeypatch.setattr(caches, "make_prompt_cache", lambda model: private_cache)
    scheduler = Scheduler(
        model=lambda tokens, cache: mx.zeros((1, tokens.shape[1], 4)),
        tokenizer=Tokenizer(),
        config=SchedulerConfig(enable_prefix_cache=False, use_memory_aware_cache=False),
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [1]
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    result = PromptLogprobs(top_k)
    request = Request(
        "decode", [1, 2, 3], SamplingParams(max_tokens=1, prompt_logprobs=result)
    )
    scheduler.add_request(request)
    assert scheduler._schedule_waiting() == [request]
    assert request._prompt_scoring_state is None
    inserted = scheduler.batch_generator.insert.call_args
    assert inserted.kwargs["caches"] == [private_cache]
    assert inserted.args == ([[3]],)
    assert result.logprobs == [
        None,
        pytest.approx(-1.38629436),
        pytest.approx(-1.38629436),
    ]
    assert all(len(top) == min(top_k, 4) for top in result.top_logprobs[1:])


def test_malformed_offsets_fall_back_to_decoded_alignment():
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    class MalformedOffsets(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [97, 98], "offset_mapping": [(0, 1)]}

    payload = legacy_prompt_logprobs("ab", MalformedOffsets(), _wire_scores([97, 98]))
    assert payload["tokens"] == ["a", "b"]
    assert payload["text_offset"] == [0, 1]


def test_unalignable_decoder_rejected():
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    tokenizer = Tokenizer()
    tokenizer.decode = lambda ids: "x" * len(ids)
    with pytest.raises(ValueError, match="cannot align prompt token offsets exactly"):
        legacy_prompt_logprobs("abcd", tokenizer, _wire_scores([1, 2]))


def test_base_engine_does_not_claim_prompt_scoring():
    from rapid_mlx.engine.base import BaseEngine

    assert BaseEngine.supports_prompt_logprobs.fget(object()) is False


def test_unsupported_completion_without_scoring_stays_404():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.routes.completions import install_unsupported_completions

    app = FastAPI()
    install_unsupported_completions(app)
    response = TestClient(app).post(
        "/v1/completions", json={"model": "stub", "prompt": "a"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_stream_preserves_keepalives(monkeypatch):
    from rapid_mlx.routes import completions

    async def stream(*args, **kwargs):
        yield ": keepalive\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(completions, "stream_completion", stream)
    events = [
        event
        async for event in completions._stream_completion_prompts(
            None, ["a", "b"], None
        )
    ]
    assert events == [": keepalive\n\n", ": keepalive\n\n", "data: [DONE]\n\n"]


def test_scoring_never_snapshots_shared_prefixes(monkeypatch):
    scheduler = Scheduler(
        model=object(),
        tokenizer=Tokenizer(),
        config=SchedulerConfig(enable_prefix_cache=False, use_memory_aware_cache=False),
    )
    request = Request(
        "private", [1, 2, 3], SamplingParams(prompt_logprobs=PromptLogprobs(0))
    )
    scheduler.add_request(request)
    request.prefix_boundary = 2
    scheduler.uid_to_request_id[1] = request.request_id
    scheduler.memory_aware_cache = MagicMock()
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {1: ([object()], [1, 2])}
    monkeypatch.setattr(scheduler, "_hybrid_checkpoints_enabled", lambda: True)
    response = SimpleNamespace(
        uid=1, progress=(2, 3), end_of_segment=True, end_of_prompt=False
    )
    scheduler._record_hybrid_checkpoints([response])
    scheduler.batch_generator.extract_cache.assert_not_called()
    scheduler._snapshot_boundary_segments([response])
    scheduler.batch_generator.extract_cache.assert_called_once_with([1])
    scheduler.memory_aware_cache.store.assert_not_called()


def test_generation_error_discards_private_scoring_state():
    from rapid_mlx.request import RequestStatus

    scheduler = Scheduler(
        model=object(),
        tokenizer=Tokenizer(),
        config=SchedulerConfig(enable_prefix_cache=False, use_memory_aware_cache=False),
    )
    request = Request(
        "error", [1, 2], SamplingParams(prompt_logprobs=PromptLogprobs(0))
    )
    scheduler.add_request(request)
    scheduler.waiting.remove(request)
    scheduler.running[request.request_id] = request
    request._prompt_scoring_state = object()
    request.prompt_cache = [object()]
    assert scheduler._recover_from_generation_error() == {"error"}
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request._prompt_scoring_state is None
    assert request.prompt_cache is None


@pytest.mark.parametrize("stream", [False, True])
def test_echo_without_scoring_keeps_text(patched_config, monkeypatch, stream):
    import json

    async def generate(**kwargs):
        return GenerationOutput(
            text="Z",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="length",
        )

    async def stream_generate(**kwargs):
        yield GenerationOutput(
            text="Z",
            new_text="Z",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="length",
        )

    client, engine = _build_completions_app(patched_config, monkeypatch)
    engine.generate = generate
    engine.stream_generate = stream_generate
    response = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "a",
            "echo": True,
            "stream": stream,
            "max_tokens": 1,
        },
    )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert "".join(c["text"] for event in events for c in event["choices"]) == "aZ"
        assert response.text.count("data: [DONE]") == 1
    else:
        assert response.json()["choices"][0]["text"] == "aZ"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 1])
@pytest.mark.parametrize("prompt", ["a  b", "a\u0301 b"])
def test_unalignable_prompt_rejected_before_generation(
    patched_config, monkeypatch, stream, as_list, max_tokens, prompt
):
    from rapid_mlx.telemetry import inference

    emit = MagicMock()
    monkeypatch.setattr(inference, "emit_capability_rejected", emit)
    client, engine = _build_completions_app(patched_config, monkeypatch)
    engine.supports_prompt_logprobs = True
    engine.tokenizer = Tokenizer()
    engine.tokenizer.decode = lambda ids: (
        "".join(chr(i) for i in ids).replace("\u0301", "").replace("  ", " ")
    )
    response = client.post(
        "/v1/completions",
        headers={"User-Agent": "codex", "X-Rapid-Client": "cli"},
        json={
            "model": "stub-model",
            "prompt": ["faithful", prompt] if as_list else prompt,
            "echo": True,
            "logprobs": 0,
            "max_tokens": max_tokens,
            "stream": stream,
        },
    )
    assert response.status_code == 400, response.text
    assert "cannot align prompt token offsets exactly" in response.json()["detail"]
    assert response.headers["content-type"] == "application/json"
    engine.stream_generate.assert_not_called()
    engine.generate.assert_not_called()
    emit.assert_called_once()
    assert emit.call_args.args == ("logprobs_unsupported",)
    assert emit.call_args.kwargs["caller_agent"] == "codex"
    assert emit.call_args.kwargs["caller_client"] == "cli"


@pytest.mark.parametrize(
    "prompt,offsets,expected",
    [
        ("a  b", [(0, 1), (1, 3), (3, 4)], ["a", "  ", "b"]),
        ("a\u0301 b", [(0, 2), (2, 3), (3, 4)], ["a\u0301", " ", "b"]),
    ],
)
def test_normalizing_tokenizer_uses_exact_source_mapping(prompt, offsets, expected):
    from rapid_mlx.prompt_logprobs import legacy_prompt_logprobs

    tokenizer = Tokenizer()
    tokenizer.decode = lambda ids: "a b"
    tokenizer._tokenizer = lambda text, **kwargs: {
        "input_ids": [1, 2, 3],
        "offset_mapping": offsets,
    }
    scores = _wire_scores([1, 2, 3])
    payload = legacy_prompt_logprobs(prompt, tokenizer, scores)
    assert payload["tokens"] == expected
    assert payload["text_offset"] == [0, offsets[0][1], offsets[1][1]]
    assert payload["token_logprobs"] == scores.logprobs
