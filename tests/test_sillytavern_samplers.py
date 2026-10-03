# SPDX-License-Identifier: Apache-2.0
"""DRY sampler, repetition-penalty range and explicit sampler refusals.

The DRY core, the schema contract and the route helper run without MLX; the
scheduler / multimodal plumbing tests are ``requires_mlx`` (Apple lane).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rapid_mlx.api import sampler_compat as sc
from rapid_mlx.api.models import ChatCompletionRequest, CompletionRequest
from rapid_mlx.sampling_dry import (
    DEFAULT_SEQUENCE_BREAKERS,
    DRYLogitsProcessor,
    _max_useful_length,
    breaker_token_ids,
    dry_penalties,
)
from rapid_mlx.service.helpers import (
    build_extended_sampling_kwargs,
    dry_sampling_kwargs,
)

NO_BREAKERS: frozenset[int] = frozenset()


def _dry(tokens, **kw):
    params = {
        "multiplier": 1.0,
        "base": 2.0,
        "allowed_length": 2,
        "breakers": NO_BREAKERS,
    }
    params.update(kw)
    return dry_penalties(tokens, **params)


# ----------------------------------------------------------------- DRY core


def test_dry_penalizes_extending_a_verbatim_repeat():
    # "1 2 3 4 ... 1 2 3" -> continuing with 4 would repeat a 3-token run.
    assert _dry([1, 2, 3, 4, 9, 1, 2, 3]) == {4: 2.0 ** (3 - 2)}


def test_dry_ignores_repeats_shorter_than_allowed_length():
    assert _dry([1, 2, 9, 1], allowed_length=2) == {}
    assert _dry([1, 2, 9, 1], allowed_length=1) == {2: 1.0}


def test_dry_keeps_the_longest_match_per_candidate():
    tokens = [5, 1, 2, 7, 3, 1, 2, 7, 4, 3, 1, 2]
    # "3 1 2" precedes 7 (length 3); "1 2" also precedes 7 (length 2).
    assert _dry(tokens) == {7: 2.0}


def test_dry_sequence_breakers_end_matches():
    tokens = [1, 2, 3, 4, 9, 1, 2, 3]
    assert _dry(tokens, breakers=frozenset({3})) == {}  # last token breaks
    assert _dry(tokens, breakers=frozenset({4})) == {}  # candidate breaks
    # A breaker inside the run cuts the match short: only "2 3" repeats.
    assert _dry(tokens, breakers=frozenset({1})) == {4: 1.0}


def test_dry_window_and_off_switch():
    tokens = [1, 2, 3, 4, 9, 1, 2, 3]
    assert _dry(tokens, multiplier=0) == {}
    assert _dry(tokens, penalty_last_n=4) == {}
    assert _dry([7]) == {}


def test_dry_exponent_follows_the_reference_beyond_any_fixed_length():
    tokens = [1] * 400
    # base 1.01 needs a long match before the penalty saturates: the full
    # reference exponent is used, not a fixed cap.
    penalties = _dry(tokens, base=1.01, multiplier=1.0)
    assert penalties[1] == pytest.approx(1.01 ** (399 - 2), rel=1e-6)


def test_dry_scan_stops_once_the_penalty_saturates():
    assert _max_useful_length(1.0, 1.0, 2) == 2  # base 1: constant penalty
    assert _max_useful_length(2e4, 1.75, 3) == 3  # already saturated
    cap = _max_useful_length(1.0, 2.0, 2)
    assert 2.0 ** (cap - 2) >= 1e4 > 2.0 ** (cap - 3)
    # A repeat longer than the cap gets the cap's (already huge) penalty.
    assert _dry([1] * 100, base=2.0)[1] == pytest.approx(2.0 ** (cap - 2))
    assert _dry([1] * 100, base=1.0)[1] == pytest.approx(1.0)


def test_dry_penalty_never_overflows():
    assert _dry([1] * 3000, base=8.0, multiplier=9.0)[1] < float("inf")


def test_breaker_token_ids_take_the_token_a_breaker_ends_with():
    class _Fast:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) for c in text]

    class _Plain:
        def encode(self, text):
            return [ord(c) for c in text]

    class _Merging:
        def encode(self, text, add_special_tokens=False):
            return [7] if text == "a:" else ([9] if text == ":" else [])

    # "\n" -> last token of "a\n"; "ab" -> its final token only.
    assert breaker_token_ids(_Fast(), ["\n", "ab"]) == frozenset({10, 98})
    assert breaker_token_ids(_Plain(), [":"]) == frozenset({58})
    assert breaker_token_ids(_Merging(), [":", "?"]) == frozenset({9})


class _Req:
    def __init__(self, prompt, output):
        self.prompt_token_ids = prompt
        self.output_token_ids = output


def _settings(**kw):
    params = {
        "multiplier": 1.0,
        "base": 2.0,
        "allowed_length": 2,
        "breakers": NO_BREAKERS,
    }
    params.update(kw)
    return DRYLogitsProcessor(**params)


def test_bound_history_is_prompt_plus_committed_output():
    request = _Req([1, 2, 3, 4], [9, 1, 2])
    bound = _settings().bind(request)
    assert bound.history() == [1, 2, 3, 4, 9, 1, 2]
    request.output_token_ids.append(3)  # live view of committed output
    assert bound.settings.penalties(bound.history()) == {4: 2.0}
    assert _settings(penalty_last_n=2).bind(request).history() == [2, 3]
    assert _settings(penalty_last_n=6).bind(request).history() == [3, 4, 9, 1, 2, 3]
    assert _settings().bind(_Req(None, None)).history() == []


def test_bound_processor_without_penalties_returns_logits_untouched():
    bound = _settings().bind(_Req([1, 2, 3], []))
    sentinel = object()
    assert bound([5, 6], sentinel) is sentinel


@pytest.mark.requires_mlx
def test_processor_subtracts_penalties_on_mlx_logits():
    import mlx.core as mx

    bound = _settings().bind(_Req([1, 2, 3, 4, 6], [1, 2, 3]))
    logits = mx.zeros((1, 8))
    # The KV-cache view passed in is ignored (a prefix-cache hit shrinks it).
    out = bound(mx.array([3]), logits)
    assert out[0, 4].item() == pytest.approx(-2.0)
    assert out[0, 5].item() == 0.0


# ------------------------------------------------------------ schema contract


def _chat(**fields):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **fields)


def test_neutral_unsupported_samplers_are_accepted():
    neutral = {key: values[0] for key, values in sc.UNSUPPORTED_SAMPLERS.items()}
    neutral["top_n_sigma"] = -1
    neutral["typical_p"] = None
    _chat(**neutral)
    CompletionRequest(prompt="x", **neutral)


@pytest.mark.parametrize(
    "field,value",
    [
        ("xtc_probability", 0.5),
        ("dynamic_temperature", True),
        ("mirostat", 1),
        ("tfs", "0.9"),
        ("typical_p", float("nan")),
    ],
)
def test_enabled_unsupported_samplers_are_refused(field, value):
    with pytest.raises(ValidationError, match=field):
        _chat(**{field: value})
    with pytest.raises(ValidationError, match=field):
        CompletionRequest(prompt="x", **{field: value})


def test_dynamic_temperature_false_is_neutral_but_zero_is_not_a_bool():
    _chat(dynamic_temperature=False)
    with pytest.raises(ValidationError):
        _chat(dynamic_temperature=0)


def test_rep_pen_range_alias_and_precedence():
    assert _chat(rep_pen_range=512).repetition_penalty_range == 512
    request = _chat(rep_pen_range=512, repetition_penalty_range=64)
    assert request.repetition_penalty_range == 64
    assert sc.apply_sampler_compat("not-a-dict") == "not-a-dict"


def test_dry_fields_and_bounds():
    request = _chat(
        dry_multiplier=0.8,
        dry_base=1.75,
        dry_allowed_length=2,
        dry_penalty_last_n=-1,
        dry_sequence_breakers=["\n", ":"],
    )
    assert request.dry_sequence_breakers == ["\n", ":"]
    for bad in (
        {"dry_multiplier": -1},
        {"dry_base": 0.5},
        {"dry_allowed_length": 0},
        {"dry_multiplier": float("inf")},
    ):
        with pytest.raises(ValidationError):
            _chat(**bad)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (["\n"], ["\n"]),
        (json.dumps(["\n", "*"]), ["\n", "*"]),
    ],
)
def test_sequence_breakers_accept_lists_and_json(value, expected):
    assert sc.parse_sequence_breakers(value) == expected


@pytest.mark.parametrize(
    "value",
    ["not json", json.dumps({"a": 1}), [1], [""], ["x" * 33], ["a"] * 65],
)
def test_sequence_breakers_reject_bad_shapes(value):
    with pytest.raises(ValueError):
        sc.parse_sequence_breakers(value)


# ------------------------------------------------------------- route helpers


def test_build_extended_sampling_kwargs_forwards_the_range():
    request = SimpleNamespace(repetition_penalty_range=0)
    assert build_extended_sampling_kwargs(request)["repetition_context_size"] == 0
    assert "repetition_context_size" not in build_extended_sampling_kwargs(
        SimpleNamespace(repetition_penalty_range=True)
    )


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_dry_sampling_kwargs_builds_the_processor():
    assert dry_sampling_kwargs(None, SimpleNamespace(dry_multiplier=0)) == {}
    assert dry_sampling_kwargs(None, SimpleNamespace()) == {}
    engine = SimpleNamespace(tokenizer=_Tokenizer())
    request = SimpleNamespace(
        dry_multiplier=0.8,
        dry_base=None,
        dry_allowed_length=None,
        dry_penalty_last_n=-1,
        dry_sequence_breakers=None,
    )
    processor = dry_sampling_kwargs(engine, request)["dry_logits_processor"]
    assert processor.base == 1.75 and processor.allowed_length == 2
    assert processor.penalty_last_n == 0
    assert processor.breakers == frozenset(
        ord(c) for c in "".join(DEFAULT_SEQUENCE_BREAKERS)
    )
    request.dry_base, request.dry_allowed_length = 2.0, 3
    request.dry_penalty_last_n, request.dry_sequence_breakers = 256, ["#"]
    processor = dry_sampling_kwargs(engine, request)["dry_logits_processor"]
    assert (processor.base, processor.allowed_length) == (2.0, 3)
    assert processor.penalty_last_n == 256
    assert processor.breakers == frozenset({ord("#")})


def test_dry_sampling_kwargs_refuses_the_multimodal_lane():
    with pytest.raises(HTTPException, match="multimodal"):
        dry_sampling_kwargs(
            SimpleNamespace(is_mllm=True, tokenizer=_Tokenizer()),
            SimpleNamespace(dry_multiplier=1.0),
        )


def test_dry_sampling_kwargs_survives_an_unusable_tokenizer():
    request = SimpleNamespace(dry_multiplier=1.0)
    processor = dry_sampling_kwargs(None, request)["dry_logits_processor"]
    assert processor.breakers == frozenset()


# ------------------------------------------------- lanes that cannot run DRY


def test_dspark_refuses_dry():
    from rapid_mlx.spec_decode.dspark.server import _validate_greedy_request

    request = _chat(temperature=0, dry_multiplier=0.8)
    with pytest.raises(HTTPException, match="dry_multiplier"):
        _validate_greedy_request(request)


def test_native_mtp_refuses_dry():
    from rapid_mlx.speculative.native_mtp.server import (
        _validate_greedy_request,
    )

    with pytest.raises(HTTPException, match="dry_multiplier"):
        _validate_greedy_request(SimpleNamespace(temperature=0, dry_multiplier=1.0))


def test_tensorfold_refuses_dry():
    from rapid_mlx.speculative.tensorfold_qwen27_server import (
        validate_http_request,
    )

    request = SimpleNamespace(
        top_k=None,
        min_p=None,
        seed=None,
        stop=None,
        messages=[SimpleNamespace(content="hi")],
        tools=None,
        response_format=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        dry_multiplier=0.5,
    )
    with pytest.raises(HTTPException):
        validate_http_request(request)


@pytest.mark.requires_mlx
def test_ddtree_refuses_dry():
    from rapid_mlx.speculative.ddtree.server import _validate_request

    with pytest.raises(HTTPException, match="DRY"):
        _validate_request(_chat(dry_multiplier=0.5))


@pytest.mark.requires_mlx
def test_deepseek_v41_refuses_dry_and_range():
    from rapid_mlx.models.deepseek_v41_native.serving import validate_request

    fields = dict.fromkeys(
        (
            "stop",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "logit_bias",
            "video_fps",
            "video_max_frames",
            "reasoning_max_tokens",
            "reasoning_effort",
            "seed",
        )
    )
    request = SimpleNamespace(
        **fields,
        repetition_penalty_range=64,
        dry_multiplier=0.5,
        chat_template_kwargs=None,
    )
    with pytest.raises(HTTPException, match="repetition_penalty_range, dry_multiplier"):
        validate_request(request)


# --------------------------------------------------------- serving plumbing


@pytest.mark.requires_mlx
def test_scheduler_admits_range_and_dry_on_plain_and_mtp_paths():
    from unittest.mock import MagicMock

    from rapid_mlx.request import Request, SamplingParams
    from tests.test_prompt_cache_snapshot import _make_scheduler_with_cache

    scheduler = _make_scheduler_with_cache()
    scheduler.config.hybrid_cache_entries = 8
    scheduler.config.non_trimmable_exact_prefix_reuse = True
    dry = DRYLogitsProcessor(
        multiplier=1.0, base=2.0, allowed_length=2, breakers=NO_BREAKERS
    )
    request = Request(
        request_id="req-st",
        prompt="ignored",
        prompt_token_ids=[10, 20, 30, 40],
        sampling_params=SamplingParams(
            max_tokens=4, repetition_penalty=1.2, repetition_context_size=0
        ),
    )
    request.dry_logits_processor = dry
    request.prefix_boundary = 99
    scheduler.waiting.append(request)
    batch_generator = MagicMock()
    batch_generator.insert_segments.return_value = [104]
    scheduler.batch_generator = batch_generator
    scheduler._ensure_batch_generator = MagicMock(return_value=True)
    scheduler._get_request_sampler = MagicMock(return_value=MagicMock())
    scheduler._register_uid_processors = MagicMock()

    assert scheduler._schedule_waiting() == [request]

    admitted = batch_generator.insert_segments.call_args.kwargs["logits_processors"][0]
    bound = admitted[-1]
    assert bound.settings is dry and bound.request is request
    # No MTP draft transaction: DRY stays out of the MTP-safe tuple, so the
    # handoff fails closed and the request decodes without MTP.
    assert bound not in request._mtp_safe_logits_processors
    assert tuple(admitted[:-1]) == request._mtp_safe_logits_processors

    import mlx.core as mx

    # The repetition processor sees the whole context (range 0): token 10 is
    # the first prompt token, outside mlx-lm's default 20-token window only
    # for long prompts, so check the window directly via the closure.
    logits = mx.ones((1, 64))
    out = admitted[0](mx.array(list(range(30))), logits)
    assert out[0, 0].item() == pytest.approx(1 / 1.2)


@pytest.mark.requires_mlx
def test_mllm_lane_honours_the_repetition_window():
    import mlx.core as mx

    from rapid_mlx.mllm_batch_generator import (
        MLLMBatchRequest,
        _maybe_apply_penalty_processors,
    )

    req = MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="hi",
        repetition_penalty=2.0,
        repetition_context_size=0,
    )
    req.output_tokens = list(range(30))
    out = _maybe_apply_penalty_processors(req, mx.ones((1, 64)))
    assert out[0, 0].item() == pytest.approx(0.5)
    narrow = MLLMBatchRequest(
        uid=1, request_id="r1", prompt="hi", repetition_penalty=2.0
    )
    narrow.output_tokens = list(range(30))
    out = _maybe_apply_penalty_processors(narrow, mx.ones((1, 64)))
    assert out[0, 0].item() == 1.0  # outside mlx-lm's default 20-token window


@pytest.mark.requires_mlx
def test_mllm_scheduler_threads_the_window_to_the_batch_request():
    from tests.test_mllm_penalty_passthrough import _stub_scheduler

    scheduler = _stub_scheduler()
    rid = scheduler.add_request(
        prompt="hi", max_tokens=8, repetition_penalty=1.3, repetition_context_size=0
    )
    assert scheduler.requests[rid].sampling_params.repetition_context_size == 0
    assert (
        scheduler.requests[
            scheduler.add_request(prompt="hi", max_tokens=8)
        ].sampling_params.repetition_context_size
        is None
    )
