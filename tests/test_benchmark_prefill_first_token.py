"""Contracts for the isolated first-token prefill research harness."""

import json
import time

import pytest

from scripts.benchmark_prefill_first_token import (
    first_token_chunk,
    receipt,
    segment_tail_chunk,
    validate_cold,
    validate_overlap,
)


def test_single_prompt_keeps_configured_size_but_admitted_peer_arms_bound():
    processing = [[[[0] * 16000], 0, 16000]]
    assert first_token_chunk(2048, processing, 0, 8, [0.001], 500) == 2048
    assert first_token_chunk(2048, processing, 1, 8, [0.001], 500) == 224
    assert first_token_chunk(2048, processing, 1, 1, [0.001], 500) == 2048


def test_every_row_visits_checkpoint_grid_during_contention():
    processing = [[[[0]], 2008, 16000], [[[0]], 100, 16000]]
    visited = [set(), set()]
    for _ in range(200):
        chunk = first_token_chunk(2048, processing, 0, 8, [0.0036], 500)
        assert 0 < chunk <= 2048
        for row, seen in zip(processing, visited):
            row[1] += chunk
            seen.add(row[1])
    for seen in visited:
        assert {2048, 4096, 6144} <= seen


def test_existing_memory_bound_is_never_enlarged():
    processing = [[[[0]], 0, 16000], [[[0]], 0, 16000]]
    assert first_token_chunk(32, processing, 0, 8, [0.1], 500) == 32
    assert first_token_chunk(2048, processing, 0, 8, [], 500) == 128
    assert first_token_chunk(2048, processing, 0, 8, [0.001], 0) == 2048


def event(**kw):
    return "data: " + json.dumps(kw)


def stream(*, done=True, finish="stop", text="42", cached=0):
    rows = [
        event(choices=[{"delta": {"role": "assistant"}}]),
        event(choices=[{"delta": {"content": text}, "finish_reason": finish}]),
        event(
            usage={
                "prompt_tokens": 300,
                "completion_tokens": 2,
                "total_tokens": 302,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
        ),
    ]
    if done:
        rows.append("data: [DONE]")
    return rows


def test_stream_requires_terminal_output_usage_and_correct_answer():
    row = receipt(stream(), time.perf_counter_ns())
    validate_cold(row, "42")
    for changes in ({"done": False}, {"finish": None}, {"text": ""}):
        with pytest.raises(ValueError, match="incomplete"):
            receipt(stream(**changes), time.perf_counter_ns())
    with pytest.raises(ValueError, match="cache reuse"):
        validate_cold(receipt(stream(cached=256), time.perf_counter_ns()), "42")
    with pytest.raises(ValueError, match="incorrect exact answer"):
        validate_cold(receipt(stream(text="43"), time.perf_counter_ns()), "42")
    with pytest.raises(ValueError, match="incorrect exact answer"):
        validate_cold(receipt(stream(finish="length"), time.perf_counter_ns()), "42")


def test_promotion_boundary_stays_protected_until_first_generation_step():
    processing = [[[[0] * 10000], 4096, 16000], [[[9]], 283, 284]]
    assert first_token_chunk(2048, processing, 0, 8, [0.001], 500) == 224
    # Once the short row is promoted, normal decode protection owns the bound.
    assert first_token_chunk(192, processing[:1], 0, 8, [0.001], 500) == 192


def test_two_new_prompts_are_bounded_before_entering_processing_batch():
    assert first_token_chunk(2048, [], 2, 8, [0.001], 500) == 224
    assert first_token_chunk(2048, [], 2, 1, [0.001], 500) == 2048


def test_stream_errors_and_missing_usage_cannot_pass_as_results():
    with pytest.raises(ValueError, match="bad request"):
        receipt([event(error={"message": "bad request"})], time.perf_counter_ns())
    with pytest.raises(ValueError, match="invalid usage"):
        receipt(
            [
                event(choices=[{"delta": {"content": "42"}, "finish_reason": "stop"}]),
                "data: [DONE]",
            ],
            time.perf_counter_ns(),
        )


def test_short_segment_tail_reduces_padding_without_growing_cost_bound():
    long = [[[0] * 10000], 2048, 16000]
    short = [[[0] * 55, [0] * 14, [0]], 224, 294]
    assert segment_tail_chunk(224, [long, short], [], 8) == 55
    assert segment_tail_chunk(32, [long, short], [], 8) == 32
    short[0] = [[0] * 14, [0]]
    assert segment_tail_chunk(160, [long, short], [], 8) == 14
    short[0] = [[0]]
    assert segment_tail_chunk(160, [long, short], [], 8) == 160
    assert segment_tail_chunk(2048, [long], [], 8) == 2048


def test_only_admissible_queued_segments_limit_prompt_chunk():
    long = [[[0] * 10000], 2048, 16000]
    queued = [(7, [[0] * 27, [0]], 32, None)]
    assert segment_tail_chunk(224, [long], queued, 8) == 27
    assert segment_tail_chunk(224, [long], queued, 1) == 224
    assert segment_tail_chunk(224, [], queued, 8) == 224


def test_invalid_cache_usage_fails_and_server_zero_omission_is_accepted():
    row = receipt(stream(), time.perf_counter_ns())
    for value in (None, False, "0", -1):
        row["usage"]["prompt_tokens_details"]["cached_tokens"] = value
        with pytest.raises(ValueError, match="cache reuse"):
            validate_cold(row, "42")
    row["usage"]["prompt_tokens_details"] = {}
    with pytest.raises(ValueError, match="cache reuse"):
        validate_cold(row, "42")
    del row["usage"]["prompt_tokens_details"]
    validate_cold(row, "42")


def test_short_arrival_after_long_prefill_cannot_pass_as_contention():
    long = dict(started_ns=10, first_ns=30)
    validate_overlap(long, dict(started_ns=20))
    for arrival in (9, 30, 40):
        with pytest.raises(ValueError, match="before long first token"):
            validate_overlap(long, dict(started_ns=arrival))
