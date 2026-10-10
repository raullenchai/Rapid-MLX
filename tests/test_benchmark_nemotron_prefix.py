"""Negative controls for the long-chat cache qualification's verdict."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.benchmark_nemotron_prefix import compare, measure


def artifacts():
    cases = [
        "initial",
        "continue",
        "continue_again",
        "edit",
        "regenerate",
        "branch",
        "different_inventory",
    ]
    warm = dict(
        profile="nemotron",
        revision="a",
        runtime="0.6.6",
        mlx="0.32.3",
        source_commit="b",
        probe_sha256="probe",
        host={"cpu": "test", "ram_bytes": 1},
        cache="on",
        rows=[
            dict(
                case=case,
                request_sha256=case,
                prompt_tokens=20000,
                cached_tokens=19000 if case in cases[1:6] else 0,
                token_ids=[1, 2],
                answer="ok",
            )
            for case in cases
        ],
    )
    cold = copy.deepcopy(warm)
    cold["cache"] = "off"
    for row in cold["rows"]:
        row["cached_tokens"] = 0
    return warm, cold


def test_matching_sequences_with_resumed_prefixes_pass():
    compare(*artifacts())


@pytest.mark.parametrize(
    "key,value",
    [
        ("token_ids", [1, 3]),
        ("request_sha256", "different"),
        ("answer", "other"),
        ("prompt_tokens", 19999),
    ],
)
def test_rejects_different_input_or_output(key, value):
    warm, cold = artifacts()
    warm["rows"][2][key] = value
    with pytest.raises(ValueError, match="mismatch"):
        compare(warm, cold)


@pytest.mark.parametrize("index", range(1, 6))
def test_every_resumed_case_must_hit(index):
    warm, cold = artifacts()
    warm["rows"][index]["cached_tokens"] = 0
    with pytest.raises(ValueError, match="insufficient long-history reuse"):
        compare(warm, cold)


@pytest.mark.parametrize("count", [1, 18999, 20000, 20001, -1])
def test_small_prefix_and_invalid_counts_cannot_qualify(count):
    warm, cold = artifacts()
    warm["rows"][1]["cached_tokens"] = count
    with pytest.raises(ValueError):
        compare(warm, cold)


def test_rejects_contaminated_baseline_missing_cases_and_different_runtime():
    for mutate in (
        lambda a, b: b["rows"][1].update(cached_tokens=1),
        lambda a, b: a["rows"].pop(),
        lambda a, b: b.update(runtime="other"),
        lambda a, b: b.update(probe_sha256="other"),
        lambda a, b: b.update(host={"cpu": "other"}),
        lambda a, b: a.update(cache="off"),
        lambda a, b: a["rows"][0].update(cached_tokens=1),
        lambda a, b: (
            a["rows"][0].update(token_ids=[]),
            b["rows"][0].update(token_ids=[]),
        ),
        lambda a, b: (
            a["rows"][0].update(prompt_tokens=100),
            b["rows"][0].update(prompt_tokens=100),
        ),
    ):
        warm, cold = artifacts()
        mutate(warm, cold)
        with pytest.raises(ValueError):
            compare(warm, cold)


def test_measure_reads_terminal_usage_and_restores_submit_after_failure():
    job = SimpleNamespace(started_at=10, prefilled_at=11)
    scheduler = SimpleNamespace(submit=lambda _: None)
    original = scheduler.submit
    app = SimpleNamespace(
        scheduler=scheduler,
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt"),
    )
    token = SimpleNamespace(new_token_ids=[1], cached_tokens=0)
    terminal = SimpleNamespace(
        finished=True,
        output_token_ids=[1],
        output_text="ok",
        prompt_tokens=20000,
        cached_tokens=19000,
    )
    provider = SimpleNamespace(backend=SimpleNamespace(_app=app), last_outputs=[])

    def outputs(*args, **kwargs):
        scheduler.submit(job)
        yield token
        provider.last_outputs = [token, terminal]

    provider._outputs = outputs
    assert measure(provider, [])["cached_tokens"] == 19000
    assert scheduler.submit is original

    def broken(*args, **kwargs):
        raise RuntimeError("failed")

    provider._outputs = broken
    with pytest.raises(RuntimeError, match="failed"):
        measure(provider, [])
    assert scheduler.submit is original


def test_measure_rejects_incomplete_response():
    scheduler = SimpleNamespace(submit=lambda _: None)
    app = SimpleNamespace(
        scheduler=scheduler,
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt"),
    )
    provider = SimpleNamespace(backend=SimpleNamespace(_app=app), last_outputs=[])

    def outputs(*args, **kwargs):
        scheduler.submit(SimpleNamespace(started_at=10, prefilled_at=11))
        yield SimpleNamespace(new_token_ids=[1])

    provider._outputs = outputs
    with pytest.raises(RuntimeError, match="completed token sequence"):
        measure(provider, [])
