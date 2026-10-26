"""The internal shadow switch cannot change wire output or request state."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from rapid_mlx.config.server_config import ServerConfig
from rapid_mlx.domain.events import StreamEvent
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.shadow import ShadowComparison, make_shadow
from rapid_mlx.prompt_parse_context import PromptParseContext
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from tests.parser_migration.harness import route


def output(text, channel=None):
    return GenerationOutput(
        text="",
        new_text=text,
        finished=False,
        channel=channel,
        prompt_parse_context=PromptParseContext("", (), "content", False, "chatml"),
    )


def test_switch_defaults_off_and_excludes_unmigrated_contracts(monkeypatch):
    cfg = SimpleNamespace(reasoning_parser_name="qwen3")
    monkeypatch.delenv("RAPID_MLX_PARSER_SHADOW", raising=False)
    assert make_shadow(cfg, False) is None
    monkeypatch.setenv("RAPID_MLX_PARSER_SHADOW", "1")
    assert isinstance(make_shadow(cfg, False), ShadowComparison)
    assert make_shadow(cfg, True) is None
    assert make_shadow(SimpleNamespace(reasoning_parser_name="gemma4"), False) is None


def test_hash_comparison_ignores_partition_and_contains_no_transcript(caplog):
    shadow = ShadowComparison()
    with caplog.at_level("DEBUG", logger="rapid_mlx.parsing.shadow"):
        shadow.observe(
            output("<think>private</think>public"),
            [StreamEvent("reasoning", reasoning="private")],
        )
        shadow.finish([StreamEvent("content", content="public")])
    assert shadow.report["content_equal"] and shadow.report["reasoning_equal"]
    assert "private" not in caplog.text and "public" not in caplog.text
    report = shadow.report.copy()
    shadow.observe(output("late"), [])
    shadow.finish([])
    assert shadow.report == report


def test_shadow_records_real_mismatch_without_correcting_output():
    shadow = ShadowComparison()
    published = [StreamEvent("content", content="changed")]
    shadow.observe(output("original"), published)
    shadow.finish([])
    assert not shadow.report["content_equal"]
    assert published[0].content == "changed"


@pytest.mark.parametrize(
    "native",
    [
        GenerationOutput(text="", channel="content"),
        GenerationOutput(text="", tool_calls=[{"name": "read"}]),
    ],
)
def test_native_paths_are_ineligible_until_adapters_are_migrated(native):
    shadow = ShadowComparison()
    shadow.observe(native, [])
    shadow.observe(output("later"), [])
    shadow.finish([])
    assert shadow.disabled and shadow.report is None


def test_empty_stream():
    shadow = ShadowComparison()
    shadow.finish([])
    assert shadow.report["source_chars"] == 0
    assert shadow.report["content_equal"] and shadow.report["reasoning_equal"]


@pytest.mark.parametrize("method", ["feed", "finish"])
def test_diagnostic_failure_never_changes_published_events(monkeypatch, method):
    shadow = ShadowComparison()
    shadow.observe(output("hello"), [StreamEvent("content", content="hello")])

    def broken(*args):
        raise RuntimeError("private diagnostic details")

    monkeypatch.setattr(shadow.parser, method, broken)
    published = [StreamEvent("content", content="world")]
    if method == "feed":
        shadow.observe(output("world"), published)
    else:
        shadow.finish(published)
    assert shadow.disabled and shadow.report is None
    assert published[0].content == "world"


def process(monkeypatch, enabled):
    monkeypatch.setenv("RAPID_MLX_PARSER_SHADOW", "1" if enabled else "0")
    processor = StreamingPostProcessor(
        ServerConfig(reasoning_parser_name="qwen3"),
        tools_requested=False,
        enable_thinking=False,
    )
    processor.reset()
    events = []
    for text in ["<think>", "private", "</think>", "public"]:
        events.extend(processor.process_chunk(output(text)))
    events.extend(
        processor.process_chunk(
            GenerationOutput(text="", finished=True, finish_reason="stop")
        )
    )
    events.extend(processor.finalize())
    return processor, [asdict(event) for event in events]


def test_processor_shadow_publishes_only_legacy_events_and_reset_is_request_local(
    monkeypatch,
):
    off, old = process(monkeypatch, False)
    on, new = process(monkeypatch, True)
    assert old == new
    assert off._parser_shadow is None
    prior = on._parser_shadow
    assert prior.report is not None
    on.reset()
    assert on._parser_shadow is not prior
    assert on._parser_shadow.report is None
    assert prior.report is not None


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_real_routes_are_unchanged_with_shadow_switch(
    monkeypatch, surface, thinking, stream
):
    args = ("qwen3", "hermes", "<think>Private.</think>Public answer.", surface, stream)
    monkeypatch.setenv("RAPID_MLX_PARSER_SHADOW", "0")
    before = route(*args, thinking=thinking, tools_mode="absent")
    monkeypatch.setenv("RAPID_MLX_PARSER_SHADOW", "1")
    after = route(*args, thinking=thinking, tools_mode="absent")
    assert before == after
    assert "error" not in after


def test_shadow_detects_reordering_even_when_lanes_individually_match():
    shadow = ShadowComparison()
    shadow.observe(
        output("<think>A</think>B<think>C"),
        [StreamEvent("reasoning", reasoning="AC"), StreamEvent("content", content="B")],
    )
    shadow.finish([])
    assert shadow.report["content_equal"] and shadow.report["reasoning_equal"]
    assert not shadow.report["order_equal"]


def test_shadow_order_digest_ignores_chunk_boundaries():
    shadow = ShadowComparison()
    shadow.observe(output("<think>A"), [])
    shadow.observe(
        output("C</think>B"),
        [StreamEvent("reasoning", reasoning="AC"), StreamEvent("content", content="B")],
    )
    shadow.finish([])
    assert shadow.report["order_equal"]
