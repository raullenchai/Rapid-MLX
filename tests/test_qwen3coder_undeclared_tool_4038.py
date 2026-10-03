# SPDX-License-Identifier: Apache-2.0
"""Issue #4038: a call to an undeclared tool must not stream as visible markup.

goose declared 17 tools (no ``read``); the model answered with
``<tool_call>\\n<function=read>…</function></tool_call>``. origin/main kept
the span as content (an undeclared name is never executable), so the user
saw raw wire markup, ``finish_reason`` was ``stop`` and nothing was logged.

Contract after the fix:

* A name the request did not declare is still never executed (origin/main's
  contract, shared by every parser here; OpenAI never emits one).
* A LONE CANONICAL BLOCK -- ``<tool_call>``, one closed ``<function=NAME>``,
  ``</tool_call>``, only whitespace between -- with an undeclared NAME is an
  attempted call: it is removed from the content and a WARNING is logged.
  This applies only when the request declared tools (not ``tool_choice=none``),
  the block is not inside Markdown code (an open ``` / ~~~ fence or an inline
  code span on its line), and every ``<function=`` opener before it belongs
  to an earlier dropped block, so streaming can decide causally and agree
  with non-streaming at any chunk size.
* Everything else keeps origin/main's preserve-as-text behaviour: bare
  ``<function=…>`` spans (prose about the wire format), shared wrappers,
  unclosed or truncated blocks, requests without tools.

The golden file pins origin/main's non-streaming and streaming output for
every case (streaming at delta sizes 1..10000 and as one whole delta); only
the ``changed_by_4038`` cases and the ``stream_overrides`` pinned below
differ.
"""

from __future__ import annotations

import json
import logging
import pathlib

import pytest

from rapid_mlx.config import get_config
from rapid_mlx.service import helpers
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from tests.qwen3coder_stream_harness import deltas, non_stream, stream

GOLDEN = json.loads(
    (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "qwen3coder_undeclared_tool_golden_main.json"
    ).read_text()
)
REQUESTS = GOLDEN["requests"]
SIZES = GOLDEN["sizes"]
CASES = GOLDEN["cases"]
LOGGER = "rapid_mlx.tool_parsers.qwen3coder_tool_parser"


def _shape(result: tuple[list[tuple[str, str]], str]) -> dict:
    calls, content = result
    return {
        "calls": [[name, json.loads(args) if args else args] for name, args in calls],
        "content": content,
    }


def _case(name: str) -> dict:
    return next(c for c in CASES if c["name"] == name)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_non_streaming(case):
    got = _shape(non_stream(case["text"], REQUESTS[case["request"]]))
    if case["changed_by_4038"]:
        assert got == case["expected"]
        assert got["content"].count("<function=") < case["main_non_stream"][
            "content"
        ].count("<function=")
    else:
        assert got == case["main_non_stream"]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_streaming(case, size):
    got = _shape(stream(case["text"], REQUESTS[case["request"]], size))
    if "after_drop_main" in case:
        # The leading block is dropped; the rest streams exactly as
        # origin/main streams that remaining text on its own.
        after = case["after_drop_main"]
        assert case["text"].endswith(after["remainder"])
        assert case["expected"] == after["non_stream"]
        assert got == after["stream"][str(size)]
    elif case["changed_by_4038"]:
        assert got == case["expected"]
    else:
        overrides = case.get("stream_overrides", {})
        assert got == overrides.get(str(size), case["main_stream"][str(size)])


def test_stream_overrides_are_the_documented_ones():
    """Streaming results of unchanged cases that differ from origin/main.

    * shared_wrapper_rejected_sibling at coarse deltas (whole text, 13,
      10000): origin/main leaked the ADMITTED write_file call's markup into
      content as well as emitting the call. Now the content is the rejected
      sibling only; the trailing ``</tool_call>`` that shares a delta with the
      closing call is still dropped, which is origin/main's existing
      coarse-delta behaviour for text after a call (see
      declared_then_lone_undeclared, where main drops it too).
    * two whitespace bytes at delta size 1 that origin/main lost are kept.
    """
    overrides = {
        (case["name"], size): value
        for case in CASES
        for size, value in case.get("stream_overrides", {}).items()
    }
    assert set(overrides) == {
        ("shared_wrapper_rejected_sibling", "0"),
        ("shared_wrapper_rejected_sibling", "13"),
        ("shared_wrapper_rejected_sibling", "10000"),
        ("shared_wrapper_two_undeclared", "1"),
        ("text_inside_wrapper_after_function", "1"),
    }
    for (name, size), value in overrides.items():
        case = _case(name)
        assert value["calls"] == case["main_non_stream"]["calls"]
        assert "write_file" not in value["content"]
        assert len(value["content"]) >= len(case["main_non_stream"]["content"]) - len(
            "</tool_call>"
        )


def test_goose_e3_shape_is_the_issue_transcript():
    case = _case("goose_e3_lone_undeclared")
    assert case["main_non_stream"]["content"] == case["text"]
    assert case["expected"] == {
        "calls": [],
        "content": (
            "Now I'll read the README file to find information about what "
            "the wordutil package does.\n"
        ),
    }


@pytest.mark.parametrize(
    "name, drops",
    [
        ("goose_e3_lone_undeclared", 1),
        ("two_lone_undeclared", 2),
        ("lone_undeclared_then_declared", 1),
        ("declared_then_lone_undeclared", 0),
        ("tool_choice_none_lone_block", 0),
        ("bare_undeclared_zero_arg_prose", 0),
    ],
)
def test_each_drop_is_logged_once(name, drops, caplog):
    case = _case(name)
    request = REQUESTS[case["request"]]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        non_stream(case["text"], request)
    assert _drop_logs(caplog) == drops
    for size in (1, 10_000):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            stream(case["text"], request, size)
        assert _drop_logs(caplog) == drops


def _drop_logs(caplog) -> int:
    return sum(
        "did not declare" in record.getMessage() and "#4038" in record.getMessage()
        for record in caplog.records
    )


def test_log_names_the_tool_not_the_arguments(caplog):
    case = _case("goose_e3_lone_undeclared")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        non_stream(case["text"], REQUESTS[case["request"]])
    (record,) = [r for r in caplog.records if "#4038" in r.getMessage()]
    assert "'read'" in record.getMessage()
    assert "README" not in record.getMessage()


@pytest.mark.parametrize(
    "name", ["goose_e3_lone_undeclared", "truncated_lone_undeclared"]
)
def test_service_non_stream_path(name, monkeypatch):
    """The chat/messages/responses non-stream helper returns the same content
    and never re-promotes the dropped span through the generic parser."""
    case = _case(name)
    request = REQUESTS[case["request"]]

    class Request:
        tools = request["tools"]

        def model_dump(self):
            return request

    cfg = get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    monkeypatch.setattr(
        helpers,
        "parse_tool_calls",
        lambda *args, **kwargs: pytest.fail("generic fallback re-promoted the span"),
    )
    content, calls = helpers._parse_tool_calls_with_parser(case["text"], Request())
    assert calls is None
    expected = case.get("expected", case["main_non_stream"])["content"]
    assert content == expected


def _write_block(value_chars: int) -> str:
    return (
        "Writing.\n<tool_call>\n<function=write>\n<parameter=content>\n"
        + "x" * value_chars
        + "\n</parameter>\n</function>\n</tool_call>"
    )


_FRAMING = len(_write_block(0)) - len("Writing.\n")


@pytest.mark.parametrize(
    "value_chars, dropped",
    [
        (32768 - _FRAMING, True),  # block is exactly the limit
        (32768 - _FRAMING + 1, False),  # one char over
        (40_000, False),  # open block outgrows the limit mid-stream
    ],
)
@pytest.mark.parametrize("size", [997, 4096, 10_000_000])
def test_blocks_over_the_hold_limit_stay_content_in_both_paths(
    value_chars, dropped, size
):
    """Streaming must hold a block until it closes, so only blocks up to
    32768 chars are dropped; longer ones stay content in BOTH paths (well
    under the post-processor's 64 KB suppression budget)."""
    text = _write_block(value_chars)
    request = REQUESTS["tools"]
    expected = ([], "Writing.\n" if dropped else text)
    assert non_stream(text, request) == expected
    assert stream(text, request, size) == expected


def test_many_dropped_blocks_in_one_delta_do_not_recurse(caplog):
    """Batched decoding can deliver many blocks in one delta; the parser
    loops over them instead of recursing."""
    text = "<tool_call>\n<function=bad>\n</function>\n</tool_call>" * 2000 + "ok"
    request = REQUESTS["tools"]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert stream(text, request, 0) == ([], "ok")
    assert _drop_logs(caplog) == 2000


@pytest.mark.parametrize(
    "name",
    [
        "lone_undeclared_in_code_fence",
        "lone_undeclared_in_tilde_fence",
        "lone_undeclared_in_inline_code",
        "lone_undeclared_in_unclosed_fence",
        "short_fence_does_not_close",
    ],
)
def test_blocks_inside_markdown_code_are_documentation(name):
    case = _case(name)
    assert not case["changed_by_4038"]
    assert "<tool_call>" in case["main_non_stream"]["content"]


@pytest.mark.parametrize("size", [1, 7, 10_000])
def test_double_backtick_inline_code_is_preserved(size):
    """Markdown code spans use matching runs, including even-length ones."""
    block = (
        "<tool_call><function=read><parameter=path>README.md</parameter>"
        "</function></tool_call>"
    )
    text = f"Example: ``{block}`` end"
    expected = ([], text)
    assert non_stream(text, REQUESTS["tools"]) == expected
    assert stream(text, REQUESTS["tools"], size) == expected


def _undeclared_block(name: str, value: str = "") -> str:
    parameter = f"<parameter=content>\n{value}\n</parameter>\n" if value else ""
    return f"<tool_call>\n<function={name}>\n{parameter}</function></tool_call>"


@pytest.mark.parametrize("size", [1, 997, 10_000_000])
def test_released_over_limit_block_prevents_later_drop(size):
    """A preserved opener blocks later drops in both parsing paths."""
    text = (
        _undeclared_block("first_bad", "x" * 40_000)
        + "middle"
        + _undeclared_block("second_bad")
        + "end"
    )
    expected = ([], text)
    assert non_stream(text, REQUESTS["tools"]) == expected
    assert stream(text, REQUESTS["tools"], size) == expected


@pytest.mark.parametrize("size", [1, 7, 10_000])
def test_released_non_lone_wrapper_prevents_later_drop(size):
    """A wrapper with residual text is content and blocks later drops."""
    first = (
        "<tool_call>\n<function=first_bad>\n</function>\nresidual text\n</tool_call>"
    )
    text = first + "middle" + _undeclared_block("second_bad") + "end"
    expected = ([], text)
    assert non_stream(text, REQUESTS["tools"]) == expected
    assert stream(text, REQUESTS["tools"], size) == expected


def test_reused_parser_logs_every_response(caplog):
    """Logging state is per response: a parser reused for several
    non-streaming responses logs each one's drops."""
    parser = Qwen3CoderToolParser(None)
    case = _case("goose_e3_lone_undeclared")
    request = REQUESTS[case["request"]]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            parser.extract_tool_calls(case["text"], request)
    assert _drop_logs(caplog) == 3


def test_reused_parser_logs_after_a_stream(caplog):
    """A streamed response's drops are logged once (finalize re-parses the
    same text); a later, different response on the same parser logs again,
    with or without reset()."""
    parser = Qwen3CoderToolParser(None)
    case = _case("goose_e3_lone_undeclared")
    request = REQUESTS[case["request"]]
    text = case["text"]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        previous = ""
        for piece in deltas(text, 5):
            parser.extract_tool_calls_streaming(
                previous, previous + piece, piece, request=request
            )
            previous += piece
        parser.extract_tool_calls(text, request)  # the finalize re-parse
        assert _drop_logs(caplog) == 1
        parser.extract_tool_calls("Other.\n" + text, request)
        assert _drop_logs(caplog) == 2
        parser.reset()
        parser.extract_tool_calls(text, request)
        assert _drop_logs(caplog) == 3


def test_markdown_code_tracker_answers_out_of_order_queries():
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _MarkdownCodeTracker

    text = "```\nx\n```\nafter <tool_call>open"
    tracker = _MarkdownCodeTracker()
    assert tracker.inside(text, len(text)) is False
    assert tracker.inside(text, 4) is True  # earlier position, inside the fence
