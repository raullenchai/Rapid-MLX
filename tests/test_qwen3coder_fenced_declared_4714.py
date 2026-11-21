# SPDX-License-Identifier: Apache-2.0
"""Declared XML calls inside Markdown are examples, including at stream EOF."""

import json

import pytest

from rapid_mlx.config import get_config
from rapid_mlx.service import helpers
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from tests.qwen3coder_stream_harness import deltas, non_stream, stream
from tests.test_postprocessor import _make_cfg, _make_output
from tests.test_tool_choice_enforcement import _make_client, _RecordingEngine

CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
    "</parameter>\n</function>\n</tool_call>"
)
BARE = CALL.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
REQUEST = {
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
}
MULTI_TOOL_REQUEST = {
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "run",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            },
        },
        REQUEST["tools"][0],
    ],
}
EXAMPLE = f"Here is the format:\n```xml\n{CALL}\n```\n"

JSON_CALL = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13])
@pytest.mark.parametrize(
    "example",
    [
        f"Use `{CALL}`.\n",
        f"Use ``{BARE}``.\n",
        "> ~~~xml\n"
        + "\n".join("> " + line for line in CALL.splitlines())
        + "\n> ~~~\n",
    ],
)
@pytest.mark.parametrize("real_call", [False, True])
def test_quoted_declared_examples_remain_content(example, real_call, size):
    text = example + (CALL.replace("Paris", "Rome") if real_call else "")
    calls, content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert calls == ([("get_weather", '{"city": "Rome"}')] if real_call else [])
    assert content == example


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13])
@pytest.mark.parametrize(
    "example", [f"Use `{JSON_CALL}`.\n", f"> ~~~json\n> {JSON_CALL}\n> ~~~\n"]
)
@pytest.mark.parametrize("real_call", [False, True])
def test_quoted_json_examples_stay_out_of_cross_format_fallback(
    example, real_call, size
):
    text = example + (JSON_CALL.replace("Paris", "Rome") if real_call else "")
    calls, content = stream(text, REQUEST, size)
    assert [(name, json.loads(args)) for name, args in calls] == (
        [("get_weather", {"city": "Rome"})] if real_call else []
    )
    # Cross-format recovery historically leaves the real wire in content.
    assert content == text


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13])
def test_stray_backtick_before_call_with_backticks_in_argument(size):
    prefix = "I will check ` shortly. "
    text = prefix + CALL.replace("Paris", "`Paris`")
    calls, content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert calls == [("get_weather", '{"city": "`Paris`"}')]
    assert content == prefix


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13])
@pytest.mark.parametrize("quote", ["`", "``", "```"])
def test_real_calls_on_both_sides_of_inline_example(size, quote):
    example = f"\nUse {quote}{CALL}{quote}.\n"
    text = CALL.replace("Paris", '"Rome"') + example + CALL.replace("Paris", '"Oslo"')
    calls, content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert calls == [
        ("get_weather", '{"city": "Rome"}'),
        ("get_weather", '{"city": "Oslo"}'),
    ]
    # Chunked streaming suppresses the lead-in before the example's marker
    # triggers deferral. Finalization must not bring that lead-in back.
    assert content == (example if size in (None, 0) else CALL + quote + ".\n")


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13])
def test_deferred_example_does_not_repeat_pre_call_content(size):
    prefix = "Checking the weather.\n"
    example = f"\nUse `{CALL}`.\n"
    text = (
        prefix
        + CALL.replace("Paris", '"Rome"')
        + example
        + CALL.replace("Paris", '"Oslo"')
    )
    assert stream(text, REQUEST, size) == (
        [
            ("get_weather", '{"city": "Rome"}'),
            ("get_weather", '{"city": "Oslo"}'),
        ],
        prefix + (example if size == 0 else CALL + "`.\n"),
    )


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13])
@pytest.mark.parametrize("prefix", ["", "Checking the weather.\n"])
def test_deferred_example_does_not_restore_suppressed_post_tool_text(size, prefix):
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    call = CALL.replace("Paris", '"Rome"')
    suppressed = "Ordinary post-tool text must stay suppressed.\n"
    example = f"Use `{CALL}`.\n"
    pp = StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser="qwen3_coder_xml"),
        request=REQUEST,
    )
    pp.reset()
    events = pp.process_chunk(_make_output(prefix)) if prefix else []
    for piece in deltas(call, size):
        events.extend(pp.process_chunk(_make_output(piece)))
    assert any(event.type == "tool_call" for event in events)
    # This separate delta takes the ordinary post-tool suppression branch.
    assert pp.process_chunk(_make_output(suppressed)) == []
    events.extend(pp.process_chunk(_make_output(example, finished=True)))
    events.extend(pp.finalize())
    content = "".join(event.content or "" for event in events)
    assert suppressed not in content
    assert content == prefix + example
    # The held example itself matches non-stream extraction. Full-response
    # content parity is intentionally unavailable here: non-stream extraction
    # preserves the ordinary post-tool text that the adapter suppresses.
    assert content == non_stream(prefix + example, REQUEST)[1]
    assert non_stream(prefix + call + suppressed + example, REQUEST) == (
        [("get_weather", '{"city": "Rome"}')],
        prefix + suppressed + example,
    )


@pytest.mark.parametrize(
    "wire,offset",
    [
        (wire, offset)
        for wire, marker in [(CALL, "<tool_call>"), (BARE, "<function=")]
        for offset in range(len(marker) + 1)
    ],
)
@pytest.mark.parametrize("quote", ["`", "``", "```"])
@pytest.mark.parametrize("real_call", [False, True])
def test_inline_marker_split_at_every_offset(wire, offset, quote, real_call):
    example = f"Use {quote}{wire}{quote}.\n"
    text = example + (CALL.replace("Paris", '"Rome"') if real_call else "")
    split = len(f"Use {quote}") + offset
    marker = "<tool_call>" if wire.startswith("<tool_call>") else "<function="
    marker_end = len(f"Use {quote}") + len(marker)
    wire_end = len(f"Use {quote}") + len(wire)
    parser = Qwen3CoderToolParser(None)
    results = []
    previous = ""
    # Drive the parser directly: the postprocessor's marker buffer can join
    # partial markers and conceal a missed Markdown candidate in the parser.
    # Finish the example's body before its closing backticks arrive, so an
    # incorrectly admitted opener has time to emit an irreversible call.
    for piece in [
        text[:split],
        text[split:marker_end],
        text[marker_end:wire_end],
        text[wire_end:],
    ]:
        current = previous + piece
        results.append(
            parser.extract_tool_calls_streaming(
                previous, current, piece, request=REQUEST
            )
        )
        previous = current
    results.append(parser.finalize_legacy_raw_stream(text, request=REQUEST))
    calls = [
        call for result in results if result for call in result.get("tool_calls", [])
    ]
    assert [
        (call["function"]["name"], call["function"]["arguments"]) for call in calls
    ] == ([("get_weather", '{"city": "Rome"}')] if real_call else [])
    assert "".join(result.get("content", "") for result in results if result) == example


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13])
@pytest.mark.parametrize("marker", ["~~~", "```"])
@pytest.mark.parametrize("close", [True, False])
def test_blockquote_fence_ends_before_outside_call(size, marker, close):
    example = f"> {marker}xml\n> {CALL.replace(chr(10), chr(10) + '> ')}\n"
    if close:
        example += f"> {marker}\n"
    text = example + CALL.replace("Paris", "Rome")
    calls, content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert calls == [("get_weather", '{"city": "Rome"}')]
    assert content == example


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13])
def test_inline_cross_format_text_call_example_followed_by_real_call(size):
    example = '`[Calling tool: get_weather({"city":"Paris"})]`\n'
    text = example + '[Calling tool: get_weather({"city":"Rome"})]'
    calls, _ = stream(text, REQUEST, size)
    assert [(name, json.loads(args)) for name, args in calls] == [
        ("get_weather", {"city": "Rome"})
    ]


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13])
def test_inline_delimiter_in_genuine_call_arguments_is_opaque(size):
    code = f"`{JSON_CALL}`\n> ~~~xml\n> example\n> ~~~"
    run = f"<tool_call><function=run><parameter=code>{json.dumps(code)}</parameter></function></tool_call>"
    text = run + CALL
    calls, content = (
        non_stream(text, MULTI_TOOL_REQUEST)
        if size is None
        else stream(text, MULTI_TOOL_REQUEST, size)
    )
    assert calls == [
        ("run", json.dumps({"code": code})),
        ("get_weather", '{"city": "Paris"}'),
    ]
    # Wrapper-looking parameter data remains unchanged in the tool arguments.


@pytest.mark.parametrize("size", [0, 1, 7])
@pytest.mark.parametrize("wire", ["wrapped", "raw", "calling"])
def test_cross_format_call_arguments_keep_inline_delimiters(size, wire):
    args = {"city": "`Paris`"}
    payload = json.dumps({"name": "get_weather", "arguments": args})
    text = {
        "wrapped": f"<tool_call>{payload}</tool_call>",
        "raw": payload,
        "calling": f"[Calling tool: get_weather({json.dumps(args)})]",
    }[wire]
    calls, _ = stream(text, REQUEST, size)
    assert [(name, json.loads(arguments)) for name, arguments in calls] == [
        ("get_weather", args)
    ]


@pytest.mark.parametrize("size", [None, 0, 1, 7])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_inline_span_across_line_endings(size, newline):
    example = f"Use `{newline}{CALL.replace(chr(10), newline)}{newline}`."
    result = (
        non_stream(example, REQUEST) if size is None else stream(example, REQUEST, size)
    )
    assert result == ([], example)


@pytest.mark.parametrize("size", [None, 0, 1, 7])
def test_incomplete_inline_call_example_stays_content(size):
    example = "Use `<function=get_weather><parameter=city>Paris`."
    result = (
        non_stream(example, REQUEST) if size is None else stream(example, REQUEST, size)
    )
    assert result == ([], example)


def test_formatted_prose_stream_does_not_rescan_markdown():
    from unittest.mock import patch

    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _MarkdownCodeTracker

    text = "ordinary `code` prose. " * 500
    parser = Qwen3CoderToolParser(None)
    content = []
    with patch.object(
        _MarkdownCodeTracker, "code_ranges", wraps=_MarkdownCodeTracker.code_ranges
    ) as scan:
        for start in range(0, len(text), 16):
            result = parser.extract_tool_calls_streaming(
                text[:start],
                text[: start + 16],
                text[start : start + 16],
                request=REQUEST,
            )
            assert result is not None and not result.get("tool_calls")
            content.append(result.get("content", ""))
        assert scan.call_count == 0
    assert "".join(content) == text


@pytest.mark.parametrize("size", [0, 1, 7, 13])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("parallel", [True, False])
def test_deferred_calls_obey_tool_choice_and_parallel_cap(size, choice, parallel):
    request = {
        **REQUEST,
        "tool_choice": (
            {"type": "function", "function": {"name": "get_weather"}}
            if choice == "named"
            else choice
        ),
        "parallel_tool_calls": parallel,
    }
    example = f"Example: `{CALL}`\n"
    text = example + CALL.replace("Paris", "Rome") + CALL.replace("Paris", "Oslo")
    calls, content = stream(text, request, size)
    expected = [("get_weather", '{"city": "Rome"}')]
    if parallel and choice != "named":
        expected.append(("get_weather", '{"city": "Oslo"}'))
    assert calls == expected
    assert content == example


@pytest.mark.parametrize("size", [0, 1, 7, 13])
@pytest.mark.parametrize("city", ["Rome", '"Rome"'])
def test_forced_legacy_eof_preserves_argument_fragments(size, city):
    request = {
        **REQUEST,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    calls, content = stream(CALL.replace("Paris", city), request, size)
    assert calls == [("get_weather", '{"city": "Rome"}')]
    assert content == ""


@pytest.mark.parametrize("size", [0, 1, 7, 13])
def test_deferred_forced_choice_keeps_already_admitted_call(size):
    request = {
        **REQUEST,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    text = (
        CALL.replace("Paris", '"Rome"')
        + f"\nExample: `{CALL}`\n"
        + CALL.replace("Paris", '"Oslo"')
    )
    calls, _ = stream(text, request, size)
    assert calls == [("get_weather", '{"city": "Rome"}')]


@pytest.mark.parametrize("genuine_deferred", [True, False])
def test_deferred_finalize_does_not_parse_prefix_on_live_state(
    genuine_deferred, monkeypatch
):
    request = {
        **REQUEST,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    unknown = CALL.replace("get_weather", "unavailable")
    genuine = CALL.replace("Paris", '"Rome"')
    prefix = (unknown if genuine_deferred else genuine) + "\nExample: `"
    example = CALL + "`\n"
    text = prefix + example + (genuine if genuine_deferred else unknown)
    parser = Qwen3CoderToolParser(None)
    # Drive the real streaming transitions: the first block is either dropped
    # as undeclared or emitted; the inline example then holds the remaining text.
    previous = ""
    for piece in deltas(prefix, 7):
        current = previous + piece
        parser.extract_tool_calls_streaming(previous, current, piece, request=request)
        previous = current
    held = prefix + "<tool_call>"
    assert (
        parser.extract_tool_calls_streaming(
            prefix, held, "<tool_call>", request=request
        )
        is None
    )
    assert parser._markdown_held_start == len(prefix)
    assert (
        parser.extract_tool_calls_streaming(
            held, text, text[len(held) :], request=request
        )
        is None
    )
    assert parser._undeclared_dropped == int(genuine_deferred)

    live_inputs = []
    extract = parser.extract_tool_calls

    def observe_live_parse(model_output, request=None):
        live_inputs.append(model_output)
        if model_output == text:
            assert vars(parser) == expected_state
        return extract(model_output, request=request)

    monkeypatch.setattr(parser, "extract_tool_calls", observe_live_parse)
    expected_state = {**vars(parser), "_markdown_held_start": None}
    result = parser.finalize_legacy_raw_stream(text, request=request)
    # A speculative prefix parse must never mutate the response's parser.
    assert live_inputs == [text]
    assert result is not None
    assert [call["function"] for call in result["tool_calls"]] == (
        [{"name": "get_weather", "arguments": '{"city": "Rome"}'}]
        if genuine_deferred
        else []
    )
    assert [call["index"] for call in result["tool_calls"]] == (
        [0] if genuine_deferred else []
    )
    assert result["content"] == example + ("" if genuine_deferred else unknown)


def test_first_visible_opener_without_markup():
    parser = Qwen3CoderToolParser(None)
    assert parser._first_opener_pos("") == 0
    assert parser._first_opener_pos("plain") == len("plain")


def test_fence_tracker_handles_current_line_and_out_of_order_queries():
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _MarkdownCodeTracker

    text = "```xml <function=get_weather>\nafter"
    tracker = _MarkdownCodeTracker()
    assert tracker.inside_fence(text, len(text))
    assert tracker.inside_fence(text, text.index("<function="))


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "  ",
        "text ",
        "```",
        "``` ",
        "```xml\n",
        "<tool_call>\n",
        "```xml\nx\n```\n<tool_call>\n",
    ],
)
@pytest.mark.parametrize(
    "span",
    [
        "",
        " ",
        "  ",
        "x",
        "`",
        "\t",
        "\n",
        "x\n",
        "x\n  ",
        "x\n    ",
        "x\n`",
        "x\n\t",
        "x\ny",
        "x\ny\n",
        BARE,
        CALL,
        CALL + "\n",
    ],
)
def test_skipped_fence_span_matches_plain_text_line_state(prefix, span):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import _MarkdownCodeTracker

    expected = _MarkdownCodeTracker()
    expected._feed_fence(prefix)
    for char in span:
        if char == "\n":
            expected._fence_line_phase = "indent"
            expected._fence_line_indent = 0
            expected._fence_line_marker = None
            expected._fence_line_run = 0
            expected._fence_line_tail_nonspace = False
        elif expected._fence_line_phase == "indent":
            if char == " " and expected._fence_line_indent < 3:
                expected._fence_line_indent += 1
            else:
                expected._fence_line_phase = "invalid"
        elif expected._fence_line_phase == "run":
            expected._fence_line_phase = "tail"
            expected._fence_line_tail_nonspace = not char.isspace()
        elif expected._fence_line_phase == "tail" and not char.isspace():
            expected._fence_line_tail_nonspace = True

    actual = _MarkdownCodeTracker()
    text = prefix + span
    actual.skip_fence_span(text, len(prefix), len(text))

    assert actual._fence_state() == expected._fence_state()
    assert actual._fence_committed == len(text)


@pytest.mark.parametrize("size", [None, 1, 2, 3, 5, 7, 13, 0])
@pytest.mark.parametrize(
    "first, fence",
    [
        (CALL, "```xml\n"),
        (BARE, "```\n"),
    ],
    ids=["wrapped-close-same-line", "bare-close-same-line"],
)
def test_call_close_then_same_line_backticks_do_not_hide_later_call(first, fence, size):
    text = first + fence + CALL.replace("Paris", "Rome")
    got_calls, _ = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [
        ("get_weather", '{"city": "Paris"}'),
        ("get_weather", '{"city": "Rome"}'),
    ]


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13, 10000])
@pytest.mark.parametrize(
    "text, content, calls",
    [
        (EXAMPLE, EXAMPLE, 0),
        (f"```xml\n{CALL}\n```", f"```xml\n{CALL}\n```", 0),
        (f"Example:\n~~~xml\n{CALL}\n~~~\n", f"Example:\n~~~xml\n{CALL}\n~~~\n", 0),
        (f"Example:\n```xml\n{BARE}\n```\n", f"Example:\n```xml\n{BARE}\n```\n", 0),
        (f"Example:\n```xml\n{CALL}", f"Example:\n```xml\n{CALL}", 0),
        (
            "Example:\n```xml\n<tool_call>\n<function=get_weather>",
            "Example:\n```xml\n<tool_call>\n<function=get_weather>",
            0,
        ),
        (f"```xml\nx\n```\n{CALL}", "```xml\nx\n```\n", 1),
        (EXAMPLE + CALL, EXAMPLE, 1),
        (EXAMPLE + CALL + CALL, EXAMPLE, 2),
        (CALL, "", 1),
    ],
)
def test_markdown_boundary(text, content, calls, size):
    got_calls, got_content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [("get_weather", '{"city": "Paris"}')] * calls
    assert got_content == content


@pytest.mark.parametrize("size", [None, 1, 2, 3, 5, 7, 13, 0])
@pytest.mark.parametrize(
    "prefix",
    [
        "```lang`bad\nordinary\n",
        f"~~~lang`ok\n{CALL}\n~~~\n",
    ],
    ids=["backtick-in-backtick-info", "backtick-in-tilde-info"],
)
def test_fence_info_backtick_rules(prefix, size):
    real_call = CALL.replace("Paris", "Rome")
    text = prefix + real_call
    got_calls, got_content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [("get_weather", '{"city": "Rome"}')]
    assert got_content == prefix


def test_cross_format_fence_mask_uses_backtick_info_rule():
    mask = Qwen3CoderToolParser.mask_markdown_fenced_regions
    invalid = "```lang`bad\n[Calling tool: get_weather({})]\n"
    tilde = "~~~lang`ok\n[Calling tool: get_weather({})]\n~~~\n"
    assert mask(invalid) == invalid
    assert "[Calling tool:" not in mask(tilde)


def test_service_does_not_reextract_example(monkeypatch):
    class Request:
        tools = REQUEST["tools"]

        def model_dump(self):
            return REQUEST

    cfg = get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    monkeypatch.setattr(
        helpers,
        "parse_tool_calls",
        lambda *a, **kw: pytest.fail("generic fallback re-extracted the example"),
    )
    content, calls = helpers._parse_tool_calls_with_parser(EXAMPLE, Request())
    assert content == EXAMPLE
    assert not calls


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13, 10000])
def test_unclosed_parameter_in_fence_does_not_consume_later_call(size):
    example = (
        "```xml\n<tool_call>\n<function=get_weather>\n<parameter=city>Paris\n```\n"
    )
    real_call = CALL.replace("Paris", "Rome")
    text = example + real_call
    got_calls, got_content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [("get_weather", '{"city": "Rome"}')]
    assert got_content == example


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13, 10000])
def test_cross_format_recovery_ignores_only_fenced_span(size):
    text = EXAMPLE + '[Calling tool: get_weather({"city":"Rome"})]'
    got_calls, _ = stream(text, REQUEST, size)
    assert got_calls == [("get_weather", '{"city": "Rome"}')]


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13, 10000])
def test_stray_backtick_does_not_hide_declared_call(size):
    prefix = "I will check ` shortly. "
    real_call = CALL.replace("Paris", "Rome")
    text = prefix + real_call
    got_calls, got_content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [("get_weather", '{"city": "Rome"}')]
    assert got_content == prefix


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13, 10000])
@pytest.mark.parametrize(
    "run_call, code",
    [
        (
            "<function=run><parameter=code>\n```python\nprint(1)\n"
            "</parameter></function>",
            "```python\nprint(1)",
        ),
        (
            "<tool_call>\n<function=run><parameter=code>\n```python\nprint(1)\n"
            "</parameter></function>\n</tool_call>",
            "```python\nprint(1)",
        ),
        (
            "<function=run><parameter=code>\n```python\nprint(1)\n```\n"
            "</parameter></function>",
            "```python\nprint(1)\n```",
        ),
    ],
    ids=["bare-unclosed", "wrapped-unclosed", "bare-closed"],
)
def test_fence_inside_call_does_not_hide_following_call(run_call, code, size):
    text = run_call + CALL
    got_calls, got_content = (
        non_stream(text, MULTI_TOOL_REQUEST)
        if size is None
        else stream(text, MULTI_TOOL_REQUEST, size)
    )
    assert got_calls == [
        ("run", json.dumps({"code": code})),
        ("get_weather", '{"city": "Paris"}'),
    ]
    assert got_content == ""


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 0])
@pytest.mark.parametrize(
    "text",
    [
        f"Sure. {CALL.replace('Paris', 'Rome')}\n{EXAMPLE}",
        f"Sure. {BARE.replace('Paris', 'Rome')}\n{EXAMPLE}",
        f"Sure. {CALL.replace('Paris', 'Rome')} trailing text\n{EXAMPLE}",
        f"{CALL.replace('Paris', 'Rome')}\n```xml\n{CALL}\n```\n",
        f"{BARE.replace('Paris', 'Rome')}\n```xml\n{CALL}\n```\n",
    ],
    ids=[
        "wrapped-after-prose",
        "bare-after-prose",
        "wrapped-close-before-trailing-text",
        "wrapped-before-fence",
        "bare-before-fence",
    ],
)
def test_call_newlines_keep_subsequent_fenced_example_out_of_tool_calls(text, size):
    got_calls, _ = stream(text, REQUEST, size)
    assert got_calls == [("get_weather", '{"city": "Rome"}')]


@pytest.mark.parametrize("text", ["<funx", "<fun"])
def test_partial_bare_opener_is_released_as_content(text):
    assert stream(text, MULTI_TOOL_REQUEST, 1) == ([], text)


def test_fenced_opener_token_id_stays_content():
    parser = Qwen3CoderToolParser(None)
    parser.tool_call_start_token_id = 42
    text = f"```xml\n{CALL}\n```"
    result = parser.extract_tool_calls_streaming(
        "", text, text, delta_token_ids=[42], request=REQUEST
    )
    assert result == {"content": text}


def test_streaming_advance_waits_for_next_function_body():
    parser = Qwen3CoderToolParser(None)
    text = CALL + CALL
    parser._stream_text = text
    parser.is_tool_call_started = True
    parser.json_closed = True
    parser.current_tool_index = 0
    assert (
        parser._streaming_step_once("", text, text, [], REQUEST, {"get_weather"})
        is None
    )
    assert parser.current_tool_index == 1


@pytest.mark.parametrize("real_call", [False, True])
@pytest.mark.parametrize(
    "example",
    [
        EXAMPLE,
        f"Use `{CALL}`.",
        "> ~~~xml\n" + "\n".join("> " + line for line in CALL.splitlines()) + "\n> ~~~",
    ],
)
def test_chat_completions_preserves_example(real_call, example, monkeypatch):
    # The route strips outer whitespace from all assistant responses.
    example = example.rstrip()
    client = _make_client(
        _RecordingEngine(text=example + ("\n" + CALL if real_call else "")),
        tool_call_parser="qwen3_coder_xml",
    )
    monkeypatch.setattr(get_config(), "enable_auto_tool_choice", True)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Show the XML format."}],
            "enable_thinking": False,
            **REQUEST,
        },
    )
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert message["content"] == example
    assert len(message.get("tool_calls") or []) == int(real_call)
