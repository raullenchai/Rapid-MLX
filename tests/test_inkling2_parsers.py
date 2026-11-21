"""Inkling typed blocks retain their channels across arbitrary stream chunks."""

import json
from types import SimpleNamespace

import pytest

from rapid_mlx.reasoning import get_parser
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.utils.decode import IncrementalDecoder

M = "<|message_model|>"
R = "<|content_thinking|>"
T = "<|content_text|>"
C = "<|content_invoke_tool_json|>"
E = "<|end_message|>"
S = "<|content_model_end_sampling|>"
CALL = M + "read_file" + C + '{"name":"read_file","args":{"path":"sample.txt"}}' + E


def test_registered():
    assert get_parser("inkling")
    assert ToolParserManager.get_tool_parser("inkling")


@pytest.mark.parametrize(
    "raw,expected",
    [
        (M + R + "Secret." + E, ("Secret.", None)),
        (M + T + "Answer" + E + S, (None, "Answer")),
        (M + R + "Secret." + E + M + T + "Answer" + E, ("Secret.", "Answer")),
        ("Plain prose.", (None, "Plain prose.")),
        (R + "Unfinished", ("Unfinished", None)),
    ],
)
@pytest.mark.parametrize("width", [1, 7, 31, 1000])
def test_reasoning(raw, expected, width):
    parser = get_parser("inkling")()
    assert parser.extract_reasoning(raw) == expected
    reasoning = content = previous = ""
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        msg = parser.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if msg:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    msg = parser.finish_stream()
    if msg:
        reasoning += msg.reasoning or ""
        content += msg.content or ""
    assert (reasoning or None, content or None) == expected


@pytest.mark.parametrize("width", [1, 7, 31, 1000])
@pytest.mark.parametrize("count", [1, 2])
def test_tools(width, count):
    raw = M + R + "Secret." + E + CALL * count + M + T + "Done" + E + S
    reasoning = get_parser("inkling")()
    tool = ToolParserManager.get_tool_parser("inkling")()
    r, c = reasoning.extract_reasoning(raw)
    assert r == "Secret."
    result = tool.extract_tool_calls(c)
    assert result.content == "Done"
    assert [(v["name"], json.loads(v["arguments"])) for v in result.tool_calls] == [
        ("read_file", {"path": "sample.txt"})
    ] * count
    previous = tool_text = content = secret = ""
    calls = []
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        msg = reasoning.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if not msg:
            continue
        secret += msg.reasoning or ""
        delta = msg.content or ""
        out = tool.extract_tool_calls_streaming(tool_text, tool_text + delta, delta)
        tool_text += delta
        if out:
            content += out.get("content") or ""
            calls.extend(out.get("tool_calls") or [])
    assert secret == "Secret."
    assert content == "Done"
    merged = {}
    for call in calls:
        fn = merged.setdefault(call["index"], {"name": "", "arguments": ""})
        for key, value in call["function"].items():
            fn[key] += value
    assert len(merged) == count
    assert all(
        v["name"] == "read_file"
        and json.loads(v["arguments"]) == {"path": "sample.txt"}
        for v in merged.values()
    )


@pytest.mark.parametrize("width", [1, 7, 31, 1000])
@pytest.mark.parametrize("through_reasoning", [False, True])
@pytest.mark.parametrize(
    "following,secret,count",
    [
        (T + "Done", "", 1),
        (R + "Secret" + T + "Done", "Secret", 1),
        (M + "header" + T + "Done", "", 1),
        (
            C + '{"name":"read_file","args":{"path":"sample.txt"}}' + E + T + "Done",
            "",
            2,
        ),
        (E + T + "Done", "", 1),
        (S + T + "Done", "", 1),
        ("<|content_invoke_tool_text|>" + "Done", "", 1),
        ("<|content_tool_error|>" + "Done", "", 1),
    ],
)
def test_new_block_closes_tool(width, through_reasoning, following, secret, count):
    from tests.parser_migration.harness import norm

    raw = CALL[: -len(E)] + following
    reasoning = get_parser("inkling")() if through_reasoning else None
    tool = ToolParserManager.get_tool_parser("inkling")()
    if reasoning:
        r, c = reasoning.extract_reasoning(raw)
        assert (r or "") == secret
    else:
        c = raw
    complete = tool.extract_tool_calls(c)
    expected_calls = [["read_file", {"path": "sample.txt"}]] * count
    assert complete.content == "Done"
    assert [
        [v["name"], json.loads(v["arguments"])] for v in complete.tool_calls
    ] == expected_calls

    tool_text = streamed_reasoning = ""
    packets = []

    def consume(msg):
        nonlocal tool_text, streamed_reasoning
        if not msg:
            return
        streamed_reasoning += msg.reasoning or ""
        delta = msg.content or ""
        out = tool.extract_tool_calls_streaming(tool_text, tool_text + delta, delta)
        tool_text += delta
        if out:
            packets.append({"choices": [{"delta": out}]})

    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        consume(
            reasoning.extract_reasoning_streaming(raw[:pos], raw[: pos + width], chunk)
            if reasoning
            else SimpleNamespace(reasoning=None, content=chunk)
        )
    if reasoning:
        consume(reasoning.finish_stream())
        assert reasoning.tool is False
    # The new block closes and validates the call before stream finalization.
    assert tool.payload is None
    assert tool.finalize_legacy_raw_stream(tool_text) is None
    assert streamed_reasoning == (secret if through_reasoning else "")
    assert norm(packets, "chat", True) == {
        "reasoning": "",
        "content": "Done",
        "calls": expected_calls,
    }


@pytest.mark.parametrize(
    "payload", ["oops", '{"name":"f","args":', "[]", '{"args":{}}', '{"name":""}']
)
def test_malformed(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + C + payload + E + M + T + "Answer" + E
    out = parser.extract_tool_calls(raw)
    assert not out.tools_called
    assert out.content == payload + "Answer"


@pytest.mark.parametrize("key", ["args", "arguments", "parameters"])
@pytest.mark.parametrize(
    "args",
    ["not-json", '{"x":', "{} trailing", "[]", "null", "1", "true", '"text"', []],
)
@pytest.mark.parametrize("width", [1, 7, 1000])
@pytest.mark.parametrize("close", [E + T + "Answer", ""])
def test_invalid_arguments_fall_back_to_text(key, args, width, close):
    parser = ToolParserManager.get_tool_parser("inkling")()
    payload = json.dumps({"name": "f", key: args})
    raw = M + C + payload + close
    expected = payload + ("Answer" if close else "")
    complete = parser.extract_tool_calls(raw)
    assert not complete.tools_called
    assert complete.tool_calls == []
    assert complete.content == expected

    content = ""
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        out = parser.extract_tool_calls_streaming(raw[:pos], raw[:pos] + chunk, chunk)
        if out:
            assert not out.get("tool_calls")
            content += out.get("content", "")
    out = parser.finalize_legacy_raw_stream(raw)
    if out:
        assert not out.get("tool_calls")
        content += out.get("content", "")
    assert content == expected


@pytest.mark.parametrize("key", ["args", "arguments", "parameters"])
@pytest.mark.parametrize("args", ["{}", '{"text":"é", "nested":{"x":[1,true,null]}}'])
@pytest.mark.parametrize("width", [1, 7, 1000])
def test_json_object_string_arguments(key, args, width):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + C + json.dumps({"name": "f", key: args}) + E
    complete = parser.extract_tool_calls(raw)
    assert complete.tools_called
    assert complete.content is None
    assert len(complete.tool_calls) == 1
    assert json.loads(complete.tool_calls[0]["arguments"]) == json.loads(args)

    calls = []
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        out = parser.extract_tool_calls_streaming(raw[:pos], raw[:pos] + chunk, chunk)
        if out:
            assert not out.get("content")
            calls.extend(out.get("tool_calls", []))
    assert parser.finalize_legacy_raw_stream(raw) is None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == json.loads(args)


@pytest.mark.parametrize("width", [1, 3, 7, 1000])
@pytest.mark.parametrize("marker", [M, R, T, C, E, S])
def test_marker_inside_tool_argument(marker, width):
    # Escaped quotes and odd/even backslash runs must survive delta boundaries.
    text = 'before "' + marker + "\\" + marker + '\\"' + marker + " after"
    payload = json.dumps({"name": "f", "args": {"text": text}})
    raw = M + C + payload + E + M + T + "Done" + E
    from tests.parser_migration.harness import norm

    reasoning = get_parser("inkling")()
    parser = ToolParserManager.get_tool_parser("inkling")()
    _, content = reasoning.extract_reasoning(raw)
    complete = parser.extract_tool_calls(content)
    assert complete.content == "Done"
    assert json.loads(complete.tool_calls[0]["arguments"]) == {"text": text}
    assert len(complete.tool_calls) == 1
    previous = tool_text = ""
    packets = []
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        msg = reasoning.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if msg and msg.content:
            delta = msg.content
            out = parser.extract_tool_calls_streaming(
                tool_text, tool_text + delta, delta
            )
            tool_text += delta
            if out:
                packets.append({"choices": [{"delta": out}]})
    tail = reasoning.finish_stream()
    if tail and tail.content:
        out = parser.extract_tool_calls_streaming(
            tool_text, tool_text + tail.content, tail.content
        )
        tool_text += tail.content
        if out:
            packets.append({"choices": [{"delta": out}]})
    out = parser.finalize_legacy_raw_stream(tool_text)
    if out:
        packets.append({"choices": [{"delta": out}]})
    assert norm(packets, "chat", True) == {
        "reasoning": "",
        "content": "Done",
        "calls": [["f", {"text": text}]],
    }


def test_partial_delimiter_and_reset():
    parser = get_parser("inkling")()
    assert (
        parser.extract_reasoning_streaming("", "<|content_think", "<|content_think")
        is None
    )
    assert parser.finish_stream().content == "<|content_think"
    parser.reset_state()
    assert parser.extract_reasoning_streaming("", R + "x", R + "x").reasoning == "x"


def test_special_tokens_reach_parser():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens=True):
            assert skip_special_tokens is False
            return "".join([M, R, "Secret", E, M, T, "Answer", E][i] for i in ids)

    decoder = IncrementalDecoder(Tokenizer())
    text = "".join(decoder.add_token(i) for i in range(8))
    assert get_parser("inkling")().extract_reasoning(text) == ("Secret", "Answer")


@pytest.mark.parametrize("thinking", [None, False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_chat_token_stream_reasoning_parity(thinking, width):
    """Typed specials must reach reasoning parsing even after auto-disable."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes.chat import router
    from tests.parser_migration.harness import norm

    pieces = [M, R, "The user wants ", "17 times 23.", E, M, T, "391", E, S]
    token_ids = list(range(len(pieces)))

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens=True):
            return "".join(
                pieces[i]
                for i in ids
                if not skip_special_tokens or not pieces[i].startswith("<|")
            )

    class Engine:
        tokenizer = Tokenizer()
        preserve_native_tool_format = True
        is_mllm = False
        supports_guided_generation = False

        def build_prompt(self, *args, **kwargs):
            return "prompt"

        async def chat(self, **kwargs):
            raw = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            return GenerationOutput(
                text=raw,
                raw_text=raw,
                finished=True,
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=len(token_ids),
            )

        async def stream_chat(self, **kwargs):
            # Same incremental decoder used by the scheduler; default HF
            # decoding above deliberately strips all typed special tokens.
            decoder = IncrementalDecoder(self.tokenizer)
            for start in range(0, len(token_ids), width):
                ids = token_ids[start : start + width]
                delta = "".join(decoder.add_token(i) for i in ids)
                yield GenerationOutput(
                    text=decoder.get_full_text(),
                    new_text=delta,
                    tokens=ids,
                    finished=False,
                    prompt_tokens=1,
                    completion_tokens=start + len(ids),
                )
            yield GenerationOutput(
                text=decoder.get_full_text(),
                finished=True,
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=len(token_ids),
            )

    cfg = reset_config()
    cfg.model_name = "inkling-2-mini-4bit"
    cfg.reasoning_parser_name = "inkling"
    cfg.reasoning_parser = get_parser("inkling")()
    cfg.tool_call_parser = "inkling"
    cfg.enable_auto_tool_choice = True
    cfg.engine = Engine()
    app = FastAPI()
    app.include_router(router)
    body = {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": "What is 17 times 23?"}],
        "temperature": 0,
        "max_tokens": 128,
    }
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    try:
        with TestClient(app) as client:
            buffered = client.post("/v1/chat/completions", json=body)
            streamed = client.post(
                "/v1/chat/completions", json={**body, "stream": True}
            )
        assert buffered.status_code == streamed.status_code == 200
        packets = [
            json.loads(line[6:])
            for line in streamed.text.splitlines()
            if line.startswith("data: {")
        ]
        expected = {
            "reasoning": "The user wants 17 times 23.",
            "content": "391",
            "calls": [],
        }
        assert norm([buffered.json()], "chat", False) == expected
        assert norm(packets, "chat", True) == expected, streamed.text
        assert any(
            choice.get("delta", {}).get("reasoning_content")
            for packet in packets
            for choice in packet.get("choices", [])
        )
        assert "<|" not in streamed.text
    finally:
        reset_config()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_routes(surface, stream):
    from tests.parser_migration.harness import route

    result = route(
        "inkling",
        "inkling",
        M + R + "Secret." + E + CALL + M + T + "Done" + E,
        surface,
        stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
    )
    assert result["reasoning"] == "Secret."
    assert result["content"] == "Done"
    assert result["calls"] == [["read_file", {"path": "sample.txt"}]]


@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"f","args":{"nested":{"name":"other","args":{}},"x":[1,true,null],"s":"é\\"}"}}',
        '{"args":{"x":1},"name":"f"}',
        '{"name":"f"}',
    ],
)
def test_argument_deltas(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + "header_name" + C + payload + E
    deltas = []
    for i, char in enumerate(raw):
        result = parser.extract_tool_calls_streaming(raw[:i], raw[: i + 1], char)
        if result:
            deltas.extend(result.get("tool_calls", []))
    assert sum("id" in d for d in deltas) == 1
    assert deltas[0]["function"]["name"] == "f"
    args = "".join(d["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == json.loads(payload).get("args", {})
    assert all(d["index"] == 0 for d in deltas)
    assert len(deltas) == 1


def test_token_stream_and_repeated_blocks():
    parser = get_parser("inkling")()
    previous = reasoning = content = ""
    for token in [M, R, "first", E, M, T, "one", E, M, R, "second", E, M, T, "two", S]:
        msg = parser.extract_reasoning_streaming(previous, previous + token, token)
        previous += token
        if msg:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    assert (reasoning, content) == ("firstsecond", "onetwo")


def test_tool_finalize_and_reset():
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = CALL[: -len(E)]
    assert parser.extract_tool_calls_streaming("", raw, raw) is None
    final = parser.finalize_legacy_raw_stream(raw)
    assert json.loads(final["tool_calls"][0]["function"]["arguments"]) == {
        "path": "sample.txt"
    }
    assert parser.finalize_legacy_raw_stream(raw) is None
    assert parser.payload is None
    parser.reset()
    result = parser.extract_tool_calls_streaming("", CALL, CALL)
    assert result["tool_calls"][0]["index"] == 0
    assert parser.extract_tool_calls("Plain prose.").content == "Plain prose."


def test_reasoning_whitespace():
    parser = get_parser("inkling")()
    assert parser.extract_reasoning(R + "one two \n" + E + T + "Answer") == (
        "one two",
        "Answer",
    )
    previous = reasoning = ""
    for char in R + "one two \n" + E:
        msg = parser.extract_reasoning_streaming(previous, previous + char, char)
        previous += char
        if msg:
            reasoning += msg.reasoning or ""
    assert reasoning == "one two"


@pytest.mark.parametrize(
    "payload", ["oops", "[]", '{"args":{}}', '{"name":"f","args":']
)
def test_malformed_streaming(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + C + payload + E
    content = ""
    for i, char in enumerate(raw):
        out = parser.extract_tool_calls_streaming(raw[:i], raw[: i + 1], char)
        if out:
            assert not out.get("tool_calls")
            content += out.get("content", "")
    assert content == payload


def test_reasoning_request_preserves_delimiters():
    parser = get_parser("inkling")()
    assert parser.reasoning_start_str == R
    assert parser.reasoning_end_str == E
    request = SimpleNamespace(skip_special_tokens=True)
    assert parser.adjust_request(request) is request
    assert request.skip_special_tokens is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Plain prose.", False),
        (M + R + "Unfinished", True),
        (R + "Thought" + E, False),
        (R + "Thought" + S, False),
        (R + "Thought" + T + "Answer", False),
        (R + "Thought" + E + M + R + "Again", True),
    ],
)
def test_open_thinking_block(raw, expected):
    assert get_parser("inkling")().is_open_in_think(raw) is expected


def test_streaming_json_member_whitespace():
    parser = ToolParserManager.get_tool_parser("inkling")()
    payload = '{ "name" \t: \n"f", "args" \n: \t{"x": 1}}'
    raw = C + payload
    assert parser.extract_tool_calls_streaming("", raw, raw) is None
    out = parser.extract_tool_calls_streaming(raw, raw + E, E)
    call = out["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"] == {"name": "f", "arguments": '{"x": 1}'}
    assert parser.finalize_legacy_raw_stream(raw + E) is None


@pytest.mark.parametrize("close", [E, S, C, ""])
@pytest.mark.parametrize("width", [1, 7, 1000])
@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"broken","args":{"x":',
        '{"name":"broken","args":{"x":1,}}',
        '{"name":"broken","args":{"x":1}} trailing',
        '{"name":"broken","args":{"x":1},"extra":oops}',
    ],
)
def test_malformed_close_after_partial_arguments(close, width, payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = C + payload + close
    complete = parser.extract_tool_calls(raw)
    assert not complete.tools_called
    assert complete.content == payload
    content = ""
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        out = parser.extract_tool_calls_streaming(raw[:pos], raw[:pos] + chunk, chunk)
        if out:
            assert not out.get("tool_calls")
            content += out.get("content", "")
    out = parser.finalize_legacy_raw_stream(raw)
    if out:
        assert not out.get("tool_calls")
        content += out.get("content", "")
    assert content == complete.content
    following = parser.extract_tool_calls_streaming(raw, raw + CALL, CALL)
    call = following["tool_calls"][0]
    assert "content" not in following
    assert call["index"] == 0
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "sample.txt"}


def test_argument_fragments_merge_within_one_chunk():
    parser = ToolParserManager.get_tool_parser("inkling")()
    payload = '{"name":"f","args":{"text":"a<b<c"}}'
    raw = C + payload + E + T + "Done"
    out = parser.extract_tool_calls_streaming("", raw, raw)
    assert out["content"] == "Done"
    assert out["preserve_post_tool_content"] is True
    assert len(out["tool_calls"]) == 1
    call = out["tool_calls"][0]
    assert call["index"] == 0
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "f"
    assert json.loads(call["function"]["arguments"]) == {"text": "a<b<c"}
    assert parser.finalize_legacy_raw_stream(raw) is None
