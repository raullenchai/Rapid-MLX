# SPDX-License-Identifier: Apache-2.0
"""Undeclared calls remain non-executable and observable on both chat wires."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import tool_call_diagnostics as diagnostics
from rapid_mlx.api.models import ChatCompletionRequest, ChatCompletionResponse
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes import chat
from rapid_mlx.tool_call_scan import split_marked_calls

TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_shell_command", "parameters": {"type": "object"}},
    }
]
# Reduced from the actual confirmation run, preserving the undeclared name and
# native raw argument markup. Model-authored claims are not tool results.
RAW_ARGS = '<arg_key>content</arg_key><arg_value>print("OK")\n</arg_value>'
RAW = "Writing the file now.<tool_call>write_file" + RAW_ARGS + "</tool_call>"


class Engine:
    preserve_native_tool_format = True
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, raw):
        self.raw = raw

    def build_prompt(self, *args, **kwargs):
        return "prompt"

    async def chat(self, **kwargs):
        return GenerationOutput(
            text=self.raw,
            raw_text=self.raw,
            prompt_tokens=4,
            completion_tokens=10,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, **kwargs):
        for index, char in enumerate(self.raw):
            yield GenerationOutput(
                text=self.raw[: index + 1],
                new_text=char,
                prompt_tokens=4,
                completion_tokens=index + 1,
                finished=False,
            )
        yield GenerationOutput(
            text=self.raw,
            new_text="",
            prompt_tokens=4,
            completion_tokens=len(self.raw),
            finished=True,
            finish_reason="stop",
        )


@pytest.fixture
def client():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.tool_call_parser = "glm47"
    cfg.enable_auto_tool_choice = True
    app = FastAPI()
    app.include_router(chat.router)
    yield TestClient(app), cfg
    reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "raw, rejected",
    [
        (RAW, True),
        ("ordinary prose", False),
        (RAW.replace("write_file", "run_shell_command"), False),
    ],
)
def test_http_diagnostics_never_become_executable(
    client, stream, raw, rejected, caplog
):
    http, cfg = client
    cfg.engine = Engine(raw)
    before = diagnostics.rejected_tool_calls_total()
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "write and run"}],
            "tools": TOOLS,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    if stream:
        payloads = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        finals = [
            p for p in payloads if any(c.get("finish_reason") for c in p["choices"])
        ]
        assert len(finals) == 1
        body = finals[0]
        assert response.text.count("[DONE]") == 1
    else:
        body = response.json()
    if rejected:
        assert body["rejected_tool_calls"] == [
            {
                "name": "write_file",
                "raw_arguments": RAW_ARGS,
                "reason": "undeclared_tool",
            }
        ]
        assert diagnostics.rejected_tool_calls_total() == before + 1
        assert 'print("OK")' not in caplog.text
        assert "undeclared_tool" in caplog.text
        assert not any(
            c.get("message", {}).get("tool_calls")
            or c.get("delta", {}).get("tool_calls")
            for p in (payloads if stream else [body])
            for c in p["choices"]
        )
        assert body["choices"][0]["finish_reason"] == "stop"
    else:
        assert "rejected_tool_calls" not in body
        assert diagnostics.rejected_tool_calls_total() == before


def test_scanner_rejection_and_literal_safety(caplog):
    before = diagnostics.rejected_tool_calls_total()
    text = "<function=unknown><parameter=x>raw</parameter></function>"
    assert (
        split_marked_calls(
            text, r"<function=(\w+)>", "</function>", valid_names={"known"}
        )
        == []
    )
    assert diagnostics.rejected_tool_calls_total() == before + 1
    assert "name='unknown'" in caplog.text
    assert (
        split_marked_calls(
            "<function=unknown>partial",
            r"<function=(\w+)>",
            "</function>",
            valid_names={"known"},
        )
        == []
    )
    literal = "<function=known><parameter=x>literal </function> <function=unknown> text</parameter></function>"
    parsed = split_marked_calls(
        literal, r"<function=(\w+)>", "</function>", valid_names={"known"}
    )
    assert len(parsed) == 1 and "<function=unknown>" in parsed[0][1]
    assert diagnostics.rejected_tool_calls_total() == before + 1


@pytest.mark.asyncio
async def test_request_isolation_errors_and_stream_close():
    @diagnostics.observe_tool_rejections
    async def handler(name):
        diagnostics.record_rejected_tool_call(name, "raw")
        await asyncio.sleep(0)
        return ChatCompletionResponse(
            model="test",
            choices=[],
            rejected_tool_calls=diagnostics.current_rejected_tool_calls(),
        )

    first, second = await asyncio.gather(handler("first"), handler("second"))
    assert first.rejected_tool_calls[0]["name"] == "first"
    assert second.rejected_tool_calls[0]["name"] == "second"
    assert diagnostics._rejections.get() is None

    @diagnostics.observe_tool_rejections
    async def failing():
        raise ValueError("failed")

    with pytest.raises(ValueError):
        await failing()
    assert diagnostics._rejections.get() is None
    closed = []

    async def events():
        try:
            diagnostics.record_rejected_tool_call("aborted", "raw")
            yield 'data: {"choices":[]}\n\n'
        finally:
            closed.append(True)

    stream = diagnostics._observe_stream(events(), {})
    await anext(stream)
    assert diagnostics._rejections.get() is None
    await stream.aclose()
    assert closed == [True]


def test_counter_exported_without_engine():
    from rapid_mlx.routes.metrics import _render_prometheus

    cfg = reset_config()
    assert (
        f"rapid_mlx_rejected_tool_calls_total {diagnostics.rejected_tool_calls_total()}"
        in _render_prometheus(cfg)
    )


@pytest.mark.parametrize(
    "cause",
    [
        "operator_opt_out",
        "no_tools",
        "no_parser",
        "parser_unsupported",
        "tool_choice",
        "parser_auto_unsafe",
        "schema_bounds",
    ],
)
@pytest.mark.asyncio
async def test_grammar_ineligible_admission_causes(cause, monkeypatch, caplog):
    caplog.set_level("INFO")
    cfg = SimpleNamespace(tool_call_parser="hermes")
    req = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "test"}], tools=TOOLS
    )
    monkeypatch.setattr(
        chat, "_constrain_tools_opted_out", lambda: cause == "operator_opt_out"
    )
    if cause == "no_tools":
        req.tools = None
    if cause == "no_parser":
        cfg.tool_call_parser = None
    monkeypatch.setattr(
        chat, "_tool_parser_supports_grammar", lambda cfg: cause != "parser_unsupported"
    )
    if cause == "tool_choice":
        req.tool_choice = "none"
    monkeypatch.setattr(
        chat, "_tool_parser_auto_safe", lambda cfg: cause != "parser_auto_unsafe"
    )
    monkeypatch.setattr(chat, "_tools_within_grammar_bounds", lambda tools: False)
    assert await chat._offload_tool_grammar_build(None, cfg, req) is None
    assert "admission=skipped:" + cause in caplog.text
    assert chat._maybe_build_tool_grammar_processor(None, cfg, req) is None


@pytest.mark.parametrize(
    "status",
    [
        "no_tokenizer",
        "no_llguidance",
        "no_lltokenizer",
        "tokenizer_unsupported",
        "native_wire_mismatch",
        "missing_tool_name",
        "unknown_named_tool",
        "builder_declined",
        "compile_failed",
        "build_failed",
        "armed",
        "backend_missing",
    ],
)
def test_grammar_builder_logs_actual_outcome(status, monkeypatch, caplog):
    from rapid_mlx.api import tool_grammar as grammar

    caplog.set_level("INFO")
    monkeypatch.setattr(chat, "_tool_grammar_eligible", lambda *a: True)
    cfg = SimpleNamespace(tool_call_parser="hermes", reasoning_parser_name=None)
    req = SimpleNamespace(
        model="test", tools=TOOLS, tool_choice="auto", parallel_tool_calls=True
    )
    engine = SimpleNamespace(tokenizer=None if status == "no_tokenizer" else object())
    monkeypatch.setattr(
        grammar, "HAS_LLGUIDANCE", status not in ("no_llguidance", "backend_missing")
    )
    monkeypatch.setattr(grammar, "HAS_LL_TOKENIZER", status != "no_lltokenizer")
    monkeypatch.setattr(
        grammar,
        "get_lltokenizer",
        lambda t: (
            None
            if status in ("no_llguidance", "no_lltokenizer", "tokenizer_unsupported")
            else object()
        ),
    )
    monkeypatch.setattr(
        chat,
        "_auto_tool_grammar_declined_for_native_wire",
        lambda *a, **k: status == "native_wire_mismatch",
    )
    monkeypatch.setattr(grammar, "resolve_reasoning_sentinels", lambda *a, **k: ())
    monkeypatch.setattr(grammar, "model_stop_token_ids", lambda t: ())
    if status == "missing_tool_name":
        req.tools = [{"function": {}}]
    if status == "unknown_named_tool":
        req.tool_choice = {"type": "function", "function": {"name": "missing"}}

    def build(*a, **k):
        if status == "build_failed":
            raise RuntimeError("broken")
        return None if status in ("builder_declined", "backend_missing") else "grammar"

    monkeypatch.setattr(grammar, "build_tool_grammar", build)
    processor = SimpleNamespace(is_broken=lambda: status == "compile_failed")
    monkeypatch.setattr(grammar, "GrammarLogitsProcessor", lambda *a, **k: processor)
    result = chat._maybe_build_tool_grammar_processor(engine, cfg, req)
    assert (result is processor) == (status == "armed")
    if status == "backend_missing":
        status = "no_llguidance"
    assert (
        "admission=" + ("armed" if status == "armed" else "skipped:" + status)
        in caplog.text
    )


@pytest.mark.parametrize(
    "status", ["admission_full", "submission_failed", "offload_failed"]
)
@pytest.mark.asyncio
async def test_grammar_offload_failures_are_observable(status, monkeypatch, caplog):
    from concurrent.futures import Future

    caplog.set_level("INFO")
    monkeypatch.setattr(chat, "_tool_grammar_eligible", lambda *a: True)
    monkeypatch.setattr(
        chat, "_try_admit_tool_grammar_build", lambda: status != "admission_full"
    )
    monkeypatch.setattr(chat, "_release_tool_grammar_build", lambda: None)

    def submit(*a):
        if status == "submission_failed":
            raise RuntimeError("closed")
        future = Future()
        future.set_exception(RuntimeError("failed"))
        return future

    monkeypatch.setattr(
        chat, "_get_tool_grammar_build_executor", lambda: SimpleNamespace(submit=submit)
    )
    assert (
        await chat._offload_tool_grammar_build(
            None, None, SimpleNamespace(model="test")
        )
        is None
    )
    assert "admission=skipped:" + status in caplog.text


@pytest.mark.asyncio
async def test_stream_bytes_and_error_frames_keep_their_wire_type():
    async def events():
        diagnostics.record_rejected_tool_call("bad", "raw")
        for event in [
            b'data: {"error":"failed"}\n\n',
            b'data: {"object":"chat.completion.chunk","choices":[{"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]:
            yield event

    output = [event async for event in diagnostics._observe_stream(events(), {})]
    assert all(isinstance(event, bytes) for event in output)
    assert output[0] == b'data: {"error":"failed"}\n\n'
    assert json.loads(output[1][6:])["rejected_tool_calls"][0]["name"] == "bad"
    assert output[2] == b"data: [DONE]\n\n"


@pytest.mark.parametrize("stream", [False, True])
def test_optional_sdk_accepts_diagnostic_extension(client, stream):
    sdk = pytest.importorskip("openai.types.chat")
    http, cfg = client
    cfg.engine = Engine(RAW)
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": TOOLS,
            "stream": stream,
        },
    )
    assert response.status_code == 200
    if stream:
        for line in response.text.splitlines():
            if line.startswith("data: {"):
                sdk.ChatCompletionChunk.model_validate(json.loads(line[6:]))
    else:
        sdk.ChatCompletion.model_validate(response.json())


@pytest.mark.parametrize(
    "frame",
    [
        "data: {not json}\n\n",
        b'data: {"value":"\xff"}\n\n',
        "event: custom\ndata: {}\n\n",
        'data: {"custom":true}\n\n',
        'data: {"object":"chat.completion.chunk","choices":null}\n\n',
        'data: {"object":"chat.completion.chunk","choices":[null,{}]}\n\n',
        memoryview(b"custom"),
    ],
)
def test_custom_frames_pass_through_unchanged(frame):
    assert (
        diagnostics._annotate_terminal_event(frame, {"key": {"name": "bad"}}) is frame
    )


@pytest.mark.asyncio
async def test_async_iterator_without_aclose_preserves_completion_and_errors():
    class Events:
        def __init__(self, fail):
            self.started = False
            self.fail = fail

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.started:
                if self.fail:
                    raise RuntimeError("original stream error")
                raise StopAsyncIteration
            self.started = True
            diagnostics.record_rejected_tool_call("bad", "raw")
            return "data: [DONE]\n\n"

    assert [x async for x in diagnostics._observe_stream(Events(False), {})] == [
        "data: [DONE]\n\n"
    ]
    with pytest.raises(RuntimeError, match="original stream error"):
        async for _ in diagnostics._observe_stream(Events(True), {}):
            pass
    assert diagnostics._rejections.get() is None


def test_rejected_scanner_bounds_each_observation_and_requires_outer_wrapper():
    before = diagnostics.rejected_tool_calls_total()
    opener = r"<function=(\w+)>"
    # Adjacent undeclared names have separate, bounded raw arguments.
    text = "<function=first>a</function></tool_call><function=second>b</function></tool_call>"
    assert (
        split_marked_calls(text, opener, "</function>", "</tool_call>", {"known"}) == []
    )
    assert diagnostics.rejected_tool_calls_total() == before + 2
    assert (
        split_marked_calls(
            "<function=bad>x</function>",
            opener,
            "</function>",
            "</tool_call>",
            {"known"},
        )
        == []
    )
    assert diagnostics.rejected_tool_calls_total() == before + 2
    # The last close does not turn every earlier unclosed marker into a huge
    # overlapping rejected argument span. Only the last candidate is complete.
    text = "<function=bad>" * 4000 + "last</function>"
    assert split_marked_calls(text, opener, "</function>", valid_names={"known"}) == []
    assert diagnostics.rejected_tool_calls_total() == before + 3
