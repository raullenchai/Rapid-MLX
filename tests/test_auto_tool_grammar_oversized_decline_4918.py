# SPDX-License-Identifier: Apache-2.0
"""Oversized auto tools stay free-form when the template's native wire wins."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import chat

XML_TEMPLATE = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("qwen35_chat_template.jinja")
    .read_text()
)
JSON_TEMPLATE = """{# <function= <parameter= are inactive source markers #}
{% for message in messages %}
{{ message.role }}:{{ message.content or '' }}
{% for call in message.tool_calls or [] %}
<tool_call>{{ call.function | tojson }}</tool_call>
{% endfor %}
{% endfor %}"""
LENGTH_DEPENDENT_TEMPLATE = """{# <function= <parameter= source markers #}
{% for message in messages %}
{% for call in message.tool_calls or [] %}
{% if tools|length <= 1 %}
<tool_call>\n<function={{ call.function.name }}><parameter=value>x</parameter></function></tool_call>
{% else %}
<tool_call>{{ call.function | tojson }}</tool_call>
{% endif %}
{% endfor %}
{% endfor %}"""


def engine(template=XML_TEMPLATE):
    return SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))


def tool(name, parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def request(*, tools=None, choice="auto", thinking=None, history=True):
    messages = [{"role": "user", "content": "Find the weather."}]
    if history:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query":"weather"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "Sunny.",
                },
            ]
        )
    kwargs = {}
    if thinking is not None:
        kwargs["chat_template_kwargs"] = {"enable_thinking": thinking}
    return ChatCompletionRequest(
        model="test",
        messages=messages,
        tools=tools or [tool("search")],
        tool_choice=choice,
        **kwargs,
    )


def cfg(parser="hermes"):
    return SimpleNamespace(tool_call_parser=parser)


def many_tools(count=300):
    return [tool(f"tool_{index}") for index in range(count)]


def described_tools(description_bytes, count=300):
    tools = many_tools(count)
    for item in tools:
        item["function"]["description"] = "x" * description_bytes
    return tools


def oversized_bytes_tool():
    return tool(
        "large",
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "string",
                    "description": "x" * chat._TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
                }
            },
        },
    )


def nested_schema(depth):
    schema = {"type": "string"}
    for _ in range(depth):
        schema = {"allOf": [schema]}
    return schema


def oversized_depth_tool(depth=20):
    return tool("deep", nested_schema(depth))


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    reset_config()


@pytest.fixture(autouse=True)
def constrain_tools(monkeypatch):
    from rapid_mlx.api import tool_grammar

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)


@pytest.fixture(autouse=True)
def restore_tool_grammar_admission():
    prior = chat._tool_grammar_inflight
    yield
    chat._tool_grammar_inflight = prior


@pytest.mark.parametrize(
    "tools",
    [many_tools(), [oversized_bytes_tool()], [oversized_depth_tool()]],
)
async def test_gate_exempts_oversized_auto_native_xml(tools):
    req = request(tools=tools)
    assert chat._tool_grammar_eligible(cfg(), req) is False
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(cfg(), req)
    assert exc.value.status_code == 400
    assert (
        await chat._enforce_tool_grammar_bounds_or_400_async(cfg(), req, engine())
        is None
    )


async def test_json_wire_oversized_auto_still_400():
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine(JSON_TEMPLATE)
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "choice",
    ["required", {"type": "function", "function": {"name": "tool_0"}}],
)
async def test_forced_oversized_native_xml_still_400(choice):
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools(), choice=choice), engine()
        )
    assert exc.value.status_code == 400


async def test_opt_out_oversized_still_free_form(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")
    assert (
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools(), choice="required"), engine()
        )
        is None
    )


async def test_gate_probe_uses_full_request_tools(monkeypatch):
    probe = Mock(wraps=chat._template_renders_native_xml_tool_call)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    req = request(tools=many_tools())

    await chat._enforce_tool_grammar_bounds_or_400_async(cfg(), req, engine())

    assert probe.call_args.args[1] is req
    assert probe.call_args.kwargs == {}
    assert chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), req)


async def test_gate_probe_runs_off_event_loop_thread(monkeypatch):
    event_loop_thread = threading.get_ident()
    probe_threads = []
    original = chat._auto_tool_grammar_declined_for_native_wire

    def record_thread(*args, **kwargs):
        probe_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        chat, "_auto_tool_grammar_declined_for_native_wire", record_thread
    )

    await chat._enforce_tool_grammar_bounds_or_400_async(
        cfg(), request(tools=many_tools()), engine()
    )

    assert probe_threads
    assert probe_threads[0] != event_loop_thread


async def test_gate_probe_admission_refused_keeps_400(monkeypatch):
    probe = Mock(return_value=True)
    monkeypatch.setattr(chat, "_try_admit_tool_grammar_build", lambda: False)
    monkeypatch.setattr(chat, "_auto_tool_grammar_declined_for_native_wire", probe)

    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )

    assert exc.value.status_code == 400
    probe.assert_not_called()


async def test_gate_probe_submission_failure_releases_admission(monkeypatch):
    class BrokenExecutor:
        def submit(self, *args, **kwargs):
            raise RuntimeError("executor shut down")

    prior = chat._tool_grammar_inflight
    release = Mock(wraps=chat._release_tool_grammar_build)
    monkeypatch.setattr(chat, "_get_tool_grammar_build_executor", BrokenExecutor)
    monkeypatch.setattr(chat, "_release_tool_grammar_build", release)

    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )

    assert exc.value.status_code == 400
    release.assert_called_once_with()
    assert chat._tool_grammar_inflight == prior


async def test_gate_probe_success_releases_admission_and_tracks_future():
    prior = chat._tool_grammar_inflight
    tracked = []
    tracked_engine = engine()
    tracked_engine.track_tool_grammar_build = tracked.append

    await chat._enforce_tool_grammar_bounds_or_400_async(
        cfg(), request(tools=many_tools()), tracked_engine
    )

    assert len(tracked) == 1
    assert tracked[0].done()
    assert chat._tool_grammar_inflight == prior


async def test_gate_probe_tracking_failure_still_completes_and_releases_admission():
    prior = chat._tool_grammar_inflight
    tracked_engine = engine()
    tracked_engine.track_tool_grammar_build = Mock(
        side_effect=RuntimeError("tracking unavailable")
    )

    assert (
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), tracked_engine
        )
        is None
    )

    tracked_engine.track_tool_grammar_build.assert_called_once()
    assert chat._tool_grammar_inflight == prior


async def test_gate_probe_worker_failure_keeps_400(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_auto_tool_grammar_declined_for_native_wire",
        Mock(side_effect=RuntimeError("probe failed")),
    )

    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )

    assert exc.value.status_code == 400


async def test_gate_probe_cancelled_caller_propagates_and_holds_slot(monkeypatch):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    started = threading.Event()
    finish = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    prior = chat._tool_grammar_inflight

    def blocking_probe(*args, **kwargs):
        started.set()
        finish.wait(timeout=5)
        return True

    monkeypatch.setattr(
        chat, "_auto_tool_grammar_declined_for_native_wire", blocking_probe
    )
    monkeypatch.setattr(chat, "_get_tool_grammar_build_executor", lambda: pool)

    try:
        task = asyncio.create_task(
            chat._offload_native_wire_decline_probe(
                engine(), cfg(), request(tools=many_tools())
            )
        )
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        assert chat._tool_grammar_inflight == prior + 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert chat._tool_grammar_inflight == prior + 1
    finally:
        finish.set()
        pool.shutdown(wait=True)

    for _ in range(200):
        if chat._tool_grammar_inflight == prior:
            break
        await asyncio.sleep(0.005)
    assert chat._tool_grammar_inflight == prior


async def test_async_gate_propagates_non_bounds_http(monkeypatch):
    def unavailable(cfg, request):  # noqa: ARG001
        raise HTTPException(status_code=503, detail="unavailable")

    monkeypatch.setattr(chat, "_enforce_tool_grammar_bounds_or_400", unavailable)

    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )

    assert exc.value.status_code == 503


async def test_exact_probe_respects_tool_count_dependent_wire():
    req = request(tools=many_tools())
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(LENGTH_DEPENDENT_TEMPLATE), cfg(), req
    )
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), req, engine(LENGTH_DEPENDENT_TEMPLATE)
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "tools",
    [
        many_tools(chat._TOOL_NATIVE_WIRE_PROBE_MAX_TOOLS + 1),
        [
            tool(
                "too_large",
                {
                    "type": "string",
                    "description": "x" * chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES,
                },
            )
        ],
        [oversized_depth_tool(chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_DEPTH + 1)],
    ],
    ids=["tool-count", "schema-bytes", "schema-depth"],
)
async def test_above_probe_cap_400_without_template_render(monkeypatch, tools):
    probe = Mock(wraps=chat._template_renders_native_xml_tool_call)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    req = request()
    req.tools = tools

    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), req, engine(XML_TEMPLATE)
        )

    assert exc.value.status_code == 400
    probe.assert_not_called()


async def test_descriptions_above_complete_probe_cap_400_without_render(monkeypatch):
    probe = Mock(wraps=chat._template_renders_native_xml_tool_call)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    tools = described_tools(chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES // 300 + 1)
    req = request(tools=tools)

    assert not chat._tools_within_grammar_bounds(req.tools)
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), req, engine(XML_TEMPLATE)
        )

    assert exc.value.status_code == 400
    probe.assert_not_called()


async def test_modest_descriptions_within_complete_probe_cap_allow_xml_decline():
    req = request(tools=described_tools(16))

    assert (
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), req, engine(XML_TEMPLATE)
        )
        is None
    )


def test_complete_probe_bound_walks_plain_dict_tool_extensions():
    tools = [tool("plain")]
    tools[0]["extension"] = "x" * chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES

    assert chat._tools_within_grammar_bounds(tools)
    assert not chat._tools_within_grammar_bounds(
        tools,
        max_tools=chat._TOOL_NATIVE_WIRE_PROBE_MAX_TOOLS,
        max_schema_bytes=chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES,
        max_schema_depth=chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_DEPTH,
        whole_tool=True,
    )


def test_complete_probe_bound_walks_object_model_extra():
    tools = [
        SimpleNamespace(
            type="function",
            function=tool("object")["function"],
            model_extra={
                "extension": "x" * chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES
            },
        )
    ]

    assert chat._tools_within_grammar_bounds(tools)
    assert not chat._tools_within_grammar_bounds(
        tools,
        max_tools=chat._TOOL_NATIVE_WIRE_PROBE_MAX_TOOLS,
        max_schema_bytes=chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_BYTES,
        max_schema_depth=chat._TOOL_NATIVE_WIRE_PROBE_MAX_SCHEMA_DEPTH,
        whole_tool=True,
    )


def test_helper_name_scan_does_not_read_parameter_schemas(monkeypatch):
    class Function:
        name = "object_tool"

        @property
        def parameters(self):
            raise AssertionError("parameter schema was touched")

    class Tool:
        function = Function()

    probe = Mock(return_value=True)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    req = request()
    req.tools = [Tool()]
    assert chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), req)
    req.tools = [tool("dict_tool")]
    assert chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), req)


def test_helper_returns_false_for_non_auto_no_structure_and_unsafe_name():
    parser = SimpleNamespace(structure_info=lambda: None)
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), request(choice="required"), parser
    )
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), request(), parser
    )
    structured = SimpleNamespace(
        structure_info=lambda: lambda name: SimpleNamespace(begin="<tool_call>")
    )
    unsafe = request()
    unsafe.tools[0].function["name"] = "bad>name"
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), unsafe, structured
    )


def test_helper_scans_at_most_tool_count_cap(monkeypatch):
    structured = SimpleNamespace(
        structure_info=lambda: lambda name: SimpleNamespace(begin="<tool_call>")
    )
    req = request()
    req.tools = [
        tool("bad>name") for _ in range(chat._TOOL_NATIVE_WIRE_PROBE_MAX_TOOLS)
    ]
    req.tools.append(tool("search"))
    probe = Mock(return_value=True)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)

    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), req, structured
    )
    probe.assert_not_called()


async def test_oversized_auto_finds_safe_name_after_grammar_tool_cap():
    req = request(tools=many_tools(400))
    # Validation rejects unsafe names on input, so mutate after construction.
    for item in req.tools[:300]:
        item.function["name"] = "bad>name"

    assert not chat._tools_within_grammar_bounds(req.tools)
    assert (
        await chat._enforce_tool_grammar_bounds_or_400_async(cfg(), req, engine())
        is None
    )


async def test_helper_propagates_parser_construction_failure_and_gate_keeps_400(
    monkeypatch,
):
    from rapid_mlx.tool_parsers import ToolParserManager

    class BrokenParser:
        @classmethod
        def supports_grammar(cls):
            return True

        def __init__(self, tokenizer):  # noqa: ARG002
            raise RuntimeError("broken parser constructor")

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: BrokenParser)
    with pytest.raises(RuntimeError, match="broken parser constructor"):
        chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), request())
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400


def test_builder_uses_shared_decline_helper(monkeypatch):
    from rapid_mlx.api import tool_grammar

    shared = Mock(return_value=True)
    monkeypatch.setattr(chat, "_auto_tool_grammar_declined_for_native_wire", shared)
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)
    monkeypatch.setattr(tool_grammar, "get_lltokenizer", lambda tokenizer: object())
    build = Mock(return_value=None)
    monkeypatch.setattr(tool_grammar, "build_tool_grammar", build)

    assert chat._maybe_build_tool_grammar_processor(engine(), cfg(), request()) is None
    assert shared.call_args.kwargs["parser"] is not None
    build.assert_not_called()


def test_chat_route_oversized_auto_native_xml_returns_response(monkeypatch):
    async def await_direct(coro, raw_request, timeout):  # noqa: ARG001
        return await coro

    class Tokenizer:
        chat_template = XML_TEMPLATE
        model_max_length = 100_000
        bos_token = None

        def encode(self, text, add_special_tokens=True):  # noqa: ARG002
            return [0] * max(1, len(text) // 4)

    class Engine:
        is_mllm = False
        supports_guided_generation = False
        supports_tool_calls = True
        preserve_native_tool_format = False
        tokenizer = Tokenizer()

        def build_prompt(self, messages, tools=None, enable_thinking=None):  # noqa: ARG002
            return "prompt"

        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return GenerationOutput(
                text="ok",
                raw_text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                cached_tokens=0,
            )

    from rapid_mlx.middleware.auth import rate_limiter

    monkeypatch.setattr(rate_limiter, "enabled", False)
    monkeypatch.setattr(chat, "_wait_with_disconnect", await_direct)
    route_cfg = reset_config()
    route_cfg.engine = Engine()
    route_cfg.model_name = "test"
    route_cfg.model_registry = None
    route_cfg.tool_call_parser = "hermes"
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat.router)

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": many_tools(),
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize(
    "factory",
    [
        lambda name: None,
        lambda name: SimpleNamespace(begin=None),
        Mock(side_effect=ValueError("rejected name")),
    ],
)
async def test_gate_keeps_400_when_parser_factory_fails(monkeypatch, factory):
    from rapid_mlx.tool_parsers import ToolParserManager

    class Parser:
        @classmethod
        def supports_grammar(cls):
            return True

        def __init__(self, tokenizer):  # noqa: ARG002
            pass

        def structure_info(self):
            return factory

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400


def test_helper_propagates_structure_info_failure():
    parser = SimpleNamespace(
        structure_info=Mock(side_effect=RuntimeError("structure unavailable"))
    )

    with pytest.raises(RuntimeError, match="structure unavailable"):
        chat._auto_tool_grammar_declined_for_native_wire(
            engine(), cfg(), request(), parser
        )


@pytest.mark.parametrize(
    ("factory", "exception"),
    [
        (lambda name: None, AttributeError),
        (lambda name: SimpleNamespace(begin=None), AttributeError),
        (Mock(side_effect=ValueError("rejected name")), ValueError),
    ],
)
def test_helper_propagates_parser_factory_failures(factory, exception):
    parser = SimpleNamespace(structure_info=lambda: factory)

    with pytest.raises(exception):
        chat._auto_tool_grammar_declined_for_native_wire(
            engine(), cfg(), request(), parser
        )


async def test_helper_propagates_probe_failure_and_gate_keeps_400(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_template_renders_native_xml_tool_call",
        Mock(side_effect=KeyError("template blew up")),
    )
    parser = SimpleNamespace(
        structure_info=lambda: lambda name: SimpleNamespace(begin="<tool_call>")
    )
    with pytest.raises(KeyError, match="template blew up"):
        chat._auto_tool_grammar_declined_for_native_wire(
            engine(), cfg(), request(), parser
        )
    with pytest.raises(HTTPException) as exc:
        await chat._enforce_tool_grammar_bounds_or_400_async(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400


def test_builder_factory_failure_falls_back_without_building(monkeypatch):
    from rapid_mlx.api import tool_grammar
    from rapid_mlx.tool_parsers import ToolParserManager

    class Parser:
        @classmethod
        def supports_grammar(cls):
            return True

        def __init__(self, tokenizer):  # noqa: ARG002
            pass

        def structure_info(self):
            return Mock(side_effect=RuntimeError("wire info unavailable"))

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    monkeypatch.setattr(tool_grammar, "get_lltokenizer", lambda tokenizer: object())
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)
    monkeypatch.setattr(tool_grammar, "resolve_reasoning_sentinels", lambda *args: ())
    build = Mock(return_value=None)
    monkeypatch.setattr(tool_grammar, "build_tool_grammar", build)

    assert chat._maybe_build_tool_grammar_processor(engine(), cfg(), request()) is None
    build.assert_not_called()
