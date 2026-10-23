# SPDX-License-Identifier: Apache-2.0
"""Auto grammar must agree with the served assistant tool-call wire."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.api.models import ChatCompletionRequest
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


def request(choice="auto", names=("search",), **kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[
            {"role": "user", "content": "Find the weather, then summarize it."},
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
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny in Paris."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in names
        ],
        tool_choice=choice,
        **kwargs,
    )


def engine(template=XML_TEMPLATE):
    return SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))


@pytest.fixture
def build(monkeypatch):
    from rapid_mlx.api import tool_grammar

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)
    monkeypatch.setattr(tool_grammar, "get_lltokenizer", lambda tokenizer: object())
    monkeypatch.setattr(tool_grammar, "resolve_reasoning_sentinels", lambda *args: ())
    # Reaching the builder is enough to prove the new decline was not taken.
    spy = Mock(return_value=None)
    monkeypatch.setattr(tool_grammar, "build_tool_grammar", spy)
    return spy


def run(req, eng=None):
    return chat._maybe_build_tool_grammar_processor(
        eng or engine(), SimpleNamespace(tool_call_parser="hermes"), req
    )


def test_auto_hermes_native_xml_after_search_declines(build, caplog):
    with caplog.at_level("DEBUG", logger=chat.__name__):
        assert run(request()) is None
    build.assert_not_called()
    assert caplog.text.count("admission=skipped:native_wire_mismatch") == 1


def test_auto_json_template_builds(build):
    assert run(request(), engine(JSON_TEMPLATE)) is None
    build.assert_called_once()
    assert build.call_args.args[1] == "auto"


def test_auto_matching_xml_structure_builds(build, monkeypatch):
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    monkeypatch.setattr(
        HermesToolParser,
        "structure_info",
        lambda self: (
            lambda name: SimpleNamespace(begin=f"<tool_call>\n<function={name}>\n")
        ),
    )
    run(request())
    build.assert_called_once()


@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "search"}}]
)
def test_forced_modes_unchanged(build, choice):
    run(request(choice))
    build.assert_called_once()
    assert build.call_args.args[1] == "required"


def test_unprobeable_template_keeps_grammar(build):
    run(
        request(),
        engine("<function= <parameter= {{ raise_exception('bad template') }}"),
    )
    build.assert_called_once()


def test_no_structure_keeps_builder_path(build, monkeypatch):
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    monkeypatch.setattr(HermesToolParser, "structure_info", lambda self: None)
    run(request())
    build.assert_called_once()


@pytest.mark.parametrize("names", [("bad>name",), ("bad>name", "search")])
def test_unsafe_names_skip_or_use_next_safe_tool(build, monkeypatch, names):
    probe = Mock(wraps=chat._template_renders_native_xml_tool_call)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    req = request(names=tuple(f"tool_{i}" for i in range(len(names))))
    for tool, name in zip(req.tools, names, strict=True):
        tool.function["name"] = name
    run(req)
    if len(names) == 1:
        probe.assert_not_called()
        build.assert_called_once()
    else:
        assert probe.call_args.args[2] == "search"
        build.assert_not_called()


@pytest.mark.parametrize(
    "template, expected",
    [
        (XML_TEMPLATE, True),
        ({"tool_use": XML_TEMPLATE, "default": "JSON"}, True),
        ({"default": XML_TEMPLATE}, True),
        (JSON_TEMPLATE, False),
        ("plain template", False),
        (None, None),
        ("<function= <parameter= {{ raise_exception('bad template') }}", None),
    ],
)
def test_probe_results(template, expected):
    assert (
        chat._template_renders_native_xml_tool_call(
            engine(template), request(), "search"
        )
        is expected
    )


@pytest.mark.parametrize("rendered", [["only one"], ["before", "different prefix"]])
def test_probe_nonadditive_render_is_unknown(monkeypatch, rendered):
    monkeypatch.setattr(
        "transformers.utils.chat_template_utils.render_jinja_template",
        lambda *args, **kwargs: (rendered, None),
    )
    assert (
        chat._template_renders_native_xml_tool_call(engine(), request(), "search")
        is None
    )


@pytest.mark.parametrize("thinking", [True, False])
def test_probe_preserves_request_inputs(monkeypatch, thinking):
    from transformers.utils.chat_template_utils import render_jinja_template

    render = Mock(wraps=render_jinja_template)
    monkeypatch.setattr(
        "transformers.utils.chat_template_utils.render_jinja_template", render
    )
    req = request(
        chat_template_kwargs={"enable_thinking": thinking, "custom_flag": "kept"}
    )
    assert chat._template_renders_native_xml_tool_call(engine(), req, "search") is True
    histories = render.call_args.args[0]
    assert histories[0][1]["tool_calls"][0]["function"]["arguments"] == {
        "query": "weather"
    }
    assert histories[0][-1]["role"] == "tool"
    assert render.call_args.kwargs["enable_thinking"] is thinking
    assert render.call_args.kwargs["custom_flag"] == "kept"
    assert render.call_args.kwargs["tools"] == [
        t.model_dump(exclude_none=True) for t in req.tools
    ]
