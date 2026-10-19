# SPDX-License-Identifier: Apache-2.0
"""Forced tool grammars deliberately keep the parser wire on native-XML templates.

On a model whose chat template renders native XML tool calls while the
configured parser's grammar wire is JSON (for example the hermes parser), the
reasoning-gated forced path builds the parser's JSON-wire grammar for
``required`` and named choices. Unlike ``auto`` (which declines the grammar on
that mismatch), a forced grammar constrains the whole call from the opener, and
the configured parser consumes that wire, so forced calls stay well formed.
These tests pin that contract (#4881).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.routes import chat
from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

XML_TEMPLATE = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("qwen35_chat_template.jinja")
    .read_text()
)
NAMED = {"type": "function", "function": {"name": "search"}}
CFG = SimpleNamespace(tool_call_parser="hermes")


def request(choice):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice=choice,
    )


def engine(template=XML_TEMPLATE):
    return SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))


@pytest.fixture(autouse=True)
def configure_grammar(monkeypatch):
    from rapid_mlx.api import tool_grammar

    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")


@pytest.mark.parametrize("choice", ["required", NAMED])
def test_forced_grammar_deliberately_uses_parser_json_wire(monkeypatch, caplog, choice):
    from rapid_mlx.api import tool_grammar

    req = request(choice)
    normalized = chat._normalize_tool_choice_for_grammar(req.tool_choice)
    grammar = tool_grammar.build_tool_grammar(
        [req.tools[0].function],
        "required" if normalized["mode"] == "required" else normalized["name"],
        HermesToolParser(tokenizer=None),
        reasoning_sentinels=(),
    )
    assert grammar is not None
    assert r"{\"name\": \"search\"" in grammar
    assert "<function=search>" not in grammar

    monkeypatch.setattr(tool_grammar, "get_lltokenizer", lambda tokenizer: object())
    monkeypatch.setattr(tool_grammar, "resolve_reasoning_sentinels", lambda *args: ())
    monkeypatch.setattr(tool_grammar, "model_stop_token_ids", lambda tokenizer: ())
    # Run the REAL builder through the route helper; only the llguidance-backed
    # processor is replaced by a recorder, so the installed grammar is asserted.
    build = Mock(wraps=tool_grammar.build_tool_grammar)
    monkeypatch.setattr(tool_grammar, "build_tool_grammar", build)

    class RecordingProcessor:
        def __init__(self, lltok, grammar, **kwargs):
            self.grammar = grammar
            self.reasoning_gate_id = kwargs.get("reasoning_end_id")

        def is_broken(self):
            return False

    monkeypatch.setattr(tool_grammar, "GrammarLogitsProcessor", RecordingProcessor)
    with caplog.at_level("DEBUG", logger=chat.__name__):
        processor = chat._maybe_build_tool_grammar_processor(engine(), CFG, req)
    build.assert_called_once()
    assert build.call_args.args[1] == "required"
    assert "Skipping auto tool grammar" not in caplog.text
    assert isinstance(processor, RecordingProcessor)
    assert r"{\"name\": \"search\"" in processor.grammar
    assert "<function=search>" not in processor.grammar


def test_forced_json_wire_is_consumable_by_hermes():
    parsed = HermesToolParser(tokenizer=None).extract_tool_calls(
        '<tool_call>\n{"name": "search", "arguments": {"query": "weather"}}\n</tool_call>'
    )
    assert parsed.tools_called
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "search"
    assert json.loads(parsed.tool_calls[0]["arguments"]) == {"query": "weather"}
