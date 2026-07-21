# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the startup tool-grammar warmup gating (#558 perf follow-up).

``_warmup_tool_grammar`` must fire the (expensive) LLTokenizer pre-build ONLY when
the server is actually configured for grammar-capable tool calling, and must be a
safe no-op otherwise — so a text-only / non-grammar deploy pays nothing at boot.
These tests drive the gating with stubs (no model, no llguidance) and assert the
heavy path (``_do_tool_grammar_warmup``) runs exactly when expected (codex #1155).
"""

import types

import pytest

import vllm_mlx.server as server


class _StubEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _GrammarParser:
    SUPPORTS_GRAMMAR = True

    def __init__(self, tokenizer=None):
        pass


class _NonGrammarParser:
    SUPPORTS_GRAMMAR = False

    def __init__(self, tokenizer=None):
        pass


@pytest.fixture
def _patch(monkeypatch):
    """Return a helper that wires cfg/parser/llguidance stubs and spies the build."""
    calls = {"n": 0}

    def _install(*, parser_name, parser_cls, has_llg=True, has_lltok=True):
        monkeypatch.setattr(
            server,
            "get_config",
            lambda: types.SimpleNamespace(tool_call_parser=parser_name),
        )
        # ``ToolParserManager`` is imported INSIDE the function via
        # ``from .tool_parsers import ToolParserManager``, so patch it on that
        # module (not on ``server``).
        import vllm_mlx.tool_parsers as tool_parsers

        monkeypatch.setattr(
            tool_parsers.ToolParserManager,
            "get_tool_parser",
            staticmethod(lambda name: parser_cls),
        )
        # HAS_* live on the api.tool_grammar module imported inside the function.
        import vllm_mlx.api.tool_grammar as tg

        monkeypatch.setattr(tg, "HAS_LLGUIDANCE", has_llg)
        monkeypatch.setattr(tg, "HAS_LL_TOKENIZER", has_lltok)

        def _spy(tokenizer, parser_cls_arg):
            calls["n"] += 1
            return True

        monkeypatch.setattr(server, "_do_tool_grammar_warmup", _spy)

    return _install, calls


async def test_warmup_fires_for_grammar_capable_parser(_patch):
    install, calls = _patch
    install(parser_name="harmony", parser_cls=_GrammarParser)
    await server._warmup_tool_grammar(_StubEngine(tokenizer=object()))
    assert calls["n"] == 1


async def test_warmup_skips_when_no_parser_configured(_patch):
    install, calls = _patch
    install(parser_name=None, parser_cls=_GrammarParser)
    await server._warmup_tool_grammar(_StubEngine(tokenizer=object()))
    assert calls["n"] == 0


async def test_warmup_skips_when_parser_not_grammar_capable(_patch):
    install, calls = _patch
    install(parser_name="mistral", parser_cls=_NonGrammarParser)
    await server._warmup_tool_grammar(_StubEngine(tokenizer=object()))
    assert calls["n"] == 0


async def test_warmup_skips_when_llguidance_unavailable(_patch):
    install, calls = _patch
    install(parser_name="harmony", parser_cls=_GrammarParser, has_llg=False)
    await server._warmup_tool_grammar(_StubEngine(tokenizer=object()))
    assert calls["n"] == 0


async def test_warmup_skips_when_tokenizer_missing(_patch):
    install, calls = _patch
    install(parser_name="harmony", parser_cls=_GrammarParser)
    await server._warmup_tool_grammar(_StubEngine(tokenizer=None))
    assert calls["n"] == 0
