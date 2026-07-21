# SPDX-License-Identifier: Apache-2.0
"""Offline tests for harmony (gpt-oss) grammar-constrained tool calling (#558).

Extends the #558 constraint coverage from {qwen, hermes} to +gpt-oss. These
validate ``HarmonyToolParser.structure_info()`` and its composition with the
grammar builder WITHOUT a decode loop:

  * the AUTO-path opt-out (``TOOL_GRAMMAR_AUTO_SAFE``): harmony's only single-
    special-token trigger (``<|channel|>``) is SHARED with non-tool responses
    (``final``/``analysis``), so the auto grammar would force a call — harmony
    declines auto (free-form fallback) while constraining ``required``/named;
  * the harmony wire triple's structural invariants (begin starts with the
    ``<|channel|>`` trigger; the four control tokens are declared sentinels);
  * grammar ENFORCEMENT via llguidance ``LLMatcher.consume_tokens`` on the REAL
    gpt-oss tokenizer — a well-formed harmony tool call is accepted in full and
    terminates, while an off-schema argument, a hallucinated tool name, a
    bad enum value, and a header missing the mandatory space before
    ``<|constrain|>`` are rejected mid-stream. This is the load-bearing #558
    proof that the constraint is grammar-enforced, not merely post-parsed.

The enforcement tests need the gpt-oss tokenizer (whose four harmony control
tokens are single special tokens). They skip ONLY on genuine unavailability
(the ``[guided]`` extra absent, or the pinned snapshot not in the local HF
cache — preflighted with no network); any other failure (a resolver regression,
a corrupt artifact) is surfaced. The pure-Python flag/opt-out tests never skip.
"""

import importlib.util

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

# The cached gpt-oss-20b MXFP4-Q8 tokenizer. Its <|channel|>/<|constrain|>/
# <|message|>/<|call|> are single special tokens (ids 200005/200003/200008/
# 200012). Pin the revision so the enforcement proof runs against an IMMUTABLE
# artifact — a different upstream revision must not silently change what the
# enforcement tests exercise.
_TOKENIZER_MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q8"
_TOKENIZER_REVISION = "773a7da77e569019bb0fd17a554b263738d669a3"

TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
            "additionalProperties": False,
        },
    },
]


# --------------------------------------------------------------------------
# Pure-Python flag / opt-out contract (always runs, no tokenizer needed).
# --------------------------------------------------------------------------
def test_abc_tool_grammar_auto_safe_defaults_true():
    # Every single-special-token-trigger family (hermes/qwen) is auto-safe by
    # default — the ABC flag must default True so this change is non-breaking.
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    assert ToolParser.TOOL_GRAMMAR_AUTO_SAFE is True


def test_harmony_opts_out_of_auto_grammar():
    # Harmony's <|channel|> trigger is shared with final/analysis blocks, so it
    # declares itself NOT auto-safe: the auto grammar would force a call.
    from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

    assert HarmonyToolParser.TOOL_GRAMMAR_AUTO_SAFE is False


@_requires_llguidance
def test_build_tool_grammar_auto_declines_for_harmony():
    # AUTO zero-call parity: an auto-unsafe family must yield NO grammar on the
    # auto path (free-form fallback — the model keeps its zero-call freedom),
    # even though the same family DOES build a grammar for required/named. This
    # is the guard against turning harmony ``auto`` into ``required``.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    class _AutoUnsafe:
        TOOL_GRAMMAR_AUTO_SAFE = False

        def structure_info(self):
            from vllm_mlx.api.tool_grammar import StructureInfo

            def _info(name: str):
                return StructureInfo(
                    begin=f"<|channel|>commentary to=functions.{name} "
                    "<|constrain|>json<|message|>",
                    end="<|call|>",
                    trigger="<|channel|>",
                    sentinels=(
                        "<|channel|>",
                        "<|constrain|>",
                        "<|message|>",
                        "<|call|>",
                    ),
                )

            return _info

    parser = _AutoUnsafe()
    # auto -> None (declined); required -> a grammar (constrained).
    assert build_tool_grammar(TOOLS, "auto", parser) is None
    assert build_tool_grammar(TOOLS, "required", parser) is not None


@_requires_llguidance
def test_build_tool_grammar_auto_still_builds_for_auto_safe_family():
    # The opt-out is per-family: a family WITHOUT the flag (defaults auto-safe,
    # like hermes/qwen) still builds an auto grammar. Proves the gate keys on
    # the flag, not on some harmony-only short-circuit.
    from vllm_mlx.api.tool_grammar import StructureInfo, build_tool_grammar

    class _AutoSafe:  # no TOOL_GRAMMAR_AUTO_SAFE -> defaults True via getattr
        def structure_info(self):
            def _info(name: str):
                return StructureInfo(
                    begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
                    end="}\n</tool_call>",
                    trigger="<tool_call>",
                    sentinels=("<tool_call>", "</tool_call>"),
                )

            return _info

    assert build_tool_grammar(TOOLS, "auto", _AutoSafe()) is not None


# --------------------------------------------------------------------------
# Enforcement against the REAL gpt-oss tokenizer.
# --------------------------------------------------------------------------
def _pinned_snapshot_cached() -> bool:
    """True iff the pinned gpt-oss tokenizer snapshot is in the local HF cache.

    Preflight the cache BEFORE loading (codex): with ``local_files_only=True`` an
    uncached snapshot makes transformers raise a BARE ``OSError`` ("couldn't
    connect … couldn't find in cached files"), NOT the specialized
    ``LocalEntryNotFoundError`` — so an except-list keyed on the specialized
    types either over-catches every ``OSError`` (masking a corrupt-file failure)
    or misses this one (failing a normal uncached CI box). ``try_to_load_from_
    cache`` sidesteps the whole error-taxonomy problem: it returns a ``str`` path
    iff the file is cached, and a sentinel/``None`` otherwise, with NO network
    round-trip. A load-time CORRUPTION error AFTER this preflight is a real
    failure and is left to propagate (never caught).
    """
    from huggingface_hub import try_to_load_from_cache

    for fname in ("tokenizer.json", "tokenizer_config.json"):
        cached = try_to_load_from_cache(
            _TOKENIZER_MODEL, fname, revision=_TOKENIZER_REVISION
        )
        if not isinstance(cached, str):
            return False
    return True


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    if not _pinned_snapshot_cached():  # pragma: no cover - uncached CI box
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not in the "
            "local HF cache — enforcement tests require it locally"
        )
    # Cached (preflight passed) -> load offline; a corruption error propagates.
    return transformers.AutoTokenizer.from_pretrained(
        _TOKENIZER_MODEL,
        revision=_TOKENIZER_REVISION,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer via the engine's own resolver.

    ``build_lltokenizer`` has the transformers-5.x fallback (direct build from
    ``backend_tokenizer.to_str()``) the raw ``llguidance.hf.from_tokenizer``
    lacks — the same path the live server uses.

    Once ``tok`` (the real cached gpt-oss tokenizer) is available, the resolver
    MUST yield an ``LLTokenizer`` — a ``None`` here would mean the production
    resolver regressed and harmony grammar constraint is SILENTLY disabled, so we
    FAIL rather than skip (codex): skipping would let the enforcement suite go
    green while the feature is broken.
    """
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    lltokenizer = build_lltokenizer(tok)
    assert lltokenizer is not None, (
        "build_lltokenizer returned None for the real gpt-oss tokenizer — the "
        "LLTokenizer resolver regressed; harmony grammar constraint would be "
        "silently disabled"
    )
    return lltokenizer


@pytest.fixture(scope="module")
def harmony_parser(tok):
    from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

    return HarmonyToolParser(tokenizer=tok)


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)``.

    Advances real grammar state one token at a time via ``consume_tokens``,
    counting how many tokens the grammar accepts before rejecting one. A
    "fully accepted" positive test whose final ``is_accepting()`` is True proves
    the string is a COMPLETE valid derivation, not merely an accepted prefix.
    """
    from llguidance.mlx import LLMatcher

    ids = tok.encode(text, add_special_tokens=False)
    matcher = LLMatcher(lltok, grammar)
    assert not matcher.get_error(), matcher.get_error()
    accepted = 0
    for tid in ids:
        if not matcher.consume_tokens([tid]):
            break
        accepted += 1
    return accepted, len(ids), matcher.is_accepting()


@_requires_llguidance
def test_structure_info_available_and_wire_invariants(harmony_parser):
    # The parser opts IN on a tokenizer whose four control tokens are single
    # special tokens, and the wire triple satisfies the builder invariants.
    get_info = harmony_parser.structure_info()
    assert get_info is not None
    si = get_info("get_weather")
    # StructTag invariants enforced by build_tool_lark:
    assert si.begin.startswith(si.trigger)
    assert si.trigger == "<|channel|>"
    assert si.trigger in si.sentinels
    assert si.end == "<|call|>"
    # All four harmony control tokens are declared sentinels (special-token refs).
    for s in ("<|channel|>", "<|constrain|>", "<|message|>", "<|call|>"):
        assert s in si.sentinels
    # Exact header bytes, including the MANDATORY space before <|constrain|>
    # (openai-harmony rejects the header without it).
    assert si.begin == (
        "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
    )


@_requires_llguidance
def test_valid_harmony_call_is_accepted_and_terminates(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    assert grammar is not None
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": "Paris"}<|call|>',
    )
    assert accepted == total, f"valid harmony call rejected ({accepted}/{total})"
    assert accepting, "valid complete harmony call is not an accepting state"


@_requires_llguidance
def test_valid_enum_value_is_accepted(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": "P", "unit": "celsius"}<|call|>',
    )
    assert accepted == total, f"valid enum value rejected ({accepted}/{total})"
    assert accepting, "valid enum call is not an accepting state"


@_requires_llguidance
def test_off_schema_argument_is_rejected(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    # `city` must be a string; an integer must be forbidden.
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": 4',
    )
    assert accepted < total, "off-schema integer argument was NOT rejected"


@_requires_llguidance
def test_bad_enum_value_is_rejected(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    # `unit` enum is {celsius, fahrenheit}; "kelvin" must be forbidden — this is
    # the adversarial-enum proof (the live pilot's user demanded kelvin and the
    # grammar held it in-enum).
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city": "P", "unit": "kelvin',
    )
    assert accepted < total, "invalid enum value was NOT rejected by the grammar"


@_requires_llguidance
def test_hallucinated_tool_name_is_rejected(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_stock",
    )
    assert accepted < total, "hallucinated tool name was NOT rejected"


@_requires_llguidance
def test_missing_constrain_space_is_rejected(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", harmony_parser)
    # The space before <|constrain|> is mandatory in the harmony header
    # (openai-harmony's own parser rejects the no-space form). The grammar must
    # too — dropping straight from the name into <|constrain|> is rejected.
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather<|constrain|>",
    )
    assert accepted < total, "header missing the mandatory space was NOT rejected"


@_requires_llguidance
def test_named_choice_narrows_to_requested_tool(harmony_parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "get_time", harmony_parser)
    assert grammar is not None
    assert "get_time" in grammar
    assert "get_weather" not in grammar
    # A call to get_time is accepted...
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_time "
        '<|constrain|>json<|message|>{"tz": "UTC"}<|call|>',
    )
    assert accepted == total and accepting, "named get_time call rejected"
    # ...but a call to the OTHER tool (get_weather) is rejected under the named
    # get_time choice.
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        "<|channel|>commentary to=functions.get_weather",
    )
    assert accepted < total, "named get_time choice wrongly allowed get_weather"


@_requires_llguidance
def test_auto_opts_out_but_required_builds_on_real_parser(harmony_parser):
    # End-to-end opt-out parity on the REAL parser: auto declines (free-form),
    # required/named build. Mirrors the routing's builder_choice mapping.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    assert build_tool_grammar(TOOLS, "auto", harmony_parser) is None
    assert build_tool_grammar(TOOLS, "required", harmony_parser) is not None
    assert build_tool_grammar([TOOLS[0]], "get_weather", harmony_parser) is not None
