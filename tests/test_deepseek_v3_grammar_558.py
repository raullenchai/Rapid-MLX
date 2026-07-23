# SPDX-License-Identifier: Apache-2.0
"""Offline tests for grammar-constrained DeepSeek-V3 tool calling (#558 E1).

Extends the #558 constraint coverage from {qwen, hermes, gpt-oss, gemma4} to the
fourth top-tier wire family: the DeepSeek-V3 "section-wrapper" tool call. The
wire the V3 chat template emits (verified byte-for-byte against the real
DeepSeek-V3 tokenizer and copied VERBATIM from SGLang's
``deepseekv3_detector.structure_info``)::

    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {args}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>

SGLang folds BOTH the section envelope and the per-call envelope into a single
call's ``begin``/``end``; the tool NAME is a bare header identifier and the body
is a whole-object ``%json`` constrained by the tool's JSON Schema — so the
existing ``build_tool_lark`` needs NO change. These tests prove that path WITHOUT
a model or a decode loop:

  * the parser opts IN only when the tokenizer proves every one of the five
    fullwidth-pipe envelope markers is a single special token, and returns the
    exact SGLang-copied wire triple (pure-Python, hermetic tokenizer stub);
  * DISTILL OPT-OUT (the release-note nuance): the same V3 chat_template shipped
    on a **Qwen tokenizer** (``DeepSeek-R1-0528-Qwen3-8B`` /
    ``DeepSeek-R1-Distill-Qwen-*``) renders those markers as ordinary MULTI-token
    text, so ``are_single_special_tokens`` is False and ``structure_info()``
    correctly returns ``None`` (free-form-then-parse fallback). This locks E1 as
    a safe no-op on the Qwen-tokenizer distills — a regression guard;
  * grammar ENFORCEMENT via llguidance ``LLMatcher.consume_tokens`` on the REAL
    DeepSeek-V3 tokenizer: the ground-truth wire is accepted in full and
    terminates, while an off-schema argument, a bad enum value, and a
    hallucinated tool name are rejected mid-stream. This is the load-bearing
    #558 proof that the constraint is grammar-enforced, not merely post-parsed.

The enforcement tests need an ORIGINAL DeepSeek-V3-family tokenizer (whose five
section markers are single special tokens, ids 128806–128814) already in the
local HF cache. They probe the cache (never the network) for, in order,
``deepseek-ai/DeepSeek-V3`` (pinned), then ``deepseek-ai/DeepSeek-R1`` and
``deepseek-ai/DeepSeek-V3-0324``, and load the first hit with
``local_files_only=True``. They skip ONLY on genuine unavailability (the
``[guided]`` extra absent, or none of the candidates cached); a cached-but-broken
tokenizer FAILS loudly on load rather than skipping. The distill opt-out test uses
the LOCALLY CACHED ``mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit`` (tokenizer
only). The pure-Python opt-in/opt-out tests never skip.
"""

import importlib.util

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

# The five fullwidth-pipe (U+FF5C) section/per-call/sep markers. Single special
# tokens (ids 128806–128814) ONLY on the original DeepSeek-V3 / R1 tokenizers.
SENTINELS = (
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁sep｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁calls▁end｜>",
)

# Original DeepSeek-V3-family tokenizers, tried in order. ALL pinned by revision
# for an IMMUTABLE enforcement artifact (codex: an unpinned fallback would let a
# mutable upstream retag change what this auto-deploy test enforces). DeepSeek-V3
# is the primary; R1 / V3-0324 are same-layout fallbacks for a box where only one
# is cached. All three are PUBLIC (ungated) and share the section-marker layout.
_TOKENIZER_CANDIDATES = (
    ("deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb"),
    ("deepseek-ai/DeepSeek-R1", "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"),
    ("deepseek-ai/DeepSeek-V3-0324", "e9b33add76883f293d6bf61f6bd89b497e80e335"),
)

# The cached Qwen-tokenizer distill — its section markers are multi-token TEXT, so
# the parser must OPT OUT. Locks the release-note "safe no-op on distills" nuance.
# Revision-pinned (like the enforcement candidates) so the regression anchor is
# reproducible rather than tied to whatever mutable ``main`` snapshot is cached.
_DISTILL_TOKENIZER = "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
_DISTILL_REVISION = "4e0d3848a0ad8f9fb54638891e4928f04fcca978"

# get_weather: required string + optional enum (exercises %json string, enum, and
# required-vs-optional). get_time: a second tool for the named-choice narrowing.
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


def _make_parser(tokenizer=None):
    from vllm_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser

    return DeepSeekV3ToolParser(tokenizer=tokenizer)


def _wire(name="get_weather", args='{"city": "Paris"}'):
    """The DeepSeek-V3 ground-truth section-wrapper wire for one call."""
    return (
        f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n"
        f"```json\n{args}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )


# --------------------------------------------------------------------------
# Pure-Python opt-in / opt-out contract (always runs — no tokenizer / network).
# Mirrors the qwen3coder hermetic ``_FakeTokenizer`` (models exactly the surfaces
# ``are_single_special_tokens`` probes: single ADDED tokens that round-trip).
# --------------------------------------------------------------------------
class _FakeAddedToken:
    def __init__(self, content, special=False):
        self.content = content
        self.special = special


class _FakeTokenizer:
    def __init__(self, added=None):
        self._added = dict(added or {})
        self._id_to_str = {i: s for s, i in self._added.items()}
        self.added_tokens_decoder = {
            i: _FakeAddedToken(s, special=False) for s, i in self._added.items()
        }

    def encode(self, text, add_special_tokens=False):
        if text in self._added:
            return [self._added[text]]
        return [0, 1]  # ordinary multi-token text

    def decode(self, ids):
        return "".join(self._id_to_str.get(i, "<unk>") for i in ids)

    def get_vocab(self):
        return dict(self._added)


def _single_token_tokenizer():
    # The real DeepSeek-V3 layout: each of the five markers is one added token.
    return _FakeTokenizer(added=dict(zip(SENTINELS, range(128806, 128806 + 5))))


def test_structure_info_opts_out_without_tokenizer():
    # No tokenizer -> cannot prove single-token sentinels -> opt out (None).
    assert _make_parser(tokenizer=None).structure_info() is None


def test_structure_info_opts_out_on_multitoken_tokenizer():
    # A tokenizer that encodes the markers as ordinary multi-token text -> opt out
    # rather than build an unenforceable special-token grammar.
    assert _make_parser(tokenizer=_FakeTokenizer(added={})).structure_info() is None


def test_structure_info_returns_deepseek_wire_triple():
    from vllm_mlx.api.tool_grammar import StructureInfo

    get_info = _make_parser(tokenizer=_single_token_tokenizer()).structure_info()
    assert callable(get_info), "opt-in must return a name->StructureInfo factory"
    si = get_info("get_weather")
    assert isinstance(si, StructureInfo)
    # The SGLang-copied triple, byte-exact.
    assert si.begin == (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        "get_weather\n```json\n"
    )
    assert si.end == "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    assert si.trigger == "<｜tool▁calls▁begin｜>"
    # Builder invariants enforced by build_tool_lark.
    assert si.begin.startswith(si.trigger)
    assert si.trigger in si.sentinels
    # All five envelope markers are declared sentinels (special-token refs).
    assert si.sentinels == SENTINELS
    # JSON body (the default arg_style) — the arguments are a whole-object %json.
    assert si.arg_style == "json"


def test_deepseek_family_is_auto_safe_by_default():
    # DeepSeek does NOT override TOOL_GRAMMAR_AUTO_SAFE — its trigger
    # (<｜tool▁calls▁begin｜>) is a dedicated tool-call boundary, so it defaults
    # auto-safe like hermes/qwen (unlike harmony's shared <|channel|>).
    from vllm_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser

    assert DeepSeekV3ToolParser.TOOL_GRAMMAR_AUTO_SAFE is True


@_requires_llguidance
def test_build_tool_lark_builds_from_deepseek_triple():
    # The whole point of E1: the EXISTING builder consumes the DeepSeek triple
    # unchanged. build_tool_lark must produce a grammar with the section markers
    # as BARE special-token refs (never quoted byte literals) and a %json body.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    si = _make_parser(tokenizer=_single_token_tokenizer()).structure_info()(
        "get_weather"
    )
    lark = build_tool_lark(TOOLS[:1], "required", [si], single_call=True)
    assert isinstance(lark, str) and lark.strip()
    # Section markers as bare refs, not quoted literals a single token can't match.
    assert " <｜tool▁calls▁begin｜> " in lark
    assert '"<｜tool▁calls▁begin｜>"' not in lark
    assert '"<｜tool▁calls▁end｜>"' not in lark
    # JSON body is a %json object; the fenced-JSON frame is byte-string literals.
    assert "%json" in lark
    assert "```json" in lark


def test_parser_declares_section_wrapper_flag():
    # The section-wrapper soundness flag drives build_tool_grammar's >1-call
    # opt-out (finding 1). It must be set on the class.
    from vllm_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser

    assert DeepSeekV3ToolParser.TOOL_GRAMMAR_SECTION_WRAPPER is True
    # And the grammar-capability marker (#1144) must MATCH the structure_info
    # override — declared True so the marker-consistency check passes and the
    # #561 oversized-schema route stays on the constrained path.
    assert DeepSeekV3ToolParser.SUPPORTS_GRAMMAR is True
    assert DeepSeekV3ToolParser.supports_grammar() is True


@_requires_llguidance
def test_section_wrapper_gate_opts_out_when_multicall_possible():
    # FINDING 1 (soundness). The section-wrapper wire folds the WHOLE tool-calls
    # envelope into each call, so a repeated grammar tag would emit back-to-back
    # sections the single-envelope parser drops after the first. build_tool_grammar
    # must OPT OUT (None) whenever >1 call is possible (``not single_call`` ->
    # ``+``/``*``) and build the grammar ONLY on the at-most-one-call path
    # (``single_call=True``). Hermetic single-token tokenizer -> runs with no
    # network. Covers required / named / auto.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    parser = _make_parser(tokenizer=_single_token_tokenizer())
    for choice in ("required", "get_weather", "auto"):
        assert (
            build_tool_grammar(TOOLS[:1], choice, parser, single_call=False) is None
        ), f"{choice}: multi-call grammar must opt out (section-wrapper soundness)"
        assert (
            build_tool_grammar(TOOLS[:1], choice, parser, single_call=True) is not None
        ), f"{choice}: at-most-one-call grammar must build"


# --------------------------------------------------------------------------
# DISTILL OPT-OUT on the REAL cached Qwen-tokenizer distill (locks finding ①).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def distill_tok():
    transformers = pytest.importorskip("transformers")
    if _cached_repo([(_DISTILL_TOKENIZER, _DISTILL_REVISION)]) is None:
        pytest.skip(
            f"{_DISTILL_TOKENIZER} tokenizer not in the local HF cache — the "
            "distill opt-out test requires it locally"
        )
    # Cache-hit confirmed -> load offline; any load failure PROPAGATES (a corrupt
    # or incomplete cached tokenizer must fail loudly, never a false-green skip).
    return transformers.AutoTokenizer.from_pretrained(
        _DISTILL_TOKENIZER, revision=_DISTILL_REVISION, local_files_only=True
    )


def test_distill_qwen_tokenizer_markers_are_multitoken(distill_tok):
    # The V3 markers on a Qwen tokenizer are ordinary multi-token TEXT — the exact
    # condition that must drive the parser to opt out.
    from vllm_mlx.api.tool_grammar import are_single_special_tokens

    assert are_single_special_tokens(distill_tok, SENTINELS) is False


def test_distill_qwen_tokenizer_opts_out_to_none(distill_tok):
    # THE regression guard for the release-note nuance: on a Qwen-tokenizer distill
    # carrying the V3 chat_template, structure_info() returns None (safe no-op),
    # NOT a grammar the tokenizer could never satisfy.
    assert _make_parser(tokenizer=distill_tok).structure_info() is None


# --------------------------------------------------------------------------
# ENFORCEMENT against a REAL original DeepSeek-V3 tokenizer.
# --------------------------------------------------------------------------
def _cached_repo(candidates):
    """First ``(repo, revision)`` whose tokenizer snapshot is in the local HF cache.

    A DETERMINISTIC, network-free cache probe (``huggingface_hub`` returns a str
    path for a cached file, a sentinel/``None`` otherwise). It REPLACES the earlier
    approach of catching ``from_pretrained`` errors and classifying their messages
    as offline-vs-corrupt: that message-signature split couldn't reliably tell a
    genuine cache-miss from a corrupt/incomplete cached tokenizer, and the
    network-capable load added timeout/connection failure modes (codex r2/r4).
    Callers instead (1) probe the cache — a miss is an explicit typed skip; and
    (2) on a hit, load ``local_files_only=True`` and let ANY failure PROPAGATE, so
    a corrupt/incomplete snapshot fails loudly. No network, no message matching.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - hub too old for the cache probe
        return None
    for repo, revision in candidates:
        hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
        # A real cached file path -> hit. None / _CACHED_NO_EXIST sentinel -> miss.
        if isinstance(hit, str):
            return repo, revision
    return None


@pytest.fixture(scope="module")
def tok():
    """Load the first cached original DeepSeek-V3-family tokenizer.

    Probes the local HF cache for DeepSeek-V3, then R1 / V3-0324 (all pinned) and
    loads the first hit offline (``local_files_only=True``). Tokenizer + config
    files only — never weights, never the network. Skips only when NONE of the
    candidates is cached; a corrupt/incomplete cached snapshot fails loudly on
    load (never a false-green skip).
    """
    transformers = pytest.importorskip("transformers")
    cached = _cached_repo(_TOKENIZER_CANDIDATES)
    if cached is None:
        pytest.skip(
            "no original DeepSeek-V3-family tokenizer cached "
            f"({', '.join(r for r, _ in _TOKENIZER_CANDIDATES)}) — E1 "
            "enforcement tests require one locally"
        )
    repo, revision = cached
    return transformers.AutoTokenizer.from_pretrained(
        repo, revision=revision, local_files_only=True
    )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer via the engine's own resolver.

    Once ``tok`` (a real cached DeepSeek-V3 tokenizer) is available, the runtime
    bridge MUST yield an ``LLTokenizer`` — a ``None`` here would mean the
    production resolver regressed and DeepSeek grammar constraint is SILENTLY
    disabled, so we FAIL rather than skip (skipping would let the enforcement
    suite go green while the feature is broken). The narrow "bridge not
    installed" case is the only sanctioned skip.
    """
    from vllm_mlx.api.tool_grammar import HAS_LL_TOKENIZER, build_lltokenizer

    if not HAS_LL_TOKENIZER:
        pytest.skip(
            "llguidance runtime bridge (llguidance.hf / LLTokenizer) not "
            "installed — DeepSeek enforcement tests require it"
        )
    lltokenizer = build_lltokenizer(tok)
    assert lltokenizer is not None, (
        "build_lltokenizer returned None for the real DeepSeek-V3 tokenizer with "
        "the runtime bridge available — the tokenizer->llguidance integration is "
        "BROKEN (this must FAIL, not skip)."
    )
    return lltokenizer


@pytest.fixture(scope="module")
def parser(tok):
    return _make_parser(tok)


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)``.

    Advances real grammar state one token at a time via ``consume_tokens``. A
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
def test_real_tokenizer_markers_are_single_special_tokens(tok):
    # The enforcement anchor: on the original DeepSeek-V3 tokenizer every section
    # marker IS a single special token, so the parser opts in.
    from vllm_mlx.api.tool_grammar import are_single_special_tokens

    assert are_single_special_tokens(tok, SENTINELS) is True


@_requires_llguidance
def test_structure_info_opts_in_on_real_tokenizer(parser):
    get_info = parser.structure_info()
    assert get_info is not None, "parser must opt IN on the real DeepSeek tokenizer"
    si = get_info("get_weather")
    assert si.begin.startswith(si.trigger)
    assert si.trigger == "<｜tool▁calls▁begin｜>"
    assert si.sentinels == SENTINELS


@_requires_llguidance
def test_valid_deepseek_call_is_accepted_and_terminates(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS[:1], "required", parser, single_call=True)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire())
    assert accepted == total, f"valid DeepSeek call rejected ({accepted}/{total})"
    assert accepting, "valid complete DeepSeek call is not an accepting state"


@_requires_llguidance
def test_section_wrapper_gate_on_real_tokenizer(parser):
    # FINDING 1 on the REAL parser/tokenizer: multi-call opts out, single_call
    # builds — for required AND auto. Mirrors the hermetic gate test end-to-end.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    for choice in ("required", "auto"):
        assert (
            build_tool_grammar(TOOLS[:1], choice, parser, single_call=False) is None
        ), f"{choice}: multi-call grammar must opt out on the real tokenizer"
        assert (
            build_tool_grammar(TOOLS[:1], choice, parser, single_call=True) is not None
        ), f"{choice}: at-most-one-call grammar must build on the real tokenizer"


@_requires_llguidance
def test_valid_enum_value_is_accepted(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS[:1], "required", parser, single_call=True)
    accepted, total, accepting = _consume(
        grammar, lltok, tok, _wire(args='{"city": "P", "unit": "celsius"}')
    )
    assert accepted == total, f"valid enum value rejected ({accepted}/{total})"
    assert accepting, "valid enum call is not an accepting state"


@_requires_llguidance
def test_off_schema_argument_is_rejected(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS[:1], "required", parser, single_call=True)
    # `city` must be a string; an integer must be forbidden. Feed a prefix up to
    # the bad byte so the rejection is unambiguous.
    bad = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        'get_weather\n```json\n{"city": 4'
    )
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "off-schema integer argument was NOT rejected"


@_requires_llguidance
def test_bad_enum_value_is_rejected(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS[:1], "required", parser, single_call=True)
    # `unit` enum is {celsius, fahrenheit}; "kelvin" must be forbidden.
    bad = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        'get_weather\n```json\n{"city": "P", "unit": "kelvin'
    )
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "invalid enum value was NOT rejected by the grammar"


@_requires_llguidance
def test_hallucinated_tool_name_is_rejected(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS[:1], "required", parser, single_call=True)
    # Only get_weather is offered; the header name get_stock must be masked.
    bad = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock"
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "hallucinated tool name was NOT rejected"


@_requires_llguidance
def test_named_choice_narrows_to_requested_tool(parser, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    # Pass the COMPLETE tools list (both get_weather AND get_time) so the named
    # choice actually EXERCISES build_tool_grammar's internal narrowing — passing
    # only [get_time] would let this test pass even if narrowing were broken.
    grammar = build_tool_grammar(TOOLS, "get_time", parser, single_call=True)
    assert grammar is not None
    assert "get_time" in grammar
    assert "get_weather" not in grammar
    # A call to get_time is accepted + terminal...
    accepted, total, accepting = _consume(
        grammar, lltok, tok, _wire(name="get_time", args='{"tz": "UTC"}')
    )
    assert accepted == total and accepting, "named get_time call rejected"
    # ...but a call to the OTHER tool is rejected under the named get_time choice.
    bad = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather"
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "named get_time choice wrongly allowed get_weather"
