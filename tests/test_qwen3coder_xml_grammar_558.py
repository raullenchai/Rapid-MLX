# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Qwen3-Coder XML grammar constraint (#558 E3).

The JSON-body families (hermes / qwen / harmony) constrain their ``arguments``
as a single ``%json`` object. Qwen3-Coder instead emits an XML arg body —
``<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n...</function>`` — so
its ``structure_info()`` declares ``arg_style="xml"`` and ``build_tool_lark``
emits a per-parameter XML body (``_emit_xml_arg_body`` / ``_emit_xml_param_value``)
instead of a whole-object ``%json``. These tests prove that path WITHOUT a model
or a decode loop:

  * ``build_tool_lark`` golden output for ``arg_style="xml"`` (function/parameter
    frame, enum alternation, ``%json`` values with ``$defs``/``$ref``
    propagation, required vs optional ``( )?``);
  * the #1996 boundary: a tool with a TOP-LEVEL free-form string is not
    constrained at all. The value has no delimiter of its own and this wire's
    close is ordinary text, so no terminal can be lexed against it — the value
    rule this file once tested (first ``/[^<]*/``, then a lazy lexeme, finally
    ``%json``) has no correct form, and forcing ``%json`` silently truncated real
    output. Enforcement therefore drives ``XML_TOOLS_CONSTRAINABLE`` (enum +
    scalars + frame, no free string); ``test_qwen3coder_xml_freeform_string_1996``
    owns the opt-out itself;
  * grammar ENFORCEMENT via llguidance ``LLMatcher`` on the REAL Qwen3-Coder
    tokenizer: a valid XML call is accepted + terminal, prose before a forced
    call is masked at token 0, and a bad enum / off-schema scalar is rejected;
  * ROUND-TRIP: the ``qwen3_coder_xml`` parser parses the wire back to
    ``{name, arguments}`` with correct types (str / int / bool / nested object).
    The PARSER still accepts both the raw and the quoted surface — only the
    grammar stopped forcing the quoted one;
  * REGRESSION GUARD: an ``arg_style="json"`` (hermes/qwen/harmony) build is
    byte-identical to before — it never emits the XML string constructs
    (``XML_PARAM_TEXT`` / ``xml_param_value``) and still uses ``%json``.

The enforcement / round-trip tests use the ACTUAL target model tokenizer
``mlx-community/Qwen3-Coder-Next-4bit`` (pinned by revision below for an
immutable artifact — its ``<tool_call>``/``</tool_call>`` single-special-token
layout and the inner XML byte markers are fixed at this commit). They skip ONLY
on genuine unavailability (llguidance extra absent, or the tokenizer neither
cached nor reachable); any OTHER failure is surfaced, not swallowed. The
pure-Python golden / structure-triple / regression tests never skip.
"""

import importlib.util
import json

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

# The REAL Qwen3-Coder target model (the pilot verified the XML wire on it).
# Pin the revision so enforcement runs against an IMMUTABLE artifact.
_TOKENIZER_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
_TOKENIZER_REVISION = "7b9321eabb85ce79625cac3f61ea691e4ea984b5"

# A representative XML tool: required string + required enum + optional int +
# optional bool — exercises every ``_emit_xml_param_value`` branch AND
# required-vs-optional framing at the EMITTER level. Since #1996 its top-level
# ``code`` string means the GUARD opts it out, so it drives the golden/emitter
# tests and the opt-out test, never the enforcement tests.
XML_TOOLS = [
    {
        "name": "run_code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "enum": ["python", "cpp"]},
                "timeout": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
            "required": ["code", "language"],
            "additionalProperties": False,
        },
    },
]

# The same tool WITHOUT a top-level free-form string. Since #1996 that string is
# what opts a tool out of grammar entirely (the wire cannot frame a bare value —
# see ``test_qwen3coder_xml_freeform_string_1996.py``), so every ENFORCEMENT test
# below drives this shape instead: it still exercises the required enum, the
# optional ``%json`` scalars, and the call frame — everything the grammar can
# still constrain.
XML_TOOLS_CONSTRAINABLE = [
    {
        "name": "run_code",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["python", "cpp"]},
                "timeout": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
            "required": ["language"],
            "additionalProperties": False,
        },
    },
]

# A nested-object tool with a ``$ref`` into ``$defs`` — proves ``$defs`` is
# propagated into the per-value ``%json`` sub-schema so the ``$ref`` resolves.
XML_REF_TOOL = [
    {
        "name": "place",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"$ref": "#/$defs/point"}},
            "required": ["origin"],
            "$defs": {
                "point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
]


# A tool whose ONLY property is a ``$ref`` resolving to a STRING (F3). The fix
# resolves the ``$ref`` BEFORE the string-vs-``%json`` decision, so it is treated
# exactly like an inline string. Since #1996 that means it opts OUT of grammar —
# resolution order still decides, it just now decides opt-out.
XML_REF_STRING_TOOL = [
    {
        "name": "geo",
        "parameters": {
            "type": "object",
            "properties": {"city": {"$ref": "#/$defs/City"}},
            "required": ["city"],
            "$defs": {"City": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


def _xml_structure_info(name: str):
    """The Qwen3-Coder XML wire triple, exactly as ``Qwen3CoderToolParser``
    ships it (``arg_style="xml"``). Declared test-locally so the pure-Python
    golden tests need no tokenizer; the enforcement tests below drive the REAL
    parser instead."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f"<tool_call>\n<function={name}>\n",
        end="</function>\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
        arg_style="xml",
    )


def _hermes_json_structure_info(name: str):
    """A hermes ``<tool_call>`` JSON-body wire triple (``arg_style="json"``,
    the default) — the regression baseline the XML change must not perturb."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
        end="}\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
    )


# --------------------------------------------------------------------------
# Golden Lark for the XML arg body (pure Python, always runs). A CHECKED-IN
# golden — comparing against it (not another call of the same implementation)
# pins the EXACT emitted grammar even if the whole builder drifted.
# --------------------------------------------------------------------------
_XML_GOLDEN_LARK = (
    "%llguidance {}\n"
    "start: (tag_0) (SEP (tag_0))* tag_end\n"
    "tag_end: TAG_TEXT\n"
    "SEP: /[ \\t\\r\\n]*/\n"
    "TAG_TEXT: /(.|\\n)*/\n"
    "\n"
    # Strings ride %json so delimiter-looking bytes are payload inside the
    # quoted JSON string rather than the XML close (#1542).
    'tag_0: <tool_call> "\\n<function=run_code>\\n" '
    '"<parameter=code>\\n" %json {"type": "string"} "\\n</parameter>\\n" '
    '"<parameter=language>\\n" ("python" | "cpp") "\\n</parameter>\\n" '
    '( "<parameter=timeout>\\n" %json {"type": "integer"} "\\n</parameter>\\n" )? '
    '( "<parameter=verbose>\\n" %json {"type": "boolean"} "\\n</parameter>\\n" )? '
    '"</function>\\n" </tool_call>\n'
)


def test_xml_lark_matches_golden():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert lark == _XML_GOLDEN_LARK


def test_xml_lark_frame_and_sentinels():
    # The function/parameter frame: <tool_call> trigger + </tool_call> close as
    # BARE special-token refs (never quoted byte literals the single token could
    # not satisfy), the <function=NAME> header, and per-parameter blocks.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert " <tool_call> " in lark  # bare trigger ref
    assert lark.rstrip().endswith("</tool_call>")  # bare closing ref
    assert '"<tool_call>"' not in lark  # NOT a quoted literal
    assert '"</tool_call>"' not in lark
    # XML frame markers are ordinary byte-string literals (multi-token text).
    assert '"\\n<function=run_code>\\n"' in lark
    assert '"<parameter=code>\\n"' in lark
    assert '"</function>\\n"' in lark


def test_xml_string_value_uses_json_string_not_delimiter_lazy_rule():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert '"<parameter=code>\\n" %json {"type": "string"}' in lark
    assert "xml_param_value" not in lark
    assert "XML_PARAM_TEXT" not in lark
    assert "XMLSTR" not in lark
    assert "/[^<]*/" not in lark


def test_xml_required_vs_optional_framing():
    # Required params are mandatory (no quantifier); optional params are wrapped
    # in ``( ... )?``. ``code``/``language`` required; ``timeout``/``verbose`` not.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    # Required string: bare (not inside a ``( ... )?`` group).
    assert (
        '"<parameter=code>\\n" %json {"type": "string"} '
        '"\\n</parameter>\\n" "<parameter=language>' in lark
    )
    # Optional scalars: each wrapped in an optional group.
    assert (
        '( "<parameter=timeout>\\n" %json {"type": "integer"} "\\n</parameter>\\n" )?'
        in lark
    )
    assert (
        '( "<parameter=verbose>\\n" %json {"type": "boolean"} "\\n</parameter>\\n" )?'
        in lark
    )


def test_xml_enum_is_literal_alternation():
    # An enum value renders as an alternation of the literal enum values (raw
    # string form for string enums), NOT ``%json`` and NOT the lazy string rule.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_TOOLS, "required", [_xml_structure_info("run_code")])
    assert '("python" | "cpp") "\\n</parameter>\\n"' in lark


def test_xml_ref_defs_propagated_into_value_schema():
    # A ``$ref`` value must carry the parent's ``$defs`` into its per-value
    # ``%json`` sub-schema so the ``$ref`` resolves. Golden-compare the emitted
    # ``%json`` payload.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_REF_TOOL, "required", [_xml_structure_info("place")])
    expected_schema = {
        "$ref": "#/$defs/point",
        "$defs": {
            "point": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
    assert f"%json {json.dumps(expected_schema)}" in lark


def test_xml_no_string_param_tool_has_no_string_helper_rule():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    int_only = [
        {
            "name": "cfg",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        }
    ]
    lark = build_tool_lark(int_only, "required", [_xml_structure_info("cfg")])
    assert "xml_param_value" not in lark
    assert "XML_PARAM_TEXT" not in lark


# --------------------------------------------------------------------------
# REGRESSION GUARD: arg_style="json" (hermes/qwen/harmony) is byte-identical —
# the XML change must not leak the XML string constructs into JSON families.
# --------------------------------------------------------------------------
def test_json_family_grammar_has_no_xml_constructs():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_hermes_json_structure_info(t["name"]) for t in XML_TOOLS]
    lark = build_tool_lark(XML_TOOLS, "required", infos)
    # JSON body: a single ``%json`` object, none of the XML string constructs.
    assert "%json" in lark
    assert "XML_PARAM_TEXT" not in lark
    assert "xml_param_value" not in lark
    assert "XMLSTR" not in lark
    assert "<parameter=" not in lark


def test_json_family_grammar_byte_identical_to_pre_xml_baseline():
    # The exact hermes forced golden (identical to the checked-in golden in
    # test_tool_grammar_558.py). Pinning it here proves the XML feature left the
    # JSON-family grammar byte-for-byte unchanged.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    tools = [
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]
    infos = [_hermes_json_structure_info("get_weather")]
    expected = (
        "%llguidance {}\n"
        "start: (tag_0) (SEP (tag_0))* tag_end\n"
        "tag_end: TAG_TEXT\n"
        "SEP: /[ \\t\\r\\n]*/\n"
        "TAG_TEXT: /(.|\\n)*/\n"
        "\n"
        'tag_0: <tool_call> "\\n{\\"name\\": \\"get_weather\\", '
        '\\"arguments\\": " %json {"type": "object", "properties": {"city": '
        '{"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, '
        '"required": ["city"], "additionalProperties": false} "}\\n" </tool_call>\n'
    )
    assert build_tool_lark(tools, "required", infos) == expected


# --------------------------------------------------------------------------
# Real Qwen3CoderToolParser.structure_info() (pure Python, hermetic tokenizer
# stub — no network). Proves the parser opts in ONLY when the tokenizer proves
# both sentinels are single special tokens, and returns arg_style="xml".
# --------------------------------------------------------------------------
class _FakeAddedToken:
    def __init__(self, content, special=False):
        self.content = content
        self.special = special


class _FakeTokenizer:
    """Models the surfaces the ``are_single_special_tokens`` guard probes:
    ``<tool_call>``/``</tool_call>`` as distinct single ADDED (special=False)
    tokens that round-trip (the real Qwen3-Coder layout: ids 151657/151658)."""

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
    return _FakeTokenizer(added={"<tool_call>": 151657, "</tool_call>": 151658})


def _make_qwen3coder(tokenizer=None):
    from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    return Qwen3CoderToolParser(tokenizer=tokenizer)


def test_qwen3coder_structure_info_opts_out_without_tokenizer():
    # No tokenizer -> cannot prove single-token sentinels -> opt out (None).
    assert _make_qwen3coder(tokenizer=None).structure_info() is None


def test_qwen3coder_structure_info_opts_out_on_multitoken_tokenizer():
    # A tokenizer that encodes <tool_call> as ordinary multi-token text -> opt
    # out rather than build an unenforceable special-token grammar.
    assert _make_qwen3coder(tokenizer=_FakeTokenizer(added={})).structure_info() is None


def test_qwen3coder_structure_info_returns_xml_wire_triple():
    from vllm_mlx.api.tool_grammar import StructureInfo

    get_info = _make_qwen3coder(tokenizer=_single_token_tokenizer()).structure_info()
    assert callable(get_info), "opt-in must return a name->StructureInfo factory"
    si = get_info("run_code")
    assert isinstance(si, StructureInfo)
    assert si.arg_style == "xml"  # the load-bearing distinction from hermes/qwen
    assert si.trigger == "<tool_call>"
    assert si.begin == "<tool_call>\n<function=run_code>\n"
    assert si.end == "</function>\n</tool_call>"
    assert si.begin.startswith(si.trigger)  # builder invariant
    assert si.sentinels == ("<tool_call>", "</tool_call>")
    assert si.trigger in si.sentinels


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT + ROUND-TRIP on the REAL Qwen3-Coder tokenizer.
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    """Genuine network/cache-miss exceptions that are a sanctioned skip.

    A corrupt tokenizer artifact / invalid revision must FAIL the test, not
    skip it, so we skip ONLY on the specific offline/cache-miss signals.
    """
    types: list[type[BaseException]] = []
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:  # pragma: no cover - old hub without these names
        pass
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr

        types.append(_ReqConnErr)
    except Exception:  # pragma: no cover - requests not present
        pass
    return tuple(types) or (OSError,)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION
        )
    except _offline_skip_exc_types():  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not cached "
            "and no network — XML enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer via the
    module's own resolver (the spike-proven candidate-3 direct build handles the
    transformers ``from_tokenizer`` isinstance gotcha ``guided.py`` trips on).

    FINDING 5: a blanket ``skip when build_lltokenizer() is None`` let the whole
    enforcement/round-trip suite go GREEN even if the production
    tokenizer->llguidance integration was BROKEN. We now sanction the skip ONLY
    when the runtime bridge is genuinely UNAVAILABLE (``HAS_LL_TOKENIZER`` False —
    ``llguidance.hf`` / ``LLTokenizer`` not importable, the same narrow
    unavailability the ``tok`` fixture treats as an offline skip). When the bridge
    IS importable AND ``tok`` loaded (cached) but ``build_lltokenizer`` returns
    ``None``, that is a broken integration and MUST FAIL — never silently pass."""
    from vllm_mlx.api.tool_grammar import HAS_LL_TOKENIZER, build_lltokenizer

    if not HAS_LL_TOKENIZER:
        pytest.skip(
            "llguidance runtime bridge (llguidance.hf / LLTokenizer) not "
            "installed — XML enforcement tests require it"
        )
    built = build_lltokenizer(tok)
    assert built is not None, (
        "build_lltokenizer() returned None for the CACHED Qwen3-Coder tokenizer "
        "with the llguidance runtime bridge available — the production "
        "tokenizer->llguidance integration is BROKEN (finding 5: this must FAIL, "
        "not skip)."
    )
    return built


def _xml_grammar(tools, tool_choice, tok):
    """Compile the XML grammar through the REAL parser (opted-in on ``tok``)."""
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    return build_tool_grammar(tools, tool_choice, _make_qwen3coder(tok))


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)`` —
    advances real matcher state so ``is_accepting()`` proves a COMPLETE valid
    derivation, not merely an accepted prefix."""
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


def _wire(code, *, language="python", timeout=None, verbose=None):
    """Build the Qwen3-Coder XML wire for the ``run_code`` tool."""
    s = (
        "<tool_call>\n<function=run_code>\n"
        f"<parameter=code>\n{json.dumps(code, ensure_ascii=False)}\n</parameter>\n"
        f"<parameter=language>\n{language}\n</parameter>\n"
    )
    if timeout is not None:
        s += f"<parameter=timeout>\n{timeout}\n</parameter>\n"
    if verbose is not None:
        s += f"<parameter=verbose>\n{verbose}\n</parameter>\n"
    s += "</function>\n</tool_call>"
    return s


def _wire_c(*, language="python", timeout=None, verbose=None):
    """The ``XML_TOOLS_CONSTRAINABLE`` wire — same frame, no free-form string."""
    s = (
        "<tool_call>\n<function=run_code>\n"
        f"<parameter=language>\n{language}\n</parameter>\n"
    )
    if timeout is not None:
        s += f"<parameter=timeout>\n{timeout}\n</parameter>\n"
    if verbose is not None:
        s += f"<parameter=verbose>\n{verbose}\n</parameter>\n"
    s += "</function>\n</tool_call>"
    return s


@_requires_llguidance
def test_finding5_tokenizer_llguidance_integration_not_broken(tok):
    # FINDING 5: with llguidance importable AND the tokenizer loaded (cached), the
    # production ``build_lltokenizer`` MUST yield a real ``LLTokenizer`` — a
    # ``None`` here means the tokenizer->llguidance integration is broken and the
    # enforcement suite would silently pass on skips. Assert it directly so the
    # breakage surfaces as a FAILURE (only ``tok``'s narrow offline/cache-miss
    # path is a sanctioned skip; we are here only because ``tok`` loaded).
    from vllm_mlx.api.tool_grammar import HAS_LL_TOKENIZER, build_lltokenizer

    if not HAS_LL_TOKENIZER:
        pytest.skip("llguidance runtime bridge (llguidance.hf / LLTokenizer) absent")
    assert build_lltokenizer(tok) is not None, (
        "build_lltokenizer() returned None despite an available runtime bridge "
        "and a loaded tokenizer — broken integration (finding 5), not a skip."
    )


@_requires_llguidance
def test_xml_valid_call_accepted_and_terminates(tok, lltok):
    grammar = _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire_c())
    assert accepted == total, f"valid XML call rejected ({accepted}/{total})"
    assert accepting, "valid complete XML call is not an accepting (terminal) state"


@_requires_llguidance
def test_xml_tool_with_a_freeform_string_is_not_constrained_at_all(tok):
    """#1996, on the real tokenizer through the real builder.

    ``XML_TOOLS`` carries a top-level free-form ``code`` string. The wire cannot
    frame a bare value there and the ``%json`` surface silently truncated real
    output, so the whole request must fall back to free-form. This supersedes the
    three enforcement tests that used to drive string values through this
    grammar (``<``-containing code, a plain value, an embedded ``</parameter>``):
    those exercised a constrained string path that no longer exists — the
    property they were protecting (a value is never truncated) is now held by
    NOT constraining it. See ``test_qwen3coder_xml_freeform_string_1996.py``.
    """
    assert _xml_grammar(XML_TOOLS, "required", tok) is None


@_requires_llguidance
def test_xml_optional_and_scalar_params_enforced(tok, lltok):
    # Optional int/bool params present with valid scalar surface forms are
    # accepted + terminal; the enum on the required ``language`` is honored.
    grammar = _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok)
    accepted, total, accepting = _consume(
        grammar, lltok, tok, _wire_c(language="cpp", timeout=30, verbose="true")
    )
    assert accepted == total and accepting, (
        f"valid call with optional scalars rejected ({accepted}/{total})"
    )


@_requires_llguidance
def test_xml_bad_enum_value_is_rejected(tok, lltok):
    # ``language`` enum is {python, cpp}; "rust" must be masked.
    grammar = _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok)
    accepted, total, _ = _consume(grammar, lltok, tok, _wire_c(language="rust"))
    assert accepted < total, "invalid enum value was NOT rejected by the grammar"


@_requires_llguidance
def test_xml_off_schema_scalar_is_rejected(tok, lltok):
    # ``timeout`` is an integer; a non-numeric value must be masked by ``%json``.
    grammar = _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok)
    bad = (
        "<tool_call>\n<function=run_code>\n"
        "<parameter=language>\npython\n</parameter>\n"
        "<parameter=timeout>\nnot_an_int"
    )
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "off-schema (non-integer) timeout was NOT rejected"


@_requires_llguidance
def test_xml_forced_rejects_prose_before_the_call(tok, lltok):
    # Forced (required) non-reasoning: the first call sits AT the trigger with no
    # free prefix, so bare prose before it is masked at token 0.
    grammar = _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok)
    assert grammar is not None
    prose_then_call = "Sure, let me run that. " + _wire_c()
    accepted, _total, _ = _consume(grammar, lltok, tok, prose_then_call)
    assert accepted == 0, (
        f"forced XML grammar accepted {accepted} prose token(s) before the "
        "trigger — the unbounded leading prefix is back (#558 forced-leak)"
    )


# --------------------------------------------------------------------------
# ROUND-TRIP: the qwen3_coder_xml parser parses the constrained wire back to
# {name, arguments} with correct types — the surface forms the grammar emits
# are exactly what the parser type-converts.
# --------------------------------------------------------------------------
def _parse(wire, tools):
    parser = _make_qwen3coder(tokenizer=None)
    req = {
        "tools": [
            {
                "type": "function",
                "function": {"name": t["name"], "parameters": t["parameters"]},
            }
            for t in tools
        ]
    }
    res = parser.extract_tool_calls(wire, request=req)
    assert res.tools_called, "parser did not detect the tool call"
    tc = res.tool_calls[0]
    return tc["name"], json.loads(tc["arguments"])


@pytest.mark.parametrize("code", ["a < b && c > d", "vector<int> v", "print('ok')"])
def test_roundtrip_string_value_with_angle_bracket(code):
    # The constrained wire round-trips back to the EXACT string value (including
    # ``<``) — the grammar and parser agree on the surface form.
    name, args = _parse(_wire(code), XML_TOOLS)
    assert name == "run_code"
    assert args["code"] == code
    assert args["language"] == "python"


def test_roundtrip_scalar_types_int_bool():
    # int / bool params type-convert correctly (not left as strings).
    name, args = _parse(
        _wire("x", language="cpp", timeout=30, verbose="true"), XML_TOOLS
    )
    assert args["timeout"] == 30 and isinstance(args["timeout"], int)
    assert args["verbose"] is True
    assert args["language"] == "cpp"


def test_roundtrip_nested_object_value():
    # A nested-object ($ref) value round-trips into a JSON object with typed
    # fields — the ``%json`` surface form the grammar emits is what the parser
    # json.loads-decodes.
    wire = (
        "<tool_call>\n<function=place>\n"
        '<parameter=origin>\n{"x": 3, "y": 4}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    name, args = _parse(wire, XML_REF_TOOL)
    assert name == "place"
    assert args["origin"] == {"x": 3, "y": 4}


# ==========================================================================
# FAITHFUL-OR-OPT-OUT via a STRICT ALLOWLIST (#558 E3). The XML arg emitter is a
# best-effort OPT-IN: when it cannot FAITHFULLY represent a schema the WHOLE
# request opts out of grammar (``build_tool_grammar`` -> ``None``, free-form
# fallback) rather than emit a grammar that silently allows schema-invalid or
# mis-typed output. The guard is a CLOSED POSITIVE SET (``_xml_schema_representable``
# returns ``True`` only when every part of the schema uses exclusively an
# enumerated known-safe keyword/shape), so it is complete BY CONSTRUCTION — no
# unknown JSON-Schema keyword can slip through. F1/F2/F4/F5 + the codex round-2
# shapes (minProperties/propertyNames/required-undeclared/false-schema/$ref-with-
# siblings/additionalProperties-schema/…) are all subsumed as "not in the
# allowlist"; F3 stays a REAL fix. Pure-Python guard tests always run; the
# ``build_tool_grammar`` integration tests skip only on genuine tokenizer/
# llguidance unavailability.
# ==========================================================================


# ---- Pure-Python representability guard (always runs) --------------------
def test_representable_common_and_noarg_schemas():
    # The common case + a genuine no-arg tool MUST stay representable (no
    # regression): typed properties, enums, required + optional, empty body.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert rep(XML_TOOLS_CONSTRAINABLE[0]["parameters"]) is True
    assert rep(XML_REF_TOOL[0]["parameters"]) is True  # $ref -> object
    assert rep({"type": "object", "properties": {}, "additionalProperties": False})
    assert rep({}) is True  # allow-any / no-arg
    # #1996: a top-level free-form string — direct or behind a ``$ref`` — is NOT
    # representable on this wire, so these two are opt-outs, not regressions.
    assert rep(XML_TOOLS[0]["parameters"]) is False
    assert rep(XML_REF_STRING_TOOL[0]["parameters"]) is False


def test_representable_rejects_f1_property_less_but_nontrivial():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    # F1: `false` schema (accepts no instance).
    assert rep(False) is False
    # F1: property-less but has `required`.
    assert rep({"type": "object", "required": ["a"]}) is False
    # F1: top-level `$ref` with no inline properties.
    assert rep({"$ref": "#/$defs/x", "$defs": {"x": {"type": "object"}}}) is False
    # F1 (spirit): a non-object top-level type has no XML parameter body.
    assert rep({"type": "array"}) is False


def test_representable_rejects_f2_string_facets():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    for facet in ("pattern", "minLength", "maxLength", "format", "const"):
        params = {
            "type": "object",
            "properties": {
                "s": {"type": "string", facet: "x" if facet != "minLength" else 2}
            },
        }
        assert rep(params) is False, facet
    # Enum (already enforced as an alternation) stays representable.
    assert rep({"properties": {"s": {"type": "string", "enum": ["a", "b"]}}}) is True
    # A `$ref` -> string carrying a facet is caught AFTER resolution (F2 + F3).
    assert (
        rep(
            {
                "properties": {"c": {"$ref": "#/$defs/C"}},
                "$defs": {"C": {"type": "string", "pattern": "^x$"}},
            }
        )
        is False
    )


def test_representable_rejects_f4_object_level_keywords():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    props = {"a": {"type": "string"}, "b": {"type": "string"}}
    for kw, val in (
        ("dependentRequired", {"a": ["b"]}),
        ("dependencies", {"a": ["b"]}),
        ("if", {}),
        ("then", {}),
        ("else", {}),
        ("not", {}),
        ("patternProperties", {"^x": {"type": "string"}}),
        ("allOf", [{}]),
        ("anyOf", [{}]),
        ("oneOf", [{}]),
    ):
        assert rep({"type": "object", "properties": props, kw: val}) is False, kw


def test_representable_rejects_f5_delimiter_unsafe_keys():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    # The property type is an INTEGER here on purpose: since #1996 a top-level
    # string opts out on its own, which would make every case below pass for the
    # wrong reason and the control unsatisfiable. An integer isolates KEY safety.
    for bad in ("x>y", "x<y", "a:b", "a,b", "a b", "a\nb", "a{b", "a}b"):
        assert rep({"properties": {bad: {"type": "integer"}}}) is False, bad
    # A normal [\w-]+ key stays representable.
    assert rep({"properties": {"my_key-1": {"type": "integer"}}}) is True


def test_representable_unresolvable_ref_opts_out():
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    # A property `$ref` that does not resolve locally is unrepresentable.
    assert rep({"properties": {"c": {"$ref": "#/$defs/missing"}, "$defs": {}}}) is False
    assert rep({"properties": {"c": {"$ref": "http://remote/x"}}}) is False


# ---- STRICT ALLOWLIST (codex round-2): closed positive set ends whack-a-mole --
# The guard is now an ALLOWLIST — representable ONLY when every part of the schema
# uses exclusively a known-safe, enumerated keyword/shape. These prove the five
# round-2 shapes a blacklist missed all opt out BY CONSTRUCTION (they are simply
# not in the allowlist), plus a positive control that the common case is intact.
def test_representable_allowlist_positive_control():
    # The normal shape a blacklist would let through unchanged MUST still be fully
    # constrained: typed scalars + enum + a nested OBJECT via `$ref` +
    # required ⊆ properties + closed additionalProperties. Guards against the
    # allowlist over-opting-out the common case. A top-level free-form string is
    # deliberately absent — #1996 made that the one shape this wire opts out of,
    # and its own test owns that case.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": ["python", "cpp"]},
            "retries": {"type": "integer"},
            "verbose": {"type": "boolean"},
            "origin": {"$ref": "#/$defs/point"},
        },
        "required": ["language"],
        "additionalProperties": False,
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
    assert rep(schema) is True


def test_representable_rejects_round2_minmax_properties():
    # A blacklist keyed on `dependentRequired`/composition MISSED these object-size
    # keywords; the allowlist opts out because they are not top-level-allowed keys.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    base = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert rep({**base, "minProperties": 1}) is False
    assert rep({**base, "maxProperties": 2}) is False


def test_representable_rejects_round2_required_undeclared_property():
    # `required` names a property that is NOT declared -> it would never be emitted
    # and thus silently unenforced -> opt out (finding 3). (Distinct from the F1
    # property-less case: here real properties ARE present.)
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert (
        rep(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a", "b"],  # `b` undeclared
            }
        )
        is False
    )


def test_representable_rejects_round2_property_schema_false():
    # A property schema that is literally `false` (accepts NO value) is not a dict
    # in the allowlist -> opt out (finding 1). `true` likewise.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert rep({"type": "object", "properties": {"a": False}}) is False
    assert rep({"type": "object", "properties": {"a": True}}) is False


def test_representable_rejects_round2_ref_with_sibling_enum():
    # A `$ref` carrying SIBLING keys (here `enum`) would DROP those siblings on
    # resolution -> opt out (finding 4), never silently ignore the enum.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert (
        rep(
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/x", "enum": ["p", "q"]}},
                "$defs": {"x": {"type": "string"}},
            }
        )
        is False
    )


def test_representable_rejects_round2_additional_properties_schema():
    # `additionalProperties` as a SCHEMA (or `True`) can't be constrained on the
    # XML wire -> opt out; only a literal `False` is representable.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    # Integer prop: a top-level string would opt out on its own (#1996) and the
    # ``False`` control could never be True.
    base = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert rep({**base, "additionalProperties": {"type": "string"}}) is False
    assert rep({**base, "additionalProperties": True}) is False
    assert rep({**base, "additionalProperties": False}) is True  # control


def test_representable_rejects_round2_property_names():
    # `propertyNames` constrains KEY names — unenforceable on the delimiter wire
    # and not a top-level-allowed key -> opt out.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert (
        rep(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "propertyNames": {"pattern": "^[a-z]+$"},
            }
        )
        is False
    )


# ---- STRICT ALLOWLIST (codex round-3): enum axis + total required guard ------
# Round-3 folded the ENUM branch INTO the allowlist (it previously blanket-accepted
# ANY non-empty enum) and made the ``required`` guard TOTAL (it previously crashed
# on an unhashable member). These prove the three holes opt out BY CONSTRUCTION
# (no raise), plus a positive control that a clean string enum stays constrained.
def test_representable_enum_positive_control_clean_string_enum():
    # A clean string enum with only annotation keys stays representable (a literal
    # alternation — the tightest constraint). Guards against over-opting-out.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert rep(
        {"properties": {"u": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}
    )
    # A type-less enum (the values fix the alternation) also stays representable.
    assert rep({"properties": {"u": {"enum": ["a", "b"]}}}) is True
    # A numeric enum consistent with its declared type stays representable.
    assert rep({"properties": {"n": {"type": "number", "enum": [1.5, 2]}}}) is True


def test_representable_enum_value_type_mismatch_opts_out():
    # codex r3 #2: an enum whose declared type contradicts its values is
    # schema-invalid on the wire -> opt out.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert rep({"properties": {"n": {"type": "integer", "enum": ["x"]}}}) is False
    # bool is NOT an integer here (excluded from int/number).
    assert rep({"properties": {"n": {"type": "integer", "enum": [True]}}}) is False
    assert rep({"properties": {"n": {"type": "number", "enum": ["1.5"]}}}) is False
    assert rep({"properties": {"b": {"type": "boolean", "enum": [1]}}}) is False
    # An object/array declared type carrying an enum has no scalar wire -> opt out.
    assert rep({"properties": {"o": {"type": "object", "enum": [{"a": 1}]}}}) is False


def test_representable_enum_unsupported_sibling_opts_out():
    # codex r3 #2: a validation sibling next to an enum (``minLength`` / ``pattern``
    # / ``minItems`` / any non-annotation key) is NOT enforced by a bare
    # alternation -> opt out rather than silently drop it.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert rep({"properties": {"s": {"enum": ["a", "bb"], "minLength": 2}}}) is False
    assert (
        rep({"properties": {"s": {"type": "string", "enum": ["a"], "pattern": "^a$"}}})
        is False
    )
    assert rep({"properties": {"s": {"enum": ["a"], "minItems": 1}}}) is False


def test_representable_enum_delimiter_bearing_value_opts_out():
    # codex r3 #3: an enum value carrying the close delimiter (or any ``<``/``>``/
    # newline) is grammar-accepted but truncates at the parser's FIRST
    # ``</parameter>`` -> opt out rather than emit a wire-breaking literal.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    assert (
        rep({"properties": {"s": {"type": "string", "enum": ["a</parameter>b"]}}})
        is False
    )
    assert (
        rep({"properties": {"s": {"type": "string", "enum": ["ok", "<x>"]}}}) is False
    )
    assert rep({"properties": {"s": {"type": "string", "enum": ["a\nb"]}}}) is False
    # A non-string enum value whose json.dumps embeds a delimiter also opts out.
    assert rep({"properties": {"o": {"enum": [{"k": "</parameter>"}]}}}) is False


def test_representable_total_required_guard_never_raises():
    # codex r3 #4 (REAL CRASH): a malformed ``required`` must OPT OUT, never raise.
    # ``set(required)`` on an UNHASHABLE member (list/dict) raised ``TypeError``
    # OUTSIDE the builder's exception handling -> a 500 on arbitrary client JSON.
    from vllm_mlx.api.tool_grammar import _xml_schema_representable as rep

    # Integer prop for the same reason as above (#1996): the control at the end
    # of this test asserts a well-formed ``required`` stays representable.
    base = {"type": "object", "properties": {"a": {"type": "integer"}}}
    # ``required`` not a list -> opt out (no raise).
    assert rep({**base, "required": "a"}) is False
    assert rep({**base, "required": 123}) is False
    assert rep({**base, "required": {"a": True}}) is False
    # ``required`` containing an UNHASHABLE member (list / dict) -> opt out, and
    # CRITICALLY does not raise ``TypeError`` from ``set(required)``.
    assert rep({**base, "required": [["a"]]}) is False
    assert rep({**base, "required": [{"k": "v"}]}) is False
    assert rep({**base, "required": ["a", ["nested"]]}) is False
    # A non-str hashable member (int) also opts out (cannot name a property).
    assert rep({**base, "required": [1, 2]}) is False
    # Control: a well-formed required ⊆ properties stays representable.
    assert rep({**base, "required": ["a"]}) is True


# ---- F3 REAL FIX: $ref resolved BEFORE the string-vs-%json decision ------
def test_xml_ref_to_string_uses_json_string_path():
    # Grammar-string level: a `$ref` -> string property emits the LAZY raw value
    # rule, NOT `%json` (whose quotes the parser would keep). The tool's only
    # param is a `$ref` -> string, so the whole grammar has NO `%json` at all.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(
        XML_REF_STRING_TOOL, "required", [_xml_structure_info("geo")]
    )
    assert '"<parameter=city>\\n" %json' in lark
    assert "xml_param_value" not in lark


def test_xml_ref_to_object_still_uses_json():
    # The complementary case: a `$ref` -> OBJECT still rides `%json` (over the
    # original `$ref` + merged `$defs`, which llguidance resolves internally).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(XML_REF_TOOL, "required", [_xml_structure_info("place")])
    assert '"<parameter=origin>\\n" %json' in lark


@_requires_llguidance
def test_xml_ref_to_string_opts_out_and_still_parses_raw(tok):
    # F3 resolved the ``$ref`` BEFORE the string-vs-``%json`` decision so a
    # ``$ref`` -> string takes the same path as an inline string. Since #1996
    # that path is "not representable", so this tool opts out too — resolution
    # order still matters, it just now decides opt-out rather than which value
    # rule to emit. The PARSER half of F3 is unchanged and still asserted: a raw
    # unquoted value round-trips to a clean ``Paris``, no quotes preserved.
    assert _xml_grammar(XML_REF_STRING_TOOL, "required", tok) is None
    wire = (
        "<tool_call>\n<function=geo>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    name, args = _parse(wire, XML_REF_STRING_TOOL)
    assert name == "geo"
    assert args["city"] == "Paris"  # no surrounding quotes


# ---- F1/F2/F4/F5 OPT-OUT through build_tool_grammar (real parser) --------
@_requires_llguidance
def test_build_tool_grammar_opts_out_f1_toplevel_ref(tok):
    # Control: a representable XML tool DOES build a grammar (proves the opt-out
    # below is the guard, not a generic build failure).
    assert _xml_grammar(XML_TOOLS_CONSTRAINABLE, "required", tok) is not None
    # F1: a top-level `$ref` (no inline properties) would collapse to an EMPTY
    # body allowing `{}` even though the schema requires fields -> opt out.
    ref_tool = [
        {
            "name": "run_code",
            "parameters": {
                "$ref": "#/$defs/loc",
                "$defs": {
                    "loc": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    }
                },
            },
        }
    ]
    assert _xml_grammar(ref_tool, "required", tok) is None


@_requires_llguidance
def test_build_tool_grammar_opts_out_f2_string_pattern(tok):
    # F2: a string param with `pattern` cannot be enforced on the raw value path.
    pat_tool = [
        {
            "name": "run_code",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "pattern": "^[a-z]+$"}},
                "required": ["code"],
            },
        }
    ]
    assert _xml_grammar(pat_tool, "required", tok) is None


@_requires_llguidance
def test_build_tool_grammar_opts_out_f4_dependent_required(tok):
    # F4: `dependentRequired` (object-level relation) is emitted-around, not
    # enforced — it would let `a` appear without its required `b` -> opt out.
    dep_tool = [
        {
            "name": "run_code",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "dependentRequired": {"a": ["b"]},
            },
        }
    ]
    assert _xml_grammar(dep_tool, "required", tok) is None


@_requires_llguidance
def test_build_tool_grammar_opts_out_f5_unsafe_key(tok):
    # F5: a key containing `>` inserted RAW into `<parameter=KEY>` would be parsed
    # back as a DIFFERENT key -> opt out.
    bad_key_tool = [
        {
            "name": "run_code",
            "parameters": {
                "type": "object",
                "properties": {"x>y": {"type": "string"}},
                "required": ["x>y"],
            },
        }
    ]
    assert _xml_grammar(bad_key_tool, "required", tok) is None


# ---- codex round-3: end-to-end opt-out (real parser) — no crash on bad input --
@_requires_llguidance
def test_build_tool_grammar_opts_out_r3_malformed_required_no_crash(tok):
    # codex r3 #4: a malformed `required` (non-list, or a list with an UNHASHABLE
    # member) must opt out to free-form (None) WITHOUT raising — `set(required)`
    # in the guard previously raised `TypeError` OUTSIDE the builder's exception
    # handling (a 500). Reaching `build_tool_grammar` at all exercises the guard
    # site that crashed.
    for bad_required in ("code", [["code"]], [{"k": "v"}], 7):
        tool = [
            {
                "name": "run_code",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": bad_required,
                },
            }
        ]
        assert _xml_grammar(tool, "required", tok) is None, bad_required


@_requires_llguidance
def test_build_tool_grammar_opts_out_r3_bad_enum(tok):
    # codex r3 #2/#3: an enum type-mismatch / unsupported sibling / delimiter-bearing
    # value each opt the whole request out of grammar; a clean string enum builds.
    def _tool(enum_schema):
        # ``n`` is an INTEGER, not a string: since #1996 a top-level free-form
        # string opts out on its own, which would make every assertion below pass
        # for the wrong reason and the control unsatisfiable. The enum is the
        # only thing under test here.
        return [
            {
                "name": "run_code",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}, "e": enum_schema},
                    "required": ["n"],
                },
            }
        ]

    assert (
        _xml_grammar(_tool({"type": "integer", "enum": ["x"]}), "required", tok) is None
    )
    assert (
        _xml_grammar(_tool({"enum": ["a", "bb"], "minLength": 2}), "required", tok)
        is None
    )
    assert (
        _xml_grammar(
            _tool({"type": "string", "enum": ["a</parameter>b"]}), "required", tok
        )
        is None
    )
    # Control: a clean string enum still builds a grammar (not a generic failure).
    assert (
        _xml_grammar(_tool({"type": "string", "enum": ["ok", "no"]}), "required", tok)
        is not None
    )
