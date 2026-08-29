# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the hermes/qwen ``structure_info()`` overrides (#558 PR-2).

PR-1 shipped the grammar builder plus a NON-BREAKING ``structure_info() ->
None`` ABC default, and proved the builder against a *test-local* hermes-style
stub. PR-2 lands the concrete per-family overrides on the REAL
``HermesToolParser`` and ``QwenToolParser`` (which share the
``<tool_call>…</tool_call>`` JSON-body wire). These tests therefore drive the
grammar path through the ACTUAL shipped parsers — not a stub — to prove:

  * each real parser's ``structure_info()`` is TOKENIZER-AWARE: it opts into
    grammar constraint (returns a ``name -> StructureInfo`` factory whose wire
    triple is the hermes ``<tool_call>`` JSON body, with the
    ``<tool_call>``/``</tool_call>`` single special tokens declared as
    ``sentinels`` — ground-truth correction #1) ONLY when the model's tokenizer
    proves both sentinels are single tokens, and OPTS OUT (returns ``None`` ->
    free-form fallback) otherwise (no tokenizer, or a Llama-based Hermes
    tokenizer that encodes ``<tool_call>`` as ordinary multi-token text);
  * feeding a real parser through ``build_tool_grammar`` yields a Lark with the
    ``<tool_call>`` bare special-token trigger + a ``%json`` schema-constraint
    region for the ``arguments`` object;
  * grammar ENFORCEMENT via llguidance ``LLMatcher``: a well-formed hermes/qwen
    tool call is ACCEPTED in full (and is a terminal/accepting state) while a
    hallucinated tool name, an off-schema argument, and a bad enum value are
    REJECTED mid-stream.

Scope note: this PR teaches hermes/qwen to DESCRIBE their grammar; NOTHING in
the request path calls ``structure_info()`` yet (chat.py/scheduler.py routing +
the runtime ``GrammarLogitsProcessor`` are PR-3). So these tests are hermetic —
no server, no decode loop, no live network beyond the pinned tokenizer fetch —
and the overrides are pure no-behavior-change scaffolding until PR-3 wires them.

The enforcement tests need a fast (Rust) tokenizer whose
``<tool_call>``/``</tool_call>`` are single special tokens — the pilot verified
this on ``mlx-community/Qwen3.5-4B-MLX-4bit`` (pinned by revision below for an
immutable artifact). Those tests skip ONLY on genuine unavailability
(llguidance extra absent, or the tokenizer neither cached nor reachable); any
OTHER failure is surfaced, not swallowed. The pure-Python structure-triple and
Lark-structure tests never skip — they carry no optional dependency.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.requires_mlx

# llguidance is only needed by the grammar-BUILD / enforcement tests (they
# compile a Lark grammar / build an LLTokenizer). The structure-triple and
# pure-Lark-string tests need NOTHING optional, so we do NOT skip at module
# level — a repo without the [guided] extra still exercises the real parsers'
# structure_info triples and the builder's string output.
_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Pin the revision so the enforcement proof runs against an IMMUTABLE artifact
# (an unpinned Hub revision is a mutable third-party dependency). The
# tokenizer's <tool_call>/</tool_call> single-special-token layout is fixed at
# this commit; a different upstream revision must not silently change what the
# enforcement tests exercise. Same pin as tests/test_tool_grammar_558.py.
_TOKENIZER_REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"

TOOLS = [
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

# The two families that share the <tool_call>…</tool_call> JSON-body wire and
# opt into grammar constraint in this PR. Parametrizing over the REAL parser
# classes proves BOTH overrides (not a shared stub) with one test body.
_PARSER_IMPORTS = {
    "hermes": ("vllm_mlx.tool_parsers.hermes_tool_parser", "HermesToolParser"),
    "qwen": ("vllm_mlx.tool_parsers.qwen_tool_parser", "QwenToolParser"),
}


class _FakeAddedToken:
    """Stand-in for transformers' ``AddedToken`` — carries a ``special`` flag.

    GROUND TRUTH: on the real Qwen3.5 tokenizer, ``<tool_call>``/``</tool_call>``
    are ADDED tokens with ``special=False``. The guard must accept them, so it
    keys on added-token REGISTRATION (``added_tokens_decoder`` membership), NOT
    the ``special`` flag — this stub defaults ``special=False`` to hold the guard
    to that reality.
    """

    def __init__(self, content, special=False):
        self.content = content
        self.special = special


class _FakeTokenizer:
    """Minimal tokenizer stub modeling the surfaces the guard probes.

    ``structure_info()`` opts into grammar constraint only when the model's
    tokenizer proves ``<tool_call>``/``</tool_call>`` are DISTINCT single
    REGISTERED added tokens that round-trip (the hermes parser is also routed to
    Llama tokenizers where they are NOT). The guard checks: ``len(encode(s)) ==
    1``, ``decode([id]) == s`` (round-trip), the id is in
    ``added_tokens_decoder`` (explicitly-added atomic token), and distinct ids
    across sentinels. This stub lets the pure-Python tests exercise every
    opt-in/opt-out branch WITHOUT a network fetch, so they stay hermetic.

    ``added`` maps each ADDED-token string to its single id (round-trips, in
    ``added_tokens_decoder`` — modeled with ``special=False`` to match the real
    Qwen tokenizer). Any other string encodes as multi-token (opt-out path).
    ``ordinary`` maps a string to a single id that round-trips but is NOT in the
    added-token registry (ordinary-BPE opt-out case). ``collapse`` maps strings
    to a shared single id that does NOT round-trip (``[UNK]`` collapse case).
    """

    def __init__(self, added=None, ordinary=None, collapse=None):
        self._added = dict(added or {})
        self._ordinary = dict(ordinary or {})
        self._collapse = dict(collapse or {})
        # id -> source string, for round-trip decode of added + ordinary.
        self._id_to_str = {i: s for s, i in self._added.items()}
        self._id_to_str.update({i: s for s, i in self._ordinary.items()})
        # added_tokens_decoder: {id: AddedToken(special=False)} — matches the
        # real Qwen layout where <tool_call> is an added but non-special token.
        self.added_tokens_decoder = {
            i: _FakeAddedToken(s, special=False) for s, i in self._added.items()
        }

    def encode(self, text, add_special_tokens=False):
        if text in self._added:
            return [self._added[text]]
        if text in self._ordinary:
            return [self._ordinary[text]]
        if text in self._collapse:
            return [self._collapse[text]]
        return [0, 1]  # ordinary multi-token text

    def decode(self, ids):
        return "".join(self._id_to_str.get(i, "<unk>") for i in ids)


def _single_token_tokenizer():
    """Qwen3-like: both sentinels are distinct single ADDED (special=False)
    tokens that round-trip -> opt in (the real Qwen layout)."""
    return _FakeTokenizer(added={"<tool_call>": 100, "</tool_call>": 101})


def _multitoken_tokenizer():
    """Llama-Hermes-like: neither sentinel is a single token -> opt out."""
    return _FakeTokenizer(added={})


def _ordinary_vocab_tokenizer():
    """Both sentinels are single tokens that round-trip but are NOT in the
    added-token registry (ordinary BPE tokens) -> opt out."""
    return _FakeTokenizer(ordinary={"<tool_call>": 100, "</tool_call>": 101})


def _unk_collapse_tokenizer():
    """Both sentinels collapse to the SAME single [UNK] id that does not
    round-trip -> opt out (distinct-id + round-trip both fail)."""
    return _FakeTokenizer(collapse={"<tool_call>": 3, "</tool_call>": 3})


def _make_parser(family: str, tokenizer=None):
    import importlib

    module_name, cls_name = _PARSER_IMPORTS[family]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(tokenizer=tokenizer)


def _make_optin_parser(family: str):
    """A real parser whose tokenizer satisfies the single-special-token guard,
    so ``structure_info()`` opts in — used by the hermetic triple/Lark tests."""
    return _make_parser(family, tokenizer=_single_token_tokenizer())


# --------------------------------------------------------------------------
# Tokenizer-aware opt-in/opt-out guard (pure Python, always runs). This is the
# load-bearing correctness contract: the hermes wire's <tool_call> sentinels
# are single special tokens ONLY on some tokenizers (Qwen3), NOT universally
# (Llama-based Hermes). A parser must opt out where the tokenizer can't prove
# DISTINCT single REGISTERED special sentinels, else it would build an
# unenforceable grammar.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_out_without_tokenizer(family):
    # No tokenizer -> cannot prove single-token sentinels -> opt out (None),
    # so the builder falls back to free-form. NON-BREAKING for tokenizer-less
    # construction paths.
    assert _make_parser(family, tokenizer=None).structure_info() is None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_out_on_multitoken_tokenizer(family):
    # A tokenizer where <tool_call> is ordinary multi-token text (Llama-based
    # Hermes) -> opt out. Declaring a special-token sentinel there would build a
    # grammar the model's tokenizer can never satisfy — the exact bug this guard
    # prevents.
    parser = _make_parser(family, tokenizer=_multitoken_tokenizer())
    assert parser.structure_info() is None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_out_on_ordinary_vocab_token(family):
    # Both sentinels are single tokens that round-trip but are NOT registered
    # special tokens -> opt out. A single ORDINARY vocab token cannot back a
    # special-token Lark ref, so len==1 alone must not opt in (codex round-2).
    parser = _make_parser(family, tokenizer=_ordinary_vocab_tokenizer())
    assert parser.structure_info() is None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_out_on_unk_collapse(family):
    # Both sentinels collapse to the SAME single [UNK] id that does not
    # round-trip -> opt out. len==1 for each would otherwise falsely opt into an
    # unenforceable grammar where open/close are indistinguishable (codex r2).
    parser = _make_parser(family, tokenizer=_unk_collapse_tokenizer())
    assert parser.structure_info() is None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_in_on_single_token_tokenizer(family):
    # A tokenizer where both sentinels ARE distinct single added tokens that
    # round-trip (Qwen3-like) -> opt in.
    parser = _make_parser(family, tokenizer=_single_token_tokenizer())
    assert parser.structure_info() is not None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_opts_in_on_non_special_added_token(family):
    # GROUND-TRUTH regression (codex r3): on the REAL Qwen3.5 tokenizer,
    # <tool_call>/</tool_call> are ADDED tokens with AddedToken.special == False
    # (and NOT in all_special_ids). They are still distinct atomic tokens a
    # grammar special-token ref resolves against (the enforcement tests below
    # pass on exactly this tokenizer). The guard MUST key on added-token
    # REGISTRATION, not the `special` flag — gating on special==True would break
    # the feature for its own target tokenizer. `_single_token_tokenizer` models
    # special=False added tokens, so this asserting opt-in is the regression.
    from vllm_mlx.api.tool_grammar import are_single_special_tokens

    tokenizer = _single_token_tokenizer()
    # Every modeled added token carries special=False (matches real Qwen).
    assert all(at.special is False for at in tokenizer.added_tokens_decoder.values())
    assert are_single_special_tokens(tokenizer, ("<tool_call>", "</tool_call>"))
    parser = _make_parser(family, tokenizer=tokenizer)
    assert parser.structure_info() is not None


# --------------------------------------------------------------------------
# structure_info() wire triple (pure Python, always runs — hermetic tokenizer
# stub so no network is needed to exercise the opt-in path).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_returns_hermes_wire_triple(family):
    from vllm_mlx.api.tool_grammar import StructureInfo

    parser = _make_optin_parser(family)
    get_info = parser.structure_info()
    # PR-2 opt-in: the override returns a name->StructureInfo factory, not None.
    assert callable(get_info), f"{family}.structure_info() must return a callable"

    si = get_info("get_weather")
    assert isinstance(si, StructureInfo)
    # The hermes <tool_call> JSON-body wire, with the concrete tool name
    # substituted into ``begin``.
    assert si.trigger == "<tool_call>"
    assert si.begin == '<tool_call>\n{"name": "get_weather", "arguments": '
    assert si.end == "}\n</tool_call>"
    # begin MUST start with trigger (StructTag invariant the builder enforces).
    assert si.begin.startswith(si.trigger)
    # Ground-truth correction #1: <tool_call>/</tool_call> are SINGLE special
    # tokens, so both must be declared as sentinels (the trigger among them) so
    # the builder renders them as special-token refs, not byte strings.
    assert si.sentinels == ("<tool_call>", "</tool_call>")
    assert si.trigger in si.sentinels


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_substitutes_each_tool_name(family):
    # The factory substitutes whatever concrete name it is given — one triple
    # per tool, so a multi-tool request constrains each tool to ITS own schema.
    parser = _make_optin_parser(family)
    get_info = parser.structure_info()
    for name in ("get_weather", "get_time", "any_other_name"):
        si = get_info(name)
        assert si.begin == f'<tool_call>\n{{"name": "{name}", "arguments": '


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_structure_info_json_escapes_tool_name(family):
    # codex r4 nit: a name containing " or \ must be JSON-escaped so the wire is
    # well-formed JSON, not broken. json.dumps handles the quoting + escaping.
    import json

    parser = _make_optin_parser(family)
    get_info = parser.structure_info()
    weird = 'weird"na\\me'
    si = get_info(weird)
    assert si.begin == f'<tool_call>\n{{"name": {json.dumps(weird)}, "arguments": '
    # The name region round-trips as valid JSON (extract the "name": <...> part).
    body = si.begin[len("<tool_call>\n") :]  # {"name": "...", "arguments":
    name_json = body[len('{"name": ') : body.index(', "arguments": ')]
    assert json.loads(name_json) == weird
    # begin still starts with the trigger (builder invariant preserved).
    assert si.begin.startswith("<tool_call>")


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_hermes_and_qwen_share_identical_wire(family):
    # hermes and qwen intentionally emit the SAME wire triple (both are the
    # <tool_call> JSON body). Assert byte-identical triples so the two overrides
    # can never silently diverge.
    hermes_si = _make_optin_parser("hermes").structure_info()("get_weather")
    other_si = _make_optin_parser(family).structure_info()("get_weather")
    assert other_si == hermes_si


# --------------------------------------------------------------------------
# build_tool_grammar / Lark structure via the REAL parsers (needs llguidance).
# --------------------------------------------------------------------------
@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
@pytest.mark.parametrize("tool_choice", ["required", "auto"])
def test_real_parser_builds_grammar(family, tool_choice):
    # Driving the REAL parser (not a stub) through the public builder yields a
    # compiled grammar (non-None) for both required and auto.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    parser = _make_optin_parser(family)
    assert build_tool_grammar(TOOLS, tool_choice, parser) is not None


@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_real_parser_lark_has_trigger_and_schema_region(family):
    # Assemble the Lark from the REAL parser's structure_info and assert the
    # load-bearing structure: <tool_call> as a BARE special-token ref (not a
    # quoted byte literal the single <tool_call> token could never satisfy),
    # </tool_call> bare closing ref, and a %json schema-constraint region.
    # Pure ``build_tool_lark`` (string assembly) — needs NO llguidance, so this
    # test always runs (it does not compile the grammar).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    get_info = _make_optin_parser(family).structure_info()
    infos = [get_info(t["name"]) for t in TOOLS]
    lark = build_tool_lark(TOOLS, "required", infos)

    assert " <tool_call> " in lark  # space-delimited bare token ref (trigger)
    assert lark.rstrip().endswith("</tool_call>")  # bare closing token ref
    assert '"<tool_call>"' not in lark  # NOT a quoted (multi-byte) literal
    assert '"</tool_call>"' not in lark
    assert "%json" in lark  # arguments constrained by JSON Schema
    assert "get_weather" in lark
    assert "get_time" in lark


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT via offline LLMatcher (the #558 proof, real parsers).
#
# Needs a fast (Rust) tokenizer whose <tool_call>/</tool_call> are single
# special tokens. The ONLY sanctioned skip is genuine tokenizer/llguidance
# UNAVAILABILITY (no network + not cached, or the optional extra absent) — any
# OTHER failure (a real grammar regression, a matcher error, an unexpected
# tokenizer exception) propagates and FAILS the test.
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    """Typed offline / connection exceptions that are ALWAYS a skip.

    Covers every transport transformers/hub may use (codex r5): huggingface_hub
    offline errors, `requests` AND `httpx` connection/timeout exceptions, and
    the Python built-in connection errors (`ConnectionError` family, which
    `ConnectionRefusedError` subclasses). Best-effort import — a transport not
    installed is simply omitted.
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
        from requests.exceptions import Timeout as _ReqTimeout

        types += [_ReqConnErr, _ReqTimeout]
    except Exception:  # pragma: no cover - requests not present
        pass
    try:
        import httpx as _httpx

        # ConnectError/ConnectTimeout/ReadTimeout all subclass TransportError;
        # use the broad base so any transport-level failure counts as offline.
        types.append(_httpx.TransportError)
    except Exception:  # pragma: no cover - httpx not present
        pass
    # Python built-in socket-level connection errors (ConnectionRefusedError,
    # ConnectionResetError, ... all subclass ConnectionError). TimeoutError is a
    # builtin too (socket timeouts raise it).
    types += [ConnectionError, TimeoutError]
    return tuple(types)


# Phrases that UNAMBIGUOUSLY mean network/connection failure (offline). We
# deliberately EXCLUDE transformers' generic "Can't load ... for X" / "look for
# the file" wording (codex r4): those also front corrupt / incompatible /
# incomplete artifact errors, so matching them would silently SKIP a real
# regression. A genuine offline failure says it could not CONNECT — that
# phrasing never appears for a corrupt local file.
_OFFLINE_OSERROR_MARKERS = (
    "offline",
    "couldn't connect",
    "we couldn't connect",
    "connection error",
    "max retries exceeded",
    "failed to establish a new connection",
    "failed to connect",
    "name or service not known",
    "temporary failure in name resolution",
)


def _is_offline_oserror(exc: BaseException) -> bool:
    """True iff ``exc`` (a bare ``OSError`` from transformers) is an offline /
    connection-failure wrap rather than a corrupt-artifact error.

    Two independent signals, either sufficient — both narrowly scoped so a
    corrupt/incompatible tokenizer artifact (which transformers ALSO raises as a
    generic ``OSError`` with "Can't load…" wording) still FAILS, not skips
    (codex r4):

      * a TYPED connection exception anywhere in the ``__cause__``/``__context__``
        chain — ``requests``/``httpx`` transport errors, ``huggingface_hub``
        offline errors, or the Python built-in ``ConnectionError``/
        ``TimeoutError`` family (see ``_offline_skip_exc_types``) — the strongest
        signal, immune to message wording; OR
      * a message containing an UNAMBIGUOUS connection-failure phrase (never
        emitted for a corrupt local file).
    """
    typed_offline = _offline_skip_exc_types()
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if typed_offline and isinstance(cur, typed_offline):
            return True
        msg = str(cur).lower()
        if any(marker in msg for marker in _OFFLINE_OSERROR_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def test_offline_oserror_classification():
    # codex r3/r4: transformers wraps an offline cache-miss as a bare OSError.
    # The `tok` fixture must SKIP on that (offline) but FAIL on a corrupt/
    # incompatible-artifact OSError — including transformers' GENERIC "Can't
    # load tokenizer for X" wording, which fronts BOTH offline and corrupt cases
    # (so message wording alone is insufficient; a typed connection cause is
    # the load-bearing signal).
    offline_direct = OSError(
        "We couldn't connect to 'https://huggingface.co' to load this file"
    )
    assert _is_offline_oserror(offline_direct)

    # codex r5: prove the TYPED-cause path in isolation. The generic outer
    # message and the cause message both carry NO offline marker, so this can
    # only pass via typed-exception detection — deleting the isinstance branch
    # would flip it red (unlike the old version, which passed on a "Max retries"
    # string marker and left the typed path untested). ConnectionRefusedError
    # subclasses the built-in ConnectionError, which is in _offline_skip_exc_types.
    generic_wrap = OSError("Can't load tokenizer for 'x'")
    try:
        try:
            raise ConnectionRefusedError(61, "Connection refused")
        except ConnectionRefusedError as cause:
            raise generic_wrap from cause
    except OSError as raised:
        assert _is_offline_oserror(raised)

    # codex r5: an httpx transport error in the chain is also offline (transformers
    # 5.x uses httpx). Marker-free message -> only the typed path can pass.
    httpx = pytest.importorskip("httpx")
    httpx_wrap = OSError("Can't load tokenizer for 'x'")
    try:
        try:
            raise httpx.ConnectError("nope")
        except httpx.ConnectError as cause:
            raise httpx_wrap from cause
    except OSError as raised:
        assert _is_offline_oserror(raised)

    # codex r4: the SAME generic "Can't load…" wording with NO connection cause
    # is a corrupt/incompatible artifact -> must NOT skip (must fail). This is
    # the regression the over-broad "can't load" marker would have wrongly
    # skipped.
    corrupt_generic = OSError(
        "Can't load tokenizer for 'x'. Make sure it is a correct model "
        "identifier or the tokenizer files are not corrupted."
    )
    assert not _is_offline_oserror(corrupt_generic)

    # A corrupt-artifact OSError with unrelated wording -> must NOT skip.
    corrupt = OSError("Unable to load weights: file is not a valid JSON document")
    assert not _is_offline_oserror(corrupt)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION
        )
    except _offline_skip_exc_types():  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not "
            "cached and no network — enforcement tests require it"
        )
    except OSError as exc:  # pragma: no cover - transformers-wrapped offline
        # transformers commonly wraps an offline cache-miss as a bare OSError.
        # Skip ONLY when the message/chain proves offline/cache-miss; a
        # corrupt-artifact OSError has no offline marker and re-raises (fails).
        if _is_offline_oserror(exc):
            pytest.skip(
                f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not "
                "cached and no network (transformers OSError) — enforcement "
                "tests require it"
            )
        raise


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer from the fast (Rust) tokenizer.

    Mirrors ``guided.py``'s tokenizer resolution: try the wrapper's inner fast
    tokenizer, then the object itself. A slow tokenizer is the one sanctioned
    skip; a genuine ``from_tokenizer`` regression is NOT swallowed.
    """
    import llguidance.hf as llg_hf

    candidates = []
    inner = getattr(tok, "_tokenizer", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(tok)
    fast_candidates = [
        c for c in candidates if getattr(c, "is_fast", True) is not False
    ]
    if not fast_candidates:
        pytest.skip("tokenizer is not a fast tokenizer — llguidance needs one")
    last_exc = None
    for cand in fast_candidates:
        try:
            return llg_hf.from_tokenizer(cand)
        except Exception as exc:  # noqa: BLE001 - re-raised below if all fail
            last_exc = exc
    raise AssertionError(
        f"llguidance could not build an LLTokenizer from any fast candidate: "
        f"{last_exc!r}"
    )


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)``.

    Advances real grammar state one token at a time via
    ``LLMatcher.consume_tokens`` (which returns a bool per batch), counting how
    many tokens the grammar accepts before it rejects one. Because this ADVANCES
    matcher state, afterwards ``is_accepting()`` reports whether the grammar can
    TERMINATE there — so a "fully accepted" positive test proves the string is a
    COMPLETE valid derivation, not merely an accepted prefix.
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
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_valid_call_is_accepted_and_terminates(family, tok, lltok):
    # A well-formed call through the REAL parser's grammar is accepted in full
    # AND is a terminal/accepting state (a complete valid derivation).
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family, tok))
    assert grammar is not None
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    )
    assert accepted == total, f"valid {family} call rejected ({accepted}/{total})"
    assert accepting, f"valid complete {family} call is not an accepting state"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_valid_enum_value_is_accepted(family, tok, lltok):
    # Positive enum control (paired with the rejection test below): a VALID enum
    # value is accepted and terminates — so the rejection test cannot pass merely
    # because the grammar forbids the optional `unit` property entirely.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family, tok))
    accepted, total, accepting = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "c"}}\n</tool_call>',
    )
    assert accepted == total, (
        f"valid enum value rejected for {family} ({accepted}/{total})"
    )
    assert accepting, f"valid enum call is not an accepting state for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_hallucinated_tool_name_is_rejected(family, tok, lltok):
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family, tok))
    accepted, total, _ = _consume(
        grammar, lltok, tok, '<tool_call>\n{"name": "get_stockquote'
    )
    assert accepted < total, f"hallucinated tool name NOT rejected for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_off_schema_argument_is_rejected(family, tok, lltok):
    # `city` must be a string; an integer must be forbidden.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family, tok))
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": 4',
    )
    assert accepted < total, f"off-schema integer argument NOT rejected for {family}"


@_requires_llguidance
@pytest.mark.parametrize("family", ["hermes", "qwen"])
def test_bad_enum_value_is_rejected(family, tok, lltok):
    # `unit` enum is {c, f}; "kelvin" must be forbidden.
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    grammar = build_tool_grammar(TOOLS, "required", _make_parser(family, tok))
    accepted, total, _ = _consume(
        grammar,
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "kelvin',
    )
    assert accepted < total, f"invalid enum value NOT rejected for {family}"
