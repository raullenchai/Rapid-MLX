# SPDX-License-Identifier: Apache-2.0
"""Runtime tests for grammar-constrained tool calling (#558 PR-3).

PR-1 shipped the pure grammar BUILDER and PR-2 the per-family
``structure_info`` overrides (both covered by ``test_tool_grammar_558.py`` /
``test_structure_info_hermes_qwen_558.py``). PR-3 adds the RUNTIME half:

  * ``GrammarLogitsProcessor`` — the per-token mask applied to logits each
    decode step, including the CUMULATIVE-BASELINE fix (mlx-lm hands the full
    ``prompt + generated`` sequence each step; the matcher must be advanced
    ONLY over generated tokens);
  * ``build_lltokenizer`` — the ``LLTokenizer`` factory with the ``to_str()``
    fallback for the transformers ``from_tokenizer`` isinstance gotcha;
  * ``routes.chat._normalize_tool_choice_for_grammar`` — the collision-safe
    ``tool_choice`` normalization (a tool literally named ``"required"`` must
    never be misread as the ``required`` enum).

The processor / enforcement tests need a fast (Rust) tokenizer whose
``<tool_call>``/``</tool_call>`` are single special tokens — verified on
``mlx-community/Qwen3.5-4B-MLX-4bit`` (the pilot wire model). They skip ONLY
on genuine unavailability (the ``[guided]`` extra absent, or the tokenizer
neither cached nor reachable). The pure-Python normalization tests carry no
optional dependency and always run.
"""

import importlib.util

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)


@pytest.fixture(autouse=True)
def _opt_in_constrain_tools(monkeypatch):
    """#558 PR-5 ships the constraint DEFAULT-ON (``RAPID_MLX_CONSTRAIN_TOOLS``
    is now an OPT-OUT toggle). These runtime tests exercise the ENABLED path;
    pin the env to ``"1"`` explicitly so the module is deterministic regardless
    of ambient env (the default-on behavior with the var ABSENT is proven
    separately in ``test_eligible_true_by_default_when_env_unset``). The explicit
    kill-switch tests override this back to ``"0"``.
    """
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")


_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Pin the revision so the enforcement proof runs against an IMMUTABLE artifact
# (mirrors test_tool_grammar_558.py). The tokenizer's
# ``<tool_call>``/``</tool_call>`` single-special-token layout is fixed here.
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


# --------------------------------------------------------------------------
# Lightweight stand-ins mirroring the ToolDefinition / request / engine / cfg
# shapes the chat route reads, so tests can drive the REAL
# ``_maybe_build_tool_grammar_processor`` without a live server.
# --------------------------------------------------------------------------
class _FunctionTool:
    """Mirrors ``ToolDefinition`` (``.function`` is a dict). ``parameters`` is
    OMITTED from the dict entirely when not passed, so ``"parameters" in fn``
    is False (the absent case)."""

    _SENTINEL = object()

    def __init__(self, name, parameters=_SENTINEL):
        self.function = {"name": name}
        if parameters is not _FunctionTool._SENTINEL:
            self.function["parameters"] = parameters


class _RequestStub:
    def __init__(self, tools, tool_choice):
        self.tools = tools
        self.tool_choice = tool_choice


class _EngineStub:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _CfgStub:
    def __init__(self, tool_call_parser):
        self.tool_call_parser = tool_call_parser


def _http_status_of(exc):
    """Best-effort HTTP status code for a hub/requests/httpx error, else None."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    # huggingface_hub sometimes carries the code directly.
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_offline_skippable(exc) -> bool:
    """True iff ``exc`` is a TRANSIENT network/cache-miss error worth skipping.

    Skip ONLY genuine transient signals: offline/cache-miss, a raw
    connection/timeout error, or a rate-limit (429) / transient server (5xx)
    HTTP response. A PERMANENT 4xx (401/403 bad creds, 404 deleted artifact,
    422 invalid pinned revision) must FAIL — it means the test is misconfigured
    or the artifact changed, not that the network blipped (codex #558-PR3). A
    corrupt local file, a programming/API error, etc. also propagate.
    """
    # Cache-miss / explicit offline: always skippable.
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        if isinstance(exc, (LocalEntryNotFoundError, OfflineModeIsEnabled)):
            return True
    except Exception:  # pragma: no cover - old hub without these names
        pass
    # Raw connection / timeout (requests + httpx): transport failure, skippable.
    transient_types: list[type[BaseException]] = []
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr
        from requests.exceptions import Timeout as _ReqTimeout

        transient_types += [_ReqConnErr, _ReqTimeout]
    except Exception:  # pragma: no cover - requests not present
        pass
    try:
        # ``TransportError`` is the base for ALL httpx transport-layer failures
        # — ConnectError, TimeoutException, ReadError, RemoteProtocolError, etc.
        # (codex #558-PR3 nit): an interrupted response mid-download is transient
        # and must skip, not spuriously fail the network-dependent suite. HTTP
        # STATUS errors are ``HTTPStatusError`` (NOT a TransportError), so they
        # still fall through to the 4xx-fails / 429+5xx-skips branch below.
        from httpx import TransportError as _HxTransport

        transient_types += [_HxTransport]
    except Exception:  # pragma: no cover - httpx not present
        pass
    if transient_types and isinstance(exc, tuple(transient_types)):
        return True
    # HTTP-status-bearing errors (HfHubHTTPError / requests.HTTPError / httpx):
    # skip ONLY 429 + 5xx (transient); permanent 4xx must fail.
    status = _http_status_of(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    return False


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION
        )
    except Exception as exc:  # pragma: no cover - offline & uncached
        # Skip ONLY on a transient network/cache-miss signal; a permanent 4xx
        # (bad creds / deleted artifact / invalid revision) or any other error
        # must FAIL, not silently green the enforcement suite (codex #558-PR3).
        if not _is_offline_skippable(exc):
            raise
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not "
            "cached and no network — runtime enforcement tests require it"
        )


@pytest.fixture(scope="module")
def hermes_grammar(tok):
    # We reach here only with llguidance present (the enforcement tests are
    # ``@_requires_llguidance``) AND the pinned tokenizer available (``tok``
    # succeeded). Under those conditions the hermes parser DOES declare a
    # ``structure_info`` (proven in test_structure_info_hermes_qwen_558.py), so
    # ``build_tool_grammar`` returning ``None`` means the production builder
    # REGRESSED and the enforcement suite must FAIL — not silently skip and go
    # green while the feature is broken (codex #558-PR3).
    pytest.importorskip("llguidance")
    from vllm_mlx.api.tool_grammar import build_tool_grammar
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    parser = HermesToolParser(tokenizer=tok)
    grammar = build_tool_grammar(TOOLS, "required", parser)
    assert grammar is not None, (
        "build_tool_grammar returned None with llguidance present and the "
        "hermes structure_info available — the grammar builder regressed"
    )
    return grammar


@pytest.fixture(scope="module")
def lltok(tok):
    # Same contract as ``hermes_grammar``: with llguidance present and the
    # pinned fast tokenizer available, ``build_lltokenizer`` MUST succeed. A
    # ``None`` here is a regression in the LLTokenizer factory, not a sanctioned
    # skip (codex #558-PR3).
    pytest.importorskip("llguidance")
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    llt = build_lltokenizer(tok)
    assert llt is not None, (
        "build_lltokenizer returned None for the pinned fast wire tokenizer "
        "with llguidance present — the LLTokenizer factory regressed"
    )
    return llt


# --------------------------------------------------------------------------
# tool_choice normalization — collision fix (pure Python, always runs).
# --------------------------------------------------------------------------
def test_normalize_required_is_enum_not_named():
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert _normalize_tool_choice_for_grammar("required") == {"mode": "required"}


@pytest.mark.parametrize("value", ["auto", None])
def test_normalize_auto_and_unset_are_auto_mode(value):
    # PR-5: auto (and unset/None, auto by default) map to the constrainable
    # ``{"mode": "auto"}`` — the optional-call auto grammar, NOT free-form.
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert _normalize_tool_choice_for_grammar(value) == {"mode": "auto"}


def test_normalize_none_is_unconstrained():
    # ``"none"`` = the model sees no tools (the #445 handler drops them). No
    # grammar at all -> ``None``.
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert _normalize_tool_choice_for_grammar("none") is None


def test_normalize_named_object_form():
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    choice = {"type": "function", "function": {"name": "get_time"}}
    assert _normalize_tool_choice_for_grammar(choice) == {
        "mode": "named",
        "name": "get_time",
    }


def test_normalize_bare_tool_name_string_is_never_named():
    # Deferred codex #2: a bare string that happens to equal a tool name is
    # an INVALID tool_choice enum, not a named selection. It must NOT be read
    # as a named choice (that requires the object form). Under PR-5 an
    # unrecognized bare string falls through to AUTO (the model may still call
    # or decline), NOT to a named/forced choice.
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert _normalize_tool_choice_for_grammar("get_time") == {"mode": "auto"}


def test_normalize_tool_named_required_only_via_object_form():
    # A tool literally named "required": under a bare string it's the enum
    # (mode=required, forces one of ANY tool); it is selectable as a NAMED
    # single tool ONLY via the object form. The two paths can never collide.
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert _normalize_tool_choice_for_grammar("required") == {"mode": "required"}
    assert _normalize_tool_choice_for_grammar(
        {"type": "function", "function": {"name": "required"}}
    ) == {"mode": "named", "name": "required"}


@pytest.mark.parametrize(
    "bad_function",
    [
        "get_weather",  # truthy string, not a dict
        ["get_weather"],  # truthy list
        123,  # truthy int
        {"name": 42},  # dict but name is not a str
        {"name": ""},  # dict but empty name
        {},  # empty dict (no name)
    ],
)
def test_normalize_malformed_function_degrades_to_none(bad_function):
    # codex #558-PR3 blocking: a dict-shaped tool_choice whose ``function`` is
    # truthy-but-not-a-dict (or a dict without a usable name) must degrade to
    # None (free-form), NOT reach ``.get`` on a non-dict and raise -> HTTP 500.
    from vllm_mlx.routes.chat import _normalize_tool_choice_for_grammar

    assert (
        _normalize_tool_choice_for_grammar(
            {"type": "function", "function": bad_function}
        )
        is None
    )


# --------------------------------------------------------------------------
# Cheap synchronous eligibility gate — must short-circuit the offload for the
# common no-tools / auto / none traffic (codex #558-PR3). Pure Python, always
# runs (no llguidance, no tokenizer).
# --------------------------------------------------------------------------
def test_eligible_false_when_no_tools():
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=None, tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_eligible_false_when_no_parser():
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    cfg = _CfgStub(None)  # no family parser configured
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_eligible_false_for_none(choice="none"):
    # ``"none"`` (model sees no tools) must NOT enter the thread-pool offload.
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice=choice)
    assert _tool_grammar_eligible(cfg, req) is False


@pytest.mark.parametrize("choice", ["auto", None])
def test_eligible_true_for_auto_and_unset(choice):
    # PR-5 default-on: auto (and unset/None, auto by default) IS a constrainable
    # mode and MUST be eligible — the auto-path grammar is the whole point of
    # PR-5. Previously (PR-3/4) auto short-circuited to free-form.
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice=choice)
    assert _tool_grammar_eligible(cfg, req) is True


def test_eligible_true_for_required_and_named():
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    cfg = _CfgStub("hermes")
    tools = [_FunctionTool("get_time")]
    assert _tool_grammar_eligible(cfg, _RequestStub(tools, "required")) is True
    named = {"type": "function", "function": {"name": "get_time"}}
    assert _tool_grammar_eligible(cfg, _RequestStub(tools, named)) is True


@pytest.mark.parametrize("value", ["0", "off", "false", "OFF", "False", " 0 "])
def test_eligible_false_when_explicitly_opted_out(monkeypatch, value):
    # PR-5 is DEFAULT-ON / OPT-OUT: only the explicit ``0``/``off``/``false``
    # values (case-insensitive, whitespace-trimmed) disable the constraint.
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", value)
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "", "2"])
def test_eligible_true_for_non_optout_values(monkeypatch, value):
    # PR-5 OPT-OUT: anything that is NOT ``0``/``off``/``false`` leaves the
    # constraint ON — including unrecognized values and the empty string (the
    # denylist is narrow by design).
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", value)
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is True


def test_eligible_true_by_default_when_env_unset(monkeypatch):
    # PR-5 ships ON by default: with the env var ABSENT the constraint activates.
    # Overrides the module autouse opt-in fixture by deleting the var to prove
    # the true unset default, not the fixture-forced ``"1"``.
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    monkeypatch.delenv("RAPID_MLX_CONSTRAIN_TOOLS", raising=False)
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is True


def test_eligible_true_for_reasonable_schema():
    # A normal-sized, shallow schema is eligible (opt-in enabled by the module
    # autouse fixture). The PR-3b size/depth/count caps must NOT reject ordinary
    # tools.
    from vllm_mlx.routes.chat import _tool_grammar_eligible

    ok = _FunctionTool(
        "get_weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[ok], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is True


def test_eligible_false_for_oversized_schema():
    # codex #558-PR3 blocking (restored in PR-3b): a pathologically LARGE client
    # schema must be rejected before it can drive an unbounded compile on the
    # shared executor.
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
        _tool_grammar_eligible,
    )

    # A schema whose serialized size blows past the cap.
    big_props = {
        f"field_{i}": {"type": "string", "description": "x" * 64}
        for i in range(_TOOL_GRAMMAR_MAX_SCHEMA_BYTES // 32)
    }
    huge = _FunctionTool(
        "bloat",
        parameters={"type": "object", "properties": big_props},
    )
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[huge], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_eligible_false_for_unicode_escaped_oversized_schema():
    # codex #558-PR3 blocking (round-2): ``len(str(...))`` under-counts — a
    # Unicode / JSON-escaped char occupies more BYTES than source code points, so
    # a schema whose CODE-POINT length is under the cap but whose escaped-byte
    # length is over it must still be rejected. Use emoji (1 code point ->
    # ``😀`` = 12 escaped chars) so a source that is well under the cap
    # by ``len(str())`` blows past it by true serialized bytes.
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
        _tool_grammar_eligible,
        _tools_within_grammar_bounds,
    )

    # ~1/6 of the cap in code points, but each emoji escapes to 12 chars -> ~2x
    # the byte cap once serialized. A naive ``len(str())`` walker would PASS this.
    emoji_blob = "\U0001f600" * (_TOOL_GRAMMAR_MAX_SCHEMA_BYTES // 6)
    assert len(emoji_blob) < _TOOL_GRAMMAR_MAX_SCHEMA_BYTES, (
        "test setup: code-point length must be under the cap so only true-byte "
        "counting rejects it"
    )
    tool = _FunctionTool(
        "u",
        parameters={"type": "object", "description": emoji_blob, "properties": {}},
    )
    # Direct walker + full eligibility both reject it on true serialized bytes.
    assert _tools_within_grammar_bounds([tool]) is False
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[tool], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_charge_scalar_rejects_giant_int_without_rendering(monkeypatch):
    # codex #558-PR3 blocking (round-3): Python ints are ARBITRARY precision, so
    # ``json.dumps(huge_int)`` renders an attacker-sized decimal string BEFORE
    # the budget check — an event-loop-blocking allocation. The O(1)
    # ``bit_length`` preflight must reject an over-budget int WITHOUT ever
    # rendering it. Prove that DETERMINISTICALLY (no wall-clock): monkeypatch the
    # module's ``json.dumps`` to explode if it is ever handed a large int, and
    # use an int just past the byte cutoff (not a multi-megabyte monster).
    from vllm_mlx.routes import chat as chat_mod
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
        _BoundsExceededError,
        _charge_json_scalar_bytes,
        _tools_within_grammar_bounds,
    )

    _real_dumps = chat_mod.json.dumps

    def _guard_dumps(obj, *a, **k):
        # The preflight must reject an over-budget int BEFORE dumps is called.
        # Mirror production's conservative digit lower bound ``((b-1)*3)//10 + 1``
        # (codex #558-PR3 round-6 nit — ``bit_length // 4 + 1`` over-counts).
        if isinstance(obj, int) and not isinstance(obj, bool):
            _b = obj.bit_length()
            _min = (1 if _b == 0 else ((_b - 1) * 3) // 10 + 1) + (1 if obj < 0 else 0)
            if _min > _TOOL_GRAMMAR_MAX_SCHEMA_BYTES:
                raise AssertionError(
                    "json.dumps was called on an over-budget int — the "
                    "bit_length preflight failed to reject it first (codex "
                    "#558-PR3)"
                )
        return _real_dumps(obj, *a, **k)

    monkeypatch.setattr(chat_mod.json, "dumps", _guard_dumps)

    # An int whose minimum decimal-digit count (bit_length/4 + 1) exceeds the
    # byte cap: rejected by the O(1) preflight, dumps never called. Chosen just
    # past the cutoff — no multi-megabyte big-int construction.
    over = 1 << (_TOOL_GRAMMAR_MAX_SCHEMA_BYTES * 4 + 8)  # bit_length//4 > cap
    budget = [_TOOL_GRAMMAR_MAX_SCHEMA_BYTES]
    with pytest.raises(_BoundsExceededError):
        _charge_json_scalar_bytes(over, budget)  # must reject, dumps not called

    # End-to-end through the tools walker: a schema carrying the over-budget int
    # default is rejected (free-form fallback) and dumps is still never rendered.
    tool = _FunctionTool(
        "n",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer", "default": over}},
        },
    )
    assert _tools_within_grammar_bounds([tool]) is False

    # A small int is charged normally (dumps IS called, harmlessly) and fits.
    ok = _FunctionTool(
        "n",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer", "default": 42}},
        },
    )
    assert _tools_within_grammar_bounds([ok]) is True


def test_int_digit_lower_bound_does_not_over_reject_exactly_fitting_scalar():
    # codex #558-PR3 round-6 nit: the O(1) int preflight must use a TRUE lower
    # bound on decimal-digit count. ``bit_length // 4 + 1`` OVER-counts (8 and 9
    # have bit_length 4 => it claims 2 digits though they are 1), so an int that
    # exactly fits the remaining budget could be spuriously rejected. Prove the
    # corrected ``((b-1)*3)//10 + 1`` bound accepts every single-digit int with a
    # 1-byte budget, and never OVER-estimates any int's real decimal length.
    from vllm_mlx.routes.chat import _BoundsExceededError, _charge_json_scalar_bytes

    # 8 and 9 (bit_length 4) each fit a 1-byte budget — the old formula rejected.
    for v in (0, 1, 7, 8, 9):
        budget = [1]  # exactly one byte, the width of a single decimal digit
        _charge_json_scalar_bytes(v, budget)  # must NOT raise
        assert budget[0] == 0, f"{v} should consume exactly its 1 digit byte"

    # The lower bound must never exceed the true decimal length for any int, so
    # a value given a budget equal to its real rendered length always fits.
    import json as _json

    for v in (10, 99, 100, 128, 255, 999, -1, -8, -9, -100, 1 << 20, -(1 << 20)):
        true_len = len(_json.dumps(v).encode("utf-8"))
        budget = [true_len]
        _charge_json_scalar_bytes(v, budget)  # exactly fits, must NOT raise
        assert budget[0] == 0
        # One byte short must reject (via the O(1) preflight for large ints, or
        # the post-charge ``budget < 0`` check for small ones — never a false
        # accept). The lower bound is conservative, so it never over-rejects the
        # exact-fit case above but always catches a genuine overflow here.
        tight = [true_len - 1]
        with pytest.raises(_BoundsExceededError):
            _charge_json_scalar_bytes(v, tight)


def test_walker_charges_commas_between_not_before_first_member():
    # codex #558-PR3 nit: the walker must charge a ``,`` only BETWEEN members
    # (N members => N-1 commas), matching the compact ``json.dumps`` envelope, so
    # a schema materially under 64 KiB is not spuriously rejected. Verify the
    # estimate never UNDER-counts real compact bytes (safe) and does not
    # OVER-count by more than the fixed key-quote overhead (tight).
    import json

    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
        _BoundsExceededError,
        _walk_size_and_depth,
    )

    # A representative multi-member object with nested list + dict.
    obj = {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f", "k"]},
            },
            "required": ["city"],
        },
    }
    compact_bytes = len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))

    # Charged cost = starting budget minus what remains after the walk.
    budget = [_TOOL_GRAMMAR_MAX_SCHEMA_BYTES]
    _walk_size_and_depth(obj, budget, 0)
    charged = _TOOL_GRAMMAR_MAX_SCHEMA_BYTES - budget[0]

    # The walker charges keys with the same ``"..."`` quotes ``json.dumps`` emits
    # and exactly N-1 commas per container, so its estimate must EQUAL the compact
    # byte count — no under-count (an oversized schema could slip the cap) and no
    # over-count. Assert EXACT equality, not a slack bound (codex #558-PR3
    # round-7 blocking): this fixture has 5 nonempty dicts, so the old phantom-
    # first-member-comma bug over-charged by exactly 5 — a ``<= 8`` slack would
    # NOT have caught it. Exact equality does: any phantom comma breaks it.
    assert charged == compact_bytes, (
        f"walker charged {charged} != compact {compact_bytes} — a phantom comma "
        "or miscounted quote inflates/deflates the estimate (codex #558-PR3). "
        "The estimate must byte-match compact json.dumps exactly."
    )

    # Exact-boundary: a single scalar just at the cap is accepted; one byte over
    # is rejected. Build a string whose ASCII escaped length is the budget.
    from vllm_mlx.routes.chat import _charge_json_scalar_bytes

    # ``json.dumps("a"*n)`` == n + 2 (surrounding quotes). Pick n so it exactly
    # consumes the whole budget.
    n = _TOOL_GRAMMAR_MAX_SCHEMA_BYTES - 2
    at_cap = [_TOOL_GRAMMAR_MAX_SCHEMA_BYTES]
    _charge_json_scalar_bytes("a" * n, at_cap)  # exactly fits
    assert at_cap[0] == 0
    over = [_TOOL_GRAMMAR_MAX_SCHEMA_BYTES]
    with pytest.raises(_BoundsExceededError):
        _charge_json_scalar_bytes("a" * (n + 1), over)  # one byte over


def test_eligible_false_for_overdeep_schema():
    # codex #558-PR3 blocking (restored in PR-3b): a pathologically DEEP nested
    # schema is rejected.
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_DEPTH,
        _tool_grammar_eligible,
    )

    # Build a schema nested well past the depth cap.
    node: dict = {"type": "string"}
    for _ in range(_TOOL_GRAMMAR_MAX_SCHEMA_DEPTH + 5):
        node = {"type": "object", "properties": {"inner": node}}
    deep = _FunctionTool("deep", parameters=node)
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[deep], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_depth_cap_counts_containers_only_not_scalar_leaves():
    # codex #558-PR3 round-7 nit: the depth cap counts CONTAINER (object/array)
    # nesting only. A scalar leaf is not a nesting level, so it must never trip
    # the cap by being one level below the deepest container — otherwise the
    # documented container-depth cap is content-dependent (a schema whose deepest
    # container holds a scalar would reject one level shallower than one that
    # doesn't). Build a chain of EXACTLY ``MAX`` nested objects with a scalar leaf
    # at the bottom: the scalar sits at container-depth ``MAX`` and must be
    # accepted; adding ONE more container tips it over and must reject.
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_DEPTH,
        _BoundsExceededError,
        _walk_size_and_depth,
    )

    # The cap admits container nesting for depths ``0 .. MAX`` inclusive, i.e.
    # ``MAX + 1`` nested containers (the innermost is entered at depth ``MAX``,
    # and ``MAX > MAX`` is false). Build exactly that many nested objects with a
    # scalar leaf at the bottom. Under the FIXED walker the scalar leaf (visited
    # at depth ``MAX + 1``) is exempt, so the whole chain is accepted. Under the
    # OLD walker the scalar's depth-``MAX + 1`` check would REJECT this exact
    # chain — the content-dependent bug this test guards.
    n_at_cap = _TOOL_GRAMMAR_MAX_SCHEMA_DEPTH + 1
    at_cap: dict = {"leaf": "x"}
    for _ in range(n_at_cap - 1):
        at_cap = {"inner": at_cap}
    budget = [1 << 20]  # generous size budget; we test depth, not size
    _walk_size_and_depth(at_cap, budget, 0)  # scalar leaf at max depth: NO raise

    # One additional CONTAINER pushes the innermost object past the cap: it is
    # entered at depth ``MAX + 1`` (``> MAX``) and must reject.
    over = {"inner": at_cap}
    with pytest.raises(_BoundsExceededError):
        _walk_size_and_depth(over, [1 << 20], 0)


def test_eligible_false_for_oversized_tool_name():
    # codex #558-PR3 blocking (restored in PR-3b): the bound must include tool
    # NAMES, not just parameters — an oversized name is compiled into the grammar
    # too and must count against the byte cap.
    from vllm_mlx.routes.chat import (
        _TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
        _tool_grammar_eligible,
    )

    huge_name = "x" * (_TOOL_GRAMMAR_MAX_SCHEMA_BYTES + 10)
    tool = _FunctionTool(huge_name, parameters={"type": "object", "properties": {}})
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[tool], tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


def test_eligible_false_for_too_many_tools():
    # codex #558-PR3 blocking (restored in PR-3b): a pathological tool COUNT
    # (huge alternation width) must be rejected before the compile.
    from vllm_mlx.routes.chat import _TOOL_GRAMMAR_MAX_TOOLS, _tool_grammar_eligible

    tools = [
        _FunctionTool(f"tool_{i}", parameters={"type": "object", "properties": {}})
        for i in range(_TOOL_GRAMMAR_MAX_TOOLS + 1)
    ]
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=tools, tool_choice="required")
    assert _tool_grammar_eligible(cfg, req) is False


# --------------------------------------------------------------------------
# #561: oversized schema on the ACTIVE (default-on) constrained path is a HARD
# HTTP 400 — we do NOT silently drop the structural guarantee. The legacy
# free-form fallback survives ONLY on the explicit opt-OUT path.
# --------------------------------------------------------------------------
def _oversized_tool():
    from vllm_mlx.routes.chat import _TOOL_GRAMMAR_MAX_SCHEMA_BYTES

    big_props = {
        f"field_{i}": {"type": "string", "description": "x" * 64}
        for i in range(_TOOL_GRAMMAR_MAX_SCHEMA_BYTES // 32)
    }
    return _FunctionTool(
        "bloat", parameters={"type": "object", "properties": big_props}
    )


@pytest.mark.parametrize("choice", ["required", "auto", None])
def test_oversized_schema_raises_400_on_active_path(monkeypatch, choice):
    # #561 operator decision: an oversized schema under the active constraint
    # (default-on, tools + parser + a constrainable tool_choice) is a hard 400 —
    # covers required AND the new auto path.
    from fastapi import HTTPException

    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_oversized_tool()], tool_choice=choice)
    with pytest.raises(HTTPException) as ei:
        _enforce_tool_grammar_bounds_or_400(cfg, req)
    assert ei.value.status_code == 400
    assert "grammar-compile bounds" in str(ei.value.detail)


@pytest.mark.parametrize("value", ["0", "off", "false"])
def test_oversized_schema_falls_back_when_opted_out(monkeypatch, value):
    # When the operator has explicitly opted OUT, the legacy free-form fallback
    # is preserved — an oversized schema must NOT 400 (backward compat).
    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", value)
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_oversized_tool()], tool_choice="required")
    # No raise — returns None (free-form fallback path).
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


@pytest.mark.parametrize("choice", ["none"])
def test_oversized_schema_no_400_for_none_choice(monkeypatch, choice):
    # ``"none"`` is not a constrainable mode (the model sees no tools), so the
    # active path is inactive and an oversized schema does not 400.
    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub("hermes")
    req = _RequestStub(tools=[_oversized_tool()], tool_choice=choice)
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


def test_reasonable_schema_no_400_on_active_path(monkeypatch):
    # A normal, in-bounds schema must NOT 400 on the active path.
    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub("hermes")
    ok = _FunctionTool(
        "get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    req = _RequestStub(tools=[ok], tool_choice="auto")
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


# --------------------------------------------------------------------------
# #1144: the oversized -> 400 decision is gated on parser grammar-CAPABILITY,
# not merely on ``tool_call_parser`` being set. A grammar-capable parser
# (hermes/qwen, ``structure_info`` -> wire triple) keeps the #561 400; a
# non-grammar-capable parser (ABC default ``structure_info`` -> None) was never
# going to be decoder-constrained, so an oversized schema falls back to FREE-FORM
# instead of 400.
# --------------------------------------------------------------------------
# A registered parser that does NOT override ``structure_info`` (grammar-incapable).
# (``deepseek_v3`` became grammar-capable in #558 E1; ``mistral`` is the stable
# non-grammar example — same family the warmup suite uses for this role.)
_NON_GRAMMAR_PARSER = "mistral"


def test_supports_grammar_probe_matches_capability():
    # The cheap route probe reports True only for grammar-capable parsers.
    from vllm_mlx.routes.chat import _tool_parser_supports_grammar

    assert _tool_parser_supports_grammar(_CfgStub("hermes")) is True
    assert _tool_parser_supports_grammar(_CfgStub("qwen")) is True
    assert _tool_parser_supports_grammar(_CfgStub(_NON_GRAMMAR_PARSER)) is False
    # Unknown / unset parser name -> not capable (free-form), never raises.
    assert _tool_parser_supports_grammar(_CfgStub("no_such_parser_xyz")) is False
    assert _tool_parser_supports_grammar(_CfgStub(None)) is False


@pytest.mark.parametrize("parser", ["hermes", "qwen"])
@pytest.mark.parametrize("choice", ["required", "auto", None])
def test_oversized_schema_still_400_for_grammar_capable_parser(
    monkeypatch, parser, choice
):
    # #561 regression guard (#1144): a grammar-CAPABLE parser keeps the hard 400
    # on an oversized schema across every constrainable tool_choice.
    from fastapi import HTTPException

    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub(parser)
    req = _RequestStub(tools=[_oversized_tool()], tool_choice=choice)
    with pytest.raises(HTTPException) as ei:
        _enforce_tool_grammar_bounds_or_400(cfg, req)
    assert ei.value.status_code == 400
    assert "grammar-compile bounds" in str(ei.value.detail)


@pytest.mark.parametrize("choice", ["required", "auto", None])
def test_oversized_schema_falls_back_for_non_grammar_parser(monkeypatch, choice):
    # #1144 core fix: a non-grammar-capable parser (structure_info -> None) with
    # an oversized schema must NOT 400 — it was never going to be constrained, so
    # it falls back to free-form exactly like the pre-#558 behavior.
    from vllm_mlx.routes.chat import (
        _enforce_tool_grammar_bounds_or_400,
        _tool_grammar_constraint_active,
    )

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub(_NON_GRAMMAR_PARSER)
    req = _RequestStub(tools=[_oversized_tool()], tool_choice=choice)
    # Path is inactive for a non-grammar parser, so no 400 (free-form fallback).
    assert _tool_grammar_constraint_active(cfg, req) is False
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


@pytest.mark.parametrize("choice", ["required", "auto", None])
def test_normal_schema_free_form_for_non_grammar_parser(monkeypatch, choice):
    # #1144: a non-grammar-capable parser with a normal in-bounds schema stays
    # free-form — never eligible for the constrained offload, never 400.
    from vllm_mlx.routes.chat import (
        _enforce_tool_grammar_bounds_or_400,
        _tool_grammar_eligible,
    )

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub(_NON_GRAMMAR_PARSER)
    ok = _FunctionTool(
        "get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    req = _RequestStub(tools=[ok], tool_choice=choice)
    assert _tool_grammar_eligible(cfg, req) is False
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


# --------------------------------------------------------------------------
# Harmony AUTO opt-out (#558 +gpt-oss) — auto path stays free-form even though
# harmony is now grammar-CAPABLE (structure_info overridden), because its
# <|channel|> trigger is shared with non-tool responses (TOOL_GRAMMAR_AUTO_SAFE
# = False). required/named remain constrained.
# --------------------------------------------------------------------------
def test_auto_safe_probe_matches_capability():
    from vllm_mlx.routes.chat import _tool_parser_auto_safe

    # Single-special-token-trigger families are auto-safe.
    assert _tool_parser_auto_safe(_CfgStub("hermes")) is True
    assert _tool_parser_auto_safe(_CfgStub("qwen")) is True
    # Harmony opts out (shared <|channel|> trigger).
    assert _tool_parser_auto_safe(_CfgStub("harmony")) is False
    # Unknown / unset -> permissive default (not constrained anyway).
    assert _tool_parser_auto_safe(_CfgStub("no_such_parser_xyz")) is True
    assert _tool_parser_auto_safe(_CfgStub(None)) is True


@pytest.mark.parametrize("choice", ["auto", None])
def test_harmony_auto_path_inactive_free_form(choice):
    # Harmony is grammar-capable, but on AUTO the constraint path is INACTIVE
    # (declines the grammar) — so it composes like a free-form path: no grammar,
    # and (below) no #561 400 on an oversized schema.
    from vllm_mlx.routes.chat import _tool_grammar_constraint_active

    cfg = _CfgStub("harmony")
    ok = _FunctionTool(
        "get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert _tool_grammar_constraint_active(cfg, _RequestStub([ok], choice)) is False


def test_harmony_required_and_named_paths_active():
    # required/named ARE constrained for harmony (a forced call is what the
    # caller asked for).
    from vllm_mlx.routes.chat import _tool_grammar_constraint_active

    cfg = _CfgStub("harmony")
    ok = _FunctionTool(
        "get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert _tool_grammar_constraint_active(cfg, _RequestStub([ok], "required")) is True
    named = {"type": "function", "function": {"name": "get_weather"}}
    assert _tool_grammar_constraint_active(cfg, _RequestStub([ok], named)) is True


@pytest.mark.parametrize("choice", ["auto", None])
def test_harmony_auto_oversized_schema_no_400(monkeypatch, choice):
    # AUTO non-regression: harmony gaining required/named grammar support must
    # NOT start 400-ing oversized schemas on the auto path (it declines the auto
    # grammar, so an oversized schema stays free-form — exactly as before).
    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub("harmony")
    req = _RequestStub(tools=[_oversized_tool()], tool_choice=choice)
    assert _enforce_tool_grammar_bounds_or_400(cfg, req) is None


def test_harmony_required_oversized_schema_400(monkeypatch):
    # But required DOES keep the #561 hard 400 for harmony (it is constrained).
    from fastapi import HTTPException

    from vllm_mlx.routes.chat import _enforce_tool_grammar_bounds_or_400

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    cfg = _CfgStub("harmony")
    req = _RequestStub(tools=[_oversized_tool()], tool_choice="required")
    with pytest.raises(HTTPException) as ei:
        _enforce_tool_grammar_bounds_or_400(cfg, req)
    assert ei.value.status_code == 400


def test_supports_grammar_marker_declared_for_in_tree_parsers():
    # In-tree DISCOVERABILITY guard (#1149 codex): the explicit
    # ``SUPPORTS_GRAMMAR`` marker itself must match the ``structure_info``
    # override across ALL registered parsers, so the grep-able capability list
    # stays honest — a new grammar-capable in-tree parser can't ship WITHOUT the
    # marker (which would rot the list), and a non-capable parser can't ship a
    # stale ``True`` (spurious 400s). NB: this asserts the marker DIRECTLY, not
    # the derived ``supports_grammar()`` — the runtime inference net (next test)
    # would otherwise mask a missing marker and make this a tautology.
    #
    # SCOPE (#1149 codex): restrict to IN-TREE parser classes (module under
    # ``vllm_mlx.tool_parsers``). An out-of-tree parser MAY legitimately override
    # ``structure_info`` WITHOUT the marker and still work via the inference net
    # (proven in ``test_supports_grammar_infers_capability_without_marker``), so
    # asserting the marker on it would contradict that compatibility contract.
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    names = ToolParserManager.list_registered()
    assert names, "no tool parsers registered"
    checked = 0
    for name in names:
        cls = ToolParserManager.get_tool_parser(name)
        if not cls.__module__.startswith("vllm_mlx.tool_parsers"):
            continue  # out-of-tree parser: covered by the inference net, not the marker
        checked += 1
        overrides_structure_info = cls.structure_info is not ToolParser.structure_info
        assert bool(cls.SUPPORTS_GRAMMAR) == overrides_structure_info, (
            f"{cls.__name__} (parser '{name}'): SUPPORTS_GRAMMAR marker="
            f"{cls.SUPPORTS_GRAMMAR} but structure_info overridden="
            f"{overrides_structure_info}; declare the marker to match (#1144)"
        )
    assert checked, "no in-tree tool parsers checked"


def test_supports_grammar_infers_capability_without_marker():
    # Runtime ROBUSTNESS net (#1149 codex): an out-of-tree parser that overrides
    # ``structure_info`` but OMITS the marker is STILL grammar-capable via
    # structural inference — no silent regression of grammar / #561 enforcement.
    # A plain parser (no override, no marker) is NOT capable. This exercises the
    # inference branch independently of the in-tree marker guard above.
    from vllm_mlx.tool_parsers import ToolParserManager
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    class _OverrideNoMarker(ToolParser):
        EXPECTED_WIRE_FORMATS = ("tool_call_json",)

        def extract_tool_calls(self, model_output, request=None):
            return None

        def structure_info(self):
            return lambda name: None

    class _PlainNoOverride(ToolParser):
        EXPECTED_WIRE_FORMATS = ("tool_call_json",)

        def extract_tool_calls(self, model_output, request=None):
            return None

    assert _OverrideNoMarker.SUPPORTS_GRAMMAR is False  # marker deliberately unset
    assert _OverrideNoMarker.supports_grammar() is True  # inferred from override
    assert _PlainNoOverride.supports_grammar() is False
    # And the concrete in-tree capable parsers report capable.
    assert ToolParserManager.get_tool_parser("hermes").supports_grammar() is True
    assert ToolParserManager.get_tool_parser("qwen").supports_grammar() is True


def test_offline_skip_classifies_http_status():
    # codex #558-PR3 blocking: only TRANSIENT signals skip; a permanent 4xx
    # (bad creds / deleted artifact / invalid revision) must FAIL, not silently
    # green the suite.
    from vllm_mlx.routes import chat  # noqa: F401  (ensure module import order)

    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _HTTPError(Exception):
        def __init__(self, code):
            self.response = _Resp(code)

    # 429 + 5xx -> transient -> skippable.
    assert _is_offline_skippable(_HTTPError(429)) is True
    assert _is_offline_skippable(_HTTPError(500)) is True
    assert _is_offline_skippable(_HTTPError(503)) is True
    # permanent 4xx -> must FAIL (not skip).
    assert _is_offline_skippable(_HTTPError(401)) is False
    assert _is_offline_skippable(_HTTPError(403)) is False
    assert _is_offline_skippable(_HTTPError(404)) is False
    assert _is_offline_skippable(_HTTPError(422)) is False
    # a non-HTTP programming error is not skippable either.
    assert _is_offline_skippable(ValueError("boom")) is False

    # httpx transport-layer failures (interrupted response mid-download) are
    # transient -> skippable, but an httpx HTTP-STATUS error still classifies by
    # code (codex #558-PR3 nit).
    httpx = pytest.importorskip("httpx")
    assert _is_offline_skippable(httpx.ReadError("peer reset")) is True
    assert _is_offline_skippable(httpx.RemoteProtocolError("truncated")) is True
    assert _is_offline_skippable(httpx.ConnectError("refused")) is True
    req = httpx.Request("GET", "https://example.invalid")
    resp_404 = httpx.Response(404, request=req)
    resp_503 = httpx.Response(503, request=req)
    assert (
        _is_offline_skippable(
            httpx.HTTPStatusError("nf", request=req, response=resp_404)
        )
        is False
    )
    assert (
        _is_offline_skippable(
            httpx.HTTPStatusError("busy", request=req, response=resp_503)
        )
        is True
    )


def test_route_offload_gated_on_eligibility_in_source():
    # The off-loop build helper must run the cheap gate + admission SYNCHRONOUSLY
    # before submitting, so no-tools / auto traffic and at-capacity floods never
    # enter the thread pool (codex #558-PR3). Source-position tripwire — the
    # BEHAVIORAL guarantees are proven by
    # ``test_ineligible_request_never_enters_heavy_build_path``,
    # ``test_offload_at_capacity_never_submits`` and
    # ``test_admission_slot_not_released_until_compile_finishes_on_cancel``.
    # The route delegates the whole thing to ``_offload_tool_grammar_build``.
    import inspect

    from vllm_mlx.routes import chat as chat_mod

    # The route calls the extracted helper (which owns the gate + admission).
    route_src = inspect.getsource(chat_mod._create_chat_completion_impl)
    assert "_offload_tool_grammar_build(engine, cfg, request)" in route_src, (
        "the route must delegate the off-loop build to _offload_tool_grammar_build"
    )

    src = inspect.getsource(chat_mod._offload_tool_grammar_build)
    gate = src.find("_tool_grammar_eligible(cfg, request)")
    admit = src.find("_try_admit_tool_grammar_build()")
    offload = src.find("_get_tool_grammar_build_executor().submit(")
    assert gate != -1, "helper must gate the offload on _tool_grammar_eligible"
    assert admit != -1, "helper must reserve admission via _try_admit before submit"
    assert offload != -1, (
        "helper must offload the build by submitting to the dedicated bounded pool"
    )
    assert gate < admit < offload, (
        "eligibility gate then admission must precede the off-loop submit"
    )


def test_route_offload_uses_dedicated_bounded_pool_not_semaphore_in_source():
    # codex #558-PR3 blocking (restored in PR-3b): the compile must run on a
    # DEDICATED bounded pool (slot held until the compile finishes, survives
    # cancel, loop-agnostic) — NOT the default executor (unbounded queue) and
    # NOT an asyncio semaphore (permit leaks on cancel, binds to one loop).
    # Assert the structural guarantee in the offload-helper source + module
    # surface.
    import inspect

    from vllm_mlx.routes import chat as chat_mod

    src = inspect.getsource(chat_mod._offload_tool_grammar_build)
    assert "_get_tool_grammar_build_executor()" in src, (
        "the helper must dispatch the compile onto the dedicated bounded pool"
    )
    assert "asyncio.to_thread" not in src, (
        "the helper must NOT use asyncio.to_thread (unbounded default-executor "
        "queue — codex #558-PR3)"
    )
    assert "_get_tool_grammar_build_semaphore" not in src, (
        "the helper must NOT gate the offload on an event-loop-bound semaphore "
        "(codex #558-PR3: permit leaks on cancel, binds to one loop)"
    )
    assert not hasattr(chat_mod, "_get_tool_grammar_build_semaphore"), (
        "the loop-bound build semaphore must not exist"
    )
    # The slot must be released via the underlying future's done-callback, NOT a
    # try/finally around the await (which fires on cancel while the worker still
    # compiles — codex #558-PR3 blocking). Assert the submit + done-callback +
    # wrap_future shape.
    assert (
        "add_done_callback(lambda" in src and "_release_tool_grammar_build()" in src
    ), (
        "the helper must release the admission slot via the future's "
        "add_done_callback (fires only when the compile actually finishes), not "
        "a try/finally around the await"
    )
    assert "asyncio.wrap_future(" in src, (
        "the helper must await the submitted future via asyncio.wrap_future so a "
        "cancelled await does not pre-release the admission slot"
    )
    # The wrapped future MUST be shielded (codex #558-PR3 round-6 blocking): a
    # bare ``await asyncio.wrap_future(fut)`` propagates a cancelled caller into
    # ``fut.cancel()``; if the work item is still QUEUED (all workers busy) the
    # cancel succeeds, the done-callback releases admission, yet the dead
    # ``_WorkItem`` still sits in the executor's unbounded queue — a submit/cancel
    # flood grows that queue past the admission cap. ``asyncio.shield`` stops the
    # cancel from reaching the underlying future so the compile runs (and its
    # queue slot is reclaimed) before admission is released.
    assert "asyncio.shield(" in src, (
        "the helper must shield the wrapped future so a cancelled caller does not "
        "cancel a still-queued compile and release its admission slot early "
        "(codex #558-PR3 round-6 blocking)"
    )
    # The dedicated pool exists and is bounded.
    ex = chat_mod._get_tool_grammar_build_executor()
    assert ex._max_workers == chat_mod._TOOL_GRAMMAR_MAX_BUILD_CONCURRENCY


def test_offload_at_capacity_never_submits():
    # codex #558-PR3 blocking (round-2, BEHAVIORAL): the off-loop build MUST gate
    # ``submit`` on ``_try_admit_tool_grammar_build`` — at capacity, no compile is
    # submitted to the executor (so the unbounded submission queue can never
    # fill), and the request falls back to free-form (None). Drive the REAL
    # ``_offload_tool_grammar_build`` with a mock executor and assert submit is
    # not called at capacity but IS called under capacity.
    import asyncio

    from vllm_mlx.routes import chat as chat_mod

    class _MockFuture:
        def add_done_callback(self, _cb):
            # Invoke immediately so the admission slot is released like a real
            # instantly-finishing compile (keeps the counter balanced).
            _cb(self)

        def result(self):
            return None

    class _MockExecutor:
        def __init__(self):
            self.submit_calls = 0

        def submit(self, *a, **k):
            self.submit_calls += 1
            return _MockFuture()

    mock_ex = _MockExecutor()
    saved_ex_getter = chat_mod._get_tool_grammar_build_executor
    saved_inflight = chat_mod._tool_grammar_inflight
    # Patch the executor getter + the build fn (so the mock future's None result
    # path is exercised without a real compile) and asyncio.wrap_future.
    chat_mod._get_tool_grammar_build_executor = lambda: mock_ex
    saved_wrap = asyncio.wrap_future

    async def _fake_wrap(fut):
        return fut.result()

    cfg = _CfgStub("hermes")
    engine = _EngineStub(tokenizer=object())
    request = _RequestStub([_FunctionTool("get_time")], "required")
    try:
        asyncio.wrap_future = _fake_wrap  # type: ignore[assignment]

        # AT CAPACITY: pre-fill in-flight to the cap so _try_admit refuses.
        chat_mod._tool_grammar_inflight = chat_mod._TOOL_GRAMMAR_MAX_INFLIGHT
        result = asyncio.run(chat_mod._offload_tool_grammar_build(engine, cfg, request))
        assert result is None, "at capacity the offload must fall back to free-form"
        assert mock_ex.submit_calls == 0, (
            "at capacity the route must NOT submit to the executor — its "
            "submission queue would otherwise grow unbounded (codex #558-PR3)"
        )

        # UNDER CAPACITY: _try_admit succeeds -> submit IS called exactly once.
        chat_mod._tool_grammar_inflight = 0
        asyncio.run(chat_mod._offload_tool_grammar_build(engine, cfg, request))
        assert mock_ex.submit_calls == 1, (
            "under capacity the eligible request must be submitted exactly once"
        )
        # The done-callback fired synchronously, so the slot is released.
        assert chat_mod._tool_grammar_inflight == 0, (
            "admission slot must be released once the compile finishes"
        )
    finally:
        chat_mod._get_tool_grammar_build_executor = saved_ex_getter
        asyncio.wrap_future = saved_wrap  # type: ignore[assignment]
        chat_mod._tool_grammar_inflight = saved_inflight


def test_bounded_admission_rejects_at_capacity():
    # codex #558-PR3 blocking (restored in PR-3b): admission caps in-flight
    # (running + queued) compiles so the executor's submission queue can't grow
    # unbounded. Past the cap, admission is refused (request falls back to
    # free-form).
    from vllm_mlx.routes import chat as chat_mod

    # Snapshot + reset the module counter so the test is order-independent.
    saved = chat_mod._tool_grammar_inflight
    chat_mod._tool_grammar_inflight = 0
    try:
        cap = chat_mod._TOOL_GRAMMAR_MAX_INFLIGHT
        for _ in range(cap):
            assert chat_mod._try_admit_tool_grammar_build() is True
        # At capacity: further admission refused.
        assert chat_mod._try_admit_tool_grammar_build() is False
        # Release one -> a slot frees -> admission succeeds again.
        chat_mod._release_tool_grammar_build()
        assert chat_mod._try_admit_tool_grammar_build() is True
        # Release everything we hold.
        for _ in range(cap):
            chat_mod._release_tool_grammar_build()
        # Release is floored at 0 (never negative).
        chat_mod._release_tool_grammar_build()
        assert chat_mod._tool_grammar_inflight == 0
    finally:
        chat_mod._tool_grammar_inflight = saved


def test_admission_slot_not_released_until_compile_finishes_on_cancel():
    # codex #558-PR3 blocking (round-2/3, BEHAVIORAL against the REAL helper): a
    # cancelled AWAIT (client disconnect) must NOT release the admission slot
    # while the worker thread is still compiling — otherwise a disconnect flood
    # releases slots early and more than the cap run at once. We drive the ACTUAL
    # ``_offload_tool_grammar_build`` (not a hand-rolled copy of its mechanism):
    # a real bounded pool runs a BLOCKED build, we cancel the coroutine
    # mid-compile, and assert the slot stays held until the (still-running)
    # compile finishes. If production regressed to an ``await``-level
    # ``try/finally`` release, this test would go red.
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vllm_mlx.routes import chat as chat_mod

    saved = chat_mod._tool_grammar_inflight
    saved_ex_getter = chat_mod._get_tool_grammar_build_executor
    chat_mod._tool_grammar_inflight = 0
    release_gate = threading.Event()  # blocks the "compile" until we let it end
    started = threading.Event()
    finished = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)

    # Make the REAL helper's build block: it submits _maybe_build_tool_grammar_
    # processor to the pool, so patch that to a blocking function. The done-
    # callback (attached by the helper) releases the slot when this returns.
    def _blocking_build(engine, cfg, request):
        started.set()
        release_gate.wait(timeout=5)
        finished.set()
        return None  # free-form result; we only care about the slot lifecycle

    chat_mod._get_tool_grammar_build_executor = lambda: pool

    async def _drive():
        cfg = _CfgStub("hermes")
        engine = _EngineStub(tokenizer=object())
        request = _RequestStub([_FunctionTool("get_time")], "required")
        # Launch the REAL helper as a task, then cancel it mid-compile.
        task = asyncio.ensure_future(
            chat_mod._offload_tool_grammar_build(engine, cfg, request)
        )
        for _ in range(600):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set(), "the real helper never submitted the compile"
        # A slot is reserved for the in-flight compile.
        assert chat_mod._tool_grammar_inflight == 1
        # Cancel the helper coroutine (client disconnect) WHILE the compile runs.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The compile is STILL running (release_gate not set): the slot must
        # still be held — a cancelled await must NOT pre-release it.
        assert not finished.is_set(), "test setup: compile finished too early"
        assert chat_mod._tool_grammar_inflight == 1, (
            "admission slot was released on cancel while the compile was still "
            "running — production must release via the future's done-callback, "
            "not an await-level try/finally"
        )
        # Let the compile finish; the done-callback releases the slot.
        release_gate.set()
        for _ in range(600):
            if chat_mod._tool_grammar_inflight == 0:
                break
            await asyncio.sleep(0.005)
        assert chat_mod._tool_grammar_inflight == 0, (
            "the done-callback must release the slot once the compile finishes"
        )

    try:
        # Patch the build fn the helper submits so it blocks.
        _orig_build = chat_mod._maybe_build_tool_grammar_processor
        chat_mod._maybe_build_tool_grammar_processor = _blocking_build
        try:
            asyncio.run(_drive())
        finally:
            chat_mod._maybe_build_tool_grammar_processor = _orig_build
    finally:
        release_gate.set()
        pool.shutdown(wait=True)
        chat_mod._get_tool_grammar_build_executor = saved_ex_getter
        chat_mod._tool_grammar_inflight = saved


def test_cancelled_caller_does_not_cancel_a_queued_compile():
    # codex #558-PR3 round-6 blocking (BEHAVIORAL): the round-6 bug is specific to
    # a QUEUED work item — all workers busy, a second compile sits in the pool's
    # internal queue. A bare ``await asyncio.wrap_future(fut)`` would, on caller
    # cancel, call ``fut.cancel()``; for a still-queued future that SUCCEEDS,
    # firing the done-callback (releasing admission) while the dead ``_WorkItem``
    # lingers in the executor's unbounded queue. We drive the REAL helper with a
    # 1-worker pool: worker A is blocked, request B is therefore QUEUED, we cancel
    # B's coroutine, and assert B's underlying future was NOT cancelled (shielded)
    # and B's admission slot stays held until B actually runs+finishes.
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vllm_mlx.routes import chat as chat_mod

    saved = chat_mod._tool_grammar_inflight
    saved_ex_getter = chat_mod._get_tool_grammar_build_executor
    saved_build = chat_mod._maybe_build_tool_grammar_processor
    chat_mod._tool_grammar_inflight = 0

    a_started = threading.Event()
    a_gate = threading.Event()  # holds worker A (and thus the single worker) busy
    b_ran = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)  # forces B to QUEUE behind A
    chat_mod._get_tool_grammar_build_executor = lambda: pool

    def _build(engine, cfg, request):
        # Distinguish A (first) from B (second) by a per-request marker.
        if getattr(request, "_which", None) == "A":
            a_started.set()
            a_gate.wait(timeout=5)
            return None
        b_ran.set()
        return None

    async def _drive():
        cfg = _CfgStub("hermes")
        engine = _EngineStub(tokenizer=object())
        req_a = _RequestStub([_FunctionTool("get_time")], "required")
        req_a._which = "A"
        req_b = _RequestStub([_FunctionTool("get_time")], "required")
        req_b._which = "B"

        task_a = asyncio.ensure_future(
            chat_mod._offload_tool_grammar_build(engine, cfg, req_a)
        )
        for _ in range(600):
            if a_started.is_set():
                break
            await asyncio.sleep(0.005)
        assert a_started.is_set(), "worker A never occupied the single pool slot"

        # B is admitted + submitted, but the single worker is busy on A, so B's
        # work item sits QUEUED. Both slots reserved.
        task_b = asyncio.ensure_future(
            chat_mod._offload_tool_grammar_build(engine, cfg, req_b)
        )
        await asyncio.sleep(0.05)
        assert chat_mod._tool_grammar_inflight == 2, (
            "both A (running) and B (queued) should hold admission slots"
        )
        assert not b_ran.is_set(), "B must still be QUEUED behind the busy worker"

        # Client B disconnects: cancel B's coroutine while its compile is QUEUED.
        task_b.cancel()
        try:
            await task_b
        except asyncio.CancelledError:
            pass
        # THE BUG: without shield, B's queued future would be cancelled here and
        # its slot released while the dead work item lingers in the queue. With
        # shield, B's slot stays held and B still runs when the worker frees up.
        assert chat_mod._tool_grammar_inflight == 2, (
            "a cancelled caller must NOT release a still-queued compile's slot "
            "(codex #558-PR3 round-6 blocking)"
        )
        assert not b_ran.is_set(), "B should not have run yet (A still blocks)"

        # Release A; the single worker then drains the still-live B, whose done-
        # callback releases BOTH slots. B genuinely ran (was not cancelled).
        a_gate.set()
        await task_a
        for _ in range(600):
            if chat_mod._tool_grammar_inflight == 0:
                break
            await asyncio.sleep(0.005)
        assert b_ran.is_set(), (
            "the shielded queued compile B must still run to completion, not be "
            "cancelled out of the queue"
        )
        assert chat_mod._tool_grammar_inflight == 0, (
            "both slots must be released once both compiles finish"
        )

    try:
        chat_mod._maybe_build_tool_grammar_processor = _build
        try:
            asyncio.run(_drive())
        finally:
            chat_mod._maybe_build_tool_grammar_processor = saved_build
    finally:
        a_gate.set()
        pool.shutdown(wait=True)
        chat_mod._get_tool_grammar_build_executor = saved_ex_getter
        chat_mod._tool_grammar_inflight = saved


@pytest.mark.parametrize(
    "tools, tool_choice, env",
    [
        ([_FunctionTool("get_time")], "required", "0"),  # opted out
        ([], "required", "1"),  # no tools
        ([_FunctionTool("get_time")], "none", "1"),  # unconstrained choice
        # NOTE (#558 PR-5): ``auto`` is NO LONGER ineligible — it is a
        # constrainable mode (auto-path grammar). It intentionally reaches the
        # build path now, so it is covered by the auto-mode enforcement tests in
        # test_tool_grammar_558.py, not here.
    ],
)
def test_ineligible_request_never_enters_heavy_build_path(
    monkeypatch, tools, tool_choice, env
):
    # codex #558-PR3 blocking (behavioral, not source-string): an INELIGIBLE
    # request (opted out, no tools, or "none") must NEVER reach the
    # client-schema grammar compile. ``_maybe_build_tool_grammar_processor``
    # re-runs the gate, so booby-trap ``build_tool_grammar`` to explode — if the
    # gate short-circuits (as it must), it's never called and we get ``None``;
    # if the gate were bypassed, the RuntimeError would surface. This proves the
    # gate behaviorally, independent of any source-position check.
    from vllm_mlx.api import tool_grammar as tg_mod
    from vllm_mlx.routes import chat as chat_mod

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", env)

    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise RuntimeError("heavy grammar compile reached for an ineligible request")

    monkeypatch.setattr(tg_mod, "build_tool_grammar", _boom)

    cfg = _CfgStub("hermes")
    engine = _EngineStub(tokenizer=object())
    request = _RequestStub(tools, tool_choice)
    assert chat_mod._tool_grammar_eligible(cfg, request) is False, (
        "fixture inputs must be ineligible"
    )
    assert chat_mod._maybe_build_tool_grammar_processor(engine, cfg, request) is None, (
        "ineligible request must fall back to free-form without compiling"
    )


# --------------------------------------------------------------------------
# build_lltokenizer.
# --------------------------------------------------------------------------
@_requires_llguidance
def test_build_lltokenizer_from_wire_tokenizer(tok):
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    llt = build_lltokenizer(tok)
    assert llt is not None
    assert llt.vocab_size > 0


@_requires_llguidance
def test_build_lltokenizer_none_on_junk_tokenizer():
    # A tokenizer with no fast internals and no backend_tokenizer -> None
    # (caller degrades to free-form), never an exception.
    from vllm_mlx.api.tool_grammar import build_lltokenizer

    class _Junk:
        is_fast = False

    assert build_lltokenizer(_Junk()) is None


# --------------------------------------------------------------------------
# GrammarLogitsProcessor — enforcement + cumulative-baseline fix.
# --------------------------------------------------------------------------
def _feed_cumulative(proc, lltok, tok, prompt_ids, gen_text):
    """Drive the processor the way mlx-lm does: full cumulative sequence each
    step (prompt + generated-so-far). Returns (blocked, blocked_gen_idx).
    """
    import mlx.core as mx
    import numpy as np

    vocab = lltok.vocab_size
    gen_ids = tok.encode(gen_text, add_special_tokens=False)
    cumulative = list(prompt_ids)
    masked = proc(mx.array(cumulative), mx.zeros((1, vocab)))
    for k, tid in enumerate(gen_ids):
        row = np.array(masked[0])
        allowed = bool(np.isfinite(row[tid]) and row[tid] > -1e30)
        if not allowed:
            return True, k
        cumulative.append(tid)
        masked = proc(mx.array(cumulative), mx.zeros((1, vocab)))
    return False, len(gen_ids)


@_requires_llguidance
def test_processor_baselines_past_prompt(tok, hermes_grammar, lltok):
    # MUST-FIX #1: the matcher must NOT be fed prompt tokens. mlx-lm hands the
    # full cumulative sequence each step; on the first call the processor must
    # baseline ``_prompt_len`` to the prompt length so only generated tokens
    # ever reach the matcher. A valid call over a non-empty prompt must pass.
    import mlx.core as mx

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    prompt_ids = tok.encode(
        "You are a helpful assistant. What's the weather in Paris?",
        add_special_tokens=False,
    )
    assert prompt_ids, "expected a non-empty prompt for the baseline test"

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    # First call: everything present is the prompt (no token sampled yet).
    proc(mx.array(prompt_ids), mx.zeros((1, lltok.vocab_size)))
    assert proc._prompt_len == len(prompt_ids), (
        "processor did not baseline the matcher past the prompt — it would "
        "feed prompt tokens into the grammar matcher and break enforcement"
    )
    # And the matcher has consumed nothing yet (prompt is not committed to it).
    assert proc._committed == len(prompt_ids)

    # A valid hermes call over that same prompt baseline is fully allowed.
    blocked, _ = _feed_cumulative(
        proc,
        lltok,
        tok,
        prompt_ids,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    )
    assert not blocked, "valid call over a non-empty prompt was wrongly blocked"


@_requires_llguidance
def test_processor_consumes_only_new_tail_each_step(tok, hermes_grammar, lltok):
    """PERF/CORRECTNESS: each decode step feeds the matcher ONLY the newly
    generated token(s), never the already-committed prefix.

    mlx-lm hands the FULL cumulative ``prompt + generated`` sequence every
    step. A naive processor would ``consume_token`` the whole sequence each
    call — O(n^2) total consumes AND a double-consume that the matcher would
    reject. The ``_committed`` offset must gate consumption so that (a) the
    prompt baseline is never consumed and (b) exactly ONE consume happens per
    NEW generated token per step. We spy on the real matcher's
    ``consume_token`` to assert the exact call schedule.
    """
    import mlx.core as mx

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    prompt_ids = tok.encode("Weather please.", add_special_tokens=False)
    assert prompt_ids, "expected a non-empty prompt for the tail-consume test"

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)

    # Spy on the matcher's consume_token: record the (call_index -> token) log.
    # ``LLMatcher.consume_token`` is a read-only Rust attribute, so we cannot
    # patch the method in place — instead wrap the whole matcher in a thin proxy
    # that delegates every attribute to the real matcher but records each
    # ``consume_token`` call.
    consumed: list[int] = []
    real_matcher = proc._matcher

    class _SpyMatcher:
        def consume_token(self, tok_id):
            consumed.append(int(tok_id))
            return real_matcher.consume_token(tok_id)

        def __getattr__(self, name):
            return getattr(real_matcher, name)

    proc._matcher = _SpyMatcher()

    vocab = lltok.vocab_size
    gen_ids = tok.encode(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        add_special_tokens=False,
    )

    # First call: only the prompt is present — NOTHING must be consumed.
    proc(mx.array(list(prompt_ids)), mx.zeros((1, vocab)))
    assert consumed == [], (
        "processor consumed prompt tokens into the matcher — the baseline "
        "gate is broken and the grammar would reject the prompt"
    )

    # Then feed one new token per step over the full cumulative sequence.
    cumulative = list(prompt_ids)
    for k, tid in enumerate(gen_ids):
        cumulative.append(tid)
        before = len(consumed)
        proc(mx.array(cumulative), mx.zeros((1, vocab)))
        # Exactly ONE new consume per step (the new tail token), never a
        # re-consume of the already-committed prefix.
        assert len(consumed) - before == 1, (
            f"step {k}: expected exactly 1 new consume (the new tail token), "
            f"got {len(consumed) - before} — the processor is re-consuming "
            "the committed prefix (O(n^2))"
        )
        assert consumed[-1] == tid, (
            f"step {k}: consumed token {consumed[-1]} != the new tail token {tid}"
        )

    # Total consumes == number of generated tokens (linear, not quadratic).
    assert consumed == list(gen_ids), (
        "the full consume log must equal exactly the generated tokens in order "
        f"({len(consumed)} consumes for {len(gen_ids)} generated tokens)"
    )
    # ``_committed`` advanced exactly over prompt + generated.
    assert proc._committed == len(prompt_ids) + len(gen_ids)


@_requires_llguidance
def test_processor_negative_controls_over_prompt(tok, hermes_grammar, lltok):
    # The load-bearing #558 proof, exercised through the cumulative-baseline
    # path (non-empty prompt). A hallucinated tool name, an off-schema
    # argument, and a bad enum are all grammar-blocked mid-stream.
    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    prompt_ids = tok.encode("Weather please.", add_special_tokens=False)

    def _run(gen_text):
        proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
        return _feed_cumulative(proc, lltok, tok, prompt_ids, gen_text)[0]

    assert _run('<tool_call>\n{"name": "get_stockquote'), (
        "hallucinated tool name was NOT blocked"
    )
    assert _run('<tool_call>\n{"name": "get_weather", "arguments": {"city": 4'), (
        "off-schema integer argument was NOT blocked"
    )
    assert _run(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "kelvin'
    ), "invalid enum value was NOT blocked"


def _row_after_full_call(proc, lltok, tok, prompt_ids, gen_text):
    """Baseline over ``prompt_ids``, feed a full valid call, and return the
    masked-logits row at the (accepting) state that follows the last token."""
    import mlx.core as mx
    import numpy as np

    vocab = lltok.vocab_size
    gen_ids = tok.encode(gen_text, add_special_tokens=False)
    cumulative = list(prompt_ids)
    proc(mx.array(cumulative), mx.zeros((1, vocab)))  # first call = prompt baseline
    cumulative.extend(gen_ids)
    masked = proc(mx.array(cumulative), mx.zeros((1, vocab)))
    return np.array(masked[0])


def _is_allowed(row, tid):
    import math

    v = float(row[tid])
    return math.isfinite(v) and v > -1e30


@_requires_llguidance
def test_stop_token_readmitted_only_at_accepting_state(tok, hermes_grammar, lltok):
    """0.10.16 dogfood P1-①: a forced ``required`` grammar must let the model
    TERMINATE after one call. The mechanism is the Gemma-4 failure in miniature:
    a model whose learned turn-terminator is a special token that is neither the
    grammar's single EOS nor a grammar literal is masked out at the accepting
    state, so under ``(tag)+`` it can only ever emit ANOTHER call — an infinite
    loop. ``GrammarLogitsProcessor`` now re-admits the model's stop tokens when —
    and ONLY when — the matcher ``is_accepting()``, so the model can stop like
    Qwen/gpt-oss already do.

    We use ``<|im_start|>`` as the stand-in stop token: a real special token on
    this tokenizer that is NOT part of the hermes tool grammar and NOT the
    grammar EOS (``<|im_end|>``), so the byte-regex ``TAG_TEXT`` tail can never
    match it. It is therefore masked at EVERY grammar state unless our
    re-admission fires."""
    import mlx.core as mx
    import numpy as np

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    stop_id = tok.convert_tokens_to_ids("<|im_start|>")
    assert isinstance(stop_id, int) and stop_id >= 0, (
        "expected <|im_start|> to be a single special token id on this tokenizer"
    )
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    prompt_ids = tok.encode("Weather please.", add_special_tokens=False)
    call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    # (1) WITH the stop id re-admitted: masked BEFORE the call (start state is
    # NON-accepting for required ``+``), un-masked AFTER a complete call.
    proc = GrammarLogitsProcessor(
        lltok, hermes_grammar, tokenizer=tok, stop_token_ids=[stop_id]
    )
    start_row = np.array(
        proc(mx.array(list(prompt_ids)), mx.zeros((1, lltok.vocab_size)))[0]
    )
    assert not _is_allowed(start_row, stop_id), (
        "stop token was re-admitted BEFORE the mandatory first call — the "
        "is_accepting gate is broken and required's force-≥1 guarantee is lost"
    )
    # The trigger IS forced open at the start (proves the grammar is live here).
    trigger_id = tok.convert_tokens_to_ids("<tool_call>")
    if isinstance(trigger_id, int) and trigger_id >= 0:
        assert _is_allowed(start_row, trigger_id), "forced trigger was masked at start"

    end_row = _row_after_full_call(proc, lltok, tok, prompt_ids, call)
    assert _is_allowed(end_row, stop_id), (
        "stop token was NOT re-admitted after a complete call — a Gemma-4-style "
        "model could never terminate and would loop the same call forever"
    )

    # (2) CONTROL: identical processor WITHOUT stop_token_ids keeps the special
    # token masked at the same accepting state — proving the re-admission (not
    # the grammar itself) is what un-masks it, and that we did not widen the
    # grammar's own accepted language.
    proc_no_stop = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    end_row_ctrl = _row_after_full_call(proc_no_stop, lltok, tok, prompt_ids, call)
    assert not _is_allowed(end_row_ctrl, stop_id), (
        "special token was allowed at the accepting state WITHOUT re-admission — "
        "the grammar language changed unexpectedly"
    )
    # The grammar EOS is (and stays) allowed at the accepting state either way —
    # this is exactly why Qwen/gpt-oss already terminate and are not regressed.
    if isinstance(eos_id, int) and eos_id >= 0:
        assert _is_allowed(end_row_ctrl, eos_id), (
            "grammar EOS was not allowed at the accepting state — baseline "
            "termination for eos==grammar-eos families regressed"
        )


def test_model_stop_token_ids_unions_all_surfaces():
    """The helper unions every eos surface the scheduler halts on (0.10.16
    P1-①), so the processor re-admits EXACTLY the ids that end generation."""
    from vllm_mlx.api.tool_grammar import model_stop_token_ids

    class _Tok:
        _eos_token_ids = {1, 106}
        eos_token_id = [1, 50]
        eos_token_ids = (106,)
        _rapid_extra_eos_token_ids = (50, 999)

    assert model_stop_token_ids(_Tok()) == (1, 50, 106, 999)
    assert model_stop_token_ids(None) == ()

    class _Singular:
        eos_token_id = 2

    assert model_stop_token_ids(_Singular()) == (2,)

    class _Empty:
        pass

    assert model_stop_token_ids(_Empty()) == ()


@_requires_llguidance
def test_named_grammar_constrains_to_single_tool(tok, lltok):
    # A NAMED choice narrows to one tool. Building the grammar over a
    # 1-element list with "required" forces exactly that tool; a call naming a
    # DIFFERENT (also-provided) tool is rejected.
    from vllm_mlx.api.tool_grammar import (
        GrammarLogitsProcessor,
        build_tool_grammar,
    )
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    parser = HermesToolParser(tokenizer=tok)
    only_time = [t for t in TOOLS if t["name"] == "get_time"]
    grammar = build_tool_grammar(only_time, "required", parser)
    # llguidance is present (test is @_requires_llguidance) and the hermes
    # structure_info is available -> None means a builder regression, fail hard.
    assert grammar is not None, (
        "build_tool_grammar returned None for a single-tool named narrow — "
        "the grammar builder regressed"
    )

    prompt_ids = tok.encode("time?", add_special_tokens=False)

    # get_time is allowed.
    proc = GrammarLogitsProcessor(lltok, grammar, tokenizer=tok)
    blocked, _ = _feed_cumulative(
        proc,
        lltok,
        tok,
        prompt_ids,
        '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call>',
    )
    assert not blocked, "the named tool get_time was wrongly blocked"

    # get_weather (not the named target) is rejected.
    proc2 = GrammarLogitsProcessor(lltok, grammar, tokenizer=tok)
    blocked2, _ = _feed_cumulative(
        proc2, lltok, tok, prompt_ids, '<tool_call>\n{"name": "get_weather'
    )
    assert blocked2, "a non-named tool was NOT blocked under a named choice"


@_requires_llguidance
def test_tool_named_required_collision_end_to_end(tok, lltok):
    # Deferred codex #2, end-to-end: a tool literally named "required" is
    # constrainable as a single named tool (via the object form) WITHOUT the
    # grammar collapsing to the "required" enum over all tools. We build the
    # named grammar the way chat.py routing does: pre-narrow to the target +
    # "required" quantifier.
    from vllm_mlx.api.tool_grammar import (
        GrammarLogitsProcessor,
        build_tool_grammar,
    )
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    collide_tools = [
        {
            "name": "required",  # a tool whose name collides with the enum
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
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
    parser = HermesToolParser(tokenizer=tok)
    # Named choice on the "required"-named tool: narrow then "required" quant.
    narrowed = [t for t in collide_tools if t["name"] == "required"]
    grammar = build_tool_grammar(narrowed, "required", parser)
    # llguidance present (test is @_requires_llguidance) + hermes structure_info
    # available -> None is a REGRESSION that would let the exact collision this
    # test guards ship green while production falls back to unconstrained
    # generation (codex #558-PR3). Assert, don't skip.
    assert grammar is not None, (
        "build_tool_grammar returned None for the 'required'-named collision "
        "case — the grammar builder regressed"
    )

    prompt_ids = tok.encode("go", add_special_tokens=False)

    # The "required"-named tool is allowed.
    proc = GrammarLogitsProcessor(lltok, grammar, tokenizer=tok)
    blocked, _ = _feed_cumulative(
        proc,
        lltok,
        tok,
        prompt_ids,
        '<tool_call>\n{"name": "required", "arguments": {"x": "y"}}\n</tool_call>',
    )
    assert not blocked, "the tool literally named 'required' was wrongly blocked"

    # The OTHER provided tool (get_time) is rejected — the named grammar is
    # single-tool, proving it did not collapse to a force-any-tool enum.
    proc2 = GrammarLogitsProcessor(lltok, grammar, tokenizer=tok)
    blocked2, _ = _feed_cumulative(
        proc2, lltok, tok, prompt_ids, '<tool_call>\n{"name": "get_time'
    )
    assert blocked2, (
        "named choice on the 'required'-named tool leaked another tool — "
        "the string collision was not contained"
    )


@_requires_llguidance
def test_processor_reset_clears_baseline(tok, hermes_grammar, lltok):
    # reset() must clear the prompt baseline and committed counter so the
    # processor can be reused for a fresh sequence.
    import mlx.core as mx

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    prompt_ids = tok.encode("hi", add_special_tokens=False)
    proc(mx.array(prompt_ids), mx.zeros((1, lltok.vocab_size)))
    assert proc._prompt_len == len(prompt_ids)
    proc.reset()
    assert proc._prompt_len is None
    assert proc._committed == 0


@_requires_llguidance
def test_processor_preserves_padded_vocab_shape(tok, hermes_grammar, lltok):
    # codex #558-PR3: when the model head is WIDER than the tokenizer vocab
    # (padded embedding), the processor must return logits of the SAME final
    # dimension — mlx-lm concatenates every row's processed logits, so a
    # narrower return breaks batched decode. Verify shape is preserved and the
    # padded tail is -inf (never sampled).
    import mlx.core as mx
    import numpy as np

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    prompt_ids = tok.encode("hi", add_special_tokens=False)
    pad = 128
    wide_vocab = lltok.vocab_size + pad
    out = proc(mx.array(prompt_ids), mx.zeros((1, wide_vocab)))
    assert out.shape[-1] == wide_vocab, "padded-vocab shape not preserved"
    tail = np.array(out[0, lltok.vocab_size :])
    assert np.all(tail == -np.inf), "padded tail must be -inf (never sampleable)"


@_requires_llguidance
def test_processor_equal_vocab_shape_unchanged(tok, hermes_grammar, lltok):
    # When model vocab == tokenizer vocab, the mask is applied in place and the
    # shape is unchanged (no concat path).
    import mlx.core as mx

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    prompt_ids = tok.encode("hi", add_special_tokens=False)
    out = proc(mx.array(prompt_ids), mx.zeros((1, lltok.vocab_size)))
    assert out.shape[-1] == lltok.vocab_size


@_requires_llguidance
def test_processor_narrower_model_head_than_tokenizer(tok, hermes_grammar, lltok):
    # codex #558-PR3 blocking: the INVERSE of the padded case — the TOKENIZER
    # carries added tokens beyond a NARROWER model output head
    # (``model_vocab < tokenizer_vocab``). The full-vocab bitmask must not be
    # applied whole to the narrower logits (would over-cover); the processor
    # slices the packed bitmask to the model-width word count and masks cleanly,
    # returning the SAME narrow shape. Assert MASKING BEHAVIOR, not just shape:
    # the grammar must still forbid in-range tokens (some become ``-inf``) while
    # keeping at least one allowed — a shape-only check would stay green even if
    # the branch returned UNMASKED logits and silently disabled enforcement.
    import mlx.core as mx
    import numpy as np

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    proc = GrammarLogitsProcessor(lltok, hermes_grammar, tokenizer=tok)
    prompt_ids = tok.encode("hi", add_special_tokens=False)
    # Trim only a few ids off the top (not on a word boundary, to exercise the
    # ceil-division word count) so the constrained opener is still in range.
    narrow_vocab = lltok.vocab_size - 5
    out = proc(mx.array(prompt_ids), mx.zeros((1, narrow_vocab)))
    assert out.shape[-1] == narrow_vocab, "narrow-head shape not preserved"

    row = np.array(out[0])
    n_blocked = int(np.count_nonzero(~np.isfinite(row)))
    n_allowed = int(np.count_nonzero(np.isfinite(row)))
    # The mask MUST have been applied to the narrow logits: the input is
    # all-zeros (finite), so any ``-inf`` entry can only come from the grammar
    # mask. ``n_blocked > 0`` therefore proves enforcement is live on this path
    # (an unmasked passthrough would leave every entry finite). ``n_allowed > 0``
    # confirms the grammar still permits progress (not a broken all-mask). At
    # grammar position 0 the hermes lazy free-prefix allows most tokens, so we
    # do NOT assert a majority are blocked — only that masking demonstrably
    # occurred and left a valid, non-empty allowed set.
    assert n_blocked > 0, "narrow-head path returned UNMASKED logits (no enforcement)"
    assert n_allowed > 0, "narrow-head path masked EVERYTHING (grammar broken)"


@_requires_llguidance
def test_get_lltokenizer_caches(tok):
    # codex #558-PR3 (nit): the ~1s LLTokenizer build must be cached per
    # tokenizer, not rebuilt on every request.
    from vllm_mlx.api.tool_grammar import get_lltokenizer

    a = get_lltokenizer(tok)
    b = get_lltokenizer(tok)
    assert a is not None
    assert a is b, "get_lltokenizer must return the same cached instance"


@_requires_llguidance
def test_get_lltokenizer_transient_failure_is_retried(monkeypatch):
    # codex #558-PR3 nit: a TRANSIENT build failure must NOT permanently disable
    # grammar enforcement — it's retried up to the budget, and a subsequent
    # success is cached. A distinct dummy tokenizer avoids poisoning the shared
    # module-scoped ``tok`` fixture's cache.
    import vllm_mlx.api.tool_grammar as tg_mod

    class _DummyTok:
        pass

    dummy = _DummyTok()
    sentinel = object()
    calls = {"n": 0}

    def _flaky_build(_tokenizer):
        calls["n"] += 1
        # Fail the first two attempts (transient), succeed on the third.
        if calls["n"] < 3:
            return None
        return sentinel

    monkeypatch.setattr(tg_mod, "build_lltokenizer", _flaky_build)

    # First two calls hit transient failures and must NOT seal the cache.
    assert tg_mod.get_lltokenizer(dummy) is None
    assert tg_mod.get_lltokenizer(dummy) is None
    # Third call succeeds and is cached.
    assert tg_mod.get_lltokenizer(dummy) is sentinel
    # Cached: no further build calls.
    assert tg_mod.get_lltokenizer(dummy) is sentinel
    assert calls["n"] == 3, "success must be cached; no rebuild after it"


@_requires_llguidance
def test_get_lltokenizer_seals_after_budget(monkeypatch):
    # codex #558-PR3 nit: a PERSISTENT failure is eventually sealed as
    # unavailable (bounded retries) so we don't rebuild-and-fail forever.
    import vllm_mlx.api.tool_grammar as tg_mod

    class _DummyTok:
        pass

    dummy = _DummyTok()
    calls = {"n": 0}

    def _always_fail(_tokenizer):
        calls["n"] += 1
        return None

    monkeypatch.setattr(tg_mod, "build_lltokenizer", _always_fail)

    budget = tg_mod._LLTOKENIZER_MAX_BUILD_ATTEMPTS
    # Each call up to the budget retries the build.
    for _ in range(budget):
        assert tg_mod.get_lltokenizer(dummy) is None
    assert calls["n"] == budget, "must retry up to the budget"
    # Now sealed: further calls short-circuit without rebuilding.
    assert tg_mod.get_lltokenizer(dummy) is None
    assert calls["n"] == budget, "sealed after budget — no further rebuilds"


@_requires_llguidance
def test_missing_parameters_flattens_to_closed_schema_in_production_path(
    tok, monkeypatch
):
    # codex #558-PR3: drive the REAL ``_maybe_build_tool_grammar_processor`` and
    # capture the schema it flattens a no-``parameters`` tool into. Absent /
    # null ``parameters`` must become a CLOSED empty-object schema; an explicit
    # ``{}`` (allow-any) and an explicit schema must pass through verbatim.
    from vllm_mlx.api import tool_grammar as tg_mod
    from vllm_mlx.routes import chat as chat_mod

    captured = {}

    def _spy_build(
        flat_tools, mode, parser, *, single_call=False, reasoning_sentinels=()
    ):
        captured["flat_tools"] = flat_tools
        captured["mode"] = mode
        captured["single_call"] = single_call
        captured["reasoning_sentinels"] = reasoning_sentinels
        return None  # short-circuit: we only need the flattened tools

    # ``build_tool_grammar`` is imported inside the route function, so patch it
    # at its defining module (the name the local import resolves to).
    monkeypatch.setattr(tg_mod, "build_tool_grammar", _spy_build)

    Fn = _FunctionTool
    engine = _EngineStub(tok)
    cfg = _CfgStub("hermes")

    def _run(tools, tool_choice):
        captured.clear()
        request = _RequestStub(tools=tools, tool_choice=tool_choice)
        # Returns None (build stub returns None) but populates ``captured``.
        chat_mod._maybe_build_tool_grammar_processor(engine, cfg, request)
        return {t["name"]: t["parameters"] for t in captured.get("flat_tools", [])}

    # Absent parameters -> closed empty-object schema.
    got = _run([Fn("noargs")], "required")
    assert got["noargs"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    # Explicit null -> closed.
    assert (
        _run([Fn("n", parameters=None)], "required")["n"]["additionalProperties"]
        is False
    )
    # Explicit {} -> preserved verbatim (allow-any).
    assert _run([Fn("n", parameters={})], "required")["n"] == {}
    # Explicit schema -> preserved verbatim.
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert _run([Fn("n", parameters=schema)], "required")["n"] == schema


@_requires_llguidance
def test_parallel_tool_calls_false_threads_single_call(tok, monkeypatch):
    # codex #558-PR3 blocking: ``parallel_tool_calls=False`` must build an
    # exactly-one-call grammar (``single_call=True`` to the builder); ``True`` /
    # unset keep the one-or-more ``required`` grammar.
    from vllm_mlx.api import tool_grammar as tg_mod
    from vllm_mlx.routes import chat as chat_mod

    captured = {}

    def _spy_build(
        flat_tools, mode, parser, *, single_call=False, reasoning_sentinels=()
    ):
        captured["single_call"] = single_call
        return None

    monkeypatch.setattr(tg_mod, "build_tool_grammar", _spy_build)

    engine = _EngineStub(tok)
    cfg = _CfgStub("hermes")

    class _Req:
        def __init__(self, ptc):
            self.tools = [_FunctionTool("get_time")]
            self.tool_choice = "required"
            self.parallel_tool_calls = ptc

    def _single_call_for(ptc):
        captured.clear()
        chat_mod._maybe_build_tool_grammar_processor(engine, cfg, _Req(ptc))
        return captured.get("single_call")

    assert _single_call_for(False) is True, "parallel_tool_calls=False -> single"
    assert _single_call_for(True) is False, "parallel_tool_calls=True -> one-or-more"
    assert _single_call_for(None) is False, "unset -> one-or-more (OpenAI default)"


@_requires_llguidance
def test_route_threads_reasoning_sentinels_from_configured_parser(tok, monkeypatch):
    """PR-4 wiring: when a reasoning parser is configured, the chat route derives
    its single-special-token reasoning markers and threads them into
    ``build_tool_grammar`` so the grammar's free prefix tolerates ``<think>``.
    A cfg WITHOUT a reasoning parser threads ``()`` (non-reasoning grammar)."""
    from vllm_mlx.api import tool_grammar as tg_mod
    from vllm_mlx.routes import chat as chat_mod

    # Guard: the assertion is only meaningful if this tokenizer carries
    # <think>/</think> as single special tokens (it does on pinned Qwen3.5).
    if not tg_mod.are_single_special_tokens(tok, ("<think>", "</think>")):
        pytest.skip("fixture tokenizer lacks single-token <think>/</think>")

    captured = {}

    def _spy_build(
        flat_tools, mode, parser, *, single_call=False, reasoning_sentinels=()
    ):
        captured["reasoning_sentinels"] = reasoning_sentinels
        return None

    monkeypatch.setattr(tg_mod, "build_tool_grammar", _spy_build)
    engine = _EngineStub(tok)

    def _sentinels_for(reasoning_parser_name):
        captured.clear()
        cfg = _CfgStub("hermes")
        if reasoning_parser_name is not None:
            cfg.reasoning_parser_name = reasoning_parser_name
        request = _RequestStub([_FunctionTool("get_time")], "required")
        chat_mod._maybe_build_tool_grammar_processor(engine, cfg, request)
        return captured.get("reasoning_sentinels")

    # Reasoning parser configured -> <think>/</think> threaded into the builder.
    assert _sentinels_for("qwen3") == ("<think>", "</think>")
    # No reasoning parser -> empty (non-reasoning grammar, no regression).
    assert _sentinels_for(None) == ()


@_requires_llguidance
def test_broken_grammar_yields_none_so_fallback_stays_active(tok, monkeypatch):
    # codex #558-PR3: a matcher that fails to compile is ``_broken`` and masks
    # NOTHING. ``_maybe_build_tool_grammar_processor`` must return ``None`` for
    # a broken processor so the route keeps the forced-prefix / free-form
    # fallback active — never set an inert ``grammar_logits_processor`` that
    # both disables the fallback AND leaves output unconstrained.
    from vllm_mlx.api import tool_grammar as tg_mod
    from vllm_mlx.routes import chat as chat_mod

    # Real builder path, but force the processor to report itself broken.
    class _BrokenProc:
        def __init__(self, *a, **k):
            pass

        def is_broken(self):
            return True

    monkeypatch.setattr(tg_mod, "build_tool_grammar", lambda *a, **k: "GRAMMAR")
    monkeypatch.setattr(tg_mod, "GrammarLogitsProcessor", _BrokenProc)

    engine = _EngineStub(tok)
    cfg = _CfgStub("hermes")
    request = _RequestStub(tools=[_FunctionTool("get_time")], tool_choice="required")

    result = chat_mod._maybe_build_tool_grammar_processor(engine, cfg, request)
    assert result is None, (
        "a broken (non-compiling) grammar must yield None so the established "
        "fallback stays active — an inert processor would disable the fallback "
        "AND leave output unconstrained"
    )


def test_broken_processor_reports_is_broken_and_masks_nothing():
    # Unit-level guarantee behind the fallback: a GrammarLogitsProcessor built
    # on an invalid grammar sets ``is_broken()`` and returns logits unchanged.
    import importlib.util

    if importlib.util.find_spec("llguidance") is None:
        pytest.skip("llguidance ([guided] extra) not installed")

    import mlx.core as mx

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    class _FakeMatcher:
        def __init__(self, *a, **k):
            pass

        def get_error(self):
            return "unterminated grammar: deliberately invalid"

    class _FakeLLTok:
        vocab_size = 8

    import vllm_mlx.api.tool_grammar as tg

    # Patch the matcher + bitmask allocation so we don't need real llguidance
    # internals for this pure broken-path assertion.
    orig_matcher = tg.LLMatcher
    orig_alloc = tg.allocate_token_bitmask
    tg.LLMatcher = _FakeMatcher
    tg.allocate_token_bitmask = lambda n, v: None
    try:
        proc = GrammarLogitsProcessor(_FakeLLTok(), "bad-grammar")
        assert proc.is_broken() is True
        logits = mx.zeros((1, 8))
        out = proc(mx.array([1, 2, 3]), logits)
        # Broken -> logits returned unchanged (no mask), same object/shape.
        assert out.shape == logits.shape
        import numpy as np

        assert np.array_equal(np.array(out), np.array(logits))
    finally:
        tg.LLMatcher = orig_matcher
        tg.allocate_token_bitmask = orig_alloc


def test_rejected_committed_token_drops_constraint_and_stops_masking():
    # codex #558-PR3 nit: a matcher that REJECTS an already-sampled+committed
    # token is desynced from the real output stream. The processor DROPS the
    # constraint (a controlled FAIL-OPEN fallback) — latch ``_aborted``, stop
    # consuming into the invalid matcher, and stop imposing a (garbage) mask,
    # returning logits UNCHANGED so downstream free-form parsing owns the
    # request. This is deliberately fail-open, NOT fail-closed: masking from a
    # desynced matcher would be strictly worse, and this whole module is
    # best-effort (missing extra / bad grammar / unsupported tokenizer all
    # degrade to free-form, never an error). Uses fakes for EVERY llguidance
    # primitive the processor touches (LLMatcher + the three bitmask fns), so it
    # runs UNCONDITIONALLY in base CI — no ``llguidance`` extra required (codex
    # #558-PR3 nit).
    import mlx.core as mx
    import numpy as np

    from vllm_mlx.api.tool_grammar import GrammarLogitsProcessor

    class _RejectingMatcher:
        """Accepts the first committed token, then rejects everything after."""

        def __init__(self, *a, **k):
            self.n_consumed = 0
            self.n_fills = 0

        def get_error(self):
            return None  # compiled fine

        def deep_copy(self):
            # get_request_matcher caches a never-consumed template and hands each
            # request a deep_copy of it; a fresh instance (n_consumed=0) is the
            # faithful clone of the initial-state template.
            return _RejectingMatcher()

        def consume_token(self, tok_id):
            self.n_consumed += 1
            # Accept the first generated token, reject the second (simulating a
            # matcher that desyncs from an already-committed token).
            return self.n_consumed <= 1

        def is_stopped(self):
            return False

        def reset(self):
            self.n_consumed = 0

    class _FakeLLTok:
        vocab_size = 8

    import vllm_mlx.api.tool_grammar as tg

    orig_matcher = tg.LLMatcher
    orig_alloc = tg.allocate_token_bitmask
    orig_fill = tg.fill_next_token_bitmask
    orig_apply = tg.apply_token_bitmask

    fills = {"n": 0}

    def _spy_fill(matcher, bitmask, row):
        fills["n"] += 1

    def _mask_all(logits, bitmask):
        # A "real" mask would zero out most tokens; return a sentinel that is
        # clearly different from the unchanged logits so we can prove masking
        # did NOT run after the abort.
        return mx.full(logits.shape, -1.0, dtype=logits.dtype)

    tg.LLMatcher = _RejectingMatcher
    tg.allocate_token_bitmask = lambda n, v: None
    tg.fill_next_token_bitmask = _spy_fill
    tg.apply_token_bitmask = _mask_all
    try:
        proc = GrammarLogitsProcessor(_FakeLLTok(), "ok-grammar", tokenizer=None)
        assert proc.is_broken() is False

        # Step 1: baseline the prompt (1 token), then feed the first generated
        # token — accepted, matcher masks (fill runs, mask applied).
        proc(mx.array([1]), mx.zeros((1, 8)))  # prompt baseline
        out1 = proc(mx.array([1, 2]), mx.zeros((1, 8)))
        assert proc._aborted is False, "first committed token was wrongly rejected"
        assert np.array_equal(np.array(out1), np.full((1, 8), -1.0)), (
            "an accepted token must still be masked (constraint active)"
        )
        fills_before_abort = fills["n"]

        # Step 2: feed the second generated token — the matcher rejects it, so
        # the processor must latch ``_aborted`` and return logits UNCHANGED.
        logits2 = mx.zeros((1, 8))
        out2 = proc(mx.array([1, 2, 3]), logits2)
        assert proc._aborted is True, "rejected committed token must latch _aborted"
        assert np.array_equal(np.array(out2), np.array(logits2)), (
            "after abort, the mask must NOT be applied (logits unchanged)"
        )
        # No new fill happened on the aborting step (short-circuit before mask).
        assert fills["n"] == fills_before_abort, (
            "the mask branch must be skipped once aborted"
        )

        # Step 3: any subsequent call also stays dropped (logits unchanged).
        logits3 = mx.zeros((1, 8))
        out3 = proc(mx.array([1, 2, 3, 4]), logits3)
        assert np.array_equal(np.array(out3), np.array(logits3))

        # reset() clears the abort latch so the processor can be reused.
        proc.reset()
        assert proc._aborted is False
    finally:
        tg.LLMatcher = orig_matcher
        tg.allocate_token_bitmask = orig_alloc
        tg.fill_next_token_bitmask = orig_fill
        tg.apply_token_bitmask = orig_apply


def test_forced_prefix_block_is_gated_on_grammar_absence():
    # codex #558-PR3: the forced assistant prefix and the grammar are mutually
    # exclusive — combining them baselines the injected prefix away and
    # bypasses schema enforcement. The route builds ``_glp`` FIRST and gates
    # the ``_forced_prefix`` block on ``_glp is None``. Assert that structural
    # guarantee in the route source so a future edit that re-enables both
    # together fails CI (a pure behavioral test would need to drive the whole
    # ~600-line ``_create_chat_completion_impl``).
    import inspect

    from vllm_mlx.routes import chat as chat_mod

    src = inspect.getsource(chat_mod._create_chat_completion_impl)
    glp_pos = src.find("_glp = await _offload_tool_grammar_build(")
    prefix_gate = src.find("if _glp is None and request.tools")
    assert glp_pos != -1, "grammar processor build call not found in route"
    assert prefix_gate != -1, (
        "forced-prefix block must be gated on '_glp is None' — the two are "
        "mutually exclusive (codex #558-PR3)"
    )
    assert glp_pos < prefix_gate, (
        "the grammar processor must be built BEFORE the forced-prefix gate"
    )
