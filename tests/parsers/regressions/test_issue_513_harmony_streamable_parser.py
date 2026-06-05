# SPDX-License-Identifier: Apache-2.0
"""Closure regressions for issues #444 / #455 / #468 / #480 via the
openai-harmony bypass (PR #515 landing #513).

PR #514 partially fixed these at the parser layer (prefix-hold +
count-based ``<|call|>`` detection) but their END-TO-END production
fix required the router to understand harmony's tool-call protocol:
``commentary`` + ``to=functions.<name>`` + optional
``<|constrain|>json`` + body + ``<|call|>``. PR #514 confirmed
``commentary`` is multi-token (``comment``+``ary``) on production
gpt-oss-20b, which the custom token-ID-match state machine could
never identify.

PR #515 lands the SOTA fix: delegate harmony state tracking to
``openai-harmony.StreamableParser`` (same library vLLM and SGLang
delegate to). The new ``HarmonyStreamingRouter`` shim exposes the
existing ``OutputRouter`` surface so the engine streaming path is
unchanged; only the harmony format gets the new backend.

This file pins the closure of #444 / #455 / #468 / #480 by replaying
production-shape token sequences (encoded via the real harmony
encoding, NOT the synthetic test vocab used by the partial-closure
regressions in the sibling files) and asserting NO marker leak +
correct channel routing.
"""

from __future__ import annotations

import pytest

# Skip the whole module when the optional ``openai-harmony`` dep is
# missing — without it, the legacy router runs and leaks (#444 etc.
# remain xfail-strict in the sibling regression files).
openai_harmony = pytest.importorskip("openai_harmony")

from openai_harmony import (  # noqa: E402
    HarmonyEncodingName,
    load_harmony_encoding,
)

from vllm_mlx.output_router import Channel, TokenMap  # noqa: E402
from vllm_mlx.output_router_harmony import HarmonyStreamingRouter  # noqa: E402

from .._harmony_markers import HARMONY_LEAK_MARKERS  # noqa: E402


@pytest.fixture(scope="module")
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class _HarmonyDecodeAdapter:
    """Minimal tokenizer surface for ``HarmonyStreamingRouter``.

    The router uses ``tokenizer.decode`` only for the legacy router
    fallback paths (not on the openai-harmony path), so a thin
    adapter over the harmony encoding suffices.
    """

    def __init__(self, enc):
        self._enc = enc

    def decode(self, ids):
        return self._enc.decode(ids)

    def get_vocab(self):
        return {}


@pytest.fixture
def router(encoding):
    tm = TokenMap(format_tag="harmony")
    return HarmonyStreamingRouter(tm, _HarmonyDecodeAdapter(encoding))


def _encode(encoding, text: str) -> list[int]:
    """Wrap encode with allowed_special=all so structural markers
    (``<|channel|>`` etc.) round-trip as single token IDs the way
    a real gpt-oss-20b would emit them.
    """
    return encoding.encode(text, allowed_special="all")


# Production-shape emit sequences. ``StreamableParser`` is initialized
# with ``role=ASSISTANT`` so the FIRST message starts directly with
# ``<|channel|>`` (the pre-set role consumes the header). Subsequent
# messages within the same assistant turn use the explicit
# ``<|start|>assistant<|channel|>...`` form.


# ----- #444 / #480 — Harmony streaming tool call leak ------------------


def test_issue_444_480_commentary_tool_call_no_marker_leak(router, encoding):
    """#444 / #480: streaming a harmony tool call MUST NOT leak
    structural markers or channel labels into ``content`` / ``reasoning``,
    and the tool call MUST surface in ``tool_calls`` with the recipient
    + body intact.

    Pre-PR #515: the custom router state machine recognized ``analysis``
    / ``final`` channel-type words by single-token ID match. Production
    ``commentary`` is two tokens (``comment``+``ary``); the router fell
    through to CONTENT and leaked the entire markered tool-call sequence
    as content text.
    """
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}<|call|>'
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["content"] is None, (
        f"#444/#480: tool-call commentary stream must NOT leak into "
        f"content; got content={result['content']!r}"
    )
    assert result["reasoning"] is None, (
        f"#444/#480: no analysis channel in this sequence — reasoning "
        f"must stay empty; got reasoning={result['reasoning']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#444/#480: tool call must surface; got tool_calls={result['tool_calls']!r}"
    )
    tc_text = result["tool_calls"][0]
    # Tool call event carries the reconstructed wire text the
    # downstream HarmonyToolParser consumes — must include recipient
    # and body.
    assert "functions.get_weather" in tc_text, (
        f"#444/#480: tool call must carry recipient; got {tc_text!r}"
    )
    assert '{"city":"NYC"}' in tc_text, (
        f"#444/#480: tool call must carry body; got {tc_text!r}"
    )


# ----- #455 — Harmony commentary channel routing -----------------------


def test_issue_455_analysis_then_commentary_separates_channels(router, encoding):
    """#455: analysis (reasoning) + commentary (tool call) in one
    assistant turn must route to reasoning and tool_calls respectively,
    with no leak between them and no markers in either.

    Pre-PR #515: the analysis body landed in reasoning correctly, but
    the commentary tool call leaked into content (router couldn't
    transition state on multi-token ``commentary``).
    """
    text = (
        "<|channel|>analysis<|message|>"
        "I'll fetch the weather.<|end|>"
        "<|start|>assistant<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"Paris"}<|call|>'
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["content"] is None, (
        f"#455: content must stay empty; got {result['content']!r}"
    )
    assert result["reasoning"] == "I'll fetch the weather.", (
        f"#455: reasoning must carry analysis body; got {result['reasoning']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#455: tool call must surface; got tool_calls={result['tool_calls']!r}"
    )

    # Universal leak check — no harmony marker may appear in user-
    # facing channels regardless of how the route runs.
    for ch_name in ("content", "reasoning"):
        val = result.get(ch_name) or ""
        for marker in HARMONY_LEAK_MARKERS:
            assert marker not in val, (
                f"#455: marker {marker!r} leaked into {ch_name}; got {val!r}"
            )


# ----- #468 — tool_choice="required" + commentary compound -----------


def test_issue_468_compound_analysis_commentary_final_separates(router, encoding):
    """#468: assistant turn with analysis → tool call (commentary) →
    final response. All three channels must route independently with
    no cross-leak.
    """
    text = (
        "<|channel|>analysis<|message|>"
        "Need to compute the sum.<|end|>"
        "<|start|>assistant<|channel|>commentary "
        "to=functions.add <|constrain|>json<|message|>"
        '{"a":1,"b":2}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>"
        "The answer is 3.<|return|>"
    )
    tokens = _encode(encoding, text)
    result = router.feed_sequence(tokens)

    assert result["reasoning"] == "Need to compute the sum.", (
        f"#468: reasoning must carry analysis; got {result['reasoning']!r}"
    )
    assert result["content"] == "The answer is 3.", (
        f"#468: content must carry final body; got {result['content']!r}"
    )
    assert result["tool_calls"] is not None and len(result["tool_calls"]) == 1, (
        f"#468: one tool call must surface; got {result['tool_calls']!r}"
    )

    # Reconstructed tool call must include name + args.
    tc_text = result["tool_calls"][0]
    assert "functions.add" in tc_text
    assert '{"a":1,"b":2}' in tc_text

    # Universal leak check.
    for ch_name in ("content", "reasoning"):
        val = result.get(ch_name) or ""
        for marker in HARMONY_LEAK_MARKERS:
            assert marker not in val, (
                f"#468: marker {marker!r} leaked into {ch_name}; got {val!r}"
            )


# ----- General invariants ----------------------------------------------


def test_per_token_streaming_routes_one_event_per_body_token(router, encoding):
    """Pin per-token streaming behavior — analysis/final body tokens
    produce ONE routed event each (matching the engine streaming
    contract). Commentary body tokens are suppressed during streaming
    (the tool call is emitted as a single aggregated event on
    ``<|call|>``), matching the existing Channel.TOOL_CALL contract.
    """
    text = "<|channel|>final<|message|>Hi there.<|return|>"
    tokens = _encode(encoding, text)
    # Reset router for explicit per-token feed.
    router.reset()
    events_per_channel: dict[Channel, list[str]] = {
        Channel.CONTENT: [],
        Channel.REASONING: [],
        Channel.TOOL_CALL: [],
    }
    for tid in tokens:
        ev = router.feed(tid)
        if ev is None:
            continue
        events_per_channel[ev.channel].append(ev.text)

    assert events_per_channel[Channel.TOOL_CALL] == []
    assert events_per_channel[Channel.REASONING] == []
    # Joined content matches the body, and at least one event per body
    # token surfaced.
    assert "".join(events_per_channel[Channel.CONTENT]) == "Hi there."
    assert len(events_per_channel[Channel.CONTENT]) >= 2, (
        f"per-token body deltas expected; got {events_per_channel[Channel.CONTENT]!r}"
    )


def test_feed_sequence_preserves_leading_and_trailing_whitespace(router, encoding):
    """Codex round-1 BLOCKING (PR #515): ``feed_sequence`` must NOT
    ``.strip()`` accumulated content / reasoning — harmony bodies can
    legitimately begin or end with whitespace (markdown blocks, code
    fences, formatted output) and stripping silently mutates the
    response. Only the exact empty string maps to ``None``.
    """
    # Final-channel body starts with a newline and ends with two
    # spaces — both must survive.
    text = "<|channel|>final<|message|>\n```py\nprint('hi')\n```  <|return|>"
    tokens = _encode(encoding, text)
    router.reset()
    result = router.feed_sequence(tokens)

    assert result["content"] == "\n```py\nprint('hi')\n```  ", (
        f"feed_sequence must preserve surrounding whitespace; got {result['content']!r}"
    )


def test_reconstruct_tool_call_rejects_malformed_recipient(router):
    """Codex round-1 BLOCKING (PR #515): the recipient string is
    interpolated into the reconstructed ``to=<recipient>`` wire text
    consumed by ``HarmonyToolParser``. A recipient with whitespace,
    marker-like characters, or unexpected shape would break the
    downstream parser. The router must reject it loudly.
    """

    class _FakeMessage:
        def __init__(self, recipient):
            self.recipient = recipient
            self.content = []
            self.content_type = None

    # ``feed()`` only calls ``_reconstruct_tool_call_text`` when
    # ``closed.recipient`` is truthy, so empty / None recipients are
    # filtered upstream — only test the cases where reconstruction
    # actually runs (recipient is a non-empty string of unexpected
    # shape).
    bad_recipients = (
        "functions.bad name",  # whitespace
        "functions.<|call|>",  # marker
        "not_functions.x",  # wrong namespace
        "functions.",  # empty name
    )
    for bad in bad_recipients:
        with pytest.raises(ValueError, match="malformed recipient"):
            router._reconstruct_tool_call_text(_FakeMessage(bad))

    # Sanity: a well-formed recipient must NOT raise.
    good = _FakeMessage("functions.get_weather")
    out = router._reconstruct_tool_call_text(good)
    assert "to=functions.get_weather" in out


def test_reconstruct_tool_call_abstains_on_body_carrying_harmony_sentinel(router):
    """Codex round-7 BLOCKING (don't emit corrupt structured event) +
    round-9 BLOCKING (don't raise on legitimate JSON containing
    ``<|...|>`` substrings): when the body or content_type carries a
    harmony structural sentinel that would corrupt the downstream
    HarmonyToolParser regex parse, ``_reconstruct_tool_call_text``
    must ABSTAIN — return ``None`` — instead of raising.

    The caller (``feed()``) then emits a CONTENT event with the body
    bytes (codex round-10 BLOCKING — streaming visibility) so the
    user still sees the tool call's intent verbatim, matching baseline
    behavior where the legacy router leaked commentary body into
    CONTENT. The engine non-stream path is unaffected: it relies on
    ``fallback_text``, which still carries the model's raw emit.
    Matches the recipient-validation pattern's intent (don't emit a
    bogus structured event) without the round-9 cost (dropping
    legitimate OpenAI tool calls whose JSON values happen to contain
    those characters) and without the round-10 cost (silent streaming
    drop).
    """

    class _Content:
        def __init__(self, text):
            self.text = text

    class _FakeMessage:
        def __init__(self, body, recipient="functions.get_weather", ctype=None):
            self.recipient = recipient
            self.content = [_Content(body)]
            self.content_type = ctype

    abstain_bodies = (
        '{"text":"use <|call|>"}',
        '{"text":"<|message|>injected"}',
        '{"x":"<|channel|>commentary"}',
        '{"y":"<|end|>"}',
        '{"z":"<|return|>"}',
        '{"a":"<|start|>"}',
        '{"b":"<|constrain|>json"}',
    )
    for bad in abstain_bodies:
        out = router._reconstruct_tool_call_text(_FakeMessage(bad))
        assert out is None, f"body carrying sentinel must abstain (None); got {out!r}"

    # Sanity: clean JSON must reconstruct normally.
    clean = _FakeMessage('{"city":"NYC"}')
    out = router._reconstruct_tool_call_text(clean)
    assert out is not None
    assert '{"city":"NYC"}' in out
    assert "<|message|>" in out  # legitimate framing must remain
    assert out.endswith("<|call|>")  # legitimate trailing sentinel

    # A content_type that smuggles a sentinel past the
    # ``<|constrain|>`` prefix must also abstain.
    out = router._reconstruct_tool_call_text(
        _FakeMessage(
            '{"city":"NYC"}',
            ctype="<|constrain|>json<|call|>injected",
        )
    )
    assert out is None, f"content_type smuggling a sentinel must abstain; got {out!r}"

    # A normal ``<|constrain|>json`` content_type must NOT abstain.
    ok = _FakeMessage('{"city":"NYC"}', ctype="<|constrain|>json")
    out = router._reconstruct_tool_call_text(ok)
    assert out is not None
    assert "<|constrain|>json<|message|>" in out

    # Codex round-11 NIT: a bare enum like ``"json"`` (missing the
    # ``<|constrain|>`` prefix) would reconstruct non-canonical wire
    # text the downstream parser can't recognise — must abstain.
    bare_ctype = _FakeMessage('{"city":"NYC"}', ctype="json")
    assert router._reconstruct_tool_call_text(bare_ctype) is None, (
        "bare content_type missing <|constrain|> prefix must abstain"
    )
    # And a content_type with disallowed characters must also abstain.
    bad_chars_ctype = _FakeMessage('{"city":"NYC"}', ctype="<|constrain|>has space")
    assert router._reconstruct_tool_call_text(bad_chars_ctype) is None, (
        "content_type with non-safe characters must abstain"
    )


def test_sentinel_body_emits_content_event_for_streaming_visibility(
    router, encoding, monkeypatch
):
    """Codex round-10 BLOCKING (PR #515): when ``_reconstruct_tool_call_text``
    abstains for a sentinel-laden body, ``feed()`` must NOT silently
    swallow the streaming output. The body was buffered during the
    commentary deltas (we suppress per-token emission until close), so
    a silent drop would make the entire tool call invisible to the
    streaming consumer. Surface the body bytes as a CONTENT event so
    the user still sees what the model emitted — matches baseline
    behavior where the legacy router leaked commentary body into
    CONTENT verbatim.

    Force the abstain path by monkeypatching the static reconstructor
    to return None (which is what would happen on a sentinel-laden
    body for a real call); replay a normal commentary tool call and
    assert ``feed_sequence`` surfaces the body bytes via CONTENT.
    """
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}<|call|>'
    )
    tokens = _encode(encoding, text)

    monkeypatch.setattr(
        HarmonyStreamingRouter,
        "_reconstruct_tool_call_text",
        staticmethod(lambda _msg: None),
    )

    router.reset()
    result = router.feed_sequence(tokens)

    # No TOOL_CALL surfaces (abstained).
    assert result["tool_calls"] is None, (
        f"abstain path must not emit TOOL_CALL; got {result['tool_calls']!r}"
    )
    # But the body IS visible to streaming consumers via CONTENT.
    assert result["content"] is not None and '{"city":"NYC"}' in result["content"], (
        f"abstain path must surface body bytes via CONTENT; got {result['content']!r}"
    )


def test_compat_gate_rejects_unknown_tokenizer_identity():
    """Codex round-2 BLOCKING (PR #515): a 3-string probe set cannot
    prove full-vocab parity. The gate must ALSO require the
    tokenizer's reported identity to be a known gpt-oss family
    member; a tokenizer named ``mistralai/Mistral-...`` that happened
    to match the markers and probes would otherwise be wrongly
    upgraded and might corrupt uncommon body tokens.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _NotGptOssTokenizer:
        name_or_path = "mistralai/Mistral-7B-Instruct-v0.3"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            # Even if the tokenizer SOMEHOW produced matching probe IDs,
            # the identity gate must short-circuit.
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _NotGptOssTokenizer()) is False


def test_compat_gate_accepts_gpt_oss_quant_suffix_variants():
    """Codex round-2 BLOCKING (PR #515): the identity allowlist must
    tolerate the org-prefix + quant-suffix renames the mlx-community
    publishes (``mlx-community/gpt-oss-20b-MXFP4-Q8``,
    ``mlx-community/gpt-oss-120b-MXFP4-Q4`` etc.) — case-insensitive
    substring match on ``gpt-oss`` covers all of them while still
    rejecting random model names.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    # Real markers from harmony encoding.
    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_call=_id("<|call|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    class _GptOssLike:
        """A tokenizer whose ``encode`` actually delegates to the
        harmony encoding — so the marker + probe checks pass; only the
        identity check decides accept vs reject for these cases.
        """

        def __init__(self, name):
            self.name_or_path = name
            self._enc = enc

        def encode(self, text, add_special_tokens=False):
            return self._enc.encode(text, allowed_special="none")

        def decode(self, ids):
            return self._enc.decode(ids)

        def get_vocab(self):
            return {}

    # Accepted identities.
    for name in (
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "openai/gpt-oss-120b",
        "mlx-community/GPT-OSS-20b",  # case-insensitive
    ):
        assert is_openai_harmony_compatible(tm, _GptOssLike(name)) is True, (
            f"identity {name!r} must be accepted"
        )

    # Rejected identities even with matching encoder.
    for name in (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen3-0.6B",
        "",  # missing name
    ):
        assert is_openai_harmony_compatible(tm, _GptOssLike(name)) is False, (
            f"identity {name!r} must be rejected"
        )


def test_recipient_shape_accepts_digit_start_names():
    """Codex round-2 BLOCKING (PR #515): the original recipient
    shape regex ``^functions\\.[A-Za-z_]...`` rejected valid OpenAI
    tool names whose first character is a digit (e.g. ``2fa_lookup``).
    Widen to match the OpenAI spec ``[A-Za-z0-9_-]{1,64}``.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import HarmonyStreamingRouter

    class _Adapter:
        name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

    r = HarmonyStreamingRouter(TokenMap(format_tag="harmony"), _Adapter())

    class _Msg:
        def __init__(self, recipient):
            self.recipient = recipient
            self.content = []
            self.content_type = None

    # These were broken on round-1; must pass on round-2.
    for good in (
        "functions.2fa_lookup",
        "functions.0_index",
        "functions.123",
        "functions.tool-with-dash",
    ):
        out = r._reconstruct_tool_call_text(_Msg(good))
        assert f"to={good}" in out, f"good recipient {good!r} got: {out!r}"


def test_compat_gate_rejects_tokenizer_without_encode():
    """Codex round-1 BLOCKING (PR #515): the compatibility gate must
    require body-vocab parity, not just marker IDs. A tokenizer that
    lacks ``.encode`` (e.g. ``_FakeTokenizer`` in
    ``tests/test_engine_router_non_stream.py`` — only exposes
    ``decode`` + ``get_vocab``) cannot prove vocab parity → gate must
    return False so the legacy router runs.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    # Realistic harmony marker IDs (same as production gpt-oss-20b).
    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    # Use a unique identity so the result cache from earlier tests
    # doesn't short-circuit this one (codex round-3 NIT: cache is keyed
    # by lowercased ``name_or_path``).
    class _NoEncodeTokenizer:
        name_or_path = "mlx-community/gpt-oss-NO-ENCODE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

    assert is_openai_harmony_compatible(tm, _NoEncodeTokenizer()) is False


def test_compat_gate_rejects_mismatched_body_vocab():
    """Codex round-1 BLOCKING (PR #515): when markers match but body
    tokens do not, the gate must REJECT — otherwise body tokens are
    decoded through harmony's vocabulary and corrupt content / tool
    arguments. This is the 7-unit-regression smoking gun from
    pr_validate round-1: synthetic test tokenizers had ``Reason=1``
    / ``ing=2`` matching harmony's markers but had completely
    different body IDs, decoded to ``"#`` under harmony encoding.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    # Use a unique identity so the result cache doesn't return a
    # stale True from the sibling accepted-variant test.
    class _MismatchedBodyVocabTokenizer:
        name_or_path = "mlx-community/gpt-oss-MISMATCHED-BODY-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            # Wrong vocab — every body string is two arbitrary IDs.
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _MismatchedBodyVocabTokenizer()) is False


def test_compat_gate_anchored_allowlist_rejects_tail_substring_fake():
    """Codex round-11 BLOCKING (PR #515): the tokenizer-identity
    allowlist used naive ``in`` substring matching, so an identity
    like ``my-not-gpt-oss/whatever`` (or ``foo-gpt-oss-bar`` with no
    ``/`` separator) would pass while clearly not being a real
    gpt-oss family member. Anchored match
    (``(?:^|/)gpt-oss(?:-|$)``) accepts canonical names and rejects
    tail-substring fakes.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    class _CompatTokenizerBase:
        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

    # Tail-substring fakes AND non-allowlisted owner prefixes (codex
    # round-12 BLOCKING — even an anchored basename allowlist let
    # arbitrary owners pass; restrict to known owner prefixes).
    rejected_names = (
        "my-not-gpt-oss/whatever",
        "foo/bar-gpt-oss-malicious",
        "my-not-gpt-oss-20b",  # no slash, tail substring
        "notgpt-oss-fake",
        "some-user/gpt-oss-remapped",  # unknown owner with valid basename
        "evil-org/gpt-oss-20b",
        "anonymous/gpt-oss",
    )
    for name in rejected_names:

        class _Fake(_CompatTokenizerBase):
            pass

        _Fake.name_or_path = name
        assert is_openai_harmony_compatible(tm, _Fake()) is False, (
            f"tail-substring identity {name!r} must be rejected; "
            f"anchored allowlist failed"
        )

    # Canonical names — known owner prefixes (openai, mlx-community,
    # unsloth) plus bare basename. Sanity check that the allowlist
    # tightening didn't accidentally over-reject.
    accepted_names = (
        "openai/gpt-oss-20b",
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "unsloth/gpt-oss-20b-MLX-8bit",
        "gpt-oss-20b",  # bare, anchored at ^
        "gpt-oss",  # bare exact
    )
    for name in accepted_names:

        class _Real(_CompatTokenizerBase):
            pass

        _Real.name_or_path = name
        assert is_openai_harmony_compatible(tm, _Real()) is True, (
            f"canonical identity {name!r} must pass; anchored regex over-rejected"
        )


def test_compat_cache_key_segregates_by_tokenizer_instance():
    """Codex round-11 BLOCKING (PR #515): the per-identity cache used
    ``(name_or_path, marker_ids)`` only, so two distinct tokenizer
    instances with the same name + marker IDs but a remapped body
    vocab shared a single cached decision. A previous ``True`` would
    then upgrade an incompatible tokenizer and corrupt
    ``StreamableParser`` body decoding.

    Fix folded ``id(tokenizer)`` into the cache key — distinct
    instances now segregate naturally, each instance gets an
    authoritative ``_compute_compat`` pass; same-instance reuses
    cleanly (zero-encode cache hit).
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    # Instance #1: real harmony body vocab → True.
    class _RealHarmony:
        name_or_path = "mlx-community/gpt-oss-CACHE-INSTANCE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

    assert is_openai_harmony_compatible(tm, _RealHarmony()) is True

    # Instance #2: SAME name_or_path + SAME marker IDs but a
    # MISMATCHED body vocab. Must NOT reuse instance #1's True (the
    # cache key now includes ``id(tokenizer)`` so #2 gets a fresh
    # authoritative probe).
    class _RemappedBody:
        name_or_path = "mlx-community/gpt-oss-CACHE-INSTANCE-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _RemappedBody()) is False, (
        "round-11: cache must segregate by tokenizer instance; "
        "remapped tokenizer shared instance #1's stale True decision"
    )


def test_compat_gate_rejects_non_int_encode_result():
    """Codex round-8 BLOCKING (PR #515): some HF tokenizer
    configurations (``return_tensors`` defaults, wrapped Fast
    tokenizers) make ``encode()`` return a ``BatchEncoding`` / tensor
    rather than a flat ``list[int]``. ``list(...)`` on those either
    raises or yields opaque objects whose equality check against the
    harmony reference list returns False but in a way that would crash
    later consumers (``StreamableParser.process``). The gate must
    return False on anything that isn't a flat int sequence so the
    engine falls back cleanly to the legacy router.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=200005,
        harmony_message=200008,
        harmony_call=200012,
        harmony_end=200007,
        harmony_return=200002,
        harmony_start=200006,
        harmony_constrain=200003,
    )

    class _BatchEncodingShape:
        """Pretends to be a BatchEncoding — ``list(self)`` returns
        dict-style keys (``"input_ids"``, ``"attention_mask"``), not
        the integer token IDs the gate expects.
        """

        def __iter__(self):
            return iter(("input_ids", "attention_mask"))

    class _BatchEncodingTokenizer:
        name_or_path = "mlx-community/gpt-oss-BATCH-ENCODING-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return _BatchEncodingShape()

    assert is_openai_harmony_compatible(tm, _BatchEncodingTokenizer()) is False

    # And a flat tuple of non-int items (an even more degenerate
    # tokenizer config) must also fall back to False rather than
    # surface a TypeError.
    class _StringTokenIdsTokenizer:
        name_or_path = "mlx-community/gpt-oss-STRING-IDS-VARIANT"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return ("Hello", "world")

    assert is_openai_harmony_compatible(tm, _StringTokenIdsTokenizer()) is False


def test_finalize_never_synthesizes_truncated_commentary(router, encoding):
    """Codex round-6 BLOCKING resolution (PR #515): finalize() adopts
    the vLLM / SGLang safer-default — a commentary message cut off
    before ``<|call|>`` must NEVER be surfaced as a tool call, even
    when the body is syntactically valid JSON. A ``max_tokens`` /
    ``stop`` truncation must not be executed as if the model had
    emitted ``<|call|>``; the upside of rescuing a rare recoverable
    case is dwarfed by the cost of one wrongly-invoked tool. Per-token
    ``feed()`` already produced every body byte (as buffered
    commentary, never routed) — finalize only flushes parser state.
    """
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}'  # NO <|call|> — truncated right before it
    )
    tokens = _encode(encoding, text)
    router.reset()
    routed_during_stream = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None:
            routed_during_stream.append(ev)
    drained = router.finalize()

    assert all(ev.channel != Channel.TOOL_CALL for ev in routed_during_stream)
    assert drained is None, (
        f"truncated commentary must NEVER surface as a tool call; got {drained!r} "
        f"(stream events: {len(routed_during_stream)})"
    )


def test_finalize_drops_mid_json_truncation(router, encoding):
    """Sibling pin to ``test_finalize_never_synthesizes_truncated_commentary``:
    a commentary message whose body was cut off mid-JSON (e.g.
    ``{"city":"NY``) must also be dropped. Covered by the same
    safer-default rule (codex round-6 resolution, PR #515) — finalize()
    never synthesizes a tool call from a truncated commentary message,
    so any partial-JSON body is dropped by construction.
    """
    # Commentary header + truncated mid-string body.
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NY'  # cut off mid-JSON
    )
    tokens = _encode(encoding, text)
    router.reset()
    for tid in tokens:
        router.feed(tid)
    drained = router.finalize()

    assert drained is None, (
        f"mid-JSON truncation must NOT surface as a tool call; got {drained!r}"
    )


def test_finalize_drops_truncated_commentary_with_empty_body(router, encoding):
    """Sibling pin: a commentary message with no body whatsoever
    (truncated right after ``<|message|>``) must also be dropped.
    Covered by the same safer-default rule (codex round-6 resolution,
    PR #515) — finalize() never synthesizes a tool call regardless of
    body shape.
    """
    text = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
    tokens = _encode(encoding, text)
    router.reset()
    for tid in tokens:
        router.feed(tid)
    drained = router.finalize()

    assert drained is None, (
        f"empty-body truncated commentary must be dropped; got {drained!r}"
    )


def test_finalize_does_not_double_emit_completed_final(router, encoding):
    """Codex round-4 BLOCKING (PR #515): for a NORMALLY-completed
    final channel (model emitted ``<|return|>``), the per-token feed
    has already produced every body delta. ``finalize()`` must NOT
    re-emit those bytes as a second routed event — that would
    duplicate the final response.
    """
    text = "<|channel|>final<|message|>Hi there.<|return|>"
    tokens = _encode(encoding, text)
    router.reset()
    streamed = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None:
            streamed.append(ev)
    drained = router.finalize()

    assert drained is None, (
        f"completed final must not re-emit on finalize; got {drained!r}"
    )
    # The per-token stream already produced the full body.
    assert (
        "".join(e.text for e in streamed if e.channel == Channel.CONTENT) == "Hi there."
    )


def test_finalize_does_not_drain_post_eos_buffered_delta(router, encoding):
    """Codex round-6 BLOCKING resolution (PR #515): even if a future
    openai-harmony build flushes a buffered tail via
    ``last_content_delta`` during ``process_eos()``, ``finalize()``
    must NOT route it as a CONTENT / REASONING event. Per-token
    ``feed()`` already produced every body byte the model emitted;
    emitting a post-EOS delta risks duplicating the last token (if the
    build does not clear ``last_content_delta`` between the per-token
    flush and the EOS flush). The safer default matches vLLM / SGLang:
    finalize is a parser-state flush only.
    """
    # Construct a router whose underlying StreamableParser is forced
    # into a state where process_eos produces a synthetic delta.

    class _StubMessage:
        def __init__(self, channel):
            self.channel = channel
            self.recipient = None
            self.content = []
            self.content_type = None

    class _StubParser:
        def __init__(self, channel, delta):
            self._channel = channel
            self._delta = delta
            self.tokens = [1]
            self.messages: list[_StubMessage] = []

        @property
        def current_channel(self):
            return self._channel

        @property
        def last_content_delta(self):
            return self._delta

        def process_eos(self):
            self.messages.append(_StubMessage(self._channel))
            self._delta = " tail"

    router.reset()
    router._parser = _StubParser("final", "")
    drained = router.finalize()
    assert drained is None, f"post-EOS final delta must NOT drain; got {drained!r}"

    router.reset()
    router._parser = _StubParser("analysis", "")
    drained = router.finalize()
    assert drained is None, f"post-EOS analysis delta must NOT drain; got {drained!r}"


def test_compat_gate_rejects_missing_marker_ids():
    """Codex round-4 BLOCKING (PR #515): the gate must require ALL
    seven harmony markers to be present in the tokenizer's TokenMap.
    Previously a tokenizer with only ``<|channel|>`` / ``<|message|>``
    plus matching probe IDs could pass — StreamableParser would then
    crash when the model emitted a ``<|call|>`` the gate never checked.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    # Missing ``harmony_call`` — gate must reject.
    tm_missing_call = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        # harmony_call=None (left out)
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    class _GptOssTokenizer:
        name_or_path = "mlx-community/gpt-oss-MISSING-CALL-VARIANT"

        def encode(self, text, add_special_tokens=False):
            return enc.encode(text, allowed_special="none")

        def decode(self, ids):
            return enc.decode(ids)

        def get_vocab(self):
            return {}

    assert is_openai_harmony_compatible(tm_missing_call, _GptOssTokenizer()) is False


def test_compat_gate_cache_segregates_by_marker_ids():
    """Codex round-4 NIT (PR #515): a SINGLE tokenizer instance probed
    with two different ``TokenMap`` marker-ID tuples must NOT share a
    cached compatibility result — the inner per-tokenizer cache keys
    on the ``(name_lc, marker_ids)`` tuple, so distinct marker tuples
    segregate into distinct entries even on the same tokenizer.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import (
        _COMPAT_RESULT_CACHE,
        is_openai_harmony_compatible,
    )

    class _T:
        name_or_path = "mlx-community/gpt-oss-CACHE-KEY-SEGREGATION"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return [0]  # always rejected by body-vocab probe

    tm_a = TokenMap(format_tag="harmony", harmony_channel=200005)
    tm_b = TokenMap(format_tag="harmony", harmony_channel=999999)

    t = _T()  # one instance — keep alive for the WeakKeyDictionary
    is_openai_harmony_compatible(tm_a, t)
    is_openai_harmony_compatible(tm_b, t)

    inner = _COMPAT_RESULT_CACHE.get(t)
    assert inner is not None, "tokenizer must have a cache slot after probes"
    assert len(inner) == 2, (
        f"cache must keep separate inner entries per marker-ID tuple; "
        f"got {len(inner)} entries: {list(inner.keys())!r}"
    )


def test_compat_gate_caches_per_tokenizer_identity():
    """Codex round-3 NIT (PR #515): the compatibility probe is called
    on every request from the engine's streaming factory; the marker
    + body-vocab checks should only run once per tokenizer identity.

    Codex round-5 BLOCKING redesign: the previous version used an
    identity that failed the allowlist gate up-front, so ``encode``
    was never invoked even without caching — the test would pass even
    if the cache were removed. This version uses a legal gpt-oss
    identity with real marker IDs so the gate reaches the body-vocab
    probe step; only a mismatching probe response causes the False
    result, and the second call MUST be short-circuited by the cache.
    """
    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import is_openai_harmony_compatible

    enc = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    def _id(s):
        return enc.encode(s, allowed_special="all")[0]

    # Real marker IDs so steps (1) identity allowlist and (2) marker
    # parity both pass; the body-vocab probe at step (3) is what
    # rejects this tokenizer (mismatching encode).
    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_id("<|channel|>"),
        harmony_message=_id("<|message|>"),
        harmony_call=_id("<|call|>"),
        harmony_end=_id("<|end|>"),
        harmony_return=_id("<|return|>"),
        harmony_start=_id("<|start|>"),
        harmony_constrain=_id("<|constrain|>"),
    )

    call_count = {"n": 0}

    class _CountingTokenizer:
        # Allowlisted identity — substring match on "gpt-oss".
        name_or_path = "mlx-community/gpt-oss-CACHE-PROBE-INVOCATION"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            call_count["n"] += 1
            return [0]  # nonsense — guaranteed mismatch → False

    # Fresh tokenizer instance — the WeakKeyDictionary cache is keyed
    # on the instance object, so a new instance is guaranteed to start
    # with no stale entry regardless of prior test state.
    t = _CountingTokenizer()
    # First call reaches the body-vocab probe and increments
    # call_count — that's the work the cache must save on subsequent
    # calls.
    result1 = is_openai_harmony_compatible(tm, t)
    assert result1 is False
    first_call_count = call_count["n"]
    assert first_call_count > 0, (
        "test setup error: first call must reach the encode probe — "
        f"expected >0 encode invocations, got {first_call_count}"
    )

    # Second call must return the cached False without re-invoking
    # encode. If the cache were removed, call_count would double on
    # this call.
    result2 = is_openai_harmony_compatible(tm, t)
    assert result2 is False
    assert call_count["n"] == first_call_count, (
        "compat gate must cache False results per identity; "
        f"saw {call_count['n']} encode calls (expected {first_call_count} "
        "— second call should have hit the cache and not re-probed)"
    )
