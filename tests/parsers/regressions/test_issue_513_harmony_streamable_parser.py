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

    class _NoEncodeTokenizer:
        # Pass the identity gate so we exercise the no-encode path.
        name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"

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

    class _MismatchedBodyVocabTokenizer:
        # Pass the identity gate so the body-vocab probe step actually
        # runs and is what rejects.
        name_or_path = "mlx-community/gpt-oss-20b-MXFP4-Q8"

        def decode(self, ids):
            return ""

        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            # Wrong vocab — every body string is two arbitrary IDs.
            return [99001, 99002]

    assert is_openai_harmony_compatible(tm, _MismatchedBodyVocabTokenizer()) is False


def test_finalize_drains_truncated_commentary_message(router, encoding):
    """End-of-stream drain: a commentary message that was cut off mid-
    body (no ``<|call|>``) must still surface as a TOOL_CALL event when
    finalize() runs ``process_eos`` — otherwise truncated generations
    silently drop tool calls.
    """
    # Truncate right before <|call|>.
    text = (
        "<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>"
        '{"city":"NYC"}'  # NO <|call|>
    )
    tokens = _encode(encoding, text)
    router.reset()
    routed_during_stream = []
    for tid in tokens:
        ev = router.feed(tid)
        if ev is not None:
            routed_during_stream.append(ev)
    drained = router.finalize()

    # No tool call during the truncated stream — final drain surfaces it.
    assert all(ev.channel != Channel.TOOL_CALL for ev in routed_during_stream)
    assert drained is not None and drained.channel == Channel.TOOL_CALL, (
        f"truncated commentary must drain on finalize; "
        f"got {drained!r} (stream events: {len(routed_during_stream)})"
    )
    assert "functions.get_weather" in drained.text
