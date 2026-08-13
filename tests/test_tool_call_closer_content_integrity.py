# SPDX-License-Identifier: Apache-2.0
"""``</tool_call>`` must survive on the CONTENT channel.

The global output sanitizer used to strip the ``</tool_call>`` closer
unconditionally, on every surface. The token then disappeared out of
ordinary assistant prose — including text a coding agent had been asked
to write to a file.

Reproduced end-to-end before the fix, with real client binaries pointed
at a real ``rapid-mlx serve``:

* Claude Code 2.1.220 via ``/v1/messages``
* Codex 0.146.0 via ``/v1/responses``

Both were asked to write a file whose first line was::

    A tool call block ends with </tool_call> on its own.

Both wrote ``"A tool call block ends with  on its own."`` to disk —
marker excised, the double space left behind — and both then burned
turns trying to work around output they could not account for.

The same token on the REASONING channel is parser residue and stays
stripped. That split mirrors the one already drawn for ``<think>``
(``_FINAL_SANITIZER`` leaves the opener alone; the reasoning-channel
helper removes it).
"""

import pytest

from vllm_mlx.api.models import ChatCompletionChunkDelta
from vllm_mlx.api.utils import (
    sanitize_content_for_stream,
    sanitize_output,
    sanitize_reasoning_content,
    sanitize_reasoning_for_stream,
)

# The line the two agents were asked to write, verbatim.
AGENT_LINE = "A tool call block ends with </tool_call> on its own."

PROSE_CASES = [
    AGENT_LINE,
    "A parameter block ends with </parameter> here.",
    "Wrap the call in <tool_call> ... </tool_call> and stop.",
    "Emit </tool_call> to close it.",
    "</tool_call>",
]


class TestContentChannelKeepsTheCloser:
    @pytest.mark.parametrize("text", PROSE_CASES)
    def test_sanitize_output_is_identity_on_prose(self, text):
        assert sanitize_output(text) == text

    @pytest.mark.parametrize("text", PROSE_CASES)
    def test_streaming_content_delta_is_identity_on_prose(self, text):
        assert sanitize_content_for_stream(text) == text

    @pytest.mark.parametrize("text", PROSE_CASES)
    def test_chunk_delta_validator_is_identity_on_prose(self, text):
        assert ChatCompletionChunkDelta(content=text).content == text

    def test_the_exact_bytes_both_agents_lost(self):
        """The double-space signature is what made this diagnosable.

        A model self-censoring the token would rephrase; only an excision
        leaves the surrounding words butted against two spaces.
        """
        corrupted = "A tool call block ends with  on its own."
        assert sanitize_output(AGENT_LINE) != corrupted
        assert sanitize_output(AGENT_LINE) == AGENT_LINE


class TestReasoningChannelStillStrips:
    def test_final_reasoning_sanitizer_strips_closer(self):
        assert sanitize_reasoning_content("thinking </tool_call> done") == (
            "thinking  done"
        )

    def test_streaming_reasoning_sanitizer_strips_closer(self):
        assert sanitize_reasoning_for_stream("thinking </tool_call> done") == (
            "thinking  done"
        )

    def test_pure_marker_reasoning_collapses_to_none(self):
        assert sanitize_reasoning_content("</tool_call>") is None

    def test_chunk_delta_validator_strips_on_reasoning_field(self):
        delta = ChatCompletionChunkDelta(reasoning_content="</tool_call>")
        assert delta.reasoning_content is None


class TestUnrelatedMarkersAreUnaffected:
    """The fix is scoped to the one token — nothing else moved."""

    @pytest.mark.parametrize(
        "marker", ["<|im_start|>", "<|endoftext|>", "<|channel|>", "</think>"]
    )
    def test_special_tokens_still_stripped_from_content(self, marker):
        assert marker not in (sanitize_output(f"text {marker} more") or "")

    def test_gemma4_full_call_still_stripped_from_content(self):
        out = sanitize_output(
            'before <|tool_call>call:f{a:<|"|>b<|"|>}<tool_call|> after'
        )
        assert "call:f" not in (out or "")
        assert "before" in (out or "")

    def test_calling_tool_text_fallback_still_stripped(self):
        out = sanitize_output('x [Calling tool: f({"a":1})] y')
        assert "Calling tool" not in (out or "")


class TestAssistantMessageEnvelope:
    """The NON-streaming twin of the chunk-delta validator.

    ``AssistantMessage`` is built by the chat route, the Anthropic
    adapter and the Responses adapter, so a channel-blind validator here
    corrupted all three surfaces at once — which is precisely how the
    same bug showed up in Claude Code (``/v1/messages``) and Codex
    (``/v1/responses``) from one root cause.

    Located by tracing the live server: every component passed the token
    through in isolation, and only an in-process trace showed
    ``sanitize_reasoning_content`` being applied to the ``content``
    field by this validator.
    """

    def test_content_keeps_the_closer(self):
        from vllm_mlx.api.models import AssistantMessage

        msg = AssistantMessage(content=AGENT_LINE)
        assert msg.content == AGENT_LINE

    def test_reasoning_content_still_stripped(self):
        from vllm_mlx.api.models import AssistantMessage

        msg = AssistantMessage(content="ok", reasoning_content="think </tool_call> end")
        assert msg.reasoning_content == "think  end"

    def test_other_special_tokens_still_stripped_from_content(self):
        from vllm_mlx.api.models import AssistantMessage

        msg = AssistantMessage(content="answer<|im_end|>")
        assert msg.content == "answer"


class TestAnthropicThinkingBlockStillStrips:
    """The `thinking` block is the reasoning channel on /v1/messages.

    Splitting the sanitizers left `_sanitize_reasoning_channel` calling
    the now content-only `sanitize_output`, so a bare `</tool_call>`
    started surviving into `thinking`. Caught in review; pinned here.
    """

    def test_thinking_block_strips_the_closer(self):
        from vllm_mlx.api.anthropic_adapter import _thinking_block_content

        assert _thinking_block_content("x</tool_call>y", "answer") == "xy"

    def test_thinking_block_keeps_ordinary_prose(self):
        from vllm_mlx.api.anthropic_adapter import _thinking_block_content

        assert (
            _thinking_block_content("weighing the options", "answer")
            == "weighing the options"
        )


class TestStreamingSanitizerHelpersKeepTheCloser:
    """The per-delta helpers, asserted on the CLOSER specifically.

    The pre-existing #1508 branch only checks that the OPENER survives, so
    routing streamed content back through the reasoning sanitizer would
    still have passed there.

    NOTE these exercise the HELPERS, not `routes/chat.py::_fast_sse_chunk`
    itself — that closure is defined inside the streaming generator and is
    not reachable from a unit test. Its dispatch is a one-line
    `field == "reasoning_content"` branch; an HTTP-level streaming test
    would be needed to pin it end to end.
    """

    def test_content_stream_helper_keeps_the_closer(self):
        assert sanitize_content_for_stream(AGENT_LINE) == AGENT_LINE
        # ...while the reasoning half of the same helper pair does strip.
        assert "</tool_call>" not in sanitize_reasoning_for_stream(AGENT_LINE)

    def test_content_delta_and_reasoning_delta_diverge(self):
        """Both fields, one envelope — the dispatch must not collapse."""
        delta = ChatCompletionChunkDelta(
            content=AGENT_LINE, reasoning_content=AGENT_LINE
        )
        assert delta.content == AGENT_LINE
        assert "</tool_call>" not in (delta.reasoning_content or "")


class TestRescuePrefixBranchAlsoStrips:
    """The length-cut rescue branch is the OTHER call site that moved.

    raised in review, twice: my first attempt at this test passed neither
    ``finish_reason="length"`` nor rescue-shaped content, so it ran the
    ORDINARY path and would have stayed green with the rescue line
    reverted. Reviewer mutated that line in memory and confirmed it.

    Entering the branch needs all three at once:

    * ``finish_reason == "length"``
    * ``text`` that satisfies ``is_rescue_payload`` (the cutoff sentinel)
    * a reasoning trace LONGER than ``RESCUE_TAIL_LENGTH``, so a
      non-empty prefix survives the tail slice

    The assertion below is on the PREFIX, so the closer under test must
    sit in the first ``len(reasoning) - RESCUE_TAIL_LENGTH`` characters.
    """

    def test_length_cut_rescue_prefix_strips_the_closer(self):
        from vllm_mlx.api.anthropic_adapter import _thinking_block_content
        from vllm_mlx.api.constants import (
            REASONING_CUTOFF_SENTINEL,
            RESCUE_TAIL_LENGTH,
        )

        # Closer up front, then enough filler that the prefix survives
        # the tail slice.
        reasoning = "planning </tool_call> the next step. " + (
            "x" * RESCUE_TAIL_LENGTH * 2
        )
        rescue_text = f"{REASONING_CUTOFF_SENTINEL}\n\nsome tail"

        out = _thinking_block_content(reasoning, rescue_text, "length")

        assert out, "expected a non-empty thinking prefix from the rescue branch"
        assert "planning" in out, (
            f"did not enter the rescue-prefix branch; got {out[:80]!r}"
        )
        assert "</tool_call>" not in out, (
            f"closer leaked through the rescue prefix: {out[:120]!r}"
        )

    def test_ordinary_path_still_strips_the_closer(self):
        """The non-rescue path, for contrast — both must hold."""
        from vllm_mlx.api.anthropic_adapter import _thinking_block_content

        assert _thinking_block_content("x</tool_call>y", "answer") == "xy"
