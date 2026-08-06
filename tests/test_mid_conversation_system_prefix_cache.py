# SPDX-License-Identifier: Apache-2.0
"""Mid-conversation system messages must not move to the front.

Claude Code injects a ``role: "system"`` message into ``messages``
routinely — the "task tools haven't been used recently" nudge, the "date
has changed" notice, and entering or leaving plan mode — always at the
END of the array, right before the new user turn.

Hoisting those into the leading system block grows the FRONT of the
prompt, shifts everything behind it, and invalidates the prefix cache at
the system/first-user boundary. Measured on qwen3.6-35b before the fix:

    A  no nudge                  input_tokens = 775
    B  nudge mid-array           input_tokens = 803
    C  nudge appended to system  input_tokens = 803   <- B == C, hoisted

    warm-up          input=775  cached=None
    identical resend input=15   cached=760
    + one nudge      input=803  cached=None   <- whole warm prefix gone

After the fix, ``+ one nudge`` reports ``input=55 cached=760`` and a
five-turn conversation hits the cache on every turn instead of never.

The relocation target is the FOLLOWING user turn, wrapped in
``<system-reminder>`` — the tag Claude Code itself uses for reminders it
injects into user turns upstream, so models already read it as an
instruction rather than something the human typed. That turn is new on
this request anyway, so nothing previously cacheable is disturbed.
"""

import pytest

from vllm_mlx.api.models import Message
from vllm_mlx.api.responses_adapter import _merge_system_messages as _merge_raw


def _merge_system_messages(messages):
    """Relocation is opt-in and only the Anthropic lane opts in."""
    return _merge_raw(messages, relocate_mid_conversation=True)


NUDGE = (
    "The task tools haven't been used recently. If you're working on tasks "
    "that would benefit from tracking progress, consider using TaskCreate."
)
DATE = "The date has changed. Today's date is now 2026-08-05."


def roles(msgs):
    return [m.role for m in msgs]


def text(msg):
    return msg.content if isinstance(msg.content, str) else str(msg.content)


class TestLeadingSystemUnchanged:
    """The historical contract for LEADING system messages still holds."""

    def test_single_leading_system_stays_at_index_0(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base prompt"),
                Message(role="user", content="hi"),
            ]
        )
        assert roles(out) == ["system", "user"]
        assert text(out[0]) == "base prompt"

    def test_multiple_leading_systems_merge_in_order(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="first"),
                Message(role="system", content="second"),
                Message(role="user", content="hi"),
            ]
        )
        assert roles(out) == ["system", "user"]
        assert text(out[0]) == "first\n\nsecond"

    def test_no_system_is_a_passthrough(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        assert _merge_system_messages(msgs) == msgs


class TestMidConversationSystemStaysInPlace:
    def test_nudge_folds_into_the_following_user_turn(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base prompt"),
                Message(role="user", content="turn 0"),
                Message(role="assistant", content="ack"),
                Message(role="system", content=NUDGE),
                Message(role="user", content="turn 1"),
            ]
        )
        # Leading block is untouched — this is what keeps the prefix stable.
        assert roles(out) == ["system", "user", "assistant", "user"]
        assert text(out[0]) == "base prompt"
        assert NUDGE not in text(out[0])

        # ...and the nudge landed on the NEW turn, at its true position.
        last = text(out[-1])
        assert NUDGE in last
        assert "<system-reminder>" in last
        assert last.endswith("turn 1")

    def test_earlier_turns_are_byte_identical(self):
        """The property the cache actually depends on."""
        base = [
            Message(role="system", content="base prompt"),
            Message(role="user", content="turn 0"),
            Message(role="assistant", content="ack"),
        ]
        without = _merge_system_messages(base + [Message(role="user", content="t1")])
        with_nudge = _merge_system_messages(
            base
            + [
                Message(role="system", content=NUDGE),
                Message(role="user", content="t1"),
            ]
        )
        assert [text(m) for m in without[:-1]] == [text(m) for m in with_nudge[:-1]]

    def test_several_nudges_before_one_user_turn(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="assistant", content="ack"),
                Message(role="system", content=NUDGE),
                Message(role="system", content=DATE),
                Message(role="user", content="t1"),
            ]
        )
        assert roles(out) == ["system", "user", "assistant", "user"]
        assert text(out[0]) == "base"
        last = text(out[-1])
        assert NUDGE in last and DATE in last
        assert last.count("<system-reminder>") == 2

    def test_nudge_with_no_following_user_turn_falls_back_to_hoist(self):
        """Never drop an instruction on the floor."""
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
            ]
        )
        assert roles(out) == ["system", "user"]
        assert NUDGE in text(out[0])

    def test_nudge_before_an_assistant_turn_is_not_folded_into_it(self):
        """Only USER turns absorb a reminder; assistant turns are model output."""
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
                Message(role="assistant", content="ack"),
                Message(role="user", content="t1"),
            ]
        )
        assert roles(out) == ["system", "user", "assistant", "user"]
        assert text(out[2]) == "ack"
        assert NUDGE in text(out[-1])

    def test_empty_mid_system_contributes_nothing(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=""),
                Message(role="user", content="t1"),
            ]
        )
        assert roles(out) == ["system", "user", "user"]
        assert text(out[-1]) == "t1"
        assert "<system-reminder>" not in text(out[-1])


class TestTemplateContractPreserved:
    """At most ONE system message, at index 0 — the reason this
    function exists (Qwen / Llama / Gemma templates raise otherwise)."""

    def test_no_system_message_survives_past_index_0(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
                Message(role="assistant", content="ack"),
                Message(role="system", content=DATE),
                Message(role="user", content="t1"),
            ]
        )
        assert all(m.role != "system" for m in out[1:])
        assert out[0].role == "system"


class TestLaneGating:
    """Relocation is opt-in; only the Anthropic lane opts in.

    codex r1 MAJOR: Codex sends ``developer`` items as DURABLE
    instructions. Folding one into a user turn demotes it from
    template-enforced system authority to ordinary user text that a
    later user message can override — e.g. a developer item pinning
    ``dry_run=true`` followed by a user asking to ignore it. The
    Responses lane therefore keeps the historical hoist.
    """

    def test_responses_lane_still_hoists(self):
        msgs = [
            Message(role="system", content="base"),
            Message(role="user", content="t0"),
            Message(role="system", content="developer: always dry_run"),
            Message(role="user", content="t1"),
        ]
        out = _merge_raw(msgs)  # no relocate flag == Responses/Codex lane
        assert roles(out) == ["system", "user", "user"]
        assert "always dry_run" in text(out[0])

    def test_anthropic_lane_relocates(self):
        msgs = [
            Message(role="system", content="base"),
            Message(role="user", content="t0"),
            Message(role="system", content=NUDGE),
            Message(role="user", content="t1"),
        ]
        out = _merge_raw(msgs, relocate_mid_conversation=True)
        assert text(out[0]) == "base"
        assert NUDGE in text(out[-1])


class TestNeverMixRelocationAndHoisting:
    """codex r1 BLOCKING: mixing the two inverts instruction order.

    ``enter plan mode`` folded into a user turn while a later
    ``exit plan mode`` gets hoisted would render the NEWEST instruction
    as the OLDEST — the model could stay in plan mode after being told
    to leave. When any mid-conversation system has no following user
    turn we hoist them all, the historical way, and accept the lost
    cache hit on that one request.
    """

    def test_trailing_system_forces_whole_request_to_hoist(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content="enter plan mode"),
                Message(role="user", content="plan this"),
                Message(role="system", content="exit plan mode"),
            ]
        )
        merged = text(out[0])
        # Both instructions live in the leading block, in ARRIVAL order.
        assert merged.index("enter plan mode") < merged.index("exit plan mode")
        # And nothing was folded into a user turn.
        assert all("<system-reminder>" not in text(m) for m in out[1:])

    def test_no_mixing_when_a_tool_turn_trails(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
                Message(role="user", content="t1"),
                Message(role="system", content=DATE),
                Message(role="assistant", content="ack"),
            ]
        )
        assert all("<system-reminder>" not in text(m) for m in out[1:])
        assert NUDGE in text(out[0]) and DATE in text(out[0])


class TestContentShapePreserved:
    """codex r1 BLOCKING: flattening dropped non-text content blocks."""

    def test_list_content_keeps_every_block(self):
        blocks = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
                Message(role="user", content=blocks),
            ]
        )
        content = out[-1].content
        assert isinstance(content, list), "list content must not be flattened"

        # Pydantic coerces raw dicts into content-part models, so compare
        # semantically rather than by identity.
        def part(block, key):
            if isinstance(block, dict):
                return block.get(key)
            return getattr(block, key, None)

        # The reminder leads...
        assert part(content[0], "type") == "text"
        assert NUDGE in part(content[0], "text")
        # ...the original text block is still there...
        assert any(part(b, "text") == "look at this" for b in content)
        # ...and the IMAGE survived, which is the whole point: flattening
        # to a string used to drop it and an MLLM answered without it.
        images = [b for b in content if part(b, "type") == "image_url"]
        assert len(images) == 1
        url = part(images[0], "image_url")
        assert "data:image/png;base64,AAA" in str(
            url if isinstance(url, str) else getattr(url, "url", url)
        )

    def test_string_content_stays_a_string(self):
        out = _merge_system_messages(
            [
                Message(role="system", content="base"),
                Message(role="user", content="t0"),
                Message(role="system", content=NUDGE),
                Message(role="user", content="t1"),
            ]
        )
        assert isinstance(out[-1].content, str)


class TestRenderedTokenPrefixIsStable:
    """codex r1 MINOR: assert the property at the TOKEN layer.

    Comparing ``Message.content`` strings only shows the messages did not
    change. What the prefix cache actually depends on is that the
    RENDERED token stream still shares its leading tokens. A template or
    normalisation change could start framing ``<system-reminder>``
    globally, or hoist it again, and every string-level test above would
    stay green while ``cached_tokens`` silently returned to zero.
    """

    @staticmethod
    def _render(tok, messages):
        rendered = tok.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=True,
            add_generation_prompt=True,
        )
        # Transformers 5 returns ``BatchEncoding`` here while older
        # releases returned the token-id list directly.  Iterating a
        # BatchEncoding compares the keys (``input_ids``,
        # ``attention_mask``), which made this cache-prefix assertion pass
        # vacuously.  Always compare the actual rendered token stream.
        if hasattr(rendered, "get") and rendered.get("input_ids") is not None:
            return rendered["input_ids"]
        return rendered

    @staticmethod
    def _shared_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def test_leading_tokens_survive_an_injected_nudge(self):
        # ``transformers`` exposes ``apply_chat_template`` even when its
        # optional Jinja runtime is absent.  The lean pr_validate environment
        # intentionally omits that extra, so skip the integration assertion
        # there instead of reporting a product regression.  The structural
        # tests above remain mandatory in every environment.
        pytest.importorskip("jinja2")
        tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
            "mlx-community/Qwen3-0.6B-8bit"
        )

        history = [
            Message(role="system", content="You are a helpful assistant. " * 40),
            Message(role="user", content="turn 0"),
            Message(role="assistant", content="acknowledged"),
        ]
        without = self._render(tok, history + [Message(role="user", content="turn 1")])
        with_nudge = self._render(
            tok,
            _merge_system_messages(
                history
                + [
                    Message(role="system", content=NUDGE),
                    Message(role="user", content="turn 1"),
                ]
            ),
        )

        shared = self._shared_prefix(without, with_nudge)
        # The nudge lands on the LAST user turn, so everything up to that
        # turn must still match token-for-token. Anything less means it
        # moved forward again and the cache is dead.
        assert shared >= len(self._render(tok, history)) - 8, (
            f"rendered prefix diverged after only {shared} tokens; "
            f"history alone renders to {len(self._render(tok, history))}"
        )

    def test_hoisting_would_have_failed_this_test(self):
        """Pin the contrast, so the assertion above cannot pass vacuously."""
        pytest.importorskip("jinja2")
        tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
            "mlx-community/Qwen3-0.6B-8bit"
        )
        history = [
            Message(role="system", content="You are a helpful assistant. " * 40),
            Message(role="user", content="turn 0"),
            Message(role="assistant", content="acknowledged"),
        ]
        without = self._render(tok, history + [Message(role="user", content="turn 1")])
        hoisted = self._render(
            tok,
            [
                Message(role="system", content=text(history[0]) + "\n\n" + NUDGE),
                *history[1:],
                Message(role="user", content="turn 1"),
            ],
        )
        # Hoisting diverges inside the leading system block, before the
        # previously rendered history is complete.  Use the same history
        # boundary as the positive assertion above instead of a fixed token
        # count: the repeated system text is deliberately long, so a fixed
        # ``< 40`` threshold does not describe where the divergence occurs.
        assert (
            self._shared_prefix(without, hoisted) < len(self._render(tok, history)) - 8
        )
