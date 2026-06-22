# SPDX-License-Identifier: Apache-2.0
"""Regression tests for D-TOOLCHOICE-R1+QWEN3 (0.8.3 hotfix).

Three tightly-related tool-calling failures surfaced on 0.8.3 dogfood:

* **T1** — DeepSeek-R1 distill with ``tool_choice="auto"`` would fabricate
  prose like ``"The current temperature in Tokyo is 24°C"`` without ever
  emitting a tool call. The pre-existing tool-use system suffix asked the
  model to call a tool but did NOT forbid printing fake tool results when
  it decided not to.
* **T2** — DeepSeek-R1 distill with ``tool_choice="required"`` returned
  ``tool_calls[0].arguments == "{}"`` even when the user prompt clearly
  carried the required string parameter. Root cause: the
  ``_forced_tool_call_prefix`` allowlist only contained ``hermes``, so
  the ``deepseek_v31`` / ``deepseek_r1_0528`` parsers fell through to
  the post-parse synthesis fallback (which defaulted ``"{}"``).
* **T3** — qwen3 + ``tool_choice="required"`` emitted malformed JSON
  inside a literal ``<tool_call>`` envelope (e.g. ``"arguments": 4128,
  7591}``). The hermes parser couldn't extract it, the post-parse path
  synthesised a call with empty args, AND the raw ``<tool_call>...``
  text leaked into BOTH ``content`` and ``reasoning_content``.

The fix is structural, not three ad-hoc patches:

1. ``_TOOL_USE_SYSTEM_SUFFIX`` now explicitly forbids fabricating tool
   output when no tool call is made (T1).
2. ``_forced_tool_call_prefix`` recognises ``deepseek_v31`` /
   ``deepseek_r1_0528`` and injects the V3.1 envelope
   ``<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>``
   so the model continues with real arguments instead of relying on the
   empty-args synthesis fallback (T2).
3. ``_synthesize_forced_tool_call`` now accepts ``raw_text`` and runs a
   bounded ``_recover_partial_tool_args`` scan before defaulting to
   ``"{}"``. Wire-marker literals from the failed-parse text are
   scrubbed from ``cleaned_text`` (and the raw text the reasoning
   parser sees) via ``_scrub_tool_wire_literals`` so the leak in T3
   doesn't survive into either user-visible field.

The tests below exercise each layer end-to-end through the real chat
route — no parallel re-implementation of the post-parse pipeline that
can silently drift from production.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.routes.chat import (
    _contains_structural_tool_wire_leak,
    _contains_tool_wire_literal,
    _forced_tool_call_prefix,
    _recover_partial_tool_args,
    _scrub_tool_wire_literals,
    _scrub_visible_tool_wire_leaks,
    _synthesize_forced_tool_call,
)
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.service.helpers import (
    _TOOL_USE_REQUIRED_SUFFIX,
    _TOOL_USE_SYSTEM_SUFFIX,
)
from vllm_mlx.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

# ──────────────────────────────────────────────────────────────────
# Test harness — recording mock engine + client builder
# ──────────────────────────────────────────────────────────────────


class _RecordingEngine:
    """Mock engine that captures kwargs and returns a configurable
    ``GenerationOutput``. Mirrors the harness used by
    ``test_chat_route_tool_choice_enforcement.py`` so test patterns stay
    consistent across files.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    supports_tool_calls = True

    def __init__(self, text: str = "ok", raw_text: str | None = None):
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_messages: Any = None
        self._text = text
        self._raw_text = raw_text if raw_text is not None else text

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text=self._text,
            raw_text=self._raw_text,
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _make_client(
    engine: _RecordingEngine, tool_call_parser: str | None = "hermes"
) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = tool_call_parser
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


_SOLO_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


_WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────
# T1 — tool_choice="auto" must not let prose fabricate tool output
# ──────────────────────────────────────────────────────────────────


def test_t1_auto_suffix_forbids_fake_tool_output():
    """``_TOOL_USE_SYSTEM_SUFFIX`` is the auto-mode prompt. The 0.8.3
    bug was that it asked the model to "use a tool immediately" but
    did NOT block the failure mode where the model emits a fake tool
    result (``"The current temperature in Tokyo is 24°C"``). The new
    suffix carries an explicit no-fabrication clause."""
    # The clause must explicitly forbid fabricating tool output AND
    # specifically the prose shapes Theo's evidence captured.
    s = _TOOL_USE_SYSTEM_SUFFIX
    assert "fabricate" in s.lower()
    # No-fake-API clause names the most common hallucination shapes.
    assert "Tool returned:" in s
    assert "Tool output:" in s
    assert "fake JSON" in s.lower() or "fake api" in s.lower()


def test_t1_auto_suffix_is_injected_for_auto_mode():
    """End-to-end through the chat route: ``tool_choice="auto"`` must
    inject the strengthened suffix so even when the parser sees prose
    (no tool call envelope), the no-fabrication clause was in front of
    the model. Asserts the suffix is part of the rendered system
    message — the actual model behaviour is exercised by the smoke
    section of the PR body."""
    engine = _RecordingEngine(text="some answer", raw_text="some answer")
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _WEATHER_TOOL,
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    # System suffix was injected into the prompt — verify by looking
    # at the messages the engine received.
    assert engine.last_messages is not None
    sys_msgs = [m for m in engine.last_messages if m.get("role") == "system"]
    assert sys_msgs, "tool_choice='auto' must inject a system suffix"
    sys_text = " ".join(m.get("content", "") for m in sys_msgs)
    assert "fabricate" in sys_text.lower(), (
        f"auto-mode system suffix must forbid fabricating tool output; "
        f"got: {sys_text[-300:]!r}"
    )


def test_t1_required_suffix_unchanged():
    """The "required" suffix is separate from the "auto" suffix —
    only the auto-mode suffix needed the no-fabrication clause (the
    required suffix already says "a text-only response is INVALID").
    Pin that the required suffix wording isn't accidentally weakened."""
    assert "MUST call" in _TOOL_USE_REQUIRED_SUFFIX
    assert "INVALID" in _TOOL_USE_REQUIRED_SUFFIX


# ──────────────────────────────────────────────────────────────────
# T2 — tool_choice="required" + deepseek_v31 must inject a wire prefix
# ──────────────────────────────────────────────────────────────────


def test_t2_forced_prefix_deepseek_v31_inserts_envelope():
    """The V3.1 wire shape is ``<｜tool▁calls▁begin｜>``
    ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}``
    ``<｜tool▁call▁end｜><｜tool▁calls▁end｜>``. The forced prefix
    inserts everything up to the JSON body so the model continues with
    real arguments."""
    out = _forced_tool_call_prefix("deepseek_v31", "get_weather")
    assert out is not None
    assert "<｜tool▁calls▁begin｜>" in out
    assert "<｜tool▁call▁begin｜>" in out
    assert "get_weather" in out
    assert out.endswith("<｜tool▁sep｜>")


def test_t2_forced_prefix_deepseek_r1_0528_alias():
    """``deepseek_r1_0528`` is the second alias the same parser
    registers under. It must produce the same prefix as
    ``deepseek_v31`` so DeepSeek-R1 distills (which load under either
    alias depending on config) get consistent forcing."""
    a = _forced_tool_call_prefix("deepseek_v31", "x")
    b = _forced_tool_call_prefix("deepseek_r1_0528", "x")
    assert a == b
    assert a is not None


def test_t2_forced_prefix_deepseek_v1_still_returns_none():
    """The V1 ``deepseek`` parser has a different body shape and
    shares only the outer markers. We intentionally did NOT add it to
    the allowlist — leaving it on the synthesis fallback is safer than
    risking a wrong wire."""
    assert _forced_tool_call_prefix("deepseek", "x") is None
    assert _forced_tool_call_prefix("deepseek_v3", "x") is None


def test_t2_deepseek_v31_parser_consumes_assembled_envelope():
    """End-to-end shape check: when the model continues from the
    injected prefix and emits ``{"city":"Tokyo"}<｜tool▁call▁end｜>``
    ``<｜tool▁calls▁end｜>``, the parser MUST extract a non-empty
    ``arguments`` string — that's the entire point of injecting the
    prefix.
    """
    prefix = _forced_tool_call_prefix("deepseek_v31", "get_weather")
    assert prefix is not None
    # Simulate the model's continuation past the injected prefix.
    full = (
        prefix + '{"city":"Tokyo","units":"c"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    parser = DeepSeekV31ToolParser(tokenizer=None)
    parser.reset()
    result = parser.extract_tool_calls(full, None)
    assert result.tools_called
    assert result.tool_calls[0]["name"] == "get_weather"
    import json as _json

    args = _json.loads(result.tool_calls[0]["arguments"])
    assert args == {"city": "Tokyo", "units": "c"}, (
        f"deepseek_v31 forced prefix did not produce real arguments; got {args!r}"
    )


def test_t2_chat_route_wires_deepseek_v31_prefix_to_engine():
    """Through the FULL chat route: ``tool_choice="required"`` + single
    tool + ``deepseek_v31`` parser must ship a ``forced_assistant_prefix``
    in the engine kwargs. Without it, T2 fails the same way as 0.8.3."""
    engine = _RecordingEngine()
    client = _make_client(engine, tool_call_parser="deepseek_v31")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather in Tokyo in celsius?"}],
            "tools": _WEATHER_TOOL,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    # codex r2 NIT: assert the exact expected status BEFORE inspecting
    # engine kwargs — a loose ``in (200, 422)`` would let a response-
    # path regression slip through silently. The mock engine returns
    # ``text="ok"`` with ``finish_reason="stop"``, so the route's
    # post-parse path synthesises a tool call (single-tool
    # ``required``) and ships 200.
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"].get("tool_calls"), (
        f"tool_choice=required must yield tool_calls in the response; got {body!r}"
    )
    # Separately assert that the prefix WAS injected into engine kwargs
    # — this is the T2 invariant the test pins.
    assert engine.last_chat_kwargs is not None
    prefix_kw = engine.last_chat_kwargs.get("forced_assistant_prefix")
    assert prefix_kw is not None, (
        "deepseek_v31 + tool_choice=required + 1 tool must inject a "
        "forced_assistant_prefix into engine kwargs (T2 fix)"
    )
    assert "<｜tool▁call▁begin｜>" in prefix_kw
    assert "get_weather" in prefix_kw


# ──────────────────────────────────────────────────────────────────
# T3 — qwen3 tool_choice="required" wire-leak + arg recovery
# ──────────────────────────────────────────────────────────────────


# The exact malformed wire shape Theo captured on qwen3.5-4b-4bit:
# the model emits invalid JSON (``"arguments": 4128, 7591`` — bare
# positional integers instead of an object body) inside a
# ``<tool_call>`` envelope, then closes with stray ``</parameter>`` /
# ``</function>`` tags.
_QWEN3_MALFORMED_BODY = (
    "<tool_call>\n"
    '{"name": "add_numbers", "arguments": 4128, 7591}\n'
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


_QWEN3_BODY_WITH_RECOVERABLE_ARGS = (
    "<tool_call>\n"
    '{"name": "add_numbers", "arguments": {"a": 4128, "b": 7591}}\n'
    # Missing </tool_call> — parser's strict pattern fails to match
    "</function>"
)


def test_t3_hermes_parser_alone_cannot_extract_malformed_body():
    """Establishes the prerequisite for the fix: with the malformed
    JSON body the hermes parser CANNOT extract a tool call. The fix
    has to handle this case at the chat-route level — adding more
    regexes to the parser would risk false positives on prose."""
    parser = HermesToolParser(tokenizer=None)
    parser.reset()
    r = parser.extract_tool_calls(_QWEN3_MALFORMED_BODY, None)
    assert not r.tools_called
    # And the leak: the parser's content output IS the raw wire.
    assert "<tool_call>" in (r.content or "")


def test_t3_scrub_wire_literals_removes_qwen3_leak():
    """``_scrub_tool_wire_literals`` is the parser-agnostic cleanup
    that strips wire-opener literals when the post-parse path
    synthesises a forced call. After scrubbing, none of the literal
    wire markers must remain in the text."""
    out = _scrub_tool_wire_literals(_QWEN3_MALFORMED_BODY)
    for leak in (
        "<tool_call>",
        "</tool_call>",
        "<function>",
        "</function>",
        "</parameter>",
    ):
        assert leak not in out, (
            f"_scrub_tool_wire_literals left {leak!r} behind: {out!r}"
        )


def test_t3_scrub_wire_literals_idempotent_on_clean_prose():
    """The scrubber must be a no-op on text that contains no wire
    markers. Otherwise it would corrupt the auto/non-forced paths
    where prose is the legitimate model output."""
    prose = "The sky is blue. Here is a sentence about birds."
    assert _scrub_tool_wire_literals(prose) == prose


def test_t3_scrub_handles_deepseek_v31_leak():
    """The scrubber is parser-agnostic. A leaked DeepSeek envelope
    (e.g. truncated tool-call begin without a matching end) must also
    be stripped."""
    leak = "prefix <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>x<｜tool▁sep｜>{}"
    out = _scrub_tool_wire_literals(leak)
    assert "<｜tool▁calls▁begin｜>" not in out
    assert "<｜tool▁call▁begin｜>" not in out
    assert "prefix" in out


def test_t3_scrub_handles_nemotron_and_glm_and_mistral_leaks():
    """Confidence sweep — every parser wire we documented in the
    scrub allowlist must be cleared."""
    cases = [
        "<function=foo>arg=1</function>",
        "<arg_key>k</arg_value>",
        "<minimax:tool_call>foo</minimax:tool_call>",
        "<invoke name='x'>body</invoke>",
        "[TOOL_CALLS]{}[/TOOL_CALLS]",
        "<|tool_calls_section_begin|>x<|tool_calls_section_end|>",
    ]
    for c in cases:
        out = _scrub_tool_wire_literals(c)
        # The wire opener bytes must NOT survive.
        for opener in (
            "<function=",
            "<arg_key>",
            "<minimax:tool_call>",
            "<invoke",
            "[TOOL_CALLS]",
            "<|tool_calls_section_begin|>",
        ):
            assert opener not in out, f"{opener!r} survived scrub of {c!r}: {out!r}"


def test_t3_recover_partial_args_from_balanced_object():
    """When the raw text contains ``"arguments": {...}`` with a
    balanced JSON object body, the recovery routine extracts that
    object verbatim (canonicalised). This is the qwen3 "real call,
    bad wrapper" case from the evidence variation."""
    raw = (
        "garbage <tool_call>\n"
        '{"name": "add_numbers", "arguments": {"a": 4128, "b": 7591}}\n'
        "</function> trailing prose"
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"a": 4128, "b": 7591}


def test_t3_recover_partial_args_returns_none_on_qwen3_malformed():
    """The original Theo-evidence qwen3 shape has
    ``"arguments": 4128, 7591`` — NOT a JSON object. The recovery
    routine must return ``None`` so the caller falls back to ``"{}"``;
    coercing two bare integers into named params would require
    schema-side guessing that is out of scope."""
    assert _recover_partial_tool_args(_QWEN3_MALFORMED_BODY) is None


def test_t3_recover_partial_args_returns_none_on_clean_prose():
    """Defensive: prose with no ``"arguments":`` literal yields
    ``None``."""
    assert _recover_partial_tool_args("just some answer") is None
    assert _recover_partial_tool_args("") is None
    assert _recover_partial_tool_args(None) is None


def test_t3_synthesize_uses_recovered_args_when_available():
    """The high-level synth helper prefers recovered args over the
    ``"{}"`` default. With a recoverable body the call's arguments
    must NOT be the empty default."""
    raw_with_args = '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":2}}'
    tc = _synthesize_forced_tool_call("add_numbers", raw_text=raw_with_args)
    import json as _json

    assert _json.loads(tc.function.arguments) == {"a": 1, "b": 2}


def test_t3_synthesize_falls_back_to_empty_when_unrecoverable():
    """When recovery returns ``None``, the helper falls back to the
    caller-provided default (``"{}"`` by default). Pin the
    backward-compat contract — pre-0.8.3 behaviour."""
    tc = _synthesize_forced_tool_call("add_numbers", raw_text=_QWEN3_MALFORMED_BODY)
    assert tc.function.arguments == "{}"


def test_t3_chat_route_strips_wire_leak_from_content_and_reasoning():
    """End-to-end through the FULL chat route: when the model emits
    the qwen3 malformed wire AND the request is ``tool_choice=required``,
    the response MUST have:

      - ``tool_calls`` populated with the forced name
      - ``content`` clean of ``<tool_call>`` and ``</function>`` leaks
      - ``reasoning_content`` (if any) also clean of those leaks
    """

    class _QwenMalformedEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=_QWEN3_MALFORMED_BODY, raw_text=_QWEN3_MALFORMED_BODY)

    engine = _QwenMalformedEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is 4128 + 7591?"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    # Tool call is present (forced synth)
    assert msg.get("tool_calls"), (
        f"tool_choice=required must produce tool_calls; got msg={msg!r}"
    )
    assert msg["tool_calls"][0]["function"]["name"] == "add_numbers"
    # Content is leak-free — the original ``<tool_call>...`` text
    # must NOT survive into the user-visible content channel.
    content = msg.get("content") or ""
    for leak in ("<tool_call>", "</tool_call>", "</function>", "</parameter>"):
        assert leak not in content, (
            f"T3 leak: {leak!r} survived in content: {content!r}"
        )
    # And the reasoning content (if any) is also leak-free.
    reasoning = msg.get("reasoning_content") or ""
    for leak in ("<tool_call>", "</tool_call>", "</function>", "</parameter>"):
        assert leak not in reasoning, (
            f"T3 leak: {leak!r} survived in reasoning_content: {reasoning!r}"
        )


def test_t3_chat_route_scrubs_wire_leak_from_reasoning_content():
    """Forced malformed tool wire inside reasoning must not leak markers
    through ``reasoning_content``.
    """

    raw = (
        "<think>Need to call the tool.\n"
        '<tool_call>{"name":"add_numbers","arguments": 4128, 7591}</function>\n'
        "Done thinking.</think>"
    )

    class _ReasoningWireLeakEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _ReasoningWireLeakEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is 4128 + 7591?"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    assert msg["tool_calls"][0]["function"]["name"] == "add_numbers"

    reasoning = msg.get("reasoning_content") or ""
    for leak in ("<tool_call>", "</tool_call>", "</function>", "</parameter>"):
        assert leak not in reasoning, (
            f"T3 leak: {leak!r} survived in reasoning_content: {reasoning!r}"
        )
    assert "Need to call the tool." in reasoning
    assert "Done thinking." in reasoning


def test_t3_chat_route_recovers_args_when_possible():
    """When the model emits a recoverable arguments object inside an
    unclosed envelope, the synth call's arguments must reflect the
    recovered values, not ``"{}"``."""

    class _QwenRecoverableEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(
                text=_QWEN3_BODY_WITH_RECOVERABLE_ARGS,
                raw_text=_QWEN3_BODY_WITH_RECOVERABLE_ARGS,
            )

    engine = _QwenRecoverableEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is 4128 + 7591?"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    assert msg.get("tool_calls")
    import json as _json

    args = _json.loads(msg["tool_calls"][0]["function"]["arguments"])
    # The model's intended args ARE recoverable from the body — verify
    # the synth used them instead of defaulting to {}.
    assert args == {"a": 4128, "b": 7591}, (
        f"recoverable args were not used: got {args!r}"
    )
    # codex r3 BLOCKING #2: even on the recoverable shape, the
    # parser-wire markers in the raw body MUST NOT leak into the
    # user-visible fields. The recovered-args shape uses a
    # ``<tool_call>`` opener + ``</function>`` cross-family closer,
    # which exercises the phase-1.5 cross-family scrub specifically
    # (the phase-1 strict same-family pair would not match).
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    for leak in (
        "<tool_call>",
        "</tool_call>",
        "</function>",
        "</parameter>",
        '"name":',
        '"arguments":',
    ):
        assert leak not in content, (
            f"recoverable-args path leaked {leak!r} into content: {content!r}"
        )
        assert leak not in reasoning, (
            f"recoverable-args path leaked {leak!r} into reasoning: {reasoning!r}"
        )


# ──────────────────────────────────────────────────────────────────
# Regression: pre-existing happy paths must not regress
# ──────────────────────────────────────────────────────────────────


def test_pre_existing_hermes_prefix_unchanged():
    """``_forced_tool_call_prefix("hermes", ...)`` must keep emitting
    the same JSON-body shape it did before — pinned in
    ``test_chat_route_forced_tool_prefix.py`` but also asserted here
    so a future refactor doesn't silently change the wire format."""
    out = _forced_tool_call_prefix("hermes", "get_weather")
    assert out is not None
    assert out.startswith("<tool_call>")
    assert '"name": "get_weather"' in out
    assert out.endswith('"arguments": ')


def test_pre_existing_unknown_parser_still_none():
    """Defensive — unknown parsers still get ``None``."""
    assert _forced_tool_call_prefix(None, "fn") is None
    assert _forced_tool_call_prefix("future_parser", "fn") is None


@pytest.mark.parametrize(
    "parser_name",
    [
        "llama",
        "kimi",
        "glm47",
        "minimax",
        "mistral",
        "harmony",
        "gemma4",
    ],
)
def test_pre_existing_non_allowlisted_parsers_still_none(parser_name: str):
    """Channel-routed and other-shape parsers still fall through —
    the T2 fix is additive."""
    assert _forced_tool_call_prefix(parser_name, "fn") is None


# ──────────────────────────────────────────────────────────────────
# Codex r1 review-driven hardening — defense-in-depth + edge cases
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hostile_name",
    [
        "x<｜tool▁sep｜>injected",
        "x<｜tool▁call▁begin｜>injected",
        "x<｜tool▁call▁end｜>injected",
        "x<｜tool▁calls▁begin｜>injected",
        "x<｜tool▁calls▁end｜>injected",
    ],
)
def test_codex_blocking_2_deepseek_prefix_rejects_marker_injection(
    hostile_name: str,
):
    """Codex r1 BLOCKING #2 — a tool name carrying a DeepSeek envelope
    marker must NOT corrupt the wire. Upstream tool-name validation
    doesn't gate on the wire-token set, so the prefix builder is
    defense-in-depth: return ``None`` and let the synthesis fallback
    kick in (empty args, but no wire corruption)."""
    assert _forced_tool_call_prefix("deepseek_v31", hostile_name) is None
    assert _forced_tool_call_prefix("deepseek_r1_0528", hostile_name) is None


def test_codex_blocking_2_deepseek_prefix_safe_name_still_works():
    """Companion to the injection-rejection test: a clean tool name
    still produces the canonical prefix (regression guard)."""
    out = _forced_tool_call_prefix("deepseek_v31", "get_weather")
    assert out is not None
    assert "get_weather" in out


def test_codex_blocking_1_scrub_preserves_trailing_text_after_orphan_opener():
    """Codex r1 BLOCKING #1 — the previous regex pattern
    ``opener.*?(?:closer|\\Z)`` silently deleted all bytes after an
    orphan opener through EOF, losing legitimate reasoning prose. The
    two-phase scrub keeps trailing text intact while still stripping
    the opener marker itself."""
    leak = "Before opener\n<tool_call>\n{junk\nReal reasoning after the orphan opener."
    out = _scrub_tool_wire_literals(leak)
    # Wire opener gone
    assert "<tool_call>" not in out
    # But trailing prose preserved
    assert "Real reasoning after the orphan opener." in out
    # And so is the prose before it
    assert "Before opener" in out


def test_codex_blocking_1_scrub_handles_balanced_pair_normally():
    """Balanced ``<tool_call>...</tool_call>`` blocks are stripped
    whole — phase-1 of the scrub. Pin so a refactor doesn't
    accidentally break the common case."""
    leak = 'preamble <tool_call>{"name":"x","arguments":{}}</tool_call> trailing'
    out = _scrub_tool_wire_literals(leak)
    assert "<tool_call>" not in out
    assert "</tool_call>" not in out
    assert "preamble" in out
    assert "trailing" in out


def test_codex_blocking_1_scrub_handles_orphan_closer_alone():
    """Defensive: an orphan ``</tool_call>`` with no matching opener
    is also stripped — phase-2 of the scrub."""
    leak = "Some legitimate prose then a stray </tool_call> marker."
    out = _scrub_tool_wire_literals(leak)
    assert "</tool_call>" not in out
    assert "legitimate prose" in out
    assert "marker." in out


def test_codex_blocking_1_scrub_unclosed_deepseek_envelope_preserves_tail():
    """Same invariant for the DeepSeek envelope: an orphan opener
    must not erase trailing text."""
    leak = (
        "Pre-envelope words. <｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_weather"
        "<｜tool▁sep｜>{partial Followed by more analysis text."
    )
    out = _scrub_tool_wire_literals(leak)
    # Every wire marker gone
    for m in (
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
    ):
        assert m not in out, f"{m!r} survived: {out!r}"
    # Trailing prose preserved
    assert "more analysis text" in out
    assert "Pre-envelope words" in out


def test_codex_nit_recover_partial_args_finds_later_occurrence():
    """Codex r1 NIT — recovery must iterate over ``"arguments"`` so a
    leading example/prose mention doesn't block recovery of the real
    body. The leading mention has no balanced object after it;
    recovery skips it and uses the later real body."""
    raw = (
        'The tool schema example uses "arguments": (an integer here, not an object). '
        # The leading occurrence has NO object body — recovery must
        # skip it instead of returning None.
        '\nactual call:\n<tool_call>{"name":"x","arguments":{"a":42}}</tool_call>'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"a": 42}


def test_codex_nit_recover_partial_args_skips_unbalanced_then_finds_balanced():
    """Two ``"arguments":`` occurrences — the first has an unbalanced
    body (truncated), the second has a balanced one. Recovery must
    iterate past the unbalanced match instead of returning ``None``."""
    raw = (
        '{"name":"a","arguments":{"k":"unterminated  \nnext call: {"arguments":{"x":1}}'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"x": 1}


def test_codex_r2_blocking_prefers_in_wire_span_over_prose_example():
    """Codex r2 BLOCKING — when a docstring-style prose example
    appears BEFORE the actual tool wire, recovery must pick the
    wire-span occurrence, NOT the prose example. The previous
    "first-parseable" strategy would have shipped the example's
    arguments as the synthesised call's args."""
    raw = (
        # Prose example with valid JSON — this is what the previous
        # first-parseable strategy would have grabbed.
        'The tool schema requires "arguments": {"a": 0, "b": 0} '
        "shape (this is just an example).\n"
        # The actual call inside the wire span — this is what we want.
        '<tool_call>\n{"name":"add_numbers","arguments":{"a":42,"b":99}}\n'
        "</tool_call>"
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"a": 42, "b": 99}, (
        f"recovery picked the wrong occurrence (prose example over wire span); "
        f"got {parsed!r}"
    )


def test_codex_r2_blocking_falls_back_to_last_when_no_wire_span():
    """When NO ``"arguments":`` occurrence sits inside a wire-span
    opener, recovery falls back to the last (rightmost) parseable
    occurrence. Pins the bare-JSON tool emission case where the
    model didn't wrap the call at all."""
    raw = (
        # No wire markers — bare JSON only. The last balanced object
        # should win.
        'first: "arguments": {"a": 1}\nsecond: "arguments": {"a": 2}'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"a": 2}


def test_codex_r2_blocking_prefers_last_within_wire_span():
    """When MULTIPLE candidates sit inside wire spans, the LAST one
    wins (most-recent intent — the tool wire is conventionally at
    the end of the response)."""
    raw = (
        '<tool_call>{"name":"x","arguments":{"v":1}}</tool_call>'
        "\nThe model continued and emitted a second corrected call:\n"
        '<tool_call>{"name":"x","arguments":{"v":2}}</tool_call>'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"v": 2}


def test_codex_r3_blocking_1_scrub_strips_cross_family_qwen3_leak():
    """Codex r3 BLOCKING #1 — the qwen3 leak shape uses a
    ``<tool_call>`` opener with a ``</function>`` closer (mismatched
    families). The strict same-family pairs in phase 1 do not match
    that span, so the JSON body between the two markers used to
    survive into ``content``. Phase 1.5 (cross-family span) closes
    the leak."""
    leak = (
        '<tool_call>\n{"name":"add_numbers","arguments":{"a":1,"b":2}}\n</function>\n'
    )
    out = _scrub_tool_wire_literals(leak)
    # The whole span (including the JSON body) is gone — that body
    # is the malformed tool-call attempt, not legitimate prose.
    for leak_token in (
        "<tool_call>",
        "</function>",
        '"name":',
        '"arguments":',
        "add_numbers",
    ):
        assert leak_token not in out, (
            f"cross-family scrub left {leak_token!r} behind: {out!r}"
        )


def test_codex_r3_blocking_1_scrub_cross_family_preserves_pre_and_post_prose():
    """The cross-family scrub MUST NOT eat legitimate text before
    the opener or after the closer — only the wire span itself."""
    leak = (
        "Pre-wire reasoning. "
        '<tool_call>{"name":"x","arguments":{}}</function>'
        " Post-wire reasoning continues."
    )
    out = _scrub_tool_wire_literals(leak)
    assert "<tool_call>" not in out
    assert "</function>" not in out
    assert "Pre-wire reasoning" in out
    assert "Post-wire reasoning continues" in out


def test_codex_r3_blocking_1_scrub_orphan_opener_still_preserves_tail():
    """Cross-family scrub is non-greedy — it requires SOME closer
    to fire. An orphan opener with no closer anywhere in the text
    should NOT trigger phase 1.5 (regression for the r1 BLOCKING #1
    invariant that trailing prose past an orphan opener survives)."""
    leak = "<tool_call>\n{junk\nLegitimate trailing prose."
    out = _scrub_tool_wire_literals(leak)
    # Marker stripped by phase 2 (standalone).
    assert "<tool_call>" not in out
    # Trailing prose preserved (phase 1.5 didn't fire — no closer).
    assert "Legitimate trailing prose" in out


def test_codex_r4_blocking_1_recover_rejects_unrelated_tool_name():
    """Codex r4 BLOCKING #1 — when an ``expected_name`` is set, the
    recovery must NOT pick up an unrelated tool's args. A response
    containing ``{"name":"other_tool","arguments":{...}}`` must yield
    ``None`` for a synth target ``"my_target"``."""
    raw = '<tool_call>{"name":"other_tool","arguments":{"x":1}}</tool_call>'
    # Without expected_name: returns the args (pre-codex-r4 behaviour).
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    assert _json.loads(got) == {"x": 1}
    # WITH expected_name set to a non-matching tool: returns None.
    got = _recover_partial_tool_args(raw, expected_name="my_target")
    assert got is None, f"expected None when name mismatches; got {got!r}"


def test_codex_r4_blocking_1_recover_accepts_matching_tool_name():
    """The companion: when the expected name DOES match, recovery
    returns the args as before. Pin the success case so the
    name-pair gate doesn't accidentally reject correct calls."""
    raw = '<tool_call>{"name":"my_target","arguments":{"x":1}}</tool_call>'
    got = _recover_partial_tool_args(raw, expected_name="my_target")
    assert got is not None
    import json as _json

    assert _json.loads(got) == {"x": 1}


def test_codex_r4_blocking_1_recover_picks_matching_pair_from_multiple():
    """When MULTIPLE ``"arguments"`` occurrences exist, the
    ``expected_name`` gate must steer recovery to the one paired
    with the matching name even if another candidate appears later."""
    raw = (
        '<tool_call>{"name":"other","arguments":{"k":"first"}}</tool_call>'
        '<tool_call>{"name":"target","arguments":{"k":"correct"}}</tool_call>'
        '<tool_call>{"name":"third","arguments":{"k":"last"}}</tool_call>'
    )
    got = _recover_partial_tool_args(raw, expected_name="target")
    assert got is not None
    import json as _json

    assert _json.loads(got) == {"k": "correct"}


def test_codex_r4_blocking_1_synthesize_uses_name_paired_args():
    """End-to-end through ``_synthesize_forced_tool_call``: when
    raw_text contains the wrong tool's args plus the right one,
    the synth picks the right one because it passes its ``name``
    as the ``expected_name`` to recovery."""
    raw = (
        '<tool_call>{"name":"other","arguments":{"wrong":true}}</tool_call>'
        '<tool_call>{"name":"target","arguments":{"correct":true}}</tool_call>'
    )
    tc = _synthesize_forced_tool_call("target", raw_text=raw)
    import json as _json

    args = _json.loads(tc.function.arguments)
    assert args == {"correct": True}


def test_codex_r4_blocking_1_synthesize_falls_back_when_no_pair():
    """When raw_text contains NO ``"arguments"`` paired with the
    synth target name, the synth falls back to ``"{}"`` instead of
    silently picking up unrelated args."""
    raw = '<tool_call>{"name":"unrelated_tool","arguments":{"x":1}}</tool_call>'
    tc = _synthesize_forced_tool_call("target", raw_text=raw)
    assert tc.function.arguments == "{}"


def test_codex_r4_blocking_2_scrub_does_not_fire_for_tool_choice_auto():
    """Codex r4 BLOCKING #2 — the scrub must NOT fire for
    ``tool_choice="auto"`` because the model may legitimately emit
    prose discussing the tool wire format. End-to-end: ``auto`` +
    parser-extracted call + raw text containing wire-shaped prose
    keeps the prose in content."""

    # Engine returns text where the parser successfully extracts a
    # tool call AND there's wire-shaped content in the trailing prose
    # (e.g. the assistant explaining what just happened).
    PROSE_WITH_WIRE_LOOKING_TEXT = (
        '<tool_call>{"name":"get_time","arguments":{"city":"Tokyo"}}</tool_call>\n'
        "Note: tool calls use the <tool_call> envelope syntax."
    )

    class _AutoEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(
                text=PROSE_WITH_WIRE_LOOKING_TEXT,
                raw_text=PROSE_WITH_WIRE_LOOKING_TEXT,
            )

    engine = _AutoEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    cfg.enable_auto_tool_choice = True
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "time in Tokyo?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    assert msg.get("tool_calls"), "auto-mode parser-extracted call must survive"
    # The auto-mode scrub gate is OFF — the prose mention of
    # ``<tool_call>`` survives in content (legitimate model output).
    # This is the pre-0.8.3 contract; codex r4 BLOCKING #2 wants it
    # preserved so the broadened scrub doesn't strip legitimate
    # prose containing tool wire syntax under ``auto``.
    content = msg.get("content") or ""
    # The phrase "<tool_call>" in the trailing prose was legitimate
    # — the scrub must NOT have removed it under tool_choice=auto.
    assert "<tool_call>" in content, (
        f"auto-mode legitimate prose mentioning <tool_call> was stripped; "
        f"content={content!r}"
    )


def test_codex_r3_nit_recover_handles_pretty_print_whitespace():
    """Codex r3 NIT — the colon between ``"arguments"`` and ``{``
    may have arbitrary whitespace (newlines, deep indents,
    pretty-print). The previous fixed 20-char window rejected valid
    JSON with too much whitespace; the fix walks past whitespace
    unbounded before requiring ``:``."""
    raw = (
        '<tool_call>{"name": "x", "arguments"\n'
        "        \n"
        "                 :\n"
        '        {"deeply": {"nested": "values"}}}</tool_call>'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"deeply": {"nested": "values"}}


def test_codex_r2_blocking_deepseek_envelope_counts_as_wire_span():
    """DeepSeek's V3.1 envelope markers also qualify as wire spans —
    a prose mention before the envelope must not beat the real call
    inside.

    codex r5 BLOCKING #2 — exercise production behaviour by passing
    ``expected_name`` (the synth always supplies it via
    ``_synthesize_forced_tool_call``). The V3.1 wire shape uses
    ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>``, NOT JSON-quoted
    ``"name":"NAME"``, so the recovery's name-pair gate has to
    recognise the DeepSeek shape too.
    """
    raw = (
        'Documentation note: "arguments" usually look like {"city": "X"}.\n'
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
        '<｜tool▁sep｜>{"arguments":{"city":"Tokyo","units":"c"}}'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    # End-to-end via the synth helper (matches production).
    tc = _synthesize_forced_tool_call("get_weather", raw_text=raw)
    import json as _json

    args = _json.loads(tc.function.arguments)
    assert args == {"city": "Tokyo", "units": "c"}
    # Also exercise the direct recovery API with the expected name.
    got = _recover_partial_tool_args(raw, expected_name="get_weather")
    assert got is not None
    parsed = _json.loads(got)
    assert parsed == {"city": "Tokyo", "units": "c"}


def test_codex_r7_deepseek_verbose_span_pairs_name_without_fixed_lookback():
    """DeepSeek name pairing is bounded by the open wire span, not 512 bytes."""

    verbose_metadata = "x" * 900
    raw = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather"
        f"<｜tool▁sep｜>{verbose_metadata}"
        '{"arguments":{"city":"Tokyo","units":"c"}}'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    got = _recover_partial_tool_args(raw, expected_name="get_weather")
    assert got is not None
    import json as _json

    assert _json.loads(got) == {"city": "Tokyo", "units": "c"}


def test_codex_r8_no_opener_name_pair_does_not_cross_from_prose():
    """Without a wire opener, name pairing stays inside the local JSON object."""

    raw = (
        'Earlier prose says "name":"target". {"arguments":{"city":"Tokyo"}}</function>'
    )
    assert _recover_partial_tool_args(raw, expected_name="target") is None


# -----------------------------------------------------------------------
# codex r6 BLOCKING #1 — gate scrub on actual wire-leak detection
# -----------------------------------------------------------------------


def test_codex_r6_blocking_1_contains_wire_literal_detects_each_family():
    """Every marker registered in ``_TOOL_WIRE_STANDALONE_MARKERS`` must
    be recognised by ``_contains_tool_wire_literal``. The detector
    must NOT false-positive on plain prose (even prose with angle
    brackets or square brackets that look superficially similar).
    """
    leaky = [
        "OK <tool_call>",
        "OK </tool_call>",
        "OK <function=foo>",
        "OK </function>",
        "OK <｜tool▁calls▁begin｜>",
        "OK <｜tool▁sep｜>",
        "OK [TOOL_CALLS]",
        "OK <|python_tag|>",
    ]
    for s in leaky:
        assert _contains_tool_wire_literal(s), f"missed leak in {s!r}"
    clean = [
        "",
        None,
        "The weather in Tokyo is sunny.",
        "Use angle brackets like <p> for HTML.",
        "Lists in markdown look like [foo](bar).",
    ]
    for s in clean:
        assert not _contains_tool_wire_literal(s), f"false leak on {s!r}"


def test_codex_r6_blocking_1_forced_choice_with_clean_text_does_not_scrub():
    """A successful forced/required tool call whose ``cleaned_text``
    is plain prose must keep its content unmodified. Pre-fix, the
    scrub gate fired on every successful forced call regardless of
    whether the parser left wire markers behind.
    """

    raw = (
        '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":2}}</tool_call>'
        "\nTool called successfully."
    )

    class _CleanForcedEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _CleanForcedEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text

    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    assert msg["tool_calls"][0]["function"]["name"] == "add_numbers"
    assert msg.get("content") == "Tool called successfully."


def test_codex_r6_blocking_1_forced_choice_with_wire_leak_still_scrubs():
    """When the parser-extracted call leaves wire markers in
    ``cleaned_text`` (qwen3 cross-family leak), the scrub MUST still
    fire so the user-visible content doesn't carry junk.
    """
    leaky = "The result is OK. <tool_call>{}</function>"
    assert _contains_tool_wire_literal(leaky)
    cleaned = _scrub_tool_wire_literals(leaky)
    # The wire body is stripped; surrounding prose is preserved.
    assert "<tool_call>" not in cleaned
    assert "</function>" not in cleaned
    assert "The result is OK." in cleaned


def test_codex_r7_structural_leak_detector_ignores_plain_marker_mentions():
    """A literal marker token in ordinary prose is not by itself a
    parser-wire leak. The route scrub gate must require structure so
    forced synthesis does not destructively rewrite explanatory text.
    """

    prose = "I cannot call tools here, but the literal token is <tool_call>."
    assert _contains_tool_wire_literal(prose)
    assert not _contains_structural_tool_wire_leak(prose)
    closer_prose = "Use </function> to close XML in this example."
    assert _contains_tool_wire_literal(closer_prose)
    assert not _contains_structural_tool_wire_leak(closer_prose)
    balanced_prose = "Show the literal form <tool_call>...</tool_call>."
    assert _contains_tool_wire_literal(balanced_prose)
    assert not _contains_structural_tool_wire_leak(balanced_prose)
    json_example_prose = 'Mention <tool_call> near {"example": true}.'
    assert _contains_tool_wire_literal(json_example_prose)
    assert not _contains_structural_tool_wire_leak(json_example_prose)
    arguments_prose = 'The <tool_call> tag contains an "arguments" field.'
    assert _contains_tool_wire_literal(arguments_prose)
    assert not _contains_structural_tool_wire_leak(arguments_prose)
    assert _contains_structural_tool_wire_leak(
        '<tool_call>{"name":"add_numbers","arguments":{"a":1}}</function>'
    )


def test_codex_r10_visible_scrub_removes_unclosed_opener_payload_body():
    """A structural orphan opener must not leave name/arguments JSON behind."""

    out = _scrub_visible_tool_wire_leaks(
        'prefix <tool_call>{"name":"x","arguments":{"a":1}} suffix'
    )
    assert "<tool_call>" not in out
    assert '"name"' not in out
    assert '"arguments"' not in out
    assert "prefix" in out
    assert "suffix" in out


def test_codex_r12_visible_scrub_removes_unclosed_deepseek_prefix_payload():
    """DeepSeek begin/name/sep prefix must not survive payload scrubbing."""

    out = _scrub_visible_tool_wire_leaks(
        'prefix <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"arguments":{"city":"Tokyo"}} suffix'
    )
    assert "<｜tool▁call▁begin｜>" not in out
    assert "<｜tool▁sep｜>" not in out
    assert "get_weather" not in out
    assert '"arguments"' not in out
    assert "prefix" in out
    assert "suffix" in out


def test_codex_r11_broad_scrub_does_not_cross_family_strip_marker_prose():
    """Cross-family scrub needs payload evidence before deleting a span."""

    prose = "Explain <tool_call> as an opener and </function> as a closer."
    out = _scrub_tool_wire_literals(prose)
    assert "as an opener and" in out


def test_codex_r7_synth_forced_clean_marker_prose_not_scrubbed():
    """When forced synthesis happens because the parser found no call,
    clean prose that merely mentions ``<tool_call>`` must survive.
    """

    raw = "I cannot call tools here, but the literal token is <tool_call>."

    class _CleanMarkerMentionEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _CleanMarkerMentionEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    assert msg.get("content") == raw


def test_codex_r7_synth_forced_clean_closer_prose_not_scrubbed():
    """A lone closer token in ordinary prose is also not enough to scrub."""

    raw = "Use </function> to close XML in this example."

    class _CleanCloserMentionEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _CleanCloserMentionEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    assert msg.get("content") == raw


def test_codex_r8_synth_forced_clean_balanced_marker_prose_not_scrubbed():
    """Balanced marker examples without payload hints are explanatory prose."""

    raw = "Show the literal form <tool_call>...</tool_call>."

    class _CleanBalancedMentionEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _CleanBalancedMentionEngine()
    client = _make_client(engine, tool_call_parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    content = msg.get("content") or ""
    assert "Show the literal form" in content
    assert "<tool_call>" in content
    assert "..." in content


# -----------------------------------------------------------------------
# codex r6 BLOCKING #2 — don't scrub raw_text before reasoning extract
# -----------------------------------------------------------------------


def test_codex_r6_blocking_2_raw_text_not_mutated_before_reasoning():
    """The route keeps raw reasoning extraction but scrubs visible fields.

    This is the production contract: a reasoning parser may inspect the
    original raw text, but any wire markers it surfaces into
    ``content``/``reasoning_content`` must be removed before response
    construction.
    """

    raw = (
        "<think>Need a tool.\n"
        '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":2}}</function>\n'
        "Done.</think>"
    )

    class _RawEchoReasoningParser:
        def extract_reasoning(self, text, enable_thinking=None):
            return (
                "Reasoning saw "
                '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":2}}</function>',
                text,
            )

    class _RawWireEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _RawWireEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = _RawEchoReasoningParser()
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), msg
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    for leak in ("<tool_call>", "</tool_call>", "</function>", "</parameter>"):
        assert leak not in content
        assert leak not in reasoning


def test_codex_r9_raw_wire_elsewhere_does_not_scrub_reasoning_marker_example():
    """Raw structural wire elsewhere must not scrub a marker-only example."""

    raw = '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":2}}</function>'

    class _MarkerExampleReasoningParser:
        def extract_reasoning(self, text, enable_thinking=None):
            return "Use <tool_call> as the opening tag.", ""

    class _RawWireEngine(_RecordingEngine):
        def __init__(self):
            super().__init__(text=raw, raw_text=raw)

    engine = _RawWireEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = _MarkerExampleReasoningParser()
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "add 1 and 2"}],
            "tools": _SOLO_TOOL,
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    reasoning = resp.json()["choices"][0]["message"].get("reasoning_content") or ""
    assert reasoning == "Use <tool_call> as the opening tag."


# -----------------------------------------------------------------------
# codex r6 NIT — wire-span lookback bounded by nearest opener/closer
# -----------------------------------------------------------------------


def test_codex_r6_nit_wire_span_lookback_unbounded_by_distance():
    """A verbose wire body whose opener sits >256 bytes before the
    ``"arguments"`` marker must still count as in-span, so a later
    prose ``"arguments"`` example cannot beat it.
    """
    # 600+ bytes of metadata between opener and "arguments":.
    metadata = "x" * 600
    raw = (
        f"<tool_call>name=get_weather\n{metadata}\n"
        '"arguments": {"city":"Tokyo"}</tool_call>\n'
        'Then prose example: "arguments": {"city":"WRONG"}.'
    )
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    # The in-span (real) call MUST win over the later prose example,
    # despite the opener sitting well past the previous 256-byte
    # lookback boundary.
    assert parsed == {"city": "Tokyo"}


def test_codex_r6_nit_wire_span_lookback_respects_intervening_closer():
    """If a closer sits between the most recent opener and ``idx``,
    ``idx`` is NOT in the span — the bounded lookback must respect
    structural close events instead of just looking for any opener.
    """
    raw = (
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>\n'
        'Then a leading prose "arguments": {"x":99} (NOT in span).\n'
        '<tool_call>{"name":"b","arguments":{"x":2}}</tool_call>'
    )
    # Two in-span candidates; recovery picks the LAST in-span one.
    got = _recover_partial_tool_args(raw)
    assert got is not None
    import json as _json

    parsed = _json.loads(got)
    assert parsed == {"x": 2}
