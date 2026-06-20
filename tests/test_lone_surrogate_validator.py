# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-130 + F-131 (lone-surrogate input validator).

Background
----------
``json.loads`` accepts ``"\\uD800"`` as a valid JSON string and binds it
to a Python ``str`` carrying the unpaired surrogate codepoint U+D800.
HuggingFace ``tokenizers`` then raises
``TypeError: TextEncodeInput must be Union[TextInputSequence, …]`` deep
inside the chat-template render, producing two distinct failure shapes
depending on the lane:

  * **F-130 (non-stream)** — HTTP 500 with the generic
    ``"Internal server error"`` body. Tokenizer crash class.
  * **F-131 (stream)** — HTTP 200 followed by an SSE ``data:`` chunk
    carrying the raw Python ``TypeError`` text *and* the exception
    class name. Status-code-then-content-type contract violation plus
    info disclosure (HuggingFace-library fingerprinting).

The route-layer scanner in ``_scan_messages_for_lone_surrogates`` runs
BEFORE the streaming branch opens its ``StreamingResponse``, so both
lanes return a clean 400 with a precise offset before any expensive
work (tokenizer / engine / SSE generator) is invoked.

These tests pin the validator on every message slot a client can
populate and on the must-accept side-cases (paired surrogate emoji,
control-code ASCII) so a future tokenizer / pydantic schema change
cannot silently weaken the gate.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.service.helpers import (
    _find_lone_surrogate,
    _scan_messages_for_lone_surrogates,
)
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Unit: the codepoint scanner itself
# ---------------------------------------------------------------------------


class TestFindLoneSurrogate:
    @pytest.mark.parametrize(
        "s,expected_offset",
        [
            ("\uD800", 0),  # bare lone high
            ("\uDFFF", 0),  # bare lone low (other end of range)
            ("\uDC00", 0),  # bare lone low (lower end)
            ("hello \uD800 world", 6),  # mid-string lone high
            ("abc\uDFFFxyz", 3),  # mid-string lone low
            ("\uD801\uD802", 0),  # two lone surrogates — first wins
        ],
    )
    def test_detects_lone_surrogate(self, s: str, expected_offset: int):
        assert _find_lone_surrogate(s) == expected_offset

    @pytest.mark.parametrize(
        "s",
        [
            "",  # empty
            "hello world",  # pure ASCII
            "café",  # latin-1
            "日本語",  # CJK (BMP, no surrogates)
            "abc\x00def",  # NUL — orthogonal to surrogates
            "abc\x07def",  # BEL — control code, accept
            "﻿",  # BOM
            "😀",  # paired emoji → single astral codepoint after json.loads
            "Hello 😀 World",  # paired emoji embedded
            "👨‍👩‍👧‍👦",  # ZWJ family emoji
        ],
    )
    def test_accepts_valid_unicode(self, s: str):
        # The scanner returns ``None`` when the string is encodable as
        # UTF-8 — including all paired surrogates (already coalesced
        # into astral codepoints by ``json.loads``).
        assert _find_lone_surrogate(s) is None
        # Sanity: the accepted strings really do round-trip through
        # UTF-8 — pins the contract the scanner exists to enforce.
        s.encode("utf-8")


# ---------------------------------------------------------------------------
# Unit: the message-walker (every slot)
# ---------------------------------------------------------------------------


def _msg(**kwargs) -> MagicMock:
    """Tiny stand-in for a ``Message`` pydantic instance — attribute
    access matches the real model surface but we don't need the full
    schema for the walker (it only reads ``content`` /
    ``tool_call_id`` / ``tool_calls`` / ``name``)."""
    m = MagicMock(spec=["content", "tool_call_id", "tool_calls", "name"])
    m.content = kwargs.get("content")
    m.tool_call_id = kwargs.get("tool_call_id")
    m.tool_calls = kwargs.get("tool_calls")
    m.name = kwargs.get("name")
    return m


class TestScanMessagesForLoneSurrogates:
    @pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
    def test_lone_surrogate_in_content_string(self, role):
        """F-130: bare lone surrogate in any role's content string."""
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates([_msg(content="hi \uD800 there")])
        assert ei.value.status_code == 400
        assert "lone surrogate" in ei.value.detail
        assert "U+D800" in ei.value.detail
        assert "messages[0].content" in ei.value.detail

    def test_lone_surrogate_in_tool_call_id(self):
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates(
                [_msg(content="ok", tool_call_id="\uDC00abc")]
            )
        assert ei.value.status_code == 400
        assert "messages[0].tool_call_id" in ei.value.detail

    def test_lone_surrogate_in_name(self):
        """``name`` is not declared on our Message model today, but the
        walker still scans it via raw-dict access so future schema
        widening doesn't quietly drop the gate."""
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates(
                [{"role": "user", "content": "ok", "name": "u_\uD800"}]
            )
        assert ei.value.status_code == 400
        assert "messages[0].name" in ei.value.detail

    def test_lone_surrogate_in_tool_calls_arguments(self):
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates(
                [
                    _msg(
                        content="ok",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"\uD801"}',
                                },
                            }
                        ],
                    )
                ]
            )
        assert ei.value.status_code == 400
        assert "messages[0].tool_calls" in ei.value.detail

    def test_lone_surrogate_in_multimodal_text_part(self):
        """``content`` as ``list[ContentPart]``: the text part of a
        multimodal message must be scanned recursively, otherwise the
        gate is bypassed by VLM clients."""
        from vllm_mlx.api.models import ContentPart

        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates(
                [_msg(content=[ContentPart(type="text", text="bad \uD800")])]
            )
        assert ei.value.status_code == 400
        assert "messages[0].content" in ei.value.detail

    def test_lone_surrogate_in_multimodal_dict_form(self):
        """Same as above but ``content`` is a list of dicts — the form
        that bypasses ContentPart validation on permissive client
        envelopes."""
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates(
                [_msg(content=[{"type": "text", "text": "bad \uDC00"}])]
            )
        assert ei.value.status_code == 400

    def test_mid_string_offset_is_precise(self):
        """The error message must surface the exact offset so a client
        can locate the bad codepoint without re-scanning client-side."""
        with pytest.raises(HTTPException) as ei:
            _scan_messages_for_lone_surrogates([_msg(content="0123456\uD800tail")])
        assert "at offset 7" in ei.value.detail

    # ---- accept paths -------------------------------------------------------

    def test_paired_surrogate_emoji_accepted(self):
        """``"\\uD83D\\uDE00"`` (JSON) → coalesced into U+1F600 😀 by
        ``json.loads``. Must NOT trigger the gate; this is the
        regression check that the validator hasn't over-rejected
        valid astral codepoints."""
        # Simulate ``json.loads`` round-trip — the wire form of an
        # emoji embeds two surrogates that collapse to a single
        # codepoint in Python ``str``.
        s = json.loads('"hi \\uD83D\\uDE00 there"')
        assert len(s) == len("hi ") + 1 + len(" there")
        _scan_messages_for_lone_surrogates([_msg(content=s)])  # no raise

    def test_control_codes_accepted(self):
        """Control codes (NUL/BEL/VT) are an orthogonal class — out of
        scope for this PR (see F-134). The validator must not
        accidentally swallow them while reaching for surrogates."""
        _scan_messages_for_lone_surrogates([_msg(content="abc\x07def")])  # no raise

    def test_empty_messages_list_noop(self):
        """Empty list is a no-op — the role-validation block in the
        chat route runs first and 400s on its own, but the scanner
        must not crash on the edge case."""
        _scan_messages_for_lone_surrogates([])  # no raise

    def test_dict_message_form_walks(self):
        """When a caller hands raw dicts (e.g. cloud-routing path), the
        walker must still recurse — attribute-AND-dict access pattern."""
        with pytest.raises(HTTPException):
            _scan_messages_for_lone_surrogates(
                [{"role": "user", "content": "hi \uD800"}]
            )


# ---------------------------------------------------------------------------
# Integration: the chat route returns 400 (not 500 / not SSE-200) on both
# the non-stream and stream lanes BEFORE any engine work is invoked
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_config():
    """Patch select fields on the global cfg singleton and restore on exit."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


def _build_chat_app(patch_cfg, monkeypatch):
    from vllm_mlx.routes import chat as chat_route

    app = FastAPI()
    app.include_router(chat_route.router)

    engine = MagicMock()
    engine.is_mllm = False
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )

    # If the validator passes, the route would try to call the mocked
    # engine and fail downstream — but we only check the 400 path here,
    # so ``raise_server_exceptions=False`` keeps the test deterministic.
    monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)
    return TestClient(app, raise_server_exceptions=False)


class TestChatRouteLoneSurrogate:
    def test_non_stream_returns_400_before_engine_invoked(
        self, patched_config, monkeypatch
    ):
        """F-130: a bare lone surrogate in the user message returns
        400, NOT 500, and NEVER reaches the engine."""
        from vllm_mlx.routes import chat as chat_route

        # Trip-wire: if the engine is ever invoked, the test fails.
        def _explode(*_a, **_kw):
            raise AssertionError(
                "engine should never be invoked when input has a lone surrogate"
            )

        client = _build_chat_app(patched_config, monkeypatch)
        engine = chat_route.get_engine("stub-model")
        engine.chat = _explode
        engine.stream_chat = _explode

        # ``httpx``'s ``json=`` arg refuses to serialize lone surrogates
        # (it does ``json_dumps(..., ensure_ascii=False).encode("utf-8")``,
        # which raises the same ``UnicodeEncodeError`` the route is
        # defending against). Build the raw JSON body ourselves so the
        # surrogate survives the wire as the ``\\uD800`` escape exactly
        # as a malicious client would send it.
        raw = b'{"model":"stub-model","messages":[{"role":"user","content":"\\uD800"}],"max_tokens":5}'
        r = client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400, r.text
        body = r.json()
        # OpenAI-style error envelope is the project convention; the
        # detail message must surface the field and the codepoint.
        assert "lone surrogate" in json.dumps(body)
        assert "U+D800" in json.dumps(body)

    def test_stream_returns_400_before_sse_opens(
        self, patched_config, monkeypatch
    ):
        """F-131: stream=true with a lone surrogate must NOT open the
        SSE stream. The pre-fix behavior was HTTP 200 followed by a
        ``data:`` chunk carrying raw Python ``TypeError`` text — both
        the status code AND the body contract were violated."""
        from vllm_mlx.routes import chat as chat_route

        def _explode(*_a, **_kw):
            raise AssertionError(
                "engine should never be invoked when input has a lone surrogate"
            )

        client = _build_chat_app(patched_config, monkeypatch)
        engine = chat_route.get_engine("stub-model")
        engine.stream_chat = _explode
        engine.chat = _explode

        # See note on the non-stream test above — pass raw JSON bytes
        # so the lone surrogate survives the wire as the JSON escape.
        raw = (
            b'{"model":"stub-model","messages":[{"role":"user",'
            b'"content":"\\uD800"}],"max_tokens":5,"stream":true}'
        )
        r = client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        # Pre-fix: r.status_code == 200, body contained
        # ``"TextEncodeInput must be"`` and ``"TypeError"`` strings.
        assert r.status_code == 400, r.text
        # The response is plain JSON (HTTPException), NOT SSE. Be
        # explicit: a 400 with an SSE-style body would still be a
        # protocol smell, even if it carried no Python error.
        assert "text/event-stream" not in r.headers.get("content-type", "")
        body = r.text
        assert "lone surrogate" in body
        # F-131 leak guard: regardless of status code, no Python-class
        # exception names must appear in the response.
        assert "TypeError" not in body
        assert "TextEncodeInput" not in body
