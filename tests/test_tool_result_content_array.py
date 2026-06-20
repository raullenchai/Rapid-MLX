# SPDX-License-Identifier: Apache-2.0
"""F-111: ``role:"tool"`` content as OpenAI o1+ multipart array.

Pre-fix: a tool reply shipped as ``content: [{"type":"text","text":"X"}]``
(the default shape from the OpenAI o1/o3 SDK clients) was silently
dropped by every text-only chat template — the model received an
empty ``<tool_response>`` and hallucinated. Confirmed at the renderer
layer on Qwen3 (empty render) and Hermes3 (hard ``TypeError``); the
in-tree downstream pipeline also corrupted ``ContentPart`` instances
via ``json.dumps(default=str)`` inside
``_normalize_tool_call_arguments_for_template`` so the bug surfaced as
"ContentPart(type='text', text='4', ...)" in the rendered prompt
instead of "4".

Fix:
* ``vllm_mlx/routes/chat.py`` — accept ``str``, ``None`` or a text-only
  content-parts array on a ``tool`` role and 400 anything else (non-text
  parts would be silently dropped by the renderer = the F-111 footgun).
* ``vllm_mlx/api/utils.py::extract_multimodal_content`` — flatten the
  text-only array to a plain string at the API boundary so the
  ``json.dumps(default=str)`` hazard downstream never sees a pydantic
  ``ContentPart`` instance.
* ``vllm_mlx/utils/chat_template.py::_normalize_text_only_content_arrays``
  — defence-in-depth: any text-only content array reaching the
  template wrapper (engine tests, the speculative server, the gradio
  app) is flattened to a string before render.

These tests run BOTH layers — the route validator (HTTP path) and the
in-process normalizer (direct unit) — so a future regression on either
layer fails one specific test.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.models import ContentPart, Message
from vllm_mlx.api.utils import extract_multimodal_content
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.utils.chat_template import (
    _is_text_only_content_array,
    _join_text_parts,
    _normalize_text_only_content_arrays,
    apply_chat_template,
)

# ---------------------------------------------------------------------------
# Unit-level: the normalizer is the single source of truth
# ---------------------------------------------------------------------------


def test_text_only_array_detected_via_dict_parts() -> None:
    assert _is_text_only_content_array([{"type": "text", "text": "X"}]) is True
    assert (
        _is_text_only_content_array(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        )
        is True
    )


def test_text_only_array_detected_via_pydantic_parts() -> None:
    # The route hands the normalizer pydantic ``ContentPart`` instances
    # before pydantic's ``model_dump`` runs; we must accept both shapes.
    parts = [ContentPart(type="text", text="hello")]
    assert _is_text_only_content_array(parts) is True


def test_text_only_array_rejects_image_part() -> None:
    assert (
        _is_text_only_content_array(
            [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": "u"}]
        )
        is False
    )


def test_text_only_array_rejects_empty_list_and_non_list() -> None:
    assert _is_text_only_content_array([]) is False
    assert _is_text_only_content_array("plain") is False
    assert _is_text_only_content_array(None) is False


def test_join_text_parts_concatenates_verbatim() -> None:
    parts = [{"type": "text", "text": "alpha "}, {"type": "text", "text": "beta"}]
    assert _join_text_parts(parts) == "alpha beta"


def test_normalize_flattens_tool_text_array_to_string() -> None:
    msg = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [{"type": "text", "text": "4"}],
    }
    out = _normalize_text_only_content_arrays([msg])
    assert out[0]["content"] == "4"
    # original is untouched (no aliasing)
    assert msg["content"] == [{"type": "text", "text": "4"}]


def test_normalize_flattens_user_text_array_to_string() -> None:
    msg = {
        "role": "user",
        "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}],
    }
    out = _normalize_text_only_content_arrays([msg])
    assert out[0]["content"] == "hi there"


def test_normalize_preserves_string_content() -> None:
    msg = {"role": "user", "content": "plain"}
    out = _normalize_text_only_content_arrays([msg])
    assert out is not None
    assert out[0]["content"] == "plain"


def test_normalize_preserves_multimodal_content_on_user() -> None:
    # An image_url part is not a tool reply — the multimodal renderer
    # must still see the original list.
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "x.png"}},
        ],
    }
    out = _normalize_text_only_content_arrays([msg])
    assert out[0]["content"] == msg["content"]


def test_normalize_rejects_non_text_part_on_tool_role() -> None:
    bad = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [{"type": "image_url", "image_url": "x.png"}],
    }
    with pytest.raises(ValueError, match="tool-role"):
        _normalize_text_only_content_arrays([bad])


# ---------------------------------------------------------------------------
# Boundary: extract_multimodal_content flattens tool content arrays
# ---------------------------------------------------------------------------


def test_extract_multimodal_flattens_tool_array_to_string_native() -> None:
    """``preserve_native_format=True`` is the non-MLLM granite/qwen/hermes
    path; tool content must arrive at the engine as a plain string so
    ``_normalize_tool_call_arguments_for_template`` cannot
    ``json.dumps(default=str)`` it into a ``ContentPart(...)`` repr.
    """
    msgs = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        ),
        Message(
            role="tool",
            tool_call_id="c1",
            content=[ContentPart(type="text", text="payload")],
        ),
    ]
    processed, _, _ = extract_multimodal_content(msgs, preserve_native_format=True)
    tool_msg = processed[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "payload"
    # Critically: NOT a list, NOT a ``ContentPart(...)`` repr.
    assert isinstance(tool_msg["content"], str)


def test_extract_multimodal_flattens_tool_array_to_string_fallback() -> None:
    """``preserve_native_format=False`` is the prose-conversion fallback;
    the f-string-injected ``tool_content`` must NOT include a Pydantic
    repr (pre-fix it would emit ``[Tool Result (c1)]: [ContentPart(...)]``).
    """
    msgs = [
        Message(
            role="tool",
            tool_call_id="c1",
            content=[ContentPart(type="text", text="payload")],
        ),
    ]
    processed, _, _ = extract_multimodal_content(msgs, preserve_native_format=False)
    converted = processed[-1]
    assert converted["role"] == "user"
    assert converted["content"] == "[Tool Result (c1)]: payload"
    # No ContentPart repr leaked through.
    assert "ContentPart(" not in converted["content"]


# ---------------------------------------------------------------------------
# Renderer: apply_chat_template propagates a flat string into the prompt
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer that exposes the input messages so we can
    assert what the chat template sees.

    The normalization layer runs unconditionally before
    ``apply_chat_template`` is called; we capture that input.
    """

    chat_template = "x"
    all_special_tokens: list[str] = []
    additional_special_tokens: list[str] = []
    special_tokens_map: dict[str, str] = {}

    def __init__(self) -> None:
        self.seen: list[dict] | None = None

    def apply_chat_template(self, messages, **kwargs):
        # Capture and render a trivial concatenation so the test can
        # assert what the template actually got — string vs array.
        self.seen = list(messages)
        parts = []
        for m in messages:
            content = m.get("content", "")
            parts.append(f"{m.get('role')}:{content}")
        return "\n".join(parts) + "\nassistant:"


def test_apply_chat_template_normalizes_tool_array_to_string() -> None:
    tok = _FakeTokenizer()
    msgs = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "FOUR"}],
        },
    ]
    prompt = apply_chat_template(tok, msgs, model_name="test")
    assert tok.seen is not None
    seen_tool = tok.seen[-1]
    assert seen_tool["content"] == "FOUR"
    assert "FOUR" in prompt


def test_apply_chat_template_propagates_value_error_on_non_text_tool_part() -> None:
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="tool-role"):
        apply_chat_template(
            tok,
            [
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [{"type": "image_url", "image_url": "x.png"}],
                }
            ],
            model_name="test",
        )


# ---------------------------------------------------------------------------
# HTTP route: F-111 array form returns 200 (not silently dropped)
# ---------------------------------------------------------------------------


class _RecordingEngine:
    preserve_native_tool_format = True
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self) -> None:
        self.last_messages: Any = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _client() -> tuple[TestClient, _RecordingEngine]:
    cfg = reset_config()
    eng = _RecordingEngine()
    cfg.engine = eng
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app), eng


_TOOL = {
    "type": "function",
    "function": {"name": "calc", "parameters": {"type": "object"}},
}


def test_route_accepts_text_only_array_on_tool() -> None:
    client, eng = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [{"type": "text", "text": "RESULT_42"}],
                },
            ],
            "tools": [_TOOL],
        },
    )
    assert r.status_code == 200, r.text
    # Engine must have received the flattened string content — the
    # whole point of the fix is that the tool payload reaches the
    # template body, not gets dropped on the floor.
    seen = eng.last_messages
    tool_msg = next(m for m in seen if m["role"] == "tool")
    assert tool_msg["content"] == "RESULT_42"


def test_route_rejects_non_text_part_on_tool() -> None:
    client, _ = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [{"type": "image_url", "image_url": {"url": "u.png"}}],
                },
            ],
            "tools": [_TOOL],
        },
    )
    assert r.status_code == 400, r.text
    # Test client uses FastAPI's default error envelope
    # ({"detail": "..."}); the prod server wraps that as
    # {"error": {"message": "..."}} via the exception handler in
    # ``server.py``. Look at both.
    body = r.json()
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "text-only" in detail


def test_route_accepts_string_tool_content_baseline() -> None:
    """Regression sanity — the legacy string form must keep working."""
    client, eng = _client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "BASELINE_42"},
            ],
            "tools": [_TOOL],
        },
    )
    assert r.status_code == 200, r.text
    seen = eng.last_messages
    tool_msg = next(m for m in seen if m["role"] == "tool")
    assert tool_msg["content"] == "BASELINE_42"
