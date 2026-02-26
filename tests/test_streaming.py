# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming JSON encoder."""

import json

import pytest

from vllm_mlx.api.streaming import StreamingJSONEncoder, _escape_json_string


# ── _escape_json_string ──────────────────────────────────────────────────

class TestEscapeJsonString:
    def test_simple_string(self):
        assert _escape_json_string("hello") == "hello"

    def test_quotes(self):
        assert _escape_json_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash(self):
        assert _escape_json_string("a\\b") == "a\\\\b"

    def test_newline(self):
        assert _escape_json_string("line1\nline2") == "line1\\nline2"

    def test_tab(self):
        assert _escape_json_string("a\tb") == "a\\tb"

    def test_unicode(self):
        result = _escape_json_string("hello 世界")
        # json.dumps may or may not escape non-ASCII; either form is valid
        assert "hello" in result

    def test_empty(self):
        assert _escape_json_string("") == ""


# ── StreamingJSONEncoder init ─────────────────────────────────────────────

class TestEncoderInit:
    def test_basic_init(self):
        enc = StreamingJSONEncoder("id-1", "model-1", "chat.completion.chunk", created=1000)
        assert enc.response_id == "id-1"
        assert enc.model == "model-1"
        assert enc.object_type == "chat.completion.chunk"
        assert enc.created == 1000

    def test_default_created(self):
        enc = StreamingJSONEncoder("id-1", "model-1", "chat.completion.chunk")
        assert isinstance(enc.created, int)
        assert enc.created > 0

    def test_prefix_contains_metadata(self):
        enc = StreamingJSONEncoder("id-1", "mymodel", "chat.completion.chunk", created=42)
        assert '"id":"id-1"' in enc._prefix
        assert '"model":"mymodel"' in enc._prefix
        assert '"created":42' in enc._prefix


# ── encode_chat_chunk ─────────────────────────────────────────────────────

class TestEncodeChatChunk:
    def _enc(self):
        return StreamingJSONEncoder("chatcmpl-test", "test-model", "chat.completion.chunk", created=100)

    def test_role_only(self):
        result = self._enc().encode_chat_chunk(role="assistant")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        data = json.loads(result[6:-2])  # strip "data: " and "\n\n"
        assert data["choices"][0]["delta"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] is None

    def test_content_only(self):
        result = self._enc().encode_chat_chunk(content="Hello")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_content_with_special_chars(self):
        result = self._enc().encode_chat_chunk(content='say "hi"\nnewline')
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"]["content"] == 'say "hi"\nnewline'

    def test_finish_reason(self):
        result = self._enc().encode_chat_chunk(finish_reason="stop")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["delta"] == {}

    def test_finish_reason_length(self):
        result = self._enc().encode_chat_chunk(finish_reason="length")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["finish_reason"] == "length"

    def test_role_and_content(self):
        result = self._enc().encode_chat_chunk(role="assistant", content="Hi")
        data = json.loads(result[6:-2])
        delta = data["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["content"] == "Hi"

    def test_with_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        result = self._enc().encode_chat_chunk(content="tok", usage=usage)
        data = json.loads(result[6:-2])
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["total_tokens"] == 15

    def test_empty_content(self):
        result = self._enc().encode_chat_chunk(content="")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"]["content"] == ""

    def test_metadata_fields(self):
        result = self._enc().encode_chat_chunk(content="x")
        data = json.loads(result[6:-2])
        assert data["id"] == "chatcmpl-test"
        assert data["model"] == "test-model"
        assert data["object"] == "chat.completion.chunk"
        assert data["created"] == 100


# ── encode_completion_chunk ───────────────────────────────────────────────

class TestEncodeCompletionChunk:
    def _enc(self):
        return StreamingJSONEncoder("cmpl-test", "test-model", "text_completion", created=200)

    def test_basic_text(self):
        result = self._enc().encode_completion_chunk(text="Hello")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["text"] == "Hello"
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["finish_reason"] is None

    def test_custom_index(self):
        result = self._enc().encode_completion_chunk(text="x", index=2)
        data = json.loads(result[6:-2])
        assert data["choices"][0]["index"] == 2

    def test_finish_reason(self):
        result = self._enc().encode_completion_chunk(text="", finish_reason="stop")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_with_usage(self):
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        result = self._enc().encode_completion_chunk(text="tok", usage=usage)
        data = json.loads(result[6:-2])
        assert data["usage"]["total_tokens"] == 8

    def test_special_chars_in_text(self):
        result = self._enc().encode_completion_chunk(text='line1\nline2\t"quoted"')
        data = json.loads(result[6:-2])
        assert data["choices"][0]["text"] == 'line1\nline2\t"quoted"'


# ── encode_done ───────────────────────────────────────────────────────────

class TestEncodeDone:
    def test_done_message(self):
        enc = StreamingJSONEncoder("id", "model", "type", created=0)
        assert enc.encode_done() == "data: [DONE]\n\n"


# ── Valid JSON output ─────────────────────────────────────────────────────

class TestValidJson:
    """Ensure all outputs are valid JSON (parseable)."""

    def test_many_chunks_are_valid_json(self):
        enc = StreamingJSONEncoder("id", "model", "chat.completion.chunk", created=0)
        chunks = [
            enc.encode_chat_chunk(role="assistant"),
            enc.encode_chat_chunk(content="Hello "),
            enc.encode_chat_chunk(content="world!"),
            enc.encode_chat_chunk(content=""),
            enc.encode_chat_chunk(finish_reason="stop"),
        ]
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            # Parse the JSON payload
            json.loads(chunk[6:-2])  # Should not raise
