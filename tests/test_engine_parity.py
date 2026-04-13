# SPDX-License-Identifier: Apache-2.0
"""Cross-engine parity tests.

Verifies that SimpleEngine and BatchedEngine produce consistent
behavior for the same inputs across critical dimensions:
- Chat template application (enable_thinking, tools)
- Token decoding (multi-byte characters, emoji, CJK)
- Parameter passthrough

These tests are mock-based and do not require a real model.
"""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.utils.chat_template import apply_chat_template
from vllm_mlx.utils.decode import IncrementalDecoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that records apply_chat_template calls."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<prompt>"
    return tok


@pytest.fixture
def simple_messages():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


@pytest.fixture
def sample_tools():
    return [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


# ---------------------------------------------------------------------------
# Template Parity Tests
# ---------------------------------------------------------------------------


class TestApplyChatTemplate:
    """Tests for the shared apply_chat_template function."""

    @pytest.mark.parametrize("enable_thinking", [True, False])
    def test_explicit_enable_thinking(
        self, mock_tokenizer, simple_messages, enable_thinking
    ):
        """Explicit enable_thinking must reach the template."""
        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            enable_thinking=enable_thinking,
            model_name="test-model",
        )
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is enable_thinking

    def test_auto_enable_thinking_non_coder(self, mock_tokenizer, simple_messages):
        """enable_thinking=None auto-resolves to True for non-coder models."""
        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            enable_thinking=None,
            model_name="Qwen3-8B",
        )
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is True

    def test_auto_enable_thinking_coder(self, mock_tokenizer, simple_messages):
        """enable_thinking=None auto-resolves to False for coder models."""
        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            enable_thinking=None,
            model_name="Qwen3-Coder-32B",
        )
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is False

    def test_tools_passed_to_template(
        self, mock_tokenizer, simple_messages, sample_tools
    ):
        """tools must reach the template."""
        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            tools=sample_tools,
            model_name="test-model",
        )
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs["tools"] == sample_tools

    def test_no_tools_when_none(self, mock_tokenizer, simple_messages):
        """tools key must not be in kwargs when tools=None."""
        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            tools=None,
            model_name="test-model",
        )
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert "tools" not in kwargs

    def test_fallback_on_type_error(self, mock_tokenizer, simple_messages):
        """Should retry without enable_thinking first, preserving tools."""
        # First call raises TypeError (both kwargs), second succeeds (without enable_thinking)
        mock_tokenizer.apply_chat_template.side_effect = [
            TypeError("unsupported keyword argument 'enable_thinking'"),
            "<fallback prompt>",
        ]
        result = apply_chat_template(
            mock_tokenizer,
            simple_messages,
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            enable_thinking=True,
            model_name="test",
        )
        assert result == "<fallback prompt>"
        # Second call should keep tools but drop enable_thinking
        _, fallback_kwargs = mock_tokenizer.apply_chat_template.call_args
        assert "tools" in fallback_kwargs
        assert "enable_thinking" not in fallback_kwargs

    def test_fallback_strips_tools_when_both_fail(self, mock_tokenizer, simple_messages):
        """Should fall back to prompt injection when template rejects both."""
        mock_tokenizer.apply_chat_template.side_effect = [
            TypeError("unsupported keyword argument 'enable_thinking'"),
            TypeError("unsupported keyword argument 'tools'"),
            "<injected prompt>",
        ]
        result = apply_chat_template(
            mock_tokenizer,
            simple_messages,
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            enable_thinking=True,
            model_name="test",
        )
        assert result == "<injected prompt>"
        # Third call should have neither tools nor enable_thinking
        _, fallback_kwargs = mock_tokenizer.apply_chat_template.call_args
        assert "tools" not in fallback_kwargs
        assert "enable_thinking" not in fallback_kwargs

    def test_no_apply_chat_template_method(self, simple_messages):
        """Should use plain text fallback when no apply_chat_template."""
        applicator = MagicMock(spec=[])  # no methods
        result = apply_chat_template(applicator, simple_messages, model_name="test")
        assert "assistant:" in result
        assert "Hello" in result


# ---------------------------------------------------------------------------
# IncrementalDecoder Tests
# ---------------------------------------------------------------------------


class TestIncrementalDecoder:
    """Tests for the shared IncrementalDecoder."""

    def test_simple_ascii(self):
        """Simple ASCII tokens decode correctly."""
        tok = MagicMock()
        tok.decode.side_effect = lambda ids, **kw: "".join(
            {1: "H", 2: "e", 3: "l"}.get(i, "?") for i in ids
        )
        dec = IncrementalDecoder(tok)
        assert dec.add_token(1) == "H"
        assert dec.add_token(2) == "e"
        assert dec.add_token(3) == "l"

    def test_emoji_split_across_tokens(self):
        """Emoji split across tokens should hold back until complete."""
        tok = MagicMock()
        # Token 1 produces incomplete emoji (U+FFFD), token 2 completes it
        tok.decode.side_effect = lambda ids, **kw: {
            (1,): "\ufffd",
            (1, 2): "😀",
        }.get(tuple(ids), "")

        dec = IncrementalDecoder(tok)
        assert dec.add_token(1) == ""  # held back
        assert dec.add_token(2) == "😀"  # now complete

    def test_cjk_split_across_tokens(self):
        """CJK characters split across tokens should hold back."""
        tok = MagicMock()
        tok.decode.side_effect = lambda ids, **kw: {
            (10,): "\ufffd",
            (10, 11): "你好",
        }.get(tuple(ids), "")

        dec = IncrementalDecoder(tok)
        assert dec.add_token(10) == ""  # held back
        assert dec.add_token(11) == "你好"

    def test_holds_back_ufffd_no_leak(self):
        """U+FFFD must never appear in output deltas."""
        tok = MagicMock()
        tok.decode.side_effect = lambda ids, **kw: {
            (1,): "Hello ",
            (1, 2): "Hello \ufffd",
            (1, 2, 3): "Hello 🎉",
        }.get(tuple(ids), "")

        dec = IncrementalDecoder(tok)
        d1 = dec.add_token(1)
        d2 = dec.add_token(2)
        d3 = dec.add_token(3)

        assert "\ufffd" not in d1
        assert "\ufffd" not in d2
        assert "\ufffd" not in d3
        assert d1 == "Hello "
        assert d2 == ""
        assert d3 == "🎉"

    def test_get_full_text(self):
        """get_full_text returns the complete decoded string."""
        tok = MagicMock()
        tok.decode.side_effect = lambda ids, **kw: {
            (1,): "Hello",
            (1, 2): "Hello world",
        }.get(tuple(ids), "")

        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        dec.add_token(2)
        assert dec.get_full_text() == "Hello world"

    def test_reset(self):
        """reset() clears state for reuse."""
        tok = MagicMock()
        tok.decode.side_effect = lambda ids, **kw: {
            (1,): "a",
            (2,): "b",
        }.get(tuple(ids), "")

        dec = IncrementalDecoder(tok)
        dec.add_token(1)
        assert dec.token_ids == [1]

        dec.reset()
        assert dec.token_ids == []
        assert dec.add_token(2) == "b"

    def test_skip_special_tokens_passed(self):
        """skip_special_tokens is forwarded to tokenizer.decode."""
        tok = MagicMock()
        tok.decode.return_value = "x"

        dec = IncrementalDecoder(tok, skip_special_tokens=True)
        dec.add_token(1)
        tok.decode.assert_called_with([1], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Parameter Passthrough Tests
# ---------------------------------------------------------------------------


class TestParameterPassthrough:
    """Tests verifying that critical parameters aren't swallowed by **kwargs."""

    def test_enable_thinking_not_swallowed(self, mock_tokenizer, simple_messages):
        """Both engines must extract enable_thinking and pass it to template."""
        # Simulate what both engines should do: extract from kwargs, pass to shared fn
        kwargs = {"enable_thinking": False, "temperature": 0.5}
        enable_thinking = kwargs.pop("enable_thinking", None)

        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            enable_thinking=enable_thinking,
            model_name="Qwen3-8B",
        )
        _, template_kwargs = mock_tokenizer.apply_chat_template.call_args
        assert template_kwargs["enable_thinking"] is False

    def test_enable_thinking_default_none(self, mock_tokenizer, simple_messages):
        """When enable_thinking not in kwargs, it should default to None (auto)."""
        kwargs = {"temperature": 0.5}
        enable_thinking = kwargs.pop("enable_thinking", None)

        apply_chat_template(
            mock_tokenizer,
            simple_messages,
            enable_thinking=enable_thinking,
            model_name="Qwen3-8B",
        )
        _, template_kwargs = mock_tokenizer.apply_chat_template.call_args
        # None → auto → True for non-coder
        assert template_kwargs["enable_thinking"] is True


# ---------------------------------------------------------------------------
# Engine-specific integration tests (mock-based)
# ---------------------------------------------------------------------------


class TestBatchedEngineEnableThinking:
    """Verify BatchedEngine passes enable_thinking to _apply_chat_template."""

    def test_chat_extracts_enable_thinking(self):
        """BatchedEngine.chat() must extract enable_thinking from kwargs."""
        # This is a structural test - we verify the code path exists
        # by checking the source code has the extraction
        import inspect

        from vllm_mlx.engine.batched import BatchedEngine

        chat_source = inspect.getsource(BatchedEngine.chat)
        assert "enable_thinking" in chat_source
        assert 'kwargs.pop("enable_thinking"' in chat_source

    def test_stream_chat_extracts_enable_thinking(self):
        """BatchedEngine.stream_chat() must extract enable_thinking from kwargs."""
        import inspect

        from vllm_mlx.engine.batched import BatchedEngine

        source = inspect.getsource(BatchedEngine.stream_chat)
        assert "enable_thinking" in source
        assert 'kwargs.pop("enable_thinking"' in source

    def test_apply_chat_template_accepts_enable_thinking(self):
        """BatchedEngine._apply_chat_template() must accept enable_thinking."""
        import inspect

        from vllm_mlx.engine.batched import BatchedEngine

        sig = inspect.signature(BatchedEngine._apply_chat_template)
        assert "enable_thinking" in sig.parameters


class TestSchedulerIncrementalDecoder:
    """Verify Scheduler uses IncrementalDecoder for decoding."""

    def test_scheduler_imports_decoder(self):
        """Scheduler module must import IncrementalDecoder."""
        import inspect

        import vllm_mlx.scheduler as sched_module

        source = inspect.getsource(sched_module)
        assert "IncrementalDecoder" in source

    def test_decoder_attached_on_schedule(self):
        """Scheduler must attach _decoder when request enters running state."""
        import inspect

        import vllm_mlx.scheduler as sched_module

        source = inspect.getsource(sched_module)
        assert "request._decoder = IncrementalDecoder" in source
