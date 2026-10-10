# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming detokenizer optimization in scheduler."""

import pytest

pytest.importorskip("mlx")


from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
)

from tests.tokenizer_fixtures import byte_level_tokenizer

# These tests exercise real BPE encoding/decoding and mlx-lm's tokenizer loader,
# without requiring model weights, a populated host cache, or network access.
pytestmark = pytest.mark.requires_mlx


class TestStreamingDetokenizer:
    """Test streaming detokenizer correctness."""

    @pytest.fixture
    def qwen_tokenizer(self):
        """Build a byte-complete BPE tokenizer locally."""
        return byte_level_tokenizer()

    def test_naive_streaming_matches_batch(self, qwen_tokenizer):
        """Verify NaiveStreamingDetokenizer output matches batch decode."""
        text = "Hello, how are you doing today? I hope you're well!"
        tokens = qwen_tokenizer.encode(text)

        # Streaming decode
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()
        streaming_result = detok.text

        # Batch decode
        batch_result = qwen_tokenizer.decode(tokens)

        assert streaming_result == batch_result, (
            f"Streaming: {repr(streaming_result)}\nBatch: {repr(batch_result)}"
        )

    def test_last_segment_incremental(self, qwen_tokenizer):
        """Verify last_segment returns only new text."""
        text = "Hello world"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()

        segments = []
        for t in tokens:
            detok.add_token(t)
            seg = detok.last_segment
            if seg:
                segments.append(seg)

        detok.finalize()

        # Concatenated segments should equal full text
        full_from_segments = "".join(segments) + detok.last_segment
        assert full_from_segments == detok.text

    def test_reset_clears_state(self, qwen_tokenizer):
        """Verify reset() clears all state."""
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)

        # Add some tokens
        tokens = qwen_tokenizer.encode("Hello")
        for t in tokens:
            detok.add_token(t)

        # Reset
        detok.reset()

        # State should be cleared
        assert detok.tokens == []
        assert detok.offset == 0

    def test_unicode_handling(self, qwen_tokenizer):
        """Test handling of unicode characters."""
        text = "Hello 你好 مرحبا 🎉"
        tokens = qwen_tokenizer.encode(text)

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result

    def test_empty_input(self, qwen_tokenizer):
        """Test with empty input."""
        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        detok.finalize()

        assert detok.text == ""
        assert detok.tokens == []

    def test_single_token(self, qwen_tokenizer):
        """Test with single token."""
        tokens = [qwen_tokenizer.encode("Hi")[0]]

        detok = NaiveStreamingDetokenizer(qwen_tokenizer)
        detok.reset()
        detok.add_token(tokens[0])
        detok.finalize()

        batch_result = qwen_tokenizer.decode(tokens)
        assert detok.text == batch_result


class TestSchedulerDetokenizer:
    """Test scheduler's detokenizer integration."""

    @pytest.fixture
    def scheduler_mock(self):
        """Create a mock scheduler with detokenizer pool."""

        class MockScheduler:
            def __init__(self):
                self.tokenizer = byte_level_tokenizer()
                self._detokenizer_pool = {}

            def _get_detokenizer(self, request_id):
                if request_id not in self._detokenizer_pool:
                    detok = NaiveStreamingDetokenizer(self.tokenizer)
                    detok.reset()
                    self._detokenizer_pool[request_id] = detok
                return self._detokenizer_pool[request_id]

            def _cleanup_detokenizer(self, request_id):
                if request_id in self._detokenizer_pool:
                    del self._detokenizer_pool[request_id]

        return MockScheduler()

    def test_detokenizer_pool_creation(self, scheduler_mock):
        """Test that detokenizers are created on demand."""
        assert len(scheduler_mock._detokenizer_pool) == 0

        detok1 = scheduler_mock._get_detokenizer("req1")
        assert len(scheduler_mock._detokenizer_pool) == 1

        scheduler_mock._get_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 2

        # Same request ID returns same detokenizer
        detok1_again = scheduler_mock._get_detokenizer("req1")
        assert detok1 is detok1_again

    def test_detokenizer_cleanup(self, scheduler_mock):
        """Test that cleanup removes detokenizers."""
        scheduler_mock._get_detokenizer("req1")
        scheduler_mock._get_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 2

        scheduler_mock._cleanup_detokenizer("req1")
        assert len(scheduler_mock._detokenizer_pool) == 1
        assert "req1" not in scheduler_mock._detokenizer_pool

        scheduler_mock._cleanup_detokenizer("req2")
        assert len(scheduler_mock._detokenizer_pool) == 0

    def test_cleanup_nonexistent_is_safe(self, scheduler_mock):
        """Test that cleaning up nonexistent request doesn't raise."""
        scheduler_mock._cleanup_detokenizer("nonexistent")  # Should not raise

    def test_multiple_requests_independent(self, scheduler_mock):
        """Test that multiple requests have independent detokenizers."""
        detok1 = scheduler_mock._get_detokenizer("req1")
        detok2 = scheduler_mock._get_detokenizer("req2")

        # Add different tokens to each
        tokens1 = scheduler_mock.tokenizer.encode("Hello")
        tokens2 = scheduler_mock.tokenizer.encode("Goodbye")

        for t in tokens1:
            detok1.add_token(t)
        for t in tokens2:
            detok2.add_token(t)

        detok1.finalize()
        detok2.finalize()

        # Results should be independent
        assert "Hello" in detok1.text
        assert "Goodbye" in detok2.text
        assert detok1.text != detok2.text


class TestOptimizedDetokenizer:
    """Test that optimized detokenizer is used when available."""

    @pytest.fixture
    def tokenizer_wrapper(self, tmp_path):
        """Exercise mlx-lm's actual decoder detection without loading weights."""
        from mlx_lm.tokenizer_utils import load

        byte_level_tokenizer().save_pretrained(tmp_path)
        return load(tmp_path)

    def test_tokenizer_wrapper_has_optimized_detokenizer(self, tokenizer_wrapper):
        """Verify TokenizerWrapper has optimized detokenizer class."""
        assert hasattr(tokenizer_wrapper, "_detokenizer_class")
        assert hasattr(tokenizer_wrapper, "detokenizer")
        # Qwen uses BPE tokenizer
        assert tokenizer_wrapper._detokenizer_class == BPEStreamingDetokenizer

    @pytest.mark.parametrize(
        "text", ["Hello, how are you doing today?", "Test message 你好 مرحبا 🎉"]
    )
    def test_optimized_detokenizer_correctness(self, tokenizer_wrapper, text):
        """Verify optimized detokenizer produces correct output."""
        raw_tokenizer = tokenizer_wrapper._tokenizer
        tokens = raw_tokenizer.encode(text)
        assert any(
            len(piece) > 1 for piece in raw_tokenizer.convert_ids_to_tokens(tokens)
        )

        # Use optimized detokenizer
        detok = tokenizer_wrapper.detokenizer
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        # Compare with batch decode
        batch_result = raw_tokenizer.decode(tokens)
        assert batch_result == text
        assert detok.text == batch_result

    def test_scheduler_uses_optimized_detokenizer(self, tokenizer_wrapper):
        """Test that scheduler-like code uses optimized detokenizer."""
        # Simulate scheduler's _get_detokenizer logic
        _detokenizer_pool = {}

        def _get_detokenizer(tokenizer, request_id):
            if request_id not in _detokenizer_pool:
                if hasattr(tokenizer, "detokenizer"):
                    detok = tokenizer.detokenizer
                else:
                    detok = NaiveStreamingDetokenizer(tokenizer)
                detok.reset()
                _detokenizer_pool[request_id] = detok
            return _detokenizer_pool[request_id]

        # Get detokenizer - should be BPEStreamingDetokenizer
        detok = _get_detokenizer(tokenizer_wrapper, "test_req")
        assert isinstance(detok, BPEStreamingDetokenizer)

        # Verify it works correctly
        raw_tokenizer = tokenizer_wrapper._tokenizer
        tokens = raw_tokenizer.encode("Test message")
        for t in tokens:
            detok.add_token(t)
        detok.finalize()
        assert detok.text == raw_tokenizer.decode(tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
