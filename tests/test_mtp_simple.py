# SPDX-License-Identifier: Apache-2.0
"""Tests for MTP (Multi-Token Prediction) in SimpleEngine.

Uses mock model objects — no real model loading required.
"""

import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx


class MockCache:
    """Mock KVCache that supports trim and is_trimmable."""

    def __init__(self, offset=0, trimmable=True):
        self.offset = offset
        self._trimmable = trimmable
        self.state = [None]

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        self.offset = max(0, self.offset - n)

    def size(self):
        return self.offset


class _CopyableArray:
    """Wrapper around mx.array that supports .copy() for snapshot tests."""

    def __init__(self, arr):
        self._arr = arr

    def copy(self):
        return _CopyableArray(mx.array(self._arr))


class MockRNNCache:
    """Mock non-trimmable cache (like DeltaNet ArraysCache)."""

    def __init__(self, offset=0):
        self.offset = offset
        self.state = [_CopyableArray(mx.zeros((1, 4)))]

    def is_trimmable(self):
        return False

    def size(self):
        return self.offset


def _make_mock_model(accept=True, return_hidden=True):
    """Create a mock model that supports MTP operations.

    Args:
        accept: If True, verify argmax matches draft token.
        return_hidden: If True, model returns (logits, hidden) tuples.
    """
    model = MagicMock()

    # Token ID sequences for deterministic testing
    primary_id = 42
    draft_id = 99
    # If reject, verify disagrees with draft
    verify_primary_id = draft_id if accept else 55

    def mock_call(input_ids, cache=None, return_hidden=False):
        seq_len = input_ids.shape[-1]
        # Logits: make primary_id the argmax
        logits = mx.zeros((1, seq_len, 100))
        # Set primary_id position high
        logits = logits.at[:, :, primary_id].add(10.0)

        if seq_len == 2:
            # Verify pass: set position 0 to agree/disagree with draft
            verify_logits = mx.zeros((1, 2, 100))
            verify_logits = verify_logits.at[:, 0, verify_primary_id].add(10.0)
            verify_logits = verify_logits.at[:, 1, primary_id].add(10.0)
            logits = verify_logits

        if return_hidden:
            hidden = mx.ones((1, seq_len, 64))
            return logits, hidden
        return logits

    model.side_effect = mock_call
    model.__call__ = mock_call

    # mtp_forward: always predicts draft_id
    def mock_mtp_forward(hidden, token_ids, mtp_cache=None):
        draft_logits = mx.zeros((1, 1, 100))
        draft_logits = draft_logits.at[:, :, draft_id].add(10.0)
        return draft_logits

    model.mtp_forward = mock_mtp_forward

    # MTP validation attributes
    model.mtp = MagicMock()
    model.mtp.layers = [MagicMock()]
    model.make_mtp_cache = MagicMock(return_value=[])
    model.args = MagicMock()
    model.args.num_nextn_predict_layers = 1

    return model, primary_id, draft_id


def _make_mock_tokenizer(eos_token_id=0):
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = eos_token_id
    tokenizer.bos_token = None

    # Mock encode: return list of increasing integers
    def mock_encode(text, add_special_tokens=True):
        # Return a simple deterministic sequence
        return [1, 2, 3, 4, 5]

    tokenizer.encode = mock_encode

    # Mock decode for IncrementalDecoder
    tokenizer.decode = MagicMock(return_value="hello")

    # Mock _detokenizer_class for IncrementalDecoder
    mock_detok = MagicMock()
    mock_detok.return_value = mock_detok
    mock_detok.last_segment = "x"
    mock_detok.reset = MagicMock()
    mock_detok.add_token = MagicMock()
    mock_detok.finalize = MagicMock()
    mock_detok.text = "hello world"
    tokenizer._detokenizer_class = lambda tok: mock_detok

    return tokenizer


class TestMTPValidation(unittest.TestCase):
    """Test _validate_and_setup_mtp."""

    def test_validates_model_with_mtp(self):
        """Model with valid MTP head should validate."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        llm.model, _, _ = _make_mock_model()
        llm._loaded = True
        llm.enable_mtp = True

        with patch("vllm_mlx.patches.qwen3_next_mtp.validate_mtp_support", return_value=True):
            llm._validate_and_setup_mtp()

        self.assertTrue(llm._mtp_validated)

    def test_fails_validation_gracefully(self):
        """Model without MTP head should fail gracefully."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        llm.model = MagicMock()
        llm._loaded = True
        llm.enable_mtp = True

        with patch("vllm_mlx.patches.qwen3_next_mtp.validate_mtp_support", return_value=False):
            llm._validate_and_setup_mtp()

        self.assertFalse(llm._mtp_validated)

    def test_warns_when_both_draft_and_mtp(self):
        """Should warn when both draft model and MTP are set."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model", draft_model="some-draft")
        llm.model, _, _ = _make_mock_model()
        llm.draft_model = MagicMock()  # Simulate loaded draft model
        llm._loaded = True
        llm.enable_mtp = True

        with patch("vllm_mlx.patches.qwen3_next_mtp.validate_mtp_support", return_value=True):
            with self.assertLogs("vllm_mlx.models.llm", level="WARNING") as cm:
                llm._validate_and_setup_mtp()

        self.assertTrue(llm._mtp_validated)
        self.assertTrue(any("Both MTP and speculative" in msg for msg in cm.output))


class TestMTPAccept(unittest.TestCase):
    """Test MTP generation when verify accepts the draft."""

    def test_yields_two_tokens_per_step(self):
        """When draft is accepted, both primary and draft should be yielded."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        model, primary_id, draft_id = _make_mock_model(accept=True)
        llm.model = model
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=999)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = False

        # Set up cache
        cache = [MockCache(offset=5)]
        llm._prompt_cache = cache
        llm._main_cache_len = 1
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        # Collect tokens (limit to avoid infinite loop)
        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=4,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                break

        # Should have pairs of (primary, draft)
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0], primary_id)
        self.assertEqual(tokens[1], draft_id)


class TestMTPReject(unittest.TestCase):
    """Test MTP generation when verify rejects the draft."""

    def test_yields_one_token_on_reject(self):
        """When draft is rejected, only primary should be yielded per step."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        model, primary_id, draft_id = _make_mock_model(accept=False)
        llm.model = model
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=999)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = False

        cache = [MockCache(offset=5)]
        llm._prompt_cache = cache
        llm._main_cache_len = 1
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=3,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                break

        # Draft token (99) should NOT appear — only primaries
        self.assertTrue(draft_id not in tokens)
        self.assertEqual(len(tokens), 3)


class TestMTPRejectHybrid(unittest.TestCase):
    """Test MTP reject with hybrid cache (KV + RNN)."""

    def test_rnn_restored_on_reject(self):
        """Hybrid cache: RNN should be restored, KV trimmed by 2 on reject."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        model, primary_id, draft_id = _make_mock_model(accept=False)
        llm.model = model
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=999)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = False

        # Hybrid cache: one KV + one RNN
        kv_cache = MockCache(offset=5, trimmable=True)
        rnn_cache = MockRNNCache(offset=5)
        llm._prompt_cache = [kv_cache, rnn_cache]
        llm._main_cache_len = 2
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=2,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                break

        # Draft token (99) should NOT appear — only primaries
        self.assertTrue(draft_id not in tokens)


class TestMTPOptimistic(unittest.TestCase):
    """Test MTP optimistic mode (always accept)."""

    def test_always_yields_two_tokens(self):
        """Optimistic mode: always yields both primary and draft."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        # Use reject model but optimistic should still accept
        model, primary_id, draft_id = _make_mock_model(accept=False)
        llm.model = model
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=999)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = True

        cache = [MockCache(offset=5)]
        llm._prompt_cache = cache
        llm._main_cache_len = 1
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=4,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                break

        # Even with reject model, optimistic yields 2 per step
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0], primary_id)
        self.assertEqual(tokens[1], draft_id)


class TestMTPEOS(unittest.TestCase):
    """Test MTP stops on EOS from primary or draft."""

    def test_stops_on_primary_eos(self):
        """Generation stops when primary token is EOS."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        model, primary_id, draft_id = _make_mock_model(accept=True)
        llm.model = model
        # Set EOS to match primary token — will stop immediately
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=primary_id)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = False

        cache = [MockCache(offset=5)]
        llm._prompt_cache = cache
        llm._main_cache_len = 1
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=100,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                self.assertEqual(chunk.finish_reason, "stop")
                break

        # Should stop at first primary token (EOS)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], primary_id)

    def test_stops_on_draft_eos(self):
        """Generation stops when accepted draft token is EOS."""
        from vllm_mlx.models.llm import MLXLanguageModel

        llm = MLXLanguageModel("test-model")
        model, primary_id, draft_id = _make_mock_model(accept=True)
        llm.model = model
        # Set EOS to match draft token
        llm.tokenizer = _make_mock_tokenizer(eos_token_id=draft_id)
        llm._loaded = True
        llm.enable_mtp = True
        llm._mtp_validated = True
        llm.mtp_optimistic = False

        cache = [MockCache(offset=5)]
        llm._prompt_cache = cache
        llm._main_cache_len = 1
        llm._cached_token_ids = [1, 2, 3, 4, 5]

        tokens = []
        for chunk in llm._mtp_generate(
            suffix_tokens=[5],
            full_token_ids=[1, 2, 3, 4, 5],
            max_tokens=100,
            sampler=lambda lp: mx.argmax(lp, axis=-1),
            stop=None,
        ):
            tokens.append(chunk.token)
            if chunk.finished:
                self.assertEqual(chunk.finish_reason, "stop")
                break

        # Should yield primary then stop at draft (EOS)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], primary_id)
        self.assertEqual(tokens[1], draft_id)


class TestMTPFlagPassthrough(unittest.TestCase):
    """Test that MTP flags flow from CLI → server → engine → model."""

    def test_simple_engine_passes_flags(self):
        """SimpleEngine should pass MTP flags to MLXLanguageModel."""
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(
            model_name="test-model",
            enable_mtp=True,
            mtp_optimistic=True,
        )

        self.assertTrue(engine._enable_mtp)
        self.assertTrue(engine._mtp_optimistic)

    def test_server_load_model_passes_flags(self):
        """load_model should pass MTP flags to SimpleEngine."""
        with patch("vllm_mlx.server.SimpleEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_engine.is_mllm = False
            mock_engine.preserve_native_tool_format = False
            MockEngine.return_value = mock_engine

            # Mock asyncio loop
            with patch("vllm_mlx.server.asyncio") as mock_asyncio:
                mock_loop = MagicMock()
                mock_asyncio.new_event_loop.return_value = mock_loop

                from vllm_mlx.server import load_model

                load_model(
                    "test-model",
                    use_batching=False,
                    enable_mtp=True,
                    mtp_optimistic=True,
                )

                # Check SimpleEngine was called with MTP flags
                MockEngine.assert_called_once()
                call_kwargs = MockEngine.call_args[1]
                self.assertTrue(call_kwargs["enable_mtp"])
                self.assertTrue(call_kwargs["mtp_optimistic"])


if __name__ == "__main__":
    unittest.main()
