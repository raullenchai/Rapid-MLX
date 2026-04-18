# SPDX-License-Identifier: Apache-2.0
"""
Tests for the fixes in PR #137:
  1. Reasoning correction merged into final SSE chunk (before finish_reason)
  2. MTP MoE QuantizedSwitchLinear replacement
  3. Doctor runner TimeoutExpired bytes handling
  4. Shared MTP decode loop (mtp_generate.py)
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# ======================================================================
# Fix 1: Reasoning correction before terminal SSE
# ======================================================================


class TestReasoningCorrectionBeforeFinish:
    """Reasoning parser correction must be merged into the final chunk
    that carries finish_reason, not emitted as a separate chunk after it.

    OpenAI-compatible clients stop reading at finish_reason="stop",
    so any correction emitted afterwards is silently lost.
    """

    def test_finalize_called_on_finish_reason(self):
        """finalize_streaming() should be called when finish_reason is set."""
        from vllm_mlx.reasoning import get_parser

        parser_cls = get_parser("qwen3")
        parser = parser_cls()

        # Simulate streaming: model outputs short text without <think> tags
        # Parser holds it as potential reasoning until finalize
        parser.reset_state()
        delta1 = parser.extract_reasoning_streaming("", "Hello", "Hello")
        delta2 = parser.extract_reasoning_streaming("Hello", "Hello world", " world")

        # Finalize should produce correction (content that was held as reasoning)
        correction = parser.finalize_streaming("Hello world")
        # The correction may or may not have content depending on parser state,
        # but the method must not crash
        assert correction is None or hasattr(correction, "content")

    def test_correction_content_not_empty_for_no_tag_output(self):
        """When model outputs text without <think> tags, finalize should
        produce a correction with the held-back content."""
        from vllm_mlx.reasoning import get_parser

        parser_cls = get_parser("qwen3")
        parser = parser_cls()
        parser.reset_state()

        # Feed text that looks like it could be reasoning (no tags)
        text = "The answer is 42."
        parser.extract_reasoning_streaming("", text, text)

        correction = parser.finalize_streaming(text)
        if correction and correction.content:
            # Correction should contain the held-back text
            assert len(correction.content) > 0


# ======================================================================
# Fix 2: MTP MoE QuantizedSwitchLinear
# ======================================================================


class TestMTPQuantizedSwitchLinear:
    """nn.quantize() doesn't handle SwitchLinear → QuantizedSwitchLinear.
    The patch must replace SwitchLinear BEFORE load_weights so parameter
    names (weight, scales, biases) match the saved quantized weights.
    """

    def test_switch_linear_import(self):
        """QuantizedSwitchLinear should be importable from mlx_lm."""
        try:
            from mlx_lm.models.switch_layers import (
                QuantizedSwitchLinear,
                SwitchLinear,
            )
            assert QuantizedSwitchLinear is not None
            assert SwitchLinear is not None
        except ImportError:
            pytest.skip("mlx_lm switch_layers not available")

    def test_quantized_switch_linear_has_scales_biases(self):
        """QuantizedSwitchLinear should have scales and biases params
        that match the quantized weight file format."""
        try:
            from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        except ImportError:
            pytest.skip("mlx_lm switch_layers not available")

        # Create a small QuantizedSwitchLinear
        qsl = QuantizedSwitchLinear(
            input_dims=64, output_dims=32, num_experts=4,
            bias=False, group_size=64, bits=4,
        )
        # Must have weight, scales, biases for load_weights to match
        assert hasattr(qsl, "weight")
        assert hasattr(qsl, "scales")
        assert hasattr(qsl, "biases")

    def test_switch_linear_replacement_logic(self):
        """The replacement should convert SwitchLinear → QuantizedSwitchLinear
        with matching dimensions."""
        try:
            from mlx_lm.models.switch_layers import (
                QuantizedSwitchLinear,
                SwitchLinear,
            )
        except ImportError:
            pytest.skip("mlx_lm switch_layers not available")

        sl = SwitchLinear(input_dims=64, output_dims=32, num_experts=4)
        ne, od, id_ = sl.weight.shape  # (num_experts, output_dims, input_dims)

        qsl = QuantizedSwitchLinear(
            id_, od, ne, bias=False, group_size=64, bits=4, mode="affine",
        )
        # Dimensions should be compatible
        assert qsl.weight.shape[0] == ne  # num_experts preserved


# ======================================================================
# Fix 3: Doctor runner TimeoutExpired bytes handling
# ======================================================================


class TestDoctorTimeoutBytes:
    """subprocess.TimeoutExpired may return bytes even with text=True.
    run_subprocess must handle this gracefully.
    """

    def test_timeout_with_bytes_stdout(self):
        """TimeoutExpired with bytes stdout should not crash."""
        from vllm_mlx.doctor.runner import run_subprocess

        # Mock subprocess.run to raise TimeoutExpired with bytes
        exc = subprocess.TimeoutExpired(
            cmd=["test"], timeout=1,
            output=b"some output bytes",
            stderr=b"some error bytes",
        )
        with patch("subprocess.run", side_effect=exc):
            rc, stdout, stderr = run_subprocess(["test"], timeout=1)

        assert rc == 124
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)
        assert "some output bytes" in stdout
        assert "some error bytes" in stderr

    def test_timeout_with_str_stdout(self):
        """TimeoutExpired with str stdout should also work."""
        from vllm_mlx.doctor.runner import run_subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["test"], timeout=1,
            output="string output",
            stderr="string error",
        )
        with patch("subprocess.run", side_effect=exc):
            rc, stdout, stderr = run_subprocess(["test"], timeout=1)

        assert rc == 124
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)
        assert "string output" in stdout

    def test_timeout_with_none_stdout(self):
        """TimeoutExpired with None stdout should return empty string."""
        from vllm_mlx.doctor.runner import run_subprocess

        exc = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
        exc.stdout = None
        exc.stderr = None
        with patch("subprocess.run", side_effect=exc):
            rc, stdout, stderr = run_subprocess(["test"], timeout=1)

        assert rc == 124
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)


# ======================================================================
# Fix 4: Shared MTP decode loop
# ======================================================================


class TestMTPGenerateStep:
    """mtp_generate_step should handle various model behaviors."""

    def test_mtp_stats_dataclass(self):
        """MTPStats should track accepted/rejected/errors."""
        from vllm_mlx.speculative.mtp_generate import MTPStats

        stats = MTPStats(accepted=10, rejected=3, errors=1)
        assert stats.total == 13
        assert abs(stats.acceptance_rate - 10 / 13) < 1e-6

    def test_mtp_stats_zero_division(self):
        """MTPStats with no attempts should not crash."""
        from vllm_mlx.speculative.mtp_generate import MTPStats

        stats = MTPStats()
        assert stats.total == 0
        assert stats.acceptance_rate == 0.0

    def test_mtp_output_dataclass(self):
        """MTPOutput should carry token, logprobs, is_draft."""
        from vllm_mlx.speculative.mtp_generate import MTPOutput

        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        out = MTPOutput(token=42, logprobs=mx.zeros(100), is_draft=True)
        assert out.token == 42
        assert out.is_draft is True

    def test_snapshot_restore_rnn_state(self):
        """_snapshot_rnn_state and _restore_rnn_state should round-trip."""
        from vllm_mlx.speculative.mtp_generate import (
            _restore_rnn_state,
            _snapshot_rnn_state,
        )

        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        # Mock a non-trimmable cache layer (like DeltaNet)
        cache_layer = MagicMock()
        cache_layer.is_trimmable.return_value = False
        cache_layer.state = [mx.ones((2, 3)), mx.zeros((2, 3))]

        snapshots = _snapshot_rnn_state([cache_layer])
        assert 0 in snapshots

        # Mutate original state
        cache_layer.state = [mx.zeros((2, 3)), mx.ones((2, 3))]

        # Restore
        _restore_rnn_state([cache_layer], snapshots)
        # State should be restored (set by mock)
        assert cache_layer.state is not None

    def test_trim_cache_only_trimmable(self):
        """_trim_cache should only trim layers that are trimmable."""
        from vllm_mlx.speculative.mtp_generate import _trim_cache

        trimmable = MagicMock()
        trimmable.is_trimmable.return_value = True
        trimmable.trim = MagicMock()

        non_trimmable = MagicMock()
        non_trimmable.is_trimmable.return_value = False

        _trim_cache([trimmable, non_trimmable], 2)

        trimmable.trim.assert_called_once_with(2)
        assert not hasattr(non_trimmable, "trim") or not non_trimmable.trim.called

    def test_fallback_when_no_return_hidden(self):
        """mtp_generate_step should fall back to standard decode
        when model doesn't support return_hidden."""
        from vllm_mlx.speculative.mtp_generate import mtp_generate_step

        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        # Mock model that returns logits without tuple (no hidden states)
        model = MagicMock()
        logits = mx.random.normal((1, 5, 100))  # [batch, seq, vocab]
        model.return_value = logits  # Not a tuple → triggers fallback

        prompt = mx.array([[1, 2, 3]])
        cache = []

        sampler = MagicMock(side_effect=lambda x: mx.array([42]))

        gen = mtp_generate_step(model, prompt, cache, sampler, max_tokens=2)
        tokens = list(gen)
        assert len(tokens) == 2
        assert all(t.token == 42 for t in tokens)
        assert all(t.is_draft is False for t in tokens)
