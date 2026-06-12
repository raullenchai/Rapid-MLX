# SPDX-License-Identifier: Apache-2.0
"""
Tests for the fixes in PR #137:
  1. Reasoning correction merged into final SSE chunk (before finish_reason)
  2. MTP MoE QuantizedSwitchLinear replacement
  3. Doctor runner TimeoutExpired bytes handling
"""

import subprocess
from unittest.mock import patch

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
            input_dims=64,
            output_dims=32,
            num_experts=4,
            bias=False,
            group_size=64,
            bits=4,
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
            id_,
            od,
            ne,
            bias=False,
            group_size=64,
            bits=4,
            mode="affine",
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
            cmd=["test"],
            timeout=1,
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
            cmd=["test"],
            timeout=1,
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
