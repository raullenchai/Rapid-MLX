# SPDX-License-Identifier: Apache-2.0
"""Sidecar quantization inference for the Gemma 4 assistant drafter.

Deliberately a SEPARATE file from ``test_mtp_gemma4_assistant_inject.py``:
that module opens with ``pytest.importorskip("mlx.core")``, so everything in it
is skipped wherever mlx is absent -- including the Linux CI runner. These
checks are pure shape arithmetic and need no mlx, so keeping them here means
they actually execute in CI instead of silently skipping.
"""

from __future__ import annotations

import pytest

from vllm_mlx.spec_decode.mtp.gemma4_inject import _infer_sidecar_quantization

# ---------------------------------------------------------------------------
# 7. Sidecar quantization inference
#
# mlx-community publishes the assistants at several widths (the 12B alone has
# 4bit / 5bit / 6bit / 8bit / bf16 / mxfp4 / mxfp8 / nvfp4). The AssistantModel
# is built full-precision, so the module must be quantized to match whatever
# the sidecar actually carries -- read off its tensors, which are ground truth
# for the packing, rather than from a config.json a checkpoint may omit.
#
# Numbers below are the real shapes from
# ``mlx-community/gemma-4-12B-it-assistant-4bit``: pre_projection is an
# nn.Linear(2 * 3840, 1024), so full-precision is (1024, 7680) and the 4-bit
# affine packing is (1024, 7680 * 4 // 32) == (1024, 960) with scales
# (1024, 7680 // 64) == (1024, 120).
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Stand-in for an mx.array -- the inference only reads ``.shape``."""

    def __init__(self, *shape):
        self.shape = shape


_FP_OUT, _FP_IN = 1024, 7680


def test_infer_sidecar_quantization_recovers_4bit_group64():

    raw = {
        "pre_projection.weight": _FakeTensor(_FP_OUT, 960),
        "pre_projection.scales": _FakeTensor(_FP_OUT, 120),
    }
    assert _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN) == {
        "bits": 4,
        "group_size": 64,
    }


def test_infer_sidecar_quantization_returns_none_for_full_precision():
    """A bf16 sidecar carries no scales -- the module must stay unquantized."""

    raw = {"pre_projection.weight": _FakeTensor(_FP_OUT, _FP_IN)}
    assert _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN) is None


def test_infer_sidecar_quantization_rejects_truncated_quantized_sidecar():
    """Packed weight but no scales: loading it into an nn.Linear would crash
    at the first draft step, so refuse up front rather than later."""

    raw = {"pre_projection.weight": _FakeTensor(_FP_OUT, 960)}
    with pytest.raises(ValueError, match="truncated quantized sidecar"):
        _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN)


def test_infer_sidecar_quantization_rejects_wrong_out_dim():
    """A sidecar built for a different assistant size must not be mis-packed
    into this one just because the key names line up."""

    raw = {
        "pre_projection.weight": _FakeTensor(999, 960),
        "pre_projection.scales": _FakeTensor(999, 120),
    }
    with pytest.raises(ValueError, match="out-dim mismatch"):
        _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN)


def test_infer_sidecar_quantization_rejects_bits_outside_mlx_affine_set():
    """Derived bits=7 is not a width MLX affine quantization produces; feeding
    it to nn.quantize would re-open the mismatch this inference closes."""

    raw = {
        "pre_projection.weight": _FakeTensor(_FP_OUT, _FP_IN * 7 // 32),
        "pre_projection.scales": _FakeTensor(_FP_OUT, 120),
    }
    with pytest.raises(ValueError, match="outside MLX affine set"):
        _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN)


def test_infer_sidecar_quantization_rejects_non_reproducing_packing():
    """Divisibility alone is not enough -- the derived params must reproduce
    the observed shapes exactly, or a coincidental match would pass."""

    # scales imply group_size=64, but the packed weight implies bits=8 while
    # carrying a column count that bits=8 would not produce for this in-dim.
    raw = {
        "pre_projection.weight": _FakeTensor(_FP_OUT, 960),
        "pre_projection.scales": _FakeTensor(_FP_OUT, 7680),  # group_size 1
    }
    with pytest.raises(ValueError):
        _infer_sidecar_quantization(raw, _FP_OUT, _FP_IN)
