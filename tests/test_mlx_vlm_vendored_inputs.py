# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored ``prepare_inputs`` surface (``mlx_vlm_vendored/
inputs.py``, step 2c).

Provenance is enforced mechanically: every function of the vendored region
must be byte-identical to the pinned upstream 0.7.1 source (the parity
probe pattern from the 2b-1 ``kv_quant`` test). Behavior parity is checked
on the text-only path with a deterministic fake tokenizer, and the lane
wiring (``mllm_batch_generator`` / ``multimodal_processor`` resolve the
vendored module) is exercised by the stub-based suites that patch the
vendored module directly.
"""

import inspect
from dataclasses import asdict

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.mlx_vlm_vendored import inputs as vendored_inputs

_REGION_FUNCTIONS = [
    "load_image",
    "resize_image",
    "process_image",
    "estimate_num_image_tokens",
    "read_audio",
    "load_audio",
    "VideoSampling",
    "VideoMetadata",
    "load_video",
    "processor_video_sampling",
    "resolve_video_sampling",
    "process_inputs",
    "process_inputs_with_fallback",
    "prepare_inputs",
]


def test_vendored_region_is_byte_identical_to_upstream():
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")
    for name in _REGION_FUNCTIONS:
        upstream = inspect.getsource(getattr(mlx_vlm_utils, name))
        vendored = inspect.getsource(getattr(vendored_inputs, name))
        assert vendored == upstream, f"{name} diverged from pinned upstream"


def test_vendored_region_constants_match_upstream():
    """Module-level assignments of the vendored region must match too.

    Compared via ``asdict``: the two ``VideoSampling`` instances are
    different classes (vendored vs upstream), so dataclass ``__eq__``
    would be identity-false across namespaces even for equal fields.
    """
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")
    assert asdict(vendored_inputs.DEFAULT_VIDEO_SAMPLING) == asdict(
        mlx_vlm_utils.DEFAULT_VIDEO_SAMPLING
    )
    assert (
        vendored_inputs._VIDEO_SAMPLING_FIELDS == mlx_vlm_utils._VIDEO_SAMPLING_FIELDS
    )


class _FakeTokenizer:
    pad_token = None
    eos_token = "</s>"

    def __call__(
        self,
        prompts,
        add_special_tokens=False,
        padding=False,
        padding_side="left",
        return_tensors="mlx",
    ):
        class _Encoded:
            input_ids = [[1, 2, 3]]
            attention_mask = [[1, 1, 1]]

        return _Encoded()


class _BareProcessor:
    tokenizer = _FakeTokenizer()


def test_prepare_inputs_text_only_matches_upstream():
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")

    upstream_out = mlx_vlm_utils.prepare_inputs(
        _BareProcessor(), prompts=["hello world"], padding=False
    )
    vendored_out = vendored_inputs.prepare_inputs(
        _BareProcessor(), prompts=["hello world"], padding=False
    )
    assert set(vendored_out) == {"input_ids", "attention_mask"}
    assert (vendored_out["input_ids"] == upstream_out["input_ids"]).all()
    assert (vendored_out["attention_mask"] == upstream_out["attention_mask"]).all()
    assert isinstance(vendored_out["input_ids"], mx.array)


def test_prepare_inputs_sets_pad_token_when_padding():
    tokenizer = _FakeTokenizer()
    processor = _BareProcessor()
    processor.tokenizer = tokenizer
    vendored_inputs.prepare_inputs(processor, prompts=["hello world"], padding=True)
    assert tokenizer.pad_token == tokenizer.eos_token
