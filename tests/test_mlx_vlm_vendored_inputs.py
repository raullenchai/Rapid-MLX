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
    "resolve_video_sampling",
    "process_inputs",
    "process_inputs_with_fallback",
    "prepare_inputs",
]
# Excluded from the probe: ``processor_video_sampling`` and ``load_video``
# carry the documented dual-namespace / capture-release hunks (see the
# package inventory) and are behavior-tested below.


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


def test_processor_video_sampling_accepts_upstream_instance():
    """dual-namespace: a hook returning the upstream VideoSampling dataclass
    must normalize into the vendored one instead of raising TypeError."""
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")

    class _HookProcessor:
        video_processor = object()  # non-None: the hook ladder is consulted

        def video_sampling_defaults(self):
            return mlx_vlm_utils.VideoSampling(fps=4, min_frames=2, max_frames=16)

    out = vendored_inputs.processor_video_sampling(_HookProcessor())
    assert isinstance(out, vendored_inputs.VideoSampling)
    assert asdict(out) == asdict(
        mlx_vlm_utils.VideoSampling(fps=4, min_frames=2, max_frames=16)
    )


def test_processor_video_sampling_dict_hook_unchanged():
    """The upstream hook convention (plain dict) keeps working verbatim."""

    class _HookProcessor:
        video_processor = object()

        def video_sampling_defaults(self):
            return {"max_frames": 32}

    out = vendored_inputs.processor_video_sampling(_HookProcessor())
    assert isinstance(out, vendored_inputs.VideoSampling)
    assert out.max_frames == 32


def test_load_video_releases_capture_on_sampler_error(monkeypatch):
    """upstream-bugfix: the cv2 capture handle must be released even when
    the frame sampler raises mid-decode."""
    cv2 = pytest.importorskip("cv2")
    released = []

    class _FakeCap:
        def __init__(self, path):
            pass

        def isOpened(self):  # noqa: N802 - mirrors the cv2 API being faked
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_COUNT: 10,
                cv2.CAP_PROP_FPS: 2.0,
                cv2.CAP_PROP_FRAME_WIDTH: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 4,
            }[prop]

        def release(self):
            released.append(True)

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("sampler boom")

    with pytest.raises(RuntimeError, match="sampler boom"):
        vendored_inputs.load_video(
            "clip.mp4",
            vendored_inputs.VideoSampling(fps=2, min_frames=1, max_frames=4),
            frame_sampler=_boom,
        )
    assert released == [True]
