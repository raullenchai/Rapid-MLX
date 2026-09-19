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

import hashlib
import inspect
import io
import tokenize
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.mlx_vlm_vendored import inputs as vendored_inputs

_UPSTREAM_REGION_SHA256 = (
    "ac610b0e2c157de878b17ec9f5ebaa8bf2c75000e44c09d84b5c17dbaf7c7b5f"
)
_VENDOR_DEVIATION_COUNT = 7

_REGION_FUNCTIONS = [
    "load_image",
    "resize_image",
    "process_image",
    "estimate_num_image_tokens",
    "read_audio",
    "VideoSampling",
    "VideoMetadata",
    "resolve_video_sampling",
    "process_inputs",
    "process_inputs_with_fallback",
]
# Excluded from the probe: ``processor_video_sampling``, ``load_video``,
# ``load_audio``, and ``prepare_inputs`` carry the documented hunks (see
# the package inventory) and are behavior-tested below.


def test_vendored_region_is_byte_identical_to_upstream():
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")
    for name in _REGION_FUNCTIONS:
        upstream = inspect.getsource(getattr(mlx_vlm_utils, name))
        vendored = inspect.getsource(getattr(vendored_inputs, name))
        assert vendored == upstream, f"{name} diverged from pinned upstream"


def test_vendored_region_matches_reviewed_sources():
    """Fail closed if an excluded deviation function drifts unnoticed.

    Function-level parity is intentionally unavailable for the four functions
    carrying reviewed deviations. Pinning both the original upstream region
    and the complete vendored module prevents an unrelated edit from hiding in
    those large functions merely because they are excluded above.
    """
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")
    upstream_path = Path(inspect.getsourcefile(mlx_vlm_utils))
    upstream_lines = upstream_path.read_text().splitlines(keepends=True)
    upstream_region = "".join(upstream_lines[1713:2543])
    assert hashlib.sha256(upstream_region.encode()).hexdigest() == (
        _UPSTREAM_REGION_SHA256
    )

    vendored_path = Path(inspect.getsourcefile(vendored_inputs))
    vendored_source = vendored_path.read_text()
    assert hashlib.sha256(vendored_source.encode()).hexdigest() == (
        "a688bdc97b69daab6e25d189ede7b69ce7859b30df0175d624ae1ff165ef277c"
    )
    deviation_comments = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(vendored_source).readline)
        if token.type == tokenize.COMMENT
        and token.string.startswith("# VENDOR-DEVIATION")
    ]
    assert len(deviation_comments) == _VENDOR_DEVIATION_COUNT
    assert sum("(redirect)" in comment for comment in deviation_comments) == 3
    assert sum("(upstream-bugfix)" in comment for comment in deviation_comments) == 3
    assert sum("(dual-namespace)" in comment for comment in deviation_comments) == 1


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


def test_load_video_releases_capture_when_open_fails(monkeypatch):
    """The handle exists even when ``isOpened`` is false and must be released."""
    cv2 = pytest.importorskip("cv2")
    released = []

    class _FakeCap:
        def __init__(self, path):
            pass

        def isOpened(self):  # noqa: N802 - mirrors the cv2 API being faked
            return False

        def release(self):
            released.append(True)

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

    with pytest.raises(ValueError, match="Cannot open video"):
        vendored_inputs.load_video("broken.mp4")
    assert released == [True]


def test_load_audio_closes_streamed_response(monkeypatch):
    """The HTTP response must close after its bytes reach the audio decoder."""
    audio_io = pytest.importorskip("mlx_audio.audio_io")
    audio_utils = pytest.importorskip("mlx_audio.utils")
    events = []

    class _Response:
        content = b"encoded-audio"

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

        def raise_for_status(self):
            events.append("status")

    monkeypatch.setattr(
        vendored_inputs.requests,
        "get",
        lambda *args, **kwargs: _Response(),
    )
    monkeypatch.setattr(
        audio_io,
        "read",
        lambda source, dtype: (np.array([[0.25], [0.5]], dtype=np.float32), 16_000),
    )
    monkeypatch.setattr(
        audio_utils,
        "resample_audio",
        lambda audio, source_rate, target_rate: pytest.fail(
            "same-rate audio should not be resampled"
        ),
    )

    result = vendored_inputs.load_audio("https://example.invalid/audio.wav", 16_000)
    assert result.tolist() == pytest.approx([0.25, 0.5])
    assert events == ["enter", "status", "exit"]


def test_prepare_inputs_decodes_bytes_video_paths(monkeypatch):
    """upstream-bugfix: bytes video paths are fsdecoded, not str()-ed."""
    seen = []

    class _Component:
        pass  # no hook: the component-attrs fallback applies

    class _Processor:
        video_processor = _Component()

    def _fake_load_video(path, *args, **kwargs):
        seen.append(path)
        return (
            np.zeros((1, 3, 4, 4)),
            vendored_inputs.VideoMetadata(
                total_num_frames=1,
                fps=1.0,
                frames_indices=[0],
                duration=1.0,
            ),
        )

    def _fake_fallback(processor, **kwargs):
        return {"input_ids": mx.array([[1]]), "attention_mask": mx.array([[1]])}

    monkeypatch.setattr(vendored_inputs, "load_video", _fake_load_video)
    monkeypatch.setattr(vendored_inputs, "process_inputs_with_fallback", _fake_fallback)

    out = vendored_inputs.prepare_inputs(
        _Processor(), prompts=["p"], videos=[b"/tmp/clip.mp4"]
    )
    assert seen == ["/tmp/clip.mp4"]
    assert isinstance(out["input_ids"], mx.array)
