# SPDX-License-Identifier: Apache-2.0
"""Regression for F-063 — extreme-aspect-ratio / sub-patch images on VLMs.

Before the fix, ``MLLMBatchGenerator._preprocess_request`` passed any
image straight into ``mlx_vlm.utils.prepare_inputs``. When a dimension
fell below the patch-tokenizer's patch_size (14 for Qwen3-VL), the
patch grid for the short side rounded down to 0 patches and the
vision-token sequence came out empty. The language model then had no
image embedding to attend to and silently hallucinated a plausible
reply (e.g. "The image is a solid green color") — the HTTP response
came back as ``200 OK`` with ``finish_reason="stop"`` and
``content=<hallucinated text>``, indistinguishable from a real
answer.

mlx_vlm's own ``smart_resize`` only rejects extreme *aspect ratios*
(``abs(w/h) > 200``), which catches the 1×500 / 2×500 / 1×10000 case
but lets 1×100 / 2×2 / 3×2 through.

The fix pre-validates each image with PIL in
``_preprocess_request`` and raises
``ValueError("Failed to process image: image too small …")`` when
``min(w, h) < 3``. The canonical ``"Failed to process image"`` prefix
is what ``MLLMScheduler._step_no_queue.is_client_error`` matches
on — so the request is cleanly aborted *and* the route layer
(``routes/chat.py`` / ``routes/anthropic.py`` /
``routes/responses.py``) returns ``HTTP 400`` with the actionable
message instead of a silent 200.

The threshold of 3 is conservative: it admits any image that the
patch tokenizer is guaranteed to keep at least one patch for, while
rejecting the strictly broken ``min(w, h) <= 2`` cases the user
reported.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


class _StubModel:
    """Minimal VLM stub — only needs ``language_model`` and ``config``."""

    def __init__(self):
        self.language_model = object()

        class _Cfg:
            image_token_index = None

        self.config = _Cfg()


def _make_generator() -> MLLMBatchGenerator:
    return MLLMBatchGenerator(
        model=_StubModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )


def _make_request(images: list[str]) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="describe",
        images=images,
        max_tokens=8,
    )


def _write_png(tmp_path: Path, w: int, h: int) -> str:
    """Materialize a constant-green PNG of size ``w x h`` and return its path.

    ``process_image_input`` is bypassed in tests below — the path is fed
    straight into the dimension guard via ``request.images``.
    """
    img = Image.new("RGB", (w, h), color="green")
    p = tmp_path / f"{w}x{h}.png"
    img.save(p, format="PNG")
    return str(p)


def _bypass_process_image(monkeypatch):
    """Make ``process_image_input`` a no-op identity so the test can pass
    a pre-materialized PNG path directly without going through base64
    decode / tempfile registration.
    """
    from vllm_mlx.models import mllm as mllm_models

    def _identity(img):
        return img

    monkeypatch.setattr(mllm_models, "process_image_input", _identity)


def _install_no_op_prepare_inputs(monkeypatch):
    """Stub ``prepare_inputs`` so the guard is the *only* gate under
    test. If the dim guard lets a sub-patch image through, the stub
    returns an empty dict and the test asserts the guard fired — not
    that mlx_vlm crashed.
    """
    import mlx_vlm.utils as mlx_vlm_utils

    from vllm_mlx import mllm_batch_generator as gen_mod

    def _passthrough(*args, **kwargs):
        # Mimic mlx_vlm.utils.prepare_inputs return shape so the
        # caller doesn't trip on ``inputs.get("input_ids")``.
        return {
            "input_ids": None,
            "pixel_values": None,
            "attention_mask": None,
        }

    monkeypatch.setattr(mlx_vlm_utils, "prepare_inputs", _passthrough)
    if hasattr(gen_mod, "prepare_inputs"):
        monkeypatch.setattr(gen_mod, "prepare_inputs", _passthrough)


# Dimensions reported in F-063 as silently emitting empty content:
#   (1, 10000), (10000, 1), (2, 500), (1, 500), (1, 100), (2, 2)
# Plus the additional boundary ``(3, 2)`` and ``(2, 3)`` to lock down
# the asymmetry: a single dimension below the threshold is enough.
@pytest.mark.parametrize(
    "w,h",
    [
        (1, 10000),
        (10000, 1),
        (2, 500),
        (1, 500),
        (1, 100),
        (2, 2),
        (3, 2),
        (2, 3),
        (1, 1),
    ],
)
def test_sub_patch_image_rejected_with_canonical_message(monkeypatch, tmp_path, w, h):
    """``min(w, h) < 3`` images must raise the canonical
    ``Failed to process image: image too small …`` ValueError before
    ``prepare_inputs`` runs.

    The route layer's matcher (``"Failed to process image" in err_msg``
    in ``MLLMScheduler._step_no_queue``) keys off the prefix to fire
    HTTP 400; the dimension digits are required so the client sees
    *which* image was rejected.
    """
    _bypass_process_image(monkeypatch)
    _install_no_op_prepare_inputs(monkeypatch)

    img_path = _write_png(tmp_path, w, h)
    gen = _make_generator()
    req = _make_request(images=[img_path])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)

    msg = str(exc_info.value)
    assert msg.startswith("Failed to process image"), (
        f"route matcher would miss this message: {msg!r}"
    )
    assert "image too small" in msg
    assert f"{w}x{h}" in msg


# Dimensions that must NOT be rejected — the smallest legitimate sizes
# the model can still represent with at least one patch on each axis
# (3 ≤ short side ≤ patch_size cases) and well-formed common sizes.
@pytest.mark.parametrize(
    "w,h",
    [
        (3, 100),  # boundary: short side hits the threshold exactly
        (3, 3),  # minimum square
        (14, 14),  # one full patch
        (100, 100),  # typical
        (4, 50),  # asymmetric but both ≥ 3
    ],
)
def test_above_threshold_image_passes_guard(monkeypatch, tmp_path, w, h):
    """``min(w, h) >= 3`` must not trip the guard. The stubbed
    ``prepare_inputs`` returns a no-op dict — the test only verifies the
    guard didn't raise (no ``ValueError`` about ``image too small``).
    """
    _bypass_process_image(monkeypatch)
    _install_no_op_prepare_inputs(monkeypatch)

    img_path = _write_png(tmp_path, w, h)
    gen = _make_generator()
    req = _make_request(images=[img_path])

    # Should not raise — guard passes, prepare_inputs returns empty dict
    # (no real model wired in for this test).
    gen._preprocess_request(req)


def test_unreadable_image_does_not_short_circuit_guard(monkeypatch, tmp_path):
    """If PIL can't open the file, the guard must *skip* the dim check
    (continue) and let the downstream ``except (OSError, ValueError)``
    block on ``prepare_inputs`` produce the canonical
    ``Failed to process image`` ValueError — otherwise we'd swallow a
    bad-payload error here and produce no client-visible 400.
    """
    _bypass_process_image(monkeypatch)

    # Write a file that is not a valid image. PIL.Image.open will raise.
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"this is not a PNG")

    # Stub prepare_inputs to raise the canonical PIL OSError to mimic
    # what mlx_vlm does for undecodable bytes.
    import mlx_vlm.utils as mlx_vlm_utils

    from vllm_mlx import mllm_batch_generator as gen_mod

    def _raise_oserror(*args, **kwargs):
        raise OSError("broken data stream when reading image file")

    monkeypatch.setattr(mlx_vlm_utils, "prepare_inputs", _raise_oserror)
    if hasattr(gen_mod, "prepare_inputs"):
        monkeypatch.setattr(gen_mod, "prepare_inputs", _raise_oserror)

    gen = _make_generator()
    req = _make_request(images=[str(bad_path)])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)

    # The downstream wrapper (F-061/F-062 fix) should produce the canonical
    # prefix — confirms the dim guard didn't preempt the wrapper.
    msg = str(exc_info.value)
    assert msg.startswith("Failed to process image"), msg
    assert "broken data stream" in msg
