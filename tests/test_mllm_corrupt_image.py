# SPDX-License-Identifier: Apache-2.0
"""Regression for F-061 / F-062 — corrupt-image handling in the MLLM batch
generator.

Before the fix, ``MLLMBatchGenerator._preprocess_request`` called
``mlx_vlm.utils.prepare_inputs`` directly. For corrupted / unsupported
image payloads PIL raises one of:

* ``OSError("broken data stream when reading image file")``
* ``PIL.UnidentifiedImageError``
* ``ValueError("Failed to load image from <path>: cannot identify image
  file ...")`` — mlx_vlm wraps PIL errors with this message in some paths

The downstream contract used by the scheduler + route layer is:

* ``MLLMScheduler._step_no_queue`` catches ``(ValueError, RuntimeError)``
  and only flags the request as a *client* error (clean abort, no
  retry) when ``str(exc).startswith("Failed to process image")``.
* ``routes/chat.py`` / ``routes/anthropic.py`` / ``routes/responses.py``
  map that prefix to ``HTTP 400`` with the actionable message.

The bug had two faces:

* ``OSError`` is *neither* ``ValueError`` *nor* ``RuntimeError``, so it
  escaped the narrow ``except`` in ``_step_no_queue`` and reached the
  broad ``except Exception`` in ``_process_loop``. The loop logged
  ``Error in MLLM process loop: …`` and **continued without advancing
  the unprocessed-requests cursor**, retrying the same broken request
  every step (~20×/s) — single 10 KB mangled PNG → pegged MLLM worker
  (F-061).
* ``ValueError("Failed to load image from …")`` (mlx_vlm wrapping) was
  caught, but the substring matcher only looked for ``Failed to
  process image``. The request was treated as a *server* error
  (``finish_reason="length"``, ``error=None``) → silent ``200 OK`` with
  ``content=null`` and ``prompt_tokens=0`` (F-062).

The fix maps expected decoder failures to a typed ``ClientRequestError``
with a bounded public message. That makes ``_step_no_queue`` clean-abort
the request without exposing temporary paths from the underlying decoder.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from vllm_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    _temporary_vision_pixel_bounds,
)


class _StubModel:
    """Minimal VLM stub — only needs ``language_model`` and ``config``."""

    def __init__(self):
        self.language_model = object()

        class _Cfg:
            image_token_index = None

        self.config = _Cfg()


def _make_generator(
    *,
    vision_min_pixels: int = 0,
    vision_max_pixels: int = 0,
    processor=None,
) -> MLLMBatchGenerator:
    return MLLMBatchGenerator(
        model=_StubModel(),
        processor=processor or object(),
        mm_processor=None,
        enable_vision_cache=False,
        vision_min_pixels=vision_min_pixels,
        vision_max_pixels=vision_max_pixels,
    )


def _make_request(images: list[str] | None) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="describe",
        images=images or [],
        max_tokens=8,
    )


def _bypass_process_image(monkeypatch):
    """Skip the base64/url decode step so the test exercises
    ``prepare_inputs`` directly without hitting tempfile I/O.
    """
    from vllm_mlx.models import mllm as mllm_models

    def _identity(img):
        return img

    monkeypatch.setattr(mllm_models, "process_image_input", _identity)


def _install_prepare_inputs_stub(monkeypatch, raiser):
    """Patch ``prepare_inputs`` on every binding site.

    ``vllm_mlx.mllm_batch_generator._preprocess_request`` re-imports
    ``prepare_inputs`` from ``mlx_vlm.utils`` on every call (local
    ``from`` statement at the top of the method), so patching
    ``mlx_vlm.utils.prepare_inputs`` is the correct target *today*.
    If a future refactor hoists the import to module level, the
    name binding shifts to ``vllm_mlx.mllm_batch_generator``; patch
    that too so this test stays a meaningful gate either way.
    """
    import mlx_vlm.utils as mlx_vlm_utils

    from vllm_mlx import mllm_batch_generator as gen_mod

    monkeypatch.setattr(mlx_vlm_utils, "prepare_inputs", raiser)
    if hasattr(gen_mod, "prepare_inputs"):
        monkeypatch.setattr(gen_mod, "prepare_inputs", raiser)


def test_preprocess_forwards_opt_in_dynamic_resolution_bounds(monkeypatch):
    _bypass_process_image(monkeypatch)
    captured = {}

    class _ImageProcessor:
        size = {"shortest_edge": 65_536, "longest_edge": 16_777_216}

    class _Processor:
        image_processor = _ImageProcessor()

    def _prepare(*args, **kwargs):
        captured.update(args[0].image_processor.size)
        return {"input_ids": None, "pixel_values": None, "attention_mask": None}

    _install_prepare_inputs_stub(monkeypatch, _prepare)
    processor = _Processor()
    original_size = processor.image_processor.size
    gen = _make_generator(
        processor=processor,
        vision_min_pixels=65_536,
        vision_max_pixels=1_048_576,
    )
    gen._preprocess_request(_make_request(images=["image.png"]))

    assert captured["shortest_edge"] == 65_536
    assert captured["longest_edge"] == 1_048_576
    assert processor.image_processor.size is original_size


def test_pixel_cap_rejects_fixed_resolution_processor():
    with pytest.raises(ValueError, match="dynamic-resolution image processor"):
        _make_generator(vision_max_pixels=1_048_576)


def test_native_mlx_vlm_pixel_attributes_are_bounded_and_restored():
    class _ImageProcessor:
        min_pixels = 3_136
        max_pixels = 1_003_520

    class _Processor:
        image_processor = _ImageProcessor()

    processor = _Processor()
    with _temporary_vision_pixel_bounds(processor, 0, 262_144) as applied:
        assert applied
        assert processor.image_processor.min_pixels == 3_136
        assert processor.image_processor.max_pixels == 262_144

    assert processor.image_processor.min_pixels == 3_136
    assert processor.image_processor.max_pixels == 1_003_520


def test_preprocess_default_does_not_override_processor_resolution(monkeypatch):
    _bypass_process_image(monkeypatch)
    captured = {}

    def _prepare(*args, **kwargs):
        captured.update(kwargs)
        return {"input_ids": None, "pixel_values": None, "attention_mask": None}

    _install_prepare_inputs_stub(monkeypatch, _prepare)
    _make_generator()._preprocess_request(_make_request(images=["image.png"]))

    assert "min_pixels" not in captured
    assert "max_pixels" not in captured


def test_preprocess_wraps_pil_oserror_as_failed_to_process_image(monkeypatch):
    """``OSError("broken data stream …")`` must surface as
    ``ValueError("Failed to process image: …")`` so the scheduler's
    narrow ``except (ValueError, RuntimeError)`` clause catches it and
    the request is cleanly aborted instead of retrying forever.
    """
    _bypass_process_image(monkeypatch)

    def _raise_oserror(*args, **kwargs):
        raise OSError("broken data stream when reading image file")

    _install_prepare_inputs_stub(monkeypatch, _raise_oserror)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    assert str(exc_info.value).startswith("Failed to process image")
    assert "valid PNG, JPEG, GIF, or WebP" in str(exc_info.value)
    assert "broken data stream" not in str(exc_info.value)


def test_preprocess_wraps_pil_unidentified_image_as_failed_to_process_image(
    monkeypatch,
):
    """``PIL.UnidentifiedImageError`` shares the same canonical mapping —
    we don't want HTTP 500 for non-PNG bytes claiming to be PNGs.
    """
    _bypass_process_image(monkeypatch)

    from PIL import UnidentifiedImageError

    def _raise_unidentified(*args, **kwargs):
        raise UnidentifiedImageError("cannot identify image file 'X'")

    _install_prepare_inputs_stub(monkeypatch, _raise_unidentified)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,SGVsbG8="])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    assert str(exc_info.value).startswith("Failed to process image")
    assert "cannot identify image file" not in str(exc_info.value)


def test_preprocess_normalizes_failed_to_load_image_to_failed_to_process_image(
    monkeypatch,
):
    """mlx_vlm's own wrapper raises ``ValueError("Failed to load image
    from <path>: …")``. That wrapping reached the scheduler before the
    fix but missed the ``"Failed to process image"`` substring matcher,
    producing a silent 200 (F-062). Confirm the normalized prefix is
    present so ``is_client_error`` fires.
    """
    _bypass_process_image(monkeypatch)

    def _raise_failed_to_load(*args, **kwargs):
        raise ValueError(
            "Failed to load image from /tmp/xyz.png: cannot identify image file '/tmp/xyz.png'"
        )

    _install_prepare_inputs_stub(monkeypatch, _raise_failed_to_load)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    msg = str(exc_info.value)
    assert msg.startswith("Failed to process image"), (
        f"matcher would miss this message: {msg!r}"
    )
    # Underlying mlx_vlm message can contain private temporary paths and must
    # stay in the exception cause rather than crossing the public boundary.
    assert "/tmp/xyz.png" not in msg


def test_preprocess_does_not_reflect_canonical_looking_decoder_message(monkeypatch):
    """Message shape must not grant permission to cross the boundary."""
    _bypass_process_image(monkeypatch)

    def _raise_canonical(*args, **kwargs):
        raise ValueError("Failed to process image: /Users/private/secret.png")

    _install_prepare_inputs_stub(monkeypatch, _raise_canonical)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    msg = str(exc_info.value)
    assert "/Users/private/secret.png" not in msg


def test_process_image_internal_runtime_error_is_not_made_public(monkeypatch):
    """Unexpected loader bugs remain server errors even if their text looks safe."""
    from vllm_mlx.models import mllm as mllm_models

    sentinel = RuntimeError("Failed to process image: /Users/private/secret.png")

    def _raise_runtime_error(_image):
        raise sentinel

    monkeypatch.setattr(mllm_models, "process_image_input", _raise_runtime_error)
    gen = _make_generator()

    with pytest.raises(RuntimeError) as exc_info:
        gen._preprocess_request(_make_request(images=["https://example.test/a.png"]))
    assert exc_info.value is sentinel


def test_preprocess_propagates_internal_bugs_unchanged(monkeypatch):
    """Non-image exception types raised by ``prepare_inputs`` (a
    processor / tokenizer / MLX runtime bug) MUST propagate unchanged
    so the caller sees HTTP 500. Swallowing every ``Exception`` and
    re-raising as ``ValueError("Failed to process image: …")`` would
    misclassify a server bug as a client 400 and silently hide it
    from observability dashboards.
    """
    _bypass_process_image(monkeypatch)

    sentinel = AttributeError("'NoneType' object has no attribute 'image_token_index'")

    def _raise_attribute_error(*args, **kwargs):
        raise sentinel

    _install_prepare_inputs_stub(monkeypatch, _raise_attribute_error)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(AttributeError) as exc_info:
        gen._preprocess_request(req)
    # Verify the exception object is the original — not a re-raise
    # that lost the traceback / type identity.
    assert exc_info.value is sentinel


def test_preprocess_propagates_typeerror_unchanged(monkeypatch):
    """Same guarantee for ``TypeError`` — mlx-lm / mlx-vlm bugs often
    surface as TypeErrors from signature drift after a mlx upgrade.
    Misclassifying those as HTTP 400 would silently hide a regression
    surfaced by a real client image.
    """
    _bypass_process_image(monkeypatch)

    sentinel = TypeError(
        "prepare_inputs() got an unexpected keyword argument 'image_token_index'"
    )

    def _raise_type_error(*args, **kwargs):
        raise sentinel

    _install_prepare_inputs_stub(monkeypatch, _raise_type_error)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(TypeError) as exc_info:
        gen._preprocess_request(req)
    assert exc_info.value is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
