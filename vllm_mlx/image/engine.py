# SPDX-License-Identifier: Apache-2.0
"""mflux-backed image generation engine.

Thin wrapper over `mflux <https://github.com/filipstrand/mflux>`_ — an
MLX-native, line-by-line port of the FLUX / Qwen-Image model families with
built-in 4/8-bit quantization. Rapid-MLX owns request validation, the lazy
load / process-lock lifecycle and the OpenAI-compatible transport; mflux owns
the diffusion pipeline and weight loading.

Only Apache-2.0-licensed families are wired here so the whole surface stays
commercially clean:

* ``flux2-klein``      — text→image + image edit (``FLUX.2-klein-4B``),
  4B/4-step, the fast default: ~3 s @ 512² / ~10 s @ 1024² on an M3 Ultra,
  ~4 GB at 4-bit
* ``z-image``          — text→image (``Z-Image-Turbo``), 6B/8-step, the quality
  option (SOTA open-source photorealism), ~5.5 GB at 4-bit
* ``flux-schnell``     — text→image (``black-forest-labs/FLUX.1-schnell``), 12B
* ``qwen-image``       — text→image (``Qwen/Qwen-Image``), strongest text-in-image
* ``qwen-image-edit``  — instruction edit (``Qwen/Qwen-Image-Edit-2509``)

``flux2-klein`` and ``z-image`` supersede the older/larger families for the
interactive tab: Klein is ~3× faster than schnell/z-image at the same
resolution while staying smaller, and Qwen-Image (20B) is too slow to feature.

The mflux model is loaded lazily on the first ``generate`` call: the canonical
repos ship full-precision weights that mflux quantizes at load, so pulling them
at server boot would stall startup on a multi-gigabyte download.
"""

from __future__ import annotations

import gc
import io
import re
import threading
import time
from pathlib import Path

# A pre-quantized mflux repo carries a quant tag in its id — either the
# ``<n>bit`` / ``<n>-bit`` convention (``FLUX.1-schnell-mflux-4bit``) or the
# ``q<n>`` convention (``Qwen-Image-Edit-mflux-q4``). Anchored to a separator
# so a base repo like ``Qwen/Qwen-Image`` (no tag) is never misread — the
# leading ``q`` of "Qwen" is not followed by a quant digit.
_QUANT_TAG_RE = re.compile(
    r"(?:^|[-_./])(?:q[2-8]|[2-8]-?bit)(?:[-_./]|$)", re.IGNORECASE
)

# mflux/Metal graphs are not re-entrant — a single process-wide lock serializes
# every generation exactly like the video lane's ``_PROCESS_GENERATION_LOCK``.
_PROCESS_GENERATION_LOCK = threading.Lock()

# Default quantization for the on-load quantize path. 4-bit is the 32GB sweet
# spot (FLUX.1-schnell ~9GB, Qwen-Image ~12GB resident at q4).
_DEFAULT_QUANTIZE = 4


def _release_allocator_cache() -> None:
    """Return weights from a discarded mflux variant before loading another."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        # Unit-test hosts and older optional MLX builds are valid here.
        pass


class ImageRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


class ImageGenerationCancelled(ImageRuntimeError):  # noqa: N818 — a cancel, not an error condition
    """Raised when the user cancels an in-flight generation mid-denoise."""


class _ProgressReporter:
    """mflux in-loop callback that mirrors denoise progress onto the engine.

    Diffusion has a fixed, known step count, so this yields a *true*
    ``step / total`` signal (unlike an LLM token stream). Registered once per
    loaded model; it also enforces cooperative cancellation by raising out of
    the loop when the engine's cancel flag is set — the abort lands within one
    step (a second or two), not after the whole render.
    """

    def __init__(self, engine: ImageGenerationEngine) -> None:
        self._engine = engine

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:  # noqa: ANN001
        engine = self._engine
        total = getattr(config, "num_inference_steps", 0) or engine._progress.get(
            "total", 0
        )
        # ``running`` flips true only once the denoise loop actually fires, so
        # the client shows "warming up" during the cold load and "denoising"
        # (with a real step count) only while stepping — not step 1 mid-download.
        engine._progress["running"] = True
        engine._progress["step"] = int(t) + 1
        engine._progress["total"] = int(total)
        if engine._is_cancelled():
            raise ImageGenerationCancelled("Generation cancelled.")


def _detect_family(model_name: str) -> str:
    """Map an alias hf_path (or local dir) to a supported mflux family."""
    name = (model_name or "").casefold()
    # Klein first — its repos ("FLUX.2-klein-4B-mflux-4bit") also contain
    # "flux", so the distinctive "klein" / "flux2" token must win before the
    # generic FLUX.1 checks below.
    if "klein" in name or "flux2" in name or "flux.2" in name:
        return "flux2-klein"
    if "z-image" in name or "z_image" in name or "zimage" in name:
        return "z-image"
    if "qwen-image-edit" in name or "qwen_image_edit" in name:
        return "qwen-image-edit"
    if "qwen-image" in name or "qwen_image" in name:
        return "qwen-image"
    if "schnell" in name:
        return "flux-schnell"
    if "flux.1-dev" in name or "flux1-dev" in name:
        return "flux-dev"
    raise ImageRuntimeError(
        f"Unsupported image model '{model_name}'. Supported families: "
        "flux2-klein, z-image, flux-schnell, qwen-image, qwen-image-edit."
    )


# Families whose ``generate_image`` takes NO ``negative_prompt`` parameter.
# FLUX.2 Klein omits it (Flux1 / Qwen-Image / Z-Image all accept it), and
# passing an unknown kwarg raises — so the engine drops it for these.
_NO_NEGATIVE_PROMPT_FAMILIES = frozenset({"flux2-klein"})

# Per-family default denoise steps when the request pins none. Distilled/turbo
# models converge in a handful of steps; a non-distilled model needs many more.
_DEFAULT_STEPS_BY_FAMILY = {
    "flux2-klein": 4,  # distilled turbo
    "z-image": 8,  # turbo, but 8 is the sweet spot for its quality
    "flux-schnell": 4,  # distilled
    "flux-dev": 20,  # non-distilled
    "qwen-image": 20,  # non-distilled 20B
    "qwen-image-edit": 20,
}


def _looks_like_prequantized(model_name: str) -> bool:
    """A pre-quantized mflux repo / local dir is loaded via ``model_path``.

    The canonical BFL / Qwen repos ship full weights that mflux quantizes on
    load (a ~57 GB download). Community mflux repos ship already-quantized
    weights (~9-27 GB) whose id carries a quant tag; those are passed straight
    through as ``model_path`` with ``quantize=None`` — re-quantizing an
    already-quantized checkpoint makes mflux error.
    """
    if model_name and Path(model_name).expanduser().is_dir():
        return True
    return bool(_QUANT_TAG_RE.search(model_name or ""))


class ImageGenerationEngine:
    """Adapter over a single mflux model family.

    One instance owns one lazily-loaded mflux model. ``generate`` is blocking
    (the caller runs it off the event loop) and returns encoded PNG bytes so
    the transport never has to touch the filesystem.
    """

    def __init__(
        self, model_name: str, *, quantize: int | None = _DEFAULT_QUANTIZE
    ) -> None:
        self.model_name = model_name
        self.family = _detect_family(model_name)
        self.supports_generation = self.family != "qwen-image-edit"
        self.supports_editing = self.family in {"flux2-klein", "qwen-image-edit"}
        # Kept for compatibility with callers that distinguish exclusive edit
        # checkpoints. FLUX.2 supports both operations, so it is not edit-only.
        self.is_edit = self.family == "qwen-image-edit"
        self.default_steps = _DEFAULT_STEPS_BY_FAMILY.get(self.family, 4)
        self.default_edit_steps = 4 if self.family == "flux2-klein" else 20
        self.default_edit_guidance = None if self.family == "flux2-klein" else 4.0
        self.supports_negative_prompt = self.family not in _NO_NEGATIVE_PROMPT_FAMILIES
        self._prequantized = _looks_like_prequantized(model_name)
        # ``None`` when the repo is already quantized — passing a quantize width
        # for a pre-quantized checkpoint makes mflux re-quantize and error.
        self._quantize = None if self._prequantized else quantize
        self._model = None
        # FLUX.2 uses distinct mflux classes for generation and editing. Only
        # one stays resident at a time so switching modes does not duplicate
        # the checkpoint in unified memory.
        self._loaded_mode: str | None = None
        self._lock = _PROCESS_GENERATION_LOCK
        # Live denoise progress (single-flight under ``_lock``, so one snapshot
        # is unambiguous). ``request_cancel`` flips ``_cancel``; the reporter
        # reads it each step. ``_reporter`` is registered once per loaded model.
        self._progress: dict[str, float | int | bool] = {
            "running": False,
            "step": 0,
            "total": 0,
            "started_at": 0.0,
        }
        # Cancellation is scoped by a monotonic run sequence rather than a bare
        # boolean, so a cancel is tied to a specific render and can never be
        # clobbered by the next request arming itself. ``_active_seq`` is the
        # in-flight run (0 = none); ``_cancel_seq`` is the highest run a cancel
        # was requested for. A run is cancelled iff ``_cancel_seq >= its seq``.
        self._run_seq = 0
        self._active_seq = 0
        self._cancel_seq = 0
        # Guards the three seq counters, which are touched from both the request
        # thread (``request_cancel``) and the generation worker thread.
        self._state_lock = threading.Lock()
        self._reporter = _ProgressReporter(self)

    def _is_cancelled(self) -> bool:
        """True when the in-flight run has an outstanding cancel request."""
        with self._state_lock:
            return self._active_seq > 0 and self._cancel_seq >= self._active_seq

    def _model_path_for_mflux(self) -> str | None:
        """``model_path`` to hand mflux: a local directory whenever we have one.

        A pre-quantized repo / local dir is handed to mflux verbatim; a canonical
        repo is selected through ``ModelConfig`` instead (``None``) so mflux
        downloads the official weights and quantizes on load.

        Passing the bare repo id made mflux resolve it through
        ``huggingface_hub`` on every start, including a fully cached one — and
        that revision lookup has no timeout, so a poisoned-DNS network hangs the
        start rather than failing fast. Resolving the cached snapshot ourselves
        and passing the directory keeps a warm start entirely local. Falls back
        to the repo id whenever the cache can't be vouched for, so a cold or
        partial cache still pulls exactly as before.
        """
        if not self._prequantized:
            return None
        from .._download_gate import mflux_local_snapshot

        return mflux_local_snapshot(self.model_name) or self.model_name

    def _build_model(self):
        """Instantiate the backing mflux model (import-lazy)."""
        from mflux.models.common.config.model_config import ModelConfig

        model_path = self._model_path_for_mflux()

        if self.family == "flux2-klein":
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

            return Flux2Klein(
                quantize=self._quantize,
                model_path=model_path,
                model_config=ModelConfig.flux2_klein_4b(),
            )
        if self.family == "z-image":
            from mflux.models.z_image.variants.z_image import ZImage

            return ZImage(
                quantize=self._quantize,
                model_path=model_path,
                model_config=ModelConfig.z_image_turbo(),
            )
        if self.family == "qwen-image-edit":
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

            return QwenImageEdit(quantize=self._quantize, model_path=model_path)
        if self.family == "qwen-image":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

            return QwenImage(quantize=self._quantize, model_path=model_path)

        from mflux.models.flux.variants.txt2img.flux import Flux1

        config = (
            ModelConfig.schnell()
            if self.family == "flux-schnell"
            else ModelConfig.dev()
        )
        return Flux1(
            quantize=self._quantize, model_path=model_path, model_config=config
        )

    def _build_edit_model(self):
        """Instantiate the edit variant for a model that accepts input images."""
        if self.family == "qwen-image-edit":
            return self._build_model()
        if self.family == "flux2-klein":
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux2.variants.edit.flux2_klein_edit import (
                Flux2KleinEdit,
            )

            model_path = self._model_path_for_mflux()
            return Flux2KleinEdit(
                quantize=self._quantize,
                model_path=model_path,
                model_config=ModelConfig.flux2_klein_4b(),
            )
        raise ImageRuntimeError(f"{self.family} does not support image editing.")

    def _verify_weights_complete(self) -> None:
        """Refuse to build a model out of a half-downloaded checkpoint.

        mflux loads whatever ``*.safetensors`` happen to sit in a component
        directory and never reads the ``model.safetensors.index.json`` beside
        them, so an interrupted pull is not an error to it: the shards that
        arrived get loaded, the rest of the transformer keeps its randomly
        initialised weights, and the run renders noise. Nothing downstream can
        tell that apart from a bad prompt.

        Checking the index here — the one point every load path converges on,
        whatever left the snapshot short — turns that into a clear failure
        before any weight is touched.
        """
        if Path(self.model_name).expanduser().is_dir():
            # A local directory is the user's own checkpoint layout; we have no
            # index contract to hold it to.
            return
        from .._download_gate import mflux_missing_weights

        missing = mflux_missing_weights(self.model_name)
        if not missing:
            # ``[]`` verified complete, ``None`` no verdict — see that
            # function: an environment problem must not read as a broken model.
            return
        raise ImageRuntimeError(
            f"Image model '{self.model_name}' is only partially downloaded, so "
            "generating with it would produce noise rather than an image. "
            f"Missing: {', '.join(missing)}. Re-run the download to finish it."
        )

    def _ensure_loaded(self, *, for_edit: bool | None = None):
        if for_edit is None:
            for_edit = self.is_edit
        desired_mode = "edit" if for_edit else "generation"
        if self._model is not None and self._loaded_mode not in (None, desired_mode):
            self._model = None
            self._loaded_mode = None
            _release_allocator_cache()
        if self._model is None:
            self._verify_weights_complete()
            try:
                self._model = (
                    self._build_edit_model() if for_edit else self._build_model()
                )
                self._loaded_mode = desired_mode
            except ImageRuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface a clean API error
                raise ImageRuntimeError(
                    f"Failed to load image model '{self.model_name}': {exc}"
                ) from exc
            # Register the progress/cancel reporter on the model's mflux
            # callback registry (present on every txt2img/edit variant).
            registry = getattr(self._model, "callbacks", None)
            if registry is not None and hasattr(registry, "register"):
                registry.register(self._reporter)
        return self._model

    def request_cancel(self) -> None:
        """Ask the in-flight generation to stop at the next denoise step.

        Targets the currently-active run; a cancel with no render in flight is a
        no-op (``_active_seq == 0``), and one armed while a render is starting is
        preserved because the seq is only advanced under the lock.
        """
        with self._state_lock:
            self._cancel_seq = max(self._cancel_seq, self._active_seq)

    def progress_snapshot(self) -> dict:
        """A JSON-safe view of the current denoise progress (single-flight)."""
        p = self._progress
        started = float(p.get("started_at") or 0.0)
        elapsed_ms = int((time.time() - started) * 1000) if started else 0
        return {
            "running": bool(p.get("running", False)),
            "step": int(p.get("step", 0)),
            "total": int(p.get("total", 0)),
            "elapsed_ms": elapsed_ms,
            "family": self.family,
        }

    def generate(
        self,
        *,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        image_paths: list[str] | None = None,
    ) -> bytes:
        """Generate one image and return it as PNG bytes.

        ``image_paths`` selects editing for dual-capability models and is
        required by edit-only checkpoints. Unsupported combinations fail loud
        instead of silently ignoring the conditioning image.
        """
        editing = bool(image_paths)
        if not editing and not self.supports_generation:
            raise ImageRuntimeError(
                "qwen-image-edit requires at least one input image (image_paths)."
            )
        if editing and not self.supports_editing:
            raise ImageRuntimeError(
                f"{self.family} is text-to-image only and does not accept input images; "
                "use an image-edit capable model."
            )

        with self._lock:
            # Claim a run sequence and arm progress BEFORE loading, so a Cancel
            # pressed during the (possibly multi-gigabyte) cold model load is
            # honored — its seq already matches this run — instead of being lost.
            # The lock guarantees single-flight, so one snapshot is unambiguous.
            with self._state_lock:
                self._run_seq += 1
                self._active_seq = self._run_seq
            # ``running`` stays False through the cold load — the reporter flips
            # it true on the first denoise step — so the client renders the
            # "warming up" phase during load, not a bogus "denoising step 1".
            self._progress.update(
                running=False,
                step=0,
                total=int(num_inference_steps),
                started_at=time.time(),
            )
            try:
                model = self._ensure_loaded(for_edit=editing)
                # Honor a cancel that landed during the warm-up load before we
                # commit to the denoise loop.
                if self._is_cancelled():
                    raise ImageGenerationCancelled("Generation cancelled.")
                if editing and self.family == "qwen-image-edit":
                    # Edit derives its output canvas from the input image and
                    # must NOT be given an explicit width/height. mflux fixes the
                    # VAE conditioning latents to a 1024²-area canvas of the input
                    # aspect ratio (``_compute_dimensions``), while the denoised
                    # latents use ``config.width/height``. Any mismatch between
                    # the two — e.g. forcing 512×512 against 1024²-derived
                    # conditioning — desyncs the RoPE position ids and the model
                    # emits pure noise (a valid, correctly-sized PNG of static).
                    # Passing ``None`` lets mflux size the target to match the
                    # conditioning, exactly like its edit CLI.
                    result = model.generate_image(
                        image_paths=image_paths,
                        height=None,
                        width=None,
                        **self._gen_kwargs(
                            seed, prompt, num_inference_steps, guidance, negative_prompt
                        ),
                    )
                elif editing:
                    result = model.generate_image(
                        image_paths=image_paths,
                        height=height,
                        width=width,
                        **self._gen_kwargs(
                            seed, prompt, num_inference_steps, guidance, negative_prompt
                        ),
                    )
                else:
                    result = model.generate_image(
                        height=height,
                        width=width,
                        **self._gen_kwargs(
                            seed, prompt, num_inference_steps, guidance, negative_prompt
                        ),
                    )
            except ImageRuntimeError:
                raise  # cancellation + already-clean errors pass straight through
            except Exception as exc:  # noqa: BLE001 — surface a clean API error
                raise ImageRuntimeError(f"Image generation failed: {exc}") from exc
            finally:
                self._progress["running"] = False
                # Clear the start time so a completed run's snapshot doesn't keep
                # reporting an ever-growing ``elapsed_ms`` while idle.
                self._progress["started_at"] = 0.0
                with self._state_lock:
                    self._active_seq = 0

        return self._encode_png(result)

    def _gen_kwargs(
        self, seed, prompt, num_inference_steps, guidance, negative_prompt
    ) -> dict:
        """Build ``generate_image`` kwargs, omitting params a family rejects.

        FLUX.2 Klein has no ``negative_prompt`` parameter (passing it raises),
        and a guidance-distilled model degrades when forced to a fixed guidance
        — so ``guidance`` is passed only when the caller set one, otherwise each
        model uses its own trained default.
        """
        kwargs = {
            "seed": seed,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        if guidance is not None:
            kwargs["guidance"] = guidance
        if negative_prompt is not None and self.supports_negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    @staticmethod
    def _encode_png(result) -> bytes:
        """Encode an mflux ``GeneratedImage`` to PNG bytes without touching disk."""
        pil_image = getattr(result, "image", None)
        if pil_image is None:
            raise ImageRuntimeError("Image backend returned no image data.")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()
