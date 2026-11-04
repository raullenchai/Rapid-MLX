# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex FL2VA/Ref2VA adapters and fixed six-step audio/video sampler."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import minimax_h3
from .minimax_h3 import MiniMaxH3BackendError, MiniMaxH3VideoEngine, merge_lora

REPO = "Pika-Labs/Pika-H3-Flex"
REVISION = "3ff166447d5edb1406107ab8370c42000ce173e4"
ADAPTER_FILE = "fl2va/adapter.safetensors"
REF_ADAPTER_FILE = "ref2va/adapter.safetensors"
REF_ADAPTER_SHA256 = "eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5"
FILES = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "fl2va/manifest.json",
    ADAPTER_FILE,
)
ADAPTER_SHA256 = "8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327"
ADAPTER_RANK = ADAPTER_ALPHA = 128
TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
KNOTS = (1000, 750, 500, 250, 100, 40, 0)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
# The manifest's base 42ed227ee7df40d41602854ae760620d6eb651fe has
# byte-identical transformer, text_encoder, vae and audio_vae LFS payloads.
BASE_REPO, BASE_REVISION, BASE_FILES = (
    minimax_h3.REPO,
    minimax_h3.REVISION,
    minimax_h3.FILES,
)
REF2VA_FILES = ("ref2va/manifest.json", REF_ADAPTER_FILE)
REF2VA_BASE_FILES = (
    "transformer_ref/config.json",
    "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
    *(
        f"transformer_ref/diffusion_pytorch_model-{i:05d}-of-00014.safetensors"
        for i in range(1, 15)
    ),
)
PAYLOADS = ((REPO, REVISION, FILES), (BASE_REPO, BASE_REVISION, BASE_FILES))
REF2VA_PAYLOADS = (
    (BASE_REPO, BASE_REVISION, REF2VA_BASE_FILES),
    (REPO, REVISION, REF2VA_FILES),
)
# Hub get_paths_info metadata at the pinned revisions above (bytes).
REF2VA_FILE_SIZES = {
    "ref2va/manifest.json": 1040,
    "ref2va/adapter.safetensors": 2767274120,
    "transformer_ref/config.json": 546,
    "transformer_ref/diffusion_pytorch_model.safetensors.index.json": 64488,
    "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors": 4825958704,
    "transformer_ref/diffusion_pytorch_model-00002-of-00014.safetensors": 4702158032,
    "transformer_ref/diffusion_pytorch_model-00003-of-00014.safetensors": 4933368192,
    "transformer_ref/diffusion_pytorch_model-00004-of-00014.safetensors": 4567069608,
    "transformer_ref/diffusion_pytorch_model-00005-of-00014.safetensors": 4702158080,
    "transformer_ref/diffusion_pytorch_model-00006-of-00014.safetensors": 4933368232,
    "transformer_ref/diffusion_pytorch_model-00007-of-00014.safetensors": 4567069608,
    "transformer_ref/diffusion_pytorch_model-00008-of-00014.safetensors": 4702158080,
    "transformer_ref/diffusion_pytorch_model-00009-of-00014.safetensors": 4933368232,
    "transformer_ref/diffusion_pytorch_model-00010-of-00014.safetensors": 4567069608,
    "transformer_ref/diffusion_pytorch_model-00011-of-00014.safetensors": 4702158080,
    "transformer_ref/diffusion_pytorch_model-00012-of-00014.safetensors": 4933368232,
    "transformer_ref/diffusion_pytorch_model-00013-of-00014.safetensors": 4567069608,
    "transformer_ref/diffusion_pytorch_model-00014-of-00014.safetensors": 4644161920,
}
REF2VA_EXTRA_BYTES = sum(REF2VA_FILE_SIZES.values())

_EAGER_VAE_ENCODER_CLASSES: dict[type[Any], type[Any]] = {}


def _install_eager_vae_encoder(video_vae: Any) -> None:
    encoder = getattr(video_vae, "encoder", None)
    if not callable(encoder):
        raise MiniMaxH3BackendError("Video VAE encoder is unavailable")
    original_class = type(encoder)
    if original_class in _EAGER_VAE_ENCODER_CLASSES.values():
        return
    eager_class = _EAGER_VAE_ENCODER_CLASSES.get(original_class)
    if eager_class is None:

        def eager_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            import mlx.core as mx

            result = super(cast(type[Any], eager_class), self).__call__(*args, **kwargs)
            # Bound transient memory without changing numerics: only materialize.
            mx.eval(result)
            return result

        eager_class = type(
            f"Eager{original_class.__name__}",
            (original_class,),
            {"__call__": eager_call},
        )
        _EAGER_VAE_ENCODER_CLASSES[original_class] = eager_class
    encoder.__class__ = eager_class


def ensure_ref2va_assets(cancel_token=None, on_progress=None) -> None:
    """Provision only missing Ref2VA assets before loading its transformer."""
    from huggingface_hub import try_to_load_from_cache

    from .. import model_downloads
    from .._mirror import (
        _DOWNLOAD_DISK_HEADROOM,
        _cache_disk_space,
        pinned_snapshot_download,
    )

    if cancel_token is not None:
        cancel_token.check()
    missing = set()
    missing_bytes = REF2VA_EXTRA_BYTES
    for repo, revision, files in REF2VA_PAYLOADS:
        for filename in files:
            if cancel_token is not None:
                cancel_token.check()
            cached = try_to_load_from_cache(repo, filename, revision=revision)
            if isinstance(cached, str) and Path(cached).is_file():
                missing_bytes -= REF2VA_FILE_SIZES[filename]
            else:
                missing.add((repo, filename))
    if not missing_bytes:
        return
    command = "rapid-mlx pull pika-h3-flex --with-ref2va"
    if model_downloads.disabled():
        raise MiniMaxH3BackendError(
            f"Pika H3 Flex Ref2VA assets are missing and model downloads are "
            f"disabled. Run `{command}` to pre-fetch them."
        )
    try:
        cache, available = _cache_disk_space()
    except OSError as exc:
        raise MiniMaxH3BackendError(
            f"Could not check free disk space for the Pika H3 Flex Ref2VA download "
            f"({exc}). Run `{command}` to pre-fetch the assets."
        ) from exc
    # Match the CLI's 10% headroom for temporary download files.
    size = f"{missing_bytes / 1e9:.1f} GB"
    if available < int(missing_bytes * (1 + _DOWNLOAD_DISK_HEADROOM)):
        raise MiniMaxH3BackendError(
            f"Ref2VA needs an additional {size} download "
            f"(second transformer + Ref2VA adapter); only {available / 1e9:.1f} GB "
            f"free on {cache}. Free space or run `{command}` after freeing space."
        )
    logger = logging.getLogger(__name__)
    logger.info(
        "Downloading Pika H3 Flex Ref2VA assets (%s, first Ref2VA request only)...",
        size,
    )
    progress_kwargs: dict[str, Any] = {}
    # Completed files count at their declared size; the in-flight file is capped
    # below its size so mirror-then-Hub retries or resumed blobs cannot over- or
    # under-report overall progress.
    completed = 0
    current = 0
    current_size = 0
    progress_lock = threading.Lock()

    def report():
        if on_progress is not None:
            on_progress(
                "downloading", min(completed + current, missing_bytes), missing_bytes
            )

    def on_bytes(delta):
        nonlocal current
        if cancel_token is not None:
            cancel_token.check()
        with progress_lock:
            current = min(max(0, current + delta), max(0, current_size - 1))
            report()

    def start_file(size):
        nonlocal current, current_size
        with progress_lock:
            current = 0
            current_size = size

    def finish_file():
        nonlocal completed, current, current_size
        with progress_lock:
            completed += current_size
            current = 0
            current_size = 0
            report()

    if cancel_token is not None or on_progress is not None:
        from huggingface_hub.utils import tqdm

        class CancelProgress(tqdm):
            def update(self, n=1):
                on_bytes(n)
                return super().update(n)

        progress_kwargs.update(tqdm_class=CancelProgress, on_bytes=on_bytes)
    try:
        if on_progress is not None:
            on_progress("downloading", 0, missing_bytes)
        for repo, revision, files in REF2VA_PAYLOADS:
            if cancel_token is None and on_progress is None:
                pinned_snapshot_download(repo, revision, allow_patterns=list(files))
            else:
                # Check cached-file boundaries too; they may have no progress updates.
                # Leave incomplete cache blobs intact for resumable downloads.
                for filename in files:
                    if (repo, filename) not in missing:
                        continue
                    start_file(REF2VA_FILE_SIZES[filename])
                    on_bytes(0)
                    pinned_snapshot_download(
                        repo, revision, allow_patterns=[filename], **progress_kwargs
                    )
                    on_bytes(0)
                    finish_file()
    except minimax_h3.VideoGenerationCancelled:
        raise
    except Exception as exc:
        if cancel_token is not None:
            cancel_token.check()
        raise MiniMaxH3BackendError(
            f"Pika H3 Flex Ref2VA asset download failed. Retry with `{command}`: {exc}"
        ) from exc
    finally:
        if on_progress is not None:
            on_progress(None, 0, 0)
    logger.info("Pika H3 Flex Ref2VA assets downloaded (%s).", size)


def is_pika_h3_flex_model(name: str | None) -> bool:
    return bool(
        name and name.rstrip("/").rsplit("/", 1)[-1].casefold() == "pika-h3-flex"
    )


def shifted_sigmas(knots, shift: float) -> list[float]:
    if (
        len(knots) < 2
        or knots[0] != 1000
        or knots[-1] != 0
        or any(not math.isfinite(k) for k in knots)
        or any(a <= b for a, b in zip(knots, knots[1:]))
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError(
            "knots must decrease strictly from 1000 to 0; shift must be positive"
        )
    # Python float evaluates the schedule in float64 before rounding to float32.
    return [
        struct.unpack(
            "f", struct.pack("f", shift * (k / 1000) / (1 + (shift - 1) * (k / 1000)))
        )[0]
        for k in knots
    ]


def align_frames(seconds: int) -> int:
    # At 15 s the upstream grid gives 362, beyond the runtime's 15.0 s limit.
    return min(345, ((seconds * 24 - 5 + 16) // 17) * 17 + 5)


class RenoiseNoise:
    def reset(self, seed: int) -> None:
        import mlx.core as mx

        self.key = mx.random.key(seed)

    def normal(self, shape):
        import mlx.core as mx

        self.key, key = mx.random.split(self.key)
        return mx.random.normal(shape, dtype=mx.float32, key=key)


class RenoiseScheduler:
    order = 1

    def __init__(self, shift: float = VIDEO_SHIFT, noise: RenoiseNoise | None = None):
        self.noise = noise if noise is not None else RenoiseNoise()
        if noise is None:
            self.noise.reset(0)
        self.set_shift(shift)
        self.set_timesteps()

    def set_shift(self, shift):
        shifted_sigmas(KNOTS, shift)
        self.shift = float(shift)

    def set_timesteps(self, num_inference_steps=None, *, sigmas=None):
        import mlx.core as mx

        self.sigmas = mx.array(shifted_sigmas(KNOTS, self.shift), dtype=mx.float32)
        self.timesteps = 1.0 - self.sigmas[:-1]
        self.num_inference_steps = len(KNOTS) - 1
        self._step_index = 0

    def scale_noise(self, sample, timestep, noise):
        import mlx.core as mx

        time = mx.array(timestep, dtype=sample.dtype)
        while time.ndim < sample.ndim:
            time = mx.expand_dims(time, axis=-1)
        return time * sample + (1.0 - time) * noise

    def step(self, model_output, timestep, sample):
        import mlx.core as mx

        denoised = sample.astype(mx.float32) + (
            1.0 - mx.array(timestep, dtype=mx.float32)
        ) * model_output.astype(mx.float32)
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        if float(sigma_next) == 0:
            return denoised.astype(sample.dtype)
        return (
            (1.0 - sigma_next) * denoised + sigma_next * self.noise.normal(sample.shape)
        ).astype(sample.dtype)


@dataclass(frozen=True)
class AdapterSpec:
    task: str
    file: str
    sha256: str
    trunk: str
    numerics_policy: str


def adapter_spec(mode: str) -> AdapterSpec:
    if mode == "fl2va":
        return AdapterSpec(
            mode, ADAPTER_FILE, ADAPTER_SHA256, "transformer", "bf16_conditioning_v1"
        )
    if mode == "ref2va":
        return AdapterSpec(
            mode,
            REF_ADAPTER_FILE,
            REF_ADAPTER_SHA256,
            "transformer_ref",
            "fp32_conditioning_v1",
        )
    raise ValueError(f"unsupported Flex partition: {mode}")


def verify_adapter(root: Path, mode: str = "fl2va") -> None:
    spec = adapter_spec(mode)
    try:
        manifest = json.loads((root / spec.task / "manifest.json").read_text())
        expected = {
            "rank": ADAPTER_RANK,
            "alpha": ADAPTER_ALPHA,
            "trunk": spec.trunk,
            "numerics_policy": spec.numerics_policy,
            "target_modules": list(TARGET_MODULES),
            "sha256": spec.sha256,
        }
        sampling = {
            "knots": list(KNOTS),
            "video_shift": VIDEO_SHIFT,
            "audio_shift": AUDIO_SHIFT,
            "method": "renoise",
        }
        if any(manifest.get(k) != v for k, v in expected.items()) or any(
            manifest.get("sampling", {}).get(k) != v for k, v in sampling.items()
        ):
            raise MiniMaxH3BackendError("Pika H3 Flex manifest mismatch")
        digest = hashlib.sha256()
        with (root / spec.file).open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != spec.sha256:
            raise MiniMaxH3BackendError("Pika H3 Flex adapter sha256 mismatch")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise MiniMaxH3BackendError("Pika H3 Flex adapter verification failed") from exc


@dataclass(frozen=True)
class ReferenceTier:
    name: str
    short_edge: int
    max_pixels: int
    ref_short_edge: int


REFERENCE_TIERS = tuple(
    ReferenceTier(*values)
    for values in (
        ("p480", 480, 832 * 480, 1024),
        ("p544", 544, 960 * 544, 1024),
        ("p768", 768, 1344 * 768, 2048),
        ("p1080", 1088, 1920 * 1088, 2048),
        ("p1440", 1440, 2560 * 1440, 2048),
        ("p2160", 2176, 3840 * 2176, 2048),
    )
)
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}


def reference_tier(height: int, width: int) -> ReferenceTier:
    if min(height, width) <= 0:
        raise ValueError("canvas dimensions must be positive")
    return next(
        (tier for tier in REFERENCE_TIERS if tier.short_edge >= min(height, width)),
        REFERENCE_TIERS[-1],
    )


def reference_image_size(
    width: int, height: int, tier: ReferenceTier
) -> tuple[int, int]:
    if min(width, height) <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError("reference image must be within 1:4 and 4:1")
    scale = tier.ref_short_edge / min(width, height)
    height = max(32, round(height * scale / 32) * 32)
    width = max(32, round(width * scale / 32) * 32)
    return height, width


def reference_video_size(
    width: int, height: int, tier: ReferenceTier
) -> tuple[int, int]:
    # Authors' canvas rule: scale short edge to tier, cap area at max_pixels, round axes to nearest 32.
    # As in their pipeline, rounding may exceed max_pixels by < one 32-pixel row/column (p480: 448x896).
    # max_pixels is the tier's trained size target, not a memory/safety bound; overshoot is intentional.
    # Do not clamp: that would move conditioning off the trained geometry.
    if min(width, height) <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError("reference video must be within 1:4 and 4:1")
    scale = tier.short_edge / min(width, height)
    h, w = height * scale, width * scale
    if h * w > tier.max_pixels:
        scale = math.sqrt(tier.max_pixels / (h * w))
        h, w = h * scale, w * scale
    height = max(32, round(h / 32) * 32)
    width = max(32, round(w / 32) * 32)
    return height, width


def reference_audio_max_samples(num_frames: int, sample_rate: int = 32000) -> int:
    return round(num_frames / 24 * sample_rate)


def validate_reference_counts(kinds, *, require_visual: bool = True) -> None:
    if len(kinds) > 12:
        raise minimax_h3.MiniMaxH3RequestError("references maximum total is 12")
    for kind in kinds:
        if kind not in REFERENCE_LIMITS:
            raise minimax_h3.MiniMaxH3RequestError("unsupported reference type")
    for kind, limit in REFERENCE_LIMITS.items():
        if kinds.count(kind) > limit:
            raise minimax_h3.MiniMaxH3RequestError(
                f"references maximum {kind} count is {limit}"
            )
    if require_visual and not any(kind in ("image", "video") for kind in kinds):
        raise minimax_h3.MiniMaxH3RequestError(
            "references require at least one image or video"
        )


def validate_reference_durations(durations) -> None:
    totals = {"video": 0.0, "audio": 0.0}
    for kind, duration in durations:
        if (
            kind not in totals
            or not math.isfinite(duration)
            or not 2 - 1e-6 <= duration <= 15 + 1e-6
        ):
            raise ValueError(f"{kind} reference duration must be 2..15 seconds")
        totals[kind] += duration
    for kind, total in totals.items():
        if total > 15 + 1e-6:
            raise ValueError(f"{kind} reference total duration must be <= 15 seconds")


def prepare_flex_references(pipeline, references, num_frames, canvas):
    """Apply the released Flex conditioning rules using runtime media helpers."""
    import mlx.core as mx
    from mlx_vlm.models.minimax_h3 import processing as media
    from mlx_vlm.models.minimax_h3.references import (
        MiniMaxH3PreparedReference,
        validate_references,
    )

    validate_reference_counts([ref.kind for ref in references])
    validate_references(references)
    if num_frames is None:
        raise ValueError("Flex references require num_frames")
    num_frames = pipeline._validate_duration(num_frames)
    tier = reference_tier(*canvas)
    prepared, durations = [], []
    for ref in references:
        item = MiniMaxH3PreparedReference(kind=ref.kind, has_audio=ref.has_audio)
        soundtrack = None
        if ref.kind == "image":
            image = media.decode_image(ref.image)
            h, w = reference_image_size(image.shape[1], image.shape[0], tier)
            item.image = media.prepare_reference_image(image, h, w)
        elif ref.kind == "video":
            frames, decoded_fps = media.decode_video(ref.video)
            fps = float(ref.fps or decoded_fps or 24)
            durations.append(("video", len(frames) / fps))
            validate_reference_durations(durations)
            frames = media.resample_reference_frames(frames, fps)[:num_frames]
            if len(frames) < 22:
                raise ValueError(
                    "video reference must contain at least 22 frames at 24 fps"
                )
            h, w = reference_video_size(frames.shape[2], frames.shape[1], tier)
            item.frames = media.resize_lanczos(frames, h, w)
            if ref.audio is None and isinstance(ref.video, (str, Path)):
                soundtrack = media.decode_video_soundtrack(ref.video)
            item.has_audio = ref.has_audio or soundtrack is not None
        if item.has_audio:
            if ref.kind == "audio" and isinstance(ref.audio, (str, Path)):
                # Decode every accepted container (wav, mp3, m4a, flac, ogg)
                # through the same ffmpeg path as video soundtracks.
                soundtrack = media.decode_video_soundtrack(ref.audio)
                if soundtrack is None:
                    raise ValueError("audio reference has no decodable audio stream")
            if soundtrack is not None:
                waveform, rate = soundtrack
            else:
                waveform, decoded_rate = media.decode_audio(ref.audio)
                rate = (
                    ref.sample_rate
                    or decoded_rate
                    or pipeline.audio_vae.config.sampling_rate
                )
            if ref.kind == "audio":
                durations.append(("audio", waveform.shape[-1] / rate))
                validate_reference_durations(durations)
            waveform = mx.array(waveform).astype(mx.float32)
            if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
                raise ValueError(
                    "reference waveform must be mono or stereo (channels, samples)"
                )
            if waveform.shape[0] == 1:
                waveform = mx.repeat(waveform, 2, axis=0)
            target_rate = pipeline.audio_vae.config.sampling_rate
            waveform = media.resample_waveform(waveform, rate, target_rate)
            item.waveform = waveform[
                :, : reference_audio_max_samples(num_frames, target_rate)
            ]
        prepared.append(item)
    return prepared, num_frames


class PikaH3FlexVideoEngine(MiniMaxH3VideoEngine):
    supports_last_frame = True
    supports_references = True
    base_files = BASE_FILES
    steps = len(KNOTS)

    def _ensure_assets(self, partition):
        if Path(self.model_name).expanduser().is_dir():
            return
        if partition == "ref2va":
            token = getattr(self, "_cancel_token", None)
            on_progress = getattr(self, "_on_progress", None)
            if token is None and on_progress is None:
                ensure_ref2va_assets()
            else:
                ensure_ref2va_assets(cancel_token=token, on_progress=on_progress)

    def _merge_adapter(self, pipeline, mode):
        from .._mirror import pinned_snapshot_download

        root = Path(
            pinned_snapshot_download(
                REPO,
                REVISION,
                allow_patterns=list(REF2VA_FILES if mode == "ref2va" else FILES),
            )
        )
        verify_adapter(root, mode)
        merge_lora(
            pipeline.transformer,
            root / adapter_spec(mode).file,
            scale=ADAPTER_ALPHA / ADAPTER_RANK,
            allowed_suffixes=TARGET_MODULES,
            expected_rank=ADAPTER_RANK,
        )

    def _prepare_model(self, model):
        pipeline = model.pipeline
        _install_eager_vae_encoder(getattr(pipeline, "video_vae", None))
        mode = self._needed_partition
        self._merge_adapter(pipeline, mode)
        self._renoise = RenoiseNoise()
        pipeline.scheduler = RenoiseScheduler(VIDEO_SHIFT, self._renoise)
        pipeline.audio_scheduler = RenoiseScheduler(AUDIO_SHIFT, self._renoise)
        self._original_normalize = pipeline._normalize_video_latents

        def normalize_bf16(latents):
            import mlx.core as mx

            return (
                self._original_normalize(latents).astype(mx.bfloat16).astype(mx.float32)
            )

        self._bf16_normalize = normalize_bf16
        pipeline._prepare_references = lambda references, num_frames: (
            prepare_flex_references(
                pipeline, references, num_frames, self._reference_canvas
            )
        )
        self._set_partition(pipeline, mode)

    def _set_partition(self, pipeline, mode):
        pipeline.partition = mode
        pipeline._adaln_cache_key = None
        pipeline._adaln_caches = None
        pipeline._adaln_weights_freed_bytes = 0
        pipeline._normalize_video_latents = (
            self._original_normalize if mode == "ref2va" else self._bf16_normalize
        )
        self._resident_partition = mode

    def _ensure_partition(self, partition):
        if self._resident_partition == partition:
            return
        self._ensure_assets(partition)
        import mlx.core as mx
        from mlx_vlm.models.minimax_h3.weights import load_transformer

        pipeline = self._model.pipeline
        old = self._resident_partition
        self._resident_partition = None
        pipeline.transformer = None
        pipeline._adaln_caches = None
        gc.collect()
        mx.clear_cache()
        try:
            pipeline.transformer = load_transformer(
                self._model_root, partition=partition
            )
            self._merge_adapter(pipeline, partition)
        except Exception:
            pipeline.transformer = None
            gc.collect()
            mx.clear_cache()
            raise
        self._set_partition(pipeline, partition)
        logging.getLogger(__name__).info(
            "Pika H3 Flex partition switch %s -> %s", old, partition
        )

    def _prepare_request(self, seed: int) -> None:
        self._renoise.reset(seed + 1)
