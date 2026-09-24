# SPDX-License-Identifier: Apache-2.0
"""Vendored ``prepare_inputs`` surface from ``mlx_vlm/utils.py`` @ 0.7.2.

Step 2c of the mlx-vlm dependency retirement (design note:
``docs/engineering/design/2026-09-18-vendor-mllm-primitives.md``).

Provenance: the module body below is **byte-verbatim** from upstream
``utils.py`` lines 1714-2807 (``load_image`` .. ``prepare_inputs`` ..
``group_images_by_shape`` .. ``should_add_special_tokens``, an unbroken
region), region sha256
``0c3681fa511baa4c345e6caba42760c7f1632ae2f69706982e39eb2f411b1294``,
except the documented deviations in the import block below — every
non-verbatim line carries a ``# VENDOR-DEVIATION`` sentinel.
The module keeps upstream's logger name so log filtering parity holds
(``mlx_vlm.apc`` precedent from 2b-2). The upstream parity of every
vendored function is probed by ``tests/test_mlx_vlm_vendored_inputs.py``.
"""

# VENDOR-DEVIATION(redirect): this import block replaces upstream
# utils.py's module-level imports with exactly the names the vendored
# region references (ruff F821 closure). stdlib/typing/PIL/numpy/requests
# resolve identically; the two mlx_vlm imports pin upstream until those
# modules are vendored (models.base) or stay upstream by contract.
import inspect  # noqa: F401  (region-referenced)
import logging
import math
import os
import warnings
from dataclasses import dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import requests
from PIL import Image, ImageOps
from mlx_vlm.models.base import BaseImageProcessor  # VENDOR-DEVIATION(redirect)

logger = logging.getLogger(
    "mlx_vlm.utils"
)  # VENDOR-DEVIATION(redirect): pinned upstream logger name


def load_image(image_source: Union[str, Path, BytesIO, Image.Image], timeout: int = 10):
    """
    Helper function to load an image from either a URL, file path, data URI,
    or BytesIO object.
    """
    import base64

    original_source = image_source
    try:
        if isinstance(image_source, Image.Image):
            image = image_source
        elif not isinstance(image_source, (str, Path, BytesIO)):
            raise ValueError(
                f"Unsupported image source type: {type(image_source).__name__}"
            )
        else:
            if isinstance(image_source, str) and image_source.startswith("data:image/"):
                if "," not in image_source:
                    raise ValueError(
                        "Invalid data URI format - missing comma separator"
                    )
                _, data = image_source.split(",", 1)
                image_source = BytesIO(base64.b64decode(data))
            if isinstance(image_source, str) and image_source.startswith(
                ("http://", "https://")
            ):
                with requests.get(
                    image_source, stream=True, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    image_source = BytesIO(response.content)

            image = Image.open(image_source)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load image from {original_source}: {e}") from e

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(img, max_size):

    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size)


def process_image(img, resize_shape, image_processor):
    if isinstance(img, str):
        img = load_image(img)
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    if resize_shape is not None:
        if isinstance(image_processor, BaseImageProcessor):
            # warnings (not logging) so repeated calls in a batch dedupe.
            warnings.warn(
                f"resize_shape={resize_shape} is ignored because "
                f"{type(image_processor).__name__} handles its own image "
                "sizing; use the processor's sizing options instead."
            )
        else:
            img = resize_image(img, resize_shape)
    return img


def estimate_num_image_tokens(processor, height: int, width: int, **size_overrides):
    """Estimate how many language-model image tokens an image will produce.

    Computed from the processor's own sizing math without loading or
    processing any pixels, so it is cheap enough to run per candidate image
    when sizing a prompt budget or choosing a ``max_pixels`` cap.

    Args:
        processor: A processor (or bare image processor). Wrapped processors
            are unwrapped via their ``image_processor`` attribute.
        height: Source image height in pixels.
        width: Source image width in pixels.
        **size_overrides: Optional per-call overrides forwarded to the image
            processor's ``num_image_tokens``, e.g. ``max_pixels=1_000_000``.

    Raises:
        NotImplementedError: If the image processor does not expose
            ``num_image_tokens``. Only dynamic-resolution processors support
            estimation; fixed-resolution models produce a constant token
            count regardless of image size.
    """
    image_processor = getattr(processor, "image_processor", processor)
    counter = getattr(image_processor, "num_image_tokens", None)
    if counter is None:
        raise NotImplementedError(
            f"{type(image_processor).__name__} does not expose "
            "num_image_tokens; token estimation is only available for "
            "dynamic-resolution image processors."
        )
    return int(counter(height, width, **size_overrides))


def read_audio(file) -> tuple:
    """Read an audio file using miniaudio (or ffmpeg for m4a/aac/ogg/opus).

    Returns (samples_float32, sample_rate) where samples is always 2D (samples, channels).
    """
    import io as _io

    if isinstance(file, bytes):
        file = _io.BytesIO(file)

    # Check if ffmpeg is needed for certain formats
    use_ffmpeg = False
    if isinstance(file, (str, Path)):
        ext = Path(file).suffix.lstrip(".").lower()
        if ext in ("m4a", "aac", "ogg", "opus"):
            use_ffmpeg = True
    elif isinstance(file, _io.BytesIO):
        pos = file.tell()
        header = file.read(12)
        file.seek(pos)
        if header[4:8] == b"ftyp" or header[:4] == b"OggS":
            use_ffmpeg = True

    if use_ffmpeg:
        import json as _json
        import shutil
        import subprocess

        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found. Install it: brew install ffmpeg (macOS) "
                "or sudo apt install ffmpeg (Linux)"
            )

        if isinstance(file, _io.BytesIO):
            file.seek(0)
            input_data = file.read()
        else:
            input_data = None

        # Get info via ffprobe
        if ffprobe_path and input_data is not None:
            probe = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "a:0",
                    "-i",
                    "pipe:0",
                ],
                input=input_data,
                capture_output=True,
            )
        elif ffprobe_path:
            probe = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "a:0",
                    str(file),
                ],
                capture_output=True,
            )
        else:
            probe = None

        sample_rate, nchannels = 44100, 1
        if probe and probe.returncode == 0:
            info = _json.loads(probe.stdout)
            if info.get("streams"):
                stream = info["streams"][0]
                sample_rate = int(stream.get("sample_rate", 44100))
                nchannels = int(stream.get("channels", 1))

        # Decode via ffmpeg to raw PCM s16le
        cmd = [ffmpeg_path, "-v", "quiet"]
        if input_data is not None:
            cmd += ["-i", "pipe:0"]
        else:
            cmd += ["-i", str(file)]
        cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(nchannels), "pipe:1"]

        result = subprocess.run(cmd, input=input_data, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decoding failed: {result.stderr.decode()}")

        samples = np.frombuffer(result.stdout, dtype=np.int16)
    else:
        import miniaudio

        if isinstance(file, (str, Path)):
            info = miniaudio.get_file_info(str(file))
            decoded = miniaudio.decode_file(
                str(file),
                nchannels=info.nchannels,
                sample_rate=info.sample_rate,
            )
        elif isinstance(file, _io.BytesIO):
            file.seek(0)
            data = file.read()
            # Detect format from magic bytes
            if data[:4] == b"RIFF":
                info = miniaudio.wav_get_info(data)
            elif data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xfa"):
                info = miniaudio.mp3_get_info(data)
            elif data[:4] == b"fLaC":
                info = miniaudio.flac_get_info(data)
            else:
                info = miniaudio.vorbis_get_info(data)
            decoded = miniaudio.decode(
                data,
                nchannels=info.nchannels,
                sample_rate=info.sample_rate,
            )
        else:
            raise TypeError(f"Unsupported file type: {type(file)}")

        sample_rate = decoded.sample_rate
        nchannels = decoded.nchannels
        samples = np.array(decoded.samples, dtype=np.int16)

    # Reshape multi-channel and convert to float32
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels)
    audio = samples.astype(np.float32) / 32768.0

    # Ensure always 2D
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    return audio, sample_rate


def load_audio(
    file,
    sr: int,
    timeout: int = 10,
):
    """
    Helper function to load audio from either a URL, file path, or numpy array.
    """
    if isinstance(file, np.ndarray):
        audio = file.astype(np.float32, copy=False)
        return audio.mean(axis=1) if audio.ndim > 1 else audio

    from mlx_audio.audio_io import read as read_audio
    from mlx_audio.utils import resample_audio

    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, str) and file.startswith(("http://", "https://")):
        try:
            # VENDOR-DEVIATION(upstream-bugfix): close the streamed response
            # on success and every decoder/error path instead of leaking it.
            with requests.get(file, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                audio, sample_rate = read_audio(
                    BytesIO(response.content), dtype="float32"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load audio from URL: {file} with error {e}"
            ) from e
    else:
        audio, sample_rate = read_audio(file, dtype="float32")

    # Upstream 0.7.2 downmixes before resampling: ``read_audio`` returns
    # (samples, channels), while ``resample_audio`` operates on the last axis.
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return np.asarray(audio, dtype=np.float32)


@dataclass(frozen=True)
class VideoSampling:
    """How many frames to take from a clip, and at what rate.

    Every field is optional so an unset one can be filled from a
    lower-precedence source via :meth:`merge`, with
    :data:`DEFAULT_VIDEO_SAMPLING` terminating the chain.
    """

    fps: Optional[float] = None
    nframes: Optional[int] = None
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    frame_factor: Optional[int] = None

    def merge(self, fallback: "VideoSampling") -> "VideoSampling":
        """Return a copy with every unset field taken from ``fallback``."""
        return VideoSampling(
            **{
                f.name: (
                    getattr(self, f.name)
                    if getattr(self, f.name) is not None
                    else getattr(fallback, f.name)
                )
                for f in fields(self)
            }
        )


DEFAULT_VIDEO_SAMPLING = VideoSampling(
    fps=2.0, min_frames=4, max_frames=768, frame_factor=2
)


@dataclass
class VideoMetadata:
    """What the decode step knew about a clip, carried alongside its frames.

    Field names mirror ``transformers.video_utils.VideoMetadata`` so a
    processor ported from upstream can consume this unchanged.
    """

    total_num_frames: int
    fps: float
    frames_indices: List[int]
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None

    @property
    def timestamps(self) -> List[float]:
        """Seconds into the clip for each sampled frame."""
        return [idx / self.fps for idx in self.frames_indices]

    @property
    def sampled_fps(self) -> float:
        """Frame rate actually achieved, which clamping can push well below
        the requested ``fps``."""
        return len(self.frames_indices) / max(self.total_num_frames, 1e-6) * self.fps


def load_video(
    video_path: str,
    sampling: Optional[VideoSampling] = None,
    frame_sampler=None,
    **sampling_kwargs,
) -> Tuple[np.ndarray, VideoMetadata]:
    """Read a video file as a (T, C, H, W) numpy array.

    Samples ``nframes`` frames, a count derived from ``fps``, or indices from
    ``frame_sampler``. Returns source-aware :class:`VideoMetadata` alongside
    the frames. Sampling fields may be supplied as a :class:`VideoSampling`
    or as loose keyword arguments; the former wins for overlapping fields.
    """
    import cv2

    if sampling_kwargs:
        unknown = set(sampling_kwargs) - {f.name for f in fields(VideoSampling)}
        if unknown:
            raise TypeError(
                f"load_video() got unexpected keyword arguments: {sorted(unknown)}"
            )
        sampling = (sampling or VideoSampling()).merge(VideoSampling(**sampling_kwargs))
    resolved = (sampling or VideoSampling()).merge(DEFAULT_VIDEO_SAMPLING)
    fps = resolved.fps
    nframes = resolved.nframes
    min_frames = resolved.min_frames
    max_frames = resolved.max_frames
    frame_factor = resolved.frame_factor

    if video_path.startswith("file://"):
        video_path = video_path[7:]

    cap = cv2.VideoCapture(video_path)
    # VENDOR-DEVIATION(upstream-bugfix): the native capture handle leaked on
    # any exception after open (failed opens, frame sampler errors, index
    # validation, cap.read/cvtColor failures); everything from the isOpened
    # check on runs under try/finally so the handle is always released.
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps

        def _round(n):
            return round(n / frame_factor) * frame_factor

        def _floor(n):
            return math.floor(n / frame_factor) * frame_factor

        def _ceil(n):
            return math.ceil(n / frame_factor) * frame_factor

        used_frame_sampler = False
        if nframes is not None:
            n = _round(nframes)
            indices = np.linspace(0, total_frames - 1, n).round().astype(int)
        elif frame_sampler is not None:
            used_frame_sampler = True
            source_metadata = VideoMetadata(
                total_num_frames=total_frames,
                fps=video_fps,
                frames_indices=list(range(total_frames)),
                width=width,
                height=height,
                duration=duration,
            )
            indices = np.asarray(
                frame_sampler(source_metadata, fps=fps, max_frames=max_frames),
                dtype=int,
            ).reshape(-1)
            n = len(indices)
        else:
            lo = _ceil(min_frames)
            hi = _floor(min(max_frames, total_frames))
            n = total_frames / video_fps * fps
            n = min(max(n, lo), hi, total_frames)
            n = _floor(n)
            indices = np.linspace(0, total_frames - 1, n).round().astype(int)
        if not used_frame_sampler and not (frame_factor <= n <= total_frames):
            raise ValueError(
                f"nframes must be in [{frame_factor}, {total_frames}], got {n}."
            )
        if n == 0 or np.any(indices < 0) or np.any(indices >= total_frames):
            raise ValueError(
                f"Frame indices must be within a non-empty {total_frames}-frame video."
            )
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError("No frames read from the video.")

    video_np = np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2))
    metadata = VideoMetadata(
        total_num_frames=total_frames,
        fps=video_fps,
        # Truncated where a read failed part-way, so this tracks the frames
        # actually returned rather than the ones asked for.
        frames_indices=indices[: len(frames)].tolist(),
        height=int(video_np.shape[2]),
        width=int(video_np.shape[3]),
        duration=total_frames / video_fps,
    )
    return video_np, metadata


_VIDEO_SAMPLING_FIELDS = tuple(f.name for f in fields(VideoSampling))


def processor_video_sampling(processor) -> VideoSampling:
    """The frame sampling a processor asks for, if it declares any.

    A processor opts in either by exposing a ``video_sampling_defaults()``
    hook — the way to declare a cap whose attribute is named something else,
    such as Gemma 4's ``num_frames`` — or by carrying matching attributes on
    its video processor component.
    """
    component = getattr(processor, "video_processor", None)
    if component is None:
        return VideoSampling()
    for owner in (component, processor):
        hook = getattr(owner, "video_sampling_defaults", None)
        if callable(hook):
            declared = hook()
            # VENDOR-DEVIATION(dual-namespace): a hook may return the upstream
            # mlx_vlm.utils.VideoSampling while the vendored class is in
            # effect; normalize any matching-shape object by its known
            # fields instead of keying on class identity.
            if isinstance(declared, VideoSampling):
                return declared
            if not isinstance(declared, dict) and all(
                hasattr(declared, f.name) for f in fields(VideoSampling)
            ):
                declared = {
                    f.name: getattr(declared, f.name) for f in fields(VideoSampling)
                }
            return VideoSampling(**declared)
    return VideoSampling(
        **{
            name: getattr(component, name, None)
            for name in ("fps", "min_frames", "max_frames")
        }
    )


def resolve_video_sampling(processor, overrides: Dict[str, Any]) -> VideoSampling:
    """Settle how a clip gets sampled.

    Caller wins, then whatever the processor declares, then the library
    defaults. Consumes the sampling keys from ``overrides`` so they do not
    travel on to the processor, which only ever sees frames that were already
    chosen here.
    """
    explicit = VideoSampling(
        **{
            name: overrides.pop(name)
            for name in _VIDEO_SAMPLING_FIELDS
            if name in overrides
        }
    )
    return explicit.merge(processor_video_sampling(processor)).merge(
        DEFAULT_VIDEO_SAMPLING
    )


def process_inputs(
    processor,
    prompts,
    images=None,
    audio=None,
    add_special_tokens=False,
    padding=True,
    padding_side="left",
    return_tensors="mlx",
    **kwargs,
):
    # Get the process method from the processor
    process_method = getattr(processor, "process", processor)
    parameters = inspect.signature(process_method).parameters

    # Prepare arguments
    args = {
        "text": prompts,
        "images": images,
        "padding": padding,
        "return_tensors": return_tensors,
    }
    if "padding_side" in parameters:
        args["padding_side"] = padding_side

    # Add special tokens if supported
    if "add_special_tokens" in parameters:
        args["add_special_tokens"] = add_special_tokens

    for param in parameters.keys():
        if param in kwargs.keys():
            args[param] = kwargs.get(param, None)

    # Add audio if provided and supported
    if audio is not None and len(audio) > 0:
        if "audio" in parameters:
            args["audio"] = audio
        elif "audios" in parameters:
            args["audios"] = audio
        else:
            raise ValueError(
                f"Processor {processor.__class__.__name__} does not support audio parameter"
            )

    return process_method(**args)


def process_inputs_with_fallback(
    processor,
    prompts,
    images,
    audio,
    add_special_tokens=False,
    return_tensors="mlx",
    **kwargs,
):
    # First attempt with specified return_tensors
    try:
        return process_inputs(
            processor,
            prompts=prompts,
            images=images,
            audio=audio,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Failed to process inputs with error: {e}")


def prepare_inputs(
    processor,
    images=None,
    audio=None,
    videos=None,
    prompts=None,
    image_token_index=None,
    resize_shape=None,
    add_special_tokens=False,
    padding=True,
    padding_side="left",
    pad_to_uniform_size=False,
    return_tensors="mlx",
    **kwargs,
):

    has_images = images is not None and (
        not hasattr(images, "__len__") or len(images) > 0
    )
    has_audio = audio is not None and (not hasattr(audio, "__len__") or len(audio) > 0)
    has_videos = videos is not None and (
        not hasattr(videos, "__len__") or len(videos) > 0
    )
    if not has_images and not has_audio and not has_videos:
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        # Ensure pad_token exists when padding text-only inputs
        if padding and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            prompts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            padding_side=padding_side,
            return_tensors=return_tensors,
        )
        input_ids = (
            inputs.input_ids
            if isinstance(inputs.input_ids, mx.array)
            else mx.array(inputs.input_ids)
        )
        mask = (
            inputs.attention_mask
            if isinstance(inputs.attention_mask, mx.array)
            else mx.array(inputs.attention_mask)
        )
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
        }

    # Process images
    if images is not None:
        if not isinstance(images, list):
            images = [images]

        image_processor = (
            processor.image_processor if hasattr(processor, "image_processor") else None
        )
        images = [process_image(img, resize_shape, image_processor) for img in images]

        # For batching, we need uniform image sizes. Instead of padding to the
        # largest image (which adds white borders that hurt accuracy), we resize
        # all images to the model's expected input size.
        if len(images) > 1 and pad_to_uniform_size:
            # Get target size from image processor if available
            target_size = None
            if image_processor is not None and hasattr(image_processor, "size"):
                size = image_processor.size
                if isinstance(size, tuple):
                    target_size = size
                elif isinstance(size, dict):
                    target_size = (size.get("height", 384), size.get("width", 384))
                elif isinstance(size, int):
                    target_size = (size, size)

            if target_size is not None:
                # Resize all images to the target size
                resized_images = []
                for img in images:
                    if img.size != (
                        target_size[1],
                        target_size[0],
                    ):  # PIL uses (width, height)
                        img = img.resize(
                            (target_size[1], target_size[0]), Image.Resampling.BICUBIC
                        )
                    resized_images.append(img)
                images = resized_images
            else:
                # Fallback: pad to largest size (original behavior)
                max_width = max(img.width for img in images)
                max_height = max(img.height for img in images)

                padded_images = []
                for img in images:
                    if img.width != max_width or img.height != max_height:
                        padded_img = Image.new(
                            "RGB", (max_width, max_height), (255, 255, 255)
                        )
                        x_offset = (max_width - img.width) // 2
                        y_offset = (max_height - img.height) // 2
                        padded_img.paste(img, (x_offset, y_offset))
                        padded_images.append(padded_img)
                    else:
                        padded_images.append(img)
                images = padded_images

    # Process audio
    if audio is not None and len(audio) > 0:
        if not isinstance(audio, list):
            audio = [audio]

        if len(audio) > 1 and not getattr(processor, "supports_multiple_audio", False):
            print(
                "\033[33mWarning\033[0m: Single prompt with multiple audio files is not supported yet. Using the first audio file.\n"
            )
            audio = audio[:1]

        feature_extractor = getattr(processor, "feature_extractor", None)
        sr = (
            getattr(feature_extractor, "sampling_rate", 16000)
            if feature_extractor is not None
            else 16000
        )
        audio = [load_audio(audio_file, sr=sr) for audio_file in audio]

    video_fps = None
    supplied_video_metadata = kwargs.pop("video_metadata", None)
    video_metadata = None
    if has_videos:
        if not isinstance(videos, list):
            videos = [videos]
        sampling = resolve_video_sampling(processor, kwargs)
        if supplied_video_metadata is not None and len(supplied_video_metadata) != len(
            videos
        ):
            raise ValueError("Expected one video_metadata entry per video.")
        component = getattr(processor, "video_processor", None)
        frame_sampler = (
            getattr(component, "sample_frames", None)
            if getattr(component, "sample_frames_in_loader", False)
            else None
        )
        loaded, video_fps, video_metadata = [], [], []
        for video_index, v in enumerate(videos):
            if isinstance(v, (str, bytes, Path)):
                # VENDOR-DEVIATION(upstream-bugfix): upstream str()-ed bytes
                # paths into "b'/tmp/a.mp4'"; decode via the filesystem
                # encoding instead.
                arr, metadata = load_video(
                    os.fsdecode(v), sampling, frame_sampler=frame_sampler
                )
                logger.info(
                    "video %s: sampled %d of %d frames at %.2f fps "
                    "(source %.2f fps, %.1fs)",
                    v,
                    len(metadata.frames_indices),
                    metadata.total_num_frames,
                    metadata.sampled_fps,
                    metadata.fps,
                    metadata.duration,
                )
            else:
                # Already-decoded frames: nothing was sampled here, so report
                # the requested rate and describe what we were handed.
                arr = v
                metadata = VideoMetadata(
                    total_num_frames=len(v),
                    fps=sampling.fps,
                    frames_indices=list(range(len(v))),
                )
                if supplied_video_metadata is not None:
                    metadata = supplied_video_metadata[video_index]
                    if isinstance(metadata, dict):
                        metadata = VideoMetadata(**metadata)
            loaded.append(arr)
            video_fps.append(metadata.sampled_fps)
            video_metadata.append(metadata)
        videos = loaded

    model_inputs = {}

    if hasattr(processor, "image_processor") and isinstance(
        processor.image_processor, BaseImageProcessor
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]

        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        text_chunks = [
            [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            for prompt in prompts
        ]

        # Find the maximum length for padding
        max_length = max(
            sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks
        )

        # Pad and create input_ids
        input_ids = []
        for chunks in text_chunks:
            ids = chunks[0] + [image_token_index] + chunks[1]
            padding = [processor.pad_token_id] * (max_length - len(ids))
            input_ids.append(mx.array(ids + padding))

        model_inputs["input_ids"] = mx.array(input_ids)
        pixel_values = processor.image_processor.preprocess(images=images)
        model_inputs["pixel_values"] = mx.array(np.stack(pixel_values))
        model_inputs["attention_mask"] = mx.array(
            [(ids != processor.pad_token_id) for ids in input_ids]
        ).astype(mx.int32)

    else:
        if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        extra = {}
        if has_videos:
            extra["videos"] = videos
            if video_fps is not None:
                extra["fps"] = video_fps
            if video_metadata is not None:
                extra["video_metadata"] = video_metadata
        inputs = process_inputs_with_fallback(
            processor,
            images=images,
            audio=audio,
            prompts=prompts,
            add_special_tokens=add_special_tokens,
            **extra,
            **kwargs,
        )

        if "images" in inputs:
            inputs["pixel_values"] = inputs["images"]
            inputs.pop("images")

        attention_mask = inputs.get("attention_mask")
        model_inputs["attention_mask"] = (
            attention_mask
            if attention_mask is None or isinstance(attention_mask, mx.array)
            else mx.array(attention_mask)
        )

        # Convert inputs to model_inputs with mx.array if present
        for key, value in inputs.items():
            if key not in model_inputs:
                if value is None:
                    model_inputs[key] = value
                elif isinstance(value, (str, list, mx.array)):
                    model_inputs[key] = value
                else:
                    model_inputs[key] = mx.array(value)

    return model_inputs


def group_images_by_shape(
    images: List[Image.Image],
    disable_grouping: bool = False,
) -> Tuple[Dict[Tuple[int, int], List[Image.Image]], Dict[Tuple[int, int], List[int]]]:
    """
    Group images by their dimensions for efficient batch processing.

    Images with the same dimensions can be stacked and processed together,
    which is much faster than processing individually (especially on GPU).

    Args:
        images: List of PIL images to group
        disable_grouping: If True, each image gets its own group (useful for debugging)

    Returns:
        grouped_images: Dict mapping shape -> list of images with that shape
        grouped_indices: Dict mapping shape -> list of original indices

    Example:
        >>> images = [img_400x300, img_800x600, img_400x300_2]
        >>> grouped, indices = group_images_by_shape(images)
        >>> grouped
        {(300, 400): [img_400x300, img_400x300_2], (600, 800): [img_800x600]}
        >>> indices
        {(300, 400): [0, 2], (600, 800): [1]}
    """
    if disable_grouping:
        # Each image in its own group
        grouped_images = {}
        grouped_indices = {}
        for i, img in enumerate(images):
            shape = (img.height, img.width)
            # Make each shape unique by adding index
            unique_shape = (img.height, img.width, i)
            grouped_images[unique_shape] = [img]
            grouped_indices[unique_shape] = [i]
        return grouped_images, grouped_indices

    grouped_images: Dict[Tuple[int, int], List[Image.Image]] = {}
    grouped_indices: Dict[Tuple[int, int], List[int]] = {}

    for i, img in enumerate(images):
        shape = (img.height, img.width)
        if shape not in grouped_images:
            grouped_images[shape] = []
            grouped_indices[shape] = []
        grouped_images[shape].append(img)
        grouped_indices[shape].append(i)

    return grouped_images, grouped_indices


def resolve_eos_token_ids(eos_token_ids, tokenizer) -> List[int]:
    """Union configured EOS token ids with the tokenizer's own EOS.

    A checkpoint's ``eos_token_id`` can disagree with the token its chat template
    ends turns on -- Chandra OCR 2 configures ``<|endoftext|>`` but emits
    ``<|im_end|>`` -- so neither source alone is enough to stop generation.
    """
    resolved: List[int] = []
    for source in (
        eos_token_ids,
        getattr(tokenizer, "eos_token_ids", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if isinstance(source, int):
            source = [source]
        if not isinstance(source, (list, tuple, set)):
            continue
        for token_id in source:
            if isinstance(token_id, int) and token_id not in resolved:
                resolved.append(token_id)
    return resolved


class StoppingCriteria:
    def __init__(
        self,
        eos_token_ids: List[int],
        tokenizer=None,
        additional_eos_token_ids: Optional[List[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.additional_eos_token_ids = list(
            dict.fromkeys(additional_eos_token_ids or ())
        )
        self.reset(eos_token_ids)

    def add_eos_token_ids(self, new_eos_token_ids: Union[int, List[int]] = None):
        """
        Add new token IDs to the list of EOS token IDs.

        Args:
            new_eos_token_ids: Integer, string, or list of integers/strings representing token IDs to add.
                               If strings are provided, they will be converted to integers if possible.
        """
        if new_eos_token_ids is None:
            return

        if self.tokenizer is None:
            raise ValueError("Processor is not provided")

        if new_eos_token_ids is not None:
            if isinstance(new_eos_token_ids, (str, int)):
                new_eos_token_ids = [new_eos_token_ids]
            resolved = []
            for token in new_eos_token_ids:
                if isinstance(token, int):
                    resolved.append(token)
                elif isinstance(token, str):
                    resolved.append(
                        self.tokenizer.encode(" " + token, add_special_tokens=False)[-1]
                    )
            self.eos_token_ids.extend(resolved)

    def reset(self, eos_token_ids: List[int] = None):
        resolved = resolve_eos_token_ids(eos_token_ids, self.tokenizer)
        resolved.extend(
            token_id
            for token_id in self.additional_eos_token_ids
            if token_id not in resolved
        )
        if getattr(self, "eos_token_ids", None) != resolved:
            self.eos_token_ids = resolved

    def __call__(self, input_ids: mx.array) -> bool:
        return input_ids in self.eos_token_ids


class ThinkingBudgetCriteria:
    """
    Enforces a budget on thinking tokens.

    Tracks tokens within thinking blocks (between start and end tokens) and
    forces a closing sequence (e.g. ``\\n</think>``) when budget is exceeded.
    """

    def __init__(
        self,
        tokenizer,
        thinking_budget: int,
        thinking_end_token: str = "</think>",
        thinking_start_token: Optional[str] = None,
        enable_thinking: bool = False,
        prompt_preopens_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        self.prompt_preopens_thinking = prompt_preopens_thinking

        # Resolve token IDs from strings
        self.thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]

        # VENDOR-DEVIATION(upstream-bugfix): the documented default
        # ``thinking_start_token=None`` crashed at construction because
        # ``tokenizer.encode(None)`` raises; guard the encode and the span
        # comparison instead (repro against pinned upstream in
        # tests/test_mlx_vlm_vendored_generate.py).
        self.thinking_start_token_id = (
            tokenizer.encode(thinking_start_token, add_special_tokens=False)[-1]
            if thinking_start_token is not None
            else None
        )

        self._forced_sequence: List[int] = []
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            self._forced_sequence.append(newline_ids[-1])
        self._forced_sequence.append(self.thinking_end_token_id)
        self._forced_index = 0

        self.in_thinking = self.enable_thinking and self.prompt_preopens_thinking
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self.forced_token_id = None

    def reset_thinking_state(self):
        """Reset thinking state between generations."""
        self.in_thinking = self.enable_thinking and self.prompt_preopens_thinking
        self.thinking_token_count = 0
        self.budget_exceeded = False
        self._forced_index = 0
        # VENDOR-DEVIATION(upstream-bugfix): upstream left a forced token
        # captured by the previous generation pending here; a later
        # pop_forced_token_id() would inject it into the new generation
        # (repro-tested in tests/test_mlx_vlm_vendored_generate.py).
        self.forced_token_id = None

    def __call__(self, token_id: int) -> Optional[int]:
        """Process a token and return a forced token ID if budget exceeded, else None."""
        if (
            self.enable_thinking
            and self.thinking_start_token_id is not None
            and token_id == self.thinking_start_token_id
        ):
            self.in_thinking = True
            return None

        if token_id == self.thinking_end_token_id:
            self.in_thinking = False
            self.budget_exceeded = False
            self._forced_index = 0
            return None

        if self.in_thinking:
            self.thinking_token_count += 1
            if self.thinking_token_count > self.thinking_budget:
                self.budget_exceeded = True

        if self.budget_exceeded and self._forced_index < len(self._forced_sequence):
            forced = self._forced_sequence[self._forced_index]
            self._forced_index += 1
            self.forced_token_id = forced
            return forced

        self.forced_token_id = None
        return None

    def pop_forced_token_id(self) -> Optional[int]:
        """Return and clear the pending forced token ID, if any."""
        if self.forced_token_id is None or not self.enable_thinking:
            return None

        forced_token_id = self.forced_token_id
        self.forced_token_id = None
        return forced_token_id


def print_array_report(t: mx.array, label: Optional[str]) -> dict:
    """
    Return a dictionary report of an MLX array similar to PyTorch's tensor representation.
    Args:
        arr: MLX array to analyze
    Returns:
        Dictionary containing shape, dtype, value representation, and statistics
    """

    # Get basic statistics
    mean_val = mx.mean(t)
    std_val = mx.std(t)
    min_val = mx.min(t)
    max_val = mx.max(t)

    report = {
        "shape": f"{tuple(t.shape)}",
        "dtype": str(t.dtype),
        "value": repr(t),
        "mean": f"array({mean_val}, dtype={t.dtype})",
        "std": f"array({std_val}, dtype={t.dtype})",
        "min": f"array({min_val}, dtype={t.dtype})",
        "max": f"array({max_val}, dtype={t.dtype})",
        "label": label if label else "array",
    }

    # Print each field, handling 'value' specially
    print("{")
    for key, value in report.items():
        if key == "value":
            print(f" '{key}': {value},")  # No quotes around value
        else:
            print(f" '{key}': {repr(value)},")
    print("}")
    return report


def should_add_special_tokens(model_type: str, processor) -> bool:
    """Return whether tokenization should add markers outside the chat template."""
    template_owns_markers = {
        "gemma3",
        "gemma3n",
        "gemma4",
        "gemma4_unified",
        "laguna",
    }
    if model_type not in template_owns_markers:
        return True
    return getattr(processor, "chat_template", None) is None
