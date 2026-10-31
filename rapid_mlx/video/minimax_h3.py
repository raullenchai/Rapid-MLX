# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 video and synchronized audio adapter."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
REPO = "MiniMaxAI/MiniMax-H3"
REVISION = "b3c7290e66afdf293bef3b9077b7a266ef421f34"
TURBO_REPO = "lightx2v/Minimax-h3-Turbo"
TURBO_REVISION = "30f485ed874aa44c8183348ad361c5eabad318da"
TURBO_FILE = "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors"

# Exact data files in the pinned FL2VA plan. In particular, never fetch repo code.
FILES = (
    "LICENSE",
    "README.md",
    "model_index.json",
    "modular_model_index.json",
    "audio_scheduler/scheduler_config.json",
    "scheduler/scheduler_config.json",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    *(
        f"processor/{name}"
        for name in (
            "chat_template.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        )
    ),
    *(
        f"tokenizer/{name}"
        for name in (
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        )
    ),
    *(
        f"text_encoder/{name}"
        for name in (
            "chat_template.json",
            "config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        )
    ),
    *(f"text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)),
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    *(
        f"transformer/diffusion_pytorch_model-{i:05d}-of-00014.safetensors"
        for i in range(1, 15)
    ),
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
    *(f"vae/diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in range(1, 4)),
)


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()


class MiniMaxH3BackendError(RuntimeError):
    """Safe generation failure."""


class MiniMaxH3RequestError(ValueError):
    """Invalid generation request."""


def _merge_turbo_lora(transformer: Any, path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors import safe_open

    # The scale is the file's metadata alpha over the rank (8/128); the
    # rank-sized alpha in the upstream example config renders static.
    with safe_open(str(path), "np") as reader:
        metadata = reader.metadata() or {}
    try:
        alpha = float(metadata["alpha"])
        if not 0 < alpha < float("inf"):
            raise ValueError("invalid alpha")
    except (KeyError, ValueError, TypeError) as exc:
        raise MiniMaxH3BackendError("MiniMax-H3 Turbo LoRA has invalid alpha") from exc
    weights = dict(tree_flatten(transformer.parameters()))
    lora = mx.load(str(path))
    suffix = ".lora_A.default.weight"
    merged_count = 0
    for name, a in lora.items():
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        key = prefix + ".weight"
        b = lora[prefix + ".lora_B.default.weight"]
        weight = weights.pop(key)
        merged = (
            weight.astype(mx.float32)
            + (alpha / a.shape[0]) * (b.astype(mx.float32) @ a.astype(mx.float32))
        ).astype(weight.dtype)
        mx.eval(merged)
        # Swap one tensor at a time so the old copy is freed before the next.
        transformer.load_weights([(key, merged)], strict=False)
        del weight, merged
        mx.clear_cache()
        merged_count += 1
    if not merged_count:
        raise MiniMaxH3BackendError("MiniMax-H3 Turbo LoRA has no adapter weights")


def is_minimax_h3_model(name: str | None) -> bool:
    return bool(name and name.rstrip("/").rsplit("/", 1)[-1].casefold() == "minimax-h3")


class MiniMaxH3VideoEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._stopping = threading.Event()
        # MLX streams are thread-local and the pipeline keeps arrays created
        # while loading, so every load and generation must run on one thread;
        # a request served from another thread fails with "There is no
        # Stream(cpu, N) in current thread".
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-minimax-h3"
        )

    @staticmethod
    def validate_request(
        *, width: int, height: int, num_frames: int, image: Path | None
    ) -> None:
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise MiniMaxH3RequestError(
                "MiniMax-H3 dimensions must be positive multiples of 32"
            )
        if not 120 <= num_frames <= 360 or (num_frames - 5) % 17:
            raise MiniMaxH3RequestError(
                "MiniMax-H3 frames must be 17n+5 and between 120 and 360"
            )

    def stop(self) -> None:
        self._stopping.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def generate(self, **kwargs) -> None:
        try:
            future = self._executor.submit(self._generate, **kwargs)
        except RuntimeError as exc:  # executor shut down by stop()
            raise MiniMaxH3BackendError("MiniMax-H3 generation cancelled") from exc
        try:
            future.result()
        # A job still queued when stop() shuts the executor down is cancelled.
        except concurrent.futures.CancelledError as exc:
            raise MiniMaxH3BackendError("MiniMax-H3 generation cancelled") from exc

    def _generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        image: Path | None,
        last_image: Path | None = None,
        on_loaded=None,
    ) -> None:
        self.validate_request(
            width=width, height=height, num_frames=num_frames, image=image
        )
        if fps != 24:
            raise MiniMaxH3RequestError("MiniMax-H3 output is fixed at 24 fps")
        if last_image is not None and image is None:
            raise MiniMaxH3RequestError("last image requires a first image")

        def progress(*_args):
            if self._stopping.is_set():
                raise MiniMaxH3BackendError("MiniMax-H3 generation cancelled")

        progress()
        try:
            from mlx_vlm.generate import (
                VideoGenerationRequest,
                generate_video,
                load_video_generation_model,
            )

            if self._model is None:
                from .._mirror import pinned_snapshot_download

                local = Path(self.model_name).expanduser()
                path = (
                    str(local)
                    if local.is_dir()
                    else pinned_snapshot_download(
                        REPO, REVISION, allow_patterns=list(FILES)
                    )
                )
                progress()
                # Both workflows use the same FL2VA pipeline; the wrapper's
                # workflow only labels results, so one loaded model serves both.
                turbo = pinned_snapshot_download(
                    TURBO_REPO, TURBO_REVISION, allow_patterns=[TURBO_FILE]
                )
                model = load_video_generation_model(str(path), workflow="fl2va")
                _merge_turbo_lora(model.pipeline.transformer, Path(turbo) / TURBO_FILE)
                # The 8-step adapter was distilled with a video shift of 6.
                model.pipeline.scheduler.set_shift(6.0)
                self._model = model
                if on_loaded:
                    on_loaded()
            progress()
            self._model.workflow = "fl2va" if image is not None else "t2va"
            request = VideoGenerationRequest(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                seed=seed,
                # mlx-vlm spaces `steps` sigmas, so 9 gives the adapter's 8
                # denoising forwards.
                steps=9,
                image=image,
                last_image=last_image,
                progress_callback=progress,
                # Keep the AdaLN weights after caching their projections: the
                # cache is keyed by the timestep table, which differs between
                # text- and frame-conditioned runs, so dropped weights would
                # make the next request in the other mode fail until reload.
                extra={"drop_adaln_weights": False},
            )
            output_path.unlink(missing_ok=True)
            result = generate_video(self._model, request)
            progress()
            import numpy as np

            from .encoding import encode_rgb_video

            frames = np.array(result.frames)
            if frames.ndim == 5 and frames.shape[0] == 1:
                frames = frames[0]
            if np.issubdtype(frames.dtype, np.floating):
                frames = np.rint(np.clip(frames, 0.0, 1.0) * 255)
            frames = np.ascontiguousarray(np.clip(frames, 0, 255).astype(np.uint8))
            encode_rgb_video(
                frames,
                output_path,
                int(round(result.fps or fps)),
                audio=None
                if result.audio is None
                else np.array(result.audio, dtype=np.float32),
                sampling_rate=result.sampling_rate,
            )
            if not output_path.is_file() or not output_path.stat().st_size:
                raise MiniMaxH3BackendError(
                    "MiniMax-H3 generation completed without an MP4 output"
                )
        except MiniMaxH3BackendError:
            output_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            logger.exception("MiniMax-H3 generation failed")
            raise MiniMaxH3BackendError(
                f"MiniMax-H3 generation failed ({type(exc).__name__}); check server logs"
            ) from exc
        finally:
            # A job leaves tens of GB of activations in MLX's buffer cache;
            # release them so the next job starts from the resident weights.
            _clear_mlx_cache()
