# SPDX-License-Identifier: Apache-2.0
"""480p T2V orchestration on the shared persistent video worker."""

from __future__ import annotations

import gc
import json
import logging
import math
import tempfile
import time
from pathlib import Path

from ..cancel import cleanup_on_cancel
from ..engine import VideoGenerationEngine
from . import (
    ALIAS,
    REPO,
    REVISION,
    FlowMatchScheduler,
    PipelineConfig,
    validate_request,
)

logger = logging.getLogger(__name__)


class HunyuanVideo15Engine(VideoGenerationEngine):
    native_fps = 24
    validate_request = staticmethod(validate_request)

    def __init__(self, model_id):
        if str(model_id) not in (REPO, ALIAS) and not Path(model_id).is_dir():
            raise ValueError(
                "Expected the pinned HunyuanVideo 1.5 480p T2V repository or a local snapshot"
            )
        super().__init__(str(model_id))
        self.timings = {}

    def generate_sync(self, *, on_loaded=None, **kwargs):
        # Notify after actual weight loading, while keeping the base class's
        # persistent worker, atomic output staging and cancellation cleanup.
        return super().generate_sync(_on_weights_loaded=on_loaded, **kwargs)

    def _load_sync(self, cancel_check=lambda: None, on_progress=None):
        if self._model_path is None:
            if Path(self.model_id).is_dir():
                root = Path(self.model_id)
            else:
                from huggingface_hub.utils import tqdm

                from ...model_downloads import snapshot_download

                class DownloadProgress(tqdm):
                    def update(self, n=1):
                        cancel_check()
                        result = super().update(n)
                        if on_progress is not None:
                            on_progress("downloading", self.n, self.total or 0)
                        return result

                logger.info(
                    "HunyuanVideo 1.5: upstream direct checkpoint (~53 GB), native 480p T2V"
                )
                root = Path(
                    snapshot_download(
                        REPO, revision=REVISION, tqdm_class=DownloadProgress
                    )
                )
            self.config = PipelineConfig.from_path(root)
            self._model_path = str(root)
        return Path(self._model_path)

    @cleanup_on_cancel
    def _generate_sync(
        self,
        *,
        prompt,
        width=848,
        height=480,
        frames=121,
        fps=24,
        steps=50,
        guidance_scale=None,
        seed=42,
        negative_prompt="",
        cancel_token=None,
        on_progress=None,
        _on_weights_loaded=None,
    ):
        import mlx.core as mx
        import numpy as np

        from ..._mlx_cache_limit import bounded_cache_limit
        from ..encoding import encode_rgb_video
        from .loading import load_component
        from .text import encode_prompts
        from .transformer import Transformer
        from .vae import VAE

        check = cancel_token.check if cancel_token is not None else lambda: None

        def progress(stage, current, total):
            check()
            if on_progress is not None:
                on_progress(stage, current, total)

        validate_request(width=width, height=height, num_frames=frames)
        if fps != self.native_fps:
            raise ValueError("HunyuanVideo 1.5 output is fixed at 24 fps")
        if guidance_scale is not None and (
            not math.isfinite(guidance_scale) or not 1 <= guidance_scale <= 30
        ):
            raise ValueError("guidance_scale must be finite and between 1 and 30")
        if not 1 <= steps <= 1000:
            raise ValueError("steps must be between 1 and 1000")
        start = time.perf_counter()
        progress("loading", 0, 1)
        root = self._load_sync(check, on_progress)
        c = self.config
        scale = float(
            c.guider["guidance_scale"] if guidance_scale is None else guidance_scale
        )
        scheduler = FlowMatchScheduler(c.scheduler, steps)
        enabled = c.guider.get("enabled", True) and (
            scale != (0 if c.guider.get("use_original_formulation", False) else 1)
        )
        model = vae = embeddings = None
        try:
            with bounded_cache_limit(1024**3):
                # Text encoders live only during this call; all outputs are evaluated
                # before returning, so their weights are freed before loading the DiT.
                embeddings = encode_prompts(
                    root, [prompt, negative_prompt] if enabled else [prompt], c, check
                )
                check()
                model = Transformer(c.transformer)
                load_component(
                    model, root / "transformer", dtype=mx.bfloat16, cancel_check=check
                )
                self.timings["load_seconds"] = time.perf_counter() - start
                progress("loading", 1, 1)
                if _on_weights_loaded is not None:
                    _on_weights_loaded()
                latent_shape = (
                    1,
                    c.vae["latent_channels"],
                    (frames - 1) // c.vae["temporal_compression_ratio"] + 1,
                    height // c.vae["spatial_compression_ratio"],
                    width // c.vae["spatial_compression_ratio"],
                )
                latents = mx.random.normal(
                    latent_shape, key=mx.random.key(seed), dtype=mx.float32
                ).astype(mx.bfloat16)
                zeros = mx.zeros(
                    (1, c.vae["latent_channels"] + 1, *latent_shape[2:]),
                    dtype=mx.bfloat16,
                )
                mx.eval(latents, zeros)
                start = time.perf_counter()
                for index, timestep in enumerate(scheduler.timesteps):
                    progress("generating", index, steps)
                    x = mx.concatenate([latents, zeros], axis=1)
                    t = mx.array([float(timestep)], dtype=mx.bfloat16)

                    def predict(embedding, x=x, t=t):
                        return model(x, t, *embedding, cancel_check=check)

                    prediction = predict(embeddings[0])
                    if enabled and int(c.guider.get("start", 0) * steps) <= index < int(
                        c.guider.get("stop", 1) * steps
                    ):
                        uncond = predict(embeddings[1])
                        # Diffusers supports either CFG formulation via the config.
                        factor = (
                            scale + 1
                            if c.guider.get("use_original_formulation", False)
                            else scale
                        )
                        prediction = uncond + factor * (prediction - uncond)
                    latents = scheduler.step(
                        prediction.astype(mx.float32), index, latents.astype(mx.float32)
                    ).astype(mx.bfloat16)
                    mx.eval(latents)
                progress("generating", steps, steps)
                self.timings["denoise_seconds"] = time.perf_counter() - start
                # Release the DiT as well before decode. Lazy graph references must
                # go too (including the closure's last input/prediction).
                del x, prediction, predict
                uncond = None
                model = embeddings = None
                gc.collect()
                mx.clear_cache()
                check()
                start = time.perf_counter()
                vae = VAE(c.vae)
                load_component(
                    vae,
                    root / "vae",
                    dtype=mx.bfloat16,
                    key_map=lambda k: k if k.startswith("decoder.") else None,
                    cancel_check=check,
                )
                self.timings["vae_load_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                pixels = vae.tiled_decode(
                    latents / c.vae["scaling_factor"], check, on_progress
                )
                self.timings["decode_seconds"] = time.perf_counter() - start
                pixels = np.clip((pixels[0].transpose(1, 2, 3, 0) + 1) / 2, 0, 1)
                pixels = np.rint(pixels * 255).astype(np.uint8)
                check()
                self.timings["peak_memory_bytes"] = int(mx.get_peak_memory())
            target = self.output_dir
            if target is not None:
                target.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix="rapid-mlx-hunyuan-", suffix=".mp4", dir=target, delete=False
            )
            handle.close()
            path = Path(handle.name)
            try:
                progress("encoding", 0, 1)
                encode_rgb_video(pixels, path, fps)
                progress("encoding", 1, 1)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            logger.info("HunyuanVideo 1.5 metrics: %s", json.dumps(self.timings))
            return path
        finally:
            model = vae = embeddings = None
            gc.collect()
            mx.clear_cache()
