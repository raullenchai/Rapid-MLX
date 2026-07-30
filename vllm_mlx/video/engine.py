"""Single-worker CogVideoX generation engine.

MLX arrays and streams are deliberately confined to one persistent worker
thread. Loading on one arbitrary executor thread and generating on another can
cross MLX stream ownership boundaries and crash the process.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)
_COGVIDEOX_TOKENIZER_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"


def _has_sentencepiece_model(model_path: str | Path) -> bool:
    root = Path(model_path)
    return any(
        candidate.is_file()
        for candidate in (
            root / "tokenizer_spiece.model",
            root / "tokenizer" / "spiece.model",
            root / "spiece.model",
        )
    )


def _resolve_tokenizer_path(model_path: str, snapshot_download) -> str:
    """Return a checkpoint containing the T5 SentencePiece model.

    The converted MLX repositories currently omit ``spiece.model`` even
    though their tokenizer metadata is present. Reuse the tokenizer from the
    declared upstream base model rather than modifying the HF cache snapshot.
    """
    if _has_sentencepiece_model(model_path):
        return model_path
    logger.info(
        "CogVideoX MLX checkpoint has no spiece.model; fetching tokenizer from %s",
        _COGVIDEOX_TOKENIZER_REPO,
    )
    return snapshot_download(
        _COGVIDEOX_TOKENIZER_REPO,
        allow_patterns=[
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/added_tokens.json",
        ],
    )


class VideoBackendUnavailableError(RuntimeError):
    """Raised when the optional CogVideoX runtime is not installed."""


class VideoGenerationEngine:
    def __init__(self, model_id: str, *, output_dir: str | Path | None = None):
        self.model_id = model_id
        self.output_dir = Path(output_dir) if output_dir else None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rapid-mlx-video"
        )
        self._pipeline = None
        self._model_path: str | None = None
        self._closed = False

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        width: int = 672,
        height: int = 384,
        frames: int = 5,
        fps: int = 8,
        steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int = 42,
    ) -> Path:
        if self._closed:
            raise RuntimeError("video engine is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._generate_sync(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            ),
        )

    def generate_sync(self, *, output_path: Path, **kwargs) -> None:
        """Generate from a non-MLX caller thread using the persistent worker."""
        if self._closed:
            raise RuntimeError("video engine is closed")
        kwargs.setdefault("negative_prompt", "")
        kwargs.setdefault("steps", 50)
        kwargs.setdefault("guidance_scale", 6.0)
        generated = self._executor.submit(
            lambda: self._generate_sync(**kwargs)
        ).result()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated), output_path)

    def _load_sync(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import mlx.core as mx  # noqa: F401
            from huggingface_hub import snapshot_download
            from videox_fun_mlx.models.cogvideox_transformer3d import (
                CogVideoXTransformer3DModel,
            )
            from videox_fun_mlx.models.cogvideox_vae import (
                AutoencoderKLCogVideoX,
            )
            from videox_fun_mlx.models.t5_encoder import T5Encoder
            from videox_fun_mlx.models.tokenizer import T5Tokenizer
            from videox_fun_mlx.pipeline.pipeline_cogvideox_fun_inpaint import (
                CogVideoXFunInpaintPipeline,
            )
            from videox_fun_mlx.pipeline.scheduler import DDIMScheduler
        except ImportError as exc:
            raise VideoBackendUnavailableError(
                "CogVideoX requires the rapid-mlx[video] dependencies and the "
                "VideoX-Fun-mlx runtime. Install the extra, then install "
                "https://github.com/dgrauet/VideoX-Fun-mlx and ensure its "
                "repository root is on PYTHONPATH."
            ) from exc

        model_path = snapshot_download(self.model_id)
        logger.info("Loading CogVideoX video pipeline from %s", model_path)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path)
        transformer = CogVideoXTransformer3DModel.from_pretrained(model_path)
        text_encoder = T5Encoder.from_pretrained(model_path)
        tokenizer_path = _resolve_tokenizer_path(model_path, snapshot_download)
        tokenizer = T5Tokenizer(tokenizer_path)
        self._pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            transformer=transformer,
            scheduler=DDIMScheduler(num_inference_steps=50),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self._model_path = model_path
        return self._pipeline

    def _generate_sync(self, **kwargs) -> Path:
        import mlx.core as mx
        import numpy as np

        pipe = self._load_sync()
        height = kwargs.pop("height")
        width = kwargs.pop("width")
        frames = kwargs.pop("frames")
        fps = kwargs.pop("fps")
        steps = kwargs.pop("steps")

        video = mx.zeros((1, frames, height, width, 3))
        mask = mx.ones((1, frames, height, width, 1))
        output = pipe(
            video=video,
            mask=mask,
            num_inference_steps=steps,
            **kwargs,
        )
        mx.eval(output)
        pixels = (
            (np.asarray(output[0].astype(mx.float32)) * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        # The temporal VAE can decode a padded latent window (currently 8
        # frames for a 5-frame input). Discard only the padded tail so the
        # response honors the requested frame count exactly.
        pixels = pixels[:frames]

        target_dir = self.output_dir
        if target_dir is not None:
            target_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix="rapid-mlx-video-",
            suffix=".mp4",
            dir=target_dir,
            delete=False,
        )
        handle.close()
        output_path = Path(handle.name)
        try:
            import imageio.v3 as iio

            iio.imwrite(output_path, pixels, fps=fps, codec="libx264")
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return output_path

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
