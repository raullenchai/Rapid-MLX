"""HDR IC-LoRA pipeline.

Two-stage IC-LoRA pipeline that produces linear HDR video output via
LogC3 inverse compression, mirroring upstream
``ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline``. Subclasses
:class:`ICLoraPipeline` so it inherits low-RAM streaming, modality
tiling, bind-time LoRA fusion, and the standard two-stage decode flow.

The HDR LoRA is trained so the VAE decoder output (mapped to ``[0, 1]``)
holds the LogC3-compressed signal. ``LogC3.decompress`` then recovers
the linear HDR signal in ``[0, infinity)``.

Outputs:
    - ``<output>``: SDR mp4 preview produced by the regular streaming
      decode path (clips highlights at 1.0).
    - ``<output>.hdr.npz``: float32 ``(F, H, W, 3)`` linear HDR tensor.
      User tooling converts to EXR / TIFF / etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.hdr import apply_hdr_decode_postprocess
from ltx_core_mlx.loader.hdr_metadata import HdrLoraConfig, read_hdr_lora_config
from ltx_core_mlx.utils.memory import aggressive_cleanup

from .ic_lora import ICLoraPipeline, _resolve_lora_path

logger = logging.getLogger(__name__)

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


class HDRICLoraPipeline(ICLoraPipeline):
    """IC-LoRA pipeline with HDR LogC3 output.

    Auto-detects the HDR transform from LoRA safetensors metadata
    (matches upstream behavior). All standard IC-LoRA flags are
    inherited: ``low_ram``, ``tile_count``, ``conditioning_attention_strength``,
    ``skip_stage_2``, etc.
    """

    def __init__(
        self,
        *args,
        hdr_lora_config: HdrLoraConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize HDR IC-LoRA pipeline.

        Args:
            *args: Forwarded to :class:`ICLoraPipeline`.
            hdr_lora_config: Explicit HDR config. When ``None`` (default),
                auto-detected from the first LoRA's safetensors metadata.
            **kwargs: Forwarded to :class:`ICLoraPipeline`.

        Raises:
            ValueError: No HDR LoRA detected and ``hdr_lora_config`` is
                not provided, or LoRAs declare conflicting transforms.
        """
        super().__init__(*args, **kwargs)

        if hdr_lora_config is not None:
            self.hdr_config = hdr_lora_config
        else:
            self.hdr_config = self._detect_hdr_config()

        if self.hdr_config is None:
            raise ValueError(
                "HDRICLoraPipeline requires an HDR LoRA. None of the provided "
                "LoRAs declares an `hdr_transform` in their safetensors metadata. "
                "Pass `hdr_lora_config=HdrLoraConfig(...)` explicitly to override "
                "auto-detection, or use `ICLoraPipeline` for non-HDR LoRAs."
            )

        if self.hdr_config.reference_downscale_factor != 1:
            self.reference_downscale_factor = self.hdr_config.reference_downscale_factor

        logger.info(
            "HDR IC-LoRA: transform=%s, reference_downscale_factor=%d",
            self.hdr_config.hdr_transform,
            self.reference_downscale_factor,
        )

    def _detect_hdr_config(self) -> HdrLoraConfig | None:
        """Scan attached LoRAs for HDR metadata; return the first hit.

        Raises:
            ValueError: Two LoRAs declare conflicting HDR transforms.
        """
        detected: HdrLoraConfig | None = None
        for lora_path, _ in self._lora_paths:
            cfg = read_hdr_lora_config(lora_path)
            if cfg is None:
                continue
            if detected is None:
                detected = cfg
            elif cfg != detected:
                raise ValueError(
                    f"Conflicting HDR configurations across LoRAs: have "
                    f"{detected!r}, got {cfg!r} from {lora_path}. Cannot "
                    f"combine HDR LoRAs with different transforms."
                )
        return detected

    def _decode_to_hdr(self, video_latent: mx.array) -> mx.array:
        """VAE-decode latents and apply the HDR inverse transform.

        Mirrors upstream ``HDRICLoraPipeline._decode_to_hdr``.

        Args:
            video_latent: ``(B, 128, F', H', W')`` video latent tensor.

        Returns:
            Linear HDR float32 tensor of shape ``(F, H, W, 3)``.
        """
        assert self.vae_decoder is not None, "VAE decoder must be loaded before HDR decode"

        decoded = self.vae_decoder.decode(video_latent)  # (1, 3, F, H, W)
        normalized = (decoded.astype(mx.float32) + 1.0) * 0.5
        hdr = apply_hdr_decode_postprocess(normalized, transform=self.hdr_config.hdr_transform)

        hdr_fhwc = hdr[0].transpose(1, 2, 3, 0)
        _materialize(hdr_fhwc)
        return hdr_fhwc

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        video_conditioning: list[tuple[str, float]],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
    ) -> str:
        """Generate HDR IC-LoRA video. Saves SDR mp4 preview + HDR ``.npz``.

        Returns:
            Path to the SDR mp4 preview. The HDR tensor is saved next to
            it at ``<output_path stem>.hdr.npz``.
        """
        video_latent, audio_latent = self.generate(
            prompt=prompt,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            images=images,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=skip_stage_2,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()

        hdr_fhwc = self._decode_to_hdr(video_latent)
        hdr_path = Path(output_path).with_suffix(".hdr.npz")
        np.savez_compressed(hdr_path, video=np.asarray(hdr_fhwc))
        logger.info("HDR tensor saved: %s (shape=%s)", hdr_path, tuple(hdr_fhwc.shape))
        del hdr_fhwc
        aggressive_cleanup()

        return self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)


__all__ = ["HDRICLoraPipeline", "_resolve_lora_path"]
