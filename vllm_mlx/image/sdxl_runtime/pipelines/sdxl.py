"""StableDiffusionXLPipeline: text-to-image with SDXL on Apple silicon, in MLX.

Wires the two CLIP text encoders, the SDXL UNet, the VAE, and a Euler scheduler.
``from_diffusers`` converts an official ``stable-diffusion-xl-base`` folder into MLX
models, so loading, encoding, denoising (with classifier-free guidance and SDXL's
size micro-conditioning), and decoding all run natively on Metal.

Tensors are channels-last. Images come back as ``(B, H, W, 3)`` in ``[-1, 1]``.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx

from ..caching import DeepCache
from ..models.autoencoder_kl_sd import AutoencoderKLSD
from ..models.clip_text import CLIPTextModel
from ..models.unet_sdxl import SDXLUNet
from ..schedulers.euler import EulerConfig, EulerDiscreteScheduler
from ..utils import prepare_image

MAX_TOKENS = 77


class StableDiffusionXLPipeline:
    """SDXL base text-to-image pipeline (channels-last ``(B, H, W, C)``)."""

    def __init__(
        self,
        unet: SDXLUNet,
        vae: AutoencoderKLSD,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModel,
        tokenizer,
        tokenizer_2,
        scheduler: EulerDiscreteScheduler,
    ):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler

    # --- loading -------------------------------------------------------------
    @classmethod
    def from_diffusers(
        cls,
        folder: str | Path,
        *,
        dtype: mx.Dtype = mx.float16,
        quantize_unet: int | None = None,
    ) -> StableDiffusionXLPipeline:
        """Load + convert a ``stable-diffusion-xl-base`` folder into MLX.

        The UNet and text encoders run in ``dtype`` (fp16 by default, SDXL's native
        precision); the VAE runs in fp32 because SDXL's VAE overflows fp16.
        ``quantize_unet`` (4/8) weight-quantizes the UNet to save memory.
        """
        from ..converters import get_converter

        folder = Path(folder)
        unet = cast(
            SDXLUNet,
            get_converter("UNet2DConditionModel").convert(
                folder / "unet", dtype=dtype, quantize=quantize_unet
            ),
        )
        vae = cast(
            AutoencoderKLSD,
            get_converter("AutoencoderKL").convert(folder / "vae", dtype=mx.float32),
        )
        te = cast(
            CLIPTextModel,
            get_converter("CLIPTextModel").convert(
                folder / "text_encoder", dtype=dtype
            ),
        )
        te2 = cast(
            CLIPTextModel,
            get_converter("CLIPTextModelWithProjection").convert(
                folder / "text_encoder_2", dtype=dtype
            ),
        )
        tok = _load_tokenizer(folder / "tokenizer")
        tok2 = _load_tokenizer(folder / "tokenizer_2")
        scheduler = EulerDiscreteScheduler(
            EulerConfig(
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="epsilon",
                timestep_spacing="leading",
                steps_offset=1,
            )
        )
        return cls(unet, vae, te, te2, tok, tok2, scheduler)

    # --- text encoding -------------------------------------------------------
    def encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Tokenize + encode with both CLIPs.

        Returns ``(prompt_embeds (1, 77, 2048), pooled (1, 1280))``: the two encoders'
        penultimate hidden states concatenated, and the bigG projected pooled embed.
        """

        def ids(tok) -> mx.array:
            enc = tok(
                prompt,
                padding="max_length",
                max_length=MAX_TOKENS,
                truncation=True,
                return_tensors="np",
            )
            return mx.array(enc["input_ids"].astype("int32"))

        hs1, _ = self.text_encoder(ids(self.tokenizer))
        hs2, pooled = self.text_encoder_2(ids(self.tokenizer_2))
        embeds = mx.concatenate([hs1[-2], hs2[-2]], axis=-1)  # (1, 77, 768+1280)
        return embeds, pooled

    def release_text_encoder_models(self) -> None:
        """Release both CLIP encoders after prompt encoding to reduce peak memory."""
        self.text_encoder = None  # type: ignore[assignment]
        self.text_encoder_2 = None  # type: ignore[assignment]
        gc.collect()
        mx.clear_cache()

    # --- generation ----------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        image=None,
        strength: float = 0.8,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 0,
        cache_interval: int = 1,
        tile_vae: bool = False,
        release_text_encoders: bool = False,
        progress: bool = True,
        on_step: Callable[[int, int], None] | None = None,
    ) -> mx.array:
        """Generate an image ``(1, height, width, 3)`` in ``[-1, 1]``.

        Pass ``image`` (a path, PIL image, or HWC/BHWC array) for image-to-image;
        ``strength`` controls how far the result may move from the input (0 < strength
        <= 1). ``height`` / ``width`` must be multiples of 8. ``cache_interval`` enables
        DeepCache (1 = off/exact; 2 ≈ 1.5-1.8x by skipping the deep UNet blocks on every
        other step). ``tile_vae`` decodes the VAE in tiles to bound memory at high res.
        ``release_text_encoders`` reclaims both CLIP encoders before denoising.
        """
        if height % 8 or width % 8:
            raise ValueError("height and width must be multiples of 8.")
        if image is not None and not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in the interval (0, 1].")

        use_cfg = guidance_scale > 1.0
        pe, pooled = self.encode_prompt(prompt)
        if use_cfg:
            npe, npooled = self.encode_prompt(negative_prompt)
            context = mx.concatenate([pe, npe], axis=0)
            text_embeds = mx.concatenate([pooled, npooled], axis=0)
        else:
            context, text_embeds = pe, pooled
        n = context.shape[0]
        if release_text_encoders:
            self.release_text_encoder_models()
        time_ids = mx.broadcast_to(
            mx.array(
                [[float(height), float(width), 0.0, 0.0, float(height), float(width)]]
            ),
            (n, 6),
        ).astype(context.dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        steps = self.scheduler.timesteps
        assert steps is not None
        if image is None:
            latents = mx.random.normal(
                (1, height // 8, width // 8, self.vae.config.latent_channels),
                key=mx.random.key(seed),
            )
            latents = latents * self.scheduler.init_noise_sigma
        else:
            input_image = prepare_image(
                image, height=height, width=width, dtype=mx.float32
            )
            if input_image.shape[0] != 1:
                raise ValueError(
                    "SDXL image-to-image currently accepts exactly one input image."
                )
            latent_key, noise_key = mx.random.split(mx.random.key(seed))
            clean_latents = (
                self.vae.encode(input_image).sample(latent_key)
                * self.vae.scaling_factor
            )
            noise = mx.random.normal(clean_latents.shape, key=noise_key)
            denoise_steps = max(
                1, min(num_inference_steps, int(num_inference_steps * strength))
            )
            begin_index = num_inference_steps - denoise_steps
            assert self.scheduler.sigmas is not None
            latents = self.scheduler.add_noise_sigma(
                clean_latents, noise, self.scheduler.sigmas[begin_index]
            )
            self.scheduler.set_begin_index(begin_index)
            steps = steps[begin_index:]
            mx.eval(latents)

        cache = DeepCache(cache_interval) if cache_interval > 1 else None
        for i, t in enumerate(steps):
            scaled = self.scheduler.scale_model_input(latents, t)
            tt = mx.broadcast_to(t, (n,))
            if use_cfg:
                x2 = mx.concatenate([scaled, scaled], axis=0)
                out = self.unet(x2, tt, context, text_embeds, time_ids, cache=cache)
                cond, uncond = out[:1], out[1:]
                noise = uncond + guidance_scale * (cond - uncond)
            else:
                noise = self.unet(
                    scaled, tt, context, text_embeds, time_ids, cache=cache
                )
            latents = self.scheduler.step(noise, t, latents)
            mx.eval(latents)
            if on_step is not None:
                on_step(i + 1, len(steps))
            if progress:
                print(f"  step {i + 1}/{len(steps)}", end="\r")
        if progress:
            print()

        image = self.vae.decode(
            latents.astype(mx.float32) / self.vae.scaling_factor, tile=tile_vae
        )
        mx.eval(image)
        return image


def _load_tokenizer(folder: Path):
    try:
        from transformers import CLIPTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The SDXL pipeline needs `transformers` for tokenization. "
            "Install it with `pip install transformers`."
        ) from exc
    return CLIPTokenizer.from_pretrained(str(folder))
