# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

# fmt: off

import gc
import logging
import math
import time
from collections.abc import Callable
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from . import model_io
from .model_io import (
    _DEFAULT_MODEL,
    load_flux,
    load_mmdit,
    load_t5_encoder,
    load_t5_tokenizer,
    load_text_encoder,
    load_tokenizer,
    load_vae_decoder,
    load_vae_encoder,
)
from .sampler import FluxSampler, ModelSamplingDiscreteFlow

logger = logging.getLogger(__name__)


def bytes2gigabytes(n: int) -> float:
    return n / 1024**3

MMDIT_CKPT = {
    "argmaxinc/mlx-stable-diffusion-3-medium": "argmaxinc/mlx-stable-diffusion-3-medium",
    "argmaxinc/mlx-stable-diffusion-3.5-large": "argmaxinc/mlx-stable-diffusion-3.5-large",
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized",
    "argmaxinc/mlx-FLUX.1-schnell": "argmaxinc/mlx-FLUX.1-schnell",
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
    "argmaxinc/mlx-FLUX.1-dev": "argmaxinc/mlx-FLUX.1-dev",
}

T5_MAX_LENGTH = {
    "argmaxinc/mlx-stable-diffusion-3-medium": 512,
    "argmaxinc/mlx-stable-diffusion-3.5-large": 512,
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 512,
    "argmaxinc/mlx-FLUX.1-schnell": 256,
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 256,
    "argmaxinc/mlx-FLUX.1-dev": 512,
}


class DiffusionPipeline:
    def __init__(
        self,
        w16: bool = False,
        shift: float = 1.0,
        use_t5: bool = True,
        model_version: str = "argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode: bool = True,
        a16: bool = False,
        local_ckpt=None,
        on_step: Callable[[int, int], None] | None = None,
    ):
        model_io.LOCAl_SD3_CKPT = local_ckpt
        self.float16_dtype = mx.float16
        model_io._FLOAT16 = self.float16_dtype
        self.dtype = self.float16_dtype if w16 else mx.float32
        self.activation_dtype = self.float16_dtype if a16 else mx.float32
        self.use_t5 = use_t5
        self.mmdit_ckpt = MMDIT_CKPT[model_version]
        self.low_memory_mode = low_memory_mode
        self.model = _DEFAULT_MODEL
        self.model_version = model_version
        self.sampler = ModelSamplingDiscreteFlow(shift=shift)
        self.latent_format = SD3LatentFormat()
        self.use_clip_g = True
        self.on_step = on_step
        self.check_and_load_models()

    def load_mmdit(self, only_modulation_dict=False):
        if only_modulation_dict:
            return load_mmdit(
                float16=True if self.dtype == self.float16_dtype else False,
                key=self.mmdit_ckpt,
                model_key=self.model_version,
                low_memory_mode=self.low_memory_mode,
                only_modulation_dict=only_modulation_dict,
            )
        self.mmdit = load_mmdit(
            float16=True if self.dtype == self.float16_dtype else False,
            key=self.mmdit_ckpt,
            model_key=self.model_version,
            low_memory_mode=self.low_memory_mode,
            only_modulation_dict=only_modulation_dict,
        )

    def check_and_load_models(self):
        if not hasattr(self, "mmdit"):
            self.load_mmdit()
        if not hasattr(self, "decoder"):
            self.decoder = load_vae_decoder(
                float16=True if self.dtype == self.float16_dtype else False,
                key=self.mmdit_ckpt,
            )
        if not hasattr(self, "clip_l"):
            self.clip_l = load_text_encoder(
                self.model,
                float16=True if self.dtype == self.float16_dtype else False,
                model_key="clip_l",
            )
            self.tokenizer_l = load_tokenizer(
                self.model,
                merges_key="tokenizer_l_merges",
                vocab_key="tokenizer_l_vocab",
                pad_with_eos=True,
            )
        if self.use_clip_g and not hasattr(self, "clip_g"):
            self.clip_g = load_text_encoder(
                self.model,
                float16=True if self.dtype == self.float16_dtype else False,
                model_key="clip_g",
            )
            self.tokenizer_g = load_tokenizer(
                self.model,
                merges_key="tokenizer_g_merges",
                vocab_key="tokenizer_g_vocab",
                pad_with_eos=False,
            )
        if self.use_t5 and not hasattr(self, "t5_encoder"):
            self.set_up_t5()

    def set_up_t5(self):
        if not hasattr(self, "t5_encoder") or self.t5_encoder is None:
            self.t5_encoder = load_t5_encoder(
                float16=True if self.dtype == self.float16_dtype else False,
                low_memory_mode=self.low_memory_mode,
            )
        if not hasattr(self, "t5_tokenizer") or self.t5_tokenizer is None:
            self.t5_tokenizer = load_t5_tokenizer(
                max_context_length=T5_MAX_LENGTH[self.model_version]
            )
        self.use_t5 = True

    def unload_t5(self):
        if self.t5_encoder is not None:
            del self.t5_encoder
            self.t5_encoder = None
        if self.t5_tokenizer is not None:
            del self.t5_tokenizer
            self.t5_tokenizer = None
        gc.collect()
        self.use_t5 = False

    def ensure_models_are_loaded(self):
        mx.eval(self.mmdit.parameters())
        mx.eval(self.clip_l.parameters())
        mx.eval(self.decoder.parameters())
        if hasattr(self, "clip_g"):
            mx.eval(self.clip_g.parameters())
        if hasattr(self, "t5_encoder") and self.use_t5:
            mx.eval(self.t5_encoder.parameters())

    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
        if negative_text is None:
            negative_text = ""
        if tokenizer.pad_with_eos:
            pad_token = tokenizer.eos_token
        else:
            pad_token = 0

        # Tokenize the text
        tokens = [tokenizer.tokenize(text)]
        if tokenizer.pad_to_max_length:
            tokens[0].extend([pad_token] * (tokenizer.max_length - len(tokens[0])))
        if negative_text is not None:
            tokens += [tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [pad_token] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        return tokens

    def encode_text(
        self,
        text: str,
        cfg_weight: float = 7.5,
        negative_text: str = "",
    ):
        tokens_l = self._tokenize(
            self.tokenizer_l,
            text,
            (negative_text if cfg_weight > 1 else None),
        )
        tokens_g = self._tokenize(
            self.tokenizer_g,
            text,
            (negative_text if cfg_weight > 1 else None),
        )

        conditioning_l = self.clip_l(tokens_l)
        conditioning_g = self.clip_g(tokens_g)
        conditioning = mx.concatenate(
            [conditioning_l.hidden_states[-2], conditioning_g.hidden_states[-2]],
            axis=-1,
        )
        pooled_conditioning = mx.concatenate(
            [conditioning_l.pooled_output, conditioning_g.pooled_output],
            axis=-1,
        )

        conditioning = mx.concatenate(
            [
                conditioning,
                mx.zeros(
                    (
                        conditioning.shape[0],
                        conditioning.shape[1],
                        4096 - conditioning.shape[2],
                    )
                ),
            ],
            axis=-1,
        )

        if self.use_t5:
            tokens_t5 = self._tokenize(
                self.t5_tokenizer,
                text,
                (negative_text if cfg_weight > 1 else None),
            )
            t5_conditioning = self.t5_encoder(tokens_t5)
            mx.eval(t5_conditioning)
        else:
            t5_conditioning = mx.zeros_like(conditioning)
        conditioning = mx.concatenate([conditioning, t5_conditioning], axis=1)

        return conditioning, pooled_conditioning

    def denoise_latents(
        self,
        conditioning,
        pooled_conditioning,
        num_steps: int = 2,
        cfg_weight: float = 0.0,
        latent_size: Tuple[int, int] = (64, 64),
        seed=None,
        image_path: Optional[str] = None,
        denoise: float = 1.0,
    ):
        # Set the PRNG state
        seed = int(time.time()) if seed is None else seed
        logger.info(f"Seed: {seed}")
        mx.random.seed(seed)

        x_T = self.get_empty_latent(*latent_size)
        if image_path is None:
            denoise = 1.0
        else:
            x_T = self.encode_image_to_latents(image_path, seed=seed)
            x_T = self.latent_format.process_in(x_T)
        noise = self.get_noise(seed, x_T)
        sigmas = self.get_sigmas(self.sampler, num_steps)
        sigmas = sigmas[int(num_steps * (1 - denoise)) :]
        extra_args = {
            "conditioning": conditioning,
            "cfg_weight": cfg_weight,
            "pooled_conditioning": pooled_conditioning,
        }
        noise_scaled = self.sampler.noise_scaling(
            sigmas[0], noise, x_T, self.max_denoise(sigmas)
        )
        latent, iter_time = sample_euler(
            CFGDenoiser(self),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
            on_step=self.on_step,
        )

        latent = self.latent_format.process_out(latent)

        return latent, iter_time

    def generate_image(
        self,
        text: str,
        num_steps: int = 2,
        cfg_weight: float = 0.0,
        negative_text: str = "",
        latent_size: Tuple[int, int] = (64, 64),
        seed=None,
        verbose: bool = True,
        image_path: Optional[str] = None,
        denoise: float = 1.0,
    ):
        # Check latent size is divisible by 2
        assert (
            latent_size[0] % 2 == 0
        ), f"Height must be divisible by 16 ({latent_size[0]*8}/16={latent_size[0]/2})"
        assert (
            latent_size[1] % 2 == 0
        ), f"Width must be divisible by 16 ({latent_size[1]*8}/16={latent_size[1]/2})"
        self.check_and_load_models()
        # Start timing
        start_time = time.time()

        # Initialize the memory log
        log = {
            "text_encoding": {
                "pre": {
                    "peak_memory": round(
                        bytes2gigabytes(mx.get_peak_memory()), 3
                    ),
                    "active_memory": round(
                        bytes2gigabytes(mx.get_active_memory()), 3
                    ),
                },
                "post": {"peak_memory": None, "active_memory": None},
            },
            "denoising": {
                "pre": {"peak_memory": None, "active_memory": None},
                "post": {"peak_memory": None, "active_memory": None},
            },
            "decoding": {
                "pre": {"peak_memory": None, "active_memory": None},
                "post": {"peak_memory": None, "active_memory": None},
            },
            "peak_memory": 0.0,
        }

        # Get the text conditioning
        text_encoding_start_time = time.time()
        if verbose:
            logger.info(
                f"Pre text encoding peak memory: {log['text_encoding']['pre']['peak_memory']}GB"
            )
            logger.info(
                f"Pre text encoding active memory: {log['text_encoding']['pre']['active_memory']}GB"
            )

        # FIXME(arda): Need the same for CLIP models (low memory mode will not succeed a second time otherwise)
        if self.use_t5 and not hasattr(self, "t5_encoder"):
            self.set_up_t5()

        conditioning, pooled_conditioning = self.encode_text(
            text, cfg_weight, negative_text
        )
        mx.eval(conditioning)
        mx.eval(pooled_conditioning)
        log["text_encoding"]["post"]["peak_memory"] = round(
            bytes2gigabytes(mx.get_peak_memory()), 3
        )
        log["text_encoding"]["post"]["active_memory"] = round(
            bytes2gigabytes(mx.get_active_memory()), 3
        )
        log["peak_memory"] = max(
            log["peak_memory"], log["text_encoding"]["post"]["peak_memory"]
        )
        log["text_encoding"]["time"] = round(time.time() - text_encoding_start_time, 3)
        if verbose:
            logger.info(
                f"Post text encoding peak memory: {log['text_encoding']['post']['peak_memory']}GB"
            )
            logger.info(
                f"Post text encoding active memory: {log['text_encoding']['post']['active_memory']}GB"
            )
            logger.info(f"Text encoding time: {log['text_encoding']['time']}s")

        # unload T5 and CLIP models after obtaining conditioning in low memory mode
        if self.low_memory_mode:
            if hasattr(self, "t5_encoder"):
                del self.t5_encoder
            if hasattr(self, "clip_g"):
                del self.clip_g
            del self.clip_l
            gc.collect()

        logger.debug(f"Conditioning dtype before casting: {conditioning.dtype}")
        logger.debug(
            f"Pooled Conditioning dtype before casting: {pooled_conditioning.dtype}"
        )
        conditioning = conditioning.astype(self.activation_dtype)
        pooled_conditioning = pooled_conditioning.astype(self.activation_dtype)
        logger.debug(f"Conditioning dtype after casting: {conditioning.dtype}")
        logger.debug(
            f"Pooled Conditioning dtype after casting: {pooled_conditioning.dtype}"
        )

        # Reset peak memory info
        mx.reset_peak_memory()

        # Generate the latents
        denoising_start_time = time.time()
        log["denoising"]["pre"]["peak_memory"] = round(
            bytes2gigabytes(mx.get_peak_memory()), 3
        )
        log["denoising"]["pre"]["active_memory"] = round(
            bytes2gigabytes(mx.get_active_memory()), 3
        )
        log["peak_memory"] = max(
            log["peak_memory"], log["denoising"]["pre"]["peak_memory"]
        )
        if verbose:
            logger.info(
                f"Pre denoise peak memory: {log['denoising']['pre']['peak_memory']}GB"
            )
            logger.info(
                f"Pre denoise active memory: {log['denoising']['pre']['active_memory']}GB"
            )

        latents, iter_time = self.denoise_latents(
            conditioning,
            pooled_conditioning,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=latent_size,
            seed=seed,
            image_path=image_path,
            denoise=denoise,
        )
        mx.eval(latents)

        log["denoising"]["post"]["peak_memory"] = round(
            bytes2gigabytes(mx.get_peak_memory()), 3
        )
        log["denoising"]["post"]["active_memory"] = round(
            bytes2gigabytes(mx.get_active_memory()), 3
        )
        log["peak_memory"] = max(
            log["peak_memory"], log["denoising"]["post"]["peak_memory"]
        )
        log["denoising"]["time"] = round(time.time() - denoising_start_time, 3)
        log["denoising"]["iter_time"] = iter_time
        if verbose:
            logger.info(
                f"Post denoise peak memory: {log['denoising']['post']['peak_memory']}GB"
            )
            logger.info(
                f"Post denoise active memory: {log['denoising']['post']['active_memory']}GB"
            )
            logger.info(f"Denoising time: {log['denoising']['time']}s")

        # unload MMDIT model after obtaining latents in low memory mode
        if self.low_memory_mode:
            del self.mmdit
            gc.collect()

        logger.debug(f"Latents dtype before casting: {latents.dtype}")
        latents = latents.astype(self.activation_dtype)
        logger.debug(f"Latents dtype after casting: {latents.dtype}")

        # Reset peak memory info
        mx.reset_peak_memory()

        # Decode the latents
        decoding_start_time = time.time()
        log["decoding"]["pre"]["peak_memory"] = round(
            bytes2gigabytes(mx.get_peak_memory()), 3
        )
        log["decoding"]["pre"]["active_memory"] = round(
            bytes2gigabytes(mx.get_active_memory()), 3
        )
        log["peak_memory"] = max(
            log["peak_memory"], log["decoding"]["pre"]["peak_memory"]
        )
        if verbose:
            logger.info(
                f"Pre decode peak memory: {log['decoding']['pre']['peak_memory']}GB"
            )
            logger.info(
                f"Pre decode active memory: {log['decoding']['pre']['active_memory']}GB"
            )
        latents = latents.astype(self.activation_dtype)
        decoded = self.decode_latents_to_image(latents)
        mx.eval(decoded)

        log["decoding"]["post"]["peak_memory"] = round(
            bytes2gigabytes(mx.get_peak_memory()), 3
        )
        log["decoding"]["post"]["active_memory"] = round(
            bytes2gigabytes(mx.get_active_memory()), 3
        )
        log["peak_memory"] = max(
            log["peak_memory"], log["decoding"]["post"]["peak_memory"]
        )
        log["decoding"]["time"] = round(time.time() - decoding_start_time, 3)
        if verbose:
            logger.info(
                f"Post decode peak memory: {log['decoding']['post']['peak_memory']}GB"
            )
            logger.info(
                f"Post decode active memory: {log['decoding']['post']['active_memory']}GB"
            )

        if verbose:
            logger.info("============= Summary =============")
            logger.info(f"Text encoder: {log['text_encoding']['time']:.1f}s")
            logger.info(f"Denoising: {log['denoising']['time']:.1f}s")
            logger.info(f"Image decoder: {log['decoding']['time']:.1f}s")
            logger.info(f"Peak memory: {log['peak_memory']:.1f}GB")

        # unload VAE Decoder model after decoding in low memory mode
        if self.low_memory_mode:
            del self.decoder
            gc.collect()

        # Convert the decoded images to uint8
        x = mx.concatenate(decoded, axis=0)
        x = (x * 255).astype(mx.uint8)

        # End timing
        end_time = time.time()
        log["total_time"] = round(end_time - start_time, 3)
        if verbose:
            logger.info(f"Total time: {log['total_time']}s")

        return Image.fromarray(np.array(x)), log

    def read_image(self, image_path: str):
        # Read the image
        img = Image.open(image_path)

        # Make sure image shape is divisible by 64
        W, H = (dim - dim % 64 for dim in (img.width, img.height))
        if W != img.width or H != img.height:
            logger.warning(
                f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}"
            )
            img = img.resize((W, H), Image.LANCZOS)  # use desired downsampling filter

        img = mx.array(np.array(img))
        img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1.0

        return mx.expand_dims(img, axis=0)

    def get_noise(self, seed, x_T):
        np.random.seed(seed)
        noise = np.random.randn(*x_T.transpose(0, 3, 1, 2).shape)
        noise = mx.array(noise).transpose(0, 2, 3, 1)
        return noise

    def get_sigmas(self, sampler, num_steps: int):
        start = sampler.timestep(sampler.sigma_max).item()
        end = sampler.timestep(sampler.sigma_min).item()
        if isinstance(sampler, FluxSampler):
            num_steps += 1
        timesteps = mx.linspace(start, end, num_steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampler.sigma(ts))
        if not isinstance(sampler, FluxSampler):
            sigs += [0.0]
        return mx.array(sigs)

    def get_empty_latent(self, *shape):
        return mx.ones([1, *shape, 16]) * 0.0609

    def max_denoise(self, sigmas):
        max_sigma = float(self.sampler.sigma_max.item())
        sigma = float(sigmas[0].item())
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def decode_latents_to_image(self, x_t):
        x = self.decoder(x_t)
        x = mx.clip(x / 2 + 0.5, 0, 1)
        return x

    def encode_image_to_latents(self, image_path: str, seed):
        image = self.read_image(image_path)
        hidden = self.encoder(image)
        mean, logvar = hidden.split(2, axis=-1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        noise = self.get_noise(seed, mean)

        return mean + std * noise


class FluxPipeline(DiffusionPipeline):
    def __init__(
        self,
        w16: bool = False,
        shift: float = 1.0,
        use_t5: bool = True,
        model_version: str = "argmaxinc/mlx-FLUX.1-schnell",
        low_memory_mode: bool = True,
        a16: bool = False,
        local_ckpt=None,
        quantize_mmdit: bool = False,
    ):
        model_io.LOCAl_SD3_CKPT = local_ckpt
        self.float16_dtype = mx.bfloat16
        model_io._FLOAT16 = self.float16_dtype
        self.dtype = self.float16_dtype if w16 else mx.float32
        self.activation_dtype = self.float16_dtype if a16 else mx.float32
        self.mmdit_ckpt = MMDIT_CKPT[model_version]
        self.low_memory_mode = low_memory_mode
        self.model = _DEFAULT_MODEL
        self.model_version = model_version
        self.sampler = FluxSampler(shift=shift)
        self.latent_format = FluxLatentFormat()
        self.use_t5 = True
        self.use_clip_g = False
        self.quantize_mmdit = quantize_mmdit
        self.check_and_load_models()

    def load_mmdit(self, only_modulation_dict=False):
        if only_modulation_dict:
            return load_flux(
                key=self.mmdit_ckpt,
                model_key=self.model_version,
                float16=True if self.dtype == self.float16_dtype else False,
                low_memory_mode=self.low_memory_mode,
                only_modulation_dict=only_modulation_dict,
            )
        self.mmdit = load_flux(
            key=self.mmdit_ckpt,
            model_key=self.model_version,
            float16=True if self.dtype == self.float16_dtype else False,
            low_memory_mode=self.low_memory_mode,
            only_modulation_dict=only_modulation_dict,
        )

    def encode_text(
        self,
        text: str,
        cfg_weight: float = 7.5,
        negative_text: str = "",
    ):
        tokens_l = self._tokenize(
            self.tokenizer_l,
            text,
            (negative_text if cfg_weight > 1 else None),
        )
        conditioning_l = self.clip_l(tokens_l[[0], :])  # Ignore negative text
        pooled_conditioning = conditioning_l.pooled_output

        tokens_t5 = self._tokenize(
            self.t5_tokenizer,
            text,
            (negative_text if cfg_weight > 1 else None),
        )
        padded_tokens_t5 = mx.zeros((1, T5_MAX_LENGTH[self.model_version])).astype(
            tokens_t5.dtype
        )
        padded_tokens_t5[:, : tokens_t5.shape[1]] = tokens_t5[
            [0], :
        ]  # Ignore negative text
        t5_conditioning = self.t5_encoder(padded_tokens_t5)
        mx.eval(t5_conditioning)
        conditioning = t5_conditioning

        return conditioning, pooled_conditioning


class CFGDenoiser(nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model: DiffusionPipeline):
        super().__init__()
        self.model = model

    def cache_modulation_params(self, pooled_text_embeddings, sigmas):
        self.model.mmdit.cache_modulation_params(
            pooled_text_embeddings, sigmas.astype(self.model.activation_dtype)
        )

    def clear_cache(self):
        self.model.mmdit.load_weights(
            self.model.load_mmdit(only_modulation_dict=True), strict=False
        )

    def __call__(
        self,
        x_t,
        timestep,
        sigma,
        conditioning,
        cfg_weight: float = 7.5,
        pooled_conditioning=None,
    ):
        if cfg_weight <= 0:
            logger.debug("CFG Weight disabled")
            x_t_mmdit = x_t.astype(self.model.activation_dtype)
        else:
            x_t_mmdit = mx.concatenate([x_t] * 2, axis=0).astype(
                self.model.activation_dtype
            )
        mmdit_input = {
            "latent_image_embeddings": x_t_mmdit,
            "token_level_text_embeddings": mx.expand_dims(conditioning, 2),
            "timestep": mx.broadcast_to(timestep, [len(x_t_mmdit)]),
        }

        mmdit_output = self.model.mmdit(**mmdit_input)
        eps_pred = self.model.sampler.calculate_denoised(sigma, mmdit_output, x_t_mmdit)
        if cfg_weight <= 0:
            return eps_pred
        else:
            eps_text, eps_neg = eps_pred.split(2)
            return eps_neg + cfg_weight * (eps_text - eps_neg)


class LatentFormat:
    """Base class for latent format conversion"""

    def __init__(self):
        self.scale_factor = 1.0
        self.shift_factor = 0.0

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class SD3LatentFormat(LatentFormat):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609


class FluxLatentFormat(LatentFormat):
    def __init__(self):
        super().__init__()
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def sample_euler(
    model: CFGDenoiser,
    x,
    sigmas,
    extra_args=None,
    on_step: Callable[[int, int], None] | None = None,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args

    timesteps = model.model.sampler.timestep(sigmas).astype(
        model.model.activation_dtype
    )
    model.cache_modulation_params(extra_args.pop("pooled_conditioning"), timesteps)

    iter_time = []
    total = len(sigmas) - 1
    try:
        for i in range(total):
            start_time = time.perf_counter()
            denoised = model(x, timesteps[i], sigmas[i], **extra_args)
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            # Euler method
            x = x + d * dt
            mx.eval(x)
            end_time = time.perf_counter()
            iter_time.append(round((end_time - start_time), 3))
            if on_step is not None:
                on_step(i + 1, total)
    finally:
        # ``cache_modulation_params`` replaces live modulation weights with
        # timestep-specific values. Cancellation and model errors must restore
        # them too or the retained model poisons the next request.
        model.clear_cache()

    return x, iter_time

# fmt: on
