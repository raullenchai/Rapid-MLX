"""CogVideoX-Fun Inpaint Pipeline for MLX.

Complete pipeline matching the original VideoX-Fun implementation:
- Classifier-free guidance (CFG)
- Correct RoPE (grid_type="slice" for CogVideoX 1.5)
- Proper VAE decode normalization
- Dynamic CFG support
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
from videox_fun_mlx.models.embeddings import get_3d_rotary_pos_embed
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler


# VAE spatial downscale factor (4 down blocks with stride 2 = 2^3 = 8 for 3 non-final blocks)
VAE_SCALE_FACTOR_SPATIAL = 8


def _resize_mask_to_latent(mask: mx.array, latent_shape: tuple) -> mx.array:
    """Resize a binary mask to match latent spatial dimensions via nearest neighbor."""
    B, D, H, W, _ = mask.shape
    _, D_t, H_t, W_t, _ = latent_shape

    mask_np = np.array(mask)
    d_idx = np.round(np.linspace(0, D - 1, D_t)).astype(int)
    h_idx = np.round(np.linspace(0, H - 1, H_t)).astype(int)
    w_idx = np.round(np.linspace(0, W - 1, W_t)).astype(int)
    resized = mask_np[:, d_idx][:, :, h_idx][:, :, :, w_idx]
    return mx.array(resized)


class CogVideoXFunInpaintPipeline:
    """Video inpainting pipeline for CogVideoX-Fun on MLX."""

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: DDIMScheduler,
        text_encoder=None,
        tokenizer=None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_cfg: bool = True,
        max_length: int = 226,
    ) -> mx.array:
        """Encode text prompt (and negative prompt for CFG) using T5.

        Returns:
            If do_cfg: (2, max_length, d_model) — [negative, positive] stacked.
            Else: (1, max_length, d_model).
        """
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("text_encoder and tokenizer required")

        pos_embeds = self.text_encoder(self.tokenizer(prompt, max_length=max_length))

        if do_cfg:
            neg_embeds = self.text_encoder(self.tokenizer(negative_prompt, max_length=max_length))
            return mx.concatenate([neg_embeds, pos_embeds], axis=0)

        return pos_embeds

    def _prepare_rotary_embeddings(
        self,
        height: int,
        width: int,
        num_latent_frames: int,
    ) -> Optional[Tuple[mx.array, mx.array]]:
        """Compute 3D RoPE matching the original pipeline exactly."""
        if not self.transformer._config.get("use_rotary_positional_embeddings"):
            return None

        p = self.transformer._config["patch_size"]
        p_t = self.transformer._config.get("patch_size_t")

        grid_height = height // (VAE_SCALE_FACTOR_SPATIAL * p)
        grid_width = width // (VAE_SCALE_FACTOR_SPATIAL * p)

        if p_t is None:
            # CogVideoX 1.0: linspace grid
            base_h = self.transformer.patch_embed.sample_height // p
            base_w = self.transformer.patch_embed.sample_width // p
            crops = ((0, 0), (base_h, base_w))
            return get_3d_rotary_pos_embed(
                embed_dim=self.transformer.transformer_blocks[0].attn1.dim_head,
                crops_coords=crops,
                grid_size=(grid_height, grid_width),
                temporal_size=num_latent_frames,
            )
        else:
            # CogVideoX 1.5: slice grid
            base_h = self.transformer.patch_embed.sample_height // p
            base_w = self.transformer.patch_embed.sample_width // p
            base_num_frames = (num_latent_frames + p_t - 1) // p_t
            return get_3d_rotary_pos_embed(
                embed_dim=self.transformer.transformer_blocks[0].attn1.dim_head,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_h, base_w),
            )

    def __call__(
        self,
        video: mx.array,
        mask: mx.array,
        prompt: Optional[str] = None,
        negative_prompt: str = "",
        prompt_embeds: Optional[mx.array] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Run video inpainting.

        Args:
            video: (B, D, H, W, C) input video in channels-last, values in [0, 1].
            mask: (B, D, H, W, 1) binary mask (1 = inpaint region).
            prompt: Text prompt.
            negative_prompt: Negative prompt for CFG.
            prompt_embeds: Pre-computed embeddings. If CFG, shape (2*B, L, D).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale. 1.0 = no guidance.
            seed: Random seed.

        Returns:
            (B, D_out, H, W, C) output video, values in [0, 1].
        """
        do_cfg = guidance_scale > 1.0
        B, D_vid, H_vid, W_vid, C_vid = video.shape

        # 1. Encode prompt
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_embeds must be provided")
            prompt_embeds = self.encode_prompt(prompt, negative_prompt, do_cfg)
            mx.eval(prompt_embeds)

        if seed is not None:
            mx.random.seed(seed)

        # 2. Normalize video to [-1, 1] for VAE and encode
        video_norm = video * 2 - 1  # [0,1] -> [-1,1]
        posterior = self.vae.encode(video_norm)
        latents = posterior.mode() * self.vae.scaling_factor
        latent_cf = latents.transpose(0, 1, 4, 2, 3)  # NDHWC -> NFCHW
        B_lat, F_lat, C_lat, H_lat, W_lat = latent_cf.shape

        # 3. Prepare inpaint conditioning
        is_full_mask = mx.mean(mask).item() > 0.99
        if is_full_mask:
            mask_latent_1ch = mx.zeros((B_lat, F_lat, 1, H_lat, W_lat))
            masked_video_latents_cf = mx.zeros((B_lat, F_lat, C_lat, H_lat, W_lat))
        else:
            masked_video = video_norm * (1 - mask)  # mask in [-1,1] space
            masked_posterior = self.vae.encode(masked_video)
            masked_video_latents = masked_posterior.mode() * self.vae.scaling_factor
            masked_video_latents_cf = masked_video_latents.transpose(0, 1, 4, 2, 3)
            mask_latent = _resize_mask_to_latent(mask, latents.shape)
            mask_latent_1ch = mask_latent.transpose(0, 1, 4, 2, 3)

        inpaint_cf = mx.concatenate([mask_latent_1ch, masked_video_latents_cf], axis=2)

        # For CFG: duplicate inpaint conditioning
        if do_cfg:
            inpaint_cf = mx.concatenate([inpaint_cf, inpaint_cf], axis=0)

        # 4. Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 5. Start from noise
        noise = mx.random.normal(latent_cf.shape)
        current = self.scheduler.add_noise(latent_cf, noise, self.scheduler.timesteps[0])

        # 6. RoPE
        image_rotary_emb = self._prepare_rotary_embeddings(H_vid, W_vid, F_lat)

        # 7. Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # CFG: duplicate latents
            if do_cfg:
                latent_model_input = mx.concatenate([current, current], axis=0)
            else:
                latent_model_input = current

            t_input = mx.broadcast_to(mx.array([float(t)]), (latent_model_input.shape[0],))

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input,
                inpaint_latents=inpaint_cf,
                image_rotary_emb=image_rotary_emb,
            )

            # CFG: combine unconditional and conditional predictions
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = mx.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            current = self.scheduler.step(noise_pred, t, current)
            mx.eval(current)

        # 8. Decode latents
        decoded_latents = current.transpose(0, 1, 3, 4, 2)  # NFCHW -> NDHWC
        decoded_latents = decoded_latents / self.vae.scaling_factor
        output = self.vae.decode(decoded_latents)

        # 9. Normalize to [0, 1] (matching diffusers: (x / 2 + 0.5).clamp(0, 1))
        output = mx.clip(output / 2 + 0.5, 0, 1)

        return output

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pipeline from a pretrained model directory."""
        import os
        from videox_fun_mlx.models.t5_encoder import T5Encoder
        from videox_fun_mlx.models.tokenizer import T5Tokenizer

        vae = AutoencoderKLCogVideoX.from_pretrained(model_path)
        transformer = CogVideoXTransformer3DModel.from_pretrained(model_path)

        text_encoder = None
        tokenizer = None
        t5_file = os.path.join(model_path, "text_encoder.safetensors")
        t5_dir = os.path.join(model_path, "text_encoder")
        if os.path.exists(t5_file) or os.path.isdir(t5_dir):
            print("Loading T5 text encoder...")
            text_encoder = T5Encoder.from_pretrained(model_path)
        spiece_candidates = [
            os.path.join(model_path, "tokenizer_spiece.model"),
            os.path.join(model_path, "tokenizer", "spiece.model"),
        ]
        for sp in spiece_candidates:
            if os.path.exists(sp):
                tokenizer = T5Tokenizer(model_path)
                break

        scheduler_config = {}
        sched_file = os.path.join(model_path, "scheduler_scheduler_config.json")
        if os.path.exists(sched_file):
            import json

            with open(sched_file) as f:
                scheduler_config = json.load(f)
            for k in ("_class_name", "_diffusers_version", "trained_betas"):
                scheduler_config.pop(k, None)
        scheduler_config.update(kwargs)
        scheduler = DDIMScheduler(**scheduler_config)

        return cls(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
