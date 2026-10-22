"""Keyframe interpolation pipeline — two-stage interpolation between reference frames.

Matches reference architecture: stage 1 at half resolution with optional CFG guidance,
neural upscale 2x, then stage 2 refinement at full resolution.

Keyframe images are re-encoded by the VAE at each stage's resolution (matching
reference ``image_conditionings_by_adding_guiding_latent``), rather than
downsampling pre-encoded latents.

Ported from ltx-pipelines keyframe_interpolation.py
"""

from __future__ import annotations

import mlx.core as mx
from PIL import Image

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop


def _encode_keyframe(
    vae_encoder: VideoEncoder,
    image: Image.Image | str,
    height: int,
    width: int,
) -> mx.array:
    """Encode a keyframe image at a specific resolution.

    Args:
        vae_encoder: VAE encoder.
        image: PIL Image or path.
        height: Target pixel height.
        width: Target pixel width.

    Returns:
        Patchified keyframe tokens (1, H*W, 128).
    """
    img_tensor = prepare_image_for_encoding(image, height, width)
    # (1, 3, H, W) -> (1, 3, 1, H, W) for single-frame video encoding
    latent = vae_encoder.encode(img_tensor[:, :, None, :, :])
    mx.eval(latent)  # Force evaluation to avoid graph buildup
    # (1, 128, 1, H', W') -> (1, H'*W', 128) tokens
    tokens = latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
    return tokens


class KeyframeInterpolationPipeline(TI2VidTwoStagesPipeline):
    """Two-stage keyframe interpolation pipeline.

    Stage 1: Generate at half resolution with keyframe conditioning + optional CFG.
             When ``dev_transformer`` is specified, uses the non-distilled model
             for higher quality interpolation (matching the reference).
    Stage 2: Neural upscale 2x, re-apply keyframe conditioning at full resolution,
             refine with distilled model (dev + LoRA fusion, or standalone distilled).

    Args:
        model_dir: Path to model weights.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Filename of the dev (non-distilled) transformer weights
            inside model_dir (e.g. ``transformer-dev.safetensors``). When provided,
            stage 1 uses this model and stage 2 fuses the distilled LoRA on top.
        distilled_lora: Filename of the distilled LoRA weights inside model_dir.
            Required when ``dev_transformer`` is set.
        distilled_lora_strength: Strength for the distilled LoRA fusion (default 1.0).
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        dev_transformer: str | None = None,
        distilled_lora: str | None = None,
        distilled_lora_strength: float = 1.0,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )
        self._dev_transformer = dev_transformer
        self._distilled_lora = distilled_lora
        self._distilled_lora_strength = distilled_lora_strength

    def interpolate(
        self,
        prompt: str,
        keyframe_images: list[Image.Image | str],
        keyframe_indices: list[int],
        keyframe_strengths: list[float] | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = 1.0,
        negative_prompt_embeds: tuple[mx.array, mx.array] | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video interpolating between keyframes using two-stage pipeline.

        Args:
            prompt: Text prompt.
            keyframe_images: List of keyframe images (PIL or paths).
            keyframe_indices: Pixel frame indices for each keyframe (0-based).
            keyframe_strengths: Per-keyframe conditioning strength in ``[0, 1]``
                (matches upstream ``ImageConditioningInput.strength``). Defaults
                to ``1.0`` for each keyframe.
            height: Final video height.
            width: Final video width.
            num_frames: Total number of pixel frames.
            frame_rate: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.
            cfg_scale: CFG guidance scale for stage 1 (1.0 = no guidance).
            negative_prompt_embeds: Optional (video_neg, audio_neg) for CFG.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        if keyframe_strengths is None:
            keyframe_strengths = [1.0] * len(keyframe_images)
        elif len(keyframe_strengths) != len(keyframe_images):
            raise ValueError(
                f"keyframe_strengths length ({len(keyframe_strengths)}) must match "
                f"keyframe_images length ({len(keyframe_images)})"
            )

        # Compute half-res latent dimensions (matching reference: height//2, width//2
        # with integer division by spatial compression factor 32).
        half_h, half_w = height // 2, width // 2
        F_half, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)

        # VAE-compatible encoding resolution (latent dims * 32, always 32-aligned)
        enc_h_half = H_half * 32
        enc_w_half = W_half * 32

        # Upscaled resolution (upsampler doubles spatial latent dims)
        H_up, W_up = H_half * 2, W_half * 2
        up_h, up_w = H_up * 32, W_up * 32

        # --- Encode keyframes at both resolutions via ImageConditioner block ---
        _materialize = getattr(mx, "eval")  # noqa: B009

        def _encode_all_keyframes(encoder) -> tuple[list, list]:
            half = [_encode_keyframe(encoder, img, enc_h_half, enc_w_half) for img in keyframe_images]
            full = [_encode_keyframe(encoder, img, up_h, up_w) for img in keyframe_images]
            _materialize(*(half + full))  # materialize before encoder is freed
            return half, full

        kf_tokens_half, kf_tokens_full = self.image_conditioner(_encode_all_keyframes, free_after=True)
        aggressive_cleanup()

        # --- Text encoding (load Gemma, encode, free) ---
        use_dev = self._dev_transformer is not None
        if cfg_scale != 1.0:
            video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)
        else:
            self._load_text_encoder()
            video_embeds, audio_embeds = self._encode_text(prompt)
            neg_video_embeds = None
            neg_audio_embeds = None
            mx.eval(video_embeds, audio_embeds)
            # Free text encoder before loading transformer
            self.prompt_encoder.free()
            aggressive_cleanup()

        # --- Load transformer (dev model required) + upsampler only ---
        # The distilled model hallucinates during keyframe interpolation.
        # The dev model + CFG is required for quality results.
        if not use_dev:
            raise ValueError(
                "Keyframe interpolation requires the dev (non-distilled) model. "
                "The distilled model hallucinates unrelated content during interpolation.\n"
                "Use: --dev-transformer transformer-dev.safetensors "
                "--distilled-lora <distilled LoRA of your model> --cfg-scale 3.0\n"
                f"Model dir: {self.model_dir} (must contain transformer-dev.safetensors plus the "
                "version-appropriate distilled LoRA, e.g. ltx-2.5-22b-distilled-lora-450.safetensors "
                "for LTX-2.5 or ltx-2.3-22b-distilled-lora-384.safetensors for LTX-2.3, "
                "e.g. dgrauet/ltx-2.3-mlx-q8)"
            )
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with keyframe conditioning ---
        F = F_half  # already computed above
        video_shape_1 = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # Build keyframe conditioning items at half resolution.
        # Each keyframe encodes a single pixel frame.
        video_kf_conditions_half = [
            VideoConditionByKeyframeIndex(
                frame_idx=kf_idx,
                keyframe_latent=tokens,
                spatial_dims=(F, H_half, W_half),
                frame_rate=frame_rate,
                strength=kf_strength,
                num_pixel_frames=1,
            )
            for tokens, kf_idx, kf_strength in zip(kf_tokens_half, keyframe_indices, keyframe_strengths)
        ]

        # Build noised state via the canonical upstream order:
        #     init (zeros) -> apply conditionings -> noise.
        # The noiser respects denoise_mask: keyframe tokens (mask=0) stay
        # clean, generation tokens (mask=1) get pure noise at sigma=1.0.
        video_state_1 = create_noised_state(
            base_shape=video_shape_1,
            conditionings=video_kf_conditions_half,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
        )
        audio_state_1 = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused (no conditionings)
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
        )

        # Stage 1 sigma schedule: dev model uses LTX2Scheduler (dynamic schedule),
        # distilled model uses predefined DISTILLED_SIGMAS.
        if use_dev:
            s1_steps = stage1_steps or 20  # Reference default for non-distilled
            num_tokens = F * H_half * W_half
            sigmas_1 = ltx2_schedule(s1_steps, num_tokens=num_tokens)
        else:
            sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        if cfg_scale != 1.0 or video_guider_params is not None:
            # Use explicitly provided negative embeds, or the auto-encoded DEFAULT_NEGATIVE_PROMPT
            video_neg = negative_prompt_embeds[0] if negative_prompt_embeds else neg_video_embeds
            audio_neg = negative_prompt_embeds[1] if negative_prompt_embeds else neg_audio_embeds

            # Pass through guider params (STG, rescale, modality).
            if video_guider_params is not None:
                vgp = video_guider_params
            else:
                vgp = MultiModalGuiderParams(cfg_scale=cfg_scale)
            if audio_guider_params is not None:
                agp = audio_guider_params
            else:
                agp = MultiModalGuiderParams(cfg_scale=cfg_scale)
            video_factory = create_multimodal_guider_factory(vgp, negative_context=video_neg)
            audio_factory = create_multimodal_guider_factory(agp, negative_context=audio_neg)

            self._pre_denoise_flush(video_state_1, audio_state_1)
            output_1 = guided_denoise_loop(
                model=x0_model,
                video_state=video_state_1,
                audio_state=audio_state_1,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                video_guider_factory=video_factory,
                audio_guider_factory=audio_factory,
                sigmas=sigmas_1,
            )
        else:
            self._pre_denoise_flush(video_state_1, audio_state_1)
            output_1 = denoise_loop(
                model=x0_model,
                video_state=video_state_1,
                audio_state=audio_state_1,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
            )
        if self.low_memory:
            aggressive_cleanup()

        # Extract generated tokens (without appended keyframe tokens)
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]

        # --- Fuse distilled LoRA for stage 2 (if using dev model) ---
        if use_dev and self._distilled_lora:
            self._fuse_distilled_lora(self.dit)

        # --- Upscale with normalize/denormalize wrapping (matching reference) ---
        # Reference: un_normalize -> upsampler -> normalize using VAE encoder stats.
        # Without this, the upsampler produces grid/weave artifacts.
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))

        # Reload encoder for normalization stats (encoder was freed earlier).
        # unpatchify returns PyTorch layout (B, C, F, H, W); stats expect MLX (B, F, H, W, C).
        def _denorm_upscale_renorm(encoder) -> mx.array:
            v_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
            v_denorm = encoder.denormalize_latent(v_mlx).transpose(0, 4, 1, 2, 3)
            v_up = self.upsampler(v_denorm)
            v_up_mlx = encoder.normalize_latent(v_up.transpose(0, 2, 3, 4, 1))
            return v_up_mlx.transpose(0, 4, 1, 2, 3)

        video_upscaled = self.image_conditioner(_denorm_upscale_renorm, free_after=True)
        mx.async_eval(video_upscaled)
        if self.low_memory:
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Upscaled resolution with keyframe conditioning ---
        # H_up/W_up already computed above from H_half*2, W_half*2
        H_full, W_full = H_up, W_up
        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        # Stage 2 orchestration matches upstream `create_noised_state`:
        #     init (initial_latent=upscaled) -> apply conditionings -> noise.
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        # Build keyframe conditioning items at full resolution.
        video_kf_conditions_full = [
            VideoConditionByKeyframeIndex(
                frame_idx=kf_idx,
                keyframe_latent=tokens,
                spatial_dims=(F, H_full, W_full),
                frame_rate=frame_rate,
                strength=kf_strength,
                num_pixel_frames=1,
            )
            for tokens, kf_idx, kf_strength in zip(kf_tokens_full, keyframe_indices, keyframe_strengths)
        ]

        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=video_kf_conditions_full,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens_up,
        )

        # Audio: no conditionings, just noise on stage-1 audio latent.
        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        # Stage 2 denoising: simple (no CFG), matching reference
        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Free transformer + upsampler before decode phase
        if self.low_memory:
            self.dit = None
            self.upsampler = None
            aggressive_cleanup()

        # Extract generated tokens (without appended keyframe tokens)
        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        keyframe_images: list[Image.Image | str] | None = None,
        keyframe_indices: list[int] | None = None,
        keyframe_strengths: list[float] | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = 1.0,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        **kwargs: object,
    ) -> str:
        """Generate two-stage keyframe interpolation and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            keyframe_images: Keyframe images (PIL or paths).
            keyframe_indices: Pixel frame indices for each keyframe.
            height: Final video height.
            width: Final video width.
            num_frames: Total number of pixel frames.
            frame_rate: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.
            cfg_scale: CFG guidance scale for stage 1.
            video_guider_params: Full video guider params (STG, rescale, modality).
            audio_guider_params: Full audio guider params.

        Returns:
            Path to output video file.
        """
        if keyframe_images is None or keyframe_indices is None:
            raise ValueError("keyframe_images and keyframe_indices are required")

        video_latent, audio_latent = self.interpolate(
            prompt=prompt,
            keyframe_images=keyframe_images,
            keyframe_indices=keyframe_indices,
            keyframe_strengths=keyframe_strengths,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        # Free any remaining heavy components from generation phase
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand
        self._load_decoders()

        result = self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)

        if self.low_memory:
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        return result
