"""Audio-to-Video two-stage pipeline — dev model + CFG + audio conditioning.

Matches the reference architecture:
  Stage 1: Dev model + CFG at half resolution, audio frozen (encoded from input).
  Stage 2: Dev + distilled LoRA fused, refine video + audio at full resolution.

Requires the dev model + distilled LoRA weights (e.g. dgrauet/ltx-2.3-mlx-q8).

Ported from ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
)
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE, TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop


class A2VidPipelineTwoStage(TI2VidTwoStagesPipeline):
    """Audio-to-Video two-stage generation pipeline.

    Stage 1: Dev model + CFG at half spatial resolution, audio frozen.
    Stage 2: Dev + distilled LoRA fused, refine video + audio at full resolution.

    Inherits from TI2VidTwoStagesPipeline for dev model loading, LoRA fusion,
    upsampler, VAE encoder, and decoder management.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength (default 1.0).
    """

    def _denoise_stage1(
        self,
        x0_model: X0Model,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        neg_video_embeds: mx.array,
        neg_audio_embeds: mx.array,
        sigmas: list[float],
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
    ) -> object:
        """Run Stage 1 denoising with Euler + CFG. Override for HQ (res2s)."""
        # Video: full guidance (ref LTX_2_3_PARAMS)
        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        # Audio: no guidance (frozen in Stage 1)
        audio_gp = MultiModalGuiderParams()

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        self._pre_denoise_flush(video_state, audio_state)
        return guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
        )

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        audio_path: str | Path | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int = 30,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        image: str | None = None,
        images=None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> str:
        """Generate video from audio and save to file.

        Uses the original input audio for the output (not VAE-decoded audio)
        for maximum fidelity.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            audio_path: Path to input audio file (required).
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            frame_rate: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps (default: 20).
            stage2_steps: Stage 2 denoising steps.
            cfg_scale: CFG guidance scale for stage 1 (default: 3.0).
            stg_scale: STG guidance scale for stage 1 (default: 1.0, upstream LTX_2_3_PARAMS).
            image: Optional reference image for I2V conditioning (first frame).
            audio_start_time: Start time in seconds for audio.
            audio_max_duration: Max audio duration.

        Returns:
            Path to the output video file.
        """
        if audio_path is None:
            raise ValueError("audio_path is required for A2VidPipelineTwoStage")

        if audio_max_duration is None:
            audio_max_duration = num_frames / frame_rate

        # --- Encode audio ---
        self._load_audio_encoder()
        assert self.audio_encoder is not None
        assert self.audio_processor is not None

        audio_data = load_audio(
            audio_path,
            target_sample_rate=16000,
            start_time=audio_start_time,
            max_duration=audio_max_duration,
        )
        if audio_data is None:
            raise ValueError(f"No audio found in {audio_path}")

        audio_latent = encode_audio(
            audio_data.waveform,
            audio_data.sample_rate,
            self.audio_encoder,
            self.audio_processor,
        )

        # Patchify audio to tokens
        audio_T = compute_audio_token_count(num_frames, frame_rate)
        audio_latent = audio_latent[:, :, :audio_T, :]
        audio_tokens, _ = self.audio_patchifier.patchify(audio_latent)  # (1, audio_T, 128)
        mx.synchronize()

        # Free audio encoder via composition block
        if self.low_memory:
            self.audio_conditioner.free()

        # --- Text encoding (positive + negative for CFG) ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load DiT (defer VAE encoder + upsampler to after Stage 1 for memory) ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        # --- Stage 1: Half resolution with CFG, audio frozen ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at half resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand.
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:

            def _encode_combined(encoder):
                conds = combined_image_conditionings(
                    resolved_images,
                    enc_h=enc_h_half,
                    enc_w=enc_w_half,
                    spatial_dims=(F, H_half, W_half),
                    video_encoder=encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
                mx.synchronize()
                return conds

            conditionings_1 = self.image_conditioner(_encode_combined, free_after=self.low_memory)
            if self.low_memory:
                aggressive_cleanup()

        # Stage 1 video: legacy_scalar_blend=True for bit-exact match with the
        # legacy create_initial_state → apply_conditioning → noise_latent_state flow.
        video_state_1 = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        # Audio frozen in Stage 1 (denoise_mask=0 = preserve)
        audio_state_1 = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=mx.zeros((1, audio_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=audio_positions,
        )

        # Stage 1 denoising
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        output_1 = self._denoise_stage1(
            x0_model=x0_model,
            video_state=video_state_1,
            audio_state=audio_state_1,
            video_embeds=video_embeds,
            audio_embeds=audio_embeds,
            neg_video_embeds=neg_video_embeds,
            neg_audio_embeds=neg_audio_embeds,
            sigmas=sigmas_1,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale + (optional) full-res I2V conditioning via ImageConditioner block ---
        if self.upsampler is None:
            self._load_upsampler()
        assert self.upsampler is not None

        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        H_full = H_half * 2
        W_full = W_half * 2

        def _upscale_and_optionally_encode(encoder):
            v_mlx = video_half.transpose(0, 2, 3, 4, 1)
            v_denorm = encoder.denormalize_latent(v_mlx).transpose(0, 4, 1, 2, 3)
            v_up = self.upsampler(v_denorm)
            v_up_renorm = encoder.normalize_latent(v_up.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
            mx.synchronize()
            conds: list = []
            if resolved_images:
                enc_h_full = H_full * 32
                enc_w_full = W_full * 32
                conds = combined_image_conditionings(
                    resolved_images,
                    enc_h=enc_h_full,
                    enc_w=enc_w_full,
                    spatial_dims=(F, H_full, W_full),
                    video_encoder=encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
            return v_up_renorm, conds

        video_upscaled, conditionings_2 = self.image_conditioner(
            _upscale_and_optionally_encode, free_after=self.low_memory
        )
        if self.low_memory:
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        # Stage 2 video: scalar-blend bit-matches legacy inline arithmetic.
        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        # Stage 2 audio: default mask path matches legacy noise_latent_state.
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens,
        )

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

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))

        # --- Decode and save ---
        if self.low_memory:
            self.dit = None
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()

        # Use original audio for output (higher fidelity than VAE-decoded)
        # Trim to exact video duration to ensure sync
        import tempfile

        video_duration = num_frames / frame_rate
        audio_data_48k = load_audio(
            audio_path,
            target_sample_rate=48000,
            start_time=audio_start_time,
            max_duration=video_duration,
        )
        if audio_data_48k is not None:
            # Trim to exact sample count for video duration
            max_samples = int(video_duration * 48000)
            waveform_48k = audio_data_48k.waveform[:, :, :max_samples]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
                temp_audio = _tmp.name
            self._save_waveform(waveform_48k, temp_audio, sample_rate=48000)
        else:
            temp_audio = None

        self.video_decoder_block.decode_and_stream(
            video_latent,
            output_path,
            frame_rate=frame_rate,
            audio_path=temp_audio,
        )

        if temp_audio is not None:
            Path(temp_audio).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
