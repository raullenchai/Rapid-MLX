"""HQ two-stage pipeline — res_2s second-order sampler for Stage 1.

Same architecture as TI2VidTwoStagesPipeline but uses the res_2s second-order sampler
instead of Euler for Stage 1 denoising, producing higher quality at fewer steps.
Supports guidance (CFG/STG) with the res_2s sampler.

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py
"""

from __future__ import annotations

import mlx.core as mx
from mlx_arsenal.diffusion import TeaCacheController

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE, TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop, res2s_denoise_loop

# TeaCache calibration constants for the HQ res_2s path (LTX-2 stage 1, 30
# steps, 384x576x65 reference shape, MLX bf16 q8). Calibrated 2026-04-27 from
# a 5-prompt run (145 deltas) via scripts/calibrate_teacache.py --hq. The
# robust fitter (scripts/fit_teacache_poly.py) picked degree 1.
#
# res_2s has fundamentally different per-step dynamics from Euler:
# - SDE noise injection between stage 1 and stage 2 inflates delta_in
#   (HQ median 0.66 vs Euler median 0.08).
# - Pearson(delta_in, delta_out) is 0.62 here vs Euler's 0.41 — the polynomial
#   is more predictive, justifying a more aggressive default threshold.
# - The pol(delta) values cluster around 0.8 because delta_in is large and
#   the slope is ~1.27, which produces a "cliff" in skip-rate vs threshold:
#   below ~0.8 nothing skips; at ~1.0 most steps skip. Default 1.0 lands at
#   the sweet spot (~52% skip per simulation, ~2x speedup expected).
LTX2_HQ_TEACACHE_COEFFICIENTS: list[float] = [
    1.2692083808655041,
    -0.033401134092491416,
]
LTX2_HQ_TEACACHE_THRESH: float = 1.0  # tune per use case


def _build_hq_teacache_controller(num_steps: int, thresh: float | None) -> TeaCacheController:
    """Construct an HQ-specific TeaCacheController.

    Mirrors :func:`ltx_pipelines_mlx.ti2vid_two_stages._build_teacache_controller`
    but uses ``LTX2_HQ_TEACACHE_COEFFICIENTS`` / ``LTX2_HQ_TEACACHE_THRESH``
    so res_2s gets coefficients fit on its own dynamics.
    """
    if not LTX2_HQ_TEACACHE_COEFFICIENTS:
        raise RuntimeError(
            "TeaCache coefficients for the LTX-2 HQ path are not calibrated yet — "
            "run scripts/calibrate_teacache.py --hq to generate them, then paste "
            "the values into LTX2_HQ_TEACACHE_COEFFICIENTS in this file."
        )
    return TeaCacheController(
        num_steps=num_steps,
        rel_l1_thresh=thresh if thresh is not None else LTX2_HQ_TEACACHE_THRESH,
        coefficients=LTX2_HQ_TEACACHE_COEFFICIENTS,
    )


class TI2VidTwoStagesHQPipeline(TI2VidTwoStagesPipeline):
    """HQ two-stage generation with res_2s second-order sampler.

    Inherits from TI2VidTwoStagesPipeline and overrides Stage 1 to use the res_2s
    sampler for higher quality at fewer steps. Stage 2 is identical.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength.
    """

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int = 15,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 0.0,
        image: str | None = None,
        images=None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        enable_teacache: bool = False,
        teacache_thresh: float | None = None,
        tap: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using HQ two-stage pipeline with res_2s sampler.

        Same as TI2VidTwoStagesPipeline.generate_two_stage but uses res_2s sampler
        for Stage 1 instead of Euler. ``enable_teacache`` / ``teacache_thresh``
        / ``tap`` are forwarded to ``res2s_denoise_loop`` exactly as in the
        Euler path.
        """
        # --- Text encoding ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load DiT + VAE encoder + upsampler ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with res_2s sampler + guidance ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at half resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand
        # (frame_idx=0, strength=1.0).
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:
            conditionings_1 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_half,
                enc_w=enc_w_half,
                spatial_dims=(F, H_half, W_half),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
                model_dir=self.model_dir,
            )

        # Stage 1 video/audio: legacy_scalar_blend=True for bit-exact match
        # (see ti2vid_two_stages.py for rationale).
        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        # Stage 1 sigma schedule (dynamic for dev model)
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guider params (HQ defaults: no STG, lower rescale)
        if video_guider_params is None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                rescale_scale=0.45,
                modality_scale=3.0,
                stg_blocks=[],
            )
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=stg_scale,
                rescale_scale=1.0,
                modality_scale=3.0,
                stg_blocks=[],
            )

        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        # Stage 1: res_2s with guidance
        teacache_controller = None
        if enable_teacache:
            teacache_controller = _build_hq_teacache_controller(stage1_steps, teacache_thresh)
            teacache_controller.reset()
        self._pre_denoise_flush(video_state, audio_state)
        output_1 = res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            teacache=teacache_controller,
            tap=tap,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale with denormalize/renormalize ---
        # Strip any appended keyframe tokens (multi-anchor with frame_idx>0
        # appends via VideoConditionByKeyframeIndex; only the base
        # F*H*W tokens are spatial latent we need to unpatchify).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution for Stage 2 (re-encode at upscaled dims)
        conditionings_2: list = []
        if resolved_images:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_full,
                enc_w=enc_w_full,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
                model_dir=self.model_dir,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        # Stage 2 video: legacy_scalar_blend=True bit-matches the legacy inline
        # ``noise * sigma + video_tokens * (1 - sigma)`` arithmetic.
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

        # Stage 2 audio: default (mask path) matches legacy noise_latent_state.
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

        # Stage 2: simple denoising (no CFG)
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

        # Strip appended keyframe tokens before unpatchify (see stage 1).
        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent
