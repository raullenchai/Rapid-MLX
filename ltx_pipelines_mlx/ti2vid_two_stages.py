"""Two-stage pipeline — dev model + CFG at half res, upscale, distilled LoRA refine.

Matches the reference architecture:
  Stage 1: Dev (non-distilled) model + CFG guidance at half resolution.
  Stage 2: Dev + distilled LoRA fused, simple denoising at full resolution.

Requires the dev model + distilled LoRA weights (e.g. dgrauet/ltx-2.3-mlx-q8).

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx_arsenal.diffusion import TeaCacheController

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors
from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop

# Reference defaults
DEFAULT_CFG_SCALE = 3.0


# TeaCache calibration constants for LTX-2 stage 1 (dev model, 30 Euler steps,
# 480x704x97 reference shape, MLX bf16 q8). Calibrated 2026-04-27 from a
# 5-prompt x 30-step run (145 deltas) on a fresh host. The robust fitter
# (scripts/fit_teacache_poly.py) picked degree 1 — higher degrees were
# non-monotone on the observed delta range and only marginally better in RMSE.
#
# Per-step input/output drift correlation in LTX-2 stage 1 is moderate
# (Pearson 0.41), and average output L1 ~ 0.56 — much higher than upstream
# DiTs (HunyuanVideo, Flux) where TeaCache typically uses thresh 0.15-0.4.
# Threshold 0.5 here yields ~22% skip rate per generation (~1.2x speedup);
# tune via `teacache_thresh=` for more aggressive caching at the cost of
# quality drift (1.0 ≈ 55% skip / ~2x, 1.5 ≈ 69% / ~3x).
LTX2_TEACACHE_COEFFICIENTS: list[float] = [
    1.3641334114092996,
    0.40915524073366694,
]
LTX2_TEACACHE_THRESH: float = 0.5


def _build_teacache_controller(num_steps: int, thresh: float | None) -> TeaCacheController:
    """Construct a TeaCacheController for LTX-2 stage 1.

    Args:
        num_steps: Number of denoising steps for stage 1.
        thresh: Optional override for the default ``rel_l1_thresh``.

    Returns:
        Configured ``TeaCacheController``.

    Raises:
        RuntimeError: If ``LTX2_TEACACHE_COEFFICIENTS`` is empty — calibration
            not yet run.
    """
    if not LTX2_TEACACHE_COEFFICIENTS:
        raise RuntimeError(
            "TeaCache coefficients for LTX-2 are not calibrated yet — run "
            "scripts/calibrate_teacache.py to generate them, then paste the "
            "values into LTX2_TEACACHE_COEFFICIENTS in this file."
        )
    return TeaCacheController(
        num_steps=num_steps,
        rel_l1_thresh=thresh if thresh is not None else LTX2_TEACACHE_THRESH,
        coefficients=LTX2_TEACACHE_COEFFICIENTS,
    )


def _remap_lora_keys(lora_sd: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap LoRA keys from ComfyUI/diffusion_model format to MLX model format."""
    return {LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(k): v for k, v in lora_sd.items()}


class TI2VidTwoStagesPipeline(BasePipeline):
    """Two-stage generation: dev model + CFG at half-res, upscale, distilled LoRA refine.

    Stage 1: Dev model + CFG guidance at half resolution (Euler sampler).
    Stage 2: Dev + distilled LoRA fused, simple denoising at full resolution.

    Requires ``dev_transformer`` and ``distilled_lora`` — the two-stage pipeline
    needs the dev model for quality generation at half resolution with CFG.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename (e.g. ``transformer-dev.safetensors``).
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength (default 1.0).
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        dev_transformer: str = "transformer-dev.safetensors",
        distilled_lora: str = "ltx-2.3-22b-distilled-lora-384.safetensors",
        distilled_lora_strength: float = 1.0,
        tile_count=None,
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
        self._tile_count = tile_count
        self.upsampler: LatentUpsampler | None = None

    def _fuse_distilled_lora(self, dit: LTXModel) -> None:
        """Fuse distilled LoRA weights into a loaded transformer in-place.

        In ``low_ram_streaming`` mode, in-place LoRA fusion would
        materialize the full transformer (defeating streaming). Instead
        we swap to the pre-fused ``transformer-distilled.safetensors``
        produced by mlx-forge — equivalent at default LoRA strength.
        """
        if self.low_ram_streaming:
            self._swap_to_distilled_streamer()
            return

        lora_stem = Path(self._distilled_lora).stem
        lora_path = self._resolve_safetensors(self.model_dir, lora_stem)
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Distilled LoRA not found: {lora_path}\n"
                "Two-stage requires the distilled LoRA for Stage 2.\n"
                "Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        lora_raw = dict(mx.load(str(lora_path)))
        lora_remapped = _remap_lora_keys(lora_raw)

        import mlx.utils

        flat_params = mlx.utils.tree_flatten(dit.parameters())
        flat_model = {k: v for k, v in flat_params if isinstance(v, mx.array)}

        model_sd = StateDict(sd=flat_model, size=0, dtype=set())
        lora_sd = StateDict(sd=lora_remapped, size=0, dtype=set())
        lora_with_strength = LoraStateDictWithStrength(lora_sd, self._distilled_lora_strength)

        fused = apply_loras(model_sd, [lora_with_strength])
        dit.load_weights(list(fused.sd.items()))
        aggressive_cleanup()

    def _swap_to_distilled_streamer(self) -> None:
        """Switch the streamer to a distilled-LoRA-fused dev model.

        Used in ``low_ram_streaming`` mode at the stage 1 → stage 2
        transition. Two strategies depending on requested LoRA
        strength:

        - **strength == 1.0**: swap to the pre-fused
          ``transformer-distilled.safetensors`` (mlx-forge produces it
          at LoRA strength 1.0). Cheaper — just opens a different
          mmap'd file, no per-block fusion at bind time.

        - **strength != 1.0**: keep streaming the dev model and
          attach a ``BlockLoraSource`` to the wrapper. Each bind() now
          dequantizes block weights, adds the LoRA delta at custom
          strength, and re-quantizes (handled by
          ``apply_loras`` already used by the non-streaming path).
          Slower per bind but supports arbitrary LoRA strength.
        """
        from ltx_core_mlx.loader.block_streaming import BlockLoraSource, BlockStreamer
        from ltx_core_mlx.loader.sd_ops import (
            LTXV_LORA_BLOCK_PREFIX,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )

        if abs(self._distilled_lora_strength - 1.0) <= 1e-6:
            distilled_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
            if not distilled_path.exists():
                raise FileNotFoundError(
                    f"Pre-fused distilled transformer not found in {self.model_dir} "
                    "(expected transformer-distilled*.safetensors). "
                    "low_ram_streaming for two-stage at LoRA strength 1.0 requires "
                    "the distilled file. Use: --model dgrauet/ltx-2.3-mlx-q8"
                )
            new_streamer = BlockStreamer(distilled_path, block_prefix="transformer.transformer_blocks.")
            old_streamer = object.__getattribute__(self.dit, "_streamer")
            object.__setattr__(self.dit, "_streamer", new_streamer)
            old_streamer.close()
            aggressive_cleanup()
            return

        # Custom strength → bind-time LoRA fusion.
        lora_path = self.model_dir / self._distilled_lora
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Distilled LoRA not found at {lora_path}. "
                "low_ram_streaming with a non-default LoRA strength requires "
                "the LoRA safetensors. Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        lora_source = BlockLoraSource(
            lora_path,
            block_prefix=LTXV_LORA_BLOCK_PREFIX,
            strength=self._distilled_lora_strength,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
        existing = list(object.__getattribute__(self.dit, "_lora_sources"))
        existing.append(lora_source)
        object.__setattr__(self.dit, "_lora_sources", existing)
        aggressive_cleanup()

    def _load_upsampler(self) -> None:
        """Load the spatial upsampler from config and weights.

        Raises:
            FileNotFoundError: when no upsampler weights file is present in the
                model dir. Stage 2 upscales the latent through this module
                unconditionally; with a randomly-initialised ``LatentUpsampler``
                the output decodes into a periodic pixel-shuffle grid ("mosaic")
                with no other error. A missing file must therefore fail loud
                rather than silently degrade.
        """
        import json

        # Try new v1.1+ naming, then old naming, then the LTX-2.5 split-layout
        # names (``spatial_upscaler_x2.safetensors``, ``ltx-2.5-*.safetensors``).
        weights_path = self.model_dir / "spatial_upscaler_x2_v1_1.safetensors"
        for stem in [
            "ltx-2.3-spatial-upscaler-x2",
            "spatial_upscaler_x2_v1_1",
            "spatial_upscaler_x2",
            "ltx-2.5-spatial-upscaler-x2",
            "ltx-2.5-latent-spatial-upscaler-x2",
        ]:
            resolved = self._resolve_safetensors(self.model_dir, stem)
            if resolved.exists():
                weights_path = resolved
                break

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Spatial upsampler weights not found in {self.model_dir} "
                "(looked for spatial_upscaler_x2_v1_1.safetensors, "
                "ltx-2.3-spatial-upscaler-x2*.safetensors and the LTX-2.5 "
                "spatial_upscaler_x2.safetensors). The two-stage and "
                "distilled pipelines require it for stage-2 upscaling; without "
                "it the latent is upscaled by an untrained module and the "
                "output degrades into a periodic 'mosaic' grid. Download it, "
                "e.g.: hf download dgrauet/ltx-2.3-mlx-q8 "
                f"spatial_upscaler_x2_v1_1.safetensors --local-dir {self.model_dir}"
            )

        config_path = self.model_dir / f"{weights_path.stem}_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self.upsampler = LatentUpsampler.from_config(config)
        else:
            self.upsampler = LatentUpsampler()

        raw = load_split_safetensors(weights_path)
        # Old format keys are prefixed with the stem; new format uses bare keys.
        stem_prefix = weights_path.stem + "."
        if raw and all(k.startswith(stem_prefix) for k in raw):
            raw = {k[len(stem_prefix) :]: v for k, v in raw.items()}
        self.upsampler.load_weights(list(raw.items()))
        aggressive_cleanup()

    def load(self) -> None:
        """Load DiT + VAE encoder + upsampler (skip decoders for memory).

        Decoders are loaded on-demand in ``generate_and_save()``.

        Skips reloading the text encoder: ``generate_two_stage`` encodes
        the prompt and frees Gemma BEFORE calling :meth:`load`. Loading
        Gemma again here would thrash the Metal heap (7.5 GB load/mmap +
        free) right before the 10 GB DiT — a documented cause of macOS
        GPU watchdog crashes under sustained system contention.
        """
        if self._loaded:
            return

        # DiT (dev model)
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        # VAE encoder (for denorm/renorm + optional I2V)
        self._load_vae_encoder()

        # Upsampler
        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def generate_two_stage(
        self,
        prompt: str,
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
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        enable_teacache: bool = False,
        teacache_thresh: float | None = None,
        tap: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1 (default: 20).
            stage2_steps: Denoising steps for stage 2.
            cfg_scale: CFG guidance scale for stage 1 (default: 3.0).
            stg_scale: STG guidance scale for stage 1 (default: 1.0, upstream LTX_2_3_PARAMS).
            image: Optional reference image for I2V conditioning.
            video_guider_params: Optional full guider params for video.
            audio_guider_params: Optional full guider params for audio.
            enable_teacache: When True, instantiate a TeaCacheController
                from the 'ltx2' arsenal preset and use it to skip stage 1
                transformer forwards whose modulated input is sufficiently
                close to the previous step's. Default False (no caching).
            teacache_thresh: Optional override for the preset's default
                ``rel_l1_thresh``. Higher = more skipping = faster but
                lossier. Ignored when ``enable_teacache=False``.
            tap: Optional per-step instrumentation hook forwarded to
                ``guided_denoise_loop``. Used by the calibration script.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
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

        # --- Stage 1: Half resolution with CFG ---
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

        # Stage 1: scalar-blend-then-cond for video matches legacy create_initial_state
        # → apply_conditioning → noise_latent_state(sigma=1) flow bit-exact for both T2V
        # and I2V (sigma=1 makes scaled_mask quantization a no-op anyway, but the scalar
        # path is consistent with stage 2 and avoids any future drift). Audio Stage 1
        # has no conditioning and clean=zeros so either flow gives identical noise.
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

        # Optional modality tiling: split video tokens into spatial/temporal
        # tiles for memory savings on long videos. Audio is replicated
        # across tiles. Composes with low_ram_streaming.
        stage1_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
            stage1_dit = TiledLTXModel(self.dit, tiler_1)

        x0_model = X0Model(stage1_dit)

        # Build guider params
        if video_guider_params is None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            )
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=stg_scale,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            )

        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        teacache_controller = None
        if enable_teacache:
            teacache_controller = _build_teacache_controller(stage1_steps, teacache_thresh)
            teacache_controller.reset()

        self._pre_denoise_flush(video_state, audio_state)
        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
            teacache=teacache_controller,
            tap=tap,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale with denormalize/renormalize ---
        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        mx.eval(video_upscaled)

        # Derive full-resolution dims from actual upscaled shape
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

        # Free VAE encoder + upsampler before Stage 2 denoising
        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        # Stage 2 video: scalar blend BEFORE conditionings to bit-match legacy
        # ``noise * 0.05 + video_tokens * 0.95`` Python-scalar arithmetic. Without
        # the flag, noise_latent_state's bf16 ``mask * sigma`` blend quantizes 0.05
        # to ~0.05005 → ~3e-3 input error → 27 dB G1 PSNR drift after VAE decode.
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

        # Stage 2 audio: legacy used noise_latent_state (bf16 mask path),
        # so leave legacy_scalar_blend=False (default) to preserve bit-equivalence.
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

        # Stage 2 reuses the same x0_model as stage 1 by default. With
        # modality tiling enabled, stage 2 has a different latent shape
        # (full resolution) so we need a fresh tiler+wrapper.
        stage2_x0_model = x0_model
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_2 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_full, W_full))
            stage2_x0_model = X0Model(TiledLTXModel(self.dit, tiler_2))

        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = denoise_loop(
            model=stage2_x0_model,
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
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        image: str | None = None,
        images=None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        enable_teacache: bool = False,
        teacache_thresh: float | None = None,
    ) -> str:
        """Generate two-stage video+audio and save to file.

        ``stage1_steps`` defaults to ``None`` so subclasses can apply
        their own default (Euler: 30, HQ res_2s: 15) without being
        overridden by this parent method's signature.
        """
        gen_kwargs: dict = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            image=image,
            images=images,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            enable_teacache=enable_teacache,
            teacache_thresh=teacache_thresh,
        )
        if stage1_steps is not None:
            gen_kwargs["stage1_steps"] = stage1_steps
        video_latent, audio_latent = self.generate_two_stage(**gen_kwargs)

        # Free transformer + encoder to make room for decoders
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand
        self._load_decoders()

        result = self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)

        # Free decoders so a subsequent generate_and_save call on the same
        # pipeline instance doesn't stack DiT (~26 GB q8) on top of decoders
        # and OOM the 32 GB envelope.
        if self.low_memory:
            self.vae_decoder = None
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        return result
