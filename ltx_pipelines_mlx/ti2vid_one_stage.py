"""TI2VidOneStagePipeline — dev one-stage T2V/I2V at full target resolution.

Mirrors upstream ``ltx_pipelines.ti2vid_one_stage.TI2VidOneStagePipeline``
verbatim (file name + class name): dev (non-distilled) transformer + CFG/STG guidance, run **once at
the target resolution** (no half-res stage 1, no upsampler, no stage 2
refine).

Use this pipeline when you want the quality of the dev model with CFG
but don't want to rely on the spatial upsampler — for example when
running at native distilled-training resolutions (≤ 480x704) where the
upscale path of :class:`TI2VidTwoStagesPipeline` adds latency without quality
benefit, or when you need direct full-res latents (e.g. for downstream
post-processing).

Trade-offs vs the other generate variants:

- vs ``--two-stage`` / ``--hq``: same dev model + CFG, but **slower**
  (full-res ~30 Euler steps vs half-res ~30 + 3 upscale steps) and
  higher peak memory. Quality on small target resolutions is similar;
  on large targets the upscale path generally wins.
- vs ``--distilled``: ~3-4x slower per step (CFG = 2-4 forwards/step),
  but better fidelity / prompt following thanks to the dev weights.
- vs default one-stage (distilled at target): adds CFG/STG quality at
  significant memory and time cost.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .scheduler import ltx2_schedule
from .ti2vid_two_stages import DEFAULT_CFG_SCALE, TI2VidTwoStagesPipeline
from .utils.helpers import create_noised_state
from .utils.samplers import guided_denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


class TI2VidOneStagePipeline(TI2VidTwoStagesPipeline):
    """Dev model + CFG one-stage T2V/I2V at full target resolution.

    Reuses :class:`TI2VidTwoStagesPipeline` for text encoding, dev transformer
    loading, and VAE encoder loading — but skips the upsampler and
    stage 2 refine. The forward runs once at the user-requested
    resolution.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        dev_transformer: str = "transformer-dev.safetensors",
        tile_count=None,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            dev_transformer=dev_transformer,
            # No distilled LoRA — we never run stage 2.
            distilled_lora="",
            distilled_lora_strength=0.0,
            tile_count=tile_count,
        )

    def load(self) -> None:
        """Load dev DiT + VAE encoder. No upsampler needed (no stage 2).

        Skips reloading the text encoder: ``generate_one_stage_dev``
        encodes the prompt and frees Gemma BEFORE calling :meth:`load`.
        Loading Gemma again here would just thrash the Metal heap
        (7.5 GB load/mmap + free) right before DiT — a documented cause
        of macOS GPU watchdog crashes under sustained system contention.
        """
        if self._loaded:
            return

        if self.dit is None:
            self.dit = self._load_dev_transformer()

        # VAE encoder is needed only for I2V (image conditioning).
        # We still load it eagerly here for symmetry with TI2VidTwoStagesPipeline.
        self._load_vae_encoder()

        self._loaded = True

    def generate_one_stage_dev(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        image: str | None = None,
        images=None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        tap: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video at full target resolution with dev model + CFG.

        Args:
            prompt: Text prompt.
            height: Target video height (must be divisible by 32).
            width: Target video width (must be divisible by 32).
            num_frames: Number of frames (must satisfy ``(F-1) % 8 == 0``).
            seed: Random seed.
            num_steps: Denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
            image: Optional reference image for I2V (legacy single-anchor).
            images: Optional list of :class:`ImageConditioningInput` for
                multi-anchor I2V (matches upstream ``combined_image_conditionings``).
            video_guider_params: Optional full guider params for video.
            audio_guider_params: Optional full guider params for audio.
            tap: Optional per-step instrumentation hook.

        Returns:
            Tuple of (video_latent, audio_latent) at target resolution.
        """
        # --- Text encoding (positive + negative for CFG) ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load dev DiT + VAE encoder ---
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        # --- Single stage at full target resolution ---
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at target resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand.
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h = H * 32
        enc_w = W * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings: list = []
        if resolved_images:
            conditionings = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h,
                enc_w=enc_w,
                spatial_dims=(F, H, W),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
                model_dir=self.model_dir,
            )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings,
            spatial_dims=(F, H, W),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H, W),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        num_tokens = F * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)

        stage_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler = VideoModalityTiler(self._tile_count, latent_shape=(F, H, W))
            stage_dit = TiledLTXModel(self.dit, tiler)

        x0_model = X0Model(stage_dit)

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

        # We can free the VAE encoder before the heavy denoise loop;
        # decoders are loaded on-demand in generate_and_save.
        if self.low_memory and image is None:
            self.image_conditioner.free()
            aggressive_cleanup()

        self._pre_denoise_flush(video_state, audio_state)
        output = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
            tap=tap,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens = output.video_latent[:, : F * H * W, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)
        _materialize(video_latent, audio_latent)
        return video_latent, audio_latent

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        image: str | None = None,
        images=None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        **_unused_kwargs,
    ) -> str:
        """Generate dev one-stage video+audio and save to file.

        ``**_unused_kwargs`` absorbs flags that don't apply (stage1_steps,
        stage2_steps, enable_teacache, etc.) so the CLI can forward
        everything uniformly.
        """
        video_latent, audio_latent = self.generate_one_stage_dev(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            image=image,
            images=images,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        # Free generation components to make room for decoders
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()

        result = self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)

        if self.low_memory:
            self.vae_decoder = None
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        return result


__all__ = ["TI2VidOneStagePipeline"]
