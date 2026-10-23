"""IC-LoRA pipeline — reference video conditioning with two-stage generation.

Ported from ltx-pipelines/src/ltx_pipelines/ic_lora.py

Two-stage video generation pipeline with In-Context (IC) LoRA support.
Allows conditioning the generated video on control signals such as depth maps,
human pose, or image edges via the video_conditioning parameter.
The specific IC-LoRA model should be provided via the loras parameter.
Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
by 2x and refines with additional denoising steps for higher quality output.
Both stages use distilled models for efficiency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.loader import (
    LTXV_LORA_BLOCK_PREFIX,
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.video import load_video_frames_normalized
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
)
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop

logger = logging.getLogger(__name__)


class ICLoraPipeline(BasePipeline):
    """Two-stage video generation pipeline with IC-LoRA reference conditioning.

    Conditions the generated video on a reference video (e.g., depth, pose, edges)
    via VideoConditionByReferenceLatent. Stage 1 generates at half resolution with
    the IC-LoRA fused into the transformer, then Stage 2 upscales and refines
    without the LoRA (clean distilled model).

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        lora_paths: List of (lora_path, strength) tuples for IC-LoRA weights.
        low_memory: Aggressive memory management.
    """

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )
        # `vae_encoder` is provided as a property by BasePipeline (proxies
        # to the ImageConditioner block). Only `upsampler` is unique here.
        self.upsampler: LatentUpsampler | None = None

        # Resolve LoRA paths (download from HuggingFace if needed)
        self._lora_paths = [(_resolve_lora_path(p), s) for p, s in (lora_paths or [])]

        # Read reference downscale factor from LoRA metadata.
        # IC-LoRAs trained with low-resolution reference videos store this factor
        # so inference can resize reference videos to match training conditions.
        self.reference_downscale_factor = 1
        for lora_path, _ in self._lora_paths:
            scale = read_lora_reference_downscale_factor(lora_path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora_path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale

    def load(self) -> None:
        """Load generation components: DiT, VAE encoder, upsampler.

        Does NOT load decoders (VAE decoder, audio decoder, vocoder) to save
        memory. Those are loaded on-demand in generate_and_save().
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # DiT (largest component)
        if self.dit is None:
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self._resolve_safetensors(model_dir, "transformer-distilled")
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

        # VAE encoder (for encoding control videos and I2V images)
        self._load_vae_encoder()

        # Upsampler (for Stage 2)
        if self.upsampler is None:
            import json

            name = "spatial_upscaler_x2_v1_1"
            config_path = model_dir / f"{name}_config.json"
            weights_path = model_dir / f"{name}.safetensors"
            if config_path.exists():
                config = json.loads(config_path.read_text()).get("config", {})
                self.upsampler = LatentUpsampler.from_config(config)
            else:
                self.upsampler = LatentUpsampler()
            if weights_path.exists():
                weights = load_split_safetensors(weights_path, prefix=f"{name}.")
                self.upsampler.load_weights(list(weights.items()))
            aggressive_cleanup()

        self._loaded = True

    def _fuse_loras(self) -> None:
        """Fuse all LoRA weights into the transformer.

        Reads LoRA files, applies ComfyUI key remapping, fuses deltas into
        model weights, and re-quantizes. In ``low_ram_streaming`` mode
        the LoRAs are attached as ``BlockLoraSource`` to the streaming
        wrapper instead of being fused in-place — fusion happens at
        each block bind.
        """
        if not self._lora_paths:
            return

        assert self.dit is not None

        if self.low_ram_streaming:
            from ltx_core_mlx.loader.block_streaming import BlockLoraSource

            sources: list = list(object.__getattribute__(self.dit, "_lora_sources"))
            for lora_path, strength in self._lora_paths:
                sources.append(
                    BlockLoraSource(
                        lora_path,
                        block_prefix=LTXV_LORA_BLOCK_PREFIX,
                        strength=strength,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )
                logger.info(f"Attached LoRA streamer: {lora_path} (strength={strength})")
            object.__setattr__(self.dit, "_lora_sources", sources)
            return

        import mlx.utils

        model_weights = dict(mlx.utils.tree_flatten(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())

        loader = SafetensorsStateDictLoader()
        lora_sds = []
        for lora_path, strength in self._lora_paths:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
            logger.info(f"Loaded LoRA: {lora_path} (strength={strength})")

        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)

        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()

        logger.info(f"Fused {len(self._lora_paths)} LoRA(s) into transformer")

    def _reload_clean_transformer(self) -> None:
        """Reload the transformer without LoRA for Stage 2.

        The reference implementation uses separate ModelLedgers for Stage 1
        (with LoRA) and Stage 2 (clean distilled). We achieve the same by
        discarding the fused transformer and reloading from disk.

        In ``low_ram_streaming`` mode we just clear the LoraSources from
        the streaming wrapper instead of full reload — the underlying
        block weights are streamed fresh from the safetensors.
        """
        if self.low_ram_streaming and self.dit is not None:
            old_sources = list(object.__getattribute__(self.dit, "_lora_sources"))
            object.__setattr__(self.dit, "_lora_sources", [])
            for src in old_sources:
                src.close()
            aggressive_cleanup()
            logger.info("Cleared streaming LoRA sources for Stage 2")
            return

        self.dit = None
        aggressive_cleanup()

        transformer_path = self.model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
        self.dit = self._load_transformer_with_optional_streaming(transformer_path)
        logger.info("Reloaded clean transformer for Stage 2")

    def _create_conditionings(
        self,
        images,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        *,
        frame_rate: float,
        video_encoder=None,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: mx.array | None = None,
    ) -> list[object]:
        """Create conditioning items for video generation.

        Builds image conditionings (I2V) and IC-LoRA reference video conditionings.
        Mirrors upstream ``ltx_pipelines.ic_lora.ICLoraPipeline._create_conditionings``.

        Args:
            images: Optional list of :class:`ImageConditioningInput` (upstream-iso
                multi-anchor) for I2V. Legacy ``list[tuple[str, int, float]]``
                tuples are accepted and normalized.
            video_conditioning: List of (video_path, strength) for IC-LoRA reference.
            height: Stage output height (pixels).
            width: Stage output width (pixels).
            num_frames: Number of pixel frames.
            video_encoder: VAE encoder to use for image encoding (defaults to
                ``self.vae_encoder``). Upstream takes this as explicit arg so
                callers can pass an offloaded/streamed instance.
            conditioning_attention_strength: Scalar attention weight in [0, 1].
            conditioning_attention_mask: Optional pixel-space mask (1, 1, F, H, W).

        Returns:
            List of conditioning items. IC-LoRA conditionings are appended last.
        """
        if video_encoder is None:
            video_encoder = self.vae_encoder
        assert video_encoder is not None

        conditionings: list[object] = []

        # Image conditionings (I2V) — upstream-iso multi-anchor pattern.
        # frame_idx==0 → VideoConditionByLatentIndex (replace);
        # frame_idx>0 → VideoConditionByKeyframeIndex (guide).
        if images:
            from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
            from ltx_pipelines_mlx.utils.args import ImageConditioningInput

            normalized = [
                img if isinstance(img, ImageConditioningInput) else ImageConditioningInput(*img) for img in images
            ]
            # Stage spatial latent dims (F, H, W) for keyframe positions
            F_lat, H_lat, W_lat = compute_video_latent_shape(num_frames, height, width)
            conditionings.extend(
                combined_image_conditionings(
                    normalized,
                    enc_h=height,
                    enc_w=width,
                    spatial_dims=(F_lat, H_lat, W_lat),
                    video_encoder=video_encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
            )

        # IC-LoRA reference video conditionings (delegated to iclora_utils for parity
        # with upstream `ltx_pipelines.ic_lora.ICLoraPipeline._create_conditionings`).
        append_ic_lora_reference_video_conditionings(
            conditionings,
            video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            reference_downscale_factor=self.reference_downscale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
        )

        if video_conditioning:
            logger.info(f"Added {len(video_conditioning)} video conditioning(s)")

        return conditionings

    def generate(
        self,
        prompt: str,
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
        conditioning_attention_mask: mx.array | None = None,
        skip_stage_2: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Generate video with IC-LoRA reference conditioning.

        Args:
            prompt: Text prompt.
            video_conditioning: List of (video_path, strength) tuples for IC-LoRA
                reference video conditioning (e.g., depth maps, poses, edges).
            height: Output video height.
            width: Output video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            images: Optional list of (image_path, frame_index, strength) for I2V.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning
                attention. 0.0 = ignore, 1.0 = full conditioning. Default 1.0.
            conditioning_attention_mask: Optional pixel-space mask (1, 1, F, H, W)
                matching reference video dimensions. Values in [0, 1].
            skip_stage_2: Skip upscale + refine, output at half resolution.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        # Load text encoder, encode, free, then load generation components.
        # Done manually instead of _encode_text_and_load() to avoid loading
        # decoders (which we don't need until generate_and_save).
        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # Load DiT + VAE encoder + upsampler (no decoders)
        self.load()

        assert self.dit is not None
        assert self.vae_encoder is not None

        # Fuse LoRA into transformer for Stage 1
        self._fuse_loras()

        # --- Stage 1: Half-resolution generation with IC-LoRA ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # Encode conditionings before denoising (reduce peak memory)
        stage_1_conditionings = self._create_conditionings(
            images=images,
            video_conditioning=video_conditioning,
            height=half_h,
            width=half_w,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
        )

        # Build noised state via canonical upstream order:
        #     init (zeros) -> apply conditionings (image replace + ref append) -> noise.
        # For pipelines without reference conditionings (image-only), this is
        # bit-equivalent to the previous code path. With reference appends,
        # the noise is generated for the FULL post-condition shape (including
        # appended ref tokens, masked out), shifting the gen-token noise
        # pattern slightly.
        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=stage_1_conditionings,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
        )

        # Denoise stage 1
        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        self._pre_denoise_flush(video_state, audio_state)
        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Extract only generation tokens (exclude appended reference tokens)
        gen_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens, (F, H_half, W_half))

        if skip_stage_2:
            audio_latent = self.audio_patchifier.unpatchify(output_1.audio_latent)
            return video_half, audio_latent

        # --- Stage 2: Upscale + refine (no IC-LoRA, clean distilled model) ---
        # Upscale with denormalize/renormalize wrapping (matching reference).
        # Reference: un_normalize -> upsampler -> normalize using VAE encoder stats.
        # Without this, the upsampler produces garbage for Stage 2.
        assert self.upsampler is not None
        assert self.vae_encoder is not None
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_up_mlx = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_up_mlx.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        mx.eval(video_upscaled)

        # Derive full-resolution latent dims from the ACTUAL upscaled shape,
        # not the target height/width (which may round differently).
        # The upsampler doubles H_half and W_half, so H_full = H_half * 2.
        H_full = H_half * 2
        W_full = W_half * 2
        enc_h_full = H_full * 32
        enc_w_full = W_full * 32

        # Encode I2V images at upscaled resolution (if any) before freeing encoder.
        # Upstream-iso multi-anchor pattern: frame_idx==0 → LatentIndex, else → KeyframeIndex.
        conditionings_2 = []
        if images:
            from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
            from ltx_pipelines_mlx.utils.args import ImageConditioningInput

            normalized = [
                img if isinstance(img, ImageConditioningInput) else ImageConditioningInput(*img) for img in images
            ]

            F_full, H_full_lat, W_full_lat = compute_video_latent_shape(num_frames, enc_h_full, enc_w_full)
            conditionings_2.extend(
                combined_image_conditionings(
                    normalized,
                    enc_h=enc_h_full,
                    enc_w=enc_w_full,
                    spatial_dims=(F_full, H_full_lat, W_full_lat),
                    video_encoder=self.vae_encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
            )

        # Free VAE encoder + upsampler + fused DiT before loading clean transformer
        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
        # Reload clean transformer without LoRA (matches reference: separate model ledgers)
        self._reload_clean_transformer()

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        # Stage 2 video: scalar-blend bit-matches legacy inline arithmetic.
        # IC-LoRA Stage 1 already accepts a small RNG-shift drift due to the
        # reference-token shape change; Stage 2 stays bit-exact since cond_2
        # is at most a LatentIndex replace (no shape change).
        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=conditionings_2,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens_up,
            legacy_scalar_blend=True,
        )

        # Audio refined in stage 2
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

        video_latent = self.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

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
        """Generate IC-LoRA conditioned video+audio and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            video_conditioning: List of (video_path, strength) for IC-LoRA reference.
            height: Output video height.
            width: Output video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            images: Optional list of (image_path, frame_index, strength) for I2V.
            conditioning_attention_strength: Attention strength for conditioning.
            skip_stage_2: Skip upscale + refine.

        Returns:
            Path to the output video file.
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

        # Free generation components to make room for decoders
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self.image_conditioner.free()
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand (not loaded during generate to save memory)
        self._load_decoders()

        return self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)

    # Upstream parity: ``pipe(prompt=..., ...)`` style invocation.
    __call__ = generate


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> mx.array:
    """Load a mask video file as a pixel-space attention mask tensor.

    Mirrors upstream ``ltx_pipelines.ic_lora._load_mask_video``:

    1. Decode the first ``num_frames`` frames at ``(height, width)``.
    2. Channel-average to grayscale.
    3. Remap from ``[-1, 1]`` (video loader range) back to ``[0, 1]``.
    4. Clip to ``[0, 1]``.

    Args:
        mask_path: Path to mask video file (any ffmpeg-readable format).
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Number of frames to load.

    Returns:
        ``mx.array`` of shape ``(1, 1, F, H, W)``, bfloat16, values in ``[0, 1]``.
    """
    # load_video_frames_normalized returns shape (1, 3, F, H, W) in [-1, 1].
    frames = load_video_frames_normalized(mask_path, height, width, num_frames)
    # Channel-average (RGB → grayscale): (1, 3, F, H, W) → (1, 1, F, H, W).
    mask = frames.mean(axis=1, keepdims=True)
    # [-1, 1] → [0, 1].
    mask = (mask + 1.0) / 2.0
    # Clip safety.
    mask = mx.clip(mask, 0.0, 1.0)
    return mask.astype(mx.bfloat16)


def _resolve_lora_path(path: str) -> str:
    """Resolve a LoRA path — download from HuggingFace if needed.

    Supports:
        - Local file paths: returned as-is if they exist.
        - HuggingFace repo IDs: downloads the repo and finds the .safetensors file.
          Example: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"

    Args:
        path: Local path or HuggingFace repo ID.

    Returns:
        Resolved local path to the .safetensors file.
    """
    local = Path(path)
    if local.exists():
        return str(local)

    # Try HuggingFace download
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading LoRA from HuggingFace: {path}")
    repo_dir = Path(snapshot_download(path))

    # Find the .safetensors file in the downloaded repo
    safetensors_files = list(repo_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {repo_dir}")
    if len(safetensors_files) > 1:
        logger.warning(f"Multiple .safetensors files found, using first: {safetensors_files[0].name}")
    return str(safetensors_files[0])
