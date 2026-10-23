"""Two-stage lip-dubbing pipeline with IC-LoRA + appended audio reference conditioning.

Mirrors upstream ``ltx_pipelines.lipdub.LipDubPipeline``: takes a reference video
(provides visual structure via IC-LoRA + audio source via VAE-encoded reference
tokens) and produces a lip-dubbed output where the generated video matches the
reference audio.

Frame count is derived from the reference video metadata (snapped to ``8k+1``),
not user-supplied. Stage 2 keeps the stage-1 audio latent unchanged (upstream's
``frozen=True`` semantics) and only refines the video.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions

from .ic_lora import ICLoraPipeline
from .iclora_utils import append_ic_lora_reference_video_conditionings
from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .utils.helpers import create_noised_state
from .utils.samplers import denoise_loop

logger = logging.getLogger(__name__)

_mx_materialize = getattr(mx, "eval")  # noqa: B009

TIME_SCALE = 8


def _snap_frames_to_8k1(frames: int) -> int:
    """Round ``frames`` down to the nearest ``8k+1``.

    Mirrors upstream ``ltx_pipelines.lipdub._snap_frames_to_8k1``. The audio+video
    VAE temporal compression factor is 8, so the model requires ``(F-1) % 8 == 0``.
    """
    return ((frames - 1) // TIME_SCALE) * TIME_SCALE + 1


def patchify_lipdub_audio_reference_latent(
    vae_latents: mx.array,
    audio_patchifier,
    negative_positions: bool = True,
) -> tuple[mx.array, mx.array]:
    """Patchify audio VAE latents and build RoPE positions (optionally shifted negative).

    Mirrors upstream ``ltx_pipelines.lipdub.patchify_lipdub_audio_reference_latent``:
    reference audio tokens get positions in negative time relative to the target audio
    sequence so the model interprets them as off-screen context rather than overlapping
    target tokens.

    Args:
        vae_latents: Audio VAE latent ``(1, 8, T, 16)``.
        audio_patchifier: :class:`AudioPatchifier` instance.
        negative_positions: If True, shift positions to ``[-aud_dur - 0.04, -0.04]``.

    Returns:
        Tuple of (patchified tokens ``(1, T, 128)``, positions ``(1, T, 1)``).
    """
    tokens, T = audio_patchifier.patchify(vae_latents)
    positions = compute_audio_positions(T)  # (1, T, 1)

    if negative_positions:
        aud_dur = float(mx.max(positions).item())
        positions = positions - (aud_dur + 0.04)

    return tokens, positions.astype(mx.float32)


class LipDubPipeline(ICLoraPipeline):
    """Two-stage lip-dub pipeline with IC-LoRA video reference + appended audio reference."""

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
    ):
        if not lora_paths or len(lora_paths) != 1:
            raise ValueError("LipDub requires exactly one --lora (the lip-dub IC-LoRA).")
        super().__init__(
            model_dir=model_dir,
            lora_paths=lora_paths,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )

    def _encode_reference_audio_vae_latent(self, video_path: str) -> mx.array:
        """Decode audio from ``video_path``, run through audio VAE encoder.

        Returns the audio VAE latent ``(1, 8, T, 16)``.
        Mirrors upstream ``LipDubPipeline._encode_reference_audio_vae_latent``.
        """
        audio_data = load_audio(video_path, target_sample_rate=16000, mono=False)
        if audio_data is None:
            raise ValueError(f"No audio stream found in {video_path}")

        def _encode(encoder, processor) -> mx.array:
            latent = encode_audio(audio_data.waveform, audio_data.sample_rate, encoder, processor)
            _mx_materialize(latent)
            return latent

        return self.audio_conditioner(_encode, free_after=False)

    def generate_lipdub(
        self,
        prompt: str,
        reference_video_path: str,
        height: int = 480,
        width: int = 704,
        reference_strength: float = 1.0,
        images: list[tuple[str, int, float]] | None = None,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
    ) -> tuple[mx.array, mx.array, float]:
        """Run the lipdub pipeline. Returns ``(video_latent, audio_latent, frame_rate)``.

        Args:
            prompt: Text prompt.
            reference_video_path: Path to reference video (provides visual + audio).
            height: Output video height.
            width: Output video width.
            reference_strength: IC-LoRA reference video strength.
            images: Optional I2V anchor images (upstream-iso multi-anchor).
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps (default: full distilled schedule).
            stage2_steps: Stage 2 denoising steps (default: full STAGE_2 schedule).
        """
        meta = probe_video_info(reference_video_path)
        num_frames = _snap_frames_to_8k1(meta.num_frames)
        frame_rate = float(meta.fps)
        logger.info(
            f"LipDub: reference video {meta.num_frames} frames at {frame_rate} fps -> {num_frames} frames (8k+1 snap)"
        )

        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        _mx_materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        ref_audio_latent = self._encode_reference_audio_vae_latent(reference_video_path)
        ref_audio_tokens, ref_audio_positions = patchify_lipdub_audio_reference_latent(
            ref_audio_latent,
            self.audio_patchifier,
            negative_positions=True,
        )
        if self.low_memory:
            self.audio_conditioner.free()
            aggressive_cleanup()

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        self._fuse_loras()

        # ===== Stage 1 (half-res) =====
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        stage_1_conditionings: list = []
        if images:
            normalized = [
                img if isinstance(img, ImageConditioningInput) else ImageConditioningInput(*img) for img in images
            ]
            stage_1_conditionings.extend(
                combined_image_conditionings(
                    normalized,
                    enc_h=H_half * 32,
                    enc_w=W_half * 32,
                    spatial_dims=(F, H_half, W_half),
                    video_encoder=self.vae_encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
            )
        append_ic_lora_reference_video_conditionings(
            stage_1_conditionings,
            [(reference_video_path, reference_strength)],
            height=half_h,
            width=half_w,
            num_frames=num_frames,
            video_encoder=self.vae_encoder,
            reference_downscale_factor=self.reference_downscale_factor,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
        )

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
            spatial_dims=(F, H_half, W_half),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
        )
        ref_cond = AudioConditionByReferenceLatent(
            patchified=ref_audio_tokens,
            positions=ref_audio_positions,
            strength=1.0,
        )
        audio_state = ref_cond.apply(audio_state, num_noisy_tokens=audio_T)

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

        gen_video_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        s1_audio_latent_tokens = output_1.audio_latent[:, :audio_T, :]

        # ===== Upscale =====
        video_half = self.video_patchifier.unpatchify(gen_video_tokens, (F, H_half, W_half))
        assert self.upsampler is not None
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx).transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = self.vae_encoder.normalize_latent(video_upscaled.transpose(0, 2, 3, 4, 1))
        video_upscaled = video_up_mlx.transpose(0, 4, 1, 2, 3)
        _mx_materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2
        enc_h_full = H_full * 32
        enc_w_full = W_full * 32

        stage_2_conditionings: list = []
        if images:
            normalized = [
                img if isinstance(img, ImageConditioningInput) else ImageConditioningInput(*img) for img in images
            ]
            stage_2_conditionings.extend(
                combined_image_conditionings(
                    normalized,
                    enc_h=enc_h_full,
                    enc_w=enc_w_full,
                    spatial_dims=(F, H_full, W_full),
                    video_encoder=self.vae_encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
            )
        append_ic_lora_reference_video_conditionings(
            stage_2_conditionings,
            [(reference_video_path, reference_strength)],
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=self.vae_encoder,
            reference_downscale_factor=self.reference_downscale_factor,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
        )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None

        # IC-LoRA stays fused for stage 2 (upstream keeps loras attached on self.stage).
        # Intentional divergence from ic_lora.py which reloads a clean transformer.

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=stage_2_conditionings,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens_up,
        )

        # Stage 2 audio is frozen (upstream noise_scale=0, frozen=True).
        # sigma=0 + initial_latent=s1 keeps the audio unchanged through Euler steps.
        audio_state_2 = create_noised_state(
            base_shape=s1_audio_latent_tokens.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=0.0,
            initial_latent=s1_audio_latent_tokens,
        )
        audio_state_2 = ref_cond.apply(audio_state_2, num_noisy_tokens=audio_T)

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

        video_latent = self.video_patchifier.unpatchify(
            output_2.video_latent[:, : F * H_full * W_full, :], (F, H_full, W_full)
        )
        # Upstream discards stage-2 audio output and decodes s1 audio.
        audio_latent = self.audio_patchifier.unpatchify(s1_audio_latent_tokens)

        return video_latent, audio_latent, frame_rate

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        reference_video_path: str,
        height: int = 480,
        width: int = 704,
        reference_strength: float = 1.0,
        images: list[tuple[str, int, float]] | None = None,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        **_unused,
    ) -> str:
        video_latent, audio_latent, frame_rate = self.generate_lipdub(
            prompt=prompt,
            reference_video_path=reference_video_path,
            height=height,
            width=width,
            reference_strength=reference_strength,
            images=images,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()
        result = self._decode_and_save_video(video_latent, audio_latent, output_path, frame_rate=frame_rate)
        if self.low_memory:
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()
        return result


__all__ = ["LipDubPipeline", "patchify_lipdub_audio_reference_latent"]
