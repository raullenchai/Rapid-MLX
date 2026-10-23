"""Retake / Extend pipeline — edit an existing video.

Matches the reference architecture: single-stage with dev model + CFG
guidance. Default: 30 steps, cfg=3.0, full multi-modal guidance.

The class hosts both:

- ``retake`` / ``retake_from_video`` — regenerate a temporal region of
  an existing video (matches upstream ``ltx_pipelines.retake``).
- ``extend`` / ``extend_from_video`` — append or prepend frames to an
  existing video. No upstream equivalent; folded into this file rather
  than a separate ``ExtendPipeline`` class so the file structure stays
  isomorphic with upstream's ``retake.py``.

Both modes share the same dev transformer, text encoding (positive +
negative for CFG), and ``guided_denoise_loop`` invocation — they only
differ in how the ``denoise_mask`` is constructed and whether new
tokens are appended to the source latent.

Ported from ltx-pipelines/src/ltx_pipelines/retake.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    TemporalRegionMask,
    noise_latent_state,
)
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.image import load_video_frames
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.scheduler import ltx2_schedule
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

# Reference defaults (LTX_2_3_PARAMS)
DEFAULT_CFG_SCALE = 3.0
DEFAULT_STG_SCALE = 1.0  # upstream LTX_2_3_PARAMS default


@dataclass(frozen=True)
class _SourceMeta:
    """Source video metadata kept after the encoder is freed."""

    height: int
    width: int
    frame_rate: float
    num_frames: int


class RetakePipeline(BasePipeline):
    """Retake pipeline: regenerate a time segment while preserving the rest.

    Uses the dev (non-distilled) model with CFG guidance for quality output.
    Single-stage pipeline (no upsampler).

    Args:
        model_dir: Path to model weights.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        dev_transformer: str = "transformer-dev.safetensors",
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self._dev_transformer = dev_transformer

    def _encode_source_video(
        self,
        video_path: str | Path,
    ) -> tuple[mx.array, mx.array, _SourceMeta]:
        """Probe the source video, encode video + audio latents via blocks, free.

        Composes :class:`ImageConditioner` (video VAE encoder) and
        :class:`AudioConditioner` (audio VAE + processor) blocks rather
        than reaching for inherited ``self._load_*`` methods.
        """
        video_path = str(video_path)
        info = probe_video_info(video_path)
        # Round down to VAE-compatible frame count (1 + 8k).
        k = max(1, (info.num_frames - 1) // 8)
        vae_compatible_frames = 1 + k * 8

        # Encode video via ImageConditioner (loads → encodes → frees)
        video_tensor = load_video_frames(video_path, info.height, info.width, vae_compatible_frames)

        def _encode_video(encoder) -> mx.array:
            latent = encoder.encode(video_tensor)
            mx.synchronize()
            return latent

        video_latent = self.image_conditioner(_encode_video, free_after=self.low_memory)
        if self.low_memory:
            del video_tensor
            aggressive_cleanup()

        # Encode audio via AudioConditioner (loads → encodes → frees) if present
        audio_latent: mx.array | None = None
        if info.has_audio:
            audio_data = load_audio(
                video_path,
                target_sample_rate=16000,
                max_duration=vae_compatible_frames / info.fps,
            )
            if audio_data is not None:

                def _encode_audio(enc, proc) -> mx.array:
                    return encode_audio(audio_data.waveform, audio_data.sample_rate, enc, proc)

                audio_latent = self.audio_conditioner(_encode_audio, free_after=self.low_memory)
                if self.low_memory:
                    aggressive_cleanup()
            elif self.low_memory:
                self.audio_conditioner.free()
        elif self.low_memory:
            self.audio_conditioner.free()

        if audio_latent is None:
            audio_T = compute_audio_token_count(vae_compatible_frames, frame_rate=info.fps)
            audio_latent = mx.zeros((1, 8, audio_T, 16), dtype=mx.bfloat16)

        meta = _SourceMeta(
            height=info.height,
            width=info.width,
            frame_rate=info.fps,
            num_frames=vae_compatible_frames,
        )
        return video_latent, audio_latent, meta

    def retake_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video file.

        Args:
            prompt: Text prompt for the regenerated segment.
            video_path: Path to the source video file.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
            regenerate_audio: If True, regenerate audio in the retake region.
                If False, preserve original audio entirely.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        video_latent, audio_latent, meta = self._encode_source_video(video_path)
        return self.retake(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            start_frame=start_frame,
            end_frame=end_frame,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            frame_rate=meta.frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            regenerate_audio=regenerate_audio,
        )

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        direction: str = "after",
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
    ) -> tuple[mx.array, mx.array]:
        """Append or prepend ``extend_frames`` latent frames to a source video.

        Mirrors :meth:`retake_from_video`'s shape but instead of regenerating
        an interior region, grows the latent (and audio) on one side.

        Args:
            prompt: Text prompt describing the new frames.
            video_path: Path to the source video file.
            extend_frames: Number of latent frames to add.
            direction: ``"before"`` or ``"after"``.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        video_latent, audio_latent, meta = self._encode_source_video(video_path)
        return self.extend(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            extend_frames=extend_frames,
            direction=direction,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            frame_rate=meta.frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
        )

    def retake(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        start_frame: int,
        end_frame: int,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video.

        Args:
            prompt: Text prompt for the regenerated segment.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            height: Video height.
            width: Video width.
            num_frames: Total number of pixel frames.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
            regenerate_audio: If True, regenerate audio in the retake region.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        # --- Text encoding (positive + negative for CFG) ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load dev transformer ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute positions
        video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # Create video state with temporal mask (1 = regenerate, 0 = preserve)
        region = TemporalRegionMask(start_frame, end_frame)
        denoise_mask = region.create_mask(F, tokens_per_frame)

        video_state = LatentState(
            latent=source_tokens,
            clean_latent=source_tokens,
            denoise_mask=denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio state
        if regenerate_audio:
            # Apply same temporal mask to audio
            audio_tokens_per_video_frame = audio_T / F
            audio_start = round(start_frame * audio_tokens_per_video_frame)
            audio_end = round(end_frame * audio_tokens_per_video_frame)

            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)
            audio_mask = audio_mask.at[:, audio_start:audio_end, :].add(
                mx.ones((1, audio_end - audio_start, 1), dtype=mx.bfloat16)
            )
        else:
            # Preserve all audio (mask=0)
            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)

        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=audio_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # --- Guided denoising ---
        num_tokens = F * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guidance (ref LTX_2_3_PARAMS)
        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_gp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

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
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent

    def extend(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        extend_frames: int,
        direction: str = "after",
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video by adding ``extend_frames`` latent frames.

        Same dev model + CFG flow as :meth:`retake`; the difference lives
        only in how the denoise mask + clean latent are constructed (new
        tokens are appended/prepended rather than regenerated in place).

        Args:
            prompt: Text prompt for the new frames.
            source_video_latent: ``(B, C, F, H, W)`` source video latent.
            source_audio_latent: ``(B, 8, T, 16)`` source audio latent.
            extend_frames: Number of latent frames to add.
            direction: ``"before"`` or ``"after"``.
            height: Source video height.
            width: Source video width.
            num_frames: Total source pixel-frame count.
            frame_rate: Source frame rate (used to size the audio extension).
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        B = source_video_latent.shape[0]
        F_source = source_video_latent.shape[2]
        _, H, W = compute_video_latent_shape(1, height, width)
        F_total = F_source + extend_frames
        tokens_per_frame = H * W

        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute total pixel frames for audio token count (each latent frame ≈ 8 pixel frames).
        total_pixel_frames = num_frames + extend_frames * 8
        extend_audio_T = max(0, compute_audio_token_count(total_pixel_frames, frame_rate=frame_rate) - audio_T)
        audio_total_T = audio_T + extend_audio_T

        new_video_shape = (B, extend_frames * tokens_per_frame, 128)

        if direction == "after":
            video_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                    mx.ones((B, new_video_shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_video = mx.concatenate(
                [source_tokens, mx.zeros(new_video_shape, dtype=mx.bfloat16)],
                axis=1,
            )
            audio_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate(
                [audio_tokens, mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16)],
                axis=1,
            )
        else:  # before
            video_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, new_video_shape[1], 1), dtype=mx.bfloat16),
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_video = mx.concatenate(
                [mx.zeros(new_video_shape, dtype=mx.bfloat16), source_tokens],
                axis=1,
            )
            audio_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate(
                [mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16), audio_tokens],
                axis=1,
            )

        video_positions = compute_video_positions(F_total, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_total_T)

        video_state = LatentState(
            latent=clean_video,
            clean_latent=clean_video,
            denoise_mask=video_denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        audio_state = LatentState(
            latent=clean_audio,
            clean_latent=clean_audio,
            denoise_mask=audio_denoise_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        num_tokens = F_total * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_gp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

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
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F_total, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
