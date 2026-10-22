"""Experimental single-stage Audio-to-Video pipeline for MLX.

The official LTX-2 distribution currently exposes A2V as a two-stage
pipeline.  This local variant keeps the same frozen input-audio conditioning
used by :class:`A2VidPipelineTwoStage`, but denoises video once at the target
resolution and skips the spatial upsampler/refine stage.

It is intended for low-resolution drafts.  At the same large output size a
full-resolution one-stage pass can be slower than the standard half-resolution
stage 1 plus short stage 2 refinement.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines_mlx.scheduler import ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE
from ltx_pipelines_mlx.utils.helpers import create_noised_state


class A2VidPipelineOneStage(A2VidPipelineTwoStage):
    """Dev + CFG A2V at the requested resolution without stage 2."""

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
            distilled_lora="",
            distilled_lora_strength=0.0,
            tile_count=tile_count,
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
        num_steps: int = 16,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        image: str | None = None,
        images=None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> str:
        if audio_path is None:
            raise ValueError("audio_path is required for A2VidPipelineOneStage")
        if audio_max_duration is None:
            audio_max_duration = num_frames / frame_rate

        # Encode the selected source-audio segment and freeze it during denoise.
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
        audio_t = compute_audio_token_count(num_frames, frame_rate)
        audio_latent = audio_latent[:, :, :audio_t, :]
        audio_tokens, _ = self.audio_patchifier.patchify(audio_latent)
        mx.synchronize()
        if self.low_memory:
            self.audio_conditioner.free()

        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        latent_frames, latent_height, latent_width = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, latent_frames * latent_height * latent_width, 128)
        video_positions = compute_video_positions(latent_frames, latent_height, latent_width, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_t)

        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings: list = []
        if resolved_images:

            def _encode_images(encoder):
                result = combined_image_conditionings(
                    resolved_images,
                    enc_h=latent_height * 32,
                    enc_w=latent_width * 32,
                    spatial_dims=(latent_frames, latent_height, latent_width),
                    video_encoder=encoder,
                    frame_rate=frame_rate,
                    model_dir=self.model_dir,
                )
                mx.synchronize()
                return result

            conditionings = self.image_conditioner(_encode_images, free_after=self.low_memory)
            if self.low_memory:
                aggressive_cleanup()

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings,
            spatial_dims=(latent_frames, latent_height, latent_width),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=mx.zeros((1, audio_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=audio_positions,
        )

        num_tokens = latent_frames * latent_height * latent_width
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        stage_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler = VideoModalityTiler(
                self._tile_count,
                latent_shape=(latent_frames, latent_height, latent_width),
            )
            stage_dit = TiledLTXModel(self.dit, tiler)

        output = self._denoise_stage1(
            x0_model=X0Model(stage_dit),
            video_state=video_state,
            audio_state=audio_state,
            video_embeds=video_embeds,
            audio_embeds=audio_embeds,
            neg_video_embeds=neg_video_embeds,
            neg_audio_embeds=neg_audio_embeds,
            sigmas=sigmas,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
        )
        generated_tokens = output.video_latent[:, : latent_frames * latent_height * latent_width, :]
        video_latent = self.video_patchifier.unpatchify(
            generated_tokens,
            (latent_frames, latent_height, latent_width),
        )
        mx.synchronize()

        if self.low_memory:
            self.dit = None
            self._loaded = False
            aggressive_cleanup()
        self._load_decoders()

        # Preserve the original waveform in the mp4, matching two-stage A2V.
        video_duration = num_frames / frame_rate
        audio_data_48k = load_audio(
            audio_path,
            target_sample_rate=48000,
            start_time=audio_start_time,
            max_duration=video_duration,
        )
        temp_audio: str | None = None
        if audio_data_48k is not None:
            max_samples = int(video_duration * 48000)
            waveform_48k = audio_data_48k.waveform[:, :, :max_samples]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio = tmp.name
            self._save_waveform(waveform_48k, temp_audio, sample_rate=48000)

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


__all__ = ["A2VidPipelineOneStage"]
