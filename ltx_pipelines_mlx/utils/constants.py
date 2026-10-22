"""Pipeline constants and default parameters for LTX-2.3."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_core_mlx.components.guiders import MultiModalGuiderParams

DEFAULT_IMAGE_CRF = 33
VIDEO_LATENT_CHANNELS = 128

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


@dataclass
class PipelineParams:
    """Default parameters for an LTX pipeline run.

    Args:
        seed: Random seed for reproducibility. None means random.
        stage_1_height: First-stage target height in pixels.
        stage_1_width: First-stage target width in pixels.
        num_frames: Number of video frames to generate.
        frame_rate: Output video frame rate.
        num_inference_steps: Number of denoising steps.
        video_guider_params: Guidance parameters for the video modality.
        audio_guider_params: Guidance parameters for the audio modality.
    """

    seed: int | None = None
    stage_1_height: int = 1088
    stage_1_width: int = 1920
    num_frames: int = 257
    frame_rate: int = 24
    num_inference_steps: int = 30
    video_guider_params: MultiModalGuiderParams = field(
        default_factory=MultiModalGuiderParams,
    )
    audio_guider_params: MultiModalGuiderParams = field(
        default_factory=MultiModalGuiderParams,
    )


LTX_2_3_PARAMS = PipelineParams(
    num_inference_steps=30,
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    ),
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    ),
)

LTX_2_3_HQ_PARAMS = PipelineParams(
    num_inference_steps=15,
    stage_1_height=1088 // 2,
    stage_1_width=1920 // 2,
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
)
