"""ltx-pipelines — Generation pipelines for LTX-2.3 on MLX.

Public pipeline classes mirror upstream Lightricks/LTX-2 pipelines 1:1:

| Public class | Upstream equivalent |
|---|---|
| ``TI2VidOneStagePipeline`` | ``ti2vid_one_stage.TI2VidOneStagePipeline`` |
| ``TI2VidTwoStagesPipeline`` | ``ti2vid_two_stages.TI2VidTwoStagesPipeline`` |
| ``TI2VidTwoStagesHQPipeline`` | ``ti2vid_two_stages_hq.TI2VidTwoStagesHQPipeline`` |
| ``DistilledPipeline`` | ``distilled.DistilledPipeline`` |
| ``ICLoraPipeline`` | ``ic_lora.ICLoraPipeline`` |
| ``HDRICLoraPipeline`` | ``hdr_ic_lora.HDRICLoraPipeline`` |
| ``LipDubPipeline`` | ``lipdub.LipDubPipeline`` |
| ``KeyframeInterpolationPipeline`` | ``keyframe_interpolation.KeyframeInterpolationPipeline`` |
| ``A2VidPipelineOneStage`` | local experimental A2V draft pipeline |
| ``A2VidPipelineTwoStage`` | ``a2vid_two_stage.A2VidPipelineTwoStage`` |
| ``RetakePipeline`` | ``retake.RetakePipeline`` (extend folded in) |

The :class:`BasePipeline` class lives in the private ``_base`` module —
it's the shared inheritance parent for all the public pipelines, with
no upstream counterpart (upstream uses pure composition). I2V is
supported on every public pipeline by passing ``image=...`` to
``generate*``; there is no dedicated ``ImageToVideoPipeline``.
"""

from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.a2vid_one_stage import A2VidPipelineOneStage
from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.hdr_ic_lora import HDRICLoraPipeline
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines_mlx.lipdub import LipDubPipeline
from ltx_pipelines_mlx.retake import RetakePipeline
from ltx_pipelines_mlx.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines_mlx.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)

__all__ = [
    "A2VidPipelineOneStage",
    "A2VidPipelineTwoStage",
    # Composition blocks (mirror upstream utils/blocks.py)
    "AudioConditioner",
    "AudioDecoder",
    "BasePipeline",
    "DistilledPipeline",
    "HDRICLoraPipeline",
    "ICLoraPipeline",
    "ImageConditioner",
    "KeyframeInterpolationPipeline",
    "LipDubPipeline",
    "PromptEncoder",
    "RetakePipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesHQPipeline",
    "TI2VidTwoStagesPipeline",
    "VideoDecoder",
    "VideoUpsampler",
]
