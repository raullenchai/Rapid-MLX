"""Vendored generation core (upstream ``mlx_vlm.generate`` @ 0.7.2).

VENDOR-DEVIATION(subset-exports): the upstream ``generate/__init__.py``
eagerly imports every modality module (dispatch, image, audio, video,
diffusion, edit_image). Only the text AR core is vendored so far, so this
init re-exports just the vendored surface; the modality modules land in
later step-3 slices and the export list grows to full parity then.
"""

from .ar import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    PromptProcessingBatch,
    batch_generate,
    generate_step,
)
from .common import (
    GenerationResult,
    PromptCacheState,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from .types import GenerateKwargs, ProcessorLike

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerateKwargs",
    "GenerationResult",
    "PromptCacheState",
    "PromptProcessingBatch",
    "ProcessorLike",
    "batch_generate",
    "generate_step",
    "generation_stream",
    "maybe_quantize_kv_cache",
    "wired_limit",
]
