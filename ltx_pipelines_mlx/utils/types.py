"""Pipeline-level types matching upstream ltx-pipelines/utils/types.py.

These dataclasses and protocols form the orchestration vocabulary used
by ``DiffusionStage`` and the helper functions in
``ltx_pipelines_mlx.utils.helpers``. They mirror the upstream LTX-2
abstractions 1:1 so that pipeline code can be read side-by-side with
the upstream Python files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx

if TYPE_CHECKING:
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage.

    Carries everything needed to build the initial noised latent state
    and run the denoising loop for a single modality (video or audio).

    Attributes:
        context: Pre-projected text embeddings, shape ``(B, Nt, dim)``.
        conditionings: List of conditioning items applied to the initial
            state before noising (e.g. ``VideoConditionByKeyframeIndex``).
        noise_scale: Scalar in ``[0, 1]`` controlling how much noise is
            mixed in. ``1.0`` = pure noise (typical stage 1 with
            ``initial_latent=None``); ``stage_2_sigmas[0]`` = partial
            noise mixed with ``initial_latent``.
        frozen: When True the state is built but its ``denoise_mask`` is
            zeroed so all tokens are treated as preserved. Used when an
            audio modality is passed for cross-modal context only and
            should not be re-denoised.
        initial_latent: Optional starting latent. ``None`` means start
            from zeros (typical stage 1). Provided as the upscaled
            stage-1 output for stage 2.
    """

    context: mx.array
    conditionings: list[Any] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: mx.array | None = None


class Denoiser(Protocol):
    """Protocol for a denoiser that receives the transformer at call time.

    Mirrors upstream ``ltx_pipelines.utils.types.Denoiser``. The
    transformer is not stored — it is passed as the first argument so
    the caller (a denoising loop or a pipeline block) controls its
    lifecycle.

    Args:
        transformer: The diffusion model (typically an X0Model wrapper).
        video_state: Current video latent state, or ``None`` if absent.
        audio_state: Current audio latent state, or ``None`` if absent.
        sigmas: 1-D array of sigma values for each diffusion step.
        step_index: Index of the current denoising step.

    Returns:
        ``(denoised_video, denoised_audio)`` arrays (either may be ``None``).
    """

    def __call__(
        self,
        transformer: Any,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: mx.array,
        step_index: int,
    ) -> tuple[mx.array | None, mx.array | None]: ...
