"""Multi-modal guidance system for diffusion generation.

Ported from ltx-core/src/ltx_core/components/guiders.py

Provides CFG (classifier-free guidance), STG (spatio-temporal guidance),
and modality-level guidance with optional rescaling. The distilled model
has CFG baked in; these guiders are needed for the full (non-distilled) model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import mlx.core as mx


def projection_coef(to_project: mx.array, project_onto: mx.array) -> mx.array:
    """Compute the scalar projection coefficient of to_project onto project_onto.

    Args:
        to_project: Array to project, shape (B, ...).
        project_onto: Array to project onto, shape (B, ...).

    Returns:
        Scalar coefficient (B, 1) such that proj = coef * project_onto.
    """
    batch_size = to_project.shape[0]
    positive_flat = to_project.reshape(batch_size, -1)
    negative_flat = project_onto.reshape(batch_size, -1)
    dot_product = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
    squared_norm = mx.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
    return dot_product / squared_norm


@dataclass(frozen=True)
class MultiModalGuiderParams:
    """Parameters for the multi-modal guider."""

    cfg_scale: float = 1.0
    """CFG (Classifier-free guidance) scale controlling how strongly the model adheres to the prompt."""

    stg_scale: float = 0.0
    """STG (Spatio-Temporal Guidance) scale controls how strongly the model reacts to the perturbation."""

    stg_blocks: list[int] | None = field(default_factory=list)
    """Which transformer blocks to perturb for STG."""

    rescale_scale: float = 0.0
    """Rescale scale controlling how strongly the model rescales the prediction after guidance."""

    modality_scale: float = 1.0
    """Modality scale controlling how strongly the model reacts to the modality perturbation."""

    skip_step: int = 0
    """Skip step controlling how often the model skips the guidance step."""


def _params_for_sigma_from_sorted_dict(
    sigma: float,
    params_by_sigma: Sequence[tuple[float, MultiModalGuiderParams]],
) -> MultiModalGuiderParams:
    """Return params for the given sigma from a sorted (sigma_upper_bound -> params) structure.

    Keys are sorted descending (bin upper bounds). Bin i is (key_{i+1}, key_i].
    Get all keys >= sigma; use last in list (smallest such key = upper bound of
    bin containing sigma), or first entry if sigma is above all keys.

    Args:
        sigma: Current noise level.
        params_by_sigma: Sequence of (sigma_bound, params) sorted descending by sigma_bound.

    Returns:
        MultiModalGuiderParams for the bin containing sigma.
    """
    if not params_by_sigma:
        raise ValueError("params_by_sigma must be non-empty")
    sigma = float(sigma)
    keys_desc = [k for k, _ in params_by_sigma]
    keys_ge_sigma = [k for k in keys_desc if k >= sigma]
    # sigma above all keys: use first bin (max key)
    key = keys_ge_sigma[-1] if keys_ge_sigma else keys_desc[0]
    return next(p for k, p in params_by_sigma if k == key)


@dataclass(frozen=True)
class MultiModalGuider:
    """Multi-modal guider with constant params per instance.

    For sigma-dependent params, use MultiModalGuiderFactory.build_from_sigma(sigma)
    to obtain a guider for each step.
    """

    params: MultiModalGuiderParams
    negative_context: mx.array | None = None

    def calculate(
        self,
        cond: mx.array,
        uncond_text: mx.array | float,
        uncond_perturbed: mx.array | float,
        uncond_modality: mx.array | float,
    ) -> mx.array:
        """Apply multi-modal guidance to compute the guided prediction.

        The guider calculates the guidance as:
            cond + (cfg_scale - 1) * (cond - uncond_text)
                 + stg_scale * (cond - uncond_perturbed)
                 + (modality_scale - 1) * (cond - uncond_modality)

        With optional rescaling to preserve the norm of the conditioned prediction.

        Args:
            cond: Conditioned prediction.
            uncond_text: Unconditional (text-dropped) prediction for CFG.
            uncond_perturbed: Perturbed prediction for STG.
            uncond_modality: Modality-dropped prediction for modality guidance.

        Returns:
            Guided prediction.
        """
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )

        if self.params.rescale_scale != 0:
            factor = mx.sqrt(mx.var(cond)) / (mx.sqrt(mx.var(pred)) + 1e-8)
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor

        return pred

    def do_unconditional_generation(self) -> bool:
        """Returns True if the guider is doing unconditional generation (CFG active)."""
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        """Returns True if the guider is doing perturbed generation (STG active)."""
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        """Returns True if the guider is doing isolated modality generation."""
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        """Returns True if the guider should skip guidance at this step."""
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0


@dataclass(frozen=True)
class MultiModalGuiderFactory:
    """Factory that creates a MultiModalGuider for a given sigma.

    Single source of truth: _params_by_sigma (schedule). Use constant() for
    one params for all sigma, from_dict() for sigma-binned params.
    """

    negative_context: mx.array | None = None
    _params_by_sigma: tuple[tuple[float, MultiModalGuiderParams], ...] = ()

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        negative_context: mx.array | None = None,
    ) -> MultiModalGuiderFactory:
        """Build a factory with constant params (same guider for all sigma).

        Args:
            params: Guidance parameters to use at all sigma levels.
            negative_context: Optional negative text embeddings for CFG.

        Returns:
            Factory that produces the same guider for all sigma.
        """
        return cls(
            negative_context=negative_context,
            _params_by_sigma=((float("inf"), params),),
        )

    @classmethod
    def from_dict(
        cls,
        sigma_to_params: Mapping[float, MultiModalGuiderParams],
        negative_context: mx.array | None = None,
    ) -> MultiModalGuiderFactory:
        """Build a factory from a dict of sigma_value -> MultiModalGuiderParams.

        Keys are sorted descending and used for bin lookup in params(sigma).

        Args:
            sigma_to_params: Mapping from sigma upper bounds to guidance params.
            negative_context: Optional negative text embeddings for CFG.

        Returns:
            Factory with sigma-dependent guidance parameters.
        """
        if not sigma_to_params:
            raise ValueError("sigma_to_params must be non-empty")
        sorted_items = tuple(sorted(sigma_to_params.items(), key=lambda x: x[0], reverse=True))
        return cls(negative_context=negative_context, _params_by_sigma=sorted_items)

    def params(self, sigma: float | mx.array) -> MultiModalGuiderParams:
        """Return params effective for the given sigma.

        Args:
            sigma: Current noise level (scalar or 0-d array).

        Returns:
            MultiModalGuiderParams for the bin containing sigma.
        """
        sigma_val = float(sigma.item() if isinstance(sigma, mx.array) else sigma)
        return _params_for_sigma_from_sorted_dict(sigma_val, self._params_by_sigma)

    def build_from_sigma(self, sigma: float | mx.array) -> MultiModalGuider:
        """Return a MultiModalGuider with params effective for the given sigma.

        Args:
            sigma: Current noise level.

        Returns:
            MultiModalGuider configured for this sigma level.
        """
        return MultiModalGuider(
            params=self.params(sigma),
            negative_context=self.negative_context,
        )


def create_multimodal_guider_factory(
    params: MultiModalGuiderParams | MultiModalGuiderFactory,
    negative_context: mx.array | None = None,
) -> MultiModalGuiderFactory:
    """Create or return a MultiModalGuiderFactory.

    Pass constant params for a single-params factory (uses
    MultiModalGuiderFactory.constant), or an existing MultiModalGuiderFactory.
    When given a factory, returns it as-is unless negative_context is provided.
    For sigma-dependent params use MultiModalGuiderFactory.from_dict(...) and
    pass that as params.

    Args:
        params: Either MultiModalGuiderParams for constant guidance, or an
            existing MultiModalGuiderFactory.
        negative_context: Optional negative text embeddings for CFG.

    Returns:
        A MultiModalGuiderFactory ready to produce guiders.
    """
    if isinstance(params, MultiModalGuiderFactory):
        if negative_context is not None and params.negative_context is not negative_context:
            return MultiModalGuiderFactory.from_dict(
                dict(params._params_by_sigma),
                negative_context=negative_context,
            )
        return params
    return MultiModalGuiderFactory.constant(params, negative_context=negative_context)
