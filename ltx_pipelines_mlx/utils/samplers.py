"""Euler denoising loop for joint audio+video diffusion.

Ported from ltx-pipelines denoising loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_arsenal.diffusion import euler_step
from tqdm import tqdm

from ltx_core_mlx.components.guiders import MultiModalGuiderFactory
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, apply_denoise_mask
from ltx_core_mlx.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.res2s import get_res2s_coefficients, phi


def _channelwise_normalize(x: mx.array) -> mx.array:
    """Normalize noise to zero mean and unit std per channel.

    Matches reference _channelwise_normalize + global normalization in _get_new_noise.
    Input x has shape (B, N, C) where N = num_tokens.
    """
    # Global normalization first: zero-mean, unit-std
    x = (x - mx.mean(x)) / (mx.std(x) + 1e-8)
    # Per-channel normalization over token dimension
    mean = mx.mean(x, axis=1, keepdims=True)
    std = mx.std(x, axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


@dataclass
class DenoiseOutput:
    """Output of the denoising loop."""

    video_latent: mx.array  # (B, N_video, C)
    audio_latent: mx.array  # (B, N_audio, C)


def _is_uniform_mask(mask: mx.array) -> bool:
    """Check if denoise mask is all-ones (full denoise, no conditioning)."""
    return bool(mx.all(mask == 1.0).item())


def _compute_per_token_timesteps(
    sigma: float,
    denoise_mask: mx.array,
) -> mx.array:
    """Compute per-token timesteps from sigma and denoise mask.

    Preserved tokens (mask=0) get timestep=0, generated tokens (mask=1)
    get timestep=sigma.

    Args:
        sigma: Current noise level scalar.
        denoise_mask: (B, N, 1) mask.

    Returns:
        Per-token timesteps (B, N).
    """
    return (denoise_mask * sigma).squeeze(-1)


def denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float] | None = None,
    video_positions: mx.array | None = None,
    audio_positions: mx.array | None = None,
    video_attention_mask: mx.array | None = None,
    audio_attention_mask: mx.array | None = None,
    show_progress: bool = True,
) -> DenoiseOutput:
    """Run the Euler denoising loop for joint audio+video.

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Text embeddings for video conditioning.
        audio_text_embeds: Text embeddings for audio conditioning.
        sigmas: Sigma schedule (defaults to DISTILLED_SIGMAS).
            The schedule already includes the terminal 0.0, so pairs are
            formed directly: ``zip(sigmas[:-1], sigmas[1:])``.
        video_positions: Positional embeddings for video.
        audio_positions: Positional embeddings for audio.
        video_attention_mask: Attention mask for video.
        audio_attention_mask: Attention mask for audio.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    if sigmas is None:
        sigmas = DISTILLED_SIGMAS

    # Resolve positions: explicit params override, then fall back to state
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions

    # Resolve attention masks from state
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent
    audio_x = audio_state.latent

    # sigmas already includes the terminal value (e.g. 0.0), so iterate
    # consecutive pairs directly — no extra phantom step.
    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising", disable=not show_progress)

    # Determine whether we need per-token timesteps (for conditioning masks).
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    for sigma, sigma_next in iterator:
        # Build sigma / per-token timesteps
        sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
        B = video_x.shape[0]

        call_kwargs: dict = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=mx.broadcast_to(sigma_arr, (B,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )

        # Pass per-token timesteps when mask is not uniform
        if not video_uniform:
            call_kwargs["video_timesteps"] = _compute_per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, audio_state.denoise_mask)

        # Predict x0
        video_x0, audio_x0 = model(**call_kwargs)

        # Apply denoise mask: blend with clean latent
        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        # Euler step
        video_x = euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)

        # Force computation for memory efficiency
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()

    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


# --- Ancestral (SDE) Euler sampler (LTX-2.5) ---

# Port of the official LTX-2.5 ``EulerAncestralDiffusionStep`` (ltx-core
# ``components/diffusion_steps.py``) and ``euler_ancestral_denoising_loop``
# (ltx-pipelines ``utils/samplers.py``). Used for distilled stage 1 from
# LTX-2.5 on: each step takes a deterministic Euler step to an intermediate
# ``sigma_down <= sigma_next`` and then renoises back up to ``sigma_next``,
# rescaling the signal component by ``alpha_next / alpha_down`` so the
# transition stays variance-preserving. ``eta`` interpolates between a plain
# Euler step (``eta=0``, ``sigma_down == sigma_next``, no noise added) and a
# fully ancestral step (``eta=1``).


def ancestral_euler_step(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array | None = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> mx.array:
    """Advance one ancestral (SDE) Euler step.

    Mirrors ``EulerAncestralDiffusionStep.step`` 1:1 (rectified-flow
    parameterization, ``alpha = 1 - sigma``):

    - ``downstep_ratio = 1 + (sigma_next / sigma - 1) * eta``
    - ``sigma_down = sigma_next * downstep_ratio``
    - ``x_next = (sigma_down / sigma) * x + (1 - sigma_down / sigma) * x0``
    - renoise: ``alpha_next = 1 - sigma_next``, ``alpha_down = 1 - sigma_down``,
      ``renoise_coeff = sqrt(max(sigma_next^2 - sigma_down^2 * alpha_next^2 /
      alpha_down^2, 0))`` and
      ``x_next = (alpha_next / alpha_down) * x_next + noise * s_noise * renoise_coeff``.

    All math runs in float32 (inputs are cast); the result is returned in
    float32 — the caller casts back to the model dtype (bfloat16), matching
    the reference loop's ``model_dtype`` handling.

    Args:
        sample: Current noisy latent x_t.
        denoised: Denoised prediction x_0.
        sigma: Current noise level.
        sigma_next: Next noise level in the schedule.
        noise: Noise tensor for the renoise term. Required when ``eta > 0``;
            unused (and may be ``None``) when ``eta == 0``.
        eta: Stochastic noise injection strength (0=deterministic, 1=maximum).
        s_noise: Noise multiplier for the injected term.

    Returns:
        Updated latent x_{t-1} in float32, or ``denoised`` when ``sigma_next == 0``.
    """
    if sigma_next == 0:
        return denoised.astype(mx.float32)
    if eta > 0 and noise is None:
        raise ValueError("ancestral_euler_step requires a noise tensor when eta > 0")

    x = sample.astype(mx.float32)
    d = denoised.astype(mx.float32)

    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio

    # Euler step to sigma_down, expressed as an interpolation between x and x0.
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * d

    if eta > 0:
        # Renoise from sigma_down back up to sigma_next.
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_coeff = mx.sqrt(
            mx.maximum(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0)
        )
        x_next = (alpha_next / alpha_down) * x_next + noise.astype(mx.float32) * s_noise * renoise_coeff
    return x_next


def ancestral_denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float] | None = None,
    video_positions: mx.array | None = None,
    audio_positions: mx.array | None = None,
    video_attention_mask: mx.array | None = None,
    audio_attention_mask: mx.array | None = None,
    show_progress: bool = True,
    noise_seed: int = -1,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> DenoiseOutput:
    """Run the ancestral (SDE) Euler denoising loop for joint audio+video.

    Port of the official ``euler_ancestral_denoising_loop``: identical model
    call pattern to :func:`denoise_loop`, but each non-terminal step draws
    fresh Gaussian noise (same shape as the latent, bfloat16 like the latent
    state) and advances via :func:`ancestral_euler_step` instead of the plain
    Euler step.

    Noise semantics (reference-verbatim):

    - One generator seeded from ``noise_seed`` (the pipeline passes
      ``seed + 10000``, ``ANCESTRAL_NOISE_SEED_OFFSET``).
    - Per step, video noise is drawn first, then audio, from that seeded
      stream. In MLX this is realised with a single base key split into
      per-step keys, each split again into (video, audio) subkeys.
    - ``eta=0`` disables noise entirely (``draw_noise=False``) — the loop then
      reduces to the deterministic Euler loop, and ``noise_seed`` has no
      effect (matches the reference's ``draw_noise=stepper.eta > 0`` gate).

    Steps run in float32; latents are held in bfloat16 (``model_dtype``),
    matching the reference. On a terminal ``sigma_next == 0`` the denoised
    prediction is taken directly and the loop breaks.

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Text embeddings for video conditioning.
        audio_text_embeds: Text embeddings for audio conditioning.
        sigmas: Sigma schedule (defaults to DISTILLED_SIGMAS).
        video_positions: Positional embeddings for video (defaults to state).
        audio_positions: Positional embeddings for audio (defaults to state).
        video_attention_mask: Attention mask for video (defaults to state).
        audio_attention_mask: Attention mask for audio (defaults to state).
        show_progress: Whether to show tqdm progress bar.
        noise_seed: Seed for the per-step SDE noise generator (pipeline seed +
            ``ANCESTRAL_NOISE_SEED_OFFSET`` in the distilled wiring).
        eta: Stochastic noise injection strength (0=deterministic, 1=maximum).
        s_noise: Noise multiplier for the injected term.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    if sigmas is None:
        sigmas = DISTILLED_SIGMAS

    # Resolve positions / attention masks from state (same as denoise_loop).
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent
    audio_x = audio_state.latent

    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising (ancestral)", disable=not show_progress)

    # Whether per-token timesteps are needed (conditioning masks present).
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    # Reference-verbatim: one seeded generator per loop; video noise drawn
    # first, audio second. ``noise_seed=-1`` (the reference default) is
    # normalized to a valid MLX key.
    step_keys = mx.random.split(mx.random.key(noise_seed % (1 << 64)), len(steps))

    draw_noise = eta > 0
    for step_idx, (sigma, sigma_next) in enumerate(iterator):
        # Build sigma / per-token timesteps (same call pattern as denoise_loop).
        sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
        B = video_x.shape[0]

        call_kwargs: dict = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=mx.broadcast_to(sigma_arr, (B,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        if not video_uniform:
            call_kwargs["video_timesteps"] = _compute_per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, audio_state.denoise_mask)

        # Predict x0 (bfloat16, matching the deterministic loop).
        video_x0, audio_x0 = model(**call_kwargs)

        # Apply denoise mask: blend with clean latent.
        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        # Terminal step: take the denoised prediction directly (reference
        # ``_ancestral_euler_denoising_loop`` short-circuit + break).
        if sigma_next == 0:
            video_x = video_x0.astype(mx.bfloat16)
            audio_x = audio_x0.astype(mx.bfloat16)
            mx.async_eval(video_x, audio_x)
            break

        # Fresh noise per step: bfloat16 (the latent state's dtype) like the
        # reference ``_get_plain_noise``, cast to float32 inside the step.
        if draw_noise:
            keys = mx.random.split(step_keys[step_idx], 2)
            video_noise = mx.random.normal(video_x.shape, dtype=mx.bfloat16, key=keys[0]).astype(mx.float32)
            audio_noise = mx.random.normal(audio_x.shape, dtype=mx.bfloat16, key=keys[1]).astype(mx.float32)
        else:
            video_noise = None
            audio_noise = None

        # Ancestral step in float32.
        video_next = ancestral_euler_step(
            video_x, video_x0, sigma, sigma_next, video_noise, eta=eta, s_noise=s_noise
        )
        audio_next = ancestral_euler_step(
            audio_x, audio_x0, sigma, sigma_next, audio_noise, eta=eta, s_noise=s_noise
        )

        if draw_noise:
            # Re-apply the conditioning mask after noise injection (reference
            # ``post_process_latent``) so preserved tokens stay exactly clean.
            # Uniform masks (T2V) are identity — skip the extra casts.
            if not video_uniform:
                video_next = apply_denoise_mask(
                    video_next,
                    video_state.clean_latent.astype(mx.float32),
                    video_state.denoise_mask.astype(mx.float32),
                )
            if not audio_uniform:
                audio_next = apply_denoise_mask(
                    audio_next,
                    audio_state.clean_latent.astype(mx.float32),
                    audio_state.denoise_mask.astype(mx.float32),
                )

        # Back to the model dtype (bfloat16) for the next model call.
        video_x = video_next.astype(mx.bfloat16)
        audio_x = audio_next.astype(mx.bfloat16)

        # Force computation for memory efficiency.
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()

    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


# --- Res2s second-order sampler ---

# Re-export for backward compatibility and tests
_res2s_phi = phi


def _res2s_coefficients(h: float, c2: float = 0.5) -> tuple[float, float, float]:
    """Compute res_2s Runge-Kutta coefficients for a given step size."""
    return get_res2s_coefficients(h, {}, c2)


def _res2s_sde_coeff(
    sigma_next: float,
    sigma_up_fraction: float = 0.5,
) -> tuple[float, float, float]:
    """Compute SDE coefficients for variance-preserving noise injection.

    Returns:
        (alpha_ratio, sigma_down, sigma_up).
    """
    sigma_up = min(sigma_next * sigma_up_fraction, sigma_next * 0.9999)
    sigma_signal = 1.0 - sigma_next
    sigma_residual = max(0.0, sigma_next**2 - sigma_up**2) ** 0.5
    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio if alpha_ratio > 0 else sigma_next
    return alpha_ratio, sigma_down, sigma_up


def _sde_step(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array,
    eta: float = 0.5,
) -> mx.array:
    """Apply Res2s SDE noise injection step.

    Ported from Res2sDiffusionStep.step() in ltx-core.

    Args:
        sample: Current noisy sample.
        denoised: Denoised prediction from the model.
        sigma: Current sigma.
        sigma_next: Next sigma in the schedule.
        noise: Random noise for stochastic injection.
        eta: Stochastic noise injection strength (0=deterministic, 1=maximum).
    """
    if sigma_next == 0:
        return denoised.astype(mx.bfloat16)

    sigma_up = min(sigma_next * eta, sigma_next * 0.9999)
    sigma_signal = 1.0 - sigma_next
    sigma_residual = max(0.0, sigma_next**2 - sigma_up**2) ** 0.5
    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio if alpha_ratio > 0 else sigma_next

    if sigma_up == 0:
        return denoised.astype(mx.bfloat16)

    eps_next = (sample - denoised) / (sigma - sigma_next) if sigma != sigma_next else mx.zeros_like(sample)
    denoised_next = sample - sigma * eps_next
    x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
    return x_noised.astype(mx.bfloat16)


def res2s_denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float] | None = None,
    video_positions: mx.array | None = None,
    audio_positions: mx.array | None = None,
    video_attention_mask: mx.array | None = None,
    audio_attention_mask: mx.array | None = None,
    show_progress: bool = True,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    video_guider_factory: MultiModalGuiderFactory | None = None,
    audio_guider_factory: MultiModalGuiderFactory | None = None,
    tap: callable | None = None,
    teacache=None,
) -> DenoiseOutput:
    """Run the res_2s second-order denoising loop for joint audio+video.

    Ported from ltx-pipelines res2s_audio_video_denoising_loop. Uses a
    second-order exponential integrator with SDE noise injection at both
    substep and step levels, plus optional iterative anchor refinement.

    The algorithm:
    1. Evaluate model at current sigma -> x0 prediction
    2. Compute epsilon = x0 - x_anchor (denoised direction)
    3. Compute substep x_mid = x_anchor + h * a21 * epsilon
    4. Inject SDE noise at substep (sigma -> sub_sigma)
    5. Optionally refine anchor via bong iteration
    6. Evaluate model at sub_sigma -> second x0 prediction
    7. Combine: x_next = x_anchor + h * (b1 * eps1 + b2 * eps2)
    8. Inject SDE noise at step level (sigma -> sigma_next)

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Text embeddings for video conditioning.
        audio_text_embeds: Text embeddings for audio conditioning.
        sigmas: Sigma schedule.
        video_positions: Positional embeddings for video.
        audio_positions: Positional embeddings for audio.
        video_attention_mask: Attention mask for video.
        audio_attention_mask: Attention mask for audio.
        show_progress: Whether to show tqdm progress bar.
        bongmath: Enable iterative anchor refinement for small steps.
        bongmath_max_iter: Max iterations for bong refinement.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    import math

    if sigmas is None:
        sigmas = list(DISTILLED_SIGMAS)
    else:
        sigmas = list(sigmas)

    # Resolve positions and attention masks from state
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent.astype(mx.float32)
    audio_x = audio_state.latent.astype(mx.float32)

    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    n_full_steps = len(sigmas) - 1

    # Inject minimal sigma to avoid division by zero (matching reference)
    if sigmas[-1] == 0:
        sigmas = sigmas[:-1] + [0.0011, 0.0]

    # Step sizes in log-space: h_i = -log(sigma_{i+1} / sigma_i)
    # Only compute for the n_full_steps pairs used in the loop (skip the
    # terminal 0.0 pair which is handled separately at the end).
    hs = [-math.log(sigmas[i + 1] / sigmas[i]) for i in range(n_full_steps)]

    phi_cache: dict = {}
    c2 = 0.5

    # Set up audio guider factory if video guider is provided but audio is not
    if video_guider_factory is not None and audio_guider_factory is None:
        audio_guider_factory = MultiModalGuiderFactory(
            negative_context=None,
            _params_by_sigma=video_guider_factory._params_by_sigma,
        )

    def _predict(
        v_x: mx.array,
        a_x: mx.array,
        sig: float,
        run_pass: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run model prediction with optional guidance, then apply denoise mask.

        ``run_pass`` is an optional indirection used to plumb TeaCache (capture
        residuals on the compute path, replace the block stack on skip). Default
        invokes ``model(**kwargs)`` directly. When provided, called as
        ``run_pass(label, kwargs)`` for each guidance pass and returns
        ``(video_x0, audio_x0)``.
        """
        if run_pass is None:

            def run_pass(_label, kwargs):
                return model(**kwargs)

        sig_arr = mx.array([sig], dtype=mx.bfloat16)
        B = v_x.shape[0]
        base_kwargs: dict = dict(
            video_latent=v_x.astype(mx.bfloat16),
            audio_latent=a_x.astype(mx.bfloat16),
            sigma=mx.broadcast_to(sig_arr, (B,)),
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        if not video_uniform:
            base_kwargs["video_timesteps"] = _compute_per_token_timesteps(sig, video_state.denoise_mask)
        if not audio_uniform:
            base_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sig, audio_state.denoise_mask)

        if video_guider_factory is None:
            # Simple prediction (no guidance)
            kw = {**base_kwargs, "video_text_embeds": video_text_embeds, "audio_text_embeds": audio_text_embeds}
            v_x0, a_x0 = run_pass("cond", kw)
        else:
            # Guided prediction (CFG/STG/modality)
            video_guider = video_guider_factory.build_from_sigma(sig)
            audio_guider = audio_guider_factory.build_from_sigma(sig)

            # 1. Conditioned prediction
            cond_kw = {**base_kwargs, "video_text_embeds": video_text_embeds, "audio_text_embeds": audio_text_embeds}
            cond_v, cond_a = run_pass("cond", cond_kw)

            # 2. Unconditional prediction for CFG
            neg_v: mx.array | float = 0.0
            neg_a: mx.array | float = 0.0
            if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
                neg_v_embeds = (
                    video_guider.negative_context if video_guider.negative_context is not None else video_text_embeds
                )
                neg_a_embeds = (
                    audio_guider.negative_context if audio_guider.negative_context is not None else audio_text_embeds
                )
                neg_kw = {**base_kwargs, "video_text_embeds": neg_v_embeds, "audio_text_embeds": neg_a_embeds}
                neg_v, neg_a = run_pass("uncond", neg_kw)

            # 3. Perturbed prediction for STG
            ptb_v: mx.array | float = 0.0
            ptb_a: mx.array | float = 0.0
            if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
                perturbation_list: list[Perturbation] = []
                if video_guider.do_perturbed_generation():
                    perturbation_list.append(
                        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
                    )
                if audio_guider.do_perturbed_generation():
                    perturbation_list.append(
                        Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
                    )
                ptb_config = PerturbationConfig(perturbations=perturbation_list)
                ptb_kw = {
                    **base_kwargs,
                    "video_text_embeds": video_text_embeds,
                    "audio_text_embeds": audio_text_embeds,
                    "perturbations": BatchedPerturbationConfig(perturbations=[ptb_config] * B),
                }
                ptb_v, ptb_a = run_pass("ptb", ptb_kw)

            # 4. Isolated modality prediction
            mod_v: mx.array | float = 0.0
            mod_a: mx.array | float = 0.0
            if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
                mod_perturbations = [
                    Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                    Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                ]
                mod_kw = {
                    **base_kwargs,
                    "video_text_embeds": video_text_embeds,
                    "audio_text_embeds": audio_text_embeds,
                    "perturbations": BatchedPerturbationConfig(
                        perturbations=[PerturbationConfig(perturbations=mod_perturbations)] * B
                    ),
                }
                mod_v, mod_a = run_pass("mod", mod_kw)

            # 5. Apply guiders
            v_x0 = video_guider.calculate(cond_v, neg_v, ptb_v, mod_v)
            a_x0 = audio_guider.calculate(cond_a, neg_a, ptb_a, mod_a)

        v_x0 = apply_denoise_mask(v_x0, video_state.clean_latent, video_state.denoise_mask)
        a_x0 = apply_denoise_mask(a_x0, audio_state.clean_latent, audio_state.denoise_mask)
        return v_x0.astype(mx.float32), a_x0.astype(mx.float32)

    desc = "Denoising (res2s guided)" if video_guider_factory is not None else "Denoising (res2s)"
    iterator = tqdm(range(n_full_steps), desc=desc, disable=not show_progress)

    for step_idx in iterator:
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        h = hs[step_idx]

        x_anchor_v = video_x
        x_anchor_a = audio_x

        # --- TeaCache decision (per outer step, gated on stage-1 input) ---
        gate_signal = None
        if tap is not None or teacache is not None:
            B_v = video_x.shape[0]
            sig_arr = mx.array([sigma], dtype=mx.bfloat16)
            stage1_video_timesteps = (
                _compute_per_token_timesteps(sigma, video_state.denoise_mask) if not video_uniform else None
            )
            gate_signal = model.model.compute_gate_signal(
                video_latent=video_x.astype(mx.bfloat16),
                audio_latent=audio_x.astype(mx.bfloat16),
                timestep=mx.broadcast_to(sig_arr, (B_v,)),
                video_timesteps=stage1_video_timesteps,
            )

        should_compute_full = True
        if teacache is not None:
            should_compute_full = teacache.should_compute(step_idx, gate_signal)

        captured_residuals: dict = {"stage1": {}, "stage2": {}}
        cached_residuals = teacache.previous_residual if (teacache is not None and not should_compute_full) else None

        def _make_capture_tap(stage: str, label: str):
            def _t(v_res, a_res):
                captured_residuals[stage][label] = (v_res, a_res)  # noqa: B023

            return _t

        def _make_override(stage: str, label: str):
            v_res, a_res = cached_residuals[stage][label]  # noqa: B023

            def _o(v_hidden, a_hidden):
                return v_hidden + v_res, a_hidden + a_res

            return _o

        def _stage_run_pass(stage: str):
            if should_compute_full:  # noqa: B023

                def _rp(label: str, kwargs: dict):
                    pass_tap = _make_capture_tap(stage, label) if (tap is not None or teacache is not None) else None
                    return model(**kwargs, tap=pass_tap)
            else:

                def _rp(label: str, kwargs: dict):
                    return model(**kwargs, block_stack_override=_make_override(stage, label))

            return _rp

        # Stage 1: evaluate at current point
        denoised_v1, denoised_a1 = _predict(video_x, audio_x, sigma, run_pass=_stage_run_pass("stage1"))

        if tap is not None and "cond" in captured_residuals["stage1"]:
            v_res, a_res = captured_residuals["stage1"]["cond"]
            tap(step_idx, gate_signal, v_res, a_res)

        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

        # Substep sigma: geometric mean (exact for c2=0.5)
        sub_sigma = math.sqrt(sigma * sigma_next)

        # Epsilon = x0 - x_anchor (denoised direction, NOT velocity)
        eps_1_v = denoised_v1 - x_anchor_v
        eps_1_a = denoised_a1 - x_anchor_a

        # Substep x
        x_mid_v = x_anchor_v + h * a21 * eps_1_v
        x_mid_a = x_anchor_a + h * a21 * eps_1_a

        # SDE noise at substep (channel-normalized to match reference)
        mx.random.seed(step_idx * 10000 + 1)
        sub_noise_v = _channelwise_normalize(mx.random.normal(video_x.shape).astype(mx.float32))
        sub_noise_a = _channelwise_normalize(mx.random.normal(audio_x.shape).astype(mx.float32))
        x_mid_v = _sde_step(x_anchor_v, x_mid_v, sigma, sub_sigma, sub_noise_v).astype(mx.float32)
        x_mid_a = _sde_step(x_anchor_a, x_mid_a, sigma, sub_sigma, sub_noise_a).astype(mx.float32)

        # Bong iteration: refine anchor for stability at small step sizes
        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                x_anchor_v = x_mid_v - h * a21 * eps_1_v
                eps_1_v = denoised_v1 - x_anchor_v
                x_anchor_a = x_mid_a - h * a21 * eps_1_a
                eps_1_a = denoised_a1 - x_anchor_a

        # Stage 2: evaluate at substep
        denoised_v2, denoised_a2 = _predict(x_mid_v, x_mid_a, sub_sigma, run_pass=_stage_run_pass("stage2"))

        if teacache is not None and should_compute_full:
            teacache.cache_residual(captured_residuals)

        eps_2_v = denoised_v2 - x_anchor_v
        eps_2_a = denoised_a2 - x_anchor_a

        # Final combination
        x_next_v = x_anchor_v + h * (b1 * eps_1_v + b2 * eps_2_v)
        x_next_a = x_anchor_a + h * (b1 * eps_1_a + b2 * eps_2_a)

        # SDE noise at step level (channel-normalized to match reference)
        mx.random.seed(step_idx * 10000 + 2)
        step_noise_v = _channelwise_normalize(mx.random.normal(video_x.shape).astype(mx.float32))
        step_noise_a = _channelwise_normalize(mx.random.normal(audio_x.shape).astype(mx.float32))
        video_x = _sde_step(x_anchor_v, x_next_v, sigma, sigma_next, step_noise_v).astype(mx.float32)
        audio_x = _sde_step(x_anchor_a, x_next_a, sigma, sigma_next, step_noise_a).astype(mx.float32)

        mx.async_eval(video_x, audio_x)

    # Final cleanup: if original schedule ended at 0, do one last denoise.
    # TeaCache is bypassed for this terminal step — it's a one-shot denoise
    # outside the controller's num_steps range.
    if sigmas[-1] == 0:
        video_x0, audio_x0 = _predict(video_x, audio_x, sigmas[n_full_steps])
        video_x = video_x0
        audio_x = audio_x0
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()
    return DenoiseOutput(
        video_latent=video_x.astype(mx.bfloat16),
        audio_latent=audio_x.astype(mx.bfloat16),
    )


# --- Guided denoising with CFG/STG/modality guidance ---


def guided_denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    video_guider_factory: MultiModalGuiderFactory,
    audio_guider_factory: MultiModalGuiderFactory | None = None,
    sigmas: list[float] | None = None,
    video_positions: mx.array | None = None,
    audio_positions: mx.array | None = None,
    video_attention_mask: mx.array | None = None,
    audio_attention_mask: mx.array | None = None,
    show_progress: bool = True,
    tap: callable | None = None,
    teacache=None,  # mlx_arsenal.diffusion.TeaCacheController-compatible
) -> DenoiseOutput:
    """Run the Euler denoising loop with multi-modal guidance (CFG/STG).

    This extends denoise_loop() with classifier-free guidance (CFG),
    spatio-temporal guidance (STG), and modality guidance. For each step,
    depending on the guider configuration:

    1. Run the model with positive (conditioned) context.
    2. Optionally run with negative (unconditioned) context for CFG.
    3. Optionally run with perturbed attention for STG.
    4. Optionally run with isolated modality for modality guidance.
    5. Combine predictions using the guider's calculate() method.
    6. Apply the Euler step as usual.

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Positive text embeddings for video conditioning.
        audio_text_embeds: Positive text embeddings for audio conditioning.
        video_guider_factory: Factory producing video guiders per sigma.
        audio_guider_factory: Factory producing audio guiders per sigma.
            If None, uses video_guider_factory for audio as well.
        sigmas: Sigma schedule (defaults to DISTILLED_SIGMAS).
        video_positions: Positional embeddings for video.
        audio_positions: Positional embeddings for audio.
        video_attention_mask: Attention mask for video.
        audio_attention_mask: Attention mask for audio.
        show_progress: Whether to show tqdm progress bar.
        tap: Optional per-step instrumentation hook called as
            ``tap(step_idx, gate_signal, video_block_residual,
            audio_block_residual)`` after the conditioned-pass forward.
            Used by the TeaCache calibration script. Has no effect on
            control flow.
        teacache: Optional TeaCache controller. When provided, the loop
            calls ``teacache.should_compute(step_idx, gate_signal)`` once
            per step using block 0's modulated input from the conditioned
            pass; on True, all guidance passes run normally and their
            block residuals are cached as a dict keyed by pass label
            (``cond``, ``uncond``, ``ptb``, ``mod``); on False, the
            previous step's cached residuals replace the block stack via
            ``block_stack_override``. The transformer head still runs
            on every pass.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    if sigmas is None:
        sigmas = DISTILLED_SIGMAS

    if audio_guider_factory is None:
        # Copy params schedule but drop negative_context — video embeds have a
        # different dimension than audio embeds, so reusing the video negative
        # context for audio would cause a shape mismatch.
        audio_guider_factory = MultiModalGuiderFactory(
            negative_context=None,
            _params_by_sigma=video_guider_factory._params_by_sigma,
        )

    # Resolve positions: explicit params override, then fall back to state
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions

    # Resolve attention masks from state
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent
    audio_x = audio_state.latent

    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising (guided)", disable=not show_progress)

    # Determine whether we need per-token timesteps (for conditioning masks).
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    # Track last denoised for skip_step
    last_video_x0: mx.array | None = None
    last_audio_x0: mx.array | None = None

    for step_idx, (sigma, sigma_next) in enumerate(iterator):
        # Build guiders for this sigma level
        video_guider = video_guider_factory.build_from_sigma(sigma)
        audio_guider = audio_guider_factory.build_from_sigma(sigma)

        # Check if both guiders want to skip this step
        if (
            video_guider.should_skip_step(step_idx)
            and audio_guider.should_skip_step(step_idx)
            and last_video_x0 is not None
            and last_audio_x0 is not None
        ):
            video_x = euler_step(video_x, last_video_x0, sigma, sigma_next)
            audio_x = euler_step(audio_x, last_audio_x0, sigma, sigma_next)
            mx.async_eval(video_x, audio_x)
            continue

        # Build common model kwargs
        sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
        B = video_x.shape[0]

        base_kwargs: dict = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=mx.broadcast_to(sigma_arr, (B,)),
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )

        # Per-token timesteps for conditioning
        if not video_uniform:
            base_kwargs["video_timesteps"] = _compute_per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            base_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, audio_state.denoise_mask)

        # --- Compute the gate signal once per step (cheap, runs prelude only).
        # Used both by tap and by teacache decisions. None when neither hook needs it.
        gate_signal = None
        if tap is not None or teacache is not None:
            gate_signal = model.model.compute_gate_signal(
                video_latent=video_x,
                audio_latent=audio_x,
                timestep=mx.broadcast_to(sigma_arr, (B,)),
                video_timesteps=base_kwargs.get("video_timesteps"),
                video_positions=video_positions,
            )

        should_compute_full = True
        if teacache is not None:
            should_compute_full = teacache.should_compute(step_idx, gate_signal)

        # Pass-level helpers: capture residual on compute path, override on skip.
        # Each loop iteration creates fresh dicts so no state leaks across steps.
        captured_residuals: dict = {}
        cached_residuals = teacache.previous_residual if (teacache is not None and not should_compute_full) else None

        def _make_capture_tap(label: str):
            """Return a tap callback that stores (v_res, a_res) under label."""

            # Closures are constructed and consumed within the same loop iteration — safe.
            def _capture(v_res: mx.array, a_res: mx.array) -> None:
                captured_residuals[label] = (v_res, a_res)  # noqa: B023

            return _capture

        def _make_override(label: str):
            """Return a block_stack_override that adds the cached residual."""
            # cached_residuals is bound at definition time (read from previous step).
            v_res, a_res = cached_residuals[label]  # noqa: B023

            def _override(v_hidden: mx.array, a_hidden: mx.array) -> tuple[mx.array, mx.array]:
                return v_hidden + v_res, a_hidden + a_res

            return _override

        def _run_pass(label: str, kwargs: dict) -> tuple[mx.array, mx.array]:
            """Run one guidance pass, capturing or replaying residuals."""
            if should_compute_full:  # noqa: B023
                # Always capture residuals on compute path so teacache (or tap) can use them.
                pass_tap = _make_capture_tap(label) if (tap is not None or teacache is not None) else None
                return model(**kwargs, tap=pass_tap)
            else:
                return model(**kwargs, block_stack_override=_make_override(label))

        # --- 1. Conditioned prediction (positive context) ---
        cond_kwargs = {
            **base_kwargs,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
        }
        cond_video_x0, cond_audio_x0 = _run_pass("cond", cond_kwargs)

        if tap is not None and "cond" in captured_residuals:
            v_res, a_res = captured_residuals["cond"]
            tap(step_idx, gate_signal, v_res, a_res)

        # --- 2. Unconditional prediction for CFG ---
        neg_video_x0: mx.array | float = 0.0
        neg_audio_x0: mx.array | float = 0.0

        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            neg_video_embeds = (
                video_guider.negative_context if video_guider.negative_context is not None else video_text_embeds
            )
            neg_audio_embeds = (
                audio_guider.negative_context if audio_guider.negative_context is not None else audio_text_embeds
            )

            neg_kwargs = {
                **base_kwargs,
                "video_text_embeds": neg_video_embeds,
                "audio_text_embeds": neg_audio_embeds,
            }
            neg_video_x0, neg_audio_x0 = _run_pass("uncond", neg_kwargs)

        # --- 3. Perturbed prediction for STG ---
        ptb_video_x0: mx.array | float = 0.0
        ptb_audio_x0: mx.array | float = 0.0

        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbation_list: list[Perturbation] = []
            if video_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=video_guider.params.stg_blocks,
                    )
                )
            if audio_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        blocks=audio_guider.params.stg_blocks,
                    )
                )
            perturbation_config = PerturbationConfig(perturbations=perturbation_list)
            batched_perturbations = BatchedPerturbationConfig(perturbations=[perturbation_config] * B)

            ptb_kwargs = {
                **base_kwargs,
                "video_text_embeds": video_text_embeds,
                "audio_text_embeds": audio_text_embeds,
                "perturbations": batched_perturbations,
            }
            ptb_video_x0, ptb_audio_x0 = _run_pass("ptb", ptb_kwargs)

        # --- 4. Isolated modality prediction ---
        mod_video_x0: mx.array | float = 0.0
        mod_audio_x0: mx.array | float = 0.0

        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            mod_perturbation_list = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            mod_perturbation_config = PerturbationConfig(perturbations=mod_perturbation_list)
            mod_batched_perturbations = BatchedPerturbationConfig(perturbations=[mod_perturbation_config] * B)

            mod_kwargs = {
                **base_kwargs,
                "video_text_embeds": video_text_embeds,
                "audio_text_embeds": audio_text_embeds,
                "perturbations": mod_batched_perturbations,
            }
            mod_video_x0, mod_audio_x0 = _run_pass("mod", mod_kwargs)

        if teacache is not None and should_compute_full:
            teacache.cache_residual(captured_residuals)

        # --- 5. Apply guiders ---
        if video_guider.should_skip_step(step_idx) and last_video_x0 is not None:
            video_x0 = last_video_x0
        else:
            video_x0 = video_guider.calculate(cond_video_x0, neg_video_x0, ptb_video_x0, mod_video_x0)

        if audio_guider.should_skip_step(step_idx) and last_audio_x0 is not None:
            audio_x0 = last_audio_x0
        else:
            audio_x0 = audio_guider.calculate(cond_audio_x0, neg_audio_x0, ptb_audio_x0, mod_audio_x0)

        # Apply denoise mask: blend with clean latent
        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        # Track for skip_step
        last_video_x0 = video_x0
        last_audio_x0 = audio_x0

        # Euler step
        video_x = euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)

        # Force computation for memory efficiency
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()

    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)
