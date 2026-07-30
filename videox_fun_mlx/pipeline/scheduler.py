"""CogVideoX DDIM Scheduler for MLX.

Supports v-prediction, trailing timestep spacing, and zero-SNR rescaling
as used by CogVideoXDDIMScheduler in diffusers.
"""

import mlx.core as mx


class DDIMScheduler:
    """DDIM scheduler compatible with CogVideoXDDIMScheduler.

    Args:
        num_train_timesteps: Number of diffusion steps used during training.
        beta_start: Starting value of beta schedule.
        beta_end: Ending value of beta schedule.
        beta_schedule: Type of beta schedule ("scaled_linear" or "linear").
        prediction_type: "epsilon" or "v_prediction".
        rescale_betas_zero_snr: Whether to rescale betas for zero terminal SNR.
        timestep_spacing: "leading" or "trailing".
        set_alpha_to_one: Whether to set final alpha to 1.
        num_inference_steps: Default number of denoising steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
        rescale_betas_zero_snr: bool = True,
        timestep_spacing: str = "trailing",
        set_alpha_to_one: bool = True,
        clip_sample: bool = False,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.clip_sample = clip_sample

        if beta_schedule == "scaled_linear":
            betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        if rescale_betas_zero_snr:
            betas = self._rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)

        # Final alpha (for prev_t at t=0)
        self.final_alpha_cumprod = mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self._timesteps = None
        self.set_timesteps(num_inference_steps)

    @staticmethod
    def _rescale_zero_terminal_snr(betas: mx.array) -> mx.array:
        """Rescale betas so that the terminal SNR is zero (last alpha_cumprod = 0)."""
        alphas = 1.0 - betas
        alphas_cumprod = mx.cumprod(alphas)
        # sqrt(alpha_bar)
        alphas_bar_sqrt = mx.sqrt(alphas_cumprod)
        # Last element must be 0
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1]
        alphas_bar_sqrt = (
            (alphas_bar_sqrt - alphas_bar_sqrt_T) * alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        )
        alphas_cumprod = alphas_bar_sqrt**2
        # Reconstruct betas
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = mx.concatenate([alphas_cumprod[:1], alphas])
        betas = 1 - alphas
        return betas

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute the timestep schedule."""
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = mx.round(mx.arange(num_inference_steps, 0, -1) * step_ratio).astype(mx.int32) - 1
            timesteps = mx.clip(timesteps, 0, self.num_train_timesteps - 1)
        else:  # "leading"
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = mx.arange(0, num_inference_steps) * step_ratio
            timesteps = timesteps[::-1]

        self._timesteps = timesteps

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def _get_prev_timestep(self, timestep: int) -> int:
        if self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            return int(timestep - step_ratio)
        else:
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            return timestep - step_ratio

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
    ) -> mx.array:
        """Perform one DDIM denoising step.

        Supports both epsilon-prediction and v-prediction.
        """
        t = int(timestep.item()) if isinstance(timestep, mx.array) else int(timestep)
        prev_t = self._get_prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

        sqrt_alpha_t = mx.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_t = mx.sqrt(1.0 - alpha_prod_t)
        sqrt_alpha_t_prev = mx.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_t_prev = mx.sqrt(1.0 - alpha_prod_t_prev)

        if self.prediction_type == "v_prediction":
            # v = sqrt(alpha_t) * eps - sqrt(1-alpha_t) * x0
            # => x0 = sqrt(alpha_t) * sample - sqrt(1-alpha_t) * v
            # => eps = sqrt(alpha_t) * v + sqrt(1-alpha_t) * sample  (not used directly)
            pred_x0 = sqrt_alpha_t * sample - sqrt_one_minus_alpha_t * model_output
            pred_eps = sqrt_alpha_t * model_output + sqrt_one_minus_alpha_t * sample  # for direction
        else:
            # epsilon prediction
            pred_x0 = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
            pred_eps = model_output

        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -1, 1)

        # DDIM step (eta=0, deterministic)
        pred_sample = sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * pred_eps

        return pred_sample

    def add_noise(
        self,
        original: mx.array,
        noise: mx.array,
        timestep,
    ) -> mx.array:
        """Add noise to original samples at the given timestep level."""
        t = int(timestep.item()) if isinstance(timestep, mx.array) else int(timestep)
        # Mirror diffusers add_noise: alphas cast to the sample dtype BEFORE
        # the sqrt, so bf16 samples stay bf16. (step() deliberately keeps the
        # torch fp32-promotion behavior; the reference pipeline casts latents
        # back after each step — pipeline_cogvideox_fun_inpaint.py:1170.)
        alpha_prod_t = self.alphas_cumprod[t].astype(original.dtype)

        sqrt_alpha_prod = mx.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = mx.sqrt(1.0 - alpha_prod_t)

        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
