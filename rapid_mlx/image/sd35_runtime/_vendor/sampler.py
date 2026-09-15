#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import mlx.core as mx
import mlx.nn as nn


class ModelSamplingDiscreteFlow(nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(mx.arange(1, timesteps + 1, 1))
        self.sigmas = ts

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: mx.array):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class FluxSampler(nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Flux models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(mx.arange(0, timesteps + 1, 1))
        self.sigmas = ts

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: mx.array):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image
