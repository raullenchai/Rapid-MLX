"""Small diagonal Gaussian used by the SDXL VAE encoder."""

import mlx.core as mx


class DiagonalGaussian:
    """A diagonal Gaussian parameterized by concatenated ``[mean, logvar]``."""

    def __init__(self, parameters: mx.array):
        self.mean, logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self, key: mx.array | None = None) -> mx.array:
        key = key if key is not None else mx.random.key(0)
        return self.mean + self.std * mx.random.normal(self.mean.shape, key=key)

    def mode(self) -> mx.array:
        return self.mean

    def kl(self) -> mx.array:
        return 0.5 * mx.sum(self.mean**2 + mx.exp(self.logvar) - 1.0 - self.logvar)
