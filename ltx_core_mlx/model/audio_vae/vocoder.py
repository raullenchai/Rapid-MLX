"""BigVGAN v2 vocoder — mel spectrogram to waveform.

Ported from ltx-core/src/ltx_core/model/audio_vae/vocoder.py

Weight key structure (after stripping "vocoder." prefix):

Base vocoder:
    conv_pre.{weight,bias}                             — Conv1d(128→1536, k=7)
    ups.{0..5}.{weight,bias}                           — ConvTranspose1d
    resblocks.{0..17}.convs{1,2}.{0,1,2}.{weight,bias} — Conv1d
    resblocks.{0..17}.acts{1,2}.{0,1,2}.act.{alpha,beta}  — SnakeBeta
    resblocks.{0..17}.acts{1,2}.{0,1,2}.{upsample.filter,downsample.lowpass.filter}
    act_post.act.{alpha,beta}                          — SnakeBeta
    act_post.{upsample.filter,downsample.lowpass.filter}
    conv_post.weight                                   — Conv1d(24→2, k=7), no bias

BWE generator (under bwe_generator.*):
    Same structure with different channel sizes.

mel_stft:
    mel_stft.mel_basis                                 — (64, 257)
    mel_stft.stft_fn.forward_basis                     — (514, 1, 512)
    mel_stft.stft_fn.inverse_basis                     — (514, 1, 512)
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# SnakeBeta activation
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/b) * sin^2(a * x).

    Weights are stored in LOG-SCALE. Forward applies exp() to get actual values.
    Weight keys: ``act.alpha``, ``act.beta``.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initialized to zeros — exp(0) = 1.0, matching reference default
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C)"""
        alpha = mx.exp(self.alpha).reshape(1, 1, -1)
        beta = mx.exp(self.beta).reshape(1, 1, -1)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


# ---------------------------------------------------------------------------
# Anti-aliased Activation1d (wraps SnakeBeta)
# ---------------------------------------------------------------------------


class LowPassKernel(nn.Module):
    """Holds the low-pass filter kernel.

    Weight key: ``filter`` — shape (1, K, 1) in MLX Conv1d format (O, K, I).
    Pre-transposed by mlx-forge from PyTorch (1, 1, K).
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.filter = mx.ones((1, kernel_size, 1))


class DownSample1d(nn.Module):
    """Anti-aliased 2x downsampler with low-pass filter.

    Weight keys:
        lowpass.filter — (1, 1, K)
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.lowpass = LowPassKernel(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) -> (B, T//2, C)"""
        B, T, C = x.shape
        # Reshape for grouped conv1d: (B*C, T, 1)
        x = x.transpose(0, 2, 1).reshape(B * C, T, 1)

        # Replicate pad — matches reference LowPassFilter1d(padding_mode='replicate')
        K = self.lowpass.filter.shape[1]
        even = 1 if K % 2 == 0 else 0
        pad_left = K // 2 - even
        pad_right = K // 2
        left_edge = mx.repeat(x[:, :1, :], pad_left, axis=1)
        right_edge = mx.repeat(x[:, -1:, :], pad_right, axis=1)
        x = mx.concatenate([left_edge, x, right_edge], axis=1)

        # Apply filter — already in MLX (O=1, K, I=1) format
        x = mx.conv1d(x, self.lowpass.filter, stride=2)

        # Reshape back: (B, C, T') -> (B, T', C)
        T_out = x.shape[1]
        return x.reshape(B, C, T_out).transpose(0, 2, 1)


class UpSample1d(nn.Module):
    """2x upsample with anti-aliasing filter.

    Weight key: ``filter`` — shape (1, K, 1) in MLX Conv1d format.
    Pre-transposed by mlx-forge from PyTorch (1, 1, K).
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.filter = mx.ones((1, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) -> (B, T*2, C)"""
        B, T, C = x.shape
        # Insert zeros between samples: (B, T, C) -> (B, T*2, C)
        x_up = mx.zeros((B, T * 2, C))
        # .at[<strided>].add() mis-indexes the source on Metal in mlx 0.31.2
        # (ml-explore/mlx#3477, fixed upstream by #3483 but unreleased). The
        # destination is freshly zeroed, so assignment is equivalent to add and
        # is correct on all backends/versions. See issue #34.
        x_up[:, ::2, :] = x

        # Reshape for grouped conv1d: (B*C, T*2, 1)
        x_up = x_up.transpose(0, 2, 1).reshape(B * C, T * 2, 1)

        K = self.filter.shape[1]
        pad = K // 2
        left_edge = mx.repeat(x_up[:, :1, :], pad, axis=1)
        right_edge = mx.repeat(x_up[:, -1:, :], pad - 1, axis=1)
        x_up = mx.concatenate([left_edge, x_up, right_edge], axis=1)

        # Filter already in MLX (O=1, K, I=1) format
        x_up = mx.conv1d(x_up, self.filter)

        T_out = x_up.shape[1]
        return x_up.reshape(B, C, T_out).transpose(0, 2, 1) * 2.0


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> activation -> downsample.

    Weight keys:
        act.alpha, act.beta         — SnakeBeta params
        upsample.filter             — (1, K, 1) MLX Conv1d format
        downsample.lowpass.filter   — (1, K, 1) MLX Conv1d format
    """

    def __init__(self, channels: int, up_ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(kernel_size)
        self.downsample = DownSample1d(kernel_size)
        self.up_ratio = up_ratio

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C)"""
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# AMPBlock1 (anti-aliased multi-periodicity residual block)
# ---------------------------------------------------------------------------


class AMPBlock1(nn.Module):
    """Anti-aliased multi-periodicity block with Activation1d.

    Weight keys:
        convs1.{0,1,2}.{weight,bias}
        convs2.{0,1,2}.{weight,bias}
        acts1.{0,1,2}.act.{alpha,beta}
        acts1.{0,1,2}.upsample.filter
        acts1.{0,1,2}.downsample.lowpass.filter
        acts2.{0,1,2}.act.{alpha,beta}
        acts2.{0,1,2}.upsample.filter
        acts2.{0,1,2}.downsample.lowpass.filter
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.acts1 = []
        self.acts2 = []

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.acts1.append(Activation1d(channels))
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                )
            )
            self.acts2.append(Activation1d(channels))
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            residual = x
            x = act1(x)
            x = conv1(x)
            x = act2(x)
            x = conv2(x)
            x = x + residual
        return x


# ---------------------------------------------------------------------------
# BigVGAN Vocoder
# ---------------------------------------------------------------------------


class BigVGANVocoder(nn.Module):
    """BigVGAN v2 vocoder: mel spectrogram -> waveform.

    Base vocoder: 128-mel -> 16kHz stereo (2-ch output)
        upsample_rates = [5, 2, 2, 2, 2, 2] -> 160x
        upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        channels: 1536 -> 768 -> 384 -> 192 -> 96 -> 48 -> 24

    BWE generator: 128-mel -> 48kHz stereo
        upsample_rates = [6, 5, 2, 2, 2] -> 240x
        upsample_kernel_sizes = [12, 11, 4, 4, 4]  (weight shapes show actual kernel)
        channels: 512 -> 256 -> 128 -> 64 -> 32 -> 16
    """

    def __init__(
        self,
        in_channels: int = 128,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple[int, ...] = (5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (11, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        out_channels: int = 2,
        apply_final_activation: bool = True,
    ):
        super().__init__()
        self._apply_final_activation = apply_final_activation

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)

        # Upsample layers (directly indexed, no interleaved activations)
        self.ups = []
        channels = upsample_initial_channel

        # Flat resblocks list (3 per upsample stage)
        self.resblocks = []

        for _i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(nn.ConvTranspose1d(channels, out_ch, kernel_size=kernel, stride=rate, padding=padding))

            # 3 resblocks per stage (one per kernel size)
            for _j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(out_ch, rk, rd))

            channels = out_ch

        self.act_post = Activation1d(channels)
        self.conv_post = nn.Conv1d(channels, out_channels, kernel_size=7, padding=3, bias=False)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

    def __call__(self, mel: mx.array) -> mx.array:
        """Convert mel spectrogram to waveform.

        Args:
            mel: (B, T, n_mels) mel spectrogram in MLX layout,
                 or (B, C, T, n_mels) for stereo processing.

        Returns:
            Waveform (B, T_audio) or (B, C, T_audio).
        """
        process_channels = False
        if mel.ndim == 4:
            B, C, T, M = mel.shape
            mel = mel.reshape(B * C, T, M)
            process_channels = True

        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            # Activation1d before upsample conv
            x = self.ups[i](x)

            # Average resblocks for this stage
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)  # (B, T_audio, 2)
        if self._apply_final_activation:
            x = mx.tanh(x)

        if process_channels:
            # x is (B*C, T_audio, 2) — but for stereo mel input, output is already 2-ch
            x = x.reshape(B, C, x.shape[1], x.shape[2])

        return x

    @property
    def hop_length(self) -> int:
        """Total upsample ratio = product of all upsample rates."""
        return math.prod([5, 2, 2, 2, 2, 2])  # 160
