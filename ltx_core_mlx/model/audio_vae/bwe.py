"""Bandwidth Extension (BWE) — 16kHz to 48kHz.

Ported from ltx-core BWE module. Combines:
1. Hann-windowed sinc 3x resampler
2. MelSTFT for computing mel spectrogram of the resampled signal
3. BWE generator (separate BigVGAN) for high-frequency residual

Weight key structure (under vocoder.* prefix, alongside base vocoder):
    bwe_generator.*         — BigVGAN with ratios [6, 5, 2, 2, 2]
    mel_stft.mel_basis      — (64, 257)
    mel_stft.stft_fn.forward_basis  — (514, 1, 512)
    mel_stft.stft_fn.inverse_basis  — (514, 1, 512)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ltx_core_mlx.model.audio_vae.vocoder import BigVGANVocoder

# ---------------------------------------------------------------------------
# Hann-sinc resampler (no learned weights)
# ---------------------------------------------------------------------------


class HannSincResampler:
    """3x upsampler using Hann-windowed sinc interpolation.

    Matches reference UpSample1d(ratio=3, window_type="hann").
    Not an nn.Module — no learnable parameters.

    Reference implementation:
        1. Replicate-pads input by ``width`` on each side
        2. Applies conv_transpose1d(stride=ratio) with the sinc kernel
        3. Scales by ratio and slices [pad_left:-pad_right]

    MLX equivalent:
        1. Replicate-pads input by ``width``
        2. Zero-inserts (stride) between samples
        3. Applies forward conv1d with the sinc kernel
        4. Scales by ratio and slices to match reference output
    """

    def __init__(self, upsample_factor: int = 3):
        self.upsample_factor = upsample_factor
        self._rolloff = 0.99
        self._lowpass_filter_width = 6
        self._width = int(np.ceil(self._lowpass_filter_width / self._rolloff))  # 7
        kernel = self._build_kernel(upsample_factor)
        self.kernel = mx.array(kernel[:, None])  # (K, 1) for conv1d
        # Padding/slicing params matching reference UpSample1d (Hann path)
        self._pad = self._width  # replicate-pad on input: 7
        self._kernel_size = 2 * self._width * upsample_factor + 1  # 43
        self._pad_left = 2 * self._width * upsample_factor  # 42
        self._pad_right = self._kernel_size - upsample_factor  # 40

    def _build_kernel(self, ratio: int) -> np.ndarray:
        """Build Hann-windowed sinc filter matching reference exactly.

        Reference formula (UpSample1d, window_type="hann"):
            time_axis = (arange(kernel_size) / ratio - width) * rolloff
            time_clamped = clip(time_axis, -lpfw, lpfw)
            window = cos(time_clamped * pi / lpfw / 2) ** 2
            kernel = sinc(time_axis) * window * rolloff / ratio
        """
        kernel_size = 2 * self._width * ratio + 1  # 43
        idx = np.arange(kernel_size, dtype=np.float64)
        time_axis = (idx / ratio - self._width) * self._rolloff
        time_clamped = np.clip(time_axis, -self._lowpass_filter_width, self._lowpass_filter_width)
        window = np.cos(time_clamped * np.pi / self._lowpass_filter_width / 2) ** 2
        kernel = (np.sinc(time_axis) * window * self._rolloff / ratio).astype(np.float32)
        return kernel

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample: (B, T) -> (B, T * factor).

        Matches reference: replicate-pad -> conv_transpose1d -> scale -> slice.
        Implemented as: replicate-pad -> zero-insert -> full conv1d -> scale -> slice.
        """
        B, T = x.shape
        ratio = self.upsample_factor

        # 1. Replicate-pad input (matches F.pad(x, (pad, pad), mode='replicate'))
        first = mx.repeat(x[:, :1], self._pad, axis=1)  # (B, pad)
        last = mx.repeat(x[:, -1:], self._pad, axis=1)  # (B, pad)
        x_padded = mx.concatenate([first, x, last], axis=1)  # (B, T + 2*pad)
        T_padded = x_padded.shape[1]

        # 2. Zero-insert between samples (conv_transpose1d style):
        #    output length = (T_padded - 1) * ratio + 1
        zi_len = (T_padded - 1) * ratio + 1
        upsampled = mx.zeros((B, zi_len))
        # .at[<strided>].add() mis-indexes the source on Metal in mlx 0.31.2
        # (ml-explore/mlx#3477, fixed upstream by #3483 but unreleased). The
        # destination is freshly zeroed, so assignment is equivalent to add and
        # is correct on all backends/versions. See issue #34.
        upsampled[:, ::ratio] = x_padded

        # 3. Full convolution via zero-pad + valid conv1d
        #    Full conv output = zi_len + K - 1
        upsampled = upsampled[:, :, None]  # (B, zi_len, 1)
        K = self.kernel.shape[0]
        upsampled = mx.pad(upsampled, [(0, 0), (K - 1, K - 1), (0, 0)])
        filt = self.kernel[None, :, :]  # (1, K, 1)
        result = mx.conv1d(upsampled, filt, padding=0)
        result = result.squeeze(-1)  # (B, zi_len + K - 1)

        # 4. Scale by ratio (matching reference: self.ratio * conv_transpose1d(...))
        result = result * ratio

        # 5. Slice to match reference output: [pad_left:-pad_right]
        result = result[:, self._pad_left : -self._pad_right]

        return result[:, : T * ratio]


# ---------------------------------------------------------------------------
# STFT function (loads basis from weights)
# ---------------------------------------------------------------------------


class STFTFunction(nn.Module):
    """STFT using pre-computed basis matrices.

    Weight keys (MLX Conv1d format, pre-transposed by mlx-forge):
        forward_basis  — (n_fft+2, n_fft, 1)  i.e. (O, K, I)
        inverse_basis  — (n_fft+2, n_fft, 1)
    """

    def __init__(self, n_fft: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.forward_basis = mx.zeros((n_fft + 2, n_fft, 1))
        self.inverse_basis = mx.zeros((n_fft + 2, n_fft, 1))


# ---------------------------------------------------------------------------
# MelSTFT (loads mel_basis and stft_fn from weights)
# ---------------------------------------------------------------------------


class MelSTFT(nn.Module):
    """Mel spectrogram transform for BWE input.

    Weight keys:
        mel_basis                   — (64, 257)
        stft_fn.forward_basis       — (514, 1, 512)
        stft_fn.inverse_basis       — (514, 1, 512)

    Note: hop_length=80 matches the BWE config (not 160 from the audio VAE
    preprocessing). This ensures BWE generator output length matches the
    3x-resampled skip connection: mel_frames * 240 == vocoder_output * 3.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 80,
        n_mels: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_basis = mx.zeros((n_mels, n_fft // 2 + 1))
        self.stft_fn = STFTFunction(n_fft)

    def __call__(self, waveform: mx.array) -> mx.array:
        """Compute mel spectrogram.

        Args:
            waveform: (B, T) waveform.

        Returns:
            (B, T_frames, n_mels) mel spectrogram.
        """
        B, T = waveform.shape

        # Use the loaded forward_basis for STFT
        # forward_basis: (n_fft+2, 1, n_fft) — real and imag interleaved
        # Reshape waveform for conv: (B, T, 1)
        x = waveform[:, :, None]

        # Causal padding: left-only, matching reference _STFTFn
        left_pad = max(0, self.n_fft - self.hop_length)  # 512 - 80 = 432
        x = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])

        # Apply STFT basis via conv1d
        # forward_basis in MLX Conv1d format: (O, K, I) = (n_fft+2, n_fft, 1)
        # Pre-transposed by mlx-forge
        basis = self.stft_fn.forward_basis  # (514, 512, 1)

        stft_out = mx.conv1d(x, basis, stride=self.hop_length)  # (B, T', 514)

        # Split real and imaginary parts
        n_fft_bins = self.n_fft // 2 + 1  # 257
        real = stft_out[:, :, :n_fft_bins]
        imag = stft_out[:, :, n_fft_bins:]

        # Magnitude
        mag = mx.sqrt(real * real + imag * imag + 1e-9)  # (B, T', 257)

        # Apply mel filterbank
        mel = mag @ self.mel_basis.T  # (B, T', n_mels)

        # Log mel
        return mx.log(mx.maximum(mel, 1e-5))


# ---------------------------------------------------------------------------
# Full vocoder + BWE pipeline
# ---------------------------------------------------------------------------


class VocoderWithBWE(nn.Module):
    """Full audio pipeline: mel -> 16kHz waveform -> 48kHz waveform.

    Combines base vocoder, 3x Kaiser-sinc resampler, and BWE generator.
    BWE output = clamp(resampled_base + bwe_residual, -1, 1).

    The forward pass must run in fp32: bfloat16 accumulation errors compound
    through 108 sequential convolutions (BigVGAN v2) and degrade spectral
    metrics (mel_l1, MRSTFT) by 40-90%. Call ``upcast_weights_to_fp32()`` once
    after ``load_weights()`` to promote all parameters to fp32.

    Weight key hierarchy (all under vocoder.* prefix):
        conv_pre, ups, resblocks, act_post, conv_post — base vocoder
        bwe_generator.*                                — BWE BigVGAN
        mel_stft.*                                     — MelSTFT for BWE
    """

    def __init__(self):
        super().__init__()
        # Base vocoder: mel -> 16kHz (stereo, 2-ch output)
        # This is a PEER module — base vocoder keys are at the same level,
        # NOT nested under a "vocoder" attribute. The base vocoder keys
        # (conv_pre, ups, resblocks, etc.) live directly in this module.

        # We need the base vocoder conv_pre, ups, resblocks, act_post, conv_post
        # to load as direct attributes of this class (not nested).
        # Use composition and expose the relevant attributes.

        # NOTE: The vocoder.safetensors has the base vocoder keys at top level
        # (after stripping "vocoder." prefix). So we DON'T wrap in a sub-module.
        # Instead we construct the BigVGAN components directly.

        # Import here to build the base vocoder inline
        from ltx_core_mlx.model.audio_vae.vocoder import (
            Activation1d,
            AMPBlock1,
        )

        # --- Base vocoder (keys at top level) ---
        in_channels = 128
        upsample_initial_channel = 1536
        upsample_rates = (5, 2, 2, 2, 2, 2)
        upsample_kernel_sizes = (11, 4, 4, 4, 4, 4)
        resblock_kernel_sizes = (3, 7, 11)
        resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)

        self.ups = []
        self.resblocks = []
        channels = upsample_initial_channel

        for _i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    out_ch,
                    kernel_size=kernel,
                    stride=rate,
                    padding=padding,
                )
            )
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(out_ch, rk, rd))
            channels = out_ch

        self.act_post = Activation1d(channels)
        self.conv_post = nn.Conv1d(channels, 2, kernel_size=7, padding=3, bias=False)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # --- BWE generator ---
        self.bwe_generator = BigVGANVocoder(
            in_channels=128,
            upsample_initial_channel=512,
            upsample_rates=(6, 5, 2, 2, 2),
            upsample_kernel_sizes=(12, 11, 4, 4, 4),
            out_channels=2,
            apply_final_activation=False,
        )

        # --- Mel STFT for BWE ---
        self.mel_stft = MelSTFT()

        # --- Resampler (no weights) ---
        self._resampler = HannSincResampler(upsample_factor=3)

    def upcast_weights_to_fp32(self) -> None:
        """Promote all parameters to fp32 in-place.

        Required after ``load_weights()`` — running the forward in bf16/fp16
        compounds accumulation errors through 108 sequential convolutions and
        degrades spectral metrics (mel_l1, MRSTFT) by 40-90%. MLX has no
        autocast; the weights themselves must be fp32 for the conv kernels to
        run in fp32.
        """
        from mlx.utils import tree_map

        self.update(tree_map(lambda p: p.astype(mx.float32), self.parameters()))

    def _run_base_vocoder(self, mel: mx.array) -> mx.array:
        """Run base vocoder: mel (B, T, 64) -> waveform (B, T_audio, 2)."""
        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
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
        x = mx.tanh(x)
        return x

    def __call__(self, mel: mx.array) -> mx.array:
        """Full pipeline: mel -> 48kHz stereo waveform.

        Reference: ltx-core VocoderWithBWE.forward

        The stereo mel channels are CONCATENATED (not processed separately):
        (B, 2, T, 64) → rearrange to (B, T, 128) → vocoder → (B, T_audio, 2).

        Args:
            mel: (B, 2, T, n_mels) stereo mel spectrogram.

        Returns:
            (B, 2, T_audio) waveform at 48kHz.
        """
        input_dtype = mel.dtype
        mel = mel.astype(mx.float32)

        B, C, T, M = mel.shape  # (B, 2, T, 64)

        # 1. Run base vocoder: concatenate stereo channels for input
        # (B, 2, T, 64) → transpose mel_bins to front → (B, 2, 64, T) → rearrange → (B, 128, T) → (B, T, 128)
        mel_concat = mel.transpose(0, 1, 3, 2)  # (B, 2, 64, T)
        mel_concat = mel_concat.reshape(B, C * M, T)  # (B, 128, T)
        mel_concat = mel_concat.transpose(0, 2, 1)  # (B, T, 128)

        waveform_16k = self._run_base_vocoder(mel_concat)  # (B, T_audio_16k, 2)
        waveform_16k = waveform_16k.transpose(0, 2, 1)  # (B, 2, T_audio_16k)

        # Compute output length before padding
        length_16k = waveform_16k.shape[-1]
        output_length = length_16k * 3  # 3x upsample (16kHz → 48kHz)

        # Pad to multiple of hop_length for exact mel frame count
        hop = self.mel_stft.hop_length
        remainder = length_16k % hop
        if remainder != 0:
            pad_amount = hop - remainder
            waveform_16k = mx.pad(waveform_16k, [(0, 0), (0, 0), (0, pad_amount)])

        # 2. Compute mel of vocoder output: (B, 2, T) → (B*2, T) → mel → (B, 2, n_mels, T')
        flat_wav = waveform_16k.reshape(B * C, -1)  # (B*2, T)
        bwe_mel = self.mel_stft(flat_wav)  # (B*2, T', n_mels)
        T_frames = bwe_mel.shape[1]
        bwe_mel = bwe_mel.reshape(B, C, T_frames, M)  # (B, 2, T', 64)

        # 3. Run BWE generator on mel: (B, 2, T', 64) → same rearrange as base vocoder
        bwe_mel_concat = bwe_mel.transpose(0, 1, 3, 2)  # (B, 2, 64, T')
        bwe_mel_concat = bwe_mel_concat.reshape(B, C * M, T_frames)  # (B, 128, T')
        bwe_mel_concat = bwe_mel_concat.transpose(0, 2, 1)  # (B, T', 128)

        residual = self.bwe_generator(bwe_mel_concat)  # (B, T_bwe, 2)
        residual = residual.transpose(0, 2, 1)  # (B, 2, T_bwe)

        # 4. Resample base vocoder output to 48kHz
        skip_channels = []
        for c in range(C):
            resampled = self._resampler(waveform_16k[:, c, :])  # (B, T_48k)
            skip_channels.append(resampled)
        skip = mx.stack(skip_channels, axis=1)  # (B, 2, T_48k)

        # 5. Add residual and clip
        min_len = min(skip.shape[-1], residual.shape[-1])
        output = skip[:, :, :min_len] + residual[:, :, :min_len]
        output = mx.clip(output, -1.0, 1.0)[:, :, :output_length]
        return output.astype(input_dtype)
