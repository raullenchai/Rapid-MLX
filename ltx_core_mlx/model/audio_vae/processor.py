"""Audio processing — STFT and mel filterbank.

Matches reference: torchaudio.transforms.MelSpectrogram with
mel_scale="slaney", norm="slaney", center=True, pad_mode="reflect", power=1.0.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class AudioProcessor:
    """STFT-based audio processor with mel filterbank.

    Converts raw waveforms to mel spectrograms matching the reference
    torchaudio MelSpectrogram configuration.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT window size (reference default: 1024).
        hop_length: Hop length for STFT.
        n_mels: Number of mel bands.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank (default: sample_rate / 2).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        if f_max is None:
            f_max = sample_rate / 2.0

        # Build Slaney mel filterbank with area normalization
        self.mel_basis = mx.array(self._build_mel_filterbank_slaney(sample_rate, n_fft, n_mels, f_min, f_max))

        # STFT window (Hann, matching torchaudio default)
        self.window = mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32))

    @staticmethod
    def _build_mel_filterbank_slaney(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
        """Build a Slaney-scale mel filterbank with area normalization.

        Matches torchaudio.transforms.MelSpectrogram(mel_scale="slaney", norm="slaney").

        Returns:
            Array of shape (n_mels, n_fft // 2 + 1).
        """

        # Slaney mel scale: linear below 1000 Hz, logarithmic above
        def hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
            f = np.asarray(f, dtype=np.float64)
            # np.where evaluates both branches; clamp to avoid log(0) on the DC bin
            f_safe = np.maximum(f, 1e-10)
            result = np.where(
                f < 1000.0,
                3.0 * f / 200.0,  # Linear region
                15.0 + 27.0 * np.log(f_safe / 1000.0) / np.log(6.4),  # Log region
            )
            return result

        def mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
            m = np.asarray(m, dtype=np.float64)
            result = np.where(
                m < 15.0,
                200.0 * m / 3.0,  # Linear region
                1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0),  # Log region
            )
            return result

        n_freqs = n_fft // 2 + 1
        mel_min = hz_to_mel_slaney(np.array(f_min))
        mel_max = hz_to_mel_slaney(np.array(f_max))
        mel_points = np.linspace(float(mel_min), float(mel_max), n_mels + 2)
        hz_points = mel_to_hz_slaney(mel_points)

        # Frequency bins
        fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)
        for i in range(n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]

            # Rising slope
            if center > lower:
                rising = (fft_freqs - lower) / (center - lower)
                filterbank[i] += np.maximum(0, rising) * (fft_freqs <= center)

            # Falling slope
            if upper > center:
                falling = (upper - fft_freqs) / (upper - center)
                filterbank[i] += np.maximum(0, falling) * (fft_freqs > center)

            # Slaney area normalization: normalize each filter by its bandwidth
            enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
            filterbank[i] *= enorm

        return filterbank.astype(np.float32)

    def waveform_to_mel(self, waveform: mx.array) -> mx.array:
        """Convert waveform to mel spectrogram.

        Matches reference: center=True with reflect padding, power=1.0 (magnitude).

        Args:
            waveform: (B, T) or (B, C, T) waveform.

        Returns:
            Mel spectrogram of shape (B, C, T', n_mels) or (B, T', n_mels).
        """
        squeeze_channel = False
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]
            squeeze_channel = True

        B, C, T = waveform.shape
        mels = []

        # Center padding with reflect (matches torchaudio center=True, pad_mode="reflect")
        pad_amount = self.n_fft // 2

        for b in range(B):
            channel_mels = []
            for c in range(C):
                signal = waveform[b, c]

                # Reflect padding for center=True
                left_pad = signal[1 : pad_amount + 1][::-1]
                right_pad = signal[-(pad_amount + 1) : -1][::-1]
                padded = mx.concatenate([left_pad, signal, right_pad])

                # Frame the signal
                T_padded = padded.shape[0]
                num_frames = (T_padded - self.n_fft) // self.hop_length + 1
                indices = mx.arange(self.n_fft)[None, :] + mx.arange(num_frames)[:, None] * self.hop_length
                frames = padded[indices] * self.window

                # FFT -> magnitude (power=1.0)
                spec = mx.fft.rfft(frames)
                mag = mx.abs(spec)

                # Mel filterbank
                mel = mag @ self.mel_basis.T
                mel = mx.log(mx.maximum(mel, 1e-5))
                channel_mels.append(mel)

            mels.append(mx.stack(channel_mels, axis=0))

        result = mx.stack(mels, axis=0)
        if squeeze_channel:
            result = result[:, 0]
        return result
