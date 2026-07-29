"""Mel-spectrogram renderer — numpy-only port of the TensorRT release's spec.py.

Same algorithm and constants as optimized/tensorRT/scripts/spec.py (the 3-band
tinted stereo mel spectrogram from our run dashboards), with the torch pieces
(STFT, resample, filterbank matmul) rewritten in numpy so the MLX release stays
torch-free. Numerics match: periodic Hann window, center=True reflect padding,
Slaney mel scale, librosa-compatible filterbank, linear resample with
align_corners=False sampling positions.

Exposes render_spectrogram_png(pcm_int16, sample_rate, width, height).
"""
from __future__ import annotations
import io
import math
from functools import lru_cache

import numpy as np
from PIL import Image


# 3-band tint
SPEC_BANDS = [
    (0, 200, (1.0, 0.0, 0.0)),       # Bass -> Red
    (200, 1500, (0.0, 1.0, 0.0)),    # Mid  -> Green
    (1500, 16000, (0.0, 0.0, 1.0)),  # High -> Blue
]
_BAND_COLORS = np.array([c for _, _, c in SPEC_BANDS], dtype=np.float32)

# Slaney mel-scale constants (linear below 1 kHz, log above), matches librosa default.
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    log_term = np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz >= _MIN_LOG_HZ, _MIN_LOG_MEL + log_term, hz / _F_SP)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= _MIN_LOG_MEL,
                    _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels - _MIN_LOG_MEL)),
                    _F_SP * mels)


def _mel_frequencies(n_mels, fmax, fmin=0.0):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))


@lru_cache(maxsize=8)
def _mel_filterbank(n_mels, n_fft, sr, fmax):
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filt = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_f - lo) / max(ce - lo, 1e-10)
        right = (hi - fft_f) / max(hi - ce, 1e-10)
        filt[i] = np.maximum(0, np.minimum(left, right))
    enorm = (2.0 / (pts[2:n_mels + 2] - pts[0:n_mels])).astype(np.float32)
    filt *= enorm[:, None]
    return filt


def _stft_power(y, n_fft, hop):
    """|STFT|^2 with periodic Hann + center=True reflect pad (torch.stft-compatible).
    Returns (n_fft//2+1, n_frames) float32."""
    y = np.asarray(y, dtype=np.float32)
    pad = n_fft // 2
    y = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    # periodic Hann (torch.hann_window default), not numpy's symmetric hanning
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    frames = y[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).T.astype(np.float32)


def _melspectrogram(y_ch, sr, n_mels=30, fmax=16000, hop_length=2048, n_fft=2048):
    power = _stft_power(y_ch, n_fft, hop_length)
    return _mel_filterbank(n_mels, n_fft, sr, fmax) @ power


def _power_to_db(S, top_db=80.0):
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-10))
    return np.maximum(log_spec - log_spec.max(), -top_db)


def _mel_channel(y_ch, sr, n_mels=30):
    """dB-scaled mel + band-tinted RGB for one mono channel."""
    S = _melspectrogram(y_ch, sr, n_mels=n_mels, fmax=16000, hop_length=2048)

    S_db = _power_to_db(S)
    np.clip(S_db, -60, 0, out=S_db)
    S_db += 60.0
    S_db /= 60.0
    np.power(S_db, 0.6, out=S_db)

    mel_f = _mel_frequencies(n_mels, fmax=16000)
    n_frames = S.shape[1]
    band_norms = np.empty((3, n_frames), dtype=np.float32)
    for i, (flo, fhi, _) in enumerate(SPEC_BANDS):
        mask = (mel_f >= flo) & (mel_f < fhi)
        if mask.any():
            power = np.sum(S[mask], axis=0)
            db = 10.0 * np.log10(power + 1e-10)
            np.clip(db, -20, None, out=db)
            db -= -20
            mx = db.max()
            if mx > 0:
                db /= mx
            band_norms[i] = db
        else:
            band_norms[i] = 0.0

    rgb = band_norms.T @ _BAND_COLORS
    for c in range(3):
        mx = rgb[:, c].max()
        if mx > 0:
            rgb[:, c] /= mx

    return S_db, rgb


def _resample_to_32k(pcm_int16, sr_in):
    """Resample (T, 2) int16 PCM to (2, T') float32 at 32 kHz.
    Linear interpolation at align_corners=False sample positions (matches the
    torch.nn.functional.interpolate call in the TRT original)."""
    target_sr = 32000
    y = pcm_int16.astype(np.float32).T / 32768.0      # (2, T)
    if sr_in == target_sr:
        return y, target_sr
    in_len = y.shape[-1]
    new_len = int(round(in_len * target_sr / sr_in))
    scale = in_len / new_len
    pos = np.clip((np.arange(new_len, dtype=np.float64) + 0.5) * scale - 0.5, 0, in_len - 1)
    out = np.stack([np.interp(pos, np.arange(in_len), ch) for ch in y]).astype(np.float32)
    return out, target_sr


def render_spectrogram_png(pcm_int16, sample_rate=44100, width=1200, height=240,
                           n_mels=30) -> bytes:
    """Render the 3-band tinted stereo mel spectrogram to PNG bytes.

    Args:
        pcm_int16: (T, 2) int16 numpy array — stereo PCM.
        sample_rate: input PCM sample rate (default 44100).
        width, height: final image size.
        n_mels: mel filterbank size (30 matches the dashboards).

    Returns: PNG bytes ready to base64 / save / etc.
    """
    if pcm_int16.ndim != 2 or pcm_int16.shape[1] != 2:
        raise ValueError(f"expected (T, 2) PCM, got {pcm_int16.shape}")
    y, sr = _resample_to_32k(pcm_int16, sample_rate)
    S_L, rgb_L = _mel_channel(y[0], sr, n_mels=n_mels)
    S_R, rgb_R = _mel_channel(y[1], sr, n_mels=n_mels)

    nf = min(S_L.shape[1], S_R.shape[1])
    S_L, S_R = S_L[:, :nf], S_R[:, :nf]
    rgb_L, rgb_R = rgb_L[:nf], rgb_R[:nf]
    nm = S_L.shape[0]

    # L channel: flip vertically so bass-band is at bottom.
    S_L = S_L[::-1]

    img = np.empty((nm * 2, nf, 3), dtype=np.float32)
    img[:nm] = S_L[:, :, np.newaxis] * rgb_L[np.newaxis, :, :]
    img[nm:] = S_R[:, :, np.newaxis] * rgb_R[np.newaxis, :, :]

    np.clip(img, 0, 1, out=img)
    img *= 255
    pil = Image.fromarray(img.astype(np.uint8)).resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
