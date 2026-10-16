# WSOLA adapted from audiotsm: https://github.com/Muges/audiotsm
#
# The MIT License
#
# Copyright (c) 2017 Muges
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Pitch-preserving, offline WSOLA for mono TTS waveforms."""

import numpy as np
from numpy.fft import irfft, rfft


def time_stretch(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Change tempo by ``speed`` using 32 ms frames with 50% overlap.

    Input and output are mono float32 samples. Pitch is preserved with WSOLA
    for clips of at least two frames (64 ms). Shorter non-empty clips are too
    short to align, so they are linearly resampled (which shifts their pitch).
    Frame alignment follows audiotsm's correlation against the previous
    frame's natural progression; normalized Hann overlap-add joins the frames.
    """
    if audio.ndim != 1:
        raise ValueError("time_stretch requires mono audio")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_length = max(2, round(sample_rate * 0.032 / 2) * 2)
    if speed == 1.0 or not len(audio):
        return audio

    # Never drop non-empty audio (round(0.5) == 0).
    output_length = max(1, round(len(audio) / speed))
    if len(audio) < 2 * frame_length:  # too short to search for alignment
        return np.interp(
            np.linspace(0, len(audio) - 1, output_length), np.arange(len(audio)), audio
        ).astype(np.float32)

    hop = frame_length // 2
    tolerance = hop
    last_start = len(audio) - frame_length
    # Pad for the previous frame's natural progression near the clip end.
    source = np.pad(audio, (0, frame_length + hop + tolerance))
    output = np.zeros(output_length + frame_length, dtype=np.float32)
    weights = np.zeros_like(output)
    # Positive endpoints allow normalization even at the clip boundaries.
    window = np.hanning(frame_length + 2)[1:-1].astype(np.float32)
    previous = 0
    for position in range(0, output_length, hop):
        expected = min(round(position * speed), last_start)
        start = expected
        if position:
            lower = max(0, expected - tolerance)
            upper = min(last_start, expected + tolerance)
            natural = source[previous + hop : previous + hop + frame_length]
            candidates = source[lower : upper + frame_length]
            # Circular correlation agrees with linear correlation over valid
            # starts; no padding for the unused full-correlation tail is needed.
            fft_length = 1 << (len(candidates) - 1).bit_length()
            correlation = irfft(
                rfft(candidates, fft_length) * rfft(natural, fft_length).conj(),
                fft_length,
            )[: upper - lower + 1]
            start = lower + int(np.argmax(correlation))
        frame = source[start : start + frame_length]
        output[position : position + frame_length] += frame * window
        weights[position : position + frame_length] += window
        previous = start
    return output[:output_length] / weights[:output_length]
