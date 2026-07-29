# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stability AI — vendored from github.com/Stability-AI/stable-audio-3
"""Waveform-to-latent encoding utilities for the standalone MLX models."""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


PATCH_SIZE = 256
ENCODER_STRIDE = 16
SAMPLES_PER_LATENT = PATCH_SIZE * ENCODER_STRIDE
SAME_ENCODER_PAD_MODULO = {
    "same-s": 32,
    "same-l": 16,
}


@dataclass(frozen=True)
class EncodedAudio:
    """Latents and validity metadata produced from a padded waveform batch."""

    latents: mx.array
    padding_mask: mx.array
    valid_latent_lengths: tuple[int, ...]
    source_samples: int
    padded_samples: int


def patch_audio(audio, *, patch_size: int = PATCH_SIZE):
    """Apply SAME's patched pretransform to a ``[B, C, T]`` waveform batch."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if audio.ndim != 3:
        raise ValueError(f"Expected audio shaped [B, C, T], got {audio.shape}.")
    batch, channels, samples = (int(value) for value in audio.shape)
    if samples % patch_size != 0:
        raise ValueError(
            f"Audio length {samples} must be divisible by patch_size={patch_size}."
        )
    patch_count = samples // patch_size
    return (
        audio.reshape(
            batch,
            channels,
            patch_count,
            patch_size,
        )
        .transpose(0, 1, 3, 2)
        .reshape(
            batch,
            channels * patch_size,
            patch_count,
        )
    )


def encode_audio(
    encoder,
    audio,
    *,
    valid_sample_lengths: tp.Sequence[int] | np.ndarray | None = None,
    pad_modulo: int = SAME_ENCODER_PAD_MODULO["same-l"],
    patch_size: int = PATCH_SIZE,
    encoder_stride: int = ENCODER_STRIDE,
    chunked: bool = False,
    chunk_size: int = 128,
    overlap: int = 32,
    chunk_batch_size: int = 1,
) -> EncodedAudio:
    """Encode a waveform batch with a SAME-S or SAME-L MLX encoder.

    The caller owns file decoding, resampling, and channel conversion. ``audio``
    must be a batch of equal-length waveforms shaped ``[B, C, T]``. The current
    SAME encoders expect 44.1 kHz stereo input.
    """

    audio = mx.array(audio)
    if audio.ndim != 3:
        raise ValueError(f"Expected audio shaped [B, C, T], got {audio.shape}.")
    batch_size, channels, source_samples = (int(value) for value in audio.shape)
    if batch_size < 1 or source_samples < 1:
        raise ValueError("Audio batches and waveforms must be non-empty.")
    if channels != 2:
        raise ValueError(f"SAME encoders expect stereo audio, got {channels} channels.")
    if patch_size <= 0 or encoder_stride <= 0:
        raise ValueError("patch_size and encoder_stride must be positive.")
    if pad_modulo <= 0 or pad_modulo % encoder_stride != 0:
        raise ValueError("pad_modulo must be a positive multiple of encoder_stride.")

    valid_lengths = _normalize_valid_lengths(
        valid_sample_lengths,
        batch_size=batch_size,
        source_samples=source_samples,
    )
    sample_alignment = patch_size * pad_modulo
    padded_samples = math.ceil(source_samples / sample_alignment) * sample_alignment
    if padded_samples != source_samples:
        audio = mx.pad(
            audio,
            ((0, 0), (0, 0), (0, padded_samples - source_samples)),
        )

    patches = patch_audio(audio, patch_size=patch_size)
    latents = _encode_patches(
        encoder,
        patches,
        encoder_stride=encoder_stride,
        pad_modulo=pad_modulo,
        chunked=chunked,
        chunk_size=chunk_size,
        overlap=overlap,
        chunk_batch_size=chunk_batch_size,
    )

    samples_per_latent = patch_size * encoder_stride
    valid_latent_lengths = tuple(
        min(
            int(latents.shape[-1]),
            math.ceil(length / samples_per_latent),
        )
        for length in valid_lengths
    )
    positions = mx.arange(int(latents.shape[-1]))[None, :]
    padding_mask = positions < mx.array(valid_latent_lengths)[:, None]
    return EncodedAudio(
        latents=latents,
        padding_mask=padding_mask,
        valid_latent_lengths=valid_latent_lengths,
        source_samples=source_samples,
        padded_samples=padded_samples,
    )


def _normalize_valid_lengths(
    valid_sample_lengths: tp.Sequence[int] | np.ndarray | None,
    *,
    batch_size: int,
    source_samples: int,
) -> tuple[int, ...]:
    if valid_sample_lengths is None:
        return (source_samples,) * batch_size
    values = tuple(int(value) for value in valid_sample_lengths)
    if len(values) != batch_size:
        raise ValueError(
            "valid_sample_lengths must contain one value per audio batch item."
        )
    if any(value < 0 or value > source_samples for value in values):
        raise ValueError(
            f"Valid sample lengths must be between 0 and {source_samples}."
        )
    return values


def _encode_patches(
    encoder,
    patches,
    *,
    encoder_stride: int,
    pad_modulo: int,
    chunked: bool,
    chunk_size: int,
    overlap: int,
    chunk_batch_size: int,
):
    total_patches = int(patches.shape[-1])
    if total_patches % encoder_stride != 0:
        raise ValueError(
            f"Patch length {total_patches} must be divisible by "
            f"encoder_stride={encoder_stride}."
        )
    if total_patches % pad_modulo != 0:
        raise ValueError(
            f"Patch length {total_patches} must be divisible by "
            f"pad_modulo={pad_modulo}."
        )

    total_latents = total_patches // encoder_stride
    if not chunked or total_latents <= chunk_size:
        return encoder(patches)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if chunk_batch_size < 1:
        raise ValueError("chunk_batch_size must be positive.")
    chunk_patches = chunk_size * encoder_stride
    if chunk_patches % pad_modulo != 0:
        raise ValueError(
            "chunk_size produces a patch length incompatible with pad_modulo."
        )

    hop_latents = chunk_size - overlap
    chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
    final_start = total_latents - chunk_size
    if chunk_starts[-1] != final_start:
        chunk_starts.append(final_start)

    batch_size = int(patches.shape[0])
    encoded_chunks = []
    for offset in range(0, len(chunk_starts), chunk_batch_size):
        batch_starts = chunk_starts[offset : offset + chunk_batch_size]
        chunk_inputs = mx.concatenate(
            [
                patches[
                    ...,
                    start * encoder_stride : (start + chunk_size) * encoder_stride,
                ]
                for start in batch_starts
            ],
            axis=0,
        )
        encoded_batch = encoder(chunk_inputs)
        mx.eval(encoded_batch)
        encoded_chunks.extend(
            encoded_batch[index * batch_size : (index + 1) * batch_size]
            for index in range(len(batch_starts))
        )

    return _stitch_encoded_chunks(
        encoded_chunks,
        chunk_starts=chunk_starts,
        total_latents=total_latents,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _stitch_encoded_chunks(
    encoded_chunks,
    *,
    chunk_starts: list[int],
    total_latents: int,
    chunk_size: int,
    overlap: int,
):
    # Floor division intentionally gives the later chunk the extra latent when
    # an overlap is odd. The interval clipping below prevents gaps or duplicates.
    half_overlap = overlap // 2
    intervals = []
    last_index = len(chunk_starts) - 1
    for index, (start, chunk) in enumerate(
        zip(chunk_starts, encoded_chunks, strict=True)
    ):
        is_first = index == 0
        is_last = index == last_index
        output_start = total_latents - chunk_size if is_last else start
        left = 0 if is_first else half_overlap
        right = chunk_size if is_last else chunk_size - half_overlap
        intervals.append((output_start + left, output_start + right, left, chunk))

    pieces = []
    cursor = 0
    output_shape = tuple(int(value) for value in encoded_chunks[0].shape[:-1])
    for index, (target_start, target_end, left, chunk) in enumerate(intervals):
        next_start = (
            intervals[index + 1][0] if index + 1 < len(intervals) else target_end
        )
        target_end = min(target_end, next_start)
        clipped_start = max(target_start, cursor)
        if clipped_start > cursor:
            pieces.append(
                mx.zeros(
                    (*output_shape, clipped_start - cursor),
                    dtype=encoded_chunks[0].dtype,
                )
            )
            cursor = clipped_start
        if target_end <= clipped_start:
            continue
        source_start = left + clipped_start - target_start
        source_end = source_start + target_end - clipped_start
        pieces.append(chunk[..., source_start:source_end])
        cursor = target_end

    if cursor < total_latents:
        pieces.append(
            mx.zeros(
                (*output_shape, total_latents - cursor),
                dtype=encoded_chunks[0].dtype,
            )
        )
    return mx.concatenate(pieces, axis=-1)
