# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex FL2VA adapter and fixed six-step audio/video sampler."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from . import minimax_h3
from .minimax_h3 import MiniMaxH3BackendError, MiniMaxH3VideoEngine, merge_lora

REPO = "Pika-Labs/Pika-H3-Flex"
REVISION = "3ff166447d5edb1406107ab8370c42000ce173e4"
ADAPTER_FILE = "fl2va/adapter.safetensors"
FILES = ("LICENSE", "NOTICE", "README.md", "fl2va/manifest.json", ADAPTER_FILE)
ADAPTER_SHA256 = "8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327"
ADAPTER_RANK = ADAPTER_ALPHA = 128
TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
KNOTS = (1000, 750, 500, 250, 100, 40, 0)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
# The manifest's base 42ed227ee7df40d41602854ae760620d6eb651fe has
# byte-identical transformer, text_encoder, vae and audio_vae LFS payloads.
BASE_REPO, BASE_REVISION, BASE_FILES = (
    minimax_h3.REPO,
    minimax_h3.REVISION,
    minimax_h3.FILES,
)
PAYLOADS = ((REPO, REVISION, FILES), (BASE_REPO, BASE_REVISION, BASE_FILES))


def is_pika_h3_flex_model(name: str | None) -> bool:
    return bool(
        name and name.rstrip("/").rsplit("/", 1)[-1].casefold() == "pika-h3-flex"
    )


def shifted_sigmas(knots, shift: float) -> list[float]:
    if (
        len(knots) < 2
        or knots[0] != 1000
        or knots[-1] != 0
        or any(not math.isfinite(k) for k in knots)
        or any(a <= b for a, b in zip(knots, knots[1:]))
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError(
            "knots must decrease strictly from 1000 to 0; shift must be positive"
        )
    # Python float evaluates the schedule in float64 before rounding to float32.
    return [
        struct.unpack(
            "f", struct.pack("f", shift * (k / 1000) / (1 + (shift - 1) * (k / 1000)))
        )[0]
        for k in knots
    ]


def align_frames(seconds: int) -> int:
    # At 15 s the upstream grid gives 362, beyond the runtime's 15.0 s limit.
    return min(345, ((seconds * 24 - 5 + 16) // 17) * 17 + 5)


class RenoiseNoise:
    def reset(self, seed: int) -> None:
        import mlx.core as mx

        self.key = mx.random.key(seed)

    def normal(self, shape):
        import mlx.core as mx

        self.key, key = mx.random.split(self.key)
        return mx.random.normal(shape, dtype=mx.float32, key=key)


class RenoiseScheduler:
    order = 1

    def __init__(self, shift: float = VIDEO_SHIFT, noise: RenoiseNoise | None = None):
        self.noise = noise if noise is not None else RenoiseNoise()
        if noise is None:
            self.noise.reset(0)
        self.set_shift(shift)
        self.set_timesteps()

    def set_shift(self, shift):
        shifted_sigmas(KNOTS, shift)
        self.shift = float(shift)

    def set_timesteps(self, num_inference_steps=None, *, sigmas=None):
        import mlx.core as mx

        self.sigmas = mx.array(shifted_sigmas(KNOTS, self.shift), dtype=mx.float32)
        self.timesteps = 1.0 - self.sigmas[:-1]
        self.num_inference_steps = len(KNOTS) - 1
        self._step_index = 0

    def scale_noise(self, sample, timestep, noise):
        import mlx.core as mx

        time = mx.array(timestep, dtype=sample.dtype)
        while time.ndim < sample.ndim:
            time = mx.expand_dims(time, axis=-1)
        return time * sample + (1.0 - time) * noise

    def step(self, model_output, timestep, sample):
        import mlx.core as mx

        denoised = sample.astype(mx.float32) + (
            1.0 - mx.array(timestep, dtype=mx.float32)
        ) * model_output.astype(mx.float32)
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        if float(sigma_next) == 0:
            return denoised.astype(sample.dtype)
        return (
            (1.0 - sigma_next) * denoised + sigma_next * self.noise.normal(sample.shape)
        ).astype(sample.dtype)


def verify_adapter(root: Path) -> None:
    try:
        manifest = json.loads((root / "fl2va/manifest.json").read_text())
        expected = {
            "rank": ADAPTER_RANK,
            "alpha": ADAPTER_ALPHA,
            "trunk": "transformer",
            "numerics_policy": "bf16_conditioning_v1",
            "target_modules": list(TARGET_MODULES),
            "sha256": ADAPTER_SHA256,
        }
        sampling = {
            "knots": list(KNOTS),
            "video_shift": VIDEO_SHIFT,
            "audio_shift": AUDIO_SHIFT,
            "method": "renoise",
        }
        if any(manifest.get(k) != v for k, v in expected.items()) or any(
            manifest.get("sampling", {}).get(k) != v for k, v in sampling.items()
        ):
            raise MiniMaxH3BackendError("Pika H3 Flex manifest mismatch")
        digest = hashlib.sha256()
        with (root / ADAPTER_FILE).open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != ADAPTER_SHA256:
            raise MiniMaxH3BackendError("Pika H3 Flex adapter sha256 mismatch")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise MiniMaxH3BackendError("Pika H3 Flex adapter verification failed") from exc


class PikaH3FlexVideoEngine(MiniMaxH3VideoEngine):
    supports_last_frame = True
    steps = len(KNOTS)

    def _prepare_model(self, model):
        from .._mirror import pinned_snapshot_download

        root = Path(
            pinned_snapshot_download(REPO, REVISION, allow_patterns=list(FILES))
        )
        verify_adapter(root)
        merge_lora(
            model.pipeline.transformer,
            root / ADAPTER_FILE,
            scale=ADAPTER_ALPHA / ADAPTER_RANK,
            allowed_suffixes=TARGET_MODULES,
            expected_rank=ADAPTER_RANK,
        )
        self._renoise = RenoiseNoise()
        pipeline = model.pipeline
        pipeline.scheduler = RenoiseScheduler(VIDEO_SHIFT, self._renoise)
        pipeline.audio_scheduler = RenoiseScheduler(AUDIO_SHIFT, self._renoise)
        normalize = pipeline._normalize_video_latents

        def normalize_bf16(latents):
            import mlx.core as mx

            return normalize(latents).astype(mx.bfloat16).astype(mx.float32)

        pipeline._normalize_video_latents = normalize_bf16

    def _prepare_request(self, seed: int) -> None:
        self._renoise.reset(seed + 1)
