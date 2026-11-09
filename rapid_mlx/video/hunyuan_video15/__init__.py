# SPDX-License-Identifier: Apache-2.0
"""HunyuanVideo 1.5 checkpoint contract; safe to import without MLX."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
REVISION = "286be7ce72277246578a3e3cc2487e95ddae5bcf"
ALIAS = "hunyuan-video-1.5-480p"


def is_hunyuan_video15_model(name: str | None) -> bool:
    if not name:
        return False
    if name in (REPO, ALIAS):
        return True
    try:
        index = Path(name) / "model_index.json"
        return (
            index.is_file()
            and json.loads(index.read_text()).get("_class_name")
            == "HunyuanVideo15Pipeline"
        )
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class PipelineConfig:
    transformer: dict
    vae: dict
    scheduler: dict
    guider: dict
    width: int = 848
    height: int = 480
    frames: int = 121
    fps: int = 24
    steps: int = 50

    @classmethod
    def from_path(cls, root: Path):
        def read(file):
            return json.loads((root / file).read_text())

        index = read("model_index.json")
        if index.get("_class_name") != "HunyuanVideo15Pipeline":
            raise ValueError("Expected HunyuanVideo15Pipeline (480p T2V)")
        transformer, vae = read("transformer/config.json"), read("vae/config.json")
        scheduler, guider = (
            read("scheduler/scheduler_config.json"),
            read("guider/guider_config.json"),
        )
        if (
            transformer.get("task_type") != "t2v"
            or transformer.get("target_size") != 640
            or transformer.get("use_meanflow", False)
        ):
            raise ValueError(
                "Only HunyuanVideo 1.5 480p non-distilled T2V is implemented"
            )
        if (
            transformer["in_channels"] != 2 * vae["latent_channels"] + 1
            or transformer["out_channels"] != vae["latent_channels"]
        ):
            raise ValueError("Transformer/VAE latent channels disagree")
        if sum(transformer["rope_axes_dim"]) != transformer["attention_head_dim"]:
            raise ValueError("Rotary dimensions must sum to attention_head_dim")
        if (
            vae["spatial_compression_ratio"] != 16
            or vae["temporal_compression_ratio"] != 4
        ):
            raise ValueError(
                "Only 16x spatial / 4x temporal VAE compression is implemented"
            )
        unsupported = (
            "use_dynamic_shifting",
            "invert_sigmas",
            "stochastic_sampling",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        )
        if (
            scheduler.get("_class_name") != "FlowMatchEulerDiscreteScheduler"
            or any(scheduler.get(k, False) for k in unsupported)
            or scheduler.get("shift_terminal") is not None
        ):
            raise ValueError("Unsupported HunyuanVideo scheduler configuration")
        if (
            guider.get("_class_name") != "ClassifierFreeGuidance"
            or guider.get("guidance_rescale", 0) != 0
        ):
            raise ValueError("Only plain classifier-free guidance is implemented")
        if not math.isfinite(scheduler["shift"]) or scheduler["shift"] <= 0:
            raise ValueError("Scheduler shift must be finite and positive")
        return cls(transformer, vae, scheduler, guider)


def validate_request(*, width, height, num_frames, image=None):
    if image is not None:
        raise ValueError("HunyuanVideo 1.5 currently supports text-to-video only")
    if not (
        min(width, height) >= 256
        and min(width, height) <= 480
        and max(width, height) <= 848
        and width % 16 == height % 16 == 0
    ):
        raise ValueError(
            "HunyuanVideo 1.5 480p dimensions must be multiples of 16, short edge 256..480 and long edge <=848"
        )
    if not 5 <= num_frames <= 121 or num_frames % 4 != 1:
        raise ValueError("HunyuanVideo 1.5 frames must be 4n+1, between 5 and 121")


class FlowMatchScheduler:
    """Diffusers' explicit-sigma shifted flow Euler schedule (float32)."""

    def __init__(self, config, steps):
        import numpy as np

        if not 1 <= steps <= 1000:
            raise ValueError("steps must be between 1 and 1000")
        sigmas = np.linspace(1, 0, steps + 1)[:-1].astype(np.float32)
        shift = config["shift"]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])
        self.timesteps = sigmas * config.get("num_train_timesteps", 1000)

    def step(self, model_output, index, sample):
        # Both the scheduler's state and accumulation are float32 upstream.
        return (
            sample + (self.sigmas[index + 1] - self.sigmas[index]).item() * model_output
        )
