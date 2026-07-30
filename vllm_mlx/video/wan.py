# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 / 2.2 adapter for the unified video-generation lane."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

WAN_REVISIONS: dict[str, str] = {
    "Anes1032/Wan2.2-TI2V-5B-mlx-q8": ("9624723c94ddf509832555c45e223a035baa7d1c"),
    "rickylin20260522/Wan2.2-TI2V-5B-mlx": ("592b2473f27cd6f466cdd9f2c0f5750a77b37b59"),
    "Anes1032/Wan2.2-I2V-A14B-mlx-q8": ("633f50fc3e16e7faf76713dcf07b0bea730f02c9"),
    "rickylin20260522/Wan2.2-T2V-A14B-mlx": (
        "225358452f995a6807acaebff9dfc4976c39c8c8"
    ),
}

_SCHEDULERS = frozenset({"euler", "dpm++", "unipc"})
_TILING_MODES = frozenset(
    {"auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"}
)
_MAX_LORA_STRENGTH = 4.0
_SAFE_DEFAULT_MAX_AREA = 704 * 1280
_WAN_MODEL_TYPES = frozenset({"t2v", "i2v", "ti2v"})


class WanRequestError(ValueError):
    """A caller-fixable Wan request error."""


class WanBackendError(RuntimeError):
    """A safe, operator-fixable Wan configuration error."""


def is_wan_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    folded = model_name.casefold()
    return "wan2.1" in folded or "wan2.2" in folded or "/wan2" in folded


def _parse_loras(spec: str | None) -> list[tuple[str, float]] | None:
    if not spec:
        return None
    parsed: list[tuple[str, float]] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        path, separator, tail = part.rpartition(":")
        if separator and path:
            try:
                strength = float(tail)
            except ValueError as exc:
                raise WanBackendError(
                    f"invalid Wan LoRA strength in {part!r}; expected path[:strength]"
                ) from exc
            if not math.isfinite(strength) or not 0 <= strength <= _MAX_LORA_STRENGTH:
                raise WanBackendError(
                    f"Wan LoRA strength must be between 0 and {_MAX_LORA_STRENGTH}"
                )
            parsed.append((path, strength))
            continue
        parsed.append((part, 1.0))
    return parsed or None


def _env_choice(name: str, default: str, allowed: frozenset[str]) -> str:
    value = os.environ.get(name, default)
    if value not in allowed:
        raise WanBackendError(
            f"${name} must be one of {', '.join(sorted(allowed))}; got {value!r}"
        )
    return value


def _env_steps() -> int | None:
    raw = os.environ.get("RAPID_MLX_WAN_STEPS")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise WanBackendError("$RAPID_MLX_WAN_STEPS must be an integer") from exc
    if not 1 <= value <= 500:
        raise WanBackendError("$RAPID_MLX_WAN_STEPS must be between 1 and 500")
    return value


def _resolve_model_path(model_name: str) -> Path:
    local_override = os.environ.get("RAPID_MLX_WAN_MODEL_DIR")
    candidate = Path(local_override or model_name).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if local_override:
        logger.error("Configured Wan model directory does not exist: %s", candidate)
        raise WanBackendError(
            "$RAPID_MLX_WAN_MODEL_DIR does not point to a directory; "
            "check the server log for the resolved path"
        )

    from huggingface_hub import snapshot_download

    from ..model_aliases import resolve_model

    repository = resolve_model(model_name)
    revision = WAN_REVISIONS.get(repository)
    if revision is None:
        raise WanBackendError(
            f"remote Wan checkpoint {repository!r} is not registered at a pinned "
            "revision; use a registered alias or RAPID_MLX_WAN_MODEL_DIR"
        )
    try:
        return Path(snapshot_download(repository, revision=revision))
    except Exception as exc:
        logger.exception("Could not resolve Wan checkpoint %s", repository)
        raise WanBackendError(
            f"could not resolve Wan checkpoint {repository!r}: "
            f"{type(exc).__name__}; check network access and repository permissions"
        ) from exc


class WanVideoEngine:
    """Adapter over ``mlx_video.generate_wan.generate_video``.

    ``mlx-video-with-audio`` supports converted Wan 2.1 and 2.2 checkpoint
    layouts. The checkpoint config determines T2V/I2V/TI2V, native frame
    rate and the safe pixel-area ceiling.
    """

    family = "wan"

    def __init__(self, model_name: str) -> None:
        self.model_name = str(model_name)
        self.model_path = _resolve_model_path(self.model_name)
        self.steps = _env_steps()
        self.scheduler = _env_choice("RAPID_MLX_WAN_SCHEDULER", "unipc", _SCHEDULERS)
        self.tiling = _env_choice("RAPID_MLX_WAN_TILING", "auto", _TILING_MODES)
        self.loras = _parse_loras(os.environ.get("RAPID_MLX_WAN_LORA"))
        self.loras_high = _parse_loras(os.environ.get("RAPID_MLX_WAN_LORA_HIGH"))
        self.loras_low = _parse_loras(os.environ.get("RAPID_MLX_WAN_LORA_LOW"))
        self.config = self._read_config()
        if self.model_type not in _WAN_MODEL_TYPES:
            raise WanBackendError(
                "the Wan checkpoint config.json has unsupported model_type "
                f"{self.model_type!r}; expected t2v, i2v, or ti2v"
            )

    def _read_config(self) -> dict:
        config_path = self.model_path / "config.json"
        if not config_path.is_file():
            logger.warning(
                "Wan checkpoint %s has no config.json; family-specific FPS and "
                "area guards are unavailable",
                self.model_path,
            )
            return {}
        try:
            with config_path.open() as config_file:
                config = json.load(config_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WanBackendError(
                "the Wan checkpoint has an unreadable config.json"
            ) from exc
        if not isinstance(config, dict):
            raise WanBackendError("the Wan checkpoint config.json must be an object")
        return config

    @property
    def native_fps(self) -> int:
        configured = self.config.get("sample_fps")
        if configured is not None:
            try:
                value = float(configured)
            except (TypeError, ValueError):
                value = 0
            if math.isfinite(value) and value > 0:
                rounded = round(value)
                if rounded >= 1:
                    return rounded
        version = str(self.config.get("model_version") or "")
        return 24 if version.startswith("2.2") or "2.2" in self.model_name else 16

    @property
    def max_area(self) -> int:
        try:
            value = int(self.config.get("max_area") or 0)
        except (TypeError, ValueError):
            value = 0
        # Upstream Wan 2.1/2.2 T2V configs use 0 to mean unrestricted.
        # The public API must retain a finite safety bound, so local/custom
        # configs without a positive checkpoint limit get the TI2V 5B ceiling.
        return value if value > 0 else _SAFE_DEFAULT_MAX_AREA

    @property
    def model_type(self) -> str:
        return str(self.config.get("model_type") or "").casefold()

    def validate_request(
        self, *, width: int, height: int, num_frames: int, image: Path | None
    ) -> None:
        if num_frames % 4 != 1:
            lower = max(1, ((num_frames - 1) // 4) * 4 + 1)
            upper = lower + 4
            raise WanRequestError(
                f"Wan num_frames must be 4n+1; got {num_frames}. "
                f"Nearest valid values are {lower} and {upper}."
            )
        if self.max_area and width * height > self.max_area:
            raise WanRequestError(
                f"{width}x{height} exceeds this Wan checkpoint's "
                f"{self.max_area}-pixel ceiling"
            )
        if self.model_type == "t2v" and image is not None:
            raise WanRequestError(
                "this Wan checkpoint is text-to-video only; omit input_reference"
            )
        if self.model_type == "i2v" and image is None:
            raise WanRequestError(
                "this Wan checkpoint is image-to-video only and requires "
                "input_reference"
            )

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        image: Path | None,
    ) -> None:
        self.validate_request(
            width=width, height=height, num_frames=num_frames, image=image
        )
        try:
            from mlx_video.generate_wan import generate_video
        except ImportError as exc:
            raise WanBackendError(
                "Wan generation requires mlx-video-with-audio>=0.1.36; "
                "install the rapid-mlx video extra"
            ) from exc

        generate_video(
            model_dir=str(self.model_path),
            prompt=prompt,
            image=str(image) if image is not None else None,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=self.steps,
            seed=seed,
            output_path=str(output_path),
            scheduler=self.scheduler,
            tiling=self.tiling,
            loras=self.loras,
            loras_high=self.loras_high,
            loras_low=self.loras_low,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise WanBackendError("Wan generation completed without an MP4 output")
