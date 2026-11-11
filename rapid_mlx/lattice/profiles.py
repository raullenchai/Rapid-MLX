# SPDX-License-Identifier: Apache-2.0
"""Immutable facts from existing device information and already-loaded config."""

from __future__ import annotations

import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib.metadata import PackageNotFoundError, version

from ..chip_tier import classify_chip_tier


@cache
def mlx_version() -> str:
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


@cache
def rapid_version() -> str:
    try:
        return version("rapid-mlx")
    except PackageNotFoundError:
        return "unknown"


@cache
def _device_key(name) -> str:
    match = (
        re.fullmatch(r"Apple (M[1-9][0-9]*)(?: (Pro|Max|Ultra))?", name.strip())
        if isinstance(name, str)
        else None
    )
    return " ".join(filter(None, match.groups())) if match else "unknown"


def device_key(info: Mapping) -> str:
    return _device_key(info.get("device_name", ""))


@cache
def macos_version() -> str:
    return platform.mac_ver()[0]


@dataclass(frozen=True)
class HardwareProfile:
    architecture: str = "unknown"
    gpu_generation: int | None = None
    tier: str = "unknown"
    gpu_cores: int | None = None
    memory_size: int | None = None


@dataclass(frozen=True)
class ModelProfile:
    family: str = "unknown"
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_experts: int = 0
    top_k: int = 0
    bits: int | None = None
    group_size: int | None = None
    attention_kinds: tuple[str, ...] = ()
    quant_mode: str = "unknown"

    @property
    def geometry(self) -> tuple:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
            self.bits,
            self.group_size,
        )


def model_profile(config: Mapping) -> ModelProfile:
    text = config.get("text_config") or config.get("language_config") or config
    quant = (
        config.get("quantization")
        or text.get("quantization")
        or config.get("quantization_config")
        or {}
    )
    family = text.get("model_type", "unknown")
    family = (
        family
        if isinstance(family, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", family)
        else "unknown"
    )

    def integer(mapping, *keys):
        return next((mapping[k] for k in keys if type(mapping.get(k)) is int), None)

    kinds = tuple(
        sorted(
            {
                k
                for k in text.get("layer_types", ())
                if k
                in {
                    "full_attention",
                    "linear_attention",
                    "sliding_attention",
                    "gated_delta_net",
                }
            }
        )
    )
    mode = quant.get("mode", "affine" if quant.get("bits") else "unknown")
    return ModelProfile(
        family,
        integer(text, "hidden_size", "d_model"),
        integer(text, "moe_intermediate_size", "intermediate_size"),
        integer(text, "num_experts", "num_local_experts", "n_routed_experts") or 0,
        integer(text, "num_experts_per_tok", "num_experts_per_token", "top_k") or 0,
        integer(quant, "bits"),
        integer(quant, "group_size"),
        kinds,
        mode if mode in {"affine", "mxfp4", "mxfp8"} else "unknown",
    )


@dataclass(frozen=True)
class LatticeContext:
    hardware: HardwareProfile
    mlx: str
    macos: str
    model: ModelProfile = ModelProfile()
    mode: str = "plain"
    width: int = 1
    batch: int = 1
    has_mtp_head: bool = False
    rapid_mlx: str = "unknown"


@lru_cache(maxsize=128)
def _context(
    name,
    architecture,
    cores,
    memory,
    library_version,
    macos,
    model,
    mode,
    width,
    batch,
    has_mtp_head,
    rapid_mlx,
) -> LatticeContext:
    tier = classify_chip_tier(name)
    arch = re.fullmatch(r"applegpu_g(\d+)[a-z]*", architecture)
    # Older MLX device_info omitted architecture. Only known Apple generations
    # have this fallback; never extrapolate an ISA for an unknown future chip.
    generation = (
        int(arch[1])
        if arch
        else (
            {1: 13, 2: 14, 3: 15, 4: 16, 5: 17}.get(tier.generation)
            if tier.generation is not None
            else None
        )
    )
    return LatticeContext(
        HardwareProfile(architecture, generation, tier.variant, cores, memory),
        library_version,
        macos,
        model,
        mode,
        width,
        batch,
        has_mtp_head,
        rapid_mlx,
    )


def runtime_context(
    info: Mapping,
    *,
    config: Mapping | None = None,
    model=ModelProfile(),
    library_version=None,
    macos=None,
    mode="plain",
    width=1,
    batch=1,
    has_mtp_head=False,
    rapid_mlx=None,
) -> LatticeContext:
    return _context(
        device_key(info),
        str(info.get("architecture", "unknown")),
        info.get("gpu_cores"),
        info.get("memory_size"),
        mlx_version() if library_version is None else library_version,
        macos_version() if macos is None else macos,
        model_profile(config) if config is not None else model,
        mode,
        width,
        batch,
        has_mtp_head,
        rapid_version() if rapid_mlx is None else rapid_mlx,
    )
