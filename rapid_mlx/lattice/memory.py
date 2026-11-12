# SPDX-License-Identifier: Apache-2.0
"""Universal memory constants, frozen at setup without device probes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any

from . import resolve, runtime_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryPolicy:
    metal_cache_floor_bytes: int = 2 * 1024**3
    metal_cache_cap_bytes: int = 32 * 1024**3
    metal_cache_divisor: int = 4
    memory_pressure_margin: float = 0.05
    memory_pressure_ceiling: float = 0.99
    memory_pressure_fallback_bytes: int = 200 * 1024**3
    memory_check_interval: int = 16

    def metal_cache_limit(self, soft_limit_bytes: int) -> int:
        cache = max(
            self.metal_cache_floor_bytes,
            min(
                self.metal_cache_cap_bytes, soft_limit_bytes // self.metal_cache_divisor
            ),
        )
        return min(cache, soft_limit_bytes) if soft_limit_bytes > 0 else cache

    def pressure_threshold(
        self, utilization: float, device_info: Callable[[], Mapping]
    ) -> int:
        """Read existing device facts once; preserve both legacy fallback paths."""
        try:
            info = device_info()
            recommended = info.get(
                "max_recommended_working_set_size", info.get("memory_size", 0)
            )
            memory = (
                recommended if recommended > 0 else self.memory_pressure_fallback_bytes
            )
            return int(
                memory
                * min(
                    utilization + self.memory_pressure_margin,
                    self.memory_pressure_ceiling,
                )
            )
        except Exception as exc:
            logger.debug("Memory pressure device fallback: %s", exc)
            return self.memory_pressure_fallback_bytes


@lru_cache(maxsize=1)
def memory_policy() -> MemoryPolicy:
    """Resolve once at engine setup; universal cells need no hardware facts.

    The immutable result carries decisions into the engine loop. Any resolution
    failure freezes the legacy constants instead, including on no-MLX hosts.
    """
    try:
        context = runtime_context({})
        values: dict[str, Any] = {
            field.name: resolve(field.name, context).value
            for field in fields(MemoryPolicy)
        }
        policy = MemoryPolicy(**values)
        if policy.metal_cache_floor_bytes > policy.metal_cache_cap_bytes:
            # Each knob validates alone; reject a contradictory pair as a whole
            # so a lowered cap is never silently overridden by the floor.
            raise ValueError(
                "metal_cache_floor_bytes exceeds metal_cache_cap_bytes "
                f"({policy.metal_cache_floor_bytes} > {policy.metal_cache_cap_bytes})"
            )
        return policy
    except Exception as exc:
        logger.debug("Lattice memory policy resolution failed; using defaults: %s", exc)
        return MemoryPolicy()
