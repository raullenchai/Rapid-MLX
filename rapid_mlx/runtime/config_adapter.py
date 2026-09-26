# SPDX-License-Identifier: Apache-2.0
"""Rollback adapter between legacy startup values and the central resolver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .effective_config import (
    EffectiveRuntimeConfig,
    RuntimeConfigOverride,
    RuntimeConstraint,
    RuntimeDefault,
    RuntimeField,
    RuntimeProfileValue,
    RuntimeResolutionError,
    resolve_effective_runtime_config,
)


class RuntimeParityError(RuntimeResolutionError):
    """Raised when the new resolver differs from the legacy startup result."""


@dataclass(frozen=True, slots=True)
class RuntimeLaunchValues:
    """Typed values at the CLI/Server engine-construction boundary."""

    prefill_step_size: int
    max_num_seqs: int
    gpu_memory_utilization: float | None
    enable_prefix_cache: bool
    kv_cache_dtype: str

    def value_for(self, field: RuntimeField):
        return {
            RuntimeField.PREFILL_STEP_SIZE: self.prefill_step_size,
            RuntimeField.MAX_NUM_SEQS: self.max_num_seqs,
            RuntimeField.GPU_MEMORY_UTILIZATION: self.gpu_memory_utilization,
            RuntimeField.ENABLE_PREFIX_CACHE: self.enable_prefix_cache,
            RuntimeField.KV_CACHE_DTYPE: self.kv_cache_dtype,
        }[field]

    def as_defaults(self, source_id: str) -> tuple[RuntimeDefault, ...]:
        return tuple(
            RuntimeDefault(field, self.value_for(field), source_id)
            for field in RuntimeField
        )

    @classmethod
    def from_effective(cls, config: EffectiveRuntimeConfig) -> RuntimeLaunchValues:
        return cls(
            prefill_step_size=cast(
                int, config.get(RuntimeField.PREFILL_STEP_SIZE).value
            ),
            max_num_seqs=cast(int, config.get(RuntimeField.MAX_NUM_SEQS).value),
            gpu_memory_utilization=cast(
                float | None,
                config.get(RuntimeField.GPU_MEMORY_UTILIZATION).value,
            ),
            enable_prefix_cache=cast(
                bool, config.get(RuntimeField.ENABLE_PREFIX_CACHE).value
            ),
            kv_cache_dtype=cast(str, config.get(RuntimeField.KV_CACHE_DTYPE).value),
        )


DEFAULT_RUNTIME_LAUNCH_VALUES = RuntimeLaunchValues(
    prefill_step_size=2048,
    max_num_seqs=256,
    gpu_memory_utilization=None,
    enable_prefix_cache=True,
    kv_cache_dtype="bf16",
)


def resolve_with_legacy_parity(
    *,
    surface: str,
    legacy: RuntimeLaunchValues,
    defaults: tuple[RuntimeDefault, ...],
    profile_values: tuple[RuntimeProfileValue, ...] = (),
    overrides: tuple[RuntimeConfigOverride, ...] = (),
    constraints: tuple[RuntimeConstraint, ...] = (),
) -> EffectiveRuntimeConfig:
    """Resolve centrally and fail closed if the legacy adapter disagrees."""

    if not surface.strip():
        raise RuntimeParityError("runtime parity surface must not be empty")
    config = resolve_effective_runtime_config(
        defaults=defaults,
        profile_values=profile_values,
        overrides=overrides,
        constraints=constraints,
    )
    resolved = RuntimeLaunchValues.from_effective(config)
    if resolved != legacy:
        mismatches = [
            f"{field.value}: legacy={legacy.value_for(field)!r}, "
            f"central={resolved.value_for(field)!r}"
            for field in RuntimeField
            if legacy.value_for(field) != resolved.value_for(field)
        ]
        raise RuntimeParityError(
            f"{surface} runtime parity mismatch: " + "; ".join(mismatches)
        )
    return config
