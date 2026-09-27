# SPDX-License-Identifier: Apache-2.0
"""Immutable, provenance-carrying runtime configuration contract.

This module is intentionally disconnected from the production startup path.
It defines the contract and precedence rules used by migration 002; adapters
can compare it with the existing CLI and Server resolvers before either caller
is switched over.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias, TypeVar

RuntimeValue: TypeAlias = bool | int | float | str | None


class RuntimeResolutionError(ValueError):
    """Raised when runtime inputs cannot produce an authoritative result."""


class RuntimeField(str, Enum):
    """Initial migration surface shared by CLI, Server, and Desktop."""

    PREFILL_STEP_SIZE = "prefill_step_size"
    MAX_NUM_SEQS = "max_num_seqs"
    GPU_MEMORY_UTILIZATION = "gpu_memory_utilization"
    ENABLE_PREFIX_CACHE = "enable_prefix_cache"
    KV_CACHE_DTYPE = "kv_cache_dtype"


class RuntimeValueSource(str, Enum):
    GLOBAL_DEFAULT = "global_default"
    PERFORMANCE_PROFILE = "performance_profile"
    USER_OVERRIDE = "user_override"
    COMPATIBILITY = "compatibility"
    SAFETY = "safety"


class RuntimeReasonCode(str, Enum):
    GLOBAL_DEFAULT = "global_default"
    PROFILE_RECOMMENDATION = "profile_recommendation"
    EXPLICIT_USER_OVERRIDE = "explicit_user_override"
    COMPATIBILITY_FALLBACK = "compatibility_fallback"
    SAFETY_FALLBACK = "safety_fallback"


class RuntimeTraceAction(str, Enum):
    SELECTED = "selected"
    OVERRIDDEN = "overridden"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class RuntimeDefault:
    field: RuntimeField
    value: RuntimeValue
    source_id: str


@dataclass(frozen=True, slots=True)
class RuntimeProfileValue:
    field: RuntimeField
    value: RuntimeValue
    source_id: str


@dataclass(frozen=True, slots=True)
class RuntimeConfigOverride:
    """One explicit operator value, separate from model selection."""

    field: RuntimeField
    value: RuntimeValue
    source_id: str


@dataclass(frozen=True, slots=True)
class RuntimeConstraint:
    """A hard compatibility or safety adjustment applied after overrides."""

    field: RuntimeField
    value: RuntimeValue
    source: RuntimeValueSource
    source_id: str
    reason_code: RuntimeReasonCode


@dataclass(frozen=True, slots=True)
class RuntimeTraceEntry:
    value: RuntimeValue
    source: RuntimeValueSource
    source_id: str
    reason_code: RuntimeReasonCode
    action: RuntimeTraceAction


@dataclass(frozen=True, slots=True)
class EffectiveRuntimeValue:
    field: RuntimeField
    value: RuntimeValue
    source: RuntimeValueSource
    source_id: str
    reason_code: RuntimeReasonCode
    trace: tuple[RuntimeTraceEntry, ...]


@dataclass(frozen=True, slots=True)
class EffectiveRuntimeConfig:
    """Authoritative immutable result, ordered by :class:`RuntimeField`."""

    fields: tuple[EffectiveRuntimeValue, ...]

    def __post_init__(self) -> None:
        actual = tuple(item.field for item in self.fields)
        expected = tuple(RuntimeField)
        if actual != expected:
            raise RuntimeResolutionError(
                "effective config must contain every runtime field in canonical order"
            )

    def get(self, field: RuntimeField) -> EffectiveRuntimeValue:
        if not isinstance(field, RuntimeField):
            raise RuntimeResolutionError(f"unknown runtime field: {field!r}")
        return self.fields[list(RuntimeField).index(field)]


@dataclass(frozen=True, slots=True)
class _FieldPolicy:
    validate: Callable[[RuntimeValue], bool]
    expected: str


class _RuntimeInput(Protocol):
    @property
    def field(self) -> RuntimeField: ...

    @property
    def value(self) -> RuntimeValue: ...

    @property
    def source_id(self) -> str: ...


_RuntimeInputT = TypeVar("_RuntimeInputT", bound=_RuntimeInput)


def _positive_int(value: RuntimeValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _optional_utilization(value: RuntimeValue) -> bool:
    return value is None or (isinstance(value, float) and 0.0 < value <= 1.0)


def _boolean(value: RuntimeValue) -> bool:
    return isinstance(value, bool)


def _kv_cache_dtype(value: RuntimeValue) -> bool:
    return isinstance(value, str) and value in {"bf16", "int8", "int4"}


_FIELD_POLICIES: dict[RuntimeField, _FieldPolicy] = {
    RuntimeField.PREFILL_STEP_SIZE: _FieldPolicy(_positive_int, "a positive integer"),
    RuntimeField.MAX_NUM_SEQS: _FieldPolicy(_positive_int, "a positive integer"),
    RuntimeField.GPU_MEMORY_UTILIZATION: _FieldPolicy(
        _optional_utilization, "None or a float in (0, 1]"
    ),
    RuntimeField.ENABLE_PREFIX_CACHE: _FieldPolicy(_boolean, "a boolean"),
    RuntimeField.KV_CACHE_DTYPE: _FieldPolicy(
        _kv_cache_dtype, "one of: bf16, int8, int4"
    ),
}


def _validate_input(
    *, field: object, value: RuntimeValue, source_id: str, kind: str
) -> RuntimeField:
    if not isinstance(field, RuntimeField):
        raise RuntimeResolutionError(f"unknown runtime field: {field!r}")
    if not isinstance(source_id, str) or not source_id.strip():
        raise RuntimeResolutionError(f"{kind} source_id must not be empty")
    policy = _FIELD_POLICIES[field]
    if not policy.validate(value):
        raise RuntimeResolutionError(
            f"invalid {field.value} from {source_id!r}: expected {policy.expected}"
        )
    return field


def _unique_by_field(
    items: tuple[_RuntimeInputT, ...], *, kind: str
) -> dict[RuntimeField, _RuntimeInputT]:
    result: dict[RuntimeField, _RuntimeInputT] = {}
    for item in items:
        field = _validate_input(
            field=item.field,
            value=item.value,
            source_id=item.source_id,
            kind=kind,
        )
        if field in result:
            raise RuntimeResolutionError(f"duplicate {kind} for {field.value}")
        result[field] = item
    return result


def resolve_effective_runtime_config(
    *,
    defaults: tuple[RuntimeDefault, ...],
    profile_values: tuple[RuntimeProfileValue, ...] = (),
    overrides: tuple[RuntimeConfigOverride, ...] = (),
    constraints: tuple[RuntimeConstraint, ...] = (),
) -> EffectiveRuntimeConfig:
    """Resolve defaults < profile < explicit user < hard constraints.

    Inputs are tuples so the complete request is immutable and reproducible.
    Unknown, duplicate, missing, or incompatible inputs fail closed.
    """

    default_by_field = _unique_by_field(defaults, kind="default")
    missing = [field.value for field in RuntimeField if field not in default_by_field]
    if missing:
        raise RuntimeResolutionError("missing runtime defaults: " + ", ".join(missing))
    profile_by_field = _unique_by_field(profile_values, kind="profile value")
    override_by_field = _unique_by_field(overrides, kind="override")
    constraint_by_field = _unique_by_field(constraints, kind="constraint")

    for candidate_constraint in constraints:
        expected_reason = {
            RuntimeValueSource.COMPATIBILITY: RuntimeReasonCode.COMPATIBILITY_FALLBACK,
            RuntimeValueSource.SAFETY: RuntimeReasonCode.SAFETY_FALLBACK,
        }.get(candidate_constraint.source)
        if (
            expected_reason is None
            or candidate_constraint.reason_code is not expected_reason
        ):
            raise RuntimeResolutionError(
                "constraints must pair a compatibility/safety source with its "
                "matching fallback reason"
            )

    resolved: list[EffectiveRuntimeValue] = []
    for field in RuntimeField:
        default = default_by_field[field]
        assert isinstance(default, RuntimeDefault)
        trace = [
            RuntimeTraceEntry(
                value=default.value,
                source=RuntimeValueSource.GLOBAL_DEFAULT,
                source_id=default.source_id,
                reason_code=RuntimeReasonCode.GLOBAL_DEFAULT,
                action=RuntimeTraceAction.SELECTED,
            )
        ]

        profile = profile_by_field.get(field)
        if profile is not None:
            assert isinstance(profile, RuntimeProfileValue)
            trace.append(
                RuntimeTraceEntry(
                    value=profile.value,
                    source=RuntimeValueSource.PERFORMANCE_PROFILE,
                    source_id=profile.source_id,
                    reason_code=RuntimeReasonCode.PROFILE_RECOMMENDATION,
                    action=RuntimeTraceAction.OVERRIDDEN,
                )
            )

        override = override_by_field.get(field)
        if override is not None:
            assert isinstance(override, RuntimeConfigOverride)
            trace.append(
                RuntimeTraceEntry(
                    value=override.value,
                    source=RuntimeValueSource.USER_OVERRIDE,
                    source_id=override.source_id,
                    reason_code=RuntimeReasonCode.EXPLICIT_USER_OVERRIDE,
                    action=RuntimeTraceAction.OVERRIDDEN,
                )
            )

        field_constraint = constraint_by_field.get(field)
        if field_constraint is not None:
            trace.append(
                RuntimeTraceEntry(
                    value=field_constraint.value,
                    source=field_constraint.source,
                    source_id=field_constraint.source_id,
                    reason_code=field_constraint.reason_code,
                    action=RuntimeTraceAction.FALLBACK,
                )
            )

        final = trace[-1]
        resolved.append(
            EffectiveRuntimeValue(
                field=field,
                value=final.value,
                source=final.source,
                source_id=final.source_id,
                reason_code=final.reason_code,
                trace=tuple(trace),
            )
        )

    return EffectiveRuntimeConfig(fields=tuple(resolved))
