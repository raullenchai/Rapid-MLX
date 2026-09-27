# SPDX-License-Identifier: Apache-2.0
"""Contract tests for migration 002's effective runtime configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rapid_mlx.runtime.effective_config import (
    EffectiveRuntimeConfig,
    EffectiveRuntimeValue,
    RuntimeConfigOverride,
    RuntimeConstraint,
    RuntimeDefault,
    RuntimeField,
    RuntimeProfileValue,
    RuntimeReasonCode,
    RuntimeResolutionError,
    RuntimeTraceAction,
    RuntimeTraceEntry,
    RuntimeValueSource,
    resolve_effective_runtime_config,
)


def _defaults() -> tuple[RuntimeDefault, ...]:
    return (
        RuntimeDefault(RuntimeField.PREFILL_STEP_SIZE, 2048, "scheduler:v1"),
        RuntimeDefault(RuntimeField.MAX_NUM_SEQS, 256, "scheduler:v1"),
        RuntimeDefault(RuntimeField.GPU_MEMORY_UTILIZATION, None, "server:v1"),
        RuntimeDefault(RuntimeField.ENABLE_PREFIX_CACHE, True, "scheduler:v1"),
        RuntimeDefault(RuntimeField.KV_CACHE_DTYPE, "bf16", "scheduler:v1"),
    )


@pytest.mark.parametrize(
    ("profiles", "overrides", "constraints", "expected", "source", "actions"),
    [
        ((), (), (), 2048, RuntimeValueSource.GLOBAL_DEFAULT, ["selected"]),
        (
            (RuntimeProfileValue(RuntimeField.PREFILL_STEP_SIZE, 512, "profile:qwen"),),
            (),
            (),
            512,
            RuntimeValueSource.PERFORMANCE_PROFILE,
            ["selected", "overridden"],
        ),
        (
            (RuntimeProfileValue(RuntimeField.PREFILL_STEP_SIZE, 512, "profile:qwen"),),
            (RuntimeConfigOverride(RuntimeField.PREFILL_STEP_SIZE, 1024, "cli"),),
            (),
            1024,
            RuntimeValueSource.USER_OVERRIDE,
            ["selected", "overridden", "overridden"],
        ),
        (
            (RuntimeProfileValue(RuntimeField.PREFILL_STEP_SIZE, 512, "profile:qwen"),),
            (RuntimeConfigOverride(RuntimeField.PREFILL_STEP_SIZE, 4096, "api"),),
            (
                RuntimeConstraint(
                    RuntimeField.PREFILL_STEP_SIZE,
                    1024,
                    RuntimeValueSource.COMPATIBILITY,
                    "mlx-lm:0.31",
                    RuntimeReasonCode.COMPATIBILITY_FALLBACK,
                ),
            ),
            1024,
            RuntimeValueSource.COMPATIBILITY,
            ["selected", "overridden", "overridden", "fallback"],
        ),
    ],
)
def test_precedence_is_table_driven_and_traceable(
    profiles, overrides, constraints, expected, source, actions
):
    config = resolve_effective_runtime_config(
        defaults=_defaults(),
        profile_values=profiles,
        overrides=overrides,
        constraints=constraints,
    )
    result = config.get(RuntimeField.PREFILL_STEP_SIZE)
    assert result.value == expected
    assert result.source is source
    assert [entry.action.value for entry in result.trace] == actions
    assert result.source_id == result.trace[-1].source_id
    assert result.reason_code is result.trace[-1].reason_code


def test_all_supported_value_shapes_resolve_with_field_provenance():
    config = resolve_effective_runtime_config(defaults=_defaults())
    assert tuple(item.field for item in config.fields) == tuple(RuntimeField)
    assert config.get(RuntimeField.MAX_NUM_SEQS).value == 256
    assert config.get(RuntimeField.GPU_MEMORY_UTILIZATION).value is None
    assert config.get(RuntimeField.ENABLE_PREFIX_CACHE).value is True
    assert config.get(RuntimeField.KV_CACHE_DTYPE).value == "bf16"
    assert all(item.source_id for item in config.fields)
    assert all(
        item.reason_code is RuntimeReasonCode.GLOBAL_DEFAULT for item in config.fields
    )


def test_safety_constraint_is_visible_after_explicit_override():
    result = resolve_effective_runtime_config(
        defaults=_defaults(),
        overrides=(
            RuntimeConfigOverride(
                RuntimeField.GPU_MEMORY_UTILIZATION, 0.95, "desktop:settings"
            ),
        ),
        constraints=(
            RuntimeConstraint(
                RuntimeField.GPU_MEMORY_UTILIZATION,
                0.75,
                RuntimeValueSource.SAFETY,
                "memory-pressure:v1",
                RuntimeReasonCode.SAFETY_FALLBACK,
            ),
        ),
    ).get(RuntimeField.GPU_MEMORY_UTILIZATION)
    assert result.value == 0.75
    assert result.source is RuntimeValueSource.SAFETY
    assert [entry.source for entry in result.trace] == [
        RuntimeValueSource.GLOBAL_DEFAULT,
        RuntimeValueSource.USER_OVERRIDE,
        RuntimeValueSource.SAFETY,
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (RuntimeField.PREFILL_STEP_SIZE, 0),
        (RuntimeField.PREFILL_STEP_SIZE, True),
        (RuntimeField.MAX_NUM_SEQS, -1),
        (RuntimeField.GPU_MEMORY_UTILIZATION, 1),
        (RuntimeField.GPU_MEMORY_UTILIZATION, 1.1),
        (RuntimeField.ENABLE_PREFIX_CACHE, 1),
        (RuntimeField.KV_CACHE_DTYPE, "fp32"),
    ],
)
def test_invalid_values_fail_closed(field, value):
    defaults = tuple(
        RuntimeDefault(
            item.field, value if item.field is field else item.value, item.source_id
        )
        for item in _defaults()
    )
    with pytest.raises(RuntimeResolutionError, match=f"invalid {field.value}"):
        resolve_effective_runtime_config(defaults=defaults)


@pytest.mark.parametrize(
    "bad_input, message",
    [
        (
            (RuntimeDefault(RuntimeField.PREFILL_STEP_SIZE, 1024, "duplicate"),),
            "duplicate default",
        ),
        (
            (RuntimeDefault(RuntimeField.PREFILL_STEP_SIZE, 1024, ""),),
            "source_id must not be empty",
        ),
    ],
)
def test_duplicate_and_unattributed_inputs_fail_closed(bad_input, message):
    with pytest.raises(RuntimeResolutionError, match=message):
        resolve_effective_runtime_config(defaults=_defaults() + bad_input)


def test_non_string_source_id_fails_closed():
    invalid = RuntimeDefault.__new__(RuntimeDefault)
    object.__setattr__(invalid, "field", RuntimeField.PREFILL_STEP_SIZE)
    object.__setattr__(invalid, "value", 1024)
    object.__setattr__(invalid, "source_id", 7)
    with pytest.raises(RuntimeResolutionError, match="source_id must not be empty"):
        resolve_effective_runtime_config(defaults=_defaults()[:-1] + (invalid,))


def test_missing_and_unknown_fields_fail_closed():
    with pytest.raises(RuntimeResolutionError, match="missing runtime defaults"):
        resolve_effective_runtime_config(defaults=_defaults()[:-1])

    unknown = RuntimeDefault.__new__(RuntimeDefault)
    object.__setattr__(unknown, "field", "new_unreviewed_knob")
    object.__setattr__(unknown, "value", 1)
    object.__setattr__(unknown, "source_id", "test")
    with pytest.raises(RuntimeResolutionError, match="unknown runtime field"):
        resolve_effective_runtime_config(defaults=_defaults() + (unknown,))


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (RuntimeValueSource.USER_OVERRIDE, RuntimeReasonCode.SAFETY_FALLBACK),
        (
            RuntimeValueSource.COMPATIBILITY,
            RuntimeReasonCode.SAFETY_FALLBACK,
        ),
    ],
)
def test_constraint_source_and_reason_must_match(source, reason):
    with pytest.raises(RuntimeResolutionError, match="constraints must pair"):
        resolve_effective_runtime_config(
            defaults=_defaults(),
            constraints=(
                RuntimeConstraint(
                    RuntimeField.MAX_NUM_SEQS,
                    1,
                    source,
                    "constraint:test",
                    reason,
                ),
            ),
        )


def test_config_and_nested_trace_are_immutable():
    config = resolve_effective_runtime_config(defaults=_defaults())
    with pytest.raises(FrozenInstanceError):
        config.fields = ()
    with pytest.raises(FrozenInstanceError):
        config.fields[0].trace[0].value = 1


def test_effective_config_rejects_partial_or_reordered_construction():
    trace = RuntimeTraceEntry(
        1,
        RuntimeValueSource.GLOBAL_DEFAULT,
        "test",
        RuntimeReasonCode.GLOBAL_DEFAULT,
        RuntimeTraceAction.SELECTED,
    )
    field = EffectiveRuntimeValue(
        RuntimeField.MAX_NUM_SEQS,
        1,
        trace.source,
        trace.source_id,
        trace.reason_code,
        (trace,),
    )
    with pytest.raises(RuntimeResolutionError, match="canonical order"):
        EffectiveRuntimeConfig((field,))


def test_get_rejects_unknown_field():
    config = resolve_effective_runtime_config(defaults=_defaults())
    with pytest.raises(RuntimeResolutionError, match="unknown runtime field"):
        config.get("prefill_step_size")
