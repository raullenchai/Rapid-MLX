# SPDX-License-Identifier: Apache-2.0
"""Parity fixtures backed by the production CLI and compatibility resolvers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rapid_mlx.cli as cli
from rapid_mlx.kv_cache_dtype import (
    KVCacheQuantizationUnsupportedError,
    resolve_kv_cache_dtype,
)
from rapid_mlx.runtime.config_adapter import (
    DEFAULT_RUNTIME_LAUNCH_VALUES,
    RuntimeLaunchValues,
    resolve_with_legacy_parity,
)
from rapid_mlx.runtime.effective_config import (
    RuntimeConfigOverride,
    RuntimeConstraint,
    RuntimeField,
    RuntimeProfileValue,
    RuntimeReasonCode,
    RuntimeValueSource,
)


def _resolve(*, legacy, profiles=(), overrides=(), constraints=()):
    return resolve_with_legacy_parity(
        surface="legacy-fixture",
        legacy=legacy,
        defaults=DEFAULT_RUNTIME_LAUNCH_VALUES.as_defaults("runtime-defaults:v1"),
        profile_values=profiles,
        overrides=overrides,
        constraints=constraints,
    )


@pytest.mark.parametrize(
    ("model", "configured", "expected"),
    [
        ("qwen3.5-4b-4bit", 2048, 512),
        ("qwen3.5-4b-6bit", 2048, 1024),
        ("gemma-4-12b-4bit", 2048, 512),
    ],
)
def test_model_profile_prefill_matches_production_resolver(model, configured, expected):
    legacy_prefill = cli._resolve_prefill_step_size(
        model_name=model,
        configured=configured,
        user_set_explicit=False,
    )
    assert legacy_prefill == expected
    legacy = RuntimeLaunchValues(
        prefill_step_size=legacy_prefill,
        max_num_seqs=256,
        gpu_memory_utilization=None,
        enable_prefix_cache=True,
        kv_cache_dtype="bf16",
    )
    config = _resolve(
        legacy=legacy,
        profiles=(
            RuntimeProfileValue(
                RuntimeField.PREFILL_STEP_SIZE,
                expected,
                f"model-profile:{model}",
            ),
        ),
    )
    assert config.get(RuntimeField.PREFILL_STEP_SIZE).value == expected


def test_explicit_prefill_and_machine_memory_flags_match_legacy_values():
    legacy_prefill = cli._resolve_prefill_step_size(
        model_name="qwen3.5-4b-4bit",
        configured=1536,
        user_set_explicit=True,
    )
    legacy = RuntimeLaunchValues(legacy_prefill, 8, 0.75, False, "int8")
    config = _resolve(
        legacy=legacy,
        overrides=(
            RuntimeConfigOverride(RuntimeField.PREFILL_STEP_SIZE, 1536, "cli"),
            RuntimeConfigOverride(RuntimeField.MAX_NUM_SEQS, 8, "cli"),
            RuntimeConfigOverride(RuntimeField.GPU_MEMORY_UTILIZATION, 0.75, "cli"),
            RuntimeConfigOverride(RuntimeField.ENABLE_PREFIX_CACHE, False, "cli"),
            RuntimeConfigOverride(RuntimeField.KV_CACHE_DTYPE, "int8", "cli"),
        ),
    )
    assert RuntimeLaunchValues.from_effective(config) == legacy


def test_reasoning_workload_adjustment_matches_production_resolver():
    decision = resolve_kv_cache_dtype("int4", reasoning=True, explicit=True)
    assert decision.dtype == "int8"
    legacy = RuntimeLaunchValues(2048, 256, None, True, decision.dtype)
    config = _resolve(
        legacy=legacy,
        overrides=(RuntimeConfigOverride(RuntimeField.KV_CACHE_DTYPE, "int4", "cli"),),
        constraints=(
            RuntimeConstraint(
                RuntimeField.KV_CACHE_DTYPE,
                decision.dtype,
                RuntimeValueSource.SAFETY,
                "workload:reasoning-quality-floor",
                RuntimeReasonCode.SAFETY_FALLBACK,
            ),
        ),
    )
    resolved = config.get(RuntimeField.KV_CACHE_DTYPE)
    assert resolved.value == decision.dtype
    assert resolved.source is RuntimeValueSource.SAFETY


def test_automatic_incompatible_kv_fallback_matches_production_resolver():
    hf_config = {"model_type": "gemma4", "sliding_window": 1024}
    decision = resolve_kv_cache_dtype(
        "int4",
        explicit=False,
        model_name="gemma-4-12b",
        hf_config=hf_config,
    )
    assert decision.dtype == "bf16"
    config = _resolve(
        legacy=DEFAULT_RUNTIME_LAUNCH_VALUES,
        overrides=(
            RuntimeConfigOverride(
                RuntimeField.KV_CACHE_DTYPE, "int4", "profile:auto-kv"
            ),
        ),
        constraints=(
            RuntimeConstraint(
                RuntimeField.KV_CACHE_DTYPE,
                decision.dtype,
                RuntimeValueSource.COMPATIBILITY,
                "compat:gemma4-sliding-window",
                RuntimeReasonCode.COMPATIBILITY_FALLBACK,
            ),
        ),
    )
    assert config.get(RuntimeField.KV_CACHE_DTYPE).source_id.startswith("compat:")


def test_explicit_incompatible_kv_request_still_fails_closed():
    with pytest.raises(KVCacheQuantizationUnsupportedError):
        resolve_kv_cache_dtype(
            "int4",
            explicit=True,
            model_name="gemma-4-12b",
            hf_config={"model_type": "gemma4", "sliding_window": 1024},
        )


def _cli_args(**overrides):
    values = {
        "model": "example/model",
        "_original_alias": "example-alias",
        "max_num_seqs": 256,
        "gpu_memory_utilization": None,
        "reasoning": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_effective_config_keeps_untouched_defaults():
    config = cli._resolve_cli_effective_runtime_config(
        args=_cli_args(),
        prefill_step_size=2048,
        prefill_user_set_explicit=False,
        enable_prefix_cache=True,
        kv_cache_decision=None,
        kv_quant_explicit=False,
        argv=[],
    )

    assert RuntimeLaunchValues.from_effective(config) == DEFAULT_RUNTIME_LAUNCH_VALUES
    assert all(
        field.source is RuntimeValueSource.GLOBAL_DEFAULT for field in config.fields
    )


def test_cli_effective_config_records_auto_profile_values():
    config = cli._resolve_cli_effective_runtime_config(
        args=_cli_args(max_num_seqs=16, gpu_memory_utilization=0.5),
        prefill_step_size=1024,
        prefill_user_set_explicit=False,
        enable_prefix_cache=False,
        kv_cache_decision=SimpleNamespace(requested="int8", dtype="int8"),
        kv_quant_explicit=False,
        argv=[],
    )

    assert RuntimeLaunchValues.from_effective(config) == RuntimeLaunchValues(
        1024, 16, 0.5, False, "int8"
    )
    assert all(
        config.get(field).source is RuntimeValueSource.PERFORMANCE_PROFILE
        for field in RuntimeField
    )


@pytest.mark.parametrize(
    ("reasoning", "expected_source"),
    [
        (False, RuntimeValueSource.COMPATIBILITY),
        (True, RuntimeValueSource.SAFETY),
    ],
)
def test_cli_effective_config_records_flags_and_kv_fallback(reasoning, expected_source):
    config = cli._resolve_cli_effective_runtime_config(
        args=_cli_args(
            max_num_seqs=8,
            gpu_memory_utilization=0.75,
            reasoning=reasoning,
        ),
        prefill_step_size=1536,
        prefill_user_set_explicit=True,
        enable_prefix_cache=False,
        kv_cache_decision=SimpleNamespace(requested="int4", dtype="int8"),
        kv_quant_explicit=True,
        argv=[
            "--max-num-seqs=8",
            "--gpu-memory-utilization",
            "--disable-prefix-cache",
        ],
    )

    assert RuntimeLaunchValues.from_effective(config) == RuntimeLaunchValues(
        1536, 8, 0.75, False, "int8"
    )
    assert config.get(RuntimeField.PREFILL_STEP_SIZE).source is (
        RuntimeValueSource.USER_OVERRIDE
    )
    assert config.get(RuntimeField.KV_CACHE_DTYPE).source is expected_source
