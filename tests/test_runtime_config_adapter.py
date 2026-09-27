# SPDX-License-Identifier: Apache-2.0
"""Parity and rollback contract for runtime resolver migration stage 2."""

from __future__ import annotations

import pytest

from rapid_mlx.runtime.config_adapter import (
    DEFAULT_RUNTIME_LAUNCH_VALUES,
    RuntimeLaunchValues,
    RuntimeParityError,
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

BASELINE = DEFAULT_RUNTIME_LAUNCH_VALUES


@pytest.mark.parametrize(
    ("surface", "legacy", "profiles", "overrides", "constraints"),
    [
        ("cli", BASELINE, (), (), ()),
        ("server", BASELINE, (), (), ()),
        (
            "cli:qwen3.5-4b-4bit:m3-16:long-context",
            RuntimeLaunchValues(512, 256, None, True, "bf16"),
            (
                RuntimeProfileValue(
                    RuntimeField.PREFILL_STEP_SIZE,
                    512,
                    "profile:qwen3.5-4b-4bit",
                ),
            ),
            (),
            (),
        ),
        (
            "server:explicit-flags",
            RuntimeLaunchValues(1024, 8, 0.75, False, "int8"),
            (),
            (
                RuntimeConfigOverride(RuntimeField.PREFILL_STEP_SIZE, 1024, "cli"),
                RuntimeConfigOverride(RuntimeField.MAX_NUM_SEQS, 8, "cli"),
                RuntimeConfigOverride(RuntimeField.GPU_MEMORY_UTILIZATION, 0.75, "cli"),
                RuntimeConfigOverride(RuntimeField.ENABLE_PREFIX_CACHE, False, "cli"),
                RuntimeConfigOverride(RuntimeField.KV_CACHE_DTYPE, "int8", "cli"),
            ),
            (),
        ),
        (
            "cli:gemma4:incompatible-kv",
            BASELINE,
            (),
            (
                RuntimeConfigOverride(
                    RuntimeField.KV_CACHE_DTYPE, "int4", "profile:auto-kv"
                ),
            ),
            (
                RuntimeConstraint(
                    RuntimeField.KV_CACHE_DTYPE,
                    "bf16",
                    RuntimeValueSource.COMPATIBILITY,
                    "compat:gemma4-sliding-window",
                    RuntimeReasonCode.COMPATIBILITY_FALLBACK,
                ),
            ),
        ),
    ],
)
def test_representative_legacy_and_central_results_match(
    surface, legacy, profiles, overrides, constraints
):
    config = resolve_with_legacy_parity(
        surface=surface,
        legacy=legacy,
        defaults=BASELINE.as_defaults("runtime-defaults:v1"),
        profile_values=profiles,
        overrides=overrides,
        constraints=constraints,
    )
    assert RuntimeLaunchValues.from_effective(config) == legacy


def test_parity_mismatch_fails_closed_with_field_evidence():
    with pytest.raises(RuntimeParityError) as caught:
        resolve_with_legacy_parity(
            surface="cli",
            legacy=RuntimeLaunchValues(1024, 8, None, False, "bf16"),
            defaults=BASELINE.as_defaults("runtime-defaults:v1"),
        )
    message = str(caught.value)
    assert "cli runtime parity mismatch" in message
    assert "prefill_step_size: legacy=1024, central=2048" in message
    assert "max_num_seqs: legacy=8, central=256" in message
    assert "enable_prefix_cache: legacy=False, central=True" in message


def test_surface_id_is_required():
    with pytest.raises(RuntimeParityError, match="surface must not be empty"):
        resolve_with_legacy_parity(
            surface=" ",
            legacy=BASELINE,
            defaults=BASELINE.as_defaults("runtime-defaults:v1"),
        )


@pytest.mark.parametrize("field", list(RuntimeField))
def test_launch_values_have_a_typed_value_for_every_contract_field(field):
    assert BASELINE.value_for(field) is not ...
