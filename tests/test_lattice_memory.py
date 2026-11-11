"""Universal memory policies preserve legacy arithmetic and freeze overrides."""

import json
import logging
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults, memory

LEGACY = {
    "metal_cache_floor_bytes": 2 * 1024**3,
    "metal_cache_cap_bytes": 32 * 1024**3,
    "metal_cache_divisor": 4,
    "memory_pressure_margin": 0.05,
    "memory_pressure_ceiling": 0.99,
    "memory_pressure_fallback_bytes": 200 * 1024**3,
    "memory_check_interval": 16,
}


@pytest.fixture(autouse=True)
def isolate_policy(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    memory.memory_policy.cache_clear()
    yield
    memory.memory_policy.cache_clear()


def legacy_cache_limit(soft):
    cache = max(2 * 1024**3, min(32 * 1024**3, soft // 4))
    return min(cache, soft) if soft > 0 else cache


def legacy_pressure_threshold(info, utilization):
    try:
        info = info()
        recommended = info.get(
            "max_recommended_working_set_size", info.get("memory_size", 0)
        )
        memory = recommended if recommended > 0 else 200 * 1024**3
        return int(memory * min(utilization + 0.05, 0.99))
    except Exception:
        return 200 * 1024**3


@pytest.mark.parametrize(
    "soft",
    [-1, 0, 1, 1024, 8 * 1024**3, 96 * 1024**3, 96 * 1024**3 + 3, 256 * 1024**3],
)
def test_default_cache_formula(soft):
    policy = memory.memory_policy()
    assert asdict(policy) == LEGACY
    assert policy.metal_cache_limit(soft) == legacy_cache_limit(soft)


@pytest.mark.parametrize("utilization", [0, 0.5, 0.9, 0.95, 1])
@pytest.mark.parametrize(
    "info",
    [
        {},
        {"memory_size": 0},
        {"memory_size": 8 * 1024**3},
        {
            "memory_size": 256 * 1024**3,
            "max_recommended_working_set_size": 96 * 1024**3,
        },
        {"memory_size": 256 * 1024**3, "max_recommended_working_set_size": 0},
        {"max_recommended_working_set_size": None},
    ],
)
def test_default_pressure_formula(info, utilization):
    probe = Mock(return_value=info)
    assert memory.memory_policy().pressure_threshold(
        utilization, probe
    ) == legacy_pressure_threshold(lambda: info, utilization)
    probe.assert_called_once_with()


@pytest.mark.parametrize(
    "hardware",
    [
        lattice.HardwareProfile(),
        lattice.HardwareProfile(gpu_generation=14, tier="base"),
        lattice.HardwareProfile(gpu_generation=99, tier="Ultra"),
    ],
)
def test_universal_cells(hardware):
    context = replace(lattice.runtime_context({}), hardware=hardware)
    for name, value in LEGACY.items():
        decision = lattice.resolve(name, context, environ={})
        assert decision.value == value
        assert decision.source.startswith("builtin:")
        cell = next(
            cell for cell in defaults.builtin_profile().cells if name in cell["knobs"]
        )
        assert "hardware" not in cell


@pytest.mark.parametrize("source", ["env", "overlay"])
def test_overrides(monkeypatch, tmp_path, source):
    knobs = dict(
        metal_cache_floor_bytes=1,
        metal_cache_cap_bytes=100,
        metal_cache_divisor=2,
        memory_pressure_margin=0.1,
        memory_pressure_ceiling=0.8,
        memory_pressure_fallback_bytes=1000,
        memory_check_interval=3,
    )
    if source == "env":
        monkeypatch.setenv("RAPID_MLX_BENCHMARK_KNOBS", json.dumps(knobs))
    else:
        path = tmp_path / "overlay.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile_version": "test",
                    "generated_at": "2026-11-11T00:00:00Z",
                    "cells": [
                        {
                            "id": "memory",
                            "knobs": knobs,
                            "provenance": {"evidence": "test override"},
                        }
                    ],
                }
            )
        )
        profile = defaults.load_profile(path, lattice.KNOBS)
        assert len(profile.cells) == 1
        monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)
    policy = memory.memory_policy()
    assert asdict(policy) == knobs
    assert policy.metal_cache_limit(99) == 49
    assert policy.pressure_threshold(0.5, lambda: {"memory_size": 100}) == 60
    assert policy.pressure_threshold(0.9, lambda: {}) == 800
    assert policy.pressure_threshold(0.5, Mock(side_effect=RuntimeError)) == 1000


def test_resolution_is_cached(monkeypatch):
    resolve = Mock(wraps=memory.resolve)
    context = Mock(wraps=memory.runtime_context)
    monkeypatch.setattr(memory, "resolve", resolve)
    monkeypatch.setattr(memory, "runtime_context", context)
    policy = memory.memory_policy()
    monkeypatch.setenv("RAPID_MLX_BENCHMARK_KNOBS", '{"memory_check_interval": 1}')
    for soft in range(100):
        assert memory.memory_policy() is policy
        policy.metal_cache_limit(soft)
        policy.pressure_threshold(0.5, lambda: {})
    assert resolve.call_count == len(LEGACY)
    assert {call.args[0] for call in resolve.call_args_list} == LEGACY.keys()
    context.assert_called_once_with({})
    assert policy.memory_check_interval == 16


@pytest.mark.parametrize("stage", ["runtime_context", "resolve"])
def test_resolution_failure_fallback(monkeypatch, caplog, stage):
    failure = Mock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr(memory, stage, failure)
    with caplog.at_level(logging.DEBUG, logger=memory.__name__):
        policy = memory.memory_policy()
        assert asdict(policy) == LEGACY
        for soft in (0, 1, 8 * 1024**3, 96 * 1024**3, 256 * 1024**3):
            assert memory.memory_policy().metal_cache_limit(soft) == legacy_cache_limit(
                soft
            )
    failure.assert_called_once()
    assert "using defaults" in caplog.text


def test_device_failure_fallback(caplog):
    probe = Mock(side_effect=RuntimeError("device unavailable"))
    with caplog.at_level(logging.DEBUG, logger=memory.__name__):
        assert memory.memory_policy().pressure_threshold(0.5, probe) == 200 * 1024**3
    probe.assert_called_once_with()
    assert "device fallback" in caplog.text


@pytest.mark.parametrize("name", LEGACY)
def test_schema_domains(name):
    knob = lattice.KNOBS[name]
    assert knob.valid(LEGACY[name])
    invalid = [True, False, None, "1", -1, float("nan"), float("inf")]
    if type(LEGACY[name]) is int:
        invalid += [0, 1.0]
        assert knob.valid(1) and knob.valid(10**100)
    else:
        invalid += [1.01]
        assert knob.valid(1) and knob.valid(0.5)
        assert knob.valid(0) == (name == "memory_pressure_margin")
    for value in invalid:
        assert not knob.valid(value), (name, value)
        context = lattice.runtime_context({})
        assert (
            lattice.resolve(
                name,
                context,
                environ={"RAPID_MLX_BENCHMARK_KNOBS": json.dumps({name: value})},
            ).value
            == LEGACY[name]
        )
