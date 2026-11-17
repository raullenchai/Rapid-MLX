# SPDX-License-Identifier: Apache-2.0
"""Scheduling knob domains, profile validation and override precedence."""

import json
from pathlib import Path

import jsonschema
import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults


@pytest.mark.parametrize(
    "name,default,candidates,valid,invalid",
    [
        (
            "prefix_boundary_min_tokens",
            0,
            (0, 128, 256, 512),
            [0, 1, 128, 256, 512],
            [-1, True, 128.0, "128"],
        ),
        (
            "admission_window_ms",
            8.0,
            (0, 8),
            [0, 8, 8.0, 0.5],
            [-1, True, "8", float("nan"), float("inf")],
        ),
    ],
)
def test_scheduling_knob_contract(
    monkeypatch, tmp_path, name, default, candidates, valid, invalid
):
    ctx = lattice.runtime_context({})
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    knob = lattice.KNOBS[name]
    assert knob.default == default
    assert knob.candidates == candidates
    assert lattice.resolve(name, ctx, environ={}) == lattice.Decision(
        default, "default"
    )
    paths = [
        Path("proto/model-runtime/v1/lattice-profile.schema.json"),
        Path("rapid_mlx/catalog/schemas/lattice-profile.schema.json"),
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    schema = json.loads(paths[0].read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for value in valid + invalid:
        accepted = any(type(value) is type(v) and value == v for v in valid)
        assert knob.valid(value) == accepted
        data = {
            "schema_version": 1,
            "profile_version": "test.1",
            "generated_at": "2026-11-17T00:00:00Z",
            "cells": [
                {
                    "id": "scheduling",
                    "knobs": {name: value},
                    "provenance": {"evidence": "test only"},
                }
            ],
        }
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(data))
        profile = defaults.load_profile(path, lattice.KNOBS)
        assert bool(profile.cells) == accepted
        # JSON Schema permits integral floats; runtime deliberately rejects them
        # for integer knobs. JSON itself does not represent NaN or infinity.
        if type(value) is not float or value in (0.5, 8.0):
            assert validator.is_valid(data) == accepted
        decision = lattice.resolve(
            name, ctx, environ={"RAPID_MLX_BENCHMARK_KNOBS": json.dumps({name: value})}
        )
        assert decision.value == (value if accepted else default)


@pytest.mark.parametrize(
    "name,value", [("prefix_boundary_min_tokens", 256), ("admission_window_ms", 0.5)]
)
def test_scheduling_env_and_overlay(monkeypatch, name, value):
    ctx = lattice.runtime_context({})
    profile = defaults.Profile("test.1", ({"id": "test", "knobs": {name: value}},))
    monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)
    knob = lattice.KNOBS[name]
    assert lattice.resolve(name, ctx, environ={}).value == value
    assert lattice.resolve(name, ctx, environ={knob.env: "0"}) == lattice.Decision(
        0, "env"
    )
    for bad in ("bad", "nan", "inf", "-1"):
        assert lattice.resolve(name, ctx, environ={knob.env: bad}).value == value
    assert (
        lattice.resolve(
            name,
            ctx,
            environ={
                knob.env: "0",
                "RAPID_MLX_BENCHMARK_KNOBS": json.dumps({name: value}),
            },
        ).source
        == "benchmark"
    )


def measured_context():
    return lattice.runtime_context(
        {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
        config={
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
        library_version="0.32.4",
        macos="27.0.1",
        mode="mtp",
    )


@pytest.fixture
def isolated_policy(monkeypatch):
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    for name in ("prefix_boundary_min_tokens", "admission_window_ms"):
        monkeypatch.delenv(lattice.KNOBS[name].env, raising=False)
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)


def test_short_request_builtin_measured_cell(isolated_policy):
    ctx = measured_context()
    assert ctx.model.intermediate_size is None
    for name, value in (
        ("prefix_boundary_min_tokens", 256),
        ("admission_window_ms", 0),
    ):
        decision = lattice.resolve(name, ctx, environ={})
        assert (decision.value, decision.source) == (
            value,
            "builtin:short-request-17-max-36g-qwen38-27b",
        )
        knob = lattice.KNOBS[name]
        assert (
            lattice.resolve(name, ctx, environ={knob.env: str(knob.default)}).value
            == knob.default
        )


@pytest.mark.parametrize(
    "section,changes",
    [
        *(("hardware", {"gpu_generation": g}) for g in [None, 13, 14, 15, 16, 18]),
        *(("hardware", {"tier": tier}) for tier in ["base", "Pro", "Ultra", "unknown"]),
        *(
            ("hardware", {"memory_size": memory})
            for memory in [None, 24 * 2**30, 48 * 2**30]
        ),
        ("model", {"family": "qwen3_5_moe_text"}),
        ("model", {"family": "unknown"}),
        ("model", {"hidden_size": 4096}),
        ("model", {"bits": 8}),
        ("model", {"bits": None}),
        ("model", {"group_size": 128}),
        ("model", {"quant_mode": "mxfp4"}),
        (None, {"mlx": "0.32.3"}),
        (None, {"macos": "26.5.2"}),
        *((None, {"mode": mode}) for mode in ["plain", "compiled", "unknown"]),
        *((None, {"batch": batch}) for batch in [2, 8]),
        (None, {"width": 3}),
    ],
)
def test_short_request_builtin_excludes_other_facts(isolated_policy, section, changes):
    from dataclasses import replace

    ctx = measured_context()
    if section:
        changes = {section: replace(getattr(ctx, section), **changes)}
    ctx = replace(ctx, **changes)
    for name in ("prefix_boundary_min_tokens", "admission_window_ms"):
        decision = lattice.resolve(name, ctx, environ={})
        assert decision.value == lattice.KNOBS[name].default
        assert decision.source in {"default", "capability"}


@pytest.mark.parametrize(
    "formats,mode,batch,expected",
    [
        ([(4, 64, "affine")], "mtp", 1, (256, 0)),
        ([(4, 64, "affine")], "plain", 1, (0, 8)),
        ([(4, 64, "affine")], "mtp", 2, (0, 8)),
        ([(4, 64, "affine"), (8, 64, "affine")], "mtp", 1, (0, 8)),
        ([(4, 128, "affine")], "mtp", 1, (0, 8)),
        ([(4, 64, "mxfp4")], "mtp", 1, (0, 8)),
        ([], "mtp", 1, (0, 8)),
    ],
)
def test_short_request_live_model_adapter(
    monkeypatch, isolated_policy, formats, mode, batch, expected
):
    import sys
    from types import ModuleType, SimpleNamespace

    from rapid_mlx.lattice.scheduling import scheduling_value, skip_prefix_boundary

    mx = ModuleType("mlx.core")
    mx.device_info = lambda: {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30}
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr("rapid_mlx.lattice.profiles.mlx_version", lambda: "0.32.4")
    monkeypatch.setattr("rapid_mlx.lattice.profiles.macos_version", lambda: "27.0.1")
    modules = [
        (str(i), SimpleNamespace(weight=object(), bits=b, group_size=g, mode=m))
        for i, (b, g, m) in enumerate(formats)
    ]
    modules.append(("norm", SimpleNamespace(weight=object())))
    target = SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120),
        named_modules=lambda: modules,
    )
    model = SimpleNamespace(language_model=target)
    assert (
        tuple(
            scheduling_value(name, model, mode=mode, batch=batch)
            for name in ("prefix_boundary_min_tokens", "admission_window_ms")
        )
        == expected
    )
    if batch == 1:
        assert skip_prefix_boundary(model, 255, mode=mode) == (expected[0] == 256)
        assert not skip_prefix_boundary(target, 256, mode=mode)
