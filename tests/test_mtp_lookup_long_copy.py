# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for qualified lookup copy ceilings."""

import json
from pathlib import Path

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults
from rapid_mlx.lattice.mtp import lookup_max_copy
from rapid_mlx.spec_decode.mtp.prompt_lookup import (
    CopyDraftGate,
    PromptLookupIndex,
)


@pytest.mark.parametrize("ceiling,earned", [(31, 31), (63, 63)])
def test_copy_ceiling_and_existing_gate_growth(ceiling, earned):
    index = PromptLookupIndex(list(range(100)), min_ngram=2)
    gate = CopyDraftGate()
    widths = []
    for _ in range(20):
        match = index.propose([0, 1], max_tokens=ceiling)
        assert len(match.tokens) == ceiling
        width = gate.width_cap(len(match.tokens))
        widths.append(width)
        gate.observe(
            is_copy_draft=True,
            committed=width + 1,
            round_ms=1,
            accepted=width,
            proposed=width,
        )
    assert widths[:3] == [2, 2, 2]
    assert max(widths) == earned <= ceiling


@pytest.fixture
def lookup_context(monkeypatch):
    ctx = lattice.runtime_context({}, mode="mtp", has_mtp_head=True)
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.setattr("rapid_mlx.lattice.mtp._model_context", lambda _: ctx)
    for key in (
        "RAPID_MLX_BENCHMARK_KNOBS",
        "RAPID_MLX_MTP_LOOKUP_MAX_COPY",
    ):
        monkeypatch.delenv(key, raising=False)
    return ctx


def test_lookup_defaults_preserve_family_policy(lookup_context):
    assert lookup_max_copy(object(), 31) == 31
    assert lookup_max_copy(object(), 24) == 24
    assert lattice.resolve("mtp_lookup_max_copy", lookup_context).value == 31


@pytest.mark.parametrize("ceiling", [31, 63])
def test_lookup_env_and_benchmark_precedence(monkeypatch, lookup_context, ceiling):
    monkeypatch.setenv("RAPID_MLX_MTP_LOOKUP_MAX_COPY", str(ceiling))
    assert lookup_max_copy(object(), 24) == (24 if ceiling == 31 else ceiling)
    monkeypatch.setenv(
        "RAPID_MLX_BENCHMARK_KNOBS",
        json.dumps({"mtp_lookup_max_copy": 63}),
    )
    assert lookup_max_copy(object(), 24) == 63


@pytest.mark.parametrize("value", [True, 0, 32, 47, 48, 64, "63", 63.0])
def test_invalid_ceiling_rejected(lookup_context, value):
    assert not lattice.KNOBS["mtp_lookup_max_copy"].valid(value)
    decision = lattice.resolve(
        "mtp_lookup_max_copy",
        lookup_context,
        environ={
            "RAPID_MLX_BENCHMARK_KNOBS": json.dumps({"mtp_lookup_max_copy": value})
        },
    )
    assert decision == lattice.Decision(31, "default")


@pytest.mark.parametrize(
    "change", [{"has_mtp_head": False}, {"mode": "plain"}, {"batch": 2}]
)
def test_lookup_scope_cannot_be_overridden(change):
    ctx = lattice.runtime_context({}, mode="mtp", has_mtp_head=True)
    from dataclasses import replace

    ctx = replace(ctx, **change)
    name = "mtp_lookup_max_copy"
    decision = lattice.resolve(
        name, ctx, environ={"RAPID_MLX_BENCHMARK_KNOBS": json.dumps({name: 63})}
    )
    assert decision.source == "capability"
    assert decision.value == lattice.KNOBS[name].default


def test_lookup_schemas_match_runtime_domains():
    paths = [
        Path("proto/model-runtime/v1/lattice-profile.schema.json"),
        Path("rapid_mlx/catalog/schemas/lattice-profile.schema.json"),
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    properties = json.loads(paths[0].read_text())["$defs"]["cell"]["properties"][
        "knobs"
    ]["properties"]
    assert properties["mtp_lookup_max_copy"] == {"enum": [31, 63]}


def test_lookup_overlay_and_invalid_env(monkeypatch, lookup_context, tmp_path):
    cell = {
        "id": "test-lookup",
        "knobs": {"mtp_lookup_max_copy": 63},
        "provenance": {"evidence": "test only"},
    }
    data = {
        "schema_version": 1,
        "profile_version": "test.1",
        "generated_at": "2026-11-17T00:00:00Z",
        "cells": [cell],
    }
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(data))
    profile = defaults.load_profile(path, lattice.KNOBS)
    assert profile.cells
    monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)
    assert lookup_max_copy(object(), 31) == 63
    monkeypatch.setenv("RAPID_MLX_MTP_LOOKUP_MAX_COPY", "64")
    assert lookup_max_copy(object(), 31) == 63
    monkeypatch.setenv("RAPID_MLX_MTP_LOOKUP_MAX_COPY", "31")
    assert lookup_max_copy(object(), 31) == 31
    # Runtime and JSON schema validate the same actual overlay document.
    import jsonschema

    schema = json.loads(
        Path("proto/model-runtime/v1/lattice-profile.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(data)
    cell["knobs"]["mtp_lookup_max_copy"] = 64
    assert list(validator.iter_errors(data))
    path.write_text(json.dumps(data))
    assert not defaults.load_profile(path, lattice.KNOBS).cells


def measured_context():
    return lattice.runtime_context(
        {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
        config={
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "quantization": {"bits": 4, "group_size": 64},
        },
        library_version="0.32.4",
        macos="27.0.1",
        mode="mtp",
        has_mtp_head=True,
    )


def test_lookup_builtin_measured_cell(monkeypatch):
    ctx = measured_context()
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    decision = lattice.resolve("mtp_lookup_max_copy", ctx, environ={})
    assert (decision.value, decision.source) == (
        63,
        "builtin:mtp-lookup-17-max-36g-qwen38-27b",
    )
    # Explicit rollback to the unchanged default wins over enrollment.
    assert (
        lattice.resolve(
            "mtp_lookup_max_copy", ctx, environ={"RAPID_MLX_MTP_LOOKUP_MAX_COPY": "31"}
        ).value
        == 31
    )


@pytest.mark.parametrize(
    "section,changes",
    [
        *(("hardware", {"gpu_generation": g}) for g in [None, 13, 14, 15, 16, 18]),
        ("hardware", {"tier": "Pro"}),
        ("hardware", {"tier": "Ultra"}),
        ("hardware", {"memory_size": 48 * 2**30}),
        ("hardware", {"memory_size": None}),
        ("model", {"family": "qwen3_5_moe_text"}),
        ("model", {"family": "unknown"}),
        ("model", {"hidden_size": 4096}),
        ("model", {"bits": 8}),
        ("model", {"bits": None}),
        ("model", {"group_size": 128}),
        ("model", {"quant_mode": "mxfp4"}),
        (None, {"mlx": "0.32.3"}),
        (None, {"macos": "26.5.2"}),
        (None, {"mode": "plain"}),
        (None, {"batch": 2}),
        (None, {"width": 3}),
        (None, {"has_mtp_head": False}),
    ],
)
def test_lookup_builtin_other_facts_retain_31(monkeypatch, section, changes):
    from dataclasses import replace

    ctx = measured_context()
    if section:
        changes = {section: replace(getattr(ctx, section), **changes)}
    ctx = replace(ctx, **changes)
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    assert lattice.resolve("mtp_lookup_max_copy", ctx, environ={}).value == 31


@pytest.mark.parametrize(
    "formats,expected",
    [
        ([(4, 64, "affine")], 63),
        ([(4, 64, "affine"), (8, 64, "affine")], 31),
        ([(4, 128, "affine")], 31),
        ([(4, 64, "mxfp4")], 31),
        ([], 31),
    ],
)
def test_lookup_adapter_uses_installed_dense_formats(
    monkeypatch, lookup_context, formats, expected
):
    from types import SimpleNamespace

    # Even declared affine4 facts cannot qualify mixed or unavailable weights.
    monkeypatch.setattr(
        "rapid_mlx.lattice.mtp._model_context", lambda _: measured_context()
    )
    modules = [
        (str(i), SimpleNamespace(weight=object(), bits=b, group_size=g, mode=m))
        for i, (b, g, m) in enumerate(formats)
    ]
    modules.append(("norm", SimpleNamespace(weight=object())))
    model = SimpleNamespace(named_modules=lambda: modules)
    assert lookup_max_copy(SimpleNamespace(language_model=model), 31) == expected


def test_lookup_adapter_without_installed_head_preserves_policy(monkeypatch):
    monkeypatch.setattr("rapid_mlx.lattice.mtp._model_context", lambda _: None)
    assert lookup_max_copy(object(), 31) == 31
    assert lookup_max_copy(object(), 24) == 24


def test_lookup_measured_cell_through_installed_model_adapter(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    # Explicit inert modules keep this CPU contract independent of native MLX.
    mx = ModuleType("mlx.core")
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    from rapid_mlx.lattice import defaults
    from rapid_mlx.lattice.mtp import lookup_max_copy

    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"device_name": "Apple M5 Max", "memory_size": 36 * 2**30},
        raising=False,
    )
    monkeypatch.setattr("rapid_mlx.lattice.profiles.mlx_version", lambda: "0.32.4")
    monkeypatch.setattr("rapid_mlx.lattice.profiles.macos_version", lambda: "27.0.1")
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.setattr(lattice, "local_calibration", lambda _: {})
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    monkeypatch.delenv("RAPID_MLX_MTP_LOOKUP_MAX_COPY", raising=False)
    # Real installed text args omit quantization; the module carries it.
    weight = SimpleNamespace(weight=object(), bits=4, group_size=64, mode="affine")
    target = SimpleNamespace(
        mtp=object(),
        mtp_forward=lambda: None,
        make_mtp_cache=lambda: [],
        # Live text args on hardware expose no intermediate size (profile 0).
        args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120),
        named_modules=lambda: [("model.layers.0.mlp.up_proj", weight)],
    )
    assert lookup_max_copy(SimpleNamespace(language_model=target), 31) == 63
    weight.bits = 8
    assert lookup_max_copy(target, 31) == 31
