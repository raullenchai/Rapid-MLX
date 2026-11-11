"""Representation extensions preserve existing profile decisions."""

import hashlib
import itertools
import json
from dataclasses import asdict

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults


def builtin_contexts():
    """Current hardware/software/model/workload grid, including missing facts."""
    for (
        generation,
        tier,
        memory,
        mlx,
        macos,
        model,
        mode,
        width,
        batch,
    ) in itertools.product(
        (None, 14, 15, 16, 17, 18),
        ("unknown", "base", "Pro", "Max", "Ultra"),
        (None, 36 * 2**30, 48 * 2**30, 256 * 2**30),
        ("unknown", "0.32.2", "0.32.3", "0.32.4"),
        ("unknown", "26.5.2", "27.0.1"),
        (
            lattice.ModelProfile(),
            lattice.ModelProfile("qwen35_moe", *lattice.QWEN35_AFFINE4),
            lattice.ModelProfile(
                "qwen3_5_moe_text", *lattice.QWEN35_AFFINE4, quant_mode="affine"
            ),
        ),
        ("plain", "mtp"),
        (1, 3, 9, 64),
        (1, 2),
    ):
        yield lattice.LatticeContext(
            lattice.HardwareProfile(
                gpu_generation=generation, tier=tier, memory_size=memory
            ),
            mlx,
            macos,
            model,
            mode,
            width,
            batch,
            has_mtp_head=True,
        )


def test_builtin_resolution_snapshot(monkeypatch):
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    digest = hashlib.sha256()
    selected = set()
    count = 0
    for ctx in builtin_contexts():
        decisions = {}
        for name in lattice.KNOBS:
            decision = lattice.resolve(name, ctx, environ={})
            decisions[name] = asdict(decision)
            if decision.source.startswith("builtin:"):
                selected.add(decision.source.removeprefix("builtin:"))
        digest.update(json.dumps(decisions, sort_keys=True).encode())
        count += 1
    assert selected == {cell["id"] for cell in defaults.builtin_profile().cells}
    assert count == 69120
    # Recorded before changing representation on 7b25e1b4c8dcf80cadc7021bff69a48f76db5aca.
    assert (
        digest.hexdigest()
        == "fd5f021b90cf747db8e2f9be32a87ba3d1a756066b97af0813c8eb0f19ab0e11"
    )


def legacy_valid(default, value):
    """Validator from the pre-extension registry, used only as an oracle."""
    import sys

    if default == ():
        return (
            isinstance(value, (tuple, list))
            and 3 <= len(value) <= 6
            and all(
                type(v) in (int, float) and abs(v) <= sys.float_info.max for v in value
            )
            and all(v > 0 for v in value[:-1])
            and value[-1] >= 0
        )
    if type(default) is int:
        if default == 0:
            return type(value) is int and value >= 0
        return type(value) is int and 1 <= value <= 4
    if type(default) is bool:
        return type(value) is bool
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(type(v) is int for v in value)
        and tuple(value) in ((1, 1), (2, 4))
    )


def test_existing_validator_domains():
    import sys

    values = [
        None,
        False,
        True,
        "1",
        {},
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        10**100,
        0.0,
        1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        (),
        [],
    ]
    for length in range(1, 8):
        for scalar in (
            -1,
            0,
            1,
            1.5,
            True,
            "1",
            None,
            float("nan"),
            float("inf"),
            sys.float_info.max,
        ):
            values.extend(([scalar] * length, (scalar,) * length))
    values.extend(([1, 1], [2, 4], [True, True], [1.0, 1.0], [1, 2, 0], [1, 2, -1]))

    class TupleValue(tuple):
        pass

    class ListValue(list):
        pass

    values.extend((TupleValue((1, 1)), ListValue([2, 4])))
    for name, knob in lattice.KNOBS.items():
        for value in values:
            assert knob.valid(value) == legacy_valid(knob.default, value), (name, value)


def test_validators_do_not_depend_on_default():
    from dataclasses import replace

    knob = lattice.Knob(
        lambda c: True, lambda c: None, lattice.int_range(0, 64), default=False
    )
    assert knob.valid(64) and not knob.valid(True) and not knob.valid(65)
    assert replace(knob, default="changed").valid(64)
    mode = lattice.Knob(
        lambda c: True, lambda c: None, lattice.enum_values("off", "on")
    )
    assert mode.valid("off") and not mode.valid(False)


def representation_cell():
    return {
        "id": "universal",
        "model": {"geometry": {"hidden_size": 2048, "num_experts": 256, "top_k": 8}},
        "workload": {"flattened_rows": [0, 64]},
        "knobs": {"lane_simd_probe": False},
        "provenance": {"evidence": "representation test; no gate enrollment"},
    }


def profile_document(cells):
    return {
        "schema_version": 1,
        "profile_version": "test",
        "generated_at": "2026-11-11T00:00:00Z",
        "cells": cells,
    }


def load_cells(tmp_path, cells):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile_document(cells)))
    return defaults.load_profile(path, lattice.KNOBS)


def test_universal_partial_geometry_and_rows(tmp_path):
    from dataclasses import replace

    cell = load_cells(tmp_path, [representation_cell()]).cells[0]
    ctx = lattice.runtime_context(
        {},
        model=lattice.ModelProfile(hidden_size=2048, num_experts=256, top_k=8),
        flattened_rows=64,
    )
    for hardware in (
        lattice.HardwareProfile(),
        lattice.HardwareProfile(gpu_generation=99, tier="Ultra"),
        lattice.HardwareProfile(architecture="unclassified"),
    ):
        for rows, expected in (
            (None, False),
            (0, True),
            (1, True),
            (64, True),
            (65, False),
        ):
            assert (
                defaults.matches(
                    cell, replace(ctx, hardware=hardware, flattened_rows=rows)
                )
                is expected
            )
    for fact in ("hidden_size", "num_experts", "top_k"):
        for value in (
            None,
            0,
            getattr(ctx.model, fact) - 1,
            getattr(ctx.model, fact) + 1,
        ):
            assert not defaults.matches(
                cell, replace(ctx, model=replace(ctx.model, **{fact: value}))
            )
    for fact in ("intermediate_size", "bits", "group_size"):
        for value in (None, 1, 99):
            assert defaults.matches(
                cell, replace(ctx, model=replace(ctx.model, **{fact: value}))
            )
    assert ctx.flattened_rows == 64 and ctx.width == 1
    assert lattice.runtime_context({}, flattened_rows=1) != lattice.runtime_context(
        {}, flattened_rows=64
    )


def test_each_geometry_fact_can_be_selected_alone(tmp_path):
    from dataclasses import replace

    for fact in (
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "bits",
        "group_size",
    ):
        raw = representation_cell()
        raw.pop("workload")
        raw["model"]["geometry"] = {fact: 8}
        cell = load_cells(tmp_path, [raw]).cells[0]
        ctx = lattice.LatticeContext(
            lattice.HardwareProfile(), "unknown", "unknown", lattice.ModelProfile()
        )
        assert not defaults.matches(cell, ctx)
        assert defaults.matches(
            cell, replace(ctx, model=replace(ctx.model, **{fact: 8}))
        )
    raw["model"]["geometry"] = {}
    assert defaults.matches(load_cells(tmp_path, [raw]).cells[0], ctx)


def test_universal_specific_conflict(tmp_path, monkeypatch):
    from dataclasses import replace

    universal = representation_cell()
    universal.pop("model")
    universal.pop("workload")
    specific = dict(
        universal,
        id="specific",
        hardware=[{"gpu_generation": 14, "tier": "base"}],
        knobs={"lane_simd_probe": True},
    )
    ctx = lattice.LatticeContext(
        lattice.HardwareProfile(gpu_generation=14, tier="base"), "unknown", "unknown"
    )
    for layer in ("builtin_profile", "overlay_profile"):
        for cells in ([universal, specific], [specific, universal]):
            with monkeypatch.context() as patch:
                patch.setattr(
                    lattice, layer, lambda cells=cells: load_cells(tmp_path, cells)
                )
                decision = lattice.resolve("lane_simd_probe", ctx, environ={})
                assert decision == lattice.Decision(
                    True, "default", "conflicting lattice profile cells", "test"
                )
                assert (
                    lattice.resolve(
                        "lane_simd_probe",
                        replace(ctx, hardware=lattice.HardwareProfile()),
                        environ={},
                    ).value
                    is False
                )


def profile_validator():
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        Path("proto/model-runtime/v1/lattice-profile.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )


def test_profile_schema_builtin_and_extensions():
    from pathlib import Path

    validator = profile_validator()
    validator.validate(
        json.loads(Path("rapid_mlx/lattice/profiles/builtin.json").read_text())
    )
    validator.validate(profile_document([representation_cell()]))
    assert (
        Path("proto/model-runtime/v1/lattice-profile.schema.json").read_bytes()
        == Path("rapid_mlx/catalog/schemas/lattice-profile.schema.json").read_bytes()
    )


def test_malformed_extensions_rejected(tmp_path):
    import copy

    validator = profile_validator()
    cases = (
        [("hardware", value) for value in (None, [], {}, "any", [{}])]
        + [
            ("model", {"geometry": value})
            for value in (
                None,
                [2048],
                {"hidden_siz": 2048},
                {"hidden_size": True},
                {"bits": 0},
                {"top_k": None},
                {"num_experts": "256"},
            )
        ]
        + [
            ("workload", {"flattened_rows": value})
            for value in (
                64,
                [],
                [0],
                [0, 64, 65],
                [-1, 64],
                [False, 64],
                [0, "64"],
                [0, None],
            )
        ]
    )
    for key, value in cases:
        bad = copy.deepcopy(representation_cell())
        bad[key] = value
        assert list(validator.iter_errors(profile_document([bad]))), (key, value)
        assert not load_cells(tmp_path, [bad]).cells, (key, value)
    bad = representation_cell()
    bad["workload"]["flattened_rows"] = [65, 64]
    assert not load_cells(tmp_path, [bad]).cells


def test_benchmark_context_schema_accepts_optional_row_fact(monkeypatch):
    from pathlib import Path

    import jsonschema

    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    schema = json.loads(
        Path("proto/community-benchmark/v1/benchmark-run.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(
        {"$ref": "#/$defs/lattice/properties/context", "$defs": schema["$defs"]}
    )
    context = lattice.dump(
        lattice.runtime_context({}, library_version="unknown", macos="unknown")
    )
    context = json.loads(json.dumps(context))
    for rows in (None, 0, 64):
        context["flattened_rows"] = rows
        validator.validate(context)
    context.pop("flattened_rows")
    validator.validate(context)
    for rows in (-1, True, "64"):
        context["flattened_rows"] = rows
        assert list(validator.iter_errors(context))
