"""Data profiles select registered values only, within explicitly declared cells."""

import json
import logging
import tomllib
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults

NAME = "mtp_max_depth"


@pytest.fixture(autouse=True)
def isolate_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(defaults, "overlay_path", lambda: tmp_path / "profile.json")
    defaults.overlay_profile.cache_clear()
    yield
    defaults.overlay_profile.cache_clear()


def cell(**changes):
    return dict(
        id="test",
        hardware=[dict(gpu_generation=15, tier="Ultra")],
        knobs={NAME: 3},
        provenance={"evidence": "reviewed measurement"},
        **changes,
    )


def document(cells=None, **changes):
    return dict(
        schema_version=1,
        profile_version="test.1",
        generated_at="2026-11-10T00:00:00Z",
        cells=[cell()] if cells is None else cells,
        **changes,
    )


def load(tmp_path, data):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    return defaults.load_profile(path, lattice.KNOBS)


def context(**changes):
    return replace(
        lattice.LatticeContext(
            lattice.HardwareProfile(gpu_generation=15, tier="Ultra"),
            "0.32.4",
            "26.5.2",
            mode="mtp",
            has_mtp_head=True,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("extra", 1),
        ("schema_version", 2),
        ("schema_version", True),
        ("profile_version", "private/path"),
        ("generated_at", None),
        ("generated_at", "bad"),
        ("cells", {}),
    ],
)
def test_invalid_document(tmp_path, caplog, key, value):
    data = document()
    data[key] = value
    with caplog.at_level(logging.DEBUG):
        assert load(tmp_path, data) == defaults.Profile()
    assert "Rejected lattice profile" in caplog.text


@pytest.mark.parametrize("data", [[], None, {}, {"cells": []}])
def test_missing_document_structure(tmp_path, data):
    assert load(tmp_path, data) == defaults.Profile()


@pytest.mark.parametrize("raw", ["{", "[]" * 3, '"\\ud800"'])
def test_unreadable_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    assert defaults.load_profile(path, lattice.KNOBS) == defaults.Profile()


def test_missing_and_unreadable_file(tmp_path):
    assert (
        defaults.load_profile(tmp_path / "missing", lattice.KNOBS) == defaults.Profile()
    )
    assert defaults.load_profile(tmp_path, lattice.KNOBS) == defaults.Profile()


@pytest.mark.parametrize(
    "key,value",
    [
        ("extra", True),
        ("id", "bad/path"),
        ("hardware", []),
        ("hardware", {}),
        ("hardware", [dict(gpu_generation=True, tier="Ultra")]),
        ("hardware", [dict(gpu_generation=15, tier="Unknown")]),
        ("hardware", [dict(gpu_generation=15, tier="Ultra", other=1)]),
        ("hardware", [dict(gpu_generation=15, tier="Ultra", gpu_cores=[80, 60])]),
        ("hardware", [dict(gpu_generation=15, tier="Ultra", gpu_cores=80)]),
        ("hardware", [dict(gpu_generation=15, tier="Ultra", gpu_cores=[0, 80])]),
        ("software", {"mlx": "invalid"}),
        ("software", {"mlx": 3}),
        ("software", {"mlx": ""}),
        ("software", {"unknown": "==1"}),
        ("model", {"unknown": 3}),
        ("model", {"geometry": [1]}),
        ("model", {"geometry": [True] * 6}),
        ("model", {"family": 3}),
        ("model", {"attention_kinds": "full_attention"}),
        ("model", {"attention_kinds": [False]}),
        ("workload", {"mode": "bad/path"}),
        ("workload", {"width": 0}),
        ("workload", {"extra": 1}),
        ("provenance", {}),
        ("provenance", {"evidence": "test", "unknown": True}),
        ("provenance", None),
        ("provenance", {"evidence": ""}),
        ("knobs", {"unknown": True}),
        ("knobs", {}),
        ("knobs", []),
        ("knobs", {NAME: True}),
        ("knobs", {NAME: 0}),
        ("knobs", {NAME: 5}),
        ("knobs", {NAME: "3"}),
        ("knobs", {"affine8_attention": 1}),
        ("knobs", {"qwen35_affine4_rows": [True, True]}),
        ("knobs", {"qwen35_affine4_rows": [2, 3]}),
        ("knobs", {"qwen35_affine4_rows": "bad"}),
        ("capabilities", "affine8_attention"),
        ("capabilities", ["unknown"]),
        ("capabilities", [NAME]),
        ("capabilities", [[]]),
    ],
)
def test_invalid_cell_does_not_poison_valid_sibling(tmp_path, caplog, key, value):
    bad = cell()
    bad[key] = value
    with caplog.at_level(logging.DEBUG):
        profile = load(tmp_path, document([bad, dict(cell(), id="good")]))
    assert [e["id"] for e in profile.cells] == ["good"]
    assert "Rejected lattice cell" in caplog.text


@pytest.mark.parametrize("bad", [[], None, {}, 1])
def test_cell_structure(tmp_path, bad):
    assert load(tmp_path, document([bad])).cells == ()


def test_duplicate_ids_rejected(tmp_path):
    assert load(tmp_path, document([cell(), cell()])).cells == ()


def test_tuple_values_normalized(tmp_path):
    e = cell()
    e["knobs"] = {"qwen35_affine4_rows": [2, 4]}
    assert load(tmp_path, document([e])).cells[0]["knobs"] == {
        "qwen35_affine4_rows": (2, 4)
    }


def test_matching_ranges_sets_versions_models_workloads(tmp_path):
    e = cell(
        software={"mlx": ">=0.32.3,<0.33", "macos": "==26.5.2", "rapid_mlx": ">=1.0"},
        model={
            "family": "qwen3_5_moe_text",
            "geometry": list(lattice.QWEN35_AFFINE4),
            "quant_mode": "affine",
            "attention_kinds": ["full_attention"],
        },
        workload={"mode": "mtp", "width": 1, "batch": 1},
    )
    e["hardware"].append(dict(gpu_generation=17, tier="Max"))
    e["hardware"][0].update(gpu_cores=[60, 80], memory_size=[256, 512])
    e = load(tmp_path, document([e])).cells[0]
    ctx = context(
        rapid_mlx="1.0.2",
        model=lattice.ModelProfile(
            "qwen3_5_moe_text",
            *lattice.QWEN35_AFFINE4,
            attention_kinds=("full_attention",),
            quant_mode="affine",
        ),
    )
    for cores in (60, 80):
        qualified = replace(
            ctx, hardware=replace(ctx.hardware, gpu_cores=cores, memory_size=256)
        )
        assert defaults.matches(e, qualified)
        assert not defaults.matches(e, replace(qualified, mlx="unknown"))
        assert not defaults.matches(e, replace(qualified, mlx="0.34"))
        assert not defaults.matches(e, replace(qualified, mode="plain"))
        assert not defaults.matches(e, replace(qualified, model=lattice.ModelProfile()))
    for cores in (None, 59, 81):
        assert not defaults.matches(
            e,
            replace(
                ctx, hardware=replace(ctx.hardware, gpu_cores=cores, memory_size=256)
            ),
        )
    assert not defaults.matches(e, ctx)
    assert defaults.matches(
        e, replace(ctx, hardware=lattice.HardwareProfile(gpu_generation=17, tier="Max"))
    )
    assert not defaults.matches(
        e, replace(ctx, hardware=lattice.HardwareProfile(gpu_generation=16, tier="Pro"))
    )


def test_full_precedence_and_dump(monkeypatch):
    path = defaults.overlay_path()
    path.write_text(json.dumps(document()))
    ctx = context()
    decision = lattice.resolve(NAME, ctx, environ={})
    assert (decision.value, decision.source, decision.profile_version) == (
        3,
        "overlay:test",
        "test.1",
    )
    assert lattice.dump(ctx)["knobs"][NAME] == {
        "value": 3,
        "source": "overlay:test",
        "reason": None,
        "profile_version": "test.1",
    }
    monkeypatch.setattr(lattice, "local_calibration", lambda c: {NAME: 2})
    assert lattice.resolve(NAME, ctx, environ={}).source == "local"
    assert lattice.resolve(NAME, ctx, environ={lattice.KNOBS[NAME].env: "4"}).value == 4
    assert (
        lattice.resolve(
            NAME,
            replace(ctx, has_mtp_head=False),
            environ={lattice.KNOBS[NAME].env: "4"},
        ).source
        == "capability"
    )


def test_overlay_cannot_qualify_unsafe_runtime():
    e = cell(capabilities=["affine8_attention"])
    e["knobs"] = {"affine8_attention": True}
    defaults.overlay_path().write_text(json.dumps(document([e])))
    assert (
        lattice.resolve(
            "affine8_attention",
            context(mlx="99"),
            environ={"RAPID_MLX_AFFINE8_ATTENTION": "1"},
        ).source
        == "capability"
    )


def test_overlay_conflicts_and_identical_cells(monkeypatch):
    first = cell()
    second = dict(cell(), id="second")
    profile = defaults.Profile("v", (first, second))
    monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)
    assert lattice.resolve(NAME, context(), environ={}).source == "overlay:second"
    second["knobs"] = {NAME: 4}
    assert lattice.resolve(NAME, context(), environ={}).value == 1


def test_overlay_path_convention(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The fixture replaces the path provider; exercise the original separately.
    assert (
        ORIGINAL_OVERLAY_PATH() == tmp_path / ".config/rapid-mlx/lattice-profile.json"
    )


ORIGINAL_OVERLAY_PATH = defaults.overlay_path


def test_builtin_resource_and_package_declaration():
    resource = files("rapid_mlx.lattice").joinpath("profiles/builtin.json")
    data = json.loads(resource.read_text())
    assert data["schema_version"] == 1
    assert len(defaults.builtin_profile().cells) == len(data["cells"]) == 22
    config = tomllib.loads(Path("pyproject.toml").read_text())
    assert (
        "lattice/profiles/*.json"
        in config["tool"]["setuptools"]["package-data"]["rapid_mlx"]
    )


# Golden values recorded from origin/main (1cbe83308), independent of profile data.
@pytest.mark.parametrize(
    "cell_id,generation,tier,mlx,macos,memory,values",
    [
        (
            "singleton-15-ultra",
            15,
            "Ultra",
            "0.32.4",
            "26.5.2",
            256,
            {"qwen35_affine4_bank_views": True, "qwen35_residual_norm": True},
        ),
        (
            "singleton-16-pro",
            16,
            "Pro",
            "0.32.4",
            "26.5.2",
            48,
            {"qwen35_affine4_bank_views": False, "qwen35_residual_norm": False},
        ),
        (
            "singleton-17-max",
            17,
            "Max",
            "0.32.4",
            "27.0.1",
            36,
            {"qwen35_affine4_bank_views": True, "qwen35_residual_norm": True},
        ),
        (
            "affine4-m4-pro-rows",
            16,
            "Pro",
            "future",
            "future",
            48,
            {"qwen35_affine4_rows": (2, 4)},
        ),
        (
            "affine8-m3-ultra",
            15,
            "Ultra",
            "0.32.3",
            "26.5.2",
            256,
            {"affine8_attention": True},
        ),
        (
            "mtp-depth-17-max-36g-qwen35-moe",
            17,
            "Max",
            "0.32.3",
            "27.0.1",
            36,
            {NAME: 4},
        ),
        (
            "mtp-depth-15-ultra-256g-qwen35-moe",
            15,
            "Ultra",
            "0.32.4",
            "26.5.2",
            256,
            {NAME: 3},
        ),
    ],
)
@pytest.mark.parametrize("cores", [None, 60, 80])
def test_origin_main_golden(
    cell_id, generation, tier, mlx, macos, memory, values, cores
):
    ctx = context(
        hardware=lattice.HardwareProfile(
            gpu_generation=generation,
            tier=tier,
            gpu_cores=cores,
            memory_size=memory * 2**30,
        ),
        mlx=mlx,
        macos=macos,
        model=lattice.ModelProfile(
            "qwen3_5_moe_text", *lattice.QWEN35_AFFINE4, quant_mode="affine"
        ),
    )
    for knob, expected in values.items():
        active = replace(ctx, mode="mtp" if knob == NAME else "plain")
        decision = lattice.resolve(knob, active, environ={})
        assert decision.value == expected
        assert decision.source == "builtin:" + cell_id
        assert decision.profile_version == defaults.builtin_profile().version


def test_rapid_version_metadata(monkeypatch):
    from rapid_mlx.lattice import profiles

    profiles.rapid_version.cache_clear()
    monkeypatch.setattr(profiles, "version", lambda name: "1.2.3")
    assert profiles.rapid_version() == "1.2.3"
    profiles.rapid_version.cache_clear()

    def missing(name):
        raise profiles.PackageNotFoundError(name)

    monkeypatch.setattr(profiles, "version", missing)
    assert profiles.rapid_version() == "unknown"
    profiles.rapid_version.cache_clear()


@pytest.mark.parametrize(
    "cell_id,key,version",
    [
        ("singleton-15-ultra", "mlx", "0.32.4"),
        ("singleton-16-pro", "mlx", "0.32.4"),
        ("singleton-17-max", "mlx", "0.32.4"),
        ("affine8-m3-ultra", "mlx", "0.32.3"),
        ("affine8-m3-ultra", "macos", "26.5.2"),
        ("mtp-depth-17-max-36g-qwen35-moe", "mlx", "0.32.3"),
        ("mtp-depth-17-max-36g-qwen35-moe", "macos", "27.0.1"),
        ("mtp-depth-15-ultra-256g-qwen35-moe", "mlx", "0.32.4"),
        ("mtp-depth-15-ultra-256g-qwen35-moe", "macos", "26.5.2"),
    ],
)
@pytest.mark.parametrize("suffix", ["", "+custom", ".0", ".1", "rc1", ".post1"])
def test_migrated_software_pins_match_exactly(cell_id, key, version, suffix):
    e = next(e for e in defaults.builtin_profile().cells if e["id"] == cell_id)
    hw = e["hardware"][0]
    ctx = context(
        hardware=lattice.HardwareProfile(
            gpu_generation=hw["gpu_generation"],
            tier=hw["tier"],
            memory_size=hw.get("memory_size", [256 * 2**30])[0],
        ),
        model=lattice.ModelProfile(
            "qwen3_5_moe_text", *lattice.QWEN35_AFFINE4, quant_mode="affine"
        ),
        mode="plain" if cell_id.startswith("singleton") else "mtp",
        **{k: v.lstrip("=") for k, v in e["software"].items()},
    )
    ctx = replace(ctx, **{key: version + suffix})
    assert defaults.matches(e, ctx) == (suffix == "")
    for name, value in e["knobs"].items():
        decision = lattice.resolve(name, ctx, environ={})
        assert decision.value == (
            value if suffix == "" else lattice.KNOBS[name].default
        )
        assert decision.source == (
            "builtin:" + cell_id
            if suffix == ""
            else "capability"
            if name == "affine8_attention"
            else "default"
        )
        if name == "affine8_attention" and suffix:
            override = lattice.resolve(
                name, ctx, environ={lattice.KNOBS[name].env: "1"}
            )
            assert override.value is False
            assert override.source == "capability"


def test_base_tier_uses_existing_fact_vocabulary(tmp_path):
    e = cell()
    e["hardware"] = [dict(gpu_generation=15, tier="base")]
    profile = load(tmp_path, document([e]))
    ctx = lattice.runtime_context({"device_name": "Apple M3"})
    assert defaults.matches(profile.cells[0], ctx)


def test_overlay_overrides_builtin_but_not_local_or_env(monkeypatch):
    ctx = context(
        hardware=lattice.HardwareProfile(
            gpu_generation=15, tier="Ultra", memory_size=256 * 2**30
        ),
        model=lattice.ModelProfile(
            "qwen3_5_moe_text", *lattice.QWEN35_AFFINE4, quant_mode="affine"
        ),
    )
    assert lattice.resolve(NAME, ctx, environ={}).value == 3
    e = cell()
    e["knobs"] = {NAME: 4}
    defaults.overlay_path().write_text(json.dumps(document([e])))
    defaults.overlay_profile.cache_clear()
    assert lattice.resolve(NAME, ctx, environ={}).value == 4
    monkeypatch.setattr(lattice, "local_calibration", lambda c: {NAME: 2})
    assert lattice.resolve(NAME, ctx, environ={}).value == 2
    assert lattice.resolve(NAME, ctx, environ={lattice.KNOBS[NAME].env: "1"}).value == 1
