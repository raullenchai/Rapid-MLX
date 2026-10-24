"""Recipe epochs preserve completed payloads and the retained quality prefix."""

import json
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx


@pytest.fixture(autouse=True)
def restore_cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_amend as a
from scripts.deepseek_v41_gptq_io import atomic_json, digest
from scripts.deepseek_v41_gptq_quality import NativeBuild, bind_quality
from tests.test_deepseek_v41_gptq_quality import fixture


def metadata(tmp_path):
    parts, scratch = tmp_path / "parts", tmp_path / "scratch"
    parts.mkdir()
    scratch.mkdir()
    manifest = {"config": {"num_hidden_layers": 40}}
    atomic_json(parts / "source-manifest.json", manifest)
    run = dict(recipe=b.recipe(), source_identity=digest(manifest))
    run["identity"] = digest(run)
    atomic_json(parts / "run.json", run)
    atomic_json(
        scratch / "checkpoint.json", dict(next_layer=0, identity=run["identity"])
    )
    return parts, scratch, run


@pytest.mark.parametrize(
    "bad",
    [
        "phase",
        "working",
        "output",
        "boundary",
        "first",
        "damping",
        "past",
        "identity",
        "reason",
    ],
)
def test_rejects_unsafe_amendment(tmp_path, bad):
    parts, scratch, run = metadata(tmp_path)
    cp = dict(next_layer=0, identity=run["identity"])
    new = {**run["recipe"], "resident_budget_gb": 216.0}
    first, reason = 0, "approved cap"
    if bad == "phase":
        cp["phase"] = "ffn"
    if bad == "working":
        (scratch / "working-0").mkdir()
    if bad == "output":
        (parts / "model-layer-0.safetensors").touch()
    if bad == "boundary":
        (scratch / "boundary-1").mkdir()
    if bad == "first":
        first = 1
    if bad == "damping":
        new["damping"] = 0.2
    if bad == "past":
        cp["next_layer"] = first = 1
        new["gate_up_3bit_layers"] = [0]
        atomic_json(
            parts / "layer-0-complete.json",
            dict(run_identity=run["identity"], files={}, entries=[]),
        )
    if bad == "identity":
        cp["identity"] = "unknown"
    if bad == "reason":
        reason = " "
    atomic_json(scratch / "checkpoint.json", cp)
    before = (parts / "run.json").read_bytes()
    with pytest.raises(ValueError):
        a.amend_recipe(parts, scratch, new, first, reason)
    assert (parts / "run.json").read_bytes() == before


def test_journal_recovery_and_chain_rejection(tmp_path, monkeypatch):
    parts, scratch, original = metadata(tmp_path)
    new = {**original["recipe"], "resident_budget_gb": 216.0}
    real = a.atomic_json

    def crash(path, value):
        if path.name == "checkpoint.json":
            raise RuntimeError("interrupted")
        real(path, value)

    monkeypatch.setattr(a, "atomic_json", crash)
    with pytest.raises(RuntimeError):
        a.amend_recipe(parts, scratch, new, 0, "approved")
    monkeypatch.setattr(a, "atomic_json", real)
    run = a.amend_recipe(parts, scratch, new, 0, "approved")
    assert len(run["recipe_amendments"]) == 1
    assert (
        json.loads((scratch / "checkpoint.json").read_text())["identity"]
        == run["identity"]
    )
    with pytest.raises(ValueError):
        a.validate_record(run, 0, dict(run_identity=original["identity"]))
    broken = deepcopy(run)
    broken["recipe_amendments"][0]["old_identity"] = "unknown"
    with pytest.raises(ValueError):
        a.epochs(broken)
    newer = {**new, "cache_limit_gb": 0.5}
    run = a.amend_recipe(parts, scratch, newer, 0, "second")
    assert len(a.epochs(run)) == 3


def test_boundary_continue_assemble_and_retained_prefix(tmp_path):
    mx.set_default_device(mx.cpu)
    source, c, _, _, _ = fixture(tmp_path)
    c = b.recipe(
        {
            **c,
            "pilot": False,
            "pilot_compare_gate_up": False,
            "dense_8bit_layers": [0, 1, 2, 3],
        }
    )
    reference = json.loads((tmp_path / "reference/reference.json").read_text())
    for label in ("baseline", "amended"):
        root = tmp_path / label
        root.mkdir()
        parts, scratch = root / "parts", root / "scratch"
        with b.MemoryGuard(c) as guard:
            b.initialize(
                source,
                c,
                [np.array([1, 2, 5, 8], np.int32)],
                scratch,
                parts,
                "calibration",
                guard,
                token_map=list(range(32)),
                heldout_reference=reference,
                separation=dict(
                    reference=str(tmp_path / "reference"), corpus_hashes=[]
                ),
            )
            bind_quality(source, parts, tmp_path / "reference", scratch / "heldout")
            for layer in range(2):
                b.run_layer(source, c, parts, scratch, layer, guard)
        before = {
            p.name: p.read_bytes()
            for p in parts.glob("*")
            if p.is_file() and ("layer-0" in p.name or "layer-1" in p.name)
        }
        active = c
        if label == "amended":
            active = {**c, "gate_up_3bit_layers": [2, 3], "resident_budget_gb": 216.0}
            a.amend_recipe(parts, scratch, active, 2, "approved synthetic change")
            assert all(
                (parts / name).read_bytes() == content
                for name, content in before.items()
            )
        # Resume via the production CLI in a fresh CPU process for each layer.
        atomic_json(root / "recipe.json", active)
        for layer in range(2, 4):
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/deepseek_v41_gptq.py",
                    "run",
                    "--cpu",
                    "--source",
                    str(tmp_path / "source"),
                    "--manifest",
                    str(parts / "source-manifest.json"),
                    "--recipe",
                    str(root / "recipe.json"),
                    "--parts",
                    str(parts),
                    "--scratch",
                    str(scratch),
                    "--layer",
                    str(layer),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        with b.MemoryGuard(active) as guard:
            b.assemble(source, active, parts, root / "build", guard)
        if label == "amended":
            run = json.loads((parts / "run.json").read_text())
            config = json.loads((root / "build/config.json").read_text())
            assert (
                config["rapid_quantization"]["gptq"]["recipe_amendments"]
                == run["recipe_amendments"]
            )
            for layer in range(4):
                record = json.loads(
                    (parts / f"layer-{layer}-complete.json").read_text()
                )
                a.validate_record(run, layer, record)
                unknown = {**record, "run_identity": "unknown"}
                with pytest.raises(ValueError):
                    a.validate_record(run, layer, unknown)
                bits = config["quantization"]["modules"][
                    f"layers.{layer}.ffn.experts.gate_proj"
                ]["bits"]
                assert bits == (2 if layer < 2 else 3)
            # Readers/assembly reject a foreign identity even with valid file hashes.
            record_path = parts / "layer-0-complete.json"
            original = json.loads(record_path.read_text())
            atomic_json(record_path, {**original, "run_identity": "unknown"})
            with pytest.raises(ValueError):
                NativeBuild(parts, source).verify(0)
            with b.MemoryGuard(active) as guard, pytest.raises(ValueError):
                b.assemble(source, active, parts, root / "bad-build", guard)
            atomic_json(record_path, original)
    for layer in range(4):
        name = f"model-layer-{layer}.safetensors"
        old = (tmp_path / "baseline/build" / name).read_bytes()
        new = (tmp_path / "amended/build" / name).read_bytes()
        assert (old == new) == (layer < 2)


@pytest.mark.skipif(
    not os.environ.get("DSV41_LAUNCHER"),
    reason="owner launcher is external; set DSV41_LAUNCHER to opt in",
)
def test_launcher_checks_recorded_recipe_at_next_layer(tmp_path):
    from pathlib import Path

    parts, scratch, run = metadata(tmp_path)
    old = b.recipe(
        {
            **run["recipe"],
            "pilot": False,
            "dense_8bit_layers": list(range(8)),
            "memory_limit_gb": 180.0,
            "cache_limit_gb": 0.5,
        }
    )
    run = dict(recipe=old, source_identity=run["source_identity"])
    run["identity"] = digest(run)
    atomic_json(parts / "run.json", run)
    atomic_json(
        scratch / "checkpoint.json", dict(next_layer=0, identity=run["identity"])
    )
    new = {
        **old,
        "resident_budget_gb": 216.0,
        "gate_up_3bit_layers": [15, 35, 36, 37, 38, 39],
    }
    a.amend_recipe(parts, scratch, new, 0, "approved")
    src, dst, calibration = [
        tmp_path / n for n in ("full.json", "studio.json", "calibration.json")
    ]
    calibration.touch()
    atomic_json(dst, old)
    text = Path(os.environ["DSV41_LAUNCHER"]).read_text()
    guard = text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    for c, succeeds in ((new, True), (old, False)):
        atomic_json(src, c)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                guard,
                str(src),
                str(dst),
                str(calibration),
                str(parts),
                str(scratch),
            ],
            capture_output=True,
        )
        assert (result.returncode == 0) == succeeds, result.stderr.decode()
    assert json.loads(dst.read_text()) == new


@pytest.mark.parametrize(
    "field", ["source_identity", "calibration_identity", "recipe", "identity"]
)
def test_unamended_run_rejects_modified_identity_fields(tmp_path, field):
    _, _, run = metadata(tmp_path)
    assert a.epochs(run) == [(0, run["identity"], run["recipe"])]
    original = run["identity"]
    if field == "recipe":
        run["recipe"]["damping"] *= 2
    else:
        run[field] = "modified"
    with pytest.raises(ValueError, match="authenticate"):
        a.epochs(run)
    with pytest.raises(ValueError, match="authenticate"):
        a.validate_record(run, 0, {"run_identity": original})


@pytest.mark.parametrize("kind", ["limit", "large", "manifest"])
def test_amendment_rejects_out_of_range_layers_before_commit(tmp_path, kind):
    parts, scratch, run = metadata(tmp_path)
    new = {**run["recipe"], "gate_up_3bit_layers": [40 if kind == "limit" else 1000000]}
    if kind == "manifest":
        atomic_json(
            parts / "source-manifest.json", {"config": {"num_hidden_layers": 1000001}}
        )
    before = (
        (parts / "run.json").read_bytes(),
        (scratch / "checkpoint.json").read_bytes(),
    )
    with pytest.raises(ValueError, match="outside source|manifest identity"):
        a.amend_recipe(parts, scratch, new, 0, "invalid layer")
    assert before == (
        (parts / "run.json").read_bytes(),
        (scratch / "checkpoint.json").read_bytes(),
    )


def test_amendment_cannot_undershoot_committed_payloads(tmp_path):
    from scripts.deepseek_v41_gptq_io import file_hash

    parts, scratch, run = metadata(tmp_path)
    payload = parts / "layer.bin"
    payload.write_bytes(b"data" * 4)
    atomic_json(
        parts / "layer-0-complete.json",
        dict(
            run_identity=run["identity"],
            files={"layer.bin": file_hash(payload)},
            entries=[
                dict(
                    name="weight",
                    shape=[4],
                    dtype="F32",
                    sources=[],
                    component="fixed_bf16",
                    bits=None,
                    group_size=64,
                    rows=None,
                    suffix="weight",
                )
            ],
        ),
    )
    atomic_json(
        scratch / "checkpoint.json", dict(next_layer=1, identity=run["identity"])
    )
    before = [
        (parts / "run.json").read_bytes(),
        (scratch / "checkpoint.json").read_bytes(),
    ]
    with pytest.raises(ValueError, match="below committed payload"):
        a.amend_recipe(
            parts,
            scratch,
            {**run["recipe"], "resident_budget_gb": 15 / 1e9},
            1,
            "too small",
        )
    assert before == [
        (parts / "run.json").read_bytes(),
        (scratch / "checkpoint.json").read_bytes(),
    ]
    accepted = a.amend_recipe(
        parts,
        scratch,
        {**run["recipe"], "resident_budget_gb": 16 / 1e9},
        1,
        "exact committed size",
    )
    assert accepted["recipe"]["resident_budget_gb"] == 16 / 1e9


@pytest.mark.parametrize("budget", [True, None, float("nan")])
def test_amendment_rejects_invalid_budget_before_metadata_changes(tmp_path, budget):
    parts, scratch, run = metadata(tmp_path)
    before = (parts / "run.json").read_bytes()
    with pytest.raises(ValueError, match="invalid memory cap"):
        a.amend_recipe(
            parts,
            scratch,
            {**run["recipe"], "resident_budget_gb": budget},
            0,
            "invalid cap",
        )
    assert (parts / "run.json").read_bytes() == before
