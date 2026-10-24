"""CPU qualification of RTN ranking, original teacher propagation and recovery."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx
from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_quality as q
from scripts import deepseek_v41_gptq_scan as s
from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    atomic_json,
    digest,
    inventory,
    load_state,
)
from tests.test_deepseek_v41_streaming_teacher import write_source


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(1017)
    yield
    mx.set_default_device(previous)


def fixture(tmp_path):
    write_source(tmp_path / "source")
    source = PartialSource(tmp_path / "source", inventory(tmp_path / "source"))
    c = b.recipe(dict(prefill_chunk=4, ffn_chunk=8, keep_mtp=False))
    windows = [np.array([1, 2, 5, 8, 13, 3, 4], np.int32)]
    return source, c, windows


def test_rtn_error_matches_explicit_gate_weighted_formula():
    rng = np.random.default_rng(7)
    w = mx.array(rng.normal(size=(12, 64)).astype(np.float32)).astype(mx.bfloat16)
    x = mx.array(rng.normal(size=(11, 64)).astype(np.float32))
    gates = np.linspace(0.1, 0.9, 11, dtype=np.float32)
    guard = SimpleNamespace(check=lambda: None)
    row = s.rtn_errors(w, lambda: iter([(x[:5], gates[:5]), (x[5:], gates[5:])]), guard)
    wf = np.asarray(w.astype(mx.float32)).astype(np.float64)
    xf = np.asarray(x).astype(np.float64)
    energy = np.sum((xf @ wf.T) ** 2 * gates[:, None])
    assert row["calibration_output_energy"] == pytest.approx(energy, rel=1e-5)
    for bits, key in [
        (2, "rtn_relative_output_error"),
        (3, "three_bit_rtn_relative_output_error"),
    ]:
        quant = mx.dequantize(
            *mx.quantize(w, bits=bits, group_size=64),
            bits=bits,
            group_size=64,
            dtype=mx.float32,
        )
        error = np.sum((xf @ (wf - np.asarray(quant)).T) ** 2 * gates[:, None]) / energy
        assert row[key] == pytest.approx(error, rel=1e-5)
    assert row["error_token_count"] == 11
    empty = s.rtn_errors(w, lambda: iter([]), guard)
    assert empty["calibration_output_energy"] == empty["rtn_relative_output_error"] == 0


def test_scan_original_boundary_resume_and_binding(tmp_path, monkeypatch):
    source, c, windows = fixture(tmp_path)
    root = tmp_path / "scan"
    with b.MemoryGuard(c) as guard:
        s.run_scan(
            source,
            root,
            c,
            windows,
            ["calib"],
            {"calibration": "one"},
            guard,
            through=1,
            token_map=list(range(32)),
        )
        original = tmp_path / "original"
        q.reference_init(
            source,
            original,
            windows,
            ["calib"],
            {},
            c,
            guard,
            token_map=list(range(32)),
        )
        q.reference_run(source, original, 1, c, guard)
        h = load_state(root / "boundary-2" / "window-0")["h"]
        expected = load_state(original / "boundary-2" / "window-0")["h"]
        assert np.array_equal(
            np.asarray(h.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
        )
        assert not (root / "boundary-0").exists()
        assert not (root / "boundary-1").exists()
        with pytest.raises(ValueError, match="complete scan"):
            b.select_recipe(root, source, c, scan=True)
        with pytest.raises(ValueError, match="identity changed"):
            s.run_scan(
                source, root, c, windows, ["calib"], {"calibration": "two"}, guard
            )
        with pytest.raises(ValueError, match="identity changed"):
            s.run_scan(
                source,
                root,
                {**c, "resident_budget_gb": 211},
                windows,
                ["calib"],
                {"calibration": "one"},
                guard,
            )
        real = s.scan_layer

        def interrupted(*args, **kwargs):
            real(*args, **kwargs)
            raise RuntimeError("power loss before commit")

        monkeypatch.setattr(s, "scan_layer", interrupted)
        with pytest.raises(RuntimeError, match="power loss"):
            s.run_scan(
                source,
                root,
                c,
                windows,
                ["calib"],
                {"calibration": "one"},
                guard,
                through=2,
            )
        assert (
            json.loads((root / "scan-checkpoint.json").read_text())["next_layer"] == 2
        )
        assert (root / "boundary-2").exists()
        monkeypatch.setattr(s, "scan_layer", real)
        s.run_scan(source, root, c, windows, ["calib"], {"calibration": "one"}, guard)
        before = (root / "layer-3-stats.json").read_bytes()
        s.run_scan(source, root, c, windows, ["calib"], {"calibration": "one"}, guard)
        assert before == (root / "layer-3-stats.json").read_bytes()
    rows = s.validate_records(root, source, c)
    assert len(rows) == 4
    assert all(len(r["projections"]) == 8 for r in rows)
    selected = b.recipe(b.select_recipe(root, source, c, scan=True))
    assert selected["pilot"] is False
    assert selected["dense_8bit_layers"] == [0, 1, 2, 3]
    b.output_plan(source, selected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deepseek_v41_gptq.py",
            "select-recipe",
            "--scan",
            str(root),
            "--output",
            str(tmp_path / "selected.json"),
        ],
    )
    b.main()
    assert json.loads((tmp_path / "selected.json").read_text()) == selected
    with b.MemoryGuard(selected) as guard:
        b.initialize(
            source,
            selected,
            windows,
            tmp_path / "full-state",
            tmp_path / "full-parts",
            "full-calibration",
            guard,
            token_map=list(range(32)),
        )
    full_run = json.loads((tmp_path / "full-parts" / "run.json").read_text())
    assert full_run["recipe"] == selected
    assert full_run["source_identity"] == digest(source.manifest)
    row = rows[0]
    row["projections"]["expert.0.w1"]["calibration_output_energy"] += 1
    atomic_json(root / "layer-0-stats.json", row)
    with pytest.raises(ValueError, match="corrupt"):
        s.validate_records(root, source, c)


def test_constructed_rankings_equal_pilot_selection(tmp_path, monkeypatch):
    source = SimpleNamespace(
        args=SimpleNamespace(n_layers=10, n_routed_experts=1), manifest={"source": 1}
    )
    c = b.recipe(dict(resident_budget_gb=104 / 1e9))
    scan, pilot = tmp_path / "scan", tmp_path / "pilot"
    scan.mkdir()
    pilot.mkdir()
    run = s.scan_metadata(source, c, {"calibration": "a"})
    atomic_json(scan / "run.json", run)
    atomic_json(
        scan / "scan-checkpoint.json", dict(identity=run["identity"], next_layer=10)
    )
    pilot_identity = digest(dict(recipe=c))
    atomic_json(pilot / "run.json", dict(identity=pilot_identity, recipe=c))
    for i in range(10):
        # Benefit ranks opposite dense movement; tied scores break by layer ID.
        projections = {
            f"expert.0.{p}": dict(
                rtn_relative_output_error=0.4,
                three_bit_rtn_relative_output_error=0.2,
                calibration_output_energy=(10 - i) * 100,
            )
            for p in ("w1", "w3")
        }
        row = dict(
            layer=i,
            run_identity=run["identity"],
            block_output_4bit_relative_error=i // 2,
            projections=projections,
        )
        row["sha256"] = digest(row)
        atomic_json(scan / f"layer-{i}-stats.json", row)
        real = {
            k: {
                **v,
                "gptq_relative_output_error": v["rtn_relative_output_error"],
                "three_bit_gptq_relative_output_error": v[
                    "three_bit_rtn_relative_output_error"
                ],
            }
            for k, v in projections.items()
        }
        atomic_json(
            pilot / f"layer-{i}-stats.json",
            {**row, "run_identity": pilot_identity, "projections": real},
        )
    monkeypatch.setattr(
        b,
        "output_plan",
        lambda source, c: (
            [],
            dict(resident_tensor_bytes=100 + 2 * len(c["gate_up_3bit_layers"])),
        ),
    )
    result = b.select_recipe(scan, source, c, scan=True)
    assert result == b.select_recipe(pilot, source, c)
    assert result["dense_8bit_layers"] == list(range(2, 10))
    assert result["gate_up_3bit_layers"] == [0, 1]
    with pytest.raises(ValueError, match="source/recipe/identity"):
        s.validate_records(scan, SimpleNamespace(manifest={"source": 2}), c)
    monkeypatch.setattr(
        b, "output_plan", lambda source, c: ([], dict(resident_tensor_bytes=105))
    )
    with pytest.raises(ValueError, match="exceed budget"):
        b.select_recipe(scan, source, c, scan=True)


def test_calibration_subset_windows_heldout_exclusion_and_binding(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{str(i): i + 1 for i in range(20)}}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for name in ("a", "b"):
        (tmp_path / (name + ".txt")).write_text(" ".join(str(i) for i in range(20)))
    (tmp_path / "held.txt").write_text("held out text")
    slices, held = tmp_path / "calibration.json", tmp_path / "heldout.json"
    atomic_json(slices, {"slices": {"a": ["a.txt"], "b": ["b.txt"]}})
    atomic_json(held, {"slices": {"held": ["held.txt"]}})
    source = SimpleNamespace(source=tmp_path, args=SimpleNamespace(max_seq_len=100))
    windows, names, provenance = s.calibration_subset(source, slices, held, 4, 2)
    assert len(windows) == len(names) == 4
    assert windows[0].tolist() == [1, 2, 3, 4]
    assert windows[1].tolist() == [17, 18, 19, 20]
    (tmp_path / "a.txt").write_text(" ".join(str(i) for i in reversed(range(20))))
    assert s.calibration_subset(source, slices, held, 4, 2)[2] != provenance
    with pytest.raises(ValueError, match="insufficient"):
        s.calibration_subset(source, slices, held, 11, 2)
    # Reject copied held-out text, even under another name/path.
    (tmp_path / "a.txt").write_text((tmp_path / "held.txt").read_text())
    with pytest.raises(ValueError, match="held-out"):
        s.calibration_subset(source, slices, held, 1, 1)
    atomic_json(slices, {"slices": {"a": ["held.txt"]}})
    with pytest.raises(ValueError, match="held-out"):
        s.calibration_subset(source, slices, held, 1, 1)


@pytest.mark.parametrize("kind", ["movement", "projection"])
def test_scan_measurements_reject_nonfinite_values(tmp_path, kind):
    source = SimpleNamespace(
        args=SimpleNamespace(n_layers=1, n_routed_experts=1), manifest={}
    )
    c = b.recipe()
    run = s.scan_metadata(source, c, {})
    atomic_json(tmp_path / "run.json", run)
    atomic_json(
        tmp_path / "scan-checkpoint.json", dict(identity=run["identity"], next_layer=1)
    )
    row = dict(
        layer=0,
        run_identity=run["identity"],
        block_output_4bit_relative_error=float("nan") if kind == "movement" else 0,
        projections={
            f"expert.0.{p}": dict(
                rtn_relative_output_error=float("nan") if kind == "projection" else 0,
                three_bit_rtn_relative_output_error=0,
                calibration_output_energy=1,
            )
            for p in ("w1", "w3")
        },
    )
    row["sha256"] = digest(row)
    atomic_json(tmp_path / "layer-0-stats.json", row)
    with pytest.raises(ValueError, match="invalid scan"):
        s.validate_records(tmp_path, source, c)


@pytest.mark.parametrize("kind", ["partial", "checkpoint", "through"])
def test_scan_rejects_uncommitted_state(tmp_path, kind):
    source = SimpleNamespace(args=SimpleNamespace(n_layers=1), manifest={})
    c = b.recipe()
    windows = [np.array([1, 2], np.int32)]
    provenance = dict(window_names=["a"], actual_token_hashes=[digest([1, 2])])
    run = s.scan_metadata(source, c, provenance)
    if kind == "partial":
        (tmp_path / "partial").touch()
    else:
        atomic_json(tmp_path / "run.json", run)
        atomic_json(
            tmp_path / "scan-checkpoint.json",
            dict(
                identity="bad" if kind == "checkpoint" else run["identity"],
                next_layer=0,
            ),
        )
        atomic_json(tmp_path / "reference.json", {})
    with pytest.raises(ValueError):
        s.run_scan(source, tmp_path, c, windows, ["a"], {}, None, through=99)
