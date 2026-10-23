"""Offline allocation rejection and CLI contracts on small artifacts."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import deepseek_v41_rd_allocate as alloc
from scripts import deepseek_v41_rd_layer_allocate as layer
from scripts import deepseek_v41_variant_alloc as variant
from scripts.deepseek_v41_contrastive import contrastive
from tests.test_deepseek_v41_allocation_phase2 import slices
from tests.test_deepseek_v41_allocation_phase4 import selection
from tests.test_deepseek_v41_merge_calibration import raw, rewrite, write_pass


def test_contrastive_invalid_policy_and_matching():
    r, c, names, roles = slices()
    with pytest.raises(ValueError, match="thresholds"):
        contrastive(r, c, names, roles, hot_threshold=0)
    roles["extra"] = "reference"
    names[0] = "extra"
    with pytest.raises(ValueError, match="language-matched"):
        contrastive(r, c, names, roles)
    r, c, names, roles = slices()
    roles.pop("medical")
    with pytest.raises(ValueError, match="explicit role"):
        contrastive(r, c, names, roles)


@pytest.mark.parametrize(
    "bad",
    ["never_missing", "shape", "mask", "never", "projection", "protected", "cost"],
)
def test_layer_invalid_contracts(bad):
    sel = selection()
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], (1, 6, 4))
    costs = np.broadcast_to([0, 30, 42, 54], q.shape).copy()
    kw = {}
    if bad == "never_missing":
        del sel["never_routed"]
        kw.update(fallback=True, allow_hot_pruning=True)
    elif bad == "shape":
        q = q[:, :, :2]
    elif bad == "mask":
        sel["candidate"] = np.zeros((2, 6), bool)
    elif bad == "never":
        sel["never_routed"] = np.ones((1, 6), bool)
    elif bad == "projection":
        kw["projection_q"] = np.zeros((1, 6, 2, 3))
    elif bad == "protected":
        sel["protected"][0, 0] = True
    elif bad == "cost":
        costs[0, 1, 1] += 1
    with pytest.raises(ValueError):
        layer.layer_allocate(sel, q, costs, 0, 1000, topk=1, **kw)


def test_global_and_audit_reject_bad_shapes():
    with pytest.raises(ValueError, match="expected sensitivity"):
        alloc.allocate(np.ones(2), np.ones(2), np.ones(2), 0, 1)
    with pytest.raises(ValueError, match="mass arrays"):
        alloc.pruning_audit({"layers": {}}, np.ones(2), np.ones(2), [], {})
    with pytest.raises(ValueError, match="hot calibration"):
        alloc.pruning_audit(
            {"layers": {"0": {"kept_expert_ids": [0]}}},
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
            ["cold"],
            {"cold": "cold"},
        )


@pytest.mark.parametrize("bad", ["cost", "ids", "bits", "fixed"])
def test_variant_rejects_invalid_base(bad):
    cost = {
        "expert_bytes": np.broadcast_to([0, 30, 42, 54], (1, 2, 4)).tolist(),
        "fixed_non_expert_bytes": 0,
    }
    base = {
        "layers": {"0": {"kept_expert_ids": [0, 1], "bits": 2}},
        "fixed_non_expert_bytes": 0,
        "resident_tensor_bytes": 60,
    }
    if bad == "cost":
        cost["expert_bytes"][0][0][0] = 1
    if bad == "ids":
        base["layers"]["0"]["kept_expert_ids"] = [1, 0]
    if bad == "bits":
        base["layers"]["0"]["bits"] = 8
    if bad == "fixed":
        base["fixed_non_expert_bytes"] = 1
    base["provenance"] = {"costs_sha256": alloc.inventory_digest(cost)}
    with pytest.raises(ValueError):
        variant.variant_allocation(base, cost, [])


def test_merge_rejects_corruption_and_read_race(tmp_path, monkeypatch):
    from scripts import deepseek_v41_merge_calibration as merge

    a = write_pass(tmp_path / "a.npz", raw(1))
    b = write_pass(tmp_path / "b.npz", raw(2))
    for update in (
        {"keep_experts": np.array(3)},
        {"counts_by_slice": np.zeros((3, 2, 4))},
    ):
        write_pass(b, raw(2))
        rewrite(b, arrays_update=update)
        with pytest.raises(ValueError, match="inconsistent"):
            merge.merge_calibration([a, b], tmp_path / "out.npz")
    write_pass(b, raw(2))
    rewrite(b, meta_update={"keep_experts": 0})
    with pytest.raises(ValueError, match="keep_experts"):
        merge.merge_calibration([a, b], tmp_path / "out.npz")
    original = merge._read_pass

    def changed(p):
        result = original(p)
        p.write_bytes(p.read_bytes() + b"x")
        return result

    monkeypatch.setattr(merge, "_read_pass", changed)
    with pytest.raises(ValueError, match="changed while reading"):
        merge.merge_calibration([a, b], tmp_path / "out.npz")


@pytest.mark.parametrize(
    "script",
    [
        "deepseek_v41_rd_allocate",
        "deepseek_v41_variant_alloc",
        "deepseek_v41_rd_synthetic",
        "deepseek_v41_merge_calibration",
    ],
)
def test_direct_cli_help(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"),
            run_name="__main__",
        )
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    sm = {"model_revision": "r", "routing_topk": 1, "config_sha256": "config"}
    qm = {
        "model_revision": "r",
        "config_sha256": "config",
        "group_size": 64,
        "bits_order": [0, 2, 3, 4],
    }
    payload = np.broadcast_to([0, 30, 42, 54], (1, 2, 4)).copy()
    score = dict(
        robust_reap=np.ones((1, 2)),
        split_keep_overlap=np.ones((1, 3)),
        metadata=json.dumps(sm),
    )
    error = dict(
        q=np.broadcast_to([1.0, 0.3, 0.1, 0.01], payload.shape),
        completed=np.ones((1, 2), bool),
        expert_weight_bytes=payload,
        metadata=json.dumps(qm),
    )
    cost = dict(
        model_revision="r",
        group_size=64,
        fixed_non_expert_bytes=0,
        router_row_bytes_by_layer={"0": 0},
        expert_bytes=payload.tolist(),
    )
    paths = [tmp_path / n for n in ("score.npz", "error.npz", "cost.json")]
    argv = [
        "allocate",
        "--scores",
        str(paths[0]),
        "--errors",
        str(paths[1]),
        "--costs",
        str(paths[2]),
        "--granularity",
        "legacy-expert",
        "--topk",
        "1",
        "--min-kept",
        "1",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    def write():
        np.savez(paths[0], **score)
        np.savez(paths[1], **error)
        paths[2].write_text(json.dumps(cost))

    return score, error, cost, argv, write


@pytest.mark.parametrize(
    "bad",
    [
        "overlap_threshold",
        "overlap",
        "completed",
        "revision",
        "topk",
        "format",
        "cost_group",
        "cost_shape",
        "missing_source",
        "legacy_preset",
    ],
)
def test_cli_rejects_unusable_artifacts(cli_inputs, bad):
    score, error, cost, argv, write = cli_inputs
    if bad == "overlap_threshold":
        argv.extend(["--min-split-overlap", "2"])
    elif bad == "overlap":
        score["split_keep_overlap"][:] = 0
    elif bad == "completed":
        error["completed"][:] = False
    elif bad in ("revision", "format"):
        meta = json.loads(error["metadata"])
        meta["model_revision" if bad == "revision" else "group_size"] = "bad"
        error["metadata"] = json.dumps(meta)
    elif bad == "topk":
        argv[argv.index("--topk") + 1] = "2"
    elif bad == "cost_group":
        cost["group_size"] = 128
    elif bad == "cost_shape":
        cost["expert_bytes"] = []
    elif bad == "missing_source":
        argv[argv.index("--granularity") + 1] = "layer"
        argv.extend(["--max-prune-fraction", "0.1"])
    elif bad == "legacy_preset":
        argv.extend(["--preset", "T256"])
    write()
    with pytest.raises(ValueError):
        alloc.main()


@pytest.mark.parametrize("mismatch", [False, True])
def test_synthetic_plans_record_assumptions(tmp_path, monkeypatch, mismatch):
    from scripts import deepseek_v41_rd_synthetic as synth

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    costs = tmp_path / "costs.json"
    costs.write_text(
        json.dumps(
            {
                "expert_bytes": np.broadcast_to([0, 30, 42, 54], (1, 384, 4)).tolist(),
                "fixed_non_expert_bytes": 0,
            }
        )
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthetic",
            "--source",
            str(source),
            "--costs",
            str(costs),
            "--output-dir",
            str(out),
        ],
    )
    monkeypatch.setattr(synth, "CheckpointIndex", lambda p: object())
    monkeypatch.setattr(synth, "mtp_byte_costs", lambda *a: {"4": 12, "8": 24})

    # Exercise real scoring/allocation; only source-shard planning is an I/O boundary.
    def plan(_index, _config, result):
        return (
            [],
            None,
            {
                "resident_tensor_bytes": result.get("resident_tensor_bytes", 0)
                + int(mismatch)
            },
        )

    monkeypatch.setattr(synth, "plan", plan)
    monkeypatch.setattr(synth, "layout_report", lambda entries, report: report)
    if mismatch:
        with pytest.raises(ValueError, match="bytes disagree"):
            synth.main()
    else:
        synth.main()
        for target in ("T128", "T256"):
            result = json.loads((out / f"{target}-synthetic.json").read_text())
            assert result["synthetic_only"] is True
            assert "Not measured" in result["assumptions"]
            assert result["keep_mtp"] == (target == "T256")
            assert result["resident_tensor_bytes"] <= result["budget_bytes"]
        assert json.loads((out / "reference-plan.json").read_text())["reference"]
    costs.write_text(json.dumps({"expert_bytes": [[[0, 30, 42, 54]]]}))
    with pytest.raises(ValueError, match="384 experts"):
        synth.main()


def test_merge_cli_and_invalid_counts(tmp_path, monkeypatch):
    from scripts import deepseek_v41_merge_calibration as merge

    a = write_pass(tmp_path / "a.npz", raw(1))
    b = write_pass(tmp_path / "b.npz", raw(2))
    output = tmp_path / "merged.npz"
    monkeypatch.setattr(sys, "argv", ["merge", str(a), str(b), "--output", str(output)])
    merge.main()
    with np.load(output) as result:
        assert result["tokens"].item() == 48
    rewrite(b, arrays_update={"counts_halves_by_slice": np.ones((3, 2, 2, 4), float)})
    with pytest.raises(ValueError, match="invalid counts"):
        merge.merge_calibration([a, b], tmp_path / "bad.npz")


def test_single_reference_requires_matching_language():
    r, c, names, roles = slices()
    names = ["chat", "medical", "cold_history_zh", "general_reference_en"]
    roles = dict(zip(names, ["hot", "hot", "cold", "reference"]))
    with pytest.raises(ValueError, match="language-matched"):
        contrastive(r, c, names, roles)
    names[-1] = "general_reference_zh"
    roles = dict(zip(names, ["hot", "hot", "cold", "reference"]))
    assert contrastive(r, c, names, roles)["candidate"][0, 1]


@pytest.mark.parametrize("key", ["config_sha256", "checkpoint_sha256"])
def test_allocator_rejects_same_label_different_identity(cli_inputs, key):
    score, error, cost, argv, write = cli_inputs
    meta = json.loads(error["metadata"])
    meta[key] = "different"
    error["metadata"] = json.dumps(meta)
    write()
    with pytest.raises(ValueError, match="hashes do not match"):
        alloc.main()


def test_allocator_rejects_colliding_budget_output_names(cli_inputs):
    score, error, cost, argv, write = cli_inputs
    argv.extend(["--budgets-gb", "200.00001", "200.00002"])
    write()
    with pytest.raises(ValueError, match="duplicate allocation output"):
        alloc.main()
    assert not list(Path(argv[argv.index("--output-dir") + 1]).glob("*.json"))


def test_merge_failed_write_never_publishes_and_can_retry(tmp_path, monkeypatch):
    from scripts import deepseek_v41_merge_calibration as merge

    a = write_pass(tmp_path / "a.npz", raw(1))
    b = write_pass(tmp_path / "b.npz", raw(2))
    out = tmp_path / "merged.npz"
    original = merge.save_scores

    def broken(handle, *args):
        handle.write(b"partial archive")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(merge, "save_scores", broken)
    with pytest.raises(OSError, match="disk failure"):
        merge.merge_calibration([a, b], out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npz", "b.npz"]
    monkeypatch.setattr(merge, "save_scores", original)
    merge.merge_calibration([a, b], out)
    with np.load(out) as result:
        assert result["tokens"].item() == 48


@pytest.mark.parametrize("role", ["hott", "", None, 0])
def test_contrastive_rejects_unsupported_roles(role):
    r, c, names, roles = slices()
    roles["medical"] = role
    with pytest.raises(ValueError, match="unsupported slice role"):
        contrastive(r, c, names, roles)


@pytest.mark.parametrize(
    "parameter", ["hot_threshold", "cold_ratio", "min_cold_routes", "ratio_tolerance"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_contrastive_rejects_nonfinite_thresholds(parameter, value):
    with pytest.raises(ValueError, match="thresholds"):
        contrastive(*slices(), **{parameter: value})


@pytest.mark.parametrize("value", [1.5, True, np.bool_(True)])
def test_contrastive_rejects_noninteger_route_threshold(value):
    with pytest.raises(ValueError, match="thresholds"):
        contrastive(*slices(), min_cold_routes=value)


@pytest.mark.parametrize("value", [2, 2.0, np.int64(2)])
def test_contrastive_accepts_integer_route_threshold(value):
    assert contrastive(*slices(), min_cold_routes=value)["candidate"][0, 1]


@pytest.mark.parametrize("change", ["missing", "revision", "config", "payload"])
def test_variant_rejects_unbound_inventory(change):
    cost = {"model_revision": "r", "config_sha256": "c", "expert_bytes": [1]}
    base = {"provenance": {"costs_sha256": alloc.inventory_digest(cost)}}
    if change == "missing":
        base.clear()
    elif change == "revision":
        cost["model_revision"] = "other"
    elif change == "config":
        cost["config_sha256"] = "other"
    else:
        cost["expert_bytes"] = [2]
    with pytest.raises(ValueError, match="base provenance"):
        variant.variant_allocation(base, cost, [])


def test_allocation_failed_later_budget_publishes_nothing(cli_inputs):
    _, _, _, argv, write = cli_inputs
    argv.extend(["--budgets-gb", "1", "0"])
    write()
    with pytest.raises(ValueError):
        alloc.main()
    assert not list(Path(argv[argv.index("--output-dir") + 1]).iterdir())


@pytest.mark.parametrize("failure", ["existing", "fsync", "link", "nan"])
def test_allocation_publication_failure_preserves_existing(
    tmp_path, monkeypatch, failure
):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    outputs = [(first, {"a": 1}), (second, {"b": 2})]
    if failure == "existing":
        second.write_text("original")
    elif failure == "nan":
        outputs[1][1]["b"] = float("nan")
    else:
        original = getattr(alloc.os, failure)
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk failure")
            return original(*args)

        monkeypatch.setattr(alloc.os, failure, fail_second)
    with pytest.raises((OSError, ValueError)):
        alloc.publish_allocations(outputs)
    assert not first.exists()
    assert list(tmp_path.iterdir()) == ([second] if failure == "existing" else [])
    if failure == "existing":
        assert second.read_text() == "original"
