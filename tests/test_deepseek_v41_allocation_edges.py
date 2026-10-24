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
    names[3] = "extra"
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
    monkeypatch.setattr(alloc, "validate_legacy_inventory", lambda *args: None)
    sm = {
        "model_revision": "r",
        "routing_topk": 1,
        "config_sha256": "config",
        "slice_names": ["hot"],
    }
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
        assert not list(out.iterdir())
    else:
        synth.main()
        for target in ("T128", "T256"):
            result = json.loads((out / f"{target}-synthetic.json").read_text())
            assert result["synthetic_only"] is True
            assert "Not measured" in result["assumptions"]
            assert result["keep_mtp"] == (target == "T256")
            assert result["resident_tensor_bytes"] <= result["budget_bytes"]
        assert json.loads((out / "reference-plan.json").read_text())["reference"]
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with pytest.raises(FileExistsError):
            synth.main()
        assert before == {p.name: p.read_bytes() for p in out.iterdir()}
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


@pytest.mark.parametrize("release", [False, True])
def test_legacy_inventory_verifies_actual_source(tmp_path, release):
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    from scripts.deepseek_v41_expert_quant_error import inventory
    from scripts.deepseek_v41_rd_build import plan
    from scripts.repack_deepseek_v41_native import CheckpointIndex
    from tests.test_deepseek_v41_quant_error import tiny_source

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 1
    (source / "config.json").write_text(json.dumps(config))
    index = CheckpointIndex(source)
    cost = inventory(index, config, 4)
    if release:
        _, _, report = plan(
            index, config, {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 2}}}
        )
        cost["fixed_non_expert_bytes"] = sum(
            v
            for k, v in report["components"].items()
            if k not in ("router", "routed_experts", "engram_ssd")
        )
        cost["policy"] = "release: test"
        cost["nonexpert_bits"] = 8
    config_hash = alloc.file_hash(source / "config.json")
    metadata = {
        "index_sha256": alloc.file_hash(source / "model.safetensors.index.json")
    }
    alloc.validate_legacy_inventory(source, cost, config_hash, metadata)
    with pytest.raises(ValueError, match="requires --source"):
        alloc.validate_legacy_inventory(None, cost, config_hash, metadata)
    with pytest.raises(ValueError, match="identity"):
        alloc.validate_legacy_inventory(source, cost, "wrong", metadata)
    with pytest.raises(ValueError, match="identity"):
        alloc.validate_legacy_inventory(source, cost, config_hash, {})
    cost["fixed_non_expert_bytes"] -= 1
    with pytest.raises(ValueError, match="fixed_non_expert_bytes mismatch"):
        alloc.validate_legacy_inventory(source, cost, config_hash, metadata)


@pytest.mark.parametrize("cold_name", ["cold", "cold_"])
def test_unqualified_cold_slice_cannot_use_single_reference(cold_name):
    r, counts, names, roles = slices()
    names[2] = cold_name
    roles[cold_name] = roles.pop("cold_en")
    with pytest.raises(ValueError, match="explicit language suffix"):
        contrastive(r, counts, names, roles)


@pytest.mark.parametrize("replacement", [True, False])
def test_allocation_rollback_preserves_replaced_or_removed_output(
    tmp_path, monkeypatch, replacement
):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    original = alloc.os.link

    def link(source, destination):
        if destination == second:
            first.unlink()
            if replacement:
                first.write_text("other writer")
            raise OSError("second link failed")
        original(source, destination)

    monkeypatch.setattr(alloc.os, "link", link)
    with pytest.raises(OSError, match="second link failed"):
        alloc.publish_allocations([(first, {"a": 1}), (second, {"b": 2})])
    assert list(tmp_path.iterdir()) == ([first] if replacement else [])
    if replacement:
        assert first.read_text() == "other writer"


def test_contrastive_rejects_fractional_route_counts():
    r, counts, names, roles = slices()
    counts = counts.astype(float)
    counts[2, 0, 0, 1] = 2.5
    with pytest.raises(ValueError):
        contrastive(r, counts, names, roles)
    counts[2, 0, 0, 1] = 2.0
    assert contrastive(r, counts, names, roles)["candidate"][0, 1]


def test_directory_sync_failure_closes_descriptor(tmp_path, monkeypatch):
    import os

    from scripts import deepseek_v41_merge_calibration as merge

    descriptors = []
    original = os.fsync

    def fail(descriptor):
        descriptors.append(descriptor)
        raise OSError("directory sync failed")

    monkeypatch.setattr(merge.os, "fsync", fail)
    with pytest.raises(OSError, match="directory sync failed"):
        merge.fsync_directory(tmp_path)
    monkeypatch.setattr(merge.os, "fsync", original)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_publication_syncs_directory_on_success_and_rollback(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(alloc, "fsync_directory", synced.append)
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    alloc.publish_allocations([(first, {"a": 1})])
    assert synced == [tmp_path]
    first.unlink()
    original = alloc.os.link

    def fail_second(source, destination):
        if destination == second:
            raise OSError("link failed")
        original(source, destination)

    monkeypatch.setattr(alloc.os, "link", fail_second)
    with pytest.raises(OSError):
        alloc.publish_allocations([(first, {"a": 1}), (second, {"b": 2})])
    assert synced == [tmp_path, tmp_path]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["unlink", "directory_sync", "temporary"])
def test_allocation_rollback_attempts_every_cleanup(tmp_path, monkeypatch, failure):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
    first, second, third = (
        roots[0] / "one.json",
        roots[1] / "two.json",
        roots[1] / "three.json",
    )
    original_link = alloc.os.link
    original_unlink = Path.unlink
    sync_calls = []
    failed_temporary = []

    def link(source, destination):
        if destination == third:
            raise OSError("publication failure")
        original_link(source, destination)

    def unlink(path, *args, **kwargs):
        if failure == "unlink" and path == first:
            raise OSError("unlink failure")
        if failure == "temporary" and path.name.startswith("tmp"):
            failed_temporary.append(path)
            raise OSError("temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    def sync(path):
        sync_calls.append(path)
        if failure == "directory_sync":
            raise OSError("directory sync failure")

    monkeypatch.setattr(alloc.os, "link", link)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(alloc, "fsync_directory", sync)
    with pytest.raises(
        alloc.AllocationRollbackError, match="rollback incomplete"
    ) as caught:
        alloc.publish_allocations([(first, {}), (second, {}), (third, {})])
    assert str(caught.value.__cause__) == "publication failure"
    assert len(caught.value.failures) >= 1
    assert set(sync_calls) == set(roots)
    assert not second.exists() and not third.exists()
    assert first.exists() == (failure == "unlink")
    if failure == "temporary":
        assert len(failed_temporary) == 3


def test_temporary_cleanup_failure_rolls_back_published_output(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    original = Path.unlink
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        if path.name.startswith("tmp"):
            calls += 1
            if calls == 1:
                raise OSError("temporary unlink failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="temporary unlink failed"):
        alloc.publish_allocations([(output, {})])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "overlap",
    [np.array(1), np.ones(3), np.ones((2, 3)), np.ones((1, 2)), np.full((1, 3), 2)],
)
def test_allocator_rejects_malformed_stability_evidence(cli_inputs, overlap):
    score, _, _, _, write = cli_inputs
    score["split_keep_overlap"] = overlap
    write()
    with pytest.raises(ValueError, match="unstable calibration"):
        alloc.main()


@pytest.mark.parametrize(
    "completed",
    [np.array(True), np.ones((1, 1), bool), np.ones((1, 2), int), np.ones((2,), bool)],
)
def test_allocator_rejects_malformed_completion_evidence(cli_inputs, completed):
    _, error, _, _, write = cli_inputs
    error["completed"] = completed
    write()
    with pytest.raises(ValueError, match="incomplete"):
        alloc.main()


def test_legacy_inventory_rejects_source_changed_during_use(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    from scripts import deepseek_v41_expert_quant_error as quant
    from scripts.repack_deepseek_v41_native import CheckpointIndex
    from tests.test_deepseek_v41_quant_error import tiny_source

    source = tmp_path / "source"
    config = tiny_source(source)
    cost = quant.inventory(CheckpointIndex(source), config)
    config_path = source / "config.json"
    config_hash = alloc.file_hash(config_path)
    metadata = {
        "index_sha256": alloc.file_hash(source / "model.safetensors.index.json")
    }
    original = quant.inventory

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        config_path.write_bytes(config_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(quant, "inventory", changed)
    with pytest.raises(ValueError, match="source checkpoint config"):
        alloc.validate_legacy_inventory(source, cost, config_hash, metadata)


@pytest.mark.parametrize("kind", ["unsigned", "float", "sum"])
def test_allocators_reject_cost_overflow(kind):
    if kind == "sum":
        costs = np.broadcast_to([0, 2**61, 2**61 + 1, 2**61 + 2], (1, 6, 4)).copy()
    else:
        costs = np.broadcast_to(
            np.array(
                [0, 2**62, 2**63, 2**63 + 1],
                dtype=np.uint64 if kind == "unsigned" else float,
            ),
            (1, 6, 4),
        ).copy()
        if kind == "float":
            costs[..., 3] = float(2**64)
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], costs.shape)
    with pytest.raises(ValueError, match="int64"):
        alloc.allocate(np.ones((1, 6)), q, costs, 0, 1e20, 1, 1)
    with pytest.raises(ValueError, match="int64"):
        layer.layer_allocate(selection(), q, costs, 0, 1e20, topk=1)
    cost = {"expert_bytes": costs.tolist(), "fixed_non_expert_bytes": 0}
    base = {
        "layers": {"0": {"kept_expert_ids": [0], "bits": 2}},
        "provenance": {"costs_sha256": alloc.inventory_digest(cost)},
    }
    with pytest.raises(ValueError, match="int64|invalid layer cost"):
        variant.variant_allocation(base, cost, [])


def test_never_routed_detection_does_not_sum_large_counts():
    r, counts, names, roles = slices()
    counts = counts.astype(np.uint64)
    counts[0, 0, 0, 3] = np.uint64(2**63)
    counts[0, 1, 0, 3] = np.uint64(2**63)
    assert counts.sum(axis=(0, 1))[0, 3] == 0
    result = contrastive(r, counts, names, roles)
    assert not result["never_routed"][0, 3]


@pytest.mark.parametrize("fraction", [np.nan, np.inf, -np.inf])
def test_layer_prune_fraction_already_rejects_nonfinite_values(fraction):
    costs = np.broadcast_to([0, 30, 42, 54], (1, 6, 4))
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], costs.shape)
    with pytest.raises(ValueError, match="invalid layer RD inputs"):
        layer.layer_allocate(
            selection(), q, costs, 0, 1000, max_prune_fraction=fraction, topk=1
        )


@pytest.mark.parametrize(
    "artifact,key",
    [
        ("score", "model_revision"),
        ("score", "slice_names"),
        ("score", "routing_topk"),
        ("error", "group_size"),
        ("error", "bits_order"),
        ("error", "config_sha256"),
    ],
)
@pytest.mark.parametrize("mode", ["missing", "wrong_type"])
def test_allocator_metadata_errors_are_actionable(cli_inputs, artifact, key, mode):
    score, error, _, _, write = cli_inputs
    data = score if artifact == "score" else error
    meta = json.loads(data["metadata"])
    if mode == "missing":
        del meta[key]
    else:
        meta[key] = None
    data["metadata"] = json.dumps(meta)
    write()
    with pytest.raises(ValueError, match=f"{artifact} metadata.*{key}"):
        alloc.main()


@pytest.mark.parametrize(
    "artifact,value",
    [
        ("score", None),
        ("error", []),
        ("score", {"model_revision": "r", "routing_topk": 1, "slice_names": [1]}),
        (
            "score",
            {"model_revision": "r", "routing_topk": 1, "slice_names": ["a", "a"]},
        ),
        (
            "error",
            {
                "model_revision": "r",
                "config_sha256": "c",
                "group_size": 64,
                "bits_order": ["0"],
            },
        ),
    ],
)
def test_allocator_rejects_invalid_metadata_schema(cli_inputs, artifact, value):
    score, error, _, _, write = cli_inputs
    (score if artifact == "score" else error)["metadata"] = json.dumps(value)
    write()
    with pytest.raises(ValueError, match="metadata"):
        alloc.main()


@pytest.mark.parametrize(
    "artifact,key", [("score", "split_keep_overlap"), ("error", "completed")]
)
def test_allocator_reports_missing_arrays(cli_inputs, artifact, key):
    score, error, _, _, write = cli_inputs
    del (score if artifact == "score" else error)[key]
    write()
    with pytest.raises(ValueError, match=f"{artifact} artifact missing arrays"):
        alloc.main()


@pytest.mark.parametrize(
    "key,value", [("group_size", 128), ("bits_order", [0, 1, 3, 4])]
)
def test_allocator_rejects_well_typed_wrong_quantization_metadata(
    cli_inputs, key, value
):
    _, error, _, _, write = cli_inputs
    meta = json.loads(error["metadata"])
    meta[key] = value
    error["metadata"] = json.dumps(meta)
    write()
    with pytest.raises(ValueError, match="quantization error format"):
        alloc.main()


def test_layer_cli_requires_source_and_roles_after_schema_validation(cli_inputs):
    score, _, _, argv, write = cli_inputs
    argv[argv.index("--granularity") + 1] = "layer"
    argv.extend(["--max-prune-fraction", "0.1"])
    score["reap_halves"] = np.ones((1, 2, 1, 2))
    score["counts_halves_by_slice"] = np.ones((1, 2, 1, 2), int)
    write()
    with pytest.raises(ValueError, match="needs --score reap"):
        alloc.main()


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "resident",
        "layer",
        "routed",
        "router",
        "fixed",
        "breakdown_resident",
        "payload",
        "engram",
        "component_type",
    ],
)
def test_variant_rejects_stale_base_accounting(bad):
    cost = {
        "expert_bytes": np.broadcast_to([0, 30, 42, 54], (1, 2, 4)).tolist(),
        "fixed_non_expert_bytes": 100,
    }
    base = {
        "layers": {
            "0": {"kept_expert_ids": [0, 1], "bits": 2, "routed_and_router_bytes": 60}
        },
        "fixed_non_expert_bytes": 100,
        "resident_tensor_bytes": 160,
        "provenance": {"costs_sha256": alloc.inventory_digest(cost)},
        "byte_breakdown": {
            "components": {
                "routed_experts": 60,
                "router": 0,
                "shared": 100,
                "engram_ssd": 10,
            },
            "resident_tensor_bytes": 160,
            "total_payload_bytes": 170,
            "shared_engram_bytes": 10,
        },
    }
    assert (
        variant.variant_allocation(base, cost, ["0=3/3/3"])["resident_tensor_bytes"]
        == 184
    )
    breakdown = base["byte_breakdown"]
    if bad == "missing":
        del base["resident_tensor_bytes"]
    elif bad == "resident":
        base["resident_tensor_bytes"] += 1
    elif bad == "layer":
        base["layers"]["0"]["routed_and_router_bytes"] += 1
    elif bad in ("routed", "router", "fixed", "component_type"):
        key = {
            "routed": "routed_experts",
            "router": "router",
            "fixed": "shared",
            "component_type": "shared",
        }[bad]
        breakdown["components"][key] += 0.5 if bad == "component_type" else 1
    else:
        key = {
            "breakdown_resident": "resident_tensor_bytes",
            "payload": "total_payload_bytes",
            "engram": "shared_engram_bytes",
        }[bad]
        breakdown[key] += 1
    with pytest.raises(ValueError, match="base"):
        variant.variant_allocation(base, cost, ["0=3/3/3"])
