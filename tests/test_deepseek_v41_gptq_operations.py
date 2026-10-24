"""Failure and CLI contracts for staging and full-run coordination."""

import json
import runpy
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_assemble_stage as s
from scripts import deepseek_v41_gptq_fullrun as f


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_staging_rejects_mutation_and_unexpected_group(tmp_path, monkeypatch):
    path = tmp_path / "data"
    with pytest.raises(ValueError, match="regular"):
        s.signature(path)
    path.write_bytes(b"original")
    real = s.file_hash

    def mutate(p):
        value = real(p)
        p.write_bytes(b"mutated")
        return value

    monkeypatch.setattr(s, "file_hash", mutate)
    with pytest.raises(ValueError, match="changed during hashing"):
        s.checked_hash(path)
    with pytest.raises(ValueError, match="unexpected"):
        s.check_binding(
            {"entry_digests": {}, "groups": {"bad": {}}}, {"entry_digests": {}}
        )


@pytest.mark.parametrize("command", ["verify-layers", "stage-other", "assemble-staged"])
def test_stage_cli_dispatch(tmp_path, monkeypatch, command):
    (tmp_path / "run.json").write_text(json.dumps({"recipe": b.recipe()}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = ["stage", command, "--parts", str(tmp_path)]
    if command == "verify-layers":
        args += ["--cache", str(tmp_path / "cache")]
    else:
        args += [
            "--source",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--staging",
            str(tmp_path),
        ]
    if command == "assemble-staged":
        args += [
            "--destination",
            str(tmp_path / "out"),
            "--verify-cache",
            str(tmp_path / "cache"),
        ]
    monkeypatch.setattr(sys, "argv", args)
    source = object()
    monkeypatch.setattr(s, "PartialSource", lambda *a: source)
    guard = SimpleNamespace(report=lambda: {})
    monkeypatch.setattr(s, "MemoryGuard", lambda *a: nullcontext(guard))
    mocks = {}
    for name in ("verify_layers", "stage_other", "assemble_staged"):
        mocks[name] = Mock(return_value={"layers": {"0": {}}})
        monkeypatch.setattr(s, name, mocks[name])
    s.main()
    assert mocks[command.replace("-", "_")].call_count == 1
    if command == "stage-other":
        recipe = tmp_path / "recipe.json"
        recipe.write_text(json.dumps({"memory_limit_gb": 2}))
        monkeypatch.setattr(sys, "argv", [*args, "--recipe", str(recipe)])
        with pytest.raises(ValueError, match="CURRENT"):
            s.main()


@pytest.mark.parametrize("module", [s, f])
def test_script_entrypoint_help(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__file__, "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(module.__file__, run_name="__main__")
    assert result.value.code == 0


@pytest.mark.parametrize("host", ["-option", "bad;command", ""])
def test_remote_rejects_unsafe_host(host):
    with pytest.raises(ValueError, match="host"):
        f.Remote(host, "python3")


def test_remote_transports_and_failures(tmp_path, monkeypatch):
    calls = []
    result = SimpleNamespace(returncode=0, stdout='{"ok": true}')

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return result

    monkeypatch.setattr(f.subprocess, "run", run)
    remote = f.Remote("example.test", "python3")
    assert remote.call("info", path="data") == {"ok": True}
    assert json.loads(calls[-1][1]["input"])["path"] == "data"
    remote.transfer(tmp_path / "a", "/work/a", True)
    assert calls[-1][0][0] == "rsync"
    remote.transfer("/work/a", tmp_path / "a", False)
    assert calls[-1][0][-1] == str(tmp_path / "a")
    result.returncode = 1
    with pytest.raises(RuntimeError, match="remote info failed"):
        remote.call("info", path="data")
    with pytest.raises(RuntimeError, match="rsync failed"):
        remote.transfer(tmp_path / "a", "/work/a", True)


@pytest.mark.parametrize("failure", [False, True])
def test_driver_cli(tmp_path, monkeypatch, failure):
    args = [
        "fullrun",
        "--host",
        "example.test",
        "--remote-work",
        "/work/build",
        "--remote-python",
        "python3",
        "--destination",
        str(tmp_path / "out"),
    ]
    for name in ("source", "manifest", "recipe", "calibration", "reference-source"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", args)
    run = Mock(side_effect=ValueError("fixture") if failure else None)
    monkeypatch.setattr(f, "Driver", lambda *a: SimpleNamespace(run=run))
    monkeypatch.setattr(f, "Remote", lambda *a: object())
    if failure:
        with pytest.raises(SystemExit) as exc:
            f.main()
        assert exc.value.code == 1
    else:
        f.main()
    assert run.called
    args[args.index("/work/build")] = "relative"
    with pytest.raises(SystemExit) as exc:
        f.main()
    assert exc.value.code == 2


def test_shard_sink_rejects_ranges_before_writing(tmp_path):
    from scripts.deepseek_v41_gptq_io import ShardSink
    from scripts.deepseek_v41_rd_build import Entry

    entry = Entry("bank", (2, 2), "F32", (), "test")
    sink = ShardSink(tmp_path / "out.safetensors", [entry])
    try:
        for expert in (-1, 2, True):
            with pytest.raises(ValueError, match="expert output range"):
                sink.put("bank", mx.ones((2,)), expert)
        with pytest.raises(ValueError, match="range size"):
            sink.put("bank", mx.ones((3,)))
        with pytest.raises(ValueError, match="dtype"):
            sink.put("bank", mx.ones((2, 2), mx.uint32))
        with pytest.raises(ValueError, match="incomplete"):
            sink.finish()
        sink.put("bank", mx.ones((2,)), 0)
        before = sink.file.tell()
        for expert, tensor in ((0, mx.zeros((2,))), (None, mx.zeros((2, 2)))):
            with pytest.raises(ValueError, match="duplicate"):
                sink.put("bank", tensor, expert)
            assert sink.file.tell() == before
        sink.put("bank", mx.ones((2,)), 1)
        sink.finish()
    finally:
        sink.close()
    sink = ShardSink(tmp_path / "whole.safetensors", [entry])
    try:
        sink.put("bank", mx.ones((2, 2)))
        with pytest.raises(ValueError, match="duplicate"):
            sink.put("bank", mx.zeros((2,)), 0)
        sink.finish()
    finally:
        sink.close()
    sink = ShardSink(tmp_path / "unfinished.safetensors", [entry])
    sink.close()


@pytest.mark.parametrize("kind", ["quantized", "expert_type", "path", "size", "header"])
def test_partial_source_rejects_invalid_inventory(tmp_path, kind):
    from scripts.deepseek_v41_gptq_io import PartialSource, inventory
    from tests.test_deepseek_v41_streaming_teacher import write_source

    write_source(tmp_path / "source")
    manifest = inventory(tmp_path / "source")
    if kind == "quantized":
        manifest["config"]["quantization"] = {"bits": 4}
    if kind == "expert_type":
        manifest["config"]["quantization_config"]["expert_dtype"] = "bad"
    if kind == "path":
        next(iter(manifest["tensors"].values()))["path"] = "../outside"
    if kind == "size":
        manifest["shards"]["source.safetensors"]["size"] += 1
    if kind == "header":
        manifest["shards"]["source.safetensors"]["header_sha256"] = "bad"
    with pytest.raises(ValueError):
        source = PartialSource(tmp_path / "source", manifest)
        source.verify(0)


def test_invalid_shard_component_and_path():
    from scripts.deepseek_v41_gptq_io import required_shards

    with pytest.raises(ValueError, match="component"):
        required_shards({"weight_map": {"embed.weight": "s"}}, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        required_shards({"weight_map": {"embed.weight": "../s"}}, "top")


def test_scan_rejects_bad_window_and_base_recipe():
    from scripts import deepseek_v41_gptq_scan as scan

    source = SimpleNamespace(args=SimpleNamespace(max_seq_len=4))
    with pytest.raises(ValueError, match="window"):
        scan.calibration_subset(source, None, None, 0)
    for recipe in (
        {**b.recipe(), "dense_bits": 8},
        {**b.recipe(), "low_token_bump": True},
    ):
        with pytest.raises(ValueError, match="scan requires"):
            scan.scan_metadata(source, recipe, {})


def test_atomic_destination_publication_rollback_and_race(tmp_path):
    from scripts.deepseek_v41_gptq_io import atomic_destination

    target = tmp_path / "build"
    with (
        pytest.raises(OSError, match="interrupted"),
        atomic_destination(target) as stage,
    ):
        stage.mkdir()
        (stage / "partial").write_text("unfinished")
        assert not target.exists()
        raise OSError("interrupted")
    assert not target.exists() and not list(tmp_path.glob(".build.partial-*"))
    with atomic_destination(target) as stage:
        stage.mkdir()
        (stage / "complete").write_text("done")
        assert not target.exists()
    assert (target / "complete").read_text() == "done"
    competitor = tmp_path / "competitor"
    with pytest.raises(FileExistsError), atomic_destination(competitor) as stage:
        stage.mkdir()
        competitor.mkdir()
        inode = competitor.stat().st_ino
    assert competitor.stat().st_ino == inode
    assert not list(tmp_path.glob(".competitor.partial-*"))
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError), atomic_destination(broken):
        pytest.fail("must not enter a symlink destination")
    assert broken.is_symlink()


def test_exclusive_publication_platform_contract(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    rename = Mock(return_value=0)
    monkeypatch.setattr(
        io.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(renameat2=rename)
    )
    with monkeypatch.context() as patch:
        patch.setattr(io.sys, "platform", "linux")
        io.rename_directory_exclusive(tmp_path / "source", tmp_path / "dest")
        assert rename.call_args.args[-1] == 1
        assert rename.call_args.args[1].startswith(b"/")
    with monkeypatch.context() as patch:
        patch.setattr(io.sys, "platform", "unsupported")
        with pytest.raises(RuntimeError, match="unavailable"):
            io.rename_directory_exclusive(tmp_path / "source", tmp_path / "dest")


@pytest.mark.parametrize("expert,shape", [(None, (4,)), (0, (1, 2))])
def test_shard_sink_rejects_equal_bytes_wrong_shape(tmp_path, expert, shape):
    from scripts.deepseek_v41_gptq_io import ShardSink
    from scripts.deepseek_v41_rd_build import Entry

    sink = ShardSink(
        tmp_path / "bank.safetensors", [Entry("bank", (2, 2), "F32", (), "test")]
    )
    try:
        before = sink.file.tell()
        with pytest.raises(ValueError, match="range shape"):
            sink.put("bank", mx.ones(shape), expert)
        assert sink.file.tell() == before and not sink.written
    finally:
        sink.close()


def test_publication_syncs_parent_after_file_is_visible(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io
    from scripts.deepseek_v41_rd_build import Entry

    observed = []
    real = io.sync_parent

    def sync(path):
        assert path.exists()
        observed.append(path.name)
        real(path)

    monkeypatch.setattr(io, "sync_parent", sync)
    io.atomic_json(tmp_path / "record.json", {})
    io.save_state(tmp_path / "state", {"value": mx.ones((1,))})
    sink = io.ShardSink(
        tmp_path / "bank.safetensors", [Entry("bank", (2,), "F32", (), "test")]
    )
    sink.put("bank", mx.ones((2,)))
    sink.finish()
    with io.atomic_destination(tmp_path / "complete") as stage:
        stage.mkdir()
    assert observed == [
        "record.json",
        io.state_snapshot(tmp_path / "state")[1].name,
        "state.json",
        "bank.safetensors",
        "complete",
    ]


def test_directory_sync_closes_descriptor_after_failure(tmp_path, monkeypatch):
    import os

    real_close = os.close
    closed = []

    def close(fd):
        closed.append(fd)
        real_close(fd)

    def fail(fd):
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(f.os, "fsync", fail)
    monkeypatch.setattr(f.os, "close", close)
    with pytest.raises(OSError, match="directory sync"):
        f.sync_parent(tmp_path / "file")
    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])


def test_remote_operation_deadlines_and_keepalives(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        response = (
            {"command_finished": True, "returncode": 0, "timed_out": False}
            if json.loads(kw.get("input", "{}")).get("op") == "run"
            else True
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(response))

    monkeypatch.setattr(f.subprocess, "run", run)
    remote = f.Remote("example.test", "python3")
    remote.call("info", path="data")
    assert calls[-1][1]["timeout"] == f.METADATA_TIMEOUT
    assert "ServerAliveInterval=30" in calls[-1][0]
    assert "ServerAliveCountMax=3" in calls[-1][0]
    remote.call("run", argv=["python3"], cwd="/work", log="/work/log")
    assert calls[-1][1]["timeout"] > f.COMMAND_TIMEOUT
    assert json.loads(calls[-1][1]["input"])["command_timeout"] == f.COMMAND_TIMEOUT
    remote.transfer(tmp_path / "a", "/work/a", True)
    assert calls[-1][1]["timeout"] == f.TRANSFER_TIMEOUT
    assert "ServerAliveInterval=30" in calls[-1][0][2]

    def timeout(*args, **kwargs):
        raise f.subprocess.TimeoutExpired("fixture", 1, stderr="sensitive-output")

    monkeypatch.setattr(f.subprocess, "run", timeout)
    for action in (
        lambda: remote.call("info", path="data"),
        lambda: remote.transfer(tmp_path / "a", "/work/a", True),
    ):
        with pytest.raises(RuntimeError, match="timed out") as error:
            action()
        assert "sensitive-output" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [b"bad", (2**63).to_bytes(8, "little"), (10).to_bytes(8, "little") + b"{}"],
)
def test_inventory_rejects_unbounded_headers(tmp_path, payload):
    from scripts.deepseek_v41_gptq_io import PartialSource, inventory
    from tests.test_deepseek_v41_streaming_teacher import write_source

    root = tmp_path / "source"
    write_source(root)
    manifest = inventory(root)
    shard = root / "source.safetensors"
    shard.write_bytes(payload)
    manifest["shards"][shard.name]["size"] = len(payload)
    with pytest.raises(ValueError, match="header"):
        inventory(root)
    with pytest.raises(ValueError, match="header"):
        PartialSource(root, manifest).verify(0)


def test_header_rejects_truncation_during_read(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    path = tmp_path / "shard"
    path.write_bytes((2).to_bytes(8, "little") + b"{}")
    real_open = type(path).open

    class Truncated:
        def __enter__(self):
            self.stream = real_open(path, "rb")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            return self.stream.read(size) if size == 8 else b""

    monkeypatch.setattr(type(path), "open", lambda *a, **kw: Truncated())
    with pytest.raises(ValueError, match="truncated safetensors header"):
        io.read_header(path)


@pytest.mark.parametrize("stop", [None, 0, -1, "partial"])
def test_shard_sink_handles_short_writes(tmp_path, stop):
    from scripts.deepseek_v41_gptq_io import ShardSink
    from scripts.deepseek_v41_rd_build import Entry

    path = tmp_path / "out.safetensors"
    sink = ShardSink(path, [Entry("value", (4,), "F32", (), "test")])
    real = sink.file

    class Writer:
        def __getattr__(self, name):
            return getattr(real, name)

        def write(self, data):
            if stop == "partial":
                return real.write(data[:3])
            return stop

    sink.file = Writer()
    try:
        if stop == "partial":
            sink.put("value", mx.arange(4, dtype=mx.float32))
            sink.finish()
            assert mx.array_equal(mx.load(str(path))["value"], mx.arange(4))
        else:
            with pytest.raises(OSError, match="short shard write"):
                sink.put("value", mx.zeros((4,)))
            assert not sink.written and not path.exists()
            with pytest.raises(ValueError, match="incomplete"):
                sink.finish()
    finally:
        sink.close()


def test_direct_assembly_rejects_cross_volume_before_conversion(tmp_path, monkeypatch):
    parts = tmp_path / "parts"
    parts.mkdir()
    destination = tmp_path / "output"
    real_stat = type(parts).stat
    monkeypatch.setattr(
        type(parts),
        "stat",
        lambda p, *a, **kw: (
            SimpleNamespace(st_dev=-1) if p == parts else real_stat(p, *a, **kw)
        ),
    )
    conversion = Mock()
    monkeypatch.setattr(b, "_assemble", conversion)
    with pytest.raises(ValueError, match="same filesystem"):
        b.assemble(None, None, parts, destination, None)
    conversion.assert_not_called()
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [parts]


def test_recipe_mutable_defaults_and_overrides_are_isolated():
    first = b.recipe()
    first["dense_8bit_layers"].append(123)
    assert 123 not in b.recipe()["dense_8bit_layers"]
    overrides = {"gate_up_3bit_layers": [1]}
    second = b.recipe(overrides)
    second["gate_up_3bit_layers"].append(2)
    assert overrides == {"gate_up_3bit_layers": [1]}


def test_restart_preserves_shared_array_identity(tmp_path):
    from scripts.deepseek_v41_gptq_io import load_state, save_state

    value = mx.arange(4).astype(mx.bfloat16)
    separate = mx.array(value)
    shared = SimpleNamespace(cache=value)
    state = {"value": value, "nested": [shared, value, separate], "shared": shared}
    path = tmp_path / "state"
    save_state(path, state)
    loaded = load_state(path)
    assert loaded["value"] is loaded["nested"][0].cache
    assert loaded["value"] is loaded["nested"][1]
    assert loaded["shared"] is loaded["nested"][0]
    assert loaded["value"] is not loaded["nested"][2]
    assert loaded["value"].dtype == mx.bfloat16
    assert mx.array_equal(loaded["value"], value)
    from scripts.deepseek_v41_gptq_io import state_snapshot

    assert len(mx.load(str(state_snapshot(path)[1]))) == 2


def test_state_publication_keeps_previous_generation_after_interruption(
    tmp_path, monkeypatch
):
    from scripts import deepseek_v41_gptq_io as io

    path = tmp_path / "state"
    io.save_state(path, {"value": mx.array([1])})
    previous = io.state_snapshot(path)[1]
    real = io.atomic_json
    monkeypatch.setattr(io, "atomic_json", Mock(side_effect=OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        io.save_state(path, {"different": mx.array([2, 3])})
    assert io.load_state(path)["value"].item() == 1
    assert io.state_snapshot(path)[1] == previous
    monkeypatch.setattr(io, "atomic_json", real)
    io.save_state(path, {"different": mx.array([2, 3])})
    assert io.load_state(path)["different"].tolist() == [2, 3]
    payload = io.state_snapshot(path)[1]
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        io.load_state(path)
    io.atomic_json(path.with_suffix(".json"), {"tree": {}, "tensor_sha256": "../bad"})
    with pytest.raises(ValueError, match="invalid state payload"):
        io.load_state(path)


def test_memory_guard_restores_limit_if_cache_setup_fails(monkeypatch):
    memory = Mock(return_value=123)
    monkeypatch.setattr(mx, "set_memory_limit", memory)
    monkeypatch.setattr(mx, "set_cache_limit", Mock(side_effect=RuntimeError("cache")))
    with pytest.raises(RuntimeError, match="cache"):
        b.MemoryGuard(b.recipe())
    assert memory.call_args.args == (123,)


def test_verified_copy_isolates_output_and_rejects_corruption(tmp_path):
    from scripts.deepseek_v41_gptq_io import copy_verified, file_hash

    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"original")
    copy_verified(source, target, file_hash(source))
    assert source.stat().st_ino != target.stat().st_ino
    target.write_bytes(b"mutated")
    assert source.read_bytes() == b"original"
    with pytest.raises(ValueError, match="copied shard payload mismatch"):
        copy_verified(source, target, "incorrect")


def test_memory_guard_restores_limits_for_accounting_error(monkeypatch):
    memory, cache = Mock(return_value=123), Mock(return_value=456)
    monkeypatch.setattr(mx, "set_memory_limit", memory)
    monkeypatch.setattr(mx, "set_cache_limit", cache)
    monkeypatch.setattr(b.MemoryGuard, "check", Mock(side_effect=OSError("accounting")))
    with pytest.raises(OSError, match="accounting"), b.MemoryGuard(b.recipe()):
        pass
    assert memory.call_args.args == (123,)
    assert cache.call_args.args == (456,)


def test_projection_error_matches_weighted_gram_objective():
    import numpy as np

    from scripts.deepseek_v41_gptq_quant import gram

    rng = np.random.default_rng(5037)
    x = rng.normal(size=(7, 32)).astype(np.float32)
    weight = mx.array(rng.normal(size=(3, 32)).astype(np.float32))
    gates = np.array([0.01, 0.1, 0.2, 0.4, 0.7, 0.9, 1.8], np.float32)
    packed = mx.quantize(weight, bits=2, group_size=32)
    result = b.projection_errors(weight, packed, 2, lambda: iter([(x, gates)]), 32)
    w = np.asarray(weight).astype(np.float64)
    q = np.asarray(mx.dequantize(*packed, bits=2, group_size=32)).astype(np.float64)
    h, _ = gram([(x, gates)], 32)
    energy = np.sum((x @ w.T) ** 2 * gates[:, None])
    error = np.sum((x @ (w - q).T) ** 2 * gates[:, None])
    assert result["calibration_output_energy"] == pytest.approx(energy, rel=1e-5)
    assert result["gptq_relative_output_error"] == pytest.approx(
        error / energy, rel=1e-5
    )
    assert error == pytest.approx(
        np.trace((w - q) @ np.asarray(h) @ (w - q).T), rel=1e-5
    )
    assert energy != pytest.approx(np.sum((x @ w.T) ** 2 * gates[:, None] ** 2))


def test_restart_reclaims_only_unreferenced_payload_generations(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    path = tmp_path / "state"
    io.save_state(path, {"value": mx.array([1])})
    old = io.state_snapshot(path)[1]
    unrelated = tmp_path / "state.not-a-generation.safetensors"
    unrelated.write_bytes(b"preserve")
    real = io.atomic_json
    monkeypatch.setattr(io, "atomic_json", Mock(side_effect=OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        io.save_state(path, {"value": mx.array([2])})
    assert len(list(tmp_path.glob("state.*.safetensors"))) == 3
    assert io.load_state(path)["value"].item() == 1
    assert set(tmp_path.glob("state.*.safetensors")) == {old, unrelated}
    monkeypatch.setattr(io, "atomic_json", real)
    io.save_state(path, {"value": mx.array([3])})
    current = io.state_snapshot(path)[1]
    assert not old.exists()
    assert set(tmp_path.glob("state.*.safetensors")) == {current, unrelated}
    assert io.load_state(path)["value"].item() == 3


def test_verified_copy_syncs_file_before_directory(tmp_path, monkeypatch):
    import stat

    from scripts import deepseek_v41_gptq_io as io

    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"payload")
    synced = []
    real_sync = io.os.fsync

    def sync(fd):
        synced.append("directory" if stat.S_ISDIR(io.os.fstat(fd).st_mode) else "file")
        real_sync(fd)

    monkeypatch.setattr(io.os, "fsync", sync)
    io.copy_verified(source, target, io.file_hash(source))
    assert synced == ["file", "directory"]


def test_assembly_flushes_supporting_files_before_publication(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    events = []
    sync_files, rename = io.sync_files, io.rename_directory_exclusive

    def sync(paths):
        paths = list(paths)
        events.append(("sync", sorted(path.name for path in paths)))
        sync_files(paths)

    def publish(source, destination):
        assert events == [("sync", ["README.md", "layer-0.json"])]
        events.append(("publish", destination.name))
        rename(source, destination)

    monkeypatch.setattr(io, "sync_files", sync)
    monkeypatch.setattr(io, "rename_directory_exclusive", publish)
    with io.atomic_destination(tmp_path / "build") as stage:
        (stage / "statistics").mkdir(parents=True)
        (stage / "README.md").write_text("report")
        (stage / "statistics" / "layer-0.json").write_text("{}")
    assert events[-1] == ("publish", "build")


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "/outside", "a/b", "a\\b", 7]
)
def test_artifact_names_cannot_escape_parts(tmp_path, monkeypatch, name):
    from scripts import deepseek_v41_gptq_io as io

    monkeypatch.setattr(io, "file_hash", lambda _: pytest.fail("unsafe artifact read"))
    with pytest.raises(ValueError, match="unsafe checkpoint artifact"):
        io.verify_artifact(tmp_path, {"files": {name: "hash"}})


def test_artifact_symlink_cannot_escape_parts(tmp_path, monkeypatch):
    from scripts import deepseek_v41_gptq_io as io

    parts = tmp_path / "parts"
    parts.mkdir()
    (tmp_path / "outside").write_text("private")
    (parts / "layer.bin").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(io, "file_hash", lambda _: pytest.fail("outside artifact read"))
    with pytest.raises(ValueError, match="escapes parts"):
        io.verify_artifact(parts, {"files": {"layer.bin": "hash"}})


@pytest.mark.parametrize("termination", ["watchdog", "process_group"])
def test_atomic_assembly_cleans_after_abrupt_exit(tmp_path, termination):
    import subprocess
    import time

    from scripts.deepseek_v41_gptq_io import atomic_destination

    target = tmp_path / "assembled"
    retained = tmp_path / "checkpoint.json"
    retained.write_text("committed boundary")
    code = """import os,signal,sys
from pathlib import Path
from scripts.deepseek_v41_gptq_io import atomic_destination
with atomic_destination(Path(sys.argv[1])) as stage:
    stage.mkdir()
    (stage / "payload").write_bytes(b"unpublished")
    if sys.argv[2] == "watchdog":
        os._exit(86)
    os.killpg(os.getpgrp(), signal.SIGTERM)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(target), termination],
        start_new_session=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == (86 if termination == "watchdog" else -15)
    deadline = time.monotonic() + 10
    while list(tmp_path.glob(".assembled.partial-*")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not list(tmp_path.glob(".assembled.partial-*"))
    assert not target.exists()
    assert retained.read_text() == "committed boundary"
    with atomic_destination(target) as stage:
        stage.mkdir()
        (stage / "payload").write_bytes(b"complete")
    assert (target / "payload").read_bytes() == b"complete"


def test_atomic_assembly_rejects_failed_cleanup_helper(tmp_path, monkeypatch):
    import io

    from scripts import deepseek_v41_gptq_io as storage

    helper = SimpleNamespace(stdout=io.BytesIO(b""))
    monkeypatch.setattr(
        storage.subprocess, "Popen", lambda *a, **k: nullcontext(helper)
    )
    with (
        pytest.raises(RuntimeError, match="cleanup helper"),
        storage.atomic_destination(tmp_path / "assembled"),
    ):
        pytest.fail("must not start writing without cleanup supervision")
    assert not list(tmp_path.iterdir())
