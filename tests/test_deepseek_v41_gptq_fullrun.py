"""CPU-only lifecycle and interrupted-export/resume contract; no SSH or MLX."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import deepseek_v41_gptq_fullrun as f


@pytest.fixture(autouse=True)
def local_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "SCRATCH_ROOT", tmp_path / "driver-scratch")


def manifest(n=40):
    tensors = {f"layers.{i}.weight": {"path": f"layer-{i}.bin"} for i in range(n)}
    tensors.update(
        {
            "embed.weight": {"path": "embed.bin"},
            "head.weight": {"path": "head.bin"},
            "mtp.weight": {"path": "mtp.bin"},
            "layers.0.scale": {"path": "shared.bin"},
            f"layers.{n - 1}.scale": {"path": "shared.bin"},
            "layers.1.head_scale": {"path": "head.bin"},
        }
    )
    return {
        "config": {"text_config": {"num_hidden_layers": n}},
        "source_revision": "test",
        "tensors": tensors,
        "shards": {r["path"]: {"size": 4} for r in tensors.values()},
    }


def test_lifecycle_shared_and_nonlayer():
    p = f.lifecycle(manifest())
    assert {"embed.bin", "head.bin", "mtp.bin"} <= set(f.needed(p, 0, 40))
    assert "layer-1.bin" in f.needed(p, 0, 40)
    assert "layer-2.bin" not in f.needed(p, 0, 40)
    assert "shared.bin" not in f.deletable(
        p, 0, {"identity": "r", "next_layer": 1}, "r"
    )
    assert "shared.bin" in f.deletable(p, 39, {"identity": "r", "next_layer": 40}, "r")
    assert "head.bin" not in f.deletable(
        p, 39, {"identity": "r", "next_layer": 40}, "r"
    )


@pytest.mark.parametrize(
    "cp",
    [
        {"identity": "r", "next_layer": 0, "phase": "ffn"},
        {"identity": "other", "next_layer": 1},
        {"identity": "r", "next_layer": True},
    ],
)
def test_no_cleanup_before_authoritative_commit(cp):
    with pytest.raises(ValueError):
        f.deletable(f.lifecycle(manifest()), 0, cp, "r")


@pytest.mark.parametrize(
    "change", ["traversal", "missing", "unknown_layer", "unmapped"]
)
def test_bad_manifest_fails_closed(change):
    m = manifest()
    if change == "traversal":
        m["tensors"]["layers.0.weight"]["path"] = "../victim"
    if change == "missing":
        m["tensors"]["layers.0.weight"]["path"] = "absent"
    if change == "unknown_layer":
        m["tensors"]["layers.40.weight"] = {"path": "head.bin"}
    if change == "unmapped":
        m["shards"]["unmapped"] = {"size": 4}
    with pytest.raises(ValueError):
        f.lifecycle(m)


class FakeRemote:
    def __init__(self, root, args):
        self.root, self.a = root, args
        self.events = []
        self.fail_export = False
        self.no_commit = False
        self.fail_init = False
        self.free = f.HEADROOM + 10000000
        self.runs = []
        self.path(args.remote_work).mkdir(parents=True)
        root = Path(f.__file__).parent.parent
        for name in f.tool_manifest(root):
            target = self.path(args.remote_work + "/repo/" + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)

    def path(self, p):
        return self.root / str(p).lstrip("/")

    def call(self, op, **kw):
        self.events.append((op, kw))
        path = self.path(kw["path"]) if "path" in kw else None
        if op == "tools":
            return f.tool_manifest(path)
        if op == "free":
            return self.free
        if op == "json":
            return json.loads(path.read_text()) if path.exists() else None
        if op == "info":
            return (
                {"size": path.stat().st_size, "sha256": f.sha(path)}
                if path.exists()
                else None
            )
        if op == "mkdir":
            path.mkdir(parents=True, exist_ok=True)
            return True
        if op == "lock":
            path.mkdir()
            return True
        if op == "unlock":
            path.rmdir()
            return True
        if op == "delete":
            root = self.path(kw["root"]).resolve()
            assert path.resolve() != root and path.resolve().is_relative_to(root)
            cp = json.loads(self.path(kw["checkpoint"]).read_text())
            assert cp["identity"] == kw["identity"] and cp["next_layer"] > kw["layer"]
            # Export marker and all its hashes must exist before ANY deletion.
            marker = self.a.destination / f"layer-{kw['layer']}-export.json"
            record = json.loads(marker.read_text())
            assert all(
                f.sha(self.a.destination / n) == h for n, h in record["files"].items()
            )
            path.unlink(missing_ok=True)
            return True
        if op == "recover_init":
            assert not (path / "state/checkpoint.json").exists()
            archived = []
            for name in ("parts", "state"):
                p = path / name
                if p.exists():
                    target = path / ("abandoned-init-test-" + name)
                    p.rename(target)
                    archived.append(target.name)
            return archived
        if op == "run":
            argv = kw["argv"]
            cmd = argv[2]

            def arg(name):
                return argv[argv.index(name) + 1]

            parts = self.path(arg("--parts"))
            state = self.path(arg("--scratch"))
            if cmd == "init":
                assert not parts.exists() and not state.exists()
                parts.mkdir(parents=True)
                state.mkdir(parents=True)
                recipe = f.resolved_recipe(self.path(arg("--recipe")))
                source = json.loads(self.path(arg("--manifest")).read_text())
                reference = json.loads(
                    (self.path(arg("--reference")) / "reference.json").read_text()
                )
                metadata = {
                    "recipe": recipe,
                    "source_identity": f.digest(source),
                    "heldout_reference_identity": reference["identity"],
                }
                self.identity = f.digest(metadata)
                f.atomic(parts / "run.json", {**metadata, "identity": self.identity})
                if self.fail_init:
                    self.fail_init = False
                    raise RuntimeError("interrupted init")
                f.atomic(
                    state / "checkpoint.json",
                    {"identity": self.identity, "next_layer": 0},
                )
                return True
            layer = int(arg("--layer"))
            self.runs.append(layer)
            cp = json.loads((state / "checkpoint.json").read_text())
            assert cp["next_layer"] == layer
            m = json.loads(self.a.manifest.read_text())
            for name in f.needed(f.lifecycle(m), layer, 40):
                assert (self.path(arg("--source")) / name).exists()
            files = {}
            for name in [
                f"model-layer-{layer}.safetensors",
                f"layer-{layer}-stats.json",
            ]:
                (parts / name).write_text(str(layer))
                files[name] = f.sha(parts / name)
            f.atomic(
                parts / f"layer-{layer}-complete.json",
                {"run_identity": self.identity, "files": files},
            )
            with (parts / "heldout-drift.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "layer": layer,
                            "candidate_identity": f.digest(
                                dict(kind="gptq_parts", run=self.identity)
                            ),
                            "reference_identity": "ref",
                        }
                    )
                    + "\n"
                )
            if not self.no_commit:
                f.atomic(
                    state / "checkpoint.json",
                    {"identity": self.identity, "next_layer": layer + 1},
                )
            return True
        raise AssertionError(op)

    def transfer(self, source, target, upload):
        self.events.append(("transfer", {"upload": upload, "source": str(source)}))
        if (
            self.fail_export
            and not upload
            and str(source).endswith("model-layer-0.safetensors")
        ):
            self.fail_export = False
            raise RuntimeError("interrupted export")
        source = Path(source) if upload else self.path(source)
        target = self.path(target) if upload else Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


@pytest.fixture
def setup(tmp_path):
    m = manifest()
    source = tmp_path / "source"
    source.mkdir()
    for name in m["shards"]:
        (source / name).write_bytes(b"data")
    for name in ("config.json", "tokenizer.json"):
        (source / name).write_text("{}")
    mp = tmp_path / "source-manifest.json"
    f.atomic(mp, m)
    recipe = tmp_path / "recipe.json"
    f.atomic(recipe, {"pilot": False, "dense_8bit_layers": list(range(8))})
    corpus = tmp_path / "calibration.json"
    f.atomic(corpus, {"slices": {"general": ["text.txt"]}})
    (tmp_path / "text.txt").write_text("corpus")
    reference = tmp_path / "reference"
    reference.mkdir()
    f.atomic(
        reference / "reference.json",
        {"identity": "ref", "source_identity": f.digest(m)},
    )
    for i in range(41):
        boundary = reference / f"boundary-{i}"
        boundary.mkdir()
        (boundary / "tensor").write_text(str(i))
        f.atomic(
            boundary / "complete.json",
            {"identity": "ref", "files": {"tensor": f.sha(boundary / "tensor")}},
        )
    a = SimpleNamespace(
        manifest=mp,
        recipe=recipe,
        source=source,
        reference_source=reference,
        calibration=corpus,
        destination=tmp_path / "collected",
        remote_work="/m5",
        remote_python="/python",
    )
    return a, FakeRemote(tmp_path / "remote", a)


def test_full_run_then_idempotent_resume(setup):
    a, r = setup
    f.Driver(a, r).run()
    assert r.runs == list(range(40))
    assert len(list(a.destination.glob("layer-*-export.json"))) == 40
    assert (
        json.loads((a.destination / "fullrun-complete.json").read_text())["next_layer"]
        == 40
    )
    assert r.path("/m5/stage/embed.bin").exists()
    assert r.path("/m5/stage/head.bin").exists()
    assert not r.path("/m5/stage/shared.bin").exists()
    before = list(r.runs)
    f.Driver(a, r).run()
    assert r.runs == before


def test_resume_after_checkpoint_before_export(setup):
    a, r = setup
    r.fail_export = True
    with pytest.raises(RuntimeError, match="interrupted export"):
        f.Driver(a, r).run()
    assert (
        json.loads(r.path("/m5/fullrun/state/checkpoint.json").read_text())[
            "next_layer"
        ]
        == 1
    )
    assert r.path("/m5/stage/layer-0.bin").exists()
    assert not any(op == "delete" for op, _ in r.events)
    f.Driver(a, r).run()
    assert r.runs == list(range(40))  # layer zero never reruns after commit


def test_drift_and_completion_without_checkpoint_never_export_or_delete(setup):
    a, r = setup
    r.no_commit = True
    with pytest.raises(ValueError, match="commit exactly one layer"):
        f.Driver(a, r).run()
    assert not list(a.destination.glob("layer-*-export.json"))
    assert not any(op == "delete" for op, _ in r.events)


def test_transfer_requires_disk_reserve(setup):
    a, r = setup
    r.free = f.HEADROOM + 3
    driver = f.Driver(a, r)
    with pytest.raises(ValueError, match="insufficient M5 disk"):
        driver.upload(a.source / "layer-0.bin", "/m5/stage/layer-0.bin")
    assert not any(op == "transfer" for op, _ in r.events)


def test_bad_transfer_hash_blocks_cleanup(setup):
    a, r = setup
    driver = f.Driver(a, r)
    with pytest.raises(ValueError, match="local source hash mismatch"):
        driver.upload(
            a.source / "layer-0.bin",
            "/m5/stage/layer-0.bin",
            {"size": 4, "sha256": "bad"},
        )
    assert not any(op == "delete" for op, _ in r.events)


def test_changed_recipe_fails_closed_on_resume(setup):
    a, r = setup
    r.fail_export = True
    with pytest.raises(RuntimeError):
        f.Driver(a, r).run()
    c = json.loads(a.recipe.read_text())
    c["dense_8bit_layers"] = list(range(1, 9))
    f.atomic(a.recipe, c)
    with pytest.raises(ValueError, match="inputs changed"):
        f.Driver(a, r).run()
    assert not any(op == "delete" for op, _ in r.events)


def test_resume_interrupted_initialization_without_deleting_partial_state(setup):
    a, r = setup
    r.fail_init = True
    with pytest.raises(RuntimeError, match="interrupted init"):
        f.Driver(a, r).run()
    assert r.path("/m5/fullrun/parts/run.json").exists()
    assert not r.path("/m5/fullrun/state/checkpoint.json").exists()
    f.Driver(a, r).run()
    assert r.runs == list(range(40))
    assert r.path("/m5/fullrun/abandoned-init-test-parts/run.json").exists()


def test_remote_helper_header_hash_and_checkpoint_gate(tmp_path):
    """Execute the actual remote helper locally, still without ssh or a model."""
    import hashlib
    import struct
    import subprocess
    import sys

    shard = tmp_path / "shard"
    header = b'{"tensor": {}}'
    raw = struct.pack("<Q", len(header)) + header
    shard.write_bytes(raw + b"payload")

    def request(**q):
        return subprocess.run(
            [sys.executable, "-c", f.REMOTE],
            input=json.dumps(q),
            text=True,
            capture_output=True,
        )

    result = request(
        op="info",
        path=str(shard),
        meta={
            "size": shard.stat().st_size,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "sha256": f.sha(shard),
        },
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "size": len(raw) + 7,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "sha256": f.sha(shard),
    }
    cp = tmp_path / "checkpoint.json"
    f.atomic(cp, {"identity": "run", "next_layer": 0})
    q = dict(
        op="delete",
        path=str(shard),
        root=str(tmp_path),
        checkpoint=str(cp),
        identity="run",
        layer=0,
    )
    assert request(**q).returncode != 0
    assert shard.exists()
    f.atomic(cp, {"identity": "run", "next_layer": 1})
    assert request(**q).returncode == 0
    assert not shard.exists()


def test_source_header_mismatch_after_transfer_fails_closed(setup):
    a, r = setup
    driver = f.Driver(a, r)
    # Fake transfer supplies a size/SHA but cannot supply the expected source header.
    with pytest.raises(ValueError, match="verification failed"):
        driver.upload(
            a.source / "layer-0.bin",
            "/m5/stage/layer-0.bin",
            {"size": 4, "header_sha256": "wrong"},
        )
    assert not any(op == "delete" for op, _ in r.events)


@pytest.mark.parametrize("kind", ["empty", "layer_name", "coverage", "size"])
def test_invalid_lifecycle_metadata(kind):
    m = manifest()
    if kind == "empty":
        m["shards"] = {}
    elif kind == "layer_name":
        m["tensors"]["layers.bad.weight"] = {"path": "head.bin"}
    elif kind == "coverage":
        m["tensors"].pop("layers.2.weight")
        m["tensors"]["head.extra"] = {"path": "layer-2.bin"}
    else:
        m["shards"]["head.bin"]["size"] = 0
    with pytest.raises(ValueError):
        f.lifecycle(m)


@pytest.mark.parametrize("kind", ["layers", "recipe", "unknown"])
def test_driver_rejects_invalid_config(setup, kind):
    a, r = setup
    if kind == "layers":
        f.atomic(a.manifest, manifest(4))
    elif kind == "recipe":
        f.atomic(a.recipe, {"pilot": True})
    else:
        f.atomic(a.recipe, {"unknown": 1})
    with pytest.raises(ValueError):
        f.Driver(a, r)


@pytest.mark.parametrize(
    "kind",
    ["source", "boundary", "size", "destination", "remote", "tooling", "initialized"],
)
def test_driver_preflight_rejections(setup, kind, monkeypatch):
    a, r = setup
    driver = f.Driver(a, r)
    if kind == "source":
        f.atomic(
            a.reference_source / "reference.json",
            {"identity": "ref", "source_identity": "bad"},
        )
    elif kind == "boundary":
        f.atomic(a.reference_source / "boundary-0/complete.json", {"identity": "bad"})
    elif kind == "size":
        (a.source / "layer-0.bin").write_bytes(b"changed")
    elif kind == "destination":
        a.destination.mkdir()
        (a.destination / "unbound").touch()
    elif kind == "remote":
        path = r.path(driver.parts + "/run.json")
        path.parent.mkdir(parents=True)
        f.atomic(path, {})
    elif kind == "tooling":
        r.path(a.remote_work + "/repo/scripts/deepseek_v41_gptq.py").write_text(
            "changed"
        )
    else:
        real = r.call

        def changed(op, **kw):
            result = real(op, **kw)
            if op == "json" and kw.get("path") == driver.parts + "/run.json" and result:
                result["source_identity"] = "bad"
            return result

        monkeypatch.setattr(r, "call", changed)
    with pytest.raises(ValueError):
        driver.run()
    assert not any(op == "delete" for op, _ in r.events)


def test_upload_size_and_reference_identity_rejected(setup):
    a, r = setup
    driver = f.Driver(a, r)
    with pytest.raises(ValueError, match="size"):
        driver.upload(a.source / "layer-0.bin", "/work/file", {"size": 1})
    driver.ref_identity = "different"
    with pytest.raises(ValueError, match="boundary identity"):
        driver.stage_reference(0)


@pytest.mark.parametrize(
    "kind",
    [
        "uncommitted",
        "files",
        "remote_hash",
        "transfer_hash",
        "drift_hash",
        "drift_rows",
    ],
)
def test_export_rejects_corrupt_state_without_cleanup(setup, monkeypatch, kind):
    a, r = setup
    driver = f.Driver(a, r)
    r.fail_export = True
    with pytest.raises(RuntimeError, match="interrupted export"):
        driver.run()
    r.events.clear()
    if kind == "uncommitted":
        f.atomic(
            r.path(driver.state + "/checkpoint.json"),
            {"identity": driver.identity, "next_layer": 0},
        )
    elif kind == "files":
        path = r.path(driver.parts + "/layer-0-complete.json")
        value = json.loads(path.read_text())
        value["files"] = {}
        f.atomic(path, value)
    elif kind == "remote_hash":
        r.path(driver.parts + "/model-layer-0.safetensors").write_bytes(b"changed")
    elif kind in ("transfer_hash", "drift_hash"):
        real = r.transfer

        def corrupt(source, target, upload):
            real(source, target, upload)
            if not upload and (
                (kind == "transfer_hash" and str(source).endswith(".safetensors"))
                or (kind == "drift_hash" and str(source).endswith(".jsonl"))
            ):
                Path(target).write_bytes(b"changed")

        monkeypatch.setattr(r, "transfer", corrupt)
    else:
        path = r.path(driver.parts + "/heldout-drift.jsonl")
        path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError):
        driver.export(0)
    assert not any(op == "delete" for op, _ in r.events)


def test_metadata_transfer_corruption_is_rejected(setup, monkeypatch):
    a, r = setup
    driver = f.Driver(a, r)
    a.destination.mkdir()
    path = r.path(driver.parts + "/run.json")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    real = r.transfer

    def corrupt(source, target, upload):
        real(source, target, upload)
        Path(target).write_text("changed")

    monkeypatch.setattr(r, "transfer", corrupt)
    with pytest.raises(ValueError, match="metadata hash"):
        driver.collect_metadata()
    assert not (a.destination / "run.json").exists()


@pytest.mark.parametrize(
    "slices",
    [
        {"../../outside": ["text.txt"]},
        {"a/b": ["text.txt"]},
        {"": ["text.txt"]},
        [],
        {},
    ],
)
def test_unsafe_calibration_names_fail_before_remote_actions(setup, slices):
    a, remote = setup
    f.atomic(a.calibration, {"slices": slices})
    with pytest.raises(ValueError):
        f.Driver(a, remote).run()
    assert remote.events == []


@pytest.mark.parametrize(
    "kind", ["absolute", "parent", "symlink", "malformed", "empty"]
)
def test_calibration_paths_cannot_escape_manifest_directory(setup, tmp_path, kind):
    a, remote = setup
    outside = tmp_path / "outside.txt"
    outside.write_text("must not upload")
    directory = tmp_path / "calibration"
    directory.mkdir()
    a.calibration = directory / "manifest.json"
    if kind == "absolute":
        files = [str(outside)]
    elif kind == "parent":
        files = ["../outside.txt"]
    elif kind == "symlink":
        (directory / "link.txt").symlink_to(outside)
        files = ["link.txt"]
    elif kind == "malformed":
        files = "outside.txt"
    else:
        files = []
    f.atomic(a.calibration, {"slices": {"general": files}})
    with pytest.raises(ValueError):
        f.Driver(a, remote).run()
    assert remote.events == []


def test_remote_command_timeout_terminates_started_process(tmp_path):
    import os
    import subprocess
    import sys

    pid = tmp_path / "child.pid"
    command = (
        "import os,time; from pathlib import Path; Path("
        + repr(str(pid))
        + ").write_text(str(os.getpid())); time.sleep(10)"
    )
    result = subprocess.run(
        [sys.executable, "-c", f.REMOTE],
        input=json.dumps(
            dict(
                op="run",
                argv=[sys.executable, "-c", command],
                cwd=str(tmp_path),
                log=str(tmp_path / "command.log"),
                command_timeout=0.5,
            )
        ),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0
    completion = json.loads(result.stdout)
    assert completion["command_finished"] and completion["timed_out"]
    assert completion["returncode"] != 0
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


@pytest.mark.parametrize(
    "dependency",
    [
        "scripts/deepseek_v41_rd_build.py",
        "rapid_mlx/models/deepseek_v41_native/model.py",
    ],
)
def test_remote_imported_dependency_change_rejected_before_conversion(
    setup, dependency
):
    args, remote = setup
    remote.path(args.remote_work + "/repo/" + dependency).write_text("changed")
    with pytest.raises(ValueError, match="tooling differs"):
        f.Driver(args, remote).run()
    assert not remote.runs
    assert not any(op == "delete" for op, _ in remote.events)


def test_remote_tool_manifest_matches_local_and_detects_extra_code(tmp_path):
    import subprocess
    import sys

    for name in ("scripts/entry.py", "rapid_mlx/helper.py", "rapid_mlx/kernel.metal"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    cached = tmp_path / "scripts/__pycache__/entry.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"cache")
    expected = f.tool_manifest(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", f.REMOTE],
        input=json.dumps({"op": "tools", "path": str(tmp_path)}),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == expected
    (tmp_path / "rapid_mlx/extra.py").write_text("extra")
    assert f.tool_manifest(tmp_path) != expected


@pytest.mark.parametrize("volume", ["destination", "scratch"])
def test_local_export_disk_preflight_prevents_cleanup(setup, monkeypatch, volume):
    args, remote = setup
    driver = f.Driver(args, remote)
    remote.fail_export = True
    with pytest.raises(RuntimeError, match="interrupted export"):
        driver.run()
    remote.fail_export = False
    remote.events.clear()
    monkeypatch.setattr(
        f.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=0
            if (volume == "destination" or Path(path) != args.destination)
            else f.HEADROOM + 10000000
        ),
    )
    with pytest.raises(ValueError, match="local.*export space"):
        driver.export(0)
    assert not any(op == "delete" for op, _ in remote.events)


@pytest.mark.parametrize("failure", ["timeout", "exit", "json"])
def test_remote_command_uncertainty_stays_set_until_verified_success(
    monkeypatch, failure
):
    import subprocess

    remote = f.Remote("example.test", "python3")

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("ssh", 1)
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0, stdout="invalid"
        )

    monkeypatch.setattr(f.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError)):
        remote.call("run", argv=["python3"], cwd="/work", log="/work/log")
    assert remote.indeterminate
    monkeypatch.setattr(
        f.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="true"),
    )
    remote.call("info", path="data")
    assert remote.indeterminate


def test_driver_retains_remote_lock_after_indeterminate_command(setup, monkeypatch):
    args, remote = setup
    original = remote.call

    def call(op, **kw):
        if op == "run":
            remote.indeterminate = True
            raise RuntimeError("uncertain remote process")
        return original(op, **kw)

    monkeypatch.setattr(remote, "call", call)
    with pytest.raises(RuntimeError, match="uncertain remote"):
        f.Driver(args, remote).run()
    assert remote.path(args.remote_work + "/fullrun.lock").is_dir()
    assert not any(op in ("unlock", "delete") for op, _ in remote.events)


def test_export_flushes_payloads_before_committing_marker(setup, monkeypatch):
    import os

    args, remote = setup
    driver = f.Driver(args, remote)
    remote.fail_export = True
    with pytest.raises(RuntimeError, match="interrupted export"):
        driver.run()
    remote.fail_export = False
    synced = set()
    real_sync, real_atomic = f.os.fsync, f.atomic

    def sync(fd):
        synced.add(os.fstat(fd).st_ino)
        real_sync(fd)

    def atomic(path, value):
        if path.name == "layer-0-export.json":
            for name in ("model-layer-0.safetensors", "layer-0-stats.json"):
                assert (args.destination / name).stat().st_ino in synced
            assert args.destination.stat().st_ino in synced
        return real_atomic(path, value)

    monkeypatch.setattr(f.os, "fsync", sync)
    monkeypatch.setattr(f, "atomic", atomic)
    driver.export(0)
    assert (args.destination / "layer-0-export.json").exists()


@pytest.mark.parametrize("returncode", [0, 7])
def test_remote_helper_attests_completed_command_and_releases_uncertainty(
    tmp_path, monkeypatch, returncode
):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f.REMOTE],
        input=json.dumps(
            {
                "op": "run",
                "argv": [sys.executable, "-c", f"raise SystemExit({returncode})"],
                "cwd": str(tmp_path),
                "log": str(tmp_path / "command.log"),
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "command_finished": True,
        "returncode": returncode,
        "timed_out": False,
    }
    monkeypatch.setattr(f.subprocess, "run", lambda *a, **kw: result)
    remote = f.Remote("example.test", "python3")
    if returncode:
        with pytest.raises(RuntimeError, match="finished unsuccessfully"):
            remote.call("run", argv=[], cwd="/work", log="/work/log")
    else:
        assert remote.call("run", argv=[], cwd="/work", log="/work/log") is True
    assert not remote.indeterminate


def test_remote_rejects_unattested_success_response(monkeypatch):
    monkeypatch.setattr(
        f.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="true"),
    )
    remote = f.Remote("example.test", "python3")
    with pytest.raises(RuntimeError, match="completion is unverified"):
        remote.call("run", argv=[], cwd="/work", log="/work/log")
    assert remote.indeterminate


def test_exports_ignore_remote_working_space_after_commit(setup, monkeypatch):
    args, remote = setup
    driver = f.Driver(args, remote)
    remote.fail_export = True
    with pytest.raises(RuntimeError, match="interrupted export"):
        driver.run()
    remote.fail_export = False
    remote.free = 0
    monkeypatch.setattr(
        f.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=f.LOCAL_EXPORT_RESERVE + 1000000),
    )
    driver.export(0)
    driver.collect_metadata()
    assert (args.destination / "layer-0-export.json").is_file()
    assert (args.destination / "run.json").is_file()


@pytest.mark.parametrize("enough", [False, True])
def test_metadata_download_space_and_durable_publication(setup, monkeypatch, enough):
    import stat

    args, remote = setup
    driver = f.Driver(args, remote)
    args.destination.mkdir()
    source = remote.path(driver.parts + "/run.json")
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monkeypatch.setattr(
        f.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=f.LOCAL_EXPORT_RESERVE + (2 if enough else 1)),
    )
    events = []
    real_sync, real_log = f.os.fsync, driver.log

    def sync(fd):
        events.append("directory" if stat.S_ISDIR(f.os.fstat(fd).st_mode) else "file")
        real_sync(fd)

    def log(event, **kwargs):
        if event == "metadata_verified":
            assert events == ["file", "directory"]
        real_log(event, **kwargs)

    monkeypatch.setattr(f.os, "fsync", sync)
    monkeypatch.setattr(driver, "log", log)
    if enough:
        driver.collect_metadata()
        assert (args.destination / "run.json").read_text() == "{}"
    else:
        with pytest.raises(ValueError, match="local metadata export space"):
            driver.collect_metadata()
        assert not (args.destination / "run.json").exists()
        assert not any(op == "transfer" for op, _ in remote.events)


@pytest.mark.parametrize(
    "kind", ["outside", "root", "directory", "symlink", "parent_symlink", "missing"]
)
def test_remote_cleanup_is_confined_to_authorized_files(tmp_path, kind):
    import subprocess
    import sys

    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("keep")
    cp = tmp_path / "checkpoint.json"
    f.atomic(cp, {"identity": "run", "next_layer": 1})
    target = root / "entry"
    if kind == "outside":
        target = outside
    elif kind == "root":
        target = root
    elif kind == "directory":
        target.mkdir()
    elif kind == "symlink":
        target.symlink_to(outside)
    elif kind == "parent_symlink":
        target.symlink_to(tmp_path, target_is_directory=True)
        target = target / "outside"
    result = subprocess.run(
        [sys.executable, "-c", f.REMOTE],
        input=json.dumps(
            dict(
                op="delete",
                path=str(target),
                root=str(root),
                checkpoint=str(cp),
                identity="run",
                layer=0,
            )
        ),
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) is (kind == "missing")
    assert outside.read_text() == "keep" and root.is_dir()


@pytest.mark.parametrize("primary_failure", [False, True])
def test_unlock_failure_preserves_conversion_error(setup, monkeypatch, primary_failure):
    args, remote = setup
    remote.fail_export = primary_failure
    original = remote.call

    def call(op, **kwargs):
        if op == "unlock":
            raise RuntimeError("unlock unavailable")
        return original(op, **kwargs)

    monkeypatch.setattr(remote, "call", call)
    expected = "interrupted export" if primary_failure else "unlock unavailable"
    with pytest.raises(RuntimeError, match=expected) as caught:
        f.Driver(args, remote).run()
    if primary_failure:
        assert caught.value.__notes__ == [
            "Remote unlock also failed: unlock unavailable"
        ]


def test_artifact_locks_exclude_other_processes_and_aliases(tmp_path, monkeypatch):
    import subprocess
    import sys

    from scripts import deepseek_v41_gptq_lock as locks

    monkeypatch.setattr(locks, "LOCK_ROOT", tmp_path / "locks")
    artifact = tmp_path / "parts"
    artifact.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(artifact, target_is_directory=True)
    probe = """import sys
from pathlib import Path
from scripts import deepseek_v41_gptq_lock as locks
locks.LOCK_ROOT=Path(sys.argv[1])
with locks.artifact_locks([Path(sys.argv[2])]): pass
"""

    def contender():
        return subprocess.run(
            [sys.executable, "-c", probe, str(locks.LOCK_ROOT), str(alias)],
            capture_output=True,
            text=True,
        )

    with locks.artifact_locks([artifact]):
        assert contender().returncode != 0
        with pytest.raises(RuntimeError, match="busy"), locks.artifact_locks([alias]):
            pass
    assert contender().returncode == 0
    # Exceptions also release descriptors; lock files stay to preserve inodes.
    with pytest.raises(ValueError), locks.artifact_locks([artifact]):
        raise ValueError("interrupted")
    assert contender().returncode == 0
    assert len(list(locks.LOCK_ROOT.iterdir())) == 1


def test_artifact_reservation_blocks_standalone_between_driver_commands(
    tmp_path, monkeypatch
):
    from scripts import deepseek_v41_gptq_lock as locks

    monkeypatch.setattr(locks, "LOCK_ROOT", tmp_path / "locks")
    paths = [tmp_path / "parts", tmp_path / "state"]
    with pytest.raises(ValueError, match="owner"):
        locks.reserve_artifacts(paths, "")
    locks.reserve_artifacts(paths, "driver")
    for owner in (None, "another-driver"):
        with (
            pytest.raises(RuntimeError, match="reserved"),
            locks.artifact_locks(paths, owner),
        ):
            pass
    args = SimpleNamespace(parts=paths[0], scratch=paths[1])
    monkeypatch.setenv(locks.OWNER_ENV, "driver")
    # A second child must still acquire the OS lock, even with the token.
    with (
        locks.command_locks(args),
        pytest.raises(RuntimeError, match="busy"),
        locks.command_locks(args),
    ):
        pass
    monkeypatch.delenv(locks.OWNER_ENV)
    with pytest.raises(RuntimeError, match="reserved"), locks.command_locks(args):
        pass
    locks.reserve_artifacts(paths, "driver", release=True)
    with locks.command_locks(args):
        pass
    with locks.command_locks(SimpleNamespace()):
        pass


def test_remote_reservation_and_standalone_cli_share_lock(tmp_path):
    pytest.importorskip("mlx.core")
    import os
    import subprocess
    import sys

    from scripts import deepseek_v41_gptq_lock as locks

    parts = tmp_path / "parts"
    repo = Path(__file__).resolve().parents[1]
    q = dict(
        path=str(tmp_path / "fullrun.lock"),
        repo=str(repo),
        resources=[str(parts)],
        owner="test-driver",
    )

    def remote(op):
        return subprocess.run(
            [sys.executable, "-c", f.REMOTE],
            input=json.dumps(dict(op=op, **q)),
            text=True,
            capture_output=True,
        )

    def cli(owner=None):
        env = dict(os.environ)
        env.pop(locks.OWNER_ENV, None)
        if owner:
            env[locks.OWNER_ENV] = owner
        return subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/deepseek_v41_gptq.py"),
                "amend-recipe",
                "--parts",
                str(parts),
                "--scratch",
                str(tmp_path / "state"),
                "--recipe",
                str(tmp_path / "missing-recipe"),
                "--from-layer",
                "0",
                "--reason",
                "test",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

    # A standalone job already holding the lock rejects the driver's reservation.
    with locks.artifact_locks([parts]):
        assert remote("lock").returncode != 0
        assert not Path(q["path"]).exists()
    assert remote("lock").returncode == 0
    try:
        rejected = cli()
        assert "conversion artifact is reserved" in rejected.stderr
        assert not parts.exists()
        # Owned child passes the lock and reaches normal argument validation.
        assert "FileNotFoundError" in cli("test-driver").stderr
        assert remote("lock").returncode != 0
    finally:
        assert remote("unlock").returncode == 0
    assert "FileNotFoundError" in cli().stderr


def test_consumed_reference_is_released_before_next_layer(setup, monkeypatch):
    args, remote = setup
    driver = f.Driver(args, remote)
    stage_reference = driver.stage_reference

    def stage(layer):
        if layer == 2:
            # Layer one is committed and exported before its input is reclaimed.
            assert remote.runs == [0, 1]
            assert (args.destination / "layer-1-export.json").exists()
            root = remote.path(driver.reference)
            assert not any((root / "boundary-1").iterdir())
            assert (root / "boundary-0/complete.json").exists()
            assert (root / "boundary-2/complete.json").exists()
            assert (args.reference_source / "boundary-1/complete.json").exists()
            raise RuntimeError("reference retention checked")
        stage_reference(layer)

    monkeypatch.setattr(driver, "stage_reference", stage)
    with pytest.raises(RuntimeError, match="reference retention checked"):
        driver.run()
