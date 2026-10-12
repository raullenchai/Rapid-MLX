# SPDX-License-Identifier: Apache-2.0
"""Durable adapter prefixes are optional, isolated and disposable."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import disk_caches
from rapid_mlx.speculative.tensorfold_prefix_disk import (
    BUDGET_ENV,
    ENABLE_ENV,
    PrefixDisk,
    install_prefix_disk,
)


@pytest.fixture
def runtime():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("tensorfold.engine.prefix_snapshots")
    from tensorfold.engine.alternating_kv import AlternatingKVCache

    return mx, AlternatingKVCache


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    disk_caches.configure(False)
    monkeypatch.delenv(disk_caches.ENV_VAR, raising=False)
    monkeypatch.setenv(ENABLE_ENV, "1")
    yield
    disk_caches.configure(False)


def make_disk(tmp_path, runtime, budget=4 * 1024**2):
    mx, cls = runtime
    item = cls()
    item.keys = mx.ones((1, 2, 4, 8))
    item.values = mx.full((1, 2, 4, 8), 2.0)
    item.offset = 4
    disk = PrefixDisk(tmp_path / "prefixes", "qualified-model", [item], budget)
    return disk, item


def stored(disk, item):
    disk.save([1, 2, 3, 4], [item])
    files = list(disk.directory.glob("*.safetensors"))
    assert len(files) == 1
    return files[0]


def rewrite(path, runtime, mutation):
    mx, _ = runtime
    arrays, meta = mx.load(str(path), return_metadata=True)
    mutation(meta, arrays)
    mx.save_safetensors(str(path), arrays, metadata=meta)
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_real_snapshot_restart_roundtrip(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)
    # A new adapter instance knows only the live model's empty cache schema.
    fresh = PrefixDisk(disk.directory, disk.model_id, [item], disk.budget)
    tokens, cache = fresh.load(path)
    assert tokens == [1, 2, 3, 4]
    assert type(cache[0]) is type(item)
    mx, _ = runtime
    assert mx.array_equal(cache[0].keys, item.keys).item()
    assert mx.array_equal(cache[0].values, item.values).item()
    assert cache[0].offset == 4
    assert cache[0].spare_keys is None


@pytest.mark.parametrize(
    "kind",
    [
        "checksum",
        "missing-checksum",
        "layers-json",
        "missing-layers",
        "class",
        "fields",
        "offset",
        "extra-array",
        "wrong-model",
        "tokens",
    ],
)
def test_bad_snapshot_is_a_cache_miss(tmp_path, runtime, kind):
    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)
    if kind == "checksum":
        path.with_suffix(".sha256").write_text("invalid")
    elif kind == "missing-checksum":
        path.with_suffix(".sha256").unlink()
    else:

        def corrupt(meta, arrays):
            layers = json.loads(meta["layers"])
            if kind == "layers-json":
                meta["layers"] = "{"
            elif kind == "missing-layers":
                del meta["layers"]
            elif kind == "wrong-model":
                meta["model"] = "another-model"
            elif kind == "tokens":
                meta["tokens"] = "[true, 2, 3, 4]"
            elif kind == "extra-array":
                arrays["unknown"] = runtime[0].zeros((1,))
            else:
                if kind == "class":
                    layers[0]["class"] = "os:system"
                elif kind == "fields":
                    layers[0]["plain"]["__class__"] = "untrusted"
                elif kind == "offset":
                    layers[0]["plain"]["offset"] = 999
                meta["layers"] = json.dumps(layers)

        rewrite(path, runtime, corrupt)
    assert disk.load(path) is None


@pytest.mark.parametrize("policy", ["cli", "env", "budget", "free-space"])
def test_policy_prevents_writes(tmp_path, runtime, monkeypatch, policy):
    disk, item = make_disk(tmp_path, runtime)
    if policy == "cli":
        disk_caches.configure(True)
    elif policy == "env":
        monkeypatch.setenv(disk_caches.ENV_VAR, "1")
    elif policy == "budget":
        disk.budget = 1
    else:
        monkeypatch.setattr("rapid_mlx.memory_cache._free_disk_bytes", lambda _: 0)
    disk.save([1, 2, 3, 4], [item])
    assert not list(disk.directory.glob("*.safetensors"))


def test_disabled_policy_prevents_reads(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)
    disk_caches.configure(True)
    assert disk.load(path) is None


def test_restore_requires_strict_matching_prefix(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    stored(disk, item)
    inserted = []
    scheduler = SimpleNamespace(
        checkpoints=SimpleNamespace(
            longest=lambda prompt, usable: 0,
            insert=lambda *args, **kw: inserted.append(args),
        ),
        prompt_memory=None,
    )
    disk.restore(scheduler, [1, 2, 3, 4])
    disk.restore(scheduler, [1, 2, 99, 4, 5])
    disk.restore(scheduler, [1, 2, 3, 4, 5], usable=lambda _: False)
    assert not inserted
    disk.restore(scheduler, [1, 2, 3, 4, 5])
    assert inserted[0][0] == [1, 2, 3, 4]


def test_default_off_does_not_touch_app(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    install_prefix_disk(object(), Path("unused"))


def test_bad_budget_disables_persistence(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "not-an-integer")
    install_prefix_disk(object(), Path("unused"))


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "dtype",
        "truncated",
        "flipped",
        "oversize",
        "symlink",
        "checksum-symlink",
    ],
)
def test_unsafe_tensor_is_a_miss(tmp_path, runtime, kind):
    disk, item = make_disk(tmp_path, runtime)
    # Geometry must come from live state, independently of the file under test.
    disk = PrefixDisk(disk.directory, disk.model_id, [item], disk.budget)
    path = stored(disk, item)
    if kind in ("shape", "dtype"):

        def mutate(meta, arrays):
            arrays["0.keys"] = (
                runtime[0].ones((1, 3, 4, 8))
                if kind == "shape"
                else arrays["0.keys"].astype(runtime[0].int32)
            )

        rewrite(path, runtime, mutate)
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:20])
    elif kind == "flipped":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif kind == "oversize":
        with path.open("ab") as f:
            f.truncate(disk.budget + 1)
    else:
        victim = path if kind == "symlink" else path.with_suffix(".sha256")
        other = tmp_path / "outside"
        victim.rename(other)
        victim.symlink_to(other)
    assert disk.load(path) is None


def test_identity_changes_when_contents_change(tmp_path):
    from rapid_mlx.speculative.tensorfold_prefix_disk import _identity

    target = tmp_path / "model"
    target.mkdir()
    weights = target / "weights.safetensors"
    weights.write_bytes(b"original")
    template = target / "chat_template.jinja"
    template.write_text("first")
    app = SimpleNamespace(
        _model=SimpleNamespace(prefill_key="same"),
        engine=SimpleNamespace(
            prefill_plan=SimpleNamespace(step=2048, name="grid2048")
        ),
        tokenizer=SimpleNamespace(chat_template="first"),
    )
    first = _identity(target, None, app)
    weights.write_bytes(b"replaced")
    second = _identity(target, None, app)
    assert first != second
    template.write_text("second")
    assert second != _identity(target, None, app)


def test_row_state_geometry_and_bfloat16_roundtrip(tmp_path, runtime):
    from tensorfold.families.nemotron_h.state_cache import RowStateCache

    mx, _ = runtime
    item = RowStateCache(2)
    item.cache = [mx.ones((1, 3, 8), dtype=mx.bfloat16), mx.ones((1, 2, 4, 8))]
    disk = PrefixDisk(tmp_path / "states", "model", [item], 4 * 1024**2)
    path = stored(disk, item)
    loaded = disk.load(path)
    assert loaded is not None
    assert mx.array_equal(loaded[1][0].cache[0], item.cache[0]).item()
    assert loaded[1][0].cache[0].dtype == mx.bfloat16

    def mutate(meta, arrays):
        arrays["0._cache.1"] = mx.ones((1, 2, 4, 9))

    rewrite(path, runtime, mutate)
    assert disk.load(path) is None


def test_unknown_live_class_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="live cache class"):
        PrefixDisk(tmp_path, "model", [SimpleNamespace()], 1024)


def test_global_disable_does_no_io(tmp_path, runtime, monkeypatch):
    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)

    def unexpected(*args, **kwargs):
        raise AssertionError("disk I/O attempted while globally disabled")

    monkeypatch.setattr(disk, "_files", unexpected)
    monkeypatch.setattr(disk, "_private_directory", unexpected)
    monkeypatch.setattr(
        "rapid_mlx.speculative.tensorfold_prefix_disk._read_file", unexpected
    )
    monkeypatch.setenv(disk_caches.ENV_VAR, "1")
    disk.save([1, 2, 3, 4], [item])
    disk.restore(SimpleNamespace(checkpoints=object()), [1, 2, 3, 4, 5])
    assert disk.load(path) is None
    install_prefix_disk(object(), Path("unused"))


def test_lock_contention_skips_entire_write(tmp_path, runtime):
    from rapid_mlx.runtime.cache import _exclusive_cache_lock

    disk, item = make_disk(tmp_path, runtime)
    disk.directory.mkdir(mode=0o700)
    with _exclusive_cache_lock(str(disk.directory), operation="test") as locked:
        assert locked
        disk.save([1, 2, 3, 4], [item])
        assert list(disk.directory.iterdir()) == []
    stored(disk, item)


def test_interrupted_publication_is_a_miss_and_recoverable(
    tmp_path, runtime, monkeypatch
):
    disk, item = make_disk(tmp_path, runtime)
    original = Path.replace

    def interrupt(path, target):
        if path.name == "digest":
            raise OSError("simulated crash before checksum publication")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt)
        disk.save([1, 2, 3, 4], [item])
    path = next(disk.directory.glob("*.safetensors"))
    assert disk.load(path) is None
    disk.save([1, 2, 3, 4], [item])
    assert disk.load(path) is not None
    assert not list(disk.directory.glob(".stage-*"))


def test_budget_eviction_and_abandoned_stage(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime, budget=1024**2 + 1500)
    # The conservative header allowance leaves room for only one entry's
    # reservation before publication. Any previous entry must be evicted.
    first = stored(disk, item)
    abandoned = disk.directory / ".stage-abandoned"
    abandoned.mkdir()
    (abandoned / "partial").write_bytes(b"interrupted")
    disk.save([5, 6, 7, 8], [item])
    assert not first.exists()
    assert not abandoned.exists()
    files = list(disk.directory.glob("*.safetensors"))
    assert len(files) == 1
    assert sum(p.stat().st_size for p in disk.directory.iterdir()) <= disk.budget
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].with_suffix(".sha256").stat().st_mode & 0o777 == 0o600
    assert disk.directory.stat().st_mode & 0o777 == 0o700


def test_symlink_directory_does_not_write(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    disk.directory.symlink_to(outside, target_is_directory=True)
    disk.save([1, 2, 3, 4], [item])
    assert list(outside.iterdir()) == []


def test_seam_drift_fails_closed(tmp_path, monkeypatch):
    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    scheduler = SimpleNamespace(
        _persist="original", _read_disk_block="original", snapshot_dir=None
    )
    app = SimpleNamespace(scheduler=scheduler, checkpoints=object())
    monkeypatch.setattr(module, "SUPPORTED_REVISION", "next-revision")
    install_prefix_disk(app, tmp_path)
    assert scheduler._persist == "original"
    assert scheduler._read_disk_block == "original"
    assert scheduler.snapshot_dir is None
    assert not hasattr(app, "_rapid_prefix_disk")


@pytest.mark.parametrize(
    "kind",
    [
        "huge-header",
        "bad-header-json",
        "huge-shape",
        "overlap",
        "trailing",
        "nonfinite",
        "empty-slots",
    ],
)
def test_structural_corruption_never_calls_native_loader(
    tmp_path, runtime, monkeypatch, kind
):
    import struct

    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + length])
    if kind == "huge-header":
        data = struct.pack("<Q", 2**63) + data[8:]
    elif kind == "bad-header-json":
        data = data[:8] + b"!" + data[9:]
    elif kind in ("huge-shape", "overlap", "empty-slots"):
        if kind == "huge-shape":
            header["0.keys"]["shape"][2] = 2**60
        elif kind == "overlap":
            header["0.keys"]["data_offsets"] = header["0.values"]["data_offsets"]
        else:
            layers = json.loads(header["__metadata__"]["layers"])
            layers[0]["arrays"] = []
            layers[0]["plain"].update(keys=None, values=None)
            header["__metadata__"]["layers"] = json.dumps(layers)
        raw = json.dumps(header).encode()
        data = struct.pack("<Q", len(raw)) + raw + data[8 + length :]
    elif kind == "trailing":
        data += b"trailing"
    else:
        rewrite(
            path,
            runtime,
            lambda meta, arrays: arrays.update(
                {"0.keys": runtime[0].full((1, 2, 4, 8), float("nan"))}
            ),
        )
        data = path.read_bytes()
    path.write_bytes(data)
    path.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        runtime[0], "load", lambda *a, **kw: pytest.fail("native file loader invoked")
    )
    assert disk.load(path) is None


def test_successful_install_uses_override_and_never_enables_sessions(
    tmp_path, runtime, monkeypatch
):
    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    disk, item = make_disk(tmp_path, runtime)
    scheduler = SimpleNamespace(
        snapshot_dir=None,
        session_dir=None,
        disk_blocks=None,
        session_blocks=None,
        _thread=SimpleNamespace(is_alive=lambda: False),
    )
    app = SimpleNamespace(
        scheduler=scheduler,
        checkpoints=object(),
        context_window=32768,
        _model=SimpleNamespace(),
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: [10]),
        engine=SimpleNamespace(prefill_prefix=lambda ids: [item]),
    )
    import types

    from tensorfold.server.scheduler import Scheduler

    scheduler._keep_checkpoints = types.MethodType(
        Scheduler._keep_checkpoints, scheduler
    )
    scheduler.prompt_memory = None
    scheduler.checkpoints = SimpleNamespace(insert=lambda *a, **kw: None)
    app.close = lambda: None
    app.system_prefix_len = lambda messages, tools, prompt_ids, thinking=None: 0
    scheduler._open_job = lambda job: None
    scheduler.submit = lambda job: None
    scheduler.on_engine = lambda fn: fn(app.engine)
    monkeypatch.setattr(module, "_qualified", lambda _: True)
    monkeypatch.setattr(module, "_assets_unchanged", lambda *a: True)
    monkeypatch.setattr(module, "_identity", lambda *a: "qualified")
    monkeypatch.setenv(module.DIRECTORY_ENV, str(disk.directory))
    install_prefix_disk(app, tmp_path)
    assert scheduler.snapshot_dir == disk.directory
    assert (
        scheduler.session_dir
        is scheduler.disk_blocks
        is scheduler.session_blocks
        is None
    )
    job = SimpleNamespace(
        vision=None,
        _rapid_disk_system_len=5,
        prompt_ids=[1, 2, 3, 4, 5],
        shared_prefix_lens=(5,),
        stream=SimpleNamespace(history_checkpoints=[([1, 2, 3, 4], [item])]),
    )
    scheduler._keep_checkpoints(job, {4})
    assert app._rapid_prefix_disk.wait(5)
    assert (
        app._rapid_prefix_disk.load(next(disk.directory.glob("*.safetensors")))
        is not None
    )


def test_changed_seam_source_disables_feature(monkeypatch, runtime):
    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    monkeypatch.setattr(module, "_digest", lambda _: "changed-source")
    assert not module._qualified(SimpleNamespace())


@pytest.mark.parametrize("phase", ["before-tensor", "before-checksum"])
def test_process_crash_releases_lock_and_leaves_only_a_miss(tmp_path, runtime, phase):
    import subprocess
    import sys

    disk, item = make_disk(tmp_path, runtime)
    script = r"""
import os, sys
from pathlib import Path
import mlx.core as mx
from tensorfold.engine.alternating_kv import AlternatingKVCache
from rapid_mlx.speculative.tensorfold_prefix_disk import PrefixDisk
c = AlternatingKVCache()
c.keys = mx.ones((1, 2, 4, 8))
c.values = mx.full((1, 2, 4, 8), 2.0)
c.offset = 4
disk = PrefixDisk(Path(sys.argv[1]), 'qualified-model', [c], 4 * 1024**2)
original = Path.replace
def crash(path, target):
    if (sys.argv[2] == 'before-checksum' and path.name == 'digest') or (
            sys.argv[2] == 'before-tensor' and path.suffix == '.safetensors'):
        os._exit(51)
    return original(path, target)
Path.replace = crash
disk.save([1, 2, 3, 4], [c])
raise RuntimeError('crash point not reached')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(disk.directory), phase],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 51, result.stdout + result.stderr
    assert list(disk.directory.glob(".stage-*"))
    for path in disk.directory.glob("*.safetensors"):
        assert disk.load(path) is None
    # The OS released the permanent transaction lock. A new writer cleans the
    # abandoned staging tree and replaces an incomplete publication safely.
    disk.save([1, 2, 3, 4], [item])
    assert not list(disk.directory.glob(".stage-*"))
    assert disk.load(next(disk.directory.glob("*.safetensors"))) is not None


def test_two_processes_share_one_budget_and_publish_valid_pairs(tmp_path, runtime):
    import subprocess
    import sys

    disk, _ = make_disk(tmp_path, runtime)
    script = r"""
import sys
from pathlib import Path
import mlx.core as mx
from tensorfold.engine.alternating_kv import AlternatingKVCache
from rapid_mlx.speculative.tensorfold_prefix_disk import PrefixDisk
c = AlternatingKVCache()
c.keys = mx.ones((1, 2, 4, 8))
c.values = mx.full((1, 2, 4, 8), 2.0)
c.offset = 4
disk = PrefixDisk(Path(sys.argv[1]), 'qualified-model', [c], 4 * 1024**2)
for i in range(12):
    disk.save([int(sys.argv[2]), i, 3, 4], [c])
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(disk.directory), str(i)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in (1, 2)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
    files = list(disk.directory.glob("*.safetensors"))
    assert 0 < len(files) <= 8
    assert (
        sum(p.stat().st_size for p in disk.directory.iterdir() if p.is_file())
        <= disk.budget
    )
    assert len(list(disk.directory.glob("*.sha256"))) == len(files)
    assert all(disk.load(p) is not None for p in files)
    assert not list(disk.directory.glob(".stage-*"))


def test_writer_removes_kv_capacity_beyond_system_prefix(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    mx, _ = runtime
    item.keys = mx.concatenate([item.keys, mx.full((1, 2, 3, 8), 99.0)], axis=2)
    item.values = mx.concatenate([item.values, mx.full((1, 2, 3, 8), 99.0)], axis=2)
    path = stored(disk, item)
    tokens, restored = disk.load(path)
    assert tokens == [1, 2, 3, 4]
    assert restored[0].keys.shape == (1, 2, 4, 8)
    assert restored[0].values.shape == (1, 2, 4, 8)
    assert mx.all(restored[0].keys == 1).item()
    assert mx.all(restored[0].values == 2).item()
    # Trimming a serialization copy must not mutate the live inference cache.
    assert item.keys.shape[2] == item.values.shape[2] == 7
    assert item.offset == 4


def test_write_validation_does_not_allocate_another_mlx_cache(
    tmp_path, runtime, monkeypatch
):
    disk, item = make_disk(tmp_path, runtime)

    def unexpected(*args, **kwargs):
        raise AssertionError("writer allocated a second MLX cache")

    import numpy as np

    with monkeypatch.context() as patch:
        patch.setattr(np, "frombuffer", unexpected)
        path = stored(disk, item)
    assert disk.load(path) is not None


def test_checkpoint_replacement_during_identity_hash_disables_install(
    tmp_path, runtime
):
    import subprocess
    import sys

    target = tmp_path / "model"
    target.mkdir()
    (target / "weights.safetensors").write_bytes(b"old")
    script = r"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
from tensorfold.engine.alternating_kv import AlternatingKVCache
from rapid_mlx.speculative import tensorfold_prefix_disk as module
c = AlternatingKVCache()
c.keys = c.values = mx.ones((1, 2, 4, 8))
c.offset = 4
scheduler = SimpleNamespace(snapshot_dir=None, _persist='original', _read_disk_block='original')
engine = SimpleNamespace(prefill_prefix=lambda ids: [c])
scheduler.on_engine = lambda fn: fn(engine)
app = SimpleNamespace(scheduler=scheduler, checkpoints=object(), context_window=32768,
                      _model=SimpleNamespace(mtp=None), engine=engine,
                      tokenizer=SimpleNamespace(encode=lambda *a, **kw: [10]))
module._qualified = lambda _: True
os.environ[module.ENABLE_ENV] = '1'
os.environ[module.DIRECTORY_ENV] = sys.argv[2]
def changed_identity(*args):
    (Path(sys.argv[1]) / 'weights.safetensors').write_bytes(b'new')
    return 'new-contents-for-already-loaded-old-model'
module._identity = changed_identity
module.install_prefix_disk(app, Path(sys.argv[1]))
assert not hasattr(app, '_rapid_prefix_disk')
assert scheduler._persist == scheduler._read_disk_block == 'original'
assert scheduler.snapshot_dir is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(target), str(tmp_path / "snapshots")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "weights.safetensors").read_bytes() == b"new"
    assert not (tmp_path / "snapshots").exists()


def test_asset_guard_rejects_changed_symlink_even_to_old_blob(tmp_path):
    import time

    from rapid_mlx.speculative.tensorfold_prefix_disk import _assets_unchanged

    target = tmp_path / "model"
    target.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"old")
    link = target / "weights.safetensors"
    link.symlink_to(blob)
    app = SimpleNamespace(_model=SimpleNamespace(mtp=None))
    # All assets predate a hypothetical future process birth.
    assert _assets_unchanged(target, None, app, time.time() + 60)
    started_at = time.time() - 60
    assert not _assets_unchanged(target, None, app, started_at)
    # An old blob mtime cannot hide a fresh link status-change time.
    import os

    os.utime(blob, (started_at - 60, started_at - 60))
    assert not _assets_unchanged(target, None, app, started_at)


def hf_artifact(tmp_path, *, digest="a" * 64, revision="b" * 40):
    repo = tmp_path / "models--pilot--model"
    blob = repo / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"original")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    link = snapshot / "model.safetensors"
    link.symlink_to(blob)
    return snapshot, blob, link


def test_hf_lfs_identity_does_not_read_weights(tmp_path, monkeypatch):
    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    snapshot, blob, _ = hf_artifact(tmp_path)
    monkeypatch.setattr(module, "_digest", lambda _: pytest.fail("weight read"))
    identity = module._artifact(snapshot)
    source = identity["contents"][0][1]
    assert source["sha256"] == blob.name
    assert source["revision"] == snapshot.name
    assert source["generation"][2] == len(b"original")
    assert source["source"] == "hf-lfs-sha256-generation"
    assert identity == module._artifact(snapshot)


@pytest.mark.parametrize("kind", ["overwrite", "replace", "relink"])
def test_hf_generation_invalidates_same_name_size_and_mtime(tmp_path, kind):
    import os

    from rapid_mlx.speculative.tensorfold_prefix_disk import _artifact

    snapshot, blob, link = hf_artifact(tmp_path)
    before = _artifact(snapshot)
    info = blob.stat()
    if kind == "overwrite":
        blob.write_bytes(b"modified")
    elif kind == "replace":
        replacement = blob.with_name("replacement")
        replacement.write_bytes(b"modified")
        replacement.replace(blob)
    else:
        link.unlink()
        link.symlink_to(blob)
    os.utime(blob, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert blob.stat().st_size == info.st_size
    assert blob.stat().st_mtime_ns == info.st_mtime_ns
    assert _artifact(snapshot) != before


@pytest.mark.parametrize(
    "kind", ["local", "git-sha1", "bad-revision", "outside", "blob-link", "config"]
)
def test_non_lfs_layouts_always_hash_contents(tmp_path, kind, monkeypatch):
    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    snapshot, blob, link = hf_artifact(
        tmp_path,
        digest="a" * (40 if kind == "git-sha1" else 64),
        revision="not-a-commit" if kind == "bad-revision" else "b" * 40,
    )
    if kind == "local":
        link.unlink()
        link.write_bytes(b"original")
    elif kind in ("outside", "blob-link"):
        outside = tmp_path / ("c" * 64)
        outside.write_bytes(b"original")
        if kind == "outside":
            link.unlink()
            link.symlink_to(outside)
        else:
            blob.unlink()
            blob.symlink_to(outside)
    elif kind == "config":
        link.rename(link.with_suffix(".json"))
    calls = []
    original = module._digest
    monkeypatch.setattr(module, "_digest", lambda p: (calls.append(p), original(p))[1])
    before = module._artifact(snapshot)
    assert len(calls) == 1
    assert before["contents"][0][1] == hashlib.sha256(b"original").hexdigest()
    next(snapshot.iterdir()).write_bytes(b"modified")
    assert module._artifact(snapshot) != before
    assert len(calls) == 2


def test_asset_guard_rejects_indirect_blob_symlink(tmp_path):
    import time

    from rapid_mlx.speculative.tensorfold_prefix_disk import _assets_unchanged

    snapshot, blob, _ = hf_artifact(tmp_path)
    other = tmp_path / "old-blob"
    other.write_bytes(b"old")
    blob.unlink()
    blob.symlink_to(other)
    app = SimpleNamespace(_model=SimpleNamespace(mtp=None))
    assert not _assets_unchanged(snapshot, None, app, time.time() + 60)


def test_asset_guard_rejects_hf_blob_directory_swap(tmp_path):
    import time

    from rapid_mlx.speculative.tensorfold_prefix_disk import _assets_unchanged

    snapshot, blob, _ = hf_artifact(tmp_path)
    replacement = blob.parent.with_name("replacement-blobs")
    replacement.mkdir()
    (replacement / blob.name).write_bytes(b"modified")
    time.sleep(0.01)
    started_at = time.time()
    blob.parent.rename(blob.parent.with_name("original-blobs"))
    replacement.rename(blob.parent)
    app = SimpleNamespace(_model=SimpleNamespace(mtp=None))
    assert not _assets_unchanged(snapshot, None, app, started_at)


def test_asset_guard_rejects_replaced_parent_alias(tmp_path):
    import time

    from rapid_mlx.speculative.tensorfold_prefix_disk import _assets_unchanged

    for name in ("original", "replacement"):
        model = tmp_path / name / "model"
        model.mkdir(parents=True)
        (model / "weights.safetensors").write_bytes(name.encode())
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "original")
    time.sleep(0.01)
    started_at = time.time()
    alias.unlink()
    alias.symlink_to(tmp_path / "replacement")
    app = SimpleNamespace(_model=SimpleNamespace(mtp=None))
    assert not _assets_unchanged(alias / "model", None, app, started_at)


def test_interim_hf_identity_snapshot_is_a_miss(tmp_path, runtime):
    from rapid_mlx.speculative.tensorfold_prefix_disk import _identity

    target, _, _ = hf_artifact(tmp_path)
    app = SimpleNamespace(
        _model=SimpleNamespace(mtp=None),
        engine=SimpleNamespace(prefill_plan=SimpleNamespace(step=2048)),
        tokenizer=SimpleNamespace(chat_template="template"),
    )
    current = _identity(target, None, app)
    legacy = json.loads(current)
    legacy["format"] = 4  # Interim policy lacked the container/ancestor guard.
    disk, item = make_disk(tmp_path, runtime)
    disk.model_id = json.dumps(legacy, sort_keys=True)
    path = stored(disk, item)
    assert disk.load(path) is not None
    fresh = PrefixDisk(disk.directory, current, [item], disk.budget)
    assert fresh.load(path) is None


def test_pinned_mtp_lookahead_is_refused_or_recorded(tmp_path, runtime, monkeypatch):
    import types

    from tensorfold.engine.family_prefill import FamilyPrefill
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot
    from tensorfold.families.nemotron_h.model import NemotronH
    from tensorfold.families.nemotron_h.mtp import MTPCache

    mx = runtime[0]

    class Feed(FamilyPrefill):
        prefill_guard = None
        prefill_chunks = 0

        def __init__(self):
            # No weights or model are loaded. Tiny deterministic tensors isolate
            # the pinned prefill + absorb_draft_context data flow.
            model = SimpleNamespace(mtp=object(), prompt_pass=False)
            model.prefill = lambda inputs, cache: mx.zeros((1, inputs.shape[1], 1))
            model.model = SimpleNamespace(
                backbone=SimpleNamespace(
                    embeddings=lambda ids: ids.astype(mx.float32)[..., None]
                )
            )
            model._trim_chained = NemotronH._trim_chained

            def head(hidden, embeddings, cache, tail):
                rows = embeddings.reshape(1, 1, -1, 1)
                cache.update_and_fetch(rows, rows)

            model._head_step = head
            model.absorb_draft_context = types.MethodType(
                NemotronH.absorb_draft_context, model
            )
            self.model = model

    feed = Feed()
    template, checkpoint = MTPCache(), MTPCache()
    feed._family_feed([7] * 16, [template], [(0, 16)])
    prompt = [7] * 512 + [999, 4]
    feed._family_feed(prompt, [checkpoint], [(0, 512)])
    assert checkpoint.keys[0, 0, 511, 0].item() == 999
    disk = PrefixDisk(tmp_path / "mtp", "review", [template], 4 * 1024**2)
    disk.save(prompt[:512], [checkpoint])
    assert not list(disk.directory.glob("*.safetensors"))
    disk.save(prompt[:512], [checkpoint], prompt=prompt, shared_prefix_len=512)
    assert not list(disk.directory.glob("*.safetensors"))
    disk.save(prompt[:512], [checkpoint], prompt=prompt, shared_prefix_len=513)
    path = next(disk.directory.glob("*.safetensors"))
    assert json.loads(disk._metadata(path)["lookahead"]) == 999
    restored = disk.load(path)[1][0]
    assert restored.keys[0, 0, 511, 0].item() == 999
    # Trusted synthetic reference only: exercise the pinned writer/reader too.
    # Production never invokes this import-by-name native loader.
    native = save_snapshot(
        tmp_path / "native-reference", "review", prompt[:512], [checkpoint]
    )
    native_tokens, native_cache = load_snapshot(native, "review")
    assert native_tokens == prompt[:512]
    assert mx.array_equal(native_cache[0].keys[..., :512, :], restored.keys).item()
    entries = []
    scheduler = SimpleNamespace(
        checkpoints=SimpleNamespace(
            longest=lambda *a: 0, insert=lambda *a, **kw: entries.append(a)
        ),
        prompt_memory=None,
    )
    disk.restore(scheduler, prompt)
    assert len(entries) == 1
    entries.clear()
    disk.restore(scheduler, [7] * 512 + [998, 4])
    assert entries == []
    original = disk._verified_bytes

    def replaced(path):
        rewrite(path, runtime, lambda meta, arrays: meta.update(lookahead="998"))
        return original(path)

    monkeypatch.setattr(disk, "_verified_bytes", replaced)
    disk.restore(scheduler, prompt)
    assert entries == []


def test_background_slow_fsync_drops_busy_and_close_is_bounded(
    tmp_path, runtime, monkeypatch
):
    import os
    import threading
    import time

    disk, item = make_disk(tmp_path, runtime)
    blocked, release = threading.Event(), threading.Event()
    original = os.fsync

    def slow(fd):
        blocked.set()
        assert release.wait(5)
        return original(fd)

    monkeypatch.setattr(os, "fsync", slow)
    started = time.monotonic()
    from tensorfold.server.scheduler import Scheduler

    accepted = []
    scheduler = SimpleNamespace(
        snapshot_dir=disk.directory,
        checkpoints=SimpleNamespace(insert=lambda *a, **kw: None),
        _persist=lambda tokens, cache: accepted.append(disk.submit(tokens, cache)),
    )
    job = SimpleNamespace(
        vision=None,
        prompt_ids=[1, 2, 3, 4, 5],
        stream=SimpleNamespace(history_checkpoints=[([1, 2, 3, 4], [item])]),
    )
    Scheduler._keep_checkpoints(scheduler, job, {4})
    assert accepted == [True]
    assert time.monotonic() - started < 0.2
    assert blocked.wait(2)
    assert disk.reserved_bytes > 0
    assert not disk.submit([1, 2, 3, 4], [item])
    entries = []
    scheduler = SimpleNamespace(
        checkpoints=SimpleNamespace(
            longest=lambda *a: 0, insert=lambda *a, **kw: entries.append(a)
        ),
        prompt_memory=None,
    )
    disk.restore(scheduler, [1, 2, 3, 4, 5])
    assert entries == []
    started = time.monotonic()
    disk.close()
    assert time.monotonic() - started < 0.2
    release.set()
    assert disk.wait(5)
    assert disk.reserved_bytes == 0
    assert not list(disk.directory.glob("*.safetensors"))


def test_actual_mid_write_enospc_is_a_miss(tmp_path, runtime, monkeypatch):
    import errno

    disk, item = make_disk(tmp_path, runtime)

    def fail(path, payload):
        path.write_bytes(b"half-written tensor")
        raise OSError(errno.ENOSPC, "injected disk full after write")

    monkeypatch.setattr(disk, "_write_host", fail)
    assert disk.submit([1, 2, 3, 4], [item])
    assert disk.wait(5)
    assert list(disk.directory.glob("*.safetensors")) == []
    assert list(disk.directory.glob(".stage-*")) == []


def test_install_real_pinned_qualification_identity_and_engine(
    runtime, monkeypatch, tmp_path
):
    import tensorfold
    from tensorfold.server.scheduler import Scheduler

    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    disk, item = make_disk(tmp_path, runtime)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.snapshot_dir = scheduler.session_dir = scheduler.disk_blocks = (
        scheduler.session_blocks
    ) = None
    scheduler.checkpoints = SimpleNamespace(insert=lambda *a, **kw: None)
    scheduler.prompt_memory = None
    app = SimpleNamespace(
        scheduler=scheduler,
        checkpoints=scheduler.checkpoints,
        context_window=32768,
        _model=SimpleNamespace(mtp=None),
        tokenizer=SimpleNamespace(
            encode=lambda *a, **kw: [10], chat_template="model-free"
        ),
        engine=SimpleNamespace(
            prefill_prefix=lambda ids: [item], prefill_plan=SimpleNamespace(step=512)
        ),
        close=lambda: None,
    )
    app.system_prefix_len = lambda messages, tools, prompt_ids, thinking=None: 0
    scheduler.on_engine = lambda fn: fn(app.engine)
    monkeypatch.setenv(module.DIRECTORY_ENV, str(disk.directory))
    assert module._qualified(scheduler)
    target = Path(tensorfold.__file__)
    module.install_prefix_disk(app, target)
    assert scheduler.snapshot_dir == disk.directory
    assert json.loads(app._rapid_prefix_disk.model_id)["target"]["contents"]
    assert scheduler.session_dir is None
    job = SimpleNamespace(
        vision=None,
        _rapid_disk_system_len=5,
        prompt_ids=[1, 2, 3, 4, 5],
        shared_prefix_lens=(5,),
        stream=SimpleNamespace(history_checkpoints=[([1, 2, 3, 4], [item])]),
    )
    scheduler._keep_checkpoints(job, {4})
    assert app._rapid_prefix_disk.wait(5)
    assert app._rapid_prefix_disk.load(next(disk.directory.glob("*.safetensors")))
    app.close()


def test_copy_admission_refuses_before_materialization(tmp_path, runtime, monkeypatch):
    disk, item = make_disk(tmp_path, runtime)
    admitted = []
    memory = SimpleNamespace(allow_load=lambda size: admitted.append(size) or False)
    monkeypatch.setattr(disk, "_capture", lambda *a: pytest.fail("unadmitted copy"))
    assert not disk.submit([1, 2, 3, 4], [item], memory=memory)
    assert admitted and admitted[0] >= 3 * (item.keys.nbytes + item.values.nbytes)
    assert disk.reserved_bytes == 0
    assert not disk.directory.exists()


@pytest.fixture
def profile_disk_app(tmp_path, runtime, monkeypatch):
    import queue
    import threading
    import types

    import tensorfold
    from mlx_lm.tokenizer_utils import load
    from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
    from tensorfold.families.nemotron_h.mtp import MTPCache
    from tensorfold.server.app import ChatApp
    from tensorfold.server.prompt_blocks import PromptBlocks
    from tensorfold.server.scheduler import Scheduler

    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    target = (
        Path.home()
        / ".cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8"
    )
    tokenizer = load(target, {"local_files_only": True})
    plan = PrefillPlan(
        2048, *message_markers(tokenizer)[:1], assistant=message_markers(tokenizer)[1]
    )

    def cache(size):
        items = []
        for cls in (runtime[1], MTPCache):
            item = cls()
            item.keys = runtime[0].ones((1, 1, size, 1))
            item.values = item.keys
            item.offset = size
            items.append(item)
        return items

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.snapshot_dir = scheduler.session_dir = scheduler.disk_blocks = (
        scheduler.session_blocks
    ) = None
    inserted = []
    scheduler.checkpoints = SimpleNamespace(
        insert=lambda *a, **kw: inserted.append(a), longest=lambda *a: 0
    )
    scheduler.prompt_memory = None
    scheduler._stop = threading.Event()
    scheduler._queue = queue.Queue()
    scheduler._thread = SimpleNamespace(is_alive=lambda: False)
    app = SimpleNamespace(
        scheduler=scheduler,
        checkpoints=scheduler.checkpoints,
        context_window=32768,
        _model=SimpleNamespace(mtp=None),
        tokenizer=tokenizer,
        tokenizer_lock=threading.Lock(),
        enable_thinking=True,
        late_system="system",
        effort_for=lambda _: None,
        engine=SimpleNamespace(prefill_plan=plan, prefill_prefix=lambda ids: cache(16)),
    )
    app.system_prefix_len = types.MethodType(PromptBlocks.system_prefix_len, app)
    app.close = types.MethodType(ChatApp.close, app)
    app.save_sessions = types.MethodType(ChatApp.save_sessions, app)
    scheduler.on_engine = lambda fn: fn(app.engine)
    monkeypatch.setenv(module.DIRECTORY_ENV, str(tmp_path / "profile-cache"))
    # Keep the pinned read call site but omit inference for this fixture.
    scheduler._open_job = types.MethodType(
        lambda self, job: self._read_disk_block(job.prompt_ids), scheduler
    )
    module.install_prefix_disk(app, Path(tensorfold.__file__))
    assert hasattr(app, "_rapid_prefix_disk")
    yield app, cache, inserted
    app.close()


def test_actual_profile_rejects_assistant_prefix_and_accepts_system(
    profile_disk_app, monkeypatch
):
    from tensorfold.server.prompt_blocks import PromptBlocks
    from tensorfold.server.text import render_prompt_ids

    app, cache, inserted = profile_disk_app
    disk = app._rapid_prefix_disk

    def job(messages):
        prompt = render_prompt_ids(
            app.tokenizer, messages, enable_thinking=True, late_system=app.late_system
        )
        shared = app.system_prefix_len(messages, None, prompt, thinking=True)
        lens = tuple(n for n in (shared - 2048, shared - 512, shared) if n >= 512)
        points = {app.engine.prefill_plan.chunks(prompt).floor(n) for n in lens} - {0}
        n = max(x for x in points if x < shared)
        j = SimpleNamespace(
            prompt_ids=prompt,
            shared_prefix_lens=lens,
            vision=None,
            stream=SimpleNamespace(history_checkpoints=[(prompt[:n], cache(n))]),
        )
        app.scheduler.submit(j)
        return j, points, n

    system = {"role": "system", "content": "Shared instructions. " * 600}
    bad = [
        system,
        {"role": "assistant", "content": "PRIVATE_REPLY_MARKER " * 1800},
        {"role": "user", "content": "Continue please."},
    ]
    j, points, n = job(bad)
    assert "PRIVATE_REPLY_MARKER" in app.tokenizer.decode(j.prompt_ids[:n])
    app.scheduler._keep_checkpoints(j, points)
    assert disk.wait(5)
    assert not list(disk.directory.glob("*.safetensors"))
    good = [system, {"role": "user", "content": "Question?"}]
    j, points, n = job(good)
    app.scheduler._keep_checkpoints(j, points)
    assert disk.wait(5)
    assert list(disk.directory.glob("*.safetensors"))
    inserted.clear()
    app.scheduler._open_job(j)
    assert inserted and len(inserted[0][0]) == n
    j, _, _ = job(bad)
    inserted.clear()
    monkeypatch.setattr(
        disk, "_read_prefix", lambda *a: pytest.fail("ineligible disk read")
    )
    app.scheduler._open_job(j)
    assert inserted == []
    # The independent two-probe cap cannot include user text matching the runtime probe.
    probe_user = [system, {"role": "user", "content": "⁣probe PRIVATE_USER"}]
    j, _, _ = job(probe_user)
    raw = PromptBlocks.system_prefix_len(app, probe_user, None, j.prompt_ids, True)
    assert j._rapid_disk_system_len < raw


def test_inaccessible_directory_and_other_fs_errors_are_misses(
    tmp_path, runtime, monkeypatch
):
    import os

    disk, item = make_disk(tmp_path, runtime)
    parent = tmp_path / "restricted"
    parent.mkdir()
    disk.directory = parent / "cache"
    disk.directory.mkdir()
    scheduler = SimpleNamespace(
        checkpoints=SimpleNamespace(longest=lambda *a: 0), prompt_memory=None
    )
    os.chmod(parent, 0)
    try:
        disk.restore(scheduler, [1, 2, 3, 4, 5])
    finally:
        os.chmod(parent, 0o700)
    assert disk.read_reserved_bytes == 0

    def inaccessible(*args, **kwargs):
        raise PermissionError("injected directory error")

    monkeypatch.setattr(disk, "_private_directory", inaccessible)
    disk.restore(scheduler, [1, 2, 3, 4, 5])
    assert disk.read_reserved_bytes == 0
    disk.submit([1, 2, 3, 4], [item])
    assert disk.wait(5)


@pytest.mark.parametrize("closing", [False, True])
def test_stalled_read_bounds_request_and_real_app_shutdown(
    profile_disk_app, runtime, monkeypatch, closing
):
    import threading
    import time

    from rapid_mlx.speculative import tensorfold_prefix_disk as module

    app, cache, inserted = profile_disk_app
    disk = app._rapid_prefix_disk
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(module, "READ_TIMEOUT", 0.1)

    def stalled(*args):
        entered.set()
        assert release.wait(3)
        return b"late discarded bytes"

    monkeypatch.setattr(disk, "_read_prefix", stalled)
    owner = threading.Thread(target=lambda: disk.restore(app.scheduler, [1, 2, 3]))
    app.scheduler._thread = owner
    owner.start()
    assert entered.wait(2)
    assert disk.read_reserved_bytes == 3 * disk.budget
    start = time.monotonic()
    if closing:
        app.close()  # real pinned ChatApp.close -> Scheduler.stop
    else:
        owner.join(0.4)
    assert not owner.is_alive() and time.monotonic() - start < 0.4
    assert not disk._reading.acquire(blocking=False)
    assert disk.read_reserved_bytes > 0
    release.set()
    deadline = time.monotonic() + 2
    while (
        disk.read_reserved_bytes or disk._reading.locked()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert disk.read_reserved_bytes == 0 and inserted == []
    assert disk._reading.acquire(blocking=False)
    disk._reading.release()
