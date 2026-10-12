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
    scheduler._persist([1, 2, 3, 4], [item])
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
