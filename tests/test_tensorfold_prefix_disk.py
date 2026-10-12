# SPDX-License-Identifier: Apache-2.0
"""Durable adapter prefixes are optional, isolated and disposable."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import disk_caches
from rapid_mlx.speculative.tensorfold_prefix_disk import (
    BUDGET_ENV, ENABLE_ENV, PrefixDisk, install_prefix_disk,
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
    template, item = cls(), cls()
    item.keys = mx.ones((1, 2, 4, 8))
    item.values = mx.full((1, 2, 4, 8), 2.0)
    item.offset = 4
    disk = PrefixDisk(tmp_path / "prefixes", "qualified-model", [template], budget)
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
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_real_snapshot_restart_roundtrip(tmp_path, runtime):
    disk, item = make_disk(tmp_path, runtime)
    path = stored(disk, item)
    # A new adapter instance knows only the live model's empty cache schema.
    fresh = PrefixDisk(disk.directory, disk.model_id, [runtime[1]()], disk.budget)
    tokens, cache = fresh.load(path)
    assert tokens == [1, 2, 3, 4]
    assert type(cache[0]) is type(item)
    mx, _ = runtime
    assert mx.array_equal(cache[0].keys, item.keys).item()
    assert mx.array_equal(cache[0].values, item.values).item()
    assert cache[0].offset == 4
    assert cache[0].spare_keys is None


@pytest.mark.parametrize("kind", ["checksum", "missing-checksum", "layers-json", "missing-layers",
                                  "class", "fields", "offset", "extra-array", "wrong-model", "tokens"])
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
            if kind == "layers-json": meta["layers"] = "{"
            elif kind == "missing-layers": del meta["layers"]
            elif kind == "wrong-model": meta["model"] = "another-model"
            elif kind == "tokens": meta["tokens"] = "[true, 2, 3, 4]"
            elif kind == "extra-array": arrays["unknown"] = runtime[0].zeros((1,))
            else:
                if kind == "class": layers[0]["class"] = "os:system"
                elif kind == "fields": layers[0]["plain"]["__class__"] = "untrusted"
                elif kind == "offset": layers[0]["plain"]["offset"] = 999
                meta["layers"] = json.dumps(layers)
        rewrite(path, runtime, corrupt)
    assert disk.load(path) is None


@pytest.mark.parametrize("policy", ["cli", "env", "budget", "free-space"])
def test_policy_prevents_writes(tmp_path, runtime, monkeypatch, policy):
    disk, item = make_disk(tmp_path, runtime)
    if policy == "cli": disk_caches.configure(True)
    elif policy == "env": monkeypatch.setenv(disk_caches.ENV_VAR, "1")
    elif policy == "budget": disk.budget = 1
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
    scheduler = SimpleNamespace(checkpoints=SimpleNamespace(
        longest=lambda prompt, usable: 0,
        insert=lambda *args, **kw: inserted.append(args)), prompt_memory=None)
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
