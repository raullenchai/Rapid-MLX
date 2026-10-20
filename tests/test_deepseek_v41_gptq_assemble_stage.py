"""CPU-only staged assembly must retain the real assembler's complete bytes."""

import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_amend as amend
from scripts import deepseek_v41_gptq_assemble_stage as s
from scripts.deepseek_v41_gptq_io import PartialSource, atomic_json, inventory
from tests.test_deepseek_v41_gptq_quality import fixture as quality_fixture


@pytest.fixture(scope="module", params=[False, True], ids=["original", "amended"])
def run(request, tmp_path_factory):
    mx.set_default_device(mx.cpu)
    root = tmp_path_factory.mktemp("staged-assembly")
    # Use the same source/reference fixture as the amendment integration tests.
    source, c, _, _, _ = quality_fixture(root)
    # The shared quality fixture has no draft tensors: add a small real MTP
    # group to exercise dspark paths, config and index identity as well.
    raw = dict(mx.load(str(source.source / "source.safetensors")))
    raw["mtp.proj.weight"] = mx.ones((64, 64), dtype=mx.bfloat16)
    raw["mtp.norm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
    mx.save_safetensors(str(source.source / "source.safetensors"), raw)
    source = PartialSource(source.source, inventory(source.source))
    c = b.recipe(
        {
            **c,
            "pilot": False,
            "pilot_compare_gate_up": False,
            "dense_8bit_layers": [0, 1, 2, 3],
            "keep_mtp": True,
        }
    )
    parts, scratch = root / "parts", root / "scratch"
    staging = root / "early-stage"
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
        )
        # Stage before any layer is done; an allowed layer-only amendment must
        # preserve this work and the cached records of the retained prefix.
        s.stage_other(source, parts, staging, guard)
        for layer in range(2):
            b.run_layer(source, c, parts, scratch, layer, guard)
        cache = root / "early-cache.json"
        s.verify_layers(parts, cache)
        if request.param:
            c = {**c, "gate_up_3bit_layers": [2, 3], "resident_budget_gb": 216.0}
            amend.amend_recipe(parts, scratch, c, 2, "synthetic retained prefix")
        for layer in range(2, 4):
            b.run_layer(source, c, parts, scratch, layer, guard)
        direct = root / "direct"
        b.assemble(source, c, parts, direct, guard)
    return SimpleNamespace(
        root=root,
        source=source,
        c=c,
        parts=parts,
        staging=staging,
        cache=cache,
        direct=direct,
    )


def file_bytes(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("trust", [False, True])
def test_byte_identity_and_no_rehash(run, tmp_path, monkeypatch, trust):
    staging = run.staging
    if trust:
        # A prior assembly hardlink changes ctime; exercise the fast trust path
        # on a fresh stage whose complete signature still matches its manifest.
        staging = tmp_path / "fresh-stage"
        with b.MemoryGuard(run.c) as guard:
            s.stage_other(run.source, run.parts, staging, guard)
    s.verify_layers(run.parts, run.cache)

    def unexpected(*args, **kwargs):
        pytest.fail("cached layer rehashed or staged group rewritten")

    monkeypatch.setattr(s, "verify_artifact", unexpected)
    monkeypatch.setattr(s, "write_group", unexpected)
    real_hash = s.file_hash
    hashed = []

    def track(path):
        hashed.append(str(path))
        return real_hash(path)

    monkeypatch.setattr(s, "file_hash", track)
    with b.MemoryGuard(run.c) as guard:
        s.assemble_staged(
            run.source,
            run.c,
            run.parts,
            tmp_path / "out",
            guard,
            staging,
            run.cache,
            trust_stage_hash=trust,
        )
    assert file_bytes(tmp_path / "out") == file_bytes(run.direct)
    for group in s.other_groups(run.source, run.c):
        relative = s.group_path(group)
        assert (tmp_path / "out" / relative).stat().st_ino == (
            staging / relative
        ).stat().st_ino
    assert not any("model-layer-" in name for name in hashed)
    assert not any(
        str(tmp_path / "out" / s.group_path(g)) in hashed
        for g in s.other_groups(run.source, run.c)
    )
    if trust:
        assert not any(str(staging) in name for name in hashed)


def test_resume_partial_group(run, tmp_path, monkeypatch):
    staging = tmp_path / "stage"
    real = s.write_group
    calls = []

    def crash(target, es, index, config, row_chunk):
        calls.append(target)
        if len(calls) == 2:
            target.with_suffix(".partial").write_bytes(b"incomplete group")
            raise RuntimeError("interruption")
        return real(target, es, index, config, row_chunk=row_chunk)

    monkeypatch.setattr(s, "write_group", crash)
    with b.MemoryGuard(run.c) as guard, pytest.raises(RuntimeError):
        s.stage_other(run.source, run.parts, staging, guard)
    manifest = s.read_json(staging / "stage-manifest.json")
    assert len(manifest["groups"]) == 1
    first = calls[0]
    inode = first.stat().st_ino
    calls.clear()
    monkeypatch.setattr(s, "write_group", real)
    with b.MemoryGuard(run.c) as guard:
        result = s.stage_other(run.source, run.parts, staging, guard)
        s.stage_other(run.source, run.parts, staging, guard)
    assert first.stat().st_ino == inode
    assert len(result["groups"]) == len(s.other_groups(run.source, run.c))
    assert not list(staging.rglob("*.partial"))


@pytest.mark.parametrize(
    "change", ["recipe", "source", "file", "size", "entries", "missing"]
)
@pytest.mark.parametrize("trust", [False, True])
def test_stale_stage_rejected(run, tmp_path, change, trust):
    staging = tmp_path / "stage"
    with b.MemoryGuard(run.c) as guard:
        s.stage_other(run.source, run.parts, staging, guard)
    path = staging / "stage-manifest.json"
    manifest = s.read_json(path)
    if change == "recipe":
        manifest["recipe"]["engram_bits"] = 8
    elif change == "source":
        manifest["source_digest"] = "changed source"
    elif change == "entries":
        manifest["entry_digests"][next(iter(manifest["groups"]))] = "changed entries"
    elif change == "missing":
        manifest["groups"].pop(next(iter(manifest["groups"])))
    else:
        group = next(iter(manifest["groups"]))
        target = staging / s.group_path(group)
        with target.open("r+b") as f:
            if change == "size":
                f.truncate(10)
            else:
                f.seek(-1, 2)
                byte = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([byte[0] ^ 1]))
    atomic_json(path, manifest)
    with b.MemoryGuard(run.c) as guard, pytest.raises(ValueError):
        s.assemble_staged(
            run.source,
            run.c,
            run.parts,
            tmp_path / "out",
            guard,
            staging,
            tmp_path / "cache.json",
            trust_stage_hash=trust,
        )
    assert not (tmp_path / "out").exists()


def test_changed_current_recipe_and_source_rejected(run, tmp_path):
    stage = s.read_json(run.staging / "stage-manifest.json")
    c = {**run.c, "engram_bits": 8}
    with pytest.raises(ValueError, match="binding"):
        s.check_binding(
            stage, s.stage_binding(run.source, c, s.other_groups(run.source, c))
        )
    manifest = deepcopy(run.source.manifest)
    manifest["source_revision"] = "new revision"
    source = PartialSource(run.source.source, manifest)
    with pytest.raises(ValueError, match="binding"):
        s.check_binding(
            stage, s.stage_binding(source, run.c, s.other_groups(source, run.c))
        )


def test_cache_only_verifies_new_or_changed_layers(run, tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    real = s.verify_artifact
    calls = []

    def track(parts, record):
        calls.append(record)
        return real(parts, record)

    monkeypatch.setattr(s, "verify_artifact", track)
    s.verify_layers(run.parts, cache)
    assert len(calls) == 4
    calls.clear()
    s.verify_layers(run.parts, cache)
    assert not calls
    # Use independent copies so shared fixture/hardlinked outputs stay immutable.
    import shutil

    parts = tmp_path / "parts"
    shutil.copytree(run.parts, parts)
    cache2 = tmp_path / "cache2.json"
    s.verify_layers(parts, cache2)
    calls.clear()
    part = parts / "model-layer-0.safetensors"
    # Metadata-only change requires re-verification even for identical payload.
    st = part.stat()
    os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
    s.verify_layers(parts, cache2)
    assert len(calls) == 1
    calls.clear()
    with part.open("r+b") as f:
        f.seek(-1, 2)
        old = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([old[0] ^ 1]))
    os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="artifact changed"):
        s.verify_layers(parts, cache2)
    assert len(calls) == 1


def test_cache_record_and_epoch_validation(run, tmp_path, monkeypatch):
    cache = s.verify_layers(run.parts, tmp_path / "cache.json")
    record = s.read_json(run.parts / "layer-0-complete.json")
    changed = {**record, "extra_audit_field": "new digest"}
    calls = []
    real = s.verify_artifact
    monkeypatch.setattr(
        s, "verify_artifact", lambda p, r: (calls.append(r), real(p, r))
    )
    current = s.read_json(run.parts / "run.json")
    s.verify_layer(run.parts, current, 0, changed, cache)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        s.verify_layer(
            run.parts, current, 0, {**record, "run_identity": "foreign"}, cache
        )


def test_cli_cpu_end_to_end(run, tmp_path):
    import subprocess
    import sys

    tool = str(s.Path(s.__file__).resolve())
    manifest = tmp_path / "source-manifest.json"
    atomic_json(manifest, run.source.manifest)
    common = [
        "--source",
        str(run.source.source),
        "--manifest",
        str(manifest),
        "--parts",
        str(run.parts),
        "--staging",
        str(tmp_path / "stage"),
    ]
    commands = [
        ["stage-other", *common],
        [
            "verify-layers",
            "--parts",
            str(run.parts),
            "--cache",
            str(tmp_path / "cache.json"),
        ],
        [
            "assemble-staged",
            *common,
            "--destination",
            str(tmp_path / "out"),
            "--verify-cache",
            str(tmp_path / "cache.json"),
            "--trust-stage-hash",
        ],
    ]
    for args in commands:
        result = subprocess.run(
            [sys.executable, tool, *args], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert file_bytes(tmp_path / "out") == file_bytes(run.direct)
