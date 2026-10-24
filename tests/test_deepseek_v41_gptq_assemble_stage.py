"""CPU-only staged assembly must retain the real assembler's complete bytes."""

import json
import os
from copy import deepcopy
from types import SimpleNamespace

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
from scripts import deepseek_v41_gptq_amend as amend
from scripts import deepseek_v41_gptq_assemble_stage as s
from scripts.deepseek_v41_gptq_io import PartialSource, atomic_json, inventory
from tests.test_deepseek_v41_gptq_quality import fixture as quality_fixture


@pytest.fixture(scope="module", params=[False, True], ids=["original", "amended"])
def run(request, tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield _staged_run(request, tmp_path_factory)
    finally:
        mx.set_default_device(previous)


def _staged_run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("staged-assembly")
    # Use the same source/reference fixture as the amendment integration tests.
    source, c, _, _, _ = quality_fixture(root)
    # The shared quality fixture has no draft tensors: add a small real MTP
    # group to exercise dspark paths, config and index identity as well.
    raw = dict(mx.load(str(source.source / "source.safetensors")))
    raw["mtp.proj.weight"] = mx.ones((64, 64), dtype=mx.bfloat16)
    raw["mtp.norm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
    mx.save_safetensors(str(source.source / "source.safetensors"), raw)
    index_path = source.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update(
        {name: "source.safetensors" for name in raw if name.startswith("mtp.")}
    )
    atomic_json(index_path, index)
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
        # Exercise the fast trust path on a fresh stage whose complete
        # signature still matches its manifest.
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
        assert (tmp_path / "out" / relative).stat().st_ino != (
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


def test_assembly_rejects_identity_destination_and_budget(run, tmp_path, monkeypatch):
    wrong = {**run.c, "memory_limit_gb": 1}
    with pytest.raises(ValueError, match="identity"):
        b.assemble(run.source, wrong, run.parts, tmp_path / "direct", None)
    with pytest.raises(FileExistsError):
        b.assemble(run.source, run.c, run.parts, run.direct, None)
    with pytest.raises(ValueError, match="identity"):
        s.assemble_staged(
            run.source,
            wrong,
            run.parts,
            tmp_path / "staged",
            None,
            run.staging,
            tmp_path / "cache",
        )
    with pytest.raises(FileExistsError):
        s.assemble_staged(
            run.source,
            run.c,
            run.parts,
            run.direct,
            None,
            run.staging,
            tmp_path / "cache",
        )
    real = b.layout_report

    def over(*args):
        return {**real(*args), "resident_tensor_bytes": 1e30}

    monkeypatch.setattr(b, "layout_report", over)
    monkeypatch.setattr(s, "layout_report", over)
    with pytest.raises(ValueError, match="resident"):
        b.assemble(run.source, run.c, run.parts, tmp_path / "direct", None)
    with pytest.raises(ValueError, match="resident"):
        s.assemble_staged(
            run.source,
            run.c,
            run.parts,
            tmp_path / "staged",
            None,
            run.staging,
            tmp_path / "cache",
        )


def test_staging_rejects_binding_cache_and_verification_races(
    run, tmp_path, monkeypatch
):
    current = s.read_json(run.parts / "run.json")
    with pytest.raises(ValueError, match="source identity"):
        s.stage_other(SimpleNamespace(manifest={}), run.parts, tmp_path / "stage", None)
    cache = tmp_path / "cache.json"
    atomic_json(cache, {})
    with pytest.raises(ValueError, match="cache identity"):
        s.load_cache(run.parts, cache, current)
    record = s.read_json(run.parts / "layer-0-complete.json")
    with pytest.raises(ValueError, match="unsafe"):
        s.verify_layer(
            run.parts,
            current,
            0,
            {**record, "files": {"../bad": "sha"}},
            {"layers": {}},
        )
    real = s.signature
    count = 0

    def raced(path):
        nonlocal count
        count += 1
        result = real(path)
        if count > len(record["files"]):
            result["mtime_ns"] += 1
        return result

    monkeypatch.setattr(s, "signature", raced)
    with pytest.raises(ValueError, match="changed during verification"):
        s.verify_layer(run.parts, current, 0, record, {"layers": {}})


@pytest.mark.parametrize("race", [False, True])
def test_staged_cross_device_and_prelink_mutation(run, tmp_path, monkeypatch, race):
    real = s.signature
    seen = {}

    def changed(path):
        result = real(path)
        if path.is_relative_to(run.staging):
            seen[path] = seen.get(path, 0) + 1
            if not race:
                result["device"] = -1
            elif seen[path] >= 6:
                result["mtime_ns"] += 1
        return result

    monkeypatch.setattr(s, "signature", changed)
    with (
        b.MemoryGuard(run.c) as guard,
        pytest.raises(ValueError, match="before linking" if race else "filesystem"),
    ):
        s.assemble_staged(
            run.source,
            run.c,
            run.parts,
            tmp_path / "out",
            guard,
            run.staging,
            tmp_path / "cache",
        )


def test_assembly_retains_heldout_reports(run, tmp_path):
    import shutil

    parts = tmp_path / "parts"
    shutil.copytree(run.parts, parts)
    for name in ("heldout-drift.jsonl", "heldout-reference.json"):
        (parts / name).write_text("{}\n")
    with b.MemoryGuard(run.c) as guard:
        s.assemble_staged(
            run.source,
            run.c,
            parts,
            tmp_path / "out",
            guard,
            run.staging,
            tmp_path / "cache",
        )
    assert (tmp_path / "out/gptq-heldout-drift.jsonl").read_text() == "{}\n"
    assert (tmp_path / "out/gptq-heldout-reference.json").read_text() == "{}\n"


@pytest.mark.parametrize("mode", ["direct", "staged"])
@pytest.mark.parametrize("failure", ["copy", "metadata"])
def test_failed_assembly_can_retry_without_partial_destination(
    run, tmp_path, monkeypatch, mode, failure
):
    target = tmp_path / "published"
    module = b if mode == "direct" else s

    def assemble(guard):
        if mode == "direct":
            return b.assemble(run.source, run.c, run.parts, target, guard)
        return s.assemble_staged(
            run.source, run.c, run.parts, target, guard, run.staging, tmp_path / "cache"
        )

    with b.MemoryGuard(run.c) as guard:
        with monkeypatch.context() as patch:
            if failure == "copy":

                def fail_copy(*args):
                    raise OSError("injected copy failure")

                patch.setattr(module, "copy_verified", fail_copy)
            else:
                real = module.atomic_json

                def fail_metadata(path, value):
                    if path.name == "config.json":
                        raise OSError("injected metadata failure")
                    return real(path, value)

                patch.setattr(module, "atomic_json", fail_metadata)
            with pytest.raises(OSError, match="injected"):
                assemble(guard)
        assert not target.exists()
        assert not list(tmp_path.glob(".published.partial-*"))
        assemble(guard)
    assert file_bytes(target) == file_bytes(run.direct)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("mutation", ["name", "shape", "counts"])
def test_assembly_rejects_modified_completion_plan(run, tmp_path, staged, mutation):
    import shutil

    from scripts.deepseek_v41_gptq_io import file_hash

    parts = tmp_path / "parts"
    shutil.copytree(run.parts, parts)
    path = parts / "layer-0-complete.json"
    record = json.loads(path.read_text())
    if mutation == "counts":
        stats_path = parts / "layer-0-stats.json"
        stats = json.loads(stats_path.read_text())
        stats["routed_tokens"] = [-1]
        atomic_json(stats_path, stats)
        record["files"][stats_path.name] = file_hash(stats_path)
    elif mutation == "name":
        record["entries"][0]["name"] = "unplanned.weight"
    else:
        record["entries"][0]["shape"][0] += 1
    atomic_json(path, record)
    output = tmp_path / "out"
    with (
        b.MemoryGuard(run.c) as guard,
        pytest.raises(ValueError, match="completed layer"),
    ):
        if staged:
            s.assemble_staged(
                run.source,
                run.c,
                parts,
                output,
                guard,
                run.staging,
                tmp_path / "cache.json",
            )
        else:
            b.assemble(run.source, run.c, parts, output, guard)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_published_checkpoint_mutation_cannot_change_parts(run, tmp_path, mode):
    target = tmp_path / "published"
    original = (run.parts / "model-layer-0.safetensors").read_bytes()
    with b.MemoryGuard(run.c) as guard:
        if mode == "direct":
            b.assemble(run.source, run.c, run.parts, target, guard)
        else:
            s.assemble_staged(
                run.source,
                run.c,
                run.parts,
                target,
                guard,
                run.staging,
                tmp_path / "cache",
            )
    (target / "model-layer-0.safetensors").write_bytes(b"modified output")
    assert (run.parts / "model-layer-0.safetensors").read_bytes() == original


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_assembly_rejects_metadata_added_after_inventory(run, tmp_path, mode):
    import shutil

    root = tmp_path / "source"
    shutil.copytree(run.source.source, root)
    (root / "tokenizer_config.json").write_text('{"changed":true}')
    source = PartialSource(root, run.source.manifest)
    target = tmp_path / "published"
    with (
        b.MemoryGuard(run.c) as guard,
        pytest.raises(ValueError, match="metadata identity"),
    ):
        if mode == "direct":
            b.assemble(source, run.c, run.parts, target, guard)
        else:
            s.assemble_staged(
                source, run.c, run.parts, target, guard, run.staging, tmp_path / "cache"
            )
    assert not target.exists()
