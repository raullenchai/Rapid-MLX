#!/usr/bin/env python3
"""CPU-only incremental staging and hash verification for a live GPTQ run.

Never writes parts/run state. Staging and cache have independent advisory locks;
committed input files must remain immutable, while assembly reads them.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from scripts.deepseek_v41_gptq import MemoryGuard, allocation, output_plan, recipe
from scripts.deepseek_v41_gptq_amend import epochs, validate_record
from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    atomic_destination,
    atomic_json,
    conversion_record,
    copy_bound_metadata,
    copy_verified,
    digest,
    entries_json,
    verify_artifact,
)
from scripts.deepseek_v41_rd_build import (
    existing_parent,
    group_path,
    groups_for,
    header,
    layout_report,
    write_group,
)
from scripts.deepseek_v41_rd_common import file_hash

# Only these recipe fields influence non-layer output_plan entries. MTP has its
# own fixed 4-bit/group-64 policy; dense/routed precision and caps affect layers.
# Entry digests independently guard against changes to the planner itself.
BOUND_FIELDS = ("engram_bits", "engram_group_size", "keep_mtp")
VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular immutable file: {path}")
    st = path.stat()
    return dict(
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        inode=st.st_ino,
        device=st.st_dev,
        ctime_ns=st.st_ctime_ns,
    )


def checked_hash(path, expected=None):
    before = signature(path)
    sha = file_hash(path)
    if signature(path) != before:
        raise ValueError(f"file changed during hashing: {path}")
    if expected is not None and sha != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
    return dict(**before, sha256=sha)


def other_groups(source, c):
    entries, _ = output_plan(source, c)
    return {
        g: es for g, es in groups_for(entries).items() if not g.startswith("layer-")
    }


def stage_binding(source, c, groups):
    return dict(
        schema_version=VERSION,
        source_digest=digest(source.manifest),
        recipe_fields=list(BOUND_FIELDS),
        recipe={k: c[k] for k in BOUND_FIELDS},
        entry_digests={g: digest(entries_json(es)) for g, es in groups.items()},
    )


def check_binding(stage, binding):
    if any(stage.get(k) != v for k, v in binding.items()):
        raise ValueError("staging source/recipe/entry binding mismatch")
    if set(stage.get("groups", {})) - set(binding["entry_digests"]):
        raise ValueError("unexpected staged group")


def check_stage_file(staging, group, es, record, trust=False):
    path = staging / group_path(group)
    expected_size = 8 + len(header(es)) + sum(e.nbytes for e in es)
    current = signature(path)
    if record["size"] != expected_size or current["size"] != expected_size:
        raise ValueError(f"staged size mismatch: {group}")
    if not (trust and all(record.get(k) == v for k, v in current.items())):
        checked_hash(path, record["sha256"])
    return record["sha256"]


def stage_other(source, parts, staging, guard):
    """Publish each complete group then its manifest record; resume partial writes."""
    parts, staging = Path(parts), Path(staging)
    with locked(staging / ".stage.lock"):
        run = read_json(parts / "run.json")
        epochs(run)
        c = recipe(run["recipe"])
        if run["source_identity"] != digest(source.manifest):
            raise ValueError("staging source identity mismatch")
        groups = other_groups(source, c)
        binding = stage_binding(source, c, groups)
        manifest_path = staging / "stage-manifest.json"
        stage = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {**binding, "groups": {}}
        )
        check_binding(stage, binding)
        atomic_json(manifest_path, stage)
        # Source inventory pins full shard payloads as well as sizes and headers.
        source.verify("top")
        if c["keep_mtp"]:
            source.verify("mtp")
        for layer in sorted(
            {int(g.split("-")[1]) for g in groups if g.startswith("engram-")}
        ):
            source.verify(layer)
        started = time.monotonic()
        for group, es in groups.items():
            target = staging / group_path(group)
            if group in stage["groups"]:
                check_stage_file(staging, group, es, stage["groups"][group])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # write_group already uses exclusive .partial + fsync + rename.
            # An unrecorded final file is an interrupted publication: rewrite it.
            partial = target.with_suffix(".partial")
            if partial.exists() or partial.is_symlink():
                partial.unlink()
            write_group(target, es, source.index, source.config, row_chunk=65536)
            component = (
                int(group.split("-")[1])
                if group.startswith("engram-")
                else ("mtp" if group == "mtp" else "top")
            )
            source.verify(component)
            guard.check()
            stage["groups"][group] = checked_hash(target)
            stage["profile"] = dict(
                seconds=time.monotonic() - started, **guard.report()
            )
            atomic_json(manifest_path, stage)
            print(json.dumps(dict(staged_group=group, **stage["profile"])), flush=True)
        # A concurrent compatible layer-only amendment is permitted, but a change
        # to staged semantics must never be silently accepted.
        current = read_json(parts / "run.json")
        epochs(current)
        check_binding(
            stage,
            stage_binding(
                source, current["recipe"], other_groups(source, current["recipe"])
            ),
        )
        return stage


def load_cache(parts, cache, run):
    binding = dict(
        schema_version=VERSION,
        parts=str(parts.resolve()),
        source_digest=run["source_identity"],
    )
    value = read_json(cache) if cache.exists() else {**binding, "layers": {}}
    if any(value.get(k) != v for k, v in binding.items()):
        raise ValueError("verification cache identity mismatch")
    return value


def verify_layer(parts, run, layer, record, cache):
    # Always authenticate epoch before even considering a cached hash.
    validate_record(run, layer, record)
    paths = {}
    for name in record["files"]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("unsafe layer artifact name")
        paths[name] = parts / name
    before = {name: signature(path) for name, path in paths.items()}
    key = str(layer)
    wanted = dict(
        layer=layer,
        record_digest=digest(record),
        files={
            name: dict(**meta, sha256=record["files"][name])
            for name, meta in before.items()
        },
    )
    if cache["layers"].get(key) != wanted:
        verify_artifact(parts, record)
        if {name: signature(path) for name, path in paths.items()} != before:
            raise ValueError(f"layer changed during verification: {layer}")
        cache["layers"][key] = wanted
    return wanted


def verify_layers(parts, cache):
    """Verify only complete records; persist progress after every layer."""
    parts, cache = Path(parts), Path(cache)
    with locked(cache.with_name(cache.name + ".lock")):
        run = read_json(parts / "run.json")
        epochs(run)
        value = load_cache(parts, cache, run)
        for path in sorted(parts.glob("layer-*-complete.json")):
            layer = int(path.name.split("-")[1])
            verify_layer(parts, run, layer, read_json(path), value)
            atomic_json(cache, value)
        return value


def assemble_staged(
    source,
    c,
    parts,
    destination,
    guard,
    staging,
    verify_cache,
    *,
    trust_stage_hash=False,
):
    parts, destination, staging, verify_cache = map(
        Path, (parts, destination, staging, verify_cache)
    )
    with (
        locked(staging / ".stage.lock"),
        locked(verify_cache.with_name(verify_cache.name + ".lock")),
        atomic_destination(destination) as output,
    ):
        return _assemble_staged(
            source,
            c,
            parts,
            output,
            guard,
            staging,
            verify_cache,
            trust_stage_hash,
        )


def _assemble_staged(
    source, c, parts, destination, guard, staging, verify_cache, trust_stage_hash
):
    """Assembly tail kept aligned with assemble(); byte-identity tests guard drift."""
    from scripts.deepseek_v41_gptq_amend import epochs, validate_record
    from scripts.deepseek_v41_rd_build import group_path

    run = json.loads((parts / "run.json").read_text())
    epochs(run)
    if run["recipe"] != c or run["source_identity"] != digest(source.manifest):
        raise ValueError("assembly recipe/source identity mismatch")
    cache = load_cache(parts, verify_cache, run)
    entries, _ = output_plan(source, c)
    layer_groups, completion = {}, {}
    for i in range(source.args.n_layers):
        record = json.loads((parts / f"layer-{i}-complete.json").read_text())
        validate_record(run, i, record)
        verify_layer(parts, run, i, record, cache)
        atomic_json(verify_cache, cache)
        completion[str(i)] = record
        from scripts.deepseek_v41_gptq import validated_layer_entries

        layer_groups[f"layer-{i}"] = validated_layer_entries(
            source, run, i, record, parts
        )
    other = {
        g: es for g, es in groups_for(entries).items() if not g.startswith("layer-")
    }
    stage = read_json(staging / "stage-manifest.json")
    check_binding(stage, stage_binding(source, c, other))
    if set(stage["groups"]) != set(other):
        raise ValueError("incomplete staging manifest")
    staged_hashes, staged_signatures = {}, {}
    device = existing_parent(destination.parent).stat().st_dev
    for group, es in other.items():
        path = staging / group_path(group)
        if signature(path)["device"] != device:
            raise ValueError("staging and destination must share a filesystem")
        staged_hashes[str(group_path(group))] = check_stage_file(
            staging, group, es, stage["groups"][group], trust_stage_hash
        )
        staged_signatures[group] = signature(path)
    all_entries = [e for es in [*layer_groups.values(), *other.values()] for e in es]
    totals = {}
    for e in all_entries:
        totals[e.component] = totals.get(e.component, 0) + e.nbytes
    costs = layout_report(
        all_entries,
        dict(
            components=totals,
            resident_tensor_bytes=sum(
                v for k, v in totals.items() if k != "engram_ssd"
            ),
            shared_engram_bytes=totals.get("engram_ssd", 0),
            total_payload_bytes=sum(totals.values()),
        ),
    )
    if costs["resident_tensor_bytes"] > c["resident_budget_gb"] * 1e9:
        raise ValueError("assembled resident payload exceeds budget")
    print(json.dumps(dict(assembly_costs=costs)), flush=True)
    destination.mkdir(parents=True)
    mapping, mtp_mapping = {}, {}
    for group, es in {**layer_groups, **other}.items():
        target = destination / group_path(group)
        target.parent.mkdir(parents=True, exist_ok=True)
        if group.startswith("layer-"):
            # Staged hash checks can take minutes: recheck the cached file
            # signatures immediately before copying each immutable layer.
            layer = int(group.split("-")[1])
            verify_layer(parts, run, layer, completion[str(layer)], cache)
            copy_verified(
                parts / target.name,
                target,
                completion[str(layer)]["files"][target.name],
            )
        else:
            # Copy only the exact staged payload verified above.
            path = staging / group_path(group)
            if signature(path) != staged_signatures[group]:
                raise ValueError(f"staged file changed before linking: {group}")
            copy_verified(path, target, staged_hashes[str(group_path(group))])
            guard.check()
        for e in es:
            (mtp_mapping if group == "mtp" else mapping)[e.name] = target.name
    config = json.loads(json.dumps(source.config))
    config.pop("quantization_config", None)
    config["model_type"] = "deepseek_v41"
    modules = {
        e.name.rsplit(".", 1)[0]: dict(bits=e.bits, group_size=e.group_size)
        for e in all_entries
        if e.bits and e.suffix == "weight" and e.component != "engram_ssd"
    }
    config["quantization"] = dict(
        bits=8,
        group_size=64,
        modules={n: v for n, v in modules.items() if not n.startswith("mtp.")},
        engram_bits=c["engram_bits"],
        engram_group_size=c["engram_group_size"],
    )
    actual_allocation = allocation(source, c)
    for e in all_entries:
        if e.component == "routed_experts" and e.suffix == "weight":
            layer_id, projection = e.name.split(".")[1], e.name.split(".")[-2]
            actual_allocation["layers"][layer_id]["projection_bits"][projection] = (
                e.bits
            )
    config["rapid_quantization"] = dict(
        format="native-packed-affine",
        allocation=actual_allocation,
        shared_engram_identity=digest(
            dict(
                source=digest(source.manifest),
                bits=c["engram_bits"],
                group_size=c["engram_group_size"],
            )
        ),
        mtp=bool(mtp_mapping),
        vision=False,
        bytes=costs,
        gptq=dict(
            recipe=c,
            recipe_amendments=run.get("recipe_amendments", []),
            calibration_manifest_sha256=run.get(
                "calibration_manifest_sha256", run["calibration_identity"]
            ),
            calibration_content_sha256=run["calibration_identity"],
            source_revision=run["source_revision"],
            run_identity=run["identity"],
            heldout_reference_identity=run.get("heldout_reference_identity"),
        ),
    )
    if mtp_mapping:
        draft = {**config, **config.get("text_config", {})}
        draft["quantization"] = dict(
            bits=8,
            group_size=64,
            modules={n: v for n, v in modules.items() if n.startswith("mtp.")},
        )
        atomic_json(destination / "dspark" / "config.json", draft)
        atomic_json(
            destination / "dspark" / "model.safetensors.index.json",
            dict(weight_map=mtp_mapping),
        )
    copy_bound_metadata(source, destination)
    if (parts / "engram_token_map.json").exists():
        shutil.copyfile(
            parts / "engram_token_map.json", destination / "engram_token_map.json"
        )
    atomic_json(
        destination / "model.safetensors.index.json",
        dict(
            metadata=dict(
                total_size=costs["total_payload_bytes"] - totals.get("mtp", 0)
            ),
            weight_map=mapping,
        ),
    )
    atomic_json(destination / "config.json", config)
    atomic_json(
        destination / "conversion-provenance.json",
        conversion_record(source, destination),
    )
    atomic_json(destination / "gptq-provenance.json", run)
    statistics = destination / "gptq-statistics"
    statistics.mkdir()
    for i in range(source.args.n_layers):
        shutil.copyfile(parts / f"layer-{i}-stats.json", statistics / f"layer-{i}.json")
    for name in ("heldout-drift.jsonl", "heldout-reference.json"):
        if (parts / name).exists():
            shutil.copyfile(parts / name, destination / ("gptq-" + name))
    atomic_json(destination / "gptq-layer-provenance.json", completion)
    (destination / "README.md").write_text(
        "# Experimental native GPTQ build\n\nSee gptq-provenance.json and per-layer calibration reports. Quality qualification pending.\n"
    )
    known = {
        name: sha
        for record in completion.values()
        for name, sha in record["files"].items()
    }
    known.update(staged_hashes)
    pinned = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(destination))
            pinned.append(
                dict(
                    name=relative,
                    size=path.stat().st_size,
                    sha256=known.get(relative) or file_hash(path),
                )
            )
    atomic_json(destination / "gptq-artifacts.json", dict(files=pinned))
    return costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-layers")
    verify.add_argument("--parts", type=Path, required=True)
    verify.add_argument("--cache", type=Path, required=True)
    for command in ("stage-other", "assemble-staged"):
        p = sub.add_parser(command)
        for name in ("manifest", "source", "parts", "staging"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument(
            "--cpu",
            action="store_true",
            help="accepted for assemble compatibility; always CPU",
        )
        p.add_argument(
            "--recipe",
            type=Path,
            help="optional for staging; must equal CURRENT run.json recipe",
        )
        p.add_argument(
            "--scratch", type=Path, help="accepted for assemble compatibility; unused"
        )
        if command == "assemble-staged":
            p.add_argument("--destination", type=Path, required=True)
            p.add_argument("--verify-cache", type=Path, required=True)
            p.add_argument("--trust-stage-hash", action="store_true")
    args = parser.parse_args()
    from scripts.deepseek_v41_gptq_lock import command_locks

    with command_locks(args):
        execute_command(args)


def execute_command(args):
    mx.set_default_device(mx.cpu)
    if args.command == "verify-layers":
        value = verify_layers(args.parts, args.cache)
        print(
            json.dumps(dict(verified_layers=sorted(map(int, value["layers"])))),
            flush=True,
        )
        return
    current = read_json(args.parts / "run.json")["recipe"]
    c = recipe(read_json(args.recipe) if args.recipe else current)
    if c != current:
        raise ValueError("recipe must match CURRENT run.json")
    source = PartialSource(args.source, read_json(args.manifest))
    # Separate CPU process, capped at 8 GB irrespective of the live GPU run's cap.
    # Do not change recipe metadata: guard caps only the staging process.
    limits = {
        **c,
        "memory_limit_gb": min(c["memory_limit_gb"], 8.0),
        "cache_limit_gb": min(c["cache_limit_gb"], 0.25),
    }
    with MemoryGuard(limits) as guard:
        if args.command == "stage-other":
            stage_other(source, args.parts, args.staging, guard)
        else:
            assemble_staged(
                source,
                c,
                args.parts,
                args.destination,
                guard,
                args.staging,
                args.verify_cache,
                trust_stage_hash=args.trust_stage_hash,
            )
        print(json.dumps(guard.report()), flush=True)


if __name__ == "__main__":
    main()
