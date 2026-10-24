"""Validated, append-only recipe epochs for layer-streamed GPTQ runs."""

import json
import math
from copy import deepcopy
from datetime import datetime, timezone

from scripts.deepseek_v41_gptq_fullrun import atomic as atomic_json
from scripts.deepseek_v41_gptq_fullrun import digest


def allowed_change(old, new, first):
    if set(old) != set(new):
        raise ValueError("amendment requires complete recipes")
    for key in ("resident_budget_gb", "memory_limit_gb", "cache_limit_gb"):
        if (
            type(new[key]) not in (int, float)
            or not math.isfinite(new[key])
            or new[key] <= 0
        ):
            raise ValueError("invalid memory cap")
    for c in (old, new):
        layers = c["gate_up_3bit_layers"]
        if (
            not isinstance(layers, list)
            or any(type(i) is not int or i < 0 for i in layers)
            or len(layers) != len(set(layers))
        ):
            raise ValueError("invalid gate_up_3bit_layers")
    allowed = {
        "resident_budget_gb",
        "memory_limit_gb",
        "cache_limit_gb",
        "gate_up_3bit_layers",
    }
    if any(old[k] != new[k] for k in old if k not in allowed):
        raise ValueError("recipe change can affect finished layers")
    # Changing membership below the boundary changes finished tensors. Entries
    # already in force below the boundary may remain unchanged.
    if any(
        i < first
        for i in set(old["gate_up_3bit_layers"]) ^ set(new["gate_up_3bit_layers"])
    ):
        raise ValueError("gate/up recipe change affects finished layers")
    if old == new:
        raise ValueError("recipe amendment makes no change")


def epochs(run):
    """Replay and authenticate every epoch, including the original run digest."""
    amendments = run.get("recipe_amendments", [])
    if not isinstance(amendments, list):
        raise ValueError("invalid recipe amendment chain")
    state = deepcopy(run)
    state.pop("identity", None)
    state.pop("recipe_amendments", None)
    if amendments:
        state["recipe"] = amendments[0]["old_recipe"]
    result = [(0, digest(state), state["recipe"])]
    for index, record in enumerate(amendments):
        first = record["from_layer"]
        if type(first) is not int or first < result[-1][0]:
            raise ValueError("invalid amendment boundary")
        if (
            record["old_identity"] != result[-1][1]
            or record["old_recipe"] != state["recipe"]
        ):
            raise ValueError("broken recipe amendment chain")
        if not record["reason"].strip() or not record["timestamp"]:
            raise ValueError("missing amendment audit information")
        allowed_change(state["recipe"], record["new_recipe"], first)
        state["recipe"] = record["new_recipe"]
        state["recipe_amendments"] = amendments[: index + 1]
        result.append((first, digest(state), state["recipe"]))
    if digest(state) != run["identity"] or state["recipe"] != run["recipe"]:
        raise ValueError("run identity does not authenticate recipe amendment chain")
    return result


def layer_epoch(run, layer):
    if type(layer) is not int or layer < 0:
        raise ValueError("invalid layer")
    return next(e for e in reversed(epochs(run)) if e[0] <= layer)


def validate_record(run, layer, record):
    if record["run_identity"] != layer_epoch(run, layer)[1]:
        raise ValueError("unknown or wrong layer run identity")


def amend_recipe(parts, scratch, new, first, reason):
    """Caller must stop the driver and hold its exclusive large-model lock.

    run.json is the transaction journal: if interrupted between the two atomic
    replacements, retrying this exact amendment repairs only checkpoint identity.
    Readers fail closed during that interval. No layer artifacts are rewritten.
    """
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from scripts.deepseek_v41_gptq_io import entries_from_json, verify_artifact

    run_path, checkpoint_path = parts / "run.json", scratch / "checkpoint.json"
    run = json.loads(run_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    epochs(run)
    manifest = json.loads((parts / "source-manifest.json").read_text())
    if digest(manifest) != run["source_identity"]:
        raise ValueError("source manifest identity mismatch")
    count = ModelArgs.from_dict(manifest["config"]).n_layers
    layers = new.get("gate_up_3bit_layers")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(layers, list)
        or any(type(i) is not int or not 0 <= i < count for i in layers)
    ):
        raise ValueError("amended layer ID outside source model")
    if type(first) is not int or first < 0 or first != checkpoint["next_layer"]:
        raise ValueError("from-layer must equal checkpoint next_layer")
    if checkpoint.get("phase") not in (None, "boundary"):
        raise ValueError("cannot amend mid-layer")
    # Attention begins before the FFN phase is recorded. A working directory or
    # orphaned next-layer output therefore also means this is not a clean stop.
    if (scratch / f"working-{first}").exists() or (
        scratch / f"boundary-{first + 1}"
    ).exists():
        raise ValueError("cannot amend mid-layer or with uncommitted layer work")
    if any(
        (parts / name).exists()
        for name in (
            f"layer-{first}-complete.json",
            f"model-layer-{first}.safetensors",
            f"layer-{first}-stats.json",
            f"layer-{first}-profile.json",
        )
    ):
        raise ValueError("uncommitted layer output exists")
    if not reason.strip():
        raise ValueError("amendment reason is required")
    budget = new.get("resident_budget_gb")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("invalid memory cap")
    committed_bytes = 0
    for layer in range(first):
        record = json.loads((parts / f"layer-{layer}-complete.json").read_text())
        validate_record(run, layer, record)
        verify_artifact(parts, record)
        committed_bytes += sum(
            entry.nbytes
            for entry in entries_from_json(record["entries"])
            if entry.component != "engram_ssd"
        )
    if committed_bytes > budget * 1e9:
        raise ValueError("amended resident budget is below committed payload size")
    if checkpoint["identity"] != run["identity"]:
        last = run.get("recipe_amendments", [{}])[-1]
        if not (
            last.get("old_identity") == checkpoint["identity"]
            and last.get("from_layer") == first
            and last.get("new_recipe") == new
            and last.get("reason") == reason
        ):
            raise ValueError("checkpoint/run identity mismatch")
    else:
        allowed_change(run["recipe"], new, first)
        record = dict(
            old_recipe=run["recipe"],
            old_identity=run["identity"],
            new_recipe=new,
            from_layer=first,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        run.setdefault("recipe_amendments", []).append(record)
        run["recipe"] = new
        run.pop("identity")
        run["identity"] = digest(run)
        epochs(run)
        atomic_json(run_path, run)
    checkpoint["identity"] = run["identity"]
    atomic_json(checkpoint_path, checkpoint)
    return run
