"""Original-prefix RTN sensitivity ranking; never exports quantized weights.

RTN errors are ranking proxies, not estimates of achieved GPTQ quality. Dense
movement is sum squared block-output difference / sum squared reference output,
with identical fixed recipe auxiliaries and original routed weights in both
branches. Only the unmodified source branch advances the teacher boundary.
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from scripts import deepseek_v41_gptq as b
from scripts import deepseek_v41_gptq_quality as q
from scripts.deepseek_v41_gptq_io import atomic_json, digest, load_state
from scripts.deepseek_v41_rd_common import file_hash, read_slices


def calibration_subset(source, slices, heldout, tokens=512, windows_per_slice=2):
    """Nonoverlapping beginning/end windows from every calibration domain."""
    from tokenizers import Tokenizer

    if not 0 < tokens <= source.args.max_seq_len or windows_per_slice < 1:
        raise ValueError("invalid scan window sizes")
    paths, excluded = read_slices(slices, []), read_slices(heldout, [])
    excluded_paths = {p.resolve() for ps in excluded.values() for p in ps}
    excluded_hashes = {file_hash(p) for p in excluded_paths}
    tokenizer = Tokenizer.from_file(str(source.source / "tokenizer.json"))
    windows, names, evidence = [], [], {}
    for name, files in paths.items():
        hashes = [file_hash(p) for p in files]
        if any(
            p.resolve() in excluded_paths for p in files
        ) or excluded_hashes.intersection(hashes):
            raise ValueError("held-out text overlaps scan calibration")
        ids = tokenizer.encode("\n\n".join(p.read_text() for p in files)).ids
        if len(ids) < tokens * windows_per_slice:
            raise ValueError(f"{name}: insufficient independent scan tokens")
        starts = np.linspace(0, len(ids) - tokens, windows_per_slice, dtype=int)
        evidence[name] = dict(files=hashes, starts=starts.tolist(), length=len(ids))
        for i, start in enumerate(starts):
            windows.append(np.asarray(ids[start : start + tokens], np.int32))
            names.append(f"slice_{len(evidence) - 1}_window_{i}")
    provenance = dict(
        calibration_manifest_sha256=file_hash(slices),
        heldout_manifest_sha256=file_hash(heldout),
        tokenizer_sha256=file_hash(source.source / "tokenizer.json"),
        files=evidence,
        tokens=tokens,
        windows_per_slice=windows_per_slice,
        token_hashes=[digest(w.tolist()) for w in windows],
    )
    return windows, names, provenance


def scan_metadata(source, c, provenance):
    if c["group_size"] != 64 or c["dense_bits"] != 4 or c["gate_up_bits"] != 2:
        raise ValueError("scan requires dense4/gate-up2/group64 base recipe")
    if c["dense_8bit_layers"] or c["gate_up_3bit_layers"] or c["low_token_bump"]:
        raise ValueError("scan requires an unselected base recipe without token bumps")
    run = dict(
        schema_version=1,
        kind="rtn_scan",
        source_identity=digest(source.manifest),
        recipe=c,
        calibration=provenance,
        backend="cpu" if mx.default_device() == mx.cpu else "metal",
        mlx_version=mx.__version__,
    )
    run["identity"] = digest(run)
    return run


def validate_records(root, source, c):
    root = Path(root)
    run = json.loads((root / "run.json").read_text())
    if (
        run.get("kind") != "rtn_scan"
        or digest({k: v for k, v in run.items() if k != "identity"}) != run["identity"]
        or run["source_identity"] != digest(source.manifest)
        or run["recipe"] != c
    ):
        raise ValueError("scan source/recipe/identity mismatch")
    checkpoint = json.loads((root / "scan-checkpoint.json").read_text())
    if checkpoint != dict(identity=run["identity"], next_layer=source.args.n_layers):
        raise ValueError("selection requires a complete scan")
    records = []
    for i in range(source.args.n_layers):
        row = json.loads((root / f"layer-{i}-stats.json").read_text())
        if (
            row.get("layer") != i
            or row.get("run_identity") != run["identity"]
            or digest({k: v for k, v in row.items() if k != "sha256"})
            != row.get("sha256")
        ):
            raise ValueError("mixed or corrupt scan records")
        movement = row["block_output_4bit_relative_error"]
        expected = {
            f"expert.{e}.{p}"
            for e in range(source.args.n_routed_experts)
            for p in ("w1", "w3")
        }
        if (
            set(row["projections"]) != expected
            or not math.isfinite(movement)
            or movement < 0
        ):
            raise ValueError("invalid scan measurements")
        for p in row["projections"].values():
            for k in (
                "rtn_relative_output_error",
                "three_bit_rtn_relative_output_error",
                "calibration_output_energy",
            ):
                if not math.isfinite(p[k]) or p[k] < 0:
                    raise ValueError("invalid scan projection error")
        records.append(row)
    return records


def rtn_errors(weight, chunks, guard):
    """Exact routed, gate-weighted squared projection errors, no Hessian."""
    w = weight.astype(mx.float32)
    deltas = [
        w
        - mx.dequantize(
            *mx.quantize(weight, bits=bits, group_size=64),
            bits=bits,
            group_size=64,
            dtype=mx.float32,
        )
        for bits in (2, 3)
    ]
    mx.eval(deltas)
    energy, errors, count = 0.0, [0.0, 0.0], 0
    for x, gates in chunks():
        x = x.astype(mx.float32)
        g = mx.array(gates).astype(mx.float32)[:, None]
        y = x @ w.T
        energy += float(mx.sum(y * y * g).item())
        for i, delta in enumerate(deltas):
            err = x @ delta.T
            errors[i] += float(mx.sum(err * err * g).item())
        count += len(x)
        guard.check()
    return dict(
        rtn_relative_output_error=errors[0] / max(energy, 1e-30),
        three_bit_rtn_relative_output_error=errors[1] / max(energy, 1e-30),
        calibration_output_energy=energy,
        error_token_count=count,
        error_method="rtn_ranking_proxy",
    )


def linear(weight):
    module = nn.Linear(weight.shape[-1], weight.shape[-2], bias=False)
    module.weight = weight
    b.install_cpu_affine(module)
    return module


class RoutedInputs:
    """Map each spill once; sync outputs once per branch rather than per expert."""

    def __init__(self, inp, args, c):
        self.inp, self.args, self.c = inp, args, c
        total = inp["total"]
        paths = inp["paths"]
        self.x = np.memmap(
            paths["x"], mode="r", dtype=np.uint16, shape=(total, args.dim)
        )
        self.y = np.memmap(
            paths["y"], mode="r+", dtype=np.float32, shape=(total, args.dim)
        )
        ids = np.memmap(
            paths["ids"],
            mode="r",
            dtype=np.int32,
            shape=(total, args.n_activated_experts),
        )
        gates = np.memmap(paths["gates"], mode="r", dtype=np.float32, shape=ids.shape)
        self.routes = []
        for expert in range(args.n_routed_experts):
            rows, slots = np.where(ids == expert)
            self.routes.append((rows, np.array(gates[rows, slots])))

    def chunks(self, expert=None):
        rows, gates = (
            (np.arange(self.inp["total"]), None)
            if expert is None
            else self.routes[expert]
        )
        for start in range(0, len(rows), self.c["ffn_chunk"]):
            r = rows[start : start + self.c["ffn_chunk"]]
            x = mx.array(np.array(self.x[r])).view(mx.bfloat16)
            yield r, x, None if gates is None else gates[start : start + len(r)]

    def add(self, modules, guard, expert=None):
        for rows, x, gates in self.chunks(expert):
            y = modules[2](
                b.intermediate(
                    x, *modules[:2], self.args.swiglu_limit, routed=expert is not None
                )
            ).astype(mx.float32)
            if gates is not None:
                y *= mx.array(gates)[:, None]
            mx.eval(y)
            self.y[rows] += np.asarray(y)
            guard.check()

    def close(self):
        self.y.flush()
        del self.x, self.y, self.routes


def scan_layer(source, layer, windows, target, work, c, guard):
    """Three attention branches; one source read per expert for all branches.

    Disk spills bound activation memory. Original routing drives gate/up errors;
    each dense comparator reroutes on its own perturbed attention output.
    """
    if work.exists():
        shutil.rmtree(work)
    if target.exists():
        shutil.rmtree(target)  # only the uncommitted next boundary owned by scan
    work.mkdir(parents=True)
    blocks, inputs = [], []
    for branch in range(3):
        block = b.bounded_block(source, layer, c, original=branch == 0)
        if branch == 2:
            for name in b.DENSE:
                module = b.module_at(block, name)
                b.set_module(
                    block,
                    name,
                    b.packed_module(
                        module.weight,
                        mx.quantize(module.weight, bits=4, group_size=64),
                        4,
                        64,
                    ),
                )
            b.install_cpu_affine(block)
        blocks.append(block)
        inp = q.attention_inputs(
            block, windows, work / f"branch-{branch}", source.args, c, guard
        )
        with inp["paths"]["y"].open("wb") as f:
            f.truncate(inp["total"] * source.args.dim * 4)
        inputs.append(inp)
    routed = [RoutedInputs(inp, source.args, c) for inp in inputs]
    reports = {}
    for expert in range(source.args.n_routed_experts):
        modules = []
        for projection in ("w1", "w3", "w2"):
            w = source.tensor(
                f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"
            )
            if projection != "w2":

                def chunks(expert=expert):
                    for _, x, gates in routed[0].chunks(expert):
                        yield x, gates

                reports[f"expert.{expert}.{projection}"] = rtn_errors(w, chunks, guard)
            modules.append(linear(w))
        for rows in routed:
            rows.add(modules, guard, expert)
        del modules, w
        gc.collect()
        mx.clear_cache()
        if expert % 32 == 0:
            print(
                json.dumps(dict(scan_layer=layer, expert=expert, **guard.report())),
                flush=True,
            )
    for branch, (block, inp) in enumerate(zip(blocks, inputs)):
        shared = [
            b.module_at(block, "ffn.shared_experts." + p) for p in ("w1", "w3", "w2")
        ]
        routed[branch].add(shared, guard)
        routed[branch].close()
        out = target if branch == 0 else work / f"output-{branch}"
        b.finalize_windows(
            block,
            inp["stage"],
            inp["paths"],
            out,
            inp["total"],
            source.args,
            {**c, "pilot": True},
            guard,
            run_identity=digest(dict(source=source.manifest, recipe=c, branch=branch)),
            layer=layer,
        )
    numerator = denominator = 0.0
    for i in range(len(windows)):
        ref = load_state(work / "output-1" / f"window-{i}")["h"].astype(mx.float32)
        candidate = load_state(work / "output-2" / f"window-{i}")["h"].astype(
            mx.float32
        )
        numerator += float(mx.sum((candidate - ref) ** 2).item())
        denominator += float(mx.sum(ref * ref).item())
        guard.check()
    return dict(
        block_output_4bit_relative_error=numerator / max(denominator, 1e-30),
        projections=reports,
        error_method="rtn_ranking_proxy",
    )


def run_scan(
    source, root, c, windows, names, provenance, guard, through=None, token_map=None
):
    root = Path(root)
    provenance = {
        **provenance,
        "window_names": names,
        "actual_token_hashes": [
            digest(np.asarray(w, np.int32).tolist()) for w in windows
        ],
    }
    run = scan_metadata(source, c, provenance)
    if not (root / "run.json").exists():
        # Fail closed on partial initialization; never adopt an unrelated cache.
        if root.exists() and any(root.iterdir()):
            raise ValueError(
                "scan directory has incomplete initialization; use a new directory"
            )
        source.verify("top")
        q.reference_init(
            source, root, windows, names, provenance, c, guard, token_map=token_map
        )
        atomic_json(
            root / "scan-checkpoint.json", dict(identity=run["identity"], next_layer=0)
        )
        atomic_json(root / "run.json", run)
    if json.loads((root / "run.json").read_text()) != run:
        raise ValueError("scan source/calibration/recipe/backend identity changed")
    cp = json.loads((root / "scan-checkpoint.json").read_text())
    if (
        cp["identity"] != run["identity"]
        or not 0 <= cp["next_layer"] <= source.args.n_layers
    ):
        raise ValueError("scan checkpoint mismatch")
    reference = json.loads((root / "reference.json").read_text())
    stop = source.args.n_layers - 1 if through is None else through
    if not 0 <= stop < source.args.n_layers:
        raise ValueError("invalid scan through-layer")
    for layer in range(cp["next_layer"], stop + 1):
        started = time.monotonic()
        source.verify(layer)
        previous, target = root / f"boundary-{layer}", root / f"boundary-{layer + 1}"
        q.verify_boundary(previous, reference["identity"])
        row = scan_layer(
            source,
            layer,
            q.windows_at(previous, len(windows)),
            target,
            root / "scan-working",
            c,
            guard,
        )
        q.commit_boundary(
            target, dict(identity=reference["identity"], boundary=layer + 1)
        )
        row.update(
            layer=layer,
            run_identity=run["identity"],
            seconds=time.monotonic() - started,
        )
        row["sha256"] = digest(row)
        atomic_json(root / f"layer-{layer}-stats.json", row)
        atomic_json(
            root / "scan-checkpoint.json",
            dict(identity=run["identity"], next_layer=layer + 1),
        )
        # Only adjacent ORIGINAL boundaries are needed. Commit before pruning.
        shutil.rmtree(previous)
        shutil.rmtree(root / "scan-working")
        gc.collect()
        mx.clear_cache()
        print(
            json.dumps(
                dict(
                    scan_completed_layer=layer,
                    seconds=row["seconds"],
                    projected_total_minutes=row["seconds"] * source.args.n_layers / 60,
                    **guard.report(),
                )
            ),
            flush=True,
        )


def add_parser(sub):
    p = sub.add_parser("scan", description=__doc__)
    for name in ("source", "manifest", "slices", "heldout-slices", "scan"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--windows-per-slice", type=int, default=2)
    p.add_argument("--through-layer", type=int)
    p.add_argument("--cpu", action="store_true")


def execute(a):
    from scripts.deepseek_v41_gptq_io import PartialSource

    if a.cpu:
        mx.set_default_device(mx.cpu)
    elif not mx.metal.is_available():
        raise RuntimeError(
            "scan needs Metal; use --cpu only for tiny CPU qualification"
        )
    c = b.recipe(json.loads(a.recipe.read_text()) if a.recipe else None)
    mx.random.seed(c["seed"])
    source = PartialSource(a.source, json.loads(a.manifest.read_text()))
    windows, names, provenance = calibration_subset(
        source, a.slices, a.heldout_slices, a.tokens, a.windows_per_slice
    )
    with b.MemoryGuard(c) as guard:
        run_scan(source, a.scan, c, windows, names, provenance, guard, a.through_layer)
