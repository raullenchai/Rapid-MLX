"""Held-out, layer-major boundary drift and mixed-prefix logit evaluation.

Reference and candidate state use the streaming teacher's native attention/HC
arithmetic and typed cache serializer. Expert banks are read one expert at a
time. No held-out activation enters a quantization Hessian.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from scripts import deepseek_v41_gptq as b
from scripts.deepseek_v41_gptq_io import (
    PartialSource,
    atomic_json,
    digest,
    load_state,
    save_state,
    state_snapshot,
    verify_artifact,
)
from scripts.deepseek_v41_rd_common import file_hash, read_slices


class TensorFile:
    """Read a single native tensor or expert row without mapping a whole bank."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = 8 + n

    def array(self, name, expert=None):
        t = self.header[name]
        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "U32": "<u4", "I32": "<i4"}[
            t["dtype"]
        ]
        shape = tuple(t["shape"])
        offset = self.offset + t["data_offsets"][0]
        if expert is not None:
            if not 0 <= expert < shape[0]:
                raise ValueError("expert row outside native bank")
            offset += expert * int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
            shape = shape[1:]
        return np.memmap(self.path, mode="r", dtype=dtype, offset=offset, shape=shape)

    def tensor(self, name, expert=None):
        raw = self.array(name, expert)
        value = mx.array(np.array(raw))
        del raw
        return (
            value.view(mx.bfloat16) if self.header[name]["dtype"] == "BF16" else value
        )


def residual_mmap(path):
    path = Path(path)
    tree, payload = state_snapshot(path)
    key = tree["dict"]["h"]["array"]
    tensor_file = TensorFile(payload)
    if tensor_file.header[key]["dtype"] != "BF16":
        raise ValueError("quality boundaries must store BF16 residual streams")
    return tensor_file.array(key)


def windows_at(root, count):
    return [Path(root) / f"window-{i}" for i in range(count)]


def commit_boundary(directory, metadata):
    files = {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(Path(directory).iterdir())
        if p.is_file() and p.name != "complete.json"
    }
    atomic_json(Path(directory) / "complete.json", {**metadata, "files": files})


def verify_boundary(directory, identity):
    record = json.loads((Path(directory) / "complete.json").read_text())
    if record["identity"] != identity:
        raise ValueError("boundary identity mismatch")
    verify_artifact(directory, record)
    return record


def _float(raw):
    return (np.asarray(raw, np.uint32) << 16).view(np.float32)


def array_metrics(reference, candidate):
    """All HC streams form one vector per token; zero/zero has cosine one."""
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("boundary shapes differ")
    columns = (
        int(np.prod(reference.shape[-2:]))
        if reference.ndim == 4
        else reference.shape[-1]
    )
    r = np.asarray(reference, np.float64).reshape(-1, columns)
    q = np.asarray(candidate, np.float64).reshape(r.shape)
    if not np.isfinite(r).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite hidden state")
    rr, qq = (v * v for v in (r, q))
    energy = rr.sum(axis=1)
    norm = np.sqrt(energy * qq.sum(axis=1))
    squared = ((q - r) ** 2).sum(axis=1)
    cosine = np.divide(
        (r * q).sum(axis=1), norm, out=np.zeros_like(norm), where=norm > 0
    )
    cosine[squared == 0] = 1
    return dict(
        squared_error=float(squared.sum()),
        reference_energy=float(energy.sum()),
        cosine_sum=float(np.clip(cosine, -1, 1).sum()),
        tokens=len(r),
    )


def finish_metrics(totals):
    energy, error = totals["reference_energy"], totals["squared_error"]
    if not energy and error:
        raise ValueError("relative error undefined for a zero reference")
    return dict(
        relative_frobenius_error=float(np.sqrt(error / energy)) if energy else 0.0,
        mean_token_cosine_similarity=totals["cosine_sum"] / totals["tokens"],
        tokens=totals["tokens"],
    )


def boundary_metrics(reference, candidate, names, check=lambda: None):
    per_slice = {}
    aggregate = dict(squared_error=0.0, reference_energy=0.0, cosine_sum=0.0, tokens=0)
    for i, name in enumerate(names):
        r = residual_mmap(Path(reference) / f"window-{i}")
        q = residual_mmap(Path(candidate) / f"window-{i}")
        if r.shape != q.shape or r.ndim != 4 or r.shape[0] != 1:
            raise ValueError("expected matching [1,tokens,HC,dim] boundaries")
        totals = dict(squared_error=0.0, reference_energy=0.0, cosine_sum=0.0, tokens=0)
        for start in range(0, r.shape[1], 32):
            a = _float(r[:, start : start + 32]).reshape(-1, r.shape[2] * r.shape[3])
            c = _float(q[:, start : start + 32]).reshape(a.shape)
            values = array_metrics(a, c)
            for key, value in values.items():
                totals[key] += value
            check()
        per_slice[name] = finish_metrics(totals)
        for key, value in totals.items():
            aggregate[key] += value
        del r, q
    worst_error = max(per_slice, key=lambda n: per_slice[n]["relative_frobenius_error"])
    worst_cosine = min(
        per_slice, key=lambda n: per_slice[n]["mean_token_cosine_similarity"]
    )
    return {
        **finish_metrics(aggregate),
        "worst_slice_relative_frobenius_error": per_slice[worst_error][
            "relative_frobenius_error"
        ],
        "worst_slice_error_name": worst_error,
        "worst_slice_mean_token_cosine_similarity": per_slice[worst_cosine][
            "mean_token_cosine_similarity"
        ],
        "worst_slice_cosine_name": worst_cosine,
        "slices": per_slice,
    }


class NativeBuild:
    """Layer-local reader for both complete native builds and exported GPTQ parts."""

    def __init__(self, root, source, *, pending_layer=None):
        self.root, self.source = Path(root), source
        self.pending_layer = pending_layer
        self.files = {}
        self.mapping = None
        self.run = None
        if (self.root / "run.json").exists() and not (
            self.root / "model.safetensors.index.json"
        ).exists():
            self.run = json.loads((self.root / "run.json").read_text())
            if self.run["source_identity"] != digest(source.manifest):
                raise ValueError("native parts/source identity mismatch")
            from scripts.deepseek_v41_gptq_amend import epochs

            epochs(self.run)
            self.config = source.config
            self.identity = digest(dict(kind="gptq_parts", run=self.run["identity"]))
        else:
            self.config = json.loads((self.root / "config.json").read_text())
            self.mapping = json.loads(
                (self.root / "model.safetensors.index.json").read_text()
            )["weight_map"]
            self.identity = digest(
                dict(
                    config=self.config,
                    index=self.mapping,
                    path=str(self.root.resolve()),
                )
            )
        self.qconfig = self.config.get("quantization", {})

    def identity_for(self, layer):
        if self.run:
            from scripts.deepseek_v41_gptq_amend import layer_epoch

            return digest(dict(kind="gptq_parts", run=layer_epoch(self.run, layer)[1]))
        return self.identity

    @property
    def prefix_identity(self):
        # Keep the existing cache namespace so a retained pre-amendment prefix
        # remains usable after earlier reference boundaries have been released.
        return self.identity_for(0) if not self.run else self.original_identity()

    def original_identity(self):
        from scripts.deepseek_v41_gptq_amend import epochs

        return digest(dict(kind="gptq_parts", run=epochs(self.run)[0][1]))

    def file_for(self, name):
        filename = (
            self.mapping[name]
            if self.mapping is not None
            else f"model-layer-{name.split('.')[1]}.safetensors"
        )
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("unsafe indexed native shard")
        if filename not in self.files:
            self.files[filename] = TensorFile(self.root / filename)
        return self.files[filename]

    def has(self, name):
        try:
            return name in self.file_for(name).header
        except (KeyError, FileNotFoundError):
            return False

    def tensor(self, name, expert=None):
        return self.file_for(name).tensor(name, expert)

    def shape(self, name):
        return self.file_for(name).header[name]["shape"]

    def verify(self, layer):
        if self.run and layer != self.pending_layer:
            record = json.loads(
                (self.root / f"layer-{layer}-complete.json").read_text()
            )
            from scripts.deepseek_v41_gptq_amend import validate_record

            validate_record(self.run, layer, record)
            verify_artifact(self.root, record)
        prefix = f"layers.{layer}."
        names = [
            n
            for n in self.file_for(prefix + "ffn.experts.gate_proj.weight").header
            if n.startswith(prefix)
        ]
        return digest({n: file_hash(self.file_for(n).path) for n in names[:1]})

    def linear(self, name, columns, expert=None):
        weight = self.tensor(name + ".weight", expert)
        if not self.has(name + ".scales"):
            module = nn.Linear(columns, weight.shape[-2], bias=False)
            module.weight = weight
        else:
            triple = (
                weight,
                self.tensor(name + ".scales", expert),
                self.tensor(name + ".biases", expert),
            )
            bits = weight.shape[-1] * 32 // columns
            group = columns // triple[1].shape[-1]
            module = b.packed_module(
                SimpleNamespace(shape=(weight.shape[-2], columns)), triple, bits, group
            )
        b.install_cpu_affine(module)
        return module

    def block(self, layer, c):
        from mlx.utils import tree_flatten

        block = b.bounded_block(self.source, layer, c, original=True)
        prefix = f"layers.{layer}."

        def pred(name, module):
            base = prefix + name
            if not hasattr(module, "to_quantized") or not self.has(base + ".scales"):
                return False
            columns = module.weight.shape[-1]
            return dict(
                bits=self.shape(base + ".weight")[-1] * 32 // columns,
                group_size=columns // self.shape(base + ".scales")[-1],
            )

        nn.quantize(block, class_predicate=pred)
        values = []
        for name, expected in tree_flatten(block.parameters()):
            value = self.tensor(prefix + name)
            if name.startswith("attn.wo_a.") and value.shape != expected.shape:
                value = value.reshape(expected.shape)
            if name.startswith("ffn.gate.") and value.shape != expected.shape:
                setattr(block.ffn.gate, name.rsplit(".", 1)[-1], value)
            values.append((name, value))
        block.load_weights(values, strict=True)
        if block.engram is not None:
            if self.run:
                from scripts.deepseek_v41_gptq_amend import layer_epoch

                r = layer_epoch(self.run, layer)[2]
                block.engram.embed = b.AffineSourceEngram(
                    self.source.index, layer, r["engram_bits"], r["engram_group_size"]
                )
            else:
                from rapid_mlx.models.deepseek_v41_native.engram import (
                    DiskQuantizedEngramEmbedding,
                )

                base = prefix + "engram.embed"
                if self.has(base + ".scales"):
                    keys = {s: base + "." + s for s in b.SUFFIXES}
                    block.engram.embed = DiskQuantizedEngramEmbedding(
                        {key: self.file_for(key).path for key in keys.values()},
                        weight_key=keys["weight"],
                        scales_key=keys["scales"],
                        biases_key=keys["biases"],
                        num_embeddings=self.shape(base + ".weight")[0],
                        dim=self.source.index.tensors[base + ".weight"].shape[-1],
                        bits=self.qconfig["engram_bits"],
                        group_size=self.qconfig.get(
                            "engram_group_size", self.qconfig["group_size"]
                        ),
                        cache_rows=0,
                    )
                else:
                    raise ValueError("native Engram must use the SSD affine layout")
        b.install_cpu_affine(block)
        block.eval()
        mx.eval(block.parameters())
        return block

    def expert_ids(self, layer):
        allocation = self.config.get("rapid_quantization", {}).get("allocation", {})
        spec = allocation.get("layers", {}).get(str(layer), {})
        return spec.get(
            "kept_expert_ids",
            list(range(self.shape(f"layers.{layer}.ffn.experts.gate_proj.weight")[0])),
        )

    def expert(self, layer, slot, projection):
        names = dict(w1="gate_proj", w3="up_proj", w2="down_proj")
        columns = (
            self.source.args.moe_inter_dim
            if projection == "w2"
            else self.source.args.dim
        )
        return self.linear(
            f"layers.{layer}.ffn.experts.{names[projection]}", columns, slot
        )


class ErrorProbe(nn.Module):
    def __init__(self, module, weight, guard):
        super().__init__()
        self.module, self.guard = module, guard
        self.reference = weight.astype(mx.float32)
        q = mx.dequantize(
            *(getattr(module, s) for s in b.SUFFIXES),
            bits=module.bits,
            group_size=module.group_size,
            dtype=mx.float32,
        )
        rtn = mx.dequantize(
            *mx.quantize(weight, bits=module.bits, group_size=module.group_size),
            bits=module.bits,
            group_size=module.group_size,
            dtype=mx.float32,
        )
        self.deltas = [self.reference - rtn, self.reference - q]
        self.count, self.energy, self.errors = 0, 0.0, [0.0, 0.0]
        mx.eval(self.reference, self.deltas)

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        original = xf @ self.reference.T
        self.energy += float(mx.sum(original * original).item())
        for i, delta in enumerate(self.deltas):
            error = xf @ delta.T
            self.errors[i] += float(mx.sum(error * error).item())
        self.count += len(xf)
        self.guard.check()
        return self.module(x)


def attention_inputs(block, windows, work, args, c, guard, source=None, layer=None):
    work.mkdir(parents=True, exist_ok=True)
    probes = {}
    if source is not None:
        for name in b.DENSE[:4]:
            module = b.module_at(block, name)
            if hasattr(module, "scales"):
                probes[name] = ErrorProbe(
                    module, source.tensor(f"layers.{layer}.{name}.weight"), guard
                )
                b.set_module(block, name, probes[name])
    stage = []
    for i, path in enumerate(windows):
        state = load_state(path)
        out = b.attention_window(block, state, args, c, guard)
        out["h"] = out["h"].astype(mx.bfloat16)
        target = work / f"attention-{i}"
        save_state(target, out)
        stage.append(target)
        del state, out
        gc.collect()
        mx.clear_cache()
    reports = {}
    for name, probe in probes.items():
        module = probe.module
        reports[name] = dict(
            status="measured",
            tokens=probe.count,
            output_energy=probe.energy,
            error_scope="heldout_quantized_prefix_inputs",
            rtn_relative_output_error=probe.errors[0] / max(probe.energy, 1e-30),
            gptq_relative_output_error=probe.errors[1] / max(probe.energy, 1e-30),
        )
        b.set_module(block, name, module)
    paths = {key: work / f"{key}.bin" for key in ("x", "ids", "gates", "y")}
    total = sum(load_state(p)["h"].shape[1] for p in stage)
    counts = b.spill_ffn(block, stage, paths, total, args, c, guard)
    return dict(
        stage=stage, paths=paths, total=total, counts=counts, projections=reports
    )


def heldout_errors(weight, module, chunks):
    if not hasattr(module, "scales"):
        return dict(status="source_precision", tokens=None)
    report = b.projection_errors(
        weight,
        tuple(getattr(module, s) for s in b.SUFFIXES),
        module.bits,
        chunks,
        module.group_size,
    )
    energy = report.pop("calibration_output_energy")
    tokens = report.pop("error_token_count")
    return dict(
        status="measured" if energy else "unobserved",
        output_energy=energy,
        error_scope="heldout_quantized_prefix_inputs",
        tokens=tokens,
        **{key: value if energy else None for key, value in report.items()},
    )


def add_outputs(module, modules, inputs, args, c, guard, expert=None):
    for rows, x, gates in b.ffn_chunks(
        inputs["paths"], inputs["total"], args, c, expert
    ):
        y = module(
            b.intermediate(x, *modules, args.swiglu_limit, routed=expert is not None)
        ).astype(mx.float32)
        if gates is not None:
            y *= mx.array(gates)[:, None]
        mx.eval(y)
        output = np.memmap(
            inputs["paths"]["y"],
            mode="r+",
            dtype=np.float32,
            shape=(inputs["total"], args.dim),
        )
        output[rows] += np.asarray(y)
        output.flush()
        del output
        guard.check()


def apply_layer(
    source, layer, windows, target, work, c, guard, native=None, projection_errors=False
):
    """One layer; attention windows then an expert-major FFN pass over mmap rows."""
    started = time.monotonic()
    if target.exists():
        shutil.rmtree(target)
    if work.exists():
        shutil.rmtree(work)
    block = (
        native.block(layer, c)
        if native
        else b.bounded_block(source, layer, c, original=True)
    )
    inputs = attention_inputs(
        block,
        windows,
        work,
        source.args,
        c,
        guard,
        source if projection_errors else None,
        layer,
    )
    with inputs["paths"]["y"].open("wb") as f:
        f.truncate(inputs["total"] * source.args.dim * 4)
    ids = (
        native.expert_ids(layer)
        if native
        else list(range(source.args.n_routed_experts))
    )
    reports = inputs["projections"]
    for slot in range(len(ids) + 1):
        shared = slot == len(ids)
        eid = None if shared else slot
        source_id = None if shared else ids[slot]

        def chunks(inputs=inputs, eid=eid):
            for _, x, g in b.ffn_chunks(
                inputs["paths"], inputs["total"], source.args, c, eid
            ):
                yield x, g

        modules = []
        for projection in ("w1", "w3", "w2"):
            base = f"layers.{layer}.ffn." + (
                f"shared_experts.{projection}"
                if shared
                else f"experts.{source_id}.{projection}"
            )
            if shared:
                module = b.module_at(block, "ffn.shared_experts." + projection)
            elif native:
                module = native.expert(layer, slot, projection)
            else:
                w = source.tensor(base + ".weight")
                module = nn.Linear(w.shape[-1], w.shape[0], bias=False)
                module.weight = w
                b.install_cpu_affine(module)
            if projection_errors:

                def observed(
                    modules=modules, projection=projection, shared=shared, chunks=chunks
                ):
                    for x, g in chunks():
                        yield (
                            (
                                b.intermediate(
                                    x,
                                    *modules[:2],
                                    source.args.swiglu_limit,
                                    routed=not shared,
                                )
                                if projection == "w2"
                                else x
                            ),
                            g,
                        )

                reports[
                    "shared." + projection
                    if shared
                    else f"expert.{source_id}.{projection}"
                ] = heldout_errors(source.tensor(base + ".weight"), module, observed)
            modules.append(module)
        add_outputs(modules[2], modules[:2], inputs, source.args, c, guard, eid)
        del modules, module
        gc.collect()
        mx.clear_cache()
        if slot % 32 == 0 or shared:
            print(
                json.dumps(
                    dict(
                        quality_layer=layer,
                        expert="shared" if shared else slot,
                        seconds=time.monotonic() - started,
                        **guard.report(),
                    )
                ),
                flush=True,
            )
    b.finalize_windows(
        block,
        inputs["stage"],
        inputs["paths"],
        target,
        inputs["total"],
        source.args,
        {**c, "pilot": True},
        guard,
        run_identity=native.identity_for(layer) if native else digest(source.manifest),
        layer=layer,
    )
    # BF16 is both the native residual dtype and the portable boundary format.
    for path in windows_at(target, len(windows)):
        state = load_state(path)
        if state["h"].dtype != mx.bfloat16:
            state["h"] = state["h"].astype(mx.bfloat16)
            save_state(path, state)
        del state
    del block, inputs
    shutil.rmtree(work)
    gc.collect()
    mx.clear_cache()
    return reports


def quality_recipe(reference, c):
    return {**c, "prefill_chunk": reference["prefill_chunk"], "pilot": True}


def planning_with_quality(source, estimate, reference, baseline=False):
    result = dict(estimate)
    qc = {"prefill_chunk": reference["prefill_chunk"]}
    state = max(
        b.boundary_estimate(source.args, reference["lengths"], i, qc)
        for i in range(source.args.n_layers + 1)
    )
    spill = sum(reference["lengths"]) * (
        source.args.dim * 6 + source.args.n_activated_experts * 8
    )
    allowance = (8 if baseline else 6) * state + 2 * spill + 64 * 2**20
    projections = [
        (name, t)
        for name, t in source.index.tensors.items()
        if any(name.endswith("." + p + ".weight") for p in b.DENSE[:4])
    ]
    # Source attention weights are FP8/BF16, not packed FP4 expert matrices.
    ram = 12 * max(
        sum(
            int(np.prod(t.shape))
            for name, t in projections
            if name.startswith(f"layers.{i}.")
        )
        for i in range(source.args.n_layers)
    )
    result["heldout_local_disk_allowance_bytes"] = allowance
    result["heldout_ram_allowance_bytes"] = ram
    result["estimated_peak_bytes"] += ram
    result["local_disk_with_export_bytes"] += allowance
    result["local_disk_with_all_outputs_bytes"] += allowance
    return result


def reference_init(source, root, windows, names, provenance, c, guard, token_map=None):
    from rapid_mlx.models.deepseek_v41_native.engram import EngramHasher
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
        make_identity_pre_mix,
    )
    from rapid_mlx.models.deepseek_v41_native.load import load_token_map
    from rapid_mlx.models.deepseek_v41_native.model import SharedState

    root = Path(root)
    if len(set(names)) != len(names) or len(windows) != len(names) or not windows:
        raise ValueError("invalid reference windows")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("unsafe held-out slice name")
    lengths = [len(w) for w in windows]
    if any(
        np.asarray(w).ndim != 1
        or not 0 < len(w) <= source.args.max_seq_len
        or np.any(np.asarray(w) < 0)
        or np.any(np.asarray(w) >= source.args.vocab_size)
        for w in windows
    ):
        raise ValueError("invalid held-out token window")
    estimates = [
        b.boundary_estimate(source.args, lengths, i, c)
        for i in range(source.args.n_layers + 1)
    ]
    estimate = dict(
        residual_bytes_per_boundary=sum(lengths)
        * source.args.hc_mult
        * source.args.dim
        * 2,
        all_boundary_bytes=sum(estimates),
        boundary_bytes=estimates,
        logits_bytes=sum(lengths) * source.args.vocab_size * 4,
        working_ffn_bytes=sum(lengths)
        * (source.args.dim * 6 + source.args.n_activated_experts * 8),
    )
    print(json.dumps(dict(reference_storage_estimate=estimate)), flush=True)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    metadata = dict(
        schema_version=1,
        source_identity=digest(source.manifest),
        source_revision=source.manifest["source_revision"],
        names=names,
        lengths=lengths,
        token_hashes=[digest(np.asarray(w, np.int32).tolist()) for w in windows],
        prefill_chunk=c["prefill_chunk"],
        provenance=provenance,
        backend="cpu_float32_blas" if mx.default_device() == mx.cpu else "native_metal",
        mlx_version=mx.__version__,
        storage_estimate=estimate,
    )
    metadata["identity"] = digest(metadata)
    atomic_json(root / "reference.json", metadata)
    atomic_json(root / "source-manifest.json", source.manifest)
    hasher = None
    if source.args.engram_layer_ids:
        if token_map is None:
            token_map = load_token_map(
                str(source.source), source.args, cache_dir=str(root)
            )
        else:
            atomic_json(root / "engram_token_map.json", token_map)
        hasher = EngramHasher(source.args, token_map)
    boundary = root / "boundary-0"
    boundary.mkdir()
    embed = source.tensor("embed.weight").astype(mx.bfloat16)
    for i, ids in enumerate(windows):
        ids = np.asarray(ids, np.int32)
        h = embed[mx.array(ids)[None]]
        h = mx.broadcast_to(
            h[:, :, None], (1, len(ids), source.args.hc_mult, source.args.dim)
        )
        state = dict(
            ids=mx.array(ids),
            h=h,
            pre=make_identity_pre_mix(1, len(ids), source.args.hc_mult),
            hashes=None
            if hasher is None
            else mx.array(hasher(ids[None], 0, np.zeros((1, len(ids)), np.int64))),
            shared=SharedState(),
            selections=[],
        )
        save_state(boundary / f"window-{i}", state)
        guard.check()
    commit_boundary(boundary, dict(identity=metadata["identity"], boundary=0))
    atomic_json(
        root / "checkpoint.json", dict(identity=metadata["identity"], next_layer=0)
    )
    return metadata


def reference_run(source, root, through, c, guard):
    root = Path(root)
    reference = json.loads((root / "reference.json").read_text())
    if reference["source_identity"] != digest(source.manifest):
        raise ValueError("reference/source mismatch")
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    if checkpoint["identity"] != reference["identity"]:
        raise ValueError("reference checkpoint identity mismatch")
    if not -1 <= through < source.args.n_layers:
        raise ValueError("invalid reference layer")
    c = quality_recipe(reference, c)
    for layer in range(checkpoint["next_layer"], through + 1):
        source.verify(layer)
        previous = root / f"boundary-{layer}"
        verify_boundary(previous, reference["identity"])
        target = root / f"boundary-{layer + 1}"
        started = time.monotonic()
        apply_layer(
            source,
            layer,
            windows_at(previous, len(reference["names"])),
            target,
            root / "working",
            c,
            guard,
        )
        commit_boundary(
            target, dict(identity=reference["identity"], boundary=layer + 1)
        )
        atomic_json(
            root / "checkpoint.json",
            dict(identity=reference["identity"], next_layer=layer + 1),
        )
        print(
            json.dumps(
                dict(
                    reference_completed_layer=layer,
                    seconds=time.monotonic() - started,
                    **guard.report(),
                )
            ),
            flush=True,
        )
    return reference


def append_row(path, row):
    """Append once; interrupted terminal writes are safely discarded on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = []
    if path.exists():
        data = path.read_bytes()
        end = data.rfind(b"\n") + 1
        if end != len(data):
            with path.open("r+b") as f:
                f.truncate(end)
        prior = [json.loads(line) for line in data[:end].splitlines()]
    key = (
        row["reference_identity"],
        row["candidate_identity"],
        row.get("baseline_identity"),
        row["prefix_start"],
        row["layer"],
    )
    for old in prior:
        old_key = (
            old["reference_identity"],
            old["candidate_identity"],
            old.get("baseline_identity"),
            old["prefix_start"],
            old["layer"],
        )
        if old_key == key:
            if old != row:
                raise ValueError("metrics row changed for a committed layer")
            print(json.dumps(old), flush=True)
            return
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)


def prefix_boundary(
    source,
    reference_root,
    reference,
    native,
    layer,
    scratch,
    c,
    guard,
    start=0,
    projection_errors=False,
):
    root = Path(scratch) / (
        "prefix-"
        + digest(
            dict(
                build=native.prefix_identity,
                start=start,
                reference=reference["identity"],
            )
        )[:20]
    )
    root.mkdir(parents=True, exist_ok=True)
    previous = Path(reference_root) / f"boundary-{start}"
    previous_identity = reference["identity"]
    reports = None
    first = start
    retained = sorted(
        (
            p
            for p in root.glob("boundary-*")
            if (p / "complete.json").exists()
            and int(p.name.split("-")[-1]) <= layer + 1
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if retained:
        previous = retained[-1]
        first = int(previous.name.split("-")[-1])
        previous_identity = digest(
            dict(
                reference=reference["identity"],
                build=native.identity_for(first - 1),
                start=start,
                boundary=first,
            )
        )
        record = verify_boundary(previous, previous_identity)
        if native.has(f"layers.{first - 1}.ffn.experts.gate_proj.weight") and record[
            "layer_sha256"
        ] != native.verify(first - 1):
            raise ValueError("native layer changed after prefix checkpoint")
        report_path = previous / "projection-errors.json"
        reports = json.loads(report_path.read_text()) if report_path.exists() else None
    for i in range(first, layer + 1):
        target = root / f"boundary-{i + 1}"
        identity = digest(
            dict(
                reference=reference["identity"],
                build=native.identity_for(i),
                start=start,
                boundary=i + 1,
            )
        )
        if not previous.exists():
            raise ValueError(
                "prefix checkpoint missing; provide earlier layer tensors to replay"
            )
        verify_boundary(previous, previous_identity)
        layer_hash = native.verify(i)
        reports = apply_layer(
            source,
            i,
            windows_at(previous, len(reference["names"])),
            target,
            root / "working",
            c,
            guard,
            native,
            projection_errors,
        )
        atomic_json(target / "projection-errors.json", reports)
        commit_boundary(
            target, dict(identity=identity, boundary=i + 1, layer_sha256=layer_hash)
        )
        if previous.parent == root and previous != target:
            shutil.rmtree(previous)
        previous, previous_identity = target, identity
    return previous, reports


def drift(
    source,
    reference_root,
    native,
    layers,
    scratch,
    metrics,
    c,
    guard,
    baseline=None,
    start=0,
    projection_errors=True,
    update_parts_stats=False,
):
    reference_root, scratch = Path(reference_root), Path(scratch)
    reference = json.loads((reference_root / "reference.json").read_text())
    if reference["source_identity"] != digest(source.manifest):
        raise ValueError("reference/source mismatch")
    c = quality_recipe(reference, c)
    rows = []
    for layer in layers:
        if layer < start or layer >= source.args.n_layers:
            raise ValueError("invalid drift layer or prefix start")
        for boundary in (layer, layer + 1):
            verify_boundary(
                reference_root / f"boundary-{boundary}", reference["identity"]
            )
        row = dict(
            schema_version=1,
            layer=layer,
            prefix_start=start,
            cumulative_scope="full_prefix" if start == 0 else "block",
            reference_identity=reference["identity"],
            candidate_identity=native.identity_for(layer),
            baseline_identity=baseline.identity if baseline else None,
            reference_backend=reference["backend"],
            execution_backend="cpu_float32_blas"
            if mx.default_device() == mx.cpu
            else "native_metal",
            metrics_definition="relative Frobenius norm; cosine over flattened HC streams per token",
        )
        for label, build in (("candidate", native), ("baseline", baseline)):
            if build is None:
                continue
            cumulative, reports = prefix_boundary(
                source,
                reference_root,
                reference,
                build,
                layer,
                scratch,
                c,
                guard,
                start,
                projection_errors,
            )
            local = cumulative
            if layer != start:
                local = scratch / f"local-{label}-{layer}"
                build.verify(layer)
                apply_layer(
                    source,
                    layer,
                    windows_at(
                        reference_root / f"boundary-{layer}", len(reference["names"])
                    ),
                    local,
                    scratch / "local-work",
                    c,
                    guard,
                    build,
                )
            row[label] = dict(
                cumulative=boundary_metrics(
                    reference_root / f"boundary-{layer + 1}",
                    cumulative,
                    reference["names"],
                    guard.check,
                ),
                local=boundary_metrics(
                    reference_root / f"boundary-{layer + 1}",
                    local,
                    reference["names"],
                    guard.check,
                ),
            )
            if reports:
                atomic_json(
                    scratch / f"{label}-layer-{layer}-heldout-errors.json",
                    dict(
                        reference_identity=reference["identity"],
                        build_identity=build.identity_for(layer),
                        layer=layer,
                        projections=reports,
                    ),
                )
                if label == "candidate" and update_parts_stats:
                    backfill_stats(build, layer, reports, reference)
            if local != cumulative:
                shutil.rmtree(local)
        append_row(metrics, row)
        rows.append(row)
    return rows


def validate_reference(source, reference):
    if reference["source_identity"] != digest(source.manifest):
        raise ValueError("held-out reference/source identity mismatch")
    tokenizer = source.source / "tokenizer.json"
    if (
        tokenizer.exists()
        and file_hash(tokenizer) != reference["provenance"]["tokenizer_sha256"]
    ):
        raise ValueError("held-out reference/calibration tokenizer mismatch")


def validate_separation(reference_root, reference, windows, corpus_hashes):
    """Check calibration evidence before committing or resuming a build."""
    heldout_hashes = {
        sha
        for s in reference["provenance"]["slices"].values()
        for sha in s["corpus_sha256"]
    }
    if set(corpus_hashes) & heldout_hashes:
        raise ValueError("calibration and held-out corpus files overlap")
    if windows is None:
        return
    calibration = [np.asarray(w, np.int32).tobytes() for w in windows]
    for path in windows_at(
        Path(reference_root) / "boundary-0", len(reference["names"])
    ):
        ids = np.asarray(load_state(path)["ids"], np.int32)
        if any(ids.tobytes() in raw for raw in calibration):
            raise ValueError("held-out token window appears in calibration data")


def bind_quality(
    source, parts, reference_root, scratch, baseline=None, *, windows=None, slices=None
):
    """Bind held-out content separately from machine-local staging paths."""
    parts, reference_root, scratch = Path(parts), Path(reference_root), Path(scratch)
    reference = json.loads((reference_root / "reference.json").read_text())
    validate_reference(source, reference)
    run = json.loads((parts / "run.json").read_text())
    if (
        run.get("heldout_reference_identity", reference["identity"])
        != reference["identity"]
    ):
        raise ValueError("held-out reference changed; start a new run")
    calibration_hashes = (
        {file_hash(p) for files in read_slices(slices, []).values() for p in files}
        if slices is not None
        else set()
    )
    validate_separation(reference_root, reference, windows, calibration_hashes)
    config = dict(
        reference=str(reference_root.resolve()),
        scratch=str(scratch.resolve()),
        baseline=None if baseline is None else str(Path(baseline).resolve()),
        reference_identity=reference["identity"],
    )
    old = parts / "quality-config.json"
    if (
        old.exists()
        and json.loads(old.read_text())["reference_identity"] != reference["identity"]
    ):
        raise ValueError("quality cache identity changed")
    atomic_json(old, config)
    atomic_json(parts / "heldout-reference.json", reference)
    return reference


def prepare_for_build(source, parts, layer, block, c, guard):
    config = json.loads((parts / "quality-config.json").read_text())
    reference_root, scratch = Path(config["reference"]), Path(config["scratch"])
    reference = json.loads((reference_root / "reference.json").read_text())
    if config["reference_identity"] != reference["identity"]:
        raise ValueError("held-out reference identity changed")
    for boundary in (layer, layer + 1):
        verify_boundary(reference_root / f"boundary-{boundary}", reference["identity"])
    native = NativeBuild(parts, source, pending_layer=layer)
    qc = quality_recipe(reference, c)
    scratch.mkdir(parents=True, exist_ok=True)
    previous = reference_root / "boundary-0"
    if layer:
        previous, _ = prefix_boundary(
            source, reference_root, reference, native, layer - 1, scratch, qc, guard
        )
    root = scratch / (
        "prefix-"
        + digest(
            dict(build=native.prefix_identity, start=0, reference=reference["identity"])
        )[:20]
    )
    root.mkdir(parents=True, exist_ok=True)
    work = scratch / f"fitting-layer-{layer}"
    if work.exists():
        shutil.rmtree(work)
    inputs = attention_inputs(
        block,
        windows_at(previous, len(reference["names"])),
        work,
        source.args,
        qc,
        guard,
        source,
        layer,
    )
    return dict(
        config=config,
        reference=reference,
        reference_root=reference_root,
        scratch=scratch,
        native=native,
        c=qc,
        root=root,
        work=work,
        inputs=inputs,
    )


def finish_for_build(source, parts, layer, block, context, ffn_stats, guard):
    reference, inputs, qc = context["reference"], context["inputs"], context["c"]
    target = context["root"] / f"boundary-{layer + 1}"
    if target.exists():
        shutil.rmtree(target)
    b.finalize_windows(
        block,
        inputs["stage"],
        inputs["paths"],
        target,
        inputs["total"],
        source.args,
        qc,
        guard,
        run_identity=digest(
            dict(
                reference=reference["identity"],
                build=context["native"].identity_for(layer),
            )
        ),
        layer=layer,
    )
    reports = {
        **inputs["projections"],
        **{name: report["heldout"] for name, report in ffn_stats.items()},
    }
    atomic_json(target / "projection-errors.json", reports)
    identity = digest(
        dict(
            reference=reference["identity"],
            build=context["native"].identity_for(layer),
            start=0,
            boundary=layer + 1,
        )
    )
    commit_boundary(
        target,
        dict(
            identity=identity,
            boundary=layer + 1,
            layer_sha256=context["native"].verify(layer),
        ),
    )
    baseline = (
        NativeBuild(context["config"]["baseline"], source)
        if context["config"]["baseline"]
        else None
    )
    drift(
        source,
        context["reference_root"],
        context["native"],
        [layer],
        context["scratch"],
        parts / "heldout-drift.jsonl",
        qc,
        guard,
        baseline,
    )


def release_previous_quality(context, layer):
    previous = context["root"] / f"boundary-{layer}"
    if previous.exists():
        shutil.rmtree(previous)
    shutil.rmtree(context["work"])


def backfill_stats(native, layer, reports, reference):
    if native.run is None:
        raise ValueError(
            "held-out backfill writes only GPTQ parts, never complete builds"
        )
    path = native.root / f"layer-{layer}-stats.json"
    stats = json.loads(path.read_text())
    if set(stats["projections"]) != set(reports):
        raise ValueError("held-out projection report is incomplete")
    for name, report in stats["projections"].items():
        report["calibration"] = dict(
            error_scope="calibration",
            **{
                k: v
                for k, v in report.items()
                if k
                in (
                    "rtn_relative_output_error",
                    "gptq_relative_output_error",
                    "calibration_output_energy",
                )
            },
        )
        report["heldout"] = reports[name]
    stats["heldout_reference_identity"] = reference["identity"]
    atomic_json(path, stats)
    complete_path = native.root / f"layer-{layer}-complete.json"
    record = json.loads(complete_path.read_text())
    record["files"][path.name] = file_hash(path)
    atomic_json(complete_path, record)


def logit_cache(
    source, reference_root, boundary, destination, reference, guard, *, mode
):
    """Write a partial-prefix milestone cache, not a checkpoint qualification."""
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    provenance = reference["provenance"]
    metadata = dict(
        schema_version=1,
        cache_kind="gptq_partial_prefix",
        source_identity=digest(source.manifest),
        mode=mode,
        model_revision=source.manifest["source_revision"],
        tokenizer_sha256=provenance["tokenizer_sha256"],
        slices={},
        quality_reference_identity=reference["identity"],
        top_policy="original_unchanged",
        execution_backend=reference["backend"],
    )
    norm, head = source.tensor("norm.weight"), source.tensor("head.weight")
    mx.eval(norm, head)
    for i, name in enumerate(reference["names"]):
        state = load_state(Path(boundary) / f"window-{i}")
        ids = np.asarray(state["ids"])
        np.save(destination / f"{name}-tokens.npy", ids)
        logits = np.lib.format.open_memmap(
            destination / f"{name}-logits.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), source.args.vocab_size),
        )
        for start in range(0, len(ids), 8):
            end = min(len(ids), start + 8)
            h = hc_pre_norm(
                state["h"][:, start:end],
                state["pre"][:, start:end],
                norm,
                source.args.norm_eps,
            )
            value = h.astype(mx.float32) @ head.T
            mx.eval(value)
            logits[start:end] = np.asarray(value).reshape(-1, source.args.vocab_size)
            guard.check()
        logits.flush()
        del logits, state
        np.save(
            destination / f"{name}-mass.npy",
            np.ones((len(ids), source.args.n_layers), np.float32),
        )
        metadata["slices"][name] = dict(
            tokens=len(ids),
            corpus_sha256=provenance["slices"][name]["corpus_sha256"],
            logits_sha256=file_hash(destination / f"{name}-logits.npy"),
            tokens_sha256=file_hash(destination / f"{name}-tokens.npy"),
        )
    atomic_json(destination / "metadata.json", metadata)
    return destination


def compare_milestone(reference, candidate):
    """Compare authenticated partial-prefix caches without claiming a built checkpoint."""
    from scripts.evaluate_deepseek_v41_rd import summarize_metrics, token_metrics

    roots = [Path(reference), Path(candidate)]
    metas = [json.loads((root / "metadata.json").read_text()) for root in roots]
    rm, cm = metas
    keys = ("source_identity", "quality_reference_identity", "tokenizer_sha256")
    if (
        any(m.get("cache_kind") != "gptq_partial_prefix" for m in metas)
        or rm.get("mode") != "unpruned_reference"
        or any(not rm.get(key) or rm[key] != cm.get(key) for key in keys)
        or not rm["slices"]
        or set(rm["slices"]) != set(cm["slices"])
    ):
        raise ValueError("milestone cache provenance mismatch")
    results, all_kl, all_flips = {}, [], []
    for name in rm["slices"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe milestone slice name")
        if rm["slices"][name]["corpus_sha256"] != cm["slices"][name]["corpus_sha256"]:
            raise ValueError("milestone corpus mismatch")
        arrays = []
        for root, meta in zip(roots, metas):
            entry = meta["slices"][name]
            loaded = {}
            for kind in ("tokens", "logits"):
                path = root / f"{name}-{kind}.npy"
                if file_hash(path) != entry.get(f"{kind}_sha256"):
                    raise ValueError("milestone cache payload changed")
                loaded[kind] = np.load(path, mmap_mode="r", allow_pickle=False)
            ids, logits = loaded["tokens"], loaded["logits"]
            if (
                ids.ndim != 1
                or not len(ids)
                or ids.dtype.kind not in "iu"
                or entry["tokens"] != len(ids)
                or logits.ndim != 2
                or logits.shape[0] != len(ids)
            ):
                raise ValueError("invalid milestone cache shape")
            arrays.append(loaded)
        if not np.array_equal(arrays[0]["tokens"], arrays[1]["tokens"]):
            raise ValueError("milestone token IDs differ")
        n = len(arrays[0]["tokens"])
        kl, flips = np.empty(n), np.empty(n)
        for start in range(0, n, 8):
            kl[start : start + 8], flips[start : start + 8] = token_metrics(
                arrays[0]["logits"][start : start + 8],
                arrays[1]["logits"][start : start + 8],
            )
        results[name] = dict(tokens=n, **summarize_metrics(kl, flips))
        all_kl.append(kl)
        all_flips.append(flips)
    return dict(
        mode="gptq_partial_prefix",
        kl_estimator="full_vocabulary",
        **summarize_metrics(np.concatenate(all_kl), np.concatenate(all_flips)),
        slices=results,
        mean_domain_kl_nats=float(np.mean([v["kl_nats"] for v in results.values()])),
        worst_domain_kl_nats=max(v["kl_nats"] for v in results.values()),
    )


def milestone(
    source,
    reference_root,
    native,
    last,
    scratch,
    output,
    c,
    guard,
    *,
    logits_reference=None,
):

    reference_root, scratch = Path(reference_root), Path(scratch)
    if Path(output).exists():
        raise FileExistsError(output)
    reference = json.loads((reference_root / "reference.json").read_text())
    if reference["source_identity"] != digest(source.manifest):
        raise ValueError("reference/source mismatch")
    if not -1 <= last < source.args.n_layers:
        raise ValueError("invalid last quantized layer")
    c = quality_recipe(reference, c)
    scratch.mkdir(parents=True, exist_ok=True)
    if last < 0:
        boundary = reference_root / f"boundary-{source.args.n_layers}"
        verify_boundary(boundary, reference["identity"])
    else:
        boundary, _ = prefix_boundary(
            source, reference_root, reference, native, last, scratch, c, guard
        )
        for layer in range(last + 1, source.args.n_layers):
            source.verify(layer)
            target = scratch / f"original-tail-boundary-{layer + 1}"
            apply_layer(
                source,
                layer,
                windows_at(boundary, len(reference["names"])),
                target,
                scratch / "tail-working",
                c,
                guard,
            )
            if boundary.parent == scratch and boundary.name.startswith(
                "original-tail-"
            ):
                shutil.rmtree(boundary)
            boundary = target
    if logits_reference is None:
        logits_reference = reference_root / "logits"
        if not (logits_reference / "metadata.json").exists():
            original = reference_root / f"boundary-{source.args.n_layers}"
            verify_boundary(original, reference["identity"])
            logit_cache(
                source,
                reference_root,
                original,
                logits_reference,
                reference,
                guard,
                mode="unpruned_reference",
            )
    candidate = logit_cache(
        source,
        reference_root,
        boundary,
        output,
        reference,
        guard,
        mode="packed_candidate",
    )
    result = compare_milestone(Path(logits_reference), candidate)
    result.update(
        last_quantized_layer=last,
        quality_reference_identity=reference["identity"],
        candidate_identity=native.identity_for(last) if last >= 0 else None,
    )
    atomic_json(Path(output) / "quality.json", result)
    print(json.dumps(result), flush=True)
    return result


def parse_layers(value):
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
    if not match or int(match[2] or match[1]) < int(match[1]):
        raise ValueError("layers must be an inclusive range, e.g. 0-3")
    return list(range(int(match[1]), int(match[2] or match[1]) + 1))


def add_parsers(sub):
    for command in ("reference", "drift", "milestone"):
        p = sub.add_parser(command, help=__doc__.splitlines()[0])
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--reference", type=Path, required=True)
        p.add_argument("--recipe", type=Path)
        p.add_argument("--cpu", action="store_true")
        if command == "reference":
            choice = p.add_mutually_exclusive_group()
            choice.add_argument("--slices", type=Path)
            choice.add_argument("--tokens-from-cache", type=Path)
            p.add_argument("--tokens", type=int, default=512)
            p.add_argument("--prefill-chunk", type=int, default=512)
            p.add_argument("--through-layer", type=int)
            p.add_argument("--write-logits", action="store_true")
        else:
            p.add_argument("--build", type=Path, required=True)
            p.add_argument("--scratch", type=Path, required=True)
            if command == "drift":
                p.add_argument("--layers", required=True)
                p.add_argument("--block-start", type=int, default=0)
                p.add_argument("--baseline", "--rtn-build", type=Path)
                p.add_argument("--metrics", type=Path, required=True)
                p.add_argument("--skip-projection-errors", action="store_true")
                p.add_argument(
                    "--update-parts-stats",
                    action="store_true",
                    help="add held-out errors to existing GPTQ parts and rehash completion records",
                )
            else:
                p.add_argument("--last-quantized-layer", type=int, required=True)
                p.add_argument("--output", type=Path, required=True)
                p.add_argument("--logits-reference", type=Path)


def execute(a):
    source = PartialSource(a.source, json.loads(a.manifest.read_text()))
    c = b.recipe(json.loads(a.recipe.read_text()) if a.recipe else None)
    if a.cpu:
        mx.set_default_device(mx.cpu)
    with b.MemoryGuard(c) as guard:
        if a.command == "reference":
            c = {**c, "prefill_chunk": a.prefill_chunk}
            if not (a.reference / "reference.json").exists():
                if a.tokens_from_cache:
                    provenance = json.loads(
                        (a.tokens_from_cache / "metadata.json").read_text()
                    )
                    if (
                        provenance["mode"] != "unpruned_reference"
                        or provenance["model_revision"]
                        != source.manifest["source_revision"]
                    ):
                        raise ValueError("original logit cache provenance mismatch")
                    names = list(provenance["slices"])
                    windows = [
                        np.load(
                            a.tokens_from_cache / f"{n}-tokens.npy", allow_pickle=False
                        )
                        for n in names
                    ]
                elif a.slices:
                    slices = read_slices(a.slices, [])
                    names = list(slices)
                    windows, _ = b.calibration_windows(source, a.slices, a.tokens)
                    provenance = dict(
                        tokenizer_sha256=file_hash(source.source / "tokenizer.json"),
                        slices={
                            n: dict(corpus_sha256=[file_hash(p) for p in paths])
                            for n, paths in slices.items()
                        },
                    )
                else:
                    raise ValueError(
                        "new reference needs --slices or --tokens-from-cache"
                    )
                reference_init(
                    source, a.reference, windows, names, provenance, c, guard
                )
            reference = reference_run(
                source,
                a.reference,
                source.args.n_layers - 1
                if a.through_layer is None
                else a.through_layer,
                c,
                guard,
            )
            if a.write_logits:
                logit_cache(
                    source,
                    a.reference,
                    a.reference / f"boundary-{source.args.n_layers}",
                    a.reference / "logits",
                    reference,
                    guard,
                    mode="unpruned_reference",
                )
        else:
            native = NativeBuild(a.build, source)
            if a.command == "drift":
                baseline = NativeBuild(a.baseline, source) if a.baseline else None
                drift(
                    source,
                    a.reference,
                    native,
                    parse_layers(a.layers),
                    a.scratch,
                    a.metrics,
                    c,
                    guard,
                    baseline,
                    a.block_start,
                    not a.skip_projection_errors,
                    a.update_parts_stats,
                )
            else:
                milestone(
                    source,
                    a.reference,
                    native,
                    a.last_quantized_layer,
                    a.scratch,
                    a.output,
                    c,
                    guard,
                    logits_reference=a.logits_reference,
                )
