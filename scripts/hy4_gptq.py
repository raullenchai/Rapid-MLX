#!/usr/bin/env python3
"""Resumable Hy4 native-affine build and layer-streamed held-out ruler."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v41_rd_common import file_hash
from scripts.hy4_gptq_adapter import (
    Hy4Adapter,
    Recipe,
    disk_switch,
    initial_hidden,
    load_layer,
    logits,
)
from scripts.streamed_gptq_io import (
    RangeSink,
    TensorReader,
    artifact_lock,
    atomic_json,
    digest,
    load_boundary,
    mx_module,
    save_boundary,
    slice_evidence,
    verify_receipt,
)


def layers_prefix(value, count):
    layers = list(range(count)) if value is None else [int(i) for i in value.split(",")]
    if not layers or layers != list(range(len(layers))) or len(layers) > count:
        raise ValueError("only-layers must be a contiguous prefix starting at zero")
    return layers


def calibration(adapter, manifest, tokens):
    """Render local conversation records with the frozen chat template, offline."""
    from transformers import AutoTokenizer

    if type(tokens) is not int or not 1 <= tokens <= 8192:
        raise ValueError("invalid calibration window size")
    paths, evidence = slice_evidence(manifest)
    required = {"english", "chinese", "code", "tools"}
    if not required.issubset(paths):
        raise ValueError("calibration needs english, chinese, code and tools slices")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter.source, local_files_only=True, trust_remote_code=False
    )
    windows = []
    for files in paths.values():
        for path in files:
            raw = path.read_text()
            # Tool slices may supply complete conversations + tool schemas.
            record = (
                json.loads(raw)
                if path.suffix == ".json"
                else {"messages": [{"role": "user", "content": raw}]}
            )
            if "messages" not in record or not record["messages"]:
                raise ValueError("calibration conversation is empty")
            for effort in ("high", "no_think"):
                # Render text first: tokenize=True returns an Encoding
                # object rather than a list of ids on newer tokenizers.
                text = tokenizer.apply_chat_template(
                    record["messages"],
                    tools=record.get("tools"),
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=effort,
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    raise ValueError("chat template produced no tokens")
                windows.append(np.asarray(ids[:tokens], np.int32))
    return windows, evidence


class NativeStore:
    def __init__(self, root, adapter, layers):
        self.root, self.adapter, self.layers = Path(root), adapter, layers
        self.root.mkdir(parents=True, exist_ok=True)
        self.weight_map = {}
        self.specs = {}

    def register(self, names, filename):
        for name in names:
            if name in self.weight_map and self.weight_map[name] != filename:
                raise ValueError("duplicate native output name")
            self.weight_map[name] = filename

    def publish_index(self):
        atomic_json(
            self.root / "model.safetensors.index.json", {"weight_map": self.weight_map}
        )

    def reader(self):
        self.publish_index()
        return TensorReader(self.root)

    def entries(self, name):
        shape = self.adapter.shape(name)
        spec = self.adapter.recipe.spec(name)
        if spec:
            bits, group, _ = spec
            base = name.removesuffix(".weight")
            self.specs[base] = dict(bits=bits, group_size=group, mode="affine")
            return {
                base + "." + suffix: (dtype, (*shape[:-1], cols))
                for suffix, dtype, cols in [
                    ("weight", "U32", shape[-1] * bits // 32),
                    ("scales", "BF16", shape[-1] // group),
                    ("biases", "BF16", shape[-1] // group),
                ]
            }
        original, _ = self.adapter.names[name]
        dtype = self.adapter.reader.index.tensors[original].dtype
        if dtype.startswith("F8_"):
            dtype = "BF16"
        return {name: (dtype, shape)}

    def sink(self, name):
        entries = self.entries(name)
        filename = "model-" + digest(name)[:24] + ".safetensors"
        self.register(entries, filename)
        return RangeSink(self.root / filename, entries)

    def put(self, sink, name, start, values, *, bank=False):
        mx = mx_module()
        spec = self.adapter.recipe.spec(name)
        names = (
            [name]
            if spec is None
            else [
                name.removesuffix(".weight") + "." + s
                for s in ("weight", "scales", "biases")
            ]
        )
        for key, value in zip(names, values, strict=True):
            mx.eval(value)
            if bank:
                value = value[None]
            raw = np.array(
                value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
            )
            sink.put(key, start, raw)

    def rtn(self, name, check):
        mx = mx_module()
        spec = self.adapter.recipe.spec(name)
        shape = self.adapter.shape(name)
        # Only matrix row chunks are materialized; expert banks use GPTQ below.
        chunk = 4096 if len(shape) == 2 else shape[0]
        sink = self.sink(name)
        try:
            for start in range(0, shape[0], chunk):
                stop = min(shape[0], start + chunk)
                value = self.adapter.tensor(
                    name, rows=(start, stop) if len(shape) == 2 else None
                )
                values = (
                    mx.quantize(
                        value.astype(mx.bfloat16), bits=spec[0], group_size=spec[1]
                    )
                    if spec
                    else (value,)
                )
                self.put(sink, name, start, values)
                del value, values
                check()
            sink.finish()
        finally:
            sink.close()

    def config(self):
        config = json.loads(json.dumps(self.adapter.config))
        text = config.get("text_config", config)
        count = len(self.layers)
        for key in ("quantization_config", "quantization"):
            config.pop(key, None)
            text.pop(key, None)
        text["num_hidden_layers"] = count
        for key in ("layer_types", "mlp_layer_types"):
            text[key] = getattr(self.adapter.args, key)[:count]
        text["engram_layer_ids"] = [
            i for i in self.adapter.args.engram_layer_ids if i <= count
        ]
        config["quantization"] = dict(
            bits=8, group_size=64, mode="affine", **self.specs
        )
        return config


def routed_calibration(adapter, block, layer, windows, previous, work, check):
    import mlx.nn as nn

    mx = mx_module()
    original = block.mlp
    captured = []

    class Capture(nn.Module):
        def __call__(self, x):
            captured.append(adapter.routed_inputs(block_holder, x))
            return mx.zeros_like(x)

    # The adapter routes through the original MoE, never through this probe.
    from types import SimpleNamespace

    block_holder = SimpleNamespace(mlp=original)
    block.mlp = Capture()
    paths = []
    try:
        for i, tokens in enumerate(windows):
            hidden = load_boundary(previous[i])
            result = adapter.forward(block, hidden, tokens)
            mx.eval(result)
            x, ids, weights = captured.pop()
            path = work / f"routed-{i}.npz"
            np.savez(
                path,
                x=np.array(x.astype(mx.float32)),
                indices=np.array(ids),
                weights=np.array(weights),
            )
            paths.append(path)
            del hidden, result, x, ids, weights
            check()
    finally:
        block.mlp = original
    return paths


def expert_chunks(paths, expert, adapter=None, fitted=None):
    mx = mx_module()
    for path in paths:
        with np.load(path) as data:
            positions, slots = np.where(data["indices"] == expert)
            x = mx.array(data["x"][positions]).astype(mx.bfloat16)
            weights = data["weights"][positions, slots]
        if len(positions):
            if fitted:
                x = adapter.intermediate(*fitted, x)
            yield x, weights


def quantize_experts(store, layer, paths, check):
    from scripts.streamed_gptq_quant import QuantConfig, gptq, gram

    mx = mx_module()
    adapter = store.adapter
    base = f"model.layers.{layer}.mlp.switch_mlp."
    names = [base + p + ".weight" for p in ("gate_proj", "up_proj", "down_proj")]
    sinks = [store.sink(n) for n in names]
    stats = []
    config = QuantConfig()
    try:
        for expert in range(adapter.args.n_routed_experts):
            fitted = []
            expert_stats = {}
            for name, sink in zip(names, sinks, strict=True):
                spec = adapter.recipe.spec(name)
                bits, group, _ = spec
                weight = adapter.tensor(name, expert=expert)
                down = name.endswith(".down_proj.weight")
                chunks = expert_chunks(paths, expert, adapter, fitted if down else None)
                h, count = gram(chunks, weight.shape[-1], check=check)
                if count:
                    triple, metrics = gptq(
                        weight, h, bits, config, count=count, check=check
                    )
                else:
                    triple = mx.quantize(weight, bits=bits, group_size=group)
                    metrics = dict(tokens=0, fallback="unobserved_rtn")
                store.put(sink, name, expert, triple, bank=True)
                expert_stats[name.rsplit(".", 2)[1]] = metrics
                if not down:
                    fitted.append(mx.dequantize(*triple, bits=bits, group_size=group))
                del weight, h, triple
                check()
            stats.append(expert_stats)
        for sink in sinks:
            sink.finish()
    finally:
        for sink in sinks:
            sink.close()
    return stats


def source_identity(adapter):
    paths = {t.path for t in adapter.reader.index.tensors.values()}
    paths.update(
        adapter.source / n
        for n in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if (adapter.source / n).exists()
    )
    return {p.name: file_hash(p) for p in sorted(paths)}


def tool_identity():
    root = Path(__file__).resolve().parents[1]
    names = [
        f"scripts/{n}.py"
        for n in (
            "hy4_gptq",
            "hy4_gptq_adapter",
            "streamed_gptq_io",
            "streamed_gptq_quant",
            "streamed_gptq_memory",
        )
    ]
    names += [
        f"rapid_mlx/models/{n}.py"
        for n in ("hy_v4_flash", "hy_v4_flash_config", "hy_v4_flash_ops")
    ]
    return {name: file_hash(root / name) for name in names}


def build(adapter, destination, windows, evidence, layers, check=lambda: None):
    """Commit each layer and its quantized-prefix residuals as one restart unit."""
    if Path(destination).resolve().is_relative_to(adapter.source.resolve()):
        raise ValueError("destination cannot be inside source")
    layers_prefix(",".join(map(str, layers)), adapter.args.num_hidden_layers)
    if not windows or any(
        t.ndim != 1
        or not 0 < len(t) <= 8192
        or t.dtype.kind not in "iu"
        or np.any(t < 0)
        or np.any(t >= adapter.args.vocab_size)
        for t in windows
    ):
        raise ValueError("invalid calibration token windows")
    with artifact_lock(destination):
        return _build(adapter, destination, windows, evidence, layers, check)


def _build(adapter, destination, windows, evidence, layers, check):
    mx = mx_module()
    destination = Path(destination)
    if destination.resolve().is_relative_to(adapter.source.resolve()):
        raise ValueError("destination cannot be inside source")
    store = NativeStore(destination, adapter, layers)
    state = destination / "build-state"
    state.mkdir(exist_ok=True)
    identity = digest(
        dict(
            source=source_identity(adapter),
            tools=tool_identity(),
            recipe=asdict(adapter.recipe),
            layers=layers,
            evidence=evidence,
            tokens=[digest(t.tolist()) for t in windows],
        )
    )
    run_path = state / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["identity"] != identity:
        raise ValueError("resume identity mismatch")
    if not run_path.exists() and any(destination.glob("*.safetensors")):
        raise ValueError("destination contains an unrelated checkpoint")
    atomic_json(
        run_path,
        dict(
            identity=identity,
            evidence=evidence,
            token_hashes=[digest(t.tolist()) for t in windows],
            layers=layers,
            recipe=asdict(adapter.recipe),
        ),
    )
    previous = []
    for layer in [None, *layers]:
        key = "global" if layer is None else f"layer-{layer}"
        marker = state / f"{key}.json"
        if marker.exists():
            receipt = json.loads(marker.read_text())
            verify_receipt(destination, receipt, identity)
            expected_names = {
                key for name in adapter.names_for(layer) for key in store.entries(name)
            }
            if (
                set(receipt["weight_map"]) != expected_names
                or receipt["specs"] != store.specs
                or len(receipt["boundaries"]) != len(windows)
            ):
                raise ValueError("resume plan mismatch")
            store.weight_map.update(receipt["weight_map"])
            previous = [destination / p for p in receipt["boundaries"]]
            continue
        before = set(store.weight_map)
        for name in adapter.names_for(layer):
            if ".switch_mlp." not in name:
                store.rtn(name, check)
        reader = store.reader()
        boundaries = []
        stats = []
        if layer is not None:
            block = load_layer(adapter, layer, reader)
            if adapter.args.mlp_layer_types[layer] == "sparse":
                paths = routed_calibration(
                    adapter, block, layer, windows, previous, state, check
                )
                stats = quantize_experts(store, layer, paths, check)
                block.mlp.switch_mlp = disk_switch(adapter, layer, store.reader())
                for path in paths:
                    path.unlink()
        for i, tokens in enumerate(windows):
            hidden = (
                initial_hidden(adapter, tokens, reader)
                if layer is None
                else adapter.forward(block, load_boundary(previous[i]), tokens)
            )
            path = state / f"{key}-window-{i}.safetensors"
            save_boundary(path, hidden)
            boundaries.append(str(path.relative_to(destination)))
            del hidden
            check()
        if layer is not None:
            del block
        new_map = {n: store.weight_map[n] for n in store.weight_map.keys() - before}
        files = set(new_map.values()) | set(boundaries)
        receipt = dict(
            identity=identity,
            weight_map=new_map,
            specs=dict(store.specs),
            boundaries=boundaries,
            files={n: file_hash(destination / n) for n in sorted(files)},
            expert_stats=stats,
        )
        atomic_json(marker, receipt)
        previous = [destination / p for p in boundaries]
        gc.collect()
        mx.clear_cache()
    store.publish_index()
    atomic_json(destination / "config.json", store.config())
    for pattern in (
        "tokenizer*.json",
        "chat_template*.jinja",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        for path in adapter.source.glob(pattern):
            shutil.copyfile(path, destination / path.name)
    size = sum((destination / f).stat().st_size for f in set(store.weight_map.values()))
    report = dict(
        recipe=asdict(adapter.recipe),
        layers=layers,
        tensor_count=len(store.weight_map),
        safetensors_bytes=size,
        identity=identity,
    )
    atomic_json(destination / "size-report.json", report)
    return report


def quality(adapter, destination, windows, evidence, work, check=lambda: None):
    if Path(work).resolve().is_relative_to(adapter.source.resolve()):
        raise ValueError("quality work cannot be inside source")
    with artifact_lock(destination), artifact_lock(work):
        return _quality(adapter, destination, windows, evidence, work, check)


def _quality(adapter, destination, windows, evidence, work, check):
    from scripts.evaluate_deepseek_v41_rd import summarize_metrics, token_metrics

    destination = Path(destination)
    run = json.loads((destination / "build-state/run.json").read_text())
    train = {h for values in run["evidence"].values() for h in values}
    if train & {h for values in evidence.values() for h in values}:
        raise ValueError("held-out corpus overlaps calibration")
    if set(run["token_hashes"]) & {digest(t.tolist()) for t in windows}:
        raise ValueError("held-out token windows overlap calibration")
    if run["recipe"] != json.loads(json.dumps(asdict(adapter.recipe))):
        raise ValueError("quality recipe differs from build")
    # Receipts bind every candidate tensor to this source and build epoch.
    identity = digest(
        dict(
            source=source_identity(adapter),
            tools=tool_identity(),
            recipe=asdict(adapter.recipe),
            layers=run["layers"],
            evidence=run["evidence"],
            tokens=run["token_hashes"],
        )
    )
    if identity != run["identity"]:
        raise ValueError("quality source identity mismatch")
    expected_map = {}
    for key in ["global", *(f"layer-{i}" for i in run["layers"])]:
        receipt = json.loads((destination / "build-state" / f"{key}.json").read_text())
        verify_receipt(destination, receipt, identity)
        expected_map.update(receipt["weight_map"])
    if (
        json.loads((destination / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        != expected_map
    ):
        raise ValueError("candidate index differs from committed build")
    expected_store = NativeStore(destination, adapter, run["layers"])
    for layer in [None, *run["layers"]]:
        for name in adapter.names_for(layer):
            expected_store.entries(name)
    if json.loads((destination / "config.json").read_text()) != expected_store.config():
        raise ValueError("candidate config differs from committed build")
    reader = TensorReader(destination)
    # Finish the reference before creating any candidate layer.
    references = []
    for i, result in enumerate(
        logits(adapter, windows, Path(work) / "reference", run["layers"], check=check)
    ):
        path = Path(work) / f"reference-logits-{i}.npy"
        np.save(path, result)
        references.append(path)
    metrics = []
    for path, candidate in zip(
        references,
        logits(
            adapter, windows, Path(work) / "candidate", run["layers"], reader, check
        ),
        strict=True,
    ):
        reference = np.load(path, mmap_mode="r")
        metrics.append(token_metrics(reference, candidate))
    kl = np.concatenate([m[0] for m in metrics])
    flips = np.concatenate([m[1] for m in metrics])
    report = dict(
        summarize_metrics(kl, flips),
        tokens=len(kl),
        layers=run["layers"],
        truncated=len(run["layers"]) != adapter.args.num_hidden_layers,
        reference="layer_streamed_fp8_dequantized",
        evidence=evidence,
    )
    atomic_json(Path(work) / "quality.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "quality"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--slices", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        choices=("hy4-gptq-v1", "hy4-gptq-lowbit", "hy4-gptq-engram8"),
        default="hy4-gptq-v1",
    )
    parser.add_argument("--keep-embeddings-bf16", action="store_true")
    parser.add_argument("--gate-up-3bit-layers", default="")
    parser.add_argument("--only-layers")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--memory-limit-gb", type=float, default=55)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if not 0 < args.memory_limit_gb < 60:
        parser.error("memory-limit-gb must be positive and below 60")
    if args.command == "quality" and (
        args.work is None or args.only_layers is not None
    ):
        parser.error("quality needs --work and uses the build layer count")
    adapter = Hy4Adapter(
        args.source,
        Recipe(
            args.recipe,
            args.keep_embeddings_bf16,
            tuple(int(i) for i in args.gate_up_3bit_layers.split(",") if i),
        ),
    )
    windows, evidence = calibration(adapter, args.slices, args.tokens)
    from scripts.streamed_gptq_memory import MemoryGuard

    with MemoryGuard(
        dict(memory_limit_gb=args.memory_limit_gb, cache_limit_gb=0.25)
    ) as guard:
        if args.command == "build":
            layers = layers_prefix(args.only_layers, adapter.args.num_hidden_layers)
            report = build(
                adapter, args.destination, windows, evidence, layers, guard.check
            )
        else:
            if args.work.resolve().is_relative_to(args.source.resolve()):
                parser.error("quality work cannot be inside source")
            report = quality(
                adapter, args.destination, windows, evidence, args.work, guard.check
            )
        report.update(guard.report())
    atomic_json(
        (args.destination / "size-report.json")
        if args.command == "build"
        else (args.work / "quality.json"),
        report,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
