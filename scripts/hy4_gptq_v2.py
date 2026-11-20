"""Opt-in Hy4 v2 policy; legacy recipe serialization stays unchanged."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.hy4_gptq_adapter import Recipe


def default_bits(layer, projection):
    return 6 if projection == "down_proj" or 20 <= layer <= 26 else 4


def parse_bits_map(value):
    """JSON object (inline or file): {"20": {"gate_proj": 4, "up_proj": 6}}.

    Entries replace the v2 default for that zero-based layer/projection bank.
    All experts in a bank share bits and group 64, as required by gather_qmm.
    """
    raw = json.loads(
        value if value.lstrip().startswith(("{", "[")) else Path(value).read_text()
    )
    if not isinstance(raw, dict):
        raise ValueError("bits map must be a JSON object")
    entries = []
    for layer, projections in raw.items():
        if not str(layer).isdigit() or not isinstance(projections, dict):
            raise ValueError(
                "bits map needs nonnegative layer IDs and projection objects"
            )
        for projection, bits in projections.items():
            if (
                projection not in ("gate_proj", "up_proj", "down_proj")
                or type(bits) is not int
                or bits not in (2, 3, 4, 6, 8)
            ):
                raise ValueError(
                    "bits map needs gate/up/down_proj and native affine bits"
                )
            entries.append((int(layer), projection, bits))
    if len({(i, p) for i, p, _ in entries}) != len(entries):
        raise ValueError("duplicate bits map layer/projection")
    return tuple(sorted(entries))


@dataclass(frozen=True)
class V2Recipe(Recipe):
    name: str = "hy4-gptq-v2"
    bits_map: tuple[tuple[int, str, int], ...] = ()
    routing_weight_squared: bool = True
    min_tokens: int = 1024
    calibration_token_budget: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.name != "hy4-gptq-v2":
            raise ValueError("v2 options require hy4-gptq-v2")
        if (
            type(self.routing_weight_squared) is not bool
            or type(self.min_tokens) is not int
            or self.min_tokens < 1
        ):
            raise ValueError("invalid v2 weighting or minimum tokens")
        if self.calibration_token_budget is not None and (
            type(self.calibration_token_budget) is not int
            or self.calibration_token_budget < 1
        ):
            raise ValueError("calibration token budget must be positive")
        # Also validate callers constructing the dataclass without the CLI parser.
        parsed = parse_bits_map(
            json.dumps(
                {
                    str(i): {p: b for j, p, b in self.bits_map if j == i}
                    for i, _, _ in self.bits_map
                }
            )
        )
        if parsed != self.bits_map:
            raise ValueError("bits map must be unique and sorted")

    def spec(self, name):
        from rapid_mlx.models.hy_v4_flash_config import checkpoint_key

        canonical = checkpoint_key(name)
        spec = super().spec(name)
        if canonical and spec and ".switch_mlp." in canonical:
            layer = int(canonical.split(".")[2])
            projection = canonical.split(".")[-2]
            for i, p, bits in self.bits_map:
                if (i, p) == (layer, projection):
                    return bits, 64, "gptq"
        return spec


def budget_windows(records, tokens, budget=None):
    """Round-robin contiguous windows, without duplicating short source records."""
    baseline = sum(min(tokens, len(ids)) for ids in records)
    budget = 4 * baseline if budget is None else budget
    if type(budget) is not int or budget < 1:
        raise ValueError("calibration token budget must be positive")
    windows, used = [], 0
    for start in range(0, max(map(len, records)), tokens):
        for ids in records:
            if start < len(ids) and used < budget:
                window = np.asarray(
                    ids[start : start + min(tokens, budget - used)], np.int32
                )
                windows.append(window)
                used += len(window)
        if used == budget:
            break
    return windows, dict(
        v1_tokens=baseline,
        requested_tokens=budget,
        actual_tokens=used,
        source_exhausted=used < budget,
        router_top_up=False,
    )


def routing_weights(weights, squared):
    weights = np.asarray(weights, dtype=np.float32)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("invalid routing weight")
    return weights**2 if squared else weights


def low_token_policy(count, cols):
    """Linear damping ramp from 0.1 at cols to 0.2 at half cols."""
    if count < 0.5 * cols:
        return "rtn", None
    return "gptq", 0.1 + 0.1 * max(0.0, 2 * (1 - count / cols))


def coverage(stats, min_tokens):
    counts = [s["gate_proj"]["tokens"] for s in stats]
    fraction = sum(c >= min_tokens for c in counts) / len(counts)
    return dict(
        min_tokens=min_tokens,
        fraction=fraction,
        target=0.98,
        target_met=fraction >= 0.98,
        unseen=sum(c == 0 for c in counts),
    )


def estimate_sizes(adapter, layers):
    """Payload estimates including BF16 scales/biases; excludes runtime workspace/KV."""
    totals = {
        "v1_disk_bytes": 0,
        "v2_disk_bytes": 0,
        "v1_resident_weight_bytes": 0,
        "v2_resident_weight_bytes": 0,
    }
    legacy = Recipe(keep_embeddings_bf16=adapter.recipe.keep_embeddings_bf16)
    for layer in [None, *layers]:
        for name in adapter.names_for(layer):
            shape = adapter.shape(name)
            elements = math.prod(shape)
            for label, recipe in (("v1", legacy), ("v2", adapter.recipe)):
                spec = recipe.spec(name)
                if spec:
                    bits, group, _ = spec
                    size = elements * bits // 8 + elements // group * 4
                else:
                    original, _ = adapter.names[name]
                    dtype = adapter.reader.index.tensors[original].dtype
                    size = (
                        elements
                        * {
                            "BF16": 2,
                            "F16": 2,
                            "F32": 4,
                            "I32": 4,
                            "I64": 8,
                            "F8_E4M3": 2,
                            "F8_E4M3FN": 2,
                        }[dtype]
                    )
                totals[label + "_disk_bytes"] += size
                if not name.endswith(".multi_head_embedding.embedding.weight"):
                    totals[label + "_resident_weight_bytes"] += size
    totals["disk_delta_bytes"] = totals["v2_disk_bytes"] - totals["v1_disk_bytes"]
    totals["resident_delta_bytes"] = (
        totals["v2_resident_weight_bytes"] - totals["v1_resident_weight_bytes"]
    )
    totals["resident_weight_target_met"] = totals["v2_resident_weight_bytes"] <= 100e9
    return totals


def quantize_experts(store, layer, paths, check):
    from scripts.hy4_gptq import expert_chunks
    from scripts.streamed_gptq_io import mx_module
    from scripts.streamed_gptq_quant import QuantConfig, gptq, gram

    mx = mx_module()
    adapter, recipe = store.adapter, store.adapter.recipe
    if not isinstance(recipe, V2Recipe):
        recipe = V2Recipe(keep_embeddings_bf16=recipe.keep_embeddings_bf16)
    base = f"model.layers.{layer}.mlp.switch_mlp."
    names = [base + p + ".weight" for p in ("gate_proj", "up_proj", "down_proj")]
    sinks = [store.sink(n) for n in names]
    stats = []
    try:
        for expert in range(adapter.args.n_routed_experts):
            fitted, expert_stats = [], {}
            for name, sink in zip(names, sinks, strict=True):
                bits, group, _ = recipe.spec(name)
                weight = adapter.tensor(name, expert=expert)
                down = name.endswith(".down_proj.weight")
                chunks = (
                    (x, routing_weights(w, recipe.routing_weight_squared))
                    for x, w in expert_chunks(
                        paths, expert, adapter, fitted if down else None
                    )
                )
                h, count = gram(chunks, weight.shape[-1], check=check)
                method, damping = low_token_policy(count, weight.shape[-1])
                if method == "gptq":
                    config = QuantConfig(
                        group_size=group,
                        damping=damping,
                        low_token_damping=damping,
                        min_tokens=recipe.min_tokens,
                    )
                    triple, metrics = gptq(
                        weight, h, bits, config, count=count, check=check
                    )
                else:
                    triple = mx.quantize(weight, bits=bits, group_size=group)
                    metrics = dict(
                        tokens=count, fallback="low_token_rtn", cols=weight.shape[-1]
                    )
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
    print(
        json.dumps(dict(layer=layer, coverage=coverage(stats, recipe.min_tokens))),
        flush=True,
    )
    return stats
