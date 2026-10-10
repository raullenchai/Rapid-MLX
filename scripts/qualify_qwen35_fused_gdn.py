#!/usr/bin/env python3
"""Local-only native-oracle GDN qualification; no speed or universal parity claim.

Run as python -m scripts.qualify_qwen35_fused_gdn. Each arm owns a fresh prompt
cache and a persistent shadow GDN cache, using shared real-model hidden inputs.
The model follows the first arm in the requested order. All decode steps compare
native normalized output, projected output, convolution and recurrent state.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_bits(left: np.ndarray, right: np.ndarray) -> dict:
    """Compare storage bits, including signed zero, without dtype conversion."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("oracle shape/dtype disagreement")
    indices = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    return {
        "shape": list(left.shape),
        "storage_dtype": str(left.dtype),
        "stock_sha256": digest(left.tobytes()),
        "fused_sha256": digest(right.tobytes()),
        "differing_elements": int(indices.size),
        "indices_sha256": digest(indices.astype("<i8").tobytes()),
        "first_indices": indices[:64].tolist(),
    }


def snapshot_identity(path: Path) -> dict:
    # Check before resolve(): symlinked cold-cache snapshots retain their HF SHA.
    if path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise ValueError("model must name an immutable cached snapshots/<40-hex-SHA>")
    files = sorted(path.glob("*.safetensors"))
    if not files or not all(p.is_file() for p in files):
        raise ValueError("snapshot has no complete local weights")
    return {
        "repository": path.parent.parent.name.removeprefix("models--").replace(
            "--", "/"
        ),
        "revision": path.name,
        "config_sha256": digest((path / "config.json").read_bytes()),
        "weights": [{"name": p.name, "size": p.stat().st_size} for p in files],
    }


def source_inventory() -> dict:
    from mlx_lm.models import gated_delta, qwen3_5, qwen3_next

    from rapid_mlx import qwen35_fused_gdn_decode
    from rapid_mlx.kernels import qwen4_fused_gdn_decode

    root = Path(__file__).resolve().parent.parent
    sources = {
        "harness": Path(__file__),
        "adapter": Path(qwen35_fused_gdn_decode.__file__),
        "kernel": Path(qwen4_fused_gdn_decode.__file__),
        "native_layer": Path(inspect.getfile(qwen3_5)),
        "native_recurrence": Path(inspect.getfile(gated_delta)),
        "native_normalization": Path(inspect.getfile(qwen3_next)),
    }
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines(),
        "source_sha256": {k: digest(p.read_bytes()) for k, p in sources.items()},
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("mlx", "mlx-metal", "mlx-lm", "numpy")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
    }


def qualify(model_path: Path, histories: list[int], steps: int, out: Path) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from rapid_mlx import qwen35_fused_gdn_decode as fused

    identity = snapshot_identity(model_path)
    # No tokenizer/model network access: only an existing local snapshot.
    model, tokenizer = load(model_path, tokenizer_config={"local_files_only": True})
    enrolled = fused.install_qwen35_fused_gdn_decode(model)
    if not enrolled:
        return {"model": identity, "status": "not_admitted", "trajectories": []}
    layers = {
        id(m): name
        for name, m in model.named_modules()
        if getattr(m, fused._TAG, False)
    }
    if len(layers) != enrolled:
        raise RuntimeError("enrollment count disagreement")
    families = {type(m) for _, m in model.named_modules() if id(m) in layers}
    if len(families) != 1:
        raise RuntimeError("qualification expects one native layer class")
    family = families.pop()
    production = family.__call__
    native = family._rapid_qwen35_fused_gdn_original_call
    kernel = fused.fused_gdn_decode
    calls = 0

    def counted_kernel(*a, **kw):
        nonlocal calls
        result = kernel(*a, **kw)
        calls += 1
        return result

    class Capture(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.value = None

        def __call__(self, value):
            self.value = value
            return self.inner(value)

    captures = {}
    for _, layer in model.named_modules():
        if id(layer) in layers:
            captures[id(layer)] = (layer, layer.out_proj)
    for layer, projection in captures.values():
        layer.out_proj = Capture(projection)

    def storage(array):
        if array.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            raise ValueError(f"unexpected tensor dtype {array.dtype}")
        bits = np.array(
            array.view(mx.uint32 if array.dtype == mx.float32 else mx.uint16)
        )
        exponent = {
            mx.bfloat16: 0x7F80,
            mx.float16: 0x7C00,
            mx.float32: 0x7F800000,
        }[array.dtype]
        if np.any((bits & exponent) == exponent):
            raise ValueError("non-finite oracle/candidate tensor")
        return bits

    trajectories = []
    shadow = {}
    context = {}
    row_count = 0
    with gzip.open(out / "rows.jsonl.gz", "wt") as rows:

        def audited(layer, inputs, mask=None, cache=None):
            nonlocal row_count
            if id(layer) not in layers:
                return native(layer, inputs, mask, cache)
            if context["step"] == -1:
                result = native(layer, inputs, mask, cache)
                mx.eval(result, *cache.state)
                shadow[id(layer)] = copy.deepcopy(cache)
                return result
            if id(layer) not in shadow:
                raise RuntimeError("decode has no independent native prefill cache")
            if not fused._eligible(layer, inputs, mask, cache):
                raise RuntimeError("decode unexpectedly outside production admission")
            primary = context["order"][0]
            caches = {primary: cache, context["order"][1]: shadow[id(layer)]}
            results, normalized = {}, {}
            before = calls
            for arm in context["order"]:
                fn = native if arm == "stock" else production
                results[arm] = fn(layer, inputs, mask, caches[arm])
                normalized[arm] = layer.out_proj.value
                # Materialize each arm completely before starting the next.
                mx.eval(results[arm], normalized[arm], *caches[arm].state)
            if calls - before != 1:
                raise RuntimeError("production fused arm silently fell back")
            tensors = {
                "normalized_output": (normalized["stock"], normalized["fused"]),
                "projected_output": (results["stock"], results["fused"]),
                "conv": (caches["stock"][0], caches["fused"][0]),
                "state": (caches["stock"][1], caches["fused"][1]),
            }
            metrics = {}
            for name, (a, b) in tensors.items():
                if a.dtype != b.dtype:
                    raise ValueError(f"oracle/candidate dtype disagreement: {name}")
                metrics[name] = {
                    "dtype": str(a.dtype),
                    **compare_bits(storage(a), storage(b)),
                }
            row = {**context, "layer": layers[id(layer)], "metrics": metrics}
            rows.write(json.dumps(row, sort_keys=True) + "\n")
            row_count += 1
            if any(m["differing_elements"] for m in metrics.values()):
                witness = {
                    f"{arm}_{name}": pair[i]
                    for name, pair in tensors.items()
                    for i, arm in enumerate(("stock", "fused"))
                }
                witness["hidden_inputs"] = inputs
                mx.save_safetensors(
                    str(out / f"mismatch-{len(trajectories)}.safetensors"), witness
                )
                for name, (a, b) in tensors.items():
                    np.save(
                        out / f"mismatch-{len(trajectories)}-{name}-indices.npy",
                        np.flatnonzero(
                            storage(a).reshape(-1) != storage(b).reshape(-1)
                        ),
                    )
                raise ValueError(
                    f"native mismatch at {layers[id(layer)]} step {context['step']}"
                )
            return results[primary]

        fused.fused_gdn_decode = counted_kernel
        family.__call__ = audited
        try:
            text = "Record 17: copper key in Kyoto. Record 23: blue lantern in Oslo. "
            body = tokenizer.encode(text, add_special_tokens=False)
            for history in histories:
                prompt = (body * (history // len(body) + 1))[:history]
                for order in (("stock", "fused"), ("fused", "stock")):
                    shadow.clear()
                    cache = make_prompt_cache(model)
                    context = {"history": history, "order": list(order), "step": -1}
                    start = row_count
                    tokens = []
                    logits = None
                    error = None
                    try:
                        # Chunked native prefill; fresh independent caches per order.
                        for offset in range(0, history, 256):
                            logits = model(
                                mx.array([prompt[offset : offset + 256]]), cache=cache
                            )
                            mx.eval(
                                logits,
                                *[v for c in cache for v in c.state if v is not None],
                            )
                        for step in range(steps):
                            context["step"] = step
                            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                            logits = model(mx.array([[token]]), cache=cache)
                            mx.eval(
                                logits,
                                *[v for c in cache for v in c.state if v is not None],
                            )
                            tokens.append(token)
                        if row_count - start != steps * enrolled:
                            raise RuntimeError("missing decode comparisons")
                    except (ValueError, RuntimeError) as exc:
                        error = str(exc)
                    trajectory = {
                        "history": history,
                        "order": list(order),
                        "steps_completed": len(tokens),
                        "rows": row_count - start,
                        "tokens": tokens,
                        "text": tokenizer.decode(tokens),
                        "error": error,
                    }
                    trajectories.append(trajectory)
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in trajectory.items()
                                if k not in ("tokens", "text")
                            }
                        ),
                        flush=True,
                    )
                    del cache, logits
                    shadow.clear()
                    mx.clear_cache()
        finally:
            family.__call__ = production
            fused.fused_gdn_decode = kernel
            for layer, projection in captures.values():
                layer.out_proj = projection
    exact = all(t["error"] is None for t in trajectories)
    for i in range(0, len(trajectories), 2):
        exact &= trajectories[i]["tokens"] == trajectories[i + 1]["tokens"]
    return {
        "model": identity,
        "enrolled_layers": enrolled,
        "threadgroup_y": next(iter(captures.values()))[
            0
        ]._rapid_qwen35_fused_gdn_threadgroup_y,
        "status": "exact" if exact else "mismatch",
        "rows": row_count,
        "rows_sha256": digest((out / "rows.jsonl.gz").read_bytes()),
        "trajectories": trajectories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--histories", type=int, nargs="+", default=[128, 4096])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.histories) < 2 or args.steps < 1:
        parser.error("histories >= 2 and steps >= 1 required")
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = source_inventory()
    inventory["execution"] = {
        "pid": os.getpid(),
        "arms": "serial mx.eval barriers",
        "hardware_exclusive": False,
        "performance_claim": False,
    }
    inventory["process_inventory_before"] = subprocess.check_output(
        ["ps", "-axo", "pid,comm"], text=True
    ).splitlines()
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    results = []
    # Cooperative process exclusion only; unrelated GPU jobs are not stopped.
    with open("/private/tmp/rapid-mlx-qwen35-qualification.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, model in enumerate(args.model):
            out = args.output / str(index)
            out.mkdir()
            results.append(
                qualify(model.expanduser().absolute(), args.histories, args.steps, out)
            )
    report = {
        "inventory": inventory,
        "histories": args.histories,
        "steps": args.steps,
        "results": results,
        "exact": all(r["status"] == "exact" for r in results),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
