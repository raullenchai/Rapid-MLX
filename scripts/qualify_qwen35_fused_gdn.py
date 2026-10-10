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


def weight_identity(path: Path) -> dict:
    """Verify cached content-addressed bytes, rather than trusting file sizes."""
    blob = path.resolve()
    if blob.parent.name != "blobs" or not re.fullmatch(r"[0-9a-f]{64}", blob.name):
        raise ValueError("weights must reference content-addressed cached blobs")
    hasher = hashlib.sha256()
    with path.open("rb") as shard:
        for chunk in iter(lambda: shard.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != blob.name or path.resolve() != blob:
        raise ValueError(f"cached weight SHA-256 mismatch: {path.name}")
    return {"name": path.name, "size": path.stat().st_size, "sha256": actual}


def snapshot_identity(path: Path) -> dict:
    # Check before resolve(): symlinked cold-cache snapshots retain their HF SHA.
    if path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise ValueError("model must name an immutable cached snapshots/<40-hex-SHA>")
    files = sorted(path.glob("*.safetensors"))
    if not files or not all(p.is_file() for p in files):
        raise ValueError("snapshot has no complete local weights")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        manifest = json.loads(index.read_text()).get("weight_map")
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("checkpoint index has no weight manifest")
        shards = set(manifest.values())
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or not (path / name).is_file()
            for name in shards
        ):
            raise ValueError("checkpoint index references missing or invalid shards")
    elif len(files) != 1:
        raise ValueError("sharded checkpoint requires a complete weight index")
    return {
        "repository": path.parent.parent.name.removeprefix("models--").replace(
            "--", "/"
        ),
        "revision": path.name,
        "config_sha256": digest((path / "config.json").read_bytes()),
        "index_sha256": digest(index.read_bytes()) if index.is_file() else None,
        "weights": [weight_identity(p) for p in files],
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
        # Production catches an unpacking failure and falls back to native.
        # Count only the exact return contract that it can consume.
        output, conv, state = kernel(*a, **kw)
        calls += 1
        return output, conv, state

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
        width = {
            mx.bfloat16: mx.uint16,
            mx.float16: mx.uint16,
            mx.float32: mx.uint32,
        }.get(array.dtype, mx.uint8)
        return np.array(array.view(width))

    def finite(array, bits):
        exponent = {
            mx.bfloat16: 0x7F80,
            mx.float16: 0x7C00,
            mx.float32: 0x7F800000,
        }.get(array.dtype)
        return exponent is not None and not bool(np.any((bits & exponent) == exponent))

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
            differing_indices = {}
            for name, (a, b) in tensors.items():
                a_bits, b_bits = storage(a), storage(b)
                metric = {
                    "stock_dtype": str(a.dtype),
                    "fused_dtype": str(b.dtype),
                    "stock_shape": list(a.shape),
                    "fused_shape": list(b.shape),
                    "stock_sha256": digest(a_bits.tobytes()),
                    "fused_sha256": digest(b_bits.tobytes()),
                    "finite": finite(a, a_bits) and finite(b, b_bits),
                    "error": None,
                }
                if a.dtype != b.dtype or a.shape != b.shape:
                    metric["error"] = "oracle/candidate shape/dtype disagreement"
                else:
                    metric.update(compare_bits(a_bits, b_bits))
                    differing_indices[name] = np.flatnonzero(
                        a_bits.reshape(-1) != b_bits.reshape(-1)
                    )
                metric["exact"] = bool(
                    metric["error"] is None
                    and metric["finite"]
                    and metric.get("differing_elements") == 0
                )
                metrics[name] = metric
            row = {**context, "layer": layers[id(layer)], "metrics": metrics}
            rows.write(json.dumps(row, sort_keys=True) + "\n")
            row_count += 1
            if not all(m["exact"] for m in metrics.values()):
                witness = {
                    f"{arm}_{name}": pair[i]
                    for name, pair in tensors.items()
                    for i, arm in enumerate(("stock", "fused"))
                }
                witness["hidden_inputs"] = inputs
                mx.save_safetensors(
                    str(out / f"mismatch-{len(trajectories)}.safetensors"), witness
                )
                for name, indices in differing_indices.items():
                    np.save(
                        out / f"mismatch-{len(trajectories)}-{name}-indices.npy",
                        indices,
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
                    except Exception as exc:
                        # Operational failures become failed receipts. Process
                        # controls (KeyboardInterrupt/SystemExit) still escape.
                        error = f"{type(exc).__name__}: {exc}"
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
            try:
                result = qualify(
                    model.expanduser().absolute(), args.histories, args.steps, out
                )
            except Exception as exc:
                result = {
                    "model": str(model),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "trajectories": [],
                }
            results.append(result)
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
