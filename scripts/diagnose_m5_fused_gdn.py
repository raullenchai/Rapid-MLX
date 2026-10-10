#!/usr/bin/env python3
"""Compare real-weight fused GDN calls against a stock-driven trajectory."""

import argparse
import copy
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from rapid_mlx import qwen35_fused_gdn_decode as f
    from rapid_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
    from rapid_mlx.moe_fusion import fuse_gate_up
    from rapid_mlx.patches.qwen3_5_eager_dispatch import install_qwen3_5_eager_dispatch
    from rapid_mlx.qwen35_moe_router import install_qwen35_moe_router

    model, tok = load(str(a.model))
    install_qwen3_5_eager_dispatch()
    fusion = {
        "gate_up": fuse_gate_up(model),
        "input": fuse_gdn_in_proj(model),
        "router": install_qwen35_moe_router(model),
        "decode": f.install_qwen35_fused_gdn_decode(model),
    }
    layers = {
        id(m): name for name, m in model.named_modules() if getattr(m, f._TAG, False)
    }
    rows = []
    counts = {}
    classes = {type(m) for _, m in model.named_modules() if id(m) in layers}

    def metrics(x, y):
        return {
            "exact": bool(mx.array_equal(x, y).item()),
            "max_abs": float(
                mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()
            ),
        }

    for cls in classes:
        original = cls.__call__

        def checked(self, inputs, mask=None, cache=None, _original=original):
            eligible = f._eligible(self, inputs, mask, cache)
            count = counts.get(id(self), 0)
            other = copy.deepcopy(cache) if eligible and count < 3 else None
            setattr(self, f._TAG, False)
            try:
                stock = _original(self, inputs, mask, cache)
            finally:
                setattr(self, f._TAG, True)
            if other is not None:
                fused = _original(self, inputs, mask, other)
                mx.eval(stock, fused, *cache.state, *other.state)
                rows.append(
                    {
                        "layer": layers[id(self)],
                        "step": count,
                        "output": metrics(stock, fused),
                        "conv": metrics(cache[0], other[0]),
                        "state": metrics(cache[1], other[1]),
                    }
                )
                counts[id(self)] = count + 1
            return stock

        cls.__call__ = checked
    prompt = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Implement a Python LRU cache with O(1) get and put, plus tests.",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    out = list(
        stream_generate(model, tok, prompt, max_tokens=8, sampler=make_sampler(temp=0))
    )
    if not layers or any(counts.get(layer_id, 0) != 3 for layer_id in layers):
        raise RuntimeError(
            "did not observe three eligible calls on every enrolled layer"
        )
    result = {
        "fusion": fusion,
        "comparisons": rows,
        "tokens": [int(x.token) for x in out],
        "all_exact": all(
            v["exact"]
            for r in rows
            for k, v in r.items()
            if k in ["output", "conv", "state"]
        ),
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "comparisons": len(rows),
                "all_exact": result["all_exact"],
                "first_mismatches": [
                    r
                    for r in rows
                    if not all(r[k]["exact"] for k in ["output", "conv", "state"])
                ][:3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
