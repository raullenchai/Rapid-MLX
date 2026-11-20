#!/usr/bin/env python3
"""Metal-host-only Hy4 probe. Compare --flush none/aux in fresh processes.
GC counts omit C++ graph nodes/MTLBuffers; logical bytes can count aliases twice.
--flush evaluates selected state each step; --clear-every tests allocator caching.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HY4_ENGRAM_SSD_OFFLOAD"] = "1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/Volumes/RTL-2T/models-cold/hy4-flash-gptq-v1")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-limit-gb", type=float, default=34.4)
    p.add_argument("--clear-every", type=int, default=0)
    p.add_argument("--prompt", default="Solve a hard science problem step by step.")
    modes = ["none", "offset", "slot1", "slot2", "aux", "all"]
    p.add_argument("--flush", choices=modes, default="none")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    if not Path(args.model).is_dir():
        p.error("model must be an existing local directory")

    import mlx.core as mx

    if not mx.metal.is_available():
        p.error("requires a Metal host; do not run in the sandbox")
    from mlx_lm.generate import BatchGenerator

    from rapid_mlx.moe_fusion import fuse_gate_up
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    model, tokenizer = load_model_with_fallback(args.model)
    fuse_gate_up(model)  # Same post-load MoE fusion as BatchedEngine.
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    readers = [
        layer.engram_layer.multi_head_embedding.embedding
        for layer in model.layers
        if layer.engram_layer is not None
    ]
    if not readers or not all(hasattr(r, "_cache") for r in readers):
        raise RuntimeError("Engram SSD offload was not installed")
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    class Probe:
        calls = 0
        decode_calls = 0

        def __getattr__(self, name):
            return getattr(model, name)

        def __call__(self, inputs, cache=None):
            logits = model(inputs, cache=cache)
            self.calls += 1
            self.decode_calls += int(inputs.shape[1] == 1)
            if cache is not None and args.flush != "none":
                roots = []
                for layer in cache:
                    for slot, c in enumerate(layer.caches):
                        selected = args.flush in ("all", f"slot{slot}")
                        selected |= args.flush == "aux" and slot > 0
                        if args.flush == "offset":
                            roots.append(c.offset)
                        elif selected:
                            roots.extend([c.keys, c.values, c.offset])
                mx.async_eval([x for x in roots if isinstance(x, mx.array)])
            return logits

    probe = Probe()
    generator = BatchGenerator(
        probe,
        max_tokens=args.steps + 2,
        stop_tokens=[],
        prefill_batch_size=args.batch_size,
        completion_batch_size=args.batch_size,
        prefill_step_size=max(2, len(tokens)),
    )
    baseline = mx.get_active_memory()

    def report(label):
        # Do not evaluate cache state here: that would mask the suspected leak.
        cache = generator._generation_batch.prompt_cache
        sizes, roots = [0] * 3, [0] * 3
        for layer in cache:
            for slot, c in enumerate(layer.caches):
                arrays = [
                    x for x in (c.keys, c.values, c.offset) if isinstance(x, mx.array)
                ]
                sizes[slot] += sum(x.nbytes for x in arrays)
                roots[slot] += len(arrays)
        active = mx.get_active_memory()
        record = dict(
            label=label,
            decode_calls=probe.decode_calls,
            cache_updates_per_slot=probe.calls * len(model.layers),
            active_bytes=active,
            active_delta_bytes=active - baseline,
            cache_bytes=mx.get_cache_memory(),
            gc_mx_arrays=sum(isinstance(x, mx.array) for x in gc.get_objects()),
            cache_roots_per_slot=roots,
            logical_bytes_per_slot=sizes,
            engram_lru_rows=[len(r._cache) for r in readers],
            engram_pending=sum(r._pending is not None for r in readers),
            engram_misses=[r.cache_misses for r in readers],
            flush=args.flush,
        )
        print(json.dumps(record), flush=True)

    try:
        report("loaded")
        generator.insert([list(tokens) for _ in range(args.batch_size)])
        last = 0
        while probe.decode_calls < args.steps:
            # next() uses the serving generation stream and actual _step path.
            generator.next()
            if args.clear_every and probe.decode_calls % args.clear_every == 0:
                mx.synchronize(generator.stream)
                mx.clear_cache()
            if probe.decode_calls // 200 > last // 200:
                mx.synchronize(generator.stream)
                report("decode")
                last = probe.decode_calls
        mx.synchronize(generator.stream)
        report("done")
    except Exception:
        report("error")
        raise
    finally:
        generator.close()
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
