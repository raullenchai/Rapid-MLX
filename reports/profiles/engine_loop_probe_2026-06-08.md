# engine_loop B=1 timing probe — 2026-06-08

## Outcome

The hypothesis that the rapid-mlx HTTP B=1 gap to mlx-vlm lives in the
asyncio dispatch around `scheduler.step()` is **FALSIFIED**.

On both a small dense model (Qwen 3.5 4B 4-bit) and the same hybrid+MoE
model the original investigation used (Qwen 3.6 35B-A3B 4-bit), the
combined asyncio + executor + dispatch overhead is well under 2% of the
per-token wall clock. Rewriting the engine loop to a dedicated GPU
thread + `queue.Queue` model (the mlx-vlm pattern) would close less
than 1% of the gap, not the predicted ~33%.

The real gap lives **inside `Scheduler.step()`** — somewhere in
`vllm_mlx/scheduler.py` and the code it calls — not in the asyncio
plumbing around it.

## Method

`vllm_mlx/engine_core.py:_engine_loop` was instrumented with three
separated timings (env-gated via `RAPID_MLX_PROBE_ENGINE_LOOP=1`):

| Phase | Measurement |
|---|---|
| `step_inside_ms` | `time.perf_counter()` around `self.scheduler.step()` **inside** the executor thread |
| `step_await_ms` | `time.perf_counter()` around `await loop.run_in_executor(_executor, _timed_step)` on the asyncio loop |
| `dispatch_ms` | `time.perf_counter()` from start of output distribution through `await asyncio.sleep(0)` |
| executor RTT (derived) | `step_await_ms - step_inside_ms` — asyncio + GIL + thread context switch |

Aggregated and emitted as a structured `[engine_loop_probe]` INFO log
line every 32 steps. Zero overhead when the env var is unset.

Harness: `scripts/probe_engine_loop.py` — spawns
`python3.12 -m vllm_mlx.cli serve <alias>` on port 8765, fires a 256-token
warmed-up streaming chat completion at B=1, parses the probe lines, and
prints a summary.

Hardware: M3 Ultra 256 GB, macOS 26.2, Python 3.12.

## Numbers

### Qwen 3.5 4B 4-bit (`qwen3.5-4b`)

```
step_inside_ms   : avg=7.013  min=6.986  max=7.037
step_await_ms    : avg=7.083  min=7.056  max=7.108
executor RTT     : avg=0.071
loop_dispatch_ms : avg=0.043  min=0.041  max=0.046
per_token_ms     : 7.126  -> 140.33 tok/s
breakdown        : inside=98.4%  rtt=1.0%  dispatch=0.6%
end-to-end HTTP  : 128.52 tok/s
```

### Qwen 3.6 35B-A3B 4-bit (`qwen3.6-35b`)

```
step_inside_ms   : avg=14.205  min=14.129  max=14.411
step_await_ms    : avg=14.288  min=14.213  max=14.499
executor RTT     : avg=0.084
loop_dispatch_ms : avg=0.048  min=0.047  max=0.049
per_token_ms     : 14.337  -> 69.75 tok/s
breakdown        : inside=99.1%  rtt=0.6%  dispatch=0.3%
end-to-end HTTP  : 65.79 tok/s
```

## What changed about the picture

The earlier investigation (perf-b1-asyncio-loop-root-cause-2026-06-08
memory entry) measured rapid-mlx's HTTP path at 65.3 tok/s and an
external "rapid-mlx Scheduler.step sync loop" at 97.9 tok/s on the same
model and hardware, and concluded the 33% delta lived in the asyncio
wrapper.

The probe contradicts both halves of that delta:

1. `scheduler.step()` measured **inside** the running engine averages
   **14.2 ms/token** (= 70.4 tok/s), not 10.2 ms / 97.9 tok/s. The
   "sync loop" bench was measuring something cheaper than what the
   engine actually executes per step — likely raw
   `mlx_lm.generate.BatchGenerator.next()` without our scheduler /
   sampler / KV / prefix-cache wrappers, plus without the prefill +
   KV-cache state that long-running decode accumulates.

2. asyncio executor RTT is **84 μs**, not 5 ms.

End-to-end HTTP (65.8 tok/s) ≈ engine decode (69.8 tok/s) × ~94%, where
the ~6% is prefill + TTFT + HTTP framing on the 256-token request
budget — nothing to attribute to the engine loop.

## Where the gap to mlx-vlm actually lives

mlx-vlm runs Qwen 3.6 35B-A3B at ~91.6 tok/s on the same hardware. Our
engine averages 69.8 tok/s. The 21.8 tok/s delta is **3.3 ms/token of
extra time inside `Scheduler.step()` or below it** — i.e. in code that
mlx-vlm doesn't run.

Candidates (have NOT been validated; this is the next probe):

1. **Scheduler-layer Python overhead.** `vllm_mlx/scheduler.py:step()` does
   request bookkeeping, finish detection, sampler fast-path gating
   (PR #521), spec-decode hooks, prefix-cache check, KV-cache stitching.
   mlx-vlm's `BatchGenerator.next()` is leaner.
2. **mlx-lm vs mlx-vlm BatchGenerator divergence.** mlx-vlm's
   `_greedy_argmax_step` (`mlx_vlm/generate/ar.py:886-914`) skips the
   full logits tensor when `top_p=1, temp=0` — saves the
   vocab-wide softmax. Our hybrid path doesn't use this. PR #521's
   dense fast path doesn't fire for hybrid (Qwen 3.6 hybrid is
   `is_hybrid=True`).
3. **Hybrid state-machine cost.** Qwen 3.6 hybrid (mixed attention +
   DeltaNet) does per-layer routing. mlx-vlm may have a tighter
   implementation. Already on the radar — see
   `perf_hybrid_b1_sampler_hypothesis_falsified_2026-06-06.md`.

## Decision

- **Do NOT rewrite the engine loop** for a dedicated GPU thread +
  `queue.Queue` (mlx-vlm pattern). The estimated upside is < 1%, not
  worth the concurrency-rewrite risk.
- **Next perf PR**: drill inside `Scheduler.step()` with the same
  inside/outside split methodology. Likely targets in order: hybrid
  state machine, sampler fast-path generalisation, prefix-cache check
  cost.
- **Keep the probe** — env-gated, zero overhead when off, makes
  follow-up perf PRs cheaper to validate.

## Repro

```
RAPID_MLX_PROBE_ENGINE_LOOP=1 python3.12 -m vllm_mlx.cli serve qwen3.6-35b --port 8765
# or:
python3.12 scripts/probe_engine_loop.py --alias qwen3.6-35b --max-tokens 256
```

Raw log artefacts: `/tmp/probe_engine_loop_qwen3.5-4b.log`,
`/tmp/probe_engine_loop_qwen3.6-35b.log`.
