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

---

## Phase 2 — scheduler sub-phase probe (same day)

After the engine-loop probe ruled out the asyncio wrapper, `Scheduler.step()`
was instrumented with a second probe (same env var
`RAPID_MLX_PROBE_ENGINE_LOOP=1`) that splits each step into six measurable
buckets: `bg_next` (`self.batch_generator.next()` — mlx-lm's
`BatchGenerator.next()`), `process_resp` (detokenise + stop check +
output construction + cleanup_finished), `schedule` (admit waiting +
prompt token accounting), `snapshot` (hybrid prompt + boundary KV cache
hooks), `aborts`, `periodic` (`mx.eval` + `mx.clear_cache` + memory
log), and `other`.

### Numbers (B=1, M3 Ultra, default sampling = temperature 0.7, top_p model default)

Qwen 3.6 35B-A3B 4-bit::

    bg_next       : avg=14.302 ms (99.3%)
    process_resp  : avg= 0.082 ms ( 0.6%)
    schedule      : avg= 0.002 ms ( 0.0%)
    snapshot      : avg= 0.002 ms ( 0.0%)
    aborts        : avg= 0.001 ms ( 0.0%)
    periodic      : avg= 0.015 ms ( 0.1%)
    other         : avg= 0.003 ms ( 0.0%)
    TOTAL         : avg=14.406 ms      -> 69.4 tok/s engine
    end-to-end HTTP: 64.9 tok/s

99.3% of per-step time is inside `BatchGenerator.next()` — the same mlx-lm
call mlx-vlm also makes. Our rapid-mlx-specific Python wrapper code
(`process_resp` + `schedule` + `snapshot` + `aborts` + `periodic` +
`other`) is **under 1% combined**. So the gap to mlx-vlm is **not in
our wrapper code** either; it lives inside `BatchGenerator.next()`.

### Where inside `BatchGenerator.next()` — sampler cost isolated

Re-ran the same probe with `temperature=0`, `top_p=1.0` (greedy), so
the sampler degenerates to argmax:

Qwen 3.6 35B-A3B 4-bit, greedy::

    bg_next       : avg= 9.690 ms (99.2%)
    TOTAL         : avg= 9.769 ms      -> 102.4 tok/s engine
    end-to-end HTTP: 92.4 tok/s

Qwen 3.5 4B 4-bit, greedy::

    bg_next       : avg= 5.786 ms (99.0%)
    TOTAL         : avg= 5.847 ms      -> 171.0 tok/s engine
    end-to-end HTTP: 165 tok/s (vs 128 with sampling)

Summary (HTTP, B=1):

| Model | sampled tok/s | greedy tok/s | sampler cost / token |
|---|---|---|---|
| qwen3.5-4b 4-bit | 128.5 | 165 | ~1.2 ms |
| qwen3.6-35b-a3b 4-bit | 64.9 | 92.4 | ~4.6 ms |

The greedy 92.4 tok/s on Qwen 3.6 35B-A3B matches mlx-vlm's reported
91.6 tok/s within noise. The earlier "rapid-mlx is 30% behind mlx-vlm"
framing was driven by **default sampling overhead in `BatchGenerator`'s
full softmax + top_p mask + cumulative sort**, which mlx-vlm avoids
when its `_greedy_argmax_step` fast path
(`mlx_vlm/generate/ar.py:886-914`) detects neutral sampler knobs.

## Final conclusion

The B=1 gap to mlx-vlm has three components, in order of size:

1. **Sampler cost** (4.6 ms / token on 35B-A3B, 1.2 ms on 4B): the
   sampler runs full softmax + top_p cumulative when sampling
   parameters are neutral but not explicitly identified as such. **This
   is the only addressable bottleneck.** A `_greedy_argmax_step`-style
   shortcut that skips the full softmax when the sampler is detected
   as degenerate (temp=0 OR top_p ≥ 0.999 with other knobs neutral)
   would close it. mlx-vlm has this; mlx-lm and rapid-mlx do not.
2. **rapid-mlx wrapper Python overhead** (~0.1 ms total): scheduler
   bookkeeping + collector dispatch + asyncio loop. Combined, less
   than 1%. Not worth touching.
3. **asyncio dispatch** (~0.1 ms): falsified hypothesis from earlier
   today. Not the bottleneck.

## Phase 3 — greedy logsumexp-skip experiment (also same day)

Followed Phase 2 with an implementation of the suspected fix:
``_install_greedy_skip_fastpath`` in ``scheduler.py``, which detects
all-greedy sampler batches and monkey-patches ``GenerationBatch._step``
to skip the ``logprobs = logits - mx.logsumexp(logits, axis=-1)``
normalisation. Argmax is invariant under additive shifts along the
vocab axis, so the sampled token is byte-identical.

### Empirical result: the patch is a no-op

Qwen 3.6 35B-A3B 4-bit, B=1, greedy (temp=0, top_p=1):

| Variant | bg_next ms | HTTP tok/s |
|---|---|---|
| no patch | 9.690 | 92.39 |
| with `RAPID_MLX_GREEDY_SKIP_FAST_PATH=1` patch installed | 9.838 | 91.77 |

Within noise. The patch fired (`[greedy_skip_fastpath] installed`
appears in the log, and the `_greedy_step` path branched correctly),
but the saving wasn't measurable.

### Why the patch doesn't help

The Phase 2 numbers had me chasing the wrong op. The 4.6 ms gap
between sampled and greedy at the BatchGenerator level **was never in
``mx.logsumexp``**. It is in ``mx.random.categorical`` (and other
sampler-chain ops) that mlx-lm already short-circuits when
``make_sampler(temp=0)`` is called:

```python
# mlx-lm sample_utils.py:46
def make_sampler(temp=0.0, top_p=0.0, ...):
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)
    ...  # builds the categorical chain only when temp != 0
```

So mlx-lm's `_step` at `temp=0` already runs the cheapest possible
sampler — argmax on logprobs (which is equivalent to argmax on logits).
The wasted ``mx.logsumexp`` is presumably (a) cheap on its own at the
M3 Ultra vocab dims (the GPU runs it in parallel with downstream ops
via MLX's lazy graph), or (b) elided by MLX's eval-time fusion. Either
way: removing it doesn't move the wall.

### Cross-check: where the 4.6 ms gap really lives

Re-ran Phase 1 with `temp=0.7, top_p=1.0` (sampling with the most
trivial possible chain — no `apply_top_p`, no `min_p`, no `top_k`):

```
qwen3.6-35b 4-bit, B=1, temp=0.7, top_p=1.0:
  bg_next: 14.139 ms  (essentially identical to default sampling 14.302 ms)
```

Setting `top_p=1.0` skips `apply_top_p` entirely (the mlx-lm sampler
chain becomes a single `categorical_sampling(logits, 0.7) =
mx.random.categorical(logits * (1/0.7))`). Yet bg_next stays at
~14.14 ms.

So the cost is **``mx.random.categorical`` itself** — the 150k-vocab
Gumbel-trick categorical sample on Apple MTL. Not amortisable by
anything we can do in our layer.

### Decision

- **Revert the greedy fast path** (not shipped in this PR).
  ``scheduler.py`` returns to its pre-Phase-3 state; only the Phase 1
  + 2 probes remain.
- **Re-frame the "B=1 ~30% gap"**: at greedy we already have parity
  with mlx-vlm (92.4 tok/s vs 91.6). The gap to mlx-vlm under
  sampling is **unverified** — the original investigation cited 91.6
  tok/s "greedy or top-p 0.95 — both ~same", but Phase 3 leaves us
  doubting whether the top-p number was correct. mlx-vlm uses the same
  mlx-lm sample_utils internally; there's no architectural reason it
  would be faster at top-p 0.95 than we are.
- **Follow-up to validate** (not this PR): bench mlx-vlm directly at
  `temp=0.7, top_p=0.95` on the same model + hardware. If mlx-vlm is
  also ~65 tok/s there, we have parity end-to-end and the "B=1 30%
  gap" framing was an artefact of comparing rapid-mlx-default-sampling
  vs mlx-vlm-greedy. If mlx-vlm is still 91 tok/s there, the gap is
  inside their forward pass (model code differences) and we go look
  at that.

The probes are the durable artefact. They take seconds to re-run
whenever a new perf hypothesis appears, and have already disproven
two of them in one day.

