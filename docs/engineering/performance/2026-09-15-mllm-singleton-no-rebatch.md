# Serialized-MLLM singleton no-rebatch: methodology and interpretation

Date: 2026-09-15 · Machine: Mac Studio M3 Ultra, 256 GB unified memory ·
Model: `mlx-community/Qwen3.6-35B-A3B-4bit`, revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46` (offline resolution, default HF cache)

## What changed

On the serialized hybrid MLLM lane the structural B=1 policy means every
batch holds exactly one request, yet the legacy path still merged that
request's regular cache leaves into batched wrappers, which every decode
step then extracted and re-merged around. The fast path keeps the
prefill-written regular leaves (`cache_layout="singleton_regular"`) when:

* `MLLMSchedulerConfig.allow_arrays_cache` is on and all three batch
  limits are 1 (existing structural gate);
* `_next()` admission has no active batch (queued requests stay queued);
* the flag is `auto` (default; `off` is the operator rollback);
* every cache leaf is an exact regular `KVCache`/`ArraysCache` class —
  unknown, wrapped, subclassed, or quantized shapes fail closed into the
  legacy merge path.

Terminal extraction for singleton-regular batches builds detached leaves
from explicit contiguous copies evaluated on the worker stream; upstream
`ArraysCache.extract` returns lazy slices that would alias live state.

## Harness

`scripts/bench_qwen36_mllm_singleton.py` drives the production rollback
lever (two `BatchedEngine` instances per phase: `off` then `auto`) over
the tracked 22-case manifest `evals/prompts/qwen36_mllm_runtime.json`
(20 media cases, 4 per category — OCR, grounded control, structured JSON,
accessibility description, two-image comparison — plus a long
warm-prefix-capable text case that qualifies APC warm resumes on the
fast path, and one text-only fallback case protecting the native-text
routing boundary, run with `no_hybrid=True` so the request stays on the
measured lane). The primary deterministic gate is per-case output
equality between phases on the same exact build; manifest checkers
(required/forbidden terms, JSON shape) guard against two equally wrong
outputs passing. The harness exits non-zero unless the fast path
engaged and the warm case qualified.

```bash
RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_mllm_singleton \
  --model <snapshot-path> --pairs 3 --output <result.json>
RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_mllm_singleton \
  --model <snapshot-path> --pairs 2 --apc on --output <result-apc.json>
RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_mllm_singleton \
  --model <snapshot-path> --lifecycle --abort-iterations 50 --pairs 0
```

## Results

See `docs/benchmarks/results/2026-09-15-qwen36-mllm-singleton.json`
(schema 2) for the final-head paired runs (per-case medians, memory,
checker passes, singleton batch counts) and the lifecycle scenario
output.

Cold A/B (`--apc off`, 3 measured passes per case):

* Paired exactness: **22/22 cases bit-identical** between `off` and
  `auto` (SHA-256 over full response text, all passes identical within
  a phase); 88 singleton-regular batches counted in the `auto` phase,
  0 in `off`.
* End-to-end latency: median **−11.7%** elapsed across the 22 cases
  (per-case −19.5%…+0.5%) — the per-token batch extract/re-merge
  round-trip is gone on the serialized lane.
* Decode throughput: median **+23.6%** generation tok/s.
* TTFT: unchanged within noise (median −0.9%; prefill work is
  identical).
* Memory: indistinguishable (candidate cache bytes slightly lower;
  peak identical).

APC-warm A/B (`--apc on`, 2 sends per case, second send warm):

* Same paired exactness: 22/22 bit-identical, fast path engaged, 110
  singleton batches in `auto`.
* APC warm resumes interact correctly with the fast path: 4/4
  qualified cases hit the warm cache in both phases
  (204 tokens saved per warm send), and the `auto` phase keeps the
  warm-TTFT win (median TTFT **−41.4%**, median elapsed **−59.1%**
  across all cases including their cold first send).

Lifecycle: mid-generation cancellation, post-abort recovery byte-match,
queued concurrency serialization (26 queued requests processed), and a
50-iteration randomized abort/recovery soak (seed 1729) with byte-exact
probe responses.

## Interpretation and scope

The win is specific to requests that remain on the serialized
`MLLMScheduler` — media requests, and text-only requests when no
qualified native text engine is started. Text-only traffic routed to the
shared-weight native-cache text engine is out of scope and its separate
evidence is not claimed here. Rollback for operators:

```text
rapid-mlx serve --mllm-singleton-fastpath off
```

Regression tripwires live in `tests/test_mllm_singleton_fastpath.py`
(eligibility, merge-path fallbacks, queued admission, extend refusal,
detached extraction).
