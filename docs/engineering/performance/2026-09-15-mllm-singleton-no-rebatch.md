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
the tracked 21-case manifest `evals/prompts/qwen36_mllm_runtime.json`
(20 media cases, 4 per category — OCR, grounded control, structured JSON,
accessibility description, two-image comparison — plus one text-only
fallback case protecting the native-text routing boundary, run with
`no_hybrid=True` so the request stays on the measured lane). The primary
deterministic gate is per-case output equality between phases on the same
exact build; manifest checkers (required/forbidden terms, JSON shape)
guard against two equally wrong outputs passing.

```bash
RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_mllm_singleton \
  --model <snapshot-path> --pairs 3 --output <result.json>
RAPID_MLX_MEDIA_ROOT=<repo-root> python -m scripts.bench_qwen36_mllm_singleton \
  --model <snapshot-path> --lifecycle --abort-iterations 50 --pairs 0
```

## Results

See `docs/benchmarks/results/2026-09-15-qwen36-mllm-singleton.json` for the
exact-head paired run (per-case medians, memory, checker passes) and the
lifecycle scenario output.

* Paired exactness: **21/21 cases bit-identical** between `off` and
  `auto` (SHA-256 over full response text, 3 measured passes per case,
  all passes identical within a phase).
* Decode throughput: median **+35%** generation tok/s across the 21 cases
  (per-case ×1.27–×1.49), matching the spike prediction for removing the
  per-token batch extract/re-merge round-trip on the serialized lane.
* TTFT: unchanged within noise (median −1.6%; prefill work is identical).
* Memory: indistinguishable (candidate active bytes slightly lower;
  peak identical).
* Lifecycle: mid-generation cancellation, post-abort recovery byte-match,
  queued concurrency serialization, and a 50-iteration randomized
  abort/recovery soak with byte-exact probe responses.

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
