# Dense 27B hybrid vision: concurrent text qualification

Issue #4324. Mac Studio M3 Ultra, 256 GB; Python 3.12.13, vision runtime
0.7.2, text runtime 0.31.3. Checkpoint `mlx-community/Qwen3.8-27B-4bit`, revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; baseline commit
`dfaa9cc407a0abb80dd0b5aa0afd46c6d249c47b`.

## Behavior and scope

The vision-capable model keeps one loaded set of weights. Its qualified dense
27B language backbone now uses the existing shared-weight native-cache text
scheduler for pure-text requests. Media remains on the serialized vision
scheduler. Both schedulers use the model-owning executor and preserve separate
MRoPE state. Exact architecture and layer-geometry checks fail closed for other
hybrid backbones. Requested speculative decoding and `--no-hybrid` retain their
existing behavior; the default speculative alias route is unchanged.

## HTTP measurement

SSE `/v1/chat/completions`, greedy, thinking disabled, 300 tokens per request,
`ignore_eos=true`, B simultaneous clients, two repetitions. Prefix and disk
caches disabled, all three requested batch limits 4. Aggregate throughput is
actual completion tokens divided by group wall time. TTFT starts before the
HTTP request and stops at the first content/reasoning delta, excluding role
and keepalive events. Each phase had an 8-token warmup.

| Policy | B=1 tok/s | B=2 aggregate tok/s | B=4 aggregate tok/s | Slowest TTFT at B=4 |
| --- | --- | --- | --- | --- |
| Serialized vision baseline | 29.78 / 28.92 | 29.23 / 31.13 | 29.57 / 29.61 | 30.64 / 30.45 s |
| Shared-weight native text | 23.39 / 23.25 | 50.33 / 49.60 | 69.62 / 66.57 | 0.57 / 0.60 s |

B=4 median aggregate throughput improved about **2.30×**; the slowest first
content delta fell from about 30.5 seconds to about 0.59 seconds. **Single-client
throughput decreased about 20.6%** relative to the serialized singleton fast
path. This qualification targets multi-client throughput and queue latency;
it does not claim a single-client speedup. Operators can retain the serialized
path with `--mllm --no-hybrid --no-spec-decode`.

These are measurements on a shared development host, not an isolated hardware
maximum. Other work may affect absolute timing. Only one benchmark server was
sent requests at a time; the candidate phase preceded the baseline phase.
Earlier contention-heavy exploratory samples were discarded as a separate
experiment, not mixed into this table. Both repetitions of each final cell
are retained in the [machine-readable receipt](../../benchmarks/results/2026-10-10-hybrid-vision-text-batching.json).

## Reproduction

Use the same runtime and cached checkpoint revision for both revisions:

```bash
python -m rapid_mlx.cli serve mlx-community/Qwen3.8-27B-4bit \
  --no-spec-decode --disable-prefix-cache --disable-disk-caches \
  --max-num-seqs 4 --prefill-batch-size 4 --completion-batch-size 4
mkdir -p /private/tmp/hybrid-vision-text-batching-bench
python scripts/benchmark_hybrid_vision_text_batching.py \
  --model mlx-community/Qwen3.8-27B-4bit --tokens 300 --reps 2 \
  --output /private/tmp/hybrid-vision-text-batching-bench/result.json
```

Run the candidate first, then the baseline at the commit above,
keeping other GPU inference idle to match the recorded phase order. The recorded baseline
used the same candidate dependency environment with native-text startup
qualification disabled to preserve the exact pre-change serialized policy.
The probe requires final usage and exactly the requested token count; incomplete
or error streams fail instead of producing throughput numbers.

## Correctness and lifecycle

Real HTTP dogfood passed streaming and nonstreaming text, image/text
concurrency (a 224×224 red image recognized as red while a text request returned
81), text before/after media (56), three mid-stream disconnect/recovery cycles,
concurrent requests with different token limits (8/48/16/32), a 3035-token prompt recalling
731, and Anthropic nonstreaming compatibility. The server remained healthy.
Different batching/cache implementations can change greedy wording; no
cross-lane bit-exactness claim is made.

Direct-engine dogfood additionally verified shared language-model object identity,
and four concurrent requests actually completing at 8/16/32/64 tokens while
retaining their individual code numbers (731–734). Each prompt asked to begin
with its code and then count every integer from 1 to 100 without abbreviation;
greedy generation with thinking disabled reached each configured limit.

With prefix caching enabled and `hybrid_cache_entries=8`, two consecutive groups
of four requests shared a 700-token prompt (60 repeats of a forest/ocean sentence,
code 731, and a distinct request suffix). All eight answers recalled 731. Cold
requests reused zero tokens; all four warm requests reused 672 tokens each.
Scheduler counters recorded four hits, 2688 tokens saved, zero non-trimmable
skips, and eight bounded cache entries after the warm group. These checks use
`BatchedEngine(..., force_mllm=True, no_spec_decode=True)` with all three batch
limits 4 and `SchedulerConfig(enable_prefix_cache=True, hybrid_cache_entries=8)`;
they are correctness evidence, separate from the cache-disabled HTTP timings.

Focused regression tests cover qualified startup, operator/speculative overrides,
malformed geometry, layer layout, shared model/executor identity, cache namespace
contracts, lane-local MRoPE restoration, fallback, cancellation, and shutdown.
Disabling the new dense eligibility helper makes
`test_mllm_start_activates_native_text_lane_only_after_qualification[dense27b]`
fail because the native scheduler never activates.
