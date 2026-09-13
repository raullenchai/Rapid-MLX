# Qwen3.6-35B-A3B native text-cache qualification

Date: 2026-09-13

Target: `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`

Host: Mac Studio, Apple M3 Ultra, 256 GB unified memory

OS: macOS 26.5.2 (25F84)

Runtime: MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17

Measured candidate: `8060a79a2`, based on `531317fb7`. The final benchmark
was rerun from the same committed implementation and benchmark harness.

## Decision

Use the standard text scheduler and its native cache containers for text-only
requests to the qualified vision-capable Qwen3.6 checkpoint. Media requests
remain on the existing multimodal scheduler. Both schedulers share the exact
same loaded language model and single MLX executor, so the optimization does
not load or copy a second set of weights.

Enrollment fails closed on the exact qualified model geometry: 40 layers,
30 linear-attention layers, 10 full-attention layers, 256 experts with top-8
routing, and the measured GatedDeltaNet head dimensions. Speculative decoding
and unknown layouts remain on the existing path. The existing `--no-hybrid`
operator override retains the prior multimodal-only path.

The vision language module retains MRoPE position state on the model rather
than in each cache. The wrapper therefore swaps and restores lane-local state
around each complete forward call. Both scheduler loops use the same
single-thread executor, making this state boundary atomic with model execution.

## Performance

The checkpoint was loaded once. Prefix caching was disabled. One warmup per
lane was excluded, then six adjacent baseline/candidate pairs alternated order.
Each pair used the same greedy, thinking-off coding request. The response text
was byte-identical in all six pairs.

| Metric | Existing MLLM cache | Native text cache |
| --- | ---: | ---: |
| Median decode | 72.86 tok/s | 107.63 tok/s |
| Observed range | 72.31–78.99 tok/s | 106.76–108.81 tok/s |
| Positive adjacent pairs |  | 6 / 6 |

The median paired speedup was **1.482x (+48.2%)**.

Startup active memory was 20.403 GB. It was 20.403 GB after the baseline
warmup and 20.420 GB after the native-cache warmup, a 17 MB increase rather
than another model-sized allocation. The complete performance, quality, and
vision campaign peaked at 21.847 GB active memory.

## Quality and lifecycle gates

Exact wording is not a valid universal quality requirement because the two
cache implementations can change low-order floating-point results. The gate
therefore combines exact checks where the contract is structural with scored
behavior elsewhere:

- a compact Python task produced the same parseable O(n) implementation and
  three asserts on both lanes;
- compact JSON and the requested tool call were byte-identical;
- arithmetic reasoning reached the correct 0% conclusion on both lanes;
- both creative-writing responses contained the requested setting, ice, and
  music, although their wording differed;
- one deterministic question from each of 57 MMLU subjects produced identical
  answer choices on both lanes: 50/57 correct, zero candidate regressions and
  zero candidate-only fixes.

A real screenshot request stayed on the media scheduler and read its main
heading as `Share Compute`. The immediately following text request returned
the required compact JSON on the native-cache lane. A separate same-process
dogfood run submitted one image request and one text request concurrently;
both completed correctly. A long text stream was then cancelled after four
chunks, the abort was accepted, both schedulers returned to zero running and
waiting requests, and the next text request succeeded.

The two lanes expose separate cache statistics, while aggregate lifecycle and
request counters include both. Pause, resume, cancellation, cache clearing,
and shutdown operate on both schedulers. Candidate-start failure is optional:
cleanup is best-effort and the already-running multimodal scheduler remains
authoritative.

## Rejected alternatives

- A separately loaded text model reached similar throughput but duplicates
  model ownership and memory, so it is not a product path.
- Sharing the model with the standard scheduler while retaining the vision
  cache containers stayed near 69 tok/s. Changing the scheduler alone was not
  the source of the gain.
- Replacing the depthwise convolution, removing the single-row cache repack,
  or replacing full attention did not help; the latter regressed to about
  52 tok/s.
- Replacing only the linear-attention caches inside the multimodal scheduler
  reached about 75 tok/s. It preserved the media execution shape but captured
  only a small fraction of the available improvement.

## Reproduction

Resolve the already-cached immutable checkpoint and dataset snapshots, then
run:

```bash
RAPID_MLX_MEDIA_ROOT=/path/to/media-root PYTHONPATH=. \
python3.12 scripts/large-model-run.py \
  --working-set-gb 32 --reserve-gb 12 -- \
  python3.12 scripts/benchmark_qwen36_native_text_cache.py \
    --model /path/to/qwen36/snapshot \
    --pairs 6 --max-tokens 256 --quality-max-tokens 512 \
    --mmlu-path /path/to/mmlu/all/test.parquet --mmlu-samples 57 \
    --image-path /path/to/screenshot.jpg --image-expect 'Share Compute'
```

The harness exits nonzero unless both lanes pass all five behavioral checks,
the candidate has no MMLU regression, at least five of six adjacent pairs
improve, the paired median is at least 1.30x, and the optional image plus
text-recovery checks pass. Absolute throughput should be measured on an
otherwise idle GPU; the adjacent same-process ratio is the landing metric.
