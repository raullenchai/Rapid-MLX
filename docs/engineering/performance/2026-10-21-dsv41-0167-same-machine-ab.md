# DeepSeek V4.1 Flash 1017: 0.16.7 same-machine decode A/B

Date: 2026-10-21. Owner: Harbor.

## Environment and method

M3 Ultra, 256 GB unified memory; macOS 26.5.2; Python 3.12.14;
MLX 0.32.4; mlx-lm 0.31.3. Model: DeepSeek V4.1 Flash 1017 GPTQ v2,
revision `c521050a` (`c521050aba97719393865adf611ad745c7f4f551`).

Single-stream server defaults, greedy (`temperature=0`), 256 max completion
tokens, one warm-up request and two measured requests per prompt. The five
[checked-in prompts](fixtures/2026-10-21-dsv41-0167-prompts/) contain 686–985
prompt tokens; `p1008` is a historical label, not this prompt's token count.
These are a separate prompt set from the
[draft-depth measurements](2026-10-21-dsv41-draft-depth.md).

The [benchmark](../../../scripts/bench_dsv41_same_machine_ab.py) measures SSE
arrival times. Decode tok/s is `(completion_tokens - 1) / (t_last - t_first)`,
using server-reported completion usage and the first/last nonempty content or
reasoning delta. TTFT runs from request start to the first such delta. The table
reports the arithmetic mean of the two per-request decode rates, excluding
warm-up, prefill and model loading. The harness fails with a nonzero exit if
server-reported completion usage is missing or any measured round has
unmeasurable decode throughput; it never averages fewer rounds than requested.

## Results

The baseline is 0.16.6 from PyPI. The 0.16.7 measurements used release-candidate
commit `95f2215d4`, installed into a fresh environment before the version bump
(its version string still read 0.16.6). They are not a separate measurement of
the published 0.16.7 wheel; commands below reproduce with both published builds.

| Build / depth | p1008 | qa | summarize | code | structured |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.16.6 PyPI, default | 17.6 | 23.7 | 20.4 | 26.0 | 33.9 |
| 0.16.7 candidate, default (depth 2) | 32.6 | 40.1 | 34.3 | 42.7 | 44.9 |
| 0.16.7 candidate, depth 0 (no draft) | 28.1 | 27.8 | 27.7 | 27.9 | 27.6 |
| Default speedup vs 0.16.6 | 1.86x | 1.69x | 1.68x | 1.64x | 1.32x |

Default decode improves by **1.32x–1.86x**, depending on prompt class. TTFT is
approximately 3.7–4.7 s for these prompts. Steady-state process RSS is about
204 GB for all builds; peak system wired memory was 218.7 GB for the candidate
and 218.4 GB for 0.16.6. System free memory bottomed at about 0.1 GB during
loading as file cache filled. One baseline load exited silently; its retry
succeeded. That isolated load failure was not reproduced.

Greedy sanity: depth 2 output was byte-identical to the earlier depth 2 path
before the draft-kernel optimization on all five prompts. Output versus 0.16.6
or depth 0 can differ at near-tie tokens (for example, punctuation): multi-row
verification changes floating-point reduction order. Cross-depth token identity
is not guaranteed. The inspected outputs remained coherent and correct,
including the code diagnosis, JSON and summaries.

## Reproduction

Run from the repository root on an otherwise idle machine with the same model
revision. The alias pins the checkpoint and colocated draft revision above;
retain server logs to confirm it. Use the shared model cache for both builds.
The following creates two fresh Python 3.12 environments, installs from PyPI
rather than the checkout, and records resolved dependencies:

```sh
mkdir -p bench-results
python3.12 -m venv .venv-ab-0166
python3.12 -m venv .venv-ab-0167
.venv-ab-0166/bin/python -m pip install --index-url https://pypi.org/simple \
  'rapid-mlx==0.16.6' 'mlx==0.32.4' 'mlx-lm==0.31.3'
.venv-ab-0167/bin/python -m pip install --index-url https://pypi.org/simple \
  'rapid-mlx==0.16.7' 'mlx==0.32.4' 'mlx-lm==0.31.3'
.venv-ab-0166/bin/python -m pip freeze > bench-results/0166-packages.txt
.venv-ab-0167/bin/python -m pip freeze > bench-results/0167-packages.txt
export RAPID_MLX_TELEMETRY=0 RAPID_MLX_DSV41_LANES=1
unset RAPID_MLX_DSV41_DRAFT_DEPTH
export GPU_LOCK="$(pwd)/bench-results/gpu.lock"
```

In terminal A, start **one** server command below at a time. macOS `lockf` holds
the lock for the server's full lifetime, including load and warm-up. All GPU
jobs on the machine must use this same lock path. Wait for readiness, run its
matching client command in terminal B, then stop the server with Ctrl-C and
wait for exit before starting the next build. Both builds use default depth;
only the third run overrides depth to zero.

```sh
lockf -k "$GPU_LOCK" .venv-ab-0166/bin/rapid-mlx serve \
  deepseek-v41-flash-1017 --host 127.0.0.1 --port 8000
lockf -k "$GPU_LOCK" .venv-ab-0167/bin/rapid-mlx serve \
  deepseek-v41-flash-1017 --host 127.0.0.1 --port 8000
RAPID_MLX_DSV41_DRAFT_DEPTH=0 lockf -k "$GPU_LOCK" \
  .venv-ab-0167/bin/rapid-mlx serve \
  deepseek-v41-flash-1017 --host 127.0.0.1 --port 8000
```

In terminal B, define this function from the same repository root. It copies
prompts into each result directory because the benchmark saves the first
measured output beside its input. JSON lines retain per-round TTFT, usage,
decode interval and throughput. Each round, error and summary line also includes
`_schema_version` and `_methodology_hash` (SHA-256 of the benchmark script).
A failed request returns a nonzero status.

```sh
bench() {
  label="$1"
  mkdir -p "bench-results/$label"
  cp docs/engineering/performance/fixtures/2026-10-21-dsv41-0167-prompts/*.txt \
    "bench-results/$label/"
  for prompt in p1008 qa summarize code structured; do
    .venv-ab-0167/bin/python -m scripts.bench_dsv41_same_machine_ab \
      --base http://127.0.0.1:8000/v1 --model deepseek-v41-flash-1017 \
      --prompt-file "bench-results/$label/$prompt.txt" \
      --label "$label/$prompt" --max-tokens 256 --warmup 1 --rounds 2 \
      > "bench-results/$label/$prompt.jsonl" || return 1
  done
}

# Run each line only while its matching server is ready in terminal A.
bench 0166-default
bench 0167-default-depth2
bench 0167-depth0
```

These commands are reproduction instructions, not additional measured results.
Two rounds establish this same-machine observation, not a confidence interval
or a guarantee for other hardware, contexts or concurrent workloads.
