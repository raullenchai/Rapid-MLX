# Restored hybrid boundary retention

When a non-trimmable cache restores almost the entire prompt, saving another
prompt or completion entry can displace the boundary needed by another session.
An exact match to the longer entry cannot restore the shorter recurrent state.
Apply the existing boundary-retention policy to a restored, live marked entry
after prefill split planning, provided the pending tail is at most 64 tokens
and there is no useful new boundary ahead of it.

The scheduler still forwards the same cache and pending tokens. A missing
boundary falls back to the existing cache-write policy. A rejected cached insert
clears the restored retention state and rearms cold boundary capture. Requests
without a valid caller boundary, dense caches, longer tails, and meaningful new
boundaries retain their existing behavior. A persisted entry's marker alone does
not establish its original provenance; eligibility also requires the current
request's valid caller boundary.

## Experimental environment

- M5 Max, 36 GiB unified memory, macOS 27.0.1; no concurrent inference service.
- Python 3.12.13, MLX 0.32.3, mlx-lm 0.31.3, mlx-vlm 0.7.2,
  transformers 5.15.1.
- `mlx-community/Qwen3.6-35B-A3B-4bit`, immutable snapshot
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`.
- Baseline engine `dfaa9cc407a0abb80dd0b5aa0afd46c6d249c47b`; candidate runtime
  `0f4bee5fbb184e925355c57ae253e47ce39ab519`.
- Ordinary text HTTP route, FCFS, four sequences, completion batch four,
  prefill step 2048, decode stall target 500 ms, eight hybrid entries.
- Prefix budget 2 GiB; additional unmodified-server capacity control at 4 GiB.
  Checkpoints, blocked prefill, fused decode, compiled decode, lane matmul,
  prompt host cache, disk caches and speculative decoding disabled.

For seeds 9201–9204, generate independent A/B/C/D documents using
`random.Random(str(seed) + name)` and 32,700 choices from
`river stone tree water light metal wind green`. Wrap each document as one user
message, naming a distinct secret code at its beginning and asking to repeat
that code and explain its location. The actual rendered prompt is 32,752 tokens.
Clear the prefix cache between seeds. Seed A/B, repeat A/B individually, then
release A/B/C/D requests through one barrier. Use temperature zero, thinking
disabled and at most 32 output tokens. Record every stream's terminal marker,
finish reason, usage, output hash and cached-token count, plus server cache
status before/after each cohort. Do not discard early stops or failures.

Two seed groups submit warm threads first and two submit cold threads first;
the barrier does not guarantee network arrival or scheduler execution order.
Repeat the baseline and candidate with reversed seed order. Run a separate
unmodified-server control that skips individual repeats to obtain naturally
warm cohorts before duplicate writes can change their cache residency.

## Results

Each cell is the median of four document sets. Forward and reversed seed orders
are reported separately; the thread-submission groups do not guarantee actual
arrival order.

| Measurement | Baseline, forward | Candidate, forward | Baseline, reversed | Candidate, reversed |
| --- | ---: | ---: | ---: | ---: |
| Solo A TTFT (s) | 0.245 | 0.239 | 0.247 | 0.240 |
| Solo B TTFT (s) | 9.527 | 0.239 | 9.533 | 0.238 |
| Cohort A TTFT (s) | 33.201 | 4.855 | 33.120 | 4.868 |
| Cohort B TTFT (s) | 5.405 | 4.855 | 5.398 | 4.869 |
| Cohort C TTFT (s) | 33.201 | 24.181 | 33.121 | 24.159 |
| Cohort D TTFT (s) | 33.201 | 24.181 | 33.120 | 24.160 |
| Cohort makespan (s) | 33.779 | 24.629 | 33.703 | 24.609 |

Baseline solo B missed in all eight sets; the candidate's solo A/B and cohort
A/B each reused 32,736 tokens in all eight sets. Every measured long response
finished with 32 output tokens and a complete terminal stream, with identical
prompt/completion/total usage across paired requests. Both candidate passes
retained full text: all 64 responses returned their document's code within the
first 100 characters. The 32 paired seed/solo output hashes matched exactly.
Cohort hashes matched in 11/16 and 13/16 comparisons respectively; changing cache
residency changes batch execution, so these results do not claim numerical
losslessness. The reversed baseline also retained text and answered all 32 codes
correctly; the original baseline retained hashes rather than full text.

The unmodified 4 GiB capacity control let both solo repeats hit, at medians
0.251/0.246 seconds. It accumulated five entries (3,506 MiB) before the cohort;
duplicate whole-prompt/completion writes displaced useful boundaries and cold
prefill reclamation removed additional entries. All four cohort requests missed,
with a median 41.980-second makespan. Increasing the byte budget alone therefore
did not reproduce the candidate's cache residency in this workload.

The unmodified 2 GiB native warm control skipped individual repeats, retaining
both seeded boundaries until the cohort. Its median makespan was 24.666 seconds;
all 24 measured responses returned the correct code. All eight warm cohort
output hashes matched each candidate pass. Cohort totals matched 15/16 in each
comparison, with one cold-lane disagreement per pass; these controls support
the narrower cache-residency explanation without establishing losslessness.

Two earlier fetch-time prototypes failed to improve actual HTTP cache hits;
their results are excluded from the table. A successful intermediate prototype
was also excluded because the table qualifies the final runtime revision.
Independent review found and fixed eviction after planning and cached-insert
retry handling before the final experiments. The focused M4 scheduler/cache
suite passed 155 tests; removing the restored-boundary arm made all four targeted
retention/alignment regressions fail.

## Reproduction

Use an already available immutable snapshot in the default model cache. In each
revision's checkout, launch the same server configuration with the existing
controlled-environment helper:

```python
import subprocess
import sys
from pathlib import Path
from scripts.benchmark_hybrid_checkpoints import child_environment, server_command

model = Path.home() / (
    ".cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/"
    "snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
)
assert model.is_dir()
command = server_command(sys.executable, str(model), 8649) + [
    "--max-num-seqs", "4", "--completion-batch-size", "4",
    "--decode-stall-target-ms", "500", "--scheduling-policy", "fcfs",
]
subprocess.run(command, env=child_environment(0, 0), check=True)
```

After `/health` reports ready, run this client from a separate terminal:

```python
import concurrent.futures
import json
import random
import threading
import time
import httpx
from scripts.benchmark_hybrid_checkpoints import stream_receipt

base = "http://127.0.0.1:8649"

def ask(messages, barrier=None):
    with httpx.Client(timeout=240, trust_env=False) as client:
        if barrier is not None:
            barrier.wait(timeout=30)
        start = time.perf_counter()
        with client.stream("POST", base + "/v1/chat/completions", json={
            "model": "m5-qualification", "messages": messages,
            "temperature": 0, "max_tokens": 32, "enable_thinking": False,
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            response.raise_for_status()
            return stream_receipt(response.iter_lines(), start,
                                  require_full_budget=False)

ask([{"role": "user", "content": "Say READY."}])
with httpx.Client(timeout=10, trust_env=False) as client:
    for seed in [9201, 9202, 9203, 9204]:  # Reverse this for the second pass.
        client.post(base + "/v1/cache/clear").raise_for_status()
        docs = {}
        for name in "ABCD":
            rng = random.Random(str(seed) + name)
            text = " ".join(rng.choice([
                "river", "stone", "tree", "water", "light", "metal", "wind", "green"
            ]) for _ in range(32700))
            docs[name] = [{"role": "user", "content":
                f"Document {seed}{name}. Secret code: {seed}{name}.\n" + text +
                f"\nState the secret code for document {seed}{name}, then explain how you found it."
            }]
        for phase in ["seed", "solo"]:  # Omit solo for the native warm control.
            for name in "AB":
                print(json.dumps(dict(seed=seed, phase=phase, lane=name,
                                      **ask(docs[name]))), flush=True)
        print(client.get(base + "/v1/status").json())
        names = "ABCD" if seed in [9201, 9204] else "CDAB"
        barrier = threading.Barrier(4)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(ask, docs[name], barrier): name for name in names}
            for job in concurrent.futures.as_completed(jobs):
                print(json.dumps(dict(seed=seed, phase="cohort", lane=jobs[job],
                                      **job.result())), flush=True)
        print("cohort makespan", time.perf_counter() - start)
        print(client.get(base + "/v1/status").json())
```

For the capacity control, change only
`RAPID_MLX_PREFIX_CACHE_MAX_BYTES` to `4294967296` in the child environment.
Keep the full command, dependency versions, source revision, model snapshot and
receipt files with each run. The HTTP output hash covers concatenated content
and reasoning fields, not raw token IDs. Full response text must additionally
be retained when checking the secret-code task.

## Limits

These results qualify one hybrid model and one M5 configuration. M4 focused
tests validate scheduler contracts but do not establish an M4 speedup. No M3
performance claim follows. Cohort outputs may change when cold/warm residency
changes batch execution; report hash disagreement rather than calling the
optimization numerically lossless. The secret-code task is a narrow correctness
check, not a general model-quality evaluation.
