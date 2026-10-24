# Restored hybrid boundary retention

When a non-trimmable cache restores almost the entire prompt, saving another
prompt or completion entry can displace the boundary needed by another session.
An exact match to the longer entry cannot restore the shorter recurrent state.
Apply the existing boundary-retention policy to a restored, live marked entry
after prefill split planning, provided the pending tail is at most 64 tokens,
the current caller boundary is at or after the restored position, and there is
no useful new boundary ahead of it.

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
- Baseline engine `dfaa9cc407a0abb80dd0b5aa0afd46c6d249c47b`.
- Candidate HTTP forward source `9b906fd0a4b29ce5b1245cacd035af14be8c8800`;
  reversed source `32c670b7e6821297e0d3765ec77eb6ea5e0dfa1e` differs only
  by test formatting. Both include the atomic retention decision and narrow
  caller-boundary guard.
- Final implementation `ce1b7299b0706a0b2a99e7517d6b81e24781339c` adds two
  local aliases for dynamically attached request metadata. Independent review
  verified that removing only those identity aliases produces the same scheduler
  AST as the measured revision. Final focused tests and type checks ran on this
  implementation; the HTTP timings retain their actual source revisions.
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
| Solo A TTFT (s) | 0.245 | 0.238 | 0.247 | 0.238 |
| Solo B TTFT (s) | 9.527 | 0.236 | 9.533 | 0.237 |
| Cohort A TTFT (s) | 33.201 | 4.826 | 33.120 | 4.828 |
| Cohort B TTFT (s) | 5.405 | 4.826 | 5.398 | 4.828 |
| Cohort C TTFT (s) | 33.201 | 24.107 | 33.121 | 24.085 |
| Cohort D TTFT (s) | 33.201 | 24.108 | 33.120 | 24.086 |
| Cohort makespan (s) | 33.779 | 24.558 | 33.703 | 24.534 |

Baseline solo B missed in all eight sets; the candidate's solo A/B and cohort
A/B each reused 32,736 tokens in all eight sets. Every measured long response
finished with 32 output tokens and a complete terminal stream, with identical
prompt/completion/total usage across paired requests. Both candidate passes
retained full text: all 64 responses returned their document's code within the
first 100 characters. The 32 paired seed/solo output hashes matched exactly.
Cohort hashes matched in 10/16 and 12/16 comparisons respectively; changing cache
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
output hashes matched each candidate pass. All 16 cohort hashes, including the
cold lanes, matched in each comparison. These controls support the narrower
cache-residency explanation without establishing general losslessness.

Two earlier fetch-time prototypes failed to improve actual HTTP cache hits;
their results are excluded from the table. A successful intermediate prototype
was also excluded because the table qualifies the recorded runtime revision.
Independent review found and fixed eviction after planning, cached-insert retry
handling and a concurrent-eviction window before the reported experiments.
The final M5 scheduler/cache suite passed 159 tests; removing the restored-boundary
arm made all four targeted retention/alignment regressions fail. Replacing the
atomic implementation with the earlier split-lock implementation made the
deterministic concurrent-eviction regression fail. The pinned Python 3.11 type
budget passed without growth or new suppressions. Focused coverage exercised all
44 changed executable scheduler lines (100%), including missing restored metadata
and an unavailable cache backend. These focused tests and HTTP
experiments are distinct from the repository's complete unit and CI gates.

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

These results qualify one hybrid model and one M5 configuration. M4 contract
tests do not establish an M4 speedup. No M3
performance claim follows. Cohort outputs may change when cold/warm residency
changes batch execution; report hash disagreement rather than calling the
optimization numerically lossless. The secret-code task is a narrow correctness
check, not a general model-quality evaluation.

## M3 Ultra integration rerun (2026-10-23)

The branch was rebased onto `f9899bf9b`. The rejected-cache retry now clears
restored retention metadata before the existing cold-prefill cohort admission
check. A regression covers both immediate retry and deferral followed by retry.
The focused scheduler, snapshot, pressure and Metal admission suites pass 327
cases on CPU, with all 44 changed executable scheduler lines covered.

The native small-model experiment compares that main revision with scheduler
source `741158ac8cde9272a551297fda800fbcbad1e618`. Later changes in this rerun
are tests and this evidence document only. Hardware is M3 Ultra, 256 GiB, macOS
26.5.2; Python 3.12.13, MLX 0.32.4, mlx-lm 0.31.3, Transformers 5.15.1.
Weights are the existing `mlx-community/LFM2.5-1.2B-Instruct-4bit` snapshot
`dee2f8a2786e6648bb644a7ca40652842490034b`; no download was needed.

This is a native scheduler experiment, not an HTTP or semantic-answer test.
Create two distinct 2,048-token documents by encoding 500 repetitions of
`Document A: river stone tree water light ` and
`Document B: green metal wind forest lake `, then truncating each to 2,048
tokens. Append exactly 1, 16 or 64 tokens from repeated encodings of
` Please repeat the document name.` Set the caller boundary to 2,048.
Prefill each boundary with the real model in 512-token chunks and store the
resulting native cache with `message_boundary=True`. The state size is
25,247,744 bytes per document. Pressure budget is 63,119,360 bytes, sufficient
for two entries but not a third; capacity control is 201,981,952 bytes.

Use the real Scheduler with one sequence, prefill batch one, completion batch
one, memory-aware cache, eight hybrid entries and no paged cache. For each
case, run an unmeasured B request to warm kernels, clear, measure a cold B,
clear, seed the two native boundaries, then measure A followed by B. Each
request uses greedy decoding and 16 output tokens with EOS ignored. Preserve
output token IDs/text, finish reason, cached tokens, TTFT, total request time,
entry count, ledger bytes and A/B residency. Synthetic truncated prompts are
for cache-state comparison and do not establish answer quality.

Run 20 fresh-process captures per arm in alternating base/candidate then
candidate/base order, forming ten ABBA blocks. Hold the large-model lock over
the entire experiment. Compare log total-request-time ratios using the existing
ten-block one-sided 95% Student-t critical value 1.8331129326536335 and unchanged
5% regression limit. Do not pool the three tails or pressure/capacity cases.
Other CPU-heavy processes were observed during this run; the model lock does
not establish a quiet host or attribute timing variation to this patch.

All 40 captures completed. The candidate retained both boundaries after A in
60/60 pressure cases; baseline did so in 0/60. B reused 2,048 tokens in all
candidate pressure cases versus zero on baseline. All 720 measured responses
finished with 16 tokens. All 120 cold-control pairs and 120 capacity-control
warm pairs matched native output tokens. Pressure pairs matched 100/120:
B with a 64-token tail differs between cold baseline and warm candidate in all
20 captures. All 120 candidate pressure outputs match the naturally warm
baseline capacity control. This supports a residency/execution explanation;
it does not establish numerical losslessness or semantic answer correctness.

| Tail tokens | Pressure B median TTFT, base → candidate | Pressure A request-time 95% bounds | Capacity A upper bound | Cold upper bounds, two controls |
| --- | --- | --- | --- | --- |
| 1 | 255.99 → 6.95 ms | +6.13% to +9.91% (FAIL) | +5.70% (INCONCLUSIVE) | +0.69%, +2.10% |
| 16 | 264.76 → 15.98 ms | +3.73% to +9.45% (INCONCLUSIVE) | +5.14% (INCONCLUSIVE) | +1.36%, +1.10% |
| 64 | 269.56 → 20.57 ms | -4.53% to +4.57% (PASS) | +2.37% (PASS) | +1.74%, +1.59% |

Pressure A median total request times were 37.62 → 40.49 ms, 50.06 → 52.56 ms,
and 55.66 → 57.13 ms respectively. All capacity B upper bounds were below
+1.20%. The 5% limit and 20 captures per arm were not relaxed. The measured
FAIL and INCONCLUSIVE results remain recorded despite possible contention.
This small-model experiment does not replace the prior full HTTP/model/chip
matrix and does not clear its qualification hold. Keep the PR draft and open;
#4451 remains the tracker for a quiet-window diagnosis and qualification.

Raw receipts are named `4459-paired-00-baseline.log` through
`4459-paired-19-candidate.log`; the harness, driver, analysis and source-binding
manifest are retained with them. The harness SHA-256 is
`12103dd34d6b41857bcc5364dc1e3025553b6b4d24de7f99f3b357d70d39ed58`.
The raw baseline environment field read a shared remote-tracking reference
that advanced during the run. The separate source-binding manifest verifies
all 3,727 archived file blobs against the fixed baseline commit
`f9899bf9b64c03af5b8be0600a4b2786270e190a`; the measured source never changed.

## User-value rerun after removing duplicate lookup (2026-10-23)

Rebased onto main `36b7356ef`. Runtime source `356a3b012` reuses the
already-built boundary key and entry during MRU touch, removing a duplicate
prefix tuple allocation and dictionary lookup while retaining one lock-held
liveness decision. Focused session-prefix and snapshot CPU tests: 145 passed;
changed scheduler coverage: 46/46 executable lines. Lint and formatting pass.

The same native small-hybrid protocol above completed 20 fresh-process captures
per arm in ten ABBA blocks, under the model lock. The baseline archive was
verified blob-for-blob against `36b7356ef`; source labels are fixed commit IDs,
not moving remote references. The cached model snapshot and hardware are the
same. This run used Python 3.12.14, MLX 0.32.2, mlx-lm 0.31.3 and Transformers
5.16.1. Both arms used those versions; differences from the earlier experiment
preclude attributing between-experiment timing changes solely to this edit.
Other CPU-heavy test workers overlapped the run. This is not a verified quiet
window, and timing causality is unresolved.

| Tail | Pressure B median TTFT, base → candidate | Pressure A elapsed median, base → candidate | Pressure A one-sided 95% upper bound |
| --- | --- | --- | --- |
| 1 | 257.41 → 7.07 ms | 39.14 → 40.68 ms | +8.75% (INCONCLUSIVE) |
| 16 | 266.36 → 15.51 ms | 51.84 → 52.99 ms | +6.77% (INCONCLUSIVE) |
| 64 | 271.47 → 20.55 ms | 56.27 → 57.61 ms | +5.55% (INCONCLUSIVE) |

Capacity A upper bounds pass at +0.52%, +0.78%, +3.19%. Capacity cold tail 16
(+5.57%) and capacity B tail 64 (+8.90%) remain inconclusive. The unchanged
5% gate has not passed. No current-head validator or merge authorization is
claimed.

Candidate residency remains 60/60 pressure cases versus 0/60 baseline. All
720 responses finish with 16 output tokens. Cold-control and capacity warm
pairs match 120/120 each. Pressure pairs match 100/120: in this software
configuration B with tail 16 differs between cold and warm execution. All 120
candidate pressure outputs match the naturally warm baseline capacity control.
This remains a synthetic cache-state test, not semantic-answer qualification.

Decision: keep the unique work as a draft. Alternating reused documents near
the hybrid-cache budget gain approximately 250 ms TTFT on the second document
while paying 1.15–1.54 ms on the first; frequency in deployed workloads is
unknown. The earlier long-document evidence saves about 9.3 seconds. Those
workload benefits justify retention work, but do not waive the regression gate
or establish a generic speedup. Next action is quiet-window qualification of
this scoped optimization, not broader cache redesign.

Raw captures use the `value1-4459-paired-` prefix; the harness, fixed-source
driver, analysis, coverage and host-observation receipts are preserved with
them. No remote branch or unrelated workload was modified by the experiment.
