# Restored hybrid boundary retention

Latest qualification: the [noise-floor and repeated-control follow-up](#noise-floor-and-repeated-controls-2026-10-24) resolves the recorded performance hold under the owner’s revised acceptance rule. Earlier dispositions below are historical.

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

## Long-document current-main A/B (2026-10-24)

**Disposition: BLOCKED; retain draft.** The requested workload shows a clear
user benefit, so closing it as having no measurable gain would discard useful
work. It does not meet the required identical-output/no-regression decision
rule. No exact-head validator, MERGE-SAFE, readiness or merge is claimed.

Baseline is main `ccc630179f57c8a8dec15327fa70b7a87394216a`; candidate runtime is
`5863e624e09fff4fb6f5196b1a03fe30a7a4b708`. The branch rebased cleanly. Its old
commits were consolidated to remove machine-local author/committer addresses;
the complete candidate Git tree is unchanged from the measured checkout:
`6e671e2f4d13a8c994674d2e95134dc74c652bc8`. Later changes are evidence only.
The baseline was an archive of that fixed main commit, not a moving reference.
Main history, release notes and related merged cache PRs were checked; the
restored-boundary retention mechanism remains unique.

Hardware: M3 Ultra, 256 GiB, macOS 26.5.2. Python 3.12.14, MLX 0.32.4,
mlx-lm 0.31.3, Transformers 5.15.1 and httpx 0.28.1 are identical in both arms.
The already-cached Qwen3.6-35B-A3B-4bit snapshot is the same immutable revision
listed above. No weights were downloaded. Each owned loopback server ran on
port 8736 under the large-model lock and was terminated before the next arm.
The lock was released after the baseline pressure arm when a harness port
probe rejected the just-closed socket; the probe was corrected to allow socket
reuse, and the remaining arms reacquired the lock after another job finished.
No measured request was retried or discarded.

Use the HTTP workload and document generation in Reproduction above, with port
8736. Each arm runs seeds 9201–9204: seed A/B, solo A/B, then barrier-release
A/B/C/D (submission order ABCD for 9201/9204, CDAB for 9202/9203). Each rendered
prompt contains 32,752 tokens. Arm order is baseline 2 GiB, candidate 2 GiB,
candidate 4 GiB, baseline 4 GiB. The cache is cleared between seeds; one short
unmeasured warmup precedes each arm. A submission barrier does not guarantee
network arrival order. Both controls use the same cache-entry limit of eight,
FCFS, four sequences, batch size four, prefill step 2048, temperature zero and
32 output tokens, with the optional kernels/caches disabled as above.

[Auditable receipts](2026-10-24-hybrid-boundary-retention.json.gz) retain every
response's text, reasoning, output hash, complete usage hash, normalized token
usage hash, terminal/finish status and timings, plus cache counters/ledger after
each phase. Complete usage hashes intentionally differ when cached-token
metadata differs; normalized usage covers prompt/completion/total tokens.
All 128 measured responses complete with 32 output tokens, and all 128 include
the correct document code. All 64 paired normalized usage hashes match.

Each timing below is a median across four document sets, in seconds.

| Measurement | Main 2 GiB | Candidate 2 GiB | Main 4 GiB | Candidate 4 GiB |
| --- | ---: | ---: | ---: | ---: |
| Cold A TTFT | 18.232 | 18.567 | 17.962 | 18.483 |
| Cold B TTFT | 18.830 | 18.593 | 18.062 | 18.741 |
| Solo A TTFT | 0.323 | 0.293 | 0.288 | 0.287 |
| Solo B TTFT | 18.622 | 0.390 | 0.369 | 0.380 |
| Solo A elapsed | 0.698 | 0.646 | 0.599 | 0.633 |
| Solo B elapsed | 19.186 | 0.759 | 0.696 | 0.737 |
| Cohort A TTFT | 62.763 | 8.792 | 60.392 | 8.940 |
| Cohort B TTFT | 9.775 | 8.792 | 9.555 | 8.940 |
| Cohort C TTFT | 62.763 | 42.958 | 60.392 | 42.878 |
| Cohort D TTFT | 62.764 | 42.956 | 60.392 | 42.877 |
| Cohort makespan | 63.441 | 43.441 | 60.990 | 43.350 |

At 2 GiB, candidate B solo reuses 32,736 tokens in 4/4 sets versus 0/4 on main;
A solo hits in both. Both candidate cohort A/B reuse 32,736 tokens in all sets,
whereas main reuses only B. At 4 GiB, both solo repeats hit in both arms, but
main still loses cohort A's hit. Solo-phase eviction-counter increments are
3 versus 0 at 2 GiB and 2 versus 0 at 4 GiB in every set. Cohort increments are
5 versus 2 and 3 versus 1 respectively. These counters include restored-entry
removal; they are not exclusively capacity evictions. The ledger remains within
the configured byte budget in every recorded status.

Exact output hashes match 31/32 paired responses at each budget: all eight
cold and eight solo pairs, and 15/16 cohort pairs. In both budgets the mismatch
is cohort D, seed 9203. Both responses correctly identify `9203D`; main's
explanation begins “The secret code is embedded directly within the text of the
document”, while the candidate begins “The secret code is explicitly stated in
the very first sentence of”. Retention changes cache residency and batch
execution; this experiment does not establish numerical losslessness. The
repeated mismatch fails the owner's exact-output condition even though the
narrow answer-quality check passes. Guaranteeing identical decoding across
changed batch/cache paths would require a separate numerical investigation,
outside this PR's retention-only scope and explicit kernel non-goals.

Other CPU test workloads overlapped these arms. The 4 GiB candidate solo elapsed
medians are approximately 5.7% (A) and 5.8% (B) higher, so no regression-free
claim is made. This four-set sequential experiment is not the historical
20-capture paired statistical gate and does not waive its 5% limit. It establishes
the pressure benefit (B TTFT about 98% lower; cohort makespan about 32% lower),
not a quiet-window nonregression pass. Attribution of the control timing
variation remains unresolved.

Focused CPU tests on the rebased runtime: 145 passed. Changed executable
scheduler coverage: 46/46 lines, 100%; Ruff and whitespace checks pass. All
owned servers exited and the lock was released. The draft and #4451 stay open;
Vector owns the numerical-identity investigation and a genuinely quiet paired
nonregression qualification before exact-head validation and green CI can
permit landing. No serving defaults, API, kernels or cache formats changed.

## Noise floor and repeated controls (2026-10-24)

**Performance qualification passes the owner's follow-up decision rule.**
The follow-up replaces the prior strict output-identity decision with a
same-main noise comparison and three repeated larger-cache controls. This
supersedes the preceding BLOCKED performance disposition for this workload;
exact-head validation and green CI remain separate merge gates. No numerical
losslessness or generic cross-model/chip performance claim follows.

### Fixed source and protocol

Baseline: `b2822fd6fa6af8b7c789e333044aa69435d08a83`. Candidate:
`2a236e0cc6be9cbaed451daa371c18c1f1cba28b`. The intervening main changes concern
mirror metadata; the scheduler and benchmark helper are unchanged from the
preceding experiment. Both revisions use the same cached model snapshot,
M3 Ultra 256 GiB, Python 3.12.14, MLX 0.32.4, mlx-lm 0.31.3, Transformers
5.15.1 and httpx 0.28.1. Nothing was downloaded into the model cache.

Repeat the same 32-response arm described above: four document sets, cold
A/B seeds, solo A/B repeats and four-request cohorts, with the same alternating
submission-order groups, 32,752-token prompts and 32-token output cap. A full
main-only pass consists of one 2 GiB and one 4 GiB arm (64 responses); repeat
that pass twice at the same fixed main revision. The two 4 GiB main arms also
serve as the first two timing controls. Add a third main 4 GiB arm and three
candidate 4 GiB arms, for 256 measured responses in total.

Actual arm order: main 2 GiB run 1, main 4 GiB run 1, candidate 4 GiB run 1,
main 2 GiB run 2, candidate 4 GiB run 2, main 4 GiB run 2, main 4 GiB run 3,
candidate 4 GiB run 3. Thus the three timing pairs use main/candidate,
candidate/main, main/candidate order. Each arm starts a fresh owned server and
uses the same short warmup and cache clears. The GPU lock covers the entire
sequence, with loopback port 8736 and no other owned GPU work. Foreign CPU
workers were observed; the timing assessment uses the requested measured
run-to-run spread and does not assert a quiet host.

[Source-bound summary and all receipts](2026-10-24-hybrid-boundary-followup.jsonl.gz)
are losslessly gzip-compressed and contain a summary JSON record followed by
one JSON record per response or cache status. Read them with `gzip -dc`; compression
keeps raw data from exhausting the code-review diff budget. They include full output text, hashes, usage, request IDs, timing samples,
cache ledgers, mismatch identities and the original-response audit. All 256
responses completed with the correct code, 32 output tokens, length finish and
terminal stream marker. Every recorded cache ledger stays within its budget.
All 64 main/main and 96 new main/candidate normalized usage comparisons match.

### Output noise and the two original differences

The two fresh identical-main passes differ in **7/64** output hashes: four
2 GiB cohort responses (9201 A, 9202 C, 9203 D, 9204 D), and three 4 GiB cohort
responses (9202 C/D and 9203 D). All are cold lanes with zero cached tokens in
both runs. Cold-seed and solo-repeat hashes are identical. The original
candidate comparison's **2/64** differences are below this measured noise
floor. New 4 GiB main/candidate comparisons differ in 5/32, 1/32 and 2/32
responses respectively, **8/96** overall, also below the main-only rate.
These are descriptive counts for this workload, not a population confidence
bound or a guarantee that future outputs will match.

Both original differences concern 9203 D and share their first **24 of 32
output tokens**. Both correctly state `9203D` and then describe its presence
near the start of the supplied document. The different explanations are valid
continuations of that answer, not garbage or context substitution. Both have
identical prompt/completion/total usage, the requested length cap and a complete
stream; neither is a transport-truncated response. Fresh main **itself** emits
the candidate's exact variant at 4 GiB run 1 and 2 GiB run 2, and emits the
original main variant in the other corresponding pass.

The requested boundary audit corrects the proposed causal explanation: **D
itself did not restore a different boundary**. Both original cache-fetch logs
show a MISS and all 32,752 tokens scheduled for prefill. Its cohort neighbor A
changes from zero cached tokens on main to 32,736 on the candidate, while B
hits in both. The cohort therefore has different cache residency and batch
execution, but attributing D's difference to its own restored state would be
incorrect. The unchanged-main repeats reproduce both variants without this
patch, including variation with identical per-request cached-token counts.
This supports ordinary batch-dependent numeric variation, and the narrower
noise-level/semantic acceptance rule; it is not evidence of a corrupted
restored boundary or a proof of the precise numerical cause.

### Three larger-cache controls

Each run median is computed across the four document sets. All values below
are seconds; the spread column is the minimum–maximum of the three run
medians. The comparison uses the median of those three medians. No failed or
slow request was excluded. The summary field
`no_median_regression_beyond_observed_run_range` is true when the signed median
increase does not exceed the observed range; decreases are improvements rather
than regressions. It is not an absolute-change or output-identity predicate.

| Measurement | Main run medians | Candidate run medians | Main median [spread] | Candidate median [spread] |
| --- | --- | --- | --- | --- |
| Cold A elapsed | 18.576, 18.387, 18.395 | 18.595, 18.424, 18.319 | 18.395 [18.387–18.576] | 18.424 [18.319–18.595] |
| Cold B elapsed | 18.902, 18.432, 18.590 | 18.659, 18.437, 18.439 | 18.590 [18.432–18.902] | 18.439 [18.437–18.659] |
| Solo A TTFT | 0.296, 0.300, 0.304 | 0.292, 0.279, 0.284 | 0.300 [0.296–0.304] | 0.284 [0.279–0.292] |
| Solo B TTFT | 0.390, 0.372, 0.405 | 0.385, 0.374, 0.384 | 0.390 [0.372–0.405] | 0.384 [0.374–0.385] |
| Solo A elapsed | 0.612, 0.624, 0.630 | 0.621, 0.616, 0.607 | 0.624 [0.612–0.630] | 0.616 [0.607–0.621] |
| Solo B elapsed | 0.736, 0.708, 0.742 | 0.741, 0.664, 0.718 | 0.736 [0.708–0.742] | 0.718 [0.664–0.741] |
| Cohort makespan | 61.474, 61.001, 61.018 | 43.116, 42.474, 42.437 | 61.018 [61.001–61.474] | 42.474 [42.437–43.116] |

The alleged larger-cache warm elapsed regression does not reproduce: candidate
medians are **1.2% lower for A and 2.4% lower for B**, with overlapping run
ranges. The small cold A elapsed increase (+0.16%) is within the observed
spread. Every measured timing median increase is smaller than the larger of
its two observed run-median ranges; improvements are retained as improvements,
not treated as failures because their ranges do not overlap. The full TTFT,
elapsed and cohort timing samples are in the receipt file. This is the owner's
requested three-run noise assessment, not the historical twenty-capture
statistical gate or a new universal 5% guarantee.

Together with the original 2 GiB benefit (B-repeat TTFT 18.622 → 0.390 seconds,
32,736 cached tokens instead of zero, cohort makespan 63.441 → 43.441 seconds),
these controls resolve the two performance blockers under the revised rule.
Focused CPU regressions still pass 145 tests; changed executable scheduler
coverage remains 46/46 lines (100%), with Ruff and whitespace checks passing.
No runtime code was changed to force output identity or improve a timing sample.
All owned servers exited and the GPU lock was released after the final arm.

The final integration rebase onto `73794395a` adds unrelated main changes; the
measured text scheduler, memory cache, engine, server and benchmark-helper
sources are unchanged. Receipt archives preserve the complete readable data;
SHA-256 of the decompressed original and follow-up artifacts respectively:

- `538aa22194e734b8aca88238fc4ce73e3dc94ea2f068790b0c407e6702e72b18`
- `5f75b7d86b757d5df38a55b96eeb61d2c3d78c56d5c68c61060c230dc5bf9448`
