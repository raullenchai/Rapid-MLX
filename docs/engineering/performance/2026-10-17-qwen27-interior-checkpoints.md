# Guarded Qwen27 interior checkpoints

Owner: Vector. Qualification: 2026-10-17, Mac Studio M3 Ultra, 256 GiB.
Implementation: `54b5312451f67f351d4a805eb241d793f327479d`.

**Measured result: native-token parity with one lane. The current history-aware
policy has model-free replay and pressure-regression evidence; timings below
belong to the measured implementation above.**
Base: `7be1c00fe45fa0f7c2ee23804075f81f9d1a3aff`. Measurements started on
`76b8296cd1cf3a009de045d719dd4823974afae2`. Main advanced during the
campaign with release notes, an unrelated model profile, and a Desktop video fix.
The branch was rebased again. The Qwen27 profile, CLI, TensorFold modules, pinned
runtime, and frozen before/fixed adapters retain the measured behavior; both
adapter files match final base/implementation blobs byte for byte. Tests,
coverage, and the mypy budget were rerun on the final base.

## Problem and compatibility boundary

A recurrent state cannot be truncated to undo an edit. The Qwen27 TensorFold
adapter already retains message-boundary states, but an edit inside an approximately
8,000-token user message can precede every retained boundary. Such a request
replays the whole message even though the deterministic prefill plan contains
usable interior chunk starts.

This change retains one optional tail checkpoint for the latest long message,
using the existing engine factory and public `begin_stream(checkpoints_at=...)`
argument. The runtime, deterministic chunk plan, scheduler-selected boundaries,
system reservations, byte budget, and slot limit retain their existing behavior.
No runtime source or private method is patched. The extra cut is enabled only
with one lane. Multiple lanes use the native engine factory with the same plan.
The lane argument, model `parallel` argument, memory fraction, executor sizing,
and server environment selection remain those of the base commit.

## Runtime evidence and lane decision

All runtime references below are relative to `src/tensorfold/` at TensorFold
0.6.6 revision `cb2ebf0540f42604e2759b2ddef497861e928248`.

- `server/scheduler.py:149` constructs one `tensorfold-engine` thread targeting
  `_run`. `_run` calls `_loop` at line 311. The loop admits jobs before advancing
  a prefill at lines 326–330. `_room` at line 425 counts live streams plus
  unfinished prefills; `_admit` at line 430 can admit several prefills.
- `server/prompt_fill.py:71` starts a prefill and line 124 calls
  `engine.begin_stream`. `_fill` at line 158 advances one generator with `next`
  at line 178. `_end_fill` at line 195 transfers completed snapshots via
  `_keep_checkpoints` at line 205. `server/scheduler.py:539` implements that
  transfer and line 548 inserts the retained snapshots in the shared store.
- `server/checkpoints.py:186`, `:192`, and `:232` protect each `peek`, `match`,
  and `insert` individually with the store lock. `nbytes` at line 172 and
  `__len__` at line 272 read current entries; these controls do not reserve
  capacity for unfinished streams.

Thus engine work runs on one scheduler thread, but a stream's admission,
generator advancement, and store insertion are separate scheduler operations.
With several lanes, another prefill can run between them. A Python mutex around
an individual adapter method would not make the whole lifecycle atomic. The
one-cut policy also stores its ownership on the engine, while several prefills
share that engine. Supporting extras across lanes would require stream-specific
ownership and capacity reservations or a lifecycle serialization policy.

The chosen boundary is simpler: lanes 2, 4, and 8 use the unmodified native
engine class. They create no optional-cut owner, admission, or release code.
Lane 1 permits only one active stream or prefill (`scheduler.py:428`), so the
existing lifecycle guard remains applicable. Model-free factory tests exercise
all four lane values and assert native class identity and absence of optional
state for multiple lanes. The multi-lane cases fail against the rebased adapter
before the lane decision and pass afterward. This avoids relying on a claim
that a single scheduler thread makes interleaved lifecycles atomic.

## Retention policy

The adapter owns at most one optional cut. On a compatible resume from that
exact cut, promotion uses the native selection counterfactual with this optional
entry excluded. Public `store.peek(ids, usable=...)` rejects its length and
accepts only actual chunk starts from public `prompt_chunks`
(`engine/lane_engine.py:388-393`), returning the same best strict-prefix hit
and prior `last_prompt` as native matching (`server/checkpoints.py:175-208`).
The scheduler's usable predicate is chunk membership; its `take` flag changes
cache transfer, not the selected hit (`server/prompt_fill.py:80-94`).
The hook calls public `choose_checkpoints(history_len, cached_main,
last_prompt_main, ids)` (`server/checkpoints.py:28-37`), floors those choices,
unions the floored nonzero shared-prefix lengths using public `PromptChunks.floor`
(`engine/prefill_plan.py:84-91`), and filters strictly between
`cached_main` and `len(ids)`, exactly as `server/prompt_fill.py:57-58,97-99`.
Ownership is relinquished iff the owned length is in this set. The registry
supplies the actual submitted history and shared-prefix lengths; missing IDs
mean zero history and no shared cuts. Raw assistant markers do not identify
native selections.

Promotion replaces the exact resident key through public `store.insert`, using
its existing cache and the current prompt. This copies no arrays and replaces
rather than double-charges the entry (`server/checkpoints.py:232-245`). It also
sets the creation length (`born`) that main would store: merely relinquishing
ownership leaves the old length, which changes preferential eviction
(`:275-283`). Native match already refreshes recency/last_prompt (`:192-210`).
If the promoted length is also a floored shared prefix, insertion preserves
native pinning (`server/scheduler.py:545-548`). Pinned entries are protected;
missing entries clear stale ownership.

Every other restored cut remains optional and releases on an incompatible
prompt or when a later native boundary supersedes it. There is no second resumed owner. Candidate selection uses the latest
existing full-step grid start strictly before the final message boundary and
last original selection, excluding all original selections. The final message
must span two 2,048-token steps and the scheduler must already request retention.
Admission conservatively counts all current entries, original selections, and
one extra slot. Before begin_stream and again in the generator's finally,
slot/byte pressure releases the optional resident before native insertion.
Pending snapshots use their actual charged sizes. An extra that cannot fit is
removed; original/system and interrupted-prefill progress remain intact.

Release uses public exact-length peek/match(take=True), checking pinning first.
The scheduler exhausts/throws/closes the wrapper before transferring snapshots
(`server/prompt_fill.py:174-224,231-242`; `server/scheduler.py:539-548`). The
store getter remains weak; no runtime source or private method is replaced.

### Product request paths

The CLI routes this backend to the HTTP provider
(`rapid_mlx/cli.py:6949-6950,7016`). Its _prompt_boundaries renders messages
without the generation suffix and computes shared system lengths
(`rapid_mlx/speculative/tensorfold_qwen27_server.py:76-121`). The provider builds
ChatJob(history_len=..., shared_prefix_lens=...) and submits directly to the
scheduler. Before submission it registers `job_id -> (history_len, shared_prefix_lens)`, removing
the entry in finally on completion, cancellation, submission failure, or worker
error. Runtime `server/prompt_fill.py:104-105` constructs
`LaneStream(stream_id=job.job_id)` and `:124-125` passes it to begin_stream.
**This product path supplies authoritative selection context to the hook.**
The measured campaign invoked the CLI HTTP route; its saved command receipts
and public Scheduler.submit audits record nonzero history/shared lengths.

The separate TensorFoldQwen27Backend.stream convenience API calls
app.chat([], prompt=list(prompt_ids))
(in `rapid_mlx/speculative/tensorfold_qwen27.py`). Runtime
`server/prompts.py:15-17,72-79` returns a RenderedPrompt with default history zero
for explicit prompt requests; `server/app.py:317` sets system_len to zero.
Consequently choose_checkpoints selects nothing for that cold path, and this
adapter admits no optional cut. Matching main there means boundary-only native
behavior, with no claimed interior-edit acceleration. These statements apply
to the qualified pinned runtime, not to a different message-based caller.

## Measurement method

Fresh `rapid-mlx serve` processes, temperature zero, streaming HTTP,
200-token output cap (2,048 for the original thinking case), fixed identical
request bodies, memory-only checkpoints, and no model downloads. Every model
run passes through `scripts/large-model-run.py --working-set-gb 64` with a shared
host lock. HF and Transformers are offline; ports are in 18800–18899. Only
owned processes are stopped. The coordinator holds the lock across the campaign;
every server launch also uses the wrapper with its inherited lock marker.

Before uses the exact base adapter; fixed uses the measured adapter source,
with hashes retained in scratch provenance. Mixed K=4 uses four distinct long
opening prompts, an interior edit in the latest conversation, three round-robin
follow-up rounds, and regeneration of each last request. Common K=4 omits the
interior edit. Each shape has three repeats, with before/fixed order alternating
by repeat. Mixed runs have full cache-off pairs, including every opening,
follow-up, edit, and regeneration. The original 16-request single-conversation
matrix is repeated once for each before/fixed and on/off combination.

TTFT is the first nonempty content or reasoning delta. Native output token IDs
are compared for identical request hashes. Reuse comes from terminal provider
counts. Memory observations include MLX active/cache/peak and the runtime store's
charged bytes. Charged bytes can count immutable shared arrays more than once;
MLX peak is cumulative per fresh process, not a per-request delta. Timings on
shared hardware are noisy; the common-only median drifted about 25% across
repeats for both variants. Paired comparisons, token reuse, and parity are
reported separately.

Dependencies: TensorFold 0.6.6 at the revision above; MLX 0.32.3; MLX-LM 0.31.3;
Transformers 5.15.1; model environment Python 3.12.13. Target and draft snapshot
revisions are respectively `70ae7fac63274ff2eac54152031433374cb80f2f` and
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Rebased measurements

436 HTTP requests across 22 fresh servers; 297 native-token comparisons, all identical. Mixed runs compare every request against their cache-off pair. Common on/off parity is additionally sampled by the identical mixed fixture bodies and the full single-conversation matrix. Three repeats for K=4; one repeat of the original matrix.

| Shape / operation | Common hits before → fixed (each repeat) | Reused tokens before → fixed | TTFT before → fixed, median [min,max] s |
|---|---|---|---|
| common K=4 follow-up/regenerate | 16/16, 16/16, 16/16 → 16/16, 16/16, 16/16 | 7959,8220,8480,8481,8740,8742 → 7959,8220,8480,8481,8740,8742 | 1.132 [0.123, 1.412] → 1.127 [0.124, 1.533] |
| mixed K=4 follow-up/regenerate | 16/16, 16/16, 16/16 → 16/16, 16/16, 16/16 | 7959,8220,8442,8481,8742 → 7959,8220,8442,8481,8742 | 1.092 [0.117, 1.853] → 1.094 [0.118, 1.917] |

| Shape / repeat | Common hits before → fixed | Common median TTFT before → fixed s |
|---|---|---|
| common / 1 | 16/16 → 16/16 | 1.092 → 1.092 |
| common / 2 | 16/16 → 16/16 | 1.159 → 1.129 |
| common / 3 | 16/16 → 16/16 | 1.367 → 1.358 |
| mixed / 1 | 16/16 → 16/16 | 1.094 → 1.095 |
| mixed / 2 | 16/16 → 16/16 | 1.093 → 1.096 |
| mixed / 3 | 16/16 → 16/16 | 1.088 → 1.086 |

The common-only median drifted about 25% across repeats for both variants. Paired medians remain close; no speed claim for common operations is inferred from that shared-machine drift.

| Interior operation | Common hits | Reused tokens before → fixed | TTFT before → fixed, median [min,max] s |
|---|---|---|---|
| mixed K=4 interior edit | — | 0 → 6144 | 23.995 [23.765, 25.225] → 5.946 [5.805, 6.114] |

Cache-off interior-edit TTFT before → fixed: 24.181 [23.697, 24.593] → 24.869 [23.724, 25.469] s. Common aggregates contain 48 requests per variant and shape. Prompt reuse is not a byte-capacity guarantee.

| Shape | Final native evictions before → fixed | Final retained GiB before → fixed | Maximum retained GiB before → fixed | Process MLX peak GiB before → fixed | Final optional releases before → fixed |
|---|---|---|---|---|---|
| common K=4 | 7 → 7 | 6.1006 → 6.1006 | 6.1006 → 6.1006 | 23.4986 → 23.5298 | 0 → 4 |
| mixed K=4 | 6 → 6 | 6.0347 → 6.0347 | 6.0347 → 6.0347 | 23.6063 → 23.6067 | 0 → 4 |

Final retained/MLX peak values are medians across fresh cache-on processes; maximum retained is the maximum over every request and repeat. Native evictions and optional releases show all observed counts. Common retained prefix sets, reuse, bytes, and eviction counts match the base after every common operation in all repeats.

| K=4 operation | Prompt tokens | Reused before → fixed | TTFT before → fixed (median s, three repeats) |
|---|---|---|---|
| common/opening-c0 | 7966 | 0 → 0 | 23.910 → 24.061 |
| common/opening-c1 | 7966 | 0 → 0 | 23.840 → 24.730 |
| common/opening-c2 | 7966 | 0 → 0 | 24.370 → 24.020 |
| common/opening-c3 | 7966 | 0 → 0 | 24.888 → 25.057 |
| common/followup-1-c0 | 8227 | 7959 → 7959 | 1.150 → 1.144 |
| common/followup-1-c1 | 8227 | 7959 → 7959 | 1.157 → 1.183 |
| common/followup-1-c2 | 8227 | 7959 → 7959 | 1.153 → 1.130 |
| common/followup-1-c3 | 8227 | 7959 → 7959 | 1.161 → 1.141 |
| common/followup-2-c0 | 8488 | 8220 → 8220 | 1.179 → 1.154 |
| common/followup-2-c1 | 8488 | 8220 → 8220 | 1.195 → 1.137 |
| common/followup-2-c2 | 8488 | 8220 → 8220 | 1.208 → 1.149 |
| common/followup-2-c3 | 8487 | 8220 → 8220 | 1.161 → 1.133 |
| common/followup-3-c0 | 8741 | 8481 → 8481 | 1.071 → 1.032 |
| common/followup-3-c1 | 8749 | 8481 → 8481 | 1.173 → 1.129 |
| common/followup-3-c2 | 8749 | 8481 → 8481 | 1.166 → 1.129 |
| common/followup-3-c3 | 8747 | 8480 → 8480 | 1.165 → 1.099 |
| common/regenerate-c0 | 8741 | 8481 → 8481 | 1.071 → 1.033 |
| common/regenerate-c1 | 8749 | 8742 → 8742 | 0.132 → 0.147 |
| common/regenerate-c2 | 8749 | 8742 → 8742 | 0.132 → 0.137 |
| common/regenerate-c3 | 8747 | 8740 → 8740 | 0.131 → 0.153 |
| mixed/opening-c0 | 7966 | 0 → 0 | 24.012 → 24.993 |
| mixed/opening-c1 | 7966 | 0 → 0 | 24.887 → 24.686 |
| mixed/opening-c2 | 7966 | 0 → 0 | 24.921 → 24.618 |
| mixed/opening-c3 | 7966 | 0 → 0 | 24.953 → 24.573 |
| mixed/interior-edit-c3 | 7966 | 0 → 6144 | 23.995 → 5.946 |
| mixed/followup-1-c0 | 8227 | 7959 → 7959 | 1.100 → 1.098 |
| mixed/followup-1-c1 | 8227 | 7959 → 7959 | 1.091 → 1.095 |
| mixed/followup-1-c2 | 8227 | 7959 → 7959 | 1.092 → 1.107 |
| mixed/followup-1-c3 | 8219 | 7959 → 7959 | 1.032 → 1.017 |
| mixed/followup-2-c0 | 8488 | 8220 → 8220 | 1.125 → 1.113 |
| mixed/followup-2-c1 | 8488 | 8220 → 8220 | 1.138 → 1.114 |
| mixed/followup-2-c2 | 8488 | 8220 → 8220 | 1.109 → 1.104 |
| mixed/followup-2-c3 | 8449 | 7959 → 7959 | 1.850 → 1.860 |
| mixed/followup-3-c0 | 8741 | 8481 → 8481 | 1.021 → 1.020 |
| mixed/followup-3-c1 | 8749 | 8481 → 8481 | 1.096 → 1.119 |
| mixed/followup-3-c2 | 8749 | 8481 → 8481 | 1.103 → 1.112 |
| mixed/followup-3-c3 | 8686 | 8442 → 8442 | 0.925 → 0.921 |
| mixed/regenerate-c0 | 8741 | 8481 → 8481 | 1.019 → 1.013 |
| mixed/regenerate-c1 | 8749 | 8742 → 8742 | 0.128 → 0.129 |
| mixed/regenerate-c2 | 8749 | 8742 → 8742 | 0.128 → 0.123 |
| mixed/regenerate-c3 | 8686 | 8442 → 8442 | 0.919 → 0.921 |

### Original single-conversation regression

| Case | Prompt tokens | Reused before → fixed | TTFT before → fixed s | Cache-off TTFT before → fixed s | Native parity before / fixed |
|---|---|---|---|---|---|
| cold | 7958 | 0 → 0 | 26.421 → 26.551 | 26.452 → 26.360 | 1/1 / 1/1 |
| S3 | 7958 | 0 → 6144 | 25.971 → 5.882 | 25.731 → 25.995 | 1/1 / 1/1 |
| S0 | 8190 | 7951 → 7951 | 1.119 → 1.114 | 26.435 → 26.126 | 1/1 / 1/1 |
| S1 | 7958 | 7951 → 7951 | 0.118 → 0.140 | 26.387 → 26.323 | 1/1 / 1/1 |
| S2 | 8190 | 7951 → 7951 | 0.905 → 0.905 | 26.538 → 26.923 | 1/1 / 1/1 |
| S3f | 8190 | 7951 → 7951 | 1.079 → 1.029 | 26.286 → 26.358 | 1/1 / 1/1 |
| setup-third | 8420 | 7951 → 7951 | 2.032 → 2.053 | 27.910 → 27.460 | 1/1 / 1/1 |
| S4 | 7958 | 0 → 0 | 25.831 → 25.818 | 26.193 → 26.484 | 1/1 / 1/1 |
| S5b | 8190 | 7951 → 7951 | 1.080 → 1.067 | 26.255 → 26.405 | 1/1 / 1/1 |
| S5c | 8098 | 7951 → 7951 | 0.613 → 0.614 | 26.761 → 25.834 | 1/1 / 1/1 |
| setup-thinking | 8198 | 0 → 0 | 27.078 → 27.233 | 26.858 → 27.095 | 1/1 / 1/1 |
| S5a | 8424 | 8192 → 8192 | 1.095 → 1.105 | 26.935 → 27.716 | 1/1 / 1/1 |
| S6 | 8419 | 7951 → 7951 | 2.032 → 2.068 | 28.190 → 28.260 | 1/1 / 1/1 |
| near-start | 7958 | 0 → 0 | 25.605 → 25.891 | 25.526 → 25.655 | 1/1 / 1/1 |
| long-system-cold | 7970 | 0 → 0 | 26.123 → 25.597 | 25.862 → 25.449 | 1/1 / 1/1 |
| long-system-edit | 7970 | 6144 → 6144 | 6.588 → 6.579 | 26.580 → 26.488 | 1/1 / 1/1 |

The measured implementation released the S3 interior on intervening short turns;
S4 was a miss. The current policy also misses there, matching main; these timings describe
the measured implementation.

| Original matrix, cache on | Final evictions | Final retained GiB | Maximum retained GiB | MLX peak GiB | Optional releases |
|---|---|---|---|---|---|
| before | 0 | 7.7138 | 7.7138 | 26.6771 | 0 |
| fixed | 0 | 7.7138 | 7.7138 | 26.4421 | 3 |

At the 6,144-token retained cut the observed snapshot charge is 661,403,648 bytes
(0.6160 GiB). The current adapter retains at most one optional state; promoted
history and system states use native accounting. No new peak-memory measurement
is claimed.

Across paired mixed phases, the largest observed fixed-minus-before increase is 0.6160 GiB charged retained bytes, 0.7139 GiB MLX active, and 0.3645 GiB cumulative MLX peak. These are phase comparisons, distinct from each fresh process's eventual peak above.

## Model-free verification

The real pinned plan, store, and choose_checkpoints cover shortened-history
probe-2, same-length-edit probe-3 under slot/byte pressure, raw markers with
history zero, the aligned 8,202-token boundary, and the former superseded-history
counterexample. Probe-2 regenerates 6,144 on both main and the hook; probe-3
regenerates 7,950 with zero evictions on both. With an older superseded 990-token
entry, shortened history still regenerates 6,144 on both under pressure.
The latter fails against previous HEAD (zero reuse). A further continuation
regression catches stale creation metadata: without refreshing born, native
pressure yields 7,950 on main versus 6,144 on the hook; promotion now preserves
7,950 on both with identical native eviction counts.

Registry tests check submission sees the exact job ID/history pair and cleanup
on success, generator close, worker error, and submit error. Missing IDs default
to zero. History-floor tests include zero, before/exactly at/after the cut, and
invalid prompt-end/out-of-range history. Existing tests preserve pinning,
interrupted prefill, byte/slot headroom, cache-off, lane gating and weak lifetime.
No new coverage exemptions or mypy budget allocations are used.

Current verification: 241 TensorFold tests pass; fresh diff-cover against
origin/main is 100% with zero missing changed lines for both adapter files.
Ruff check/format pass across all 12 TensorFold Python files. The pinned
Python 3.11 mypy budget remains 839 grandfathered errors across 161 files,
with no growth.

## Limits

This is latest-message retention, not a promise to keep interiors in every
conversation. Optional cuts, including raw or edited resumes without a matching
counterfactual native selection, release on incompatible turns, when a later native boundary
supersedes them, or first under pressure. Promoted
native selections have native eviction behavior. Promotion alone does not
materialize other counterfactual selections below the restored cut: a newly
introduced shorter shared prefix can be suppressed, causing later reuse to
fall below native behavior. The policy therefore does not establish universal
retention or reuse equivalence. Earlier-message edits and edits before the
retained tail cut can still miss. System-message reuse remains controlled by the existing system
checkpoints. The measured byte cost is workload-specific and residency remains
conditional. Multiple lanes preserve boundary-only retention and do not gain
this interior-edit speedup. No multi-lane interior-cut performance claim is made.

The evidence does not qualify temperature sampling, all edit locations,
long-running memory pressure, pool routing/affinity, GUI journeys, or other model
families. Cancellation, ownership, byte/slot pressure, pin promotion, and factory
lifetime receive model-free regression checks; cancellation stress on a loaded
model is outside this campaign.

## Reproduction and artifacts

Private campaign artifacts and frozen sources live under
`/private/tmp/tensorfold-qwen27-cuts-pilot/rebased/`; the parent contains the
original briefs and historical pilot data. Scratch is subject to host retention.
The tables here preserve the conclusions independently of scratch lifetime.
From the task worktree, with that existing environment:

```sh
D=/private/tmp/tensorfold-qwen27-cuts-pilot
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  "$D/venv/bin/python" scripts/large-model-run.py \
  --working-set-gb 64 --lock-timeout 14400 -- \
  "$D/venv/bin/python" "$D/rebased/campaign.py"
COVERAGE_FILE="$D/rebased/current-main-coverage.db" PYTHONDONTWRITEBYTECODE=1 \
  "$D/venv/bin/python" -m pytest tests/test_tensorfold_* -p no:cacheprovider \
  --basetemp="$D/rebased/pytest-current-main" --cov=rapid_mlx \
  --cov-report="xml:$D/rebased/current-main-coverage.xml"
"$D/venv/bin/diff-cover" "$D/rebased/current-main-coverage.xml" \
  --compare-branch=origin/main --fail-under=100
MYPY_CACHE_DIR="$D/rebased/mypy-pinned-cache" \
  "$D/rebased/mypy-venv/bin/python" scripts/check_mypy_error_budget.py
```

The K=4 input recipe is deterministic with the pinned target tokenizer:

```python
system = (
    "You are a backend engineer. Review the inventory carefully. "
    "Give a concise deployment plan of approximately 200 tokens. "
    "Do not repeat the inventory. "
)
raw = "".join(
    f"Service {i}: region west, replicas {i % 7 + 2}, queue limit {i % 19 + 50}; "
    "preserve session order and retry transient faults.\n"
    for i in range(1000)
)
user = tokenizer.decode(tokenizer.encode(raw, add_special_tokens=False)[:7900])
user += "\nFinal instruction: explain how to deploy these services safely."
opening = f"Conversation {c}: independent inventory.\n" + user
followup = (
    f"Follow-up {round} for conversation {c}: revise the plan with rollback "
    "checkpoints, health checks, staged rollout gates, dependency ordering, "
    "operator ownership, monitoring, and verification. Explain the concrete "
    "next steps in about 200 tokens."
)
```

Use conversations `c=0..3`, rounds `1..3`, thinking disabled, and append each
assistant response before the next round. Freeze those bodies from the first
base run and replay them for every variant. Regeneration resends the last
request body. The mixed edit replaces `deploy these services safely.` with
`upgrade these services safely.` in c3 immediately after the four openings,
before appending that edited response and starting follow-ups. The original
single-conversation matrix additionally tests reply edits, stripped reasoning,
removing an exchange, a near-start inventory edit, and moving the inventory to
system content. Its saved exact bodies are identified by the fixture hash in
campaign provenance.

The mypy environment uses Python 3.11 and exactly `config/mypy-requirements.txt`.
Reusing completed campaign receipts is allowed only with their exact source
hashes; archive receipts before a new measurement. A startup-only attempt was
interrupted before inference to bypass a spurious large-download confirmation;
the successful campaign uses `RAPID_MLX_AUTO_PULL=1` together with offline mode
and disabled downloads. That interruption is excluded from measurement.

## Offline policy evidence

No models were loaded or servers started for the history-registry correction.
The 22 saved campaign files (436 requests) are replayed with audited token IDs,
history/shared lengths, the pinned real plan/store/choose_checkpoints, and scalar
snapshots sized `258750464 + 65536 * prefix_length` bytes, verified against every
recorded snapshot size. Replay compares the native main-equivalent baseline,
previous HEAD `2ff83ddb2`, and the current adapter. It checks policy decisions,
retained inventories including creation metadata, bytes, and reuse; it does not
remeasure latency or output parity. Historical model tables remain unchanged.

The aligned thinking prompt admits 6,144 as well as 8,192; that admission was
never remeasured. Its next short request releases the optional state. Historical
S4 measured zero reuse, equal to main and the current policy. Previous HEAD
reused 6,144 there by preserving surplus across intervening conversations.
These corrections have replay evidence only.

Reproduce locally with the existing scratch environment:
`PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /private/tmp/tensorfold-qwen27-cuts-pilot/venv/bin/python /private/tmp/tensorfold-qwen27-cuts-pilot/review/round4_fixture_equivalence.py`.
Full decisions and differences: `review/round4-fixture-equivalence.json` under
the same scratch root. These files are subject to host retention.

Current replay: versus previous HEAD, 48 request records differ in eight files;
14 files are identical. No request reuses less than main. S4 in the two
single-conversation files changes 6,144 to zero versus previous HEAD, matching
main and the original measurement. Every other reuse count is unchanged.
Original selections, candidates, and native eviction counts are unchanged.
Selected/retained snapshots gain 6,144 for S4 and near-start in those two files
(four records). Inventories change on 34 records; charged bytes decrease by
661,403,648 on 32 records. Ownership fields change on 12 records, removed resumed
ownership on 44, cumulative releases on 40.

Against the native baseline, eight interior-edit requests improve zero to
6,144, with no reuse regression. Optional state adds 661,403,648 bytes on 64
request records. All common K=4 follow-up/regenerate inventories, charged bytes,
reuse, and native eviction counts equal main after every operation. Replay
counts administrative releases separately. Previous HEAD already disagrees
with the historical fixed measurement only at S4; this is accounted for above.
The policy is tested with scalar caches; full scheduler memory admission and
model execution are not requalified by this replay.
