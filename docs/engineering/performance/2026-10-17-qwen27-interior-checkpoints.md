# Guarded Qwen27 interior checkpoints

Owner: Vector. Qualification: 2026-10-17, Mac Studio M3 Ultra, 256 GiB.
Implementation: `54b5312451f67f351d4a805eb241d793f327479d`.

**Verdict: works with native-token parity, with the extra cut limited to lane 1.**
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

The adapter owns at most one optional cut. It considers the latest existing
full-step grid start strictly before both the final message boundary and the
last original scheduler selection, excluding all original selections. An aligned
history boundary therefore does not hide an earlier eligible interior. The
final supplied message must span at least two 2,048-token steps, the scheduler
must already request retention, and the cut must be beyond the restored state.
Admission requires space for the current store, all original selections, and one extra slot.
Pinned entries count conservatively in this admission test.

After prefill, the wrapper checks the actual byte sizes of every pending
snapshot plus the resident store. It keeps the extra only when the complete
set fits the existing eight ordinary slots and 16 GiB budget. The runtime
retains its three reserved system slots. The optional state is released before
inserting original selections if slot or byte pressure would otherwise cause
it to displace them. Release uses exact-prefix `peek` and public
`match(..., take=True)` for the owned cut; it never chooses an arbitrary victim
or removes a pinned system entry.

An incompatible prompt or short final message releases the optional cut.
Ownership is relinquished as history only when the scheduler itself selects
the owned length. Assistant marker proximity is insufficient: raw completions
can contain the same marker. The pinned scheduler filters out restored lengths
(`server/checkpoints.py:32-37`, `server/prompt_fill.py:97-99`), so a shortened
chat whose history floors to the restored cut also keeps optional ownership
until the normal release rules apply. Cancellation preserves runtime retry progress and original snapshots. The store is reached
through a weak reference so keeping the factory does not retain model caches
after backend close. Administrative transfer lookups increment runtime hit
counters, so reuse below comes from terminal provider `cached_tokens`, not the
store's lookup hit counter. Optional releases and native evictions are reported
separately.

The wrapper was reviewed for simplification; the marker inference and its
metadata reinsertion were removed. Its checks cover distinct admission, release, history promotion, interruption, pinning,
and lifecycle cases. The lane gate avoids adding a new synchronization mechanism.

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

The guarded policy intentionally releases the S3 interior on intervening short turns; S4 remains a miss. This is the expected difference from the original two-cut pilot, which held interiors from several histories.

| Original matrix, cache on | Final evictions | Final retained GiB | Maximum retained GiB | MLX peak GiB | Optional releases |
|---|---|---|---|---|---|
| before | 0 | 7.7138 | 7.7138 | 26.6771 | 0 |
| fixed | 0 | 7.7138 | 7.7138 | 26.4421 | 3 |

At the 6,144-token retained cut the observed snapshot charge is 661,403,648 bytes (0.6160 GiB). The adapter retains at most one optional state; later ordinary boundaries and existing system states retain their own runtime accounting.

Across paired mixed phases, the largest observed fixed-minus-before increase is 0.6160 GiB charged retained bytes, 0.7139 GiB MLX active, and 0.3645 GiB cumulative MLX peak. These are phase comparisons, distinct from each fresh process's eventual peak above.

## Model-free verification

- `tests/test_tensorfold_*`: 228 passed after the offline review corrections. The lane-factory regression fails for
  lanes 2/4/8 before the decision and passes afterward; lane 1 passes in both.
  The existing guard regressions cover full slots, actual pending bytes,
  pressure release, compatible edits, short turns, history promotion, pinning,
  cancellation/retry progress, cache-off forwarding, and factory lifetime.
  A focused full-slot factory replay fails against the rebased two-cut pilot
  (`[4096, 6144, 7950]` instead of `[7950]`) and passes against this guard.
- `pytest --cov=rapid_mlx` plus `diff-cover --compare-branch=origin/main
  --fail-under=100`: 76 changed executable lines, zero missing, 100%.
- Ruff check and format check pass across all 12 TensorFold source/test files.
- `scripts/check_mypy_error_budget.py` passes in its pinned Python 3.11
  environment: 839 grandfathered diagnostics across 161 files, no growth. The
  touched adapter has zero errors and no baseline allocation. No mypy budget
  or coverage exemptions were added.

## Limits

This is latest-message retention, not a promise to keep interiors in every
conversation. Intervening short turns release the optional cut; an edit to an
earlier long message can therefore still miss. Edits before the retained tail
cut also miss. System-message reuse remains controlled by the existing system
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

## Offline review corrections

Two model-free regressions failed against the reviewed implementation and pass
with the corrections. The real pinned `PrefillPlan` for 8,202 tokens with message
openers at 0, 33, and 8,192 produces starts `[0, 2048, 4096, 6144, 8192]`.
With original selection `[8192]`, the adapter now retains 6,144 as well; editing
token 7,900 resumes 6,144. A raw completion extending that optional prefix with
56 tokens, an assistant marker, and eight tokens has no scheduler selection.
It retains optional ownership and releases the old entry on the next
incompatible prompt, leaving at most one optional cut.

An offline replay compared the reviewed and corrected hooks over all 436
recorded requests in 22 rebased campaign files, using their audited prompt token
IDs/history/shared lengths, the real pinned plan/store/selection function, and
scalar snapshots. Snapshot sizes follow `258750464 + 65536 * prefix_length`
bytes, verified against every recorded snapshot size. The old-hook replay
matches every recorded fixed-run reuse count. This checks retention decisions;
it does not remeasure latency or execute model inference.

The fixtures are **not all equivalent**: `setup-thinking` in each of
`before-on-0-requests.json` and `fixed-on-0-requests.json` changes candidate
8,192 to 6,144 and selected/retained lengths `[8192]` to `[6144, 8192]`.
Retained bytes increase from 3,945,207,808 to 4,606,611,456 for that request.
The next request, `S5a`, releases the optional cut and restores the previous
inventory/bytes. Its cumulative release counter, and those of `S6`,
`near-start`, `long-system-cold`, and `long-system-edit`, increase by one in
both files. All replayed reuse counts are unchanged. The other 20 files have
identical candidate/admission, ownership, inventory, bytes, and release results.
Removing only the marker heuristic changes none of the 436 requests; no
measured case is lost. Shortened histories without scheduler selection may
release sooner, which preserves the optional-cut ownership contract.

No models were loaded or servers started for these corrections. The changed
`setup-thinking` admission requires an owner decision on any model rerun;
earlier measurements do not measure this new admission.
