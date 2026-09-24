# Qwen Auto Runtime

Date: 2026-09-23
Status: proposed; implementation starts in the next session
Owner: Atlas
Host: Studio for implementation and reproducible qualification; M4 Pro 48 GB
and M1 Max 64 GB for product gates

## Decision

Build a fail-closed, per-request runtime plan for qualified vision-capable
Qwen checkpoints:

- load the target model once;
- route requests containing media to the MLLM scheduler;
- route text-only requests to the fastest qualified shared-weight text
  scheduler: MTP when the exact artifact and runtime pass qualification,
  ordinary native-cache autoregressive decode otherwise;
- keep all MLX work on the existing single model-owner executor;
- fall back at startup, never by replaying a request after generation began.

The intended user promise is: **the default Qwen server is both fast for text
and able to accept images, without a restart or a second copy of target
weights.** This is a staged destination, not a claim about today's code.

Step 3c of the mlx-vlm vendoring project is not a prerequisite and remains
frozen. The runtime will use the existing, narrow drafter/injector provider
boundary. Unknown artifacts, cache layouts, drafters, or model implementations
stay on the current lane.

## Why this is the next high-ROI project

The first 24 hours of 0.15.0 telemetry covered 78 reporting installs. The
signal is small and opt-out biased, but it is directionally strong:

| Signal | Observation | Consequence |
| --- | ---: | --- |
| Qwen3.5 4B Q4 | 16 installs | qualify after the architecture is proven; it is the default starter and has a prior Metal-wedge history |
| Qwen3.8 27B Q4 | 15 installs | highest-value large-model target; 48/64 GB safety is mandatory |
| Qwen3.6 35B Q4 | 5 installs | best first proof because a shared-weight native text lane already exists and has a committed receipt |
| Chat | 279 inference milestones | optimize the dominant request surface |
| Image endpoint | 26 milestones | image generation is a real secondary surface, but it is not evidence for VLM image input |
| Image-input rejection | 7 events from 2 installs | current text-vs-vision lane selection leaks into user failures |
| Heavy users | 8 installs above 100 requests; 2 above 1,000 | cancellation, memory admission, and recovery matter as much as single-request speed |
| Common hosts | M1 Max 64 GB and M4 Pro 48 GB | qualification must cover constrained large-model hosts, not only the 256 GB Studio |

Today, the CLI injects an alias-owned MTP default before serving-lane
resolution. A requested speculative method forces a vision-capable checkpoint
onto the text-only process lane. Conversely, the existing Qwen3.6
shared-weight native text engine starts only inside the MLLM process and only
when `spec_decode == "none"`. The two valuable capabilities therefore exclude
each other before weights are loaded.

## Verified starting point

The repository already owns the pieces that make this project bounded:

1. The MLLM path is the sole serving and benchmark path for media.
2. Cache, APC, vision-cache, and input-preparation primitives are vendored.
3. The text AR generation core is vendored (step 3a).
4. The speculative coordinator core is vendored (step 3b).
5. `BatchedEngine` already runs an MLLM scheduler and an optional
   `AsyncEngineCore` over one loaded Qwen3.6 language model and one executor.
6. Media requests remain authoritative on MLLM; pause, abort, cache clear,
   stats, and shutdown already span both schedulers.
7. The exact Qwen3.6-35B Q4 receipt measured 107.63 tok/s on the native-cache
   text lane versus 72.86 tok/s on serialized MLLM, a 1.482x gain, with only
   17 MB additional active memory and no second target weight copy.

The ownership gained in 3a/3b reduces upstream drift and gives Rapid a place
to fix coordinator/cache behavior. It does **not** by itself install MTP into
the dual-lane engine. That integration must be qualified explicitly.

## Current facts that must not be blurred

### Memory

- MLLM has count-based admission but no projected D-METAL-CAP admission.
- The text `Scheduler` has projected KV admission, in-flight reservations,
  pressure eviction, and actionable 503 errors.
- MLLM's exact text cache and media-boundary cache already share one byte
  ceiling. They are not two independent 20% budgets.
- The additional text scheduler owns a separate retained prefix-cache budget.
  That cross-scheduler duplication is the real budget gap.
- Both schedulers share the process-wide MLX allocator and Metal cap.

### Artifact identity

The config metadata at the locally cached, commit-pinned revisions shows that
all three priority artifacts are the `qwen3_5` / `qwen3_5_moe` family, not Qwen
Flash-Next:

| Alias | Revision | Outer/text model type | Layers | MTP metadata |
| --- | --- | --- | ---: | --- |
| `qwen3.5-4b-4bit` | `32f3e8e…` | `qwen3_5` / `qwen3_5_text` | 32 | one head |
| `qwen3.6-35b-4bit` | `38740b8…` | `qwen3_5_moe` / `qwen3_5_moe_text` | 40 | one head |
| `qwen3.8-27b-4bit` | `aa985c2…` | `qwen3_5` / `qwen3_5_text` | 64 | one head |

`qwen3.8-flash-next-4bit` is the separate `qwen4_exp` path. It is not evidence
for or against the 27B artifact.

The current Qwen3.8 alias declares its target repository as its MTP sidecar.
The cached pinned snapshot contains three sharded target files **and** the
accepted nested `mtp/model.safetensors` sidecar. Looking only at the snapshot
root produces a false negative; the B0 probe must walk the provider's declared
layouts and report the selected file shape without reading model weights.

### B0 artifact trust model

B0 verifies the bytes it already reads. For `config.json` and the safetensors
index, a 40-hex Hugging Face cache-object name must equal the Git blob SHA-1 and
a 64-hex name must equal the raw SHA-256. A mismatch fails before the runtime
capability is minted.

B0 deliberately does not open 15–20 GiB tensor shards. Target and MTP tensor
facts are therefore Hugging Face cache-object provenance, not byte-content
verification. Qualification binds every cache-object name and observed file
size, and reports `tensor_byte_integrity: unchecked`. Repointing, renaming, or
size drift fails closed at the point-in-time revalidation boundary.

Optional `.sha256` declarations are bounded metadata reads. A regular receipt
must fit within 4 KiB; a receipt symlink must additionally resolve through the
same repository's canonical Hub cache-object layout. Receipts that alias the
candidate tensor inode, arbitrary symlinks, and oversized files are rejected
before any content read.

This contract trusts the Hugging Face downloader/cache CAS invariant. It does
not detect pre-existing corruption, content restored after observation, or an
equal-size byte mutation under the same cache-object name. Full tensor-byte
integrity would require a separate opt-in verification pass and is outside B0.

## Runtime-plan contract

Introduce one immutable `QwenRuntimePlan`, resolved before loading weights and
revalidated after loading them. It is the source of truth for boot logs,
status, routing, and tests.

Suggested fields:

```text
target_lane: vision | text
text_mode: none | native_ar | mtp
selection_source: operator | alias_default | qualified_auto | fallback
qualification_id: nullable stable receipt id
reason: closed enum
media_enabled: bool
fallback_chain: ordered tuple
```

Do not infer eligibility from a model name. A qualification row binds:

- public alias and backing repository;
- commit-pinned target and optional drafter revisions;
- every target/MTP cache-object name and observed size, with tensor-byte
  integrity explicitly unchecked;
- outer and language `model_type`;
- quantization and weight layout;
- exact layer/cache geometry;
- required injector and cache API probes;
- supported sampling, logits processors, tools, and KV configuration;
- tested runtime versions and hardware classes;
- benchmark/quality receipt.

An exact alias may select a row. A raw path may select it only when its
resolved commit-pinned provenance and size receipt matches the row. Unknown and
mutable provenance fails closed.

## Selection semantics

Explicit operator intent remains stronger than automation during the first
release:

| Input | Initial behavior |
| --- | --- |
| no flags, exact qualified Qwen alias | preserve the current lane through B2; after B3 memory-pressure qualification, use `qualified_auto`: MLLM for media, best qualified shared-weight text mode |
| `--mllm` | force the vision target and suppress alias-owned speculative defaults; preserve the already-qualified shared-weight native AR text companion |
| `--no-mllm` / forced text | text-only; preserve current decoder-selection rules |
| `--no-spec-decode` | dual runtime may use qualified native AR, never MTP |
| explicit `--speculative-config` | preserve current text-only semantics until that exact dual-MTP combination has its own receipt |
| conflicting explicit flags | fail before loading weights |
| unqualified alias or raw path | preserve current serving-lane decision |

The important ordering change is that an *implicit* alias MTP default no
longer forces the process lane before the auto runtime can be considered. An
explicit speculative request remains explicit and does not silently acquire
vision semantics in the first release.

## Boot and request flow

```text
alias + flags + hardware
        |
        v
pre-load RuntimePlan ---- no exact receipt ----> current lane resolution
        |
        v
load one VLM target on model-owner executor
        |
        +--> start MLLM scheduler (authoritative fallback)
        |
        +--> run exact geometry/cache/template/injector probes
                    |
             pass  |  fail
                    |   +----------------------> MLLM-only fallback
                    v
          start shared-weight text scheduler
                    |
                    v
request has media? ---- yes ----> MLLM scheduler
        |
        no
        v
native AR or MTP text scheduler
```

Both schedulers must continue to submit every model forward to the same
single-thread executor. The Qwen MRoPE fields (`_position_ids` and
`_rope_deltas`) are lane-local state temporarily installed around an entire
forward. MTP injection must not weaken that transaction.

Startup failure may follow the declared fallback chain only while the target is
known to be unmodified. Once injection mutates the shared language model, any
later failure must tear down and reload a clean target before falling back, or
fail startup. Continuing on that same instance is forbidden. Once a request is
committed to a scheduler, there is no cross-lane replay: replay can duplicate
tokens, tools, side effects, and request IDs.

## MTP integration boundary

The first prototype should reuse the production text scheduler, not invent a
third speculative engine:

1. load the VLM target once;
2. resolve the row's pinned drafter revision to a verified local snapshot; do
   not pass a mutable repository ID to the provider;
3. apply the existing Rapid `dispatch_mtp_inject` provider to the loaded Qwen
   language module on the model-owner executor;
4. wrap that same language module with a qualified cache wrapper;
5. construct the companion `AsyncEngineCore` with `spec_decode="mtp"`;
6. require a startup install probe to prove the live scheduler reports MTP.

All fallible, non-mutating validation must happen before step 3. Injection is a
transaction boundary: because the current provider replaces the model class,
attaches draft state, and installs process-global patches without an uninject
operation, failure after step 3 cannot safely discard only the candidate. It
must tear down and reload the target before following the fallback chain, or
abort startup. A future same-instance fallback requires an explicit, tested
uninject transaction.

This path loads only the draft head/sidecar in addition to the already-loaded
target. It does not load a second target model. Concrete drafter/model
vendoring is unnecessary if the existing provider can satisfy the closed
contract.

If the loaded mlx-vlm language class cannot satisfy the existing injector
without broad model-class copying, the MTP row is a no-go and the auto runtime
ships native AR for that artifact. Reopening 3c is not an accepted workaround.

## Memory and cache policy

The auto runtime needs one engine-owned memory view, but it should arrive in
two risk-sized steps.

For the first canary:

- retain the existing process-wide Metal limit;
- disable retained device prefix state on the companion text scheduler, or
  give it an explicit slice from the one engine-wide cache budget;
- keep live request caches and model/draft weights accounted by the existing
  Metal admission preflight;
- expose per-lane active requests, cache bytes, and cap violations.

Before default-on enrollment on 48 GB hardware:

- add a shared reservation ledger owned by `BatchedEngine`;
- make both schedulers charge in-flight KV/workspace projections to it;
- add an MLLM projection for text tokens plus conservatively bounded media
  prefill/workspace;
- register MLLM exact/media and text retained caches against one byte budget;
- evict inactive retained state before rejecting, never active request state;
- return an actionable 503 before prefill rather than poisoning an MLLM batch.

Do not add a second Metal limit or independently compute another 20%-of-RAM
cache ceiling.

## Work slices

### B0 — truth probe and runtime-plan skeleton

No default behavior change.

- Add the pure `QwenRuntimePlan` resolver and stable reason enums.
- Preserve whether speculative config was explicit or alias-injected.
- Expose the resolved plan in boot logs and local status.
- Add a model-free fixture for exact qualification rows.
- Reproduce Qwen3.6 and Qwen3.8 current default boot from cached commit-pinned
  artifacts, including Qwen3.8's nested MTP-sidecar resolution.

### B1 — generalize the existing shared-weight native AR lane

- Rename the Qwen3.6-specific wrapper only after behavior-equivalent tests.
- Move geometry/cache eligibility to exact qualification rows.
- Keep Qwen3.6 as the only enabled row initially.
- Convert provenance-bound artifact facts immediately before loading, then
  repeat the same fresh rebind/reprobe after loading and before publishing the
  runtime.
  The B0 receipt is point-in-time only; it does not provide atomic protection
  against a hostile same-user process that mutates and restores cache paths.
- Add template/token parity, interleaved MRoPE, cancellation, lifecycle, and
  failed-candidate cleanup tests.
- Ship as opt-in/canary; do not change alias defaults yet.

### B2 — Qwen3.6 dual MTP feasibility and qualification

- Apply the existing Qwen3.5-family injector to the already-loaded language
  model and start a speculative companion scheduler.
- Prove MTP is genuinely installed before publishing the plan.
- Compare dual MTP, dual native AR, current text-only MTP, and serialized MLLM
  on the same target/drafter revisions.
- If it passes, expose `qualified_auto` as an explicit canary for Qwen3.6 Q4;
  do not change the no-flags default yet.

### B3 — cross-lane memory admission

- Add the engine-owned reservation and retained-cache ledger.
- Give MLLM fail-before-prefill Metal admission.
- Run the mixed media/text pressure campaign on 48 GB before enrolling a
  larger cohort.
- Preserve `/v1/cache/export` and `/v1/cache/import` for the companion text
  lane, or explicitly keep default-on blocked while the MLLM engine rejects
  those APIs.
- Only after those gates pass, enable the no-flags `qualified_auto` default for
  the qualified Qwen3.6 row.

### B4 — enroll Qwen3.8 27B Q4

- Pin and verify the current nested sidecar/head identity first.
- Add its exact 64-layer dense geometry only after cache and injector probes
  pass on the real artifact.
- Qualify on M4 Pro 48 GB and M1 Max 64 GB.
- Default it on only if it retains the current MTP benefit and media parity.

### B5 — evaluate Qwen3.5 4B Q4

- Treat its explicit `is_hybrid=false` alias pin as a safety decision, not a
  stale label. The prior Metal handle-exhaustion wedge must be reproduced as a
  negative regression test.
- Enroll only after a long-chain/abort soak on 48 GB and 64 GB hosts.
- Do not assume the 35B cache row generalizes to the dense 4B model.

Each slice gets its own PR. B0/B1 should remain reviewable and behavior-neutral;
artifact enrollment and default changes must not hide inside refactors.

## Qualification gates

An artifact may enter `qualified_auto` only when all gates pass:

### Correctness and capability

- coding, reasoning, creative, strict JSON, tool calls, and seeded sampling;
- no regression on the tracked subject set used by the existing Qwen3.6
  receipt;
- real screenshot request, immediate text recovery, then another image;
- byte equality where deterministic contracts require it; semantic/checker
  parity where different valid wording is expected;
- processor and tokenizer render identical token IDs for text-only prompts;
- unknown cache leaves, KV quantization, adapters, or unsupported processors
  fail closed to a qualified fallback.

### Performance

- native AR enrollment: paired decode speed at least 1.30x serialized MLLM,
  matching the existing Qwen3.6 gate;
- MTP enrollment: no evaluated workload below native AR and at least 1.15x
  median decode improvement on the target hardware class;
- no median TTFT regression above 10% without a documented user-visible gain;
- text TTFT/decode measured both alone and during a real media prefill to make
  shared-executor head-of-line blocking visible.

### Memory and lifecycle

- one target-weight identity and no model-sized second allocation;
- recorded startup, peak, post-request, and post-cache-clear active memory;
- one 1080p image plus concurrent long text fits the 48 GB qualification host;
- a synthetic over-cap burst produces actionable 503s and does not abort
  unrelated queued requests;
- 50 randomized abort/recovery iterations; both schedulers return to zero;
- pause/resume, cache clear, reload, and shutdown cover both schedulers;
- text-lane cache export/import retains its current API behavior when the
  process also owns an MLLM scheduler;
- interleaved media/text/MTP forwards preserve MRoPE and rollback state.

## Observability and rollout

The first release should add local truth before high-volume telemetry:

- `/v1/status`: resolved plan, selection reason, qualification id, candidate
  fallback reason, per-lane request/cache/admission counters;
- `/metrics`: request count, failure count, cap violations, and active/waiting
  gauges by the closed lane enum;
- PostHog: add at most a closed `runtime_plan` enum to `model_served`, after
  registry/privacy review. Do not add per-request events or prompt-derived
  fields. Do not attach a single request's lane to an aggregate inference
  milestone unless the local counter is also keyed by lane.

Roll out exact artifact by exact artifact: internal flag, opt-in canary,
hardware qualification, then alias default. `auto` must always have an `off`
escape hatch that restores the pre-project lane decision after restart.

## Explicit non-goals

- resuming broad step 3c vendoring;
- loading a second target model;
- enrolling an entire family from structural probes alone;
- moving an active request between schedulers;
- raising hybrid MLLM batching above the qualified singleton policy;
- building a new speculative algorithm, adaptive speculative controller, or
  cross-lane fairness scheduler;
- expanding media prefix caching or rewriting the existing Metal governor;
- Desktop-specific routing based on `caller_agent`;
- treating `/v1/images` generation usage as evidence for VLM image input.

## Consultation record

Claude Fable 5.1 was used for two read-only adversarial reviews. Its useful
findings were the MLLM Metal-admission gap, shared-executor head-of-line risk,
the need for exact artifact qualification, and the warning against mid-request
fallback. Source inspection corrected two review errors before they entered
this design:

- MLLM exact text and media-boundary caches already share one ceiling; the
  independent budget is the companion text scheduler's cache.
- Qwen3.8 27B is `qwen3_5_text`; it is not the `qwen4_exp` Flash-Next model.

## Next session: first concrete action

Implement B0 only. Start from the immutable `QwenRuntimePlan` and provenance
marker for implicit versus explicit speculative selection. Add the offline
Qwen3.6/Qwen3.8 boot truth probes before changing an alias default. Do not
start B1 until B0 can explain, in one status payload, exactly which target,
lane, decoder, qualification row, and fallback the process will use.
