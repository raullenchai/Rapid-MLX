# Atlas handoff — Qwen Auto Runtime

- **Owner:** Atlas
- **Branch:** `atlas/b0-object-identity-fix`
- **Base:** B0 integration at `3fa352de5`
- **Host:** Studio; product qualification also requires M4 Pro 48 GB and M1
  Max 64 GB
- **Status:** B0 foundation and behavior-neutral boot/status composition are
  complete; no product behavior or defaults changed

## Goal

Make qualified vision-capable Qwen aliases choose the best request lane
automatically: media on MLLM, text on shared-weight MTP when qualified and on
shared-weight native AR otherwise. Load target weights once and fail closed to
the current path.

## Decisions

- Step 3c is frozen and is not a dependency.
- Resolve one frozen runtime plan before load and revalidate it against the
  loaded artifact.
- Exact artifact receipts, not family-name probes, enable automatic behavior.
- Preserve explicit flag semantics in the first release. Only alias-owned
  implicit defaults may be reordered by the auto planner.
- Explicit `--mllm` preserves the current qualified shared-weight native AR
  text companion; it is not redefined as serialized MLLM-only operation.
- B2 exposes dual MTP only as an explicit canary. No-flags `qualified_auto`
  remains disabled until B3 cross-lane admission and pressure gates pass.
- Startup fallback is allowed; mid-request replay across lanes is forbidden.
- MTP injection is a mutation boundary. A post-injection startup failure must
  reload a clean target or abort; it cannot fall back on the mutated instance.
- Drafter resolution must use the qualification row's commit-pinned revision.
  Passing a mutable repository ID to the provider is forbidden.
- MLLM text/media caches already share a ceiling. The memory gap is between
  MLLM and the companion text scheduler, plus MLLM's missing projected Metal
  admission.
- Default-on also requires cache-persistence parity: the current MLLM guard on
  `/v1/cache/export` and `/v1/cache/import` cannot erase the companion text
  lane's existing API when a dual runtime is selected.

## Verified facts

- Qwen3.6 Q4 already has a shared-weight native AR text lane: 1.482x decode
  versus serialized MLLM, +17 MB, no second target copy.
- Current MTP default selection forces vision-capable aliases to a text-only
  process before the shared-weight lane can start.
- Cached exact configs identify priority aliases as Qwen3.5-family models:
  Qwen3.5 4B `qwen3_5_text`, Qwen3.6 35B `qwen3_5_moe_text`, Qwen3.8 27B
  `qwen3_5_text`.
- Qwen3.8 27B is not the `qwen4_exp` Flash-Next path.
- The cached Qwen3.8 target/declared-sidecar snapshot at `aa985c2…` has three
  root target shards plus a nested `mtp/model.safetensors` sidecar whose
  declared SHA-256 matches its HF cache-object name. That is provenance, not a
  recomputation over tensor bytes. A root-only scan is a false negative. The
  truth layer reports locator shape without claiming runtime qualification.
- Qwen3.5 4B's pinned snapshot has a real one-shard safetensors index. The
  current MTP locator also returns its root target `model.safetensors` as a
  fallback, so truth classifies that shape as `root_model_ambiguous` and never
  treats it as head eligibility.

## Integrated B0 foundation

- CLI normalization now retains whether speculative selection came from an
  explicit config, legacy flag, alias-owned default, or no selection. Existing
  normalized values and precedence remain unchanged.
- `rapid_mlx.qwen_runtime_plan` provides commit-pinned target/mode provenance,
  closed selection and fallback reasons, fail-closed exact qualification,
  mutation-boundary recovery, and fresh verified-target reload proof.
- `rapid_mlx.runtime.qwen_artifact` provides an offline canonical Hub snapshot
  binding, redacted artifact truth/receipts, and the private capability seam
  that alone can mint the planner's verified target identity.
- The production Qwen MTP injector and artifact truth share one dependency-free
  locator/classifier. Exact pinned config/index fixtures cover Qwen3.5 4B,
  Qwen3.6 35B, and Qwen3.8 27B without model loads or tensor reads.
- The production seam remains behavior-neutral: it constructs only the exact
  legacy plan after final lane/companion startup, passes `auto_enabled=False`,
  registers no qualification rows, and never constructs
  `ResolvedQwenArtifact`.

## B0 boot/status composition

- CLI provenance maps into closed `SpeculativeIntent` values without changing
  normalization. `load_model` captures only the original explicit lane before
  alias and automatic routing mutate the effective flags.
- `BatchedEngine.start()` records the final text/vision lane and optional
  Qwen3.6 shared-weight companion. Qwen classification uses exact loaded/local
  config model-type pairs, never names.
- Text MTP status comes from the completed boot dispatch receipt plus the same
  config-vetted profile gate used by the lazy scheduler. It does not eagerly
  create a request generator or read the scheduler field that remains unset
  until first inference.
- The frozen plan describes the selected decoder. A separate closed
  `qwen_runtime_activation` reports the current lazy generator as
  `pending_first_request`, `active`, `fallback_native_ar`, or
  `not_applicable`; closing/recreating the generator can return it to pending.
- One frozen legacy plan is logged and stored per successful boot. Stop and
  reload clear and recompute the plan, optional artifact truth, snapshot source,
  and MTP dispatch receipt. Teardown clears them before fallible scheduler
  cleanup, so a failed stop cannot expose stale facts.
- `engine.get_stats()` and `/v1/status` expose only the JSON-safe plan and an
  optional redacted artifact truth. Truth is published only when the exact
  already-selected source revalidates as canonical, commit-pinned Hub
  provenance;
  a programmatic canonical HF-cache snapshot path derives its repo id from the
  validated cache entry, while arbitrary local paths and mutable repo ids omit
  truth.

## B0 trust-boundary correction

- Config and index metadata fail closed unless a 40-hex cache-object name
  equals Git blob SHA-1 or a 64-hex name equals raw SHA-256 for the bytes the
  probe already read.
- Every target shard and MTP object binds its canonical HF cache-object name and
  observed size into receipts and qualification identities. Repointing,
  renaming, truncation, and size drift are rejected by fresh revalidation.
- Status and qualification state expose `tensor_byte_integrity: unchecked`.
  B0 never opens tensor content and trusts the Hugging Face downloader/cache
  CAS invariant. It cannot detect pre-existing, restored, or equal-size tensor
  corruption under the same cache-object name.
- Optional `.sha256` declarations are read only from bounded regular files or
  canonical same-repository Hub cache objects. Arbitrary receipt symlinks,
  candidate-tensor inode aliases, and files over 4 KiB are rejected before
  content reads.
- Serialized status calls the stable digest a `provenance_receipt_id` and its
  issuer a `provenance_authority`. The private `VerifiedQwenTarget` /
  `verification_id` composition names remain temporarily for stacked B1 branch
  compatibility; their docstrings and values now define provenance only, not
  tensor-byte verification.
- The production route remains behavior-neutral: auto selection is disabled
  and no qualification rows are registered.

## Next concrete action

Integrate this B0 commit, then use emitted legacy plans/truth to validate the
provider contract before enabling any qualification row or automatic lane
selection. Do not change alias defaults or begin MTP dual-lane integration
until that evidence is reviewed.

## Integration verification

- `562 passed`: runtime-plan, artifact-provenance/private-capability,
  boot-plan/status, CLI provenance, speculative-config, shared locator,
  injector/install, batched-family, and MTP CLI focused suites.
- `310 passed` under focused coverage; exact changed-production line coverage is
  100% for `qwen_runtime_plan.py`, `runtime/qwen_artifact.py`, and the receipt
  extractor.
- The artifact conversion tests exercise the integrated core private mint,
  target/MTP cache-object repoints, renames, truncation/size drift, metadata
  object-name mismatches, the tensor-open guard, and the explicit same-size
  non-guarantee.
- Ruff format/check, targeted pinned mypy 2.3.1, and `git diff --check` pass.

Later B2/B3 product gates still require M4 Pro 48 GB and M1 Max 64 GB as
specified in the design.

## Risk to carry forward

The architecture shares one executor and mutates model-global MRoPE/rollback
state transactionally. Any speculative companion must prove interleaved media
and text correctness. Memory admission must become cross-lane before Qwen3.8
is default-on; two schedulers cannot independently spend the same Metal and
retained-cache headroom.
