# Atlas handoff — Qwen Auto Runtime

- **Owner:** Atlas
- **Branch:** `atlas/qwen-auto-runtime-plan`
- **Base:** `origin/main` at `ec98a7ba2`
- **Host:** Studio; product qualification also requires M4 Pro 48 GB and M1
  Max 64 GB
- **Status:** B0 foundation integrated; production boot/status composition is
  not wired; no product behavior or defaults changed

## Goal

Make qualified vision-capable Qwen aliases choose the best request lane
automatically: media on MLLM, text on shared-weight MTP when qualified and on
shared-weight native AR otherwise. Load target weights once and fail closed to
the current path.

## Decisions

- Step 3c is frozen and is not a dependency.
- Resolve one immutable runtime plan before load and revalidate it against the
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
- Drafter resolution must use and verify the qualification row's immutable
  revision. Passing a mutable repository ID to the provider is forbidden.
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
  root target shards plus a nested `mtp/model.safetensors` sidecar with a
  matching blob/SHA receipt. A root-only scan is a false negative. The truth
  layer reports locator shape without claiming runtime qualification.
- Qwen3.5 4B's pinned snapshot has a real one-shard safetensors index. The
  current MTP locator also returns its root target `model.safetensors` as a
  fallback, so truth classifies that shape as `root_model_ambiguous` and never
  treats it as head eligibility.

## Integrated B0 foundation

- CLI normalization now retains whether speculative selection came from an
  explicit config, legacy flag, alias-owned default, or no selection. Existing
  normalized values and precedence remain unchanged.
- `rapid_mlx.qwen_runtime_plan` provides immutable target/mode identities,
  closed selection and fallback reasons, fail-closed exact qualification,
  mutation-boundary recovery, and fresh verified-target reload proof.
- `rapid_mlx.runtime.qwen_artifact` provides an offline canonical Hub snapshot
  binding, redacted artifact truth/receipts, and the private capability seam
  that alone can mint the planner's verified target identity.
- The production Qwen MTP injector and artifact truth share one dependency-free
  locator/classifier. Exact pinned config/index fixtures cover Qwen3.5 4B,
  Qwen3.6 35B, and Qwen3.8 27B without model loads or tensor reads.
- The foundation is intentionally dormant: no serve/benchmark boot path imports
  the planner, constructs `ResolvedQwenArtifact`, stores a resolved plan on the
  engine, emits plan/fallback boot logs, or publishes it through
  `engine.get_stats()` and `/v1/status`.

## Next concrete action

Add the behavior-preserving B0 composition seam to the existing serve boot:

1. after immutable snapshot resolution, bind and probe the exact local target,
   convert it to `VerifiedQwenTarget`, and construct `ResolvedQwenArtifact`;
2. translate current lane selection plus retained CLI provenance into the
   legacy plan, call `resolve_qwen_runtime_plan`, and retain the resolved plan
   on the engine without changing the selected lane;
3. emit the plan/reason/fallback in boot logs and expose the stored redacted
   payload through `engine.get_stats()` and `/v1/status`;
4. add offline boot-composition tests proving current Qwen3.6 and Qwen3.8
   defaults, explicit flags, and unqualified/raw paths are byte-for-byte
   behavior-equivalent.

Do not change alias defaults in B0. Do not begin MTP dual-lane integration
until this boot composition proves the real provider contract.

## Integration verification

- `219 passed`: runtime-plan, artifact-truth/private-capability, CLI provenance,
  speculative-config, shared locator, and injector/install focused suites.
- The artifact conversion test exercises the integrated core private mint,
  rather than only a stand-in contract.
- Ruff format/check and `git diff --check` pass for the integrated Python diff.

The missing B0 verification is deliberately the same as the missing code:
offline Qwen3.6/Qwen3.8 boot-composition tests through the real serve resolver,
boot log, stored engine plan, and status payload. Later B2/B3 product gates
still require M4 Pro 48 GB and M1 Max 64 GB as specified in the design.

## Risk to carry forward

The architecture shares one executor and mutates model-global MRoPE/rollback
state transactionally. Any speculative companion must prove interleaved media
and text correctness. Memory admission must become cross-lane before Qwen3.8
is default-on; two schedulers cannot independently spend the same Metal and
retained-cache headroom.
