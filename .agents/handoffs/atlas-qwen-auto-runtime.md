# Atlas handoff — Qwen Auto Runtime

- **Owner:** Atlas
- **Branch:** `atlas/qwen-auto-runtime-plan`
- **Base:** `origin/main` at `ec98a7ba2`
- **Host:** Studio; product qualification also requires M4 Pro 48 GB and M1
  Max 64 GB
- **Status:** design complete; no product behavior changed

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
  root target shards plus an accepted nested `mtp/model.safetensors` sidecar.
  A root-only scan is a false negative; the offline probe must cover declared
  nested layouts without reading weights.

## Next concrete action

Implement design slice B0 from
`docs/engineering/design/2026-09-23-qwen-auto-runtime.md`:

1. add the pure `QwenRuntimePlan` and closed reasons;
2. preserve implicit versus explicit speculative selection provenance;
3. expose plan and fallback in local status/boot logs;
4. add exact offline qualification fixtures;
5. reproduce current Qwen3.6 and Qwen3.8 default boot from immutable cached
   artifacts.

Do not change alias defaults in B0. Do not begin MTP dual-lane integration
until the boot truth probes establish the real provider contract.

## Verification plan

- focused resolver and routing unit tests;
- Qwen3.6/Qwen3.8 immutable offline boot probes;
- ruff, mypy budget, `git diff --check`;
- Codex review before implementation PR;
- later product gates on M4 Pro 48 GB and M1 Max 64 GB as specified in the
  design.

## Risk to carry forward

The architecture shares one executor and mutates model-global MRoPE/rollback
state transactionally. Any speculative companion must prove interleaved media
and text correctness. Memory admission must become cross-lane before Qwen3.8
is default-on; two schedulers cannot independently spend the same Metal and
retained-cache headroom.
