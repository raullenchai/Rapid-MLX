# Qwen Auto Runtime B0 artifact truth/status handoff

- Receiving role: Atlas
- Branch: `atlas/qwen-auto-runtime-b0-truth-status`
- Base: `ec98a7ba221e0c9ced3c38fe0f064717bb7d9a1d` (`origin/main`)
- Host: Studio

## Completed scope

- `probe_qwen_artifact` remains offline and content-free. It reads canonical
  JSON config/index metadata, optional sidecar `.sha256` receipts, and file
  metadata; it never downloads, imports/loads a model, or opens tensor bytes.
- `verify_hub_snapshot_binding` is the sole provenance mint. It validates the
  configured Hugging Face cache root, canonical repo id, exact full 40-hex
  snapshot revision, exact resolved directory, and optional canonical
  subfolder. Its binding constructor is private and arbitrary temp paths,
  mutable/spoofed revisions, URLs, userinfo, and path-like sources fail closed.
- Status is redacted: only a validated canonical Hub repo/revision/subfolder
  can be emitted. Local paths and unvalidated source strings are never echoed.
- `QwenArtifactTruth` is private-token minted. Only a probe whose resolver
  binding matches the exact snapshot retains the capability required to mint
  a runtime target; direct construction and `dataclasses.replace` cannot forge
  verified repo/revision/receipt fields.
- Subfolder containment is anchored to the resolved requested revision root.
  Blob/sibling-revision directory escapes and a revision entry symlinked to a
  sibling commit fail closed, while a symlinked repo-cache ancestor remains
  supported.
- Target receipts preserve the canonical index digest, extracted shard set,
  missing-shard state, full ordered layer layout, and—where available—the
  nested MTP symlink/blob id, declared SHA-256, and file size.
- `scripts/extract_qwen_artifact_receipt.py` reproduces the receipt against an
  existing canonical Hub snapshot without network or tensor reads.
- The dependency-free `rapid_mlx.qwen_artifact_layout` owns current locator
  precedence and closed path classification. Both the production injector and
  artifact truth use it, preventing duplicated locator behavior.
- `to_verified_runtime_target` owns the fail-closed mapping to the core
  target-only identity, including full ordered `layer_types`, quantization,
  target-weight receipt, and cache geometry. It mints the opaque verified
  wrapper only through core's private `_mint_verified_qwen_target` seam with
  authority `rapid_mlx.qwen_artifact:hub-snapshot-v1`; no caller reconstructs
  target identity ad hoc.

## Verified facts

- Qwen3.8-27B revision `aa985c29…` is `qwen3_5` / `qwen3_5_text`, not
  `qwen4_exp`. Its target is a three-shard indexed checkpoint and its nested
  `mtp/model.safetensors` is an HF-cache symlink with a matching blob/SHA
  receipt. Removing that nested file leaves no MTP locator candidate.
- Qwen3.6-35B is a four-shard indexed checkpoint with 40 ordered layers, 256
  experts, and 80 per-module quantization overrides in the exact config.
- Qwen3.5-4B's pinned snapshot has a real index whose weight map names the one
  root `model.safetensors`; its canonical index digest is retained in the
  fixture receipt. It is therefore a one-shard `indexed_safetensors` target.
  The current production locator also returns that target trunk as its final
  fallback.

## Product bug / risk

- `_find_mtp_weights_file` accepts a root `model.safetensors` by filename
  alone. For Qwen3.5-4B this is the target trunk, not an MTP head. Truth reports
  `root_model_ambiguous`, attaches no head-like receipt, and exposes no
  `accepted`/`eligible` field. Runtime qualification must categorically reject
  this state until stronger sidecar evidence exists.

## Status integration decision

`/v1/status` already has a future seam through `engine.get_stats()`, but the
current engine does not yet own the resolved RuntimePlan and verified artifact
binding. This slice does not fabricate a placeholder from `cfg.model_name`.
Publish the stored truth only when the integration resolver owns it.

## Verification

- `343 passed`: artifact truth plus MTP self-contained locator, injector,
  batched-family capability, CLI wiring, and spec-decode suites.
- Ruff passes on all changed Python files; `git diff --check` passes.
- The conversion seam was exercised directly against corrected core commit
  `e381ee37491e958967d514c7643758348777d957`; private minting succeeds and the
  public `VerifiedQwenTarget` constructor rejects.

## Next concrete action

Cherry-pick the follow-up after the initial truth commit and corrected core,
then wire the resolver to retain the resulting `VerifiedQwenTarget`. Re-mint a
fresh matching wrapper after reload; do not persist a boolean proof or derive
identity from a mutable repo-only locator.
