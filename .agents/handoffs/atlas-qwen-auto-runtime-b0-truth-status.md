# Qwen Auto Runtime B0 artifact truth/status handoff

- Receiving role: Atlas
- Branch: `atlas/qwen-auto-runtime-b0-truth-status`
- Base: `ec98a7ba221e0c9ced3c38fe0f064717bb7d9a1d` (`origin/main`)
- Host: Studio

## Completed scope

- Added an offline `probe_qwen_artifact` helper. It reads only an
  already-resolved `config.json`, a safetensors index, and file metadata. It
  never imports/loads a model, reads tensor bytes, contacts the network, or
  returns an absolute path.
- Added complete config fixtures copied from the pinned Qwen3.5-4B,
  Qwen3.6-35B, and Qwen3.8-27B cached revisions plus content-free snapshot
  shape manifests.
- Added closed artifact facts for model types, exact geometry, MTP head count,
  quantization overrides, target shard completeness, and today's MTP locator.
- Added an explicit resolver trust boundary. A 40-hex revision supplied by a
  caller is `declared_immutable_unverified`; only a resolver that bound this
  exact directory to source and revision may set `provenance_verified=True`.

## Verified facts

- Qwen3.8-27B revision `aa985c29…` is `qwen3_5` / `qwen3_5_text`, 64 layers;
  it is not `qwen4_exp`.
- That snapshot's target is a three-shard indexed checkpoint and its
  `mtp/model.safetensors` is an HF-cache symlink accepted by the current
  `_find_mtp_weights_file` contract without reading its contents.
- The same root target-shard layout with the nested MTP file removed is not
  accepted as an MTP sidecar.
- Qwen3.6-35B has four target shards, 40 layers, 256 experts, and 80
  per-module quantization overrides in its exact config.
- Qwen3.5-4B's pinned snapshot has an index whose weight map names the single
  root `model.safetensors`; it is therefore `indexed_safetensors` with one
  shard, not an index-free single-file layout.

## Product bugs / risks found

- `_find_mtp_weights_file` accepts a root `model.safetensors` by filename
  alone. On the Qwen3.5-4B target snapshot this is the target trunk, not proof
  of an MTP head. The probe reports this as `root_model_ambiguous`; a runtime
  qualification row must not treat it as an accepted head without stronger
  artifact provenance/content-shape validation.
- Existing HF sidecar resolution can download a mutable repo id without an
  immutable revision. B0 truth deliberately reports repo-only and unverified
  40-hex declarations as non-immutable.

## Status integration decision

`/v1/status` already has a clean future seam: it builds its payload from
`engine.get_stats()`. No engine/runtime-plan object owns verified artifact
provenance yet, so this slice does not add a misleading placeholder or derive
identity from `cfg.model_name`. Integrate `truth.to_status_dict()` only after
the B0 RuntimePlan/resolver stores the trusted truth on the engine.

## Next concrete action

Cherry-pick this slice into the B0 integration branch, then have the resolver
construct the probe with source, immutable revision, and its explicit verified
binding. Publish that stored object through the existing status stats seam;
do not re-probe from a mutable repo name in the route.
