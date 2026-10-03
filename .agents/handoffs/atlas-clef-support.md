# Clef System One integration handoff

- Receiving role: Atlas
- Branch: `atlas/clef-support` (based on `origin/main` at `f4ffa41f`)
- PR: https://github.com/raullenchai/Rapid-MLX/pull/4061 (draft)
- Scope: Cloudflare Clef and Clef-Flash as a System One backend, using the
  official joint schema head on Torch/MPS. No deployment or release change.

## Verified facts

- Both Cloudflare models publish the same `joint_schema_model.py` source
  (SHA-256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`).
  Their release revisions are pinned in `ClefBackend`.
- The model card specifies Apache-2.0 weights/code, a Jev/SystemOne compatible
  `/v1/systemone` response, and text, image, and video input. The official
  runtime was validated on an H200 with Torch 2.11 and Transformers 5.10.2.
- The new tests exercise the route contract, media bounds, pinned checkpoint
  selection, and a genuine forward pass through the official joint head on a
  tiny CPU backbone. The wheel contains the vendored source, license, and notice.

## Remaining validation

- Run `rapid-mlx system-one clef-flash` on a Mac with Metal access and a
  writable default Hugging Face cache. This sandbox has no Metal device and
  cannot write the required default cache, so no full-weight inference was
  claimed. The 9B Flash snapshot is about 19.1 GB on Hugging Face.
- Send one text `noul`/`choice`/`score` request, one PNG data URL request,
  and one frame-array video request; compare probabilities with Cloudflare's
  reference script on the same checkpoint. Record memory, load time, and
  latency before declaring Mac production support. Then repeat with 27B only
  if Flash is sound.
