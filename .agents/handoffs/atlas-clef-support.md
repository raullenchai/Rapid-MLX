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
- An earlier sandboxed session could not access Metal or the Hugging Face
  cache. A later unrestricted session on the same M3 Ultra confirmed MPS and
  loaded the real pinned 9B checkpoint. Text, image, video-frame, and rank
  requests all returned 200; see
  `docs/engineering/performance/2026-10-03-clef-flash-m3-ultra-dogfood.md`
  for inputs, outputs, timings, and memory.
- Local Clef tests pass (7/7); repository Ruff lint/format and the pinned
  shrink-only mypy budget pass after CI fixes. The optional Torch tests skip
  in the default test matrix when the `[clef]` extra is absent.

## Remaining qualification

- The 27B Clef checkpoint has not been downloaded or run. Cloudflare only
  validated its reference runtime on H200; one successful M3 Ultra run is not
  a Mac hardware qualification matrix.
- Synthetic video color frames work, but a two-frame order-reversal probe
  was ambiguous. Real video decision quality and concurrency remain unmeasured.
- The PR remains draft until its changed-lines coverage gate is addressed;
  the pinned vendor source is currently counted as uncovered by that gate.
