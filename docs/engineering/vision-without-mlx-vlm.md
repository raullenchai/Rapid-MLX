# Vision without mlx-vlm: vendoring the remaining vision surface

Date: 2026-10-02
Status: draft (design only — no code in this change)
Upstream pin: `mlx-vlm==0.7.2` (exact pin in `pyproject.toml`, shared with the
signed Desktop sidecar)
Counts and sizes measured at `9c56a62c1` against the PyPI
`mlx_vlm-0.7.2-py3-none-any.whl`.

## Why

The `[vision]` extra costs ~322 MB (torch + torchvision + opencv + transformers
chains) and exists solely to satisfy `import mlx_vlm`. mlx-vlm 0.7.2 itself
does **not** require torch (verified `requires_dist`: mlx, transformers,
jinja2, sentencepiece, miniaudio, tqdm, Pillow, requests, llguidance,
mlx-audio, opencv, fastapi, starlette, uvicorn, websockets, numpy) — torch
enters through the transitive import chain a real `import mlx_vlm` pulls
(processors/server modules that reach torch-dependent code; see the Doctor
comment in `rapid_mlx/doctor/env_health.py:1991`), so we pin torch defensively.

oMLX's answer is to ship mlx-vlm as a pinned **base** dependency and drop
torch. This project's ruling is different: **remove mlx-vlm entirely** and
vendor the vision surface, extending the approved vendor plan in
`docs/engineering/design/2026-09-18-vendor-mllm-primitives.md` (phases 1–3
already moved the cache/APC/AR/speculative core in-tree). Vendoring decouples
us from upstream's release cadence, brings vision under our review + mypy
budget + golden gates, and avoids mlx-vlm's heavy base chain
(mlx-audio, miniaudio, llguidance, gradio-under-ui).

A prerequisite already landed: on a base wheel (no `[vision]`), text-capable
VLM checkpoints now degrade to the text lane with one warning
(`fix(serve)` @ `93277e801`, PR for branch `fix/vlm-text-degrade`) — Gemma 4 *,
Qwen3-VL/3.5/3.6/3.8 backbones serve via `mlx_lm` 0.31.3 or the vendored
`gemma4_vendored`/`muse_glimmer`/`qwen4_exp` loaders. What remains below is
the **vision-serving** surface: image/video input, the MLLM-lane model
classes, processors, and the spec-decode + diffusion lanes.

## 1. mlx_vlm import inventory (75 sites, 24 files)

Grouped by what removal actually requires. "Hard" = the module cannot be
imported without mlx-vlm installed.

### A. Already vendored (transitional dual-namespace recognition — no action)

| File | Sites | Notes |
|---|---|---|
| `mllm_batch_generator.py` | 4 | vendored-first cache recognition; upstream import is a fallback (`VENDOR-DEVIATION(dual-namespace)`) |
| `engine/batched.py` | 1 | same shim |
| `quantized_batch_cache.py` | 2 | cache-type recognition |
| `mllm_cache_compat.py` | 1 | cache-type recognition |
| `hybrid_state_checkpoints.py` | 1 | `mlx_vlm_vendored.cache` + upstream fallback |

These `except ImportError` fallbacks vanish mechanically once phase 3 unifies
types (one revert each).

### B. Text lane (works without mlx-vlm today)

| File | Sites | Notes |
|---|---|---|
| `models/gemma4_text.py` | 4 | PREFERRED import of `mlx_vlm.models.gemma4{,_unified}` with the vendored `gemma4_vendored/` (120 KB) fallback; text lane is mlx-vlm-free |

### C. Hard vision-path deps (the remaining vendor surface)

| File | Sites | Upstream modules needed |
|---|---|---|
| `models/mllm.py` (the MLLM lane) | 20 | `load`, `load_config`, `generate`/`stream_generate` + `models.cache` shims (already vendored underneath), `prompt_utils.apply_chat_template`/`get_chat_template`, `video_generate.process_vision_info`/`generate`, plus the ABSENT/BROKEN `import mlx_vlm` probe at :375 (this is the guard itself — becomes a vendored-presence probe) |
| `models/prism_hadamard_qwen35.py` (Bonsai 2 pack, MLLM lane) | 5 | `models.qwen3_5.{Model, ModelConfig}`, `utils.{get_model_path, StoppingCriteria}` |
| `patches/glm5_next_*.py` (3 files) | 7 | `models.glm5_next.{language, glm5_next}`, `models.mlp.DeepseekMLP`, `models.base`, `models.deepseek_v32.language`, `prompt_utils.{MODEL_CONFIG, MessageFormat}` |
| `speculative/native_mtp/*` (4 files) | 13 | `models.cache` (vendored), `generate.ar` (vendored 3a), `speculative.drafters.*` (phase 3c), `models.glm5_next.language`, `models.linear`, `utils.get_model_path`, `load` |
| `speculative/dflash/*` (2 files) | 6 | `load`, `generate`, `stream_generate`, `prompt_utils.apply_chat_template`, `speculative.drafters`, `utils.get_model_path` |
| `spec_decode/dspark/*` (2 files) | 3 | `load`, `speculative.drafters`, `prompt_utils.apply_chat_template` |
| `routes/health.py` | 2 | `utils.{load_config, get_model_path, ...}` for cache-clear/health probes |
| `runtime/diffusion_lane.py` | 3 | `generate.diffusion.{...}`, `utils.load` (diffusion-LM streaming) |
| `image/hidream_runtime/runtime.py` | 1 | `mlx_vlm.load` (image-gen lane helper) |
| `benchmark.py` | 2 | cv2 import + an mlx_vlm probe for image mode |

### D. Probes and strings (repoint, don't vendor)

| File | Sites | Notes |
|---|---|---|
| `doctor/env_health.py` | 1 (+:1991 comment) | `"mlx-vlm": "mlx_vlm"` extra-probe mapping; becomes "vendored vision present" check after removal |
| `models/mllm.py:375` | (counted above) | the REAL `import mlx_vlm` status probe — after removal this probes the vendored package |

**Net:** the true vendor surface is ~50 hard sites concentrated in 8 files
(`mllm.py` alone is 20), plus the per-family model classes/processors those
sites load from upstream.

## 2. torch usage in `[vision]`

- **Our serving code never imports torch.** Grep across `rapid_mlx/`: the only
  `import torch` sites are offline conversion tools
  (`system_one/convert_clm.py:123`, `models/deepseek_v41_native/convert.py:240`)
  and an audio checkpoint-security probe (`audio/sa3/.../checkpoint_security.py:20`)
  — none on a serve path. All other "torch" hits are comments
  (`torch_dtype` config strings, torch-reference notes in vendored loaders).
- **`import mlx_vlm` does pull torch transitively** (env_health.py:1991
  documents this), which is why `torch>=2.3.0` + `torchvision>=0.18.0` are
  pinned in `[vision]` since the extra was introduced (e5b029b1d).
- **A vendored vision path needs no torch**, mirroring oMLX: mlx-vlm's own
  processors bypass HF `AutoProcessor`, and the vendored slices we need
  (`models.cache`, `generate.ar`, apc, inputs, kv_quant) are mlx + stdlib
  only. The one genuine native-image dependency is **opencv** for video frame
  decode (`mllm.py:1465`, `benchmark.py:47`) — keep `opencv-python` in the
  extra (or gate it behind a `[video]` slice) and drop torch/torchvision.
- Verification gate for the torch removal: `import` the vendored vision
  package in a venv with **no torch installed** and run the per-family golden
  suite; Doctor's torch row flips to "not required" for vision serving.

## 3. Licence and attribution

mlx-vlm is **MIT**; its `LICENSE` is already vendored verbatim in
`rapid_mlx/models/mlx_vlm_vendored/` under the provenance contract
(`__init__.py`: verbatim copies except documented import-redirect hunks,
per-file sha256 against the 0.7.2 tag, `diff`-reviewable). Extending it:

1. Copy upstream files verbatim; keep their headers.
2. Record per-file sha256 + hunk list in the slice's `__init__.py`
   (the existing contract, one section per slice).
3. Add the slice to the vendor table in
   `2026-09-18-vendor-mllm-primitives.md` and to this doc's tracking table.
4. Ship the upstream MIT `LICENSE` (already in-tree) — no NOTICE change
   needed beyond naming mlx-vlm and its copyright holders in the slice docs;
   our distribution already carries the license text.
5. The `VENDOR-DEVIATION(...)` marker convention stays the only permitted
   edit class, enforced by a diff test against the recorded hashes.

## 4. Size estimate (measured from the 0.7.2 wheel, uncompressed)

| Slice | Size |
|---|---|
| Already vendored (`models/mlx_vlm_vendored/` on disk) | 1.8 MB |
| Already vendored (`models/gemma4_vendored/`) | 120 KB |
| `models/qwen3_5/` (Bonsai pack + Qwen3.5 VL) | 118 KB |
| `models/gemma4/` (vision classes; text side done) | 149 KB |
| `models/qwen3_vl/` | 90 KB |
| `models/gemma4_unified/` (vision side) | 36 KB |
| `models/glm5_next/` + `deepseek_v32/` + shared `mlp/mla/linear/switch_layers` | ~250 KB (est.) |
| `generate/` (vision dispatch + diffusion; AR core done) | 321 KB total |
| `utils.py` + `tokenizer_utils` + `prompt_utils` | 154 KB |
| `speculative/` (drafters; coordinator done) | 485 KB total |

**Estimated new vendor: ~2–3 MB of Python** across 6–10 family slices —
against the ~322 MB `[vision]` extra today. Dropping torch/torchvision
(~200 MB installed) and keeping only opencv + pillow leaves a vision extra of
roughly 90–120 MB. `models/` totals 8.5 MB upstream; we vendor only served
families, so unknown arches keep failing closed exactly as today
(`_text_lane_loads_model_type` / `_VENDORED_TEXT_FALLBACK_MODEL_TYPES`).

## 5. Test strategy

- **Golden outputs per family** (the #1247 GOLDEN gate pattern): for each
  vendored family, a fixed prompt set (`tests/golden_prompts.py` L1/L2) at
  fixed seed through the real checkpoint, comparing token IDs and logprobs
  against goldens captured from the pinned upstream 0.7.2 lane. One golden
  file per (family, checkpoint, prompt id); re-captured at every pin bump.
  Existing per-slice tests to mirror: `test_mlx_vlm_vendored_cache.py`,
  `test_mlx_vlm_vendored_generate.py`, `test_gemma4_unified_routing.py`,
  `test_qwen4_exp_vendored.py`.
- **Structural (no-MLX, CI-fast):** config-parity tests (vendored `TextConfig`
  defaults vs upstream), weight-shim parity (state-dict rename tables),
  import-guard tests (`test_gemma4_text_import_guard.py` pattern), and the
  sha256 verbatim-diff test from the provenance contract.
- **L1/L2 coherence smoke:** the existing `--no-mllm` Gemma-4/Qwen3.5/Bonsai
  lanes extend to vision-on-vendored (image prompt → golden first tokens),
  keeping the Desktop sidecar byte-identical to the CLI lane.
- **Probe-mode:** vision requests on a base wheel keep the existing
  capability rejection contract (image → `capability_rejected` event, video →
  `video_input_unsupported`), unchanged by this plan.

## 6. Migration order (extends the 2026-09-18 phases)

1. **4a — templating/processors:** `prompt_utils`, `tokenizer_utils`,
   `MODEL_CONFIG`/`MessageFormat` (glm5_next processor patch). No model
   classes yet; unblocks every chat path.
2. **4b — per-family model classes,** one slice per PR under the review-diff
   cap: qwen3_5 (unblocks the Bonsai 2 pack on the MLLM lane) → glm5_next +
   deepseek_v32 → gemma4 vision + gemma4_unified vision → qwen3_vl →
   minimax_m3_vl / unlimited_ocr / quantized_verifier as served.
3. **4c — generate/ vision loop:** `stream_generate`/`generate` vision
   dispatch + `video_generate` + the cv2 boundary in `mllm.py`;
   `models/mllm.py`'s 20 import sites collapse to vendored imports.
4. **4d — probes/lanes:** `routes/health.py`, `doctor/env_health.py`,
   `benchmark.py`, `diffusion_lane.py`, `hidream_runtime`, the spec-decode
   drafters (3c leftovers), and the mllm.py:375 status probe re-pointed to
   vendored presence.
5. **4e — dependency flip:** drop `mlx-vlm` (and torch/torchvision) from
   `[vision]`; keep opencv + pillow. Desktop sidecar rebuild in the same
   release (signed pin shared with CLI). Telemetry-gated: any regression
   reverts to the byte-verbatim upstream pin for one release.

Each slice ships behind the dual-namespace shim so upstream stays authoritative
until the slice's goldens pass, then the redirect flips.

## 7. Risks

- **Upstream drift:** new model families (qwen4_exp, muse_glimmer …) arrive
  upstream faster than we vendor; mitigation — the fail-closed arch gate means
  unsupported arches keep the loud `[vision]`-required error, never silent
  breakage. The fail-closed degrade predicate
  (`checkpoint_serves_text_without_vision`) already embodies this.
- **Processor edge cases:** chat templates and image-preprocessing vary per
  family and are the most behavior-sensitive code to vendor; 4a lands first
  precisely so goldens exercise templates before any model class moves.
- **Video/opencv:** cv2 stays a hard video dependency; a later `[video]`
  slice could vendor a minimal decoder, but that is out of scope here.
- **Sidecar parity:** Desktop ships the same vendored tree; a mismatch is a
  coherence-gate failure, same as today's shared `mlx-vlm==0.7.2` pin.
- **mypy budget:** vendored files enter the reviewed baseline
  (`config/mypy-error-baseline.txt`) like the existing 1.8 MB slice did.
- **Review-diff cap:** verbatim copies + documented hunks only; enforced by
  test, not review vigilance.

## 8. Upgrade tracking

| Slice | Upstream files | Vendored PR | Goldens captured at |
|---|---|---|---|
| cache/APC/AR/inputs/kv_quant/fp8/sample_utils/base/linear | 0.7.2 | #3545, #3554, #3558, #3563, #3566, #3575 (see 2026-09-18 doc) | existing vendored suites |
| gemma4 text (nonunified + unified/assistant) | 0.7.2 | `models/gemma4_vendored/` | L1/L2 text lanes |
| muse_glimmer, qwen4_exp text | 0.7.2 | `models/{muse_glimmer,qwen4_exp}.py` | per-family vendored tests |
| qwen3_5 | 0.7.2 | phase 4b (this plan) | TBD |
| glm5_next + deepseek_v32 | 0.7.2 | phase 4b | TBD |
| gemma4/gemma4_unified/qwen3_vl vision | 0.7.2 | phase 4b | TBD |
| prompt_utils/tokenizer_utils | 0.7.2 | phase 4a | TBD |
| generate/ vision + video | 0.7.2 | phase 4c | TBD |

Pin-bump procedure: bump the recorded sha256 set, re-run per-family goldens
against the new upstream lane, review the verbatim diff, tighten the mypy
baseline. A slice whose goldens cannot be reproduced fails the bump.

## 9. oMLX comparison

| | oMLX | This plan |
|---|---|---|
| mlx-vlm | pinned **base** dependency | **removed**, vendored |
| torch | not shipped (custom processors bypass HF AutoProcessor) | dropped from `[vision]` (same reasoning, verified above) |
| New upstream families | free on bump | require a vendored slice (fail-closed until then) |
| Patch independence | none (upstream cadence) | full (VENDOR-DEVIATION process) |
| Install size | mlx-vlm base chain (transformers, mlx-audio, miniaudio, llguidance, …) | +2–3 MB in-tree Python; opencv + pillow only |
| Owner ruling | n/a | mlx-vlm removed — hence vendor, not pin |
