# Vendoring the mlx-vlm cache/APC/vision primitives into the native MLLM lane

Date: 2026-09-18
Status: approved (user ruling: follow the original Phase B plan)
Upstream pin: `mlx-vlm==0.7.1` (exact pin in `pyproject.toml`, shared with the
Desktop sidecar)

## Why

The native serialized lane (`MLLMBatchGenerator`, the serving path since the
runtime-independence PRs) is functionally and performance-wise the primary
generation path, but it still leans on five mlx-vlm modules for cache and
vision primitives. Every lane-level feature PR (#3496 sampling parity,
#3502 singleton no-rebatch, #3505 media prefix cache, #3534 legacy-generation
retirement) has had to reason about behavior of code we do not own and cannot
patch independently of upstream's release cadence. Vendoring brings those
primitives under this repo's review, tests, and mypy budget while keeping the
upstream pin for what remains (model loading, processors, templating, and the
speculative-decode runtime).

This is Phase B **step 2** of the mlx-vlm dependency retirement plan:

1. ✅ Step 1 — retire the legacy generation surface (#3534): benchmark on the
   native lane, `MLXMultimodalLM.generate/stream_generate/chat/stream_chat`
   deprecated.
2. **Step 2 (this doc) — vendor the cache/vision primitives.**
3. Step 3 — vendor model implementations + the speculative-decode runtime
   (`mlx_vlm.generate.ar`, `mlx_vlm.speculative.drafters`, model-class hooks).
   The dflash/native_mtp servers move off mlx-vlm **here**, not earlier: their
   `generate/stream_generate` calls are the speculative engine itself
   (`draft_model`/`draft_kind` ride into `mlx_vlm.generate`; the DFlash hooks
   live on mlx-vlm model classes), so they cannot move before step 3 exists.

## What gets vendored (upstream 0.7.1, verbatim unless noted)

| Upstream module | Lines | Deps | Used by |
|---|---|---|---|
| `models/cache.py` | 3,615 | mlx + stdlib only | `mllm_batch_generator.py` (ArraysCache/KVCache eligibility + detached extraction), `engine/batched.py` |
| `apc.py` | 4,994 | mlx, numpy, stdlib, + `apc_coordinator`/`apc_storage`/`kv_quant`/`_stream_cleanup` | `mllm_batch_generator.py` (prefix cache engine) |
| `apc_coordinator.py` | 251 | stdlib | apc |
| `apc_storage.py` | 96 | stdlib | apc |
| `kv_quant.py` | 186 | stdlib top-level, **lazy `.turboquant`** (7k lines, imports `.models.cache`) | apc |
| `_stream_cleanup.py` | 10 | mlx | apc |
| `apc_adapters.py` | 795 | mlx, stdlib, lazy `from .apc` | `mllm_batch_generator.py` (`clone_cache_entry`, `Capability`, `resolve_capability`) |
| `vision_cache.py` | 81 | mlx, stdlib | `mllm_batch_generator.py` (`VisionFeatureCache`) |
| `utils.py::prepare_inputs` + helpers | ~900 | mlx, numpy, PIL; lazy cv2/audio imports | `mllm_batch_generator.py` (prompt→model inputs) |

Total ≈ 11k lines. NOT vendored (stays on the pinned dependency): model
classes, `load`/`load_config`, `prompt_utils` templating, the speculative
runtime.

## PR split (tooling-driven, revised per design review)

`pr_validate`'s codex review caps the diff at 200 KB; the full surface is
~400 KB+. A design consultation (adversarial review of the type-namespace
problem below) revised the original cache-first split into a
**types → engine → adapters** ordering. The controlling observation: the
dangerous interaction is not file size but *dispatch* — upstream
`apc_adapters.clone_cache_entry` and friends dispatch on exact upstream type
objects, so a vendored-typed cache flowing into an upstream-dispatching
consumer silently disables fast paths (measured: `_snap_exact_text_prefix`
returns `None`, 7 tests fail). Ordering the PRs so that recognition is
widened before production is moved keeps every slice behavior-neutral.

- **2a — types + seam** (this PR): vendor `models/cache.py` verbatim; widen
  every lane recognition site to a three-namespace union (vendored, upstream
  mlx-vlm, mlx-lm) via `mllm_cache_compat`; dual-namespace contract tests. No
  producer emits vendored-typed caches yet, so behavior is unchanged by
  construction.
- **2b — engine**: vendor `apc.py` + coordinator/storage/kv_quant/
  _stream_cleanup (~5.4k lines alone, hence its own PR). `kv_quant` lands
  here, not with the cache types: its `from_legacy()` lazily imports
  `.turboquant` (7k lines with its own `.models.cache` dependency), so the
  dependency needs an explicit home — a vendored slice if it fits the
  review-diff cap, otherwise a documented redirect to the pinned upstream.
- **2c — adapters + vision + inputs**: vendor `apc_adapters.py` (its
  `clone_cache_entry` dispatch moves to the resolver), `vision_cache.py`,
  `prepare_inputs` + helpers; contract test: an upstream-typed cache
  vendored-clones into a real cache and vice versa.

Deviations from verbatim are allowed **only** in lane dispatch tuples and
are marked `# VENDOR-DEVIATION(dual-namespace):` so a grep finds them all;
step 3 ends the transition with one mechanical revert to byte-verbatim
(gated on a telemetry counter for upstream-namespace caches reaching zero).

## Type-namespace duality (the design problem this solves)

Upstream mlx-vlm 0.6.4+ ships cache classes that are structurally identical
to mlx-lm's but are distinct class objects; the repo already carries
`mllm_cache_compat.first_incompatible_mllm_cache_type` to bridge that split.
Vendoring creates a *third* namespace: the vendored module re-executes
`cache.py`, so `VendoredKVCache is not mlx_vlm.models.cache.KVCache`. Every
exact-type dispatch site (`type(leaf) in qualified`, clone/extract
dispatches) must therefore accept all three namespaces during the
transition. Rejected alternatives: a startup monkeypatch aliasing the
vendored types onto upstream's (silent-failure risk, breaks cross-process
persistence) and deferring the dispatch-sensitive adapters to last while
flipping tests early (the exact failure measured above).

## Layout and provenance contract

```
rapid_mlx/models/mlx_vlm_vendored/
  __init__.py          # provenance header, upstream tag, redirect inventory
  cache.py             # 2a
  apc.py               # 2b
  apc_coordinator.py   # 2b
  apc_storage.py       # 2b
  kv_quant.py          # 2b (lazy .turboquant dep — see PR split)
  _stream_cleanup.py   # 2b
  apc_adapters.py      # 2c
  vision_cache.py      # 2c
  inputs.py            # 2c (prepare_inputs + helpers from utils.py)
```

Rules (modeled on the `gemma4_vendored` precedent):

- Copies are **verbatim** from the upstream 0.7.1 tag except *import
  redirects* and *in-source `VENDOR-DEVIATION(upstream-bugfix)` hunks* for
  defects that reproduce against the pinned upstream, each documented in the
  module body and in the package `__init__.py` inventory. No other logic
  edits, no formatting, no type-annotation drives. Any future behavior
  change must re-qualify against upstream (diff against the pinned tag must
  show only the documented hunks).
- The `__init__.py` header lists upstream tag + file hashes so CI could later
  add a provenance check.
- All former `from mlx_vlm...` imports in the lane redirect to
  `rapid_mlx.models.mlx_vlm_vendored`.

## Verification

- Mechanical: `diff` vendored file vs upstream 0.7.1 → only the documented
  redirect hunks.
- Behavioral: the vendored classes must be drop-ins. Lane unit tests
  (`test_mllm_batch_generator*`, `test_prefix_cache*`, engine tests) pass
  unchanged — they exercise the real code paths against fakes and small
  models.
- Parity probe: import both upstream and vendored modules; assert symbol
  identity for the pure pieces (e.g. `should_quantize_kv_layer` outputs on a
  layer matrix) where mlx-vlm is installed.
- mypy: vendored files enter the existing budget discipline (clean-cache
  count vs base, no unexplained growth beyond the vendored baseline).
- Full `pr_validate` codex loop per PR as usual.

## Risks

- **Silent divergence**: mitigated by the verbatim + documented-redirects
  contract and the provenance header.
- **Security/bug fixes land upstream**: we now own applying them; the exact
  pin makes this a deliberate, reviewable event instead of a silent
  transitive drift.
- **Import cycles**: `apc_adapters` lazily imports `apc`; both live in the
  vendored package so the relative import keeps working. `cache.py` has zero
  internal deps.
