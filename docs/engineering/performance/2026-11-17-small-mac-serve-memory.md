# Small-Mac serve memory admission: corrected field evidence

Date: 2026-11-17. Inspected base checkout:
`52642bde6c59e3e8fb011560ae7c2c9886ecc870`.
All memory quantities here use GiB (2³⁰ bytes); the CLI labels these GB.

## Corrected conclusion

The earlier conclusion to keep both refusals was wrong for the documented
16 GiB image models. Successful field workloads disprove the assumption that
an admission estimate exceeding Metal's recommendation establishes that a
model cannot run. Catalog `min_memory_gb` is now the single source for both
the published minimum and preflight admission at or above that minimum.
There are no per-model exceptions and no changes to the recorded estimates.
This is permission to attempt loading, not a guarantee against runtime memory
pressure for every resolution, context length or concurrent workload.

## Field evidence

Source: user-supplied aggregate PostHog results, 2026-09-01 through
2026-11-17. Image successes are `inference_bucket_reached`, `result=ok`,
on `/v1/images/generations`. Counts are approximate installs, not controlled
benchmarks; no installation identifiers are retained here.

| Model and hardware | Successful workloads | Preflight refusals |
| --- | --- | --- |
| `flux2-klein-4b`, 16 GiB | About 40 installs on 0.15.0–1.0.3; nine on 1.0.1–1.0.3 across server and Desktop; several reached 5–9 images and one 50–99 | 1.0.1: three installs; 1.0.2: four; 1.0.3: two |
| `z-image-turbo`, 16 GiB | About 16 installs, including 1.0.0–1.0.2; one reached 500–999 images | 1.0.1: two installs; 1.0.3: one |
| `qwen3.5-4b-4bit`, 8 GiB | Successful requests on 0.15.7 (server) and 1.0.1 (Desktop) | Prior brief reports three installs |

Image refusals above were `insufficient_memory` at `preflight`. The supplied
README image matrix advertised 16 GiB for both image models, matching their
`aliases.json` profiles. Qwen's current profile documents 16 GiB, so 8 GiB is
below its documented minimum. Its successes challenge treating the estimate
as a physical impossibility, but do not establish a new qualified minimum.
The 8 GiB uncached Qwen refusal remains under the existing below-minimum
footprint policy; its 16 GiB admission now follows the catalog contract.

## Root cause and mixed results

Previously `catalog_memory_fit` could refuse a model even after its catalog
minimum was met: an image admission charge (or text weights/resident size)
was compared against `recommendedMaxWorkingSetSize` as a hard ceiling.
That recommendation is not a proof of a physical limit. Without a Metal
report, meeting the catalog minimum passed. A complete cached checkpoint
also bypassed ordinary admission checks. These code paths explain how the
same version and physical RAM could admit one installation and refuse
another; the supplied aggregates do not identify which path each install
used. Desktop can also reject an image before CLI startup through its
resident model admission path; correcting CLI preflight alone is insufficient.

| Model | Download size | Admission estimate | Fallback GPU budget | Current policy |
| --- | ---: | ---: | ---: | --- |
| `flux2-klein-4b`, 16 GiB | 4.30 GiB | 14.2 GiB | 10.67 GiB | Pass: meets catalog 16 GiB minimum |
| `z-image-turbo`, 16 GiB | — | 13.1 GiB | 10.67 GiB | Pass: meets catalog 16 GiB minimum |
| `flux3-klein`, 16 GiB | — | — | 10.67 GiB | Refuse uncached: below catalog 24 GiB minimum |
| `qwen3.5-4b-4bit`, 8 GiB | 2.85 GiB | 6.0 GiB | 5.33 GiB | Refuse uncached: below catalog 16 GiB minimum, estimate exceeds budget |

`runtime/resident_models.py::measured_image_estimate_bytes` records FLUX's
historical **13.15 GiB peak active memory at 1024×1024**, plus **1 GiB cache
allowance**, rounded up to **14.2 GiB**, introduced in
`3d8b8c3e48df6629cf0671c1ca99fbfe24726708` (#5437). This larger-host
observation does not measure the minimum working set on a 16 GiB Mac.
MLX reports active and cached memory separately; free buffer-cache growth
alone cannot be claimed to explain an active-memory peak. No constrained
16 GiB measurement was made, so the precise contribution of allocator,
activation and runtime differences remains unproven.

The cached FLUX checkpoint revision
`7ee1b3aa8178a1240050490072196a57da2bf2a9` has these tensor payloads,
counted from safetensors headers without loading arrays:

| Component | Bytes | GiB |
| --- | ---: | ---: |
| Transformer | 2,180,004,864 | 2.0303 |
| Text encoder | 2,262,920,448 | 2.1075 |
| VAE | 165,078,086 | 0.1537 |
| Total | 4,608,003,398 | 4.2915 |

The 8.86 GiB difference between tensors and recorded active peak is an
aggregate, not an experimentally isolated activation requirement. Qwen's
6 GiB estimate comes from the curated 6144 MiB footprint in
`model_recommendations.json`; no new Qwen measurement was made.

## Policy and regression coverage

The public catalog payload and preflight both read the resolved profile's
`min_memory_gb`. At or above it, the preflight admits every catalog model,
regardless of the reported Metal recommendation or cache status. Below it,
existing enforced floors and modality/footprint rules remain. Models without
a documented minimum retain the estimate-based check. Estimates remain
available for runtime residency planning; this change does not alter runtime
allocation limits, download behavior or catalog minima.

The improved image refusal explanation (model, text encoder, VAE and default
image generation headroom) and fitting same-task suggestion are retained.
Regression tests cover both image models on 16 GiB, FLUX3 refusal there,
Qwen below and at its documented minimum, alias/repository parity, cache and
Metal probe variants, and agreement between published catalog minima and
admission for every catalog profile at and above its minimum.

## Attempted default generation

Existing interpreter: Python 3.12.13, MLX 0.32.4, mflux 0.23.0, mlx-lm 0.31.3.
The following command used offline mode and the existing HF cache only:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /private/tmp/website-poshog-dl-resume/.venv/bin/python - <<'PY'
import mlx.core as mx
from rapid_mlx.image.engine import ImageGenerationEngine
mx.reset_peak_memory()
engine = ImageGenerationEngine('Runpod/FLUX.2-klein-4B-mflux-4bit')
image = engine.generate(prompt='A red apple on a wooden table', width=1024, height=1024, num_inference_steps=4, seed=0)
print('PNG bytes:', len(image), 'peak active GiB:', mx.get_peak_memory() / 2**30)
PY
```

Result: exit 1 at `import mlx.core`, with
`ImportError: [metal::load_device] No Metal device available.` No weights were
downloaded and no new peak was measured. The 13.15 GiB number above is the
historical recorded active peak, not a measurement from this command.
MLX distinguishes [active and cached allocation memory](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp);
an active-memory peak alone does not include its cache allowance.


## Verification of the correction

Using `/private/tmp/website-poshog-dl-resume/.venv/bin/python`:

- `-m pytest tests/test_catalog_memory_preflight.py tests/test_byom_preflight.py tests/test_pull_preflight_gates.py tests/test_readme_image_matrix.py tests/test_model_profiles_ssot.py -q` → 281 passed.
- `-m pytest tests/test_catalog_identity_estimates.py -q` → 218 passed.
- `-m pytest tests/test_community_benchmark_workspace.py -q -k 'floor or metal_limit or memory or gpu_budget or raw_estimate'` → 39 passed, 327 deselected.
- `ruff check .` → passed; `ruff format --check .` → 2428 files already formatted.
- `git diff --check` → passed.

A broader exploratory run including `tests/test_resident_models.py` aborted
in a native dependency import through `rapid_mlx/api/guided.py` during
`test_dynamic_resident_auto_detected_hybrid_gets_bounded_prefix_reuse`.
The full residency suite is therefore unverified in this session. A separate
broad benchmark-workspace run was interrupted after 303 passing tests across
the combined run; the memory-focused selection above completed successfully.
No new live generation or 16 GiB hardware measurement was performed.


## Desktop admission correction

Desktop already obtains image `min_memory_gb` from `rapid-mlx models --json`
through `ModelCatalog.imageEntries`, which sets `ModelEntry.minimumMemoryGB`.
No Swift model table was added. The former image residency charge was at
least `16 × 0.8 = 12.8` GiB on a 16 GiB minimum, but Desktop launched the
resident pool with `floor(16 × 0.8) = 12` GiB. Artifact-based estimates could
raise the charge further, and a resident capacity rejection had no fallback.

Image loads now compare the catalog minimum with physical RAM before a
resident request or process transition. Known image models receive the same
rounded residency budget as the supported minimum Mac, independently of disk
size. Unknown models keep artifact-based estimates. The separate live-pressure
warning remains. Thus FLUX2 Klein 4B and Z-Image Turbo receive a 12 GiB charge
and pass capacity on 16 GiB, while FLUX3 Klein's 24 GiB minimum refuses there.
The supplied field evidence also includes successful Desktop generations on
1.0.1 and 1.0.2; those successes support the catalog contract.

A shared test fixture covers every image catalog minimum: Python checks it
against the live public catalog and CLI fit result; Swift parses it through
the production catalog parser and checks physical capacity plus the resident
ceiling below, at, and above each minimum.

Desktop follow-up verification:

- In `apps/rapid-mac`, `swift test --disable-sandbox --filter 'ImageCatalogTests|ModelResidencyTests|ImageGenerationResolutionTests'` → 56 tests passed across 3 suites.
- Python preflight/catalog selection above → 281 passed; catalog identity estimates → 218 passed.
- `ruff check .`, `ruff format --check .`, and `git diff --check` → passed.
- These are simulated-capacity regression checks; no new live image generation was performed.
