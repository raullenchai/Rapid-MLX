# CUA sidecar native framework packaging

- **Owner:** Vector, with Harbor review for release packaging
- **Branch/worktree:** `harbor/cua-sidecar-pyobjc` at
  `/private/tmp/harbor-desk-cua-sidecar-pyobjc`
- **Base:** observation API stack head `e7b395e2c`
- **Status:** implementation complete locally; intentionally not pushed pending
  manager review

## Verified facts

- A clean Desktop sidecar lacks `ApplicationServices` and `Quartz`, so native
  Computer Use reports an unavailable platform before TCC can be evaluated.
- The two direct framework requirements at 12.2.2 resolve a five-distribution
  closure on Python 3.12: core, Cocoa, CoreText, ApplicationServices, and
  Quartz. A clean target import of both public modules succeeds.
- The wheel closure contains 21 runtime Mach-O extensions after removing the
  upstream `PyObjCTest` and dSYM payload. This changes the measured signing
  baseline from 173 to 194. The extensions are universal arm64/x86_64 Mach-O
  bundles and remain covered by the existing codesign sweep.
- The new optional extra uses Darwin markers. Server imports on other platforms
  keep their existing lazy behavior.
- The Desktop dependency inventory now names all five PyObjC distributions.
  The core and ApplicationServices wheels expose `License: MIT` metadata but no
  standalone license file, so the complete shared project copyright and MIT
  permission notice is an explicit source-controlled build input staged at
  `licenses/PyObjC-MIT.txt` inside the sidecar.

## Reference and repository checks

- Checked the existing package extras, sidecar constraints, trim phase, smoke
  checks, signing enumeration, and distribution constraint tests first. The
  change follows those established mechanisms rather than adding a parallel
  installer.
- Checked the primary inference server precedents and MLX-native packaging
  scope; they do not package macOS Accessibility bindings into a signed Desktop
  sidecar, so no applicable mechanism was available there.
- Used official package metadata to verify the framework dependency closure;
  no external source or asset was copied.

## Verification

- `python3.12 -m pytest -q tests/test_cua_packaging.py tests/test_sidecar_distribution_constraints.py`
  — 9 passed.
- Ruff check and format check passed for the touched tests.
- `bash -n apps/rapid-mac/scripts/build-sidecar.sh` and `git diff --check`
  passed.
- Clean Python 3.12 target install imported both native framework modules and
  measured 21 post-trim Mach-Os.
- The explicit license notice was staged into the release-shaped scratch tree,
  remained readable, and matched its source byte-for-byte.

## Remaining risk and next action

- A release-shaped local build completed the embedded interpreter, minimal
  FFmpeg build, package resolution, and install. It then stopped in the
  pre-existing mflux validation because this sandboxed session exposes no Metal
  device. The failure occurred before the shared trim/count/sign/smoke stages:
  `ImportError: [metal::load_device] No Metal device available`.
- On that exact staging tree, the equivalent PyObjC trim left 21 distribution-
  owned Mach-Os. All 21 accepted the build script's ad-hoc hardened-runtime
  signing options, passed `codesign --verify --strict`, and the two framework
  imports succeeded both before and after signing. This independently supports
  the 173 + 21 = 194 baseline, but the canonical runner must still execute the
  complete count and Developer ID signing/notarization path.
- Manager reviews the local diff, then decides whether to publish it as the next
  stacked draft PR.
