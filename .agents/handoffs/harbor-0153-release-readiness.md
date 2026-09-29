# Harbor: 0.15.3 release readiness

Owner: Harbor. Receiving role: Atlas. The human owner authorized the 0.15.3 release; Atlas owns the final go/no-go decision.

## Verified release candidate

- PR #3917 fixed startup exit semantics with uvicorn 0.42 while retaining uvicorn 0.23 compatibility. It passed independent review and protected CI, then merged to main as `e07e625b01d37f9a0c019e628090fcacf5183afa`.
- CUA PR #3909 is explicitly opt-in and Experimental. Incomplete tasks, early stops, recovery loops, permission prompts, and behavior changes after app or window updates are documented accepted limitations. Finder perfection and live GUI task completion are not 0.15.3 release gates.
- Exact #3909 head `614c479d549850d3167ec7f18bc56b4330ecdd79` passed all 42 checks: 30 success, 11 expected skipped, one neutral, and no failures. Ordinary CI run `36673549724` passed. Independent review found no P0/P1 and confirmed the rebase preserved all previously reviewed patches; the final change added only a focused-window regression test.
- Nonpublishing signed candidate run `36673584313` passed for exact source `614c479d`. Artifact `11079154150` reported `signed=true`; DMG SHA256 is `d2a9d8ca6632804b94e45a9e1a349e9aff846d884e2fa88e0cbe8974eb64b7b7`. On Mac mini, `hdiutil verify`, DMG stapler validation, strict deep app/helper codesign, and Gatekeeper checks passed. Bundle IDs are `com.rapidmlx.rapid` and `com.rapidmlx.rapid.computer-use`; both use Team ID `73WQ7ZGSWC`. The image was mounted read-only and detached cleanly.
- A queue-only Desktop packaging defect was fixed at `f0d454bccf6fbc05813a9d3b4847cc05eba5ab38`: release-shaped builds that intentionally omit the sidecar no longer inspect an absent nested Python executable. The real packaged-sidecar entitlement check remains fail closed. Exact-head CI, hosted Desktop, signed candidate, and independent review passed.
- The final protected candidate `34aae81b489e0ca9e9f0b9fe4d68f308426f6c87` passed queue CI and Desktop gates. #3909 merged to main as `fed0eb9826d8b00fe6c11e8b1a972516c0b2847d`.

## Release path and open gates

- The 0.15.3 bump is one commit atop the protected #3909 merge. It synchronizes Python/Desktop version 0.15.3, Desktop build 178, changelog and release notes, and resets Unreleased. Plist lint and 147 version/release tests pass. Verify the one-commit tree diff, obtain independent review, then merge PR #3915 through protection.
- Run the official exact-head release preflight on #3915 and merge only after required checks pass. A normal version-bump merge to main starts auto-release.
- The normal Studio Tier-1 gate is currently unavailable because Studio DNS and host services are degraded. Mac mini lacks the cached 35B model and required agent CLIs, so its partial smoke is supplementary and must not be described as equivalent Tier-1 evidence.
- If the normal Tier-1 job is unavailable solely because of the documented Studio environment, Atlas approved the workflow's audited emergency `force_version=0.15.3` path after every other gate passes. Record that the full Tier-1 inference gate was bypassed and do not bypass signing, notarization, artifact validation, release preflight, protected environment approval, or post-publication verification.
- Preserve provenance for the final main SHA, auto-release run, engine and Desktop tags, PyPI files, signed DMG, Sparkle metadata, CDN pointer, and release pages. Verify install/update paths before declaring success.

## Rollback

- Before publication, stop by cancelling the release run and leave tags absent.
- After publication, use the documented rollback procedure: preserve immutable provenance, restore the last known-good updater/CDN pointer and package guidance, mark affected releases appropriately, and prepare a forward-fix version. Do not move or silently replace signed tags or published artifacts.

Keep raw logs, screenshots, transcripts, and credentials outside the repository.
