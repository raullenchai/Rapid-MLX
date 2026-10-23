# Desktop runtime trust and legacy migration

Owner: Harbor (release packaging), Pixel (engine discovery); Atlas coordinates
release readiness. Security audit branch: `atlas/security-audit`.

## Verified behavior and changes

The October 2026 release workflow calls the `desktop-releasable` composite,
which builds the complete app with `scripts/build.sh` and packages it with
`scripts/dmg.sh`. The sidecar is copied into `Contents/Resources/rapid-mlx`
before the app's outer code-signature resource seal is generated. The legacy
slim-DMG script remains in the repository, but the current release workflow
does not call it and the former BootstrapCoordinator implementation is absent.

Previously, ServerLocator could execute a shell launcher from user-writable
Application Support merely because it was executable. Its writable VERSION
could outrank the bundled engine. This allowed code substitution by a process
already able to write as the user; it is not by itself a privilege escalation.
Checking only the Python Mach-O signature would not authenticate the shell
launcher, Python sources, or imported modules.

Automatic discovery now uses only the bundled engine. Legacy runtime overrides
are ignored even when the bundle is absent. RAPID_BIN remains an explicit
choice for development and integration tests. This closes the automatic
Application Support substitution path; it does not claim protection against
all malware already running as the user or modifying their launch environment.

The sidecar build verifies cached and downloaded Python archives against a
reviewed, committed SHA-256 before tar parses or extracts them. Overrides of the
Python release require an explicit matching PBS_SHA256. The cache key includes
version, tag, and digest. Both base and legacy video-extra mlx-arsenal requirements
match the release-tested sidecar pin, 0.12.1. This is a reproducibility constraint,
not a claim that newer releases are malicious.

## Rollout and recovery

Ship through the ordinary signed full-app release pipeline after human release
authorization. Verify a fixture with a newer writable runtime VERSION still
selects the bundled engine, and a missing bundle never launches that runtime.
Verify explicit RAPID_BIN still selects a developer checkout. Native regression
coverage lives in ServerLocatorVersionSelectionTests and
AboutPanelEngineIdentityTests; archive and dependency tests live in
`tests/test_sidecar_supply_chain.py`.

Users of a historical slim installation must install the current full desktop
app. Existing runtime-override directories can remain on disk; no deletion or
model migration is needed. Developers using SKIP_SIDECAR=1 must set RAPID_BIN.
A malformed or missing bundled engine is recovered by reinstalling the full app.

If compatibility problems occur, prefer reinstalling a known-good signed full
app. Reverting locator hardening restores the writable-code execution path and
requires an explicit security-risk decision by Atlas and the human release owner.
No production deployment is part of this audit.

If independent engine updates are restored in future, Harbor must provide an
app-anchored signed manifest covering the entire runtime tree and a verified
publish/launch protocol. A digest stored beside a writable runtime or a version
comparison is not an authenticity check. Pixel must keep failures recoverable
without silently executing unverified content.
