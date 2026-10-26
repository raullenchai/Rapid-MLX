# Desktop runtime verification and upgrades

Desktop selects the engine inside its full application bundle. Before selecting
it, Desktop validates the app's code signature and resource seal against the
release Team ID and bundle identifier pinned in source. The seal includes the launcher,
Python modules and native dependencies. Runtime symlinks must stay inside the
sealed application. A background launch task warms authentication. Engine launches
require a successful check for that binary path in the current app session,
including server starts, capability probes, catalogs and downloads. Concurrent
callers share that check; failures are never cached. The launch
path remains inside the app even when the launcher is an internal symlink.
Missing, changed, unsigned or differently signed bundles are refused.
Verification reads the bundled payload once per session and binary path.
Dictation capture only stats cached snapshot paths and artifact identities inventoried
by an earlier catalog refresh; it neither starts the CLI nor authenticates the bundle.
Reference-file identity and timestamps must also match, so repointing `refs/main`
cannot authorize a stale snapshot.

## Upgrading a slim installation

Use the normal signed full-app update, or install the full Desktop app from the
release download. The update replaces the application and supplies its bundled
engine. It leaves model caches, preferences and the old Application Support
`runtime-override` directory intact. That old engine is no longer selected,
regardless of its version. No deletion or copying of the old engine is needed.
The current release assembler rejects signed/official builds that request
`SKIP_SIDECAR=1`, so an update cannot intentionally omit the replacement engine.

A missing or unverifiable engine shows recovery guidance and a download action.
Install the full app and reopen it; reopening an incomplete app alone does not
provision an engine. A failed update should retain the existing app through the
whole-app updater's replacement/rollback behavior. This change does not replace
that updater or install engines from mutable sidecar metadata.

## Explicit developer override

To run a local checkout or an engine-less development build, explicitly select
an executable in the app's launch environment:

```sh
RAPID_BIN="$HOME/dev/rapid-mlx/.venv/bin/rapid-mlx" \
  "/Applications/Rapid-MLX Desktop.app/Contents/MacOS/Rapid"
```

Quit an already running app first. `RAPID_BIN` intentionally bypasses bundled
engine authentication and appears as an override in the About panel. Only set
it for code you intend to execute. Unset it to restore normal bundled selection.
Desktop never searches PATH or automatically trusts legacy managed directories.
Ad-hoc signed development bundles also require `RAPID_BIN`; production bundles
must match the pinned release signing identity.

## Verification

`DesktopRuntimeTrustTests` signs real fixture apps and checks intact runtimes,
changed resources, signature replacement, symlink containment and the explicit
override. Its migration case starts with an existing slim-install directory,
adds the full signed app payload, resolves and executes the new engine, and
checks that the old managed engine remains unchanged. Release assembly refusal
is exercised as a subprocess. No model or GPU is required.

## Threat model and residual race

Installation remains prompt-free. Desktop verifies the complete application
signature and resource digests before granting session authority to an
app-owned binary path. An unsigned payload, a different signing identity, or
resources changed before the first successful verification are refused.
Authentication failures are retried, never retained as authority.

The owner-approved session cache means changes made after successful
verification during the running session are not detected until relaunch.
This extends the previously accepted same-user verification-to-use race:
static verification does not make the bundle immutable or isolate it from
another process running as the user. Full-app updates should be followed by
relaunch. No privileged installation helper or administrator authorization is
introduced, and the pinned requirement and strict/all-architectures/nested-code
flags are unchanged. The explicit developer override remains separate.

## Dictation latency regression guard

The required Desktop build runs `DesktopRuntimeTrustTests/dictationCaptureLatency`
as a named CI step, excluding it from the following full-suite invocation to
avoid creating the large fixture twice. Standalone full Swift runs include it. It drives the real
`toggleFromUI` controller through recorder start using fake capture, warms a
real ad-hoc signed app with 24,000 runtime resource files (375 MiB), and asserts
capture within 300 ms with zero catalog loads, bundled launches, or full
bundle checks after warm-up. The designated test signature replaces only the
release identity, not Security.framework validation or its flags. Repeated
real launches assert that successful validation is reused; deletion of the
selected cache snapshot, a weight/config file, or a symlinked blob must refuse
capture without invoking the catalog. Empty snapshots are not accepted.
The budget allows CI scheduling noise; counters catch regressions even on
hardware that verifies faster than the budget. No GPU or microphone is used.
