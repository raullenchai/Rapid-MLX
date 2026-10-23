# Desktop runtime verification and upgrades

Desktop selects the engine inside its full application bundle. Before selecting
it, Desktop validates the app's code signature and resource seal against the
release Team ID and bundle identifier pinned in source. The seal includes the launcher,
Python modules and native dependencies. Runtime symlinks must stay inside the
sealed application. Server starts recheck the seal before probing the engine.
Missing, changed, unsigned or differently signed bundles are refused.
Verification reads the bundled payload and adds some startup latency.

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
