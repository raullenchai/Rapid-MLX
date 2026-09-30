# Pixel handoff: startup removable-volume I/O

- Owner: Pixel
- Branch: `pixel/cua-startup-volume-io`
- Base: `pixel/cua-starter-availability` at `7a98fb04`
- Scope: keep startup model-link maintenance ordered before chat catalog
  discovery while moving its potentially blocking filesystem calls off
  MainActor, so the main window and model-free CUA remain usable.
- Non-goals: cache relocation, model downloads, changing symlink ownership or
  cleanup rules, and changing chat catalog semantics.

## Evidence

- Studio sampled the main thread inside
  `restorePersistedSession -> installAllSnapshotSymlinks ->
  removeStaleStubIfOwned -> attributesOfItem -> getxattr` for over a minute.
- The configured Hugging Face cache path is a symlink to a removable volume;
  macOS removable-volume authorization can therefore block the synchronous
  metadata probe.
- Maintenance now runs in a utility-priority detached task. The restore chain
  still awaits its result before catalog discovery, preventing cache
  read/write races, while the await yields MainActor to SwiftUI and CUA.
- Restore and catalog-refresh callers share one maintenance operation for the
  current `DownloadManager.cacheGeneration`. A later cache generation runs
  maintenance again before its first catalog read, preserving runtime download
  and model-folder change repair semantics. Completed generations advance
  monotonically, so an older waiter cannot schedule stale maintenance after a
  newer pass. There is no separate post-commit symlink repair hook in the
  current Sources tree.
- Regressions directly exercise the production coordinator: one deliberately
  blocks maintenance while observing MainActor progress; another proves
  same-generation single-flight ordering and a second maintenance pass after
  the generation changes.

## Private reference check

Reviewed the required desktop references' startup patterns and Rapid's own
download/catalog workers. Adopted the established rule that removable or
user-selected storage I/O runs off the UI actor while dependent discovery
preserves ordering. No reference code or assets were copied.
