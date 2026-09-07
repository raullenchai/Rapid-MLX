# Atlas handoff: Always-on service P0

- **Owner:** Atlas
- **Branch:** `atlas/always-on-p0`
- **Base:** `origin/main` at `b915cadf`
- **Host:** local Apple silicon Mac
- **Scope:** reliable system-service configuration, credentials, bounded logs,
  diagnostics, and upgrade rollback

## Implemented facts

- LaunchDaemon definitions now invoke a stable `service run --config` entry
  point instead of embedding model/runtime flags.
- Active, pending, and previous JSON definitions live outside the replaceable
  venv under root-owned system Application Support. They are schema-validated,
  atomically replaced, and forbidden from containing API-key flags.
- `service configure` stages without disrupting the active engine;
  `service apply` promotes only after validation and restores the prior config
  and service when bootstrap/readiness fails.
- `service credential` reads keys only from stdin into a service-owned 0600
  non-symlink file. The runtime injects the key via environment immediately
  before spawning the server; status/config never reveal its value.
- The runtime supervises the server, forwards termination signals, and bounds
  stdout/stderr by size, backup count, and age.
- `service upgrade` snapshots the installed environment, upgrades as the
  service account, gates acceptance on launchd `/readyz`, and restores the
  snapshot on failure.
- Status reports config digest/error/pending state, credential presence,
  launchd state/run count, and suspected startup failure loops.

## Verification

- `ruff check`: pass
- focused mypy with skipped external imports: 14 service modules, pass
- service/config/assets tests: 142 passed
- wider CLI selection: 166 passed; two environment-only failures remain
  (subprocess argcomplete returned no aliases and local Python lacks pydantic).

## Remaining work

- Run the real install/configure/apply/credential/upgrade/reboot matrix on a
  sacrificial supported Mac before merge; unit tests never mutate launchd.
- Connect service config into the planned global `EffectiveRuntimeConfig`
  provenance DTO when that migration begins.
- Empty-engine boot, request-driven model loading, and automatic memory-pressure
  budgets belong to the adaptive multi-model residency milestone, not this
  lifecycle patch.
- Release-directory/symlink upgrades could reduce downtime further; this P0
  uses the documented in-place environment snapshot because it preserves the
  existing installer contract.

## Rollback

Legacy argv-in-plist installations remain readable by status/restart. Rolling
the code back after installing a config-backed plist requires
uninstall/reinstall with the older CLI; models, caches, credentials, and logs
are intentionally preserved.
