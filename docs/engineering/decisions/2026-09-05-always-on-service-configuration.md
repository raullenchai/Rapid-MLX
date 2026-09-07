# Always-on service configuration and transactional apply

- **Status:** Accepted for implementation
- **Owner:** Atlas
- **Date:** 2026-09-05

## Context

The first `rapid-mlx service` release encoded its model, bind, and runtime flags
directly in a root-owned LaunchDaemon plist. A change required
uninstall/install, while overwriting the plist without re-bootstrap created
split-brain between the file and launchd's in-memory job. Secrets were correctly
refused but there was no supported authenticated headless deployment.

oMLX demonstrates the usability value of persisted settings, but its reported
stale-restart and invalid-setting crash-loop failures show that persistence
without a validation and activation boundary is insufficient. vLLM and SGLang
reinforce that readiness, metrics, and explicit effective launch parameters are
part of the production contract rather than UI-only state.

## Decision

The plist is a stable bootstrap definition. It invokes
`rapid-mlx service run --config <absolute path>` and contains neither model
settings nor credentials. The config is versioned JSON, atomically written
under system Application Support, root-owned, and non-writable to other users.
It remains readable for unprivileged status diagnostics and deliberately lives
outside the replaceable Python virtual environment.

Configuration changes use two phases:

1. `service configure` validates and writes a pending candidate.
2. `service apply` saves the active definition, stops the job, promotes the
   candidate, bootstraps it, and gates success on `/readyz`. Any failure restores
   and starts the previous definition.

An optional API key lives in a separate mode-0600 non-symlink credential file.
The runtime launcher reads it immediately before `execve` and exposes it only as
`RAPID_MLX_API_KEY` to the server process. Config and status surfaces reveal
only whether a credential exists.

## Consequences

- Normal model, bind, and serve-flag changes do not rewrite privileged launchd
  state and cannot silently take effect only after reboot.
- Invalid candidates never disturb the active service; unhealthy candidates
  have a deterministic rollback target.
- The service-level schema is deliberately narrower than the planned global
  `EffectiveRuntimeConfig`. A later adapter should feed this source into that
  resolver rather than create competing precedence rules.
- Service-account and executable changes remain uninstall/install operations.
- The stable runtime supervises the server process and bounds stdout/stderr by
  size, backup count, and age. Release-directory upgrades remain separate
  operational work and should not be hidden inside this config transaction.

## Rollback

The prior argv-in-plist form remains readable by status/restart for installed
legacy services. Code rollback requires uninstall/install because an older CLI
does not implement the stable `service run` entry point.
