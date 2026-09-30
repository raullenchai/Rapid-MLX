# Pixel handoff: model-free Computer Use server

- Owner: Pixel
- Branch: `pixel/cua-model-free-server`
- Base: `pixel/cua-create-recovery` (`aca4dc06`)
- Receiving roles: Vector for the CLI contract; Atlas for stack ordering; Harbor for packaged-sidecar smoke.

## Goal

Opening Computer Use lazily starts one app-owned, authenticated CUA sidecar
without selecting, downloading, or loading a chat model. The process remains
owned across tab navigation and is reaped with the app.

## Contract

- CLI: `rapid-mlx serve --cua-only --host 127.0.0.1 --port N`
- Auth: `RAPID_MLX_API_KEY`
- Parent watchdog: `RAPID_MLX_WATCHDOG_PPID`
- Readiness must report `ready: true`, `model: null`, and
  `model_loaded: false`; the Desktop then verifies authenticated
  `/v1/cua/capabilities` before exposing the client.

## Reference check

- Reviewed the existing Rapid app-owned server, process-group, port allocation,
  bearer, watchdog, and split signal/reap shutdown paths and reused those
  primitives instead of creating a second process abstraction.
- Open WebUI Desktop's local server lifecycle guards duplicate start and keeps
  server status in a main-process owner. Cherry Studio's service lifecycle puts
  long-lived side effects in app-owned services. LM Studio separates daemon
  availability from model residency. Adapted those proven boundaries to a
  native, lazy app-owned CUA manager; rejected component-owned process lifetime
  because navigation cancellation can orphan or churn the service.
- No proprietary code or assets were copied.

## Known integration item

The current server stack can classify a full-screen Notification Center layer
as click occlusion. That backend correction is tracked separately and is not
part of this GUI lifecycle branch.

## Verification

Run focused CUA manager/client/view-model tests, App termination ordering tests,
the full CUA Swift filter, `git diff --check`, and a combined smoke against the
Vector `--cua-only` server commit before merging the stack.
