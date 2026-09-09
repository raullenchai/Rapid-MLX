# Share Compute desktop boundary

Rapid Desktop exposes QuickSilver provider mode behind the experimental
**Share Compute** gate. Enabling the gate only adds the workspace; connecting,
downloading a supported model, and joining the pool remain separate explicit
actions.

## Process and credential boundary

- Desktop launches the bundled `rapid-mlx share … --quicksilver` supervisor in
  its own process group.
- A first-run `qsppk-` provider key is sent as one bounded line over the
  child's stdin. It is never placed in argv, the child environment, Desktop
  preferences, or a Desktop-owned file.
- Python owns the existing `0600` QuickSilver node cache. After a successful
  Desktop registration it also writes a separate non-secret, `0600` marker
  bound to model, alias, and worker. Swift decodes only that bounded marker;
  it never opens or decodes the node cache containing the share credential.
- Python publishes a field-whitelisted, credential-scanned, atomically replaced
  `0600` status snapshot under `~/.rapid-mlx/quicksilver/`. Desktop consumes
  that snapshot instead of parsing human log text.

## Model residency and teardown

Share Compute takes the same exclusive large-model lease used by local
benchmarks. The embedded chat/media server is fully stopped before the pool
supervisor starts, so two large models cannot compete for unified memory. On
explicit stop or provider failure, Desktop releases the lease and restores the
previous local model. Disabling the experimental gate invalidates an in-flight
join as well as stopping an established session.

App termination signals Share Compute, the embedded server, and downloads
before waiting on any of them. The pool supervisor has a bounded graceful exit
followed by process-group SIGKILL, and its nested serve process also carries the
Desktop parent watchdog.

## Deliberate initial scope

The first Desktop surface supports the three catalog mappings declared by the
QuickSilver provider implementation and requires weights to be downloaded
before joining. It does not install QuickSilver's login LaunchAgent. Background
login service ownership needs cross-process residency arbitration with Desktop;
shipping the CLI service toggle directly would permit a hidden pool model to
contend with a later Desktop model.
