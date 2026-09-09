# Pixel → Atlas: Collective Compute Desktop integration

- Owner/host: Pixel / Studio Mac
- Branch: `pixel/collective-compute-ui-mock`
- Base: `origin/main@b2264d61b`
- Server prerequisite: merged PR #3251 (`rapid-mlx share --quicksilver`)
- Interactive mock: `docs/plans/collective-compute-ui-mock.html`

## Goal and boundary

Expose QuickSilver provider mode as an Experimental Desktop surface named
Collective Compute. This pass is a product/interaction mock only: it does not
change Swift, Python, a public API, credentials, services, or release state.

## Verified design facts

- The Experimental toggle adds or removes one sidebar destination and starts no
  work by itself, matching Video, Computer Use, and Benchmark.
- The primary path is choose model + worker, join, watch startup, observe node
  health, and leave. First registration asks for the provider key in a sheet;
  the everyday surface never displays it.
- The mock covers setup, empty-key validation, startup, online, transient relay
  semantics, revoked credential recovery, explicit stop, dark mode, and the
  720-point window floor.
- Online values stay inside #3251's available contract: catalog/local alias,
  worker, node, payout account, heartbeat-backed inflight count, connection
  duration, and lifecycle. Earnings remain a link to QuickSilver because #3251
  exposes no balance, price, or accumulated-payout summary.
- `--install-service` is a secondary, default-off “Keep available after login”
  control under Connection details rather than part of first-run registration.
- Browser interaction assertions pass for registration focus and validation,
  setup → startup → online, connection details, the Experimental gate, light
  and dark appearance, and a 720-point-wide layout. No console/page error was
  emitted.

## Architecture risk requiring Atlas disposition

The Desktop may already own a resident chat/media server, while #3251 always
spawns another `rapid-mlx serve` child. Invoking that CLI unchanged can keep two
large models resident and cause avoidable memory pressure or a system stall.
The proposed GUI contract is exclusive large-model residency: stop/pause the
current Desktop model before joining the pool, let one provider supervisor own
the pool model, then offer to restore the prior Desktop session after leaving.
This lifecycle must be coordinated through `ServerManager`; the UI must not
silently launch a second 27B server.

The GUI also needs structured lifecycle state. Parsing human CLI logs would be
fragile and could regress redaction. Atlas should choose either a small
Desktop-facing provider supervisor contract around the existing QuickSilver
primitives or a machine-readable event/status mode that preserves #3251's
secret handling and fail-closed behavior.

## Next concrete action

After product approval of the mock, Atlas should disposition the residency and
status-contract choices. Pixel can then implement the gate, sidebar surface,
all user-visible states, accessibility identifiers, and Swift tests on a fresh
feature branch; Vector should verify real model residency, relay lifecycle,
credential redaction, cancellation, and restore behavior before queueing.
