# AX-first GUI golden flows

`scripts/gui-golden-flows.sh` runs the release journeys against a built
Rapid-MLX Desktop app without loading a real model.

**Journeys** — a user walking a path end to end:

1. fresh install, consent, onboarding, and steady-state shell;
2. Settings mutation and persistence across an app relaunch;
3. basic chat, persisted conversation row, and restored transcript;
4. a deliberately slow stream and semantic **Stop generating** action;
5. model start, a one-shot sidecar crash, automatic respawn, and ready state.

**Invariants** — properties that must hold, not paths a user walks. These were
added after a release where every escaped defect landed on a surface no journey
covered, and each one names the defect it would have caught:

6. `update-state` — Settings → App must name the version the app actually is.
7. `no-dead-controls` — every Settings panel must expose controls of its own.
8. `catalog-integrity` — a model that cannot chat must never be offered as one.

The distinction matters. A journey answers *"can someone do this?"*; an
invariant answers *"is this still true everywhere?"*. The three defects below
were all invisible to journey-shaped tests:

| Flow | Would have caught | Why a journey missed it |
| --- | --- | --- |
| `update-state` | [#1612](https://github.com/raullenchai/Rapid-MLX/issues/1612) — the fallback update manifest sat at 0.11.0 for four releases | Nothing in a journey compares what the panel *says* to what the bundle *is* |
| `no-dead-controls` | [#1595](https://github.com/raullenchai/Rapid-MLX/pull/1595) dead recovery buttons, [#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) toggles that reported success without changing value, [#1605](https://github.com/raullenchai/Rapid-MLX/issues/1605) a tray item that reported nowhere | A journey visits the controls it needs; these were the ones nobody scripted |
| `catalog-integrity` | [#1603](https://github.com/raullenchai/Rapid-MLX/issues/1603) — eight video-generation models offered as chat models, dead-ending *after* a download of up to 64 GB | The picker renders them perfectly; the bug is that they are there at all |

### Current baseline

Run on 2026-08-07, on a build of this checkout:

| Flow | Result |
| --- | --- |
| `update-state` | **PASS** — panel reads "Up to date — v0.12.6 is the latest release.", matching `CFBundleShortVersionString` |
| `catalog-integrity` | **PASS** — `fake-video-alias` reaches neither the chat surface nor Model Management |
| `no-dead-controls` | **PASS** — all six Settings panels expose controls of their own; see the red → green note below |

The first two were measured against `main`. `no-dead-controls` was red on `main`
and is green as of the identifier work described below; the run recorded here is
the one that made it green.

#### `no-dead-controls`: red → green

This flow shipped red, on purpose, and has since been driven green by fixing
the product rather than the assertion. Worth recording, because a gate that has
never moved is a gate nobody has evidence about.

It first failed on **`tools`**: Settings → Tools rendered three tool toggles, a
backend radio group and a browsing toggle, and **none of them carried an
identifier**. The controls worked — real `AXCheckBox`/`AXRadioButton` with
correct values — they were simply unaddressable. Naming them took that panel
`0 → 8`, and the flow then failed one panel further along, on **`privacy`**,
which had the same gap: a telemetry toggle and three policy `Link`s, all
unnamed. Naming those took `privacy` `0 → 4`. Final measured run:

```
[gui-golden]   models: 2 identified controls
[gui-golden]   modelManagement: 16 identified controls
[gui-golden]   tools: 8 identified controls
[gui-golden]   appearance: 3 identified controls
[gui-golden]   privacy: 4 identified controls
[gui-golden]   app: 5 identified controls
[gui-golden] PASS — no-dead-controls
```

`app` was never bare — the loop simply died at `privacy` before reaching it.
Confirmed rather than assumed: `Settings.App.{UpToDate,RecheckCTA,
ExportDiagnostics,HideDockOnCloseToggle,ResetDockOnboardingCTA}`.

#### What this flow does NOT prove

It counts identifiers; it does not press them. A panel can be fully addressable
and still contain a control that does nothing observable, so green here means
"reachable", not "works".

That is not hypothetical. `Settings.Privacy.TelemetryToggle` is addressable and
is a real `AXCheckBox`, and `AXPress` on it **does** flip the stored preference
(`com.rapidmlx.rapid.telemetry.enabled` `0 → 1`, a client ID is minted, the
shared `~/.rapid-mlx/telemetry-client-id` appears) — but the switch itself does
not re-render, so its AX value stays `0` until you leave the panel and come
back, at which point it reads `1`. The cause is that
`SettingsView.telemetryEnabledBinding`'s getter reads `TelemetryConfig.isEnabled`,
a plain `static var` over `UserDefaults.standard`, which gives SwiftUI no
dependency to invalidate on. To a user, that is a consent switch that appears to
snap back to off while they are in fact opted in. It is pre-existing (the
control was previously unaddressable, which is exactly why nothing caught it)
and it is the same family as [#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608)
in the table above. **It is not fixed and not yet filed** — it was found while
naming the control and is reported in the PR that did so, deliberately left out
of an identifiers-only change because it touches consent semantics. This flow
will not catch it; a successor that presses each control and asserts the value
moved would.

Two notes on writing assertions here, both learned the hard way while adding
these:

- The first version of `no-dead-controls` counted every `Settings.*` identifier
  on the panel. The six `Settings.Category.*` buttons appear on *every* panel,
  so the count was never below six and the flow went green on a completely
  unlabelled Tools panel. Count the panel's **own** controls.
- `catalog-integrity` proves it discriminates rather than trivially passing:
  the non-video `fake-alias` appears 9 times in the same tree where
  `fake-video-alias` appears 0 times. A test that asserts an absence must show
  that the corresponding presence is detectable.

Every journey gets a unique bundle identifier and throwaway `HOME` through
`dogfood-isolate.sh`. The fake sidecar emits deterministic SSE and JSONL
lifecycle evidence, so the suite does not download a model or put meaningful
pressure on unified memory.

## Run

Build the current checkout, then run all flows:

```bash
cd apps/rapid-mac
SKIP_SIDECAR=1 BUNDLE_MODEL=0 ./scripts/build.sh
./scripts/gui-golden-flows.sh
```

Run one journey or retain its isolated persona for diagnosis:

```bash
./scripts/gui-golden-flows.sh --flow slow-stream-stop
./scripts/gui-golden-flows.sh --flow chat-restore --keep
./scripts/gui-golden-flows.sh --flow no-dead-controls
```

The suite needs a **local login session** — not SSH or tmux. It also needs the
screen to stay awake: when the session goes idle, `CGSSessionScreenIsLocked`
flips to `Yes`, every app reports zero windows through AX, and `screencapture`
returns wallpaper. That looks exactly like a broken app. Hold the session with
`caffeinate -dimsu -t <seconds>` for the length of the run — `-u` is the
load-bearing flag, since plain `-d` stops display sleep but not the idle path —
and re-read the lock state before trusting any window assertion.

Set `RAPID_GUI_SOURCE_APP` to test a release candidate bundle and
`RAPID_GUI_GOLDEN_OUT` to choose the artifact directory. Each run records AX
trees, actions, fake-sidecar events, logs, and a top-level `result.json`.

## Why this is not coordinate automation

The checked-in `rapid-ax.swift` helper talks directly to macOS Accessibility.
It finds controls by stable `AXIdentifier`, performs `AXPress`, sets native text
values, and serializes roles/descriptions/values for assertions. Peekaboo is
kept for permission checks, window discovery, menu interaction, and screenshots.
The only coordinate fallback is the documented first-run SwiftUI consent-sheet
fallback, derived from AX bounds, for older accessibility stacks.

This makes normal actions independent of window position, resolution, theme,
and most layout changes. It also avoids Peekaboo snapshot publication failures
seen while the ready-state UI is updating. The app does not need to own the
foreground for ordinary AX actions; menu operations and screenshots may briefly
activate it, so release CI should still use a dedicated logged-in macOS session.

## Adding a flow

Prefer a stable `.accessibilityIdentifier(...)` in product code, then assert
observable user state rather than sleeps or pixels. Keep model behavior behind
the fake sidecar unless the purpose of the flow is real inference quality. A
real-model dogfood pass remains a separate, explicitly memory-budgeted release
stage.
