# AX-first GUI golden flows

`scripts/gui-golden-flows.sh` runs the release journeys against a built
Rapid-MLX Desktop app without loading a real model.

**Journeys** — a user walking a path end to end:

1. fresh install, consent, onboarding, and steady-state shell;
2. Settings mutation and persistence across an app relaunch;
3. basic chat, persisted conversation row, and restored transcript;
4. a deliberately slow stream and semantic **Stop generating** action;
5. model start, a one-shot sidecar crash, automatic respawn, and ready state.
6. a memory-constrained user can see and select an honestly labelled sub-1B
   fallback instead of being sent back to a chooser whose smallest visible
   model is the one that just failed the live-memory guard.

**Invariants** — properties that must hold, not paths a user walks. These were
added after a release where every escaped defect landed on a surface no journey
covered, and each one names the defect it would have caught:

7. `update-state` — Settings → App must name the version the app actually is.
8. `no-dead-controls` — every Settings panel must expose controls of its own.
9. `catalog-integrity` — a model that cannot chat must never be offered as one.

The distinction matters. A journey answers *"can someone do this?"*; an
invariant answers *"is this still true everywhere?"*. The three defects below
were all invisible to journey-shaped tests:

| Flow | Would have caught | Why a journey missed it |
| --- | --- | --- |
| `update-state` | [#1612](https://github.com/raullenchai/Rapid-MLX/issues/1612) — the fallback update manifest sat at 0.11.0 for four releases | Nothing in a journey compares what the panel *says* to what the bundle *is* |
| `no-dead-controls` | [#1595](https://github.com/raullenchai/Rapid-MLX/pull/1595) dead recovery buttons, [#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) toggles that reported success without changing value, [#1605](https://github.com/raullenchai/Rapid-MLX/issues/1605) a tray item that reported nowhere | A journey visits the controls it needs; these were the ones nobody scripted |
| `catalog-integrity` | [#1603](https://github.com/raullenchai/Rapid-MLX/issues/1603) — eight video-generation models offered as chat models, dead-ending *after* a download of up to 64 GB | The picker renders them perfectly; the bug is that they are there at all |

### Current baseline

Run against `main` on 2026-08-07, on a build of this checkout:

| Flow | Result |
| --- | --- |
| `update-state` | **PASS** — panel reads "Up to date — v0.12.6 is the latest release.", matching `CFBundleShortVersionString` |
| `catalog-integrity` | **PASS** — `fake-video-alias` reaches neither the chat surface nor Model Management |
| `no-dead-controls` | **FAILS on `tools`** — see below |

`no-dead-controls` fails today, correctly: Settings → Tools renders three tool
toggles, a backend radio group and a browsing toggle, and **none of them carry
an identifier**. The controls work — they are real `AXCheckBox`/`AXRadioButton`
with correct values — they are simply unaddressable. That is a coverage gap in
product code, not a harness bug, and it is tracked separately. Expect this flow
to go green when those identifiers land; until then it is the gate for that work.

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

### Low-memory recovery

The normal model picker intentionally hides sub-1B models: they fall below the
default quality and tool-use floor, and presenting them beside normal choices
without context makes a faster but worse answer look like a product failure.
That policy cannot govern a recovery path. If the live memory guard says the
starter is unsafe and tells the user to “pick a smaller model,” onboarding must
actually contain one.

`low-memory-choice` pins the visible half of that contract through AX:

1. open fresh onboarding and advance to **Choose your first model**;
2. find `Quickstart.Choice.qwen3-0.6b-4bit` under **LOWEST MEMORY**;
3. assert that the card says **less accurate** and **not recommended for
   tools**, so lower memory is not presented as equivalent quality;
4. select it through AX and retain the before/after trees as evidence.

The warning-to-switch half is deterministic Swift coverage rather than a host-
RAM-dependent GUI trick. `QuickstartView.lowMemoryRecoveryChoice(for:)` replays
the original live-memory snapshot against the fallback footprint and exposes
`Quickstart.Memory.SwitchToLowMemory` only when the replacement falls below the
85% danger line. Under heavier pressure the button is absent, avoiding a false
promise or a warning loop; **Cancel** still returns to the chooser where the
low-memory category remains visible.

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
./scripts/gui-golden-flows.sh --flow low-memory-choice
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
