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

## AX structural baselines

Ten settled states across the five journeys are also fingerprinted as
**structural baselines**, committed under
`Tests/GUIGoldenFlows/__Snapshots__/<flow>.<state>.txt`. `scripts/ax-baseline.py`
normalises a raw AX dump into an indented tree and the suite fails on any
difference, so a PR that removes a button, reparents a control, renames an
identifier, drops an icon or flips an enabled state produces a reviewable diff
instead of passing silently.

**This is the cheap layer of appearance testing and it is structural only.** It
cannot see colour, spacing, typography or anything else that never reaches the
accessibility layer; the PNG snapshots in `Tests/RapidTests/__Snapshots__` stay
the pixel-level check.

The normaliser keeps hierarchy, role, subrole, `accessibilityIdentifier`,
`AXTitle`/`AXDescription`/`AXHelp`, enabled state, sibling order below the
window level, and the *kind* of each value (`bool:true`, `bool:false`,
`number`, `text`, `empty`). It drops or rewrites everything that is legitimately
volatile: screen coordinates and sizes, pids, top-level window z-order, value
contents, and version numbers, byte sizes, token rates, durations, dates, clock
times, UUIDs, `/Users/<name>` paths and the fake model alias wherever they
appear in text. `Settings.App.UpToDate` carries the release version and a
conversation row identifier carries a fresh UUID — recording those verbatim
would make the baselines flap every release and every run.

Two further things are dropped because they flap without any product change,
both found by comparing real recorded baselines rather than by reasoning:

- **Everything below a window-control button.** The traffic lights are AppKit's,
  and their anonymous `AXGroup` descendants are realized lazily: two dumps taken
  seconds apart in the *same* run recorded one group under `AXZoomButton` in
  `settings-root` and two in `models-idle`. The buttons themselves stay, so a
  missing close box is still a diff; their private innards do not.
- **Relative day headings.** A transcript is filed under `Today` — until a run
  straddles local midnight, at which point the identical UI says `Yesterday`
  and every baseline holding one goes red at 00:00 for no reason.

An intended UI change is a deliberate commit:

```bash
./scripts/gui-golden-flows.sh --update-baselines
git diff apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__
```

Recording is **only** ever done by `--update-baselines`. A missing baseline is a
failure, not a free pass: recording on absence would mean a typo'd snapshot name,
or one somebody forgot to `git add`, sails through CI green while comparing
against nothing. (This deliberately diverges from the PNG convention in
`Tests/RapidTests/SnapshotHelpers.swift`.)

Inspect a single normalised tree without running a journey:

```bash
python3 scripts/ax-baseline.py normalize --scrub fake-alias /tmp/…/steady.json
```

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

Fingerprint only *settled* states. Baselines taken mid-transition flap: the
crash-recovery tree captured while the sidecar was still restarting contained a
transient "Starting …" banner in one run and not the next. `wait_send_idle`
exists for this — `ChatView.SendOrStopButton` publishes `AXHelp` only while the
readiness gate is closed, so the absence of that attribute is a
copy-independent "ready and not streaming" signal. If a new state turns out to
be irreducibly unstable, exclude it and say why rather than loosening the
comparison.

**Never assert that text appears *somewhere* in the tree when a specific place
is what you mean.** `chat-restore` failed roughly one run in two for a reason
worth repeating. `start_model` gated on `SendOrStopButton.description ==
"Send message"`, which is the button's label for the whole startup — its hint
still read "… is still starting." So the flow pressed Send into a closed
readiness gate, the press was dropped, and the draft stayed in the composer.
`assert_tree_text "golden restore marker"` then *found* the prompt — in the
composer — and reported success for a message that was never sent. The run only
failed later, on the reply that never came, which is why it looked like a
flake rather than a broken assertion.

Both halves are now fixed: `start_model` waits on `wait_send_idle`, and
`send_prompt` requires the composer to actually drain. The general rule: an
assertion that a string is present anywhere is satisfied by the input field,
the placeholder, the tooltip and the sidebar. Say *which element*.
