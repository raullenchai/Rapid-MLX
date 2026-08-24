# Stateful SwiftUI extraction inventory

This inventory ranks Desktop surfaces where product behavior is still owned by
SwiftUI views. It guides small extraction PRs; it is not a proposal to replace
SwiftUI, create a new framework, or move rendering-only state out of views.

Snapshot: `main` at `a9c0b000` (2026-08-23). Re-check the source before starting
an item, and update this document when ownership moves.

## Ranking method

Order is evidence-first, without a synthetic score:

1. confirmed escaped regressions and severity of the user-visible failure;
2. breadth of user impact, especially first run, model lifecycle, and data loss;
3. asynchronous or cross-surface ownership currently coordinated in a view;
4. availability of a small seam that can move to deterministic Swift tests.

Pure presentation state such as hover, animation phase, popover visibility, and
measured geometry is not an extraction target unless it causes a documented
behavioral or visual escape. The recommended layer below is the cheapest layer
that should own the non-rendering contract; targeted GUI or visual coverage may
still be needed for presentation boundaries.

## Ranked inventory

### 1. Audio and Dictation readiness/lifecycle

- **Sources:** [`AudioView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/AudioView.swift),
  [`DictationView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/DictationView.swift),
  and [`DictationController.swift`](../../../apps/rapid-mac/Sources/Rapid/Dictation/DictationController.swift).
- **View-owned concern:** readiness actions, model-load tasks, in-flight model
  aliases, download-status reactions, voice-preview task identity/cancellation,
  and the handoff from download to warmup remain coordinated across view
  callbacks.
- **Impact/evidence:** Silent first-use download and premature hotkey arming
  escaped to dogfood; see the
  [regression ledger](escaped-gui-regressions.md#dictations-first-use-silently-downloaded-a-model--pr-2188)
  and [hotkey record](escaped-gui-regressions.md#dictation-hotkey-armed-before-the-model-was-ready--pr-2193).
- **Extract next:** one Audio/Dictation lifecycle state machine that accepts
  catalog, download, serving, warmup, selection, and cancellation events and
  emits readiness plus permitted effects. Keep audio playback itself behind a
  small effect protocol.
- **Cheapest effective layer:** table-driven Swift unit tests for transitions
  and stale-event rejection; contract tests for emitted load/cancel effects;
  native journeys only for visible readiness and real shortcut behavior.

### 2. Model Management and model-picker lifecycle

- **Sources:** [`SettingsModelManagementPanel.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/SettingsModelManagementPanel.swift)
  and [`ModelPickerBar.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/ModelPickerBar.swift).
- **View-owned concern:** both surfaces own catalog loading/generation,
  download-job reconciliation, deletion confirmation/result state, hardware
  fit decisions, and refresh-after-mutation behavior. Model Management also
  owns folder, favorites, filtering, storage, and capability reconciliation.
- **Impact/evidence:** This is the control plane for download, start, delete,
  storage location, and model discoverability. Incorrect reconciliation can
  show stale availability or repeat a destructive action; the two-step
  Download/Start escape is recorded in
  [PR #2053](https://github.com/raullenchai/Rapid-MLX/pull/2053).
- **Extract next:** a shared catalog/action coordinator with explicit refresh
  generations and outcomes for download, delete, folder change, and cache
  reconciliation. Presentation-only query/sort/hover state can remain local.
- **Cheapest effective layer:** Swift unit tests for action/state transitions
  and stale generations; integration tests against `DownloadManager` and
  catalog clients; native journeys for confirmation sheets and file panels.

### 3. App-shell launch and model handoff

- **Source:** [`ContentView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/ContentView.swift).
- **View-owned concern:** selected alias, section routing, telemetry gate,
  onboarding handoff, launch auto-start, catalog generations, pending download
  continuation, readiness actions, and server-state reactions meet in the app
  shell.
- **Impact/evidence:** Launch restoration previously displayed an unsolicited
  low-memory sheet; see the
  [ledger record](escaped-gui-regressions.md#app-launch-showed-an-unsolicited-low-memory-sheet--pr-2053).
  Failures here affect every session and can cross onboarding, Chat, Images,
  Audio, and Settings.
- **Extract next:** a launch/handoff policy that turns persisted selection,
  consent, catalog, download, and server events into explicit intents. Do not
  move navigation or sheet rendering into the policy.
- **Cheapest effective layer:** Swift unit tests for launch and handoff decision
  tables; integration tests for emitted server/download calls; relaunch
  XCUITest for the window/sheet boundary.

### 4. First-run onboarding orchestration

- **Source:** [`QuickstartView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/QuickstartView.swift).
- **View-owned concern:** `QuickstartCoordinator` already owns most phase logic,
  but the view still coordinates catalog/download tasks, cancel requests,
  hardware snapshot, server reactions, selection mirroring, and recovery
  actions across a large surface.
- **Impact/evidence:** Visible step skipping and noisy cached variants escaped
  in a released novice journey; see the
  [step record](escaped-gui-regressions.md#onboarding-skipped-a-visible-step--issue-2033)
  and [variant record](escaped-gui-regressions.md#onboarding-showed-noisy-cached-model-variants--issue-2033).
- **Extract next:** finish the existing coordinator boundary by feeding it
  typed catalog/download/server events and returning effects, rather than
  adding a second onboarding state object.
- **Cheapest effective layer:** extend existing coordinator unit tests and add
  effect-contract tests; native journey/geometry coverage owns visible timing,
  focus, and layout.

### 5. Chat composer and conversation-scoped UI state

- **Source:** [`ChatView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/ChatView.swift).
- **View-owned concern:** attachment ownership is now extracted, but the view
  still coordinates input routing, model capability rejection, send admission,
  conversation-instruction drafts, focus/scroll requests, and message edit and
  deletion sessions.
- **Impact/evidence:** Async attachment completion crossed conversation
  ownership before extraction; see the
  [ledger record](escaped-gui-regressions.md#async-chat-attachments-crossed-conversation-ownership--pr-2265).
- **Extract next:** after the active attachment-journey work lands, isolate
  composer admission and conversation-instruction edit sessions. Do not fold
  rendering, Markdown, scrolling, or AppKit text composition into that type.
- **Cheapest effective layer:** Swift unit tests for admission/session identity
  and immutable submissions; native journeys for picker, paste, drag/drop,
  keyboard focus, and message menus.

### 6. Sidebar conversation/folder operations

- **Source:** [`SidebarView.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/SidebarView.swift).
- **View-owned concern:** rename sessions, folder prompts, archive visibility,
  delete confirmations, collapsed folders, drag/drop targets, and day-boundary
  refresh are held together in view state.
- **Impact/evidence:** No escaped regression is recorded yet, but these actions
  mutate conversation organization and include destructive confirmation and
  drag/drop boundaries. A stale session can rename, move, or delete the wrong
  item.
- **Extract next:** a conversation-operation session keyed by stable IDs, with
  explicit begin/commit/cancel and stale-target rejection. Keep hover and row
  expansion presentation local unless persistence is intended.
- **Cheapest effective layer:** Swift unit tests for ID ownership and operation
  transitions; native journeys for focus, menus, confirmation, and drag/drop.

### 7. Connector editing and restart reconciliation

- **Sources:** [`SettingsConnectorsPanel.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/SettingsConnectorsPanel.swift)
  and [`MCPServerEditorSheet.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/MCPServerEditorSheet.swift).
- **View-owned concern:** edit/remove sessions, enable changes, restart state,
  validation feedback, and editor drafts cross the panel/sheet boundary. The
  source already documents a user-hit lifetime bug caused by putting durable
  reload state in `@State`.
- **Impact/evidence:** Misowned state can lose edits or show a connector as
  applied before the required restart/reload completes. The source comment is
  the current evidence; no escaped-regression ledger record exists yet.
- **Extract next:** a connector edit transaction with validated draft,
  replacement identity, persistence result, and restart-required outcome.
- **Cheapest effective layer:** Swift unit tests for validation and transaction
  outcomes; integration tests for config persistence/restart effects; native
  journey for sheet focus and confirmation.

### 8. Performance-settings edit/reload state

- **Source:** [`SettingsPerformancePanel.swift`](../../../apps/rapid-mac/Sources/Rapid/UI/SettingsPerformancePanel.swift).
- **View-owned concern:** selected alias, catalog refresh, effective config
  bindings, apply errors, launched flags, and reload actions are coordinated in
  the view while some durable state already lives in `SettingsRouter`.
- **Impact/evidence:** Incorrect ownership can apply a setting to the wrong
  alias or display saved settings that are not active. No escaped regression is
  currently recorded, so this ranks below surfaces with observed failures.
- **Extract next:** an alias-keyed edit session that distinguishes persisted,
  effective, launched, dirty, and reload-required values.
- **Cheapest effective layer:** table-driven Swift unit tests for edit/reload
  decisions and alias switching; integration tests for persistence; native
  journey only for control binding and error presentation.

## Maintenance rule

Update this inventory in the same PR that materially moves one of these
boundaries. Link the new type and tests, narrow or remove the view-owned concern,
and re-rank only when new escape evidence or user impact changes the order.
New surfaces should enter the list only when they own non-rendering behavior;
the count of `@State` properties alone is not evidence.
