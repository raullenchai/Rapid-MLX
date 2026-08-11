# User Flows — Manual Smoke Test Reference

This document is the authoritative reference for every user-facing flow in Rapid for macOS, the contract each flow has to honour, and the result of the most recent end-to-end smoke pass. Future contributors land here before touching the chat surface, server lifecycle, model-management, Connectors, or Quick Ask panel.

Every entry has the same shape:

- **Trigger** — what the user does
- **Expected** — what should happen, end-to-end
- **Touches** — code surfaces involved (paths into the repo)
- **Last smoke result** — verified, blocked, or known-issue, with date
- **Known issues** — open bugs filed against this flow

> **Contract refresh (2026-08-11).** Flow definitions were reconciled against
> `main` through **v0.12.9**. Verification dates below remain historical evidence;
> they do not imply that an older build defines today's surface. The deterministic
> GUI journeys live in `scripts/gui-golden-flows.sh` and run in CI; flows that still
> require a physical keyboard or an external service say so explicitly.

The smoke pass that backs the "Last smoke result" column runs against a fresh `Rapid-MLX Desktop.app` built by `scripts/build.sh` and driven by `scripts/walkthrough.sh` (F1–F10 via **AppleScript AXIdentifier** clicks — see the identifier inventory near the end) plus `scripts/release-smoke.sh` for structural launch. Flows that depend on real keyboard or pointer input require macOS Accessibility permission for the terminal hosting the harness, plus a real hardware key on the items that exercise the global Carbon hotkey.

**Driving the UI: use AXIdentifier clicks, not raw coordinates.** `cliclick c:X,Y` lands at the right pixel but does **not** trigger SwiftUI button `onClick` / `.onSubmit` handlers — synthetic `CGEvent`s don't reach SwiftUI's gesture recognizers. Typing (`cliclick t:`) and filling via a control that IS reachable (e.g. an example-prompt chip) work; the send button and Enter/Cmd+Return do not fire from cliclick. Re-confirmed on v0.10.6, 2026-07-21. The supported path is `scripts/walkthrough.sh`, which clicks controls by their `.accessibilityIdentifier` via `System Events`. To verify a chat end-to-end without the GUI send, hit the sidecar directly: the app spawns `…/rapid-mlx … serve <alias> --port 8000`; read its per-launch bearer from the sidecar process env (`RAPID_MLX_API_KEY`, `RAPID_MLX_WATCHDOG_PPID` == app pid) and `curl localhost:8000/v1/chat/completions -H "Authorization: Bearer $KEY"`.

---

## Flow 1 — First launch & onboarding

First run has **two distinct surfaces**, shown in order:

**(a) The Quickstart wizard** (`UI/QuickstartView.swift:551`, driven by `QuickstartCoordinator`) — replaced the old single dense setup card in **v0.10.0**. It short-circuits the chat surface for a brand-new user with no model on disk. Three steps (progress dots `total: 3`):

- **Step 0 — Welcome** (`welcomeStep`): brand mark + "WELCOME TO RAPID-MLX / Fast, free AI that runs on your Mac." Primary **Get started** (`Quickstart.GetStarted`) and **Skip for now** (`Quickstart.Skip`, Esc-bound).
- **Step 1 — Choose your first model** (`chooseModelStep`): a recommended starter card (**`bonsai-1.7b-2bit`**, tool-capable ternary ~0.5 GB — swapped in from the old 0.6B in v0.10.2), "OR PICK A BIGGER ONE" trade-up cards, and **Browse all models →** (`Quickstart.BrowseAll`).
- **Step 2 — Download / Starting** progress card (`Phase.downloading` → `.starting` → `.ready`): on-brand progress with live percentage and a "runs privately on your Mac" note (v0.10.3). Also handles `lowDiskWarning` and `failed` phases (`Quickstart.Retry`). Starter pulls from the R2 mirror first (zero-config; v0.7.4), with an instant-on path when weights are bundled.

**(b) The OnboardingTour** (`UI/OnboardingTour.swift:89`) — a 4-page centred feature sheet shown **after** the server reaches ready (gated by `OnboardingState.hasSeen`, wired at `ContentView.swift:937`). Pages: (1) Pick a model, (2) Set a system prompt, (3) Tools the model can call, (4) Ask without switching apps (Quick Ask ⌥Space). Skip / Next / Esc fire correctly since v0.8.14. Does not re-show once completed.

**Expected (invariants).**

1. Within ~15 s the SwiftUI scene renders; the main window appears at its default frame (historically 1200×820; a wider default is in use on recent builds).
2. On a genuinely-first install any stale `sessions.json` from a prior user is archived (v0.8.14) — no stranger's chats in the first view. A fresh session is written with an empty `messages` array and `alias: ""`.
3. A single `MenuBarExtra` glyph appears (see F7 — consolidated to one icon in v0.8.20; visibility fix v0.10.0).
4. No crash report lands in `~/Library/Logs/DiagnosticReports/Rapid*`.
5. The Quick Ask global hotkey registers with `Option+Space`; the chord blob lands in `UserDefaults["rapid.quickask.v1.chord"]`.

**Touches.** `UI/QuickstartView.swift`, `UI/OnboardingComponents.swift`, `UI/OnboardingTour.swift`, `Server/QuickstartCoordinator.*`, `Chat/SessionStore.swift`, `QuickAsk/GlobalHotkey.swift`, `Server/BundledModel.swift`, `scripts/release-smoke.sh`.

**Last smoke result.** Structural launch of v0.10.6 verified 2026-07-21 (cliclick dogfood): app renders, bottom bar reads "Rapid Desktop 0.10.6 — up to date", starter model loads to **Ready**. Wizard step transitions + Skip: re-smoke pending on a truly-fresh Application Support.

**Known issues.** None open. (Historical: the two-surface ordering above is intentional, not a bug.)

---

## Flow 2 — Quick Ask global hotkey

**Trigger.** Press the configured global chord (default `Option+Space`) from anywhere, with or without Rapid frontmost.

**Expected.**

1. A non-activating `NSPanel` (720×480) drops in at screen-centre; the caller app keeps focus.
2. The panel's compose field is first-responder; typing lands in `QuickAskView`.
3. `Return` submits and the panel streams a reply against the running rapid-mlx server.
4. `Esc` dismisses the panel; `quickAskEscMonitor` consumes the key without delivering it to the previous app.

**Touches.** `QuickAsk/GlobalHotkey.swift`, `QuickAsk/QuickAskPanel.swift`, `QuickAsk/QuickAskController.swift`, `QuickAsk/QuickAskConfig.swift`, `QuickAsk/QuickAskChordPreset.swift`, `RapidApp.swift` (`installQuickAskHotkey`).

**Last smoke result.** No user-facing changes v0.7.0→v0.10.6; default chord confirmed still `⌥Space` (`GlobalHotkey.swift:229`). Registration verified 2026-06-13. Real-keystroke fire **requires manual user verification** — see Known Issue 2-A.

**Known issues.**

- **2-A (won't fix, macOS design).** Synthetic `CGEventPost` events do not trigger `RegisterEventHotKey` handlers — Carbon hotkeys only respond to genuine `IOHID` keyboard events. Every automated harness (`cliclick`, `osascript "key code"`, `CGEventPost`) can verify Rapid *registers* the chord but not that pressing it opens the panel. Manual real-keyboard testing is the only path.

---

## Flow 3 — Send a chat message (main window)

**Trigger.** With Rapid frontmost, focus the compose editor (click or `Cmd+L`), type, then submit: **`Return`** submits; **`Shift+Return`** inserts a newline; **`Cmd+Return`** also submits (Slack / Linear shape). Up-arrow in an empty field recalls the last user message.

**Expected.**

1. The compose editor receives the keystrokes. It is `ComposeTextEditor` (an `NSViewRepresentable`, `ChatView.swift:3732`) hosting `AutosizingTextView` (the `NSTextView` subclass, `:3689`). Submit is intercepted in `Coordinator.textView(_:doCommandBy:)` (`:3837`) so `Return`/`Cmd+Return` don't double-fire against the send button (which carries no `.keyboardShortcut` by design).
2. `ChatViewModel.send(_:alias:)` appends a `.user` message and starts a stream against `http://127.0.0.1:8000/v1/chat/completions` (bearer-authed since the per-launch-secret work).
3. The assistant placeholder appears immediately (`status: .streaming`), press-feedback (spring depress) plays on the send/chips (v0.10.6), and VoiceOver announces stream start + completion (v0.8.20).
4. SSE chunks arrive on `ChatStreamClient`: `delta.reasoning_content` → `reasoning_content`, `delta.content` → `content`; the UI keeps the tail visible and the composer stays on-screen even once the conversation fills the window (macOS 14/15 clip fixed v0.8.16).
5. Text scales with the system Dynamic Type setting across the transcript and ~80 text sites (v0.10.6).
6. On end-of-stream the row flips to `.complete` with a token-count / model footer; no stray machine text is shown when a tool-capable model chose not to call a tool (v0.10.6). Busy / low-memory conditions surface a plain-language message, not a raw error (v0.10.0 / v0.8.19).
7. `sessions.json` round-trips the new user+assistant pair via the UUID-suffixed atomic write; interrupted `.streaming` rows are cleaned up on next load rather than sticking on "Thinking…" (v0.10.0).

**Touches.** `UI/ChatView.swift` (`ComposeTextEditor`, `AutosizingTextView`, `focusComposeShortcut`, `ChatView.SendOrStopButton`), `Chat/ChatViewModel.swift`, `Chat/ChatStreamClient.swift`, `Chat/SessionStore.swift`.

**Last smoke result.** Engine end-to-end verified 2026-07-21 on v0.10.6: `curl` to the app's sidecar (`:8000`, `qwen3-0.6b-4bit`, bearer) returned a correct completion. GUI keystroke→send is **cliclick-blind** (see the driving note at the top) — drive via `walkthrough.sh` (`ChatView.SendOrStopButton`) or a real keyboard.

**Known issues.**

- **3-A (partial fix landed 2026-06-13, P2 follow-up).** Original report was that Rapid's main `Window` scene didn't appear in the AX hierarchy because `System Events → tell process "Rapid" → count of windows` returned `0`. Subsequent investigation surfaced two facts that re-graded this issue from P1 to P2:
  1. Under the smoke-pass harness (an SSH+tmux session to the user's local Mac), `System Events → count of windows` also returns `0` for **Calculator** and reports `no app frontmost`. That signal is a remote-execution artifact of the AppleScript scripting bridge through SSH, not a Rapid bug. Conclusion: the System Events probe is not a valid diagnostic in this harness shape.
  2. Querying the AX surface **directly** via `AXUIElementCopyAttributeValue(AXUIElementCreateApplication(pid), kAXWindowsAttribute, ...)` returns one entry — but that entry resolves to the AXApplication itself rather than the AppKitWindow underneath. `AXMainWindow` / `AXFocusedWindow` / `AXFocusedUIElement` all behave the same way. The SwiftUI scene-graph accessibility-bridge is gated on `AXEnhancedUserInterface` being `true` — VoiceOver flips that on startup, but without VoiceOver running the bridge stays dormant. An external setter receives `-25208` (`kAXErrorNotImplemented` — the AXApplication element implements no settable `AXEnhancedUserInterface` for a foreign process; NOT `-25205` / `kAXErrorAttributeUnsupported` as earlier notes claimed). Per issue #173, on macOS 15+ the *in-process* set returns the same `-25208`, so the bridge stays dormant for non-VoiceOver users there too — the launch-time set is a benign no-op on those releases, not a regression.

  Landed fix (`fix(a11y): pin activation policy + enable AX bridge at launch`): `NSApp.setActivationPolicy(.regular)` plus a self-set of `AXEnhancedUserInterface = true` from inside the app, both in `AppDelegate.applicationDidFinishLaunching`. Post-fix probe confirms both flags take effect (`false → true` and `policy → 0`). The remaining gap — `kAXWindowsAttribute` still returning the AXApplication instead of the AppKitWindow — is a SwiftUI `Window` scene framework limitation that needs deeper work (a custom `NSAccessibilityElement` wrapper around the chat compose `NSTextView` is the likely shape). Sighted users on real keyboard + mouse are unaffected; screen-reader users see an improved surface but not a fully-navigable one.
- **3-B (open, harness-only, not a launch blocker).** `cliclick`-injected keystrokes do not reach Rapid's compose input when the test harness is an SSH+tmux session. Even after pinning the activation policy and enabling the AX bridge, `NSRunningApplication.activate` and `AXFrontmost = true` both silently no-op from this remote context (the AX `set` call returns success but the value never changes). Conclusion: the OS protects user focus from being stolen by background SSH sessions. The user driving the app locally with a real mouse and keyboard sees none of this — flows 3-7 work for them as designed.

---

## Flow 4 — Settings panel

**Trigger.** `Cmd+,` while Rapid is frontmost, or `Rapid → Settings…`.

**Expected.**

1. The Settings window opens at its last-left size (restored from `NSWindow Frame Settings`).
2. The category rail is a native, arrow-key-navigable list with VoiceOver
   semantics. `ForEach(Category.allCases)` renders exactly six categories, in
   declaration order: **Model Management, Tools, Connectors, Appearance, Privacy,
   App**. Model Management is the default. The former stand-alone Models tab was
   folded into it; there is no Permissions, Web Search, Sampling, Quick Ask,
   Keyboard, or Storage category in this app.
3. Clicking a category swaps the body. Model preferences and tool enablement use
   `UserDefaults`; provider secrets use Keychain; connector configuration is
   persisted by `MCPConfigStore`. Web-search provider settings and browse approval
   mode are subsections of **Tools**. Updates and the inference-engine path are in
   **App**.
4. `Cmd+W` closes Settings without quitting the app.

**Touches.** `UI/SettingsView.swift`, `UI/SettingsModelManagementPanel.swift`,
`UI/SettingsToolsPanel.swift`, `UI/SettingsConnectorsPanel.swift`,
`Tools/KeychainStore.swift`, `MCP/MCPConfigStore.swift`.

**Last smoke result.** The six-category contract is pinned by Swift tests and the
settings GoldenFlow. Persistence paths have focused unit coverage. Physical
keyboard traversal remains a manual-local check.

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 5 — Model picker & server start / stop

**Trigger.** Click the model picker in the chat header, choose an alias, click **Start** / **Download & start** / **Stop** on the status pill.

**Expected.**

1. The picker enumerates cached models via `ModelCatalog`, with a dedicated
   **Quickstart** row followed by **Recommended for your N GB Mac**. Each RAM tier
   has one measured smart pick and, where available, one faster/lighter
   alternative. The remaining catalog is split into runnable **All models** and
   **Not fit for this Mac** sections; it no longer exposes the retired
   Default/Speed/Quality/Coding/Vision role matrix.
2. Selecting an alias updates `ContentView.alias` and persists in the active session's `alias`.
3. **Start** → `ServerManager.start(alias:)`; the pill transitions `idle → starting → ready(alias)` over ~5–15 s. AutoStart prefers a cached runnable model over a RAM-bucketed default that would trigger a fresh large download (v0.8.14).
4. The child rapid-mlx process spawns as a process-group leader (`posix_spawn` + `POSIX_SPAWN_SETPGROUP`); shutdown sends `kill(-pgid, SIGTERM)` then `SIGKILL`. An idle-state sidecar crash surfaces and auto-respawns, bounded by a respawn budget so crash loops can't run away; **Stop** reliably stops even mid-respawn (v0.7.14 / v0.7.15).
5. **Stop** tears the subprocess down without zombies.

**Touches.** `Server/ServerManager.swift` (process-group spawn + shutdown inline ~936–1006), `Server/ModelCatalog.swift`, `UI/ModelPickerBar.swift` (`ModelPickerBar.PrimaryButton`), `UI/ContentView.swift`.

**Last smoke result.** Process-group cleanup verified 2026-06-13; qwen3-0.6b-4bit load→Ready verified 2026-07-21 (v0.10.6 dogfood). Picker/Start/Stop click test via `ModelPickerBar.PrimaryButton` — re-smoke pending.

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 6 — Download a model

**Trigger.** Click **Download** on a non-cached alias (picker or Model Management), or `DownloadManager.startDownload(alias:)`.

**Expected.**

1. `DownloadManager.startDownload` re-resolves `ServerLocator.find()` (picks up a `brew upgrade` since launch) then pulls the alias. Source is the **R2 mirror by default** (v0.7.4, faster + rate-limit-free), with the HuggingFace path as fallback.
2. Progress streams into the UI with per-tick speed + Chrome-style ETA ("683 KB/s · 5 min left", v0.7.12), smooth within a single big shard (v0.7.11), and covers the R2-mirror phase, not just HF (v0.7.9). Multi-hour downloads on slow links are no longer killed at 30 min (deadline became a stall window, v0.7.13).
3. Cancelling mid-download flips the job to `.cancelled` via the `cancellingProcesses` identity tracker, so a late cancel can't rewrite a normal completion. Quickstart models land in the correct on-disk folder (v0.8.20) and honour a custom models folder (F11).
4. On success the alias becomes available to the picker + Model Management.

**Touches.** `Server/DownloadManager.swift`, `Server/ServerLocator.swift`, `Server/ModelsFolderPreference.swift`, `Services/ModelCacheActions.swift`.

**Last smoke result.** Subprocess wiring verified 2026-06-13 (`DownloadManagerTests` real-subprocess suite). UI cancel-button click — re-smoke pending (3-B).

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 7 — Clean quit (`Cmd+Q`) & menu-bar lifecycle

**Trigger.** `Cmd+Q` with Rapid frontmost, or **Quit** from the single menu-bar extra.

**Expected.**

1. There is **one** menu-bar (tray) icon (consolidated from two in v0.8.20) exposing open, new chat, live model status, settings, quit; it renders reliably on recent macOS (visibility fix v0.10.0).
2. `AppDelegate.applicationWillTerminate` runs synchronously on the main thread.
3. `SessionStore.finalizeStreamingForTermination` flips dangling `.streaming` rows to `.complete` and strips orphan `tool_calls` markers before the flush.
4. `SessionStore.flushSync` writes the canonical envelope via the UUID-suffixed atomic-replace path.
5. `ServerManager.shutdownSync` SIGTERMs the whole rapid-mlx process group (SIGKILL after 2 s). A parent-PID watchdog means even force-quit / `kill -9` leaves **no** orphan sidecar holding the port or RAM (v0.8.14). `DownloadManager.shutdownSync` tears down in-flight pulls.
6. `CrashReporter.recordCleanShutdown` clears the launch marker so the next launch doesn't misclassify the exit.

**Touches.** `RapidApp.swift` (`AppDelegate`, `MenuBarExtra`), `Chat/SessionStore.swift`, `Server/ServerManager.swift`, `Server/DownloadManager.swift`, `Telemetry/CrashReporter.swift`.

**Last smoke result.** Verified 2026-06-13 and again 2026-07-21 (v0.10.6): `osascript quit` exits cleanly, port 8000 released, no orphan rapid-mlx, no crash report. **Instance-scoped teardown caution:** a user may have their own headless `rapid-mlx serve … --port 88xx` running; quit only the GUI app (its watchdog'd sidecar dies with it) — never bare `pkill -f rapid-mlx`.

**Known issues.** None.

---

## Flow 8 — Auto-update

**Trigger.** Background poll in `UpdateChecker` finds a release whose `version` is strictly newer than `Bundle.main.shortVersionString`.

**Expected.**

1. The menu-bar extra label changes to "Update available — v\<X.Y.Z\>"; in-app checks are reliable (extra path + clean fallback, v0.10.4).
2. Clicking opens the **Update Rapid** window offering in-app install via `Installer` plus an "Open release page" fallback. Release notes render as **formatted, CHANGELOG-driven narrative** (v0.8.3 / v0.8.18), not a raw commit list or raw Markdown.
3. The release-page URL is host-allowlisted; the DMG URL is download-host-allowlisted (rejects HTTP + foreign hosts); the version string is gated by strict-numeric grammar (`"0.5.13.evil"` rejected).
4. A failed Finder "Replace" on `/Applications` is detected and offers "Open update dialog" (v0.7.7). The "Setup didn't finish" recovery overlay promotes a **"Download update vX.Y.Z"** CTA when a newer release exists (v0.8.14).

**Touches.** `Updater/UpdateChecker.swift`, `Updater/Installer.swift`, `UI/UpdateInstallView.swift`, `UI/FailedReplaceBanner.swift`, `UI/SettingsView.swift` (App tab: `Settings.App.Update*`).

**Last smoke result.** Validation layer pinned by `UpdateCheckerTests`. End-to-end menu-bar click flow — re-smoke pending (3-B). Live latest.json for 0.10.6 confirmed servable 2026-07-21.

**Known issues.** Inherits 3-B.

---

## Flow 9 — Pop a conversation into its own window

**Trigger.** **File → Open Conversation in New Window** (⇧⌘O), or right-click a session in the sidebar → **Open in New Window**.

**Expected.**

1. A new window titled `Conversation` shows the picked session's full transcript, including the hybrid-thinking `reasoning_content` under a collapsed "Thinking" disclosure.
2. The popped window is pinned to that session UUID for life — switching `activeID` in the main window does not change it (core contract).
3. `openWindow(id:value:)` with the same UUID reactivates the existing window (value-based dedup), no stacking.
4. The popped window is read-only (no compose row); a **Reply in Main Window** header button sets `store.activeID` and re-focuses the main window.
5. Streaming updates flow in automatically (`SessionStore` is `@Observable`, re-resolved per render).
6. A deleted/unavailable session renders a "Conversation not available" state (with `PoppedConversation.CloseUnavailableWindow`) rather than crashing.

**Touches.** `UI/PoppedConversationView.swift`, `RapidApp.swift` (`WindowGroup("Conversation", …, for: UUID.self)`), `UI/SessionsSidebar.swift`.

**Last smoke result.** No user-facing changes v0.7.0→v0.10.6. Shape verified 2026-06-13 by `PoppedConversationViewTests`. End-to-end open-window flow — re-smoke pending (3-B).

**Known issues.** Inherits 3-A / 3-B. The popped scene uses `WindowGroup` (not `Window`), so its AX hierarchy may differ from the main window's.

---

## Flow 10 — Model Management (library browser)  *(NEW — v0.10.0)*

**Trigger.** **Settings → Model Management** (also reachable from the Quickstart "Browse all models →").

**Expected.** A file-manager-style cache inspector separate from the inline picker (`SettingsModelManagementPanel.swift:32`, issue #210):

1. Search + filter (All / Cached / Not cached) + sort (`…Search`, `…Filter`, `…SortMenu`).
2. One row per alias with a status badge; per-model **Download / Cancel / Delete / Retry** (`…Download.<alias>` etc.); pin favourites via a star (`…Favorite.<alias>`).
3. A **"Recommended for your N GB Mac"** section (RAM-bucketed role picks, issue #507; `…RecommendedHeader`, `…Recommended.<role>`) with segmented Accuracy / Speed meters (`…MeterLegend`).
4. A **"Total: X GB across N models"** footer (`…Footer`).
5. The models folder controls (F11) sit at the top of this panel.

**Touches.** `UI/SettingsModelManagementPanel.swift`, `Services/ModelCacheActions.swift`, `Server/ModelCatalog.swift`, `Server/DownloadManager.swift`.

**Last smoke result.** Definitions current to v0.10.6; re-smoke pending.

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 11 — Custom models folder  *(NEW — v0.10.0)*

**Trigger.** **Settings → Model Management → models-folder section** (`SettingsModelManagementPanel.swift:202`, issue #503).

**Expected.**

1. **Choose…** (`Settings.ModelManagement.ChooseFolder`) points Rapid at any folder (e.g. an external drive) for downloaded models; **Use default** (`…UseDefaultFolder`) reverts. Current path shown at `…FolderPath`.
2. Backed by `Server/ModelsFolderPreference.swift`; the new location takes effect on the next model load / download.
3. If the chosen drive is unplugged, an "unavailable" warning renders (`…FolderUnavailable`) rather than failing silently.

**Touches.** `UI/SettingsModelManagementPanel.swift`, `Server/ModelsFolderPreference.swift`, `Server/DownloadManager.swift`, `Server/ModelCatalog.swift`.

**Last smoke result.** Definitions current to v0.10.6; re-smoke pending.

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 12 — Connectors (MCP)  *(REBUILT — issue #1716)*

**Trigger.** **Settings → Connectors** (`SettingsConnectorsPanel.swift`).

**Architecture.** The engine hosts the MCP connections (`vllm_mlx/mcp/*`); the app owns
the config, the visibility and the consent. The app writes
`~/.config/rapid-mlx/mcp.json`, passes `--mcp-config` at spawn, reads state back over
`GET /v1/mcp/servers` + `/v1/mcp/tools`, and executes an approved call through
`POST /v1/mcp/execute`. The engine deliberately does **not** inject MCP tools into
`/v1/chat/completions` (`MCPClientManager.get_merged_tools` exists and is uncalled), so the
tool loop stays in `ChatViewModel` and the consent gate can live on screen.

**Expected.**

1. A master **Enable connectors** toggle (`Settings.Connectors.MasterToggle`, opt-in, default off).
   Off means `--mcp-config` is not passed at all, so the engine has no MCP subsystem — not merely
   zero servers.
2. **Add** (`…AddButton`) opens `MCPServerEditorSheet` (name, transport stdio/SSE, command/args/env
   or URL, per-server enable). Server names are validated against `[A-Za-z0-9_-]{1,32}` and must not
   contain `__`, because the engine namespaces every tool as `server__tool` and splits on the first
   `__`: the name has to stay a legal OpenAI function name and an unambiguous namespace half.
3. Each server row has a status dot, a one-line state (`…Row.Status.<name>`: *Connected · N tools*,
   the rejection reason, or *Start a model to connect*), an on/off toggle (`…Row.Toggle.<name>`) and
   an edit/remove menu (`…Row.Menu.<name>`).
4. Saving calls `POST /v1/mcp/reload`, so an edit applies **without restarting the model**. Only a
   change to the master toggle needs a restart (the flag is read once at spawn) — that raises the
   **Restart to apply** banner (`…RestartButton`). A reload that fails raises the same banner rather
   than silently swallowing the edit.
5. Connected tools are listed with per-tool switches (`Settings.Connectors.Tool.Toggle.<name>`), each
   tagged with its source server. A disabled tool is stripped from the request body AND refused at
   dispatch.
6. At inference time, the first call to each connector tool raises `MCPToolApprovalSheet`
   (`ContentView.swift`): "Run \<tool\>?" with the server, the namespaced name, the display-safe
   arguments, and **Allow once / Always allow / Don't allow**
   (`ToolApproval.MCP.{Allow,AlwaysAllow,Deny}`). **Always allow** is scoped to that one tool.
7. A blanket **Auto-approve all tool calls** switch (`…AutoApproveToggle`) plus **Reset** of
   remembered grants (`…ResetApprovals`) live in the approvals card.
8. A misbehaving connector cannot take the chat surface down: `init_mcp` is non-fatal and one bad
   config entry is dropped-and-reported rather than failing the whole load.

**Touches.** `UI/SettingsConnectorsPanel.swift`, `UI/MCPServerEditorSheet.swift`,
`UI/ContentView.swift` (`MCPToolApprovalDialog` / `MCPToolApprovalSheet`), `MCP/*`
(`MCPServerConfig`, `MCPConfigStore`, `MCPCatalog`, `MCPToolApprovalStore`, `MCPToolRegistry`,
`CompositeToolRegistry`), `Server/ServerManager.swift` (`serveArguments`,
`mcpConfigPathProvider`), and engine-side `vllm_mlx/server.py` (`init_mcp`, `reload_mcp`),
`vllm_mlx/routes/mcp_routes.py`, `vllm_mlx/mcp/config.py` (tolerant load).

**Last smoke result.** Unit-covered by `MCPConnectorsTests.swift` (config round-trip, name
validation, launch-flag gating, approval semantics, dispatch refusal) and
`tests/test_mcp_resilience.py` (non-fatal init, per-entry isolation, reload). End-to-end
connector smoke pending.

**Known issues.** The approval sheet's three buttons carry AXIdentifiers, but the sheet is only
reachable when a real connector is configured and a model actually calls one of its tools — the
golden-flow harness has no fixture connector, so that step is manual for now.

---

## Flow 13 — Built-in web tools and browse approval

**Trigger.** Enable tools under **Settings → Tools**, then let a tool-capable
model call `web_search`, `browse`, or `weather`.

**Expected.**

1. The three built-in tools are individually enabled or disabled. A disabled
   tool is omitted from the request and refused again at dispatch.
2. `web_search` uses DuckDuckGo, Brave, or Tavily as selected under Tools;
   provider keys are stored in Keychain. `weather` uses Open-Meteo and needs no
   approval.
3. `browse` accepts only HTTP(S), rejects private/loopback/reserved destinations
   before prompting, and shows the display-safe host and complete URL in a
   per-fetch approval sheet (`ToolApproval.Browse.{Allow,Deny}`). Redirects are
   checked and approved under the same policy.
4. **Approve every page automatically** changes browse from per-fetch approval
   to a persisted blanket mode. This does not approve MCP tools; connector grants
   remain per tool under **Settings → Connectors**.
5. The app ships no filesystem tools, shell execution, permissions matrix, or
   filesystem sandbox. Those capabilities must not be inferred from the engine
   or from the retired rapid-desktop contract.

**Touches.** `UI/SettingsToolsPanel.swift`, `UI/ContentView.swift`
(`BrowseApprovalSheet`), `Tools/BuiltinToolRegistry.swift`,
`Tools/BrowseApprovalStore.swift`, `Tools/BrowseSSRFGuard.swift`,
`Tools/{WebSearchTool,BrowseTool,WeatherTool}.swift`.

**Last smoke result.** Enable/disable, provider persistence, SSRF refusal, approval
decisions, redirects, and cancellation are unit-covered; the built-in-tools
GoldenFlow exercises the rendered settings and chat path.

**Known issues.** DNS rebinding and pagination/result-bound hardening remain in
#1535. A deliberately declined or failed result can still be mischaracterised by
weak models; investigation is tracked in #1582.

---

## Flow 14 — In-chat controls: system prompt & tools toggle  *(NEW surface — v0.7.x→v0.10.x)*

**Trigger.** In an active conversation, use the compose-row controls.

**Expected.**

1. **Per-conversation system prompt.** A `systemPromptHeader` chip (`ChatView.swift:530`); unset shows a **Set system prompt** link (`:519`) opening `SystemPromptSheet`, persisted per-session via `SessionStore.setSystemPrompt(id:)` and reloaded on session switch.
2. **Per-turn tools toggle.** A wrench `toolsChip` in the compose row (`ChatView.swift:1688`) opens a popover to enable/disable individual tools for the turn (shows active/count). Web search is **one tool inside this popover** (and Settings → Web Search for provider/keys) — there is **no** separate web-search compose toggle. The empty-state "Search the web" example chip (`:1278`) is a prompt seed, not a toggle.
3. **Upgrade nudge.** After ~3 messages on the bundled starter, a nudge banner above the composer offers a one-click background upgrade download with dismissal modes (v0.7.2, retuned v0.10.2) — drives F6.

**Touches.** `UI/ChatView.swift` (`systemPromptHeader`, `toolsChip`, `SystemPromptSheet`, upgrade nudge), `Chat/SessionStore.swift`, `Tools/*`.

**Last smoke result.** Definitions current to v0.10.6; re-smoke pending.

**Known issues.** Inherits 3-A / 3-B.

---

## Flow 15 — Conversation management: delete, undo, search  *(NEW/expanded)*

**Trigger.** Sidebar interactions on sessions.

**Expected.**

1. **Delete** a conversation via the sidebar; a confirmation sheet (`DeleteSessionConfirmation.swift`, `DeleteSessionConfirm.Sheet` / `…DeleteButton` / `…SkipToggle` "don't ask again").
2. **Undo delete** (v0.10.6): a "Chat deleted" toast row with an **Undo** button (`Sidebar.UndoDelete`, ⌘Z) briefly restores the conversation.
3. **Search chats** (`SessionsSidebar.swift:557`): a "Search chats" field filters sessions (`filteredSessions`, logic in `Chat/MessageSearch.swift`); in-chat find is ⌘F (`ChatView.swift:702`).
4. **Resilience banners:** corrupt `sessions.json` shows a "Show in Finder / Dismiss" banner (`SessionLoadFailureBanner`); a restored session on a since-removed model shows a stale-alias banner (`StaleSessionAliasBanner`); "N chats skipped — restore backup" recovery (v0.8.20, via Storage tab).

**Touches.** `UI/SessionsSidebar.swift`, `UI/DeleteSessionConfirmation.swift`, `Chat/MessageSearch.swift`, `Chat/SessionStore.swift`, `UI/SessionLoadFailureBanner.swift`, `UI/StaleSessionAliasBanner.swift`.

**Last smoke result.** Definitions current to v0.10.6; re-smoke pending. New chat + session switching exercised 2026-07-21 (v0.10.6 dogfood) via `Sidebar.NewChat`.

**Known issues.** Inherits 3-A / 3-B.

---

## Accessibility identifier inventory (for `scripts/walkthrough.sh`)

`walkthrough.sh` clicks controls by `.accessibilityIdentifier` via `System Events` — the supported way to drive the UI (cliclick coordinates don't fire SwiftUI handlers; see the driving note at the top). Keep this inventory in sync when adding/removing identifiers. Current set (file:line → id):

- **Quickstart wizard** — `QuickstartView.swift`: `Quickstart.GetStarted` (:739), `Quickstart.Skip` (:757), `Quickstart.BrowseAll` (:818), `Quickstart.LowDisk.Continue` (:949), `Quickstart.LowDisk.Cancel` (:961), `Quickstart.Retry` (:1000). `OnboardingComponents.swift`: `Quickstart.Footer.Back` (:100), `Quickstart.Footer.Primary` (:114), `Quickstart.Choice.<alias>` (:193 recommended, :241 compact).
- **Chat** — `ChatView.swift`: `ChatView.SendOrStopButton` (:2211), `ChatView.NotReadyHint` (:1560), `ChatView.toolCallArtifactSuppressed` (:3074), `ChatView.toolNotCalledCaption` (:3114), `…dismiss` (:3127).
- **Model picker** — `ModelPickerBar.swift`: `ModelPickerBar.PrimaryButton` (:1407/:1417/:1427, three mutually-exclusive states).
- **Sidebar** — `SessionsSidebar.swift`: `Sidebar.NewChat` (:228), `Sidebar.LoadingSessions` (:262), `EmptySidebar.StartChatting` (:287), `Sidebar.UndoDelete` (:1173). `DeleteSessionConfirmation.swift`: `DeleteSessionConfirm.SkipToggle` (:206), `…DeleteButton` (:226), `…Sheet` (:234).
- **Footer** — `ContentView.swift`: `Footer.DesktopVersionPill` (:1975).
- **Settings** — `SettingsView.swift`: `Settings.QuickAsk.LaunchAtLogin` (:569), `Settings.WebSearch.{KeyField,ClearButton,SaveButton}.<providerID>`, `Settings.App.{HideDockOnCloseToggle,ResetDockOnboardingCTA,UpdateHeadline,UpdateCTA,UpToDate,Checking,Unknown,RecheckCTA}`. `SettingsModelManagementPanel.swift`: `Settings.Models.{ShowAllModelsToggle,AutoStartOnLaunchToggle}`.
- **Model Management** (prefix `Settings.ModelManagement.`) — `SettingsModelManagementPanel.swift`: `FolderPath`, `FolderUnavailable`, `ChooseFolder`, `UseDefaultFolder`, `Search`, `SortMenu`, `Filter`, `RecommendedHeader`, `Recommended.<role>`, `Footer`, `MeterLegend`, `Favorite.<alias>`, `Delete.<alias>`, `Download.<alias>`, `Cancel.<alias>`, `Retry.<alias>`, `Row.<alias>`, `Status.<text>`, plus `Recommended.{Delete,Download,Cancel,Retry}.<alias>`.
- **Connectors** (prefix `Settings.Connectors.`) — `SettingsConnectorsPanel.swift`: `MasterToggle`, `AutoApproveToggle`, `ResetApprovals`, `RestartButton`, `SubsystemError`, `AddButton`, `ConfirmRemove`, `CancelRemove`, `Row.Status.<name>`, `Row.Toggle.<name>`, `Row.Menu.<name>`, `Row.Edit.<name>`, `Row.Remove.<name>`, `Tool.Toggle.<name>`. `MCPServerEditorSheet.swift` (prefix `Settings.Connectors.Editor.`): `Name`, `Transport`, `Command`, `URL`, `Enabled`, `AddArgument`, `AddEnv`, `Allow`, `Cancel`. Pinned by `AccessibilityIdentifierInventoryTests`.
- **MCP tool approval sheet** — `ContentView.swift`: `ToolApproval.MCP.Allow`, `ToolApproval.MCP.AlwaysAllow`, `ToolApproval.MCP.Deny`.
- **Banners / misc** — `FailedReplaceBanner` (`FailedReplaceBanner.swift:90`), `StaleSessionAliasBanner{,.Dismiss}`, `SessionLoadFailureBanner{,.ShowInFinder,.Dismiss}`, `PoppedConversation.CloseUnavailableWindow` (`PoppedConversationView.swift:88`).

- **Settings → Tools** — `SettingsToolsPanel.swift`: `Settings.Tools.Toggle.<tool>` (`web_search`, `browse`, `weather`), `Settings.Tools.WebSearch.Backend` (the radio group) + `Settings.Tools.WebSearch.Backend.<duckduckgo|brave|tavily>`, `Settings.Tools.WebSearch.KeyField.<provider>`, `Settings.Tools.WebSearch.SaveKey.<provider>`, `Settings.Tools.WebSearch.KeyDashboardLink.<provider>` (the "Get a <provider> key" link — only rendered for a provider that has a dashboard URL, which is why it was missed on the first pass), `Settings.Tools.Browse.AutoApproveToggle`.
- **Settings → Privacy** — `SettingsView.swift`: `Settings.Privacy.TelemetryToggle`, `Settings.Privacy.Link.{PrivacyPolicy,License,Credits}` (each link keyed on the document it opens, not on its label). The toggle re-renders on press as of [#1623](https://github.com/raullenchai/Rapid-MLX/issues/1623); before that it wrote the preference while appearing to snap back — see `docs/gui-golden-flows.md`.
- **Conversation row controls** — `SidebarView.swift`: `Sidebar.Conversation.{Pin,Unpin}.<UUID>` (hover control, named for the action the press performs), `Sidebar.Conversation.Menu.<UUID>` (the ··· menu button), and its items `Sidebar.Conversation.Action.{Rename,Pin,Unpin,Archive,Unarchive,Delete}` (shared with the right-click menu). Delete confirmation: `Sidebar.DeleteConversation.{Confirm,Keep}`.
- **Message actions** — `ChatView.swift`: `ChatView.Message.{Copy,Edit,Retry,CancelEdit,SaveEdit}.<message UUID>`.
- **Tool approval** — `ContentView.swift`: `ToolApproval.Browse.{Allow,Deny}` (the per-fetch `browse` approval). The enclosing sheet is deliberately **unnamed**: an accessibility modifier on a container that is not its own accessibility element applies to the elements it contains, so naming the wrapper risks stamping it over the two buttons. Wait for `ToolApproval.Browse.Allow` to assert the prompt is up.

**No identifier (can't be AX-driven):** nothing on the current surface is known
to be unreachable. Both browse and MCP approval sheets carry identifiers. The
app has no sandbox approval prompt because it ships no filesystem or shell tools.
Reaching the MCP sheet still requires a configured connector that actually calls
a tool; its controls are addressable even when that external fixture is absent.

**Keeping this inventory from rotting.** The list above used to grow only when someone remembered to add an identifier. It is now defended by a CI gate: `scripts/check_rapid_mac_ax_identifiers.py` (job `accessibility-identifiers` in `.github/workflows/rapid-mac-ci.yml`) fails any PR that *adds* an interactive control under `apps/rapid-mac/Sources/` without `.accessibilityIdentifier(...)`. A control that genuinely cannot carry one opts out with a reasoned, greppable `// ax-exempt: <why>` marker on the control's line or the line above — no such control is known on the current surface (the `confirmationDialog`/`alert` doubt was measured and closed, see above), so `rg ax-exempt apps/rapid-mac` finding nothing is correct. The gate is scoped to added lines, so the *existing* gaps below are not blocked by it; `--audit` enumerates them. See `docs/gui-golden-flows.md` § "The identifier gate".

---

## Test harness limits

Three "verified" levels, kept distinct so a reviewer doesn't conflate them:

| Level                            | Meaning                                                                                                    |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| **Wire / persistence verified**  | Data structures, file formats, HTTP/SSE shapes round-trip under the unit + integration suite. UI-independent. |
| **Structural launch verified**   | `scripts/release-smoke.sh` confirms the scene renders, the main window exists in CGWindowList, the app stays alive 15 s, no crash report. |
| **End-to-end UX verified**       | A real user — or an Accessibility-permitted `walkthrough.sh` AXIdentifier pass on a local login session — drove the flow with mouse + keyboard. |

`walkthrough.sh` raised the ceiling from level 2 by clicking via AXIdentifier instead of coordinates, but full level-3 coverage still depends on the AX bridge (3-A) exposing each control and on running from a **local login session, not SSH+tmux** (3-B). The sidecar HTTP API (bearer from the sidecar env) verifies the chat wire end-to-end independent of the GUI send path.

---

## Open follow-up items

- **3-A residual — SwiftUI `Window` scene AX bridge still returns the `AXApplication` instead of the `AppKitWindow`.** `setActivationPolicy(.regular)` + `AXEnhancedUserInterface = true` lands the floor; the deeper bridge (real `AXWindow` with navigable `AXTextArea` / `AXButton` children) still needs custom `NSAccessibilityElement` wrappers. P2 — affects VoiceOver, not sighted keyboard+mouse users. **This gates whether `walkthrough.sh`'s `first button … whose AXIdentifier is …` lookups actually resolve** — verify on a fresh v0.10.6 run before trusting the AXIdentifier walk (an older note recorded these lookups returning empty; confirm current state).
- **3-B residual — no harness-driven UI test from SSH.** Run the walk on a local login GUI session; cliclick for focus, AXIdentifier `click` for buttons, real keystrokes for submit. cliclick coordinate clicks do NOT fire SwiftUI handlers (re-confirmed v0.10.6).
- ~~**Approval dialogs lack identifiers.**~~ Closed. The shipped browse and MCP
  approval sheets carry the identifiers listed above. The retired contract's
  sandbox prompt never existed in this app because filesystem and shell tools do
  not ship here.
- **Manual-only checks.** Physical global-hotkey delivery and local-login keyboard
  traversal cannot be proven by the unattended AX lane. Keep their dated manual
  evidence separate from CI GoldenFlow results.
- **Compose editor doc drift fixed.** `ComposeTextEditor` is the `NSViewRepresentable`; the `NSTextView` subclass it hosts is `AutosizingTextView`. Keep this straight in future edits.
