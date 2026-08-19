# User Flows — Rapid for macOS

This document maps every user-facing flow in the Rapid-MLX macOS app to the
source that implements it. It was **regenerated from the current
`apps/rapid-mac/Sources/` tree**, not patched — an earlier revision had drifted
far enough from the code to describe features that do not exist (a "Quick Ask"
global hotkey, a "pop conversation into its own window" surface, a 4-page
onboarding tour) and to cite files that were deleted or renamed. The rule for
this file going forward:

> **Every flow, file path, type name, accessibility identifier, setting, and
> provider named here is verified against current source. Each flow lists the
> real files it `Touches:`. When something cannot be verified in
> `Sources/`, it is left out rather than guessed.**

No verification dates or version numbers appear below: they cannot be
regenerated from source, and stamping them was exactly how the previous file
came to lie. Treat this as a structural map, not a smoke-test log.

Every entry has the same shape:

- **Trigger** — what the user does
- **Expected** — what should happen, from source
- **Touches** — code surfaces involved

The scene graph is two SwiftUI `Window` scenes declared in
`RapidApp.swift:271` (`"main"`) and `:431` (`"settings"`); there is no
`Settings` scene and no extra `WindowGroup`. The app is menu-bar-resident:
`AppDelegate.applicationShouldTerminateAfterLastWindowClosed` returns `false`
(`RapidApp.swift:786`) and a single AppKit `NSStatusItem` tray is installed by
`MenuBarController` (`RapidApp.swift:762`) — there is intentionally **no**
SwiftUI `MenuBarExtra` (its glyph does not render on macOS 26; see the comment
at `RapidApp.swift:509`). The `"main"` window's detail pane switches between
Chat, Images, Audio, and Launch surfaces from the sidebar
(`SidebarView.swift:6` `enum SidebarSection`; routed in
`ContentView.swift:604` `detailArea`).

---

## Flow 1 — First launch & onboarding (Quickstart)

**Trigger.** First run with no chat model on disk.

**Expected.**

1. A first-run **Quickstart** wizard short-circuits the chat surface, owned by
   `QuickstartCoordinator` (constructed at `RapidApp.swift:245`) and rendered by
   `QuickstartView`. It has a welcome step (**Get started** /
   **Skip**), a model-choice step (a recommended starter plus trade-up cards and
   **Browse all models**), and a download/starting/ready progress step, with
   low-disk, low-memory, and failure branches.
2. The model-choice UI is built from `OnboardingComponents.swift` and
   `OnboardingModelSelection.swift`; the "Direction D" wide/compact layouts live
   in `OnboardingDirectionD.swift`.
3. Low-memory recovery: when the live-memory guard rejects the starter, a
   sub-1B fallback is offered and **Switch to low-memory** appears only when the
   fallback clears the danger line (`Quickstart.Memory.SwitchToLowMemory`).

**Touches.** `UI/QuickstartView.swift`, `UI/OnboardingComponents.swift`,
`UI/OnboardingModelSelection.swift`, `UI/OnboardingDirectionD.swift`,
`Server/QuickstartModel.swift`, `RapidApp.swift` (`quickstart`).

---

## Flow 2 — Telemetry consent (first run)

**Trigger.** First launch, once the shell is up, when no telemetry decision has
been recorded (`ContentView.swift:45`, `TelemetryConsent.needsDecision()`).

**Expected.** A consent sheet (`TelemetryConsentView`) offers **Share** /
**Don't share** (`TelemetryConsent.Share` / `TelemetryConsent.DontShare`); the
choice is persisted via `TelemetryConsent.record(enabled:)`
(`ContentView.swift:838`). The decision is later changeable in Settings →
Privacy (`Settings.Privacy.TelemetryToggle`).

**Touches.** `UI/TelemetryConsentView.swift`, `UI/ContentView.swift`
(`decideTelemetry`), `Telemetry/TelemetryConsent.swift`,
`Telemetry/TelemetryConfig.swift`.

---

## Flow 3 — Send a chat message

**Trigger.** With a model ready, type in the composer and submit.

**Expected.**

1. `ChatView` hosts the composer and the send/stop control
   (`ChatView.SendOrStopButton`) and the attachment button
   (`ChatView.AddAttachments`).
2. `ChatViewModel` appends the user message and streams a reply from the
   embedded rapid-mlx sidecar via `ChatStreamClient` over the loopback port
   (`ChatStreamClient.loopbackURL(port:)`, bearer-authed).
3. Reasoning-capable models expose their thinking under a per-message
   disclosure (`ChatView.Message.ReasoningDisclosure.<UUID>`), and tool calls
   render as chips (`ToolCallChip.*`).
4. Attachments: the composer accepts document attachments; the button stays
   visible on a text-only alias but rejects image paste/drop for models without
   vision (see `docs/gui-golden-flows.md` → `chat-document-attachment`).

**Touches.** `UI/ChatView.swift`, `Chat/ChatViewModel.swift`,
`Chat/ChatStreamClient.swift`, `Chat/ChatFileAttachment.swift`,
`Chat/ChatMessage.swift`.

---

## Flow 4 — Message actions

**Trigger.** Hover / focus a transcript message.

**Expected.** Per-message controls carry `ChatView.Message.<action>.<UUID>`
identifiers (`ChatView.swift:1275`, `actionIdentifier`): **Copy**, **Edit**
(with `EditField`, `SaveEdit`, `CancelEdit`), **Retry**, **Select text**
(opens the cross-paragraph `SelectTextSheet`), and **ReasoningDisclosure**.

**Touches.** `UI/ChatView.swift`, `UI/SelectTextSheet.swift`.

---

## Flow 5 — Model picker & two-step lifecycle (Download → Start → Stop)

**Trigger.** Use the model picker in the chat header, then the primary action.

**Expected.**

1. The picker (`ModelPickerBar`) enumerates aliases via `ModelCatalog`, with
   Quickstart / Recommended / HuggingFace entries
   (`ModelPickerBar.ModelMenu`, `ModelPickerBar.Alias.<alias>`, etc.).
2. The single primary button (`ModelPickerBar.PrimaryButton`) is **two-step**:
   its label is `"Download"` for an alias not yet on disk and `"Start"` once it
   is cached (`ModelPickerBar.swift:1487` `startButtonLabel =
   selectedAliasIsCached ? "Start" : "Download"`). When a model is running the
   same button becomes **Stop model**, which calls `server.stop()`
   (`ModelPickerBar.swift:1440`) and is disambiguated from the composer's "Stop
   response" (see the comment at `:1442`).
3. Readiness gating for image/audio/chat is centralised in `ModelReadiness`;
   a readiness banner offers the switch action (`Readiness.Action`,
   `Readiness.Band`).

**Touches.** `UI/ModelPickerBar.swift`, `Server/ServerManager.swift`,
`Server/ModelCatalog.swift`, `UI/ModelReadiness.swift`,
`UI/ReadinessBanner.swift`, `UI/ReadinessModelStart.swift`.

---

## Flow 6 — Download a model

**Trigger.** Choose **Download** on an uncached alias (picker or Model
Management), or a Quickstart download.

**Expected.** `DownloadManager` spawns a `rapid-mlx pull <alias>` job; progress
and cancel/dismiss surface in the download strip
(`DownloadStrip.Cancel.<alias>`, `DownloadStrip.Dismiss.<alias>`). The models
folder is honoured (see Flow 12). On success the alias becomes runnable.

**Touches.** `Server/DownloadManager.swift`, `UI/DownloadStrip.swift`,
`Server/ServerLocator.swift`, `Services/ModelCacheActions.swift`.

---

## Flow 7 — Conversation management: sidebar, folders, export

**Trigger.** Sidebar interactions on conversation rows.

**Expected.**

1. **New chat** (`Sidebar.NewChat`); each conversation row carries
   `Sidebar.Conversation.<UUID>` and a `···` menu
   (`Sidebar.Conversation.Menu.<UUID>`).
2. Row actions: **Rename** (`…Action.Rename`,
   `Sidebar.Conversation.Rename.<UUID>`), **Delete** (`…Action.Delete`,
   confirmed via `Sidebar.DeleteConversation.{Confirm,Keep}`), **Move to
   folder** (`…Action.MoveToFolder`, `…MoveToNewFolder`,
   `…MoveToFolder.Remove`), and **Export** as **Markdown** or **JSON**
   (`Sidebar.Conversation.Action.Export.{Markdown,JSON}`).
3. **Folders**: create/rename/delete (`Sidebar.Folder.*`,
   `Sidebar.DeleteFolder.{Confirm,Keep}`); archived section toggle
   (`Sidebar.Archived.Toggle`).
4. Persistence: conversations are stored in `conversations.json` via
   `ConversationStore` (`Chat/ConversationHistory.swift:92,106`); folders in a
   separate `folders.json` via `ConversationFolderStore`
   (`Chat/ConversationFolder.swift:97`). There is **no** `sessions.json` /
   `SessionStore` in this app.
5. **Export All Chats…** is a menu command (⇧⌘E) that writes the whole library
   through `ConversationExportPanel.exportAll` (`RapidApp.swift:399`).

**Touches.** `UI/SidebarView.swift`, `Chat/ConversationHistory.swift`,
`Chat/ConversationFolder.swift`, `Chat/ConversationExport.swift`,
`UI/ConversationExportPanel.swift`, `RapidApp.swift` (Export All command).

---

## Flow 8 — Search conversations

**Trigger.** The sidebar **Search chats** control (`Toolbar.SearchChats`,
`SidebarView.swift:231`).

**Expected.** A search panel (`ConversationSearchView`,
`ConversationSearch.Panel`) with a query field (`ConversationSearch.Field`),
results keyed per conversation (`ConversationSearch.Result.<UUID>`), a clear
control, an empty state, and a **New chat** shortcut; matching is done by
`ConversationSearch`.

**Touches.** `UI/ConversationSearchView.swift`, `Chat/ConversationSearch.swift`.

---

## Flow 9 — Settings (eight categories)

**Trigger.** `Cmd+,` or the in-window settings button (`ContentView.Settings`).

**Expected.** The Settings window renders a category rail
(`Settings.Category.<rawValue>`) with exactly **eight** release categories, in
declaration order (`SettingsView.swift:59` `enum Category`): **Model
Management, Instructions, Tools, Connectors, Performance, Appearance, Privacy,
App**. A ninth **Developer** category exists only under `#if DEBUG`
(`SettingsView.swift:90`) and is compiled out of release builds. Deep-linking a
category is done through `SettingsRouter` (`RapidApp.swift:324`).

**Touches.** `UI/SettingsView.swift`, `UI/SettingsRouter.swift`,
`RapidApp.swift` (`"settings"` scene).

---

## Flow 10 — Custom instructions (global + per-conversation)

**Trigger.** Settings → Instructions, or the compose-row conversation-
instructions control (`ChatView.ConversationInstructions`).

**Expected.**

1. **Global** instructions are applied to every chat, stored in UserDefaults by
   `CustomInstructionsConfig` (`Chat/CustomInstructionsConfig.swift:9`); the
   Settings editor exposes `Settings.Instructions.Clear`.
2. **Per-conversation** instructions travel with history and are edited via
   `InstructionTextEditor`
   (`ChatView.ConversationInstructions.{Save,Clear,Cancel}`).

**Touches.** `Chat/CustomInstructionsConfig.swift`,
`UI/InstructionTextEditor.swift`, `UI/ChatView.swift`.

---

## Flow 11 — Per-model performance settings

**Trigger.** Settings → Performance.

**Expected.** A per-model panel (`Settings.Performance.Panel`) with a model
picker (`Settings.Performance.ModelPicker`; `Settings.Performance.NoModel` when
none), KV-cache mode (`…KVMode`), prefix cache (`…PrefixCache`), cache budget
(`…CacheBudget`, `…CacheBudgetAutomatic`), speculative decoding
(`…SpeculativeDecoding.Enabled`), reset (`…Reset`), and a "restart / applies
next load" notice (`…RestartNotice`, `…AppliesNextLoad`). Overrides are stored
by `ModelPerfConfigStore` and merged into launch flags at spawn
(`ModelPerfConfigStore.swift:107`; wired at `RapidApp.swift:211`).

**Touches.** `UI/SettingsPerformancePanel.swift`,
`Server/ModelPerfConfigStore.swift`, `Server/ModelPerfConfig.swift`,
`Server/ServerManager.swift`.

---

## Flow 12 — Model Management (library) & models folder

**Trigger.** Settings → Model Management.

**Expected.** A cache inspector: search/filter/sort
(`Settings.ModelManagement.{Search,Filter,SortMenu,ClearSearch}`), per-alias
rows (`…Row.<alias>`) with **Download / Cancel / Delete / Retry**
(`…{Download,Cancel,Delete,Retry}.<alias>`) and favourite pin
(`…Favorite.<alias>`); a capability-tab split (`…CapabilityTabs`); a
**Recommended for your Mac** section (`…RecommendedHeader`,
`…Recommended.{Download,Cancel,Delete,Retry}.<alias>`) with a meter legend
(`…MeterLegend`); a storage summary/footer (`…StorageSummary`, `…Footer`,
`…LargestModel`); auto-start and show-all toggles
(`Settings.Models.{AutoStartOnLaunchToggle,ShowAllModelsToggle}`); and a
custom-models-folder control (`…ChooseFolder`, `…UseDefaultFolder`,
`…FolderPath`, `…FolderUnavailable`) backed by `ModelsFolderPreference`.

**Touches.** `UI/SettingsModelManagementPanel.swift`,
`Server/ModelCatalog.swift`, `Services/ModelCacheActions.swift`,
`Server/ModelsFolderPreference.swift`, `Server/DownloadManager.swift`.

---

## Flow 13 — Built-in tools & browse approval

**Trigger.** Settings → Tools, then let a tool-capable model call a tool.

**Expected.**

1. Exactly **three** built-in tools are dispatched by
   `BuiltinToolRegistry` (`BuiltinToolRegistry.swift:46`): `web_search`,
   `browse`, and `weather` (an unknown name is refused with the list). Each is
   individually toggleable (`Settings.Tools.Toggle.<tool>`) with a details
   disclosure (`Settings.Tools.Details.<name>`,
   `Settings.Tools.DetailsBody.<name>`).
2. `weather` (Open-Meteo) needs no approval; `web_search` runs against the
   configured backend (Flow 14).
3. `browse` is SSRF-guarded (`BrowseSSRFGuard`) and gated by a per-fetch
   approval sheet with **three** buttons — **Deny / Always allow / Allow once**
   (`ToolApproval.Browse.{Deny,AlwaysAllow,Allow}`, `ContentView.swift:1273`).
   A blanket auto-approve lives at
   `Settings.Tools.Browse.AutoApproveToggle`.
4. The app ships no filesystem or shell tools.

**Touches.** `UI/SettingsToolsPanel.swift`, `Tools/BuiltinToolRegistry.swift`,
`Tools/WebSearchTool.swift`, `Tools/BrowseTool.swift`,
`Tools/BrowseApprovalStore.swift`, `Tools/BrowseSSRFGuard.swift`,
`Tools/WeatherTool.swift`, `UI/ContentView.swift` (browse approval sheet).

---

## Flow 14 — Web-search provider configuration

**Trigger.** Settings → Tools → web-search backend
(`Settings.Tools.WebSearch.Backend`).

**Expected.** There are **five** providers, in the order the radio renders
them (`WebSearchProvider.swift:63` `enum WebSearchProvider`): **Keenable**
(the zero-setup, keyless default — `WebSearchConfig` defaults to `.keenable`,
`WebSearchProvider.swift:211`), **Parallel**, **Tavily**, **Brave Search**, and
**DuckDuckGo** (keyless backstop). Keys, when a provider takes one, are stored
in the macOS Keychain (per-provider `keychainAccount`), never UserDefaults;
pasting a key for a key-requiring provider while on a keyless backend
auto-promotes the selection (`setAPIKey`, `WebSearchProvider.swift:272`).

**Touches.** `Tools/WebSearchProvider.swift`, `UI/SettingsToolsPanel.swift`,
`Tools/KeychainStore.swift`, `Tools/WebSearchClients.swift`.

---

## Flow 15 — Connectors (MCP)

**Trigger.** Settings → Connectors.

**Expected.**

1. A master **Enable connectors** toggle (`Settings.Connectors.MasterToggle`,
   opt-in). The flag is read once at spawn, so toggling it raises a **Restart
   to apply** banner (`…RestartButton`).
2. **Add / edit** a server through `MCPServerEditorSheet` (name, transport,
   command/URL, env, enable — `Settings.Connectors.Editor.*`). Rows expose
   status, toggle, and an edit/remove menu (`…Row.{Status,Toggle,Menu,Edit,
   Remove}.<name>`).
3. Connected tools have per-tool switches
   (`Settings.Connectors.Tool.Toggle.<name>`); a blanket auto-approve
   (`…AutoApproveToggle`) and **Reset approvals** (`…ResetApprovals`) live in
   the approvals card.
4. At inference the first call to a connector tool raises an approval sheet
   with **three** buttons (`ToolApproval.MCP.{Deny,AlwaysAllow,Allow}`,
   `ContentView.swift:1376`). Editing or removing a server revokes its
   remembered grant (`MCPToolApprovalStore.revokeGrants`, wired at
   `RapidApp.swift:187`).

**Touches.** `UI/SettingsConnectorsPanel.swift`, `UI/MCPServerEditorSheet.swift`,
`UI/ContentView.swift` (MCP approval sheet), `MCP/MCPConfigStore.swift`,
`MCP/MCPCatalog.swift`, `MCP/MCPToolApprovalStore.swift`,
`MCP/MCPToolRegistry.swift`, `MCP/CompositeToolRegistry.swift`,
`Server/ServerManager.swift` (`mcpConfigPathProvider`).

---

## Flow 16 — Images tab (text→image and image edit)

**Trigger.** Sidebar → Images (`Sidebar.Images`).

**Expected.** A dedicated `ImagesView`: empty-state hero
(`Images.EmptyState`), a model picker filtered to image aliases
(`Images.ModelPicker`, `Images.Model.<alias>`), aspect/resolution controls
(`Images.Aspect.<ar>`, `Images.Resolution.<res>`), a readiness switch when the
sidecar is on another model (`Readiness.Action`), prompt + generate
(`Images.Prompt`, `Images.Generate`), an in-flight progress card with cancel
(`Images.Cancel`), a result stage and gallery filmstrip (`Images.Stage`,
`Images.Gallery`, `Images.Gallery.Thumb.<n>`), and save/edit result actions
(`Images.Result.{Save,Edit}`). Edit mode imports a source
(`Images.Edit.Import`, `Images.Edit.Source`) and exits back to generation
(`Images.Edit.Exit`). rapid-mlx serves one model per process, so selecting an
image alias reloads the sidecar.

**Touches.** `UI/ImagesView.swift`, `Images/ImageGenViewModel.swift`,
`Images/ImageClient.swift`.

---

## Flow 17 — Audio tab (speech & transcription)

**Trigger.** Sidebar → Audio (`Sidebar.Audio`).

**Expected.** `AudioView` with a mode switch (`Audio.Mode`) and an empty state
that can jump to Model Management (`Audio.EmptyState`,
`Audio.EmptyState.OpenModelManagement`). **Speech (TTS)**: text
(`Audio.Speech.Text`), voice picker + preview (`Audio.Speech.VoicePicker`,
`Audio.Speech.PreviewVoice.<voice>`), speed (`Audio.Speech.Speed`), generate /
play / save (`Audio.Speech.{Generate,Play,Save,LoadVoices}`).
**Transcription (STT)**: file picker, run, result, copy, save
(`Audio.Transcription.{FilePicker,Run,Result,Copy,Save}`).

**Touches.** `UI/AudioView.swift`, `Audio/AudioViewModel.swift`,
`Audio/AudioClient.swift`.

---

## Flow 18 — Dictation (background global hotkey)

**Trigger.** A press-and-hold global hotkey, armed at launch
(`RapidApp.swift:302`, `dictation.bootstrap()`) so it works with Rapid's window
closed.

**Expected.** `DictationController` records, transcribes against the sidecar,
and injects text at the cursor. It is built on a `CGEventTap`
(`DictationHotkey.swift:4`), needing Microphone and Accessibility grants
(`Dictation.GrantMicrophone`, `Dictation.GrantAccessibility`). The Dictation
settings surface (`DictationView`) exposes enable/arm/hotkey/model
(`Dictation.{Enable,Arm,Hotkey,Model}`), a vocabulary manager
(`Dictation.{NewTerm,AddTerm,RemoveTerm.<term>,Suggestion.<name>}`), a
transcript-fix sheet (`Dictation.Fix.*`), and history controls
(`Dictation.{CopyTranscript,ArchiveAudio,ClearHistory}`).

**Touches.** `Dictation/DictationController.swift`,
`Dictation/DictationHotkey.swift`, `Dictation/DictationRecorder.swift`,
`Dictation/DictationInjector.swift`, `Dictation/DictationVocabulary.swift`,
`Dictation/DictationHistory.swift`, `UI/DictationView.swift`.

---

## Flow 19 — Launch / integrations page

**Trigger.** Sidebar → Launch (`Sidebar.Launch`).

**Expected.** `LaunchView` (`SidebarView.swift:1365`) renders
`ConnectToolsView`, which lists integration targets from `IntegrationCatalog`
and offers copy/reveal of connection details
(`Launch.Integration.Copy.<tool.id>`, `ConnectTools.{Copy,Reveal}.<label>`,
`ConnectTools.Close`) — i.e. connecting external clients to the local server.

**Touches.** `UI/SidebarView.swift` (`LaunchView`), `UI/ConnectToolsView.swift`,
`Server/IntegrationCatalog.swift`.

---

## Flow 20 — App updates (Sparkle)

**Trigger.** Background check at launch, or Settings → App.

**Expected.**

1. Updates are **Sparkle-owned**: `SparkleUpdateController` runs the check,
   background download, EdDSA verification, and install-on-quit
   (`RapidApp.swift:334`, gated on `sparkleUpdater.isEnabled`). There is **no**
   in-app installer type in `Updater/` — the directory is `InstallTracker`,
   `SparkleUpdateController`, and `UpdateChecker`.
2. `UpdateChecker` is read-only: it GETs a static manifest and drives the
   version pill and the Settings → App status
   (`Settings.App.{Checking,RecheckCTA,UpdateCTA,AutomaticUpdatesToggle}`); it
   installs nothing (`RapidApp.swift:64`).
3. `InstallTracker` detects the "Finder Replace into /Applications silently
   failed because Rapid was still running" footgun; `FailedReplaceBanner`
   surfaces recovery (`FailedReplace.{OpenUpdate,Dismiss}`).
4. Manifest URLs are host-allowlisted (`updateReleaseHostAllowlist`,
   `updateDownloadHostAllowlist`, `RapidApp.swift:16,33`).

**Touches.** `Updater/SparkleUpdateController.swift`,
`Updater/UpdateChecker.swift`, `Updater/InstallTracker.swift`,
`UI/FailedReplaceBanner.swift`, `UI/SettingsView.swift` (App category),
`RapidApp.swift`.

---

## Flow 21 — Dock-visibility on close & menu-bar residency

**Trigger.** Close the main window; or use the tray.

**Expected.** The first native close reaches a Dock-visibility prompt
(`MainWindowCloseInterceptor`, installed at `RapidApp.swift:557`) offering keep
/ hide-on-close plus "don't ask again" (`DockVisibilityPrompt`,
`DockVisibilityPromptStore`). A persisted "hide always" choice boots the app in
`.accessory` policy (`RapidApp.swift:695`). The app stays alive behind the
single `MenuBarController` tray, which exposes open / new chat / status /
settings / quit; a Dock click re-opens the main window
(`applicationShouldHandleReopen`, `RapidApp.swift:819`).

**Touches.** `UI/MainWindowCloseInterceptor.swift`,
`UI/DockVisibilityPrompt.swift`, `UI/DockVisibilityPromptStore.swift`,
`UI/MenuBarController.swift`, `RapidApp.swift` (`AppDelegate`).

---

## Flow 22 — Clean quit & termination

**Trigger.** `Cmd+Q`, or **Quit** from the tray.

**Expected.** `applicationWillTerminate` runs `runStandardTermination`
(`RapidApp.swift:938`): stop the in-flight stream, signal then reap the server
process group and the download children (overlapping grace windows), then flush
`ConversationStore` and `ConversationFolderStore`. `CrashReporter` clears the
launch marker so the next launch does not misclassify the exit. A watchdog'd
sidecar dies with the app; a user's own headless `rapid-mlx serve` is untouched.

**Touches.** `RapidApp.swift` (`runTerminationSequence`,
`runStandardTermination`), `Server/ServerManager.swift`,
`Server/DownloadManager.swift`, `Chat/ConversationHistory.swift`,
`Chat/ConversationFolder.swift`, `Telemetry/CrashReporter.swift`.

---

## Flow 23 — Appearance / theme

**Trigger.** Settings → Appearance (`Settings.Appearance.ThemePicker`).

**Expected.** A theme override persisted by `AppearanceConfig` and re-applied
after `NSApp` boots (`RapidApp.swift:718`) so a saved "Light" choice survives a
dark host.

**Touches.** `UI/SettingsView.swift` (Appearance), `UI/AppearanceConfig.swift`.

---

## Flow 24 — Missing-runtime overlay

**Trigger.** The embedded rapid-mlx binary cannot be resolved
(`ServerState.missing`, routed by `ContentView.mainAreaBranch`,
`ContentView.swift:1195`).

**Expected.** An install/recovery overlay replaces the chat surface with
recheck / download-update / quit actions
(`MissingRuntime.{Recheck,RecheckStatus,DownloadUpdate,Quit}`).

**Touches.** `UI/ContentView.swift` (missing-runtime branch).

---

## Flow 25 — Developer panel (DEBUG only)

**Trigger.** Settings → Developer — **present only in `#if DEBUG` builds**
(`SettingsView.swift:90`).

**Expected.** State-erasing rehearsal actions (`Settings.Developer.Panel`,
`…Reonboard`, `…{Confirm,Cancel}Reonboard`) with scoped erase toggles
(`Settings.Developer.Scope.{Conversations,Preferences,Telemetry}`), driven by
`ReonboardingReset` (`ReonboardingScope`, `Services/ReonboardingReset.swift`).

**Touches.** `UI/SettingsDeveloperPanel.swift`, `Services/ReonboardingReset.swift`.

---

## Removed in this regeneration (fictional / stale)

These appeared in the prior file and are **not present in `Sources/`**; the
greps that prove their absence are recorded so a future edit doesn't reintroduce
them:

- **"Quick Ask" global hotkey / `QuickAskView` / `GlobalHotkey`.** No such type
  or directory. `grep -rn "struct QuickAskView\|GlobalHotkey" Sources` → none
  (the string "QuickAsk" survives only in unrelated doc-comments).
- **"Pop a conversation into its own window" / `PoppedConversationView.swift`.**
  Absent (`find … -iname "*popped*"` → none). There is one `"main"` window and
  one `"settings"` window; no per-conversation `WindowGroup`.
- **`OnboardingTour` 4-page sheet.** No `OnboardingTour.swift`; onboarding is
  the Quickstart wizard (Flow 1).
- **`sessions.json` / `SessionStore` / `SessionsSidebar.swift` /
  `DeleteSessionConfirmation.swift` / `MessageSearch.swift` /
  `SessionLoadFailureBanner.swift` / `StaleSessionAliasBanner.swift`.** None
  exist. The store is `conversations.json` via `ConversationStore`
  (`Chat/ConversationHistory.swift`).
- **`Updater/Installer.swift` / `UpdateInstallView.swift` / in-app installer.**
  Absent; updates are Sparkle-only (Flow 20).
- **Three-provider web search (DDG/Brave/Tavily only).** There are five, with a
  keyless Keenable default (Flow 14).
- **Six Settings categories.** There are eight in release (Flow 9).
- **Dead AX ids** `Settings.QuickAsk.LaunchAtLogin`, `DeleteSessionConfirm.*`,
  and the two-button `ToolApproval.Browse.{Allow,Deny}` — the browse and MCP
  approval sheets each carry **three** buttons now.

---

## Accessibility identifier inventory

This is the authoritative AX inventory, generated by scanning
`.accessibilityIdentifier(...)` string literals in `apps/rapid-mac/Sources/`.
Identifiers built from interpolation keep their placeholder (e.g. `.<alias>`,
`.<UUID>`, `.<name>`). Grouped by the file that declares them. When adding or
removing a control, update this list; the `accessibility-identifiers` CI job
(`scripts/check_rapid_mac_ax_identifiers.py`) fails any PR that adds an
interactive control under `Sources/` without an identifier.

**Chat — `UI/ChatView.swift`:** `ChatView.SendOrStopButton`,
`ChatView.AddAttachments`, `ChatView.ConversationInstructions`,
`ToolCallChip.<action>`, `ToolCallChip.Toggle.<call.id>`,
`ChatView.Message.<action>.<UUID>` (action ∈ {`EditField`, `CancelEdit`,
`SaveEdit`, `Edit`, `Copy`, `SelectText`, `Retry`, `ReasoningDisclosure`}).
`UI/SelectTextSheet.swift`: `SelectText.Done`. `UI/JumpToBottomButton.swift`:
`Transcript.JumpToBottom`. `UI/InstructionTextEditor.swift`:
`ChatView.ConversationInstructions.{Save,Clear,Cancel}`,
`Settings.Instructions.Clear`, `<id>.Count`.

**ContentView — `UI/ContentView.swift`:** `ContentView.Settings`,
`ContentView.LogDrawer`, `ContentView.ToggleLogs`, `Footer.DesktopVersionPill`,
`MemoryWarning.{Confirm,Cancel}`,
`MissingRuntime.{Recheck,RecheckStatus,DownloadUpdate,Quit}`,
`ToolApproval.Browse.{Allow,AlwaysAllow,Deny}`,
`ToolApproval.MCP.{Allow,AlwaysAllow,Deny}`.

**Sidebar — `UI/SidebarView.swift`:** `Sidebar.NewChat`, `Sidebar.Images`,
`Sidebar.Audio`, `Sidebar.Launch`, `Sidebar.Residency`,
`Sidebar.ResidentModel.<id>`, `Sidebar.Archived.Toggle`, `Toolbar.SearchChats`,
`Sidebar.Conversation.<UUID>`, `Sidebar.Conversation.Menu.<UUID>`,
`Sidebar.Conversation.Rename.<UUID>`,
`Sidebar.Conversation.Action.{Rename,Delete,MoveToFolder,MoveToNewFolder,
MoveToFolder.Remove,Export.JSON,Export.Markdown}`,
`Sidebar.DeleteConversation.{Confirm,Keep}`,
`Sidebar.Folder.{Action.Rename,Action.Delete,NameField,Toggle.<slug>,
Prompt.Confirm,Prompt.Cancel}`, `Sidebar.DeleteFolder.{Confirm,Keep}`.

**Conversation search — `UI/ConversationSearchView.swift`:**
`ConversationSearch.{Panel,Field,Clear,Close,Empty,NewChat,Result.<UUID>}`.

**Model picker — `UI/ModelPickerBar.swift`:** `ModelPickerBar.PrimaryButton`,
`ModelPickerBar.ModelMenu`, `ModelPickerBar.ModelInfo`,
`ModelPickerBar.RefreshCatalog`, `ModelPickerBar.Alias.<alias>`,
`ModelPickerBar.Quickstart.<alias>`, `ModelPickerBar.Recommended.<alias>`,
`ModelPickerBar.HuggingFace.<repo>`,
`ModelPickerBar.Context.{Download,Delete}.<alias>`,
`ModelPickerBar.CustomAlias.{Open,Field,Use,Cancel}`,
`ModelPickerBar.Delete.{Confirm,Cancel}`. Readiness:
`UI/ReadinessBanner.swift` `Readiness.Action`; `UI/Components/LifecycleBand.swift`
`Readiness.Band`. Downloads: `UI/DownloadStrip.swift`
`DownloadStrip.{Cancel,Dismiss}.<alias>`.

**Quickstart / onboarding — `UI/QuickstartView.swift`:**
`Quickstart.{GetStarted,Skip,BrowseAll,Ready.StartChatting,ResumeNotice,
Review.Footnote,Step2.Kicker}`,
`Quickstart.BrowseAll.{Search,Filter,SortMenu,Sort.<order>,List,Count}`,
`Quickstart.CachedModel.<alias>`, `Quickstart.YourPick.<alias>`,
`Quickstart.Download.Cancel`, `Quickstart.Failure.BackToModelSelection`,
`Quickstart.LowDisk.{Continue,Cancel}`,
`Quickstart.Memory.{SwitchToLowMemory,LoadAnyway,Cancel}`.
`UI/OnboardingComponents.swift`: `Quickstart.Choice.<alias>`,
`Quickstart.CatalogRow.<alias>`. `UI/OnboardingDirectionD.swift`:
`Quickstart.{Footer.Back,Footer.Primary,Progress,Compare,SelectionSummary,
Rail.ThisMac,Step2.Kicker,Subject.{Bytes,Percent,Rail,Rate}}`.

**Settings rail / App / Privacy / Appearance — `UI/SettingsView.swift`:**
`Settings.Category.<rawValue>`,
`Settings.App.{AutomaticUpdatesToggle,Checking,RecheckCTA,UpdateCTA,
ExportDiagnostics,HideDockOnCloseToggle,ResetDockOnboardingCTA}`,
`Settings.Appearance.ThemePicker`, `Settings.Privacy.TelemetryToggle`,
`Settings.Privacy.Link.{PrivacyPolicy,License,Credits,MTPLX}`.

**Model Management — `UI/SettingsModelManagementPanel.swift`:**
`Settings.ModelManagement.{Search,ClearSearch,Filter,SortMenu,Sort.<order>,
Row.<alias>,Status.<text>,Download.<alias>,Cancel.<alias>,Delete.<alias>,
Retry.<alias>,ConfirmDelete,Favorite.<alias>,KeepOnDisk,CapabilityTabs,
RecommendedHeader,Recommended.<primary?>,Recommended.Download.<alias>,
Recommended.Cancel.<alias>,Recommended.Delete.<alias>,Recommended.Retry.<alias>,
MeterLegend,StorageSummary,LargestModel,Footer,VisibleCount,ChooseFolder,
UseDefaultFolder,FolderPath,FolderUnavailable}`,
`Settings.Models.{AutoStartOnLaunchToggle,ShowAllModelsToggle}`.

**Performance — `UI/SettingsPerformancePanel.swift`:**
`Settings.Performance.{Panel,ModelPicker,NoModel,KVMode,PrefixCache,CacheBudget,
CacheBudgetAutomatic,SpeculativeDecoding.Enabled,Reset,RestartNotice,
AppliesNextLoad}`.

**Tools — `UI/SettingsToolsPanel.swift`:**
`Settings.Tools.Toggle.<function.name>`, `Settings.Tools.Details.<name>`,
`Settings.Tools.DetailsBody.<name>`, `Settings.Tools.WebSearch.Backend`,
`Settings.Tools.Browse.AutoApproveToggle`.

**Connectors — `UI/SettingsConnectorsPanel.swift`:**
`Settings.Connectors.{MasterToggle,AddButton,AutoApproveToggle,ResetApprovals,
RestartButton,SubsystemError,ConfirmRemove,CancelRemove,
Row.Status.<name>,Row.Toggle.<name>,Row.Menu.<name>,Row.Edit.<name>,
Row.Remove.<name>,Tool.Toggle.<name>}`. `UI/MCPServerEditorSheet.swift`:
`Settings.Connectors.Editor.{Name,Transport,Command,URL,Enabled,Allow,Cancel}`.

**Developer (DEBUG) — `UI/SettingsDeveloperPanel.swift`:**
`Settings.Developer.{Panel,Reonboard,ConfirmReonboard,CancelReonboard,
Scope.Conversations,Scope.Preferences,Scope.Telemetry}`.

**Images — `UI/ImagesView.swift`:**
`Images.{EmptyState,ModelPicker,Model.<alias>,Aspect.<ar>,Resolution,
Resolution.<res>,Prompt,Generate,Cancel,Stage,Gallery,Gallery.Thumb.<n>,
Starter.<index>,Result.Save,Result.Edit,Edit.Import,Edit.Source,Edit.Exit}`.

**Audio — `UI/AudioView.swift`:**
`Audio.{Mode,EmptyState,EmptyState.OpenModelManagement,
Speech.{Text,VoicePicker,VoiceOption.<voice>,PreviewVoice.<voice>,Speed,
Generate,Play,Save,LoadVoices},
Transcription.{FilePicker,Run,Result,Copy,Save}}`.

**Dictation — `UI/DictationView.swift`:**
`Dictation.{Enable,Arm,Hotkey,Model,NewTerm,AddTerm,RemoveTerm.<term>,
Suggestion.<name>,Fix,Fix.Heard,Fix.Correction,Fix.Apply,Fix.Cancel,
CopyTranscript,ArchiveAudio,ClearHistory,GrantMicrophone,GrantAccessibility,
Error}`.

**Launch / Connect — `UI/ConnectToolsView.swift`:**
`Launch.Integration.Copy.<tool.id>`, `ConnectTools.{Copy.<label>,Reveal.<label>,
Close}`.

**Telemetry consent — `UI/TelemetryConsentView.swift`:**
`TelemetryConsent.{Share,DontShare}`.

**Banners / misc:** `UI/FailedReplaceBanner.swift`
`FailedReplaceBanner`, `FailedReplace.{OpenUpdate,Dismiss}`;
`UI/Components/QuietIconButton.swift` `Sheet.Close`; `AboutPanel.swift`
`About.Link.<label>`.

**Debug-only design specimens — `DevSnapshot.swift`:**
`DevSnapshot.Specimen.*` (button/toggle/icon style specimens; not a user
surface).

---

## Notes on driving the UI

The deterministic GUI journeys live in `scripts/gui-golden-flows.sh` and drive
the app by `AXIdentifier` (see `docs/gui-golden-flows.md`). The dictation
global-hotkey path (`Dictation/DictationHotkey.swift`) and the dictation/audio
capture paths depend on real Microphone/Accessibility grants and a local GUI
login session, so they are exercised by hand rather than in the unattended
lane. (Audio is sidebar-driven and has no global hotkey — only dictation
registers one.)
