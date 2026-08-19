import AppKit
import SwiftUI

/// Hosts that ``release.htmlURL`` is allowed to point at when the
/// "Update available" menu item is clicked. Anything else is silently
/// ignored — a compromised update manifest can't turn the CTA into a
/// phishing redirect (codex round 3). Lower-cased ASCII hostnames only.
///
/// `rapidmlx.com` and `www.rapidmlx.com` were added when the updater
/// stopped proxying GitHub Releases and started reading a static
/// manifest on `dl.rapidmlx.com` (see `UpdateChecker.swift`). The
/// manifest's `html_url` now points at the public landing page
/// (`https://rapidmlx.com/desktop`), not at GitHub — the repo is
/// private and most users can't open the release page on github.com
/// anyway.
let updateReleaseHostAllowlist: Set<String> = [
    "github.com",
    "www.github.com",
    "rapidmlx.com",
    "www.rapidmlx.com",
]

/// Hosts a manifest's `dmg_url` is allowed to name. Sparkle owns the
/// actual download now (its own appcast, its own EdDSA verification), so
/// nothing in-process fetches this URL any more — the check survives as a
/// manifest-shape gate: a payload naming an unexpected download host is
/// treated as a malformed manifest rather than quietly accepted into the
/// version status the UI renders.
///
/// The ``UpdateCheckerTests`` superset invariant requires every host in
/// ``updateReleaseHostAllowlist`` to appear here too; keep the two in sync.
/// Lower-cased ASCII hostnames only.
let updateDownloadHostAllowlist: Set<String> = [
    "github.com",
    "www.github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "dl.rapidmlx.com",
    "rapidmlx.com",
    "www.rapidmlx.com",
]

/// The main window's size on a launch that has no saved frame.
///
/// Named rather than inline because the width is not a taste choice: Direction
/// D's onboarding enters its two-column layout at
/// ``OnboardingD/columnsMinWidth`` (1290pt), and the previous default of
/// 1200×820 sat below it — so every fresh install met Step 2 in the medium
/// STACKED layout and no user ever saw the composition Paper 05.1.A specifies
/// until they resized the window themselves. 1440×900 is Paper's own desktop
/// frame, and it is the smallest round size that clears the breakpoint.
///
/// This is a DEFAULT, not a policy. It applies only where SwiftUI has no frame
/// to restore: ``AppDelegate/attachMainWindow(_:)`` sets the
/// `Rapid.MainWindow` autosave name, which restores a saved frame synchronously
/// and therefore wins for every returning user. Nothing resizes the window when
/// onboarding appears or changes state, and narrowing it still reaches the
/// medium and compact tiers exactly as before.
enum MainWindowDefaults {
    static let width: CGFloat = 1440
    static let height: CGFloat = 900
}

struct RapidApp: App {
    @Environment(\.openWindow) private var openWindow

    /// The single source of truth for the embedded rapid-mlx child. We
    /// build it once at app launch so all windows / scenes share state.
    @State private var server: ServerManager
    /// Per-window-but-shared chat controller — single window for now, so
    /// keeping a process-wide instance is fine.
    @State private var chatViewModel: ChatViewModel
    /// Images-tab controller — text→image / image-edit against an image-gen
    /// alias. It keeps its own UI state while sharing the resident-model
    /// sidecar with Chat.
    @State private var imageGen: ImageGenViewModel
    /// Local speech-to-text and text-to-speech workflows. Like Images, this
    /// shares the one embedded server and swaps models only on explicit work.
    @State private var audio: AudioViewModel
    /// Owned by the app, not the Audio tab: dictation must keep working
    /// while Rapid's own window is closed.
    @State private var dictation: DictationController
    /// Self-update poller. GETs a public static manifest on R2 at
    /// `https://dl.rapidmlx.com/latest.json`. See ``UpdateChecker``.
    @State private var updater: UpdateChecker
    /// Sparkle owns the whole update pipeline — check, download, EdDSA
    /// verification, install-on-quit — for release builds carrying an injected
    /// public key. ``UpdateChecker`` above no longer installs anything; it is
    /// the read-only version/status source the pill and Settings render.
    @State private var sparkleUpdater: SparkleUpdateController
    /// Persisted sampling knobs exposed via Settings → Sampling.
    @State private var sampling: SamplingConfig
    /// App-wide custom instructions shared by Settings and every chat turn.
    @State private var customInstructions: CustomInstructionsConfig
    /// Persisted theme override exposed via Settings → Appearance.
    @State private var appearance: AppearanceConfig
    /// Deep-link channel into the Settings window.
    @State private var settingsRouter: SettingsRouter
    /// Side-car downloader — spawns ``rapid-mlx pull <alias>`` jobs.
    @State private var downloads: DownloadManager
    /// Detects the "Finder Replace into /Applications silently failed
    /// because Rapid-MLX was still running" footgun (issue #251).
    @State private var installTracker: InstallTracker
    /// First-launch Quickstart owner (Flow A).
    @State private var quickstart: QuickstartCoordinator
    /// Opt-in "hide Dock icon, keep running in the background" prompt.
    @State private var dockPromptStore: DockVisibilityPromptStore
    /// View → Keep Window on Top toggle (session state, not persisted).
    @State private var keepWindowOnTop: Bool = false
    /// Web-search backend + API key, shared by the tool runner and Settings.
    @State private var webSearch: WebSearchConfig
    /// Per-fetch approval gate for the ``browse`` tool, shared by the tool
    /// runner (which suspends on it) and the SwiftUI approval sheet.
    @State private var browseApproval: BrowseApprovalStore
    /// MCP connectors (issue #1716) — the config file the engine reads, the
    /// live state it reports back, the per-tool consent gate, and the registry
    /// that ties them into the chat loop.
    @State private var mcpConfig: MCPConfigStore
    @State private var mcpCatalog: MCPCatalog
    @State private var mcpApproval: MCPToolApprovalStore
    @State private var mcpTools: MCPToolRegistry
    /// Per-model performance overrides (issue #1717) — the KV-cache and
    /// prefix-cache flags the app hands the engine at spawn.
    @State private var perfConfig: ModelPerfConfigStore

    /// AppKit delegate — installs the menu-bar tray + tears down the
    /// subprocess before the process image dies.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    init() {
        // Install the crash reporter FIRST — every other init step can
        // fatalError under bad disk / permissions state, and we want
        // those abortions to leave a marker for the next launch.
        CrashReporter.install()
        // Migrate only users who had explicitly changed the legacy
        // telemetry toggle. An absent value remains undecided/off and is
        // handled by the first-run consent sheet in ContentView.
        TelemetryConsent.synchronizeExistingDecision()
        // Sweep orphan rapid-mlx processes from previous sessions BEFORE
        // anything else looks at our serve port.
        //
        // Detached, NOT inline. This runs inside ``RapidApp.init`` —
        // i.e. before the first window is drawn — and the sweep forks
        // ``lsof`` (plus a ``ps`` per pid it finds) and can wait out a
        // SIGTERM grace when it actually finds an orphan. Inline, that
        // was dead time on the main thread with no UI on screen: the
        // app appeared to hang on launch (Dock icon bouncing, no
        // window) for as long as the sweep took.
        //
        // Fire-and-forget is correct here rather than something the
        // launch path awaits. The sweep is an opportunistic cleanup of
        // a PREVIOUS session's leftovers; nothing in this launch's
        // startup depends on its result. The real guarantee that we
        // don't collide with an orphan lives in
        // ``ServerManager.start`` → ``PortAllocator.allocate()``,
        // which re-sweeps each candidate port and bind-probes it
        // before spawning. If this background sweep hasn't finished by
        // the time the user picks a model, the allocator simply finds
        // the orphan itself and reaps it there.
        // Held as an awaitable handle rather than a fire-and-forget task, so
        // ``ServerManager.start`` can wait it out instead of racing it onto
        // the same port. Still detached — launch never blocks on it.
        PortSweep.startLaunchSweep(port: PortAllocator.candidatePorts.first ?? 8000)
        let manager = ServerManager()
        let samplingConfig = SamplingConfig()
        let customInstructionsConfig = CustomInstructionsConfig()
        let appearanceConfig = AppearanceConfig()
        // Apply the persisted theme override before the first window
        // renders so the user doesn't see a flash of the wrong mode.
        appearanceConfig.apply()
        // Built-in tools. The registry owns the two stores the tools consult
        // at dispatch time, and the app re-publishes them into the environment
        // so Settings + the approval sheet bind to the same instances.
        let webSearchConfig = WebSearchConfig()
        let browseApprovalStore = BrowseApprovalStore()
        let builtinRegistry = BuiltinToolRegistry(
            browseApproval: browseApprovalStore,
            webSearch: webSearchConfig
        )
        _webSearch = State(initialValue: webSearchConfig)
        _browseApproval = State(initialValue: browseApprovalStore)

        // Issue #1716: MCP connectors. The config store owns the file the
        // engine child reads; the catalog reads back what that child actually
        // connected to; the approval store gates every call the model makes.
        // All three are republished into the environment so Settings and the
        // approval dialog bind to the same instances the tool runner consults.
        let mcpConfigStore = MCPConfigStore()
        let mcpApprovalStore = MCPToolApprovalStore()
        let mcpCatalog = MCPCatalog { [weak manager] in
            // Only report an endpoint once the child answered /healthz —
            // polling a starting server just logs connection refusals, and a
            // stopped one has no bearer to authenticate with.
            guard let manager, manager.servingAlias != nil,
                  let bearer = manager.activeBearer else { return nil }
            return (host: manager.host, port: manager.activePort, bearer: bearer)
        }
        let mcpRegistry = MCPToolRegistry(catalog: mcpCatalog, approval: mcpApprovalStore)
        let toolRegistry = CompositeToolRegistry(builtin: builtinRegistry, mcp: mcpRegistry)
        // Resolved at each spawn rather than captured now — see
        // ``ServerManager/mcpConfigPathProvider``.
        manager.mcpConfigPathProvider = { [weak mcpConfigStore] in
            MainActor.assumeIsolated { mcpConfigStore?.launchConfigPath }
        }
        // Editing what a connector runs (or removing it) drops any "always
        // allow" remembered against it, so consent can't transfer to code the
        // user never approved.
        mcpConfigStore.onServerReconfigured = { [weak mcpApprovalStore] serverName in
            mcpApprovalStore?.revokeGrants(forServer: serverName)
        }
        // The config file is hand-editable, and an edit made while the app was
        // closed never hits the hook above. Reconcile once at launch against
        // the loaded connectors' fingerprints so a hand-swapped command drops
        // its grant before any tool can run.
        mcpApprovalStore.reconcileGrants(
            against: Dictionary(
                mcpConfigStore.servers.map { ($0.name, $0.executionFingerprint) },
                uniquingKeysWith: { first, _ in first }
            )
        )
        _mcpConfig = State(initialValue: mcpConfigStore)
        _mcpCatalog = State(initialValue: mcpCatalog)
        _mcpApproval = State(initialValue: mcpApprovalStore)
        _mcpTools = State(initialValue: mcpRegistry)

        // Issue #1717: per-model performance overrides. Resolved at each spawn
        // for the alias being started — see ``ServerManager/
        // perfLaunchFlagsProvider``. Empty for every alias the user has not
        // touched, so an install that never opens the panel spawns the same
        // argv as before.
        let perfConfigStore = ModelPerfConfigStore()
        manager.perfLaunchFlagsProvider = { [weak perfConfigStore] alias in
            MainActor.assumeIsolated { perfConfigStore?.launchFlags(forAlias: alias) ?? [] }
        }
        manager.perfConfigProvider = { [weak perfConfigStore] alias in
            MainActor.assumeIsolated {
                let recommended = RAMBucketedDefault.launchFlags(
                    forAlias: alias,
                    physicalRAMGB: MacHardware.detect().physicalRAMGB
                )
                let merged = ServerManager.mergedPerformanceFlags(
                    recommended: recommended,
                    userOverrides: perfConfigStore?.launchFlags(forAlias: alias) ?? []
                )
                let value = ModelPerfConfig(launchFlags: merged)
                return value.isEmpty ? nil : value
            }
        }
        _perfConfig = State(initialValue: perfConfigStore)

        let chat = ChatViewModel(
            tools: toolRegistry,
            sampling: samplingConfig,
            customInstructions: customInstructionsConfig,
            server: manager
        )
        let updateChecker = UpdateChecker()
        let sparkleUpdateController = SparkleUpdateController()
        let downloadsInstance = DownloadManager(binaryPath: manager.binaryPath)
        // #253: let ``ServerManager.start(alias:)`` await any in-flight
        // background pull for the same alias before spawning serve.
        manager.attachDownloads(downloadsInstance)
        _server = State(initialValue: manager)
        _downloads = State(initialValue: downloadsInstance)
        _installTracker = State(initialValue: InstallTracker())
        _quickstart = State(initialValue: QuickstartCoordinator())
        let dockPrompt = DockVisibilityPromptStore()
        _dockPromptStore = State(initialValue: dockPrompt)
        AppDelegate.shared.dockPromptStore = dockPrompt
        _chatViewModel = State(initialValue: chat)
        _imageGen = State(initialValue: ImageGenViewModel(server: manager))
        _audio = State(initialValue: AudioViewModel(server: manager))
        _dictation = State(initialValue: DictationController(server: manager))
        _updater = State(initialValue: updateChecker)
        _sparkleUpdater = State(initialValue: sparkleUpdateController)
        _sampling = State(initialValue: samplingConfig)
        _customInstructions = State(initialValue: customInstructionsConfig)
        _appearance = State(initialValue: appearanceConfig)
        _settingsRouter = State(initialValue: SettingsRouter())
        // Hand the live singletons to the delegate so the shutdown hook
        // and the AppKit menu-bar tray can reach them without rebuilding
        // the SwiftUI environment.
        AppDelegate.shared.server = manager
        AppDelegate.shared.downloads = downloadsInstance
        AppDelegate.shared.updater = updateChecker
        AppDelegate.shared.sparkleUpdater = sparkleUpdateController
        AppDelegate.shared.chat = chat
        AppDelegate.shared.appearance = appearanceConfig
    }

    var body: some Scene {
        Window("Rapid-MLX", id: "main") {
            ContentView()
                // Lock the whole app to the Rapid brand amber — the ⚡ energy
                // hue. Steel-blue (`brand`) is demoted to the info/tool/data
                // lane per the rapidmlx.com design system (rapid-desktop #632).
                .tint(RapidTheme.brandAmber)
                .environment(server)
                .environment(downloads)
                .environment(chatViewModel)
                .environment(imageGen)
                .environment(audio)
                .environment(dictation)
                .environment(updater)
                .environment(sparkleUpdater)
                .environment(sampling)
                .environment(customInstructions)
                .environment(appearance)
                .environment(settingsRouter)
                .environment(installTracker)
                .environment(quickstart)
                .environment(dockPromptStore)
                .environment(webSearch)
                .environment(browseApproval)
                .environment(mcpConfig)
                .environment(mcpCatalog)
                .environment(mcpApproval)
                .environment(mcpTools)
                .environment(perfConfig)
                .task {
                    // Dictation is a background service: arm it at launch so
                    // the hotkey works without ever opening the Audio tab.
                    await dictation.bootstrap()
                }
                .task {
                    // DEV-ONLY: render real screens to PNG when
                    // RAPID_DEV_SNAPSHOT_DIR is set, then quit. Inert
                    // (returns immediately) in normal use.
                    await DevSnapshot.runIfRequested(
                        server: server, downloads: downloads, chat: chatViewModel,
                        updater: updater, sampling: sampling, appearance: appearance,
                        settingsRouter: settingsRouter, installTracker: installTracker,
                        quickstart: quickstart, dockPromptStore: dockPromptStore)
                }
                .task {
                    // Register the AppKit→SwiftUI bridges so the Dock
                    // reopen hook and the menu-bar tray can materialise
                    // the main / settings scenes through ``openWindow``.
                    AppDelegate.openMainWindow = {
                        openWindow(id: "main")
                    }
                    AppDelegate.openSettingsWindow = {
                        openWindow(id: "settings")
                    }
                    AppDelegate.openSettingsWindowAt = { category in
                        // `route(to:open:)` assigns the pending category
                        // BEFORE opening — that ordering is load-bearing, see
                        // SettingsRouter.
                        settingsRouter.route(to: category) {
                            openWindow(id: "settings")
                        }
                    }
                }
                .task {
                    if sparkleUpdater.isEnabled {
                        // Sparkle owns the six-hour schedule, background
                        // download, signature validation, and install-on-quit.
                        sparkleUpdater.start()
                    }
                    // ONE check, at launch — deliberately not a timer.
                    //
                    // Sparkle owns "is there a newer version", and running a
                    // second six-hour poll beside its own was pure duplication.
                    // This single call still earns its keep: it refreshes the
                    // version pill and the Settings status, and it is the only
                    // thing that reports this build's version to the update
                    // endpoint, which is where the per-version install
                    // distribution comes from. Losing that would mean losing
                    // the only answer to "how many installs are still on an old
                    // build" (see #1944).
                    //
                    // A relaunch re-runs it, which is frequent enough for a
                    // distribution metric and for a status pill nobody watches
                    // continuously.
                    await updater.check()
                }
                .task(id: server.servingAlias) {
                    // When the server transitions to ``.ready(alias)``,
                    // fetch the per-alias ``ServerModelProfile`` and let
                    // ``SamplingConfig.applyServerProfile`` decide whether
                    // to apply its curated ``recommended_sampling``.
                    // Clear the stale reasoning parser BEFORE the async
                    // fetch so a chat send during an alias swap can't bump
                    // ``max_tokens`` for the wrong alias.
                    sampling.clearActiveReasoningParser()
                    guard let alias = server.servingAlias else { return }
                    let baseURL = ChatStreamClient.loopbackURL(port: server.activePort)
                    let bearer = server.activeBearer
                    guard let profile = await ServerProfileFetcher.fetch(
                        baseURL: baseURL,
                        alias: alias,
                        bearer: bearer
                    ) else {
                        return
                    }
                    guard !Task.isCancelled,
                          server.servingAlias == alias else { return }
                    sampling.applyServerProfile(profile)
                }
                .task {
                    // Flush any crash markers the previous launch left,
                    // then post one session_start if the user opted in.
                    await CrashReporter.flushPendingCrashReports()
                    await TelemetrySession.sendStartIfNeeded()
                }
        }
        .defaultSize(width: MainWindowDefaults.width, height: MainWindowDefaults.height)
        .windowResizability(.contentMinSize)
        .commands {
            // Replace the system-default "About" item with our own.
            CommandGroup(replacing: .appInfo) {
                Button("About Rapid-MLX") {
                    AboutPanel.show(server: server)
                }
            }
            // File → Export All Chats…  The per-conversation export lives on
            // the sidebar row menu; this is the whole-library backup, and the
            // menu bar is where a user looks for "get my data out" when they
            // don't have one particular chat in mind.
            CommandGroup(after: .newItem) {
                Button("Export All Chats…") {
                    ConversationExportPanel.exportAll(
                        conversations: chatViewModel.conversations,
                        folders: chatViewModel.folders
                    )
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
            // ⌘, → our Window-based Settings (replaces the default that
            // targeted the removed ``Settings`` scene).
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    openWindow(id: "settings")
                }
                .keyboardShortcut(",", modifiers: .command)
            }
            // View → Keep Window on Top (session state, not persisted).
            CommandGroup(after: .windowArrangement) {
                Toggle("Keep Window on Top", isOn: $keepWindowOnTop)
                    .keyboardShortcut("t", modifiers: [.command, .option])
                    .onChange(of: keepWindowOnTop) { _, newValue in
                        Self.applyWindowOnTop(newValue)
                    }
            }
        }

        // Settings window. A real ``Window`` scene (not the SwiftUI
        // ``Settings`` scene) so the tray's "Settings…" item can open it
        // reliably via ``openWindow(id:)`` — the ``Settings`` scene's
        // ``showSettingsWindow:`` selector isn't reachable from a
        // status-item menu action. ⌘, is re-wired in ``.commands``.
        Window("Settings", id: "settings") {
            SettingsView()
                .tint(RapidTheme.brandAmber)
                .environment(chatViewModel)
                .environment(sampling)
                .environment(customInstructions)
                .environment(appearance)
                .environment(settingsRouter)
                .environment(server)
                // Settings → Developer resets the wizard, and SwiftUI traps
                // rather than warns when an @Environment observable is
                // missing — so the Settings scene needs this even though the
                // panel that reads it only exists in a debug build.
                .environment(quickstart)
                .environment(downloads)
                .environment(updater)
                .environment(sparkleUpdater)
                .environment(dockPromptStore)
                .environment(webSearch)
                .environment(browseApproval)
                .environment(mcpConfig)
                .environment(mcpCatalog)
                .environment(mcpApproval)
                .environment(mcpTools)
                .environment(perfConfig)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 900, height: 720)
    }

    /// Walk ``NSApp.windows`` and flip the main chat window's level to
    /// ``.floating`` when on, ``.normal`` when off, matching by SwiftUI's
    /// window identifier ("main").
    static func applyWindowOnTop(_ enabled: Bool) {
        let target: NSWindow.Level = enabled ? .floating : .normal
        for window in NSApp.windows {
            guard window.identifier?.rawValue == "main" else { continue }
            window.level = target
        }
    }
}

/// AppKit delegate whose only job is to tear down the embedded child
/// before the process exits. SwiftUI's pure `App` lifecycle has no
/// equivalent synchronous hook — `onDisappear` and scene phase changes
/// fire on app activation transitions, not on terminate.
///
/// ``@MainActor`` is required by Swift 6 strict concurrency — AppKit
/// delegate callbacks all land on the main thread anyway, and the
/// static ``shared`` holder needs an isolation domain to stop being
/// a "non-Sendable shared mutable state" diagnostic.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Shared instance used by the `RapidApp` initialiser. AppKit
    /// constructs the delegate via the SwiftUI adaptor, so we route
    /// through a static holder rather than passing the manager into
    /// the delegate's init.
    static let shared = AppDelegate()

    weak var server: ServerManager?
    weak var downloads: DownloadManager?
    /// Hand from ``RapidApp.init`` so ``applicationWillTerminate`` can
    /// cancel the in-flight chat stream task before shutting the child
    /// down, and the menu-bar tray's "New Chat" can reach it.
    weak var chat: ChatViewModel?
    /// Persisted theme override. Re-applied from
    /// ``applicationDidFinishLaunching`` so the user's "Light" choice
    /// takes effect on the first window even when the host system is
    /// in Dark Mode. ``RapidApp.init``'s eager ``apply()`` runs too
    /// early — NSApp's appearance machinery isn't wired up yet — so
    /// the recorded value silently drops on launch.
    weak var appearance: AppearanceConfig?
    /// #502: live state the AppKit menu-bar tray (``MenuBarController``)
    /// reads on menu open and glyph repaint. Handed from ``RapidApp.init``
    /// like ``server`` / ``sessionStore`` above; weak because the SwiftUI
    /// ``@State`` on ``RapidApp`` owns each for the app's lifetime.
    weak var updater: UpdateChecker?
    weak var sparkleUpdater: SparkleUpdateController?
    /// The single AppKit menu-bar (tray) surface. Installed in
    /// ``applicationDidFinishLaunching`` and held strongly so the
    /// ``NSStatusItem`` slot stays alive for the app's lifetime —
    /// releasing it would tear the tray icon out of the menu bar.
    /// There is intentionally no SwiftUI ``MenuBarExtra`` counterpart:
    /// its glyph does not render on macOS 26 (#502), and two surfaces
    /// at once is the #475 double-icon bug.
    private var menuBarController: MenuBarController?

    /// Issue #260: persisted choice + Dock-icon activation-policy
    /// driver. Stored on the delegate so the main-window close
    /// interceptor can reach it without rebuilding the SwiftUI
    /// environment chain (the delegate runs outside the view tree).
    /// Wired in ``RapidApp.init``.
    var dockPromptStore: DockVisibilityPromptStore?

    /// Strong reference to the main window's close interceptor. The
    /// chained ``NSWindowDelegate`` proxy lives for the app lifetime
    /// — releasing it would let the SwiftUI scene's default close
    /// path fire without ever running the #260 prompt. Slot is
    /// populated by ``ContentView``'s ``WindowAccessor`` on first
    /// window appear; AppDelegate stores it because the slot must
    /// survive scene re-mount across hide/show cycles.
    var mainWindowCloseInterceptor: MainWindowCloseInterceptor?

    /// Attach the AppKit-only main-window behaviours once SwiftUI has
    /// materialised its concrete ``NSWindow``. Repeated accessor callbacks
    /// are expected; installation is idempotent for the same window and is
    /// repeated when SwiftUI creates a replacement after a normal close.
    func attachMainWindow(_ window: NSWindow) {
        let needsInstall = MainWindowCloseInterceptor.shouldReinstall(
            currentAttachedWindow: mainWindowCloseInterceptor?.attachedWindow,
            newWindow: window
        )
        guard let visibleFrame = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame else {
            if needsInstall, let store = dockPromptStore {
                mainWindowCloseInterceptor = MainWindowCloseInterceptor(window: window, store: store)
            }
            return
        }

        if needsInstall {
            // Setting the name restores any saved frame synchronously. Clamp
            // immediately afterwards so a monitor that was unplugged while
            // the app was closed cannot strand the restored window.
            window.setFrameAutosaveName("Rapid.MainWindow")
            window.setFrame(WindowFrameClamp.clamp(frame: window.frame, to: visibleFrame), display: false)
            if let store = dockPromptStore {
                mainWindowCloseInterceptor = MainWindowCloseInterceptor(window: window, store: store)
            }
        } else if WindowFrameClamp.isStranded(frame: window.frame, in: visibleFrame) {
            // The onboarding sheet can cause a later AppKit reposition after
            // the initial autosave restore. Only fight genuinely stranded
            // frames on repeat callbacks, preserving deliberate edge parking.
            window.setFrame(WindowFrameClamp.clamp(frame: window.frame, to: visibleFrame), display: true)
        }
    }

    /// AppKit instantiates the delegate via the SwiftUI adaptor's
    /// `init()`. Returning the shared singleton on `init()` is not
    /// straightforward, so we let AppKit make its own instance and
    /// keep the static singleton holding the manager reference. The
    /// adaptor-created delegate forwards through to the singleton.
    override init() {
        super.init()
    }

    /// #173: whether an ``AXEnhancedUserInterface`` set result warrants a
    /// stderr diagnostic. ``.success`` obviously doesn't; ``.notImplemented``
    /// (``-25208``) is the EXPECTED macOS-15+/26 result — the in-process
    /// set is a no-op there and the SwiftUI ``Window`` bridge stays dormant
    /// for non-VoiceOver users, so a line on every launch is pure noise.
    /// Every OTHER non-success is genuinely unexpected (a real contract
    /// change worth a canary line). Pure + ``static`` so a unit test can
    /// pin the policy without standing up ``NSApplication``.
    /// ``nonisolated`` because it touches no actor state — lets the unit
    /// test call it synchronously off the main actor.
    nonisolated static func shouldLogAXBridgeResult(_ err: AXError) -> Bool {
        err != .success && err != .notImplemented
    }

    /// Force ``.regular`` activation policy and enable the SwiftUI
    /// accessibility bridge so the main ``Window`` scene reaches
    /// VoiceOver, other assistive tech, and automation harnesses.
    ///
    /// SwiftUI's default activation-policy heuristic is not pinned
    /// across macOS releases — some pick ``.regular``, some pick
    /// ``.accessory``. Setting it explicitly here removes the
    /// ambiguity. The menubar-resident behaviour is preserved because
    /// ``applicationShouldTerminateAfterLastWindowClosed`` returns
    /// false (same shape Ollama uses) and the AppKit menu-bar tray
    /// (``MenuBarController``) keeps the app reachable with no window
    /// open.
    ///
    /// The ``AXEnhancedUserInterface`` flip is the load-bearing
    /// half. Without it, ``AXMainWindow`` / ``AXFocusedWindow`` /
    /// ``AXFocusedUIElement`` on the application AX element all
    /// resolve back to the AXApplication itself instead of the
    /// underlying ``Window`` scene — the SwiftUI scene-graph
    /// accessibility-bridge stays dormant. VoiceOver normally
    /// flips this attribute on startup; without VoiceOver running,
    /// the window hierarchy is invisible to every assistive
    /// surface, breaking screen readers and any CI smoke harness
    /// that walks the AX tree to drive the chat surface. Setting
    /// it here from inside the app is the historically-correct call
    /// site: an EXTERNAL setter always gets ``-25208`` /
    /// ``kAXErrorNotImplemented`` because the AXApplication element
    /// exposes no settable ``AXEnhancedUserInterface`` to a foreign
    /// process. Note the in-process set is itself best-effort — on macOS
    /// 15+ (and 26 / Tahoe) it ALSO returns ``kAXErrorNotImplemented``
    /// and the bridge stays dormant for non-VoiceOver users (issue
    /// #173). It is kept because it still takes effect on macOS 14 and
    /// is a benign no-op (not a regression) where it doesn't — the
    /// prior comment mislabelled ``-25208`` as "attribute not writable"
    /// (that is ``-25205`` / ``kAXErrorAttributeUnsupported``, a
    /// different code).
    /// Runs after ``NSApp`` is initialised but BEFORE the first window
    /// or Dock icon renders. Issue raullenchai/Rapid-MLX#845
    /// (v0.8.0→v0.8.2 hotfix): the previous landing point for this
    /// flip was ``RapidApp.init()``, but SwiftUI's ``App.init()`` runs
    /// before ``NSApplicationMain`` initialises ``NSApp``, so the
    /// implicitly-unwrapped global force-unwrapped ``nil`` → SIGTRAP
    /// for every user with ``hideAlways`` persisted. Doing the flip
    /// here means ``NSApp`` is alive AND the user with
    /// ``hideAlways`` still avoids the brief "Dock icon flashes then
    /// disappears" jolt that motivated the eager flip in the first
    /// place.
    ///
    /// ``applicationDidFinishLaunching`` re-asserts the policy as a
    /// defence-in-depth backstop (covers the rare case where AppKit
    /// resets us between will- and did-FinishLaunching).
    ///
    /// Why the optional-chained ``NSApp?.``: under production launch
    /// ``NSApp`` is always alive at this delegate hook (that's the
    /// whole reason this code lives here and not in ``RapidApp.init``).
    /// But the ``InitMustNotTouchNSAppTests`` companions invoke this
    /// method directly from ``swift test``, which never calls
    /// ``NSApplicationMain`` and therefore leaves ``NSApp`` as ``nil``.
    /// Implicitly-unwrapping it (the v0.8.0 shape that the #845 hotfix
    /// only half-fixed) SIGTRAPs the test runner in isolation —
    /// ``swift test --filter InitMustNotTouchNSAppTests`` would crash
    /// 1/3 cases; the full suite only "passed" because earlier
    /// alphabetical tests transitively initialised
    /// ``NSApplication.shared``. Safe-unwrap here is a no-op in
    /// production and lets the test harness exercise this hook
    /// without depending on alphabetical test ordering.
    func applicationWillFinishLaunching(_ notification: Notification) {
        guard let choice = dockPromptStore?.choice else { return }
        if choice == .hideAlways {
            NSApp?.setActivationPolicy(.accessory)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The pin is defensive — Resources/Info.plist has no
        // ``LSUIElement`` entry so AppKit resolves to ``.regular``
        // anyway, but keeping the explicit set means a future
        // experiment with ``LSUIElement=true`` (to chase the Ollama
        // shape) can't silently re-route the AX bridge through the
        // accessory branch. Note this combines with
        // ``applicationShouldTerminateAfterLastWindowClosed`` returning
        // false to keep the dock icon visible even after the user
        // closes the main window — that's the Slack/Linear shape, not
        // the Ollama shape (Ollama uses ``.accessory``+no dock icon).
        //
        // #260: honour the persisted "hide Dock icon on close" choice
        // — if the user previously picked Yes + Don't ask again, the
        // app should boot in ``.accessory`` so the first frame
        // doesn't briefly show the Dock icon (jarring "did the
        // setting unstick?" flash). ``RapidApp.init`` already set
        // the policy eagerly from the same source; re-asserting here
        // covers the rare case where AppKit's own initialisation
        // reset us back to ``.regular`` between init and the launch
        // delegate. Default and ``.keepAlways`` both fall through
        // to ``.regular``.
        // Safe-unwrap ``NSApp?`` mirrors PR #376's fix in the sibling
        // ``applicationWillFinishLaunching`` hook (line ~949): production
        // launch via ``NSApplicationMain`` has ``NSApp`` alive here, but
        // a ``swift test`` runner can invoke this delegate method
        // directly while ``NSApplication.shared`` is uninitialised, in
        // which case the implicitly-unwrapped global is ``nil`` and a
        // bare ``NSApp.foo`` call SIGTRAPs the runner. No test
        // exercises this hook today; the optional-chain is precautionary
        // so a future ``--filter`` companion doesn't have to chase the
        // same hotfix again.
        let dockChoice = AppDelegate.shared.dockPromptStore?.choice ?? .notAsked
        if dockChoice == .hideAlways {
            NSApp?.setActivationPolicy(.accessory)
        } else {
            NSApp?.setActivationPolicy(.regular)
        }
        // ``ignoringOtherApps: false`` keeps focus with whatever the
        // user had open if Rapid was launched non-interactively (e.g.
        // login items, ``open -ga Rapid`` from a script). A
        // user-initiated Finder launch still brings Rapid forward via
        // the Launch Services activation hint — but a background
        // launch no longer yanks focus from the user's editor. Codex
        // round 1 NIT #1.
        NSApp?.activate(ignoringOtherApps: false)
        // Re-apply the persisted theme override now that ``NSApp``
        // is fully bootstrapped — ``RapidApp.init``'s eager
        // ``apply()`` runs while NSApplicationMain is still wiring up
        // the appearance machinery, so the value is recorded but
        // doesn't propagate to the first window. Without this second
        // call, a user with "Light" persisted opens the app to a
        // dark-mode window when the host is in Dark Mode and has to
        // visit Settings → Appearance and toggle the radio (any
        // change re-runs ``apply()`` via ``didSet``) to get the
        // theme they actually saved.
        AppDelegate.shared.appearance?.apply()
        // Deferred a runloop tick so the AX server has seen NSApp's
        // own registration; ``Task { @MainActor }`` cooperates with
        // Swift 6 strict-concurrency the rest of the file relies on.
        Task { @MainActor in
            let app = AXUIElementCreateApplication(getpid())
            let err = AXUIElementSetAttributeValue(
                app,
                "AXEnhancedUserInterface" as CFString,
                kCFBooleanTrue
            )
            // ``AXError`` is a thin enum over an OSStatus. Logging on an
            // UNEXPECTED non-success means a future macOS where Apple
            // tightens the contract surfaces as a warning in the log tail
            // rather than as "VoiceOver inexplicably broken" months
            // later. Codex round 1 NIT #4.
            //
            // #173 (formerly #169): ``-25208`` == ``kAXErrorNotImplemented``
            // is the EXPECTED result on macOS 15+/26 — the SwiftUI
            // ``Window`` scene exposes no settable
            // ``AXEnhancedUserInterface`` to the in-process caller there,
            // so the bridge stays dormant for non-VoiceOver users (benign:
            // mouse/keyboard/trackpad UI unaffected; VoiceOver flips it via
            // its own path). Emitting a stderr line for that known-benign
            // code on EVERY launch is pure noise, so stay silent for it and
            // log only genuinely-unexpected codes.
            if Self.shouldLogAXBridgeResult(err) {
                fputs(
                    "Rapid: AXEnhancedUserInterface set returned err=\(err.rawValue) (\(err))\n",
                    stderr
                )
            }
        }
        // #502: install the persistent menu-bar tray AFTER the
        // activation-policy + AX setup, so the status-bar slot inherits
        // NSApp's settled appearance on the first frame. This AppKit
        // ``NSStatusItem`` is the SINGLE tray surface: SwiftUI's
        // ``MenuBarExtra`` glyph does not render on macOS 26 (Darwin
        // 25.x / Tahoe), so it was removed entirely rather than run
        // alongside this one — two surfaces at once is the #475
        // double-icon bug. The controller reads its live state
        // (server / updater / sparkleUpdater / sessionStore / quickAsk)
        // through ``AppDelegate.shared``, all populated by
        // ``RapidApp.init`` above.
        menuBarController = MenuBarController()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Codex round-2 finding: the previous implementation posted
        // ``Task { @MainActor in await store.flush() }`` and then
        // blocked the main thread with ``sem.wait`` — a deadlock,
        // because ``applicationWillTerminate`` runs on the main
        // thread and the posted task could never schedule. The
        // 2 s timeout would always fire and the user's last edits
        // would be lost. ``flushSync()`` performs the encode + atomic
        // write inline so the data lands before this delegate hook
        // returns to AppKit.
        MainActor.assumeIsolated {
            AppDelegate.runStandardTermination()
        }
        // Last write before AppKit pulls the plug — clears this
        // launch's crash marker so the NEXT launch doesn't
        // misclassify our clean exit as an unclean shutdown. Must
        // run after the data flushes above so a slow flush that
        // turns into a hang is still caught.
        CrashReporter.recordCleanShutdown()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Match Ollama / menubar-resident behaviour: closing the last
        // window keeps the app alive behind the menu-bar tray
        // (``MenuBarController``). The user quits via the tray "Quit"
        // item or Cmd-Q.
        false
    }

    /// Handle the Dock/Launchpad re-open click while the app is alive
    /// but has no visible windows (the menu-bar-resident shape we
    /// inherit from ``applicationShouldTerminateAfterLastWindowClosed``
    /// returning false). Without this hook, clicking the Dock icon
    /// after the user closed the main window via ⌘W or the red traffic
    /// light is a silent no-op: AppKit sees there's nothing to
    /// un-hide because the SwiftUI ``Window`` scene was destroyed on
    /// close, and the user is left wondering whether the app is
    /// running at all. The only escape is the menu-bar tray's "Open
    /// Rapid-MLX" item — which assumes the user knows about the tray
    /// icon, breaking the muscle-memory contract every other
    /// Mac app honours (Slack, Linear, Discord, Things all re-show
    /// their main window on Dock click, even when otherwise
    /// menubar-resident).
    ///
    /// Synthesise the same flow the tray's "Open" button runs by
    /// posting through the static ``openMainWindow`` bridge that
    /// ``RapidApp`` wires to SwiftUI's ``@Environment(\.openWindow)``
    /// on the main scene's first appearance. We cannot call
    /// ``openWindow(id:)`` directly from AppKit because that
    /// environment value isn't accessible from the delegate.
    ///
    /// Return ``true`` per the documented contract: it tells AppKit
    /// we handled the request and prevents the default "un-hide /
    /// un-minimise" pathway from also firing on the same click.
    func applicationShouldHandleReopen(
        _ sender: NSApplication,
        hasVisibleWindows flag: Bool
    ) -> Bool {
        // If the user already has a window visible (un-hidden via
        // Cmd+Tab, came back from another Space), AppKit's default
        // behaviour is correct — just bring it forward. Don't double-
        // fire the open.
        if flag {
            return true
        }
        NSApp.activate(ignoringOtherApps: true)
        AppDelegate.openMainWindow?()
        return true
    }

    /// Bridge between AppKit (``applicationShouldHandleReopen``, the
    /// menu-bar tray's "Open Rapid-MLX" item, future URL handlers) and
    /// SwiftUI's ``@Environment(\.openWindow)``. ``RapidApp`` captures
    /// its ``openWindow`` action on the main scene's first ``.task`` and
    /// writes it here; AppKit-side callers invoke it without owning an
    /// environment binding. ``nil`` until the scene has materialised
    /// once — guarded with ``?.()`` so a reopen click before first
    /// launch is a no-op, not a crash.
    static var openMainWindow: (@MainActor () -> Void)?

    /// Bridge for the Settings window, driven from the tray's
    /// "Settings…" item AND the ⌘, command. Uses a real ``Window``
    /// scene + ``openWindow(id:)`` rather than the SwiftUI ``Settings``
    /// scene's ``showSettingsWindow:`` selector — that selector is on
    /// the responder CHAIN, not ``NSApplication``, so a programmatic
    /// ``sendAction`` from the status-item menu never reached it and the
    /// tray "Settings…" item silently did nothing.
    static var openSettingsWindow: (@MainActor () -> Void)?

    /// Open Settings deep-linked to a category, for call sites outside the
    /// SwiftUI environment (the tray reads its dependencies through
    /// ``AppDelegate.shared`` and cannot see ``SettingsRouter`` directly).
    ///
    /// Exists because the tray's "Check for updates…" had nowhere to report
    /// to: it fired the check and discarded the result, so a user on the
    /// latest version clicked and saw nothing at all. Settings → App already
    /// renders that state ("Up to date — vX.Y.Z is the latest release."), so
    /// the item routes there rather than growing a second surface that could
    /// disagree with the first.
    static var openSettingsWindowAt: (@MainActor (SettingsView.Category?) -> Void)?

    /// Canonical termination ordering. Audit P1 wants the in-flight
    /// chat stream cancelled BEFORE the session envelope is
    /// normalised / flushed and BEFORE the server child is torn
    /// down — see `applicationWillTerminate` for the full rationale.
    ///
    /// Pulled out as a pure closure-driven helper so the ordering
    /// can be pinned by a unit test (the `applicationWillTerminate`
    /// hook itself isn't directly testable without standing up
    /// AppKit). Production code threads the live AppDelegate slots
    /// through; tests pass spy closures that record call order.
    ///
    /// Codex r1 NIT: prior version had the 5 calls inlined in
    /// `applicationWillTerminate`, with no test pinning the
    /// stop-first invariant against a future reorder.
    ///
    /// Teardown is SPLIT into a signal phase and a reap phase. The
    /// previous shape called a single blocking `shutdownSync()` per
    /// subsystem back-to-back, so their grace windows SUMMED: the
    /// server's 5 s SIGTERM grace (+0.5 s post-SIGKILL settle) ran to
    /// completion before the download manager even sent its first
    /// SIGTERM, then that added its own 2 s — up to ~7.5 s of blocked
    /// main thread. macOS gives `applicationWillTerminate` a finite
    /// budget before force-killing the app, and everything after the
    /// teardown (`ConversationStore.flush()`, and
    /// `CrashReporter.recordCleanShutdown()` at the call site) is
    /// exactly the work that must not be skipped — losing the latter
    /// makes the NEXT launch misreport a clean quit as a crash.
    ///
    /// Signalling every child FIRST means the two grace windows
    /// OVERLAP: the download children spend the server's 5 s grace
    /// dying, so the download reap that follows almost always finds
    /// them already gone and returns immediately. Worst case is now
    /// bounded by the longest single grace rather than their sum, and
    /// no grace period was shortened — the server's 5 s window is
    /// load-bearing for rapid-mlx's prefix-cache flush.
    static func runTerminationSequence(
        stopStream: () -> Void,
        signalServer: () -> Void,
        signalDownloads: () -> Void,
        reapServer: () -> Void,
        reapDownloads: () -> Void
    ) {
        // ORDER MATTERS — the audit P1 invariant is: stopStream BEFORE
        // any teardown so the inflight URLSessionDataTask FIN reaches
        // rapid-mlx before the child is SIGTERM'd.
        stopStream()
        // Signal phase — non-blocking. Both subsystems get their
        // SIGTERM before anyone waits, so the graces overlap.
        signalServer()
        signalDownloads()
        // Reap phase — blocking. Server first: its grace is the long
        // one, and by the time it returns the download children have
        // had that entire window to exit.
        reapServer()
        reapDownloads()
        // Drain any queued conversation-history write so the last turn /
        // edit / deletion isn't lost when the process exits before the
        // async save lands. Folders drain alongside it — the two files are
        // written in the same user actions (deleting a folder unfiles the
        // conversations in it), so flushing only one can leave the pair
        // disagreeing about where a row lives.
        ConversationStore.flush()
        ConversationFolderStore.flush()
    }

    /// The single clean-shutdown wiring, shared by ``applicationWillTerminate``
    /// and the re-onboarding relaunch (``ReonboardingReset``). Both paths end
    /// the process, so both must persist the chat, reap the server, and stop
    /// the download children — keeping this in one place stops the two from
    /// drifting (the re-onboarding path originally teardown only the server
    /// and orphaned in-flight downloads / lost the last chat edit, #1973).
    @MainActor
    static func runStandardTermination() {
        runTerminationSequence(
            stopStream: { AppDelegate.shared.chat?.stopAndPersist() },
            signalServer: { AppDelegate.shared.server?.beginShutdown() },
            signalDownloads: { AppDelegate.shared.downloads?.beginShutdown() },
            reapServer: { AppDelegate.shared.server?.shutdownSync() },
            reapDownloads: { AppDelegate.shared.downloads?.finishShutdown() }
        )
    }
}
