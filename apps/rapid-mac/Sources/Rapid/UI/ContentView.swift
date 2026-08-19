import SwiftUI

/// Main window content. Minimal menu-bar app: a model picker at the
/// top, the chat transcript in the middle, a status footer at the
/// bottom. The chat surface is gated on ``ServerState`` — before the
/// server is ready the picker's Start button owns the flow, and a
/// brand-new user with no model on disk sees the Quickstart card.
struct ContentView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// #459: hard window-width floor. The original 640pt target preserved
    /// half-screen tiling on built-in MacBook displays; 720pt is the smallest
    /// width at which the current onboarding and workspace remain usable.
    /// Enforced on the shell below, not just declared. It was a bare
    /// constant read only by a test for a while, which meant the number and
    /// the window could disagree without anything noticing — and they did:
    /// the rail's 176pt minimum plus the 440pt detail floor let the window
    /// go to ~616.
    static let minWindowWidth: CGFloat = 720
    static let minWindowHeight: CGFloat = 560

    @Environment(ServerManager.self) private var server
    @Environment(DownloadManager.self) private var downloads
    @Environment(ChatViewModel.self) private var chat
    @Environment(ImageGenViewModel.self) private var imageGen
    @Environment(AudioViewModel.self) private var audio
    @Environment(SamplingConfig.self) private var sampling
    @Environment(UpdateChecker.self) private var updater
    @Environment(QuickstartCoordinator.self) private var quickstart
    @Environment(BrowseApprovalStore.self) private var browseApproval
    @Environment(MCPCatalog.self) private var mcpCatalog
    @Environment(MCPToolApprovalStore.self) private var mcpApproval
    @Environment(\.openWindow) private var openWindow

    @State private var alias: String = ""
    /// Monotonic signal from picker row taps. Catalog initialization also
    /// writes ``alias``, so the value alone cannot distinguish automation
    /// from a real user override while launch probing is suspended.
    @State private var userSelectionRevision: UInt = 0
    /// Which detail surface the sidebar shows (chat vs the Launch page).
    @State private var section: SidebarSection = .chat
    /// Window-level conversation search, opened from the toolbar.
    @State private var showConversationSearch = false
    /// An absent telemetry preference is an undecided first run, not
    /// implicit consent.
    @State private var telemetryConsentPending = TelemetryConsent.needsDecision()
    @SceneStorage("Rapid.showLogs") private var showLogs: Bool = false
    /// Per-session "browse all models" dismissal of the Quickstart card.
    @State private var quickstartDismissedThisSession: Bool = false
    /// Monotonic composer-focus request forwarded to ``ChatView``. Bumped
    /// by the onboarding completion transaction so the user lands in their
    /// first chat with the caret already in the message field.
    @State private var composerFocusRequest: Int = 0
    @Environment(SettingsRouter.self) private var settingsRouter
    /// #223: launch-time auto-start "needs download" state — the empty
    /// state names the pending pull when non-nil.
    @State private var autoStartPendingDownload: (alias: String, sizeText: String?)?
    /// Catalog snapshot, owned here rather than in ``ChatView`` so the
    /// chat surface and the Launch page resolve readiness from the SAME
    /// inputs. Two views calling the same pure function on different
    /// catalogs would reintroduce exactly the drift ``ModelReadiness``
    /// exists to remove. Empty until the first refresh lands.
    @State private var catalogEntries: [ModelEntry] = []
    /// False until the first catalog refresh completes. Distinguishes
    /// "this alias is not downloaded" from "we don't know yet", which
    /// are different sentences and different next steps.
    @State private var catalogLoaded: Bool = false
    /// FU-1: persisted opt-out for the launch-time auto-start path.
    @AppStorage(AutoStartPreference.storageKey) private var autoStartOnLaunch: Bool = AutoStartPreference.defaultValue

    var body: some View {
        // Capture the identity owned by this alert render. A delayed dismiss
        // callback must not cancel a newer warning that has reached the queue
        // head in the meantime (#1463).
        let displayedMemoryWarning = server.pendingMemoryWarning
        // Ollama-style layout: a left sidebar (New Chat / Launch / — later —
        // history) + a detail pane. No top model-control bar; the model
        // picker lives inline in the compose box (see ChatView) and the
        // model comes up on first send (implicit lifecycle). Search belongs
        // to the sidebar column beside macOS's native collapse control.
        Group {
            if quickstartVisible {
                // Setup owns the window (Paper 05.1.A). See ``onboardingShell``.
                onboardingShell
            } else {
                productionShell
            }
        }
        // `.windowResizability(.contentMinSize)` derives the window's floor
        // from this, so the shell states it once here rather than leaving it
        // to whatever the rail and the detail happen to add up to.
        .frame(minWidth: Self.minWindowWidth, minHeight: Self.minWindowHeight)
        .background {
            // Bridge the SwiftUI scene to the AppKit behaviours that need the
            // concrete main NSWindow: close interception, frame autosave, and
            // recovery from a display-layout change. This was documented as
            // the installation point but had never been mounted (#1590).
            WindowAccessor { window in
                AppDelegate.shared.attachMainWindow(window)
            }
            .frame(width: 0, height: 0)
        }
        .overlay {
            if showConversationSearch {
                conversationSearchOverlay
            }
        }
        .onChange(of: server.state) { _, newState in
            // #223: clear the download-prompt CTA the moment the server
            // moves out of ``.idle``.
            if case .idle = newState {} else { autoStartPendingDownload = nil }
            // A stale "couldn't reach the model" banner is actioned once
            // the server reaches ``.ready`` again.
            if case .ready = newState {
                chat.clearStaleErrorBanner()
                if server.downloadProgress.hasObservedGrowth {
                    downloads.markCacheChanged()
                }
                // Issue #1716: the child has just published which MCP servers
                // it connected to. Read it once here rather than polling —
                // the state only changes on a spawn or an explicit reload,
                // and both funnel through a point that refreshes.
                Task { await mcpCatalog.refresh() }
            } else {
                // Any non-ready state means no child is answering. Drop the
                // tool list: keeping it would let the chat loop advertise —
                // and try to execute — tools with no process behind them.
                mcpCatalog.clear()
            }
        }
        .onChange(of: server.state) { _, newState in
            // Sync the picker breadcrumb when the server lands in
            // ``.ready(<alias>)`` against a different alias than shown.
            if case .ready(let serving) = newState,
               !serving.isEmpty,
               serving != alias,
               ContentView.shouldSyncChatAlias(
                   serving: serving,
                   catalogEntries: catalogEntries,
                   knownMediaAliases: Set(
                       audio.audioModels.map(\.alias) + imageGen.imageModels.map(\.alias)
                   ),
                   section: section
               ) {
                alias = serving
            }
            // Retire pending-Ready provenance the moment the app serves some
            // OTHER model. The record names the alias it is about precisely
            // so it can be falsified like this: once the user has moved on,
            // offering to confirm the old flow on the next launch would be
            // confirming something that is no longer true.
            if case .ready(let serving) = newState,
               quickstart.hasPendingReady,
               serving != quickstart.selection.alias {
                quickstart.clearPendingReady()
            }
        }
        .alert(
            displayedMemoryWarning?.title ?? "",
            isPresented: Binding(
                // #1503: when the Quickstart sheet is up it renders its OWN
                // in-sheet copy of this decision (this alert is anchored on
                // the parent, behind the full-window sheet, and cannot
                // present over it). Suppress here so we don't double-present
                // on macOS builds where an alert DOES stack above a sheet —
                // gated on the exact predicate QuickstartView presents on.
                get: {
                    server.pendingMemoryWarning != nil && !memoryWarningHandledByQuickstart
                },
                // SwiftUI writes `false` here AFTER a button action runs.
                // Both buttons already resolved the decision and cleared
                // `pendingMemoryWarning`, so an unconditional cancel would
                // immediately undo a "Load anyway" the user just picked.
                // Only treat this as a dismissal when nothing resolved it
                // (Esc, click-outside).
                set: {
                    if !$0, let warning = displayedMemoryWarning {
                        server.cancelPendingMemoryLoad(warning)
                    }
                }
            ),
            presenting: displayedMemoryWarning
        ) { warning in
            Button("Cancel", role: .cancel) {
                if let running = runningAlias { alias = running }
                server.cancelPendingMemoryLoad(warning)
            }
            .accessibilityIdentifier("MemoryWarning.Cancel")
            Button(warning.confirmTitle, role: .destructive) {
                server.confirmPendingMemoryLoad(warning)
            }
            .accessibilityIdentifier("MemoryWarning.Confirm")
        } message: { warning in
            Text(warning.message)
        }
        .sheet(isPresented: firstRunSheetPresented) {
            firstRunSheet
        }
        .onChange(of: settingsRouter.quickstartReturnGeneration) { _, _ in
            quickstartDismissedThisSession = false
        }
        // Per-fetch approval for the ``browse`` tool. Skipped entirely when
        // the user has turned on auto-approve in Settings (resolved before a
        // request is ever published), so it only appears on a real prompt.
        .modifier(BrowseApprovalDialog(store: browseApproval))
        // Issue #1716: per-tool consent for MCP connector tools. Same shape as
        // the browse sheet above — an MCP server is an arbitrary local process,
        // so "may the model run this" is a decision that belongs on screen.
        .modifier(MCPToolApprovalDialog(store: mcpApproval))
        // #1589: keyed on the consent decision rather than fire-once, so
        // the launch auto-start that stood down for the modal sheet gets
        // its turn the moment the user answers it. The value only ever
        // moves true → false (once per install), so this is a single
        // re-run, and the first pass returns at the gate without an
        // in-flight `server.start` for SwiftUI's task cancellation to
        // interrupt. Users with no pending decision see one run, as before.
        .task(id: telemetryConsentPending) { await runLaunchAutoStart() }
        // Keyed exactly as the picker's own catalog task: re-fetch when
        // the engine binary appears and whenever the set of models on
        // disk changes anywhere in the app. Routed through the shared
        // ``ModelCatalogCache`` so owning a second reader here costs no
        // extra subprocess.
        .task(id: ModelPickerBar.PickerCatalogKey(
            binaryPath: server.binaryPath,
            cacheGeneration: downloads.cacheGeneration,
            refreshEnabled: !telemetryConsentPending
        )) {
            guard !telemetryConsentPending else { return }
            await refreshCatalogSnapshot()
        }
        .task {
            while !Task.isCancelled {
                await server.refreshResidency()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    /// First-run setup, filling the window (Paper 05.1.A).
    ///
    /// ## Why this is not a sheet
    ///
    /// It was one, and that was the defect. `.sheet` on macOS is a
    /// document-modal panel: AppKit sizes it to its content's ideal width,
    /// insets it, rounds its corners and leaves the parent window visible
    /// around and behind it. So onboarding rendered as a centred card floating
    /// over a dimmed chat surface — the exact composition 05.1.A rules out
    /// ("no dimmed application, no modal card floating over a live app, no
    /// composer, no production sidebar and no status strip behind it"). Worse,
    /// the sheet settled at its 620pt minimum, which is below the 820pt
    /// breakpoint, so the rail collapsed to its compact strip on a 1440pt
    /// display and the full-height rail was effectively unreachable.
    ///
    /// Replacing the shell instead of covering it makes the geometry correct
    /// by construction: this view IS the window's content, so it fills
    /// everything below the native title bar, has no corner radius of its own,
    /// and there is nothing behind it to show through. The title bar and its
    /// traffic lights are untouched — "full window" here means the app's
    /// content area, never macOS fullscreen.
    ///
    /// Nothing about the state machine moves with it. ``quickstartVisible`` is
    /// the same predicate that gated the sheet, and every alert, dialog and
    /// task modifier stays attached to the root above this branch, so consent,
    /// approvals and the launch auto-start all behave exactly as before.
    @ViewBuilder
    private var onboardingShell: some View {
        quickstartSurface
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RapidTheme.surfaceCanvas)
            // The last resort for a dismissal request that did not come from a
            // visible control. Every Step 2 micro-stage, the hero and both
            // warning screens carry a `.cancelAction` control, and AppKit
            // resolves Escape against those first — so this only ever fires on
            // the screens that have none (downloading, Ready, a failure), where
            // it reproduces exactly what the sheet's own dismissal used to do.
            .onExitCommand {
                if quickstart.retreatWithinStep2() { return }
                quickstartDismissedThisSession = true
                quickstart.skipForNow()
            }
    }

    /// The ordinary application shell: sidebar, detail pane, download
    /// strip, log drawer and status footer.
    ///
    /// Swapped out wholesale while setup is owed, rather than dimmed
    /// behind it — see ``onboardingShell``.
    @ViewBuilder
    private var productionShell: some View {
        VStack(spacing: 0) {
            // #1588: this recovery path existed since the app was introduced
            // but was never mounted, so a failed Finder replacement was
            // detected and then silently discarded.
            FailedReplaceBanner()
            NavigationSplitView {
                SidebarView(
                selection: $section,
                chat: chat,
                onNewChat: {
                    chat.newConversation()
                    section = .chat
                },
                onSearchChats: {
                    showConversationSearch = true
                },
                onSelectConversation: { id in
                    chat.selectConversation(id)
                    section = .chat
                },
                server: server
            )
            // v1.0: the rail paints an explicit warm surface rather than
            // inheriting the system sidebar material. The material is a
            // cool translucent grey that fought the warm canvas beside
            // it — the two planes read as belonging to different apps.
            .background(RapidTheme.surfaceSidebar)
            .navigationSplitViewColumnWidth(
                min: SidebarView.columnMinWidth,
                ideal: SidebarView.columnIdealWidth,
                max: SidebarView.columnMaxWidth
            )
            } detail: {
                detailArea
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Detail floor dropped 520 → 440 alongside the narrower
                // rail, back when the window floor was 640 and the old pair
                // (190 sidebar + 520 detail) over-committed it by 70pt —
                // which is what forced horizontal clipping instead of
                // graceful compression. The floor is 720 now, so this has
                // 80pt of slack; it stays at 440 because it is the detail's
                // own minimum, not a number derived from the window.
                .frame(minWidth: 440, minHeight: Self.minWindowHeight)
                    .background(RapidTheme.surfaceCanvas)
            }
            // Background pulls are process-wide, not chat-only.  Keep their
            // progress visible whichever sidebar destination is selected.
            DownloadStrip(downloads: downloads)
            if showLogs {
                LogDrawer(server: server)
                    .frame(minHeight: 100, idealHeight: 150, maxHeight: 220)
                    .accessibilityIdentifier("ContentView.LogDrawer")
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            statusFooter
        }
    }

    private var conversationSearchOverlay: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black.opacity(0.28)
                    .ignoresSafeArea()
                    .contentShape(Rectangle())
                    .onTapGesture { showConversationSearch = false }
                    .accessibilityHidden(true)

                ConversationSearchView(
                    conversations: chat.conversations,
                    now: Date(),
                    onNewChat: {
                        showConversationSearch = false
                        chat.newConversation()
                        section = .chat
                    },
                    onSelectConversation: { id in
                        showConversationSearch = false
                        chat.selectConversation(id)
                        section = .chat
                    },
                    onDismiss: { showConversationSearch = false }
                )
                .frame(
                    width: min(680, max(480, proxy.size.width - 64)),
                    height: min(560, max(400, proxy.size.height - 80))
                )
                .padding(RapidTheme.Space.xl)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .transition(.opacity)
        .zIndex(10)
    }

    // MARK: - Readiness (the one shared lifecycle value)

    /// The window's single readiness value.
    ///
    /// Resolved once per render and handed to both the chat surface and
    /// the Launch page, so "what state is the model in" has exactly one
    /// answer at any instant. See ``ModelReadiness`` for the precedence
    /// rules and the copy contract.
    private var readiness: ModelReadiness {
        ModelReadiness.resolve(
            serverState: server.readinessState(for: alias),
            alias: alias,
            cacheState: cacheState(for: alias),
            sizeText: sizeText(for: alias),
            progress: progressSnapshot,
            failure: readinessFailure,
            downloadInFlight: downloads.isDownloading(alias)
        )
    }

    /// The failure to present on the initiating surface, if any.
    ///
    /// A resident-load rejection published by ``ServerManager`` (#1838) is the
    /// freshest, most specific signal about whether the selected model can
    /// load — the engine's own rejection reason, not flattened to a generic
    /// "couldn't start". It wins over a turn-level ``chat`` failure so a
    /// banner-initiated load that the engine rejects shows its reason here on
    /// the chat surface, not only in the log pane. When no load was rejected,
    /// fall back to the chat failure exactly as before.
    private var readinessFailure: ModelReadiness.Failure? {
        if let load = server.residentLoadFailure(for: alias) {
            return ModelReadiness.Failure(message: load.message, alias: load.alias)
        }
        return chat.lastError.map {
            ModelReadiness.Failure(
                message: $0,
                kind: chat.lastFailureKind,
                // The alias the failure is ABOUT — not the one
                // currently selected. Passing `alias` here is what
                // let a failure follow the user onto whatever model
                // they picked next.
                alias: chat.lastFailureAlias
            )
        }
    }

    /// Progress is only read while the server is actually starting.
    ///
    /// This is an observation-scope decision, not a cosmetic one.
    /// ``DownloadProgress`` republishes every 500 ms, and reading it in
    /// this view's body registers ``ContentView`` — and therefore the
    /// whole split view — as an observer. Gating on ``.starting`` keeps
    /// that half-second churn confined to the window where the banner
    /// genuinely needs it, instead of paying it for the entire session.
    private var progressSnapshot: ModelReadiness.ProgressSnapshot? {
        guard case .starting = server.state else { return nil }
        return ModelReadiness.ProgressSnapshot(
            activity: server.downloadProgress.startupActivity,
            subtitle: server.downloadProgress.progressSubtitle,
            fraction: server.downloadProgress.progressFraction
        )
    }

    /// What we know about this alias' weights — see the ``resolve`` docs
    /// for why neither unknown state resolves to "download".
    private func cacheState(for alias: String) -> ModelReadiness.CacheState {
        guard !alias.isEmpty else { return .catalogPending }
        // #223's launch-time decision already established that this
        // alias needs pulling, and it lands before the catalog snapshot
        // does. Trusting it here means a cold first launch says
        // "isn't downloaded yet" immediately instead of flashing
        // "isn't running" for the second the catalog takes to load.
        if let pending = autoStartPendingDownload, pending.alias == alias {
            return .notOnDisk
        }
        // An EMPTY catalog is not evidence that an alias is unknown.
        // ``ModelCatalog.load`` returns `[]` as its failure sentinel (the
        // `rapid-mlx models` subprocess failed), and ``catalogLoaded``
        // flips to true either way. Treating that as "the catalog has
        // spoken" would tell the user their model is unrecognised on the
        // strength of a command that did not run.
        guard catalogLoaded, !catalogEntries.isEmpty else { return .catalogPending }
        guard let entry = catalogEntries.first(where: { $0.alias == alias }) else {
            // A custom alias the user typed isn't in the catalog. We
            // genuinely don't know whether it's on disk — and unlike the
            // still-loading case, nothing is coming to tell us, so the
            // banner must not claim it is already downloaded.
            return .notInCatalog
        }
        return entry.cached ? .onDisk : .notOnDisk
    }

    /// Human download size for the readiness copy. Shares
    /// ``QuickstartView``'s formatter so onboarding and the composer
    /// quote the same number in the same units for the same model.
    private func sizeText(for alias: String) -> String? {
        guard !alias.isEmpty else { return nil }
        // The launch-time auto-start decision already formatted one for
        // the alias it picked; prefer it so the two paths can't disagree
        // by a rounding step.
        if let pending = autoStartPendingDownload,
           pending.alias == alias,
           let text = pending.sizeText {
            return text
        }
        let text = QuickstartView.sizeText(for: alias)
        return text.isEmpty ? nil : text
    }

    private func refreshCatalogSnapshot() async {
        guard let binary = server.binaryPath else { return }
        let loaded = await ModelCatalogCache.shared.entries(
            binary: binary,
            generation: downloads.cacheGeneration
        )
        guard !Task.isCancelled else { return }
        catalogEntries = loaded
        catalogLoaded = true
    }

    /// Perform the readiness banner's next-step action.
    ///
    /// Lives here rather than in ``ChatView`` because starting a model is
    /// a window-level concern — the Launch page raises the same actions.
    /// Start routes through ``ServerManager.ensureServing`` so the action also
    /// replaces a different resident model (for example after using Images).
    /// ``ensureServing`` delegates its cold-start leg to ``start``, preserving
    /// the launch flags and live free-memory guard (#1435).
    private func performReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .chooseModel:
            // The model picker in the composer owns this step; the
            // banner names it but deliberately renders no second
            // control. See ``ModelReadiness.Action.isRenderable``.
            break
        case .download(let target):
            downloadModel(target)
        case .start(let target):
            startModel(target)
        case .retry(let target):
            // Clear the failure first, or the banner would stay in its
            // failed state until the next server transition lands and
            // the user would read "Couldn't start X" while X is starting.
            chat.clearStaleErrorBanner()
            startModel(target)
        case .restart(let target):
            chat.clearStaleErrorBanner()
            restartModel(target)
        case .openModelManagement:
            settingsRouter.route(.openModelManagement) {
                openWindow(id: "settings")
            }
        }
    }

    private func startModel(_ target: String) {
        let hfPath = catalogEntries.first(where: { $0.alias == target })?.hfRepo
        // ``ensureServing`` via the shared helper, NOT ``server.start``: start is
        // cold-start only and no-ops while a different model (e.g. an Images
        // checkpoint) is resident, silently dropping the switch (#1739).
        Task {
            _ = await server.ensureServing(
                alias: target,
                hfPath: hfPath,
                estimatedMemoryGB: nil,
                replacementGroup: .assistant
            )
        }
    }

    /// Fetch the weights WITHOUT loading them. The ``download`` action only
    /// promises a download; the model becomes ``needsStart`` when the bytes
    /// land and the user starts it explicitly. Same download-only path the
    /// picker's "Download in background" uses.
    private func downloadModel(_ target: String) {
        let hfPath = catalogEntries.first(where: { $0.alias == target })?.hfRepo
        _ = downloads.startDownload(alias: target, hfPath: hfPath)
    }

    private func selectChatModel(_ target: String) {
        userSelectionRevision &+= 1
        // Selecting an on-disk model is an activation request: load it and
        // release the previous assistant immediately. An uncached selection
        // still waits for the explicit Download & start action, so browsing
        // the catalog never starts a multi-GB network transfer by itself.
        let entry = catalogEntries.first(where: { $0.alias == target })
        guard Self.activatesChatModelOnSelection(
            isResident: server.isModelResident(target),
            isCached: entry?.cached == true
        ) else { return }
        let hfPath = entry?.hfRepo
        Task {
            _ = await server.ensureServing(
                alias: target,
                hfPath: hfPath,
                estimatedMemoryGB: nil,
                replacementGroup: .assistant
            )
        }
    }

    static func activatesChatModelOnSelection(
        isResident: Bool,
        isCached: Bool
    ) -> Bool {
        isResident || isCached
    }

    private func restartModel(_ target: String) {
        let hfPath = catalogEntries.first(where: { $0.alias == target })?.hfRepo
        Task {
            await server.stop()
            _ = await server.ensureServing(alias: target, hfPath: hfPath)
        }
    }

    // MARK: - Detail routing

    /// The detail pane: the chat surface, or the Launch page, per the
    /// sidebar selection.
    @ViewBuilder
    private var detailArea: some View {
        switch section {
        case .chat:
            mainArea
        case .images:
            ImagesView(viewModel: imageGen, server: server)
        case .audio:
            AudioView(viewModel: audio, server: server)
        case .launch:
            LaunchView(
                server: server,
                alias: alias,
                readiness: readiness,
                onReadinessAction: performReadinessAction
            )
        }
    }

    // MARK: - Main area

    @ViewBuilder
    private var mainArea: some View {
        switch ContentView.mainAreaBranch(for: server.state) {
        case .chat:
            ChatView(
                viewModel: chat,
                server: server,
                alias: $alias,
                readiness: readiness,
                catalogRefreshEnabled: !telemetryConsentPending,
                onUserModelSelection: selectChatModel,
                onReadinessAction: performReadinessAction,
                composerFocusRequest: composerFocusRequest
            )
        case .missing:
            missingOverlay
        }
    }

    // MARK: - Quickstart presentation

    /// The setup surface itself. Sized by its container now that the
    /// container is the window rather than a sheet.
    @ViewBuilder
    private var quickstartSurface: some View {
        QuickstartView(
            coordinator: quickstart,
            downloads: downloads,
            server: server,
            cachedModels: catalogEntries,
            catalogLoaded: catalogLoaded,
            onSkip: {
                quickstartDismissedThisSession = true
                // Skip keeps its existing meaning — no completion flag — but
                // it does retire a pending Ready confirmation, so walking
                // away is not re-asked on the next launch.
                quickstart.skipForNow()
            },
            // No `onBrowseAll` any more. It existed to lower this sheet while a
            // Settings window took over the catalogue; Browse all models is now
            // a micro-stage inside Step 2, so there is nothing to lower and
            // nothing for the parent to know about (Paper 05.2.J · S1).
            onSeedWelcome: seedQuickstartWelcome,
            onCompleted: finishOnboardingHandoff
        )
    }

    /// Steps 1 + 2 of the Start chatting transaction: make sure the chat the
    /// user is about to land in exists, and put the welcome message in it —
    /// exactly once, enforced by the coordinator that calls this.
    ///
    /// ``ChatViewModel`` always holds an open conversation, so "ensure the
    /// session exists" is a matter of making that conversation the one on
    /// screen rather than creating anything. Returning the append's own
    /// verdict (rather than an unconditional ``true``) is what lets the
    /// coordinator tell a landed welcome from one still owed.
    private func seedQuickstartWelcome() -> Bool {
        section = .chat
        // A transcript that already holds messages is not a failed seed —
        // it is somebody's conversation, and dropping an onboarding intro
        // into the middle of it would be worse than skipping the greeting.
        // Either way nothing is left owed, so this reports success and the
        // coordinator does not record a pending welcome.
        guard chat.messages.isEmpty else { return true }
        return chat.seedAssistantWelcome(quickstart.seedMessage)
    }

    /// The parent's half of the Start chatting transaction, run once, only
    /// after the coordinator confirms it performed the completion.
    ///
    /// Order matters here. The surface is already gone by this point (the
    /// coordinator moved to ``.dismissed``, so ``quickstartVisible`` is
    /// false and the ordinary shell is back — same window, same size, no
    /// second window and nothing to fade). What is left is: land on Chat,
    /// SAY that setup finished, and only then move the caret. The
    /// announcement is posted before the focus request because a VoiceOver
    /// user who hears the field described first never learns that the thing
    /// they just activated actually worked.
    private func finishOnboardingHandoff() {
        section = .chat
        VoiceOverAnnouncer.announce("Setup complete. Opening your first chat.")
        composerFocusRequest &+= 1
    }

    /// True when the Quickstart card should render in place of the chat
    /// surface — a user who hasn't completed onboarding, hasn't ever had
    /// this app serve a model, and whose server hasn't engaged a
    /// different alias. Eligibility keys on app-owned state only (see
    /// ``QuickstartCoordinator.isEligible``), never the shared HF cache.
    private var quickstartVisible: Bool {
        // Quickstart is a Chat onboarding surface. Server/model transitions
        // inside Audio or Images must never interrupt those workflows with a
        // global onboarding sheet.
        guard ContentView.quickstartCanPresent(in: section) else { return false }
        guard !quickstartDismissedThisSession else { return false }
        // Telemetry consent comes first. Both surfaces used to be able to fire
        // together (nothing referenced the other's condition) — tolerable when
        // Quickstart merely filled the detail pane behind the consent sheet,
        // but now they compete for the same presentation channel, and macOS
        // grants that to whichever asks first. Deferring here keeps the order
        // deterministic: decide on telemetry, then pick a model.
        guard !telemetryConsentPending else { return false }
        if ContentView.serverEngagedWithDifferentAlias(
            state: server.state,
            quickstartAlias: quickstart.selection.alias
        ) {
            return false
        }
        if QuickstartCoordinator.isEligible(
            done: quickstart.done,
            legacyDone: quickstart.legacyDone,
            lastServedAlias: ServerManager.lastServedAlias(),
            serverState: server.state
        ) {
            return true
        }
        // A flow that reached Ready and was never confirmed is still owed a
        // confirmation, and ``isEligible`` cannot say so: it reads
        // ``lastServedAlias``, which that flow has by definition already
        // written. Without this clause the relaunch case is unreachable —
        // the user would be dropped straight into the app having never been
        // asked, which is the automatic hand-off under another name.
        if quickstart.hasPendingReady, !quickstart.done {
            return true
        }
        // Keep the card up while the user is mid-flow (download / start /
        // waiting to confirm Ready).
        return ContentView.quickstartRetainsSurface(phase: quickstart.phase)
    }

    /// Does onboarding still own the window in this phase?
    ///
    /// Pure so the one rule that decides whether setup is on screen can be
    /// pinned exhaustively. ``.ready`` is the load-bearing case: readiness
    /// used to release the surface, and the whole point of Onboarding V3 is
    /// that it no longer does — the screen stays until the user confirms.
    /// Only the two terminal states let the shell back through.
    static func quickstartRetainsSurface(phase: QuickstartCoordinator.Phase) -> Bool {
        switch phase {
        case .downloading, .starting, .failed, .lowDiskWarning, .ready:
            return true
        case .idle, .dismissed:
            return false
        }
    }

    static func quickstartCanPresent(in section: SidebarSection) -> Bool {
        section == .chat
    }

    /// A process-wide server transition must not overwrite the Chat model
    /// selection when Audio, Images, or Video starts its own resident model.
    /// Unknown aliases remain eligible because custom text model ids are not
    /// necessarily present in the catalog snapshot.
    static func shouldSyncChatAlias(
        serving: String,
        catalogEntries: [ModelEntry],
        knownMediaAliases: Set<String> = [],
        section: SidebarSection
    ) -> Bool {
        guard section == .chat else { return false }
        guard !knownMediaAliases.contains(serving) else { return false }
        guard let entry = catalogEntries.first(where: { $0.alias == serving }) else {
            return true
        }
        return entry.kind == .chat
    }

    /// True when the Quickstart sheet is up AND owns the pending
    /// memory-warning decision (#1503) — it renders its own in-sheet copy,
    /// so this parent-anchored (and sheet-covered) alert must stand down to
    /// avoid a double presentation. Keyed on the exact predicate
    /// ``QuickstartView`` presents on, so the two surfaces can never
    /// disagree about who owns the decision.
    private var memoryWarningHandledByQuickstart: Bool {
        quickstartVisible
            && QuickstartView.memoryWarningToPresent(
                phase: quickstart.phase,
                pending: server.pendingMemoryWarning,
                selectionAlias: quickstart.selection.alias
            ) != nil
    }

    /// Alias the server is actively serving, or ``nil`` if no live state.
    private var runningAlias: String? {
        switch server.state {
        case .ready(let a), .starting(let a):
            return a
        case .idle, .stopped, .missing, .crashed:
            return nil
        }
    }

    // MARK: - First-run telemetry consent

    private var firstRunSheetPresented: Binding<Bool> {
        Binding(
            get: { telemetryConsentPending },
            set: { presented in
                // Swallow external dismiss attempts while undecided.
                _ = presented
            }
        )
    }

    @ViewBuilder
    private var firstRunSheet: some View {
        if telemetryConsentPending {
            TelemetryConsentView(onDecision: decideTelemetry)
                .interactiveDismissDisabled()
        }
    }

    private func decideTelemetry(_ enabled: Bool) {
        TelemetryConsent.record(enabled: enabled)
        telemetryConsentPending = false
        guard enabled else { return }
        Task { await TelemetrySession.sendStartIfNeeded() }
    }

    // MARK: - Missing-sidecar overlay

    /// The missing-engine recovery (Paper 05.1 state 20).
    ///
    /// Direction D's outcome composition — tinted glyph, kicker, display
    /// headline, one amber primary and a bordered Quit — rather than the
    /// previous centred card. The red tone is deliberate and is the one place
    /// setup uses it: this is not a recoverable step of setup, it is setup
    /// being unable to run at all.
    ///
    /// **Deliberate Paper deviation.** Paper draws this state inside the
    /// full-window setup shell with the rail inert. Upstream renders it in the
    /// detail pane with the live sidebar beside it, and moving it would change
    /// which surface owns the window — a presentation-ownership change, not a
    /// visual one, and out of scope for this PR. The composition matches; the
    /// containing plane does not.
    @ViewBuilder
    private var missingOverlay: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer(minLength: 24)

            OnboardingOutcomeBlock(
                glyph: "exclamationmark.octagon",
                tone: .error,
                kicker: "SETUP COULDN'T RUN",
                title: "Setup didn't finish",
                message: "Rapid-MLX isn't fully set up yet. "
                    + "Reopen Rapid-MLX to run the one-time setup again."
            ) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                    // Same two branches, same actions, same ordering as
                    // before — only the button tiers change. The recovery
                    // action (download the update, or recheck) is the amber
                    // primary; Quit steps down, since making "give up" the
                    // most prominent control on a recoverable failure was the
                    // wrong emphasis.
                    if let release = updater.availableUpdate,
                       let downloadURL = ContentView.missingOverlayDownloadURL(for: release) {
                        OnboardingActionLane {
                            Button("Download update \(release.version)") {
                                NSWorkspace.shared.open(downloadURL)
                            }
                            .buttonStyle(.onboardingPrimary)
                            .accessibilityIdentifier("MissingRuntime.DownloadUpdate")
                            Button("Recheck") { recheckEngine() }
                                .buttonStyle(.onboardingOutline)
                                .accessibilityIdentifier("MissingRuntime.Recheck")
                            Button("Quit Rapid-MLX") { NSApp.terminate(nil) }
                                .buttonStyle(.onboardingQuiet)
                                .accessibilityIdentifier("MissingRuntime.Quit")
                        }
                    } else {
                        OnboardingActionLane {
                            Button("Recheck") { recheckEngine() }
                                .buttonStyle(.onboardingPrimary)
                                .accessibilityIdentifier("MissingRuntime.Recheck")
                            Button("Quit Rapid-MLX") { NSApp.terminate(nil) }
                                .buttonStyle(.onboardingOutline)
                                .accessibilityIdentifier("MissingRuntime.Quit")
                        }
                    }

                    // Recheck's result, once there is one. Without this the
                    // control is genuinely working — a full ``ServerLocator``
                    // re-resolution — and yet reads as dead, because a still-
                    // missing engine leaves every visible thing unchanged.
                    if let status = ServerManager.recheckStatusMessage(for: server.lastBinaryRecheck) {
                        Text(status)
                            .scaledSystemFont(13)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("MissingRuntime.RecheckStatus")
                    }

                    Text("Rapid-MLX runs AI models on your Mac. Your chats stay on "
                         + "this computer — no messages are sent to the cloud.")
                        .scaledSystemFont(13)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 600, alignment: .leading)
                }
            }

            Spacer(minLength: 24).layoutPriority(-1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.top, OnboardingD.canvasTop)
        .padding(.bottom, OnboardingD.canvasBottom)
        .padding(.leading, OnboardingD.canvasLeading)
        .padding(.trailing, OnboardingD.canvasTrailing)
        .background(RapidTheme.surfaceCanvas)
    }

    /// The missing-engine screen's Recheck.
    ///
    /// The re-resolution itself is real and always was; what this adds is
    /// saying so. The announcement matters more here than almost anywhere
    /// else in the app — the sighted feedback is a single line of text that
    /// may render identically to the last one, and a VoiceOver user pressing
    /// a button that reports nothing has no way to tell it apart from a
    /// button that does nothing.
    private func recheckEngine() {
        server.refreshBinary(userInitiated: true)
        if let status = ServerManager.recheckStatusMessage(for: server.lastBinaryRecheck) {
            VoiceOverAnnouncer.announce(status)
        }
    }

    // MARK: - Status footer

    /// The footer's right-hand readouts, widest arrangement first.
    ///
    /// ``ViewThatFits`` picks the first that does not overflow, so this is an
    /// ordering of what to give up rather than a set of width breakpoints —
    /// no magic numbers to re-tune when a chip's text changes length.
    ///
    /// What may be dropped follows the rule ``settingsContentIsCompact``
    /// already states for Settings: shed readouts, never controls. The gear,
    /// the log toggle and ``DesktopVersionPill`` are buttons — the version
    /// pill routes to Settings → App — so all three survive every width and
    /// sit outside this group. Everything inside is a number you read.
    ///
    /// The ambient system probes go first: CPU, GPU and memory describe the
    /// Mac, not this app's work. Throughput goes next. ``ServerStatusPill``
    /// is last because "is a model running" is the one piece of state the
    /// rest of the footer is meaningless without — and even it is redundant
    /// in the narrowest case, where ``ReadinessBanner`` is saying the same
    /// thing in a full sentence directly above.
    @ViewBuilder
    private var footerReadouts: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                ServerStatusPill(state: server.state)
                TokensPerSecondPill(messages: { chat.messages })
                CPUPill()
                GPUPill()
                MemoryPill()
            }
            HStack(spacing: 8) {
                ServerStatusPill(state: server.state)
                TokensPerSecondPill(messages: { chat.messages })
            }
            ServerStatusPill(state: server.state)
            EmptyView()
        }
    }

    private var statusFooter: some View {
        HStack(spacing: 8) {
            SettingsGearButton()
            Button {
                withAnimation(RapidMotion.resolve(RapidMotion.standard, reduceMotion: reduceMotion)) {
                    showLogs.toggle()
                }
            } label: {
                Image(systemName: "terminal")
                    .font(.system(size: 13))
                    .foregroundStyle(showLogs ? RapidTheme.brand : Color.secondary)
            }
            .buttonStyle(.borderless)
            .help(showLogs ? "Hide logs" : "Show logs")
            .accessibilityLabel(showLogs ? "Hide logs" : "Show logs")
            .accessibilityIdentifier("ContentView.ToggleLogs")
            Spacer(minLength: 8)
            footerReadouts
            DesktopVersionPill(updater: updater)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .frame(minHeight: 30)
        .background(.bar)
    }

    // MARK: - Launch auto-start (Flow A/B)

    private func runLaunchAutoStart() async {
        // Consent is the first user-controlled surface. Do not even inspect
        // model caches behind it: the final AutoStartDecision also knows about
        // this gate, but reaching that decision requires loading the catalog
        // first (`rapid-mlx models` + `ls`). Besides doing work the user has not
        // allowed the app to begin yet, that made first-launch UI depend on the
        // operator's existing HF cache. The task is keyed on this value and
        // runs again immediately after the user answers.
        guard !telemetryConsentPending else { return }
        guard autoStartOnLaunch else {
            autoStartPendingDownload = nil
            return
        }
        guard case .idle = server.state else { return }

        _ = BundledModel.installBundledSnapshotSymlink()
        _ = QuickstartModel.installAllSnapshotSymlinks()

        let aliasAtEntry = alias
        let userSelectionRevisionAtEntry = userSelectionRevision
        var cachedAliases: Set<String> = []
        // #1706: the full catalog membership snapshot (cached AND
        // uncached entries the sidecar can serve), used to validate the
        // stored last-served alias before we resume it. `nil` would ask
        // ``decide`` to skip the membership check; we always pass a
        // concrete set when the binary is reachable so a stored alias
        // the engine can't serve (renamed/dropped/wrong-modality) falls
        // through to the picker instead of a failed serve.
        var catalogAliases: Set<String>? = nil
        if let binary = server.binaryPath {
            let entries = await ModelCatalogCache.shared.entries(
                binary: binary,
                generation: downloads.cacheGeneration
            )
            cachedAliases = Set(entries.filter { $0.cached }.map(\.alias))
            catalogAliases = Set(entries.map(\.alias))
        }
        guard case .idle = server.state else {
            autoStartPendingDownload = nil
            return
        }
        if Self.launchSelectionWasReplaced(
            aliasAtEntry: aliasAtEntry,
            currentAlias: alias,
            userSelectionChanged: userSelectionRevision != userSelectionRevisionAtEntry
        ) {
            autoStartPendingDownload = nil
            return
        }
        let hardware = MacHardware.detect()
        // A candidate is only "too big" for auto-start if ModelSizing says so
        // AND it isn't this Mac's curated recommendation. The recommendation's
        // measured footprint is trusted over ModelSizing's estimate (which
        // over-states low-bit / MoE models — e.g. bonsai-27b-2bit reads as
        // ~14.8 GB but really fits 16 GB), so a cached tier pick must not be
        // rejected here. Mirrors ModelPickerBar.handleStartTap, the switch
        // gate above, and CacheAwareDefault.bucketedFits.
        let rejectsAlias: (String) -> Bool = { candidate in
            ModelSizing.classify(ModelSizing.estimate(alias: candidate), on: hardware) == .tooBig
                && !RAMBucketedDefault.isRecommendedPick(
                    alias: candidate, physicalRAMGB: hardware.physicalRAMGB)
        }
        let decision = AutoStartDecision.decide(
            lastServedAlias: ServerManager.lastServedAlias(),
            bundledFallbackAlias: BundledModel.firstLaunchAlias(lastServedAlias: nil),
            binaryReachable: server.binaryPath != nil,
            cachedAliases: cachedAliases,
            serverState: server.state,
            catalogAliases: catalogAliases,
            rejectsAlias: rejectsAlias,
            userOptedIn: autoStartOnLaunch,
            // #1589: the two first-run surfaces get the launch before
            // auto-start does. Nothing loads behind the modal consent
            // sheet, and nothing loads for a user Quickstart still owes
            // onboarding to — auto-start restores a last-used model, it
            // does not invent a first one.
            firstRunDecisionPending: telemetryConsentPending,
            // The SAME predicate the Quickstart sheet presents on (via
            // ``QuickstartCoordinator.isEligible``), minus the server-state
            // gate that this path is about to move. Asking it here rather
            // than re-deriving "is this a new user" is the whole fix: the
            // two paths cannot disagree if there is only one answer.
            onboardingPending: QuickstartCoordinator.onboardingOwed(
                done: quickstart.done,
                legacyDone: quickstart.legacyDone,
                lastServedAlias: ServerManager.lastServedAlias()
            ),
            // Skip a retired starter only while the rescue is still on
            // offer. Once the user has completed or dismissed Quickstart,
            // `isEligible` stops showing the card — and an unconditional
            // predicate here would then leave them with neither their
            // configured auto-start nor any rescue UI, launching into an
            // idle server for no reason they can see. Respecting `done`
            // keeps the skip and the card appearing and disappearing
            // together.
            isRetiredStarter: { alias in
                !quickstart.done && QuickstartCoordinator.retiredStarters.contains(alias)
            }
        )
        switch decision {
        case .start(let resume):
            alias = resume
            autoStartPendingDownload = nil
            // Launch-time resume: if live memory makes this model unsafe right
            // now, ``start`` defers silently instead of opening with a memory
            // modal the user never asked for (issue: annoying warning on every
            // app open). The user's own Start/first-message routes through
            // ``start`` without this flag and still gets the warning.
            await server.start(alias: resume, isLaunchAutoStart: true)
        case .promptDownload(let pending):
            let footprint = ModelSizing.estimate(alias: pending)
            let sizeText: String? = footprint.paramsBillions == nil
                ? nil
                : String(format: "~%.1f GB", footprint.weightsGB)
            autoStartPendingDownload = (alias: pending, sizeText: sizeText)
            alias = pending
        case .skip:
            autoStartPendingDownload = nil
        }
    }

    /// Catalog loading may populate an initially-empty picker while the
    /// launch probe awaits subprocesses. That is initialization, not a user
    /// override. Once launch entered with a concrete selection, however, a
    /// different non-empty alias means the user took control and auto-start
    /// must stand down.
    nonisolated static func launchSelectionWasReplaced(
        aliasAtEntry: String,
        currentAlias: String,
        userSelectionChanged: Bool = false
    ) -> Bool {
        userSelectionChanged
            || (!aliasAtEntry.isEmpty && !currentAlias.isEmpty && currentAlias != aliasAtEntry)
    }

    // MARK: - Pure helpers (testable seams)

    static func missingOverlayDownloadURL(
        for availableUpdate: UpdateChecker.Release?
    ) -> URL? {
        guard let release = availableUpdate,
              let url = URL(string: release.htmlURL),
              url.scheme?.lowercased() == "https",
              url.user == nil,
              url.password == nil,
              let host = url.host?.lowercased(),
              updateReleaseHostAllowlist.contains(host) else {
            return nil
        }
        return url
    }

    static func serverEngagedWithDifferentAlias(
        state: ServerState,
        quickstartAlias: String
    ) -> Bool {
        switch state {
        case .ready(let a), .starting(let a), .crashed(let a, _):
            return a != quickstartAlias
        case .idle, .stopped, .missing:
            return false
        }
    }

    /// Which surface the detail pane shows.
    ///
    /// The branch used to carry a ``serverReady`` payload that
    /// ``ChatView`` accepted and never read — readiness was decided
    /// three other ways inside the view. ``ModelReadiness`` now owns
    /// that question in one place, so the branch is back to the single
    /// thing it actually decides: chat surface, or install overlay.
    enum MainAreaBranch: Equatable {
        case chat
        case missing
    }

    static func mainAreaBranch(for state: ServerState) -> MainAreaBranch {
        switch state {
        case .missing:
            return .missing
        case .ready, .starting, .idle, .stopped, .crashed:
            return .chat
        }
    }
}

/// Approval dialog for the ``browse`` tool: present while a request is
/// pending, deny on external dismiss (Esc / click-outside) so a suspended
/// tool can never hang waiting on a sheet the user has closed. "Always allow"
/// enables the persisted public-web auto-approval mode that can be turned off
/// again in Settings → Tools.
private struct BrowseApprovalDialog: ViewModifier {
    let store: BrowseApprovalStore

    func body(content: Content) -> some View {
        content.sheet(isPresented: Binding(
            get: { store.pendingRequest != nil },
            set: { if !$0 && store.pendingRequest != nil { store.answer(.deny) } }
        )) {
            if let req = store.pendingRequest {
                BrowseApprovalSheet(request: req, store: store)
            }
        }
    }
}

/// URL approval sheet for ``browse``. The complete URL is shown display-safe
/// (so a model can't hide the real destination behind bidi / zero-width
/// scalars past the one-line preview) and the host is called out so the user
/// judges *where* the request goes. It is only a read-only GET whose body
/// returns to the model — but the model picks the URL, so approving the exact
/// destination is what stops silent exfiltration to an attacker-controlled
/// host.
private struct BrowseApprovalSheet: View {
    let request: BrowseApprovalStore.PendingApproval
    let store: BrowseApprovalStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Fetch this web page?")
                .font(.headline)
            Text("The model wants to fetch content from:")
                .font(.callout)
                .foregroundStyle(.secondary)

            Text(BrowseApprovalStore.displaySafe(request.host))
                .font(.system(.body, design: .monospaced).weight(.semibold))
                .textSelection(.enabled)

            ScrollView {
                Text(BrowseApprovalStore.displaySafe(request.fullURL))
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .frame(minHeight: 44, maxHeight: 140)
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            Text("The page's text is sent back to the model. Only http/https "
                + "public addresses are allowed — private and local addresses are blocked.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("Always allow applies to all future public web pages. You can turn it off in Settings → Tools.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Don't allow") { store.answer(.deny) }
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("ToolApproval.Browse.Deny")
                Button("Always allow") { store.alwaysAllow() }
                    .accessibilityIdentifier("ToolApproval.Browse.AlwaysAllow")
                Button("Allow once") { store.answer(.allowOnce) }
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("ToolApproval.Browse.Allow")
            }
        }
        .padding(20)
        .frame(width: 460)
        // Deliberately NOT `.accessibilityIdentifier("…Sheet")` on this stack.
        // An accessibility modifier on a container that is not its own
        // accessibility element is applied to the elements it contains, so a
        // wrapper identifier can be stamped onto the descendants and make the
        // name ambiguous — including over the three buttons a flow actually
        // needs to press. "The approval is up" is better asserted by waiting
        // for `ToolApproval.Browse.Allow`, which is the control the user acts
        // on rather than a wrapper around it.
    }
}

/// Approval dialog for an MCP connector tool (issue #1716). Mirrors
/// ``BrowseApprovalDialog``: present while a request is pending, deny on
/// external dismiss so a suspended tool call can never hang on a sheet the
/// user has closed.
private struct MCPToolApprovalDialog: ViewModifier {
    let store: MCPToolApprovalStore

    func body(content: Content) -> some View {
        content.sheet(isPresented: Binding(
            get: { store.pendingRequest != nil },
            set: { if !$0 && store.pendingRequest != nil { store.answer(.deny) } }
        )) {
            if let req = store.pendingRequest {
                MCPToolApprovalSheet(request: req, store: store)
            }
        }
    }
}

/// Consent sheet for one connector tool call.
///
/// Shows the server, the tool, and the arguments the MODEL chose — all three
/// matter. A tool name alone ("run `read_file`?") is not a question anyone can
/// answer: which server's `read_file`, and reading what? The arguments are
/// rendered display-safe for the same reason the browse sheet renders its URL
/// that way — a model can hide the real target behind bidi or zero-width
/// scalars, and the user has to see what the engine will actually receive.
///
/// "Always allow" is scoped to THIS tool. That is the difference from the
/// browse sheet, whose equivalent flips one global mode: a blanket grant
/// across an open-ended, user-installed tool set is not something to hand out
/// as a side effect of approving one call.
private struct MCPToolApprovalSheet: View {
    let request: MCPToolApprovalStore.PendingApproval
    let store: MCPToolApprovalStore

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Run \(request.shortName)?")
                .font(.headline)
            Text("The model wants to run a tool from the “\(request.serverName)” connector.")
                .font(.callout)
                .foregroundStyle(.secondary)

            // `toolName` is the raw server-supplied name (kept raw on the
            // request for dispatch and grant keys); scrub it for display so a
            // bidi/zero-width scalar can't spoof which tool this dialog approves.
            Text(BrowseApprovalStore.displaySafe(request.toolName))
                .font(.system(.body, design: .monospaced).weight(.semibold))
                .textSelection(.enabled)

            if !request.argumentsPreview.isEmpty {
                Text("With:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    Text(request.argumentsPreview)
                        .font(.system(.callout, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
                .frame(minHeight: 44, maxHeight: 140)
                .background(Color(nsColor: .textBackgroundColor))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            Text("Connectors are programs running on this Mac. Only allow tools "
                + "from servers you set up yourself.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("Always allow applies to this tool only. You can review and "
                + "revoke it in Settings → Connectors.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Don't allow") { store.answer(.deny) }
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("ToolApproval.MCP.Deny")
                Button("Always allow") { store.answer(.alwaysAllowTool) }
                    .accessibilityIdentifier("ToolApproval.MCP.AlwaysAllow")
                Button("Allow once") { store.answer(.allowOnce) }
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("ToolApproval.MCP.Allow")
            }
        }
        .padding(20)
        .frame(width: 460)
        // No wrapper identifier — see the note on ``BrowseApprovalSheet``.
    }
}

/// v0.4.37: ChatGPT-Desktop-style gear-icon Settings button. Lives
/// in the bottom-left corner of ``statusFooter`` so the affordance
/// is exactly where users trained by ChatGPT / Claude / VS Code
/// expect to find it. Tooltip mentions the Cmd+, hotkey so power
/// users keep their muscle memory; the click itself just opens the
/// Settings scene (no deep-link override — lands on the user's last
/// selected tab).
struct SettingsGearButton: View {
    /// ``openWindow(id: "settings")``, NOT ``@Environment(\.openSettings)``:
    /// this app declares a real ``Window("Settings", id: "settings")`` and no
    /// SwiftUI ``Settings`` scene, so `OpenSettingsAction` targets a scene that
    /// does not exist and silently does nothing. See ``SettingsRouter``.
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button {
            // No router assignment: this affordance is "just open Settings",
            // so it deliberately lands on the user's last selected tab.
            openWindow(id: "settings")
        } label: {
            Image(systemName: "gearshape")
                .font(.system(size: 14))
        }
        .buttonStyle(.borderless)
        .help("Settings — ⌘,")
        .accessibilityLabel("Settings")
        .accessibilityIdentifier("ContentView.Settings")
    }
}

/// Always-visible desktop version chip in the bottom status bar.
///
/// Replaces the v0.4-era ``CLIStatusPill`` — since v0.6.6 bundled the
/// rapid-mlx engine as a sidecar, the CLI version is engine trivia
/// the user can't act on. What they CAN act on is the desktop app's
/// own version: this pill names it and surfaces the "an update is
/// available" signal in-place.
///
/// Three states, derived purely from ``UpdateChecker``:
///
///   * ``upToDate`` — green dot, "Rapid Desktop X.Y.Z · up to date".
///     ``UpdateChecker`` resolved a release whose version is exactly
///     equal to the installed version.
///   * ``updateAvailable`` — amber dot, "Rapid Desktop X.Y.Z · update
///     A.B.C available". ``availableUpdate`` is non-nil.
///   * ``unknown`` — no dot, "Rapid Desktop X.Y.Z". First check still
///     in flight, the worker briefly failed, OR the installed build
///     is strictly newer than the manifest (dev / pre-release / stale
///     manifest). We never paint red: a transient blip masquerading
///     as "broken" is worse than a calm name-only chip, and a dev
///     build ahead of the manifest is not a fault either.
///
/// Clicks deep-link to Settings → App, which is the canonical home
/// for "update Rapid-MLX Desktop". GitHub is deliberately NOT a
/// click target: the source repo is private, so a github.com nav
/// would 404 for end users.
struct DesktopVersionPill: View {
    @Bindable var updater: UpdateChecker
    /// ``openWindow(id: "settings")``, NOT ``@Environment(\.openSettings)`` —
    /// see ``SettingsGearButton`` above and ``SettingsRouter``.
    @Environment(\.openWindow) private var openWindow
    @Environment(SettingsRouter.self) private var router

    enum PillState: Equatable {
        case upToDate(version: String)
        case updateAvailable(current: String, latest: String)
        case unknown(version: String)
    }

    /// Pure derivation from ``UpdateChecker`` state to the pill
    /// state. Lifted out so the truth-table can be pinned by tests
    /// without standing up a real SwiftUI environment.
    ///
    /// "Up to date" is reserved for the case where the manifest
    /// returned a release whose version is **exactly equal** to the
    /// installed version. Two adjacent cases used to collapse into
    /// ``.upToDate`` and produced the v0.7.4 bug:
    ///
    ///   1. Dev / pre-release build whose ``currentVersion`` is
    ///      strictly NEWER than ``latest`` (e.g. unsigned local build
    ///      at 0.7.4 while ``dl.rapidmlx.com/latest.json`` still
    ///      advertises 0.6.14 because the publish script hasn't run
    ///      for the cut tag yet).
    ///   2. Stale R2 manifest after a release where the static JSON
    ///      regeneration is still propagating through CF edge.
    ///
    /// Both should read ``.unknown`` — "Rapid Desktop X · checking
    /// for updates" without a green checkmark — rather than the
    /// reassuring "up to date" that lies to the user. The previous
    /// gate (``if latest != nil``) treated those cases as proof of
    /// currency because ``availableUpdate`` is nil whenever
    /// ``currentVersion >= latest``.
    static func resolve(
        currentVersion: String,
        availableUpdate: UpdateChecker.Release?,
        latest: UpdateChecker.Release?
    ) -> PillState {
        if let upgrade = availableUpdate {
            return .updateAvailable(current: currentVersion, latest: upgrade.version)
        }
        if let latest = latest,
           !UpdateChecker.isNewer(currentVersion, than: latest.version) {
            // Manifest released ≥ our installed version AND no
            // ``availableUpdate`` means they are equal — we are
            // truly current. (If ``latest`` were strictly newer the
            // ``availableUpdate`` branch above would have fired.)
            return .upToDate(version: currentVersion)
        }
        // Either no check has landed yet, OR our installed version
        // is strictly newer than the manifest (dev / pre-release /
        // stale manifest). Show the version calmly and skip the
        // misleading "up to date" verdict.
        return .unknown(version: currentVersion)
    }

    private var state: PillState {
        DesktopVersionPill.resolve(
            currentVersion: updater.currentVersion,
            availableUpdate: updater.availableUpdate,
            latest: updater.latest
        )
    }

    var body: some View {
        Button {
            // ``route(to:open:)`` rather than an assignment followed by an
            // open: ``SettingsView`` reads the router from ``.onAppear``, so
            // the category has to land first, and the closure form makes that
            // order impossible to invert here.
            router.route(to: .app) { openWindow(id: "settings") }
        } label: {
            HStack(spacing: 5) {
                if let tint = dotTint {
                    Circle()
                        .fill(tint)
                        .frame(width: 7, height: 7)
                }
                label
            }
            .scaledSystemFont(11)
            // One line, always. Without this the label wraps under
            // compression — in a narrow window it became a five-line green
            // block — and, worse, a view that wraps reports that it fits at
            // any width, so the footer's ``ViewThatFits`` could never tell
            // that the row had run out of room.
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule().fill(capsuleTint.opacity(0.12))
            )
            .overlay(
                Capsule().strokeBorder(capsuleTint.opacity(0.30), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .help(tooltip)
        .accessibilityIdentifier("Footer.DesktopVersionPill")
    }

    @ViewBuilder
    private var label: some View {
        switch state {
        case .upToDate(let version):
            Text("Rapid-MLX \(version) · up to date")
        case .updateAvailable(let current, let latest):
            Text("Rapid-MLX \(current) · update \(latest) available")
                .fontWeight(.medium)
        case .unknown(let version):
            Text("Rapid-MLX \(version)")
        }
    }

    /// Green / amber / nil. ``nil`` means "no dot at all" — the
    /// unknown state deliberately omits the dot rather than painting
    /// it grey so a flaky network blip doesn't look like a fault.
    private var dotTint: Color? {
        switch state {
        case .upToDate:         return RapidTheme.green
        case .updateAvailable:  return RapidTheme.amber
        case .unknown:          return nil
        }
    }

    /// Capsule fill / border tint. Mirrors ``dotTint`` for the two
    /// active states; falls back to ``.secondary`` for unknown so
    /// the capsule still reads as a real affordance.
    private var capsuleTint: Color {
        dotTint ?? .secondary
    }

    private var tooltip: String {
        switch state {
        case .upToDate(let version):
            return "Rapid-MLX \(version) is the latest release. Click to open Settings → App."
        case .updateAvailable(let current, let latest):
            return "Rapid-MLX \(latest) is available (you're on \(current)). Click to install."
        case .unknown(let version):
            return "Rapid-MLX \(version). Click to open Settings → App."
        }
    }
}

/// Bottom log drawer. Reuses the same ring buffer the v0.2 ContentView
/// rendered as a hard-coded panel; here it's collapsible so the chat
/// surface owns the full window in the common case.
///
/// v0.4.28: pins the scroll position to the bottom marker on every
/// append so the freshly-arrived log line is what the user sees when
/// they open the drawer mid-session. Previously the ScrollView anchored
/// to the top of the suffix-80 window — opening the drawer during a
/// long download showed the oldest visible line first, forcing a
/// manual scroll to see the in-flight progress.
private struct LogDrawer: View {
    @Bindable var server: ServerManager

    /// Stable id for the invisible bottom marker we scroll to on every
    /// log append. Lives outside the ``ForEach`` so adding rows can't
    /// invalidate it.
    private static let bottomAnchor = "Rapid.LogDrawer.bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    if server.logLines.isEmpty {
                        Text("(no output yet)")
                            .scaledSystemFont(11, design: .monospaced)
                            .foregroundStyle(.tertiary)
                    } else {
                        ForEach(Array(server.logLines.suffix(80).enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .scaledSystemFont(11, design: .monospaced)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    // Invisible sentinel so ``ScrollViewReader`` has a
                    // stable anchor at the bottom of the content, even
                    // when the row count is < the drawer height (in
                    // which case scrolling to a row id would be a no-op).
                    Color.clear
                        .frame(height: 1)
                        .id(Self.bottomAnchor)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .onAppear {
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
            .onChange(of: server.logLines.count) { _, _ in
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
        }
    }
}
