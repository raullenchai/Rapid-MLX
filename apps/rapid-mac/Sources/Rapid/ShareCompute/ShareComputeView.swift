import SwiftUI

/// Share Compute.
///
/// The shell owns page chrome, the tab selection, and the one piece of state
/// the manager cannot hold for us: whether a finished session's receipt is
/// still on screen. Everything about the session itself —lifecycle, residency,
/// teardown — stays in ``ShareComputeManager``; this view projects it through
/// ``ShareComputeScreen`` and renders.
struct ShareComputeView: View {
    @Bindable var manager: ShareComputeManager
    @Bindable var downloads: DownloadManager
    let catalog: [ModelEntry]
    let catalogLoaded: Bool

    // MARK: Persisted choices

    @AppStorage("Rapid.shareCompute.selectedModel") private var selectedID = "qwen3.8-27b"
    @AppStorage("Rapid.shareCompute.worker")
    private var worker = Host.current().localizedName ?? "Mac"

    // MARK: Surface state

    @State private var tab: ShareComputeTab = ShareComputeTab.harnessRequested() ?? .share
    @State private var isPickerOpen = false
    @State private var freeBytes: Int64?
    @State private var showingConnectionReview = false
    @State private var providerKey = ""
    @State private var containerSize = CGSize(width: 1_240, height: 820)
    /// A lifecycle surface a visual-review harness asked to render. Always
    /// `nil` outside golden mode — see ``ShareComputeReviewStage/requested``.
    @State private var reviewStage = ShareComputeReviewStage.requested()

    private let hardware = MacHardware.detect()

    // MARK: History

    @State private var store = ShareComputeReceiptStore()
    @State private var receipts: [ShareComputeReceipt] = []
    @State private var historyPage = 0
    @State private var selectedReceiptID: String?

    // MARK: Credits (QuickSilver ledger)

    /// Owns every ledger read. Generation counter, single-flight, interval
    /// gate and the 401 key-deletion rule all live there — see
    /// ``ShareComputeLedgerCoordinator`` for why none of it can live here.
    @State private var ledger = ShareComputeLedgerCoordinator()

    // MARK: Session tracking

    /// Facts captured when a session starts, because by the time it ends the
    /// manager has already cleared its own copies.
    @State private var sessionStartedAt: Date?
    @State private var sessionModel: ShareComputeLocalModel?
    @State private var sessionNodeID: String?
    @State private var sessionReachedPool = false
    @State private var sessionReachedWarmup = false
    /// The receipt for the session that just ended, held so Share can show the
    /// completion surface until the user leaves it.
    @State private var completedReceipt: ShareComputeReceipt?

    @Environment(\.openURL) private var openURL
    /// Routes Credits' "no key yet" button to Settings → Experimental →
    /// Share Compute — the single place a QuickSilver key is entered.
    /// Optional: a preview or a host that has not injected the router still
    /// renders; the button just falls back to opening Settings untargeted.
    @Environment(SettingsRouter.self) private var settingsRouter: SettingsRouter?
    @Environment(\.openWindow) private var openWindow

    // MARK: - Derived

    private var isNarrow: Bool { containerSize.width < 880 }

    private var allPoolModels: [ShareComputeLocalModel] {
        ShareComputeLocalModel.make(catalog: catalog).filter {
            ShareComputeModel.desktopSharing.contains($0.model)
                && $0.model.fits(hardware, catalogEntry: $0.entry)
        }
    }

    /// Share lists downloaded, locally ready models only.
    private var shareableModels: [ShareComputeLocalModel] {
        ShareComputeLocalModel.readyForSharing(allPoolModels)
    }

    private var selectedModel: ShareComputeLocalModel? {
        guard !selectedID.isEmpty else { return nil }
        return shareableModels.first { $0.id == selectedID } ?? shareableModels.first
    }

    private var screen: ShareComputeScreen {
        ShareComputeScreen.make(
            state: manager.state,
            hasCompletedSession: completedReceipt != nil
        )
    }

    private var requiresProviderKey: Bool {
        _ = manager.providerKeyRevision
        guard let selectedModel else { return true }
        return manager.requiresProviderKey(for: selectedModel.model, worker: worker)
    }

    private var relayStatus: ShareComputeRelayStatusBar.Status {
        switch screen {
        case .ready: return selectedModel == nil ? .failed : .readyToConnect
        case .preparing: return .connecting
        case .online, .stopping: return .online
        case .reconnecting: return .reconnecting
        case .failed: return .failed
        case .complete: return .readyToConnect
        }
    }

    private var summary: ShareComputeContributionSummary {
        .make(from: receipts)
    }

    /// When the elapsed clock should count from. Prefers the provider's own
    /// `connected_at` so the clock measures time IN THE POOL rather than time
    /// since the button was pressed; falls back to the local start only when
    /// the provider has not published one.
    private var poolJoinedAt: Date? {
        if let connectedAt = manager.snapshot?.connectedAt, connectedAt > 0 {
            return Date(timeIntervalSince1970: connectedAt)
        }
        return sessionStartedAt
    }

    // MARK: - Body

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                    header
                    ShareComputeTabBar(selection: $tab)
                    if case .failed(let message) = screen {
                        InlineNotice(message: message, tone: .error)
                    }
                    content
                }
                .padding(isNarrow ? RapidTheme.Space.xl : 40)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(RapidTheme.surfaceCanvas)
            .onAppear { containerSize = proxy.size }
            .onChange(of: proxy.size) { _, value in containerSize = value }
        }
        .task {
            freeBytes = DiskSpaceProbe.freeBytesForHFCache()
            receipts = store.load()
            selectedReceiptID = receipts.first?.id
            reconcileSelection()
            applyReviewPresentation()
        }
        .onChange(of: manager.state) { previous, current in
            handleStateChange(from: previous, to: current)
        }
        // The node id is published by the provider on its own schedule and
        // often arrives while the phase is already `online`, so watching the
        // state alone would miss it and leave the receipt without a node.
        .onChange(of: manager.snapshot?.nodeID) { _, nodeID in
            if let nodeID, !nodeID.isEmpty { sessionNodeID = nodeID }
        }
        // The provider key is the ledger's credential too (see
        // ``ShareComputeLedgerCoordinator``'s provider-first resolution).
        // Settings → Experimental → Share Compute is a SEPARATE window, so
        // this is the only way Credits learns the key changed there while
        // it is on screen. Watching the REVISION, not ``savedProviderKeyLabel``:
        // a replacement key that redacts to the same label (same last four
        // characters, different secret) leaves the label unchanged, and this
        // view would otherwise never notice the ledger's world moved.
        .onChange(of: manager.providerKeyRevision) { _, _ in
            ledger.providerKeyChanged(isVisible: tab == .credits)
        }
        // A download finishing makes a model ready; select it so step 2
        // completes without another click.
        .onChange(of: shareableModels.map(\.id)) { _, _ in
            reconcileSelection()
        }
        .onChange(of: catalogLoaded) { _, _ in
            reconcileSelection()
            // The catalog arrives after first paint, so a review stage that
            // needs a selected model (the picker, the review sheet) has to be
            // re-applied once there is one to select.
            applyReviewPresentation()
        }
        .onChange(of: tab) { _, _ in
            isPickerOpen = false
            freeBytes = DiskSpaceProbe.freeBytesForHFCache()
        }
        .onChange(of: downloads.cacheGeneration) { _, _ in
            freeBytes = DiskSpaceProbe.freeBytesForHFCache()
        }
        .sheet(isPresented: $showingConnectionReview) {
            ShareComputeConnectionReview(
                modelTitle: selectedModel?.title ?? "",
                requiresProviderKey: requiresProviderKey,
                savedProviderKeyLabel: manager.savedProviderKeyLabel,
                worker: $worker,
                providerKey: $providerKey,
                onCancel: dismissConnectionReview,
                onConnect: startSession
            )
        }
    }

    // MARK: - Chrome

    private var header: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Share Compute")
                    .font(RapidFont.pageTitle)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Serve live requests from this Mac through QuickSilver.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Link(destination: ShareComputeDestination.provider) {
                HStack(spacing: 6) {
                    Text("Open QuickSilver")
                    Image(systemName: "arrow.up.right.square").font(.system(size: 11))
                }
                .font(RapidFont.body)
            }
            .buttonStyle(.rapidSecondaryCompact)
            .accessibilityIdentifier("ShareCompute.OpenQuickSilver")
        }
    }

    @ViewBuilder
    private var content: some View {
        switch tab {
        case .share: shareTab
        case .credits: contributionTab
        }
    }

    // MARK: - Share

    @ViewBuilder
    private var shareTab: some View {
        if let reviewStage, reviewStage.replacesShareTab {
            reviewSurface(reviewStage)
        } else {
            liveShareTab
        }
    }

    /// The reviewable lifecycle surfaces, rendered from
    /// ``ShareComputeReviewFixture``.
    ///
    /// The SAME views the live path below mounts, given the same parameter
    /// types. Only the inputs are fixtures; the rendering under review is the
    /// shipping one. `onStop` is deliberately inert — a review capture has no
    /// session to stop, and wiring it to the real manager would let a
    /// screenshot run reach into lifecycle code it has no business touching.
    @ViewBuilder
    private func reviewSurface(_ stage: ShareComputeReviewStage) -> some View {
        switch stage {
        case .preparing:
            ShareComputePreparingView(
                modelTitle: ShareComputeReviewFixture.modelTitle,
                rows: ShareComputeReviewFixture.preparationRows,
                tracking: ShareComputeReviewFixture.preparingTracking,
                canStop: true,
                isNarrow: isNarrow,
                onStop: {}
            )
        case .online:
            ShareComputeOnlineView(
                modelTitle: ShareComputeReviewFixture.modelTitle,
                worker: ShareComputeReviewFixture.worker,
                nodeID: ShareComputeReviewFixture.nodeID,
                inflight: 3,
                poolJoinedAt: ShareComputeReviewFixture.poolJoinedAt(),
                isReconnecting: false,
                payoutAccountConnected: true,
                canStop: true,
                isNarrow: isNarrow,
                onStop: {}
            )
        case .sessionComplete:
            ShareComputeSessionCompleteView(
                receipt: ShareComputeReviewFixture.receipt(),
                isNarrow: isNarrow,
                onViewReward: {},
                onShareAgain: {}
            )
        case .download:
            ShareComputeShareTab(
                models: [], candidates: [ShareComputeLocalModel(model: ShareComputeModel.supported[0], entry: nil)],
                selected: nil, relayStatus: .failed, catalogLoaded: true, isNarrow: isNarrow,
                requiresConnection: true, hasSavedKey: false, isPickerOpen: $isPickerOpen,
                freeBytes: Int64.max, downloads: downloads, onSelect: { _ in }, onStart: {},
                onOpenQuickSilver: {}, onAddKey: {}, onGetKey: {}
            )
        case .connectionReview, .modelPicker:
            // Presented OVER the ready workbench; see `reviewPresentation`.
            readyWorkbench
        }
    }

    @ViewBuilder
    private var liveShareTab: some View {
        switch screen {
        case .ready, .failed:
            readyWorkbench
        case .preparing:
            ShareComputePreparingView(
                modelTitle: activeTitle,
                rows: ShareComputePreparationPlan.rows(
                    state: manager.state,
                    isAlreadyRegistered: isAlreadyRegistered
                ),
                tracking: .make(screen: screen),
                canStop: screen.allowsStop,
                isNarrow: isNarrow,
                onStop: manager.leave
            )
        case .online, .reconnecting, .stopping:
            ShareComputeOnlineView(
                modelTitle: activeTitle,
                worker: manager.snapshot?.worker ?? worker,
                nodeID: manager.snapshot?.nodeID,
                inflight: manager.snapshot?.inflight,
                poolJoinedAt: poolJoinedAt,
                isReconnecting: screen == .reconnecting,
                payoutAccountConnected: manager.snapshot?.payoutAccount?.isEmpty == false,
                canStop: screen.allowsStop,
                isNarrow: isNarrow,
                onStop: manager.leave
            )
        case .complete:
            if let completedReceipt {
                ShareComputeSessionCompleteView(
                    receipt: completedReceipt,
                    isNarrow: isNarrow,
                    onViewReward: { openURL(ShareComputeDestination.rewards) },
                    onShareAgain: { shareAgain(catalogID: completedReceipt.catalogID) }
                )
            }
        }
    }

    private var readyWorkbench: some View {
        ShareComputeShareTab(
            models: shareableModels,
            candidates: allPoolModels.filter { !$0.isReady },
            selected: selectedModel,
            relayStatus: relayStatus,
            catalogLoaded: catalogLoaded,
            isNarrow: isNarrow,
            requiresConnection: requiresProviderKey,
            hasSavedKey: manager.providerKeyIsStored,
            savedKeyLabel: manager.savedProviderKeyLabel,
            isPickerOpen: $isPickerOpen,
            freeBytes: freeBytes,
            downloads: downloads,
            onSelect: select,
            onStart: presentConnectionReview,
            onOpenQuickSilver: { openURL(ShareComputeDestination.provider) },
            onAddKey: openShareComputeKeySettings,
            onGetKey: { openURL(ShareComputeDestination.dashboard) }
        )
    }

    private var activeTitle: String {
        manager.activeModel?.title ?? sessionModel?.title ?? selectedModel?.title ?? ""
    }

    private var isAlreadyRegistered: Bool {
        guard let model = manager.activeModel ?? sessionModel?.model else { return false }
        return manager.hasRegistration(for: model, worker: worker)
    }

    // MARK: - My Contribution

    private var contributionTab: some View {
        ShareComputeCreditsTab(
            state: ledger.state,
            receipts: receipts,
            localSummary: summary,
            savedKeyLabel: ledger.savedKeyLabel,
            canRefresh: ledger.allowsManualRefresh,
            isRefreshing: ledger.isBusy,
            refreshHoldNote: ledger.refreshHoldNote,
            isNarrow: isNarrow,
            onOpenKeySettings: openShareComputeKeySettings,
            onRefresh: { ledger.manualRefresh() }
        )
        // Loads when Credits becomes visible and STOPS when it does not: the
        // task is cancelled on tab change, `poll` returns, and its `defer`
        // cancels any walk still in flight.
        .task(id: tab) {
            guard tab == .credits else { return }
            ledger.appear()
            await ledger.poll()
        }
    }

    // MARK: - Ledger

    /// Deep-links to the ONE place a QuickSilver key is entered. Matches the
    /// `openWindow` + `SettingsRouter` pattern every other Settings deep link
    /// in the app uses — `@Environment(\.openSettings)` is a no-op here (see
    /// ``SettingsRouter``).
    private func openShareComputeKeySettings() {
        if let settingsRouter {
            settingsRouter.route(to: .experimentalFeatures) { openWindow(id: "settings") }
        } else {
            openWindow(id: "settings")
        }
    }

    // MARK: - Review presentation

    /// Opens whichever transient surface a review stage asked for.
    ///
    /// Both go through the ordinary state the user's own click would set —
    /// ``showingConnectionReview`` and ``isPickerOpen`` — so the captured
    /// sheet and popover are the real ones, with the real dismissal,
    /// focus, and Escape behaviour. Nothing here bypasses a guard: the sheet
    /// still needs a selected model, exactly as pressing the button does.
    private func applyReviewPresentation() {
        guard let reviewStage else { return }
        switch reviewStage {
        case .connectionReview:
            if selectedModel != nil { showingConnectionReview = true }
        case .modelPicker:
            isPickerOpen = true
        case .preparing, .online, .sessionComplete, .download:
            break
        }
    }

    // MARK: - Actions

    private func select(_ model: ShareComputeLocalModel) {
        // Changing the selection never starts anything, and it is refused
        // outright while a session owns the Mac.
        guard !screen.locksModelSelection else { return }
        selectedID = model.id
    }

    private func reconcileSelection() {
        selectedID = Self.reconciledSelectionID(
            current: selectedID, shareableIDs: shareableModels.map(\.id), locked: screen.locksModelSelection
        )
    }

    static func reconciledSelectionID(current: String, shareableIDs: [String], locked: Bool) -> String {
        guard !locked, !shareableIDs.contains(current) else { return current }
        return shareableIDs.first ?? current
    }

    private func presentConnectionReview() {
        guard selectedModel != nil, !screen.locksModelSelection else { return }
        guard manager.canEditProviderKey else { return }
        isPickerOpen = false
        if !requiresProviderKey {
            startSession()
        } else {
            showingConnectionReview = true
        }
    }

    private func dismissConnectionReview() {
        providerKey = ""
        showingConnectionReview = false
    }

    private func startSession() {
        guard let model = selectedModel, manager.canEditProviderKey else { return }
        let key = providerKey
        providerKey = ""
        showingConnectionReview = false
        // A new session replaces whatever receipt was on screen; the stored
        // history keeps it.
        completedReceipt = nil
        sessionStartedAt = Date()
        sessionModel = model
        sessionNodeID = nil
        sessionReachedPool = false
        sessionReachedWarmup = false
        Task {
            await manager.join(
                model: model.model,
                worker: worker,
                providerKey: key.isEmpty ? nil : key
            )
        }
    }

    private func shareAgain(catalogID: String) {
        guard manager.canEditProviderKey else { return }
        completedReceipt = nil
        guard shareableModels.contains(where: { $0.id == catalogID }) else {
            selectedID = ""
            tab = .share
            reconcileSelection()
            return
        }
        selectedID = catalogID
        tab = .share
        presentConnectionReview()
    }

    private func handleStateChange(
        from previous: ShareComputeManager.State,
        to current: ShareComputeManager.State
    ) {
        if let nodeID = manager.snapshot?.nodeID, !nodeID.isEmpty {
            sessionNodeID = nodeID
        }
        if current == .online { sessionReachedPool = true }
        if current == .warming || current == .connecting { sessionReachedWarmup = true }

        // A provider key that was rejected reopens the review rather than
        // leaving the user on a failure with no way forward.
        if case .failed(let message) = current,
           message.localizedCaseInsensitiveContains("provider key") {
            showingConnectionReview = true
        }

        guard previous.isActive, !current.isActive else { return }
        finishSession()
    }

    private func finishSession() {
        guard let startedAt = sessionStartedAt, let model = sessionModel else { return }
        sessionStartedAt = nil
        sessionModel = nil

        // A join that never got the shared model running did not contribute
        // anything, and filing it as a session would pad the history — and the
        // recognition metrics above it — with attempts. The failure is still
        // reported, by the error notice on the Share tab.
        guard sessionReachedWarmup || sessionReachedPool else { return }

        let receipt = ShareComputeReceipt(
            id: ShareComputeReceipt.makeID(),
            catalogID: model.id,
            modelTitle: model.title,
            worker: ShareComputeModel.sanitizedWorker(worker),
            nodeID: sessionNodeID,
            startedAt: startedAt,
            endedAt: Date(),
            rewardStatus: rewardStatus,
            restoreStatus: manager.lastRestoreOutcome
        )
        receipts = store.append(receipt)
        selectedReceiptID = receipt.id
        historyPage = 0
        completedReceipt = receipt
    }

    /// The only inputs are what this Mac observed. A session that reached the
    /// pool has a provider record to go and look at; one that got as far as
    /// warm-up but never online is genuinely unresolved here.
    ///
    /// ``notEligible`` is unreachable from this path today, because
    /// ``finishSession`` declines to write a receipt for a session that never
    /// started the shared model. It stays in the enum so a receipts file
    /// written by another build still decodes — dropping the case would make
    /// one unknown value discard the user's whole history.
    private var rewardStatus: ShareComputeRewardStatus {
        sessionReachedPool ? .available : .processing
    }
}
