import Foundation
import Testing
@testable import Rapid

/// Behaviour of the Signal Split redesign.
///
/// Every test here pins a rule from the implementation handoff that a future
/// change could plausibly break without failing to compile: what Share is
/// allowed to list, when the model locks, how provider phases become visible
/// steps, and — most of all — the places where Rapid must decline to state
/// something QuickSilver owns.
@Suite("Share Compute · Signal Split")
struct ShareComputeRedesignTests {

    // MARK: - Helpers

    private static func entry(
        alias: String,
        cached: Bool,
        size: String? = nil
    ) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: "org/\(alias)", sizeOnDisk: size, cached: cached)
    }

    /// The real pool catalog, so the tests exercise the aliases the provider
    /// actually accepts rather than invented ones.
    private static let pool = ShareComputeModel.supported

    private static func receipt(
        id: String,
        catalogID: String = "qwen3.8-27b",
        model: String = "Qwen3.8 27B · 4-bit",
        started: Date,
        duration: TimeInterval = 3_600,
        reward: ShareComputeRewardStatus = .available,
        restore: ShareComputeRestoreStatus = .requested,
        node: String? = "qs-node-a8c1"
    ) -> ShareComputeReceipt {
        ShareComputeReceipt(
            id: id,
            catalogID: catalogID,
            modelTitle: model,
            worker: "Lori-Mac",
            nodeID: node,
            startedAt: started,
            endedAt: started.addingTimeInterval(duration),
            rewardStatus: reward,
            restoreStatus: restore
        )
    }

    // MARK: - Model availability

    @Test("Share lists only downloaded, locally ready models")
    func shareFiltersToReadyModels() {
        let catalog = [
            Self.entry(alias: Self.pool[0].alias, cached: true, size: "16.8 GB"),
            Self.entry(alias: Self.pool[1].alias, cached: false),
            // Nemotron is absent from the catalog entirely.
        ]
        let all = ShareComputeLocalModel.make(catalog: catalog)
        // One row per SUPPORTED model, however many that is — the join is over
        // the catalog, not over whatever `rapid-mlx ls` happened to return.
        // Hard-coding 3 here is what made adding glm-5.3-flash a test failure.
        #expect(all.count == ShareComputeModel.supported.count)

        let shareable = ShareComputeLocalModel.readyForSharing(all)
        #expect(shareable.map(\.id) == [Self.pool[0].catalogID])
        #expect(shareable[0].onDiskSize == "16.8 GB")
        // A ready model never advertises a download size — there is nothing
        // to download.
        #expect(shareable[0].estimatedDownloadBytes == nil)
    }

    @Test("Alias matching ignores case so a catalog spelling change can't hide a model")
    func aliasMatchingIsCaseInsensitive() {
        let catalog = [Self.entry(alias: Self.pool[0].alias.uppercased(), cached: true, size: "16.8 GB")]
        let all = ShareComputeLocalModel.make(catalog: catalog)
        #expect(ShareComputeLocalModel.readyForSharing(all).count == 1)
    }

    @Test("A model that is not on this Mac reports a download size, never an on-disk size")
    func downloadRequiredState() throws {
        let catalog = [Self.entry(alias: Self.pool[1].alias, cached: false)]
        let local = ShareComputeLocalModel.make(catalog: catalog)
            .first { $0.id == Self.pool[1].catalogID }
        let model = try #require(local)
        #expect(!model.isReady)
        #expect(model.onDiskSize == nil)
        // ModelSizing derives a figure from the alias; the exact number is its
        // business, but a 35B model must produce SOMETHING to warn the user with.
        #expect((model.estimatedDownloadBytes ?? 0) > 0)
    }

    // MARK: - Storage eligibility

    @Test("Storage eligibility is measured, and an unreadable volume makes no claim")
    func storageEligibility() {
        let twentyGB = Int64(20) * Int64(1 << 30)

        let plenty = ShareComputeStorageEligibility.evaluate(
            downloadBytes: twentyGB,
            freeBytes: Int64(400) * Int64(1 << 30)
        )
        #expect(plenty.isSufficient)
        #expect(plenty.label?.contains("Enough space") == true)

        let tight = ShareComputeStorageEligibility.evaluate(
            downloadBytes: twentyGB,
            freeBytes: Int64(2) * Int64(1 << 30)
        )
        #expect(!tight.isSufficient)
        #expect(tight.label?.contains("Needs") == true)

        // Fail-open: a probe that returned nothing must not render as "not
        // enough space", which would block a user who has plenty.
        let unknown = ShareComputeStorageEligibility.evaluate(
            downloadBytes: twentyGB,
            freeBytes: nil
        )
        #expect(unknown == .unknown)
        #expect(unknown.label == nil)
        #expect(!unknown.isSufficient)
    }

    // MARK: - Live Pool rows

    @Test("Row order is the catalog's own and never a ranking")
    func poolRowOrderIsStable() {
        let catalog = [
            Self.entry(alias: Self.pool[0].alias, cached: false),
            Self.entry(alias: Self.pool[1].alias, cached: true, size: "20.4 GB"),
            Self.entry(alias: Self.pool[2].alias, cached: true, size: "18.9 GB"),
        ]
        let rows = ShareComputePoolRow.make(
            locals: ShareComputeLocalModel.make(catalog: catalog),
            summary: nil,
            freeBytes: Int64(400) * Int64(1 << 30)
        )
        // Every supported model, in catalog order — readiness does NOT reorder
        // them. The summary's own `models` order is not a ranking either, and
        // resorting on every 30s refresh would make the list jump.
        #expect(rows.map(\.id) == ShareComputeModel.supported.map(\.catalogID))
    }

    @Test("A model the summary never mentions shows dashes, not zeros")
    func unreportedModelIsNotZero() {
        let catalog = ShareComputeModel.supported.map {
            Self.entry(alias: $0.alias, cached: true, size: "1 GB")
        }
        let summary = ShareComputePoolSummary(
            updatedAt: Date(timeIntervalSince1970: 1_700_000_000),
            totals: .init(connectedNodes: 4, readyNodes: 3, availableSlots: 2),
            models: [
                .init(
                    modelID: "qwen3.8-27b",
                    isEnabled: true,
                    connectedNodes: 4,
                    readyNodes: 3,
                    busyNodes: 1,
                    availableSlots: 2
                )
            ]
        )
        let rows = ShareComputePoolRow.make(
            locals: ShareComputeLocalModel.make(catalog: catalog),
            summary: summary,
            freeBytes: nil
        )
        // A short summary must not shorten the UI.
        #expect(rows.count == ShareComputeModel.supported.count)

        let reported = rows.first { $0.id == "qwen3.8-27b" }!
        #expect(reported.availability.stats?.connectedNodes == 4)
        #expect(reported.availability.acceptsConnections)

        for row in rows where row.id != "qwen3.8-27b" {
            // UNKNOWN, not zero: the pool said nothing about these.
            #expect(row.availability == .unreported)
            #expect(row.availability.stats == nil)
            #expect(!row.availability.acceptsConnections)
        }
    }

    @Test("enabled:false is a distinct disabled state that refuses connection")
    func disabledModelCannotBeConnected() {
        let catalog = [Self.entry(alias: Self.pool[0].alias, cached: true, size: "16.8 GB")]
        let stats = ShareComputePoolModelStats(
            modelID: Self.pool[0].catalogID,
            isEnabled: false,
            connectedNodes: 0,
            readyNodes: 0,
            busyNodes: 0,
            availableSlots: 0
        )
        let rows = ShareComputePoolRow.make(
            locals: ShareComputeLocalModel.make(catalog: catalog),
            summary: ShareComputePoolSummary(
                updatedAt: Date(),
                totals: .init(connectedNodes: 0, readyNodes: 0, availableSlots: 0),
                models: [stats]
            ),
            freeBytes: nil
        )
        let row = rows.first { $0.id == Self.pool[0].catalogID }!
        #expect(row.isDisabledUpstream)
        #expect(row.availability == .disabled(stats))
        // Downloaded locally, but the pool is not routing to it — so no.
        #expect(row.local.isReady)
        #expect(!row.canConnect)
        // Its counters are still real and still rendered.
        #expect(row.availability.stats?.connectedNodes == 0)
    }

    @Test("A model_id Rapid has never heard of is ignored, not rendered")
    func unknownModelIDIsSafe() {
        let catalog = [Self.entry(alias: Self.pool[0].alias, cached: true, size: "16.8 GB")]
        let summary = ShareComputePoolSummary(
            updatedAt: Date(),
            totals: .init(connectedNodes: 9, readyNodes: 9, availableSlots: 9),
            models: [
                .init(
                    modelID: "some-future-model",
                    isEnabled: true,
                    connectedNodes: 9,
                    readyNodes: 9,
                    busyNodes: 0,
                    availableSlots: 9
                )
            ]
        )
        let rows = ShareComputePoolRow.make(
            locals: ShareComputeLocalModel.make(catalog: catalog),
            summary: summary,
            freeBytes: nil
        )
        // No row for a model this Mac cannot serve…
        #expect(!rows.contains { $0.id == "some-future-model" })
        // …but the summary still carries it, and the totals still count it.
        #expect(summary.stats(for: "some-future-model")?.connectedNodes == 9)
        #expect(summary.totals.connectedNodes == 9)
    }

    // MARK: - Screen projection

    @Test("Backend phases map onto the screen the user should see")
    func screenProjection() {
        func screen(
            _ state: ShareComputeManager.State,
            completed: Bool = false
        ) -> ShareComputeScreen {
            .make(state: state, hasCompletedSession: completed)
        }

        #expect(screen(.idle) == .ready)
        #expect(screen(.idle, completed: true) == .complete)
        // Every preparation phase collapses to one screen — the step rail is
        // what distinguishes them.
        for state in [
            ShareComputeManager.State.preparing, .registering, .starting, .warming, .connecting,
        ] {
            #expect(screen(state) == .preparing)
        }
        #expect(screen(.online) == .online)
        #expect(screen(.reconnecting) == .reconnecting)
        #expect(screen(.stopping) == .stopping)
        #expect(screen(.failed("boom")) == .failed("boom"))
    }

    @Test("One model at a time: selection locks for the whole active session")
    func modelSelectionLocks() {
        #expect(!ShareComputeScreen.ready.locksModelSelection)
        #expect(!ShareComputeScreen.complete.locksModelSelection)
        #expect(!ShareComputeScreen.failed("x").locksModelSelection)
        // Preparing counts: the model is already being started by then, so
        // changing it mid-flight would desync the UI from the provider.
        #expect(ShareComputeScreen.preparing.locksModelSelection)
        #expect(ShareComputeScreen.online.locksModelSelection)
        #expect(ShareComputeScreen.reconnecting.locksModelSelection)
        #expect(ShareComputeScreen.stopping.locksModelSelection)
    }

    @Test("Stop is offered exactly while stopping is a valid thing to do")
    func stopAvailability() {
        #expect(ShareComputeScreen.preparing.allowsStop)
        #expect(ShareComputeScreen.online.allowsStop)
        #expect(ShareComputeScreen.reconnecting.allowsStop)
        // Already stopping — a second Stop is a no-op the user would read as
        // a stuck button.
        #expect(!ShareComputeScreen.stopping.allowsStop)
        #expect(!ShareComputeScreen.ready.allowsStop)
        #expect(!ShareComputeScreen.complete.allowsStop)
        #expect(!ShareComputeScreen.failed("x").allowsStop)
    }

    // MARK: - Preparing steps

    @Test("Steps before the running phase are complete, after it are waiting")
    func preparationStepOrdering() {
        let rows = ShareComputePreparationPlan.rows(
            state: .starting,
            isAlreadyRegistered: false
        )
        #expect(rows.map(\.status) == [
            .complete,    // pause current model
            .complete,    // register this Mac
            .inProgress,  // start shared model
            .waiting,     // warm up
            .waiting,     // join compute pool
        ])
    }

    @Test("A Mac that is already registered says so instead of claiming it registered again")
    func alreadyRegisteredStep() {
        let early = ShareComputePreparationPlan.rows(
            state: .preparing,
            isAlreadyRegistered: true
        )
        // Settled before the sequence even reaches it — the marker is on disk.
        #expect(early[1].status == .alreadyRegistered)
        #expect(early[0].status == .inProgress)

        let during = ShareComputePreparationPlan.rows(
            state: .registering,
            isAlreadyRegistered: true
        )
        #expect(during[1].status == .alreadyRegistered)

        let fresh = ShareComputePreparationPlan.rows(
            state: .registering,
            isAlreadyRegistered: false
        )
        #expect(fresh[1].status == .inProgress)
    }

    @Test("Online settles every step; a failure marks the first unsettled one")
    func terminalStepStates() {
        let online = ShareComputePreparationPlan.rows(state: .online, isAlreadyRegistered: false)
        let unsettled = online.filter { !$0.status.isSettled }
        #expect(unsettled.isEmpty)

        let failed = ShareComputePreparationPlan.rows(state: .failed("nope"), isAlreadyRegistered: false)
        #expect(failed[0].status == .failed)
        let notWaiting = failed.dropFirst().filter { $0.status != .waiting }
        #expect(notWaiting.isEmpty)

        // A registered Mac that fails did not fail AT registration.
        let failedRegistered = ShareComputePreparationPlan.rows(
            state: .failed("nope"),
            isAlreadyRegistered: true
        )
        #expect(failedRegistered[0].status == .failed)
        #expect(failedRegistered[1].status == .alreadyRegistered)
    }

    // MARK: - Restore

    @Test("Model restore reports only what Rapid can confirm")
    func restoreOutcome() {
        // Nothing was serving before sharing started.
        #expect(ShareComputeManager.restoreOutcome(
            pendingAlias: nil, isShuttingDown: false, hasServer: true
        ) == .notNeeded)
        // The normal path requests a restart; this does not prove readiness.
        #expect(ShareComputeManager.restoreOutcome(
            pendingAlias: "qwen3.5-4b-4bit", isShuttingDown: false, hasServer: true
        ) == .requested)
        #expect(ShareComputeRestoreStatus.requested.title == "Restart requested")
        // App shutdown abandons the restore — the receipt must not claim the
        // model came back.
        #expect(ShareComputeManager.restoreOutcome(
            pendingAlias: "qwen3.5-4b-4bit", isShuttingDown: true, hasServer: true
        ) == .skipped)
        #expect(ShareComputeManager.restoreOutcome(
            pendingAlias: "qwen3.5-4b-4bit", isShuttingDown: false, hasServer: false
        ) == .skipped)
    }

    // MARK: - Credit ownership

    @Test("Receipt status describes the local session, never a settlement state")
    func rewardOwnership() {
        for status in ShareComputeRewardStatus.allCases {
            // No case promises a figure…
            #expect(!status.detailTitle.contains("$"))
            #expect(!status.tagTitle.contains("$"))
            // …and none claims a QuickSilver-side outcome Rapid cannot see.
            // There is no contributor-earnings API, so "Available" and "no
            // accepted work" were assertions about a ledger this app has never
            // read.
            for banned in ["Available", "accepted work", "Paid", "Credited"] {
                #expect(!status.tagTitle.localizedCaseInsensitiveContains(banned))
                #expect(!status.detailTitle.localizedCaseInsensitiveContains(banned))
            }
        }
        #expect(ShareComputeRewardStatus.available.tagTitle == "Ended")
        #expect(ShareComputeRewardStatus.available.hasProviderRecord)
        // A session that never reached the pool has nothing to open.
        #expect(!ShareComputeRewardStatus.notEligible.hasProviderRecord)
    }

    // MARK: - History

    @Test("History pages five rows and clamps an out-of-range page")
    func historyPagination() {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let receipts = (0..<12).map {
            Self.receipt(id: "QS-\($0)", started: base.addingTimeInterval(TimeInterval(-$0 * 3_600)))
        }
        #expect(ShareComputeHistoryPage.size == 5)
        #expect(ShareComputeHistoryPage.pageCount(total: 12) == 3)
        #expect(ShareComputeHistoryPage.slice(receipts, page: 0).count == 5)
        #expect(ShareComputeHistoryPage.slice(receipts, page: 2).count == 2)
        // Past the end clamps onto the last page rather than showing nothing.
        #expect(ShareComputeHistoryPage.slice(receipts, page: 99).count == 2)
        #expect(ShareComputeHistoryPage.slice([], page: 0).isEmpty)

        #expect(ShareComputeHistoryPage.rangeLabel(page: 0, total: 12).contains("1–5 of 12"))
        #expect(ShareComputeHistoryPage.rangeLabel(page: 2, total: 12).contains("11–12 of 12"))
    }

    @Test("Summary counts sessions, total time, and distinct models")
    func contributionSummary() {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let receipts = [
            Self.receipt(id: "a", catalogID: "qwen3.8-27b", started: base, duration: 3_600),
            Self.receipt(id: "b", catalogID: "qwen3.6-35b", started: base, duration: 1_800),
            // Same model again — still one model contributed.
            Self.receipt(id: "c", catalogID: "qwen3.8-27b", started: base, duration: 900),
        ]
        let summary = ShareComputeContributionSummary.make(from: receipts)
        #expect(summary.sessionCount == 3)
        #expect(summary.totalShared == 6_300)
        #expect(summary.modelCount == 2)
        #expect(ShareComputeContributionSummary.make(from: []) == .empty)
    }

    @Test("Receipts persist across store instances, newest first")
    func historyPersistence() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compute-receipts-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }

        let store = ShareComputeReceiptStore(fileURL: url)
        #expect(store.load().isEmpty)

        let base = Date(timeIntervalSince1970: 1_700_000_000)
        store.append(Self.receipt(id: "QS-OLD", started: base))
        let after = store.append(Self.receipt(id: "QS-NEW", started: base.addingTimeInterval(7_200)))
        #expect(after.map(\.id) == ["QS-NEW", "QS-OLD"])

        // A fresh store reads the same history — this is the durability claim
        // the My Contribution tab makes.
        let reopened = ShareComputeReceiptStore(fileURL: url)
        let loaded = reopened.load()
        #expect(loaded.map(\.id) == ["QS-NEW", "QS-OLD"])
        #expect(loaded[0].nodeID == "qs-node-a8c1")
        #expect(loaded[0].rewardStatus == .available)
        #expect(loaded[0].restoreStatus == .requested)
        #expect(loaded[1].duration == 3_600)
    }

    @Test("Appending the same receipt id replaces rather than duplicates it")
    func receiptIDsAreUnique() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compute-receipts-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let store = ShareComputeReceiptStore(fileURL: url)
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        store.append(Self.receipt(id: "QS-1", started: base, duration: 60))
        let after = store.append(Self.receipt(id: "QS-1", started: base, duration: 120))
        #expect(after.count == 1)
        #expect(after[0].duration == 120)
    }

    @Test("A corrupt or oversized receipts file degrades to empty history")
    func corruptHistoryIsSurvivable() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("share-compute-receipts-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }

        try Data("{ not json".utf8).write(to: url)
        #expect(ShareComputeReceiptStore(fileURL: url).load().isEmpty)

        // Bounded read: a pathological file must not be pulled into memory.
        try Data(
            repeating: 0x61,
            count: ShareComputeReceiptStore.maximumFileBytes + 1
        ).write(to: url)
        #expect(ShareComputeReceiptStore(fileURL: url).load().isEmpty)
    }

    @Test("Receipt ids are stable, prefixed, and short enough to read aloud")
    func receiptIdentifiers() {
        let id = ShareComputeReceipt.makeID(
            uuid: UUID(uuidString: "8A31C0DE-0000-0000-0000-000000000000")!
        )
        #expect(id == "QS-8A31")
    }

    // MARK: - Formatting

    @Test("Durations format the way each surface needs them")
    func durationFormatting() {
        #expect(ShareComputeDuration.short(6_138) == "1h 42m")
        #expect(ShareComputeDuration.short(2_820) == "47m")
        #expect(ShareComputeDuration.precise(6_138) == "1h 42m 18s")
        #expect(ShareComputeDuration.clock(6_138) == "1:42:18")
        // A total below an hour must not read as "0h 47m".
        #expect(ShareComputeDuration.total(2_820) == "47m")
        // Negative or zero intervals never produce a negative clock.
        #expect(ShareComputeDuration.clock(-5) == "0:00:00")
    }

    // MARK: - Tabs

    @Test("The value path has shorter wording for the narrow layout")
    func narrowValuePathCopy() {
        for step in ShareComputeValueStep.path {
            // Narrow copy exists, is genuinely shorter, and never empties out.
            #expect(!step.narrowTitle.isEmpty)
            #expect(!step.narrowDetail.isEmpty)
            #expect(step.narrowTitle.count <= step.title.count)
            #expect(step.narrowDetail.count < step.detail.count)
            #expect(step.title(isNarrow: true) == step.narrowTitle)
            #expect(step.title(isNarrow: false) == step.title)
            #expect(step.detail(isNarrow: true) == step.narrowDetail)
        }
    }

    // MARK: - GUI harness hooks

    @Test("Harness overrides are inert unless golden mode is on")
    func harnessHooksAreGated() {
        // A normal launch: no golden marker, so neither override is read even
        // when the names are present in the environment.
        #expect(SidebarSection.harnessRequested(environment: [:]) == nil)
        #expect(SidebarSection.harnessRequested(
            environment: ["RAPID_GUI_INITIAL_SECTION": "shareCompute"]
        ) == nil)
        #expect(ShareComputeTab.harnessRequested(
            environment: ["RAPID_GUI_SHARE_COMPUTE_TAB": "pool"]
        ) == nil)

        // Golden mode on: the named surface is honoured.
        #expect(SidebarSection.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_INITIAL_SECTION": "shareCompute",
        ]) == .shareCompute)
        #expect(ShareComputeTab.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_SHARE_COMPUTE_TAB": "credits",
        ]) == .credits)

        // An unknown name falls through to the caller's default rather than
        // trapping or landing somewhere arbitrary.
        #expect(SidebarSection.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_INITIAL_SECTION": "nope",
        ]) == nil)
        #expect(ShareComputeTab.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_SHARE_COMPUTE_TAB": "nope",
        ]) == nil)
    }

    @Test("Narrow step copy is Paper's, not an ad-hoc abbreviation")
    func narrowValuePathMatchesPaper() {
        // Transcribed from the API-Aligned narrow artboards. These are a spec,
        // and an earlier pass had quietly shortened them further to buy room
        // the collapsed rail now provides. The wording also has to stay free of
        // "accepted work" / "rewards": the pool forwards live requests and pays
        // API credits, and neither phrase describes that.
        let expected = [
            ("Connect securely", "Register and open the relay"),
            ("Serve live requests", "Answers stream back"),
            ("Receive credits", "From metered tokens"),
        ]
        #expect(ShareComputeValueStep.path.count == expected.count)
        for (step, want) in zip(ShareComputeValueStep.path, expected) {
            #expect(step.narrowTitle == want.0)
            #expect(step.narrowDetail == want.1)
        }
    }

    // MARK: - Review harness

    @Test("Review chrome suppression needs both keys and is off by default")
    func reviewChromeSuppressionIsGated() {
        // Nothing set, and the bare key on its own: production behaviour.
        #expect(!ContentView.suppressesReviewChrome(environment: [:]))
        #expect(!ContentView.suppressesReviewChrome(
            environment: ["RAPID_GUI_SUPPRESS_REVIEW_CHROME": "1"]
        ))
        // Golden mode alone does not suppress it either — the golden-flow
        // suite asserts on the update card and must keep seeing it.
        #expect(!ContentView.suppressesReviewChrome(
            environment: ["RAPID_GUI_GOLDEN_MODE": "1"]
        ))
        #expect(ContentView.suppressesReviewChrome(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_SUPPRESS_REVIEW_CHROME": "1",
        ]))
    }

    @Test("Suppression hides the update card without changing any other rule")
    func suppressionOnlyAffectsPresentation() {
        func present(suppressed: Bool) -> Bool {
            ContentView.shouldPresentUpdateCard(
                releaseVersion: "9.9.9",
                dismissedVersion: "",
                handedOffVersion: nil,
                onboardingVisible: false,
                blockingOverlayVisible: false,
                hasAction: true,
                suppressedForReview: suppressed
            )
        }
        #expect(present(suppressed: false))
        #expect(!present(suppressed: true))
        // Callers that predate the flag keep their meaning.
        #expect(ContentView.shouldPresentUpdateCard(
            releaseVersion: "9.9.9",
            dismissedVersion: "",
            handedOffVersion: nil,
            onboardingVisible: false,
            blockingOverlayVisible: false,
            hasAction: true
        ))
    }

    @Test("Lifecycle review stages are gated, and only some replace the tab")
    func reviewStagesAreGated() {
        #expect(ShareComputeReviewStage.requested(environment: [:]) == nil)
        #expect(ShareComputeReviewStage.requested(
            environment: ["RAPID_GUI_SHARE_COMPUTE_STAGE": "online"]
        ) == nil)
        #expect(ShareComputeReviewStage.requested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_SHARE_COMPUTE_STAGE": "online",
        ]) == .online)
        // An unrecognised name leaves the tab on its live path.
        #expect(ShareComputeReviewStage.requested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1",
            "RAPID_GUI_SHARE_COMPUTE_STAGE": "nope",
        ]) == nil)

        #expect(ShareComputeReviewStage.preparing.replacesShareTab)
        #expect(ShareComputeReviewStage.online.replacesShareTab)
        #expect(ShareComputeReviewStage.sessionComplete.replacesShareTab)
        // These two present OVER the ready workbench.
        #expect(!ShareComputeReviewStage.connectionReview.replacesShareTab)
        #expect(!ShareComputeReviewStage.modelPicker.replacesShareTab)
    }

    @Test("Review fixtures are projections of real state, not invented rows")
    func reviewFixturesUseRealProjections() {
        // The preparing rail must come out of the real plan, so a change to
        // the projection rules reaches the review captures.
        #expect(
            ShareComputeReviewFixture.preparationRows
                == ShareComputePreparationPlan.rows(
                    state: .starting,
                    isAlreadyRegistered: true
                )
        )
        // Mid-sequence, and honest about the step that did not run.
        let statuses = ShareComputeReviewFixture.preparationRows.map(\.status)
        #expect(statuses.contains(.alreadyRegistered))
        #expect(statuses.contains(.inProgress))
        #expect(statuses.contains(.waiting))

        let receipt = ShareComputeReviewFixture.receipt()
        #expect(receipt.nodeID == ShareComputeReviewFixture.nodeID)
        #expect(receipt.duration > 0)
        // No fixture ever claims a reward amount.
        #expect(receipt.rewardStatus == .available)
    }

    // MARK: - Responsive shell

    @Test("The rail collapses at the narrow review width and not at the wide one")
    func railBreakpoint() {
        #expect(SidebarView.compactWidth == 64)
        #expect(720 <= SidebarView.compactBreakpoint)
        #expect(1_440 > SidebarView.compactBreakpoint)
        // The collapsed rail must leave the detail more room than the full
        // column would, which is the entire reason it exists.
        #expect(SidebarView.compactWidth < SidebarView.columnMinWidth)
    }

    @Test("Each tab states the question it answers")
    func tabsHaveDistinctJobs() {
        #expect(ShareComputeTab.allCases.count == 3)
        let hints = ShareComputeTab.allCases.map(\.accessibilityHint)
        #expect(Set(hints).count == 3)
        #expect(ShareComputeTab.share.title == "Share")
        #expect(ShareComputeTab.credits.title == "Credits")
        #expect(ShareComputeTab.livePool.title == "Live Pool")
        // Paper's reading order: what to do, what it earns, what is online.
        #expect(ShareComputeTab.allCases.map(\.rawValue) == ["share", "credits", "livePool"])
    }
}

// MARK: - Pool summary data layer

/// `GET /v1/pool/summary` — decoding, HTTP handling, and the five page states.
///
/// Nothing here touches the network: every test drives the client through its
/// injected transport. A test that reached production would be flaky, would
/// spend the endpoint's per-IP rate limit, and would assert on numbers that
/// change under it.
@Suite("Share Compute · Pool summary")
struct ShareComputePoolSummaryTests {

    /// The documented example payload, verbatim, including the all-zero pool
    /// the production service currently reports.
    static let exampleJSON = """
    {
      "updated_at": "2026-09-23T15:24:18Z",
      "totals": {
        "connected_nodes": 0,
        "ready_nodes": 0,
        "available_slots": 0
      },
      "models": [
        {
          "model_id": "glm-5.3-flash",
          "enabled": true,
          "connected_nodes": 0,
          "ready_nodes": 0,
          "busy_nodes": 0,
          "available_slots": 0
        }
      ]
    }
    """

    static let populatedJSON = """
    {
      "updated_at": "2026-09-23T15:24:18Z",
      "totals": { "connected_nodes": 32, "ready_nodes": 25, "available_slots": 15 },
      "models": [
        { "model_id": "qwen3.8-27b", "enabled": true,  "connected_nodes": 18, "ready_nodes": 14, "busy_nodes": 4, "available_slots": 9 },
        { "model_id": "qwen3.6-35b", "enabled": true,  "connected_nodes": 9,  "ready_nodes": 7,  "busy_nodes": 2, "available_slots": 4 },
        { "model_id": "nemotron-3.5-lightning", "enabled": true, "connected_nodes": 5, "ready_nodes": 4, "busy_nodes": 1, "available_slots": 2 },
        { "model_id": "glm-5.3-flash", "enabled": false, "connected_nodes": 0, "ready_nodes": 0, "busy_nodes": 0, "available_slots": 0 }
      ]
    }
    """

    private static func client(
        _ body: @escaping @Sendable (URLRequest) async throws -> (Data, URLResponse)
    ) -> ShareComputePoolSummaryClient {
        ShareComputePoolSummaryClient(transport: body)
    }

    private static func ok(_ json: String) -> @Sendable (URLRequest) async throws -> (Data, URLResponse) {
        { request in
            (
                Data(json.utf8),
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            )
        }
    }

    // MARK: Decoding

    @Test("The documented example payload decodes")
    func decodesExample() async throws {
        let summary = try await Self.client(Self.ok(Self.exampleJSON)).summary()
        // `updated_at` is the SERVER's timestamp, parsed as ISO-8601 — not the
        // moment Rapid fetched it.
        #expect(
            summary.updatedAt
                == ShareComputePoolSummaryClient.parseISO8601("2026-09-23T15:24:18Z")
        )
        #expect(summary.models.count == 1)
        #expect(summary.models[0].modelID == "glm-5.3-flash")
        #expect(summary.models[0].isEnabled)
    }

    @Test("Fractional-second timestamps parse too")
    func decodesFractionalSeconds() {
        // The service emits these under load; rejecting them would turn a good
        // response into a malformedBody exactly when the pool is busiest.
        #expect(ShareComputePoolSummaryClient.parseISO8601("2026-09-23T15:24:18.317Z") != nil)
        #expect(ShareComputePoolSummaryClient.parseISO8601("not a date") == nil)
    }

    @Test("A genuinely all-zero pool is SUCCESS, not an error")
    func allZeroIsSuccess() async throws {
        let summary = try await Self.client(Self.ok(Self.exampleJSON)).summary()
        let state = ShareComputePoolSummaryState.loaded(summary)

        #expect(summary.isPoolEmpty)
        #expect(summary.totals.connectedNodes == 0)
        // The state carries data, shows no error, and offers no stale note.
        #expect(state.summary != nil)
        #expect(state.blockingError == nil)
        #expect(state.staleNote == nil)
    }

    @Test("All four known model ids decode and keep their own counters")
    func decodesAllFourModels() async throws {
        let summary = try await Self.client(Self.ok(Self.populatedJSON)).summary()
        #expect(
            summary.models.map(\.modelID) == [
                "qwen3.8-27b",
                "qwen3.6-35b",
                "nemotron-3.5-lightning",
                "glm-5.3-flash",
            ]
        )
        #expect(summary.stats(for: "qwen3.8-27b")?.busyNodes == 4)
        #expect(summary.stats(for: "nemotron-3.5-lightning")?.availableSlots == 2)
        // enabled:false is preserved, not filtered away.
        #expect(summary.stats(for: "glm-5.3-flash")?.isEnabled == false)
        #expect(summary.enabledModelIDs == ["qwen3.8-27b", "qwen3.6-35b", "nemotron-3.5-lightning"])
    }

    @Test("Every published model id has a Rapid catalog entry")
    func knownModelIDsAreAllSupported() async throws {
        let summary = try await Self.client(Self.ok(Self.populatedJSON)).summary()
        let supported = Set(ShareComputeModel.supported.map(\.catalogID))
        for stats in summary.models {
            #expect(supported.contains(stats.modelID), "unsupported id: \(stats.modelID)")
        }
        // …and GLM specifically, with its local alias and title.
        let glm = ShareComputeModel.supported.first { $0.catalogID == "glm-5.3-flash" }
        #expect(glm?.alias == "glm5.3-flash-4bit")
        #expect(glm?.title == "GLM 5.3 Flash · 4-bit")
        #expect(glm?.shortTitle == "GLM 5.3 Flash")
    }

    @Test("Totals are used as published, never re-derived from the model rows")
    func totalsAreNotRecomputed() async throws {
        // The server counts 32 connected while the models array enumerates
        // 18+9+5+0 = 32 — but also a deliberately mismatched ready count, to
        // prove the UI reads `totals` rather than summing.
        let json = """
        {
          "updated_at": "2026-09-23T15:24:18Z",
          "totals": { "connected_nodes": 40, "ready_nodes": 31, "available_slots": 17 },
          "models": [
            { "model_id": "qwen3.8-27b", "enabled": true, "connected_nodes": 1, "ready_nodes": 1, "busy_nodes": 0, "available_slots": 1 }
          ]
        }
        """
        let summary = try await Self.client(Self.ok(json)).summary()
        #expect(summary.totals.connectedNodes == 40)
        #expect(summary.totals.readyNodes == 31)
        #expect(summary.totals.availableSlots == 17)
        // A summed re-derivation would have produced 1/1/1.
        let summed = summary.models.reduce(0) { $0 + $1.connectedNodes }
        #expect(summed == 1)
        #expect(summary.totals.connectedNodes != summed)
    }

    // MARK: HTTP + transport

    @Test("A 503 with no previous data is the only case that shows an error")
    func serviceUnavailableWithoutPreviousData() async {
        let client = Self.client { request in
            (
                Data(),
                HTTPURLResponse(url: request.url!, statusCode: 503, httpVersion: nil, headerFields: nil)!
            )
        }
        do {
            _ = try await client.summary()
            Issue.record("503 must not decode as success")
        } catch let error as ShareComputePoolSummaryError {
            #expect(error == .httpStatus(503))
            let state = ShareComputePoolSummaryState.loading.failing(error)
            #expect(state == .unavailable(.httpStatus(503)))
            #expect(state.summary == nil)
            #expect(state.blockingError?.displayMessage == "Pool data unavailable")
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test("A 503 with previous data keeps every previous value and timestamp")
    func serviceUnavailableKeepsPreviousData() async throws {
        let previous = try await Self.client(Self.ok(Self.populatedJSON)).summary()
        let state = ShareComputePoolSummaryState
            .loaded(previous)
            .beginningLoad()
            .failing(.httpStatus(503))

        #expect(state == .refreshFailed(previous, .httpStatus(503)))
        // The numbers on screen are the LAST GOOD ones — not zeros, and not an
        // error page.
        #expect(state.summary?.totals.connectedNodes == 32)
        #expect(state.summary?.updatedAt == previous.updatedAt)
        #expect(state.blockingError == nil)
        #expect(state.staleNote != nil)
    }

    @Test("A refresh in flight keeps the previous data on screen")
    func refreshingPreservesPreviousData() async throws {
        let previous = try await Self.client(Self.ok(Self.populatedJSON)).summary()
        let refreshing = ShareComputePoolSummaryState.loaded(previous).beginningLoad()
        #expect(refreshing == .refreshing(previous))
        #expect(refreshing.isRefreshing)
        #expect(refreshing.summary?.totals.connectedNodes == 32)
        // The very first load has nothing to preserve.
        #expect(ShareComputePoolSummaryState.loading.beginningLoad() == .loading)
    }

    @Test("Malformed JSON is an error, and the body never reaches the message")
    func malformedBody() async {
        let junk = #"{"totals": "not an object", "secret": "qspsk-abc123"}"#
        let client = Self.client { request in
            (
                Data(junk.utf8),
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            )
        }
        do {
            _ = try await client.summary()
            Issue.record("malformed body must not decode")
        } catch let error as ShareComputePoolSummaryError {
            guard case .malformedBody(let detail) = error else {
                Issue.record("expected malformedBody, got \(error)")
                return
            }
            // Bodies are how credentials and PII end up in logs. The error
            // carries a fixed string, never the response.
            #expect(!detail.contains("qspsk-"))
            #expect(!detail.contains("not an object"))
            #expect(error.displayMessage == "Pool data unavailable")
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test("A transport failure surfaces as unreachable")
    func transportFailure() async {
        struct Boom: Error {}
        let client = Self.client { _ in throw Boom() }
        do {
            _ = try await client.summary()
            Issue.record("transport failure must propagate")
        } catch let error as ShareComputePoolSummaryError {
            guard case .unreachable = error else {
                Issue.record("expected unreachable, got \(error)")
                return
            }
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    // MARK: Request shape

    @Test("The request is a fixed HTTPS GET carrying no credential")
    func requestCarriesNoCredential() async throws {
        actor Captured {
            var request: URLRequest?
            func set(_ value: URLRequest) { request = value }
        }
        let captured = Captured()
        let client = Self.client { request in
            await captured.set(request)
            return (
                Data(Self.exampleJSON.utf8),
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            )
        }
        _ = try await client.summary()
        let request = await captured.request

        #expect(request?.url == ShareComputePoolSummaryClient.endpoint)
        #expect(request?.url?.scheme == "https")
        #expect(request?.url?.host == "pay.quicksilverpro.io")
        #expect(request?.httpMethod == "GET")
        #expect(request?.httpBody == nil)
        // The summary is public. No provider, share, or inference key may ride
        // this request — there is not even a parameter to pass one.
        let headers = request?.allHTTPHeaderFields ?? [:]
        #expect(headers["Authorization"] == nil)
        #expect(headers["X-API-Key"] == nil)
        #expect(!headers.values.contains { $0.contains("qspsk-") || $0.contains("qsppk-") })
        // A product UA is required — Cloudflare 403s library defaults.
        #expect(headers["User-Agent"]?.hasPrefix("Rapid/") == true)
        // A bounded timeout, so a hung socket cannot outlast one refresh cycle.
        #expect((request?.timeoutInterval ?? .infinity) <= 15)
    }

    // MARK: Pacing

    @Test("Refresh never runs faster than the server's cache")
    func refreshPacing() {
        // The service caches ~15s and rate-limits by IP; anything quicker buys
        // identical bytes and spends the user's budget.
        #expect(ShareComputePoolRefresh.minimumInterval == 15)
        #expect(ShareComputePoolRefresh.recommendedInterval >= ShareComputePoolRefresh.minimumInterval)
        #expect(ShareComputePoolRefresh.interval(2) == 15)
        #expect(ShareComputePoolRefresh.interval(60) == 60)
    }

    @Test("Updated-ago always describes the server timestamp")
    func updatedLabel() {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        #expect(ShareComputePoolClock.updatedLabel(base, now: base) == "Updated just now")
        #expect(
            ShareComputePoolClock.updatedLabel(base, now: base.addingTimeInterval(120))
                == "Updated 2m ago"
        )
        #expect(
            ShareComputePoolClock.updatedLabel(base, now: base.addingTimeInterval(7_200))
                == "Updated 2h ago"
        )
        // A clock skew that puts the server ahead must not render "-3s ago".
        #expect(
            ShareComputePoolClock.updatedLabel(base, now: base.addingTimeInterval(-3))
                == "Updated just now"
        )
    }
}

// MARK: - API-aligned surface rules

/// The claims the three tabs are and are not allowed to make.
///
/// These are copy and behaviour rules, not layout checks — the screenshots
/// cover layout. What they pin is the class of regression that reads fine and
/// is false: a demand ranking, a queue depth, an invented zero, or a cash
/// promise creeping back into a string.
@Suite("Share Compute · API-aligned surfaces")
struct ShareComputeAPIAlignedSurfaceTests {

    private static func entry(alias: String, cached: Bool, size: String? = nil) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: "org/\(alias)", sizeOnDisk: size, cached: cached)
    }

    /// Two cached, two not — the fixture the review screenshots use.
    private static func rows(summary: ShareComputePoolSummary?) -> [ShareComputePoolRow] {
        let cached: Set<String> = ["qwen3.8-27b-4bit", "qwen3.6-35b"]
        let catalog = ShareComputeModel.supported.map {
            entry(alias: $0.alias, cached: cached.contains($0.alias), size: "16.8 GB")
        }
        return ShareComputePoolRow.make(
            locals: ShareComputeLocalModel.make(catalog: catalog),
            summary: summary,
            freeBytes: Int64(400) * Int64(1 << 30)
        )
    }

    // MARK: Selection drives the detail and the CTA

    @Test("Selecting a model changes the action, its enablement, and what it does")
    func selectionUpdatesTheAction() throws {
        let summary = try ShareComputePoolSummaryFixture.populated.load()
        let rows = Self.rows(summary: summary)

        // Ready + enabled → connect, naming the model.
        let ready = rows.first { $0.id == "qwen3.8-27b" }!
        let connect = ShareComputePoolAction.make(for: ready)
        #expect(connect == .connect(modelTitle: "Qwen3.8 27B"))
        #expect(connect.title == "Continue with Qwen3.8 27B")
        #expect(connect.isEnabled)
        #expect(!connect.startsDownload)

        // Selecting a different, un-downloaded model changes the verb.
        let notDownloaded = rows.first { $0.id == "nemotron-3.5-lightning" }!
        let download = ShareComputePoolAction.make(for: notDownloaded)
        #expect(download == .download(modelTitle: notDownloaded.local.title))
        #expect(download.title == "Download to serve")
        #expect(download.isEnabled)
        #expect(download.startsDownload)

        // A measured low-space volume must not start a download that cannot
        // land. The selected-model card carries the exact free/needed figures.
        let constrainedRows = ShareComputePoolRow.make(
            locals: rows.map(\.local),
            summary: summary,
            freeBytes: Int64(2) * Int64(1 << 30)
        )
        let constrained = constrainedRows.first { $0.id == "nemotron-3.5-lightning" }!
        let insufficient = ShareComputePoolAction.make(for: constrained)
        #expect(insufficient == .insufficientStorage)
        #expect(insufficient.title == "Not enough storage")
        #expect(!insufficient.isEnabled)
        #expect(!insufficient.startsDownload)

        // GLM is disabled upstream in this fixture: neither verb applies, and
        // the button must not be pressable.
        let disabled = rows.first { $0.id == "glm-5.3-flash" }!
        let unavailable = ShareComputePoolAction.make(for: disabled)
        #expect(unavailable == .unavailable)
        #expect(!unavailable.isEnabled)
        #expect(!unavailable.startsDownload)

        // A missing summary cannot authorize a connect or a download.
        let unreported = Self.rows(summary: nil).first!
        #expect(ShareComputePoolAction.make(for: unreported) == .unavailable)
        #expect(!ShareComputePoolAction.make(for: unreported).isEnabled)

        // …and the three actions are genuinely distinct, which is the whole
        // point of the picker being a control rather than a legend.
        #expect(Set([connect.title, download.title, unavailable.title]).count == 3)
        #expect(ShareComputePoolAction.make(for: nil) == ShareComputePoolAction.none)
        #expect(!ShareComputePoolAction.none.isEnabled)
    }

    @Test("Share lists only models ready on this Mac")
    func shareListsOnlyReadyModels() {
        let rows = Self.rows(summary: nil)
        let shareable = ShareComputeLocalModel.readyForSharing(rows.map(\.local))
        #expect(shareable.map(\.id) == ["qwen3.8-27b", "qwen3.6-35b"])
        // GLM is ~180 GB and not downloaded, so Share must never offer it…
        #expect(!shareable.contains { $0.id == "glm-5.3-flash" })
        // …while Live Pool still lists it, because Live Pool is where a
        // download starts.
        #expect(rows.contains { $0.id == "glm-5.3-flash" })
    }

    // MARK: Copy hygiene

    /// Vocabulary the API-aligned surfaces cannot use, because no endpoint
    /// supports it. Each one was on the previous Pool tab.
    static let bannedPhrases = [
        "requests waiting",
        "highest demand",
        "queue depth",
        "demand ranking",
        "recommended model",
        "pool alias",
        "relay routing",
        "session count",
        "rank",
    ]

    @Test("No surface string claims demand, ranking, or a queue")
    func noDemandLanguage() throws {
        // Every user-visible string these surfaces can produce, gathered from
        // the types that produce them.
        var strings: [String] = []
        strings += ShareComputeTab.allCases.map(\.title)
        strings += ShareComputeTab.allCases.map(\.accessibilityHint)
        strings += ShareComputeValueStep.path.flatMap {
            [$0.title, $0.detail, $0.narrowTitle, $0.narrowDetail]
        }
        strings += ShareComputeRewardStatus.allCases.flatMap { [$0.tagTitle, $0.detailTitle] }
        strings.append(ShareComputePoolAction.make(for: Self.rows(summary: nil).first).title)
        strings += [
            ShareComputeRelayStatusBar.Status.readyToConnect.title,
            ShareComputeRelayStatusBar.Status.online.title,
            ShareComputeRelayStatusBar.Status.reconnecting.title,
        ]
        strings.append(ShareComputePoolSummaryError.httpStatus(503).displayMessage)

        for string in strings {
            for banned in Self.bannedPhrases {
                #expect(
                    !string.localizedCaseInsensitiveContains(banned),
                    "\"\(string)\" contains banned phrase \"\(banned)\""
                )
            }
        }
    }

    // MARK: Hardware neutrality

    @Test("Aggregate pool labels are hardware-neutral")
    func aggregateLabelsAreHardwareNeutral() {
        // `totals` counts every node on the pool, and the protocol makes no
        // hardware assumption — the public reference client is a generic
        // OpenAI-compatible adapter, so a Linux box with an NVIDIA GPU serving
        // through vLLM is counted here exactly like a Mac. A pool-wide label
        // saying "Macs" is therefore not a style choice, it is wrong.
        // Word-boundary match, not `contains("Mac")` — "Machines" legitimately
        // contains those three letters, so a substring check flags the very
        // wording this test exists to require.
        let macWord = try! NSRegularExpression(pattern: "\\bMacs?\\b")
        for label in ShareComputePoolLabels.allAggregateStrings {
            let range = NSRange(label.startIndex..., in: label)
            #expect(
                macWord.firstMatch(in: label, range: range) == nil,
                "aggregate label says Mac: \(label)"
            )
        }
        #expect(ShareComputePoolLabels.connectedTotal == "Connected Machines")
        #expect(ShareComputePoolLabels.machinesOnline(1) == "1 Machine online")
        #expect(ShareComputePoolLabels.machinesOnline(34) == "34 Machines online")
        // Zero is a real reading and still gets the plural form.
        #expect(ShareComputePoolLabels.machinesOnline(0) == "0 Machines online")
        #expect(ShareComputePoolLabels.noneConnected == "No machines are connected right now.")
    }

    @Test("Local wording still says this Mac")
    func localWordingStaysMacSpecific() {
        // The opposite rule: anything describing the machine Rapid is running
        // on should keep saying "this Mac", because that one genuinely is one.
        // Neutralising these too would make the local/remote boundary — the
        // thing this whole module is built around — invisible.
        let local = ShareComputeRelayStatusBar.Status.readyToConnect.title
        #expect(local.contains("this Mac"))
    }

    // MARK: Provider key entry point

    @Test("Provider keys come from the QuickSilver dashboard over HTTPS")
    func providerKeyEntryPoint() {
        let dashboard = ShareComputeDestination.dashboard
        #expect(dashboard.scheme == "https")
        #expect(dashboard.host == "quicksilverpro.io")
        // QuickSilver publishes #compute as the stable Share Compute tab.
        #expect(dashboard.absoluteString == "https://quicksilverpro.io/dashboard/#compute")
        #expect(dashboard.fragment == "compute")
        #expect(dashboard.query == nil)
        // Credits resolve to a QuickSilver-owned origin too.
        #expect(ShareComputeDestination.credits.scheme == "https")
        #expect(ShareComputeDestination.credits.host?.hasSuffix("quicksilverpro.io") == true)
    }

    // MARK: Golden fixtures

    @Test("The review fixture carries four models including a disabled GLM")
    func populatedFixtureShape() throws {
        let summary = try ShareComputePoolSummaryFixture.populated.load()
        #expect(summary.models.count == 4)
        #expect(summary.models.contains { $0.modelID == "glm-5.3-flash" })
        #expect(summary.stats(for: "glm-5.3-flash")?.isEnabled == false)
        // Totals are NOT the sum of the rows, so a capture would expose a UI
        // that secretly re-derived them.
        let summed = summary.models.reduce(0) { $0 + $1.connectedNodes }
        #expect(summary.totals.connectedNodes != summed)
        #expect(!summary.isPoolEmpty)
    }

    @Test("A separate fixture covers the honest all-zero pool")
    func emptyFixtureShape() throws {
        let summary = try ShareComputePoolSummaryFixture.empty.load()
        #expect(summary.isPoolEmpty)
        // Every supported model is present and enabled — the pool is on, just
        // unattended. This is success, not an error.
        #expect(summary.models.count == ShareComputeModel.supported.count)
        #expect(summary.enabledModelIDs.count == ShareComputeModel.supported.count)
        let state = ShareComputePoolSummaryState.loaded(summary)
        #expect(state.blockingError == nil)
        // Its rows render real zeros, not dashes.
        let rows = Self.rows(summary: summary)
        for row in rows {
            #expect(row.availability.stats?.connectedNodes == 0)
            #expect(row.availability != .unreported)
        }
    }

    @Test("Fixtures are inert without golden mode")
    func fixturesAreGated() {
        #expect(ShareComputePoolSummaryFixture.requested(environment: [:]) == nil)
        // The name alone is not enough — golden mode must be on too.
        #expect(
            ShareComputePoolSummaryFixture.requested(environment: [
                "RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY": "populated"
            ]) == nil
        )
        #expect(
            ShareComputePoolSummaryFixture.requested(environment: [
                "RAPID_GUI_GOLDEN_MODE": "1",
                "RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY": "populated",
            ]) == .populated
        )
        // An unknown name falls through to the real client rather than trapping.
        #expect(
            ShareComputePoolSummaryFixture.requested(environment: [
                "RAPID_GUI_GOLDEN_MODE": "1",
                "RAPID_GUI_SHARE_COMPUTE_POOL_SUMMARY": "nope",
            ]) == nil
        )
    }
}

// MARK: - Contributor ledger

/// `GET /v1/pool/ledger` — credential handling, decoding, pagination, error
/// classification, and the wording rules that keep a ledger window from being
/// mistaken for a local session.
///
/// No test here touches the network or the system Keychain: the client's
/// transport and the store's Keychain are both injected.
@Suite("Share Compute · Contributor ledger")
struct ShareComputeLedgerTests {

    static let key = ShareComputeReadKey(rawValue: "qsprk-" + String(repeating: "a", count: 32))

    /// The documented sample response, verbatim.
    static let sampleJSON = """
    {
      "unit": "usd_api_credit",
      "note": "Amounts are QuickSilver API credit (spendable platform balance), not a cash payout. 1.0 == $1 of API usage.",
      "caps": { "node_monthly_cap_usd": 20.0, "pool_monthly_cap_usd": 100.0 },
      "ledger": [
        {
          "node_id": "qspnode-34f07f8fade84d42",
          "model_id": "qwen3.8-27b",
          "period_start": "2026-09-24T04:00:00+00:00",
          "period_end": "2026-09-24T04:30:00+00:00",
          "request_count": 7,
          "input_tokens": 100,
          "output_tokens": 250,
          "accrued_credit": 0.0123,
          "final_credit": 0.0123,
          "status": "credited",
          "credited_at": "2026-09-24T06:00:00+00:00",
          "updated_at": "2026-09-24T06:00:00+00:00",
          "allowance_month": "2026-09-01",
          "node_monthly_cap_usd": 20.0,
          "cursor": 42
        }
      ],
      "next_cursor": null
    }
    """

    private static func client(
        _ body: @escaping @Sendable (URLRequest) async throws -> (Data, URLResponse)
    ) -> ShareComputeLedgerClient {
        ShareComputeLedgerClient(transport: body)
    }

    private static func ok(_ json: String) -> @Sendable (URLRequest) async throws -> (Data, URLResponse) {
        { request in
            (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
    }

    private static func status(_ code: Int, _ json: String = "{}") -> @Sendable (URLRequest) async throws -> (Data, URLResponse) {
        { request in
            (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!)
        }
    }

    // MARK: Credential handling

    @Test("The read key rides exactly one header and appears nowhere else")
    func authorizationHeaderIsTheOnlyPlacement() async throws {
        actor Captured {
            var request: URLRequest?
            func set(_ value: URLRequest) { request = value }
        }
        let captured = Captured()
        let client = Self.client { request in
            await captured.set(request)
            return (Data(Self.sampleJSON.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        _ = try await client.page(key: Self.key, limit: 50, cursor: 7, nodeID: "qspnode-1")
        let request = await captured.request
        let headers = request?.allHTTPHeaderFields ?? [:]

        // Exactly the documented header, exactly once.
        #expect(headers["Authorization"] == "Bearer \(Self.key.rawValue)")
        // …and nowhere else: not the URL, not the body, not another header.
        let url = request?.url?.absoluteString ?? ""
        #expect(!url.contains(Self.key.rawValue))
        #expect(!url.contains("qsprk"))
        #expect(request?.httpBody == nil)
        for (name, value) in headers where name != "Authorization" {
            #expect(!value.contains("qsprk"), "\(name) leaked the key")
        }
        #expect(request?.httpMethod == "GET")
        #expect(request?.url?.scheme == "https")
        #expect(request?.url?.host == "pay.quicksilverpro.io")
        #expect(request?.url?.path == "/v1/pool/ledger")
        #expect(headers["User-Agent"]?.hasPrefix("Rapid/") == true)
        #expect(headers["Accept"] == "application/json")
        #expect((request?.timeoutInterval ?? .infinity) <= 10)
    }

    @Test("Query parameters are encoded, and limit is clamped locally")
    func queryEncoding() async throws {
        actor Captured {
            var urls: [URL] = []
            func add(_ value: URL) { urls.append(value) }
        }
        let captured = Captured()
        let client = Self.client { request in
            await captured.add(request.url!)
            return (Data(Self.sampleJSON.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        // An out-of-range limit is clamped here rather than spent on a 400.
        _ = try await client.page(key: Self.key, limit: 5_000, cursor: 42, nodeID: "qspnode-a&b#c")
        let url = await captured.urls.first!
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems ?? []
        let byName = Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") })
        #expect(byName["limit"] == "200")
        #expect(byName["cursor"] == "42")
        // `&`/`#` stay inside the one value instead of becoming new parameters.
        #expect(byName["node_id"] == "qspnode-a&b#c")
        #expect(items.count == 3)
        #expect(url.fragment == nil)

        // limit below 1 clamps up.
        _ = try await client.page(key: Self.key, limit: 0)
        let second = await captured.urls[1]
        let secondItems = URLComponents(url: second, resolvingAgainstBaseURL: false)!.queryItems ?? []
        #expect(secondItems.first { $0.name == "limit" }?.value == "1")
        // No cursor or node filter means no stray empty parameters.
        #expect(secondItems.count == 1)
    }

    // MARK: Read-key validation

    @Test("Read-key validation accepts qsprk and refuses everything else")
    func readKeyValidation() {
        let good = "qsprk-" + String(repeating: "b", count: 30)
        #expect(try! ShareComputeReadKey.validate(good).get().rawValue == good)
        // Surrounding whitespace is trimmed; the key itself is unchanged.
        #expect(try! ShareComputeReadKey.validate("  \(good)\n").get().rawValue == good)

        func rejection(_ raw: String) -> ShareComputeReadKeyRejection? {
            if case .failure(let r) = ShareComputeReadKey.validate(raw) { return r }
            return nil
        }
        #expect(rejection("") == .empty)
        #expect(rejection("   ") == .empty)
        // The dangerous one: a provider key must never be storable.
        #expect(rejection("qsppk-" + String(repeating: "p", count: 32)) == .wrongPrefix)
        #expect(rejection("sk-live-abcdefghijklmnop") == .wrongPrefix)
        #expect(rejection("qsprk-short") == .tooShort)
        #expect(rejection("qsprk-" + String(repeating: "c", count: 400)) == .tooLong)
        // Interior control characters are rejected, not silently stripped.
        #expect(rejection("qsprk-aaaaaaaaaaaa\nbbbb") == .containsControlCharacters)
        #expect(rejection("qsprk-aaaaaaaaaaaa\tbbbb") == .containsControlCharacters)

        // Every rejection explains itself, and the prefix message names the
        // provider-key mistake it exists to catch.
        for value in [ShareComputeReadKeyRejection.empty, .wrongPrefix, .tooShort, .tooLong, .containsControlCharacters] {
            #expect(!value.message.isEmpty)
        }
        #expect(ShareComputeReadKeyRejection.wrongPrefix.message.contains("qsppk-"))
    }

    @Test("The redacted label never contains the whole key")
    func redactedLabel() {
        let label = Self.key.redactedLabel
        #expect(!label.contains(Self.key.rawValue))
        #expect(label.hasPrefix("qsprk-"))
        #expect(label.contains("…"))
        #expect(label.count < 20)
    }

    // MARK: Keychain store

    @Test("Read key saves, reads, replaces, and deletes through the Keychain")
    func keychainRoundTrip() {
        let keychain = InMemoryKeychain()
        let store = ShareComputeReadKeyStore(keychain: keychain)

        #expect(store.load() == .missing)

        let first = try! ShareComputeReadKey.validate("qsprk-" + String(repeating: "1", count: 30)).get()
        #expect(store.save(first))
        #expect(store.load() == .found(first))
        // Stored under the dedicated account and nowhere else.
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == first.rawValue)

        // Replace.
        let second = try! ShareComputeReadKey.validate("qsprk-" + String(repeating: "2", count: 30)).get()
        #expect(store.save(second))
        #expect(store.load() == .found(second))

        // Remove.
        #expect(store.remove())
        #expect(store.load() == .missing)
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == nil)
    }

    @Test("The account name cannot collide with other Rapid credentials")
    func accountNamespacing() {
        let account = ShareComputeReadKeyStore.account
        #expect(account == "rapid.quicksilver.ledger-read-key.v1")
        // Web-search providers use `rapid.web-search.*`; the embedded engine
        // uses `embedded-engine.bearer.v1`. A collision would make one
        // feature's key readable as another's.
        #expect(!account.hasPrefix("rapid.web-search."))
        #expect(account != "embedded-engine.bearer.v1")
    }

    @Test("A tombstoned or unreadable item is not mistaken for a key")
    func degradedKeychainStates() {
        let keychain = InMemoryKeychain()
        let store = ShareComputeReadKeyStore(keychain: keychain)
        // `SystemKeychain.delete` masks an unremovable item with an empty
        // string; that must read as missing, not as a corrupt key.
        keychain.write(account: ShareComputeReadKeyStore.account, secret: "")
        #expect(store.load() == .missing)
        // A value that no longer validates is corrupted, which is distinct
        // from missing so the UI can say which happened.
        keychain.write(account: ShareComputeReadKeyStore.account, secret: "not-a-key")
        #expect(store.load() == .corrupted)
    }

    @Test("No provider-key persistence regression")
    func providerKeyIsNeverPersisted() {
        let keychain = InMemoryKeychain()
        let store = ShareComputeReadKeyStore(keychain: keychain)
        // `save` takes a validated ShareComputeReadKey, and validation refuses
        // the qsppk- prefix — so there is no call that can put a provider key
        // in the Keychain.
        let provider = "qsppk-" + String(repeating: "p", count: 32)
        if case .success = ShareComputeReadKey.validate(provider) {
            Issue.record("a provider key must never validate as a read key")
        }
        #expect(store.load() == .missing)
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == nil)
        // And the connection sheet still carries the provider key in view
        // state only — it has no Keychain account of its own.
        #expect(ShareComputeReadKeyStore.account.contains("ledger-read-key"))
    }

    // MARK: Decoding

    @Test("The documented sample response decodes completely")
    func decodesSample() async throws {
        let page = try await Self.client(Self.ok(Self.sampleJSON)).page(key: Self.key)
        #expect(page.unit == "usd_api_credit")
        #expect(page.note?.contains("not a cash payout") == true)
        #expect(page.caps.nodeMonthlyUSD == 20)
        #expect(page.caps.poolMonthlyUSD == 100)
        #expect(page.nextCursor == nil)
        #expect(page.windows.count == 1)

        let window = page.windows[0]
        #expect(window.nodeID == "qspnode-34f07f8fade84d42")
        #expect(window.modelID == "qwen3.8-27b")
        #expect(window.requestCount == 7)
        #expect(window.inputTokens == 100)
        #expect(window.outputTokens == 250)
        // Decimal, not Double: 0.0123 must round-trip exactly.
        #expect(window.finalCredit == Decimal(string: "0.0123"))
        #expect(window.accruedCredit == Decimal(string: "0.0123"))
        #expect(window.status == .credited)
        #expect(window.creditedAt != nil)
        #expect(window.allowanceMonth == "2026-09-01")
        #expect(window.nodeMonthlyCapUSD == 20)
        #expect(window.cursor == 42)
        #expect(window.id == 42)
    }

    @Test("Nullable model, accrued, and credited fields decode as absent")
    func nullableFields() async throws {
        let json = """
        {
          "unit": "usd_api_credit",
          "caps": { "node_monthly_cap_usd": 20.0, "pool_monthly_cap_usd": 100.0 },
          "ledger": [
            {
              "node_id": "qspnode-1", "model_id": null,
              "period_start": "2026-09-24T04:00:00+00:00",
              "period_end": "2026-09-24T04:30:00+00:00",
              "request_count": 1, "input_tokens": 2, "output_tokens": 3,
              "accrued_credit": null, "final_credit": 0.0,
              "status": "pending", "credited_at": null,
              "updated_at": "2026-09-24T06:00:00+00:00",
              "allowance_month": "2026-09-01",
              "node_monthly_cap_usd": 20.0, "cursor": 1
            }
          ],
          "next_cursor": null
        }
        """
        let page = try await Self.client(Self.ok(json)).page(key: Self.key)
        let window = page.windows[0]
        // A deleted node/model record. `nil`, not an empty string or a guess.
        #expect(window.modelID == nil)
        // UNKNOWN, not zero — the distinction the whole panel turns on.
        #expect(window.accruedCredit == nil)
        #expect(ShareComputeCreditFormatter.credit(window.accruedCredit) == "—")
        // final_credit is non-null and may legitimately be 0.
        #expect(window.finalCredit == 0)
        #expect(ShareComputeCreditFormatter.credit(window.finalCredit) == "$0.00")
        #expect(window.creditedAt == nil)
        // `note` is optional on the wire.
        #expect(page.note == nil)
    }

    @Test("Every known status decodes, and an unknown one does not fail")
    func statusDecoding() {
        #expect(ShareComputeLedgerStatus(rawValue: "pending") == .pending)
        #expect(ShareComputeLedgerStatus(rawValue: "credited") == .credited)
        #expect(ShareComputeLedgerStatus(rawValue: "zero") == .zero)
        // The contract names `void` explicitly, and anything else must also
        // survive — a closed enum would blank the tab when the backend adds one.
        #expect(ShareComputeLedgerStatus(rawValue: "void") == .unknown("void"))
        #expect(ShareComputeLedgerStatus(rawValue: "reversed_2027") == .unknown("reversed_2027"))
        // Unknown values are shown verbatim, not mapped onto an invented status.
        #expect(ShareComputeLedgerStatus(rawValue: "void").title == "void")
        // Settlement: only credited and zero are finished.
        #expect(ShareComputeLedgerStatus.credited.isSettled)
        #expect(ShareComputeLedgerStatus.zero.isSettled)
        #expect(!ShareComputeLedgerStatus.pending.isSettled)
        #expect(!ShareComputeLedgerStatus(rawValue: "void").isSettled)
    }

    @Test("An unknown status decodes end to end without failing the page")
    func unknownStatusSurvivesDecode() async throws {
        let json = Self.sampleJSON.replacingOccurrences(of: "\"credited\"", with: "\"void\"")
        let page = try await Self.client(Self.ok(json)).page(key: Self.key)
        #expect(page.windows.count == 1)
        #expect(page.windows[0].status == .unknown("void"))
    }

    @Test("Timestamps parse with and without fractional seconds")
    func timestampParsing() async throws {
        let json = Self.sampleJSON.replacingOccurrences(
            of: "\"2026-09-24T04:00:00+00:00\"",
            with: "\"2026-09-24T04:00:00.482+00:00\""
        )
        let page = try await Self.client(Self.ok(json)).page(key: Self.key)
        #expect(page.windows[0].periodStart != nil)
        #expect(ShareComputePoolSummaryClient.parseISO8601("2026-09-24T04:00:00+00:00") != nil)
        #expect(ShareComputePoolSummaryClient.parseISO8601("2026-09-24T04:00:00.482Z") != nil)
    }

    @Test("An empty ledger is a successful 200, not an error")
    func emptyLedgerIsSuccess() async throws {
        let json = """
        {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},"ledger":[],"next_cursor":null}
        """
        let account = try await Self.client(Self.ok(json)).account(key: Self.key)
        #expect(account.isEmpty)
        #expect(account.windows.isEmpty)
        #expect(!account.isTruncated)
        // The state machine routes it to loadedEmpty — a distinct surface, not
        // a failure and not a blank loaded table.
        let state = ShareComputeLedgerState.loadedState(account)
        #expect(state == .loadedEmpty(account))
        #expect(state.blockingError == nil)
        #expect(state.account != nil)
    }

    // MARK: Pagination

    @Test("The account walk follows next_cursor to the end")
    func paginationFollowsCursor() async throws {
        actor Cursors {
            var seen: [String?] = []
            func add(_ value: String?) { seen.append(value) }
        }
        let cursors = Cursors()
        let client = Self.client { request in
            let items = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems ?? []
            let cursor = items.first { $0.name == "cursor" }?.value
            await cursors.add(cursor)
            // Two pages, then stop.
            let next = cursor == nil ? "30" : "null"
            let json = """
            {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},
             "ledger":[{"node_id":"qspnode-1","model_id":"qwen3.8-27b",
             "period_start":"2026-09-24T04:00:00+00:00","period_end":"2026-09-24T04:30:00+00:00",
             "request_count":1,"input_tokens":1,"output_tokens":1,"accrued_credit":0.5,"final_credit":0.5,
             "status":"credited","credited_at":"2026-09-24T06:00:00+00:00",
             "updated_at":"2026-09-24T06:00:00+00:00","allowance_month":"2026-09-01",
             "node_monthly_cap_usd":20.0,"cursor":\(cursor ?? "40")}],
             "next_cursor":\(next)}
            """
            return (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        let account = try await client.account(key: Self.key)
        #expect(await cursors.seen == [nil, "30"])
        #expect(account.windows.count == 2)
        #expect(!account.isTruncated)
    }

    @Test("A repeated cursor stops the walk instead of looping forever")
    func repeatedCursorGuard() async throws {
        actor Counter {
            var calls = 0
            func bump() { calls += 1 }
        }
        let counter = Counter()
        // A backend that always answers with the same next_cursor.
        let client = Self.client { request in
            await counter.bump()
            let json = """
            {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},
             "ledger":[],"next_cursor":7}
            """
            return (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        let account = try await client.account(key: Self.key)
        // First page returns cursor 7, second page returns 7 again → stop.
        #expect(await counter.calls == 2)
        #expect(account.isTruncated)
    }

    @Test("The page ceiling bounds a never-ending cursor")
    func maximumPageGuard() async throws {
        actor Counter {
            var calls = 0
            func bump() -> Int { calls += 1; return calls }
        }
        let counter = Counter()
        let client = Self.client { request in
            let n = await counter.bump()
            // A strictly increasing cursor defeats the repeat guard, so only
            // the page ceiling can stop this.
            let json = """
            {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},
             "ledger":[],"next_cursor":\(n)}
            """
            return (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        let account = try await client.account(key: Self.key)
        #expect(await counter.calls == ShareComputeLedgerClient.maximumPages)
        #expect(account.isTruncated)
    }

    @Test("The walk asks for the maximum page size")
    func walkUsesMaximumLimit() async throws {
        actor Captured {
            var limits: [String] = []
            func add(_ value: String) { limits.append(value) }
        }
        let captured = Captured()
        let client = Self.client { request in
            let items = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems ?? []
            await captured.add(items.first { $0.name == "limit" }?.value ?? "")
            return (Data(Self.sampleJSON.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        _ = try await client.account(key: Self.key)
        // Fewer round trips against a 120/hour budget.
        #expect(await captured.limits == ["200"])
        #expect(ShareComputeLedgerClient.maximumLimit == 200)
        #expect(ShareComputeLedgerClient.defaultLimit == 50)
    }

    @Test("A cancelled walk stops and propagates cancellation")
    func cancellation() async {
        let client = Self.client { request in
            let json = """
            {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},
             "ledger":[],"next_cursor":null}
            """
            return (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        let task = Task { try await client.account(key: Self.key) }
        task.cancel()
        do {
            _ = try await task.value
            // A walk that finished before the cancel landed is acceptable;
            // what must never happen is a hang or a wrong error type.
        } catch is CancellationError {
            // The expected outcome when the cancel wins the race.
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    // MARK: Error classification

    @Test("HTTP statuses classify onto distinct, actionable errors")
    func errorClassification() async {
        func error(for code: Int, body: String = "{}") async -> ShareComputeLedgerError? {
            do {
                _ = try await Self.client(Self.status(code, body)).page(key: Self.key)
                return nil
            } catch let error as ShareComputeLedgerError {
                return error
            } catch { return nil }
        }
        #expect(await error(for: 401, body: #"{"error":{"code":"invalid_key","message":"nope"}}"#) == .unauthorized)
        #expect(await error(for: 401, body: #"{"error":{"code":"missing_key"}}"#) == .unauthorized)
        #expect(await error(for: 400, body: #"{"error":{"code":"bad_cursor"}}"#) == .badRequest(code: "bad_cursor"))
        #expect(await error(for: 400, body: #"{"error":{"code":"bad_limit"}}"#) == .badRequest(code: "bad_limit"))
        #expect(await error(for: 429) == .rateLimited)
        #expect(await error(for: 503, body: #"{"error":{"code":"db_unavailable"}}"#) == .serviceUnavailable)
        // Any other 5xx is an outage too.
        #expect(await error(for: 500) == .serviceUnavailable)

        // Only a rejected key is fixed by pasting a new one.
        #expect(ShareComputeLedgerError.unauthorized.needsNewKey)
        #expect(!ShareComputeLedgerError.rateLimited.needsNewKey)
        #expect(!ShareComputeLedgerError.serviceUnavailable.needsNewKey)
    }

    @Test("No surfaced error carries a body, a credential, or a request")
    func errorsCarryNoSecrets() async {
        let body = #"{"error":{"code":"invalid_key","message":"key qsprk-aaaa was revoked"}}"#
        do {
            _ = try await Self.client(Self.status(401, body)).page(key: Self.key)
            Issue.record("401 must throw")
        } catch let error as ShareComputeLedgerError {
            #expect(!error.displayMessage.contains("qsprk"))
            #expect(!error.displayMessage.contains("Bearer"))
            #expect(!error.displayMessage.contains("pay.quicksilverpro.io"))
            // The server's message is not echoed — it may quote the credential.
            #expect(!error.displayMessage.contains("revoked was"))
        } catch {
            Issue.record("unexpected: \(error)")
        }

        // A malformed 200 is body-free too: it is still an authenticated
        // response and may contain account data.
        do {
            _ = try await Self.client(Self.ok(#"{"ledger": "not an array", "secret":"qsprk-zzz"}"#)).page(key: Self.key)
            Issue.record("malformed body must throw")
        } catch let error as ShareComputeLedgerError {
            #expect(error == .malformedBody)
            #expect(!error.displayMessage.contains("qsprk"))
        } catch {
            Issue.record("unexpected: \(error)")
        }
    }

    // MARK: Page state

    @Test("A failed refresh keeps the previous ledger on screen")
    func staleDataPreservation() {
        let account = ShareComputeLedgerFixture.account
        let state = ShareComputeLedgerState
            .loaded(account)
            .beginningLoad()
            .failing(.serviceUnavailable)

        #expect(state == .refreshFailed(account, .serviceUnavailable))
        #expect(state.account?.windows.count == account.windows.count)
        // Not an error takeover — a quiet note over real rows.
        #expect(state.blockingError == nil)
        #expect(state.staleNote != nil)

        // Refreshing preserves too.
        #expect(ShareComputeLedgerState.loaded(account).beginningLoad() == .refreshing(account))
        // The first load has nothing to preserve.
        #expect(ShareComputeLedgerState.noReadKey.beginningLoad() == .loading)
    }

    @Test("A rejected key clears the table rather than implying live rows")
    func unauthorizedDiscardsStaleRows() {
        let account = ShareComputeLedgerFixture.account
        let state = ShareComputeLedgerState.loaded(account).failing(.unauthorized)
        // Stale rows under a dead credential would read as still updating.
        #expect(state == .unauthorized)
        #expect(state.account == nil)
        #expect(state.needsReadKey)
        #expect(state.blockingError == .unauthorized)
    }

    @Test("Every modelled page state is reachable and distinct")
    func stateCoverage() {
        let account = ShareComputeLedgerFixture.account
        let empty = ShareComputeLedgerFixture.emptyAccount
        let states: [ShareComputeLedgerState] = [
            .noReadKey, .loading, .loaded(account), .loadedEmpty(empty),
            .refreshing(account), .refreshFailed(account, .serviceUnavailable),
            .unauthorized, .rateLimited, .unavailable(.serviceUnavailable),
        ]
        #expect(Set(states.map { "\($0)" }).count == states.count)
        // Only the two credential states send the user to onboarding.
        #expect(states.filter(\.needsReadKey).count == 2)
        // Only the three no-data failures take over the surface.
        #expect(states.compactMap(\.blockingError).count == 3)
        // Rate limiting is its own state, not a generic outage.
        #expect(ShareComputeLedgerState.rateLimited.blockingError == .rateLimited)
    }

    @Test("Ledger refresh is far slower than Live Pool's")
    func refreshPacing() {
        // 120 requests/hour/IP, and one account walk costs a request per page.
        #expect(ShareComputeLedgerRefresh.minimumInterval == 300)
        #expect(ShareComputeLedgerRefresh.recommendedInterval >= 300)
        #expect(ShareComputeLedgerRefresh.interval(30) == 300)
        // Emphatically not Live Pool's cadence.
        #expect(ShareComputeLedgerRefresh.minimumInterval > ShareComputePoolRefresh.recommendedInterval * 5)
    }

    // MARK: Aggregation

    @Test("Monthly totals sum every fetched row, across pages and nodes")
    func monthlyAggregation() {
        let month = "2026-09-01"
        func window(_ cursor: Int, node: String, credit: String, requests: Int, month: String) -> ShareComputeLedgerWindow {
            ShareComputeLedgerWindow(
                nodeID: node, modelID: "qwen3.8-27b",
                periodStart: Date(), periodEnd: Date(),
                requestCount: requests, inputTokens: 10, outputTokens: 20,
                accruedCredit: Decimal(string: credit), finalCredit: Decimal(string: credit)!,
                status: .credited, creditedAt: Date(), updatedAt: Date(),
                allowanceMonth: month, nodeMonthlyCapUSD: 20, cursor: cursor
            )
        }
        let account = ShareComputeLedgerAccount(
            unit: "usd_api_credit", note: nil,
            caps: ShareComputeLedgerCaps(nodeMonthlyUSD: 20, poolMonthlyUSD: 100),
            windows: [
                // Two nodes, and a row from a DIFFERENT month that must not
                // contribute — this is the bug a first-page-only total hides.
                window(1, node: "qspnode-a", credit: "0.5", requests: 3, month: month),
                window(2, node: "qspnode-b", credit: "0.25", requests: 4, month: month),
                window(3, node: "qspnode-a", credit: "0.25", requests: 5, month: month),
                window(4, node: "qspnode-a", credit: "9.99", requests: 99, month: "2026-08-01"),
            ],
            isTruncated: false
        )
        let totals = account.totals(forAllowanceMonth: month)
        #expect(totals.windowCount == 3)
        #expect(totals.requestCount == 12)
        // Exact decimal arithmetic: 0.5 + 0.25 + 0.25 == 1.0 with no drift.
        #expect(totals.finalCredit == Decimal(string: "1.0"))
        #expect(totals.accruedCredit == Decimal(string: "1.0"))
        #expect(totals.nodeIDs.sorted() == ["qspnode-a", "qspnode-b"])
        #expect(!totals.hasPendingWindows)
        // The account exposes both nodes and the newest bucket.
        #expect(account.nodeIDs.count == 2)
        #expect(account.latestAllowanceMonth == month)
    }

    @Test("An unknown accrued amount makes the month's accrued total unknown")
    func unknownAccruedPropagates() {
        let totals = ShareComputeLedgerFixture.account.totals(
            forAllowanceMonth: ShareComputeLedgerFixture.allowanceMonth
        )
        // The fixture includes one window with a nil accrued amount, so the
        // bucket has no honest accrued sum — `nil`, never a partial total
        // presented as complete.
        #expect(totals.accruedCredit == nil)
        // final_credit is always present, so the authoritative figure stands.
        #expect(totals.finalCredit > 0)
        // And the fixture has pending work, which the panel must disclose.
        #expect(totals.hasPendingWindows)
    }

    @Test("Credit values format from Decimal without binary drift")
    func decimalFormatting() {
        #expect(ShareComputeCreditFormatter.credit(Decimal(string: "0.0123")!) == "$0.0123")
        #expect(ShareComputeCreditFormatter.credit(Decimal(0)) == "$0.00")
        #expect(ShareComputeCreditFormatter.credit(Decimal(string: "20")!) == "$20.00")
        // The classic float failure: 0.1 + 0.2 must not render $0.3000000001.
        let sum = Decimal(string: "0.1")! + Decimal(string: "0.2")!
        #expect(ShareComputeCreditFormatter.credit(sum) == "$0.30")
        // An unknown amount is a dash, never a zero.
        #expect(ShareComputeCreditFormatter.credit(nil as Decimal?) == "—")
        #expect(ShareComputeCreditFormatter.unknown == "—")

        #expect(ShareComputeCreditFormatter.count(7) == "7")
        #expect(ShareComputeCreditFormatter.count(218_000) == "218k")
        #expect(ShareComputeCreditFormatter.count(1_280_000) == "1.28M")
    }

    // MARK: Wording and links

    @Test("Ledger wording never borrows session, payout, or cash language")
    func ledgerWordingIsAccurate() {
        var strings: [String] = [
            ShareComputeLedgerStatus.pending.title,
            ShareComputeLedgerStatus.credited.title,
            ShareComputeLedgerStatus.zero.title,
        ]
        strings += ShareComputeLedgerError.allDisplayMessages
        for string in strings {
            for banned in ["payout", "cash", "session", "job", "accepted work"] {
                #expect(
                    !string.localizedCaseInsensitiveContains(banned),
                    "\"\(string)\" uses \(banned)"
                )
            }
        }
        // The unit note QuickSilver sends says it outright, and Rapid keeps it.
        #expect(ShareComputeLedgerFixture.account.note?.contains("not a cash payout") == true)
        #expect(ShareComputeLedgerFixture.account.unit == "usd_api_credit")
    }

    @Test("Dashboard deep links point at the documented hashes")
    func dashboardDeepLinks() {
        #expect(
            ShareComputeDestination.readKeyManagement.absoluteString
                == "https://quicksilverpro.io/dashboard/#compute"
        )
        #expect(
            ShareComputeDestination.credits.absoluteString
                == "https://quicksilverpro.io/dashboard#credits"
        )
        // Both HTTPS, both QuickSilver.
        for url in [ShareComputeDestination.readKeyManagement, ShareComputeDestination.credits] {
            #expect(url.scheme == "https")
            #expect(url.host == "quicksilverpro.io")
        }
        // They are different pages: #compute manages read keys, #credits is
        // account balance. Rapid renders the ledger itself, so #credits must
        // never be labelled "ledger details".
        #expect(ShareComputeDestination.readKeyManagement != ShareComputeDestination.credits)
    }

    // MARK: Fixtures

    @Test("Ledger fixtures are gated and carry no credential")
    func fixturesAreGatedAndSecretFree() {
        #expect(ShareComputeLedgerFixture.requested(environment: [:]) == nil)
        #expect(
            ShareComputeLedgerFixture.requested(environment: [
                "RAPID_GUI_SHARE_COMPUTE_LEDGER": "loaded"
            ]) == nil
        )
        #expect(
            ShareComputeLedgerFixture.requested(environment: [
                "RAPID_GUI_GOLDEN_MODE": "1",
                "RAPID_GUI_SHARE_COMPUTE_LEDGER": "loaded",
            ]) == .loaded
        )
        #expect(
            ShareComputeLedgerFixture.requested(environment: [
                "RAPID_GUI_GOLDEN_MODE": "1",
                "RAPID_GUI_SHARE_COMPUTE_LEDGER": "nope",
            ]) == nil
        )
        // The label shown beside a "loaded" capture is redacted, and no fixture
        // holds anything key-shaped.
        #expect(ShareComputeLedgerFixture.keyLabel.contains("…"))
        #expect(ShareComputeLedgerFixture.keyLabel.count < 20)
    }

    @Test("The loaded fixture exercises every row variant")
    func loadedFixtureCoversRowVariants() {
        let windows = ShareComputeLedgerFixture.account.windows
        #expect(windows.contains { $0.status == .pending })
        #expect(windows.contains { $0.status == .credited })
        #expect(windows.contains { $0.status == .zero })
        #expect(windows.contains { $0.status == .unknown("void") })
        // A deleted model record and an unknown accrued amount.
        #expect(windows.contains { $0.modelID == nil })
        #expect(windows.contains { $0.accruedCredit == nil })
        // More than one node, so the account-wide wording is exercised.
        #expect(ShareComputeLedgerFixture.account.nodeIDs.count >= 2)
        // Each state maps to the surface it names.
        #expect(ShareComputeLedgerFixture.noReadKey.state == .noReadKey)
        #expect(ShareComputeLedgerFixture.revoked.state == .unauthorized)
        #expect(ShareComputeLedgerFixture.empty.state.account?.isEmpty == true)
        #expect(ShareComputeLedgerFixture.stale.state.staleNote != nil)
        // Onboarding states show no saved-key label.
        #expect(ShareComputeLedgerFixture.noReadKey.savedKeyLabel == nil)
        #expect(ShareComputeLedgerFixture.revoked.savedKeyLabel == nil)
        #expect(ShareComputeLedgerFixture.loaded.savedKeyLabel != nil)
    }
}

extension ShareComputeLedgerError {
    /// Every user-facing message, for the copy-hygiene test.
    static var allDisplayMessages: [String] {
        [
            ShareComputeLedgerError.noReadKey,
            .unauthorized,
            .badRequest(code: "bad_cursor"),
            .rateLimited,
            .serviceUnavailable,
            .unreachable("x"),
            .malformedBody,
        ].map(\.displayMessage)
    }
}

// MARK: - Ledger races, redirects, and rate limiting

/// The pre-launch correctness fixes: a request in flight must never be able to
/// outlive the credential it was issued under, the credential must never be
/// replayed to a redirect target, and Refresh must actually be limited.
@Suite("Share Compute · Ledger safety")
@MainActor
struct ShareComputeLedgerSafetyTests {

    private struct FailingKeychain: KeychainStoring {
        let secret: String?

        func read(account: String) -> String? { secret }
        func write(account: String, secret: String) -> Bool { false }
        func delete(account: String) -> Bool { false }
    }

    // `nonisolated` so the closures the async tests hand to background work
    // can read them without hopping to the main actor.
    nonisolated static let keyA = try! ShareComputeReadKey
        .validate("qsprk-" + String(repeating: "a", count: 30)).get()
    nonisolated static let keyB = try! ShareComputeReadKey
        .validate("qsprk-" + String(repeating: "b", count: 30)).get()

    @Test("A failed Keychain save keeps the draft and an actionable entry state")
    func failedKeychainSave() {
        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: FailingKeychain(secret: nil)),
            fixture: nil
        )
        #expect(coordinator.save(draft: Self.keyA.rawValue) == .keychainUnavailable)
        #expect(coordinator.state == .unavailable(.keychainUnavailable))
        #expect(coordinator.state.needsReadKey)
        #expect(coordinator.savedKeyLabel == nil)
    }

    @Test("A failed Keychain removal does not claim the key is gone")
    func failedKeychainRemoval() async {
        let keychain = FailingKeychain(secret: Self.keyA.rawValue)
        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { _ in ShareComputeLedgerFixture.account },
            fixture: nil
        )
        coordinator.appear()
        await Self.settle()
        coordinator.removeKey()
        #expect(coordinator.state == .unavailable(.keychainRemovalFailed))
        #expect(coordinator.state.needsReadKey)
        #expect(coordinator.savedKeyLabel == Self.keyA.redactedLabel)
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == Self.keyA.rawValue)
    }

    /// A loader whose completion the test controls.
    final class Gate: @unchecked Sendable {
        private var continuations: [CheckedContinuation<Void, Never>] = []
        private let lock = NSLock()
        private(set) var calls: [ShareComputeReadKey] = []

        func wait() async {
            await withCheckedContinuation { continuation in
                lock.lock()
                continuations.append(continuation)
                lock.unlock()
            }
        }

        func releaseAll() {
            lock.lock()
            let pending = continuations
            continuations = []
            lock.unlock()
            pending.forEach { $0.resume() }
        }

        func record(_ key: ShareComputeReadKey) {
            lock.lock(); calls.append(key); lock.unlock()
        }

        var callCount: Int {
            lock.lock(); defer { lock.unlock() }
            return calls.count
        }
    }

    private static func coordinator(
        keychain: InMemoryKeychain,
        now: @escaping @Sendable () -> Date = { Date() },
        load: @escaping @Sendable (ShareComputeReadKey) async throws -> ShareComputeLedgerAccount
    ) -> ShareComputeLedgerCoordinator {
        ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: load,
            now: now,
            // Explicitly no fixture: these tests exercise the real path.
            fixture: nil
        )
    }

    /// Lets the coordinator's detached work run to completion.
    ///
    /// A real (tiny) sleep, not `Task.yield()`. The load closure is
    /// `@Sendable` and nonisolated, so a walk is a MainActor → global →
    /// MainActor round trip; yielding on the main actor does not schedule the
    /// global-executor leg, and the result never lands. That produced tests
    /// that passed for the wrong reason.
    private static func settle() async {
        try? await Task.sleep(for: .milliseconds(40))
    }

    // MARK: Race 1 — a stale 401 must not delete a newly saved key

    @Test("A late 401 from the OLD key never deletes the key saved after it")
    func staleUnauthorizedDoesNotDeleteNewKey() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()

        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            // Only the old key is rejected; the new one would have succeeded.
            if key == Self.keyA { throw ShareComputeLedgerError.unauthorized }
            return ShareComputeLedgerFixture.account
        }

        coordinator.appear()
        await Self.settle()
        #expect(gate.callCount == 1)

        // The user pastes a new key while the old request is still in flight.
        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == Self.keyB.rawValue)

        // Now the OLD request's 401 lands.
        gate.releaseAll()
        await Self.settle()

        // The new key survives — this is the bug that would otherwise delete a
        // credential the user pasted seconds earlier.
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == Self.keyB.rawValue)
        #expect(coordinator.state != .unauthorized)
        #expect(coordinator.savedKeyLabel == Self.keyB.redactedLabel)
    }

    @Test("A 401 for the key that IS stored does delete it")
    func currentUnauthorizedDeletesKey() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let coordinator = Self.coordinator(keychain: keychain) { _ in
            throw ShareComputeLedgerError.unauthorized
        }
        coordinator.appear()
        await Self.settle()

        // Generation matches and the stored key is the rejected one, so the
        // dead credential is cleared rather than left to 401 again next launch.
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == nil)
        #expect(coordinator.state == .unauthorized)
        #expect(coordinator.savedKeyLabel == nil)
    }

    // MARK: Race 2 — removal wins over a request in flight

    @Test("Removing the key while a request is in flight ends in noReadKey")
    func removalBeatsInFlightRequest() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()
        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            return ShareComputeLedgerFixture.account
        }

        coordinator.appear()
        await Self.settle()
        #expect(gate.callCount == 1)

        coordinator.removeKey()
        #expect(coordinator.state == .noReadKey)

        // The in-flight walk now succeeds — and must be discarded.
        gate.releaseAll()
        await Self.settle()

        #expect(coordinator.state == .noReadKey)
        #expect(coordinator.state.account == nil)
        #expect(coordinator.savedKeyLabel == nil)
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == nil)
    }

    // MARK: Race 3 — out-of-order completion

    @Test("An older result never overwrites a newer one")
    func outOfOrderResultsAreDropped() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()

        // Key A returns a populated ledger; key B returns an empty one. If the
        // stale A result won, the page would show rows for a key that is gone.
        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            return key == Self.keyA
                ? ShareComputeLedgerFixture.account
                : ShareComputeLedgerFixture.emptyAccount
        }

        coordinator.appear()
        await Self.settle()

        // Replacing the key issues a newer request…
        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        await Self.settle()
        #expect(gate.callCount == 2)

        // …and both land at once, oldest first.
        gate.releaseAll()
        await Self.settle()

        // The newer (key B, empty) result stands.
        #expect(coordinator.state.account?.isEmpty == true)
        #expect(coordinator.state == .loadedEmpty(ShareComputeLedgerFixture.emptyAccount))
    }

    // MARK: Cancellation

    @Test("Leaving the tab cancels the walk in flight")
    func leavingTabCancels() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()
        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            return ShareComputeLedgerFixture.account
        }

        coordinator.appear()
        await Self.settle()
        #expect(coordinator.isBusy)

        // What `.task(id: tab)` cancellation drives.
        coordinator.cancelInFlight()
        #expect(!coordinator.isBusy)

        gate.releaseAll()
        await Self.settle()
        // The result of a cancelled walk is never applied.
        #expect(coordinator.state.account == nil)
    }

    @Test("The poll loop cancels its walk when the enclosing task is cancelled")
    func pollCancellationCancelsWalk() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()
        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            return ShareComputeLedgerFixture.account
        }

        let polling = Task { await coordinator.poll() }
        await Self.settle()
        #expect(coordinator.isBusy)

        polling.cancel()
        _ = await polling.value
        // `poll`'s `defer` released the in-flight walk.
        #expect(!coordinator.isBusy)
        gate.releaseAll()
        await Self.settle()
        #expect(coordinator.state.account == nil)
    }

    // MARK: Single-flight

    @Test("Hammering Refresh does not start a second account walk")
    func manualRefreshIsSingleFlight() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()
        let coordinator = Self.coordinator(keychain: keychain) { key in
            gate.record(key)
            await gate.wait()
            return ShareComputeLedgerFixture.account
        }

        coordinator.appear()
        await Self.settle()
        #expect(gate.callCount == 1)

        // Ten clicks while the first walk is still running.
        for _ in 0..<10 {
            coordinator.manualRefresh()
            await Self.settle()
        }
        // Still one walk. Each extra walk would be N more requests against a
        // 120/hour budget, for identical data.
        #expect(gate.callCount == 1)
        // …and the control reports itself unavailable, so the UI disables it.
        #expect(!coordinator.allowsManualRefresh)

        gate.releaseAll()
        await Self.settle()
        #expect(gate.callCount == 1)
    }

    // MARK: Interval gate

    @Test("Manual refresh obeys the five-minute floor, and key changes bypass it")
    func manualRefreshInterval() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = Gate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)
        let coordinator = Self.coordinator(keychain: keychain, now: { clock }) { key in
            gate.record(key)
            return ShareComputeLedgerFixture.account
        }

        // First entry is immediate — a blank tab must not wait five minutes.
        coordinator.appear()
        await Self.settle()
        #expect(gate.callCount == 1)

        // A minute later: refused.
        clock = clock.addingTimeInterval(60)
        #expect(!coordinator.allowsManualRefresh)
        coordinator.manualRefresh()
        await Self.settle()
        #expect(gate.callCount == 1)

        // Re-entering the tab must not bypass the gate either, now that there
        // is data on screen — otherwise tab-flipping is a burst button.
        coordinator.appear()
        await Self.settle()
        #expect(gate.callCount == 1)

        // Past the floor: allowed.
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval)
        #expect(coordinator.allowsManualRefresh)
        coordinator.manualRefresh()
        await Self.settle()
        #expect(gate.callCount == 2)

        // Saving a key validates immediately regardless of the interval: the
        // user is waiting to learn whether the key they pasted works.
        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        await Self.settle()
        #expect(gate.callCount == 3)
        #expect(gate.calls.last == Self.keyB)
    }

    // MARK: Rate limiting

    @Test("A 429 keeps prior data, holds Refresh down, and says why")
    func rateLimitBehaviour() async throws {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)
        nonisolated(unsafe) var shouldFail = false
        let coordinator = Self.coordinator(keychain: keychain, now: { clock }) { _ in
            if shouldFail { throw ShareComputeLedgerError.rateLimited }
            return ShareComputeLedgerFixture.account
        }

        coordinator.appear()
        await Self.settle()
        #expect(coordinator.state.account != nil)

        // Past the floor, then a 429.
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval)
        shouldFail = true
        coordinator.manualRefresh()
        await Self.settle()

        // Prior rows survive — a 429 says nothing about their validity — and
        // the failure reads as a quiet note, not a takeover.
        #expect(coordinator.state.account != nil)
        #expect(coordinator.state.staleNote != nil)
        #expect(coordinator.state.blockingError == nil)

        // Refresh is now held down even though the interval has elapsed, so it
        // cannot be clicked into a deeper hole.
        #expect(coordinator.isRateLimited)
        #expect(!coordinator.allowsManualRefresh)
        #expect(coordinator.refreshHoldNote != nil)
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval)
        #expect(!coordinator.allowsManualRefresh, "still inside the 429 cooldown")

        // After the cooldown it returns.
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.rateLimitCooldown)
        #expect(!coordinator.isRateLimited)
        #expect(coordinator.allowsManualRefresh)
    }

    @Test("Pagination is one refresh, not one per page")
    func paginationIsOneRefresh() async throws {
        // The interval gate lives on `refresh`, never inside the walk: an
        // account with ten pages costs ten HTTP requests but is ONE refresh,
        // and must not stall five minutes between pages.
        actor Pages {
            var count = 0
            func bump() -> Int { count += 1; return count }
        }
        let pages = Pages()
        let client = ShareComputeLedgerClient { request in
            let n = await pages.bump()
            let next = n < 3 ? "\(n)" : "null"
            let json = """
            {"unit":"usd_api_credit","caps":{"node_monthly_cap_usd":20.0,"pool_monthly_cap_usd":100.0},
             "ledger":[],"next_cursor":\(next)}
            """
            return (Data(json.utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        let account = try await client.account(key: Self.keyA)
        #expect(await pages.count == 3)
        #expect(!account.isTruncated)
    }

    // MARK: Redirects

    @Test("A credential-bearing ledger request refuses every redirect")
    func redirectsAreRefused() async {
        let delegate = ShareComputeLedgerClient.NoRedirectDelegate()
        // The redirect URLSession would send, already carrying our header —
        // this is exactly the request that must never go out.
        var proposed = URLRequest(url: URL(string: "https://evil.example/v1/pool/ledger")!)
        proposed.setValue("Bearer \(Self.keyA.rawValue)", forHTTPHeaderField: "Authorization")

        let forwarded: URLRequest? = await withCheckedContinuation { continuation in
            delegate.urlSession(
                ShareComputeLedgerClient.session,
                task: ShareComputeLedgerClient.session.dataTask(
                    with: URLRequest(url: ShareComputeLedgerClient.endpoint)
                ),
                willPerformHTTPRedirection: HTTPURLResponse(
                    url: ShareComputeLedgerClient.endpoint,
                    statusCode: 302,
                    httpVersion: nil,
                    headerFields: ["Location": "https://evil.example/v1/pool/ledger"]
                )!,
                newRequest: proposed,
                completionHandler: { continuation.resume(returning: $0) }
            )
        }

        // nil means "do not follow" — the 302 surfaces as the response and the
        // Authorization header never leaves the pinned origin.
        #expect(forwarded == nil)
    }

    @Test("The ledger session is ephemeral, cookie-free, and cache-free")
    func sessionHardening() {
        let configuration = ShareComputeLedgerClient.makeConfiguration()
        // Not URLSession.shared: that session follows redirects, owns a
        // process-wide cookie jar, and writes to a shared on-disk cache.
        #expect(configuration.httpCookieStorage == nil)
        #expect(!configuration.httpShouldSetCookies)
        #expect(configuration.httpCookieAcceptPolicy == .never)
        #expect(configuration.urlCache == nil)
        #expect(configuration.requestCachePolicy == .reloadIgnoringLocalCacheData)
        #expect(configuration.timeoutIntervalForRequest == ShareComputeLedgerClient.timeout)
        // The endpoint stays pinned.
        #expect(
            ShareComputeLedgerClient.endpoint.absoluteString
                == "https://pay.quicksilverpro.io/v1/pool/ledger"
        )
    }

    @Test("Transport injection still bypasses the network entirely")
    func transportInjectionIsIntact() async throws {
        // The hardening must not have cost testability: a client built with an
        // injected transport never constructs a URLSession task.
        let client = ShareComputeLedgerClient { request in
            (
                Data(ShareComputeLedgerTests.sampleJSON.utf8),
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            )
        }
        let page = try await client.page(key: Self.keyA)
        #expect(page.windows.count == 1)
    }

    // MARK: Fixtures

    @Test("A golden fixture never reads the Keychain or issues a request")
    func fixtureShortCircuits() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        nonisolated(unsafe) var called = false
        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { _ in called = true; return ShareComputeLedgerFixture.account },
            now: { Date() },
            fixture: .revoked
        )
        coordinator.appear()
        await Self.settle()
        #expect(!called)
        #expect(coordinator.state == .unauthorized)
        // …and the stored key is untouched: a capture must not mutate storage.
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == Self.keyA.rawValue)
    }
}

extension ShareComputeLedgerSafetyTests {

    @Test("The result-acceptance rule rejects cancelled, stale, and superseded results")
    func acceptanceRule() {
        typealias C = ShareComputeLedgerCoordinator
        // The only accepting case: live task, current generation, newest issue.
        #expect(C.acceptsResult(
            capturedGeneration: 3, currentGeneration: 3,
            capturedIssue: 9, currentIssue: 9, isCancelled: false
        ))
        // Cancelled — Credits closed, or a credential change tore this down.
        #expect(!C.acceptsResult(
            capturedGeneration: 3, currentGeneration: 3,
            capturedIssue: 9, currentIssue: 9, isCancelled: true
        ))
        // Stale generation — a key was saved or removed mid-flight, so this
        // result describes a credential that is no longer in play.
        #expect(!C.acceptsResult(
            capturedGeneration: 2, currentGeneration: 3,
            capturedIssue: 9, currentIssue: 9, isCancelled: false
        ))
        // Superseded issue — a newer request exists; a slow older response
        // must never overwrite a faster newer one.
        #expect(!C.acceptsResult(
            capturedGeneration: 3, currentGeneration: 3,
            capturedIssue: 8, currentIssue: 9, isCancelled: false
        ))
        // All three wrong at once is still a rejection, not a double negative.
        #expect(!C.acceptsResult(
            capturedGeneration: 1, currentGeneration: 3,
            capturedIssue: 1, currentIssue: 9, isCancelled: true
        ))
    }
}

// MARK: - Ledger gating

/// The pre-launch gate fixes: a 429 cooldown that outranks every refresh
/// reason, a "first attempt" that means what it says, and a time gate that
/// re-enables its own button.
@Suite("Share Compute · Ledger gating")
@MainActor
struct ShareComputeLedgerGatingTests {

    nonisolated static let keyA = try! ShareComputeReadKey
        .validate("qsprk-" + String(repeating: "a", count: 30)).get()
    nonisolated static let keyB = try! ShareComputeReadKey
        .validate("qsprk-" + String(repeating: "b", count: 30)).get()

    /// A wake-timer sleep the test releases by hand, so a five-minute gate can
    /// be crossed without waiting five minutes.
    final class SleepGate: @unchecked Sendable {
        private let lock = NSLock()
        private var waiting: [UUID: CheckedContinuation<Void, Never>] = [:]
        private var cancelled: Set<UUID> = []
        private(set) var requestedDelays: [TimeInterval] = []

        /// Honours cancellation the way `Task.sleep` does.
        ///
        /// Race-safe in both directions: `onCancel` can fire before the
        /// continuation is even registered, so cancellation is recorded as a
        /// flag and the registering side resumes immediately if it finds one.
        /// Without that, a cancelled wake parks forever and the double reports
        /// timers the coordinator already let go.
        func sleep(_ delay: TimeInterval) async {
            // `NSLock.lock()` is unavailable directly in an async context, so
            // the mutations happen in synchronous helpers.
            record(delay)
            let id = UUID()
            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    enqueue(id, continuation)
                }
            } onCancel: {
                cancel(id)
            }
        }

        private func cancel(_ id: UUID) {
            lock.lock()
            cancelled.insert(id)
            let continuation = waiting.removeValue(forKey: id)
            lock.unlock()
            continuation?.resume()
        }

        private func record(_ delay: TimeInterval) {
            lock.lock(); requestedDelays.append(delay); lock.unlock()
        }

        private func enqueue(_ id: UUID, _ continuation: CheckedContinuation<Void, Never>) {
            lock.lock()
            if cancelled.remove(id) != nil {
                lock.unlock()
                continuation.resume()
                return
            }
            waiting[id] = continuation
            lock.unlock()
        }

        /// Lets every armed timer fire.
        func fire() {
            lock.lock()
            let pending = Array(waiting.values)
            waiting = [:]
            lock.unlock()
            pending.forEach { $0.resume() }
        }

        var armedCount: Int {
            lock.lock(); defer { lock.unlock() }
            return waiting.count
        }
    }

    final class Calls: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var keys: [ShareComputeReadKey] = []
        func record(_ key: ShareComputeReadKey) {
            lock.lock(); keys.append(key); lock.unlock()
        }
        var count: Int { lock.lock(); defer { lock.unlock() }; return keys.count }
    }

    /// See the note on the sibling suite's `settle`: a real sleep, because
    /// `Task.yield()` does not drain the off-actor leg of a walk.
    private static func settle() async {
        try? await Task.sleep(for: .milliseconds(40))
    }

    // MARK: 429 outranks every reason

    @Test("An active 429 cooldown blocks appear, manual, poll, and keyChanged")
    func cooldownOutranksEveryReason() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let calls = Calls()
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { key in calls.record(key); throw ShareComputeLedgerError.rateLimited },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )

        coordinator.appear()
        await Self.settle()
        #expect(calls.count == 1)
        #expect(coordinator.isRateLimited)

        // Every reason, repeatedly. The limit is per IP, so none of them may
        // spend a request.
        coordinator.appear()
        coordinator.manualRefresh()
        coordinator.appear()
        await Self.settle()
        #expect(calls.count == 1)

        // Even a brand-new credential. Pasting a different key cannot buy a
        // request from an IP-level limiter.
        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        await Self.settle()
        #expect(calls.count == 1)
        #expect(!coordinator.allowsManualRefresh)
    }

    @Test("Saving during a cooldown stores the key but does not clear the cooldown")
    func saveDuringCooldownKeepsCooldown() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let calls = Calls()
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { key in calls.record(key); throw ShareComputeLedgerError.rateLimited },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )
        coordinator.appear()
        await Self.settle()
        let cooldownDeadline = coordinator.rateLimitedUntil
        #expect(cooldownDeadline != nil)

        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        await Self.settle()

        // The key IS safely stored — that part is immediate.
        #expect(keychain.read(account: ShareComputeReadKeyStore.account) == Self.keyB.rawValue)
        #expect(coordinator.savedKeyLabel == Self.keyB.redactedLabel)
        // …but no request went out and the cooldown is untouched.
        #expect(calls.count == 1)
        #expect(coordinator.rateLimitedUntil == cooldownDeadline)
        #expect(coordinator.isRateLimited)
    }

    @Test("When the cooldown lifts, the key saved during it validates itself")
    func pendingKeyValidatesAfterCooldown() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let calls = Calls()
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)
        nonisolated(unsafe) var limited = true

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { key in
                calls.record(key)
                if limited { throw ShareComputeLedgerError.rateLimited }
                return ShareComputeLedgerFixture.account
            },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )

        coordinator.appear()
        await Self.settle()
        #expect(coordinator.save(draft: Self.keyB.rawValue) == nil)
        await Self.settle()
        #expect(calls.count == 1)

        // The cooldown expires.
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.rateLimitCooldown + 1)
        limited = false
        gate.fire()
        await Self.settle()

        // The owed validation ran on its own — the user did not paste again.
        #expect(calls.count == 2)
        #expect(calls.keys.last == Self.keyB)
        #expect(coordinator.state.account != nil)
    }

    // MARK: First attempt

    @Test("A failed first attempt does not make later tab entries free")
    func failedFirstAttemptStillCountsAsAnAttempt() async {
        for failure in [
            ShareComputeLedgerError.rateLimited,
            .serviceUnavailable,
            .malformedBody,
        ] {
            let keychain = InMemoryKeychain()
            keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
            let calls = Calls()
            let gate = SleepGate()
            nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

            let coordinator = ShareComputeLedgerCoordinator(
                store: ShareComputeReadKeyStore(keychain: keychain),
                load: { key in calls.record(key); throw failure },
                now: { clock },
                sleep: { await gate.sleep($0) },
                fixture: nil
            )

            coordinator.appear()
            await Self.settle()
            #expect(calls.count == 1, "\(failure)")
            // No data landed, so the old `state.account == nil` test would
            // have called every one of these a fresh first entry.
            #expect(coordinator.state.account == nil, "\(failure)")

            // Leave and re-enter, repeatedly, inside the interval.
            clock = clock.addingTimeInterval(30)
            for _ in 0..<5 {
                coordinator.cancelInFlight()
                coordinator.appear()
                await Self.settle()
            }
            #expect(calls.count == 1, "\(failure) — tab switching bypassed the gate")
        }
    }

    @Test("A cancelled first attempt also counts as an attempt")
    func cancelledFirstAttemptCountsAsAnAttempt() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let calls = Calls()
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { key in
                calls.record(key)
                try await Task.sleep(for: .seconds(60))
                return ShareComputeLedgerFixture.account
            },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )

        coordinator.appear()
        await Self.settle()
        #expect(calls.count == 1)

        coordinator.cancelInFlight()
        clock = clock.addingTimeInterval(10)
        coordinator.appear()
        await Self.settle()
        // `lastRefreshStartedAt` was stamped when the request was ISSUED, so
        // the cancelled attempt still spent the allowance.
        #expect(calls.count == 1)
    }

    // MARK: Observable gate expiry

    @Test("A time gate expiring publishes an observable change")
    func gateExpiryIsObservable() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { _ in ShareComputeLedgerFixture.account },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )

        coordinator.appear()
        await Self.settle()
        // Inside the interval, so the button is down and a timer is armed.
        #expect(!coordinator.allowsManualRefresh)
        #expect(coordinator.hasScheduledWake)

        // Exactly what SwiftUI does when it reads the property in a body.
        nonisolated(unsafe) var notified = false
        withObservationTracking {
            _ = coordinator.allowsManualRefresh
        } onChange: {
            notified = true
        }

        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval + 1)
        // `hasScheduledWake` records the Task before its injected sleeper has
        // installed a continuation. Wait for that handoff before releasing it.
        for _ in 0..<100 {
            if gate.armedCount != 0 { break }
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(gate.armedCount == 1)
        gate.fire()
        for _ in 0..<100 {
            if notified { break }
            try? await Task.sleep(for: .milliseconds(10))
        }

        // The view was told, without a tab switch, a status-bar tick, or any
        // other page's state changing.
        #expect(notified)
        #expect(coordinator.allowsManualRefresh)
        #expect(coordinator.refreshHoldNote == nil)
    }

    // MARK: Timer hygiene

    @Test("Repeated tab switches and saves never leave more than one wake task")
    func singleWakeTask() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { _ in throw ShareComputeLedgerError.rateLimited },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )

        coordinator.appear()
        await Self.settle()

        for index in 0..<8 {
            coordinator.cancelInFlight()
            coordinator.appear()
            coordinator.manualRefresh()
            if index % 3 == 0 {
                _ = coordinator.save(draft: Self.keyB.rawValue)
            }
            await Self.settle()
            #expect(coordinator.liveWakeTaskCount <= 1, "iteration \(index)")
            #expect(gate.armedCount <= 1, "iteration \(index)")
        }
    }

    @Test("Leaving Credits and removing the key both cancel the wake task")
    func wakeTaskIsCancelled() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        func makeCoordinator() -> ShareComputeLedgerCoordinator {
            ShareComputeLedgerCoordinator(
                store: ShareComputeReadKeyStore(keychain: keychain),
                load: { _ in throw ShareComputeLedgerError.rateLimited },
                now: { clock },
                sleep: { await gate.sleep($0) },
                fixture: nil
            )
        }

        // Leaving Credits.
        let leaving = makeCoordinator()
        leaving.appear()
        await Self.settle()
        #expect(leaving.hasScheduledWake)
        leaving.cancelInFlight()
        #expect(!leaving.hasScheduledWake)
        #expect(leaving.liveWakeTaskCount == 0)

        // Removing the key.
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let removing = makeCoordinator()
        removing.appear()
        await Self.settle()
        #expect(removing.hasScheduledWake)
        removing.removeKey()
        #expect(!removing.hasScheduledWake)
        #expect(removing.liveWakeTaskCount == 0)
        #expect(removing.state == .noReadKey)
        // The cooldown is a property of this IP's request history, so it
        // outlives the credential that triggered it.
        #expect(removing.isRateLimited)
    }

    @Test("Removing the key drops the owed validation")
    func removalDropsPendingValidation() async {
        let keychain = InMemoryKeychain()
        keychain.write(account: ShareComputeReadKeyStore.account, secret: Self.keyA.rawValue)
        let calls = Calls()
        let gate = SleepGate()
        nonisolated(unsafe) var clock = Date(timeIntervalSince1970: 1_800_000_000)

        let coordinator = ShareComputeLedgerCoordinator(
            store: ShareComputeReadKeyStore(keychain: keychain),
            load: { key in calls.record(key); throw ShareComputeLedgerError.rateLimited },
            now: { clock },
            sleep: { await gate.sleep($0) },
            fixture: nil
        )
        coordinator.appear()
        await Self.settle()
        _ = coordinator.save(draft: Self.keyB.rawValue)
        await Self.settle()
        #expect(calls.count == 1)

        coordinator.removeKey()
        clock = clock.addingTimeInterval(ShareComputeLedgerRefresh.rateLimitCooldown + 1)
        gate.fire()
        await Self.settle()

        // Nothing to validate — the key is gone.
        #expect(calls.count == 1)
        #expect(coordinator.state == .noReadKey)
    }
}
