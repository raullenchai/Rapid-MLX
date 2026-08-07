import XCTest

@testable import Rapid

/// Contract tests for ``ModelReadiness`` — the single source of truth for
/// "can the user send right now, and if not, what should they do?".
///
/// XCTest rather than swift-testing deliberately: the excluded legacy
/// suite under `Tests/RapidTests` uses `import Testing`, and the package
/// manifest records that the module does not resolve from the command
/// line in this toolchain. XCTest always does, and a test that cannot be
/// run is not a test.
///
/// Everything here is a pure value transform, so no SwiftUI host, no
/// subprocess, and no live ``ServerManager`` is needed.
final class ModelReadinessTests: XCTestCase {

    private let alias = "qwen3.5-9b-4bit"

    // MARK: - Helpers

    /// `cached: nil` is the TRANSIENT unknown — the catalog has not
    /// answered yet. The permanently-unknown alias (a custom name the
    /// catalog has loaded and does not contain) is a different input and
    /// gets its own tests below; it is never reachable through this
    /// helper, so every existing case here keeps its original meaning.
    private func resolve(
        _ state: ServerState,
        alias: String? = nil,
        cached: Bool? = true,
        sizeText: String? = nil,
        progress: ModelReadiness.ProgressSnapshot? = nil,
        failure: ModelReadiness.Failure? = nil
    ) -> ModelReadiness {
        ModelReadiness.resolve(
            serverState: state,
            alias: alias ?? self.alias,
            cacheState: cacheState(cached),
            sizeText: sizeText,
            progress: progress,
            failure: failure
        )
    }

    /// Same, for the cache states ``Bool?`` cannot express.
    private func resolveCacheState(
        _ state: ServerState,
        alias: String? = nil,
        cacheState: ModelReadiness.CacheState,
        sizeText: String? = nil
    ) -> ModelReadiness {
        ModelReadiness.resolve(
            serverState: state,
            alias: alias ?? self.alias,
            cacheState: cacheState,
            sizeText: sizeText
        )
    }

    private func cacheState(_ cached: Bool?) -> ModelReadiness.CacheState {
        switch cached {
        case .some(true):  return .onDisk
        case .some(false): return .notOnDisk
        case .none:        return .catalogPending
        }
    }

    /// Every case, for sweeps that must hold universally.
    private var allStates: [ModelReadiness] {
        [
            .engineMissing,
            .noModel,
            .needsDownload(alias: alias, sizeText: "5.0 GB"),
            .needsDownload(alias: alias, sizeText: nil),
            .needsStart(alias: alias),
            .unknownModel(alias: alias),
            .downloading(alias: alias, detail: "1.2 / 5.0 GB · 24%", fraction: 0.24),
            .downloading(alias: alias, detail: nil, fraction: nil),
            .starting(alias: alias, detail: "Warming up…"),
            .ready(alias: alias),
            .failed(alias: alias, message: "This model couldn't load.", action: .retry(alias: alias)),
            .failed(alias: nil, message: "Something went wrong.", action: nil),
        ]
    }

    // MARK: - The send gate

    /// The core Phase 1 contract: sending is possible in exactly one
    /// state. Everything else keeps the field live and the action dark.
    func testSendAllowedOnlyWhenReady() {
        for state in allStates {
            if case .ready = state {
                XCTAssertTrue(state.sendAllowed, "ready must allow send")
            } else {
                XCTAssertFalse(
                    state.sendAllowed,
                    "\(state) must not allow send"
                )
            }
        }
    }

    /// A gated state must always be able to say WHY, in both the mouse
    /// channel (tooltip) and the copy channel (banner headline). An
    /// empty explanation is the silent-gate defect this phase removes.
    func testEveryGatedStateExplainsItself() {
        for state in allStates where !state.sendAllowed {
            XCTAssertFalse(
                state.sendTooltip.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(state) has no send tooltip"
            )
            XCTAssertFalse(
                state.headline.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(state) has no headline"
            )
            XCTAssertFalse(
                state.composerPlaceholder.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(state) has no composer placeholder"
            )
        }
    }

    /// Ready is the one state that must NOT nag: no banner detail, and a
    /// plain "Send" tooltip rather than an explanation of a problem that
    /// no longer exists.
    func testReadyIsQuiet() {
        let ready = ModelReadiness.ready(alias: alias)
        XCTAssertNil(ready.detail)
        XCTAssertNil(ready.action)
        XCTAssertNil(ready.emptyStateHint)
        XCTAssertEqual(ready.sendTooltip, "Send")
        XCTAssertEqual(ready.composerPlaceholder, "Send a message…")
    }

    // MARK: - Placeholder-alias safety (the "Couldn't start ." defect)

    /// The regression that motivated most of this: an unresolved alias
    /// was interpolated into failure copy, producing "Couldn't start ."
    /// with an empty model name. No surface may render a placeholder as
    /// if it were a model.
    func testUnresolvedAliasNeverBecomesAModelName() {
        // "" plus every internal placeholder ``ModelDisplayName`` knows.
        let placeholders = ["", "   ", "loading", "Loading", "STARTING",
                            "warming up", "downloading", "unknown", "none"]
        for placeholder in placeholders {
            let state = resolve(.idle, alias: placeholder, cached: false)
            XCTAssertEqual(
                state, .noModel,
                "\(placeholder.debugDescription) must resolve to .noModel"
            )
            XCTAssertNil(state.alias)
            // And the copy must not contain the raw placeholder.
            let trimmed = placeholder.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                XCTAssertFalse(
                    state.headline.lowercased().contains(trimmed.lowercased()),
                    "headline leaked the placeholder \(placeholder.debugDescription)"
                )
            }
        }
    }

    /// A `.ready` state carrying a placeholder alias is not ready — it
    /// falls back to "choose a model" rather than claiming to be
    /// chatting with nothing.
    func testReadyWithPlaceholderAliasIsNotReady() {
        let state = resolve(.ready(alias: ""), alias: "")
        XCTAssertEqual(state, .noModel)
        XCTAssertFalse(state.sendAllowed)
    }

    /// No user-facing string may ever contain an empty-name artefact
    /// like a double space, a dangling "start ." or a trailing " —".
    func testNoCopyContainsEmptyNameArtefacts() {
        for state in allStates {
            let strings = [
                state.headline,
                state.detail ?? "",
                state.composerPlaceholder,
                state.sendTooltip,
                state.emptyStateSubtitle,
                state.emptyStateHint ?? "",
                state.accessibilityLabel,
            ]
            for string in strings where !string.isEmpty {
                XCTAssertFalse(string.contains("  "), "double space in: \(string)")
                XCTAssertFalse(string.contains(" ."), "dangling period in: \(string)")
                XCTAssertFalse(string.hasSuffix(" —"), "dangling dash in: \(string)")
                XCTAssertFalse(string.contains("()"), "empty parens in: \(string)")
            }
        }
    }

    // MARK: - Resolution precedence

    func testEngineMissingWinsOverEverything() {
        let state = resolve(
            .missing,
            cached: false,
            failure: .init(message: "boom", kind: .modelLoadFailed)
        )
        XCTAssertEqual(state, .engineMissing)
        XCTAssertFalse(state.sendAllowed)
    }

    /// An in-flight start outranks a stale failure. Otherwise pressing
    /// Retry would leave the banner reading "Couldn't start X" while X
    /// is visibly starting.
    func testInFlightStartBeatsStaleFailure() {
        let state = resolve(
            .starting(alias: alias),
            failure: .init(message: "old failure", kind: .modelLoadFailed)
        )
        guard case .starting(let name, _) = state else {
            return XCTFail("expected .starting, got \(state)")
        }
        XCTAssertEqual(name, alias)
    }

    /// A crash outranks a chat-level failure and supplies its own
    /// classified message plus a retry aimed at the crashed alias.
    func testCrashedResolvesToFailedWithRetry() {
        let state = resolve(
            .crashed(alias: alias, message: "RuntimeError: out of memory"),
            failure: .init(message: "unrelated", kind: .requestFailed)
        )
        guard case .failed(let name, let message, let action) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(name, alias)
        XCTAssertEqual(action, .retry(alias: alias))
        // Classified, not raw engine output.
        XCTAssertFalse(message.contains("RuntimeError"))
        XCTAssertEqual(
            message,
            FailureDiagnoser.diagnosis(for: .modelOutOfMemory, modelAlias: alias).message
        )
    }

    /// ``ChatViewModel.lastFailureKind`` was previously computed and
    /// discarded. It must now choose the message the user reads.
    func testChatFailureUsesStructuredDiagnosisOverRawMessage() {
        let state = resolve(
            .idle,
            failure: .init(
                message: "raw transport gibberish",
                kind: .engineNotRunning,
                alias: alias
            )
        )
        guard case .failed(_, let message, let action) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(
            message,
            FailureDiagnoser.diagnosis(for: .engineNotRunning, modelAlias: alias).message
        )
        XCTAssertEqual(action, .retry(alias: alias))
    }

    /// With no structured kind, the raw message is the fallback rather
    /// than a generic sentence that loses information.
    func testChatFailureWithoutKindKeepsItsMessage() {
        let state = resolve(.idle, failure: .init(message: "Disk is full.", alias: alias))
        guard case .failed(_, let message, _) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(message, "Disk is full.")
    }

    // MARK: - Failure attribution (a failure belongs to ONE model)

    /// Regression: `kimi-k2.6` fails to start, the user picks
    /// `bonsai-1.7b-2bit`, and the banner keeps showing Kimi's failure —
    /// including its Retry, its name in the placeholder, and its name in
    /// the Send tooltip. Every further pick stayed on Kimi too, because
    /// ``.crashed`` short-circuited before the selection was consulted.
    private let crashed = "kimi-k2.6"

    func testCrashedAliasMatchingSelectionStillFails() {
        let state = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: crashed,
            cached: true
        )
        guard case .failed(let name, _, let action) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(name, crashed)
        XCTAssertEqual(action, .retry(alias: crashed), "Retry must target the failed model")
        XCTAssertFalse(state.sendAllowed)
    }

    /// The fix: a crash on a DIFFERENT model than the one now selected
    /// must not describe the new selection. Cached → needsStart.
    func testCrashedAliasDiffersFromSelectionResolvesToNeedsStart() {
        let state = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: alias,
            cached: true
        )
        XCTAssertEqual(state, .needsStart(alias: alias))
        XCTAssertEqual(state.action, .start(alias: alias))
        assertNoTraceOf(crashed, in: state)
    }

    /// Same, uncached → needsDownload with the download CTA.
    func testCrashedAliasDiffersFromSelectionResolvesToNeedsDownload() {
        let state = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: alias,
            cached: false,
            sizeText: "5.0 GB"
        )
        XCTAssertEqual(state, .needsDownload(alias: alias, sizeText: "5.0 GB"))
        XCTAssertEqual(state.action, .downloadAndStart(alias: alias))
        assertNoTraceOf(crashed, in: state)
    }

    /// A stale chat-level failure must not follow the user either — this
    /// is the second half of the bug. Previously the resolver read
    /// `failure.alias ?? alias`, so a failure recorded against Kimi (or
    /// with no alias at all) got re-attributed to the new pick.
    func testStaleChatFailureDoesNotFollowTheNewSelection() {
        let state = resolve(
            .idle,
            alias: alias,
            cached: true,
            failure: .init(message: "Kimi blew up", kind: .modelLoadFailed, alias: crashed)
        )
        XCTAssertEqual(state, .needsStart(alias: alias))
        assertNoTraceOf(crashed, in: state)
        assertNoTraceOf("blew up", in: state)
    }

    /// A chat failure that DOES belong to the selection still shows.
    func testChatFailureMatchingSelectionStillShows() {
        let state = resolve(
            .idle,
            alias: alias,
            cached: true,
            failure: .init(message: "x", kind: .modelLoadFailed, alias: alias)
        )
        guard case .failed(let name, _, let action) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(name, alias)
        XCTAssertEqual(action, .retry(alias: alias))
    }

    /// An unattributable failure must not be pinned on the user's fresh
    /// pick. We show the selection's own (true) state instead.
    func testUnattributedFailureDoesNotBlameTheSelection() {
        let state = resolve(
            .idle,
            alias: alias,
            cached: true,
            failure: .init(message: "something went wrong", kind: nil, alias: nil)
        )
        XCTAssertEqual(state, .needsStart(alias: alias))
    }

    /// With nothing chosen there is no better thing to show, so the
    /// failure stays visible rather than silently vanishing.
    func testFailureStaysVisibleWhenNothingIsSelected() {
        let state = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: "",
            cached: nil
        )
        guard case .failed(let name, _, _) = state else {
            return XCTFail("expected .failed, got \(state)")
        }
        XCTAssertEqual(name, crashed)
    }

    /// The reported repro in full: crash, then pick three different
    /// models in a row. None may fall back to Kimi's failure.
    func testSuccessiveSelectionsAfterCrashNeverReturnToTheFailedModel() {
        let picks = [
            ("bonsai-1.7b-2bit", true),
            ("qwen3.5-9b-4bit", false),
            ("gemma-4-12b-4bit", true),
        ]
        for (pick, isCached) in picks {
            let state = resolve(
                .crashed(alias: crashed, message: "load failed"),
                alias: pick,
                cached: isCached,
                // The stale chat error rides along the whole time.
                failure: .init(message: "Kimi blew up", kind: .modelLoadFailed, alias: crashed)
            )
            XCTAssertEqual(
                state,
                isCached
                    ? .needsStart(alias: pick)
                    : .needsDownload(alias: pick, sizeText: nil),
                "selection \(pick) regressed to a stale failure"
            )
            XCTAssertEqual(state.action?.alias, pick, "CTA must target \(pick)")
            assertNoTraceOf(crashed, in: state)
        }
    }

    /// Selecting the failed model AGAIN brings its failure back — the
    /// failure is scoped, not discarded.
    func testReselectingTheFailedModelRestoresItsFailure() {
        let away = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: alias,
            cached: true
        )
        XCTAssertEqual(away, .needsStart(alias: alias))

        let back = resolve(
            .crashed(alias: crashed, message: "load failed"),
            alias: crashed,
            cached: true
        )
        guard case .failed(_, _, let action) = back else {
            return XCTFail("expected .failed on reselect, got \(back)")
        }
        XCTAssertEqual(action, .retry(alias: crashed))
    }

    /// The attribution rule on its own, so a future refactor of
    /// ``resolve`` cannot quietly change it.
    func testFailureAppliesRule() {
        // Nothing selected → always show.
        XCTAssertTrue(ModelReadiness.failureApplies(failedAlias: "a", selectedAlias: nil))
        XCTAssertTrue(ModelReadiness.failureApplies(failedAlias: nil, selectedAlias: nil))
        // Same model → show.
        XCTAssertTrue(ModelReadiness.failureApplies(failedAlias: "a", selectedAlias: "a"))
        // Different model → suppress.
        XCTAssertFalse(ModelReadiness.failureApplies(failedAlias: "a", selectedAlias: "b"))
        // Unattributable + a real selection → suppress (never blame it).
        XCTAssertFalse(ModelReadiness.failureApplies(failedAlias: nil, selectedAlias: "b"))
    }

    /// No user-facing string in ``state`` mentions ``needle``.
    private func assertNoTraceOf(
        _ needle: String,
        in state: ModelReadiness,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let corpus = [
            state.headline,
            state.detail ?? "",
            state.composerPlaceholder,
            state.sendTooltip,
            state.emptyStateSubtitle,
            state.emptyStateHint ?? "",
            state.accessibilityLabel,
        ].joined(separator: " ")
        XCTAssertFalse(
            corpus.lowercased().contains(needle.lowercased()),
            "user-facing copy still mentions “\(needle)”: \(corpus)",
            file: file, line: line
        )
    }

    // MARK: - Choose → download → start → ready

    func testNoModelWhenNothingChosen() {
        XCTAssertEqual(resolve(.idle, alias: "", cached: nil), .noModel)
        XCTAssertEqual(resolve(.stopped, alias: "", cached: nil), .noModel)
    }

    func testChosenButUncachedNeedsDownload() {
        let state = resolve(.idle, cached: false, sizeText: "5.0 GB")
        XCTAssertEqual(state, .needsDownload(alias: alias, sizeText: "5.0 GB"))
        XCTAssertEqual(state.action, .downloadAndStart(alias: alias))
        XCTAssertTrue(state.detail?.contains("5.0 GB") == true)
    }

    func testChosenAndCachedNeedsStart() {
        let state = resolve(.idle, cached: true)
        XCTAssertEqual(state, .needsStart(alias: alias))
        XCTAssertEqual(state.action, .start(alias: alias))
    }

    /// An unknown cache state must not claim a download is required. We
    /// resolve to "start", which is true either way — ``ServerManager``
    /// pulls on demand — rather than promising a multi-gigabyte wait we
    /// have no evidence for.
    func testUnknownCacheStateResolvesToStartNotDownload() {
        let state = resolve(.idle, cached: nil)
        XCTAssertEqual(state, .needsStart(alias: alias))
    }

    /// The same input stated explicitly: a still-loading catalog keeps
    /// ``needsStart`` and its "already downloaded" copy. This is the
    /// transient case the fallback was designed for, and it must not
    /// regress when the permanently-unknown case is split out below.
    func testCatalogPendingKeepsTheAlreadyDownloadedCopy() {
        let state = resolveCacheState(.idle, cacheState: .catalogPending)
        XCTAssertEqual(state, .needsStart(alias: alias))
        XCTAssertEqual(state.detail, "It's already downloaded — starting takes a few seconds.")
    }

    /// The defect: an alias the loaded catalog does NOT contain — a
    /// custom model name typed into "Type a model name…" — was falling
    /// through to ``needsStart``, whose detail asserts the weights are
    /// on disk. Nothing establishes that, and the picker chip beside the
    /// banner simultaneously showed the unknown-model glyph, so the same
    /// row said two different things.
    func testUnknownAliasDoesNotClaimItIsDownloaded() {
        let unknown = "mlx-community/Some-Custom-Repo"
        let state = resolveCacheState(
            .idle,
            alias: unknown,
            cacheState: .notInCatalog
        )
        XCTAssertEqual(state, .unknownModel(alias: unknown))
        let detail = state.detail ?? ""
        XCTAssertFalse(
            detail.lowercased().contains("already downloaded"),
            "an alias we cannot find must not be described as downloaded: \(detail)"
        )
        XCTAssertFalse(detail.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    /// Copy is the ONLY thing that changes. The step the user takes, the
    /// gate on Send, and the status colour are identical to
    /// ``needsStart`` — ``ServerManager`` pulls on demand, so Start is
    /// still the right and only button.
    func testUnknownAliasStillOffersStartAndStaysGated() {
        let unknown = "mlx-community/Some-Custom-Repo"
        let state = resolveCacheState(.idle, alias: unknown, cacheState: .notInCatalog)
        XCTAssertEqual(state.action, .start(alias: unknown))
        XCTAssertFalse(state.sendAllowed)
        XCTAssertFalse(state.isFailure)
        XCTAssertEqual(state.statusRole, .idle)
        XCTAssertEqual(state.headline, ModelReadiness.needsStart(alias: unknown).headline)
        XCTAssertEqual(
            state.composerPlaceholder,
            ModelReadiness.needsStart(alias: unknown).composerPlaceholder
        )
    }

    /// An unknown alias is only unknown while it is not running. Once the
    /// engine is serving it, the live serve-state outranks the catalog —
    /// a custom model the user started must reach ``ready`` and enable
    /// Send like any other.
    func testUnknownAliasStillReachesReadyWhenServing() {
        let unknown = "mlx-community/Some-Custom-Repo"
        let state = resolveCacheState(
            .ready(alias: unknown),
            alias: unknown,
            cacheState: .notInCatalog
        )
        XCTAssertEqual(state, .ready(alias: unknown))
        XCTAssertTrue(state.sendAllowed)
    }

    /// A blank alias is "no model chosen" regardless of how the cache
    /// state describes it — the unknown-alias branch must not intercept
    /// the empty selection and start naming a placeholder.
    func testUnknownCacheStateWithNoAliasIsStillNoModel() {
        XCTAssertEqual(
            resolveCacheState(.idle, alias: "", cacheState: .notInCatalog),
            .noModel
        )
    }

    /// An empty size string means ``ModelSizing`` had no estimate. It
    /// must be dropped, not rendered as empty parentheses.
    func testBlankSizeTextIsDropped() {
        let state = resolve(.idle, cached: false, sizeText: "   ")
        XCTAssertEqual(state, .needsDownload(alias: alias, sizeText: nil))
        XCTAssertEqual(state.detail, "It downloads once, then starts in seconds.")
    }

    func testDownloadingCarriesProgress() {
        let state = resolve(
            .starting(alias: alias),
            progress: .init(
                activity: .downloading,
                subtitle: "1.2 GB / 5.0 GB · 24% · 8.4 MB/s",
                fraction: 0.24
            )
        )
        XCTAssertEqual(
            state,
            .downloading(
                alias: alias,
                detail: "1.2 GB / 5.0 GB · 24% · 8.4 MB/s",
                fraction: 0.24
            )
        )
        XCTAssertEqual(state.progressFraction, 0.24)
        XCTAssertTrue(state.isWorking)
        XCTAssertEqual(state.statusRole, .working)
    }

    /// Loading / warming up are NOT downloading. The word "Downloading"
    /// may only appear when bytes are provably moving — the invariant
    /// ``DownloadProgress.startupActivity`` exists to protect.
    func testLoadingAndWarmingAreStartingNotDownloading() {
        for activity in [DownloadProgress.StartupActivity.loading, .warmingUp, .starting] {
            let state = resolve(
                .starting(alias: alias),
                progress: .init(activity: activity)
            )
            guard case .starting = state else {
                return XCTFail("\(activity) must resolve to .starting, got \(state)")
            }
            XCTAssertNil(state.progressFraction, "\(activity) must not show a determinate bar")
            XCTAssertFalse(
                state.headline.lowercased().contains("download"),
                "\(activity) must not say 'download'"
            )
        }
    }

    /// Only ``downloading`` gets a determinate bar. Everything else
    /// would be implying precision we do not have.
    func testOnlyDownloadingHasAProgressFraction() {
        for state in allStates {
            if case .downloading(_, _, let fraction) = state {
                XCTAssertEqual(state.progressFraction, fraction)
            } else {
                XCTAssertNil(state.progressFraction, "\(state) must not report progress")
            }
        }
    }

    func testReadyWhenServing() {
        let state = resolve(.ready(alias: alias))
        XCTAssertEqual(state, .ready(alias: alias))
        XCTAssertTrue(state.sendAllowed)
        XCTAssertEqual(state.statusRole, .ready)
        XCTAssertFalse(state.isWorking)
    }

    /// The full happy path, in order, asserting that send unlocks only
    /// at the final step.
    func testFullLifecycleSequence() {
        let steps: [(ServerState, Bool?, ModelReadiness)] = [
            (.idle, nil, .noModel),
            (.idle, false, .needsDownload(alias: alias, sizeText: nil)),
            (.ready(alias: alias), true, .ready(alias: alias)),
        ]
        for (serverState, cached, expected) in steps {
            let aliasForStep: String = {
                if case .noModel = expected { return "" }
                return alias
            }()
            let state = resolve(serverState, alias: aliasForStep, cached: cached)
            XCTAssertEqual(state, expected)
        }

        // The two in-flight steps need a progress snapshot to distinguish.
        let downloading = resolve(
            .starting(alias: alias),
            progress: .init(activity: .downloading, fraction: 0.5)
        )
        let starting = resolve(
            .starting(alias: alias),
            progress: .init(activity: .loading)
        )
        XCTAssertFalse(downloading.sendAllowed)
        XCTAssertFalse(starting.sendAllowed)
        XCTAssertTrue(resolve(.ready(alias: alias)).sendAllowed)
    }

    // MARK: - Status roles

    func testStatusRolesMapToTheFourTokens() {
        XCTAssertEqual(ModelReadiness.engineMissing.statusRole, .error)
        XCTAssertEqual(ModelReadiness.noModel.statusRole, .idle)
        XCTAssertEqual(ModelReadiness.needsStart(alias: alias).statusRole, .idle)
        XCTAssertEqual(
            ModelReadiness.needsDownload(alias: alias, sizeText: nil).statusRole, .idle)
        XCTAssertEqual(
            ModelReadiness.downloading(alias: alias, detail: nil, fraction: nil).statusRole,
            .working)
        XCTAssertEqual(ModelReadiness.starting(alias: alias, detail: nil).statusRole, .working)
        XCTAssertEqual(ModelReadiness.ready(alias: alias).statusRole, .ready)
        XCTAssertEqual(
            ModelReadiness.failed(alias: nil, message: "x", action: nil).statusRole, .error)
    }

    /// Only genuine faults take the error treatment. "You haven't
    /// started it yet" is not a fault and must not paint red.
    func testIsFailureIsReservedForFaults() {
        XCTAssertTrue(ModelReadiness.engineMissing.isFailure)
        XCTAssertTrue(ModelReadiness.failed(alias: nil, message: "x", action: nil).isFailure)
        XCTAssertFalse(ModelReadiness.noModel.isFailure)
        XCTAssertFalse(ModelReadiness.needsStart(alias: alias).isFailure)
        XCTAssertFalse(
            ModelReadiness.needsDownload(alias: alias, sizeText: nil).isFailure)
        XCTAssertFalse(ModelReadiness.ready(alias: alias).isFailure)
    }

    // MARK: - Actions

    /// ``chooseModel`` must not render a button: the picker already
    /// carries those exact words 40pt away, and a second control saying
    /// the same thing is the duplicate-action defect.
    func testChooseModelIsNamedButNotRendered() {
        let state = ModelReadiness.noModel
        XCTAssertEqual(state.action, .chooseModel)
        XCTAssertEqual(state.action?.isRenderable, false)
        XCTAssertEqual(state.action?.title, "Choose a model")
        XCTAssertNil(state.action?.alias)
    }

    func testActionableStatesRenderTheirButton() {
        let renderable: [ModelReadiness] = [
            .needsDownload(alias: alias, sizeText: nil),
            .needsStart(alias: alias),
            .failed(alias: alias, message: "x", action: .retry(alias: alias)),
        ]
        for state in renderable {
            guard let action = state.action else {
                return XCTFail("\(state) should offer an action")
            }
            XCTAssertTrue(action.isRenderable, "\(state) action should render")
            XCTAssertEqual(action.alias, alias)
            XCTAssertFalse(action.title.isEmpty)
            XCTAssertFalse(action.systemImage.isEmpty)
        }
    }

    /// In-flight states offer no action — progress is the answer, and a
    /// button there would either duplicate Stop or do nothing.
    func testInFlightStatesOfferNoAction() {
        XCTAssertNil(
            ModelReadiness.downloading(alias: alias, detail: nil, fraction: nil).action)
        XCTAssertNil(ModelReadiness.starting(alias: alias, detail: nil).action)
        XCTAssertNil(ModelReadiness.ready(alias: alias).action)
    }

    // MARK: - Terminology (one vocabulary across every surface)

    /// choose / download / start / ready — and no synonyms. The old
    /// Connect Tools header said "start a chat to generate the key"
    /// while its body said "Start a model"; this pins the vocabulary so
    /// that cannot come back.
    func testVocabularyIsConsistent() {
        let chooseCopy = ModelReadiness.noModel
        XCTAssertTrue(chooseCopy.headline.contains("chosen"))
        XCTAssertTrue(chooseCopy.detail?.contains("Choose a model") == true)
        XCTAssertTrue(chooseCopy.emptyStateSubtitle.hasPrefix("Choose a model"))
        XCTAssertTrue(chooseCopy.composerPlaceholder.contains("Choose a model"))
        XCTAssertTrue(chooseCopy.sendTooltip.contains("Choose a model"))

        let downloadCopy = ModelReadiness.needsDownload(alias: alias, sizeText: "5.0 GB")
        for text in [downloadCopy.composerPlaceholder,
                     downloadCopy.emptyStateSubtitle,
                     downloadCopy.sendTooltip] {
            XCTAssertTrue(text.contains("Download"), "expected 'Download' in: \(text)")
        }

        let startCopy = ModelReadiness.needsStart(alias: alias)
        for text in [startCopy.composerPlaceholder,
                     startCopy.emptyStateSubtitle,
                     startCopy.sendTooltip] {
            XCTAssertTrue(text.contains("Start"), "expected 'Start' in: \(text)")
        }

        // "select" / "pick" / "load" are the near-synonyms that caused
        // the drift. None may appear in readiness copy.
        let banned = ["select a model", "pick a model", "load a model", "start a chat"]
        for state in allStates {
            let corpus = [
                state.headline, state.detail ?? "", state.composerPlaceholder,
                state.sendTooltip, state.emptyStateSubtitle, state.emptyStateHint ?? "",
            ].joined(separator: " ").lowercased()
            for phrase in banned {
                XCTAssertFalse(corpus.contains(phrase), "\(state) uses banned phrase '\(phrase)'")
            }
        }
    }

    /// Every state that is ABOUT a specific model names it in the
    /// headline, so the user always knows which model the sentence
    /// refers to.
    func testHeadlineNamesTheModelWhenThereIsOne() {
        let named: [ModelReadiness] = [
            .needsDownload(alias: alias, sizeText: nil),
            .needsStart(alias: alias),
            .unknownModel(alias: alias),
            .downloading(alias: alias, detail: nil, fraction: nil),
            .starting(alias: alias, detail: nil),
            .ready(alias: alias),
        ]
        for state in named {
            XCTAssertTrue(
                state.headline.contains(alias),
                "headline should name the model: \(state.headline)"
            )
        }
    }

    /// The composer placeholder names the model only while it is BLOCKING
    /// the send — "Download <alias> first" has to say which model you are
    /// waiting on. Once ready it reverts to the neutral, approved "Send a
    /// message…": the model's identity is carried by the picker chip and
    /// the hero at that point, and repeating it in the field would be the
    /// same fact three times.
    ///
    /// The first draft of this suite asserted the alias in EVERY named
    /// state's placeholder, which contradicted ``testReadyIsQuiet`` — the
    /// implementation was right and the expectation was wrong.
    func testPlaceholderNamesTheModelOnlyWhileItBlocksSending() {
        let blocking: [ModelReadiness] = [
            .needsDownload(alias: alias, sizeText: nil),
            .needsStart(alias: alias),
            .downloading(alias: alias, detail: nil, fraction: nil),
            .starting(alias: alias, detail: nil),
        ]
        for state in blocking {
            XCTAssertFalse(state.sendAllowed)
            XCTAssertTrue(
                state.composerPlaceholder.contains(alias),
                "blocking placeholder should name the model: \(state.composerPlaceholder)"
            )
        }
        XCTAssertEqual(
            ModelReadiness.ready(alias: alias).composerPlaceholder,
            "Send a message…"
        )
    }

    /// The empty state's two approved strings survive verbatim — the
    /// visual redesign signed off on this copy and Phase 1 is not a
    /// licence to rewrite it.
    func testApprovedEmptyStateCopyIsPreserved() {
        XCTAssertEqual(ModelReadiness.noModel.emptyStateSubtitle, "Choose a model to start")
        XCTAssertEqual(
            ModelReadiness.ready(alias: alias).emptyStateSubtitle,
            "Chatting with \(alias)"
        )
        XCTAssertEqual(
            ModelReadiness.starting(alias: alias, detail: nil).emptyStateSubtitle,
            "Preparing your local model…"
        )
    }

    // MARK: - Accessibility

    /// The composed VoiceOver label must carry both halves, so a screen
    /// reader user gets the same information a sighted user reads.
    func testAccessibilityLabelComposesHeadlineAndDetail() {
        let state = ModelReadiness.needsDownload(alias: alias, sizeText: "5.0 GB")
        let label = state.accessibilityLabel
        XCTAssertTrue(label.contains(state.headline))
        XCTAssertTrue(label.contains(state.detail ?? "<missing>"))
    }

    func testAccessibilityLabelIsNeverEmpty() {
        for state in allStates {
            XCTAssertFalse(
                state.accessibilityLabel.trimmingCharacters(in: .whitespaces).isEmpty,
                "\(state) has an empty accessibility label"
            )
        }
    }

    // MARK: - #1505 follow-up: serve-state must not describe a foreign selection

    /// The Send-enabling `.ready` state may only describe the SELECTED
    /// model. A user who picks B while A is still serving must NOT get a
    /// bright Send button wired to a model that isn't running — resolve B's
    /// own state instead, so the composer offers B's Start and Send stays
    /// gated. (Regression: `.ready` previously described the serving alias
    /// unconditionally, enabling a send that ``ChatView`` would dispatch
    /// against the un-running selection.)
    func testReadyForADifferentModelResolvesTheSelection() {
        let other = "bonsai-1.7b-2bit"
        let state = resolve(.ready(alias: alias), alias: other, cached: true)
        XCTAssertEqual(state, .needsStart(alias: other))
        XCTAssertFalse(state.sendAllowed, "Send must not enable for a model that isn't the one serving")

        let cold = resolve(.ready(alias: alias), alias: other, cached: false)
        XCTAssertEqual(cold, .needsDownload(alias: other, sizeText: nil))
        XCTAssertFalse(cold.sendAllowed)
    }

    /// The launch pre-sync frame: the child is serving but the picker
    /// breadcrumb hasn't synced yet (empty selection). `.ready` must not win
    /// with an empty/foreign alias — Send stays gated until a real model is
    /// actually selected, which the `onChange(of: server.state)` sync does a
    /// frame later.
    func testReadyWithNoSelectionYetIsNotSendable() {
        let state = resolve(.ready(alias: alias), alias: "", cached: true)
        XCTAssertFalse(state.sendAllowed)
        XCTAssertEqual(state, .noModel)
    }

    /// The matching case is untouched: serving == selected → ready.
    func testReadyForTheSelectedModelStaysReady() {
        XCTAssertEqual(resolve(.ready(alias: alias), alias: alias), .ready(alias: alias))
        XCTAssertTrue(readyDescribesSelectionRef(serving: alias, selected: alias))
        XCTAssertFalse(readyDescribesSelectionRef(serving: alias, selected: "bonsai-1.7b-2bit"))
        XCTAssertFalse(readyDescribesSelectionRef(serving: alias, selected: ""))
    }

    /// `.starting` is permissive (it never enables Send): a not-yet-synced
    /// selection at launch keeps the in-flight start visible, but a
    /// deliberate pick of a DIFFERENT real model resolves that model.
    func testStartingIsPermissiveExceptForADifferentRealModel() {
        // Launch: breadcrumb lags the auto-started model — still show it.
        let launch = resolve(.starting(alias: alias), alias: "")
        if case .starting = launch {} else {
            XCTFail("a starting child with no selection yet must still show as starting, got \(launch)")
        }
        // Same model selected — unchanged.
        if case .starting = resolve(.starting(alias: alias), alias: alias) {} else {
            XCTFail("starting the selected model must show as starting")
        }
        // A different real model deliberately selected — resolve it.
        let other = "bonsai-1.7b-2bit"
        XCTAssertEqual(
            resolve(.starting(alias: alias), alias: other, cached: true),
            .needsStart(alias: other)
        )
        // codex r2: a PLACEHOLDER start (engine reports no alias) while a
        // real model B is selected must not claim "Starting B" and swallow
        // B's Start — we can't prove the start is B's, so resolve B.
        XCTAssertEqual(
            resolve(.starting(alias: ""), alias: other, cached: true),
            .needsStart(alias: other)
        )
        // …but a placeholder start with NOTHING selected stays visible.
        if case .starting = resolve(.starting(alias: ""), alias: "") {} else {
            XCTFail("a placeholder start with no selection must still show as starting")
        }
    }

    /// A turn-level error is shown in the ready composer only when it
    /// belongs to the current model (or carries no alias at all). Switching
    /// to a healthy model must not inherit the previous model's error.
    func testTurnErrorScopedToSelection() {
        XCTAssertTrue(
            ModelReadiness.turnErrorApplies(failureAlias: nil, selectedAlias: alias),
            "an unattributed turn error is about the current turn and must show"
        )
        XCTAssertTrue(
            ModelReadiness.turnErrorApplies(failureAlias: alias, selectedAlias: alias)
        )
        XCTAssertFalse(
            ModelReadiness.turnErrorApplies(failureAlias: crashed, selectedAlias: alias),
            "a prior model's turn error must not leak onto a freshly selected model"
        )
    }

    /// Local ref so the strict predicate is exercised directly, not only
    /// through the eight-case resolve matrix.
    private func readyDescribesSelectionRef(serving: String, selected: String) -> Bool {
        ModelReadiness.readyDescribesSelection(serving: serving, selected: selected)
    }
}

/// The detail-pane branch. Reduced to the one thing it decides now that
/// ``ModelReadiness`` owns readiness.
///
/// ``@MainActor`` because ``ContentView`` is a SwiftUI ``View``, which
/// Swift 6 infers as main-actor-isolated — including its statics.
@MainActor
final class MainAreaBranchTests: XCTestCase {
    func testOnlyMissingLeavesTheChatSurface() {
        XCTAssertEqual(ContentView.mainAreaBranch(for: .missing), .missing)
        for state: ServerState in [
            .idle,
            .stopped,
            .starting(alias: "a"),
            .ready(alias: "a"),
            .crashed(alias: "a", message: "boom"),
        ] {
            XCTAssertEqual(
                ContentView.mainAreaBranch(for: state), .chat,
                "\(state) should keep the chat surface mounted"
            )
        }
    }
}
