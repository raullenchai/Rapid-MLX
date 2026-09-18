import Foundation

/// Everything a successful upload changes, held in one value the view owns.
///
/// This is production state, not a test helper. The publish path used to write
/// these facts straight into scattered `@State` properties, which made two
/// real defects invisible:
///
/// 1. The contributor was taken from `receipts`, and `receipts` is only written
///    when the CLI managed to save the receipt locally. An upload that
///    succeeded but whose local write failed therefore left the session with no
///    identity at all — no portrait, no profile link, and no slug to ask the
///    server for the contributor's totals with.
/// 2. `/api/benchmarks/atomic/public` is edge-cached for 30 seconds. The
///    refresh fired immediately after publishing usually reads a feed that
///    predates the submission, so the count it returns is one *lower* than the
///    count just confirmed by the receipt — and it overwrote it, so the number
///    visibly went 7 → 8 → 7.
struct CommunityPublicationState: Equatable, Sendable {
    /// The pseudonym the service issued, kept for the whole session even when
    /// the local receipt could not be written. The server assigned it; losing
    /// it because of a local disk problem would be the client's own mistake.
    private(set) var sessionContributor: CommunityBenchmarkContributor?

    /// Server-confirmed receipts retained for this app session even when the
    /// CLI could not persist its local receipt file. This keeps the accepted
    /// run visibly Published and prevents an accidental second publish offer.
    private(set) var sessionReceipts: [String: CommunityBenchmarkReceipt] = [:]

    /// Counts this installation *knows* are at least true, because the server
    /// accepted its submissions into those scopes — **one per scope**.
    ///
    /// This used to be a single optional floor, cleared whenever
    /// `selectedAlias` changed. Publishing a result, glancing at another model
    /// and coming back therefore threw away the only proof the session had
    /// that its own submission existed, and the still-cached feed then walked
    /// the number back down. A floor is a fact about a scope; another scope
    /// becoming visible is not evidence against it.
    private(set) var floors: [CommunityBenchmarkScope: ConfirmedFloor] = [:]

    struct ConfirmedFloor: Equatable, Sendable {
        let scope: CommunityBenchmarkScope
        let count: Int
        let unit: String?
        let isBounded: Bool
        let confirmedAt: Date
    }

    /// The floor defending one scope, if this session confirmed one.
    func confirmedFloor(for scope: CommunityBenchmarkScope?) -> ConfirmedFloor? {
        guard let scope else { return nil }
        return floors[scope]
    }

    /// How long `/atomic/public` may serve a cached body. Mirrors
    /// `ATOMIC_BENCH_PUBLIC_CACHE_SECONDS` in the worker.
    static let publicFeedCacheSeconds: TimeInterval = 30

    /// The reads served by `/api/benchmarks/atomic/public`, and therefore the
    /// reads a 30-second cached body can make stale *together*.
    ///
    /// Refreshing only the observation count after the cache expired left the
    /// Community table, the coverage list and the pulse band on pre-publish
    /// numbers — including the table row that should have gained
    /// "INCLUDES YOURS" — until something else happened to re-read them.
    /// `contributorTotals` is deliberately absent: it comes from the
    /// cursor-paginated contributions route, which this cache does not serve.
    static let publicFeedBackedReads: Set<CommunityReadKind> = [
        .observations, .table, .coverage, .pulse,
    ]

    /// Floors kept at once. A session that publishes more than this has long
    /// since seen fresh reads for the oldest scopes; the cap only stops an
    /// unbounded dictionary.
    static let maximumFloors = 16

    // MARK: - Applying a receipt

    /// Result of recording one upload, so the caller knows what to do next.
    struct Outcome: Equatable, Sendable {
        /// The post-publish observation state **for the scope that was
        /// published to**, which is not necessarily what is on screen.
        var observations: CommunityDataState<CommunityObservationSummary>
        /// Set when the public result exists but the local receipt does not.
        var receiptNotSavedWarning: String?
        /// When the next read may see the submission. Nil when no retry is
        /// needed (a duplicate changes nothing).
        var retryAfter: Date?
        /// Whether ``observations`` may be written to the screen.
        ///
        /// False when the visible result changed while the upload was in
        /// flight — the number describes the run that was published, and
        /// painting it next to a different run's metrics would attribute one
        /// run's community standing to another. The floor is still recorded
        /// against the published scope, so navigating back shows it.
        var appliesToVisibleScope: Bool = true
    }

    /// Records a successful upload.
    ///
    /// `alreadyExists` is the important branch: a duplicate submission did not
    /// add anything to the corpus, so the count must not move and no floor is
    /// established — otherwise republishing the same run would inflate the
    /// number once per attempt.
    mutating func recordPublication(
        receipt: CommunityBenchmarkReceipt,
        receiptSaved: Bool,
        runID: String? = nil,
        scope: CommunityBenchmarkScope?,
        observations: CommunityDataState<CommunityObservationSummary>,
        now: Date = Date()
    ) -> Outcome {
        // Adopt the identity regardless of the local write. It came from the
        // server; the local receipt is only a cache of it.
        if let contributor = receipt.contributor {
            sessionContributor = contributor
        }
        if let runID {
            sessionReceipts[runID] = receipt
        }

        var outcome = Outcome(observations: observations)
        if !receiptSaved {
            outcome.receiptNotSavedWarning = String(
                localized: "Published successfully, but this Mac couldn’t save its local receipt. Rapid will keep this run marked Published for this session; the public version is correct."
            )
        }

        guard !receipt.alreadyExists else {
            // Nothing entered the corpus. No increment, no floor, no retry.
            return outcome
        }

        guard let scope, let current = observations.value else {
            // The count was never known, so there is nothing to increment and
            // nothing to defend against a stale read.
            return outcome
        }

        let advanced = current.incrementedAfterPublishing()
        floors[scope] = ConfirmedFloor(
            scope: scope,
            count: advanced.observationCount,
            unit: advanced.unit,
            isBounded: advanced.isBounded,
            confirmedAt: now
        )
        evictOldestFloorsIfNeeded()
        outcome.observations = .ready(advanced)
        outcome.retryAfter = now.addingTimeInterval(Self.publicFeedCacheSeconds)
        return outcome
    }

    private mutating func evictOldestFloorsIfNeeded() {
        guard floors.count > Self.maximumFloors else { return }
        let ordered = floors.sorted { $0.value.confirmedAt < $1.value.confirmedAt }
        for (scope, _) in ordered.prefix(floors.count - Self.maximumFloors) {
            floors.removeValue(forKey: scope)
        }
    }

    /// Records a successful upload against the context frozen when the user
    /// pressed Publish, and reports whether the result may be displayed.
    ///
    /// This is the fix for the publish-context race. `share(_:)` awaits a CLI
    /// subprocess that routinely takes seconds, and during that await **Run
    /// again**, **Benchmark another**, and the model picker can all change
    /// which result is on screen. Reading the scope and the count *after* the
    /// await therefore applied run A's receipt to run B's scope: B's count was
    /// incremented, B's aggregate was marked "includes yours", and the floor
    /// that defends the count against the edge cache was pinned to a scope
    /// nothing had been published to.
    ///
    /// Passing the captured context in makes that unrepresentable — the receipt
    /// can only ever be applied to the scope it is about.
    mutating func recordPublication(
        receipt: CommunityBenchmarkReceipt,
        receiptSaved: Bool,
        context: CommunityPublicationContext,
        visibleScope: CommunityBenchmarkScope?,
        now: Date = Date()
    ) -> Outcome {
        var outcome = recordPublication(
            receipt: receipt,
            receiptSaved: receiptSaved,
            runID: context.runID,
            scope: context.scope,
            observations: context.observations,
            now: now
        )
        // A nil published scope has nothing to match, so it never claims the
        // screen even if the screen also has no scope.
        outcome.appliesToVisibleScope = context.scope != nil && context.scope == visibleScope
        return outcome
    }

    // MARK: - Merging later reads

    /// Folds an incoming observation read into what this session already knows.
    ///
    /// Monotonic by construction for the scope that was published to: a read
    /// that comes back *below* the confirmed floor is a stale cached feed, not
    /// a correction, so the confirmed number is kept. A read at or above the
    /// floor is fresh enough to have seen the submission, so the floor is
    /// retired and the server becomes authoritative again.
    ///
    /// Any other scope passes through untouched — the floor only ever defends
    /// the exact scope that was published to.
    mutating func merge(
        _ incoming: CommunityDataState<CommunityObservationSummary>,
        scope: CommunityBenchmarkScope?,
        now: Date = Date()
    ) -> CommunityDataState<CommunityObservationSummary> {
        guard let scope, let floor = floors[scope] else { return incoming }

        switch incoming {
        case .loading:
            return incoming
        case .unavailable:
            // The service cannot answer, but this installation still knows its
            // own submission was accepted. Reverting to "unknown" would be
            // forgetting a fact we hold a receipt for.
            return .ready(floor.summary)
        case let .ready(value):
            if value.observationCount >= floor.count {
                // Fresh enough to include the submission. Only this scope's
                // floor retires — a fresh read about A says nothing about B.
                floors.removeValue(forKey: scope)
                return .ready(value)
            }
            // Stale cached body. Keep the confirmed count; the caller retries
            // once the edge cache can have expired.
            return .ready(floor.summary(mergingMedianFrom: value))
        }
    }

    /// Whether the edge cache can have expired since this scope's publication.
    func mayRetry(_ scope: CommunityBenchmarkScope?, at now: Date) -> Bool {
        guard let floor = confirmedFloor(for: scope) else { return false }
        return now.timeIntervalSince(floor.confirmedAt) >= Self.publicFeedCacheSeconds
    }

    /// Whether any scope is waiting on a post-cache re-read.
    func mayRetryAnything(at now: Date) -> Bool {
        floors.values.contains {
            now.timeIntervalSince($0.confirmedAt) >= Self.publicFeedCacheSeconds
        }
    }

    /// Drops one scope's floor. Not called on navigation — a floor is a fact
    /// about a scope, and looking at a different model is not evidence against
    /// it. Retained for an explicit reset.
    mutating func clearFloor(for scope: CommunityBenchmarkScope) {
        floors.removeValue(forKey: scope)
    }

    mutating func clearAllFloors() { floors.removeAll() }
}

/// The publish context, frozen at the moment the user presses Publish.
///
/// Everything here describes the run being uploaded. Nothing in it is re-read
/// after the upload starts, which is the entire point: the view's `@State` is
/// free to move on to another run while the CLI is still talking to the
/// service.
struct CommunityPublicationContext: Equatable, Sendable {
    /// The run the receipt will be about.
    let runID: String
    /// The scope that run belongs to — model, workload, protocol, this Mac,
    /// and the run's own case/metric/execution identity.
    let scope: CommunityBenchmarkScope?
    /// The observation state for `scope` when Publish was pressed.
    let observations: CommunityDataState<CommunityObservationSummary>
    /// The contribution branch the user actually saw, so the celebration
    /// cannot upgrade itself to a first-reference claim from a count that
    /// arrived afterwards.
    let branch: CommunityContributionBranch

    /// Freezes the context.
    ///
    /// `visibleScope` is the scope the displayed count belongs to. When it is
    /// not the scope being published — which a fast Run again can already have
    /// made true before the first `await` — the count is deliberately **not**
    /// carried over: it describes a different population, and using it would
    /// increment the wrong number. `CommunityPublicationState` then records the
    /// identity and skips the increment, which is the honest outcome when the
    /// pre-publish count for this scope is unknown.
    static func capture(
        runID: String,
        resultScope: CommunityBenchmarkScope?,
        visibleScope: CommunityBenchmarkScope?,
        observations: CommunityDataState<CommunityObservationSummary>,
        branch: CommunityContributionBranch
    ) -> Self {
        let matches = resultScope != nil && resultScope == visibleScope
        return Self(
            runID: runID,
            scope: resultScope,
            observations: matches ? observations : .unavailable(.boundedFeed),
            branch: matches ? branch : .unknown(.unavailable(.boundedFeed))
        )
    }
}

extension CommunityPublicationState.ConfirmedFloor {
    var summary: CommunityObservationSummary {
        CommunityObservationSummary(
            observationCount: count,
            median: nil,
            unit: unit,
            includesYours: true,
            isBounded: isBounded
        )
    }

    /// Keeps the confirmed count but adopts whatever statistics the (stale)
    /// server body carried, so the comparison area is not blanked while the
    /// cache drains.
    func summary(mergingMedianFrom stale: CommunityObservationSummary) -> CommunityObservationSummary {
        CommunityObservationSummary(
            observationCount: count,
            median: stale.median,
            observedMinimum: stale.observedMinimum,
            observedMaximum: stale.observedMaximum,
            unit: stale.unit ?? unit,
            includesYours: true,
            isBounded: isBounded || stale.isBounded
        )
    }
}
