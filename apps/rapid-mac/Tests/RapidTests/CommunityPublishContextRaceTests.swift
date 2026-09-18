import Foundation
import Testing
@testable import Rapid

/// A receipt may only ever be applied to the run it is about.
///
/// `share(_:)` awaits a CLI subprocess that talks to rapidmlx.com — seconds,
/// routinely. During that await the Result screen's **Run again**, **Benchmark
/// another** and the model picker could all change `latestResultID` and the
/// selected model. The old code read `activeObservationScope` and
/// `observations` *after* the await, so run A's receipt was applied to whatever
/// was on screen when it landed: B's count was incremented, B's aggregate was
/// marked "includes yours", and the stale-feed floor — the thing that defends a
/// confirmed count for 30 seconds — was pinned to a scope nothing had been
/// published to.
///
/// The fix is to freeze the context before the first await, which these tests
/// drive through the production types: `CommunityPublicationContext.capture`
/// and `CommunityPublicationState.recordPublication(receipt:receiptSaved:
/// context:visibleScope:)`.
@Suite("Publish context race")
struct CommunityPublishContextRaceTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let publishedAt = Date(timeIntervalSince1970: 1_789_000_000)

    private static func execution(_ dtype: String) -> CommunityExecutionIdentity {
        CommunityExecutionIdentity(
            rapidMLX: "0.13.4", computeDType: dtype,
            speculativeDecodingMethod: "none", kvCacheMode: "quantized",
            kvCacheDType: "int8", prefillBackend: "gpu"
        )
    }

    /// Run A and run B are the same model on the same Mac — they differ only in
    /// how they were executed, which is exactly the case a coarser check would
    /// miss.
    private static func scope(_ dtype: String, alias: String = "qwen3.5-9b-4bit") -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias, workload: .llm, protocolID: "rapid-community-speed",
            protocolVersion: 2, macProfile: profile,
            comparison: CommunityComparisonIdentity(
                caseID: "pp512-tg128", metricName: "decode_tps", execution: execution(dtype)
            )
        )
    }

    private static let scopeA = scope("bf16")
    private static let scopeB = scope("fp16")

    private static func receipt(alreadyExists: Bool = false) -> CommunityBenchmarkReceipt {
        try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"sub-1","already_exists":\#(alreadyExists),
             "accepted_at":"2026-09-06T04:40:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
    }

    private static func ready(_ count: Int) -> CommunityDataState<CommunityObservationSummary> {
        .ready(
            CommunityObservationSummary(
                observationCount: count, unit: "tok/s", isBounded: true
            )
        )
    }

    /// Stands in for the view's `@State`. The upload task holds no reference to
    /// it after capture — which is the property under test.
    private actor Screen {
        var visibleScope: CommunityBenchmarkScope?
        var observations: CommunityDataState<CommunityObservationSummary>

        init(scope: CommunityBenchmarkScope?, observations: CommunityDataState<CommunityObservationSummary>) {
            visibleScope = scope
            self.observations = observations
        }

        /// What "Run again" does: clears the result, then a new one arrives.
        func runAgain(producing scope: CommunityBenchmarkScope?, count: Int) {
            visibleScope = scope
            observations = .ready(
                CommunityObservationSummary(observationCount: count, unit: "tok/s", isBounded: true)
            )
        }

        func display(_ value: CommunityDataState<CommunityObservationSummary>) {
            observations = value
        }
    }

    /// Lets the test hold the "upload" open across an await point, so the
    /// interleaving is deterministic rather than a matter of scheduler luck.
    private actor UploadGate {
        private var continuation: CheckedContinuation<Void, Never>?
        private var released = false

        func wait() async {
            if released { return }
            await withCheckedContinuation { continuation = $0 }
        }

        func release() {
            released = true
            continuation?.resume()
            continuation = nil
        }
    }

    // MARK: - The race itself

    @Test("A delayed upload cannot update the scope the user navigated to")
    func delayedUploadCannotUpdateTheWrongScope() async {
        let screen = Screen(scope: Self.scopeA, observations: Self.ready(7))
        let gate = UploadGate()
        let before = CommunityPublicationState()

        // The user presses Publish on run A. The context is frozen here,
        // before any await — exactly where `share(_:)` freezes it.
        let context = CommunityPublicationContext.capture(
            runID: "run-a",
            resultScope: Self.scopeA,
            visibleScope: await screen.visibleScope,
            observations: await screen.observations,
            branch: CommunityContributionBranch.select(from: await screen.observations)
        )

        let upload = Task {
            () -> (CommunityPublicationState.Outcome, CommunityPublicationState) in
            // The CLI is talking to the service…
            await gate.wait()
            // …and only now does the receipt come back.
            var state = before
            let outcome = state.recordPublication(
                receipt: Self.receipt(), receiptSaved: true,
                context: context, visibleScope: await screen.visibleScope,
                now: Self.publishedAt
            )
            if outcome.appliesToVisibleScope {
                await screen.display(outcome.observations)
            }
            return (outcome, state)
        }

        // Meanwhile the user hits Run again and a second, differently executed
        // run finishes and takes over the screen.
        await screen.runAgain(producing: Self.scopeB, count: 3)
        await gate.release()
        let (outcome, publication) = await upload.value

        // The receipt was about run A, so it must not touch run B's number.
        #expect(!outcome.appliesToVisibleScope)
        #expect(
            await screen.observations.value?.observationCount == 3,
            "run A's receipt incremented run B's count"
        )
        #expect(await screen.observations.value?.includesYours == false)

        // The publication itself is still recorded — against A.
        #expect(outcome.observations.value?.observationCount == 8)
        #expect(publication.confirmedFloor(for: Self.scopeA) != nil)
        #expect(publication.confirmedFloor(for: Self.scopeA)?.count == 8)
        // The pseudonym is session-wide and is adopted regardless.
        #expect(publication.sessionContributor?.slug == "swift-otter-4417")
    }

    @Test("The floor defends run A's scope, not the scope that was on screen")
    func floorFollowsThePublishedScope() async {
        let screen = Screen(scope: Self.scopeA, observations: Self.ready(7))
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA,
            visibleScope: await screen.visibleScope,
            observations: await screen.observations,
            branch: .strengthen(observationCount: 7, isAtLeast: true)
        )
        await screen.runAgain(producing: Self.scopeB, count: 3)

        var publication = CommunityPublicationState()
        _ = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, context: context,
            visibleScope: await screen.visibleScope, now: Self.publishedAt
        )

        // B's refresh reads 3 and is left alone.
        #expect(
            publication.merge(
                Self.ready(3), scope: Self.scopeB,
                now: Self.publishedAt.addingTimeInterval(1)
            ).value?.observationCount == 3
        )
        // Navigating back to A shows the confirmed 8 even though the edge cache
        // is still serving the pre-publish body.
        #expect(
            publication.merge(
                Self.ready(7), scope: Self.scopeA,
                now: Self.publishedAt.addingTimeInterval(1)
            ).value?.observationCount == 8
        )
    }

    @Test("Publishing while the screen has not moved still updates it")
    func unchangedScreenStillUpdates() async {
        let screen = Screen(scope: Self.scopeA, observations: Self.ready(7))
        let gate = UploadGate()
        let before = CommunityPublicationState()
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA,
            visibleScope: await screen.visibleScope,
            observations: await screen.observations,
            branch: .strengthen(observationCount: 7, isAtLeast: true)
        )

        let upload = Task { () -> CommunityPublicationState.Outcome in
            await gate.wait()
            var state = before
            let outcome = state.recordPublication(
                receipt: Self.receipt(), receiptSaved: true, context: context,
                visibleScope: await screen.visibleScope, now: Self.publishedAt
            )
            if outcome.appliesToVisibleScope { await screen.display(outcome.observations) }
            return outcome
        }
        await gate.release()
        let outcome = await upload.value

        // The normal path is unaffected by the guard.
        #expect(outcome.appliesToVisibleScope)
        #expect(await screen.observations.value?.observationCount == 8)
        #expect(await screen.observations.value?.includesYours == true)
    }

    // MARK: - What `capture` freezes

    @Test("The captured scope is the published run's, not the visible one")
    func captureUsesTheResultScope() {
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA, visibleScope: Self.scopeA,
            observations: Self.ready(7), branch: .strengthen(observationCount: 7, isAtLeast: true)
        )
        #expect(context.runID == "run-a")
        #expect(context.scope == Self.scopeA)
        #expect(context.scope?.comparison?.execution.computeDType == "bf16")
        #expect(context.observations.value?.observationCount == 7)
    }

    @Test("A count belonging to another scope is not carried into the context")
    func captureRefusesAMismatchedCount() {
        // The screen had already moved on before Publish was even dispatched —
        // the narrow version of the same race.
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA, visibleScope: Self.scopeB,
            observations: Self.ready(3), branch: .strengthen(observationCount: 3, isAtLeast: true)
        )
        #expect(context.scope == Self.scopeA)
        // Incrementing 3 would publish a number that describes run B.
        #expect(context.observations.value == nil)
        #expect(!context.branch.allowsFirstReferenceLanguage)

        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, context: context,
            visibleScope: Self.scopeB, now: Self.publishedAt
        )
        // Identity yes; a fabricated count no.
        #expect(publication.sessionContributor?.slug == "swift-otter-4417")
        #expect(outcome.observations.value == nil)
        #expect(publication.floors.isEmpty)
        #expect(!outcome.appliesToVisibleScope)
    }

    @Test("A context with no scope never claims the screen")
    func noScopeNeverApplies() {
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: nil, visibleScope: nil,
            observations: Self.ready(7), branch: .strengthen(observationCount: 7)
        )
        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, context: context,
            visibleScope: nil, now: Self.publishedAt
        )
        // Two unknowns are not a match.
        #expect(!outcome.appliesToVisibleScope)
        #expect(publication.floors.isEmpty)
    }

    @Test("The celebration describes the published run even after the screen moves")
    func celebrationUsesTheCapturedContext() async {
        let screen = Screen(scope: Self.scopeA, observations: Self.ready(7))
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA,
            visibleScope: await screen.visibleScope,
            observations: await screen.observations,
            branch: CommunityContributionBranch.select(from: await screen.observations)
        )
        await screen.runAgain(producing: Self.scopeB, count: 0)

        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, context: context,
            visibleScope: await screen.visibleScope, now: Self.publishedAt
        )
        // The sheet reads the captured branch, captured scope and the outcome's
        // count — never live page state, which now describes run B.
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: context.branch,
            scope: context.scope!,
            observationCountAfterPublishing: outcome.observations.value?.observationCount,
            alreadyPublished: false
        )
        #expect(!celebration.headline.isEmpty)
        // Run B's screen count is 0; the celebration must not read it and
        // congratulate a first reference that never happened.
        #expect(!context.branch.allowsFirstReferenceLanguage)
        #expect(celebration.headline != "You published the first result")
    }

    @Test("A duplicate receipt on a moved screen changes nothing anywhere")
    func duplicateOnAMovedScreen() async {
        let screen = Screen(scope: Self.scopeA, observations: Self.ready(7))
        let context = CommunityPublicationContext.capture(
            runID: "run-a", resultScope: Self.scopeA,
            visibleScope: await screen.visibleScope,
            observations: await screen.observations,
            branch: .strengthen(observationCount: 7, isAtLeast: true)
        )
        await screen.runAgain(producing: Self.scopeB, count: 3)

        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(alreadyExists: true), receiptSaved: true,
            context: context, visibleScope: await screen.visibleScope, now: Self.publishedAt
        )
        #expect(!outcome.appliesToVisibleScope)
        #expect(outcome.observations.value?.observationCount == 7)
        #expect(publication.floors.isEmpty)
        #expect(await screen.observations.value?.observationCount == 3)
    }
}
