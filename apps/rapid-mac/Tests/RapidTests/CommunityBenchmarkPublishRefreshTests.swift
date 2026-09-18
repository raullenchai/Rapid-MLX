import Foundation
import Testing
@testable import Rapid

/// What a successful publication changes, and what it must not.
///
/// These drive `CommunityPublicationState` — the value the view actually owns
/// and mutates on the publish path — rather than a local re-statement of its
/// rules. A test that models the rule can only ever agree with itself; the two
/// defects below both lived in the real transition.
///
/// 1. The contributor was read out of `receipts`, which is only written when
///    the CLI managed to save the receipt to disk. An upload that succeeded
///    with a failed local write left the session with no pseudonym, hence no
///    portrait and no slug to request contributor totals with.
/// 2. `/api/benchmarks/atomic/public` is edge-cached for 30 s
///    (`ATOMIC_BENCH_PUBLIC_CACHE_SECONDS`), so the refresh fired immediately
///    after publishing usually reads a body that predates the submission. Its
///    count is one *lower* than the receipt just confirmed, and it overwrote
///    it: the number visibly went 7 → 8 → 7.
@Suite("Post-publish state")
struct CommunityBenchmarkPublishRefreshTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let scope = CommunityBenchmarkScope(
        modelAlias: "qwen3.5-9b-4bit", workload: .llm,
        protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: profile
    )
    private static let otherScope = CommunityBenchmarkScope(
        modelAlias: "gemma-4-12b-4bit", workload: .llm,
        protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: profile
    )
    private static let publishedAt = Date(timeIntervalSince1970: 1_789_000_000)

    private static func receipt(
        alreadyExists: Bool,
        contributor: Bool = true
    ) -> CommunityBenchmarkReceipt {
        let identity = contributor
            ? #""contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}"#
            : #""contributor":null"#
        return try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data("""
            {"submission_id":"sub-1","already_exists":\(alreadyExists),
             "accepted_at":"2026-09-06T04:40:00Z",\(identity)}
            """.utf8)
        )
    }

    private static func ready(_ count: Int, median: Double? = nil) -> CommunityDataState<CommunityObservationSummary> {
        .ready(
            CommunityObservationSummary(
                observationCount: count, median: median, unit: "tok/s", isBounded: true
            )
        )
    }

    // MARK: - Identity

    @Test("A first publication adopts the server-issued pseudonym")
    func firstPublicationAdoptsIdentity() {
        var state = CommunityPublicationState()
        #expect(state.sessionContributor == nil)

        let outcome = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(0), now: Self.publishedAt
        )
        #expect(state.sessionContributor?.slug == "swift-otter-4417")
        #expect(outcome.observations.value?.observationCount == 1)
        #expect(outcome.observations.value?.includesYours == true)
        // The count is no longer zero, so Ready must stop inviting a first
        // reference.
        #expect(
            CommunityContributionBranch.select(from: outcome.observations)
                != .firstReference
        )
    }

    @Test("A failed local receipt write does not discard the identity")
    func identitySurvivesAFailedLocalWrite() {
        var state = CommunityPublicationState()
        let outcome = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: false, runID: "run-123",
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // The pseudonym came from the server. Losing it because this Mac could
        // not write a file would be the client's own mistake.
        #expect(state.sessionContributor?.slug == "swift-otter-4417")
        #expect(state.sessionContributor?.name == "swift-otter")
        #expect(state.sessionReceipts["run-123"]?.submissionID == "sub-1")
        // The upload still happened, so the aggregate still moves…
        #expect(outcome.observations.value?.observationCount == 8)
        // …and the missing local copy is surfaced rather than leaving My
        // Results silently disagreeing with the server.
        #expect(outcome.receiptNotSavedWarning != nil)
    }

    @Test("A duplicate still adopts the identity it was issued")
    func duplicateAdoptsIdentity() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: true), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // The run was already public, but this install may be seeing its
        // pseudonym for the first time.
        #expect(state.sessionContributor?.slug == "swift-otter-4417")
    }

    @Test("A receipt with no contributor leaves the session unidentified")
    func noContributorMeansNoIdentity() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false, contributor: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // Never invented locally — a fabricated pseudonym would not resolve on
        // the website.
        #expect(state.sessionContributor == nil)
    }

    @Test("An identity once issued is not replaced by a later contributor-less receipt")
    func identityIsNotClearedByALaterReceipt() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(1), now: Self.publishedAt
        )
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false, contributor: false), receiptSaved: true,
            scope: Self.otherScope, observations: Self.ready(1),
            now: Self.publishedAt.addingTimeInterval(120)
        )
        #expect(state.sessionContributor?.slug == "swift-otter-4417")
    }

    /// The point of keeping the identity: the slug is what the totals request
    /// is made with. This drives the production adapter and inspects the URL
    /// it built.
    @Test("Totals are fetched with the server-issued slug from the receipt")
    func totalsUseTheServerIssuedSlug() async {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: false,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        let slug = try! #require(state.sessionContributor?.slug)

        let requested = RequestLog()
        let directory = CommunityBenchmarkAPIDirectory(
            transport: { request in
                await requested.record(request.url!)
                let body = #"""
                {"schema_version":1,"complete":true,"cursor":null,"runs":[
                  {"schema_version":1,"submission_id":"r1",
                   "accepted_at":"2026-09-06T04:40:00Z",
                   "task_type":"text_generation",
                   "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
                   "machine":{"chip":"Apple M3 Pro","memory_gib":18},
                   "execution":{"rapid_mlx":"0.13.4"},
                   "protocol":{"id":"rapid-community-speed","version":2},"cases":[]},
                  {"schema_version":1,"submission_id":"r2",
                   "accepted_at":"2026-09-01T04:40:00Z",
                   "task_type":"image_generation",
                   "model":{"repo_id":"mlx-community/z-image-turbo"},
                   "machine":{"chip":"Apple M3 Pro","memory_gib":18},
                   "execution":{"rapid_mlx":"0.13.4"},
                   "protocol":{"id":"rapid-image-speed","version":1},"cases":[]}
                ]}
                """#
                return (
                    Data(body.utf8),
                    HTTPURLResponse(
                        url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
                    )!
                )
            },
            aliasForRepoID: { $0 }
        )
        let totals = await directory.contributions(forSlug: slug)

        let url = try! #require(await requested.urls.first)
        #expect(url.path.hasSuffix("/api/benchmarks/atomic/contributors/swift-otter-4417"))
        let queryItems = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        #expect(queryItems.contains(URLQueryItem(name: "limit", value: "50")))
        #expect(!queryItems.contains(where: { $0.name == "contributor" }))
        // Exact, because the endpoint paginates to `complete`. This is the
        // number `receipts.count` got wrong when a local write failed.
        #expect(totals.value?.publishedRunCount == 2)
        #expect(totals.value?.modelCount == 2)
        #expect(totals.value?.slug == "swift-otter-4417")
    }

    private actor RequestLog {
        private(set) var urls: [URL] = []
        func record(_ url: URL) { urls.append(url) }
    }

    // MARK: - Counting

    @Test("A normal publication increments once and drops the stale median")
    func normalPublicationIncrements() {
        var state = CommunityPublicationState()
        let outcome = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7, median: 25.9), now: Self.publishedAt
        )
        #expect(outcome.observations.value?.observationCount == 8)
        // The client holds no sample population, so recomputing a median here
        // would be fabricating one.
        #expect(outcome.observations.value?.median == nil)
        #expect(state.confirmedFloor(for: Self.scope)?.count == 8)
        // A read can only be trusted once the edge cache may have expired.
        #expect(outcome.retryAfter == Self.publishedAt.addingTimeInterval(30))
    }

    @Test("A duplicate publication never moves the count")
    func duplicateDoesNotIncrement() {
        var state = CommunityPublicationState()
        let outcome = state.recordPublication(
            receipt: Self.receipt(alreadyExists: true), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // The corpus already contained this run.
        #expect(outcome.observations.value?.observationCount == 7)
        #expect(outcome.observations.value?.includesYours == false)
        // And no floor is established, so nothing defends an inflated number.
        #expect(state.floors.isEmpty)
        #expect(outcome.retryAfter == nil)

        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: .strengthen(observationCount: 7),
            scope: Self.scope,
            observationCountAfterPublishing: outcome.observations.value?.observationCount,
            alreadyPublished: true
        )
        #expect(celebration.headline == "This result is already published")
        #expect(celebration.body.contains("Nothing was added a second time"))
    }

    @Test("Republishing the same run repeatedly cannot inflate the count")
    func repeatedDuplicatesDoNotAccumulate() {
        var state = CommunityPublicationState()
        var observations = Self.ready(7)
        // The first attempt is genuinely new.
        observations = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: observations, now: Self.publishedAt
        ).observations
        #expect(observations.value?.observationCount == 8)

        // Four more presses of Publish on the same result.
        for index in 1...4 {
            observations = state.recordPublication(
                receipt: Self.receipt(alreadyExists: true), receiptSaved: true,
                scope: Self.scope, observations: observations,
                now: Self.publishedAt.addingTimeInterval(Double(index) * 5)
            ).observations
        }
        #expect(observations.value?.observationCount == 8, "duplicates incremented the count")
        // The original floor is untouched by the duplicates.
        #expect(state.confirmedFloor(for: Self.scope)?.count == 8)
        #expect(state.confirmedFloor(for: Self.scope)?.confirmedAt == Self.publishedAt)
    }

    @Test("Publishing into an unknown count establishes no floor")
    func unknownCountEstablishesNoFloor() {
        var state = CommunityPublicationState()
        let outcome = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: .unavailable(.boundedFeed), now: Self.publishedAt
        )
        // There was no number to increment, so inventing "1" would claim the
        // corpus contains only this run.
        #expect(outcome.observations.value == nil)
        #expect(state.floors.isEmpty)
    }

    // MARK: - The 30-second edge cache

    @Test("An immediate stale read does not walk the confirmed count backwards")
    func staleReadDoesNotLowerTheCount() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // The refresh fires at once and hits the cached body from before the
        // submission — the 7 → 8 → 7 flicker.
        let merged = state.merge(
            Self.ready(7, median: 25.9), scope: Self.scope,
            now: Self.publishedAt.addingTimeInterval(1)
        )
        #expect(merged.value?.observationCount == 8)
        #expect(merged.value?.includesYours == true)
        // The stale body's statistics are still usable; only its count is behind.
        #expect(merged.value?.median == 25.9)
        // Still defended, because the cache has not expired.
        #expect(state.confirmedFloor(for: Self.scope)?.count == 8)
        #expect(!state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(1)))
    }

    @Test("A read that has seen the submission retires the floor")
    func freshReadRetiresTheFloor() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        let merged = state.merge(
            Self.ready(8, median: 25.4), scope: Self.scope,
            now: Self.publishedAt.addingTimeInterval(31)
        )
        #expect(merged.value?.observationCount == 8)
        #expect(merged.value?.median == 25.4)
        // The server is authoritative again, so one publication cannot pin the
        // number for the rest of the session.
        #expect(state.floors.isEmpty)
        #expect(!state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(120)))
    }

    @Test("Others publishing meanwhile is not treated as staleness")
    func higherCountsPassThrough() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        let merged = state.merge(
            Self.ready(12), scope: Self.scope, now: Self.publishedAt.addingTimeInterval(45)
        )
        #expect(merged.value?.observationCount == 12)
        #expect(state.floors.isEmpty)
    }

    @Test("A retry is permitted only once the edge cache can have expired")
    func retryWaitsForTheCache() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        #expect(!state.mayRetry(Self.scope, at: Self.publishedAt))
        #expect(!state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(29)))
        #expect(state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(30)))
        #expect(state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(120)))
    }

    @Test("The confirmed count survives the service becoming unavailable")
    func unavailableDoesNotForgetTheReceipt() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        let merged = state.merge(
            .unavailable(.offline), scope: Self.scope,
            now: Self.publishedAt.addingTimeInterval(5)
        )
        // This installation holds a receipt for the submission; reverting to
        // "unknown" would be forgetting a fact it can prove.
        #expect(merged.value?.observationCount == 8)
        #expect(state.confirmedFloor(for: Self.scope)?.count == 8)
    }

    @Test("Loading passes through so the spinner is not replaced by a number")
    func loadingPassesThrough() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        #expect(state.merge(.loading, scope: Self.scope, now: Self.publishedAt).isLoading)
    }

    @Test("The floor defends only the scope that was published to")
    func floorIsScoped() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        // A different model's low count is a fact about that model, not a
        // stale read of this one.
        let other = state.merge(
            Self.ready(2), scope: Self.otherScope,
            now: Self.publishedAt.addingTimeInterval(1)
        )
        #expect(other.value?.observationCount == 2)
        // And the other model's unavailability is not overwritten either.
        #expect(
            state.merge(
                .unavailable(.boundedFeed), scope: Self.otherScope, now: Self.publishedAt
            ).value == nil
        )
        // The published scope is still defended.
        #expect(
            state.merge(Self.ready(7), scope: Self.scope, now: Self.publishedAt)
                .value?.observationCount == 8
        )
    }

    @Test("A comparison scope is not defended by a coverage-scope publication")
    func comparisonScopeIsDistinct() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        var comparisonScope = Self.scope
        comparisonScope.comparison = CommunityComparisonIdentity(
            caseID: "pp512-tg128", metricName: "decode_tps",
            execution: CommunityExecutionIdentity(
                rapidMLX: "0.13.4", computeDType: "bf16",
                speculativeDecodingMethod: "none", kvCacheMode: "quantized",
                kvCacheDType: "int8", prefillBackend: "gpu"
            )
        )
        // A count over one execution variant is a smaller population than the
        // coverage count, so the coverage floor must not raise it.
        let merged = state.merge(
            Self.ready(3), scope: comparisonScope,
            now: Self.publishedAt.addingTimeInterval(1)
        )
        #expect(merged.value?.observationCount == 3)
    }

    @Test("Leaving the model drops the floor")
    func clearingTheFloor() {
        var state = CommunityPublicationState()
        _ = state.recordPublication(
            receipt: Self.receipt(alreadyExists: false), receiptSaved: true,
            scope: Self.scope, observations: Self.ready(7), now: Self.publishedAt
        )
        state.clearFloor(for: Self.scope)
        #expect(state.floors.isEmpty)
        #expect(!state.mayRetry(Self.scope, at: Self.publishedAt.addingTimeInterval(600)))
        #expect(
            state.merge(Self.ready(7), scope: Self.scope, now: Self.publishedAt)
                .value?.observationCount == 7
        )
        // The identity is session-long and is not dropped with the floor.
        #expect(state.sessionContributor?.slug == "swift-otter-4417")
    }

    // MARK: - Everything a publication invalidates

    /// A tripwire, not a behavioural assertion: `refreshAfterPublishing` names
    /// these five reads explicitly. If a sixth kind is added and this fails,
    /// the publish path is the thing to go and update.
    @Test("Every community read a publication can change is known here")
    func everyReadKindIsAccountedFor() {
        #expect(
            Set(CommunityReadKind.allCases)
                == [.observations, .table, .coverage, .pulse, .contributorTotals]
        )
    }
}
