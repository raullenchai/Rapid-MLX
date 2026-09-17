import Foundation
import Testing
@testable import Rapid

/// What has to happen after `/api/benchmarks/atomic/public`'s 30-second edge
/// cache drains.
///
/// One cached body feeds **four** projections — the observation aggregate, the
/// Community table, the coverage list and the pulse band — so one stale read
/// makes all four stale together. The delayed retry refreshed only the count,
/// which left the table row that should have gained "INCLUDES YOURS" showing
/// pre-publish data until the user toggled a workload tab or relaunched the app.
///
/// The second half is floor lifetime. A confirmed floor used to be a single
/// optional, cleared on every `selectedAlias` change, so publishing into A,
/// glancing at B and coming back to A threw away the only proof the session had
/// that its own submission existed — and the still-cached feed walked A's
/// number back down.
@Suite("Post-publication cache reconciliation")
struct CommunityCacheReconciliationTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let mySlug = "swift-otter-4417"
    private static let publishedAt = Date(timeIntervalSince1970: 1_789_000_000)

    /// The alias is the display label; the identity is what matching uses, and
    /// the repo id keeps the publisher's own casing.
    private static func scope(
        _ alias: String, repo: String? = nil
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias, workload: .llm, protocolID: "rapid-community-speed",
            protocolVersion: 2, macProfile: profile,
            modelIdentity: CommunityModelIdentity(
                repoID: repo ?? "mlx-community/\(alias)"
            )
        )
    }

    private static let qwenScope = scope(
        "qwen3.5-9b-4bit", repo: "mlx-community/Qwen3.5-9B-4bit"
    )

    private static func receipt() -> CommunityBenchmarkReceipt {
        try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"sub-1","already_exists":false,
             "accepted_at":"2026-09-06T04:40:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
    }

    // MARK: - A first contribution, through the cache

    /// The cached body: `qwen` has two published runs, neither of them mine,
    /// and the pulse counts two contributors. This is what the edge serves for
    /// up to 30 seconds after my submission is accepted.
    private static let staleFeed = Self.feed(
        qwenSamples: 2, qwenContributors: [], runs: 2, contributors: ["modest-slate-wombat-545"]
    )

    /// The same endpoint after the cache drains: my run is in the aggregate,
    /// in the contributor list, and in `runs[]`.
    private static let freshFeed = Self.feed(
        qwenSamples: 3, qwenContributors: ["swift-otter-4417"],
        runs: 3, contributors: ["modest-slate-wombat-545", "swift-otter-4417"]
    )

    /// A body in the live `/atomic/public` shape: full `model` identity,
    /// `summary[].contributors`, and `runs[]` carrying nested `cases`.
    private static func feed(
        qwenSamples: Int,
        qwenContributors: [String],
        runs: Int,
        contributors: [String]
    ) -> String {
        func contributor(_ slug: String) -> String {
            let parts = slug.split(separator: "-")
            let tag = parts.last.map(String.init) ?? "000"
            let name = parts.dropLast().joined(separator: "-")
            return #"{"name":"\#(name)","tag":"\#(tag)","slug":"\#(slug)","#
                + #""url":"/leaderboard/contributors/\#(slug)"}"#
        }
        func model(_ repo: String, _ kind: String) -> String {
            #"""
            {"repo_id":"mlx-community/\#(repo)","identity_strength":"unresolved",
             "components":[{"component_id":"primary","role":"primary",
               "source":{"kind":"huggingface","repo_id":"mlx-community/\#(repo)"},
               "artifact":{"format":"mlx-safetensors"},
               "quantization":{"kind":"unknown","base_dtype":"unknown"}}],
             "pipeline_kind":"\#(kind)"}
            """#
        }
        let runList = (0..<runs).map { index in
            #"""
            {"schema_version":1,"submission_id":"run-\#(index)",
             "accepted_at":"2026-09-0\#(index + 1)T04:00:00Z",
             "contributor":\#(contributor(contributors[index % max(1, contributors.count)])),
             "task_type":"text_generation","model":\#(model("Qwen3.5-9B-4bit", "text_generation")),
             "machine":{"chip":"Apple M3 Pro","memory_gib":18,"cpu_cores":12,"gpu_cores":18,
                        "os":{"version":"15.6.1"}},
             "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
               "speculative_decoding":{"method":"none"},
               "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
             "protocol":{"id":"rapid-community-speed","version":2},
             "cases":[{"case_id":"pp512-tg128","target_prompt_tokens":512,
                       "target_output_tokens":128,"decode_tps":25.9,"ttft_ms":1490,
                       "total_seconds":6.5,"peak_memory_mib":6875}]}
            """#
        }.joined(separator: ",")
        return #"""
        {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
         "summary":[
           {"task_type":"text_generation","model":\#(model("Qwen3.5-9B-4bit", "text_generation")),
            "machine":{"chip":"Apple M3 Pro","memory_gib":18},
            "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
              "speculative_decoding":{"method":"none"},
              "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
            "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
            "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
            "samples":\#(qwenSamples),
            "contributors":[\#(qwenContributors.map(contributor).joined(separator: ","))],
            "latest_at":"2026-09-06T02:00:00Z"}
         ],
         "runs":[\#(runList)]}
        """#
    }

    /// Serves the stale body until `advance()` is called, then the fresh one —
    /// the edge cache, as a test double.
    private actor EdgeCache {
        private var isFresh = false
        private(set) var requestCount = 0
        func advance() { isFresh = true }
        func body() -> String {
            requestCount += 1
            return isFresh
                ? CommunityCacheReconciliationTests.freshFeed
                : CommunityCacheReconciliationTests.staleFeed
        }
    }

    private static func directory(_ cache: EdgeCache) -> CommunityBenchmarkAPIDirectory {
        CommunityBenchmarkAPIDirectory(
            transport: { request in
                let body = await cache.body()
                return (
                    Data(body.utf8),
                    HTTPURLResponse(
                        url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
                    )!
                )
            },
            aliasForRepoID: { $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased() }
        )
    }

    /// The four reads the retry must perform, driven off the production set so
    /// the test cannot drift from what the view actually re-reads.
    private struct Screen {
        var observations: CommunityDataState<CommunityObservationSummary> = .loading
        var table: CommunityDataState<[CommunityObservationRow]> = .loading
        var coverage: CommunityDataState<[CommunityCoverageGap]> = .loading
        var pulse: CommunityDataState<CommunityPulse> = .loading

        mutating func refresh(
            kinds: Set<CommunityReadKind>,
            directory: CommunityBenchmarkAPIDirectory,
            publication: inout CommunityPublicationState,
            scope: CommunityBenchmarkScope,
            viewerSlug: String?,
            now: Date
        ) async {
            if kinds.contains(.observations) {
                let answer = await directory.observations(for: scope, viewerSlug: viewerSlug)
                observations = publication.merge(answer, scope: scope, now: now)
            }
            if kinds.contains(.table) {
                table = await directory.table(
                    macProfile: scope.macProfile, workload: scope.workload,
                    metric: .primary(for: scope.workload), viewerSlug: viewerSlug
                )
            }
            if kinds.contains(.coverage) {
                coverage = await directory.coverageGaps(macProfile: scope.macProfile)
            }
            if kinds.contains(.pulse) {
                pulse = await directory.pulse()
            }
        }
    }

    @Test("A first contribution gains INCLUDES YOURS once the cache drains")
    func firstContributionGainsTheBadgeAfterTheCache() async {
        let cache = EdgeCache()
        let directory = Self.directory(cache)
        var publication = CommunityPublicationState()
        var screen = Screen()
        let scope = Self.qwenScope
        let kinds = CommunityPublicationState.publicFeedBackedReads

        // Before publishing: two runs, neither mine.
        await screen.refresh(
            kinds: kinds, directory: directory, publication: &publication,
            scope: scope, viewerSlug: nil, now: Self.publishedAt
        )
        #expect(screen.observations.value?.observationCount == 2)
        #expect(screen.table.value?.first?.summary.includesYours == false)
        #expect(screen.pulse.value?.contributorCount == 1)

        // Publish. The receipt issues this installation's first pseudonym.
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: scope,
            observations: screen.observations, now: Self.publishedAt
        )
        screen.observations = outcome.observations
        let slug = try! #require(publication.sessionContributor?.slug)
        #expect(slug == Self.mySlug)
        #expect(outcome.retryAfter == Self.publishedAt.addingTimeInterval(30))

        // The immediate refresh hits the cached body. The count holds at 3…
        await screen.refresh(
            kinds: kinds, directory: directory, publication: &publication,
            scope: scope, viewerSlug: slug, now: Self.publishedAt.addingTimeInterval(1)
        )
        #expect(screen.observations.value?.observationCount == 3)
        // …but the table genuinely cannot know yet: the cached body does not
        // list this contributor, and the client must not paint the badge from
        // its own memory.
        #expect(screen.table.value?.first?.summary.includesYours == false)
        #expect(publication.mayRetry(scope, at: Self.publishedAt.addingTimeInterval(1)) == false)

        // 30 seconds later the cache has drained and the retry fires — across
        // all four reads, not just the count.
        await cache.advance()
        #expect(publication.mayRetry(scope, at: Self.publishedAt.addingTimeInterval(30)))
        await screen.refresh(
            kinds: kinds, directory: directory, publication: &publication,
            scope: scope, viewerSlug: slug, now: Self.publishedAt.addingTimeInterval(30)
        )

        // No relaunch, no workload toggle: every projection now reflects the
        // contribution.
        #expect(screen.observations.value?.observationCount == 3)
        #expect(screen.observations.value?.includesYours == true)
        #expect(
            screen.table.value?.first?.summary.includesYours == true,
            "the table never gained INCLUDES YOURS after the cache drained"
        )
        #expect(screen.table.value?.first?.summary.observationCount == 3)
        #expect(screen.pulse.value?.contributorCount == 2)
        #expect(screen.pulse.value?.contributors.map(\.slug).contains(Self.mySlug) == true)
        #expect(screen.pulse.value?.publishedRunCount == 3)
        // The server is authoritative again.
        #expect(publication.confirmedFloor(for: scope) == nil)
    }

    @Test("The retry covers every read the public feed serves")
    func retryCoversEveryFeedBackedRead() {
        // Stated once, next to the cache duration it exists because of.
        #expect(
            CommunityPublicationState.publicFeedBackedReads
                == [.observations, .table, .coverage, .pulse]
        )
        // `contributorTotals` comes from the cursor-paginated contributions
        // route, which this cache does not serve.
        #expect(!CommunityPublicationState.publicFeedBackedReads.contains(.contributorTotals))
        #expect(
            CommunityPublicationState.publicFeedBackedReads
                .isSubset(of: Set(CommunityReadKind.allCases))
        )
    }

    @Test("Coverage also refreshes, so a thin pairing stops being reported")
    func coverageRefreshesToo() async {
        let cache = EdgeCache()
        let directory = Self.directory(cache)
        var publication = CommunityPublicationState()
        var screen = Screen()
        let scope = Self.qwenScope

        await screen.refresh(
            kinds: CommunityPublicationState.publicFeedBackedReads, directory: directory,
            publication: &publication, scope: scope, viewerSlug: nil, now: Self.publishedAt
        )
        // Two samples is under the threshold.
        #expect(screen.coverage.value?.map(\.observationCount) == [2])

        await cache.advance()
        await screen.refresh(
            kinds: CommunityPublicationState.publicFeedBackedReads, directory: directory,
            publication: &publication, scope: scope, viewerSlug: Self.mySlug,
            now: Self.publishedAt.addingTimeInterval(30)
        )
        #expect(screen.coverage.value?.map(\.observationCount) == [3])
    }

    // MARK: - Floors survive navigation

    @Test("Publish A at 7→8, visit B, return to A on a stale feed, still 8")
    func floorSurvivesAVisitToAnotherModel() {
        let a = Self.scope("qwen3.5-9b-4bit")
        let b = Self.scope("gemma-4-12b-4bit")
        var publication = CommunityPublicationState()

        // Publish into A: 7 becomes 8.
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: a,
            observations: .ready(
                CommunityObservationSummary(observationCount: 7, unit: "tok/s", isBounded: true)
            ),
            now: Self.publishedAt
        )
        #expect(outcome.observations.value?.observationCount == 8)
        #expect(publication.confirmedFloor(for: a)?.count == 8)

        // The user switches to B. B's own read lands and is not interfered with.
        let bRead = publication.merge(
            .ready(CommunityObservationSummary(observationCount: 4, isBounded: true)),
            scope: b, now: Self.publishedAt.addingTimeInterval(5)
        )
        #expect(bRead.value?.observationCount == 4)
        // …and A's floor is still there. This is the regression: the old code
        // cleared it on the `selectedAlias` change.
        #expect(
            publication.confirmedFloor(for: a)?.count == 8,
            "visiting another model discarded A's confirmed floor"
        )

        // Back to A, while the feed is still serving the pre-publish body.
        let aRead = publication.merge(
            .ready(
                CommunityObservationSummary(observationCount: 7, median: 25.9, isBounded: true)
            ),
            scope: a, now: Self.publishedAt.addingTimeInterval(10)
        )
        #expect(aRead.value?.observationCount == 8, "A's count was walked back down to 7")
        #expect(aRead.value?.includesYours == true)
        // The stale body's statistics are still usable.
        #expect(aRead.value?.median == 25.9)
    }

    @Test("A later server count of at least 8 retires A's floor")
    func aFreshCountRetiresTheFloor() {
        let a = Self.scope("qwen3.5-9b-4bit")
        var publication = CommunityPublicationState()
        _ = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: a,
            observations: .ready(CommunityObservationSummary(observationCount: 7, isBounded: true)),
            now: Self.publishedAt
        )
        let fresh = publication.merge(
            .ready(CommunityObservationSummary(observationCount: 8, median: 25.4, isBounded: true)),
            scope: a, now: Self.publishedAt.addingTimeInterval(35)
        )
        #expect(fresh.value?.observationCount == 8)
        #expect(fresh.value?.median == 25.4)
        #expect(publication.confirmedFloor(for: a) == nil, "the floor outlived a fresh read")
        // And it stays retired: a later read is authoritative, up or down.
        #expect(
            publication.merge(
                .ready(CommunityObservationSummary(observationCount: 9, isBounded: true)),
                scope: a, now: Self.publishedAt.addingTimeInterval(90)
            ).value?.observationCount == 9
        )
    }

    @Test("Two publications keep two independent floors")
    func floorsAreIndependent() {
        let a = Self.scope("qwen3.5-9b-4bit")
        let b = Self.scope("gemma-4-12b-4bit")
        var publication = CommunityPublicationState()
        _ = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: a,
            observations: .ready(CommunityObservationSummary(observationCount: 7, isBounded: true)),
            now: Self.publishedAt
        )
        _ = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: b,
            observations: .ready(CommunityObservationSummary(observationCount: 1, isBounded: true)),
            now: Self.publishedAt.addingTimeInterval(60)
        )
        #expect(publication.floors.count == 2)
        #expect(publication.confirmedFloor(for: a)?.count == 8)
        #expect(publication.confirmedFloor(for: b)?.count == 2)

        // Retiring one leaves the other.
        _ = publication.merge(
            .ready(CommunityObservationSummary(observationCount: 8, isBounded: true)),
            scope: a, now: Self.publishedAt.addingTimeInterval(90)
        )
        #expect(publication.confirmedFloor(for: a) == nil)
        #expect(publication.confirmedFloor(for: b)?.count == 2)
    }

    @Test("Floors are bounded, oldest evicted first")
    func floorsAreBounded() {
        var publication = CommunityPublicationState()
        let total = CommunityPublicationState.maximumFloors + 4
        for index in 0..<total {
            _ = publication.recordPublication(
                receipt: Self.receipt(), receiptSaved: true,
                scope: Self.scope("model-\(index)"),
                observations: .ready(
                    CommunityObservationSummary(observationCount: 1, isBounded: true)
                ),
                now: Self.publishedAt.addingTimeInterval(Double(index))
            )
        }
        #expect(publication.floors.count == CommunityPublicationState.maximumFloors)
        // The four oldest are gone; the newest are kept.
        #expect(publication.confirmedFloor(for: Self.scope("model-0")) == nil)
        #expect(publication.confirmedFloor(for: Self.scope("model-3")) == nil)
        #expect(publication.confirmedFloor(for: Self.scope("model-\(total - 1)")) != nil)
    }

    @Test("A pending retry is visible across scopes")
    func retryAcrossScopes() {
        let a = Self.scope("qwen3.5-9b-4bit")
        var publication = CommunityPublicationState()
        _ = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true, scope: a,
            observations: .ready(CommunityObservationSummary(observationCount: 7, isBounded: true)),
            now: Self.publishedAt
        )
        // While B is on screen the retry is still owed — to A.
        #expect(!publication.mayRetryAnything(at: Self.publishedAt.addingTimeInterval(10)))
        #expect(publication.mayRetryAnything(at: Self.publishedAt.addingTimeInterval(30)))
        #expect(
            !publication.mayRetry(Self.scope("gemma-4-12b-4bit"),
                                  at: Self.publishedAt.addingTimeInterval(30))
        )
    }
}
