import Foundation
import Testing
@testable import Rapid

/// The production adapter over the public atomic benchmark endpoints.
///
/// Fixtures below are trimmed copies of the real worker responses in
/// `landing/src/index.js` (`buildAtomicBenchmarkPublic`,
/// `handleAtomicBenchmarkContributions`), so the decoding here is pinned to the
/// shape the service actually returns rather than one invented for the test.
@Suite("Public atomic feed adapter")
struct CommunityBenchmarkAPIDirectoryTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static let publicFeed = #"""
    {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
     "summary":[
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"strong"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
        "samples":7,
        "contributors":[{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417",
                         "url":"/leaderboard/contributors/swift-otter-4417"}],
        "latest_at":"2026-09-06T04:37:42Z"},
       {"task_type":"image_generation",
        "model":{"repo_id":"mlx-community/z-image-turbo","identity_strength":"strong"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4"},
        "protocol":{"id":"rapid-image-speed","version":1},
        "case_id":"render-1024",
        "metric":{"name":"total_seconds","better":"lower","median":4.6,"best":4.4},
        "samples":2,
        "contributors":[],
        "latest_at":"2026-09-05T04:37:42Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"strong"},
        "machine":{"chip":"Apple M4 Max","memory_gib":48},
        "execution":{"rapid_mlx":"0.13.4"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":58.1,"best":60.0},
        "samples":3,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"}
     ],
     "runs":[
       {"schema_version":1,"submission_id":"run-1","accepted_at":"2026-09-06T04:37:42Z",
        "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417",
                       "url":"/leaderboard/contributors/swift-otter-4417"},
        "task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"strong"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18,"cpu_cores":12,"gpu_cores":18,
                   "os":{"version":"15.6.1"}},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
        "protocol":{"id":"rapid-community-speed","version":2},"cases":[]},
       {"schema_version":1,"submission_id":"run-2","accepted_at":"2026-09-05T04:37:42Z",
        "contributor":{"name":"modest-slate-wombat","tag":"545",
                       "slug":"modest-slate-wombat-545",
                       "url":"/leaderboard/contributors/modest-slate-wombat-545"},
        "task_type":"image_generation",
        "model":{"repo_id":"mlx-community/z-image-turbo","identity_strength":"strong"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18,"cpu_cores":12,"gpu_cores":18,
                   "os":{"version":"15.6.1"}},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
        "protocol":{"id":"rapid-image-speed","version":1},"cases":[]}
     ]}
    """#

    private static func directory(
        _ handler: @escaping @Sendable (URLRequest) async throws -> (Data, URLResponse)
    ) -> CommunityBenchmarkAPIDirectory {
        CommunityBenchmarkAPIDirectory(
            transport: handler,
            aliasForRepoID: { repo in
                repo.replacingOccurrences(of: "mlx-community/", with: "").lowercased()
            }
        )
    }

    private static func ok(_ body: String, url: URL) -> (Data, URLResponse) {
        (
            Data(body.utf8),
            HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
        )
    }

    private static func scope(
        _ alias: String,
        _ workload: CommunityWorkload,
        comparison: CommunityComparisonIdentity? = nil
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias,
            workload: workload,
            protocolID: workload == .image ? "rapid-image-speed" : "rapid-community-speed",
            protocolVersion: workload == .image ? 1 : 2,
            macProfile: profile,
            comparison: comparison
        )
    }

    /// The execution configuration the "bf16 / none / quantized" cells use.
    private static let bf16 = CommunityExecutionIdentity(
        rapidMLX: "0.13.4", computeDType: "bf16",
        speculativeDecodingMethod: "none", kvCacheMode: "quantized",
        kvCacheDType: "int8", prefillBackend: "gpu"
    )
    /// Same model, same Mac, same protocol — a different dtype.
    private static let fp16 = CommunityExecutionIdentity(
        rapidMLX: "0.13.4", computeDType: "fp16",
        speculativeDecodingMethod: "none", kvCacheMode: "quantized",
        kvCacheDType: "int8", prefillBackend: "gpu"
    )


    /// Four cells for the **same** model, Mac and task, differing only in
    /// protocol version, workload case, execution dtype and metric. Picking
    /// `first` out of this set — which the adapter used to do — compares a
    /// short prompt against a long one, or bf16 against fp16.
    private static let ambiguousFeed = #"""
    {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
     "summary":[
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"fp16",
                     "speculative_decoding":{"method":"none"},
                     "kv_cache":{"mode":"quantized","dtype":"int8"},
                     "prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":31.0,"best":33.0},
        "samples":4,"contributors":[],"latest_at":"2026-09-06T04:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
                     "speculative_decoding":{"method":"none"},
                     "kv_cache":{"mode":"quantized","dtype":"int8"},
                     "prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp2048-tg512",
        "metric":{"name":"decode_tps","better":"higher","median":12.0,"best":13.0},
        "samples":3,"contributors":[],"latest_at":"2026-09-06T03:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
                     "speculative_decoding":{"method":"none"},
                     "kv_cache":{"mode":"quantized","dtype":"int8"},
                     "prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
        "samples":5,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
                     "speculative_decoding":{"method":"none"},
                     "kv_cache":{"mode":"quantized","dtype":"int8"},
                     "prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":1},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":18.0,"best":19.0},
        "samples":2,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"}
     ],
     "runs":[]}
    """#

    /// A feed with no cell for this Mac profile at all.
    private static let otherMacFeed = #"""
    {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
     "summary":[
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
        "machine":{"chip":"Apple M4 Max","memory_gib":48},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
        "protocol":{"id":"rapid-community-speed","version":2},
        "case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":58.1,"best":60.0},
        "samples":3,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"}
     ],
     "runs":[]}
    """#

    // MARK: - Exact comparison identity

    @Test("A comparison picks the cell matching case, execution, metric and protocol")
    func comparisonMatchesFullIdentity() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        let state = await directory.observations(
            for: Self.scope(
                "qwen3.5-9b-4bit", .llm,
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128", metricName: "decode_tps", execution: Self.bf16
                )
            )
        )
        // 25.9 is the bf16 / short-prompt / v2 cell. `first` would have
        // returned 31.0 (fp16) — a 20% difference on the same hardware.
        #expect(state.value?.median == 25.9)
        #expect(state.value?.observationCount == 5)
    }

    @Test("A different execution dtype selects a different cell")
    func executionDiscriminates() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        let state = await directory.observations(
            for: Self.scope(
                "qwen3.5-9b-4bit", .llm,
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128", metricName: "decode_tps", execution: Self.fp16
                )
            )
        )
        #expect(state.value?.median == 31.0)
        #expect(state.value?.observationCount == 4)
    }

    @Test("A different workload case selects a different cell")
    func caseDiscriminates() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        let state = await directory.observations(
            for: Self.scope(
                "qwen3.5-9b-4bit", .llm,
                comparison: CommunityComparisonIdentity(
                    caseID: "pp2048-tg512", metricName: "decode_tps", execution: Self.bf16
                )
            )
        )
        #expect(state.value?.median == 12.0)
        #expect(state.value?.observationCount == 3)
    }

    @Test("A different protocol version is not comparable at all")
    func protocolVersionDiscriminates() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        var scope = Self.scope(
            "qwen3.5-9b-4bit", .llm,
            comparison: CommunityComparisonIdentity(
                caseID: "pp512-tg128", metricName: "decode_tps", execution: Self.bf16
            )
        )
        // v1 has one matching cell (median 18.0) and must not borrow v2's.
        scope = CommunityBenchmarkScope(
            modelAlias: scope.modelAlias, workload: scope.workload,
            protocolID: scope.protocolID, protocolVersion: 1,
            macProfile: scope.macProfile, comparison: scope.comparison
        )
        #expect(await directory.observations(for: scope).value?.median == 18.0)

        // A version nobody published is unavailable, not the nearest cell.
        let v9 = CommunityBenchmarkScope(
            modelAlias: scope.modelAlias, workload: scope.workload,
            protocolID: scope.protocolID, protocolVersion: 9,
            macProfile: scope.macProfile, comparison: scope.comparison
        )
        #expect(await directory.observations(for: v9).unavailableReason == .boundedFeed)
    }

    @Test("An unmatched metric name is unavailable, never a substituted cell")
    func metricDiscriminates() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        let state = await directory.observations(
            for: Self.scope(
                "qwen3.5-9b-4bit", .llm,
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128", metricName: "total_seconds", execution: Self.bf16
                )
            )
        )
        #expect(state.value == nil)
        #expect(state.unavailableReason == .boundedFeed)
    }

    @Test("Coverage without matching run identities is unavailable, never double-counted")
    func coverageRequiresMatchingRuns() async {
        let directory = Self.directory { Self.ok(Self.ambiguousFeed, url: $0.url!) }
        let state = await directory.observations(for: Self.scope("qwen3.5-9b-4bit", .llm))
        #expect(state.value == nil)
        #expect(state.unavailableReason == .boundedFeed)
    }

    @Test("Coverage counts one run once when it appears in multiple case cells")
    func coverageDeduplicatesRunsAcrossCases() async throws {
        var object = try #require(
            JSONSerialization.jsonObject(with: Data(Self.publicFeed.utf8))
                as? [String: Any]
        )
        var summary = try #require(object["summary"] as? [[String: Any]])
        var secondCase = summary[0]
        secondCase["case_id"] = "pp2048-tg512"
        summary.insert(secondCase, at: 1)
        object["summary"] = summary
        let body = try #require(
            String(
                data: JSONSerialization.data(withJSONObject: object),
                encoding: .utf8
            )
        )
        let directory = Self.directory { Self.ok(body, url: $0.url!) }

        let state = await directory.observations(
            for: Self.scope("qwen3.5-9b-4bit", .llm)
        )

        #expect(state.value?.observationCount == 1)
        #expect(state.value?.median == nil)
        #expect(state.value?.isBounded == true)
    }

    // MARK: - Empty table

    @Test("An empty filtered table is unavailable, never an empty ready array")
    func emptyTableIsUnavailable() async {
        let directory = Self.directory { Self.ok(Self.otherMacFeed, url: $0.url!) }
        let state = await directory.table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        )
        // `.ready([])` here rendered "Yours would be the first" in the UI.
        #expect(state.value == nil)
        #expect(state.unavailableReason == .boundedFeed)

        let imageState = await directory.table(
            macProfile: Self.profile, workload: .image, metric: .renderTime
        )
        #expect(imageState.value == nil)
        #expect(imageState.unavailableReason == .boundedFeed)
    }

    // MARK: - Existence vs absence

    @Test("A coverage scope yields a bounded count and deliberately no median")
    func presentScopeIsBounded() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let state = await directory.observations(for: Self.scope("qwen3.5-9b-4bit", .llm))
        let summary = try? #require(state.value)
        // The trimmed fixture carries one matching run even though its summary
        // sample count is seven. Run identities, not case cells, are counted.
        #expect(summary?.observationCount == 1)
        #expect(summary?.unit == "tok/s")
        // No comparison identity was supplied, so this is the coverage
        // question. A median would have to be picked from (or averaged over)
        // cells measured under different execution configurations.
        #expect(summary?.median == nil)
        #expect(summary?.isBounded == true)

        #expect(
            CommunityContributionBranch.select(from: state)
                == .strengthen(observationCount: 1, isAtLeast: true)
        )
    }

    @Test("A comparison scope yields that cell's median, and still no range")
    func comparisonScopeYieldsMedian() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let state = await directory.observations(
            for: Self.scope(
                "qwen3.5-9b-4bit", .llm,
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128",
                    metricName: "decode_tps",
                    execution: CommunityExecutionIdentity(
                        rapidMLX: "0.13.4", computeDType: "unknown"
                    )
                )
            )
        )
        #expect(state.value?.median == 25.9)
        // The feed publishes a median and a best, but no observed min/max.
        #expect(state.value?.observedMinimum == nil)
        #expect(state.value?.observedMaximum == nil)
    }

    @Test("A scope absent from the bounded feed is unknown, never zero")
    func absentScopeIsUnavailableNotZero() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        // The feed carries only the newest 50 runs, so absence proves nothing.
        let state = await directory.observations(for: Self.scope("gemma-4-12b-4bit", .llm))
        #expect(state.value == nil)
        #expect(state.unavailableReason == .boundedFeed)
        let branch = CommunityContributionBranch.select(from: state)
        #expect(!branch.allowsFirstReferenceLanguage)
        #expect(!branch.allowsComparisonStatistics)
    }

    @Test("A different Mac profile is a different scope")
    func macProfileScoping() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let other = CommunityBenchmarkScope(
            modelAlias: "qwen3.5-9b-4bit",
            workload: .llm,
            protocolID: "rapid-community-speed",
            protocolVersion: 2,
            macProfile: CommunityMacProfile(chip: "Apple M4 Max", memoryGiB: 48)
        )
        #expect(await directory.observations(for: other).unavailableReason == .boundedFeed)
    }

    // MARK: - Table

    @Test("The table shows only this Mac profile and the selected workload")
    func tableIsScoped() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let rows = await directory.table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        ).value
        #expect(rows?.map(\.modelAlias) == ["qwen3.5-9b-4bit"])

        let imageRows = await directory.table(
            macProfile: Self.profile, workload: .image, metric: .renderTime
        ).value
        #expect(imageRows?.map(\.modelAlias) == ["z-image-turbo"])
        #expect(imageRows?.first?.summary.unit == "s")
    }

    @Test("Unknown server workloads never fall through into the LLM table")
    func unknownWorkloadsAreDiscarded() async {
        let unsupported = Self.publicFeed.replacingOccurrences(
            of: #""task_type":"text_generation""#,
            with: #""task_type":"future_workload""#
        )
        let directory = Self.directory { request in
            Self.ok(unsupported, url: request.url!)
        }

        let state = await directory.table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        )
        #expect(state == .unavailable(.boundedFeed))
    }

    // MARK: - Coverage

    @Test("Coverage never flags a first-result opportunity from a bounded feed")
    func coverageNeverClaimsZero() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let gaps = await directory.coverageGaps(macProfile: Self.profile).value
        let unwrapped = try? #require(gaps)
        // z-image-turbo has 2 samples — under-represented, but definitely not
        // a first-result opportunity.
        #expect(unwrapped?.map(\.modelAlias) == ["z-image-turbo"])
        #expect(unwrapped?.allSatisfy { !$0.isFirstResultOpportunity } == true)
        #expect(unwrapped?.allSatisfy { $0.observationCount >= 1 } == true)
    }

    // MARK: - Pulse

    @Test("Pulse carries real contributor identities, not opaque seeds")
    func pulseCarriesIdentities() async {
        let directory = Self.directory { request in
            Self.ok(Self.publicFeed, url: request.url!)
        }
        let pulse = await directory.pulse().value
        let value = try? #require(pulse)
        #expect(value?.contributors.map(\.slug) == [
            "modest-slate-wombat-545", "swift-otter-4417",
        ])
        // Those slugs must drive the same plates the website shows.
        #expect(
            value?.contributors.map { CommunityContributorAvatar.plate(for: $0) } == [2, 12]
        )
        #expect(value?.contributorCount == 2)
        #expect(value?.publishedRunCount == 2)
        #expect(value?.modelCount == 2)
        // The feed is bounded, so every total is a floor.
        #expect(value?.isBounded == true)
    }

    // MARK: - Exact contributor totals

    @Test("Contributor totals follow the cursor until complete")
    func contributorTotalsPaginate() async {
        let page1 = #"""
        {"schema_version":1,"contributor":"swift-otter-4417","cursor":"c1","complete":false,
         "runs":[
           {"schema_version":1,"submission_id":"a","accepted_at":"2026-09-01T00:00:00Z",
            "contributor":null,"task_type":"text_generation",
            "model":{"repo_id":"mlx-community/A"},"cases":[]},
           {"schema_version":1,"submission_id":"b","accepted_at":"2026-09-02T00:00:00Z",
            "contributor":null,"task_type":"text_generation",
            "model":{"repo_id":"mlx-community/B"},"cases":[]}
         ]}
        """#
        let page2 = #"""
        {"schema_version":1,"contributor":"swift-otter-4417","cursor":null,"complete":true,
         "runs":[
           {"schema_version":1,"submission_id":"c","accepted_at":"2026-09-03T00:00:00Z",
            "contributor":null,"task_type":"image_generation",
            "model":{"repo_id":"mlx-community/A"},"cases":[]}
         ]}
        """#
        let requested = RequestLog()
        let directory = Self.directory { request in
            let url = request.url!
            await requested.record(url.absoluteString)
            let isSecondPage = url.query?.contains("cursor=c1") == true
            return Self.ok(isSecondPage ? page2 : page1, url: url)
        }

        let totals = await directory.contributions(forSlug: "swift-otter-4417").value
        let value = try? #require(totals)
        // Three runs across two pages; the count is exact, not the first page.
        #expect(value?.publishedRunCount == 3)
        #expect(value?.modelCount == 2)
        #expect(await requested.count == 2)
        #expect(
            await requested.all.allSatisfy {
                $0.contains("/api/benchmarks/atomic/contributors/swift-otter-4417")
                    && !$0.contains("contributor=")
            }
        )
    }

    @Test("An unterminated cursor reports unavailable rather than a partial total")
    func unterminatedPaginationIsUnavailable() async {
        let endless = #"""
        {"schema_version":1,"contributor":"swift-otter-4417","cursor":"more","complete":false,
         "runs":[{"schema_version":1,"submission_id":"a","accepted_at":"2026-09-01T00:00:00Z",
                  "contributor":null,"task_type":"text_generation",
                  "model":{"repo_id":"mlx-community/A"},"cases":[]}]}
        """#
        var directory = Self.directory { request in Self.ok(endless, url: request.url!) }
        directory.maximumContributionPages = 3
        let state = await directory.contributions(forSlug: "swift-otter-4417")
        // A floor would be worse than nothing on a screen that promises the
        // contributor's real public total.
        #expect(state.value == nil)
        #expect(state.unavailableReason == .incompleteAggregate)
    }

    @Test("An incomplete final page without a cursor never becomes an exact total")
    func incompletePageWithoutCursorIsUnavailable() async {
        let incomplete = #"""
        {"schema_version":1,"contributor":"swift-otter-4417","cursor":null,"complete":false,
         "runs":[{"schema_version":1,"submission_id":"a","accepted_at":"2026-09-01T00:00:00Z",
                  "contributor":null,"task_type":"text_generation",
                  "model":{"repo_id":"mlx-community/A"},"cases":[]}]}
        """#
        let directory = Self.directory { request in Self.ok(incomplete, url: request.url!) }

        let state = await directory.contributions(forSlug: "swift-otter-4417")

        #expect(state.value == nil)
        #expect(state.unavailableReason == .incompleteAggregate)
    }

    @Test("Contributor slug remains one encoded path segment")
    func contributorSlugIsSafelyEncoded() async {
        let complete = #"""
        {"schema_version":1,"contributor":"swift/otter","cursor":null,"complete":true,
         "runs":[]}
        """#
        let requested = RequestLog()
        let directory = Self.directory { request in
            await requested.record(request.url!.absoluteString)
            return Self.ok(complete, url: request.url!)
        }

        _ = await directory.contributions(forSlug: "swift/otter")

        #expect(await requested.all == [
            "https://rapidmlx.com/api/benchmarks/atomic/contributors/swift%2Fotter?limit=50"
        ])
    }

    // MARK: - Failures

    @Test("A non-2xx response degrades to unavailable, never to zero")
    func serverErrorIsUnavailable() async {
        let directory = Self.directory { request in
            (
                Data("{}".utf8),
                HTTPURLResponse(
                    url: request.url!, statusCode: 503, httpVersion: nil, headerFields: nil
                )!
            )
        }
        let state = await directory.observations(for: Self.scope("qwen3.5-9b-4bit", .llm))
        #expect(state.value == nil)
        #expect(!CommunityContributionBranch.select(from: state).allowsFirstReferenceLanguage)
        #expect(await directory.pulse().value == nil)
    }
}

/// Records the URLs a test's transport was asked for.
private actor RequestLog {
    private(set) var all: [String] = []
    var count: Int { all.count }
    func record(_ url: String) { all.append(url) }
}
