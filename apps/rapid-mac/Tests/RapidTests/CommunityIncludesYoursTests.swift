import Foundation
import Testing
@testable import Rapid

/// "INCLUDES YOURS" has to be a read of public data, not a memory.
///
/// It was hard-coded to `false` everywhere the server was actually consulted,
/// and only ever became true through ``CommunityPublicationState`` — the
/// optimistic path that exists for the 30 seconds after an upload. So the badge
/// appeared once, at publish time, and was gone on the next launch. The user's
/// contribution had not gone anywhere; the client had simply stopped asking.
///
/// `summary[].contributors` is a real field (`atomicBenchmarkSummary` in
/// `landing/src/index.js` builds it from `run.contributor`), so the question is
/// answerable from a plain GET with no local state at all.
@Suite("Includes yours")
struct CommunityIncludesYoursTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let mySlug = "swift-otter-4417"
    private static let bf16 = CommunityExecutionIdentity(
        rapidMLX: "0.13.4", computeDType: "bf16",
        speculativeDecodingMethod: "none", kvCacheMode: "quantized",
        kvCacheDType: "int8", prefillBackend: "gpu"
    )

    /// Two models. `qwen` has two execution cells; only the bf16 one carries
    /// this installation's pseudonym. `gemma` is somebody else's entirely.
    private static let feed = #"""
    {"schema_version":1,"summary":[
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"fp16",
         "speculative_decoding":{"method":"none"},
         "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":31.0,"best":33.0},
       "samples":4,
       "contributors":[{"name":"modest-slate-wombat","tag":"545",
                        "slug":"modest-slate-wombat-545",
                        "url":"/leaderboard/contributors/modest-slate-wombat-545"}],
       "latest_at":"2026-09-06T04:00:00Z"},
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
         "speculative_decoding":{"method":"none"},
         "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
       "samples":5,
       "contributors":[{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417",
                        "url":"/leaderboard/contributors/swift-otter-4417"},
                       {"name":"modest-slate-wombat","tag":"545",
                        "slug":"modest-slate-wombat-545",
                        "url":"/leaderboard/contributors/modest-slate-wombat-545"}],
       "latest_at":"2026-09-06T02:00:00Z"},
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Gemma-4-12B-4bit"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
         "speculative_decoding":{"method":"none"},
         "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":19.4,"best":20.1},
       "samples":6,
       "contributors":[{"name":"modest-slate-wombat","tag":"545",
                        "slug":"modest-slate-wombat-545",
                        "url":"/leaderboard/contributors/modest-slate-wombat-545"}],
       "latest_at":"2026-09-06T01:00:00Z"}
    ],"runs":[]}
    """#

    private static func directory(_ body: String = feed) -> CommunityBenchmarkAPIDirectory {
        let body = feedIncludingRuns(body)
        return CommunityBenchmarkAPIDirectory(
            transport: { request in
                (
                    Data(body.utf8),
                    HTTPURLResponse(
                        url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
                    )!
                )
            },
            aliasForRepoID: { $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased() }
        )
    }

    /// Coverage counts distinct submissions from `runs`, not summary-cell
    /// samples (one submission can populate several cells). Keep these compact
    /// hand-written summary fixtures realistic by materializing one distinct
    /// run per advertised sample.
    private static func feedIncludingRuns(_ body: String) -> String {
        guard let data = body.data(using: .utf8),
              var envelope = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let summaries = envelope["summary"] as? [[String: Any]],
              (envelope["runs"] as? [[String: Any]])?.isEmpty != false else { return body }
        var runs: [[String: Any]] = []
        for (cellIndex, summary) in summaries.enumerated() {
            let samples = summary["samples"] as? Int ?? 0
            for sampleIndex in 0..<samples {
                runs.append([
                    "submission_id": "fixture-\(cellIndex)-\(sampleIndex)",
                    "accepted_at": summary["latest_at"] as? String ?? "2026-09-06T02:00:00Z",
                    "task_type": summary["task_type"] as Any,
                    "model": summary["model"] as Any,
                    "machine": summary["machine"] as Any,
                    "protocol": summary["protocol"] as Any,
                ])
            }
        }
        envelope["runs"] = runs
        guard let encoded = try? JSONSerialization.data(withJSONObject: envelope, options: [.sortedKeys])
        else { return body }
        return String(decoding: encoded, as: UTF8.self)
    }

    private static func scope(
        _ alias: String, comparison: CommunityComparisonIdentity? = nil
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: alias, workload: .llm, protocolID: "rapid-community-speed",
            protocolVersion: 2, macProfile: profile, comparison: comparison
        )
    }

    // MARK: - A cold start, with no publication state whatsoever

    @Test("After a restart the exact observation still knows the contribution is yours")
    func exactObservationAfterRestart() async {
        // No `CommunityPublicationState` is constructed anywhere in this test.
        // The only thing carried across the "restart" is the slug, which comes
        // from a locally stored receipt.
        let state = await Self.directory().observations(
            for: Self.scope(
                "qwen3.5-9b-4bit",
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128", metricName: "decode_tps", execution: Self.bf16
                )
            ),
            viewerSlug: Self.mySlug
        )
        #expect(state.value?.includesYours == true)
        #expect(state.value?.observationCount == 5)
    }

    @Test("The coverage aggregate is yours when any counted cell is")
    func coverageAggregateAfterRestart() async {
        let state = await Self.directory().observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: Self.mySlug
        )
        // 4 fp16 (not mine) + 5 bf16 (mine) — the aggregate does contain one
        // of my runs.
        #expect(state.value?.observationCount == 9)
        #expect(state.value?.includesYours == true)
    }

    @Test("A cell that is not yours is not claimed")
    func otherContributorsCellIsNotYours() async {
        let state = await Self.directory().observations(
            for: Self.scope(
                "qwen3.5-9b-4bit",
                comparison: CommunityComparisonIdentity(
                    caseID: "pp512-tg128", metricName: "decode_tps",
                    execution: CommunityExecutionIdentity(
                        rapidMLX: "0.13.4", computeDType: "fp16",
                        speculativeDecodingMethod: "none", kvCacheMode: "quantized",
                        kvCacheDType: "int8", prefillBackend: "gpu"
                    )
                )
            ),
            viewerSlug: Self.mySlug
        )
        #expect(state.value?.observationCount == 4)
        #expect(state.value?.includesYours == false)
    }

    @Test("Another model's aggregate is never marked yours")
    func anotherModelIsNotYours() async {
        let state = await Self.directory().observations(
            for: Self.scope("gemma-4-12b-4bit"), viewerSlug: Self.mySlug
        )
        #expect(state.value?.observationCount == 6)
        #expect(state.value?.includesYours == false)
    }

    @Test("With no identity yet, nothing is claimed")
    func noIdentityClaimsNothing() async {
        let state = await Self.directory().observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: nil
        )
        #expect(state.value?.observationCount == 9)
        #expect(state.value?.includesYours == false)

        let empty = await Self.directory().observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: ""
        )
        #expect(empty.value?.includesYours == false)
    }

    // MARK: - The table

    @Test("The table marks only the rows this installation contributed to")
    func tableMarksTheRightRows() async {
        let state = await Self.directory().table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed,
            viewerSlug: Self.mySlug
        )
        let rows = try! #require(state.value)
        #expect(rows.count == 2)
        let mine = try! #require(rows.first { $0.modelAlias == "qwen3.5-9b-4bit" })
        let theirs = try! #require(rows.first { $0.modelAlias == "gemma-4-12b-4bit" })
        #expect(mine.summary.includesYours == true)
        // The row's count spans both execution cells, and one of them is mine.
        #expect(mine.summary.observationCount == 9)
        #expect(theirs.summary.includesYours == false)
    }

    @Test("Without a slug the table claims no row")
    func tableWithoutASlug() async {
        let state = await Self.directory().table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        )
        #expect(state.value?.allSatisfy { !$0.summary.includesYours } == true)
    }

    @Test("A row narrowed away from my cell is not marked yours")
    func narrowingRespectsTheBadge() {
        // My run is on the long prompt; the Generation speed column shows the
        // short one. The badge must follow the cells actually counted.
        let rows = CommunityTableProjection.rows(
            from: [
                CommunityTableProjection.Cell(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    protocolID: "rapid-community-speed", protocolVersion: 2,
                    caseID: CommunityBenchmarkMetrics.shortTextCaseID,
                    metricName: "decode_tps", executionKey: "bf16", samples: 5,
                    median: 25.9, unit: "tok/s",
                    contributorSlugs: ["modest-slate-wombat-545"]
                ),
                CommunityTableProjection.Cell(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    protocolID: "rapid-community-speed", protocolVersion: 2,
                    caseID: CommunityBenchmarkMetrics.longTextCaseID,
                    metricName: "decode_tps", executionKey: "bf16", samples: 3,
                    median: 12.0, unit: "tok/s",
                    contributorSlugs: [Self.mySlug]
                ),
            ],
            workload: .llm, metric: .generationSpeed, viewerSlug: Self.mySlug
        )
        #expect(rows.first?.summary.observationCount == 5)
        #expect(
            rows.first?.summary.includesYours == false,
            "a contribution to a cell this column does not display was claimed"
        )
    }

    @Test("An older protocol version's contribution is not claimed by the current one")
    func olderProtocolIsNotClaimed() {
        let rows = CommunityTableProjection.rows(
            from: [
                CommunityTableProjection.Cell(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    protocolID: "rapid-community-speed", protocolVersion: 2,
                    caseID: CommunityBenchmarkMetrics.shortTextCaseID,
                    metricName: "decode_tps", executionKey: "bf16", samples: 5,
                    median: 25.9, unit: "tok/s", contributorSlugs: []
                ),
                CommunityTableProjection.Cell(
                    modelAlias: "qwen3.5-9b-4bit", workload: .llm,
                    protocolID: "rapid-community-speed", protocolVersion: 1,
                    caseID: CommunityBenchmarkMetrics.shortTextCaseID,
                    metricName: "decode_tps", executionKey: "bf16", samples: 2,
                    median: 18.0, unit: "tok/s", contributorSlugs: [Self.mySlug]
                ),
            ],
            workload: .llm, metric: .generationSpeed, viewerSlug: Self.mySlug
        )
        #expect(rows.first?.summary.observationCount == 5)
        #expect(rows.first?.summary.includesYours == false)
    }

    // MARK: - Slug spelling

    @Test("Matching is by canonical slug, the same string the website keys on")
    func canonicalSlugMatching() async {
        // A cell whose contributor arrived with no `slug` field — older
        // receipts and the CLI payload predate it. `slug` then composes
        // `name + "-" + tag`, exactly as `normalize()` does on the website, so
        // both sides still agree about who this is.
        let legacy = #"""
        {"schema_version":1,"summary":[
          {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
           "samples":5,"contributors":[{"name":"swift-otter","tag":"4417"}],
           "latest_at":"2026-09-06T02:00:00Z"}
        ],"runs":[]}
        """#
        let state = await Self.directory(legacy).observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: Self.mySlug
        )
        #expect(state.value?.includesYours == true)
    }

    @Test("A near-miss slug is not this installation")
    func nearMissIsNotMatched() async {
        let state = await Self.directory().observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: "swift-otter-4418"
        )
        #expect(state.value?.includesYours == false)
    }

    @Test("A feed with no contributors field at all decodes and claims nothing")
    func missingContributorsField() async {
        let noField = #"""
        {"schema_version":1,"summary":[
          {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
           "samples":5,"latest_at":"2026-09-06T02:00:00Z"}
        ],"runs":[]}
        """#
        let state = await Self.directory(noField).observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: Self.mySlug
        )
        // Decoding must not fail, and a missing field is not a claim either way.
        #expect(state.value?.observationCount == 5)
        #expect(state.value?.includesYours == false)
    }

    // MARK: - Reload, after the optimistic state is gone

    @Test("A reload keeps the badge that the optimistic path set at publish time")
    func reloadPreservesTheBadge() async {
        // Publish: the optimistic path marks the count as including yours.
        var publication = CommunityPublicationState()
        let receipt = try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"sub-1","already_exists":false,
             "accepted_at":"2026-09-06T04:40:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
        let optimistic = publication.recordPublication(
            receipt: receipt, receiptSaved: true, scope: Self.scope("qwen3.5-9b-4bit"),
            observations: .ready(CommunityObservationSummary(observationCount: 8, isBounded: true))
        )
        #expect(optimistic.observations.value?.includesYours == true)

        // Relaunch: a brand-new state, and the answer comes from the server.
        let afterRestart = CommunityPublicationState()
        let read = await Self.directory().observations(
            for: Self.scope("qwen3.5-9b-4bit"), viewerSlug: Self.mySlug
        )
        var fresh = afterRestart
        let merged = fresh.merge(read, scope: Self.scope("qwen3.5-9b-4bit"))
        #expect(fresh.floors.isEmpty, "the fresh session holds no optimistic state")
        #expect(merged.value?.includesYours == true, "the badge did not survive the restart")
        #expect(merged.value?.observationCount == 9)
    }
}
