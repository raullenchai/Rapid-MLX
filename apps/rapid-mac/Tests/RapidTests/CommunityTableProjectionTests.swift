import Foundation
import Testing
@testable import Rapid

/// The Community table shows one row per model. The feed does not.
///
/// `summary[]` groups by task, model, machine, protocol, **case** and
/// **execution**, so a single model on a single Mac routinely has four or more
/// cells. Mapping cell → row produced three defects at once: duplicate
/// `ForEach` identities (undefined behaviour in SwiftUI), a short-prompt median
/// printed directly above a long-prompt one under one "Generation speed"
/// header, and a "Time to first token" column filled with `decode_tps` numbers
/// because the requested metric was never consulted.
@Suite("Community table projection")
struct CommunityTableProjectionTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private func cell(
        _ alias: String,
        workload: CommunityWorkload = .llm,
        protocolID: String? = nil,
        version: Int = 2,
        caseID: String = CommunityBenchmarkMetrics.shortTextCaseID,
        metric: String = "decode_tps",
        execution: String = "bf16",
        samples: Int,
        median: Double?,
        unit: String? = "tok/s",
        contributors: Set<String> = []
    ) -> CommunityTableProjection.Cell {
        CommunityTableProjection.Cell(
            modelAlias: alias,
            workload: workload,
            protocolID: protocolID ?? Self.protocolID(for: workload),
            protocolVersion: version,
            caseID: caseID, metricName: metric, executionKey: execution,
            samples: samples, median: median, unit: unit,
            contributorSlugs: contributors
        )
    }

    private static func protocolID(for workload: CommunityWorkload) -> String {
        switch workload {
        case .llm: return "rapid-community-speed"
        case .image: return "rapid-image-speed"
        case .video: return "rapid-video-speed"
        }
    }

    // MARK: - One row per model

    @Test("Four cells for one model produce exactly one row")
    func oneRowPerModel() {
        let rows = CommunityTableProjection.rows(
            from: [
                cell("qwen3.5-9b-4bit", execution: "fp16", samples: 4, median: 31.0),
                cell(
                    "qwen3.5-9b-4bit", caseID: CommunityBenchmarkMetrics.longTextCaseID,
                    samples: 3, median: 12.0
                ),
                cell("qwen3.5-9b-4bit", samples: 5, median: 25.9),
                cell("qwen3.5-9b-4bit", version: 1, samples: 2, median: 18.0),
            ],
            workload: .llm,
            metric: .generationSpeed
        )
        #expect(rows.count == 1)
        #expect(rows[0].modelAlias == "qwen3.5-9b-4bit")
    }

    @Test("Row identities are unique, so ForEach is well defined")
    func rowIdentitiesAreUnique() {
        let cells = ["qwen3.5-9b-4bit", "gemma-4-12b-4bit", "z-image-turbo"].flatMap { alias in
            [
                cell(alias, execution: "fp16", samples: 4, median: 31.0),
                cell(alias, execution: "bf16", samples: 5, median: 25.9),
                cell(alias, version: 1, samples: 2, median: 18.0),
            ]
        }
        let rows = CommunityTableProjection.rows(from: cells, workload: .llm, metric: .generationSpeed)
        #expect(rows.count == 3)
        #expect(Set(rows.map(\.id)).count == rows.count)
    }

    // MARK: - Honouring the requested metric

    @Test("A metric with no cells yields no rows rather than another metric's numbers")
    func requestedMetricIsHonoured() {
        let cells = [cell("qwen3.5-9b-4bit", samples: 5, median: 25.9)]
        // The feed carries only decode_tps for this model.
        #expect(CommunityTableProjection.rows(from: cells, workload: .llm, metric: .generationSpeed).count == 1)
        // Asking for TTFT must not hand back the generation-speed cell.
        #expect(
            CommunityTableProjection.rows(from: cells, workload: .llm, metric: .timeToFirstToken).isEmpty
        )
        #expect(
            CommunityTableProjection.rows(from: cells, workload: .llm, metric: .peakMemory).isEmpty
        )
    }

    @Test("Each metric column reads its own cells")
    func metricsAreSeparate() {
        let cells = [
            cell("qwen3.5-9b-4bit", metric: "decode_tps", samples: 5, median: 25.9, unit: "tok/s"),
            cell("qwen3.5-9b-4bit", metric: "ttft_ms", samples: 5, median: 1490, unit: "ms"),
            cell(
                "qwen3.5-9b-4bit", metric: "peak_memory_mib", samples: 5, median: 6875, unit: "MiB"
            ),
        ]
        let speed = CommunityTableProjection.rows(from: cells, workload: .llm, metric: .generationSpeed)
        let ttft = CommunityTableProjection.rows(from: cells, workload: .llm, metric: .timeToFirstToken)
        let memory = CommunityTableProjection.rows(from: cells, workload: .llm, metric: .peakMemory)
        #expect(speed.first?.summary.median == 25.9)
        #expect(speed.first?.summary.unit == "tok/s")
        #expect(ttft.first?.summary.median == 1490)
        #expect(memory.first?.summary.median == 6875)
        // Never more than one row each, even though three cells exist.
        #expect(speed.count == 1 && ttft.count == 1 && memory.count == 1)
    }

    @Test("A row's workload is the requested one, so tabs never leak into each other")
    func workloadIsHonoured() {
        let cells = [
            cell("qwen3.5-9b-4bit", samples: 5, median: 25.9),
            cell(
                "z-image-turbo", workload: .image, version: 1, caseID: "render-1024",
                metric: "total_seconds", samples: 2, median: 4.6, unit: "s"
            ),
        ]
        let llm = CommunityTableProjection.rows(from: cells, workload: .llm, metric: .generationSpeed)
        let image = CommunityTableProjection.rows(from: cells, workload: .image, metric: .renderTime)
        #expect(llm.map(\.modelAlias) == ["qwen3.5-9b-4bit"])
        #expect(image.map(\.modelAlias) == ["z-image-turbo"])
    }

    // MARK: - Narrowing rules

    @Test("Only the newest protocol version contributes")
    func newestProtocolVersionOnly() {
        let rows = CommunityTableProjection.rows(
            from: [
                cell("qwen3.5-9b-4bit", version: 2, samples: 5, median: 25.9),
                cell("qwen3.5-9b-4bit", version: 1, samples: 2, median: 18.0),
            ],
            workload: .llm, metric: .generationSpeed
        )
        // v1 results were measured under a different protocol; folding them in
        // would report 7 observations of a population that does not exist.
        #expect(rows.first?.summary.observationCount == 5)
        #expect(rows.first?.summary.median == 25.9)
    }

    @Test("Generation speed means the short prompt, not whichever case came first")
    func canonicalCaseOnly() {
        let rows = CommunityTableProjection.rows(
            from: [
                // Long prompt listed first — the order that used to decide it.
                cell(
                    "qwen3.5-9b-4bit", caseID: CommunityBenchmarkMetrics.longTextCaseID,
                    samples: 3, median: 12.0
                ),
                cell("qwen3.5-9b-4bit", samples: 5, median: 25.9),
            ],
            workload: .llm, metric: .generationSpeed
        )
        #expect(rows.first?.summary.median == 25.9)
        #expect(rows.first?.summary.observationCount == 5)
    }

    @Test("A model measured only on the long prompt is omitted, not mislabelled")
    func nonCanonicalOnlyModelIsOmitted() {
        let rows = CommunityTableProjection.rows(
            from: [
                cell(
                    "gemma-4-12b-4bit", caseID: CommunityBenchmarkMetrics.longTextCaseID,
                    samples: 6, median: 9.5
                )
            ],
            workload: .llm, metric: .generationSpeed
        )
        // Showing 9.5 under "Generation speed" would put a 2 048-token-prompt
        // number in a column every other row measures at 512.
        #expect(rows.isEmpty)
    }

    @Test("Execution variants sum the count and withhold the median")
    func executionVariantsWithholdTheMedian() {
        let rows = CommunityTableProjection.rows(
            from: [
                cell("qwen3.5-9b-4bit", execution: "fp16", samples: 4, median: 31.0),
                cell("qwen3.5-9b-4bit", execution: "bf16", samples: 5, median: 25.9),
            ],
            workload: .llm, metric: .generationSpeed
        )
        // Each run is in exactly one cell, so 9 is exact…
        #expect(rows.first?.summary.observationCount == 9)
        // …but no single number is the median of a bf16+fp16 population, and
        // picking either one would silently report the other's hardware story.
        #expect(rows.first?.summary.median == nil)
    }

    @Test("A single execution variant keeps its median")
    func singleVariantKeepsTheMedian() {
        let rows = CommunityTableProjection.rows(
            from: [cell("qwen3.5-9b-4bit", samples: 5, median: 25.9)],
            workload: .llm, metric: .generationSpeed
        )
        #expect(rows.first?.summary.median == 25.9)
        #expect(rows.first?.summary.observationCount == 5)
    }

    @Test("A metric with no canonical case and several cases withholds the median")
    func ambiguousCaseWithoutACanonicalOne() {
        let rows = CommunityTableProjection.rows(
            from: [
                cell(
                    "z-image-turbo", workload: .image, version: 1, caseID: "render-1024",
                    metric: "total_seconds", samples: 2, median: 4.6, unit: "s"
                ),
                cell(
                    "z-image-turbo", workload: .image, version: 1, caseID: "render-2048",
                    metric: "total_seconds", samples: 3, median: 11.2, unit: "s"
                ),
            ],
            workload: .image, metric: .renderTime
        )
        #expect(rows.count == 1)
        #expect(rows.first?.summary.observationCount == 5)
        #expect(rows.first?.summary.median == nil)
    }

    @Test("Rows are always marked bounded, so no count reads as a total")
    func rowsAreBounded() {
        let rows = CommunityTableProjection.rows(
            from: [cell("qwen3.5-9b-4bit", samples: 5, median: 25.9)],
            workload: .llm, metric: .generationSpeed
        )
        #expect(rows.first?.summary.isBounded == true)
        // …and therefore never claim to be the first.
        #expect(
            CommunityContributionBranch.select(from: .ready(rows[0].summary))
                != .firstReference
        )
    }

    // MARK: - Through the production adapter

    /// The same ambiguous fixture the adapter tests use, driven end to end.
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

    private static func directory(_ body: String) -> CommunityBenchmarkAPIDirectory {
        CommunityBenchmarkAPIDirectory(
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

    @Test("The ambiguous feed renders one row, not four")
    func ambiguousFeedThroughTheAdapter() async {
        let state = await Self.directory(Self.ambiguousFeed).table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed
        )
        let rows = try! #require(state.value)
        #expect(rows.count == 1)
        #expect(Set(rows.map(\.id)).count == 1)

        let row = rows[0]
        #expect(row.modelAlias == "qwen3.5-9b-4bit")
        // v1 (2 samples) and the long prompt (3) are excluded; the two v2
        // short-prompt cells remain: 4 fp16 + 5 bf16.
        #expect(row.summary.observationCount == 9)
        // They disagree on dtype, so the table shows the count without a
        // median rather than 31.0 or 25.9.
        #expect(row.summary.median == nil)
        #expect(row.summary.unit == "tok/s")
    }

    @Test("Asking the ambiguous feed for TTFT reports unavailable, not decode_tps")
    func ambiguousFeedWrongMetric() async {
        let state = await Self.directory(Self.ambiguousFeed).table(
            macProfile: Self.profile, workload: .llm, metric: .timeToFirstToken
        )
        // No TTFT cells exist in this bounded window. That is not evidence
        // that none were ever published.
        #expect(state.value == nil)
        if case let .unavailable(reason) = state {
            #expect(reason == .boundedFeed)
        } else {
            Issue.record("expected an unavailable state, got \(state)")
        }
    }
}
