import Foundation
import Testing
@testable import Rapid

/// "Where your Mac can help" has to name a real number.
///
/// The defect: the threshold was applied to *individual* summary cells and one
/// surviving cell was then kept per model. A model with a 4-sample fp16 cell
/// and a 5-sample bf16 cell has nine published results on that Mac — but the
/// 5-sample cell failed `samples < 5` and was dropped, the 4-sample cell
/// survived, and the card read "Only 4 published results on an Apple M3 Pro so
/// far". That number is not the model's coverage, is not the displayed cell's
/// coverage, and flips depending on which group the worker emitted first.
@Suite("Coverage projection")
struct CommunityCoverageProjectionTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let threshold = CommunityBenchmarkAPIDirectory.underRepresentedBelow

    private func cell(
        _ alias: String,
        workload: CommunityWorkload = .llm,
        protocolID: String = "rapid-community-speed",
        version: Int = 2,
        caseID: String = CommunityBenchmarkMetrics.shortTextCaseID,
        execution: String = "bf16",
        samples: Int
    ) -> CommunityTableProjection.Cell {
        CommunityTableProjection.Cell(
            modelAlias: alias, workload: workload, protocolID: protocolID,
            protocolVersion: version, caseID: caseID, metricName: "decode_tps",
            executionKey: execution, samples: samples, median: 25.9, unit: "tok/s"
        )
    }

    // MARK: - Aggregate first, threshold second

    @Test("4 + 5 execution cells are nine observations, not an under-represented four")
    func executionCellsAggregateBeforeTheThreshold() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [
                cell("qwen3.5-9b-4bit", execution: "fp16", samples: 4),
                cell("qwen3.5-9b-4bit", execution: "bf16", samples: 5),
            ],
            below: Self.threshold
        )
        // Nine is at or above the threshold, so this pairing is not thin at
        // all and must not appear.
        #expect(gaps.isEmpty, "a nine-observation model was reported as a gap")
    }

    @Test("The same cells never produce the phantom \"Only 4\" count")
    func theOldCountIsUnreachable() {
        let cells = [
            cell("qwen3.5-9b-4bit", execution: "fp16", samples: 4),
            cell("qwen3.5-9b-4bit", execution: "bf16", samples: 5),
        ]
        // Raise the threshold so the pairing *is* reported, and check the
        // number it is reported with.
        let gaps = CommunityCoverageProjection.gaps(from: cells, below: 20)
        #expect(gaps.count == 1)
        #expect(gaps.first?.observationCount == 9)
        #expect(gaps.first?.observationCount != 4, "an individual cell's count leaked through")
        #expect(gaps.first?.observationCount != 5)
    }

    @Test("Cell order cannot change the reported count")
    func orderIndependent() {
        let cells = [
            cell("qwen3.5-9b-4bit", execution: "fp16", samples: 4),
            cell("qwen3.5-9b-4bit", execution: "bf16", samples: 5),
            cell(
                "qwen3.5-9b-4bit", caseID: CommunityBenchmarkMetrics.longTextCaseID,
                execution: "bf16", samples: 3
            ),
        ]
        let forward = CommunityCoverageProjection.gaps(from: cells, below: 20)
        let reversed = CommunityCoverageProjection.gaps(from: cells.reversed(), below: 20)
        #expect(forward == reversed)
        // Every case cell counts once: each run is in exactly one summary group.
        #expect(forward.first?.observationCount == 12)
    }

    @Test("A genuinely thin pairing is still reported, with its real total")
    func thinPairingIsReported() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [
                cell("z-image-turbo", workload: .image, protocolID: "rapid-image-speed",
                     version: 1, caseID: "render-1024", samples: 1),
                cell("z-image-turbo", workload: .image, protocolID: "rapid-image-speed",
                     version: 1, caseID: "render-1024", execution: "fp16", samples: 1),
            ],
            below: Self.threshold
        )
        #expect(gaps.count == 1)
        #expect(gaps.first?.observationCount == 2)
        #expect(gaps.first?.isBounded == true)
        #expect(gaps.first?.workload == .image)
    }

    // MARK: - Never a first-result claim

    @Test("A reported gap is never a first-result opportunity")
    func neverFirstResult() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [cell("qwen3.5-9b-4bit", samples: 1)], below: Self.threshold
        )
        #expect(gaps.first?.observationCount == 1)
        #expect(gaps.allSatisfy { !$0.isFirstResultOpportunity })
    }

    @Test("A malformed zero-sample cell is dropped, not turned into \"nobody has run this\"")
    func zeroSamplesAreDropped() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [cell("qwen3.5-9b-4bit", samples: 0)], below: Self.threshold
        )
        // `observationCount == 0` is the one value that turns on FIRST RESULT
        // NEEDED, and a bounded feed can never support that claim.
        #expect(gaps.isEmpty)
    }

    // MARK: - Grouping boundaries

    @Test("Only the newest protocol version counts toward coverage")
    func newestProtocolVersionOnly() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [
                cell("qwen3.5-9b-4bit", version: 2, samples: 2),
                cell("qwen3.5-9b-4bit", version: 1, samples: 40),
            ],
            below: Self.threshold
        )
        // Forty results under the old protocol are not evidence that the
        // current one is well covered.
        #expect(gaps.first?.observationCount == 2)
    }

    @Test("Different models and workloads are separate pairings")
    func pairingsAreDistinct() {
        let gaps = CommunityCoverageProjection.gaps(
            from: [
                cell("qwen3.5-9b-4bit", samples: 2),
                cell("gemma-4-12b-4bit", samples: 1),
                cell("z-image-turbo", workload: .image, protocolID: "rapid-image-speed",
                     version: 1, caseID: "render-1024", samples: 3),
            ],
            below: Self.threshold
        )
        #expect(gaps.count == 3)
        // Fewest observations first — the model that most needs a sample.
        #expect(gaps.map(\.modelAlias) == ["gemma-4-12b-4bit", "qwen3.5-9b-4bit", "z-image-turbo"])
        #expect(Set(gaps.map(\.id)).count == 3)
    }

    // MARK: - Through the production adapter

    /// One model with two execution cells on this Mac (4 + 5 = 9) and one
    /// genuinely thin model (2).
    private static let feed = #"""
    {"schema_version":1,"summary":[
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"fp16"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":31.0,"best":33.0},
       "samples":4,"contributors":[],"latest_at":"2026-09-06T04:00:00Z"},
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
       "samples":5,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"},
      {"task_type":"image_generation","model":{"repo_id":"mlx-community/z-image-turbo"},
       "machine":{"chip":"Apple M3 Pro","memory_gib":18},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
       "protocol":{"id":"rapid-image-speed","version":1},"case_id":"render-1024",
       "metric":{"name":"total_seconds","better":"lower","median":4.6,"best":4.4},
       "samples":2,"contributors":[],"latest_at":"2026-09-05T04:00:00Z"},
      {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
       "machine":{"chip":"Apple M4 Max","memory_gib":48},
       "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
       "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
       "metric":{"name":"decode_tps","better":"higher","median":58.1,"best":60.0},
       "samples":1,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"}
    ],"runs":[]}
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

    @Test("The adapter reports the thin model and not the nine-observation one")
    func adapterAggregatesBeforeThresholding() async {
        let state = await Self.directory(Self.feed).coverageGaps(macProfile: Self.profile)
        let gaps = try! #require(state.value)
        #expect(gaps.map(\.modelAlias) == ["z-image-turbo"])
        #expect(gaps.first?.observationCount == 2)
        // The 4 + 5 model is covered, so it is absent — and it is certainly
        // never described as having only four results.
        #expect(!gaps.contains { $0.modelAlias == "qwen3.5-9b-4bit" })
    }

    @Test("Another Mac's thin coverage is not this Mac's gap")
    func otherMacIsExcluded() async {
        // The M4 Max row has one sample, which would be the thinnest pairing of
        // all if the machine filter were dropped.
        let state = await Self.directory(Self.feed).coverageGaps(macProfile: Self.profile)
        #expect(state.value?.allSatisfy { $0.modelAlias != "qwen3.5-9b-4bit" } == true)
    }

    @Test("An empty gap list is not a claim that the catalogue is covered")
    func emptyIsNotACoverageClaim() async {
        // Every pairing in this feed is comfortably above the threshold.
        let covered = #"""
        {"schema_version":1,"summary":[
          {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
           "samples":40,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"}
        ],"runs":[]}
        """#
        let gaps = await Self.directory(covered).coverageGaps(macProfile: Self.profile).value
        #expect(gaps?.isEmpty == true)
        #expect(gaps?.contains { $0.isFirstResultOpportunity } != true)

        // And the copy shown for that empty list makes no claim about the
        // catalogue. A model with NO results never appears in the bounded feed
        // at all, so an empty gap list is the one piece of evidence that cannot
        // establish full coverage.
        let band = CommunityBenchmarkCopy.coverageAllCovered
        let text = "\(band.title) \(band.message)".lowercased()
        // The exact claim that was there before: "Every model the catalogue
        // offers already has results on this Mac profile."
        #expect(!text.contains("already has results"))
        #expect(!text.contains("every model the catalogue"))
        // What it says instead is bounded to what was actually read, and it
        // disclaims the catalogue-wide reading explicitly.
        #expect(text.contains("recent published results"))
        #expect(text.contains("isn’t a statement about every model"))
    }
}
