import Foundation
import Testing
@testable import Rapid

/// A comparison claims that two numbers describe the same artifact.
///
/// When this Mac's run pins a snapshot revision and a quantization, and the
/// published row states neither, that claim is unproven: the row may be an
/// entirely different build of the same repo. The previous behaviour reported
/// the row's median anyway whenever exactly one row matched — so the run was
/// compared against a population it might have nothing to do with, and the
/// user saw "12% faster than typical" with no way to know it was meaningless.
///
/// One under-specified row is not better evidence than several. It is the same
/// missing fact with a smaller sample, so both cases withhold.
@Suite("Under-specified comparisons")
struct CommunityUnderSpecifiedComparisonTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let repo = "mlx-community/Qwen3.5-9B-4bit"

    private static let execution = CommunityExecutionIdentity(
        rapidMLX: "0.13.4", computeDType: "bf16",
        speculativeDecodingMethod: "none", kvCacheMode: "quantized",
        kvCacheDType: "int8", prefillBackend: "gpu"
    )
    private static let fourBit = CommunityModelIdentity.Quantization(
        kind: "weights", baseDType: "bf16", method: "affine",
        weightBitsX2: 8, groupSize: 64
    )

    /// One row, stating only what the live projection states today.
    private static let underSpecifiedFeed = Self.feed(models: [
        #"{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved"}"#
    ], samples: [7], medians: [25.9])

    /// Two rows that are each fully determined, and both compatible.
    private static let twoDeterminedFeed = Self.feed(
        models: [
            #"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "resolved_revision":"aaaa111",
             "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                             "weight_bits_x2":8,"group_size":64}}
            """#,
            #"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "resolved_revision":"aaaa111",
             "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                             "weight_bits_x2":8,"group_size":64}}
            """#,
        ],
        samples: [5, 3],
        medians: [25.9, 22.4]
    )

    /// One fully determined row that matches exactly.
    private static let determinedFeed = Self.feed(
        models: [
            #"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "resolved_revision":"aaaa111",
             "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                             "weight_bits_x2":8,"group_size":64}}
            """#
        ],
        samples: [6],
        medians: [25.9]
    )

    private static func feed(
        models: [String], samples: [Int], medians: [Double]
    ) -> String {
        let indexed = Array(zip(zip(models, samples), medians).enumerated())
        let cells = indexed.map { _, value in
            let (pair, median) = value
            let (model, sampleCount) = pair
            return #"""
            {"task_type":"text_generation","model":\#(model),
             "machine":{"chip":"Apple M3 Pro","memory_gib":18},
             "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
               "speculative_decoding":{"method":"none"},
               "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
             "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
             "metric":{"name":"decode_tps","better":"higher","median":\#(median),"best":30.0},
             "samples":\#(sampleCount),"contributors":[],"latest_at":"2026-09-06T02:00:00Z"}
            """#
        }.joined(separator: ",")
        let runs = indexed.flatMap { cellIndex, value -> [String] in
            let ((model, sampleCount), _) = value
            return (0..<sampleCount).map { sampleIndex in
                #"{"submission_id":"fixture-\#(cellIndex)-\#(sampleIndex)","accepted_at":"2026-09-06T02:00:00Z","task_type":"text_generation","model":\#(model),"machine":{"chip":"Apple M3 Pro","memory_gib":18},"protocol":{"id":"rapid-community-speed","version":2}}"#
            }
        }.joined(separator: ",")
        return #"{"schema_version":1,"summary":[\#(cells)],"runs":[\#(runs)]}"#
    }

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

    private static func scope(
        revision: String?, quantization: CommunityModelIdentity.Quantization
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: "qwen3.5-9b-4bit", workload: .llm,
            protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: profile,
            modelIdentity: CommunityModelIdentity(
                repoID: repo, resolvedRevision: revision, quantization: quantization
            ),
            comparison: CommunityComparisonIdentity(
                caseID: "pp512-tg128", metricName: "decode_tps", execution: execution
            )
        )
    }

    // MARK: - One under-specified candidate

    @Test("A single under-specified row yields no median, range or verdict")
    func singleUnderSpecifiedRowWithholds() async {
        let state = await Self.directory(Self.underSpecifiedFeed).observations(
            for: Self.scope(revision: "aaaa111", quantization: Self.fourBit),
            viewerSlug: nil
        )
        // Not a count with a missing median — an explicit unavailable state,
        // so the branch rule withholds every comparison statistic.
        #expect(state.value == nil)
        if case let .unavailable(reason) = state {
            #expect(reason == .ambiguousIdentity)
        } else {
            Issue.record("expected an unavailable state, got \(state)")
        }

        let branch = CommunityContributionBranch.select(from: state)
        #expect(!branch.allowsComparisonStatistics)
        // And never a first-reference claim: rows for this repo do exist.
        #expect(!branch.allowsFirstReferenceLanguage)
    }

    @Test("A run pinning only a quantization is under-specified just the same")
    func quantizationAloneIsEnoughToWithhold() async {
        let state = await Self.directory(Self.underSpecifiedFeed).observations(
            for: Self.scope(revision: nil, quantization: Self.fourBit), viewerSlug: nil
        )
        #expect(state.value == nil)
    }

    @Test("The reason explains itself without blaming the user's run")
    func reasonReadsHonestly() {
        let message = CommunityUnavailableReason.ambiguousIdentity.message
        #expect(message.contains("Published results exist"))
        #expect(message.contains("same build"))
    }

    // MARK: - Multiple matching variants

    @Test("Several fully-determined matches also withhold rather than pick one")
    func multipleDeterminedMatchesWithhold() async {
        let state = await Self.directory(Self.twoDeterminedFeed).observations(
            for: Self.scope(revision: "aaaa111", quantization: Self.fourBit),
            viewerSlug: nil
        )
        #expect(state.value == nil)
        if case let .unavailable(reason) = state {
            #expect(reason == .ambiguousIdentity)
        } else {
            Issue.record("expected an unavailable state, got \(state)")
        }
        // Neither cell's median leaks through.
        #expect(state.value?.median != 25.9)
        #expect(state.value?.median != 22.4)
    }

    // MARK: - What still works

    @Test("A fully determined single match still compares")
    func determinedMatchStillCompares() async {
        let state = await Self.directory(Self.determinedFeed).observations(
            for: Self.scope(revision: "aaaa111", quantization: Self.fourBit),
            viewerSlug: nil
        )
        #expect(state.value?.median == 25.9)
        #expect(state.value?.observationCount == 6)
        #expect(CommunityContributionBranch.select(from: state).allowsComparisonStatistics)
    }

    @Test("A run with no resolved identity still compares against today's feed")
    func coldCacheRunStillCompares() async {
        // A cold cache resolves no revision and no quantization, so nothing is
        // being claimed about provenance and the live feed's bare row is a
        // legitimate population for it. Withholding here would break the
        // feature on the shipping service over a distinction neither side
        // makes.
        let state = await Self.directory(Self.underSpecifiedFeed).observations(
            for: Self.scope(revision: nil, quantization: .unknown), viewerSlug: nil
        )
        #expect(state.value?.median == 25.9)
        #expect(state.value?.observationCount == 7)
    }

    @Test("The coverage question is unaffected — it makes no provenance claim")
    func coverageIsUnaffected() async {
        var coverage = Self.scope(revision: "aaaa111", quantization: Self.fourBit)
        coverage.comparison = nil
        let state = await Self.directory(Self.underSpecifiedFeed).observations(
            for: coverage, viewerSlug: nil
        )
        // "How much is published for this model on this Mac" is answerable
        // without knowing which build each run used, and it never prints a
        // median anyway.
        #expect(state.value?.observationCount == 7)
        #expect(state.value?.median == nil)
    }
}
