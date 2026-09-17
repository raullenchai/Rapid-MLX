import Foundation
import Testing
@testable import Rapid

/// The public feed's model identity is **flat**; the local record's is nested.
///
/// `CommunityModelIdentity.Wire` read variant facets only out of
/// `components[0]`, so every `subfolder`, `resolved_revision` and top-level
/// `quantization` the public projection sends was dropped on the floor. Two
/// published variants of one repo then decoded as the same model, and the
/// projections — which exist precisely to keep variants apart — were handed
/// identical keys and merged them.
@Suite("Public model identity decoding")
struct CommunityPublicIdentityDecodingTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static func decode(_ json: String) -> CommunityModelIdentity? {
        try! JSONDecoder()
            .decode(CommunityModelIdentity.Wire.self, from: Data(json.utf8))
            .identity
    }

    // MARK: - The flat public shape

    @Test("The flat public shape carries revision, subfolder and quantization")
    func flatShapeSurvivesDecoding() {
        let identity = try! #require(
            Self.decode(#"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit",
             "identity_strength":"unresolved",
             "subfolder":"4bit",
             "resolved_revision":"a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
             "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                             "weight_bits_x2":8,"group_size":64}}
            """#)
        )
        #expect(identity.repoID == "mlx-community/Qwen3.5-9B-4bit")
        #expect(identity.identityStrength == "unresolved")
        #expect(identity.subfolder == "4bit")
        #expect(identity.resolvedRevision == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2")
        #expect(identity.quantization.kind == "weights")
        #expect(identity.quantization.baseDType == "bf16")
        #expect(identity.quantization.method == "affine")
        #expect(identity.quantization.weightBitsX2 == 8)
        #expect(identity.quantization.groupSize == 64)
    }

    @Test("Today's live shape — repo id and identity strength only — still decodes")
    func todaysLiveShape() {
        // What `atomicPublicProjection` emits at the contract commit.
        let identity = try! #require(
            Self.decode(#"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved"}
            """#)
        )
        #expect(identity.repoID == "mlx-community/Qwen3.5-9B-4bit")
        #expect(identity.subfolder == nil)
        #expect(identity.resolvedRevision == nil)
        #expect(identity.quantization.isUnknown)
    }

    @Test("The nested local shape still wins where both are present")
    func nestedShapeTakesPrecedence() {
        // A producer that sends both: the nested block is its own most
        // specific record of what it loaded.
        let identity = try! #require(
            Self.decode(#"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "subfolder":"flat-value","resolved_revision":"ffff",
             "quantization":{"kind":"none","base_dtype":"fp16"},
             "components":[{"component_id":"primary","role":"primary",
               "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                         "subfolder":"nested-value","resolved_revision":"aaaa"},
               "artifact":{"format":"mlx-safetensors"},
               "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                               "weight_bits_x2":8}}]}
            """#)
        )
        #expect(identity.subfolder == "nested-value")
        #expect(identity.resolvedRevision == "aaaa")
        #expect(identity.quantization.kind == "weights")
    }

    @Test("Each facet falls back independently")
    func facetsFallBackIndependently() {
        // Nested source, flat quantization — a mixture a widened projection
        // could plausibly send.
        let identity = try! #require(
            Self.decode(#"""
            {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "quantization":{"kind":"weights","base_dtype":"bf16","weight_bits_x2":16},
             "components":[{"component_id":"primary","role":"primary",
               "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                         "resolved_revision":"bbbb"},
               "artifact":{"format":"mlx-safetensors"}}]}
            """#)
        )
        #expect(identity.resolvedRevision == "bbbb")
        #expect(identity.quantization.weightBitsX2 == 16)
    }

    // MARK: - Through the live feed

    /// A `/api/benchmarks/atomic/public` body in the flat shape, with two rows
    /// that differ only by the facets the old decoder discarded.
    private static let flatVariantFeed = #"""
    {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
     "summary":[
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit",
                 "identity_strength":"unresolved",
                 "resolved_revision":"aaaa111",
                 "quantization":{"kind":"weights","base_dtype":"bf16",
                                 "method":"affine","weight_bits_x2":8,"group_size":64}},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
          "speculative_decoding":{"method":"none"},
          "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
        "samples":5,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit",
                 "identity_strength":"unresolved",
                 "resolved_revision":"bbbb222",
                 "quantization":{"kind":"weights","base_dtype":"bf16",
                                 "method":"affine","weight_bits_x2":16,"group_size":64}},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
          "speculative_decoding":{"method":"none"},
          "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":14.1,"best":14.8},
        "samples":4,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"}
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

    private static func quantization(bitsX2: Int) -> CommunityModelIdentity.Quantization {
        .init(
            kind: "weights", baseDType: "bf16", method: "affine",
            weightBitsX2: bitsX2, groupSize: 64
        )
    }

    private static func scope(
        revision: String, bitsX2: Int
    ) -> CommunityBenchmarkScope {
        CommunityBenchmarkScope(
            modelAlias: "qwen3.5-9b-4bit", workload: .llm,
            protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: profile,
            modelIdentity: CommunityModelIdentity(
                repoID: "mlx-community/Qwen3.5-9B-4bit",
                resolvedRevision: revision,
                quantization: quantization(bitsX2: bitsX2)
            ),
            comparison: CommunityComparisonIdentity(
                caseID: "pp512-tg128", metricName: "decode_tps",
                execution: CommunityExecutionIdentity(
                    rapidMLX: "0.13.4", computeDType: "bf16",
                    speculativeDecodingMethod: "none", kvCacheMode: "quantized",
                    kvCacheDType: "int8", prefillBackend: "gpu"
                )
            )
        )
    }

    @Test("Flat variant rows select the right cell instead of merging")
    func flatVariantsSelectTheRightCell() async {
        let directory = Self.directory(Self.flatVariantFeed)
        let fourBit = await directory.observations(
            for: Self.scope(revision: "aaaa111", bitsX2: 8), viewerSlug: nil
        )
        #expect(fourBit.value?.median == 25.9)
        #expect(fourBit.value?.observationCount == 5)

        let eightBit = await directory.observations(
            for: Self.scope(revision: "bbbb222", bitsX2: 16), viewerSlug: nil
        )
        #expect(eightBit.value?.median == 14.1)
        #expect(eightBit.value?.observationCount == 4)
    }

    @Test("Flat variant rows are not merged into one table median")
    func flatVariantsAreNotMergedInTheTable() async {
        let rows = try! #require(
            await Self.directory(Self.flatVariantFeed).table(
                macProfile: Self.profile, workload: .llm, metric: .generationSpeed
            ).value
        )
        #expect(rows.count == 1)
        #expect(rows[0].summary.observationCount == 9)
        #expect(
            rows[0].summary.median == nil,
            "two artifacts were averaged into one median"
        )
    }

    @Test("Flat variant rows are counted separately for coverage")
    func flatVariantsAreSeparateForCoverage() async {
        let gaps = try! #require(
            await Self.directory(Self.flatVariantFeed)
                .coverageGaps(macProfile: Self.profile).value
        )
        #expect(gaps.map(\.observationCount) == [4])
    }
}
