import Foundation
import Testing
@testable import Rapid

/// A repo id is not a model.
///
/// `proto/community-benchmark/v1` identifies the primary component by its
/// source (`repo_id` + optional `subfolder` + `resolved_revision`), its
/// `quantization` facts and the `identity_strength` behind them.
/// `run_builder.unresolved_model_identity` fills all of that from the local
/// cache. Desktop decoded `repo_id` and threw the rest away, so a `4bit/`
/// subfolder, a different snapshot and a differently quantized artifact were
/// all the same thing to every consumer downstream — one count, one median, one
/// publication floor.
///
/// ## A note on the live service
///
/// `atomicValidateModel` currently **rejects** any submission whose source
/// carries `subfolder` or `resolved_revision`, or whose quantization is
/// anything but `{kind: unknown, base_dtype: unknown}`, and
/// `atomicPublicProjection` publishes only `{repo_id, identity_strength}`. So
/// on today's feed every cell is a bare repo id and these distinctions do not
/// yet arise *from the server*. They arise immediately from **local records**,
/// which do carry them — and the rules below are what keeps a widened
/// projection from silently merging variants later.
@Suite("Model identity")
struct CommunityModelIdentityTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let repo = "mlx-community/Qwen3.5-9B-4bit"

    private static let fourBit = CommunityModelIdentity.Quantization(
        kind: "weights", baseDType: "bf16", method: "affine",
        weightBitsX2: 8, groupSize: 64
    )
    private static let eightBit = CommunityModelIdentity.Quantization(
        kind: "weights", baseDType: "bf16", method: "affine",
        weightBitsX2: 16, groupSize: 64
    )

    // MARK: - Equality and keys

    @Test("Two identities differing only by revision are different models")
    func revisionDiscriminates() {
        let a = CommunityModelIdentity(repoID: Self.repo, resolvedRevision: "aaaa111")
        let b = CommunityModelIdentity(repoID: Self.repo, resolvedRevision: "bbbb222")
        #expect(a != b)
        #expect(!a.isSameVariant(as: b))
        #expect(a.canonicalKey != b.canonicalKey)
    }

    @Test("Two identities differing only by quantization are different models")
    func quantizationDiscriminates() {
        let a = CommunityModelIdentity(repoID: Self.repo, quantization: Self.fourBit)
        let b = CommunityModelIdentity(repoID: Self.repo, quantization: Self.eightBit)
        #expect(!a.isSameVariant(as: b))
        #expect(a.canonicalKey != b.canonicalKey)
    }

    @Test("Subfolder and identity strength discriminate too")
    func subfolderAndStrengthDiscriminate() {
        let root = CommunityModelIdentity(repoID: Self.repo)
        let nested = CommunityModelIdentity(repoID: Self.repo, subfolder: "4bit")
        let resolved = CommunityModelIdentity(repoID: Self.repo, identityStrength: "manifest")
        #expect(!root.isSameVariant(as: nested))
        #expect(!root.isSameVariant(as: resolved))
        #expect(Set([root, nested, resolved].map(\.canonicalKey)).count == 3)
    }

    @Test("An empty subfolder or revision is absence, not a distinct value")
    func emptyStringsAreAbsence() {
        #expect(
            CommunityModelIdentity(repoID: Self.repo, subfolder: "", resolvedRevision: "")
                == CommunityModelIdentity(repoID: Self.repo)
        )
    }

    // MARK: - Compatibility with a published cell

    @Test("A cell stating a different revision is never the same population")
    func publishedRevisionMismatchIsRejected() {
        let run = CommunityModelIdentity(repoID: Self.repo, resolvedRevision: "aaaa111")
        let cell = CommunityModelIdentity(repoID: Self.repo, resolvedRevision: "bbbb222")
        #expect(!run.isCompatible(withPublished: cell))
    }

    @Test("A cell stating a different quantization is never the same population")
    func publishedQuantizationMismatchIsRejected() {
        let run = CommunityModelIdentity(repoID: Self.repo, quantization: Self.fourBit)
        let cell = CommunityModelIdentity(repoID: Self.repo, quantization: Self.eightBit)
        #expect(!run.isCompatible(withPublished: cell))
    }

    @Test("A different repo or identity strength is never compatible")
    func repoAndStrengthAreRequired() {
        let run = CommunityModelIdentity(repoID: Self.repo)
        #expect(!run.isCompatible(withPublished: CommunityModelIdentity(repoID: "other/model")))
        #expect(
            !run.isCompatible(
                withPublished: CommunityModelIdentity(repoID: Self.repo, identityStrength: "manifest")
            )
        )
    }

    /// The live-service case, stated explicitly so the compromise is visible.
    @Test("A facet the feed omits neither matches nor blocks, and is flagged incomplete")
    func omittedFacetsAreUnderSpecified() {
        let run = CommunityModelIdentity(
            repoID: Self.repo, resolvedRevision: "aaaa111", quantization: Self.fourBit
        )
        // What `atomicPublicProjection` actually publishes today.
        let published = CommunityModelIdentity(repoID: Self.repo)
        // Requiring the revision would withhold every comparison on the live
        // feed, over a distinction the service does not record.
        #expect(run.isCompatible(withPublished: published))
        // But the match is under-specified, and callers must not present it as
        // exact when more than one cell survives.
        #expect(!run.facetsAreFullyDetermined(against: published))
        #expect(run.facetsAreFullyDetermined(against: run))

        // Symmetrically: a run whose cache could not establish a quantization
        // does not "disagree" with a cell that states one.
        let coldCache = CommunityModelIdentity(repoID: Self.repo)
        let stated = CommunityModelIdentity(repoID: Self.repo, quantization: Self.fourBit)
        #expect(coldCache.isCompatible(withPublished: stated))
        #expect(!coldCache.facetsAreFullyDetermined(against: stated))
    }

    // MARK: - Decoding

    @Test("The full local identity block decodes every facet")
    func decodesLocalIdentity() {
        let json = #"""
        {"schema_version":1,"identity_strength":"unresolved",
         "pipeline_kind":"text_generation",
         "components":[{"component_id":"primary","role":"primary",
           "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                     "subfolder":"4bit","resolved_revision":"a1b2c3d4"},
           "artifact":{"format":"mlx-safetensors"},
           "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                           "weight_bits_x2":8,"group_size":64}}]}
        """#
        let wire = try! JSONDecoder().decode(
            CommunityModelIdentity.Wire.self, from: Data(json.utf8)
        )
        let identity = try! #require(wire.identity)
        #expect(identity.repoID == "mlx-community/Qwen3.5-9B-4bit")
        #expect(identity.identityStrength == "unresolved")
        #expect(identity.subfolder == "4bit")
        #expect(identity.resolvedRevision == "a1b2c3d4")
        #expect(identity.quantization.kind == "weights")
        #expect(identity.quantization.weightBitsX2 == 8)
        #expect(identity.quantization.groupSize == 64)
        #expect(identity.quantization.displayName == "4-bit affine")
    }

    @Test("The flat summary shape decodes, with the omitted facets absent")
    func decodesSummaryShape() {
        let json = #"""
        {"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved"}
        """#
        let identity = try! #require(
            try! JSONDecoder()
                .decode(CommunityModelIdentity.Wire.self, from: Data(json.utf8)).identity
        )
        #expect(identity.repoID == "mlx-community/Qwen3.5-9B-4bit")
        #expect(identity.subfolder == nil)
        #expect(identity.resolvedRevision == nil)
        #expect(identity.quantization.isUnknown)
    }

    @Test("A 3.5-bit artifact is not rounded to 3 or 4")
    func halfBitWidths() {
        let identity = CommunityModelIdentity(
            repoID: Self.repo,
            quantization: .init(
                kind: "weights", baseDType: "bf16", method: "affine",
                weightBitsX2: 7, groupSize: 64
            )
        )
        #expect(identity.quantization.displayName == "3.5-bit affine")
    }

    // MARK: - Exact matching through the adapter

    /// Same repo, Mac, task, protocol, case, metric and execution — the cells
    /// differ **only** by revision and quantization.
    private static let variantFeed = #"""
    {"schema_version":1,"beta":true,"ranking_status":"unverified_not_ranked",
     "summary":[
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
          "components":[{"component_id":"primary","role":"primary",
            "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                      "resolved_revision":"aaaa111"},
            "artifact":{"format":"mlx-safetensors"},
            "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                            "weight_bits_x2":8,"group_size":64}}]},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
          "speculative_decoding":{"method":"none"},
          "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
        "samples":5,
        "contributors":[{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417",
                         "url":"/leaderboard/contributors/swift-otter-4417"}],
        "latest_at":"2026-09-06T02:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
          "components":[{"component_id":"primary","role":"primary",
            "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                      "resolved_revision":"bbbb222"},
            "artifact":{"format":"mlx-safetensors"},
            "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                            "weight_bits_x2":8,"group_size":64}}]},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
          "speculative_decoding":{"method":"none"},
          "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":22.4,"best":23.0},
        "samples":3,"contributors":[],"latest_at":"2026-09-06T01:00:00Z"},
       {"task_type":"text_generation",
        "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
          "components":[{"component_id":"primary","role":"primary",
            "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                      "resolved_revision":"aaaa111"},
            "artifact":{"format":"mlx-safetensors"},
            "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                            "weight_bits_x2":16,"group_size":64}}]},
        "machine":{"chip":"Apple M3 Pro","memory_gib":18},
        "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
          "speculative_decoding":{"method":"none"},
          "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
        "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
        "metric":{"name":"decode_tps","better":"higher","median":14.1,"best":14.8},
        "samples":4,"contributors":[],"latest_at":"2026-09-06T00:00:00Z"}
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
                caseID: "pp512-tg128", metricName: "decode_tps",
                execution: CommunityExecutionIdentity(
                    rapidMLX: "0.13.4", computeDType: "bf16",
                    speculativeDecodingMethod: "none", kvCacheMode: "quantized",
                    kvCacheDType: "int8", prefillBackend: "gpu"
                )
            )
        )
    }

    @Test("Exact matching selects the cell with this run's revision")
    func revisionSelectsTheCell() async {
        let state = await Self.directory(Self.variantFeed).observations(
            for: Self.scope(revision: "bbbb222", quantization: Self.fourBit),
            viewerSlug: nil
        )
        // 22.4 is the bbbb222 / 4-bit cell. Matching on repo id alone would
        // have returned 25.9, 22.4 or 14.1 depending on emission order.
        #expect(state.value?.median == 22.4)
        #expect(state.value?.observationCount == 3)
    }

    @Test("Exact matching selects the cell with this run's quantization")
    func quantizationSelectsTheCell() async {
        let state = await Self.directory(Self.variantFeed).observations(
            for: Self.scope(revision: "aaaa111", quantization: Self.eightBit),
            viewerSlug: nil
        )
        #expect(state.value?.median == 14.1)
        #expect(state.value?.observationCount == 4)
    }

    @Test("The remaining variant is reached by its own identity, and marked yours")
    func thirdVariant() async {
        let state = await Self.directory(Self.variantFeed).observations(
            for: Self.scope(revision: "aaaa111", quantization: Self.fourBit),
            viewerSlug: "swift-otter-4417"
        )
        #expect(state.value?.median == 25.9)
        #expect(state.value?.observationCount == 5)
        #expect(state.value?.includesYours == true)
    }

    @Test("A variant nobody has published is unknown, never another variant's number")
    func unpublishedVariantIsUnknown() async {
        let state = await Self.directory(Self.variantFeed).observations(
            for: Self.scope(revision: "cccc333", quantization: Self.fourBit),
            viewerSlug: nil
        )
        #expect(state.value == nil)
        #expect(!CommunityContributionBranch.select(from: state).allowsFirstReferenceLanguage)
    }

    @Test("An under-specified match over several cells withholds the comparison")
    func underSpecifiedMatchWithholdsTheMedian() async {
        // A run whose own identity is bare — no revision, unknown quantization,
        // which is what a cold cache produces. Three published cells are all
        // compatible with it, and no one of them is the population it belongs
        // to.
        let bare = CommunityBenchmarkScope(
            modelAlias: "qwen3.5-9b-4bit", workload: .llm,
            protocolID: "rapid-community-speed", protocolVersion: 2, macProfile: Self.profile,
            modelIdentity: CommunityModelIdentity(repoID: Self.repo),
            comparison: CommunityComparisonIdentity(
                caseID: "pp512-tg128", metricName: "decode_tps",
                execution: CommunityExecutionIdentity(
                    rapidMLX: "0.13.4", computeDType: "bf16",
                    speculativeDecodingMethod: "none", kvCacheMode: "quantized",
                    kvCacheDType: "int8", prefillBackend: "gpu"
                )
            )
        )
        let state = await Self.directory(Self.variantFeed).observations(
            for: bare, viewerSlug: nil
        )
        // A comparison claims two numbers describe the same artifact, and
        // nothing here establishes which of the three this run belongs to. A
        // count with a withheld median would still be read as "you are being
        // compared with these 12" — so the comparison is withheld entirely.
        #expect(state.value == nil)
        if case let .unavailable(reason) = state {
            #expect(reason == .ambiguousIdentity)
        } else {
            Issue.record("expected an unavailable state, got \(state)")
        }
        #expect(
            !CommunityContributionBranch.select(from: state).allowsComparisonStatistics
        )
    }

    // MARK: - Projections never fabricate a combined statistic

    @Test("The table sums variant counts and withholds the median")
    func tableWithholdsAcrossVariants() async {
        let state = await Self.directory(Self.variantFeed).table(
            macProfile: Self.profile, workload: .llm, metric: .generationSpeed,
            viewerSlug: "swift-otter-4417"
        )
        let rows = try! #require(state.value)
        // One row, because the alias is what the column is labelled with.
        #expect(rows.count == 1)
        #expect(rows[0].summary.observationCount == 12)
        #expect(
            rows[0].summary.median == nil,
            "a median was printed across three different artifacts"
        )
        // The badge still works: one of the counted cells is this contributor's.
        #expect(rows[0].summary.includesYours == true)
    }

    @Test("One variant alone keeps its median")
    func singleVariantKeepsItsMedian() async {
        let single = #"""
        {"schema_version":1,"summary":[
          {"task_type":"text_generation",
           "model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit","identity_strength":"unresolved",
             "components":[{"component_id":"primary","role":"primary",
               "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit",
                         "resolved_revision":"aaaa111"},
               "artifact":{"format":"mlx-safetensors"},
               "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                               "weight_bits_x2":8,"group_size":64}}]},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
           "samples":5,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"}
        ],"runs":[]}
        """#
        let rows = try! #require(
            await Self.directory(single).table(
                macProfile: Self.profile, workload: .llm, metric: .generationSpeed
            ).value
        )
        #expect(rows[0].summary.median == 25.9)
        #expect(rows[0].summary.observationCount == 5)
    }

    @Test("Coverage counts each variant separately")
    func coverageSeparatesVariants() async {
        let gaps = try! #require(
            await Self.directory(Self.variantFeed).coverageGaps(macProfile: Self.profile).value
        )
        // Three artifacts of one repo: 5, 3 and 4 samples. Two are thin, one is
        // at the threshold — and none of them is credited with the others' runs.
        #expect(gaps.count == 2)
        #expect(gaps.map(\.observationCount) == [3, 4])
        #expect(gaps.allSatisfy { $0.modelAlias == "qwen3.5-9b-4bit" })
        #expect(gaps.allSatisfy { !$0.isFirstResultOpportunity })
    }

    @Test("Coverage keyed on the alias alone would have hidden all three")
    func coverageWouldOtherwiseMerge() async {
        // The counterfactual, stated as a number: 5 + 3 + 4 = 12 is above the
        // threshold, so merging the variants would report no gap at all.
        let gaps = try! #require(
            await Self.directory(Self.variantFeed).coverageGaps(macProfile: Self.profile).value
        )
        #expect(!gaps.isEmpty, "the variants were merged into one covered pairing")
        #expect(gaps.reduce(0) { $0 + $1.observationCount } < 12)
    }

    // MARK: - Publication floors

    @Test("A floor confirmed for one variant does not defend another")
    func floorsAreKeyedOnTheVariant() {
        let a = Self.scope(revision: "aaaa111", quantization: Self.fourBit)
        let b = Self.scope(revision: "bbbb222", quantization: Self.fourBit)
        #expect(a != b)

        var publication = CommunityPublicationState()
        let receipt = try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"s","already_exists":false,"accepted_at":"2026-09-06T04:40:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
        let now = Date(timeIntervalSince1970: 1_789_000_000)
        _ = publication.recordPublication(
            receipt: receipt, receiptSaved: true, scope: a,
            observations: .ready(CommunityObservationSummary(observationCount: 5, isBounded: true)),
            now: now
        )
        #expect(publication.confirmedFloor(for: a)?.count == 6)
        #expect(publication.confirmedFloor(for: b) == nil)
        // B's real count passes through untouched.
        #expect(
            publication.merge(
                .ready(CommunityObservationSummary(observationCount: 3, isBounded: true)),
                scope: b, now: now
            ).value?.observationCount == 3
        )
    }
}
