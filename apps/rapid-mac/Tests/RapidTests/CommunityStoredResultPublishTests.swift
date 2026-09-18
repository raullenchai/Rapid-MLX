import Foundation
import Testing
@testable import Rapid

/// Publishing a stored result from My Results is a statement about **that run**.
///
/// `comparisonScope(for:)` used to start from the currently selected model:
///
/// ```swift
/// guard var scope else { return nil }   // built from `selected`
/// scope.comparison = result.comparisonIdentity
/// ```
///
/// So a receipt for a month-old gemma run, published while the Run tab happened
/// to be on qwen, was attributed to qwen — qwen's alias in the celebration,
/// qwen's protocol version in the query, qwen's count incremented, qwen's floor
/// confirmed — while carrying gemma's case and execution identity. Every field
/// now comes from the record.
@Suite("Publishing a stored result")
struct CommunityStoredResultPublishTests {
    /// The catalogue's repo → alias mapping. The only thing a scope needs that
    /// the record does not carry.
    private static let alias: @Sendable (String) -> String = {
        $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased()
    }

    private static func decode(_ json: String) -> CommunityBenchmarkResult {
        try! JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    /// Model X: an older text run, measured on an **M2 Max** under protocol v1,
    /// with a resolved snapshot revision and real quantization facts.
    private static let storedRunX = #"""
    {"schema_version":1,"run_id":"run-x",
     "started_at":"2026-08-02T10:00:00Z","completed_at":"2026-08-02T10:12:00Z",
     "collector":{"name":"rapid-mlx-community-bench","version":"0.12.0"},
     "execution":{"config_digest":"sha256:x",
       "runtime":{"mlx":"0.31.0","python":"3.12.8","rapid_mlx":"0.12.0"},
       "resources":{"max_concurrency":1,"compute_dtype":"fp16"},
       "task":{"kind":"text_generation","language":{
         "speculative_decoding":{"method":"none"},
         "kv_cache":{"mode":"standard","dtype":"fp16"},
         "prefill_backend":"gpu"}}},
     "machine":{"os":{"version":"14.6.1"},
       "profile":{"chip":"Apple M2 Max","cpu_cores":12,"gpu_cores":30,"memory_gib":32}},
     "measurements":[
       {"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4000,
        "output_tokens":129,"peak_active_memory_mib":9100,"round_index":1,
        "total_duration_ms":5200,"ttft_ms":1200}],
     "model":{"schema_version":1,"identity_strength":"unresolved",
       "pipeline_kind":"text_generation",
       "components":[{"component_id":"primary","role":"primary",
         "source":{"kind":"huggingface","repo_id":"mlx-community/Gemma-4-12B-4bit",
                   "resolved_revision":"a1b2c3d4e5f6"},
         "artifact":{"format":"mlx-safetensors"},
         "quantization":{"kind":"weights","base_dtype":"bf16","method":"affine",
                         "weight_bits_x2":8,"group_size":64}}]},
     "outcome":{"status":"completed"},
     "workload":{"protocol_id":"rapid-community-speed","protocol_version":1,
       "protocol_strength":"registered","task_type":"text_generation","concurrency":1,
       "cases":[{"case_id":"pp512-tg128","measured_rounds":5,"target_output_tokens":128,
                 "target_prompt_tokens":512,"warmup_rounds":1}]}}
    """#

    /// Model Y: what the Run tab is showing — a different model, a different
    /// Mac, a different protocol version, a different execution.
    private static func modelY() -> CommunityBenchmarkModel {
        CommunityBenchmarkModel(
            entry: ModelEntry(
                alias: "qwen3.5-9b-4bit", hfRepo: "mlx-community/Qwen3.5-9B-4bit",
                sizeOnDisk: "6.5 GB", cached: true, taskTypes: [.textGeneration]
            ),
            task: .textGeneration,
            protocolName: "Rapid Community Speed v2",
            protocolID: "rapid-community-speed",
            protocolVersion: 2,
            isFocus: true, estimatedMemoryGib: 6, memoryFit: "fits"
        )
    }

    private static let macNow = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)
    private static let publishedAt = Date(timeIntervalSince1970: 1_789_000_000)

    private static func receipt() -> CommunityBenchmarkReceipt {
        try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"sub-x","already_exists":false,
             "accepted_at":"2026-09-06T04:40:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
    }

    // MARK: - The scope is the record's

    @Test("A stored result's scope comes from the record, not the selected model")
    func scopeIsTheRecords() {
        let x = Self.decode(Self.storedRunX)
        let scope = try! #require(x.communityScope(alias: Self.alias))

        // Model — X, not Y.
        #expect(scope.modelAlias == "gemma-4-12b-4bit")
        #expect(scope.modelIdentity?.repoID == "mlx-community/Gemma-4-12B-4bit")
        // Protocol — the version the run was measured under, not the current one.
        #expect(scope.protocolID == "rapid-community-speed")
        #expect(scope.protocolVersion == 1)
        // The Mac that produced the record, not the Mac reading it.
        #expect(scope.macProfile.chip == "Apple M2 Max")
        #expect(scope.macProfile.memoryGiB == 32)
        #expect(scope.macProfile != Self.macNow)
        // Case, metric and execution.
        #expect(scope.comparison?.caseID == "pp512-tg128")
        #expect(scope.comparison?.metricName == "decode_tps")
        #expect(scope.comparison?.execution.computeDType == "fp16")
        #expect(scope.comparison?.execution.kvCacheMode == "standard")
        #expect(scope.comparison?.execution.rapidMLX == "0.12.0")
        // And the artifact facts the record resolved.
        #expect(scope.modelIdentity?.resolvedRevision == "a1b2c3d4e5f6")
        #expect(scope.modelIdentity?.quantization.weightBitsX2 == 8)
        #expect(scope.modelIdentity?.quantization.method == "affine")
    }

    @Test("A record with no machine cannot be scoped, and so is never compared")
    func noMachineMeansNoScope() {
        // `machine` is optional on the record — a run aborted before the
        // hardware snapshot has none. Falling back to *this* Mac would file
        // the result under hardware it was never measured on.
        let noMachine = #"""
        {"schema_version":1,"run_id":"run-nm","completed_at":"2026-08-02T10:12:00Z",
         "execution":{"config_digest":"sha256:x",
           "runtime":{"mlx":"0.31.0","python":"3.12.8","rapid_mlx":"0.12.0"},
           "resources":{"compute_dtype":"fp16"},"task":{"kind":"text_generation"}},
         "measurements":[{"case_id":"pp512-tg128","completed":true,
           "decode_duration_ms":4000,"output_tokens":129,"total_duration_ms":5200}],
         "model":{"schema_version":1,"identity_strength":"unresolved",
           "pipeline_kind":"text_generation",
           "components":[{"component_id":"primary","role":"primary",
             "source":{"kind":"huggingface","repo_id":"mlx-community/Gemma-4-12B-4bit"},
             "artifact":{"format":"mlx-safetensors"},
             "quantization":{"kind":"unknown","base_dtype":"unknown"}}]},
         "outcome":{"status":"completed"},
         "workload":{"protocol_id":"rapid-community-speed","protocol_version":1,
           "task_type":"text_generation",
           "cases":[{"case_id":"pp512-tg128","measured_rounds":5,"warmup_rounds":1}]}}
        """#
        #expect(Self.decode(noMachine).communityScope(alias: Self.alias) == nil)
    }

    @Test("A record with no registered protocol cannot be scoped")
    func noProtocolMeansNoScope() {
        let noProtocol = Self.storedRunX
            .replacingOccurrences(of: #""protocol_id":"rapid-community-speed","protocol_version":1,"#, with: "")
        #expect(Self.decode(noProtocol).communityScope(alias: Self.alias) == nil)
    }

    // MARK: - X's receipt must not touch Y

    @Test("Publishing stored X while Y is on screen leaves Y's count and floor alone")
    func publishingXDoesNotTouchY() async {
        let x = Self.decode(Self.storedRunX)
        let scopeX = try! #require(x.communityScope(alias: Self.alias))
        // Y is selected and its observations are on screen: 12 published runs.
        let scopeY = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.modelY(), macProfile: Self.macNow,
                latestResult: nil, alias: Self.alias
            )
        )
        let visibleY: CommunityDataState<CommunityObservationSummary> = .ready(
            CommunityObservationSummary(observationCount: 12, unit: "tok/s", isBounded: true)
        )
        #expect(scopeX != scopeY)

        // The user publishes X from My Results. The context is captured from
        // the RECORD; the visible scope is Y's, so Y's count is not carried in.
        let context = CommunityPublicationContext.capture(
            runID: x.id,
            resultScope: scopeX,
            visibleScope: scopeY,
            observations: visibleY,
            branch: CommunityContributionBranch.select(from: visibleY)
        )
        #expect(context.scope == scopeX)
        #expect(context.runID == "run-x")

        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true,
            context: context, visibleScope: scopeY, now: Self.publishedAt
        )

        // Y's number is untouched, on screen and in the state.
        #expect(!outcome.appliesToVisibleScope)
        #expect(publication.confirmedFloor(for: scopeY) == nil, "Y gained a floor it never earned")
        #expect(
            publication.merge(visibleY, scope: scopeY, now: Self.publishedAt)
                .value?.observationCount == 12
        )
        #expect(
            publication.merge(visibleY, scope: scopeY, now: Self.publishedAt)
                .value?.includesYours == false
        )
        // X's count was never known here, so nothing is incremented for X
        // either — a fabricated 13 would have been Y's number wearing X's name.
        #expect(outcome.observations.value == nil)
        #expect(publication.confirmedFloor(for: scopeX) == nil)
        // The identity is still adopted: it is a fact about this installation.
        #expect(publication.sessionContributor?.slug == "swift-otter-4417")
    }

    @Test("The Published sheet names X, the model that was published")
    func celebrationNamesX() {
        let x = Self.decode(Self.storedRunX)
        let scopeX = try! #require(x.communityScope(alias: Self.alias))
        let scopeY = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.modelY(), macProfile: Self.macNow,
                latestResult: nil, alias: Self.alias
            )
        )
        let context = CommunityPublicationContext.capture(
            runID: x.id, resultScope: scopeX, visibleScope: scopeY,
            observations: .ready(CommunityObservationSummary(observationCount: 12, isBounded: true)),
            branch: .strengthen(observationCount: 12, isAtLeast: true)
        )

        // The sheet reads `publishContext?.scope`, which is X's.
        let celebration = CommunityBenchmarkCopy.publishedCelebration(
            branchBeforePublishing: context.branch,
            scope: context.scope!,
            observationCountAfterPublishing: nil,
            alreadyPublished: false
        )
        let text = "\(celebration.headline) \(celebration.body)"
        #expect(context.scope?.scopeDescription.contains("gemma-4-12b-4bit") == true)
        // It names X's Mac too, because X was not measured on this one.
        #expect(context.scope?.scopeDescription.contains("Apple M2 Max") == true)
        #expect(!text.contains("qwen3.5-9b-4bit"))
        #expect(!text.contains("Apple M3 Pro"))
    }

    @Test("X's own floor is recorded when X's count is known")
    func xGetsItsOwnFloor() {
        let x = Self.decode(Self.storedRunX)
        let scopeX = try! #require(x.communityScope(alias: Self.alias))
        // The user is looking at X's result, so X's count is what is visible.
        let visibleX: CommunityDataState<CommunityObservationSummary> = .ready(
            CommunityObservationSummary(observationCount: 3, unit: "tok/s", isBounded: true)
        )
        let context = CommunityPublicationContext.capture(
            runID: x.id, resultScope: scopeX, visibleScope: scopeX,
            observations: visibleX, branch: .strengthen(observationCount: 3, isAtLeast: true)
        )
        var publication = CommunityPublicationState()
        let outcome = publication.recordPublication(
            receipt: Self.receipt(), receiptSaved: true,
            context: context, visibleScope: scopeX, now: Self.publishedAt
        )
        #expect(outcome.appliesToVisibleScope)
        #expect(outcome.observations.value?.observationCount == 4)
        #expect(publication.confirmedFloor(for: scopeX)?.count == 4)
        // The floor is keyed on the record's scope — M2 Max, protocol v1 —
        // so it does not defend the same model on this Mac.
        let sameModelThisMac = CommunityBenchmarkScope(
            modelAlias: "gemma-4-12b-4bit", workload: .llm,
            protocolID: "rapid-community-speed", protocolVersion: 2,
            macProfile: Self.macNow
        )
        #expect(publication.confirmedFloor(for: sameModelThisMac) == nil)
    }

    @Test("The visible result still answers for itself when it is the one published")
    func visibleResultScopesItself() {
        let x = Self.decode(Self.storedRunX)
        // Even with model Y selected, a displayed result answers for itself:
        // `observationScope` prefers the record over the selection.
        let scope = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.modelY(), macProfile: Self.macNow,
                latestResult: x, alias: Self.alias
            )
        )
        #expect(scope.modelAlias == "gemma-4-12b-4bit")
        #expect(scope.protocolVersion == 1)
        #expect(scope.macProfile.chip == "Apple M2 Max")
    }
}
