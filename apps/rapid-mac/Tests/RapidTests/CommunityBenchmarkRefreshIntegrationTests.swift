import Foundation
import Testing
@testable import Rapid

/// What the production refresh path actually asks the directory for.
///
/// These drive `CommunityBenchmarkView.observationScope(selected:macProfile:latestResult:)`
/// — the same function the view's `activeObservationScope` calls — and hand its
/// output to a directory that records the scope it received. So the assertion
/// is on the real query, not on a re-implementation of the rule.
///
/// The defect being pinned: while a completed run was on screen the view kept
/// querying the generic Ready/coverage scope, which carries no comparison
/// identity. The adapter then aggregated across every execution variant and
/// withheld the median, so the Result screen could never show a comparison no
/// matter how much data the server had.
@MainActor
@Suite("Observation refresh integration")
struct CommunityBenchmarkRefreshIntegrationTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    /// Records every scope the view's refresh path asks for.
    private actor ScopeRecorder {
        private(set) var scopes: [CommunityBenchmarkScope] = []
        func record(_ scope: CommunityBenchmarkScope) { scopes.append(scope) }
        var last: CommunityBenchmarkScope? { scopes.last }
        var count: Int { scopes.count }
    }

    private struct RecordingDirectory: CommunityBenchmarkDirectory {
        let recorder: ScopeRecorder
        var answer: CommunityObservationSummary = CommunityObservationSummary(observationCount: 5)

        func observations(
            for scope: CommunityBenchmarkScope,
            viewerSlug: String?
        ) async -> CommunityDataState<CommunityObservationSummary> {
            await recorder.record(scope)
            return .ready(answer)
        }
        func table(
            macProfile: CommunityMacProfile, workload: CommunityWorkload,
            metric: CommunityMetric, viewerSlug: String?
        ) async -> CommunityDataState<[CommunityObservationRow]> { .unavailable(.boundedFeed) }
        func coverageGaps(
            macProfile: CommunityMacProfile
        ) async -> CommunityDataState<[CommunityCoverageGap]> { .unavailable(.boundedFeed) }
        func pulse() async -> CommunityDataState<CommunityPulse> { .unavailable(.boundedFeed) }
    }

    private static func model(_ alias: String) -> CommunityBenchmarkModel {
        CommunityBenchmarkModel(
            entry: ModelEntry(
                alias: alias, hfRepo: "mlx-community/\(alias)", sizeOnDisk: "6.5 GB",
                cached: true, taskTypes: [.textGeneration]
            ),
            task: .textGeneration,
            protocolName: "Rapid Community Speed v2",
            protocolID: "rapid-community-speed",
            protocolVersion: 2,
            isFocus: true,
            estimatedMemoryGib: 6,
            memoryFit: "fits"
        )
    }

    /// A `benchmark results --json` record in the shape `run_builder` actually
    /// writes: the full `model-identity` block (subfolder, resolved revision
    /// and quantization facts, not just a repo id), the workload's registered
    /// `protocol_id` / `protocol_version`, and the execution projection the
    /// worker groups by.
    private static let completedRun = #"""
    {"schema_version":1,"run_id":"run-1",
     "started_at":"2026-09-06T04:30:00Z","completed_at":"2026-09-06T04:37:42Z",
     "collector":{"name":"rapid-mlx-community-bench","version":"0.13.4"},
     "execution":{"config_digest":"sha256:abc",
       "runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"},
       "resources":{"max_concurrency":1,"compute_dtype":"bf16"},
       "task":{"kind":"text_generation","language":{
         "speculative_decoding":{"method":"none"},
         "kv_cache":{"mode":"quantized","dtype":"int8"},
         "prefill_backend":"gpu"}}},
     "machine":{"os":{"version":"15.6.1"},
       "profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},
     "measurements":[
       {"case_id":"pp512-tg128","completed":true,"decode_duration_ms":5000,
        "output_tokens":129,"peak_active_memory_mib":6875,"round_index":1,
        "total_duration_ms":6500,"ttft_ms":1490}],
     "model":{"schema_version":1,"identity_strength":"unresolved",
       "pipeline_kind":"text_generation",
       "components":[{"component_id":"primary","role":"primary",
         "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit"},
         "artifact":{"format":"mlx-safetensors"},
         "quantization":{"kind":"unknown","base_dtype":"unknown"}}]},
     "outcome":{"status":"completed"},
     "workload":{"protocol_id":"rapid-community-speed","protocol_version":2,
       "protocol_strength":"registered","task_type":"text_generation","concurrency":1,
       "cases":[
        {"case_id":"pp512-tg128","measured_rounds":5,"target_output_tokens":128,
         "target_prompt_tokens":512,"warmup_rounds":1},
        {"case_id":"pp2048-tg512","measured_rounds":5,"target_output_tokens":512,
         "target_prompt_tokens":2048,"warmup_rounds":1}]}}
    """#

    /// The catalogue's repo → alias mapping, the one thing a scope needs that
    /// the record itself does not carry.
    private static let alias: (String) -> String = {
        $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased()
    }

    private static func decode(_ json: String) -> CommunityBenchmarkResult {
        try! JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    // MARK: - Scope sent to the directory

    @Test("A completed result queries with case, metric, protocol and execution")
    func completedResultSendsExactIdentity() async {
        let recorder = ScopeRecorder()
        let directory = RecordingDirectory(recorder: recorder)
        let scope = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.model("qwen3.5-9b-4bit"),
                macProfile: Self.profile,
                latestResult: Self.decode(Self.completedRun),
                alias: Self.alias
            )
        )
        _ = await directory.observations(for: scope)

        let sent = try! #require(await recorder.last)
        // Protocol identity.
        #expect(sent.protocolID == "rapid-community-speed")
        #expect(sent.protocolVersion == 2)
        // Comparison identity — the whole point.
        let comparison = try! #require(sent.comparison)
        #expect(comparison.caseID == "pp512-tg128")
        #expect(comparison.metricName == "decode_tps")
        #expect(comparison.execution.rapidMLX == "0.13.4")
        #expect(comparison.execution.computeDType == "bf16")
        #expect(comparison.execution.speculativeDecodingMethod == "none")
        #expect(comparison.execution.kvCacheMode == "quantized")
        #expect(comparison.execution.kvCacheDType == "int8")
        #expect(comparison.execution.prefillBackend == "gpu")
    }

    @Test("Ready — with no completed run — queries the coverage scope")
    func readyQueriesCoverageScope() async {
        let recorder = ScopeRecorder()
        let directory = RecordingDirectory(recorder: recorder)
        let scope = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.model("qwen3.5-9b-4bit"),
                macProfile: Self.profile,
                latestResult: nil,
                alias: Self.alias
            )
        )
        _ = await directory.observations(for: scope)
        // No run exists, so there is nothing to compare; asking for a
        // comparison identity here would invent one.
        #expect(await recorder.last?.comparison == nil)
    }

    @Test("A newly completed run switches the query from coverage to comparison")
    func completingARunSwitchesTheQuery() async {
        let recorder = ScopeRecorder()
        let directory = RecordingDirectory(recorder: recorder)
        let model = Self.model("qwen3.5-9b-4bit")

        // Ready.
        _ = await directory.observations(
            for: CommunityBenchmarkView.observationScope(
                selected: model, macProfile: Self.profile, latestResult: nil, alias: Self.alias
            )!
        )
        // The run finishes and becomes the displayed result.
        _ = await directory.observations(
            for: CommunityBenchmarkView.observationScope(
                selected: model, macProfile: Self.profile,
                latestResult: Self.decode(Self.completedRun), alias: Self.alias
            )!
        )

        let scopes = await recorder.scopes
        #expect(scopes.count == 2)
        #expect(scopes[0].comparison == nil)
        #expect(scopes[1].comparison?.caseID == "pp512-tg128")
        // Same model and machine throughout; only the question changed.
        #expect(scopes[0].modelAlias == scopes[1].modelAlias)
        #expect(scopes[0].macProfile == scopes[1].macProfile)
    }

    @Test("Changing model re-queries under the new model's identity")
    func modelChangeRequeries() async {
        let recorder = ScopeRecorder()
        let directory = RecordingDirectory(recorder: recorder)
        let result = Self.decode(Self.completedRun)

        _ = await directory.observations(
            for: CommunityBenchmarkView.observationScope(
                selected: Self.model("qwen3.5-9b-4bit"),
                macProfile: Self.profile, latestResult: result, alias: Self.alias
            )!
        )
        // Choosing another model clears the displayed result in the view, so
        // the next query is a coverage question about the new model.
        _ = await directory.observations(
            for: CommunityBenchmarkView.observationScope(
                selected: Self.model("gemma-4-12b-4bit"),
                macProfile: Self.profile, latestResult: nil, alias: Self.alias
            )!
        )

        let scopes = await recorder.scopes
        #expect(scopes.map(\.modelAlias) == ["qwen3.5-9b-4bit", "gemma-4-12b-4bit"])
        #expect(scopes[1].comparison == nil)
    }

    @Test("An image run sends the image protocol and its own metric")
    func imageRunIdentity() async {
        // `t2i-1024-square` is the real case id in
        // `proto/community-benchmark/v1/protocols/rapid-image-speed-v1.json`.
        let imageRun = #"""
        {"schema_version":1,"run_id":"run-2",
         "started_at":"2026-09-06T04:30:00Z","completed_at":"2026-09-06T04:37:42Z",
         "execution":{"config_digest":"sha256:def",
           "runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"},
           "resources":{"compute_dtype":"fp16"},
           "task":{"kind":"image_generation"}},
         "machine":{"os":{"version":"15.6.1"},
           "profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},
         "measurements":[{"case_id":"t2i-1024-square","completed":true,
           "peak_active_memory_mib":6246,"round_index":1,"total_duration_ms":4600}],
         "model":{"schema_version":1,"identity_strength":"unresolved",
           "pipeline_kind":"image_generation",
           "components":[{"component_id":"primary","role":"primary",
             "source":{"kind":"huggingface","repo_id":"mlx-community/z-image-turbo"},
             "artifact":{"format":"mlx-safetensors"},
             "quantization":{"kind":"unknown","base_dtype":"unknown"}}]},
         "outcome":{"status":"completed"},
         "workload":{"protocol_id":"rapid-image-speed","protocol_version":1,
           "protocol_strength":"registered","task_type":"image_generation","concurrency":1,
           "cases":[{"case_id":"t2i-1024-square","measured_rounds":1,"warmup_rounds":1}]}}
        """#
        let imageModel = CommunityBenchmarkModel(
            entry: ModelEntry(
                alias: "z-image-turbo", hfRepo: "mlx-community/z-image-turbo",
                sizeOnDisk: nil, cached: false,
                taskTypes: [.imageGeneration], operationModes: [.textToImage]
            ),
            task: .imageGeneration,
            protocolName: "Rapid Image Speed v1",
            protocolID: "rapid-image-speed",
            protocolVersion: 1,
            isFocus: true, estimatedMemoryGib: 4, memoryFit: "fits"
        )
        let scope = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: imageModel, macProfile: Self.profile,
                latestResult: Self.decode(imageRun), alias: Self.alias
            )
        )
        #expect(scope.protocolID == "rapid-image-speed")
        #expect(scope.protocolVersion == 1)
        #expect(scope.comparison?.caseID == "t2i-1024-square")
        // Image results are seconds, not tokens per second.
        #expect(scope.comparison?.metricName == "total_seconds")
        #expect(scope.comparison?.execution.computeDType == "fp16")
        // No language block on an image run.
        #expect(scope.comparison?.execution.kvCacheMode == nil)
    }

    // MARK: - End to end against the real adapter

    @Test("The exact identity reaches the adapter and selects the matching cell")
    func endToEndAgainstTheAdapter() async {
        // Two cells differing only by dtype. The run under test is bf16.
        let feed = #"""
        {"schema_version":1,"summary":[
          {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"fp16",
             "speculative_decoding":{"method":"none"},
             "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":31.0,"best":33.0},
           "samples":4,"contributors":[],"latest_at":"2026-09-06T04:00:00Z"},
          {"task_type":"text_generation","model":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"},
           "machine":{"chip":"Apple M3 Pro","memory_gib":18},
           "execution":{"rapid_mlx":"0.13.4","compute_dtype":"bf16",
             "speculative_decoding":{"method":"none"},
             "kv_cache":{"mode":"quantized","dtype":"int8"},"prefill_backend":"gpu"},
           "protocol":{"id":"rapid-community-speed","version":2},"case_id":"pp512-tg128",
           "metric":{"name":"decode_tps","better":"higher","median":25.9,"best":26.9},
           "samples":5,"contributors":[],"latest_at":"2026-09-06T02:00:00Z"}
        ],"runs":[]}
        """#
        let directory = CommunityBenchmarkAPIDirectory(
            transport: { request in
                (
                    Data(feed.utf8),
                    HTTPURLResponse(
                        url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil
                    )!
                )
            },
            aliasForRepoID: { $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased() }
        )
        let scope = try! #require(
            CommunityBenchmarkView.observationScope(
                selected: Self.model("qwen3.5-9b-4bit"),
                macProfile: Self.profile,
                latestResult: Self.decode(Self.completedRun),
                alias: Self.alias
            )
        )
        let state = await directory.observations(for: scope)
        // The bf16 cell, because that is how this run was executed. The
        // coverage scope would have returned 9 with no median.
        #expect(state.value?.median == 25.9)
        #expect(state.value?.observationCount == 5)
    }
}
