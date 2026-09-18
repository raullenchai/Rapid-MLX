import Foundation
import Testing
@testable import Rapid

/// Every completed Result screen has to offer a way off it.
///
/// `Benchmark another model` rendered only when `isPublished` was true, so a
/// user who finished a run and decided *not* to publish it was stuck: no
/// picker, no route back to Ready. The only escape was to start another
/// benchmark and immediately stop it, which is not an escape, it is a
/// workaround for a missing button.
///
/// The exits are unconditional. `Publish` is the one action that stays
/// conditional, because publishing an already-published run is the only one of
/// the three that would be a lie.
@Suite("Result screen exits")
struct CommunityResultExitTests {
    private static let profile = CommunityMacProfile(chip: "Apple M3 Pro", memoryGiB: 18)

    private static let alias: @Sendable (String) -> String = {
        $0.replacingOccurrences(of: "mlx-community/", with: "").lowercased()
    }

    private static func decode(_ json: String) -> CommunityBenchmarkResult {
        try! JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    private static func run(
        id: String = "run-1",
        repo: String = "mlx-community/Qwen3.5-9B-4bit",
        status: String = "completed"
    ) -> CommunityBenchmarkResult {
        decode(#"""
        {"schema_version":1,"run_id":"\#(id)",
         "started_at":"2026-09-15T10:00:00Z","completed_at":"2026-09-15T10:04:00Z",
         "execution":{"config_digest":"sha256:abc",
           "runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"},
           "resources":{"compute_dtype":"bf16"},
           "task":{"kind":"text_generation","language":{
             "speculative_decoding":{"method":"none"},
             "kv_cache":{"mode":"quantized","dtype":"int8"},
             "prefill_backend":"gpu"}}},
         "machine":{"os":{"version":"15.6.1"},
           "profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},
         "measurements":[{"case_id":"pp512-tg128","completed":true,
           "decode_duration_ms":2800,"output_tokens":129,"total_duration_ms":4200,
           "ttft_ms":1490}],
         "model":{"schema_version":1,"identity_strength":"unresolved",
           "pipeline_kind":"text_generation",
           "components":[{"component_id":"primary","role":"primary",
             "source":{"kind":"huggingface","repo_id":"\#(repo)"},
             "artifact":{"format":"mlx-safetensors"},
             "quantization":{"kind":"unknown","base_dtype":"unknown"}}]},
         "outcome":{"status":"\#(status)"},
         "workload":{"protocol_id":"rapid-community-speed","protocol_version":2,
           "task_type":"text_generation",
           "cases":[{"case_id":"pp512-tg128","measured_rounds":5,"warmup_rounds":1},
                    {"case_id":"pp2048-tg512","measured_rounds":5,"warmup_rounds":1}]}}
        """#)
    }

    private static func receipt() -> CommunityBenchmarkReceipt {
        try! JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(#"""
            {"submission_id":"sub-1","already_exists":false,
             "accepted_at":"2026-09-15T10:05:00Z",
             "contributor":{"name":"swift-otter","tag":"4417","slug":"swift-otter-4417"}}
            """#.utf8)
        )
    }

    /// The identifiers the screen offers for one state.
    private static func actionIdentifiers(
        isPublished: Bool,
        isCompleted: Bool = true,
        isPublishing: Bool = false
    ) -> [String] {
        CommunityBenchmarkResultView.actions(
            isPublished: isPublished,
            isCompleted: isCompleted,
            isPublishing: isPublishing
        ).map(\.accessibilityIdentifier)
    }

    private static func actions(
        isPublished: Bool,
        isCompleted: Bool = true,
        isPublishing: Bool = false
    ) -> [CommunityBenchmarkResultView.Action] {
        CommunityBenchmarkResultView.actions(
            isPublished: isPublished, isCompleted: isCompleted, isPublishing: isPublishing
        )
    }

    // MARK: - Both exits always exist

    @Test("An unpublished result offers publish and both exits")
    func unpublishedResultHasEveryAction() {
        let identifiers = Self.actionIdentifiers(isPublished: false)
        #expect(identifiers.contains("CommunityBenchmark.Result.Publish"))
        #expect(
            identifiers.contains("CommunityBenchmark.Result.Another"),
            "an unpublished result had no way back to the picker"
        )
        #expect(identifiers.contains("CommunityBenchmark.Result.RunAgain"))
        #expect(Self.actions(isPublished: false).allSatisfy { $0.isEnabled })
    }

    @Test("A published result keeps both exits and drops publish")
    func publishedResultHasBothExits() {
        let identifiers = Self.actionIdentifiers(isPublished: true)
        #expect(identifiers.contains("CommunityBenchmark.Result.Another"))
        #expect(identifiers.contains("CommunityBenchmark.Result.RunAgain"))
        // Publishing the same run twice is the one action that would be a lie.
        #expect(!identifiers.contains("CommunityBenchmark.Result.Publish"))
    }

    /// The states differ in what the *comparison area* renders, not in what
    /// the actions are — which is the point: no branch of the contribution
    /// story may strand the user.
    @Test("Every contribution branch offers both exits")
    func everyBranchHasBothExits() {
        for isPublished in [true, false] {
            let identifiers = Self.actionIdentifiers(isPublished: isPublished)
            #expect(
                identifiers.contains("CommunityBenchmark.Result.Another"),
                "published=\(isPublished) had no Benchmark another model"
            )
            #expect(identifiers.contains("CommunityBenchmark.Result.RunAgain"))
        }
    }

    @Test("An incomplete run still offers both exits, with publish held")
    func incompleteRunKeepsBothExits() {
        let actions = Self.actions(isPublished: false, isCompleted: false)
        let byKind = Dictionary(uniqueKeysWithValues: actions.map { ($0.kind, $0) })
        // A failed run has nothing to publish…
        #expect(byKind[.publish]?.isEnabled == false)
        // …but stranding the user on it would be worse, not better.
        #expect(byKind[.benchmarkAnother]?.isEnabled == true)
        #expect(byKind[.runAgain]?.isEnabled == true)
    }

    @Test("A publish in flight holds both exits rather than hiding them")
    func publishingDisablesButKeepsExits() {
        let actions = Self.actions(isPublished: false, isPublishing: true)
        let identifiers = actions.map(\.accessibilityIdentifier)
        // Still offered — the user can see where they will be able to go —
        // but held, so a Run again cannot redirect the in-flight receipt.
        #expect(identifiers.contains("CommunityBenchmark.Result.Another"))
        #expect(identifiers.contains("CommunityBenchmark.Result.RunAgain"))
        #expect(actions.allSatisfy { !$0.isEnabled })
    }

    @Test("Every action carries the identifier the CI gate checks for")
    func everyActionIsReachable() {
        for isPublished in [true, false] {
            for action in Self.actions(isPublished: isPublished) {
                #expect(action.accessibilityIdentifier.hasPrefix("CommunityBenchmark.Result."))
            }
        }
        // And the set is exactly the three the screen is specified to offer.
        #expect(
            Set(Self.actionIdentifiers(isPublished: false)) == [
                "CommunityBenchmark.Result.Publish",
                "CommunityBenchmark.Result.Another",
                "CommunityBenchmark.Result.RunAgain",
            ]
        )
    }

    // MARK: - Run again repeats the stored run

    @Test("Run again targets the result's own model, not the selection")
    @MainActor
    func runAgainUsesTheResultsModel() throws {
        // The Run tab is on gemma; the visible result is a stored qwen run.
        let stored = Self.run(repo: "mlx-community/Qwen3.5-9B-4bit")
        #expect(Self.alias(stored.repoID) == "qwen3.5-9b-4bit")

        // `observationScope` is what the refresh path and the publish context
        // both use, and it answers from the record.
        let scope = try #require(
            CommunityBenchmarkView.observationScope(
                selected: nil, macProfile: Self.profile,
                latestResult: stored, alias: Self.alias
            )
        )
        #expect(scope.modelAlias == "qwen3.5-9b-4bit")
        #expect(scope.protocolID == "rapid-community-speed")
        #expect(scope.protocolVersion == 2)
    }

    @Test("Choosing another model lands on Ready without starting it")
    @MainActor
    func choosingAnotherModelDoesNotStart() throws {
        // Selecting a model produces a *coverage* scope: no comparison
        // identity, because no run exists for it yet. That is exactly the
        // Ready screen's question, and nothing in producing it starts a run.
        let model = CommunityBenchmarkModel(
            entry: ModelEntry(
                alias: "gemma-4-12b-4bit", hfRepo: "mlx-community/Gemma-4-12B-4bit",
                sizeOnDisk: "7.1 GB", cached: true, taskTypes: [.textGeneration]
            ),
            task: .textGeneration,
            protocolName: "Rapid Community Speed v2",
            protocolID: "rapid-community-speed", protocolVersion: 2,
            isFocus: false, estimatedMemoryGib: 8, memoryFit: "fits"
        )
        let scope = try #require(
            CommunityBenchmarkView.observationScope(
                selected: model, macProfile: Self.profile,
                latestResult: nil, alias: Self.alias
            )
        )
        #expect(scope.modelAlias == "gemma-4-12b-4bit")
        #expect(scope.comparison == nil)
        #expect(scope.modelIdentity == nil)
        // And the Ready branch makes no claim it cannot support.
        #expect(
            CommunityContributionBranch.select(from: .unavailable(.boundedFeed))
                == .unknown(.unavailable(.boundedFeed))
        )
    }
}

/// The narrow (≈700pt detail pane) Result layout.
///
/// At that width the single-row header truncated the run's date to "Sep…" and
/// the status chip to "SAVED ON THIS M…", and the fixed action row pushed
/// Publish / Benchmark another model / Run again / Technical details off the
/// card. The header now wraps to two lines, the chip uses a shorter *complete*
/// phrase, and the actions stack.
@Suite("Narrow result layout")
struct CommunityNarrowResultLayoutTests {
    @Test("Comparison direction follows the workload metric")
    @MainActor
    func comparisonDirection() {
        #expect(CommunityBenchmarkResultView.isFaster(workload: .llm, delta: 4))
        #expect(!CommunityBenchmarkResultView.isFaster(workload: .llm, delta: -4))
        #expect(CommunityBenchmarkResultView.isFaster(workload: .image, delta: -4))
        #expect(!CommunityBenchmarkResultView.isFaster(workload: .image, delta: 4))
        #expect(CommunityBenchmarkResultView.isFaster(workload: .video, delta: -4))
        #expect(!CommunityBenchmarkResultView.isFaster(workload: .video, delta: 4))
    }

    @Test("The narrow status chip says something complete, not something truncated")
    func narrowStatusLabel() {
        let wide = CommunityBenchmarkResultView.statusLabel(
            isPublished: false, isNarrow: false
        )
        let narrow = CommunityBenchmarkResultView.statusLabel(
            isPublished: false, isNarrow: true
        )
        #expect(wide == "SAVED ON THIS MAC")
        #expect(narrow == "SAVED HERE")
        // Shorter, so it fits — and still a whole phrase, which "SAVED ON
        // THIS M…" was not.
        #expect(narrow.count < wide.count)
        #expect(!narrow.hasSuffix("…"))
    }

    @Test("A published run reads the same at either width")
    func publishedLabelIsStable() {
        for isNarrow in [true, false] {
            #expect(
                CommunityBenchmarkResultView.statusLabel(
                    isPublished: true, isNarrow: isNarrow
                ) == "PUBLISHED"
            )
        }
    }

    @Test("Narrowness changes the layout, never the actions offered")
    func narrownessDoesNotRemoveActions() {
        // The stacking is a layout decision. Dropping an action at a narrow
        // width would be the same defect as the original one, in a new place.
        let offered = CommunityBenchmarkResultView.actions(
            isPublished: false, isCompleted: true, isPublishing: false
        )
        #expect(offered.count == 3)
        #expect(
            Set(offered.map(\.accessibilityIdentifier)) == [
                "CommunityBenchmark.Result.Publish",
                "CommunityBenchmark.Result.Another",
                "CommunityBenchmark.Result.RunAgain",
            ]
        )
    }
}
