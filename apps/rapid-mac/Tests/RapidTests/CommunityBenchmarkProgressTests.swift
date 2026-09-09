import Foundation
import Testing
@testable import Rapid

/// The live-progress parsing/estimation that drives the run's determinate bar
/// and ETA. Pure helpers, so they are pinned here without standing up a view.
@Suite("Community Benchmark run progress")
struct CommunityBenchmarkProgressTests {
    private let tag = CommunityBenchmarkRunStatus.progressTag

    @Test("Only RS-tagged lines are treated as progress")
    func strippedProgressGating() {
        // Tagged → stripped and whitespace-collapsed.
        #expect(
            CommunityBenchmarkRunStatus.strippedProgress(
                from: "\(tag)pp512-tg128      round 1/5    46.1 tok/s"
            ) == "pp512-tg128 round 1/5 46.1 tok/s"
        )
        // Untagged (the failure document, warnings) → ignored.
        #expect(
            CommunityBenchmarkRunStatus.strippedProgress(
                from: "pp512-tg128 round 1/5 46.1 tok/s"
            ) == nil
        )
        #expect(CommunityBenchmarkRunStatus.strippedProgress(from: "\(tag)   ") == nil)
        #expect(CommunityBenchmarkRunStatus.strippedProgress(from: "Traceback…") == nil)
    }

    @Test("A step line is a completed warmup or measured round")
    func stepDetection() {
        #expect(CommunityBenchmarkRunStatus.isStepLine("pp512-tg128 round 3/5 46 tok/s"))
        #expect(CommunityBenchmarkRunStatus.isStepLine("pp512-tg128 warmup"))
        #expect(!CommunityBenchmarkRunStatus.isStepLine("Loading mlx-community/x (hf)..."))
        #expect(!CommunityBenchmarkRunStatus.isStepLine("Server ready in 4.2 s"))
    }

    @Test("Determinate-bar totals per task")
    func totals() {
        #expect(CommunityBenchmarkRunStatus.totalSteps(for: .textGeneration) == 12)
        #expect(CommunityBenchmarkRunStatus.totalSteps(for: .imageGeneration) == 2)
        // A single measured render has no useful bar.
        #expect(CommunityBenchmarkRunStatus.totalSteps(for: .videoGeneration) == nil)
    }

    @Test("ETA needs two completions and divides by intervals, not steps")
    func etaBoundaries() {
        let start = Date(timeIntervalSince1970: 1_000)
        // One completion: no interval yet → no estimate (never "~0:00 left").
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 1, totalSteps: 12, since: start, now: start
            ) == nil
        )
        // Two completions, 10 s apart → 10 s/interval × 10 remaining = 100 s.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 2,
                totalSteps: 12,
                since: start,
                now: start.addingTimeInterval(10)
            ) == "~1:40 left"
        )
        // All steps done → nothing left to estimate.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 12,
                totalSteps: 12,
                since: start,
                now: start.addingTimeInterval(60)
            ) == nil
        )
    }

    @Test("run_id is read from the CLI payload")
    func runIDDecoding() {
        let data = Data(#"{"run_id":"abc-123","measurements":[]}"#.utf8)
        #expect(CommunityBenchmarkCommand.runID(from: data) == "abc-123")
        #expect(CommunityBenchmarkCommand.runID(from: Data("not json".utf8)) == nil)
    }
}
