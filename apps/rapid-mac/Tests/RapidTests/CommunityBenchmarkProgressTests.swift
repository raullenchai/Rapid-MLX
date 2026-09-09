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

    @Test("Only a completion line (phase as 2nd token) counts as a step")
    func stepDetection() {
        // Completions: "<case-id> warmup …" / "<case-id> round N/M …".
        #expect(CommunityBenchmarkRunStatus.isStepLine("pp512-tg128 round 3/5 46 tok/s"))
        #expect(CommunityBenchmarkRunStatus.isStepLine("pp512-tg128 warmup"))
        #expect(CommunityBenchmarkRunStatus.isStepLine("t2i-1024-square warmup 12 s"))
        // Plan / status lines that merely mention warmup/rounds must NOT count.
        #expect(!CommunityBenchmarkRunStatus.isStepLine(
            "Benchmarking gemma-4-e4b-4bit (text_generation): 2 warmup + 10 measured rounds in total"
        ))
        #expect(!CommunityBenchmarkRunStatus.isStepLine(
            "Estimated time remaining: ~0:42 (from the warmup rate)"
        ))
        #expect(!CommunityBenchmarkRunStatus.isStepLine(
            "pp512-tg128 512 prompt tokens -> 128 output tokens"
        ))
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

    @Test("ETA divides by real inter-step time and counts down, not up")
    func etaBoundaries() {
        let start = Date(timeIntervalSince1970: 1_000)
        // One completion: no interval yet → no estimate (never "~0:00 left").
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 1, totalSteps: 12,
                runStartedAt: start, firstStepAt: start,
                lastStepAt: start, now: start
            ) == nil
        )
        // A two-step image run shows an estimate after its warmup; waiting
        // for two completions would make the ETA first appear at completion.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 1, totalSteps: 2,
                runStartedAt: start,
                firstStepAt: start.addingTimeInterval(20),
                lastStepAt: start.addingTimeInterval(20),
                now: start.addingTimeInterval(20)
            ) == "~0:20 left"
        )
        // Two completions 10 s apart → 10 s/step × 10 remaining = 100 s, at
        // the instant of the second completion.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 2, totalSteps: 12,
                runStartedAt: start,
                firstStepAt: start, lastStepAt: start.addingTimeInterval(10),
                now: start.addingTimeInterval(10)
            ) == "~1:40 left"
        )
        // 4 s later with no new step, the estimate COUNTS DOWN (100 - 4), it
        // does not inflate — the per-step average is fixed by completions.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 2, totalSteps: 12,
                runStartedAt: start,
                firstStepAt: start, lastStepAt: start.addingTimeInterval(10),
                now: start.addingTimeInterval(14)
            ) == "~1:36 left"
        )
        // Overdue (past the projection, no new step) → a finishing state, not
        // a stale "~0:00 left".
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 2, totalSteps: 12,
                runStartedAt: start,
                firstStepAt: start, lastStepAt: start.addingTimeInterval(10),
                now: start.addingTimeInterval(200)
            ) == "wrapping up…"
        )
        // All steps done → nothing left to estimate.
        #expect(
            CommunityBenchmarkRunStatus.eta(
                stepsDone: 12, totalSteps: 12,
                runStartedAt: start,
                firstStepAt: start, lastStepAt: start.addingTimeInterval(60),
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

    @Test("A failed run shows the error line, not the raw JSON document")
    func failureSummaryExtraction() {
        // The CLI's --json failure document → just the human error line.
        let doc = #"{"error":"image benchmark request failed with HTTP 500","run":{"run_id":"x"},"saved":true}"#
        #expect(
            CommunityBenchmarkCommand.failureSummary(from: doc)
                == "image benchmark request failed with HTTP 500"
        )
        // A warning/traceback preceding the JSON still resolves to the error.
        #expect(
            CommunityBenchmarkCommand.failureSummary(from: "warning: noisy\n\(doc)")
                == "image benchmark request failed with HTTP 500"
        )
        // Non-JSON failure (a crash) falls back to the raw text.
        #expect(
            CommunityBenchmarkCommand.failureSummary(from: "Segmentation fault: 11")
                == "Segmentation fault: 11"
        )
    }
}
