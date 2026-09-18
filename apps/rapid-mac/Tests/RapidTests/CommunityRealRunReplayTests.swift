import Foundation
import Testing
@testable import Rapid

/// The stderr of a **real** packaged benchmark, replayed through the reducer.
///
/// Captured verbatim from a **cleanly built** bundle
/// (`FORCE_SIDECAR_REBUILD=1 bash scripts/build.sh`, nothing hand-edited):
/// `Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx
///  benchmark run lfm2.5-1b-4bit --json --progress`
/// on an Apple M-series Mac, record separators and column padding included.
/// Synthetic fixtures are written to match what the code expects; this one was
/// written by the CLI, so it is the only fixture that can catch the shipped
/// runner drifting away from the parser.
@Suite("Real packaged run replay")
struct CommunityRealRunReplayTests {
    /// Every line the CLI actually emitted, in order. `\u{1e}` is
    /// `PROGRESS_TAG` from `rapid_mlx/community_bench/cli.py`.
    private static let capturedStderr: [String] = [
        "\u{1e}{\"event\":\"plan\",\"task_type\":\"text_generation\",\"protocol_id\":\"rapid-community-speed\",\"protocol_version\":2,\"cases\":[{\"case_id\":\"pp512-tg128\",\"warmup_rounds\":1,\"measured_rounds\":5},{\"case_id\":\"pp2048-tg512\",\"warmup_rounds\":1,\"measured_rounds\":5}],\"total_passes\":12}",
        "\u{1e}Benchmarking lfm2.5-1b-4bit (text_generation): 2 cases, 2 warmup + 10 measured rounds in total",
        "\u{1e}  pp512-tg128      512 prompt tokens -> 128 output tokens   (1 warmup + 5 measured)",
        "\u{1e}  pp2048-tg512     2048 prompt tokens -> 512 output tokens   (1 warmup + 5 measured)",
        "\u{1e}Loading mlx-community/LFM2.5-1.2B-Instruct-4bit (from the local Hugging Face cache)...",
        "\u{1e}Model loaded in 1 s",
        "\u{1e}pp512-tg128      warmup",
        "\u{1e}Estimated time remaining: ~24 s (from the warmup rate)",
        "\u{1e}pp512-tg128      round 1/5   260.8 tok/s",
        "\u{1e}pp512-tg128      round 2/5   261.2 tok/s",
        "\u{1e}pp512-tg128      round 3/5   260.3 tok/s",
        "\u{1e}pp512-tg128      round 4/5   261.6 tok/s",
        "\u{1e}pp512-tg128      round 5/5   262.6 tok/s",
        "\u{1e}pp2048-tg512     warmup",
        "\u{1e}pp2048-tg512     round 1/5   239.5 tok/s",
        "\u{1e}pp2048-tg512     round 2/5   239.5 tok/s",
        "\u{1e}pp2048-tg512     round 3/5   239.5 tok/s",
        "\u{1e}pp2048-tg512     round 4/5   239.5 tok/s",
        "\u{1e}pp2048-tg512     round 5/5   239.7 tok/s",
        "\u{1e}Saving result to this Mac...",
        "\u{1e}Saved to this Mac",
    ]

    @Test("A real run drives the whole designed sequence")
    func realRunSequence() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        // Two seconds per line — the real gaps, near enough for the ETA rule.
        var now = Date(timeIntervalSince1970: 1_800_000_000)
        var states: [CommunityRunProgress] = []
        for line in Self.capturedStderr {
            now = now.addingTimeInterval(2)
            guard let state = reducer.apply(line: line, at: now) else { continue }
            states.append(state)
        }

        // The denominator is the protocol's, declared by the run itself in
        // its first line, not a constant in this client.
        #expect(states.allSatisfy { $0.totalPasses == 12 })
        #expect(
            Self.capturedStderr.first?.contains(#""event":"plan""#) == true,
            "the run no longer declares its plan"
        )

        // Monotonic 0…12, advanced only by real completed-pass lines. The plan
        // header says "2 warmup + 10 measured rounds in total" and the
        // estimate line mentions "warmup" — neither may advance the bar.
        let counts = states.map(\.passesComplete)
        #expect(counts == counts.sorted())
        #expect(Set(counts).sorted() == Array(0...12))

        // The stage sequence, in order, derived from the stream's own case ids.
        var stages: [CommunityRunStage] = []
        for state in states where stages.last != state.stage { stages.append(state.stage) }
        #expect(stages == [.gettingReady, .warmingUp, .shortReplies, .longReplies, .saving])

        // No ETA before two completed passes; a real one after.
        for state in states where state.passesComplete < 2 {
            #expect(state.timeLeft == nil)
        }
        #expect(states.contains { $0.passesComplete == 2 && $0.timeLeft != nil })

        // Live measurements, with the pass they came from. A warmup carries no
        // rate, so pass 7 still shows pass 6's number rather than inventing one.
        let atSeven = states.first { $0.passesComplete == 7 }
        #expect(atSeven?.latestMeasurement?.passNumber == 6)
        #expect(states.last?.latestMeasurement?.passNumber == 12)
        #expect(states.last?.latestMeasurement?.value.hasSuffix("tok/s") == true)

        // And the run ends on Saving, not on a jump straight to Result.
        #expect(states.last?.stage == .saving)
        #expect(states.last?.isSaving == true)
        #expect(states.last?.passesComplete == 12)
        #expect(states.last?.fraction == 1)
    }

    @Test("The declared plan sizes the bar, and the client assumes nothing")
    func planEventDrivesTheDenominator() {
        // A reducer told the run is a single-pass video protocol still ends up
        // with 12 passes, because the run says so.
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .videoGeneration)
        )
        #expect(reducer.snapshot.totalPasses == nil)
        _ = reducer.apply(
            line: Self.capturedStderr[0], at: Date(timeIntervalSince1970: 1_800_000_000)
        )
        #expect(reducer.snapshot.totalPasses == 12)
        #expect(reducer.currentPlan.cases.map(\.id) == ["pp512-tg128", "pp2048-tg512"])
    }

    @Test("The plan header and the estimate line are status, not progress")
    func announcementsDoNotAdvanceTheBar() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        // The four lines that mention "warmup", "rounds" or a number but
        // report no completed work.
        for line in Self.capturedStderr.prefix(5) {
            _ = reducer.apply(line: line, at: now)
        }
        #expect(reducer.snapshot.passesComplete == 0)
        #expect(reducer.snapshot.fraction == nil)
        #expect(reducer.snapshot.stage == .gettingReady)
    }
}
