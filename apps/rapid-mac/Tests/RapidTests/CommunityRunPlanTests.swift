import Foundation
import Testing
@testable import Rapid

/// The denominator belongs to the protocol, not to the client.
///
/// `CommunityRunPlan` used to be `.text(caseCount: 2, warmupRounds: 1,
/// measuredRounds: 5)` — the shape of `rapid-community-speed` v2, written into
/// Desktop. A v3 with a third case, or with different round counts, would have
/// kept rendering "8 of 12 passes" while the run counted to something else,
/// and the stepper would have promised stages the run never enters.
///
/// The runner now declares its plan in a structured event before the first
/// pass (`_announce_plan` in `rapid_mlx/community_bench/local_runner.py`), and
/// the reducer adopts it.
@Suite("Protocol-declared run plan")
struct CommunityRunPlanTests {
    /// The exact JSON `_tagged_event_to_stderr` writes, record separator and
    /// all, for a protocol that is deliberately **not** the shipped one.
    private static func planEvent(
        taskType: String = "text_generation",
        cases: [(String, Int, Int)]
    ) -> String {
        let body = cases.map {
            #"{"case_id":"\#($0.0)","warmup_rounds":\#($0.1),"measured_rounds":\#($0.2)}"#
        }.joined(separator: ",")
        let total = cases.reduce(0) { $0 + $1.1 + $1.2 }
        return "\u{1e}" + #"{"event":"plan","task_type":"\#(taskType)","#
            + #""protocol_id":"rapid-community-speed","protocol_version":3,"#
            + #""cases":[\#(body)],"total_passes":\#(total)}"#
    }

    private static func pass(_ caseID: String, _ text: String) -> String {
        "\u{1e}\(caseID)      \(text)"
    }

    // MARK: - A protocol whose total is not 12

    @Test("A three-case protocol reports its own total, not 12")
    func threeCaseProtocol() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        // Before the event, the assumed shipped protocol.
        #expect(reducer.snapshot.totalPasses == 12)

        let state = reducer.apply(
            line: Self.planEvent(cases: [
                ("pp512-tg128", 1, 4),
                ("pp2048-tg512", 1, 4),
                ("pp8192-tg1024", 1, 4),
            ]),
            at: now
        )
        // 3 x (1 + 3) = 12 would coincide with the old constant, so this
        // protocol uses four measured rounds: 3 x (1 + 4) = 15.
        #expect(state?.totalPasses == 15)
    }

    @Test("A two-case protocol with nine measured rounds totals 20")
    func twentyPassProtocol() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        var now = Date(timeIntervalSince1970: 1_800_000_000)
        _ = reducer.apply(
            line: Self.planEvent(cases: [
                ("pp512-tg128", 1, 9),
                ("pp2048-tg512", 1, 9),
            ]),
            at: now
        )
        #expect(reducer.snapshot.totalPasses == 20)

        // Drive all twenty passes and check the caption never says 12.
        var captions: [String] = []
        for (index, declared) in [("pp512-tg128", 9), ("pp2048-tg512", 9)] {
            now = now.addingTimeInterval(2)
            _ = reducer.apply(line: Self.pass(index, "warmup"), at: now)
            captions.append(reducer.snapshot.passCaption ?? "")
            for round in 1...declared {
                now = now.addingTimeInterval(2)
                _ = reducer.apply(
                    line: Self.pass(index, "round \(round)/\(declared)   40.0 tok/s"),
                    at: now
                )
                captions.append(reducer.snapshot.passCaption ?? "")
            }
        }
        #expect(reducer.snapshot.passesComplete == 20)
        #expect(reducer.snapshot.fraction == 1)
        #expect(captions.last == "20 of 20 passes complete")
        #expect(!captions.contains { $0.contains("of 12") })
    }

    @Test("A single-case text protocol narrates no short/long split")
    func singleCaseProtocol() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        var now = Date(timeIntervalSince1970: 1_800_000_000)
        _ = reducer.apply(
            line: Self.planEvent(cases: [("pp512-tg128", 1, 4)]), at: now
        )
        #expect(reducer.snapshot.totalPasses == 5)
        // "Long replies" would be a stage this run never enters.
        let stages = reducer.currentPlan.stages
        #expect(stages == [.gettingReady, .warmingUp, .measuring, .saving])
        #expect(!stages.contains(.shortReplies))
        #expect(!stages.contains(.longReplies))

        now = now.addingTimeInterval(2)
        _ = reducer.apply(line: Self.pass("pp512-tg128", "warmup"), at: now)
        #expect(reducer.snapshot.stage == .warmingUp)
        now = now.addingTimeInterval(2)
        _ = reducer.apply(
            line: Self.pass("pp512-tg128", "round 1/4   40.0 tok/s"), at: now
        )
        #expect(reducer.snapshot.stage == .measuring)
    }

    @Test("A protocol with no warmup rounds omits the warming-up stage")
    func noWarmupProtocol() {
        let plan = CommunityRunPlan(
            workload: .llm,
            cases: [
                .init(id: "a", warmupRounds: 0, measuredRounds: 4),
                .init(id: "b", warmupRounds: 0, measuredRounds: 4),
            ]
        )
        #expect(plan.totalPasses == 8)
        #expect(plan.stages == [.gettingReady, .shortReplies, .longReplies, .saving])
    }

    // MARK: - Adoption rules

    @Test("The declared plan supersedes the assumed one exactly once")
    func adoptedOnce() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        _ = reducer.apply(line: Self.planEvent(cases: [("a", 1, 9)]), at: now)
        #expect(reducer.snapshot.totalPasses == 10)
        // A second plan event would move a denominator the user is watching.
        _ = reducer.apply(line: Self.planEvent(cases: [("a", 1, 1)]), at: now)
        #expect(reducer.snapshot.totalPasses == 10)
    }

    @Test("A CLI that sends no plan event still works on the assumed shape")
    func noPlanEvent() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        var now = Date(timeIntervalSince1970: 1_800_000_000)
        for round in 1...5 {
            now = now.addingTimeInterval(2)
            _ = reducer.apply(
                line: Self.pass("pp512-tg128", "round \(round)/5   40.0 tok/s"), at: now
            )
        }
        #expect(reducer.snapshot.totalPasses == 12)
        #expect(reducer.snapshot.passesComplete == 5)
    }

    @Test("Malformed or foreign JSON on the progress stream is ignored")
    func malformedEvents() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        for line in [
            "\u{1e}{not json",
            "\u{1e}{\"event\":\"something-else\",\"cases\":[]}",
            "\u{1e}{\"event\":\"plan\",\"cases\":[]}",
            "\u{1e}{\"event\":\"plan\"}",
        ] {
            _ = reducer.apply(line: line, at: now)
        }
        #expect(reducer.snapshot.totalPasses == 12)
        #expect(reducer.snapshot.passesComplete == 0)
    }

    @Test("An image plan keeps its own shape")
    func imagePlan() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .imageGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        _ = reducer.apply(
            line: Self.planEvent(
                taskType: "image_generation", cases: [("t2i-1024-square", 1, 1)]
            ),
            at: now
        )
        #expect(reducer.snapshot.totalPasses == 2)
        #expect(reducer.currentPlan.stages == [.gettingReady, .warmingUp, .rendering, .saving])
    }

    @Test("A single-pass video plan still refuses a denominator")
    func videoPlan() {
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .videoGeneration)
        )
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        _ = reducer.apply(
            line: Self.planEvent(
                taskType: "video_generation", cases: [("t2v-480p-81f", 0, 1)]
            ),
            at: now
        )
        // One pass: a bar that jumps 0 → 100% says less than a spinner.
        #expect(reducer.snapshot.totalPasses == nil)
        #expect(reducer.snapshot.fraction == nil)
    }
}
