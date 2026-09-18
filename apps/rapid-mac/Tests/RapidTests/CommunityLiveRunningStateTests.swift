import Foundation
import Testing
@testable import Rapid

/// The Running screen must be live *during* the run, not correct afterwards.
///
/// A real packaged run of `bonsai-1.7b-2bit` sat on "Getting ready" for 34
/// seconds — no mascot, no pass count, no stage change, no ETA — and then
/// jumped straight to Result. Every existing test passed, because they all
/// collected states into an actor and asserted once
/// `CommunityBenchmarkCommand.run` had returned. That proves the states were
/// *produced*; it says nothing about *when*, and "all at once, after the child
/// exited" satisfies it exactly.
///
/// The root cause was `FileHandle.read(upToCount:)`, which is not a streaming
/// read: it loops internally until it has the requested count or hits EOF, so
/// a 64 KB request against a slow trickle of progress lines returned nothing
/// until the process ended.
///
/// So these tests hold the child open and assert on the main actor *before*
/// letting it finish. A post-exit sleep can never be the proof.
@Suite("Live running state", .serialized)
struct CommunityLiveRunningStateTests {
    /// Stands in for the view's `@State`, with the same ordered main-actor
    /// delivery the run task uses.
    @MainActor
    private final class RunningScreen {
        var progress = CommunityRunProgress()
        var plan = CommunityRunPlan.assumed(for: .textGeneration)
        var didReachResult = false
        private(set) var passCountsSeenWhileRunning: [Int] = []

        func apply(_ state: CommunityRunProgress, plan: CommunityRunPlan) {
            progress = state
            self.plan = plan
            passCountsSeenWhileRunning.append(state.passesComplete)
        }
    }

    /// A child that emits real events, then blocks until a gate file appears.
    ///
    /// Blocking rather than sleeping: the assertions run while the process is
    /// genuinely still alive, and the test controls exactly when it may finish.
    private static func gatedChild(gate: URL) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-live-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        let script = directory.appendingPathComponent("rapid-mlx")
        var lines = ["#!/bin/sh"]
        lines.append(
            #"printf '\036{"event":"plan","task_type":"text_generation","protocol_id":"rapid-community-speed","protocol_version":2,"cases":[{"case_id":"pp512-tg128","warmup_rounds":1,"measured_rounds":5},{"case_id":"pp2048-tg512","warmup_rounds":1,"measured_rounds":5}],"total_passes":12}\n' >&2"#
        )
        lines.append(#"printf '\036Loading mlx-community/Bonsai (from the local Hugging Face cache)...\n' >&2"#)
        lines.append("sleep 0.1")
        lines.append(#"printf '\036pp512-tg128      warmup\n' >&2"#)
        lines.append("sleep 0.3")
        lines.append(#"printf '\036pp512-tg128      round 1/5   180.0 tok/s\n' >&2"#)
        // Blocked here, still running, while the test asserts.
        lines.append(#"while [ ! -f '\#(gate.path)' ]; do sleep 0.05; done"#)
        for round in 2...5 {
            lines.append(#"printf '\036pp512-tg128      round \#(round)/5   17\#(round).0 tok/s\n' >&2"#)
        }
        lines.append(#"printf '\036pp2048-tg512     warmup\n' >&2"#)
        for round in 1...5 {
            lines.append(#"printf '\036pp2048-tg512     round \#(round)/5   16\#(round).0 tok/s\n' >&2"#)
        }
        lines.append(#"printf '\036Saving result to this Mac...\n' >&2"#)
        lines.append(#"printf '\036Saved to this Mac\n' >&2"#)
        lines.append(#"echo '{"run_id":"live-run-1"}'"#)
        try lines.joined(separator: "\n").appending("\n")
            .write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path
        )
        return script
    }

    /// Waits for a main-actor condition, polling the run loop rather than
    /// sleeping past the window under test.
    @MainActor
    private static func waitUntil(
        timeout: TimeInterval = 10,
        _ condition: () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return condition()
    }

    @MainActor
    @Test("Progress is on screen while the child is still running")
    func liveDuringTheRun() async throws {
        let gate = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-gate-\(UUID().uuidString)")
        let binary = try Self.gatedChild(gate: gate)
        defer {
            try? FileManager.default.removeItem(at: binary.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: gate)
        }

        let screen = RunningScreen()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let (stream, feed) = AsyncStream<CommunityRunProgress>
            .makeStream(bufferingPolicy: .unbounded)

        // The production delivery shape: one ordered consumer on the main
        // actor, awaited before the Result transition.
        let delivery = Task { @MainActor in
            for await state in stream {
                screen.apply(state, plan: reducer.currentPlan)
            }
        }

        let run = Task { @MainActor in
            let output = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: CommunityBenchmarkCommand.benchmarkRunArguments(
                    alias: "bonsai-1.7b-2bit"
                ),
                onStandardErrorLine: { line in
                    guard let state = reducer.apply(line: line, at: Date()) else { return }
                    feed.yield(state)
                }
            )
            feed.finish()
            await delivery.value
            screen.didReachResult = true
            return output
        }

        // --- assertions while the child is BLOCKED and still alive ---
        let reachedTwoPasses = await Self.waitUntil {
            screen.progress.passesComplete >= 2
        }
        #expect(reachedTwoPasses, "two passes never reached the main actor while running")

        // The child is still blocked: Result has not been reached.
        #expect(!screen.didReachResult, "the run finished before the assertions")
        #expect(!FileManager.default.fileExists(atPath: gate.path))

        // Stage has moved beyond Getting ready.
        #expect(screen.progress.stage > .gettingReady)
        #expect(screen.progress.stage == .shortReplies)

        // The denominator came from the run's own plan event.
        #expect(screen.progress.totalPasses == 12)
        #expect(screen.progress.passCaption == "2 of 12 passes complete")

        // A determinate fraction exists, so the bar and the mascot's position
        // are real.
        let fraction = try #require(screen.progress.fraction)
        #expect(abs(fraction - 2.0 / 12.0) < 0.0001)

        // The newest measured value, with the pass it came from.
        let measurement = try #require(screen.progress.latestMeasurement)
        #expect(measurement.value == "180.0 tok/s")
        #expect(measurement.passNumber == 2)

        // Two completed passes is exactly enough timing evidence for an ETA.
        #expect(screen.progress.timeLeft != nil)

        // --- only now let the child finish ---
        FileManager.default.createFile(atPath: gate.path, contents: nil)
        let output = try await run.value

        #expect(screen.didReachResult)
        #expect(CommunityBenchmarkCommand.runID(from: output) == "live-run-1")
        // Everything read from stderr was applied before Result.
        #expect(screen.progress.passesComplete == 12)
        #expect(screen.progress.stage == .saving)
        // And the count only ever moved forwards.
        #expect(screen.passCountsSeenWhileRunning == screen.passCountsSeenWhileRunning.sorted())
    }

    @MainActor
    @Test("The first pass is on screen long before the process exits")
    func firstPassArrivesEarly() async throws {
        let gate = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-gate-\(UUID().uuidString)")
        let binary = try Self.gatedChild(gate: gate)
        defer {
            try? FileManager.default.removeItem(at: binary.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: gate)
        }

        let screen = RunningScreen()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let (stream, feed) = AsyncStream<CommunityRunProgress>
            .makeStream(bufferingPolicy: .unbounded)
        let delivery = Task { @MainActor in
            for await state in stream { screen.apply(state, plan: reducer.currentPlan) }
        }
        let started = Date()
        var planSeenAt: Date?

        let run = Task { @MainActor in
            _ = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "x"),
                onStandardErrorLine: { line in
                    guard let state = reducer.apply(line: line, at: Date()) else { return }
                    feed.yield(state)
                }
            )
            feed.finish()
            await delivery.value
        }

        // The plan event is the child's very first write, so it must be visible
        // essentially immediately — not at EOF.
        let sawPlan = await Self.waitUntil { screen.progress.totalPasses == 12 }
        planSeenAt = Date()
        #expect(sawPlan)
        let latency = try #require(planSeenAt).timeIntervalSince(started)
        #expect(
            latency < 1.5,
            "the plan event took \(latency)s to reach the screen; the child had not even finished"
        )

        FileManager.default.createFile(atPath: gate.path, contents: nil)
        _ = try await run.value
    }

    @MainActor
    @Test("Getting ready shows the CLI's own status text")
    func statusLineDuringPreparation() async throws {
        let gate = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-gate-\(UUID().uuidString)")
        let binary = try Self.gatedChild(gate: gate)
        defer {
            try? FileManager.default.removeItem(at: binary.deletingLastPathComponent())
            try? FileManager.default.removeItem(at: gate)
        }

        let screen = RunningScreen()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let (stream, feed) = AsyncStream<CommunityRunProgress>
            .makeStream(bufferingPolicy: .unbounded)
        let delivery = Task { @MainActor in
            for await state in stream { screen.apply(state, plan: reducer.currentPlan) }
        }
        let run = Task { @MainActor in
            _ = try await CommunityBenchmarkCommand.run(
                binary: binary,
                arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "x"),
                onStandardErrorLine: { line in
                    guard let state = reducer.apply(line: line, at: Date()) else { return }
                    feed.yield(state)
                }
            )
            feed.finish()
            await delivery.value
        }

        // Before any pass completes, the screen still has something true to
        // show: the loader's own message.
        let sawLoading = await Self.waitUntil {
            screen.progress.statusLine?.contains("Loading") == true
        }
        #expect(sawLoading, "the preparation phase had no live status to render")
        #expect(screen.progress.fraction == nil, "no pass has completed yet")
        #expect(screen.progress.stage == .gettingReady)

        FileManager.default.createFile(atPath: gate.path, contents: nil)
        _ = try await run.value
    }
}
