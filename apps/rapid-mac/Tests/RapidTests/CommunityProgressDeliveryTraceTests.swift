import Foundation
import Testing
@testable import Rapid

/// Diagnostic trace: where does a progress line actually arrive?
///
/// Four timestamps per line — raw arrival in the stderr callback, reducer
/// update, MainActor application, and process completion — so a delay can be
/// attributed to a stage instead of guessed at. Not a pass/fail regression
/// test; the decisive one lives in `CommunityLiveRunningStateTests`.
@Suite("Progress delivery trace", .serialized)
struct CommunityProgressDeliveryTraceTests {
    @MainActor
    private final class Screen {
        var applied: [(sequence: Int, at: Date, passes: Int)] = []
        func apply(_ state: CommunityRunProgress, sequence: Int) {
            applied.append((sequence, Date(), state.passesComplete))
        }
    }

    private static func slowChild() throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-trace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        let script = directory.appendingPathComponent("rapid-mlx")
        var lines = ["#!/bin/sh"]
        lines.append(
            #"printf '\036{"event":"plan","task_type":"text_generation","cases":[{"case_id":"a","warmup_rounds":1,"measured_rounds":5},{"case_id":"b","warmup_rounds":1,"measured_rounds":5}],"total_passes":12}\n' >&2"#
        )
        for index in 1...6 {
            lines.append("sleep 0.25")
            lines.append(#"printf '\036a      round \#(index)/5   10.\#(index) tok/s\n' >&2"#)
        }
        lines.append(#"echo '{"run_id":"trace-1"}'"#)
        try lines.joined(separator: "\n").appending("\n")
            .write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path
        )
        return script
    }

    @MainActor
    @Test("Trace when each progress line reaches the main actor")
    func traceDelivery() async throws {
        let binary = try Self.slowChild()
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let screen = Screen()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        let started = Date()
        let arrivals = ArrivalLog()
        let sequencer = ProgressSequencer()

        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "x"),
            onStandardErrorLine: { line in
                let raw = Date()
                guard let state = reducer.apply(line: line, at: raw) else { return }
                let reduced = Date()
                let sequence = sequencer.next()
                arrivals.record(sequence: sequence, raw: raw, reduced: reduced)
                // The production shape at the time this trace was written.
                Task { @MainActor in
                    screen.apply(state, sequence: sequence)
                }
            }
        )
        let completed = Date()
        // Give any still-queued hops a chance, so the trace shows whether they
        // were merely late rather than lost.
        try await Task.sleep(nanoseconds: 300_000_000)

        let log = arrivals.entries
        print("--- progress delivery trace (t=0 at spawn) ---")
        for entry in log {
            let appliedAt = screen.applied.first { $0.sequence == entry.sequence }?.at
            let appliedOffset = appliedAt.map {
                String(format: "%.3f", $0.timeIntervalSince(started))
            } ?? "NEVER"
            print(
                String(
                    format: "seq %2d  raw %.3f  reduced %.3f  mainActor %@",
                    entry.sequence,
                    entry.raw.timeIntervalSince(started),
                    entry.reduced.timeIntervalSince(started),
                    appliedOffset
                )
            )
        }
        print(String(format: "process completed at %.3f", completed.timeIntervalSince(started)))

        let appliedBeforeExit = screen.applied.filter { $0.at < completed }
        print("applied BEFORE process exit: \(appliedBeforeExit.count) of \(log.count)")

        // The diagnosis this trace exists to produce. It is an expectation so
        // the number appears in the run output either way.
        #expect(
            appliedBeforeExit.count > 0,
            "no progress reached the main actor while the child was running"
        )
    }

    /// Thread-safe arrival log; the stderr callback runs off the main actor.
    private final class ArrivalLog: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [(sequence: Int, raw: Date, reduced: Date)] = []

        func record(sequence: Int, raw: Date, reduced: Date) {
            lock.lock()
            defer { lock.unlock() }
            storage.append((sequence, raw, reduced))
        }

        var entries: [(sequence: Int, raw: Date, reduced: Date)] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }
}
