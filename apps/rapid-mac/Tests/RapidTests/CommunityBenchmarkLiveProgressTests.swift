import Foundation
import Testing
@testable import Rapid

/// The live Running experience, end to end, against a real child process.
///
/// Parser unit tests and fixture snapshots both passed while the shipping app
/// showed nothing but an elapsed clock, so neither is evidence. These spawn an
/// actual executable that streams RS-tagged progress on stderr over time and
/// finishes with a result document on stdout, then drive the *production*
/// `CommunityBenchmarkCommand.run(binary:arguments:onStandardErrorLine:)` and
/// the production reducer, and assert the observable state sequence.
///
/// Everything the run tab renders is derived from that sequence: the stage
/// stepper, `8 of 12 passes complete`, the mascot's position, the ETA and the
/// latest measurement.
@Suite("Live benchmark progress", .serialized)
struct CommunityBenchmarkLiveProgressTests {
    /// Writes a `/bin/sh` script that behaves like the packaged sidecar under
    /// `benchmark run --json --progress`.
    private static func fakeSidecar(
        _ body: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-live-progress-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        let script = directory.appendingPathComponent("rapid-mlx")
        try ("#!/bin/sh\n" + body).write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path
        )
        return script
    }

    /// The exact shape `rapid_mlx/community_bench/local_runner.py` emits for a
    /// two-case text protocol: a load line, a warmup line and five `round N/5`
    /// lines per case, each RS-tagged (`\x1e`, `PROGRESS_TAG` in `cli.py`).
    ///
    /// `printf` is used rather than `echo` so the record separator is a real
    /// 0x1E byte, and each line is followed by a short sleep so the reader sees
    /// several distinct pipe chunks rather than one buffered blob.
    private static func textProgressScript(resultRunID: String = "run-live-1") -> String {
        var lines: [String] = []
        lines.append(#"printf '\036Loading mlx-community/Qwen3.5-9B-4bit (from the local Hugging Face cache)...\n' >&2"#)
        lines.append("sleep 0.02")
        lines.append(#"printf '\036Model loaded in 4 s\n' >&2"#)
        for (caseID, tokens) in [("pp512-tg128", 46.1), ("pp2048-tg512", 18.7)] {
            lines.append("sleep 0.02")
            lines.append(#"printf '\036\#(caseID)      warmup\n' >&2"#)
            for round in 1...5 {
                lines.append("sleep 0.02")
                let rate = String(format: "%.1f", tokens + Double(round) * 0.1)
                lines.append(
                    #"printf '\036\#(caseID)      round \#(round)/5   \#(rate) tok/s\n' >&2"#
                )
            }
        }
        lines.append("sleep 0.02")
        lines.append(#"printf '\036Saving result to this Mac...\n' >&2"#)
        lines.append("sleep 0.02")
        lines.append(#"printf '\036Saved to this Mac\n' >&2"#)
        lines.append(#"cat <<'JSON'"#)
        lines.append(Self.resultDocument(runID: resultRunID))
        lines.append("JSON")
        return lines.joined(separator: "\n") + "\n"
    }

    private static func resultDocument(runID: String) -> String {
        #"""
        {"schema_version":1,"run_id":"\#(runID)",
         "started_at":"2026-09-15T10:00:00Z","completed_at":"2026-09-15T10:04:00Z",
         "execution":{"config_digest":"sha256:abc",
           "runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"},
           "resources":{"compute_dtype":"bf16"},
           "task":{"kind":"text_generation"}},
         "machine":{"os":{"version":"15.6.1"},
           "profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},
         "measurements":[{"case_id":"pp512-tg128","completed":true,
           "decode_duration_ms":2800,"output_tokens":129,"total_duration_ms":4200}],
         "model":{"schema_version":1,"identity_strength":"unresolved",
           "pipeline_kind":"text_generation",
           "components":[{"component_id":"primary","role":"primary",
             "source":{"kind":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit"},
             "artifact":{"format":"mlx-safetensors"},
             "quantization":{"kind":"unknown","base_dtype":"unknown"}}]},
         "outcome":{"status":"completed"},
         "workload":{"protocol_id":"rapid-community-speed","protocol_version":2,
           "task_type":"text_generation",
           "cases":[{"case_id":"pp512-tg128","measured_rounds":5,"warmup_rounds":1},
                    {"case_id":"pp2048-tg512","measured_rounds":5,"warmup_rounds":1}]}}
        """#
    }

    /// Collects every state the run tab would have rendered, in order.
    private actor Recorder {
        private(set) var states: [CommunityRunProgress] = []
        func record(_ state: CommunityRunProgress) { states.append(state) }
        var last: CommunityRunProgress? { states.last }
    }

    // MARK: - The whole path

    @Test("A real child process drives stage, count, ETA and latest measurement")
    func endToEndTextRun() async throws {
        let binary = try Self.fakeSidecar(Self.textProgressScript())
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        // The production reducer, behind a lock because the pipe reader hands
        // lines over from a detached task — exactly as the view's closure does.
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )

        let output = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "qwen3.5-9b-4bit"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )

        // Let the recorder drain the hops the production path also makes.
        try await Task.sleep(nanoseconds: 200_000_000)
        let states = await recorder.states

        // 1. The denominator is 12 for a two-case × (1 warmup + 5 measured)
        //    protocol, and it is known before any pass completes.
        #expect(states.allSatisfy { $0.totalPasses == 12 })

        // 2. Completed passes advance monotonically 1…12, from real events.
        let counts = states.map(\.passesComplete)
        #expect(counts == counts.sorted())
        #expect(counts.last == 12)
        #expect(Set(counts).sorted() == Array(0...12))

        // 3. The mascot and determinate bar appear only once a pass is real.
        let firstDeterminate = try #require(states.firstIndex { $0.fraction != nil })
        #expect(states[firstDeterminate].passesComplete == 1)
        #expect(states[..<firstDeterminate].allSatisfy { $0.fraction == nil })

        // 4. The stage sequence is the designed one, in order, derived from
        //    events rather than from a timer.
        var stageOrder: [CommunityRunStage] = []
        for state in states where stageOrder.last != state.stage {
            stageOrder.append(state.stage)
        }
        #expect(
            stageOrder == [
                .gettingReady, .warmingUp, .shortReplies, .longReplies, .saving,
            ]
        )

        // 5. The latest measurement is the newest real value, with its pass
        //    number — not a running average and not a fabricated figure.
        let withMeasurement = states.filter { $0.latestMeasurement != nil }
        #expect(!withMeasurement.isEmpty)
        let last = try #require(withMeasurement.last?.latestMeasurement)
        #expect(last.value == "19.2 tok/s")
        #expect(last.passNumber == 12)
        // The first measured pass of the first case is pass 2 overall
        // (warmup is pass 1) and carries that case's first rate.
        let first = try #require(withMeasurement.first?.latestMeasurement)
        #expect(first.value == "46.2 tok/s")
        #expect(first.passNumber == 2)

        // 6. Saving is reached before the process exits, so the final frame is
        //    truthful about what is happening to the archive.
        #expect(await recorder.last?.stage == .saving)
        #expect(await recorder.last?.isSaving == true)

        // 7. And the run really did produce a result document.
        #expect(CommunityBenchmarkCommand.runID(from: output) == "run-live-1")
    }

    @Test("ETA is withheld until two completed passes exist, then derived from them")
    func etaComesFromRealTimings() async throws {
        let binary = try Self.fakeSidecar(Self.textProgressScript())
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "qwen3.5-9b-4bit"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )
        try await Task.sleep(nanoseconds: 200_000_000)
        let states = await recorder.states

        // Nothing before the second completed pass may quote a time left: one
        // interval is not a rate, and a number invented here is the one the
        // user watches tick.
        for state in states where state.passesComplete < 2 {
            #expect(state.timeLeft == nil, "an ETA appeared after \(state.passesComplete) passes")
        }
        #expect(states.contains { $0.passesComplete >= 2 && $0.timeLeft != nil })
    }

    // MARK: - Other workloads

    @Test("An image run keeps its own shape and never claims 12 passes")
    func imageRunIsNotFabricated() async throws {
        let script = [
            #"printf '\036Starting local image server for z-image-turbo (loads the model)...\n' >&2"#,
            "sleep 0.02",
            #"printf '\036t2i-1024-square  warmup\n' >&2"#,
            "sleep 0.02",
            #"printf '\036t2i-1024-square  round 1/1  generating...\n' >&2"#,
            "sleep 0.02",
            #"printf '\036Saving result to this Mac...\n' >&2"#,
            "echo '{\"run_id\":\"img-1\"}'",
        ].joined(separator: "\n") + "\n"
        let binary = try Self.fakeSidecar(script)
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        let reducer = CommunityRunProgressBox(plan: CommunityRunPlan.assumed(for: .imageGeneration))
        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "z-image-turbo"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )
        try await Task.sleep(nanoseconds: 200_000_000)
        let states = await recorder.states

        #expect(states.allSatisfy { $0.totalPasses == 2 })
        #expect(states.map(\.passesComplete).last == 2)
        // Image runs have no short/long reply split to narrate.
        let stages = Set(states.map(\.stage))
        #expect(!stages.contains(.shortReplies))
        #expect(!stages.contains(.longReplies))
        #expect(stages.contains(.rendering))
        #expect(await recorder.last?.stage == .saving)
    }

    @Test("A video run stays indeterminate rather than inventing a denominator")
    func videoRunStaysIndeterminate() async throws {
        let script = [
            #"printf '\036Starting local video server for ltx-video (loads the model)...\n' >&2"#,
            "sleep 0.02",
            #"printf '\036t2v-768-24  round 1/1  generating...\n' >&2"#,
            "echo '{\"run_id\":\"vid-1\"}'",
        ].joined(separator: "\n") + "\n"
        let binary = try Self.fakeSidecar(script)
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        let reducer = CommunityRunProgressBox(plan: CommunityRunPlan.assumed(for: .videoGeneration))
        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "ltx-video"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )
        try await Task.sleep(nanoseconds: 200_000_000)
        let states = await recorder.states
        #expect(states.allSatisfy { $0.totalPasses == nil })
        #expect(states.allSatisfy { $0.fraction == nil })
        #expect(states.allSatisfy { $0.timeLeft == nil })
    }

    // MARK: - Transport robustness

    @Test("Progress survives arbitrary pipe chunking, including split lines")
    func chunkingDoesNotLoseEvents() async throws {
        // One `printf` with no trailing newline, then the rest — the reader
        // must join them rather than dropping or double-counting a pass.
        let script = [
            #"printf '\036pp512-tg128      war' >&2"#,
            "sleep 0.05",
            #"printf 'mup\n\036pp512-tg128      round 1/5   46.1 tok/s\n' >&2"#,
            "sleep 0.05",
            #"printf '\036pp512-tg128      round 2/5   46.4 tok/s\n\036pp512-tg128      round 3/5   46.2 tok/s\n' >&2"#,
            "echo '{\"run_id\":\"chunk-1\"}'",
        ].joined(separator: "\n") + "\n"
        let binary = try Self.fakeSidecar(script)
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "qwen3.5-9b-4bit"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )
        try await Task.sleep(nanoseconds: 200_000_000)
        #expect(await recorder.last?.passesComplete == 4)
    }

    @Test("Untagged stderr never advances the bar or reaches the screen")
    func untaggedStderrIsIgnored() async throws {
        let script = [
            // A traceback, a warning and a JSON failure document all arrive
            // untagged. None of them is progress.
            "echo 'Traceback (most recent call last):' >&2",
            "echo 'pp512-tg128      round 4/5   99.9 tok/s' >&2",
            #"echo '{"error":"something broke"}' >&2"#,
            "sleep 0.02",
            #"printf '\036pp512-tg128      warmup\n' >&2"#,
            "echo '{\"run_id\":\"untagged-1\"}'",
        ].joined(separator: "\n") + "\n"
        let binary = try Self.fakeSidecar(script)
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        let recorder = Recorder()
        let reducer = CommunityRunProgressBox(
            plan: CommunityRunPlan.assumed(for: .textGeneration)
        )
        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "qwen3.5-9b-4bit"),
            onStandardErrorLine: { line in
                guard let state = reducer.apply(line: line, at: Date()) else { return }
                Task { await recorder.record(state) }
            }
        )
        try await Task.sleep(nanoseconds: 200_000_000)
        let states = await recorder.states
        // Exactly one real pass, from the one tagged line.
        #expect(states.last?.passesComplete == 1)
        #expect(states.allSatisfy { $0.latestMeasurement?.value != "99.9 tok/s" })
    }

    @Test("The exact production arguments are what the child receives")
    func productionArgumentsReachTheChild() async throws {
        let script = """
        printf '%s\\n' "$@" > "$(dirname "$0")/argv.txt"
        echo '{"run_id":"argv-1"}'
        """
        let binary = try Self.fakeSidecar(script)
        defer { try? FileManager.default.removeItem(at: binary.deletingLastPathComponent()) }

        _ = try await CommunityBenchmarkCommand.run(
            binary: binary,
            arguments: CommunityBenchmarkCommand.benchmarkRunArguments(alias: "qwen3.5-9b-4bit")
        )
        let argv = try String(
            contentsOf: binary.deletingLastPathComponent().appendingPathComponent("argv.txt"),
            encoding: .utf8
        )
        .split(separator: "\n").map(String.init)
        #expect(
            argv == [
                "benchmark", "run", "qwen3.5-9b-4bit", "--json",
                "--progress", "--inherit-process-group",
            ]
        )
    }
}
