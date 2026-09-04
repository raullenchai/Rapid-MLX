import Darwin
import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Community Benchmark model-first projection")
struct CommunityBenchmarkModelTests {
    @Test("Atomic tasks select a protocol without modality tabs")
    func atomicTaskProjection() throws {
        let image = ModelEntry(
            alias: "flux2-klein-4b",
            hfRepo: "mlx-community/flux",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.imageGeneration],
            operationModes: [.textToImage]
        )
        let video = ModelEntry(
            alias: "wan2.2-ti2v-5b-q8",
            hfRepo: "mlx-community/wan",
            sizeOnDisk: nil,
            cached: false,
            taskTypes: [.videoGeneration],
            operationModes: [.textToVideo]
        )
        let audio = ModelEntry(
            alias: "qwen3-asr",
            hfRepo: "mlx-community/asr",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.speechRecognition]
        )

        let models = CommunityBenchmarkModel.models(from: [audio, video, image])
        #expect(models.map(\.entry.alias) == [image.alias, video.alias])
        #expect(models[0].protocolName == "Rapid Image Speed v1")
        #expect(models[1].protocolName == "Rapid Video Speed v1")
        #expect(models.allSatisfy { $0.isFocus })
    }

    @Test("Legacy catalog rows remain usable during the atomic migration")
    func legacyFallback() throws {
        let text = ModelEntry(
            alias: "custom-local-text",
            hfRepo: nil,
            sizeOnDisk: "2 GB",
            cached: true
        )
        let model = try #require(CommunityBenchmarkModel.models(from: [text]).first)
        #expect(model.task == .textGeneration)
        #expect(model.protocolName == "Rapid Community Speed v2")
    }

    @Test("Legacy diffusion kinds cannot invent registered protocol eligibility")
    func legacyDiffusionFallbackIsConservative() {
        let image = ModelEntry(
            alias: "legacy-image",
            hfRepo: "mlx-community/legacy-image",
            sizeOnDisk: nil,
            cached: true,
            kind: .image
        )
        let video = ModelEntry(
            alias: "ltx-2.3-mlx-q4",
            hfRepo: "mlx-community/ltx",
            sizeOnDisk: nil,
            cached: true,
            kind: .video
        )

        #expect(CommunityBenchmarkModel.models(from: [image, video]).isEmpty)
    }

    @Test("Atomic non-Wan video remains excluded from the Wan protocol")
    func nonWanAtomicVideoIsExcluded() {
        let ltx = ModelEntry(
            alias: "ltx-2.3-mlx-q4",
            hfRepo: "mlx-community/ltx",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.videoGeneration],
            operationModes: [.textToVideo]
        )

        #expect(CommunityBenchmarkModel.models(from: [ltx]).isEmpty)
    }

    @Test("CLI planning metadata is the shared memory-fit authority")
    func planningMetadata() throws {
        let image = ModelEntry(
            alias: "qwen-image",
            hfRepo: "mflux-community/qwen-image",
            sizeOnDisk: nil,
            cached: false,
            taskTypes: [.imageGeneration],
            operationModes: [.textToImage]
        )
        let metadata = CommunityBenchmarkCatalogModel(
            alias: image.alias,
            focus: true,
            estimatedMemoryGib: 64,
            memoryFit: "does_not_fit",
            protocolVersion: 1
        )
        let model = try #require(
            CommunityBenchmarkModel.models(
                from: [image], metadata: [image.alias: metadata]
            ).first
        )
        #expect(model.estimatedMemoryGib == 64)
        #expect(model.memoryFit == "does_not_fit")
    }

    @Test("CLI catalog filtering repairs an asynchronously selected alias")
    func asynchronousCatalogSelectionRepair() throws {
        let removed = ModelEntry(
            alias: "custom-local-text",
            hfRepo: "mlx-community/custom-local-text",
            sizeOnDisk: nil,
            cached: true
        )
        let retained = ModelEntry(
            alias: "another-local-text",
            hfRepo: "mlx-community/another-local-text",
            sizeOnDisk: nil,
            cached: false
        )
        let initial = CommunityBenchmarkModel.models(from: [removed, retained])
        let selectedBeforeCLIResponse = try #require(initial.first).entry.alias
        let metadata = CommunityBenchmarkCatalogModel(
            alias: retained.alias,
            focus: true,
            estimatedMemoryGib: 8,
            memoryFit: "fits",
            protocolVersion: 2
        )
        let filtered = CommunityBenchmarkModel.models(
            from: [removed, retained],
            metadata: [retained.alias: metadata]
        )

        #expect(selectedBeforeCLIResponse == removed.alias)
        #expect(
            CommunityBenchmarkModel.reconciledSelection(
                current: selectedBeforeCLIResponse,
                models: filtered
            ) == retained.alias
        )
    }

    @Test("Desktop keeps benchmark servers in its owned process group")
    func benchmarkRunTopologyFlag() {
        #expect(
            CommunityBenchmarkCommand.benchmarkRunArguments(alias: "flux2-klein-4b") == [
                "benchmark", "run", "flux2-klein-4b", "--json",
                "--inherit-process-group",
            ]
        )
        #expect(
            CommunityBenchmarkCommand.benchmarkResultsArguments() == [
                "benchmark", "results", "--limit", "8", "--json",
            ]
        )
        #expect(
            CommunityBenchmarkCommand.benchmarkSharePreviewArguments(
                runID: "00000000-0000-4000-8000-000000000001"
            ) == [
                "benchmark", "share", "00000000-0000-4000-8000-000000000001",
                "--preview", "--json",
            ]
        )
        #expect(
            CommunityBenchmarkCommand.benchmarkShareArguments(
                runID: "00000000-0000-4000-8000-000000000001",
                installID: "012345abcdef",
                payloadDigest: "sha256:aaaaaaaa",
                bodyDigest: "sha256:bbbbbbbb",
                target: "https://rapidmlx.com/api/benchmarks/atomic"
            ) == [
                "benchmark", "share", "00000000-0000-4000-8000-000000000001",
                "--yes", "--install-id", "012345abcdef",
                "--payload-digest", "sha256:aaaaaaaa",
                "--body-digest", "sha256:bbbbbbbb", "--target",
                "https://rapidmlx.com/api/benchmarks/atomic", "--json",
            ]
        )
    }

    @Test("Desktop consent preview displays the exact upload payload")
    func exactSharePreview() throws {
        let data = Data(
            #"{"schema_version":1,"target":"https://rapidmlx.com/api/benchmarks/atomic","install_id":"012345abcdef","payload_digest":"sha256:aaaaaaaa","body_digest":"sha256:bbbbbbbb","payload_json":"{\"run_id\": \"00000000-0000-4000-8000-000000000001\", \"install_id\": \"012345abcdef\", \"measurements\": [{\"round_index\": 1, \"total_duration_ms\": 123.5}], \"execution\": {\"task\": {\"max_tokens\": 128}}}"}"#.utf8
        )
        let preview = try CommunityBenchmarkCommand.decodeSharePreview(
            data, runID: "00000000-0000-4000-8000-000000000001"
        )

        #expect(preview.installID == "012345abcdef")
        #expect(preview.payloadDigest == "sha256:aaaaaaaa")
        #expect(preview.bodyDigest == "sha256:bbbbbbbb")
        #expect(
            preview.payloadJSON
                == #"{"run_id": "00000000-0000-4000-8000-000000000001", "install_id": "012345abcdef", "measurements": [{"round_index": 1, "total_duration_ms": 123.5}], "execution": {"task": {"max_tokens": 128}}}"#
        )
    }

    @Test("Contributor receipt produces the public profile identity")
    func contributorProfileIdentity() throws {
        let contributor = CommunityBenchmarkContributor(
            name: "modest-slate-wombat",
            tag: "545"
        )

        #expect(contributor.displayName == "modest-slate-wombat ·545")
        #expect(
            contributor.profileURL?.absoluteString
                == "https://rapidmlx.com/leaderboard/contributors/modest-slate-wombat-545"
        )
    }

    @Test("Contributor profile URL percent-encodes embedded slashes in the identity")
    func contributorProfileURLPercentEncodesSlash() {
        let contributor = CommunityBenchmarkContributor(
            name: "modest/slate+wombat",
            tag: "5 4"
        )
        // Mirrors the CLI client's urllib quote(f"{name}-{tag}", safe="-"): every
        // character outside [A-Za-z0-9_.~-] is percent-encoded, so "/" cannot be
        // interpreted as a path separator and the lone segment round-trips.
        #expect(
            contributor.profileURL?.absoluteString
                == "https://rapidmlx.com/leaderboard/contributors/modest%2Fslate%2Bwombat-5%204"
        )
    }

    @Test("Desktop decodes contributor identity and valid anonymous receipts")
    func contributorReceiptDecoding() throws {
        let full = try JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(
                #"{"submission_id":"00000000-0000-4000-8000-000000000001","already_exists":false,"accepted_at":"2026-09-01T20:00:00Z","contributor":{"name":"modest-slate-wombat","tag":"545"}}"#.utf8
            )
        )
        #expect(full.contributor?.displayName == "modest-slate-wombat ·545")
        #expect(
            full.contributionURL.absoluteString
                == "https://rapidmlx.com/leaderboard/contributors/modest-slate-wombat-545"
        )
        #expect(full.contributionLinkTitle == "modest-slate-wombat ·545")

        let anonymous = try JSONDecoder().decode(
            CommunityBenchmarkReceipt.self,
            from: Data(
                #"{"submission_id":"00000000-0000-4000-8000-000000000001","already_exists":true,"accepted_at":"2026-09-01T20:00:00Z","contributor":null}"#.utf8
            )
        )
        #expect(anonymous.contributor == nil)
        #expect(anonymous.contributionURL.absoluteString == "https://rapidmlx.com/leaderboard")
        #expect(anonymous.contributionLinkTitle == "View Community Benchmark")
    }

    @Test("Benchmark pipe capture bounds stdout heads and stderr tails")
    func boundedPipeCapture() throws {
        let headPipe = Pipe()
        headPipe.fileHandleForWriting.write(Data("abcdefgh".utf8))
        try headPipe.fileHandleForWriting.close()
        let head = CommunityBenchmarkCommand._testReadBoundedPipe(
            headPipe.fileHandleForReading,
            maxBytes: 4,
            retainTail: false
        )
        #expect(head.data == Data("abcd".utf8))
        #expect(head.truncated)

        let tailPipe = Pipe()
        tailPipe.fileHandleForWriting.write(Data("abcdefgh".utf8))
        try tailPipe.fileHandleForWriting.close()
        let tail = CommunityBenchmarkCommand._testReadBoundedPipe(
            tailPipe.fileHandleForReading,
            maxBytes: 4,
            retainTail: true
        )
        #expect(tail.data == Data("efgh".utf8))
        #expect(tail.truncated)
    }

    @Test("Cancellation after process exit cannot signal a stale process group")
    func cancellationAfterExitClearsTrackedChild() throws {
        let box = BenchmarkProcessBox()
        let stdout = Pipe()
        let stderr = Pipe()
        let child = try box.start(
            binary: URL(fileURLWithPath: "/usr/bin/true"),
            arguments: [],
            standardOutput: stdout,
            standardError: stderr
        )

        #expect(box.waitForCompletion(child) == nil)
        #expect(!box._testHasTrackedChild)
        box.cancel()
        #expect(!box._testHasTrackedChild)
    }

    @Test("Cancelling a benchmark reaps its descendant process group")
    func cancellationReapsDescendant() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-benchmark-cancel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("benchmark-fixture")
        let pidFile = root.appendingPathComponent("descendant.pid")
        try """
        #!/bin/sh
        sleep 30 &
        echo $! > "$1"
        wait
        """.write(to: script, atomically: true, encoding: .utf8)
        chmod(script.path, 0o755)

        let run = Task {
            try await CommunityBenchmarkCommand.run(
                binary: script,
                arguments: [pidFile.path]
            )
        }
        let spawnDeadline = Date().addingTimeInterval(2)
        while !FileManager.default.fileExists(atPath: pidFile.path),
              Date() < spawnDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        let pidText = try String(contentsOf: pidFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let descendantPID = try #require(pid_t(pidText))

        run.cancel()
        do {
            _ = try await run.value
            Issue.record("cancelled benchmark unexpectedly succeeded")
        } catch is CancellationError {
            // Expected. The cancellation handler must finish group teardown
            // before publishing this result to the view.
        } catch {
            Issue.record("cancelled benchmark returned \(error) instead of CancellationError")
        }

        #expect(!processExists(descendantPID))
    }

    @Test("Benchmark teardown is bounded when SIGKILL remains pending")
    func teardownBoundsUninterruptibleProcessGroup() {
        var signals: [Int32] = []
        var tick: TimeInterval = 0
        let origin = Date(timeIntervalSince1970: 1_000)

        let exited = BenchmarkProcessBox.boundedTermination(
            isAlive: { true },
            signal: { signals.append($0) },
            termGrace: 1,
            killGrace: 1,
            now: {
                defer { tick += 1 }
                return origin.addingTimeInterval(tick)
            },
            sleep: { _ in }
        )

        #expect(!exited)
        #expect(signals == [SIGTERM, SIGKILL])
    }

    @Test(
        "Group monitor retries EINTR and fires only on explicit ESRCH",
        .timeLimit(.minutes(1))
    )
    func monitorRetriesEINTRBeforeExit() async {
        let script = ProbeScript([EINTR, EINTR, ESRCH])
        await withCheckedContinuation { (exit: CheckedContinuation<Void, Never>) in
            ProcessGroupChild.monitorProcessGroupUntilExit(
                processGroupID: 12345,
                probe: { _ in script.next() },
                onExit: { exit.resume() }
            )
        }
        // Exactly three probes happened before onExit: both EINTRs were
        // re-probed instead of being treated as process-group exit.
        #expect(script.callCount == 3)
    }

    @Test(
        "Alive and unexpected probe errors keep the quarantine reservation",
        .timeLimit(.minutes(1))
    )
    func monitorTreatsUnexpectedProbeErrorsAsAlive() async {
        let script = ProbeScript([0, EPERM, EINVAL, ESRCH])
        await withCheckedContinuation { (exit: CheckedContinuation<Void, Never>) in
            ProcessGroupChild.monitorProcessGroupUntilExit(
                processGroupID: 12345,
                probe: { _ in script.next() },
                onExit: { exit.resume() }
            )
        }
        // onExit fired only after the ESRCH probe; success, EPERM, and the
        // unexpected EINVAL all kept the group treated as alive.
        #expect(script.callCount == 4)
    }

    private func processExists(_ pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }
}

/// Deterministic stand-in for the `kill(-pgid, 0)` liveness probe: replays
/// a scripted result sequence, then reports ESRCH forever.
private final class ProbeScript: @unchecked Sendable {
    private let lock = NSLock()
    private var results: [Int32]
    private var calls = 0

    init(_ results: [Int32]) {
        self.results = results
    }

    func next() -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        calls += 1
        return results.isEmpty ? ESRCH : results.removeFirst()
    }

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
}
