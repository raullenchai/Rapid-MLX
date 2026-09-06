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

    @Test("Full product catalog replaces the chat-only launch fallback")
    func productCatalogFeedIncludesMedia() {
        let chat = ModelEntry(
            alias: "qwen3.5-9b-4bit",
            hfRepo: "mlx-community/qwen",
            sizeOnDisk: nil,
            cached: true,
            taskTypes: [.textGeneration]
        )
        let image = ModelEntry(
            alias: "flux2-klein-4b",
            hfRepo: "mlx-community/flux",
            sizeOnDisk: nil,
            cached: false,
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

        let catalog = CommunityBenchmarkModel.resolvedCatalog(
            product: [chat, image, video],
            fallback: [chat]
        )

        #expect(Set(CommunityBenchmarkModel.models(from: catalog).map(\.entry.alias)) == [
            chat.alias, image.alias, video.alias,
        ])
    }

    @Test("Older runtimes retain the launch catalog fallback")
    func productCatalogFeedFallsBack() {
        let chat = ModelEntry(
            alias: "legacy-chat",
            hfRepo: nil,
            sizeOnDisk: nil,
            cached: true
        )

        #expect(
            CommunityBenchmarkModel.resolvedCatalog(
                product: nil,
                fallback: [chat]
            ) == [chat]
        )
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

    // MARK: - Result rows (tok/s, TTFT, localized time)

    /// Verbatim `benchmark results --json` run record from a Qwen3.5-9B-4bit
    /// text run on an M3 Pro (two cases × five completed rounds).
    private static let textRunFixture = #"""
    {"completed_at":"2026-09-06T04:37:42.144907Z","execution":{"config_digest":"sha256:069529b6e4cc5059f87292a388e7bf28e40da34228bf2a73d52ee53b4cd1cca1","runtime":{"mlx":"0.32.2","python":"3.12.8","rapid_mlx":"0.13.4"}},"machine":{"os":{"version":"15.6.1"},"profile":{"chip":"Apple M3 Pro","cpu_cores":12,"gpu_cores":18,"memory_gib":18}},"measurements":[{"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4916.7403327301145,"output_tokens":128,"peak_active_memory_mib":6875,"prompt_tokens":512,"round_index":1,"total_duration_ms":6396.939542144537,"ttft_ms":1480.1992094144225},{"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4924.277124926448,"output_tokens":128,"peak_active_memory_mib":6875,"prompt_tokens":512,"round_index":2,"total_duration_ms":6415.198541246355,"ttft_ms":1490.9214163199067},{"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4930.472332984209,"output_tokens":128,"peak_active_memory_mib":6875,"prompt_tokens":512,"round_index":3,"total_duration_ms":6422.98545781523,"ttft_ms":1492.5131248310208},{"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4946.83812558651,"output_tokens":128,"peak_active_memory_mib":6875,"prompt_tokens":512,"round_index":4,"total_duration_ms":6436.7318749427795,"ttft_ms":1489.8937493562698},{"case_id":"pp512-tg128","completed":true,"decode_duration_ms":4905.502292327583,"output_tokens":128,"peak_active_memory_mib":6875,"prompt_tokens":512,"round_index":5,"total_duration_ms":6399.486541748047,"ttft_ms":1493.984249420464},{"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20015.26966691017,"output_tokens":512,"peak_active_memory_mib":6875,"prompt_tokens":2048,"round_index":1,"total_duration_ms":25820.96404209733,"ttft_ms":5805.694375187159},{"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20225.04312451929,"output_tokens":512,"peak_active_memory_mib":6875,"prompt_tokens":2048,"round_index":2,"total_duration_ms":26035.800124518573,"ttft_ms":5810.756999999285},{"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20210.930791683495,"output_tokens":512,"peak_active_memory_mib":6875,"prompt_tokens":2048,"round_index":3,"total_duration_ms":26022.891083732247,"ttft_ms":5811.960292048752},{"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20368.177874945104,"output_tokens":512,"peak_active_memory_mib":6875,"prompt_tokens":2048,"round_index":4,"total_duration_ms":26177.824749611318,"ttft_ms":5809.646874666214},{"case_id":"pp2048-tg512","completed":true,"decode_duration_ms":20183.308540843427,"output_tokens":512,"peak_active_memory_mib":6875,"prompt_tokens":2048,"round_index":5,"total_duration_ms":26036.951416172087,"ttft_ms":5853.64287532866}],"model":{"components":[{"source":{"repo_id":"mlx-community/Qwen3.5-9B-4bit"}}]},"outcome":{"status":"completed"},"run_id":"c323a717-c37a-4107-a220-969e54246eb1","workload":{"cases":[{"case_id":"pp512-tg128","measured_rounds":5,"target_output_tokens":128,"target_prompt_tokens":512,"warmup_rounds":1},{"case_id":"pp2048-tg512","measured_rounds":5,"target_output_tokens":512,"target_prompt_tokens":2048,"warmup_rounds":1}],"task_type":"text_generation"}}
    """#

    private static func decodeRun(_ json: String) throws -> CommunityBenchmarkResult {
        try JSONDecoder().decode(CommunityBenchmarkResult.self, from: Data(json.utf8))
    }

    @Test("Result row shows median decode tok/s and TTFT for the short case")
    func resultHeadlineMatchesCLISummary() throws {
        let run = try Self.decodeRun(Self.textRunFixture)
        let summaries = run.caseSummaries

        #expect(summaries.map(\.caseID) == ["pp512-tg128", "pp2048-tg512"])
        #expect(summaries.map(\.rounds) == [5, 5])
        // Medians over the five completed rounds with the website's
        // (output_tokens - 1) / decode_duration formula, not an average of
        // total_duration across both cases ("16.2 s avg across cases").
        #expect(run.headline == "25.8 tok/s · TTFT 1491 ms")
        #expect(run.secondaryLines == ["pp2048-tg512: 25.3 tok/s · TTFT 5811 ms"])
        #expect(run.repoID == "mlx-community/Qwen3.5-9B-4bit")
    }

    @Test("Result summary ignores incomplete rounds and keeps status for failed runs")
    func resultSummaryCompletedRoundsOnly() throws {
        let measurements = [
            CommunityBenchmarkResult.Measurement(
                caseID: "pp512-tg128", completed: true, outputTokens: 101,
                ttftMS: 500, decodeDurationMS: 2_000, totalDurationMS: 2_500
            ),
            CommunityBenchmarkResult.Measurement(
                caseID: "pp512-tg128", completed: true, outputTokens: 101,
                ttftMS: 700, decodeDurationMS: 4_000, totalDurationMS: 4_700
            ),
            // Aborted round: must not drag the medians down.
            CommunityBenchmarkResult.Measurement(
                caseID: "pp512-tg128", completed: false, outputTokens: 3,
                ttftMS: 90_000, decodeDurationMS: 1, totalDurationMS: 90_001
            ),
            // Undeclared case still shows up, after the declared ones.
            CommunityBenchmarkResult.Measurement(
                caseID: "extra", completed: true, outputTokens: 11,
                ttftMS: nil, decodeDurationMS: 1_000, totalDurationMS: 1_200
            ),
        ]
        let summaries = CommunityBenchmarkResult.summarize(
            measurements: measurements, declaredOrder: ["pp512-tg128"]
        )
        #expect(summaries.count == 2)
        #expect(summaries[0].rounds == 2)
        // Even count → mean of the two middle values: (50 + 25) / 2 tok/s.
        #expect(summaries[0].decodeTokensPerSecond == 37.5)
        #expect(summaries[0].ttftMS == 600)
        #expect(summaries[0].headline == "37.5 tok/s · TTFT 600 ms")
        #expect(summaries[1].caseID == "extra")
        #expect(summaries[1].headline == "10.0 tok/s")

        // No completed rounds at all → no headline, so the row falls back to
        // the outcome status.
        let failed = try Self.decodeRun(
            Self.textRunFixture.replacingOccurrences(
                of: #""completed":true"#, with: #""completed":false"#
            ).replacingOccurrences(
                of: #""status":"completed""#, with: #""status":"failed""#
            )
        )
        #expect(failed.headline == nil)
        #expect(failed.outcome.status == "failed")
    }

    @Test("Image and video runs summarize as median wall seconds")
    func mediaResultSummaryUsesWallTime() {
        let render = CommunityBenchmarkResult.Measurement(
            caseID: "t2i-1024", completed: true, outputTokens: nil,
            ttftMS: nil, decodeDurationMS: nil, totalDurationMS: 12_345
        )
        let summaries = CommunityBenchmarkResult.summarize(
            measurements: [render], declaredOrder: []
        )
        #expect(summaries.map(\.headline) == ["12.3 s per run"])
        #expect(CommunityBenchmarkResult.CaseSummary.formatMilliseconds(5_811) == "5811 ms")
        #expect(CommunityBenchmarkResult.CaseSummary.formatMilliseconds(12_400) == "12.4 s")
    }

    @Test("Result timestamps render as localized Today / Yesterday / date")
    func completedAtRendering() throws {
        let calendar = Calendar(identifier: .gregorian)
        let locale = Locale(identifier: "en_US")
        let zone = try #require(TimeZone(identifier: "Asia/Singapore"))
        func render(_ raw: String, now: String) -> String {
            CommunityBenchmarkResult.formatCompletedAt(
                raw,
                now: CommunityBenchmarkResult.parseTimestamp(now)!,
                calendar: calendar, locale: locale, timeZone: zone
            )
            // ICU separates "9:33" and "PM" with a narrow no-break space on
            // newer systems and a plain space on older ones; the wording is
            // what this test pins, not the ICU version.
            .replacingOccurrences(of: "\u{202F}", with: " ")
            .replacingOccurrences(of: "\u{00A0}", with: " ")
        }
        // 04:37 UTC is 12:37 in Singapore on the same day.
        #expect(render("2026-09-06T04:37:42.144907Z", now: "2026-09-06T10:00:00Z") == "Today 12:37 PM")
        #expect(render("2026-09-06T04:37:42Z", now: "2026-09-07T10:00:00Z") == "Yesterday 12:37 PM")
        #expect(render("2026-09-05T13:33:00Z", now: "2026-09-20T10:00:00Z") == "Sep 5, 9:33 PM")
        #expect(render("2025-12-31T13:33:00Z", now: "2026-09-20T10:00:00Z") == "Dec 31, 2025, 9:33 PM")
        // Unparseable stamps stay raw rather than crashing or vanishing.
        #expect(render("not-a-date", now: "2026-09-20T10:00:00Z") == "not-a-date")
    }

    // MARK: - Model picker sections

    @Test("Model picker groups recommended, downloaded, then everything else")
    func pickerSections() {
        func entry(_ alias: String, cached: Bool) -> ModelEntry {
            ModelEntry(
                alias: alias, hfRepo: "mlx-community/\(alias)", sizeOnDisk: nil,
                cached: cached, taskTypes: [.textGeneration]
            )
        }
        func meta(_ alias: String, focus: Bool, fit: String) -> CommunityBenchmarkCatalogModel {
            CommunityBenchmarkCatalogModel(
                alias: alias, focus: focus, estimatedMemoryGib: 8,
                memoryFit: fit, protocolVersion: 2
            )
        }
        let catalog = [
            entry("zeta-local", cached: true),
            entry("alpha-remote", cached: false),
            entry("qwen3.5-9b-4bit", cached: false),
            entry("qwen3.8-27b-4bit", cached: false),
            entry("gemma-4-e4b-4bit", cached: true),
        ]
        let metadata = [
            "zeta-local": meta("zeta-local", focus: false, fit: "fits"),
            "alpha-remote": meta("alpha-remote", focus: false, fit: "fits"),
            "qwen3.5-9b-4bit": meta("qwen3.5-9b-4bit", focus: true, fit: "fits"),
            "qwen3.8-27b-4bit": meta("qwen3.8-27b-4bit", focus: true, fit: "does_not_fit"),
            "gemma-4-e4b-4bit": meta("gemma-4-e4b-4bit", focus: true, fit: "fits"),
        ]
        let models = CommunityBenchmarkModel.models(from: catalog, metadata: metadata)
        let sections = CommunityBenchmarkModel.pickerSections(models)

        #expect(sections.map(\.title) == [
            CommunityBenchmarkModel.recommendedSectionTitle,
            CommunityBenchmarkModel.downloadedSectionTitle,
            CommunityBenchmarkModel.allModelsSectionTitle,
        ])
        #expect(sections[0].models.map(\.entry.alias) == ["gemma-4-e4b-4bit", "qwen3.5-9b-4bit"])
        #expect(sections[1].models.map(\.entry.alias) == ["zeta-local"])
        // A focus model that does not fit is not "recommended for this Mac".
        #expect(sections[2].models.map(\.entry.alias) == ["qwen3.8-27b-4bit", "alpha-remote"])
        // Every alias lands in exactly one section so picker tags stay unique
        // and reconciliation over the flat list still finds each selection.
        let flattened = sections.flatMap { $0.models.map(\.entry.alias) }
        #expect(Set(flattened).count == flattened.count)
        #expect(Set(flattened) == Set(models.map(\.entry.alias)))
        #expect(
            CommunityBenchmarkModel.reconciledSelection(current: "zeta-local", models: models)
                == "zeta-local"
        )
        // Empty groups are omitted instead of rendering a bare header.
        let onlyRemote = CommunityBenchmarkModel.pickerSections(
            CommunityBenchmarkModel.models(from: [entry("alpha-remote", cached: false)])
        )
        #expect(onlyRemote.map(\.title) == [CommunityBenchmarkModel.allModelsSectionTitle])
    }

    // MARK: - Running feedback

    @Test("Running status names the model, the scope, and the expected duration")
    func runningStatusCopy() throws {
        let text = try #require(
            CommunityBenchmarkModel.models(from: [
                ModelEntry(
                    alias: "qwen3.5-9b-4bit", hfRepo: "mlx-community/Qwen3.5-9B-4bit",
                    sizeOnDisk: nil, cached: true, taskTypes: [.textGeneration]
                ),
            ]).first
        )
        #expect(
            CommunityBenchmarkRunStatus.description(for: text)
                == "Measuring qwen3.5-9b-4bit · 2 cases × (1 warmup + 5 rounds) · usually 2–5 minutes"
        )
        let image = try #require(
            CommunityBenchmarkModel.models(from: [
                ModelEntry(
                    alias: "flux2-klein-4b", hfRepo: "mlx-community/flux", sizeOnDisk: nil,
                    cached: false, taskTypes: [.imageGeneration], operationModes: [.textToImage]
                ),
            ]).first
        )
        #expect(
            CommunityBenchmarkRunStatus.description(for: image)
                == "Measuring flux2-klein-4b · 1 warmup + 1 measured render · usually 1–3 minutes · plus the download"
        )
        let start = Date(timeIntervalSince1970: 1_000)
        #expect(CommunityBenchmarkRunStatus.elapsed(from: start, to: start.addingTimeInterval(0)) == "0:00")
        #expect(CommunityBenchmarkRunStatus.elapsed(from: start, to: start.addingTimeInterval(65.9)) == "1:05")
        #expect(CommunityBenchmarkRunStatus.elapsed(from: start, to: start.addingTimeInterval(754)) == "12:34")
        #expect(CommunityBenchmarkRunStatus.elapsed(from: start, to: start.addingTimeInterval(-5)) == "0:00")
    }

    @Test("Only CLI round-progress lines are mirrored from stderr")
    func progressLineFilter() {
        #expect(
            CommunityBenchmarkRunStatus.progressLine(from: "pp512-tg128  round 3/5  46.1 tok/s\n")
                == "pp512-tg128 round 3/5 46.1 tok/s"
        )
        #expect(CommunityBenchmarkRunStatus.progressLine(from: "  warmup round 1/1  ") == "warmup round 1/1")
        #expect(CommunityBenchmarkRunStatus.progressLine(from: "Traceback (most recent call last):") == nil)
        #expect(CommunityBenchmarkRunStatus.progressLine(from: "Downloading shard 3/5") == nil)
        #expect(CommunityBenchmarkRunStatus.progressLine(from: "") == nil)
        #expect(
            CommunityBenchmarkRunStatus.progressLine(
                from: "round 1/5 " + String(repeating: "x", count: 300)
            ) == nil
        )
    }

    @Test("Stderr line splitter reassembles lines across chunks and bounds partial lines")
    func stderrLineSplitter() {
        let seen = LineCollector()
        let splitter = CommunityBenchmarkCommand.LineSplitter(onLine: seen.append)
        splitter.consume(Data("pp512-tg128 rou".utf8))
        splitter.consume(Data("nd 1/5 45.0 tok/s\npp512-tg128 round 2/5".utf8))
        splitter.consume(Data(" 45.2 tok/s\n\n".utf8))
        #expect(seen.drain() == ["pp512-tg128 round 1/5 45.0 tok/s", "pp512-tg128 round 2/5 45.2 tok/s", ""])

        // An unterminated line longer than the cap is dropped, including its
        // tail once the newline finally arrives, instead of growing forever.
        splitter.consume(Data(repeating: UInt8(ascii: "a"), count: 5_000))
        splitter.consume(Data("tail\nround 3/5\n".utf8))
        #expect(seen.drain() == ["round 3/5"])

        // A nil observer costs nothing and never touches the stream.
        let silent = CommunityBenchmarkCommand.LineSplitter(onLine: nil)
        silent.consume(Data("round 1/5\n".utf8))
    }

    @Test("Benchmark run forwards stderr lines without disturbing stdout capture")
    func runStreamsStderrLines() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-benchmark-stderr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("benchmark-fixture")
        try """
        #!/bin/sh
        echo 'pp512-tg128  round 1/5  45.0 tok/s' >&2
        echo 'pp512-tg128  round 2/5  45.2 tok/s' >&2
        echo '{"ok":true}'
        """.write(to: script, atomically: true, encoding: .utf8)
        chmod(script.path, 0o755)

        let lines = LineCollector()
        let output = try await CommunityBenchmarkCommand.run(
            binary: script,
            arguments: [],
            onStandardErrorLine: lines.append
        )
        #expect(String(data: output, encoding: .utf8) == "{\"ok\":true}\n")
        #expect(lines.drain() == [
            "pp512-tg128  round 1/5  45.0 tok/s",
            "pp512-tg128  round 2/5  45.2 tok/s",
        ])
    }

    private func processExists(_ pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }
}

/// Thread-safe sink for stderr lines delivered off the main actor.
private final class LineCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ line: String) {
        lock.lock()
        lines.append(line)
        lock.unlock()
    }

    func drain() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let snapshot = lines
        lines.removeAll()
        return snapshot
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
