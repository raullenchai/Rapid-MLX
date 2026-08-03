import Foundation
import Testing
@testable import Rapid

@Suite("Download/catalog hardening")
struct DownloadCatalogHardeningTests {
    @MainActor
    @Test("DownloadManager rejects option-shaped aliases before spawning rapid-mlx")
    func downloadManagerRejectsOptionAlias() {
        let mgr = DownloadManager(binaryPath: URL(fileURLWithPath: "/bin/echo"))
        #expect(!mgr.startDownload(alias: "--help"))

        let job = mgr.job(for: "--help")
        if case .failed(let message) = job?.status {
            #expect(message.contains("isn't valid"))
        } else {
            Issue.record("Expected invalid-alias failure, got \(String(describing: job?.status))")
        }
    }

    @Test("ModelCatalog parsers reject unsafe aliases and sanitize HF repo ids")
    func catalogParserSanitizesAliasesAndRepos() {
        let available = ModelCatalog.parseAvailable(
            """
            Alias                 Family
            --help                bad
            qwen3.6-27b           qwen
            mlx/community         bad
            """
        )
        #expect(available.map { $0.0 } == ["qwen3.6-27b"])

        let cached = ModelCatalog.parseCached(
            """
            Alias                 HuggingFace repo                         Size
            --help                mlx-community/Qwen3.6-27B-4bit           10 GB
            qwen3.6-27b           mlx-community/Qwen3.6-27B-4bit           10 GB
            phi-4                 https://evil.test/model?x=1              8 GB
            """
        )

        #expect(cached.count == 2)
        #expect(cached.first(where: { $0.0 == "qwen3.6-27b" })?.1 == "mlx-community/Qwen3.6-27B-4bit")
        #expect(cached.first(where: { $0.0 == "phi-4" })?.1 == nil)
    }

    @Test("parseAvailable drops audio-only aliases + the section-header phantom")
    func catalogParserDropsAudioAliases() {
        // Mirrors the real `rapid-mlx models` layout: a text section, then an
        // "Audio models (N aliases)" section whose rows carry an
        // `[audio:tts]` / `[audio:stt]` Kind column. The desktop has no audio
        // I/O surface and the shipped sidecar lacks `mlx-audio`, so these must
        // never reach the picker / Model Management. The section header's
        // first token ("Audio") also passes isSafeAlias and used to leak in
        // as a phantom alias.
        let available = ModelCatalog.parseAvailable(
            """
              Available models (3 aliases)
              ────────────────────────────────────────
              Alias                 Tools      Reasoning
              qwen3.6-27b           hermes     qwen3
              bonsai-1.7b-2bit      hermes     —
              gemma-4-26b-4bit      none       —

              Audio models (5 aliases)
              ────────────────────────────────────────
              Alias                 Kind        Family     HF id
              chatterbox            [audio:tts] chatterbox mlx-community/chatterbox-turbo-fp16
              kokoro                [audio:tts] kokoro     mlx-community/Kokoro-82M-bf16
              parakeet              [audio:stt] parakeet   mlx-community/parakeet-tdt-0.6b-v2
              whisper               [audio:stt] whisper    mlx-community/whisper-large-v3-mlx
              whisper-large-v3      [audio:stt] whisper    mlx-community/whisper-large-v3-mlx
            """
        )
        let aliases = available.map { $0.0 }
        // Text aliases survive.
        #expect(aliases == ["qwen3.6-27b", "bonsai-1.7b-2bit", "gemma-4-26b-4bit"])
        // No audio aliases, and no phantom "Audio" from the section header.
        for banned in ["Audio", "chatterbox", "kokoro", "parakeet", "whisper", "whisper-large-v3"] {
            #expect(!aliases.contains(banned), "audio/phantom alias leaked: \(banned)")
        }
    }

    @Test("ModelInfoCatalog sanitizes repo ids before UI link construction")
    func modelInfoSanitizesHuggingFaceRepo() {
        let safe = ModelInfoCatalog.info(for: "qwen3-8b", hfRepo: "mlx-community/Qwen3-8B-4bit")
        #expect(safe.hfRepo == "mlx-community/Qwen3-8B-4bit")

        let injected = ModelInfoCatalog.info(for: "qwen3-8b", hfRepo: "evil.test/model?token=secret")
        #expect(injected.hfRepo == nil)
    }

    @Test("ModelInfoCatalog bounds alias scanning for oversized user input")
    func modelInfoBoundsAliasScanning() {
        let alias = String(repeating: "x", count: ModelCatalog.maxAliasBytes + 40) + "qwen3.6-27b"
        let info = ModelInfoCatalog.info(for: alias, hfRepo: nil)
        #expect(info.family == "Unknown")
        #expect(info.contextWindow == nil)
    }

    @MainActor
    @Test("DownloadManager drops oversized child-output lines before progress parsing")
    func downloadManagerBoundsDecodedLines() {
        let mgr = DownloadManager()
        let oversized = String(repeating: "A", count: 9 * 1024)
        let data = Data("ok\n\(oversized)\n".utf8)
        #expect(mgr._testingDecodedLines(from: data) == ["ok"])
    }

    @Test("ModelCatalog caps retained stdout while still draining the child")
    func modelCatalogCapsSubprocessStdout() async throws {
        let script = try makeExecutableScript(
            """
            #!/bin/sh
            yes A | head -c 1200000
            """
        )

        let output = await ModelCatalog._testingRunRapidMlx(binary: script, args: ["models"])
        #expect(output.utf8.count == ModelCatalog.maxSubprocessStdoutBytes)
    }

    @Test("ModelCatalog terminates catalog subprocesses when the async task is cancelled")
    func modelCatalogCancelsSubprocess() async throws {
        let dir = try makeTemporaryDirectory()
        let marker = dir.appendingPathComponent("marker.txt")
        let script = try makeExecutableScript(
            """
            #!/bin/sh
            MARKER=\(shellQuote(marker.path))
            echo started > "$MARKER"
            trap 'echo terminated > "$MARKER"; exit 0' TERM
            while true; do sleep 1; done
            """
        )

        let task = Task {
            await ModelCatalog._testingRunRapidMlx(binary: script, args: ["models"])
        }
        #expect(await waitUntil(timeoutNanoseconds: 1_000_000_000) {
            FileManager.default.fileExists(atPath: marker.path)
        })

        task.cancel()
        _ = await task.value

        #expect(await waitUntil(timeoutNanoseconds: 1_000_000_000) {
            (try? String(contentsOf: marker, encoding: .utf8).contains("terminated")) == true
        })
    }
}

@MainActor
@Suite("DownloadProgress hardening")
struct DownloadProgressHardeningTests {
    @Test("Oversized progress lines are ignored before parser work")
    func oversizedProgressLineIgnored() {
        let progress = DownloadProgress()
        let line = String(repeating: "A", count: DownloadProgress.maxProgressLineBytes + 1)
        #expect(!progress.ingest(line))
        guard case .idle = progress.phase else {
            Issue.record("Expected .idle after oversized line, got \(progress.phase)")
            return
        }
    }

    @Test("Per-file parser still rejects missing tqdm bracket tail")
    func perFileRequiresBracketTail() {
        let progress = DownloadProgress()
        #expect(!progress.ingest("model.safetensors: 50%|█████| 1.0G/2.0G"))
        guard case .idle = progress.phase else {
            Issue.record("Expected .idle for malformed per-file line, got \(progress.phase)")
            return
        }
    }

    @Test("Fetching parser requires the canonical file-count header")
    func fetchingRequiresCanonicalHeader() {
        let progress = DownloadProgress()
        #expect(!progress.ingest("Fetching many files: 50%|█████| 1/2 [00:01<00:01, 1.0it/s]"))
        guard case .idle = progress.phase else {
            Issue.record("Expected .idle for malformed fetching line, got \(progress.phase)")
            return
        }
    }
}

private func makeTemporaryDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("rapid-download-catalog-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func makeExecutableScript(_ body: String) throws -> URL {
    let dir = try makeTemporaryDirectory()
    let script = dir.appendingPathComponent("fake-rapid-mlx.sh")
    try body.write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
    return script
}

private func shellQuote(_ raw: String) -> String {
    "'\(raw.replacingOccurrences(of: "'", with: "'\\''"))'"
}

private func waitUntil(
    timeoutNanoseconds: UInt64,
    _ condition: @escaping () -> Bool
) async -> Bool {
    let start = ContinuousClock.now
    let timeout = Duration.nanoseconds(Int64(timeoutNanoseconds))
    while start.duration(to: ContinuousClock.now) < timeout {
        if condition() { return true }
        try? await Task.sleep(nanoseconds: 25_000_000)
    }
    return condition()
}
