import Foundation
import Testing
@testable import Rapid

@Suite("Download/catalog hardening")
struct DownloadCatalogHardeningTests {
    @Test("Speculative presets are parsed from the alias profile table")
    func speculativePresetParsing() {
        let output = """
          Alias                  Size       Tools            Reasoning    Spec-Decode Suffix Tier DFlash DDTree Preset
          qwen3.8-27b-4bit       15.2 GiB   hermes           qwen3        ✓ MTP       n/a         —       —       MTP@rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3
          llama3-3b-4bit         1.8 GiB    llama3_json      —            ✓           unknown     —       —       Suffix
          qwen3.6-27b-4bit       15.1 GiB   hermes           qwen3        ✗ hybrid    n/a         —       —       —
        """
        let capabilities = ModelCatalog.parseSpeculativeCapabilities(output)
        #expect(capabilities["qwen3.8-27b-4bit"]?.method == .mtp)
        #expect(capabilities["qwen3.8-27b-4bit"]?.model == "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX")
        #expect(capabilities["qwen3.8-27b-4bit"]?.tokens == 3)
        #expect(capabilities["llama3-3b-4bit"]?.method == .suffix)
        #expect(capabilities["qwen3.6-27b-4bit"] == nil)

        let legacyOutput = """
          Alias                  Tools            Reasoning
          model-ending-suffix    hermes           Suffix
        """
        #expect(ModelCatalog.parseSpeculativeCapabilities(legacyOutput).isEmpty)
    }

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

    @Test("parseAvailable drops video-gen aliases + the section-header phantom")
    func catalogParserDropsVideoAliases() {
        // Same shape as the audio case. A `video-gen` model has no
        // tokenizer and no `stream_chat`, so it can never answer a chat
        // request — offering one costs the user up to 64 GiB of download
        // and ends at "Couldn't start X. Try again", forever (#1603).
        let output =
            """
              Available models (2 aliases)
              ────────────────────────────────────────
              Alias                 Tools      Reasoning
              qwen3.6-27b           hermes     qwen3
              bonsai-1.7b-2bit      hermes     —

              Video models (3 aliases)
              ────────────────────────────────────────
              Alias                 Size       Kind        HF id
              cogvideox-fun-5b-q4   13.3 GiB   [video:gen] dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4
              ltx-2.3-mlx-q4        21.2 GiB   [video:gen] notapalindrome/ltx23-mlx-av-q4
              wan2.2-t2v-a14b-bf16  64.3 GiB   [video:gen] rickylin20260522/Wan2.2-T2V-A14B-mlx
            """
        let aliases = ModelCatalog.parseAvailable(output).map { $0.0 }
        #expect(aliases == ["qwen3.6-27b", "bonsai-1.7b-2bit"])
        for banned in [
            "Video", "cogvideox-fun-5b-q4", "ltx-2.3-mlx-q4", "wan2.2-t2v-a14b-bf16",
        ] {
            #expect(!aliases.contains(banned), "video/phantom alias leaked: \(banned)")
        }

        // The same rows must be reported as *deliberately* excluded, so
        // `load` can refuse to re-admit them off `rapid-mlx ls` — which
        // carries no modality tag of its own.
        let excluded = ModelCatalog.parseExcludedAliases(output)
        #expect(excluded == ["cogvideox-fun-5b-q4", "ltx-2.3-mlx-q4", "wan2.2-t2v-a14b-bf16"])
        #expect(!excluded.contains("qwen3.6-27b"))
        #expect(!excluded.contains("Video"))
    }

    @Test("parseExcludedAliases covers audio too, and ignores untagged noise")
    func excludedAliasesCoversAudioAndIgnoresNoise() {
        let excluded = ModelCatalog.parseExcludedAliases(
            """
              Available models (1 aliases)
              ────────────────────────────────────────
              qwen3.6-27b           hermes     qwen3
              Loading model with BatchedEngine: qwen3.6-27b

              Audio models (2 aliases)
              ────────────────────────────────────────
              kokoro                [audio:tts] kokoro     mlx-community/Kokoro-82M-bf16
              whisper               [audio:stt] whisper    mlx-community/whisper-large-v3-mlx
            """
        )
        #expect(excluded == ["kokoro", "whisper"])
        // Only explicitly tagged rows count — a banner line or a plain
        // text row must never suppress a real model.
        #expect(!excluded.contains("qwen3.6-27b"))
        #expect(!excluded.contains("Loading"))
    }

    @Test("a withheld alias is not re-admitted through the cached listing")
    func excludedAliasesAreNotReadmittedFromCache() {
        // The decisive half of #1603. `rapid-mlx ls` carries no modality
        // tag, and the merge re-admits any cached alias with no row in
        // `models` — which is exactly the state a correctly-filtered
        // video model is in. Without the exclusion check the fix is
        // one-sided and the model returns through the side door.
        let available = [("qwen3.6-27b", String?.none)]
        let cached: [(String, String?, String?)] = [
            ("qwen3.6-27b", "mlx-community/Qwen3.6-27B-4bit", "16 GiB"),
            ("cogvideox-fun-5b-q4", "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4", "13.3 GiB"),
            ("kokoro", "mlx-community/Kokoro-82M-bf16", "0.3 GiB"),
            // A hand-pinned text alias with no `models` row must still be
            // surfaced — the re-admission path exists for this case.
            ("my-custom-llama", "me/llama-mlx", "8 GiB"),
        ]

        let merged = ModelCatalog.mergeAvailableAndCached(
            available: available,
            cached: cached,
            excluded: ["cogvideox-fun-5b-q4", "kokoro"]
        )
        let aliases = merged.map(\.alias)
        #expect(aliases.contains("qwen3.6-27b"))
        #expect(aliases.contains("my-custom-llama"))
        #expect(!aliases.contains("cogvideox-fun-5b-q4"))
        #expect(!aliases.contains("kokoro"))

        // Guard the guard: with no exclusions the same input re-admits
        // them, so the assertions above are testing the condition and not
        // some unrelated filter.
        let unfiltered = ModelCatalog.mergeAvailableAndCached(
            available: available,
            cached: cached,
            excluded: []
        )
        #expect(unfiltered.map(\.alias).contains("cogvideox-fun-5b-q4"))
    }

    @Test("Kind-tag matching requires a real column token, not a substring")
    func nonChatKindTagIsColumnScoped() {
        #expect(ModelCatalog.hasNonChatKindTag("kokoro [audio:tts] kokoro mlx/Kokoro"))
        #expect(ModelCatalog.hasNonChatKindTag("cog 13.3 GiB [video:gen] org/repo"))
        // Image generation joined audio and video as a non-chat kind in
        // #1705, which is what keeps `flux2-klein-4b` and `z-image-turbo`
        // out of the chat picker while the Images tab serves them. This
        // assertion asserted the opposite until that landed.
        #expect(ModelCatalog.hasNonChatKindTag("flux 4.3 GiB [image:gen] Runpod/FLUX.2-klein-4B-mflux-4bit"))
        #expect(ModelCatalog.hasNonChatKindTag("odd [image:gen] org/repo"))
        // A model whose repo or description merely contains the
        // characters must not vanish from the catalog.
        #expect(!ModelCatalog.hasNonChatKindTag("weird-model hermes org/repo[audio:tts]x"))
        #expect(!ModelCatalog.hasNonChatKindTag("weird-model hermes org/audio:tts"))
        #expect(!ModelCatalog.hasNonChatKindTag("qwen3.6-27b hermes qwen3 ✓ avoid — —"))
        #expect(!ModelCatalog.hasNonChatKindTag("odd [audio:] org/repo"))
        #expect(!ModelCatalog.hasNonChatKindTag("odd [image:] org/repo"))
        // The column-scoping property the test is named for still holds
        // for the new kind: a bare substring must not disqualify a row.
        #expect(!ModelCatalog.hasNonChatKindTag("weird-model hermes org/repo[image:gen]x"))
    }

    @Test("parseAvailable drops the human-readable size footer")
    func catalogParserDropsSizeFooter() {
        let available = ModelCatalog.parseAvailable(
            """
              Available models (1 aliases)
              ────────────────────────────────────────
              Alias                 Size       Tools
              lfm2.5-1b-4bit        563 MiB    —
              ────────────────────────────────────────
              Size is an approximate download footprint (weight+tokenizer); “—” = unknown. The exact size is confirmed at pull time.
            """
        )

        #expect(available.map { $0.0 } == ["lfm2.5-1b-4bit"])
        #expect(!available.contains(where: { $0.0 == "Size" }))
    }

    @Test("empty-cache notice never parses as a phantom \"No\" model (#1918)")
    func emptyCacheNoticeIsNotAModel() {
        // `rapid-mlx ls` prints this in place of a "Cached models" table when
        // the disk is cold. Its single-space prose used to tokenize (parseCached
        // splits on any whitespace) into a selectable phantom — alias "No",
        // repo "models", size "cached yet." — that dead-ended model start.
        let notice = """

              No models cached yet. Run 'rapid-mlx pull <alias>' or 'rapid-mlx chat <alias>' to download one.

            """
        #expect(ModelCatalog.parseCached(notice).isEmpty)
        #expect(ModelCatalog.parseAvailable(notice).isEmpty)
    }

    @Test("a genuine alias literally named \"No\" is not blacklisted (#1918)")
    func genuineNoAliasIsNotBlacklisted() {
        // The fix matches the full "No models cached yet" notice, never the
        // bare word "No", so a real model whose alias happens to be "No"
        // (row = "No" + 2+ spaces + repo) must still reach the catalog.
        let cached = ModelCatalog.parseCached(
            """
              Cached models (1 on disk)
              ────────────────────────────────────────
              Alias                 HuggingFace repo                         Size
              No                    mlx-community/No-Model-4bit               2 GB
            """
        )
        #expect(cached.map { $0.0 } == ["No"])
        #expect(cached.first?.1 == "mlx-community/No-Model-4bit")

        let available = ModelCatalog.parseAvailable(
            """
              Available models (1 aliases)
              ────────────────────────────────────────
              Alias                 Family
              No                    experimental
            """
        )
        #expect(available.map { $0.0 } == ["No"])
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

    @Test("ModelCatalog internal probes disable engine telemetry")
    func modelCatalogProbeDisablesTelemetry() async throws {
        let script = try makeExecutableScript(
            """
            #!/bin/sh
            printf '%s' "${DO_NOT_TRACK-unset}"
            """
        )

        for subcommand in ["models", "ls", "info"] {
            let output = await ModelCatalog._testingRunRapidMlx(
                binary: script,
                args: [subcommand, "fixture-alias"]
            )
            #expect(output == "1", "\(subcommand) probe inherited engine telemetry")
        }
    }

    @Test("ModelCatalog probe environment preserves caller paths but overrides telemetry")
    func modelCatalogProbeEnvironmentComposition() throws {
        let selected = URL(fileURLWithPath: "/Volumes/models")
        let env = ModelCatalog.probeEnvironment(
            ambient: [
                "DO_NOT_TRACK": "0",
                "KEEP_ME": "yes",
                ModelCatalog.extraModelRootsEnvKey: "[\"/ambient\"]",
            ],
            hubCacheOverride: selected
        )

        #expect(env["DO_NOT_TRACK"] == "1")
        #expect(env["KEEP_ME"] == "yes")
        #expect(env["HF_HUB_CACHE"] == selected.path)
        let encodedRoots = try #require(env[ModelCatalog.extraModelRootsEnvKey])
        let roots = try JSONDecoder().decode([String].self, from: Data(encodedRoots.utf8))
        #expect(roots == ["/ambient", "/Volumes/models"])
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
