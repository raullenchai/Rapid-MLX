import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Desktop behavior regressions")
struct DesktopBehaviorRegressionTests {
    @Test func unreadableCredentialsOfferSettingsRecovery() throws {
        for error in [ShareComputeLedgerError.keychainUnavailable, .providerKeyCorrupted] {
            #expect(ShareComputeLedgerState.unavailable(error).needsKey)
        }
        #expect(!ShareComputeLedgerState.unavailable(.rateLimited).needsKey)
        // Pin the view's wiring as well as the state: the recovery flag existed
        // before the unavailable note actually passed it to the Settings action.
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent(
            "Sources/Rapid/ShareCompute/ShareComputeCreditsView.swift"), encoding: .utf8)
        let unavailable = try #require(source.split(separator: "\n").first { $0.contains("case .unavailable(let error):") })
        #expect(unavailable.contains("action: state.needsKey"))
    }

    @Test func cancelledAudioRefreshPreservesCatalogAndSelections() async {
        let model = AudioViewModel(server: ServerManager(
            testingState: .idle, binaryPath: URL(fileURLWithPath: "/unused/rapid-mlx")))
        let original = ModelEntry(alias: "qwen3-tts-4bit", hfRepo: "org/voice", sizeOnDisk: nil,
                                  cached: true, kind: .audio, audioCapability: .speech, audioFamily: "qwen3_tts")
        model.audioModels = [original]
        model.catalogLoaded = true
        model.selectedSpeechAlias = original.alias
        model.selectedTranscriptionAlias = "whisper-large-v3"
        model.selectedVoice = "Ryan"
        // Suspend inside the provider so cancellation arrives after refresh's
        // initial guard, while the catalog probe is in flight.
        var probe: CheckedContinuation<[ModelEntry], Never>?
        let task = Task { @MainActor in
            await model.refreshCatalog(entriesProvider: { _ in
                await withCheckedContinuation { probe = $0 }
            })
        }
        while probe == nil { await Task.yield() }
        task.cancel()
        probe?.resume(returning: [])
        await task.value
        #expect(model.audioModels == [original])
        #expect(model.catalogLoaded)
        #expect(model.selectedSpeechAlias == original.alias)
        #expect(model.selectedTranscriptionAlias == "whisper-large-v3")
        #expect(model.selectedVoice == "Ryan")
    }

    @Test func settingsSearchMatchesDeduplicatedAliases() {
        let rows = ["whisper-1", "whisper-large-v3", "whisper"].map {
            ModelEntry(alias: $0, hfRepo: "org/whisper-large-v3", sizeOnDisk: nil,
                       cached: false, kind: .audio, audioCapability: .transcription, audioFamily: "whisper")
        }
        for ordered in [rows, rows.reversed().map { $0 }] {
            let entries = AudioViewModel.stablyDeduplicatedTranscriptionModels(ordered)
            #expect(entries.count == 1)
            for query in ["whisper-1", "WHISPER-1", "Whisper Large v3"] {
                let filtered = ModelCacheActions.filter(entries, by: .all, query: query)
                #expect(filtered.map(\.alias) == ["whisper-large-v3"])
            }
        }
    }

    @Test func cachedInventoryKeepsValidRows() throws {
        let good = #"{"alias":"valid-4bit","repo":"org/model","size_bytes":2048,"state":"ok","external":false}"#
        for bad in [
            "42",
            #"{"alias":null,"repo":"org/partial","size_bytes":0,"state":"incomplete","external":false}"#,
            #"{"alias":"../bad","repo":"org/model","size_bytes":2048,"state":"ok","external":false}"#,
            #"{"alias":"bad","repo":"org/model","size_bytes":true,"state":"ok","external":false}"#
        ] {
            let entries = try #require(ModelCatalog.parseCachedJSON("{\"cached\":[\(bad),\(good)]}"))
            #expect(entries.map(\.alias) == ["valid-4bit"])
            #expect(ModelCatalog.parseCachedJSON("{\"cached\":[\(bad)]}") == nil)
        }
        #expect(ModelCatalog.parseCachedJSON("broken") == nil)
        #expect(ModelCatalog.parseCachedJSON(#"{"cached":[]}"#)?.isEmpty == true)
    }

    @Test func diskOverviewWaitsForFirstFullRefresh() {
        let seed = [ModelEntry(alias: "chat-4b", hfRepo: nil, sizeOnDisk: "2 GiB", cached: true)]
        var state = ModelManagementDiskOverviewState()
        #expect(!state.hasCompletedRefresh)
        #expect(state.entries(in: seed) == nil)
        state.hasCompletedRefresh = true
        #expect(state.entries(in: seed) == seed)
        #expect(state.entries(in: []) == [])
    }

    @Test func sizelessAliasesHaveFriendlyNames() {
        #expect(ModelDisplayName.name(for: "deepseek-v4-flash-2bit") == "DeepSeek V4 · Flash · 2-bit")
        #expect(ModelDisplayName.name(for: "deepseek-v4-flash-4bit") == "DeepSeek V4 · Flash")
        #expect(ModelDisplayName.name(for: "unknown-model-4bit") == "unknown-model-4bit")
        #expect(ModelDisplayName.name(for: "Bonsai Image Large Model for Detailed Landscape Generation") == "Bonsai Image Large Model for Detailed Landscape Generation")
        #expect(ModelDisplayName.name(for: "deepseek-v4-flash") == "deepseek-v4-flash")
        for alias in ["fake-qwen3-tts", "qwen3-tts", "qwen3-tts-4bit", "qwen3-tts-clone",
                      "fake-deepseek-v4-flash-2bit", "fake-qwen3"] {
            #expect(ModelDisplayName.name(for: alias) == alias)
        }
        for token in ["tts", "stt", "asr", "speech", "voice", "audio"] {
            let alias = "deepseek-v4-\(token)"
            #expect(ModelDisplayName.name(for: alias) == alias)
        }
    }

    @Test func receiptsResolveCurrentTitleWithSnapshotFallback() throws {
        let current = try #require(ShareComputeModel.supported.first)
        func receipt(_ catalogID: String) -> ShareComputeReceipt {
            .init(id: "QS-TEST", catalogID: catalogID, modelTitle: "Old model title", worker: "test",
                  nodeID: nil, startedAt: .distantPast, endedAt: .distantPast,
                  rewardStatus: .available, restoreStatus: .notNeeded)
        }
        #expect(receipt(current.catalogID).displayModelTitle == current.title)
        #expect(receipt("unknown-id").displayModelTitle == "Old model title")
        #expect(receipt(current.catalogID).modelTitle == "Old model title")
    }

    @Test func displayNamesOmitRedundantSizeTokens() {
        #expect(ModelDisplayName.name(for: "gemma-4-e4b-4bit") == "Gemma 4 · E4B")
        #expect(ModelDisplayName.name(for: "gemma-4-e2b-4bit") == "Gemma 4 · E2B")
        #expect(ModelDisplayName.name(for: "gemma-4-12b-4bit") == "Gemma 4 · 12B")
        #expect(ModelDisplayName.name(for: "gemma-4-12b-e2b-4bit") == "Gemma 4 · 12B · E2B")
    }
}
