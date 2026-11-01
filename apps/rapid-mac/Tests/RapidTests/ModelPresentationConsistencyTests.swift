import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Model presentation consistency")
struct ModelPresentationConsistencyTests {
    @Test func namesAndCachedVariant() {
        #expect(ModelDisplayName.name(for: "qwen3.5-4b-4bit") == "Qwen 3.5 · 4B")
        #expect(ModelDisplayName.name(for: "whisper-large-v3-turbo") == "Whisper Large v3 Turbo")
        #expect(ModelDisplayName.name(for: "bonsai-1.7b-2bit") == "Bonsai · 1.7B · 2-bit")
        #expect(ModelDisplayName.name(for: "bonsai2-4b-2bit") == "Bonsai 2 · 4B · 2-bit")
        #expect(ModelDisplayName.name(for: "qwen3.5-9b-8bit") == "Qwen 3.5 · 9B · 8-bit")
        #expect(ModelDisplayName.name(for: "unknown-model") == "unknown-model")
        let entry = ModelEntry(alias: "bonsai-1.7b-2bit", hfRepo: nil, sizeOnDisk: nil, cached: true)
        let choice = QuickstartView.choice(forCachedModel: entry)
        #expect(choice.displayName == "Bonsai · 1.7B · 2-bit")
        #expect(QuickstartView.cachedRowSecondaryText(for: choice) == entry.alias)
    }

    @Test func variantNamesAndCatalogUniqueness() throws {
        #expect(ModelDisplayName.name(for: "qwen3-4b-thinking-2507-4bit") == "Qwen 3 · 4B · Thinking 2507")
        #expect(ModelDisplayName.name(for: "qwen3-4b-instruct-2507-4bit") == "Qwen 3 · 4B · Instruct 2507")
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("rapid_mlx/aliases.json")
        let entries = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [String: Any]])
        var names: [String: (String, String)] = [:]
        for alias in entries.keys.sorted() {
            guard let repo = entries[alias]?["hf_path"] as? String else { continue }
            let name = ModelDisplayName.name(for: alias)
            if let previous = names[name], previous.1 != repo {
                Issue.record("Name collision: \(previous.0) and \(alias) → \(name)")
            } else { names[name] = (alias, repo) }
        }
    }

    @Test func downloadSizeAcrossSurfacesAndMerge() throws {
        let alias = "qwen3.8-27b-4bit"
        let bytes: Int64 = 16_320_413_916
        let original = ModelEntry(alias: alias, hfRepo: "org/model", sizeOnDisk: nil, cached: false, sizeBytes: bytes, downloadSizeBytes: bytes)
        let row: CachedModelInventoryEntry = (alias: alias, hfRepo: "org/model", subfolder: nil, size: .bytes(16_320_439_756), discovery: nil, launchPath: nil)
        let cached = try #require(ModelCatalog.mergeAtomicAndCached(atomic: [original], cached: [row], excluded: []).first)
        #expect(cached.sizeBytes == 16_320_439_756)
        #expect(cached.downloadSizeBytes == bytes)
        let choice = QuickstartView.choice(forCatalogEntry: original)
        let expected = DownloadProgress.formatBytes(bytes)
        #expect(expected == "16.32 GB")
        for entry in [original, cached] {
            #expect(QuickstartView.rowSizeText(for: entry) == expected)
            #expect(QuickstartView.shortlistSizeText(for: choice, cached: entry.cached ? entry : nil, recommendedForPhysicalRAMGB: nil, catalogEntry: entry) == expected)
            let model = ShareComputeModel(catalogID: "test", alias: alias, title: "Test", detail: "")
            #expect(ShareComputeLocalModel(model: model, entry: entry).onDiskSize == expected)
        }
        let unknown = ModelEntry(alias: alias, hfRepo: nil, sizeOnDisk: nil, cached: false)
        #expect(QuickstartView.rowSizeText(for: unknown).hasPrefix("~"))
        #expect(QuickstartView.shortlistSizeText(for: choice, cached: nil, recommendedForPhysicalRAMGB: nil, catalogEntry: unknown).hasPrefix("~"))
    }

    @Test func cachedCatalogDownloadSizeFromJSON() throws {
        let json = #"""
        {"text":[],"atomic":{"snapshot":{"schema_version":2,"recommendation_policy_digests":[],
          "models":[{"schema_version":1,"registry_model_id":"legacy/hf/qwen","resolution_status":"unresolved","source":{"provider":"huggingface","repo_id":"mlx-community/Qwen3.5-9B-4bit"},"estimated_download_size_bytes":5977063311}],
          "aliases":[{"schema_version":2,"alias":"qwen3.5-9b-4bit","origin":"builtin","target":{"registry_model_id":"legacy/hf/qwen","resolution_status":"unresolved"},"capabilities":{"task_types":["text_generation"],"operation_modes":["chat"],"runtime_adapter":"mlx_lm","is_text_only":true},"availability":{"desktop":true,"cli":true,"server":true,"website":true},"default_execution_preset_id":null,"execution_presets":[]}]}}}
        """#
        var root = try #require(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        var envelope = try #require(root["atomic"] as? [String: Any])
        var snapshot = try #require(envelope["snapshot"] as? [String: Any])
        let digest = try #require(ModelCatalog.atomicCatalogDigest(snapshot))
        snapshot["catalog_digest"] = digest
        envelope["snapshot"] = snapshot
        envelope["shadow_report"] = ["equivalent": true, "catalog_digest": digest]
        root["atomic"] = envelope
        let payload = String(decoding: try JSONSerialization.data(withJSONObject: root), as: UTF8.self)
        let atomic = try #require(ModelCatalog.parseAtomicModelEntriesJSON(payload))
        let available = try #require(ModelCatalog.parseAvailableJSON(payload))
        let cached = try #require(ModelCatalog.parseCachedJSON(#"""
        {"cached":[{"alias":"qwen3.5-9b-4bit","repo":"mlx-community/Qwen3.5-9B-4bit","subfolder":null,"size_bytes":6010000000,"state":"ok","external":false}]}
        """#))
        let merged = try #require(ModelCatalog.mergeAvailableAndCached(
            available: available.entries, cached: cached, excluded: available.excluded, atomic: atomic
        ).first)
        #expect(merged.sizeBytes == 6_010_000_000)
        #expect(merged.downloadSizeBytes == 5_977_063_311)
        #expect(merged.displayedModelSize?.displayText == "5.98 GB")
    }

    @Test func cachedDownloadSizeMatchesExactArtifact() throws {
        let atomic = [ModelEntry(alias: "catalog", hfRepo: "org/model", sizeOnDisk: nil,
                                 cached: false, downloadSizeBytes: 5_977_063_311,
                                 sourceSubfolder: "4bit")]
        let rows: [CachedModelInventoryEntry] = [
            ("custom", "org/model", "4bit", .bytes(6_010_000_000), nil, nil),
            ("other", "org/model", "8bit", .bytes(9_000_000_000), nil, nil),
            ("unknown", "org/unknown", nil, .bytes(1_000_000_000), nil, nil),
        ]
        let merged = ModelCatalog.mergeAtomicAndCached(atomic: atomic, cached: rows, excluded: [])
        #expect(merged.first { $0.alias == "custom" }?.downloadSizeBytes == 5_977_063_311)
        #expect(merged.first { $0.alias == "other" }?.downloadSizeBytes == nil)
        #expect(merged.first { $0.alias == "unknown" }?.downloadSizeBytes == nil)
        #expect(merged.first { $0.alias == "unknown" }?.displayedModelSize?.displayText == "1 GB")
    }

    @Test func settingsAudioDeduplication() {
        let repo = "mlx-community/whisper-large-v3-mlx"
        let rows = ["whisper", "whisper-1", "whisper-large-v3", "whisper-small"].map {
            ModelEntry(alias: $0, hfRepo: $0 == "whisper-small" ? "org/small" : repo, sizeOnDisk: nil, cached: true, kind: .audio)
        }
        let result = AudioViewModel.stablyDeduplicatedTranscriptionModels(rows)
        #expect(result.count == 2)
        #expect(result.map(\.alias) == ["whisper-large-v3", "whisper-small"])
    }

    @Test func audioDeduplicationKeepsDistinctSubfolders() {
        let subfolders: [String] = ["a", "b"]
        let rows = subfolders.map {
            ModelEntry(alias: "audio-\($0)", hfRepo: "org/audio", sizeOnDisk: nil,
                       cached: true, kind: .audio, sourceSubfolder: $0)
        }
        #expect(AudioViewModel.stablyDeduplicatedTranscriptionModels(rows).map(\.alias) == ["audio-a", "audio-b"])
        #expect(AudioViewModel.deduplicatedTranscriptionModels(rows).map(\.alias) == ["audio-a", "audio-b"])
    }
    @Test func cachedHeadlineCountsAllChooserRows() {
        func entry(_ alias: String) -> ModelEntry {
            ModelEntry(alias: alias, hfRepo: "org/\(alias)", sizeOnDisk: nil, cached: true)
        }
        let initial = ["qwen3-1b-4bit", "qwen3-1b-2bit", "qwen3-1b-3bit", "qwen3-1b-6bit", "qwen3-1b-8bit", "qwen3-1b-16bit", "qwen3-1b-bf16"].map { entry($0) }
        let first = QuickstartView.shortlist(catalog: initial, selection: "qwen3-1b-4bit")
        let lanes = first.starters + first.recommended + first.lowMemory + first.tradeUps
        let distinct = Array(Set(lanes.map(\.alias))).sorted()
        #expect(distinct.count >= 4)
        let rows = initial + distinct.prefix(4).map { entry($0) }
        let list = QuickstartView.shortlist(catalog: rows, selection: "qwen3-1b-4bit")
        #expect(list.onThisMacCount == 11)
        #expect(QuickstartView.selectionNarrative(alias: "qwen3-1b-4bit", cachedModels: rows, comparableTradeUps: list.tradeUps, cachedRowCount: list.onThisMacCount).title == "11 are already\nhere.")
        let two = Array(initial.prefix(2))
        let small = QuickstartView.shortlist(catalog: two, selection: "qwen3-1b-4bit")
        #expect(small.onThisMacCount == 2)
        #expect(QuickstartView.selectionNarrative(alias: "qwen3-1b-4bit", cachedModels: two, comparableTradeUps: small.tradeUps, cachedRowCount: small.onThisMacCount).title == "Two are already\nhere.")
    }

}
