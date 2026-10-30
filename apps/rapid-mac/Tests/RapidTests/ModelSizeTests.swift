import Foundation
import Testing
@testable import Rapid

@Suite("Model size")
struct ModelSizeTests {
    @Test func byteFormatting() {
        #expect(ModelSize.bytes(787_276_896).displayText == DownloadProgress.formatBytes(787_276_896))
        #expect(ModelSize.bytes(787_276_896).displayText == "751 MB")
        #expect(ModelSize.bytes(3_083_520_685).displayText == "2.9 GB")
    }

    @Test func legacyTextIsNeverParsed() {
        #expect(ModelSize.legacy(" 2.9 GiB ")?.displayText == "2.9 GiB")
        #expect(ModelSize.legacy("1,010 MB")?.displayText == "1,010 MB")
        #expect(ModelSize.legacy("1,010 MB")?.byteCount == nil)
        #expect(ModelSize.legacy("  ") == nil)
        #expect(ModelSize.legacy(nil) == nil)
    }

    @Test func cachedInventoryAndMergePreserveBytes() throws {
        let payload = #"{"cached":[{"alias":"a","repo":"org/a","subfolder":null,"size_bytes":787276896,"state":"ok","external":false}]}"#
        let rows = try #require(ModelCatalog.parseCachedJSON(payload))
        #expect(rows.first?.size == .bytes(787_276_896))
        let mergePayload = #"{"cached":[{"alias":"a","repo":"org/a","subfolder":null,"size_bytes":2000000000,"state":"ok","external":false}]}"#
        let cached = try #require(ModelCatalog.parseCachedJSON(mergePayload))
        #expect(cached.first?.size == .bytes(2_000_000_000))
        let entry = ModelEntry(alias: "a", hfRepo: "org/a", sizeOnDisk: nil, cached: false)
        let merged = ModelCatalog.mergeAtomicAndCached(atomic: [entry], cached: cached, excluded: [])
        #expect(merged.first?.cached == true)
        #expect(merged.first?.sizeBytes == 2_000_000_000)
    }

    @Test func pickerAndSettingsUseCatalogSize() {
        let entry = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: "mlx-community/Qwen3-ASR-0.6B-5bit", sizeOnDisk: "751 MB", cached: false, sizeBytes: 787_276_896)
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == "751 MB")
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == entry.size?.displayText)
        let legacy = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: nil, sizeOnDisk: "750.8 MiB", cached: false)
        #expect(SettingsModelManagementPanel.downloadSizeLabel(legacy) == "750.8 MiB")
    }

    @Test @MainActor func readinessDisplayName() {
        let readiness = ModelReadiness.needsDownload(alias: "qwen3-asr-0.6b", sizeText: "750.8 MB")
        let headline = readiness.headline(displayName: "Qwen3 ASR 0.6B")
        #expect(headline.contains("Qwen3 ASR 0.6B"))
        #expect(!headline.contains("qwen3-asr-0.6b"))
        #expect(readiness.headline(displayName: nil).contains("qwen3-asr-0.6b"))
        #expect(readiness.action == .download(alias: "qwen3-asr-0.6b"))
        #expect(AudioViewModel.transcriptionDetails(alias: "qwen3-asr-0.6b", family: "qwen3_asr").displayName == "Qwen3 ASR 0.6B")
    }
}
