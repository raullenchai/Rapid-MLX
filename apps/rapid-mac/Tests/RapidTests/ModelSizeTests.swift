import Foundation
import Testing
@testable import Rapid

@Suite("Model size")
struct ModelSizeTests {
    @Test func byteFormatting() {
        let bytes: Int64 = 1_500_000_000
        let expected = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
        #expect(ModelSize.bytes(bytes).displayText == expected)
        #expect(ModelSize.legacy("1.5 GB")?.displayText == expected)
        #expect(DownloadProgress.formatBytes(bytes) == expected)
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowsNonnumericFormatting = false
        #expect(DownloadProgress.formatBytes(-1) == formatter.string(fromByteCount: 0))
    }

    @Test func legacyTextParsing() {
        #expect(ModelSize.legacy("338.9 MiB")?.displayText == ModelSize.bytes(355_365_273).displayText)
        let gibBytes = Int64(4.3 * Double(1 << 30))
        #expect(ModelSize.legacy("4.3 GiB")?.displayText == ByteCountFormatter.string(fromByteCount: gibBytes, countStyle: .file))
        #expect(ModelSize.legacy("1.5 GB")?.byteCount == 1_500_000_000)
        #expect(ModelSize.parseBytes("1,010 MB") == nil)
        #expect(ModelSize.legacy("about one gig")?.displayText == "about one gig")
        #expect(ModelSize.legacy("about one gig")?.byteCount == nil)
        #expect(ModelSize.legacy("  ") == nil)
        #expect(ModelSize.legacy(nil) == nil)
        #expect(ModelSize.legacy("1e99 GiB")?.byteCount == nil)
    }

    @Test func fixedFormatParsing() {
        #expect(ModelSize.parseBytes("1,07 GB") == nil)
        #expect(ModelSize.parseBytes("1.07\u{00a0}GB") == 1_070_000_000)
        #expect(ModelSize.parseBytes("1.234,5 MB") == nil)
    }

    @Test func nonPositiveNumericSizeFallsBackToLegacy() {
        for bytes in [Int64(0), -1] {
            let entry = ModelEntry(alias: "a", hfRepo: nil, sizeOnDisk: "2.0 GiB", cached: true, sizeBytes: bytes)
            #expect(entry.size?.byteCount == 2_147_483_648)
        }
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

    @Test func cachedInventoryWithoutSizeDoesNotUseEstimate() {
        let entry = ModelEntry(alias: "a", hfRepo: "org/a", sizeOnDisk: "751 MB", cached: false, sizeBytes: 787_276_896)
        let row: CachedModelInventoryEntry = (alias: "a", hfRepo: "org/a", subfolder: nil, size: nil, discovery: nil, launchPath: nil)
        let merged = ModelCatalog.mergeAtomicAndCached(atomic: [entry], cached: [row], excluded: [])
        #expect(merged.first?.cached == true)
        #expect(merged.first?.sizeBytes == nil)
        #expect(merged.first?.sizeOnDisk == nil)
    }

    @Test func pickerAndSettingsUseCatalogSize() {
        let entry = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: "mlx-community/Qwen3-ASR-0.6B-5bit", sizeOnDisk: "751 MB", cached: false, sizeBytes: 787_276_896)
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == ByteCountFormatter.string(fromByteCount: 787_276_896, countStyle: .file))
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == entry.size?.displayText)
        let legacy = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: nil, sizeOnDisk: "750.8 MiB", cached: false)
        #expect(SettingsModelManagementPanel.downloadSizeLabel(legacy) == legacy.size?.displayText)
    }

    @Test @MainActor func readinessDisplayName() {
        let readiness = ModelReadiness.needsDownload(alias: "qwen3-asr-0.6b", sizeText: "750.8 MB")
        let headline = readiness.headline(displayName: "Qwen3 ASR 0.6B")
        #expect(headline.contains("Qwen3 ASR 0.6B"))
        #expect(!headline.contains("qwen3-asr-0.6b"))
        #expect(readiness.headline(displayName: nil).contains(ModelDisplayName.name(for: "qwen3-asr-0.6b")))
        #expect(readiness.action == .download(alias: "qwen3-asr-0.6b"))
        #expect(AudioViewModel.transcriptionDetails(alias: "qwen3-asr-0.6b", family: "qwen3_asr").displayName == "Qwen3 ASR 0.6B")
    }
}
