import Foundation
import Testing
@testable import Rapid

@Suite("Model size text")
struct ModelSizeTextTests {
    @Test func producerFormats() {
        let examples: [(Int64, String, String)] = [
            (787_276_896, "750.8 MiB", "751 MB"),
            (3_083_520_685, "2.9 GiB", "2.9 GB"),
            (2_508_940_517, "2.34 GB", "2.3 GB"),
        ]
        for (bytes, raw, expected) in examples {
            #expect(ModelSizeText.display(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .binary)) == expected)
            #expect(ModelSizeText.display(raw) == expected)
            #expect(ModelSizeText.display(DownloadProgress.formatBytes(bytes)) == expected)
        }
    }

    @Test func invalidAndBlank() {
        #expect(ModelSizeText.display(nil) == nil)
        #expect(ModelSizeText.display("  ") == nil)
        #expect(ModelSizeText.display(" unknown ") == "unknown")
        #expect(ModelSizeText.bytes("-1 GB") == nil)
        #expect(ModelSizeText.bytes("1kib") == 1024)
    }

    @Test func localizedNumberShapes() {
        #expect(ModelSizeText.display("750,8 MB") == "751 MB")
        #expect(ModelSizeText.bytes("1,010 MB") == Int64(1010) * 1024 * 1024)
        #expect(ModelSizeText.bytes("1.023,3 MB") == Int64((1023.3 * 1024 * 1024).rounded()))
        #expect(ModelSizeText.bytes("1\u{00A0}010 MB") == Int64(1010) * 1024 * 1024)
        #expect(ModelSizeText.bytes("2.34 GB") == Int64((2.34 * 1024 * 1024 * 1024).rounded()))
        #expect(ModelSizeText.bytes("1,0,0 MB") == nil)
    }

    @Test func cachedCatalogUsesCanonicalSize() throws {
        let payload = #"{"cached":[{"alias":"a","repo":"org/a","subfolder":null,"size_bytes":787276896,"state":"ok","external":false}]}"#
        let rows = try #require(ModelCatalog.parseCachedJSON(payload))
        #expect(rows.first?.size == "751 MB")
    }

    @Test func catalogSizeWins() {
        let entry = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: "mlx-community/Qwen3-ASR-0.6B-5bit", sizeOnDisk: "750.8 MiB", cached: false)
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == "751 MB")
        #expect(SettingsModelManagementPanel.downloadSizeLabel(entry) == ModelSizeText.display(entry.sizeOnDisk))
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
