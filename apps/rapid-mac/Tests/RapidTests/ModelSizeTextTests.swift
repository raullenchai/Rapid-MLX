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
