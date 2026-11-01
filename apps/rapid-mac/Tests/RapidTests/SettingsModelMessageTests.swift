import Foundation
import Testing
@testable import Rapid

@Suite("SettingsModel catalog totals and live context labels")
struct SettingsModelMessageTests {
    @Test func downloadUsesCatalogSize() throws {
        let exact = ModelEntry(alias: "qwen3-asr-0.6b", hfRepo: "org/model", sizeOnDisk: "0.3 GB", cached: false, sizeBytes: 787_276_896)
        #expect(SettingsModelManagementPanel.downloadTotalBytes(exact) == 787_276_896)
        let legacy = ModelEntry(alias: "model", hfRepo: nil, sizeOnDisk: "1.5 GB", cached: false)
        #expect(SettingsModelManagementPanel.downloadTotalBytes(legacy) == 1_500_000_000)
        let unknown = ModelEntry(alias: "model", hfRepo: nil, sizeOnDisk: nil, cached: false)
        #expect(SettingsModelManagementPanel.downloadTotalBytes(unknown) == nil)

        // Check the UI actions use the tested helper, including both retry paths.
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/Rapid/UI/SettingsModelManagementPanel.swift"), encoding: .utf8)
        let starts = source.components(separatedBy: "downloads.startDownload(").dropFirst()
        #expect(starts.count == 4)
        for start in starts {
            #expect(start.prefix(180).contains("totalBytes: Self.downloadTotalBytes(entry)"))
        }
    }

    @Test func contextUsesMatchingLiveProfile() {
        let alias = "qwen3.6-35b-4bit"
        #expect(SettingsModelManagementPanel.contextLabel(alias, profile: ServerModelProfile(id: alias, contextWindow: 65_536)) == "64k")
        #expect(SettingsModelManagementPanel.contextLabel(alias, profile: ServerModelProfile(id: alias.uppercased(), contextWindow: 12_345)) == "12345")
        let fallback = SettingsModelManagementPanel.contextLabel(alias, profile: nil)
        #expect(fallback != nil)
        #expect(SettingsModelManagementPanel.contextLabel(alias, profile: ServerModelProfile(id: "other", contextWindow: 65_536)) == fallback)
        #expect(SettingsModelManagementPanel.contextLabel("unrecognized-model", profile: nil) == nil)
    }
}
