import Foundation
import Testing
@testable import Rapid

@Suite("Offline decode references — exact chip/model/default configuration")
struct DecodeExpectationsTests {
    struct Golden: Decodable {
        let alias: String
        let chip: String
        let ramGiB: Double
        let repoID: String?
        let summary: String
        let compact: String
        enum CodingKeys: String, CodingKey {
            case alias, chip, summary, compact
            case ramGiB = "ram_gib", repoID = "repo_id"
        }
    }

    @Test("Python and Desktop consume the same physical snapshot and golden vectors")
    func sharedGoldenVectors() throws {
        let snapshot = try #require(DecodeExpectations.snapshot)
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/decode_expectations_golden.json"))
        let rows = try JSONDecoder().decode([Golden].self, from: data)
        for row in rows {
            #expect(DecodeExpectations.summary(alias: row.alias, chip: row.chip, ramGiB: row.ramGiB, repoID: row.repoID, snapshot: snapshot) == row.summary)
            #expect(DecodeExpectations.summary(alias: row.alias, chip: row.chip, ramGiB: row.ramGiB, repoID: row.repoID, snapshot: snapshot, compact: true) == row.compact)
        }
    }

    @Test("Unknown repository, invalid RAM and absent bundle never claim a speed")
    func unknownIdentity() {
        for ram in [Double.nan, 0, -1] {
            #expect(DecodeExpectations.summary(alias: "qwen3.5-4b-4bit", chip: "M4", ramGiB: ram, repoID: "mlx-community/Qwen3.5-4B-MLX-4bit", snapshot: DecodeExpectations.snapshot) == "Expected decode range unavailable")
        }
        #expect(DecodeExpectations.summary(alias: "qwen3.5-4b-4bit", chip: "M4", ramGiB: 32, repoID: nil, snapshot: DecodeExpectations.snapshot) == "Expected decode range unavailable")
        #expect(DecodeExpectations.summary(alias: "qwen3.5-4b-4bit", chip: "M4", ramGiB: 32, repoID: "mlx-community/Qwen3.5-4B-MLX-4bit", snapshot: nil) == "Expected decode range unavailable")
    }

    @Test("Recommended surfaces replace unqualified legacy TPS with chip-aware references")
    @MainActor func recommendationSurfaces() throws {
        let pick = try #require(RAMBucketedDefault.picks(forPhysicalRAMGB: 32).first)
        let title = ModelPickerBar.recommendedTagline(pick: pick, isPrimary: true, repoID: "custom/foreign")
        #expect(title.contains("Expected decode range unavailable"))
        #expect(!title.contains("~41 tok/s"))
        let settings = SettingsModelManagementPanel.pickStatsLine(pick, repoID: "custom/foreign")
        #expect(settings.contains("Expected decode range unavailable"))
        #expect(settings.contains("memory"))
    }
}
