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
        let cohortConfigKeyOverride: String?
        enum CodingKeys: String, CodingKey {
            case alias, chip, summary, compact
            case ramGiB = "ram_gib", repoID = "repo_id"
            case cohortConfigKeyOverride = "cohort_config_key_override"
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
        let snapshotData = try Data(contentsOf: root.appendingPathComponent("rapid_mlx/decode_expectations.json"))
        for row in rows {
            var selected = snapshot
            if let key = row.cohortConfigKeyOverride {
                var payload = try #require(JSONSerialization.jsonObject(with: snapshotData) as? [String: Any])
                var models = try #require(payload["models"] as? [String: Any])
                var model = try #require(models[row.alias] as? [String: Any])
                var cohorts = try #require(model["cohorts"] as? [[String: Any]])
                for index in cohorts.indices where cohorts[index]["method"] as? String == "mtp" {
                    cohorts[index]["config_key"] = key
                }
                model["cohorts"] = cohorts
                models[row.alias] = model
                payload["models"] = models
                selected = try JSONDecoder().decode(DecodeExpectations.Snapshot.self, from: JSONSerialization.data(withJSONObject: payload))
            }
            #expect(DecodeExpectations.summary(alias: row.alias, chip: row.chip, ramGiB: row.ramGiB, repoID: row.repoID, snapshot: selected) == row.summary)
            #expect(DecodeExpectations.summary(alias: row.alias, chip: row.chip, ramGiB: row.ramGiB, repoID: row.repoID, snapshot: selected, compact: true) == row.compact)
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
