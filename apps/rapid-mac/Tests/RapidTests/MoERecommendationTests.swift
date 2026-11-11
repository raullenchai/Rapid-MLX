import Foundation
import Testing
@testable import Rapid

@Suite("MoE recommendations — shared Python/Desktop vectors")
struct MoERecommendationTests {
    private var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    struct Vector: Decodable {
        let ramGiB: Double
        let chip: String?
        let chipClass: String
        let primary: String
        enum CodingKeys: String, CodingKey {
            case chip, primary
            case ramGiB = "ram_gib", chipClass = "chip_class"
        }
    }

    @Test func sharedChipAndSelectionVectors() throws {
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/moe_recommendations_golden.json"))
        for row in try JSONDecoder().decode([Vector].self, from: data) {
            #expect(MacHardware.recommendationChipClass(row.chip) == row.chipClass)
            #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: row.ramGiB, chip: row.chip) == row.primary)
            #expect(RAMBucketedDefault.picks(forPhysicalRAMGB: row.ramGiB, chip: row.chip).first?.alias == row.primary)
            #expect(RAMBucketedDefault.preferenceOrder(forPhysicalRAMGB: row.ramGiB, chip: row.chip).first == row.primary)
        }
    }

    @Test func qualifiersFailClosedAndOldPoliciesKeepTheirOrder() throws {
        let source = try Data(contentsOf: root.appendingPathComponent("rapid_mlx/model_recommendations.json"))
        func policy(qualifier: Any?) throws -> Data {
            var object = try #require(JSONSerialization.jsonObject(with: source) as? [String: Any])
            var tiers = try #require(object["tiers"] as? [[String: Any]])
            for index in tiers.indices {
                if qualifier == nil { tiers[index].removeValue(forKey: "prefer_fast_chip_classes") }
                else if index == 5 { tiers[index]["prefer_fast_chip_classes"] = qualifier }
            }
            object["tiers"] = tiers
            object.removeValue(forKey: "policy_digest")
            object["policy_digest"] = try #require(ModelCatalog.atomicObjectDigest(object))
            return try JSONSerialization.data(withJSONObject: object)
        }
        let invalid: [Any] = [NSNull(), [String](), ["max"], ["base", "base"], "base", [true]]
        for value in invalid {
            #expect(RAMBucketedDefault.parseRecommendationPolicy(try policy(qualifier: value)) == nil)
        }
        let old = try #require(RAMBucketedDefault.parseRecommendationPolicy(try policy(qualifier: nil)))
        #expect(old.last?.ordered(chip: "M4 Pro").primary.alias == "qwen3.8-27b-4bit")
    }

    @Test func secondaryCapabilityComparisonUsesScoresRatherThanAliases() {
        func pick(alias: String, capability: Int) -> RAMBucketedDefault.Pick {
            RAMBucketedDefault.Pick(
                alias: alias, footprintGB: 20, capabilityPct: capability,
                tokensPerSec: nil, launchFlags: []
            )
        }
        let primary = pick(alias: "primary", capability: 87)
        #expect(RAMBucketedDefault.secondaryIsHigherCapability(
            pick(alias: "future-model", capability: 92), primary: primary
        ))
        #expect(!RAMBucketedDefault.secondaryIsHigherCapability(
            pick(alias: "equal-model", capability: 87), primary: primary
        ))
        #expect(!RAMBucketedDefault.secondaryIsHigherCapability(
            pick(alias: "qwen3.8-27b-4bit", capability: 82), primary: primary
        ))
    }

    @Test func secondaryCapabilityComparisonFollowsChipResolvedPolicy() throws {
        for ram in [48.0, 64.0, 96.0] {
            for chip in ["Apple M4", "Apple M4 Pro", "Apple M4 Max", "Apple M2 Ultra", "Unknown"] {
                let tier = RAMBucketedDefault.tier(forPhysicalRAMGB: ram, chip: chip)
                let secondary = try #require(tier.alt)
                let chipClass = MacHardware.recommendationChipClass(chip)
                #expect(RAMBucketedDefault.secondaryIsHigherCapability(secondary, primary: tier.primary)
                    == (chipClass == "base" || chipClass == "pro"))
            }
        }
        for tier in RAMBucketedDefault.tiers where tier.floorGB < 48 {
            let secondary = try #require(tier.alt)
            #expect(!RAMBucketedDefault.secondaryIsHigherCapability(secondary, primary: tier.primary))
        }
    }

    struct LabelVector: Decodable {
        let total: Double?
        let active: Double?
        let label: String?
    }

    @Test func sharedParameterLabels() throws {
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/moe_parameter_labels_golden.json"))
        for row in try JSONDecoder().decode([LabelVector].self, from: data) {
            #expect(DecodeExpectations.parameterLabel(total: row.total, active: row.active) == row.label)
        }
        #expect(DecodeExpectations.parameterLabel(total: .infinity, active: 3) == nil)
    }

    @Test func legacyCatalogCarriesParameterMetadata() throws {
        let output = #"{"text":[{"alias":"moe","hf_path":"org/moe","is_builtin":true,"is_text_only":true,"total_params_b":35,"active_params_b":3}]}"#
        let parsed = try #require(ModelCatalog.parseAvailableJSON(output))
        let profile = try #require(parsed.profiles["moe"])
        #expect(profile.totalParamsB == 35)
        #expect(profile.activeParamsB == 3)
    }
}
