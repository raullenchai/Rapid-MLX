import Foundation
import Testing
@testable import Rapid

@Suite("Image model parity")
struct ImageModelParityTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // RapidTests
        .deletingLastPathComponent() // Tests
        .deletingLastPathComponent() // rapid-mac
        .deletingLastPathComponent() // apps
        .deletingLastPathComponent() // repo root

    private static let aliasFamily: [String: String] = [
        "flux2-klein-4b": "flux2-klein",
        "flux2-klein-4b-bf16": "flux2-klein",
        "bonsai-image-4b-2bit": "bonsai-image",
        "flux-schnell": "flux-schnell",
        "z-image-turbo": "z-image",
        "qwen-image": "qwen-image-2.1",
        "qwen-image-2.1": "qwen-image-2.1",
        "qwen-image-2.1-bf16": "qwen-image-2.1",
        "qwen-image-edit": "qwen-image-edit",
        "hidream-o1-dev": "hidream-o1-dev",
        "sdxl-base": "sdxl-base",
        "sd35-large-4bit": "sd35-large"
    ]

    // rapid_mlx/image/qwen21_turbo.py defines the fixed TURBO_STEPS schedule.
    private static let literalSteps = ["qwen-image-2.1-turbo": 8]

    private static func imageAliases() throws -> Set<String> {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("rapid_mlx/aliases.json"))
        let aliases = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: [String: Any]],
            "aliases.json must contain a top-level object keyed by alias"
        )
        return Set(aliases.compactMap { $0.value["modality"] as? String == "image-gen" ? $0.key : nil })
    }

    private static func engineSteps() throws -> [String: Int] {
        let source = try String(
            contentsOf: repoRoot.appendingPathComponent("rapid_mlx/image/engine.py"),
            encoding: .utf8
        )
        let dictionary = try NSRegularExpression(
            pattern: #"(?m)^_DEFAULT_STEPS_BY_FAMILY\s*=\s*\{([^}]+)\}"#
        )
        let match = try #require(
            dictionary.firstMatch(in: source, range: NSRange(source.startIndex..., in: source)),
            "Could not find _DEFAULT_STEPS_BY_FAMILY in rapid_mlx/image/engine.py"
        )
        let bodyRange = try #require(Range(match.range(at: 1), in: source))
        let body = String(source[bodyRange])
        let entry = try NSRegularExpression(pattern: #"(?m)^\s*"([^"]+)":\s*(\d+),?\s*(?:#.*)?$"#)
        var table: [String: Int] = [:]
        for match in entry.matches(in: body, range: NSRange(body.startIndex..., in: body)) {
            let familyRange = try #require(Range(match.range(at: 1), in: body))
            let stepsRange = try #require(Range(match.range(at: 2), in: body))
            table[String(body[familyRange])] = try #require(Int(body[stepsRange]))
        }
        try #require(!table.isEmpty, "Parsed no defaults from engine.py _DEFAULT_STEPS_BY_FAMILY")
        return table
    }

    @Test("every image alias has a parity mapping")
    func completeness() throws {
        #expect(
            try Self.imageAliases() == Set(Self.aliasFamily.keys).union(Self.literalSteps.keys),
            "Add each new image-gen alias to this table, ModelDisplayName.imageNames, and ImageGenViewModel.seedSteps"
        )
    }

    @Test("every image alias has a friendly display name")
    func names() throws {
        for alias in try Self.imageAliases().sorted() {
            let expected = try #require(ModelDisplayName.imageNames[alias], "Missing image display name for \(alias)")
            #expect(ModelDisplayName.name(for: alias) == expected)
            #expect(ModelDisplayName.name(for: alias) != alias)
        }
    }

    @Test("image seed steps match engine defaults")
    func steps() throws {
        let engineTable = try Self.engineSteps()
        for alias in try Self.imageAliases().sorted() {
            let expected: Int
            if let literal = Self.literalSteps[alias] {
                expected = literal
            } else {
                let family = try #require(Self.aliasFamily[alias], "Missing engine family for \(alias)")
                expected = try #require(engineTable[family], "Missing engine default for family \(family) (\(alias))")
            }
            #expect(ImageGenViewModel.seedSteps(for: alias) == expected, "Default steps for \(alias)")
        }
    }
}
