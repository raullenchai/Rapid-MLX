import Foundation
import Testing
@testable import Rapid

/// Pin the shape of ``Localizable.xcstrings`` so a future drift can't
/// silently break the zh-Hans surface, and prove that
/// ``NSLocalizedString`` resolves through the catalog when the runtime
/// language is forced to Simplified Chinese.
@Suite("Localizable.xcstrings — catalog shape and zh-Hans resolution")
struct LocalizationTests {

    private static let photoHintCatalogKeys = [
        "image_input.unavailable.legacy_model",
        "image_input.unavailable.text_lane_forced",
        "image_input.unavailable.speculative_decode",
        "image_input.unavailable.vision_memory_insufficient",
        "image_input.unavailable.vision_runtime_unsupported",
        "image_input.unavailable.vision_features_unavailable",
        "image_input.unavailable.text_checkpoint",
        "image_input.unavailable.generic_text_lane"
    ]

    /// Look up the catalog from the test bundle. The .xcstrings file
    /// is declared as a resource on the Rapid executable target, so
    /// at test time it lives next to the test bundle's bundleURL
    /// under the host process's resource lookup chain. We probe the
    /// known SPM bundle path first, then fall back to the source
    /// tree path which is always present in a CI checkout.
    private func catalogURL() throws -> URL {
        let candidates: [URL] = [
            Bundle.module.url(forResource: "Localizable", withExtension: "xcstrings"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings")
        ].compactMap { $0 }

        return try #require(
            candidates.first { FileManager.default.fileExists(atPath: $0.path) },
            "Localizable.xcstrings not found on any candidate path"
        )
    }

    private func loadCatalog() throws -> [String: Any] {
        let url = try catalogURL()
        let data = try Data(contentsOf: url)
        let any = try JSONSerialization.jsonObject(with: data)
        return try #require(any as? [String: Any])
    }

    @Test("Catalog parses as valid xcstrings JSON with the expected top-level shape")
    func catalogShape() throws {
        let json = try loadCatalog()
        #expect(json["sourceLanguage"] as? String == "en")
        #expect(json["version"] as? String == "1.0")
        let strings = try #require(json["strings"] as? [String: Any])
        #expect(!strings.isEmpty)
    }

    @Test("Canonical user-visible keys carry a zh-Hans translation")
    func canonicalKeysTranslated() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        // Pick a few high-visibility keys spanning compose, sidebar,
        // settings, about, status — if any of these regress to
        // untranslated, the Chinese surface is visibly broken.
        let mustHaveZH: [String] = [
            "Send a message…",
            "New chat",
            "Search chats",
            "Today",
            "Previous 30 Days",
            "No chats match",
            "Settings",
            "Appearance",
            "Privacy",
            "About Rapid-MLX",
            "Ready",
            "Downloading",
            "Stopped"
        ]

        for key in mustHaveZH {
            let entry = try #require(
                strings[key] as? [String: Any],
                "Missing catalog entry for key: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(
                localizations["zh-Hans"] as? [String: Any],
                "Missing zh-Hans for key: \(key)"
            )
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(unit["state"] as? String == "translated")
            let value = try #require(unit["value"] as? String)
            #expect(!value.isEmpty)
        }
    }

    @Test("Every entry that declares a zh-Hans block has a non-empty translated value")
    func noPartialZHEntries() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        for (key, raw) in strings {
            guard
                let entry = raw as? [String: Any],
                let localizations = entry["localizations"] as? [String: Any],
                let zh = localizations["zh-Hans"] as? [String: Any]
            else {
                continue
            }
            let unit = try #require(zh["stringUnit"] as? [String: Any], "Missing stringUnit for \(key)")
            let value = unit["value"] as? String ?? ""
            #expect(!value.isEmpty, "Empty zh-Hans value for key: \(key)")
            #expect(
                (unit["state"] as? String) == "translated",
                "zh-Hans not marked translated for key: \(key)"
            )
        }
    }

    @Test("Every photo-unavailable remedy has reviewed English and zh-Hans catalog values")
    func localizedPhotoHintsUseStableCatalogKeys() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        #expect(
            Set(ImageInputAvailability.PhotoHint.allCases.map(\.rawValue))
                == Set(Self.photoHintCatalogKeys),
            "The production photo-hint key set and the reviewed catalog contract must move together."
        )

        for hint in ImageInputAvailability.PhotoHint.allCases {
            let key = hint.rawValue
            let entry = try #require(
                strings[key] as? [String: Any],
                "Missing photo-hint catalog entry for key: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            for language in ["en", "zh-Hans"] {
                let localization = try #require(
                    localizations[language] as? [String: Any],
                    "Missing \(language) photo-hint value for key: \(key)"
                )
                let unit = try #require(localization["stringUnit"] as? [String: Any])
                #expect(unit["state"] as? String == "translated")
                #expect(!(unit["value"] as? String ?? "").isEmpty)
                if language == "en" {
                    #expect(
                        unit["value"] as? String == hint.englishValue,
                        "The catalog's source copy and the fail-safe English value diverged for \(key)."
                    )
                }
            }
        }
    }

    @Test("Compiled zh-Hans catalog resolves through the production photo-hint path")
    func compiledCatalogLocalizesPhotoHints() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }

        let compiler = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: [
                "xcstringstool", "compile", try catalogURL().path,
                "--output-directory", output.path,
                "--serialization-format", "binary"
            ]
        )
        #expect(
            compiler.terminationStatus == 0,
            "xcstringstool failed: \(String(decoding: compiler.standardError, as: UTF8.self))"
        )

        let zhBundle = try #require(
            Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)),
            "xcstringstool did not emit a loadable zh-Hans localization bundle"
        )
        let memory = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: true,
            profile: ServerModelProfile(
                id: "model",
                capabilities: ["text", "vision"],
                servingLane: "text",
                servingLaneReason: "vision_memory_insufficient"
            ),
            localizationBundle: zhBundle
        )
        #expect(
            memory.unavailableMessage
                == "此模型的文字聊天可以正常使用。照片模式需要的内存超过这台 Mac 的容量；如需添加照片，请选择内存需求更低的视觉模型。"
        )

        let legacy = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: false,
            profile: nil,
            localizationBundle: zhBundle
        )
        #expect(legacy.unavailableMessage == "此模型不支持照片。要添加照片，请选择支持视觉的模型。")
    }
}
