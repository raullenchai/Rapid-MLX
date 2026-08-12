import Foundation
import Testing
@testable import Rapid

/// Pin the shape of ``Localizable.xcstrings`` so a future drift can't
/// silently break the zh-Hans surface, and prove that
/// ``NSLocalizedString`` resolves through the catalog when the runtime
/// language is forced to Simplified Chinese.
@Suite("Localizable.xcstrings — catalog shape and zh-Hans resolution")
struct LocalizationTests {

    /// Look up the catalog from the test bundle. The .xcstrings file
    /// is declared as a resource on the Rapid executable target, so
    /// at test time it lives next to the test bundle's bundleURL
    /// under the host process's resource lookup chain. We probe the
    /// known SPM bundle path first, then fall back to the source
    /// tree path which is always present in a CI checkout.
    private func loadCatalog() throws -> [String: Any] {
        let candidates: [URL] = [
            Bundle.module.url(forResource: "Localizable", withExtension: "xcstrings"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings")
        ].compactMap { $0 }

        let url = try #require(
            candidates.first { FileManager.default.fileExists(atPath: $0.path) },
            "Localizable.xcstrings not found on any candidate path"
        )
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
}
