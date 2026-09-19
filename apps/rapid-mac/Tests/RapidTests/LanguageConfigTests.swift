import AppKit
import Foundation
import Testing
@testable import Rapid

/// Contract for the Settings → Appearance language override. Pins:
///   - a fresh install follows the system (never a forced language)
///   - mutating the picker persists under the documented key
///   - a fresh instance reads the persisted value back
///   - a garbage stored value falls back to `.system`
///   - the language → `.lproj` code and locale mappings
///   - the Foundation redirect's *scoping*, which is the part that could do
///     real damage: it must never rewrite another bundle's lookups, and an
///     unknown code must degrade to "no override" rather than blanking copy
@MainActor
@Suite("LanguageConfig + AppLanguage")
final class LanguageConfigTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-language-test-")
        createdSuiteNames.append(name)
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    @Test("A fresh install follows the system, unlike the light-first theme default")
    func defaultFollowsSystem() {
        let cfg = LanguageConfig(defaults: freshDefaults())
        #expect(cfg.language == .system)
        #expect(cfg.language.localizationCode == nil)
    }

    @Test("Mutating the picker persists under the documented key")
    func mutationPersists() {
        let defaults = freshDefaults()
        let cfg = LanguageConfig(defaults: defaults)
        cfg.language = .simplifiedChinese
        #expect(defaults.string(forKey: LanguageConfig.storageKey) == "zh-Hans")
    }

    @Test("A fresh instance reads the persisted value back")
    func roundTrips() {
        let defaults = freshDefaults()
        LanguageConfig(defaults: defaults).language = .english
        #expect(LanguageConfig(defaults: defaults).language == .english)
    }

    @Test("A garbage stored value falls back to following the system")
    func garbageFallsBack() {
        let defaults = freshDefaults()
        defaults.set("klingon", forKey: LanguageConfig.storageKey)
        #expect(LanguageConfig(defaults: defaults).language == .system)
    }

    @Test("Every offered language maps to a compiled catalog column")
    func localizationCodes() {
        #expect(AppLanguage.system.localizationCode == nil)
        #expect(AppLanguage.english.localizationCode == "en")
        #expect(AppLanguage.simplifiedChinese.localizationCode == "zh-Hans")
    }

    @Test("The forced language is what formatters follow, not Locale.current")
    func resolvedLocales() {
        let cfg = LanguageConfig(defaults: freshDefaults())
        #expect(cfg.resolvedLocale == Locale.current)
        cfg.language = .english
        #expect(cfg.resolvedLocale.identifier == "en")
        cfg.language = .simplifiedChinese
        // A forced language must not inherit the host's locale: a Chinese UI
        // showing US-formatted dates is the half-translated result this
        // feature exists to avoid.
        #expect(cfg.resolvedLocale.identifier == "zh-Hans")
    }

    @Test("Picker labels name each language in itself, and are distinct")
    func displayNames() {
        // "English" and "简体中文" are deliberately NOT translated: a user who
        // cannot read the current UI language has to be able to find their own.
        #expect(AppLanguage.english.displayName == "English")
        #expect(AppLanguage.simplifiedChinese.displayName == "简体中文")
        let names = AppLanguage.allCases.map(\.displayName)
        #expect(Set(names).count == names.count)
    }

    @Test("Accessibility identifiers are stable and distinct")
    func accessibilityIdentifiers() {
        #expect(AppLanguage.system.accessibilityIdentifier == "Settings.Appearance.Language.system")
        #expect(AppLanguage.english.accessibilityIdentifier == "Settings.Appearance.Language.en")
        #expect(
            AppLanguage.simplifiedChinese.accessibilityIdentifier
                == "Settings.Appearance.Language.zh-Hans"
        )
        let ids = AppLanguage.allCases.map(\.accessibilityIdentifier)
        #expect(Set(ids).count == ids.count)
    }

    // MARK: - BundleLanguageOverride scoping

    @Test("Only the main bundle is redirected")
    func onlyMainBundleIsRedirected() {
        #expect(
            BundleLanguageOverride.shared.bundle(for: Bundle.module, code: "zh-Hans") == nil,
            "Redirecting another bundle would rewrite Sparkle/MarkdownUI lookups we do not own"
        )
    }

    @Test("A code with no compiled .lproj degrades to no override")
    func unknownCodeDegrades() {
        // The test runner is not the app bundle, so it carries no zh-Hans.lproj.
        // Resolving must return nil so the caller falls through to the process
        // localization instead of blanking every string.
        #expect(BundleLanguageOverride.shared.bundle(for: .main, code: "zh-Hans") == nil)
        #expect(BundleLanguageOverride.shared.bundle(for: .main, code: nil) == nil)
    }

    @Test("With no override active the lookup is a pure pass-through")
    func passthroughIsExact() {
        var seen: (String, String?, String?)?
        let result = BundleLanguageOverride.shared.resolve(
            receiver: .main,
            key: "New chat",
            value: "fallback",
            table: nil
        ) { _, key, value, table in
            seen = (key, value, table)
            return "PASSTHROUGH"
        }
        #expect(result == "PASSTHROUGH")
        #expect(seen?.0 == "New chat")
        #expect(seen?.1 == "fallback")
        #expect(seen?.2 == nil)
    }
}
