import AppKit
import Foundation
import Testing
@testable import Rapid

/// Contract for v0.4.25 Settings → Appearance override. Pins:
///   - mode → NSAppearance mapping (system → nil, light → aqua, dark → darkAqua)
///   - default mode is `.system` (no override) so brand-new users
///     don't get a forced theme
///   - mutating mode persists to UserDefaults under the documented key
///   - a fresh instance reads the persisted value back
///   - garbage stored value falls back to `.system` (defensive against
///     a manual defaults write or a future schema bump)
///   - displayName text covers the three cases
@MainActor
@Suite("AppearanceConfig + AppearanceMode — v0.4.25")
final class AppearanceConfigTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-appearance-test-")
        createdSuiteNames.append(name)
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    @Test("AppearanceMode → NSAppearance mapping")
    func nsAppearanceMapping() {
        #expect(AppearanceMode.system.nsAppearance == nil)
        #expect(AppearanceMode.light.nsAppearance?.name == .aqua)
        #expect(AppearanceMode.dark.nsAppearance?.name == .darkAqua)
    }

    @Test("Display names are human-friendly and distinct")
    func displayNames() {
        #expect(AppearanceMode.system.displayName == "Auto (follow system)")
        #expect(AppearanceMode.light.displayName == "Light")
        #expect(AppearanceMode.dark.displayName == "Dark")
        let names = AppearanceMode.allCases.map(\.displayName)
        #expect(AppearanceMode.system.accessibilityIdentifier == "Settings.Appearance.Theme.system")
        #expect(AppearanceMode.light.accessibilityIdentifier == "Settings.Appearance.Theme.light")
        #expect(AppearanceMode.dark.accessibilityIdentifier == "Settings.Appearance.Theme.dark")
        #expect(Set(names).count == names.count)
    }

    @Test("Default mode is .light when no value is stored — v0.5 light-first brand decision")
    func defaultIsLight() {
        let defaults = freshDefaults()
        let cfg = AppearanceConfig(defaults: defaults)
        #expect(cfg.mode == .light)
    }

    @Test("Mutating mode persists to UserDefaults")
    func mutationPersists() {
        let defaults = freshDefaults()
        let cfg = AppearanceConfig(defaults: defaults)
        cfg.mode = .dark
        let raw = defaults.string(forKey: AppearanceConfig.storageKey)
        #expect(raw == "dark")
        cfg.mode = .light
        #expect(defaults.string(forKey: AppearanceConfig.storageKey) == "light")
    }

    @Test("Fresh instance reads back the stored value")
    func roundTrips() {
        let defaults = freshDefaults()
        let writer = AppearanceConfig(defaults: defaults)
        writer.mode = .light
        let reader = AppearanceConfig(defaults: defaults)
        #expect(reader.mode == .light)
    }

    @Test("Garbage stored value falls back to the v0.5 light-first default — defensive against future schema bumps")
    func garbageFallback() {
        let defaults = freshDefaults()
        defaults.set("midnight-blue", forKey: AppearanceConfig.storageKey)
        let cfg = AppearanceConfig(defaults: defaults)
        #expect(cfg.mode == .light)
    }
}
