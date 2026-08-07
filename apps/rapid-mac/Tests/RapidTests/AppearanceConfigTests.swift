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
struct AppearanceConfigTests {
    /// Wipe the live UserDefaults key before each test so writes from
    /// one test don't leak into the next. Run in a non-parallel
    /// scheduler bucket by being inside the @Suite (Testing's default
    /// is serial within a struct, which is what we want here).
    private func freshDefaults() {
        UserDefaults.standard.removeObject(forKey: AppearanceConfig.storageKey)
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
        freshDefaults()
        let cfg = AppearanceConfig()
        #expect(cfg.mode == .light)
    }

    @Test("Mutating mode persists to UserDefaults")
    func mutationPersists() {
        freshDefaults()
        let cfg = AppearanceConfig()
        cfg.mode = .dark
        let raw = UserDefaults.standard.string(forKey: AppearanceConfig.storageKey)
        #expect(raw == "dark")
        cfg.mode = .light
        #expect(UserDefaults.standard.string(forKey: AppearanceConfig.storageKey) == "light")
    }

    @Test("Fresh instance reads back the stored value")
    func roundTrips() {
        freshDefaults()
        let writer = AppearanceConfig()
        writer.mode = .light
        let reader = AppearanceConfig()
        #expect(reader.mode == .light)
    }

    @Test("Garbage stored value falls back to the v0.5 light-first default — defensive against future schema bumps")
    func garbageFallback() {
        freshDefaults()
        UserDefaults.standard.set("midnight-blue", forKey: AppearanceConfig.storageKey)
        let cfg = AppearanceConfig()
        #expect(cfg.mode == .light)
    }
}
