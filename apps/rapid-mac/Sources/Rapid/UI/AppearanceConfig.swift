import AppKit
import Foundation
import Observation

/// v0.4.25: app-wide appearance override. macOS already follows the
/// system dark/light setting, but two scenarios call for a manual
/// switch:
///
/// 1. Screen-shares / demos where the audience reads better in light
///    mode even though the host's desktop runs dark.
/// 2. The opposite: a light-default user who wants to A/B-check
///    dark-mode polish without flipping the whole system.
///
/// The picker lives in Settings → Appearance. We persist the chosen
/// mode in UserDefaults via the same `rapid.appearance.v1` keyspace
/// pattern the rest of v0.4 settings use, and apply it by setting
/// `NSApp.appearance` — `nil` means "follow the system" so we can
/// switch back to Auto without an app restart.
@MainActor
@Observable
final class AppearanceConfig {
    /// One of "system" / "light" / "dark". Mutating this triggers the
    /// AppKit appearance swap via the `didSet`-like observer below
    /// (Observation framework dispatches the setter, then we re-apply).
    var mode: AppearanceMode {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: Self.storageKey)
            apply()
        }
    }

    /// Surface key — matches the `rapid.*.v1` pattern other v0.4
    /// settings use. Versioned so a future schema bump can migrate
    /// without colliding with the old stored value.
    static let storageKey = "rapid.appearance.v1"

    init() {
        // v0.5: light-first default. When the user has an explicit
        // saved choice — including "Auto (follow system)", whose stored
        // value is "system" — we honour it. Only a *fresh install* with
        // no persisted value at all falls back to Light (was ``.system``),
        // so the app opens light-first the first time it's launched
        // without removing the Auto / Dark options from Settings.
        if let raw = UserDefaults.standard.string(forKey: Self.storageKey),
           let saved = AppearanceMode(rawValue: raw) {
            self.mode = saved
        } else {
            self.mode = .light
        }
    }

    /// Push the current mode to `NSApp.appearance`. Safe to call on
    /// app launch (before the first window appears) — `NSApp` is the
    /// singleton; AppKit honours the override the moment any window
    /// renders.
    ///
    /// `NSApp` is declared as an implicitly-unwrapped `NSApplication!`
    /// but in a non-AppKit host (the Swift Testing harness, for
    /// example) the global hasn't been bootstrapped and force-unwrap
    /// crashes. We bind it as Optional first so the apply path stays
    /// no-op in test contexts.
    func apply() {
        guard let app = NSApp else { return }
        app.appearance = mode.nsAppearance
    }
}

/// Three-way enum for the picker. `system` returns `nil` for the
/// AppKit mapping so the appearance follows the macOS preference.
enum AppearanceMode: String, CaseIterable, Identifiable, Sendable {
    case system
    case light
    case dark

    var id: String { rawValue }

    /// Stable selector for VoiceOver, XCUITest, and AX-first dogfood agents.
    /// Keep this independent of localized display copy so automation does not
    /// fall back to screen coordinates when labels change.
    var accessibilityIdentifier: String {
        "Settings.Appearance.Theme.\(rawValue)"
    }

    /// Picker label — human-friendly, matches the macOS System
    /// Settings → Appearance row text.
    var displayName: String {
        switch self {
        case .system: return "Auto (follow system)"
        case .light:  return "Light"
        case .dark:   return "Dark"
        }
    }

    /// AppKit appearance for `NSApp.appearance`. `nil` means
    /// "don't override" — the app inherits the system setting.
    var nsAppearance: NSAppearance? {
        switch self {
        case .system: return nil
        case .light:  return NSAppearance(named: .aqua)
        case .dark:   return NSAppearance(named: .darkAqua)
        }
    }
}
