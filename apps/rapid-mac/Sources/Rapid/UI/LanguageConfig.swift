import Foundation
import Observation

/// App-wide UI-language override, the sibling of ``AppearanceConfig``.
///
/// macOS picks an app's language from the system setting, and until now that
/// was the only way to get a non-English Rapid-MLX: the desktop app compiled
/// the `zh-Hans` table but had no control for it, so a user on an English Mac
/// saw English copy and no way to ask for anything else. The picker lives in
/// Settings → Appearance and persists under the same `rapid.<name>.v1`
/// keyspace the rest of the settings use.
///
/// The chosen language applies to a **running** process, so the picker does not
/// need an "applies after restart" caveat. The switch has two halves, and a new
/// surface has to be on the right one:
///
/// 1. **SwiftUI copy** rides ``resolvedLocale`` into the `\.locale` environment
///    (see `RapidApp`). SwiftUI resolves `Text("…")` against that locale, and an
///    environment change invalidates the views reading it, so the switch lands
///    without rebuilding the view tree — which matters, because rebuilding would
///    discard view-local `@State` such as the half-typed message in
///    `ChatView.draft`.
/// 2. **Foundation and AppKit copy** — `String(localized:)`, `NSMenuItem`,
///    `NSAlert` — rides ``BundleLanguageOverride``.
///
/// A formatter that wants to follow the UI language rather than the system
/// reads ``resolvedLocale`` directly; the `\.locale` environment already covers
/// the ones SwiftUI owns.
@MainActor
@Observable
final class LanguageConfig {
    private let defaults: UserDefaults

    /// The persisted choice. Mutating it writes through and re-applies
    /// immediately, matching ``AppearanceConfig/mode``.
    var language: AppLanguage {
        didSet {
            defaults.set(language.rawValue, forKey: Self.storageKey)
            apply()
        }
    }

    /// Surface key — matches the `rapid.*.v1` pattern the rest of the settings
    /// use. Versioned so a future schema bump can migrate without colliding
    /// with the old stored value.
    static let storageKey = "rapid.language.v1"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        // Unlike the theme (which is light-first on a fresh install), a fresh
        // install must keep following the system. Defaulting to a specific
        // language would override the macOS setting for every new user.
        if let raw = defaults.string(forKey: Self.storageKey),
           let saved = AppLanguage(rawValue: raw) {
            self.language = saved
        } else {
            self.language = .system
        }
    }

    /// Push the current choice into the bundle redirect. Called on launch
    /// (before the first window renders) and on every change.
    func apply() {
        BundleLanguageOverride.activate(code: language.localizationCode)
    }

    /// Locale to format dates/numbers with so they match the UI language.
    /// `.system` defers to ``Locale/current``; a forced language answers with
    /// its own locale, because a Chinese UI showing US-formatted dates is the
    /// half-translated result this feature exists to avoid.
    var resolvedLocale: Locale {
        switch language {
        case .system: return .current
        case .english: return Locale(identifier: "en")
        case .simplifiedChinese: return Locale(identifier: "zh-Hans")
        }
    }
}

/// Languages the picker offers: follow macOS, or force one of the tables this
/// build actually ships.
///
/// Only `en` and `zh-Hans` are listed because those are the only `.lproj`
/// directories ``scripts/build.sh`` compiles from `Localizable.xcstrings`.
/// Adding a case here without adding its catalog column would offer the user a
/// language whose every string falls back to English — the exact failure the
/// catalog's `LocalizationTests` exist to prevent.
enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case system
    case english = "en"
    case simplifiedChinese = "zh-Hans"

    var id: String { rawValue }

    /// Stable selector for VoiceOver, XCUITest, and AX-first dogfood agents.
    /// Keep this independent of localized display copy so automation does not
    /// fall back to screen coordinates when labels change.
    var accessibilityIdentifier: String {
        "Settings.Appearance.Language.\(rawValue)"
    }

    /// The `.lproj` code to force, or nil to leave the process alone.
    var localizationCode: String? {
        switch self {
        case .system: return nil
        case .english, .simplifiedChinese: return rawValue
        }
    }

    /// Picker label.
    ///
    /// A language is named **in itself** — "English" and "简体中文" — because
    /// that is what a user who cannot read the current UI language is looking
    /// for. Only the "follow the system" row is translated, since it describes
    /// behaviour rather than naming a language. Reuses the existing
    /// `Auto (follow system)` catalog key so this row and the theme picker's
    /// equivalent row cannot drift apart.
    var displayName: String {
        switch self {
        case .system: return String(localized: "Auto (follow system)")
        case .english: return "English"
        case .simplifiedChinese: return "简体中文"
        }
    }
}
