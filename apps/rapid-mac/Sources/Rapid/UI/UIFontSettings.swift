import Foundation
import Observation
import SwiftUI

/// App-wide UI text size (Settings → Appearance → Text size).
///
/// Every chrome font goes through ``Font/rapid(size:weight:design:)``,
/// which multiplies by ``scale`` and never returns less than
/// ``minimumPointSize``. Because this is `@Observable` and the font
/// helpers read ``scale`` during `body`, SwiftUI's observation tracking
/// re-renders every view that used a scaled font when the user changes
/// the setting — no relaunch, no view identity reset.
///
/// Not `@MainActor`: fonts are also built from nonisolated statics
/// (``RapidFont``). Writes only happen from the Settings UI on main.
@Observable
final class UIFontSettings: @unchecked Sendable {
    static let shared = UIFontSettings()

    static let storageKey = "rapid.uiFontSettings.scale.v1"
    /// Floor for every UI font — anything smaller is unreadable for many users.
    static let minimumPointSize: CGFloat = 12
    static let defaultScale: Double = 1.0
    static let range: ClosedRange<Double> = 1.0...1.5
    /// Step of the −/+ buttons.
    static let buttonStep: Double = 0.1

    var isDefault: Bool { abs(scale - Self.defaultScale) < 0.001 }

    /// Clamped to ``range`` and rounded to 0.01 so stored values stay tidy.
    func setScale(_ value: Double) {
        scale = (min(max(value, Self.range.lowerBound), Self.range.upperBound) * 100).rounded() / 100
    }

    @ObservationIgnored private let defaults: UserDefaults

    var scale: Double {
        didSet { defaults.set(scale, forKey: Self.storageKey) }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let saved = defaults.double(forKey: Self.storageKey)
        self.scale = Self.range.contains(saved) ? saved : Self.defaultScale
    }

    func pointSize(_ base: CGFloat) -> CGFloat {
        max(Self.minimumPointSize, (base * scale).rounded())
    }

    /// macOS point sizes for the SwiftUI text styles.
    static func basePointSize(_ style: Font.TextStyle) -> CGFloat {
        switch style {
        case .largeTitle: 26
        case .title: 22
        case .title2: 17
        case .title3: 15
        case .headline, .body: 13
        case .callout: 12
        case .subheadline: 11
        default: 10 // footnote, caption, caption2
        }
    }
}

extension Font {
    /// Scaled, floored replacement for `Font.system(size:weight:design:)`.
    static func rapid(
        size: CGFloat,
        weight: Font.Weight = .regular,
        design: Font.Design = .default
    ) -> Font {
        .system(size: UIFontSettings.shared.pointSize(size), weight: weight, design: design)
    }

    /// Scaled, floored replacement for `Font.system(_:design:weight:)` / `.caption` etc.
    static func rapid(
        _ style: Font.TextStyle,
        design: Font.Design = .default,
        weight: Font.Weight? = nil
    ) -> Font {
        .rapid(
            size: UIFontSettings.basePointSize(style),
            weight: weight ?? (style == .headline ? .semibold : .regular),
            design: design
        )
    }
}

/// Settings → Appearance row. Collapsed it reads "Default"; clicking reveals
/// a 1.0–1.5× slider with −/+ steps. Clicking the value again resets to
/// default and collapses. The slider commits on release so the whole app is
/// not re-laid-out on every drag tick.
struct UIFontScaleControl: View {
    @State private var expanded = false
    @State private var draft = UIFontSettings.shared.scale

    private var settings: UIFontSettings { .shared }
    private var showsSlider: Bool { expanded || !settings.isDefault }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack {
                SettingsRowLabel(
                    title: String(localized: "Text size"),
                    description: String(localized: "Scales all interface text. Text is never smaller than 12 pt.")
                )
                Spacer()
                Button {
                    if showsSlider {
                        settings.setScale(UIFontSettings.defaultScale)
                        expanded = false
                    } else {
                        expanded = true
                    }
                    draft = settings.scale
                } label: {
                    Text(showsSlider ? Self.label(draft) : String(localized: "Default"))
                        .monospacedDigit()
                }
                .accessibilityIdentifier("Settings.Appearance.TextSizeToggle")
                .help(showsSlider ? String(localized: "Reset to default") : String(localized: "Customize"))
            }
            if showsSlider {
                HStack(spacing: RapidTheme.Space.sm) {
                    step(-UIFontSettings.buttonStep, systemImage: "minus", label: String(localized: "Decrease text size"))
                    Slider(value: $draft, in: UIFontSettings.range) { editing in
                        if !editing { settings.setScale(draft); draft = settings.scale }
                    }
                    .accessibilityLabel(String(localized: "Text size"))
                    .accessibilityValue(Self.label(draft))
                    .accessibilityIdentifier("Settings.Appearance.TextSizeSlider")
                    step(UIFontSettings.buttonStep, systemImage: "plus", label: String(localized: "Increase text size"))
                }
            }
        }
        .onChange(of: settings.scale) { _, new in draft = new }
    }

    private func step(_ delta: Double, systemImage: String, label: String) -> some View {
        Button {
            settings.setScale(settings.scale + delta)
            draft = settings.scale
        } label: {
            Image(systemName: systemImage)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel(label)
        .accessibilityIdentifier("Settings.Appearance.TextSize.\(delta < 0 ? "Decrease" : "Increase")")
    }

    static func label(_ scale: Double) -> String {
        "\((scale * 100).rounded() / 100)×"
    }
}
