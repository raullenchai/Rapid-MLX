import SwiftUI

/// A one-line contextual message with an optional trailing action.
///
/// Replaces the bare `Text(error).foregroundStyle(amberDeep)` banner
/// pattern, which gave a failure no container, no icon, and no visual
/// weight distinct from ordinary caption copy — so a real error read
/// like a footnote.
///
/// Three tones, mapped to the status tokens so the meaning of a colour
/// is the same here as in a status pill or a metric chip. Note that
/// ``warning`` is amber and shares the brand hue: that is intentional,
/// since amber already carries "something is in flight / needs
/// attention" across the product. ``error`` is the only red.
struct InlineNotice: View {
    enum Tone {
        case info
        case warning
        case error
        /// Something finished and freed/saved something. Added for the
        /// Settings migration, which had two hand-rolled green banners
        /// ("Freed 4.2 GB", "Saved to your Keychain") drawing their own
        /// container out of `RapidTheme.green.opacity(0.08)`.
        case success

        /// v1.0.2: ``info`` is AMBER, not steel blue.
        ///
        /// An inline readiness notice is a "here's what's happening"
        /// signal, and amber is the product's colour for that. Steel
        /// blue survives only for genuine links
        /// (``RapidTheme.linkLabel``), which is what "rare supporting
        /// colour" has to mean if it is to mean anything.
        ///
        /// ``warning`` now reads its colour from ``statusWarning``
        /// rather than naming the brand token directly. Same amber
        /// today; the difference is that "warning" is now a meaning the
        /// theme owns, so the five Settings call sites that reached for
        /// `Color.orange` have somewhere correct to go.
        var tint: Color {
            switch self {
            case .info:    return RapidTheme.brandPrimaryDeep
            case .warning: return RapidTheme.statusWarning
            case .error:   return RapidTheme.statusError
            case .success: return RapidTheme.statusReady
            }
        }

        /// Subtle tints only — never a saturated panel. These are
        /// notices inside a page, not cards of their own.
        var background: Color {
            switch self {
            case .info:    return RapidTheme.brandPrimaryTint
            case .warning: return RapidTheme.statusWarningTint
            case .error:   return RapidTheme.statusErrorTint
            case .success: return RapidTheme.statusReadyTint
            }
        }

        var symbol: String {
            switch self {
            case .info:    return "info.circle.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .error:   return "exclamationmark.octagon.fill"
            case .success: return "checkmark.circle.fill"
            }
        }
    }

    let message: String
    var tone: Tone = .warning
    var actionTitle: String? = nil
    var actionIdentifier: String? = nil
    var action: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
            Image(systemName: tone.symbol)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(tone.tint)
                .accessibilityHidden(true)

            Text(message)
                .font(RapidFont.secondary)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(RapidTertiaryButtonStyle(
                        link: tone == .info,
                        height: RapidTheme.ControlHeight.mini
                    ))
                    .fixedSize()
                    .accessibilityIdentifier(actionIdentifier ?? "InlineNotice.Action")
            }
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.sm)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .fill(tone.background)
        )
        .accessibilityElement(children: .contain)
    }
}
