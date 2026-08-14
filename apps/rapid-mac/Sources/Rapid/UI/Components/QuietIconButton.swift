import SwiftUI

/// A borderless glyph button with a real hit target and a hover wash.
///
/// Replaces two patterns that were spread across the surface:
///
///   * a bare `Image` inside `.buttonStyle(.plain)`, which gave the
///     user a ~12pt target and no hover feedback at all; and
///   * the permanently-filled grey `xmark.circle.fill` close control,
///     which sat at full weight in the corner of every sheet whether
///     or not the pointer was anywhere near it.
///
/// Colour follows the utility policy: neutral at rest, steel blue on
/// hover, and an explicit tint only when the control is reporting
/// state (a copy that just succeeded goes ready-green).
struct QuietIconButton: View {
    let symbol: String
    /// Spoken label. Also the tooltip unless ``help`` overrides it.
    let label: String
    var help: String? = nil
    /// Overrides the resting colour — for transient success states.
    var tint: Color? = nil
    var size: CGFloat = RapidTheme.ControlHeight.small
    var symbolSize: CGFloat = 11
    var action: () -> Void

    @Environment(\.isEnabled) private var isEnabled
    @State private var hovering = false

    private var foreground: Color {
        if let tint { return tint }
        return hovering && isEnabled
            ? RapidTheme.utilityActionHover
            : RapidTheme.utilityActionLabel
    }

    /// Disabled glyphs stay legible at ~55%. A copy control that is
    /// unavailable still has to be FINDABLE — its tooltip is what
    /// explains how to make it available, so fading it to near-nothing
    /// hides the explanation too.
    private var resolvedOpacity: Double { isEnabled ? 1.0 : 0.55 }

    var body: some View {
        // ax-exempt: callers attach action/entity-specific identifiers to this wrapper
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: symbolSize, weight: .medium))
                .foregroundStyle(foreground)
                // These buttons are overwhelmingly toggles — copy/checkmark,
                // show/hide, play/pause — and a hard glyph swap gave no signal
                // that the press registered. `.replace` cross-fades the old
                // symbol out and the new one in, which is what carries "copied"
                // once the word "Copied" is gone. Animating on the symbol name
                // keeps it to genuine changes: a button whose glyph is constant
                // never animates, and Reduce Motion drops it entirely (see
                // ``rapidAnimation``).
                .contentTransition(.symbolEffect(.replace))
                .frame(width: size, height: size)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                        .fill(hovering && isEnabled ? RapidTheme.hoverFill : .clear)
                )
                .contentShape(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                )
                .opacity(resolvedOpacity)
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: hovering)
        .rapidAnimation(RapidMotion.quick, value: symbol)
        .help(help ?? label)
        .accessibilityLabel(label)
    }
}

/// The standard sheet dismiss control: a quiet ✕ that only gains weight
/// under the pointer.
struct SheetCloseButton: View {
    var action: () -> Void

    var body: some View {
        QuietIconButton(
            symbol: "xmark",
            label: "Close",
            help: "Close — Esc",
            action: action
        )
        .accessibilityIdentifier("Sheet.Close")
    }
}
