import SwiftUI

/// A text field wearing the product's own focus treatment.
///
/// `.textFieldStyle(.roundedBorder)` draws AppKit's bezel and, on
/// focus, the system's bright blue ring — the single loudest blue in
/// the app, on a surface whose brand colour is amber. This replaces it
/// with `.plain` inside a container we draw:
///
///   * 1px hairline at rest,
///   * 2px amber border on focus (focus is a primary-attention signal,
///     so it gets the primary brand colour),
///   * no glow, no bezel,
///   * the standard input radius and a 36pt control height.
///
/// Behaviour is deliberately untouched: it is still a plain `TextField`,
/// so first-responder handling, Return via `onSubmit`, Escape via the
/// presenter's `.cancelAction`, text selection, and VoiceOver all work
/// exactly as they did. The focus ring is suppressed with
/// `.focusEffectDisabled()` only because we draw an equivalent one —
/// focusability itself is unchanged.
struct RapidTextField: View {
    let placeholder: String
    @Binding var text: String
    /// Called on Return. The presenter still owns the default-action
    /// button; this just makes Return-in-field do the same thing.
    var onSubmit: (() -> Void)? = nil

    @FocusState private var focused: Bool

    var body: some View {
        // ax-exempt: the caller owns the surface-specific identifier on this wrapper
        TextField(placeholder, text: $text)
            .textFieldStyle(.plain)
            .font(RapidFont.body)
            .focused($focused)
            .focusEffectDisabled()
            .onSubmit { onSubmit?() }
            .padding(.horizontal, RapidTheme.Space.md - 2)
            .frame(height: RapidTheme.ControlHeight.large)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .strokeBorder(
                        focused ? RapidTheme.focusRing : RapidTheme.hairlineStrong,
                        lineWidth: focused ? 2 : 1
                    )
            )
            .rapidAnimation(RapidMotion.quick, value: focused)
            .contentShape(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
            )
            // Clicking anywhere in the container focuses the field, not
            // just the glyph run — the bezel used to provide this.
            .onTapGesture { focused = true }
    }
}
