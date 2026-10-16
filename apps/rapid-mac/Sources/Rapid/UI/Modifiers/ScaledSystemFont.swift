import SwiftUI

/// A drop-in replacement for `.font(.system(size:))` that DOES scale
/// with Dynamic Type.
///
/// `Font.system(size:)` is a fixed-pixel rail: it ignores the
/// `\.dynamicTypeSize` environment entirely (see ``DynamicTypeClamp``
/// — its doc-comment calls this out as the un-migrated gap). Any label
/// pinned that way stays the same size no matter how large the user
/// sets their system text, which locks low-vision users out of the
/// app's own content.
///
/// This modifier keeps the **exact point size at the default text
/// size** (so there is no visual regression on a normal setup) while
/// scaling proportionally to `relativeTo` as the user enlarges text,
/// via `@ScaledMetric`. Because `@ScaledMetric` reads
/// `\.dynamicTypeSize`, a subtree wrapped in ``rapidChatDynamicTypeClamp()``
/// also inherits that clamp's ceiling — the two compose cleanly.
///
/// Usage: replace `.font(.system(size: 13))` with `.scaledSystemFont(13)`,
/// and `.font(.system(size: 11, weight: .medium, design: .monospaced))`
/// with `.scaledSystemFont(11, weight: .medium, design: .monospaced)`.
/// Icon glyphs (`Image(systemName:).font(.system(size:))`) are
/// intentionally left on the fixed rail — they are decorative and
/// usually `accessibilityHidden`.
private struct ScaledSystemFontModifier: ViewModifier {
    @ScaledMetric private var size: CGFloat
    private let weight: Font.Weight
    private let design: Font.Design

    init(
        size: CGFloat,
        relativeTo textStyle: Font.TextStyle,
        weight: Font.Weight,
        design: Font.Design
    ) {
        // `wrappedValue` is the size at the default Dynamic Type size,
        // so an unchanged system keeps the exact pre-migration look.
        self._size = ScaledMetric(wrappedValue: size, relativeTo: textStyle)
        self.weight = weight
        self.design = design
    }

    func body(content: Content) -> some View {
        content.font(.rapid(size: size, weight: weight, design: design))
    }
}

extension View {
    /// A `.font(.system(size:))` that scales with Dynamic Type.
    ///
    /// - Parameters:
    ///   - size: The point size at the *default* text size (preserved
    ///     exactly, so no regression on a normal system).
    ///   - textStyle: The scaling curve to track. Pick the semantic
    ///     style closest to the role: `.body` for reading content,
    ///     `.caption`/`.caption2` for chips and metadata, `.title`-ish
    ///     for display text.
    ///   - weight / design: Same as `Font.system(size:weight:design:)`.
    func scaledSystemFont(
        _ size: CGFloat,
        relativeTo textStyle: Font.TextStyle = .body,
        weight: Font.Weight = .regular,
        design: Font.Design = .default
    ) -> some View {
        modifier(
            ScaledSystemFontModifier(
                size: size,
                relativeTo: textStyle,
                weight: weight,
                design: design
            )
        )
    }
}
