import AppKit

/// The pulsing dot at the end of a streaming reply.
///
/// The attachment reserves space and nothing more — it carries no image and no
/// view. That is deliberate on two counts:
///
///   * **Layout is what we actually want from it.** As a character in the text
///     it flows with the last glyph, wraps with it, and re-positions on every
///     reflow with nobody computing a frame. A floating view chasing
///     `boundingRect(for:)` has to be re-solved on every width change, font
///     change, and flush, and is wrong for one frame each time.
///   * **A view provider would never render here.** `NSTextAttachmentViewProvider`
///     hosts its view in an `NSTextView`; ``MarkdownTextBlockView`` draws
///     fragments into its own context, so the view would be created and never
///     shown. ChatGPT can use the view route because its block *is* a text
///     view; ours is not, so we paint the circle in ``MarkdownTextBlockView``
///     instead and drive its opacity from the display link that is already
///     running for the fade.
final class TypingDotAttachment: NSTextAttachment {

    public static let diameter: CGFloat = 7

    /// Full fade cycle. Slow enough to read as breathing rather than blinking.
    public static let pulseDuration: CFTimeInterval = 1.1

    /// Opacity floor — the dot dims, it does not disappear.
    public static let minimumOpacity: CGFloat = 0.25

    public var color: NSColor = .textColor

    public convenience init(color: NSColor, pointSize: CGFloat) {
        self.init(data: nil, ofType: nil)
        self.color = color
        // An attachment's origin sits on the baseline, so a positive y raises
        // it. Centring near the x-height reads as part of the line rather than
        // as a subscript.
        let size = Self.diameter
        bounds = CGRect(
            x: 0,
            y: (pointSize * 0.30 - size / 2).rounded(),
            width: size,
            height: size
        )
    }

    /// Opacity at a given moment in the pulse.
    ///
    /// Cosine rather than a linear triangle: the dot lingers at both ends,
    /// which is what separates breathing from blinking.
    public static func opacity(at time: CFTimeInterval) -> CGFloat {
        let phase = (time.truncatingRemainder(dividingBy: pulseDuration)) / pulseDuration
        let wave = (1 + cos(2 * Double.pi * phase)) / 2   // 1 → 0 → 1
        return minimumOpacity + (1 - minimumOpacity) * CGFloat(wave)
    }
}
