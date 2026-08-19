import AppKit
import SwiftMath

/// A rendered inline formula, sized and positioned as one character.
///
/// ## Why an image and not a view
///
/// ``TypingDotAttachment`` explains the constraint this shares:
/// `NSTextAttachmentViewProvider` hosts its view inside an `NSTextView`, and
/// ``MarkdownTextBlockView`` draws fragments into its own context instead — a
/// hosted view would be created and never shown. ChatGPT can take the view
/// route because its message block *is* a text view; ours is not. So the
/// formula is rasterised once and carried as an image, which TextKit lays out
/// and draws like any other attachment: it flows with the sentence, wraps with
/// it, and needs nobody to chase its frame.
///
/// ## Why the colour is part of the cache key
///
/// `LaTeXMarkdownView` records what happens when it is not: MarkdownUI's
/// inline image provider keyed its cache on the parsed inline nodes, so an
/// appearance change never re-fired it, and since the glyph colour is baked
/// into the bitmap the reader was left with black-on-black formulas after
/// switching to Dark. Keying on the colour and the point size means a theme
/// change simply misses the cache.
enum InlineMathImage {

    private struct Key: Hashable {
        let latex: String
        let pointSize: CGFloat
        /// The *resolved* glyph colour, not the `NSColor` it came from.
        ///
        /// This is the whole point of the key and it was got wrong first
        /// time. The app renders with `.labelColor` (`TextKitMarkdownView`)
        /// or `.textColor` (`MarkdownOptions`) — dynamic catalog colours,
        /// which compare equal to themselves and hash equal in *every*
        /// appearance. `label.textColor` resolves against the current
        /// appearance, so keying on the `NSColor` handed a bitmap baked with
        /// near-black glyphs straight back in Dark: exactly the black-on-black
        /// failure ``LaTeXMarkdownView`` records, reproduced by the fix meant
        /// to prevent it. Components resolve; the object does not.
        let components: [CGFloat]
    }

    private static func key(
        latex: String, pointSize: CGFloat, color: NSColor
    ) -> Key {
        Key(
            latex: latex,
            pointSize: pointSize,
            components: color.cgColor.components ?? []
        )
    }

    /// Bounded so a long chat cannot grow it without limit. Formulas repeat
    /// far more often than not — the same symbol recurs down a derivation —
    /// so even a small cache carries most of the traffic.
    private static let capacity = 256
    // Main-actor state: rasterising drives AppKit, so every caller is already
    // here and the cache needs no lock of its own.
    @MainActor private static var cache: [Key: NSImage] = [:]
    @MainActor private static var order: [Key] = []

    /// Rasterise `latex`, or return the cached bitmap for this exact
    /// appearance. `nil` when SwiftMath cannot parse the body, which the
    /// caller renders as its original `$…$` text rather than as a gap.
    @MainActor
    static func image(
        latex: String, pointSize: CGFloat, color: NSColor
    ) -> NSImage? {
        // Bridged before it becomes the key, so two spellings of the same
        // formula — `\mod` and a registered `\bmod` — share one bitmap.
        let source = LaTeXCompatibility.normalized(latex)
        let key = Self.key(latex: source, pointSize: pointSize, color: color)
        if let hit = cache[key] { return hit }

        let label = MTMathUILabel()
        label.latex = source
        // `.text` rather than `.display`: inline math sits on the sentence's
        // baseline at the sentence's size. `MathView` makes the same call and
        // adds two points for display math only.
        label.labelMode = .text
        label.fontSize = pointSize
        label.textColor = MTColor(cgColor: color.cgColor) ?? MTColor.black
        label.textAlignment = .left
        // An unparseable body leaves `error` set; drawing it anyway paints
        // SwiftMath's own red diagnostic into the reader's sentence.
        guard label.error == nil else { return nil }

        // `fittingSize`, not `intrinsicContentSize` — #131 established that on
        // macOS `MTMathUILabel` overrides the former and leaves the latter
        // returning AppKit's no-intrinsic-metric sentinel. `MathView` measures
        // the same way, and for the same reason.
        label.invalidateIntrinsicContentSize()
        let size = label.fittingSize
        guard size.width > 0, size.height > 0,
              size.width.isFinite, size.height.isFinite else { return nil }

        label.frame = CGRect(origin: .zero, size: size)
        // `cacheDisplay(in:to:)` rather than rendering the layer directly.
        // `MTMathUILabel.draw` force-unwraps a display list that only
        // `_layoutSubviews` builds, and that runs from `layout()` — rendering
        // the layer of a label AppKit has never laid out crashes inside
        // SwiftMath rather than returning an empty bitmap.
        label.layoutSubtreeIfNeeded()
        guard let rep = label.bitmapImageRepForCachingDisplay(in: label.bounds) else {
            return nil
        }
        label.cacheDisplay(in: label.bounds, to: rep)
        let image = NSImage(size: size)
        image.addRepresentation(rep)

        store(image, for: key)
        return image
    }

    @MainActor private static func store(_ image: NSImage, for key: Key) {
        cache[key] = image
        order.append(key)
        guard order.count > capacity else { return }
        let evicted = order.removeFirst()
        cache.removeValue(forKey: evicted)
    }

    /// Test seam — the cache is process-wide, so a suite that renders at
    /// several sizes would otherwise read another case's bitmaps.
    @MainActor static func resetCache() {
        cache.removeAll()
        order.removeAll()
    }
}

/// The attachment that carries one formula.
///
/// Holds the LaTeX alongside the bitmap so a formula can be told apart from
/// any other attachment. Nothing in the renderer consults it yet — fading and
/// hit testing do not — so today it serves identification in tests and in the
/// debugger.
final class InlineMathAttachment: NSTextAttachment {
    let latex: String

    init(latex: String, image: NSImage, pointSize: CGFloat) {
        self.latex = latex
        super.init(data: nil, ofType: nil)
        self.image = image
        // An attachment's origin sits on the baseline. SwiftMath centres its
        // rendering on the math axis rather than the text baseline, so without
        // a nudge the formula rides high — a fraction ends up looking like a
        // superscript. Dropping it by the descender puts the axis back on the
        // line the words are standing on.
        let descender = NSFont.systemFont(ofSize: pointSize).descender
        bounds = CGRect(
            x: 0,
            y: descender.rounded(),
            width: image.size.width,
            height: image.size.height
        )
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}
