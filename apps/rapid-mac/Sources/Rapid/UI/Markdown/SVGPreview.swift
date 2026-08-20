import AppKit

/// Turns the SVG a model wrote into something a reader can look at.
///
/// ## Why there is no library here
///
/// AppKit renders SVG itself: `NSImage(data:)` accepts an SVG document and
/// returns an image backed by a vector representation, which re-rasterises at
/// whatever size it is drawn. Measured on this codebase's deployment target
/// family — a triangle's hypotenuse has a one-pixel antialiased edge at 64pt
/// and still one pixel at 512pt, so it is genuinely vector and not an upscaled
/// bitmap. Gradients, `transform`, `clipPath`, `<text>`, dashes, embedded
/// `<style>` CSS and `feGaussianBlur` all render.
///
/// The alternative considered was exyte/SVGView, which is what ChatGPT ships
/// for its *icons* (its code-block preview is a sandboxed WebView, a much
/// larger machine). That library is MIT and would work, but its last
/// functional commit was in 2023 — everything since is README edits — and
/// vendoring three thousand lines of unmaintained parser to do what one
/// framework call already does is a poor trade.
///
/// ## What this deliberately does not do
///
/// It does not execute anything. There is no script engine behind
/// `NSImage(data:)`, and a remote `<image xlink:href="http://…">` is **not**
/// fetched — measured against a local HTTP server that received no request
/// while the document drew. So a model-authored SVG cannot phone home, and
/// this needs no sandbox, no permission prompt and no network policy. That is
/// the whole reason this feature is small enough to be worth having.
enum SVGPreview {

    /// Widest the preview will draw, so a `viewBox` of 4000 does not produce a
    /// wall of image.
    static let maximumHeight: CGFloat = 420

    /// Refuse to even attempt anything larger. A model can emit a megabyte of
    /// path data, and the parse is synchronous on the main thread.
    static let maximumSourceBytes = 512 * 1024

    /// Is this code block worth attempting a preview for?
    ///
    /// A cheap pre-filter, not a decision: ``image(from:)`` is the authority,
    /// and this only exists so a plain Swift block does not pay for a parse on
    /// every streaming flush. Two conditions, both about the content.
    ///
    /// The language tag is deliberately ignored. Models label SVG as `svg`,
    /// `xml`, `html`, or nothing at all, and an allowlist was tried and
    /// removed: the only blocks it excluded were mis-tagged documents that
    /// render perfectly well, so it bought nothing that `NSImage` returning
    /// nil does not already buy, and cost a preview on every block a model
    /// labelled wrong.
    nonisolated static func looksLikeSVG(code: String, language: String?) -> Bool {
        guard code.utf8.count <= maximumSourceBytes else { return false }
        let head = code.prefix(2_048)
        guard head.range(of: "<svg", options: .caseInsensitive) != nil else { return false }
        // Prose or source that merely mentions `<svg` is not a document. The
        // document has to *open* with a tag — an XML prolog, comment or
        // doctype all satisfy that, a `let markup = "…"` does not.
        return code.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<")
    }

    /// The rendered document, or nil when it will not parse.
    ///
    /// Nil is the normal outcome mid-stream: half an `<svg` is not a document,
    /// and `NSImage` returns nil for it rather than drawing something partial.
    /// The caller shows the code and nothing else until it completes.
    @MainActor
    static func image(from code: String) -> NSImage? {
        guard code.utf8.count <= maximumSourceBytes else { return nil }
        guard let image = NSImage(data: Data(code.utf8)) else { return nil }
        let size = image.size
        guard size.width > 0, size.height > 0,
              size.width.isFinite, size.height.isFinite else { return nil }
        return image
    }

    /// The size to draw `image` at inside `width`, preserving its aspect and
    /// never exceeding ``maximumHeight``.
    ///
    /// Vector art has no natural pixel size, so it is scaled to the column
    /// rather than centred at its nominal dimensions — except that it is never
    /// scaled *up* past its own size, because a 24-point icon blown across a
    /// 700-point column looks like a mistake rather than a preview.
    nonisolated static func drawSize(for imageSize: CGSize, inWidth width: CGFloat) -> CGSize {
        guard imageSize.width > 0, imageSize.height > 0, width > 0 else { return .zero }
        let scale = min(
            min(width / imageSize.width, maximumHeight / imageSize.height),
            1
        )
        return CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
    }
}
