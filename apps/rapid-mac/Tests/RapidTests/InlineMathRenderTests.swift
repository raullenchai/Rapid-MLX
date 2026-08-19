import AppKit
import Testing
@testable import Rapid

/// Inline math has to actually reach pixels.
///
/// Rasterisation fails quietly in several ways — a zero-size label, a body
/// SwiftMath cannot parse, a measurement taken from the wrong property — and
/// every one of them ends as a formula-shaped hole in a sentence rather than
/// an error. #131 already lost a release to the last of those: on macOS
/// `MTMathUILabel` overrides `fittingSize` and leaves `intrinsicContentSize`
/// returning AppKit's no-intrinsic-metric sentinel, so measuring the obvious
/// way laid every formula out at zero size.
@Suite("Inline math rendering")
@MainActor
struct InlineMathRenderTests {

    private func inkCount(_ image: NSImage) -> Int {
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return 0 }
        var ink = 0
        for y in 0..<rep.pixelsHigh {
            for x in 0..<rep.pixelsWide {
                if let colour = rep.colorAt(x: x, y: y), colour.alphaComponent > 0.1 { ink += 1 }
            }
        }
        return ink
    }

    @Test("A formula rasterises to a non-empty bitmap")
    func formulaHasPixels() {
        InlineMathImage.resetCache()
        for latex in ["x_1", "\\frac{1}{2}", "e^{i\\pi}"] {
            guard let image = InlineMathImage.image(
                latex: latex, pointSize: 14, color: .black
            ) else {
                Issue.record("\(latex) produced no image at all")
                continue
            }
            #expect(image.size.width > 1, "\(latex) is \(image.size.width)pt wide")
            #expect(image.size.height > 1, "\(latex) is \(image.size.height)pt tall")
            #expect(inkCount(image) > 0, "\(latex) rasterised to a blank bitmap")
        }
    }

    /// A bigger formula must occupy more room than a smaller one — this is
    /// what catches a measurement that returns a constant or a sentinel.
    @Test("Size follows the formula")
    func sizeTracksContent() {
        InlineMathImage.resetCache()
        let small = InlineMathImage.image(latex: "x", pointSize: 14, color: .black)
        let large = InlineMathImage.image(latex: "\\frac{a+b}{c+d}", pointSize: 14, color: .black)
        guard let small, let large else {
            Issue.record("rasterisation returned nil for a valid formula")
            return
        }
        #expect(large.size.height > small.size.height)
    }

    /// The bug this key exists to prevent: glyph colour is baked into the
    /// bitmap, so a cache that ignores colour serves black-on-black formulas
    /// after a switch to Dark. `LaTeXMarkdownView` documents that exact
    /// failure in the MarkdownUI path.
    @Test("Colour and size are part of the cache key")
    func appearanceMissesTheCache() {
        InlineMathImage.resetCache()
        let black = InlineMathImage.image(latex: "x", pointSize: 14, color: .black)
        let white = InlineMathImage.image(latex: "x", pointSize: 14, color: .white)
        let bigger = InlineMathImage.image(latex: "x", pointSize: 22, color: .black)
        #expect(black !== white, "a colour change reused the previous bitmap")
        #expect(black !== bigger, "a size change reused the previous bitmap")
        // …but an identical request must still hit.
        #expect(black === InlineMathImage.image(latex: "x", pointSize: 14, color: .black))
    }

    /// An unparseable body renders as the `$…$` the author typed. Returning a
    /// broken bitmap would paint SwiftMath's red diagnostic into the sentence.
    @Test("An unparseable formula declines rather than drawing an error")
    func unparseableDeclines() {
        InlineMathImage.resetCache()
        #expect(InlineMathImage.image(latex: "\\notacommand{", pointSize: 14, color: .black) == nil)
    }

    /// End to end: a sentence with inline math must put an attachment into the
    /// attributed string, carrying the formula.
    @Test("The renderer emits an attachment for a math run")
    func rendererEmitsAttachment() {
        InlineMathImage.resetCache()
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let renderer = MarkdownTextRenderer(options: options)
        let blocks = MarkdownCompiler()
            .compile("The value $x_1$ matters.").items
            .compactMap { item -> MarkdownItem.TextBlock? in
                if case .text(let block) = item { return block }
                return nil
            }
        let string = renderer.attributedString(for: blocks)

        var found: [String] = []
        string.enumerateAttribute(
            .attachment, in: NSRange(location: 0, length: string.length)
        ) { value, _, _ in
            if let math = value as? InlineMathAttachment { found.append(math.latex) }
        }
        #expect(found == ["x_1"], "attachments found: \(found)")
    }
}
