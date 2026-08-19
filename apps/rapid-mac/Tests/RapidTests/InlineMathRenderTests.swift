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

    /// The case the app actually ships, and the one the first version of the
    /// key missed entirely.
    ///
    /// `appearanceMissesTheCache` above uses `.black` and `.white` — two
    /// static colours that genuinely differ, so it passes whether the key
    /// holds the `NSColor` or its resolved components. But the renderer is
    /// handed `.labelColor` (`TextKitMarkdownView`) or `.textColor`
    /// (`MarkdownOptions`), and a dynamic catalog colour compares equal to
    /// itself and hashes equal in *every* appearance. Keying on the object
    /// therefore served the Light bitmap back in Dark — the black-on-black
    /// failure `LaTeXMarkdownView` documents, reproduced by the key added to
    /// prevent it.
    @Test("A dynamic colour still misses the cache across appearances")
    func dynamicColourMissesAcrossAppearances() {
        InlineMathImage.resetCache()
        var light: NSImage?
        var dark: NSImage?
        NSAppearance(named: .aqua)?.performAsCurrentDrawingAppearance {
            light = InlineMathImage.image(latex: "x", pointSize: 14, color: .labelColor)
        }
        NSAppearance(named: .darkAqua)?.performAsCurrentDrawingAppearance {
            dark = InlineMathImage.image(latex: "x", pointSize: 14, color: .labelColor)
        }
        #expect(light != nil)
        #expect(dark != nil)
        #expect(light !== dark, "the Light bitmap was served back in Dark")
    }

    /// An unparseable body renders as the `$…$` the author typed. Returning a
    /// broken bitmap would paint SwiftMath's red diagnostic into the sentence.
    ///
    /// This asserts the contract, not the `label.error` guard specifically:
    /// review showed that deleting that guard leaves this passing, because a
    /// failed parse also leaves `fittingSize` at zero and the size guard
    /// declines first. The explicit error check stays as the readable reason,
    /// with the size check behind it.
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
        renderer.setBlocks(blocks)
        #expect(renderer.accessibleText == "The value $x_1$ matters.")
        #expect(!renderer.accessibleText.contains("\u{FFFC}"))
    }
}
