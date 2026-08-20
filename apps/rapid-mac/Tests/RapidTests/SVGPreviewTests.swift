import AppKit
import Foundation
import Network
import Testing
@testable import Rapid

/// Which code blocks offer a preview, and what the preview is drawn at.
///
/// The rendering itself is AppKit's, so what is worth testing is the two
/// judgements around it: whether a block is an SVG document at all, and how a
/// document with no natural pixel size is fitted to a column.
@Suite("SVG preview")
struct SVGPreviewTests {

    private let svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
          <circle cx="50" cy="50" r="40" fill="#3b82f6"/>
        </svg>
        """

    // MARK: - What counts as previewable

    @Test("An SVG document is offered a preview", arguments: [
        "svg", "xml", "html", "SVG", nil,
    ])
    func svgIsPreviewable(_ language: String?) {
        #expect(SVGPreview.looksLikeSVG(code: svg, language: language))
    }

    /// Prose or source that merely mentions the string is about SVG, not SVG.
    ///
    /// Every case here carries a language the filter *accepts*, so the
    /// open-with-a-tag rule is the only thing that can reject them. An earlier
    /// version of this test used `swift` and `markdown` tags, which a language
    /// allowlist also rejected — so deleting either guard left it green and it
    /// proved nothing about either.
    @Test("Code that only mentions SVG is not previewable", arguments: [
        #"let markup = "<svg viewBox=\"0 0 10 10\"/>""#,
        "Here is the diagram: <svg viewBox=\"0 0 10 10\"/>",
        "# How to draw an <svg> by hand",
    ])
    func mentioningSVGIsNotEnough(_ code: String) {
        #expect(!SVGPreview.looksLikeSVG(code: code, language: "xml"))
    }

    /// A mis-tagged document still previews. Models label SVG as whatever
    /// they feel like, and refusing on the tag would hide a working preview.
    @Test("A mis-tagged document is still previewable")
    func misTaggedIsStillPreviewable() {
        #expect(SVGPreview.looksLikeSVG(code: svg, language: "python"))
    }

    @Test("Nothing to preview", arguments: [
        "", "   ", "print(\"hello\")", "<html><body>hi</body></html>",
    ])
    func nonSVGIsNotPreviewable(_ code: String) {
        #expect(!SVGPreview.looksLikeSVG(code: code, language: nil))
    }

    /// The parse is synchronous on the main thread, and a model can emit a
    /// megabyte of path data.
    @Test("An oversized document is refused before parsing")
    func oversizedIsRefused() {
        let huge = "<svg>" + String(repeating: "x", count: SVGPreview.maximumSourceBytes) + "</svg>"
        #expect(!SVGPreview.looksLikeSVG(code: huge, language: "svg"))
    }

    // MARK: - Rendering

    @MainActor
    @Test("A valid document renders")
    func validDocumentRenders() {
        let image = SVGPreview.image(from: svg)
        #expect(image != nil)
        #expect(image?.size == CGSize(width: 100, height: 100))
    }

    /// The usual mid-stream state: the fence has not closed and the document
    /// is half-written. Nil, so the button never appears for it.
    @MainActor
    @Test("An incomplete or broken document renders nothing", arguments: [
        #"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circ"#,
        "<svg",
        "not markup at all",
        "",
    ])
    func brokenDocumentRendersNothing(_ code: String) {
        #expect(SVGPreview.image(from: code) == nil)
    }

    // MARK: - Fitting a vector to a column

    @Test("A large document is scaled down to the column")
    func largeIsScaledToColumn() {
        let size = SVGPreview.drawSize(for: CGSize(width: 800, height: 400), inWidth: 400)
        #expect(size == CGSize(width: 400, height: 200))
    }

    /// A 24-point icon blown across a 700-point column reads as a mistake
    /// rather than as a preview.
    @Test("A small document is not blown up")
    func smallIsNotUpscaled() {
        let size = SVGPreview.drawSize(for: CGSize(width: 24, height: 24), inWidth: 700)
        #expect(size == CGSize(width: 24, height: 24))
    }

    @Test("A very tall document is capped rather than filling the transcript")
    func tallIsCapped() {
        let size = SVGPreview.drawSize(for: CGSize(width: 100, height: 4_000), inWidth: 400)
        #expect(size.height <= SVGPreview.maximumHeight)
        // Aspect is kept while capping.
        #expect(abs(size.width / size.height - 100.0 / 4_000.0) < 0.01)
    }

    @Test("Degenerate sizes produce nothing to draw", arguments: [
        CGSize(width: 0, height: 100), CGSize(width: 100, height: 0), CGSize.zero,
    ])
    func degenerateSizes(_ imageSize: CGSize) {
        #expect(SVGPreview.drawSize(for: imageSize, inWidth: 400) == .zero)
    }

    @Test("A zero-width column produces nothing to draw")
    func zeroWidthColumn() {
        #expect(SVGPreview.drawSize(for: CGSize(width: 100, height: 100), inWidth: 0) == .zero)
    }
}

/// The properties of AppKit's own SVG support this feature is built on.
///
/// These are not testing Apple's code for its own sake — each one is a claim
/// the design rests on, and if any stops holding, the right response is a
/// different design rather than a patch.
@MainActor
@Suite("AppKit SVG rendering assumptions")
struct AppKitSVGAssumptionTests {

    /// Vector, not a bitmap at the document's nominal size. If this became an
    /// upscale, a preview on a Retina display would be visibly soft and the
    /// feature would need a real renderer.
    @Test("The document re-rasterises at the size it is drawn")
    func rendersAsVector() throws {
        let triangle = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
              <path d="M0 0 L10 10 L0 10 Z" fill="black"/>
            </svg>
            """
        let image = try #require(SVGPreview.image(from: triangle))

        func edgeWidth(at pixels: Int) -> Int {
            let rep = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
            )!
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            image.draw(in: NSRect(x: 0, y: 0, width: CGFloat(pixels), height: CGFloat(pixels)))
            NSGraphicsContext.restoreGraphicsState()
            return (0..<pixels).count {
                guard let colour = rep.colorAt(x: $0, y: pixels / 2) else { return false }
                return colour.alphaComponent > 0.05 && colour.alphaComponent < 0.95
            }
        }
        // An upscaled 10pt bitmap would smear this edge across many pixels at
        // 512; a vector re-rasterise keeps it at one.
        #expect(edgeWidth(at: 64) <= 2)
        #expect(edgeWidth(at: 512) <= 2)
    }

    /// The claim that makes this feature safe enough to ship without a
    /// sandbox: a model-authored document cannot reach the network. Measured
    /// here against a URL that would be recorded if it were ever requested.
    ///
    /// If this ever starts fetching, a model could exfiltrate the transcript
    /// through an `<image>` URL, and the feature would need a network policy.
    @Test("A remote reference is not fetched")
    func remoteReferenceIsNotFetched() throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let svg = """
            <svg xmlns="http://www.w3.org/2000/svg" \
            xmlns:xlink="http://www.w3.org/1999/xlink" \
            viewBox="0 0 100 100" width="100" height="100">
              <image xlink:href="http://127.0.0.1:\(probe.port)/leak.png" width="100" height="100"/>
            </svg>
            """
        let image = try #require(SVGPreview.image(from: svg))
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 100, pixelsHigh: 100,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        )!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(x: 0, y: 0, width: 100, height: 100))
        NSGraphicsContext.restoreGraphicsState()

        #expect(probe.requestCount == 0, "the SVG renderer reached the network")
    }
}

/// A loopback listener that counts inbound connections.
///
/// Deliberately not an HTTP server: the assertion is "nothing connected at
/// all", and accepting the socket is already enough to know that something
/// did. Keeping it to a listener means the test cannot be fooled by a reply
/// it wrote itself.
final class LocalRequestProbe: @unchecked Sendable {
    private let listener: NWListener
    let port: UInt16
    private let lock = NSLock()
    private var count = 0

    var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    init?() {
        guard let listener = try? NWListener(using: .tcp, on: .any) else { return nil }
        self.listener = listener

        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed: ready.signal()
            default: break
            }
        }
        listener.start(queue: .global())
        guard ready.wait(timeout: .now() + 2) == .success,
              let resolved = listener.port?.rawValue else {
            listener.cancel()
            return nil
        }
        port = resolved

        // Installed after `port` is assigned: the handler captures `self`, and
        // Swift will not let an escaping closure see a partially initialised
        // instance.
        let lock = self.lock
        listener.newConnectionHandler = { [weak self] connection in
            lock.lock()
            self?.count += 1
            lock.unlock()
            connection.cancel()
        }
    }

    func stop() { listener.cancel() }
}

/// The wiring: does the button appear, and does the card grow when it is
/// pressed. The drawing itself is AppKit's, but the row height is ours — and
/// a preview the block stack has not made room for is a preview drawn over
/// whatever comes next.
@MainActor
@Suite("SVG preview in a code block")
struct MarkdownCodeBlockPreviewTests {

    private let svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" width="100" height="50">
          <rect width="100" height="50" fill="teal"/>
        </svg>
        """

    private func block(_ code: String, _ language: String?) -> MarkdownCodeBlockView {
        let view = MarkdownCodeBlockView(options: MarkdownOptions())
        view.frame = NSRect(x: 0, y: 0, width: 400, height: 200)
        view.configure(code: code, language: language, options: MarkdownOptions())
        return view
    }

    private func previewButton(in view: NSView) -> NSButton? {
        view.subviews.compactMap { $0 as? NSButton }
            .first { $0.accessibilityIdentifier() == "CodeBlock.Preview" }
    }

    @Test("A Swift block offers no preview")
    func swiftBlockHasNoButton() {
        let view = block("print(\"hi\")", "swift")
        #expect(previewButton(in: view)?.isHidden == true)
    }

    @Test("An SVG block offers a preview")
    func svgBlockHasButton() {
        let view = block(svg, "svg")
        #expect(previewButton(in: view)?.isHidden == false)
    }

    /// Mid-stream the document is half-written. Offering a button that would
    /// render nothing is worse than offering none.
    @Test("A half-streamed document offers no preview yet")
    func partialDocumentHasNoButton() {
        let view = block("<svg xmlns=\"http://www.w3.org/2000/svg\"><rect wid", "svg")
        #expect(previewButton(in: view)?.isHidden == true)
    }

    /// The load-bearing one: the block stack lays rows out from
    /// `height(forWidth:)`, so a card whose body changed but whose height did
    /// not is a card drawn over whatever comes next.
    ///
    /// Preview is a *mode*. Two documents with the same `viewBox` but wildly
    /// different source lengths must preview at the same height — that is
    /// true only if the picture replaced the source, and false for any
    /// implementation that appends it.
    @Test("The preview replaces the source rather than stacking under it")
    func previewReplacesSource() throws {
        let padded = svg.replacingOccurrences(
            of: "<rect", with: String(repeating: "<!-- padding -->\n  ", count: 20) + "<rect"
        )
        let short = block(svg, "svg")
        let long = block(padded, "svg")

        let shortSource = short.height(forWidth: 400)
        let longSource = long.height(forWidth: 400)
        #expect(longSource > shortSource, "the padded document should have a taller source")

        for view in [short, long] {
            let button = try #require(previewButton(in: view))
            _ = button.target?.perform(button.action, with: button)
        }
        #expect(short.height(forWidth: 400) == long.height(forWidth: 400))
        // And the long one got shorter, which appending could never do.
        #expect(long.height(forWidth: 400) < longSource)

        // Pressing again puts each source back.
        for view in [short, long] {
            let button = try #require(previewButton(in: view))
            _ = button.target?.perform(button.action, with: button)
        }
        #expect(short.height(forWidth: 400) == shortSource)
        #expect(long.height(forWidth: 400) == longSource)
    }

    @Test("The button names the mode it switches to")
    func buttonTitleTracksState() throws {
        let view = block(svg, "svg")
        let button = try #require(previewButton(in: view))
        #expect(button.title == "Preview")
        _ = button.target?.perform(button.action, with: button)
        #expect(button.title == "Code")
    }

    /// Re-configuring with a different language — which happens on every
    /// streaming flush — must not leave a preview open on a block that is no
    /// longer an SVG.
    @Test("A block that stops being an SVG closes its preview")
    func previewClosesWhenNoLongerSVG() throws {
        let view = block(svg, "svg")
        let button = try #require(previewButton(in: view))
        _ = button.target?.perform(button.action, with: button)
        let expanded = view.height(forWidth: 400)

        view.configure(code: "print(\"hi\")", language: "swift", options: MarkdownOptions())
        #expect(button.isHidden)
        #expect(view.height(forWidth: 400) < expanded)
        // Exactly a fresh block's height — not merely "shorter than before".
        // A leaked `isShowingPreview` still reserves the image's 50 points,
        // and the swift body being shorter than the SVG's would hide that
        // behind a passing inequality.
        let fresh = block("print(\"hi\")", "swift")
        #expect(view.height(forWidth: 400) == fresh.height(forWidth: 400))
    }
}
