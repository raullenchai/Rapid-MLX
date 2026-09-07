import AppKit
import Testing
@testable import Rapid

/// `draw(_:)` must scale with what changed, not with the whole message.
///
/// The fade animator marks the view dirty on every display-link frame while
/// text streams. If `draw` ignores `dirtyRect` and redraws every fragment,
/// per-frame cost grows with the length of the answer — so a long reply is
/// slowest exactly when the machine is already busy decoding it.
@Suite("Markdown dirty-rect clipping")
@MainActor
struct MarkdownDirtyRectClippingTests {

    private func hostedView(paragraphs: Int) -> MarkdownTextBlockView {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .textColor
        let view = MarkdownTextBlockView(options: options)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 900),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        view.frame = NSRect(x: 0, y: 0, width: 700, height: 20_000)
        window.contentView?.addSubview(view)
        view.configure(
            blocks: (0..<paragraphs).map { i in
                .init(
                    runs: [InlineRun(
                        text: "Paragraph \(i). " + String(repeating: "word ", count: 40)
                    )],
                    kind: .paragraph
                )
            },
            options: options,
            streaming: true,
            fadeState: TextFadeAnimationState()
        )
        return view
    }

    /// Time one draw of a narrow strip — what a fade frame invalidates.
    private func msPerTailDraw(paragraphs: Int) -> Double {
        let view = hostedView(paragraphs: paragraphs)
        let tail = NSRect(
            x: 0, y: max(0, view.intrinsicContentSize.height - 24),
            width: 700, height: 24
        )
        let rep = view.bitmapImageRepForCachingDisplay(in: tail)!
        view.cacheDisplay(in: tail, to: rep)      // warm lazy layout

        let iterations = 30
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            view.setNeedsDisplay(tail)
            view.cacheDisplay(in: tail, to: rep)
        }
        return (CFAbsoluteTimeGetCurrent() - start) / Double(iterations) * 1000
    }

    /// The contract: redrawing one line must not get materially more
    /// expensive because the message above it got longer.
    ///
    /// Asserted as a ratio rather than an absolute millisecond figure so the
    /// test states the scaling property and does not fail on slower hardware.
    /// Without clipping this ratio tracked the paragraph ratio (12x); the
    /// bound below is loose enough for timing noise and still far under it.
    @Test("Tail redraw cost does not scale with document length")
    func tailDrawIsFlatInDocumentLength() {
        let short = msPerTailDraw(paragraphs: 5)
        let long = msPerTailDraw(paragraphs: 60)
        let ratio = long / max(short, 0.0001)

        #expect(
            ratio < 4,
            """
            redrawing a 24pt strip cost \(short) ms at 5 paragraphs and \
            \(long) ms at 60 (ratio \(ratio)). A ratio near the 12x document \
            ratio means draw() is walking the whole document again instead of \
            the dirty rect.
            """
        )
    }
}
