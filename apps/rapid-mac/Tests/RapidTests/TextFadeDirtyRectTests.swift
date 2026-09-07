import AppKit
import Testing
@testable import Rapid

/// The display-link must invalidate the fading tail, not the whole message.
/// Draw-side clipping cannot help if its caller always marks all bounds dirty.
@Suite("Text fade dirty rect")
@MainActor
struct TextFadeDirtyRectTests {

    private final class RecordingView: NSView {
        var invalidatedRects: [NSRect] = []

        override func setNeedsDisplay(_ invalidRect: NSRect) {
            invalidatedRects.append(invalidRect)
            super.setNeedsDisplay(invalidRect)
        }
    }

    @Test("A tail fade invalidates less than the full document")
    func tailFadeUsesNarrowInvalidation() throws {
        var options = MarkdownOptions.assistantTranscript()
        options.textColor = .black
        let renderer = MarkdownTextRenderer(options: options)
        let animator = TextFadeAnimator(
            textLayoutManager: renderer.textLayoutManager,
            textContentStorage: renderer.textContentStorage,
            animationState: TextFadeAnimationState()
        )
        animator.textColor = .black
        animator.contentLengthProvider = { renderer.proseLength }

        let view = RecordingView(frame: NSRect(x: 0, y: 0, width: 700, height: 20_000))
        animator.attach(to: view)

        let prefix = (0..<60).map { index in
            MarkdownItem.TextBlock(
                runs: [InlineRun(
                    text: "Paragraph \(index). " + String(repeating: "word ", count: 40)
                )],
                kind: .paragraph
            )
        }
        renderer.setBlocks(prefix)
        _ = renderer.measureHeight(width: view.bounds.width)
        animator.markAllRevealed()

        var grown = prefix
        grown.append(.init(runs: [InlineRun(text: "new tail")], kind: .paragraph))
        renderer.setBlocks(grown)
        let documentHeight = renderer.measureHeight(width: view.bounds.width)
        animator.testing_setClock { 1.0 }
        animator.contentDidGrow()

        view.invalidatedRects.removeAll()
        animator.testing_tick(at: 1.05)

        let invalidated = try #require(view.invalidatedRects.last)
        #expect(invalidated.height < documentHeight / 2)
        #expect(invalidated.maxY > documentHeight / 2)
    }
}
