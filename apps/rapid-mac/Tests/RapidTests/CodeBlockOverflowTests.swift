import CoreGraphics
import Foundation
import Testing
@testable import Rapid

/// The fenced-code-block edge fade (2026-08 dogfood: long Python
/// lines stopped dead at the message-column edge and read as
/// truncated output).
///
/// The measured facts this is built on, taken headlessly through
/// `NSHostingView` at the real 720pt message column: the block's
/// horizontal `ScrollView` gives the code its full intrinsic width
/// — the bug's sample laid out at 822pt — so nothing was clipped,
/// and `showsIndicators: false` installed no `NSScroller`, so
/// nothing said so. The fade is the resting-state affordance; these
/// pin when it appears and, just as importantly, when it must not.
@Suite("CodeBlockOverflow — code block edge fade")
struct CodeBlockOverflowTests {

    private typealias Span = CodeBlockOverflow.ContentSpan
    private typealias Fade = CodeBlockOverflow.Fade

    /// The reported case: 822pt of code in a 720pt column, unscrolled.
    private static let dogfoodContent = Span(minX: 0, maxX: 822)
    private static let column: (min: CGFloat, max: CGFloat) = (0, 720)

    private func fade(
        _ content: Span,
        viewport: (min: CGFloat, max: CGFloat) = CodeBlockOverflowTests.column,
        maxFadeWidth: CGFloat = CodeBlockOverflow.maxFadeWidth
    ) -> Fade {
        CodeBlockOverflow.fade(
            content: content,
            viewportMinX: viewport.min,
            viewportMaxX: viewport.max,
            maxFadeWidth: maxFadeWidth
        )
    }

    // MARK: - The reported bug

    @Test("Code wider than the column fades at the trailing edge")
    func trailingFadeOnOverflow() {
        let result = fade(Self.dogfoodContent)
        #expect(result.trailing == CodeBlockOverflow.maxFadeWidth)
        // Nothing is hidden to the left yet — the block is unscrolled,
        // so the leading edge must stay crisp.
        #expect(result.leading == 0)
    }

    @Test("A code block that fits gets no fade at all")
    func noFadeWhenContentFits() {
        #expect(fade(Span(minX: 0, maxX: 600)).isEmpty)
        // Exactly flush is still a fit.
        #expect(fade(Span(minX: 0, maxX: 720)).isEmpty)
    }

    // MARK: - Scroll position

    @Test("Scrolling right moves the fade to the leading edge")
    func fadeFollowsScrollOffset() {
        // Scrolled fully right: content.maxX has walked down onto the
        // viewport's trailing edge, 102pt now hidden to the left.
        let atEnd = fade(Span(minX: -102, maxX: 720))
        #expect(atEnd.trailing == 0, "nothing left to reveal — the trailing fade must clear")
        #expect(atEnd.leading == CodeBlockOverflow.maxFadeWidth)

        // Mid-scroll: hidden on both sides, so both edges fade.
        let midway = fade(Span(minX: -51, maxX: 771))
        #expect(midway.leading == CodeBlockOverflow.maxFadeWidth)
        #expect(midway.trailing == CodeBlockOverflow.maxFadeWidth)
    }

    @Test("The fade ramps down over the last points of the scroll")
    func fadeRampsRatherThanPops() {
        // 10pt of code still hidden -> a 10pt fade, not a full-width
        // one that vanishes in a single frame at the end of the swipe.
        #expect(fade(Span(minX: 0, maxX: 730)).trailing == 10)
        #expect(fade(Span(minX: 0, maxX: 744)).trailing == CodeBlockOverflow.maxFadeWidth)
    }

    @Test("A sub-point overhang is treated as flush")
    func subPointOverhangDoesNotFade() {
        // SwiftUI geometry lands on fractional values; 0.3pt of
        // residue is not "there is more to read", and fading on it
        // would leave a permanent smudge on a block that fits.
        #expect(fade(Span(minX: 0, maxX: 720.3)).isEmpty)
        #expect(fade(Span(minX: -0.4, maxX: 720)).isEmpty)
        // Past the half-point threshold it is real overflow.
        #expect(fade(Span(minX: 0, maxX: 721)).trailing == 1)
    }

    // MARK: - Degenerate geometry

    @Test("An unmeasured block does not fade")
    func unmeasuredSpanDoesNotFade() {
        // The frame before the GeometryReader reports. The default
        // span sits at the origin, so a viewport offset from the
        // window origin would otherwise look like content hidden to
        // the left and fade a block nobody has measured yet.
        #expect(fade(.unmeasured).isEmpty)
        #expect(fade(.unmeasured, viewport: (312, 1032)).isEmpty)
        #expect(Span.unmeasured.isMeasured == false)
        #expect(Span(minX: 0, maxX: 822).isMeasured)
    }

    @Test("A zero-width viewport does not fade")
    func zeroWidthViewportDoesNotFade() {
        #expect(fade(Self.dogfoodContent, viewport: (0, 0)).isEmpty)
        #expect(fade(Self.dogfoodContent, viewport: (100, 40)).isEmpty)
    }

    @Test("Neither edge can eat more than a third of a narrow block")
    func fadeIsCappedOnNarrowBlocks() {
        // A 60pt viewport with overflow on both sides: at the full
        // 24pt each, the gradient would be 48 of 60 points and the
        // block would be mostly smudge. The cap keeps the opaque
        // middle the widest region, which also keeps the gradient
        // stops monotonic.
        let narrow = fade(Span(minX: -200, maxX: 400), viewport: (0, 60))
        #expect(narrow.leading == 20)
        #expect(narrow.trailing == 20)
        #expect(narrow.leading + narrow.trailing < 60)
    }

    @Test("The fade honours a caller-supplied maximum")
    func respectsCustomMaxFadeWidth() {
        #expect(fade(Self.dogfoodContent, maxFadeWidth: 8).trailing == 8)
        // A non-positive maximum disables the affordance outright
        // rather than producing a degenerate gradient.
        #expect(fade(Self.dogfoodContent, maxFadeWidth: 0).isEmpty)
        #expect(fade(Self.dogfoodContent, maxFadeWidth: -4).isEmpty)
    }

    // MARK: - Gradient stop invariants

    @Test("Fade widths always produce monotonic gradient stops")
    func gradientStopsStayMonotonic() {
        // ChatView turns these widths into LinearGradient stop
        // locations at `leading / width` and `1 - trailing / width`.
        // Out-of-order stops are undefined behaviour, so sweep the
        // geometry that reaches that call.
        let viewportWidths: [CGFloat] = [1, 12, 60, 300, 720, 1600]
        let overhangs: [CGFloat] = [0, 0.4, 1, 10, 24, 500, 5000]
        for width in viewportWidths {
            for leadingHidden in overhangs {
                for trailingHidden in overhangs {
                    let result = fade(
                        Span(minX: -leadingHidden, maxX: width + trailingHidden),
                        viewport: (0, width)
                    )
                    let leadingStop = result.leading / width
                    let trailingStop = 1 - result.trailing / width
                    #expect(
                        leadingStop <= trailingStop,
                        """
                        stops crossed at viewport \(width), \
                        hidden \(leadingHidden)/\(trailingHidden): \
                        \(leadingStop) > \(trailingStop)
                        """
                    )
                    #expect(result.leading >= 0)
                    #expect(result.trailing >= 0)
                }
            }
        }
    }
}
