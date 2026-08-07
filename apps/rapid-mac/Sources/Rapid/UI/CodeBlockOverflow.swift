import CoreGraphics

/// Geometry behind the fenced-code-block edge fade.
///
/// ## The bug this exists to fix
///
/// A 2026-08 dogfood run answered a "merge overlapping intervals"
/// question with a Python block whose demo lines all stopped dead at
/// the same x position:
///
/// ```text
/// print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))   #[[1, 6],[8, 10],[15, 1
/// print(merge_intervals([]))                             #[[]
/// ```
///
/// It reads as truncation, and it is not. The block's horizontal
/// ``ScrollView`` hands the text its full intrinsic width — measured
/// headlessly through ``NSHostingView`` at the real 720pt message
/// column, that sample lays out at 822pt, so 102pt of code is present,
/// laid out, and one two-finger swipe away. What was missing was any
/// sign of it: the scroll view was built with `showsIndicators: false`,
/// which installs no `NSScroller` at all, and macOS overlay scrollers
/// are invisible at rest even when they exist. Clipped-and-lost and
/// scrolled-out-of-view looked identical, so the reader concluded the
/// answer was cut off.
///
/// The fix is an affordance, not a re-layout: turn the indicator on so
/// the gesture has feedback, and dissolve the text into the block edge
/// wherever content is actually hidden, so the boundary reads as "this
/// continues" instead of "this ends here".
///
/// ## Why this is a separate type
///
/// The ramp is the only part with a right and a wrong answer, and it
/// has edge cases worth pinning: an unmeasured block must not fade, a
/// block scrolled to its end must not keep a stale fade, and a block
/// narrower than two fades must not become all gradient. Factored out
/// of the view, those are ordinary value assertions instead of a
/// snapshot.
enum CodeBlockOverflow {

    /// Widest either edge fades, in points. Wide enough to read as a
    /// deliberate soft edge on a 15pt-based monospaced line, narrow
    /// enough that it never eats a whole token.
    static let maxFadeWidth: CGFloat = 24

    /// Horizontal span of a code block's scrollable content, in global
    /// coordinates.
    ///
    /// Only the two x edges travel in the preference. The block grows
    /// DOWNWARD on every streamed token, so folding height in here
    /// would republish the preference — and re-render the fade —
    /// on every single coalescer flush, for a number the fade never
    /// reads.
    struct ContentSpan: Equatable, Sendable {
        var minX: CGFloat
        var maxX: CGFloat

        static let unmeasured = ContentSpan(minX: 0, maxX: 0)

        /// A real measurement always has positive width. The default
        /// preference value does not, which is what keeps the fade off
        /// during the frame before the ``GeometryReader`` reports.
        var isMeasured: Bool { maxX > minX }
    }

    /// How wide the fade is at each edge. Zero on an edge means the
    /// content there is fully visible and the edge stays crisp.
    struct Fade: Equatable, Sendable {
        var leading: CGFloat
        var trailing: CGFloat

        static let none = Fade(leading: 0, trailing: 0)

        var isEmpty: Bool { self == .none }
    }

    /// Fade widths for a code block whose content spans ``content``
    /// inside a viewport spanning `viewportMinX...viewportMaxX`, all in
    /// the same coordinate space.
    ///
    /// Because the content span is measured INSIDE the scroll view, it
    /// tracks the scroll offset for free: scrolling right walks
    /// `content.maxX` down toward `viewportMaxX`, so the trailing fade
    /// closes itself as the user reaches the end of the line.
    static func fade(
        content: ContentSpan,
        viewportMinX: CGFloat,
        viewportMaxX: CGFloat,
        maxFadeWidth: CGFloat = CodeBlockOverflow.maxFadeWidth
    ) -> Fade {
        let viewportWidth = viewportMaxX - viewportMinX
        guard content.isMeasured, viewportWidth > 0, maxFadeWidth > 0 else {
            return .none
        }
        // A block narrower than two full fades would render as nothing
        // but gradient. Cap each edge at a third of the viewport so the
        // middle is always the widest, fully-opaque region.
        let cap = min(maxFadeWidth, viewportWidth / 3)
        return Fade(
            leading: ramp(hidden: viewportMinX - content.minX, cap: cap),
            trailing: ramp(hidden: content.maxX - viewportMaxX, cap: cap)
        )
    }

    /// Fade width for `hidden` points of content past an edge.
    ///
    /// Ramps with the hidden amount rather than snapping to full width,
    /// so the last few points of a scroll dissolve the fade smoothly
    /// instead of popping it off. Sub-point overhangs are treated as
    /// flush — SwiftUI geometry lands on fractional values, and a 0.3pt
    /// residue is not "there is more to read".
    private static func ramp(hidden: CGFloat, cap: CGFloat) -> CGFloat {
        guard hidden > 0.5 else { return 0 }
        return min(cap, hidden)
    }
}
