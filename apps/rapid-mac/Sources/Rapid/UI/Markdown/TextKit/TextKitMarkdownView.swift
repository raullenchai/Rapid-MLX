import AppKit
import SwiftUI

/// The chat transcript's markdown surface, rendered through TextKit 2.
///
/// Replaces the MarkdownUI-backed `LaTeXMarkdownView` for settled messages
/// (#1843). Streaming messages render through
/// ``StreamingTextKitMarkdownView`` — a debounced compiler feeding the same
/// block stack — so both paths are TextKit 2 now.
///
/// ## What carries over unchanged
///
/// * **Link safety** (#304/#349). `.chatLinkSafetyFilter()` is applied at the
///   same level as before: the outer container, so every block inherits the
///   allowlist through the environment. The filter works by replacing
///   `\.openURL`, which is scheme-based and does not care whether the link
///   came from `Text` or from an `NSTextView` — see the verification test.
/// * **Table accessibility** (#1824). `AccessibleMarkdownTable` needs only a
///   `TableModel`, so it is fed directly from the compiled `TableBlock`
///   instead of re-serialising the block back to markdown and re-parsing it
///   with a hand-written splitter. Same AX output, one less round trip, and
///   the 8-column cap is no longer a parse-time rejection — a 9-column table
///   still renders visually, it just has no AX representation, exactly as
///   before.
/// * **Display math** (#131). Routed to Rapid's own `MathView`; inline math
///   stays folded into prose as literal `$…$`, which is what
///   `displayMathOnly` already did.
/// * **Dynamic Type** (#546). The base point size arrives from the caller as
///   a resolved `CGFloat` — see `ChatView`'s `@ScaledMetric`. TextKit needs a
///   number, not a SwiftUI font, so the scaling happens once at the boundary
///   rather than being re-derived per block.
struct TextKitMarkdownView: View, Equatable {
    let content: String

    /// Body point size, scaled for Dynamic Type (#546).
    ///
    /// MarkdownUI does this itself — its `ScaledFontSizeModifier` wraps the
    /// theme's root `FontSize` in the one and only `@ScaledMetric`, which is
    /// why `.rapidChat` sets a fixed literal and the theme comment warns
    /// against adding a second one at the call site.
    ///
    /// TextKit has no such pass: `NSFont` takes a number. So the scaling lives
    /// here, once, inside the view — not at the call site, which would be the
    /// double-scale bug that comment is about (~15 × scale²). 15 is the same
    /// literal the theme uses, and the two must move together; the theme
    /// comment names three literals that already travel in lockstep, and this
    /// is now a fourth.
    @ScaledMetric(relativeTo: .body) private var basePointSize: CGFloat = 15

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.content == rhs.content
    }

    var body: some View {
        MarkdownBlockStack(
            result: Self.compile(content),
            options: options,
            isStreaming: false,
            fadeState: nil,
            fadeConfiguration: .off
        )
        // Same mount level as the MarkdownUI path: one filter on the
        // container, inherited by every block below it.
        .chatLinkSafetyFilter()
    }

    private var options: MarkdownOptions { Self.options(basePointSize: basePointSize) }

    static func options(basePointSize: CGFloat) -> MarkdownOptions {
        var options = MarkdownOptions.assistantTranscript()
        options.textPointSize = basePointSize
        options.textColor = .labelColor
        options.linkColor = NSColor(RapidTheme.linkLabel)
        return options
    }

    /// Compile the message body.
    ///
    /// The math split lives inside `MarkdownCompiler` — it has to happen
    /// before markdown parsing either way, because `$x_1$` handed to a
    /// CommonMark parser becomes `x`, emphasis, `1` and the subscript is gone
    /// before any math code sees it.
    ///
    /// An earlier version of this method segmented here as well, which was a
    /// second pass over the same text doing the same job. It was harmless —
    /// segmenting already-segmented markdown is idempotent — but it meant
    /// breaking either copy left the other one covering for it, so no test
    /// could see the breakage. One owner, one pass.
    static func compile(_ content: String) -> MarkdownResult {
        MarkdownCompiler().compile(content)
    }
}

extension MarkdownItem.TableBlock {
    /// Feed the existing accessibility representation (#1824).
    ///
    /// Returns nil under the same conditions `MarkdownTableAccessibility.parse`
    /// rejected: no headers, or more than eight columns (macOS 14's
    /// `TableColumnBuilder` has no dynamic-column primitive, hence the
    /// hand-unrolled `table1`…`table8`).
    var accessibilityModel: MarkdownTableAccessibility.TableModel? {
        let headerTexts = header.map { $0.map(\.text).joined() }
        guard !headerTexts.isEmpty, headerTexts.count <= 8 else { return nil }
        let rowTexts = rows.map { row -> [String] in
            var cells = row.map { $0.map(\.text).joined() }
            // Rectangular, like the old parser: short rows are padded so the
            // Table's column count stays consistent.
            while cells.count < headerTexts.count { cells.append("") }
            return Array(cells.prefix(headerTexts.count))
        }
        return .init(headers: headerTexts, rows: rowTexts)
    }
}
