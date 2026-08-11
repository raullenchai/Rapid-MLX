import AppKit
import SwiftUI

/// Bridges Rapid's ``SyntaxHighlighter`` to the range-and-colour shape the
/// TextKit 2 renderer needs.
///
/// The two sides disagree about output, not about tokenising:
/// ``SyntaxHighlighter.highlight`` returns an `AttributedString` (built for
/// SwiftUI's `Text`), while ``MarkdownTextRenderer`` applies colours to an
/// `NSMutableAttributedString` by `NSRange`. Converting here keeps Rapid's
/// 1,387-line grammar table — which covers far more languages than the port's
/// own highlighter did — as the single source of tokenising truth, per the
/// #1843 merge gate that says reuse it.
///
/// The port's `CodeHighlighter` is deliberately NOT carried over. Two
/// highlighters would drift, and this one is better.
enum CodeHighlighter {

    /// Kept so call sites read the same as before the port. Rapid's
    /// highlighter resolves its own colours from the SwiftUI environment's
    /// colour scheme, so there is nothing to select here — the cases exist
    /// only to satisfy the renderer's signature.
    enum Theme {
        case `default`
        case darkDefault
    }

    static func supports(language: String?) -> Bool {
        SyntaxHighlighter.supports(language: language)
    }

    /// Tokenise `code` and return the coloured spans.
    ///
    /// Ranges are UTF-16 offsets into `code`, which is what
    /// `NSMutableAttributedString.addAttribute` expects. `AttributedString`
    /// indices are character-based, so the conversion goes through
    /// `NSRange(_:in:)` rather than arithmetic on integers — a code block
    /// containing an emoji or a CJK character would otherwise colour the
    /// wrong bytes.
    static func ranges(
        in code: String, language: String?, theme: Theme
    ) -> [(NSRange, NSColor)] {
        guard supports(language: language) else { return [] }
        let highlighted = SyntaxHighlighter.highlight(code, language: language)

        var result: [(NSRange, NSColor)] = []
        for run in highlighted.runs {
            guard let colour = run.foregroundColor else { continue }
            let nsRange = NSRange(run.range, in: highlighted)
            guard nsRange.location != NSNotFound else { continue }
            result.append((nsRange, NSColor(colour)))
        }
        return result
    }
}
