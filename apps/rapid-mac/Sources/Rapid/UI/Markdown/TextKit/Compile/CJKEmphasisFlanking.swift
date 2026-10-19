import Foundation
import Markdown

/// Adapts emphasis flanking for CJK prose before handing it to unmodified cmark.
/// CJK punctuation behaves like a letter for flanking; a CJK neighbour also
/// permits flanking beside other punctuation. Underscores retain their usual
/// intraword restriction. Private-use letters bridge only delimiter runs whose
/// cmark open/close flags can exactly match these rules. The compiler removes
/// the bridges from every emitted string before running post-processors.
enum CJKEmphasisFlanking {
    static let sentinel = "\u{F0002}"
    static let bridge: Unicode.Scalar = "\u{F0002}"

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x1100...0x11FF, 0x20A9, 0x2329...0x232A, 0x2630...0x2637,
             0x268A...0x268F, 0x2E80...0x2E99, 0x2E9B...0x2EF3, 0x2F00...0x2FD5,
             0x2FF0...0x303E, 0x3041...0x3096, 0x3099...0x30FF, 0x3105...0x312F,
             0x3131...0x318E, 0x3190...0x31E5, 0x31EF...0x321E, 0x3220...0x3247,
             0x3250...0xA48C, 0xA490...0xA4C6, 0xA960...0xA97C, 0xAC00...0xD7A3,
             0xD7B0...0xD7C6, 0xD7CB...0xD7FB, 0xF900...0xFAFF, 0xFE10...0xFE19,
             0xFE30...0xFE52, 0xFE54...0xFE66, 0xFE68...0xFE6B, 0xFF01...0xFFBE,
             0xFFC2...0xFFC7, 0xFFCA...0xFFCF, 0xFFD2...0xFFD7, 0xFFDA...0xFFDC,
             0xFFE0...0xFFE6, 0xFFE8...0xFFEE, 0x16FE0...0x16FE4, 0x16FF0...0x16FF6,
             0x17000...0x18CDA, 0x18CFF...0x18D20, 0x18D80...0x18DF2,
             0x18E00...0x19191, 0x191A0...0x191D2, 0x1AFF0...0x1AFF3,
             0x1AFF5...0x1AFFB, 0x1AFFD...0x1AFFE, 0x1B000...0x1B128, 0x1B132,
             0x1B150...0x1B152, 0x1B155, 0x1B164...0x1B168, 0x1B170...0x1B2FB,
             0x1D300...0x1D356, 0x1D360...0x1D376, 0x1F1AE, 0x1F200, 0x1F202,
             0x1F210...0x1F219, 0x1F21B...0x1F22E, 0x1F230...0x1F231, 0x1F237,
             0x1F23B, 0x1F240...0x1F248, 0x1F260...0x1F265, 0x1F7DA, 0x20000...0x3FFFD:
            return true
        default: return false
        }
    }

    static func isWhitespace(_ scalar: Unicode.Scalar?) -> Bool {
        guard let scalar else { return true }
        return scalar.properties.generalCategory == .spaceSeparator
            || [9, 10, 12, 13].contains(scalar.value)
    }

    static func isPunctuation(_ scalar: Unicode.Scalar?) -> Bool {
        guard let scalar else { return false }
        // CommonMark includes all ASCII punctuation, including `$`, `+`, etc.
        if (33...47).contains(scalar.value) || (58...64).contains(scalar.value)
            || (91...96).contains(scalar.value) || (123...126).contains(scalar.value) { return true }
        switch scalar.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation,
             .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol: return true
        default: return false
        }
    }

    struct Flanking: Equatable {
        let canOpen: Bool
        let canClose: Bool
    }

    static func flanking(
        before: Unicode.Scalar?, after: Unicode.Scalar?, delimiter: Unicode.Scalar,
        cjkFriendly: Bool = true
    ) -> Flanking {
        let beforeCJK = before.map(isCJK) ?? false
        let afterCJK = after.map(isCJK) ?? false
        let beforePunctuation = isPunctuation(before)
        let afterPunctuation = isPunctuation(after)
        let beforeBoundary = beforePunctuation && !(cjkFriendly && beforeCJK)
        let afterBoundary = afterPunctuation && !(cjkFriendly && afterCJK)
        let left = !isWhitespace(after) && (!afterBoundary || isWhitespace(before)
            || beforeBoundary || (cjkFriendly && beforeCJK))
        let right = !isWhitespace(before) && (!beforeBoundary || isWhitespace(after)
            || afterBoundary || (cjkFriendly && afterCJK))
        return Flanking(
            canOpen: left && (delimiter != "_" || !right || beforePunctuation),
            canClose: right && (delimiter != "_" || !left || afterPunctuation)
        )
    }

    /// Try the least invasive bridge first, and preserve the run if no
    /// placement can reproduce both flags exactly.
    static func bridgePlacement(
        before: Unicode.Scalar?, after: Unicode.Scalar?, delimiter: Unicode.Scalar,
        desired: Flanking
    ) -> (before: Bool, after: Bool)? {
        for (insertBefore, insertAfter) in [(true, false), (false, true), (true, true)] {
            if desired == flanking(before: insertBefore ? bridge : before,
                                   after: insertAfter ? bridge : after,
                                   delimiter: delimiter, cjkFriendly: false) {
                return (insertBefore, insertAfter)
            }
        }
        return nil
    }

    static func prepare(_ source: String) -> String {
        guard source.contains("*") || source.contains("_"),
              source.unicodeScalars.contains(where: isCJK) else { return source }
        // Collect edits without rebuilding ordinary text or guessing code syntax.
        let scalars = source.unicodeScalars
        var candidates: [(position: String.Index, run: String.Index)] = []
        var index = scalars.startIndex
        while index < scalars.endIndex {
            let character = scalars[index]
            var end = scalars.index(after: index)
            if character == "\\", end < scalars.endIndex {
                end = scalars.index(after: end)
            } else if character == "*" || character == "_" {
                while end < scalars.endIndex, scalars[end] == character {
                    end = scalars.index(after: end)
                }
                let previous = index == scalars.startIndex ? nil : scalars[scalars.index(before: index)]
                let before = previous == "\r" || previous == "\n" ? nil : previous
                let next = end == scalars.endIndex ? nil : scalars[end]
                let after = next == "\r" || next == "\n" ? nil : next
                let desired = flanking(before: before, after: after, delimiter: character)
                if desired != flanking(before: before, after: after, delimiter: character, cjkFriendly: false),
                   let placement = bridgePlacement(before: before, after: after,
                                                   delimiter: character, desired: desired) {
                    if placement.before { candidates.append((index, index)) }
                    if placement.after { candidates.append((end, index)) }
                }
            }
            index = end
        }
        guard !candidates.isEmpty else { return source }

        let document = Document(parsing: source, options: MarkdownCompiler.parseOptions)
        let excluded = opaqueRanges(in: document, source: source,
                                    enclosing: source.startIndex..<source.endIndex)
        // Keep the run as well as the insertion point: a bridge immediately
        // after a closing delimiter can sit at the opaque range's upper bound.
        candidates.removeAll { candidate in
            excluded.contains { $0.contains(candidate.run) || $0.contains(candidate.position) }
        }
        guard !candidates.isEmpty else { return source }

        var output = ""
        var start = source.startIndex
        for candidate in candidates {
            output += source[start..<candidate.position]
            output += sentinel
            start = candidate.position
        }
        output += source[start...]
        return output
    }

    /// Parsed opaque nodes include their syntax delimiters. If a node has no
    /// source range, exclude its enclosing node, conservatively covering all
    /// of the missing node's lines instead of guessing at code boundaries.
    static func opaqueRanges(
        in markup: Markup, source: String, enclosing: Range<String.Index>
    ) -> [Range<String.Index>] {
        let range = MarkdownCompiler.sourceRange(of: markup, in: source) ?? enclosing
        switch markup {
        case is CodeBlock, is InlineCode, is HTMLBlock, is InlineHTML:
            return [range]
        default:
            return markup.children.flatMap { opaqueRanges(in: $0, source: source, enclosing: range) }
        }
    }

    static func strip(_ string: String) -> String {
        string.replacingOccurrences(of: sentinel, with: "")
    }
}
