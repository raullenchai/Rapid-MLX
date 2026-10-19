import Foundation

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
        var output = ""
        var fence: (character: Unicode.Scalar, length: Int)?
        // Keep line endings verbatim; inline code matching is intentionally
        // limited to a single line. A shorter or different-character fence cannot close.
        for (lineIndex, line) in source.components(separatedBy: "\n").enumerated() {
            if lineIndex > 0 { output += "\n" }
            let scalars = Array(line.unicodeScalars)
            let indent = scalars.prefix(while: { $0 == " " }).count
            let start = indent
            if indent <= 3, start < scalars.count,
               scalars[start] == "`" || scalars[start] == "~" {
                let character = scalars[start]
                let length = scalars[start...].prefix(while: { $0 == character }).count
                let rest = scalars.dropFirst(start + length)
                if let active = fence {
                    if character == active.character, length >= active.length,
                       rest.allSatisfy({ $0 == " " || $0 == "\t" || $0 == "\r" }) { fence = nil }
                    output += line
                    continue
                } else if length >= 3, character != "`" || !rest.contains("`") {
                    fence = (character, length)
                    output += line
                    continue
                }
            }
            if fence != nil { output += line; continue }
            var index = 0
            while index < scalars.count {
                let character = scalars[index]
                var end = index + 1
                if character == "\\", end < scalars.count {
                    end += 1
                } else if character == "`" || character == "*" || character == "_" {
                    while end < scalars.count, scalars[end] == character { end += 1 }
                    if character == "`" {
                        var cursor = end
                        while cursor < scalars.count {
                            if scalars[cursor] != "`" { cursor += 1; continue }
                            let closeStart = cursor
                            while cursor < scalars.count, scalars[cursor] == "`" { cursor += 1 }
                            if cursor - closeStart == end - index { end = cursor; break }
                        }
                    } else {
                        let before = index == 0 ? nil : scalars[index - 1]
                        let after = end == scalars.count ? nil : scalars[end]
                        let desired = flanking(before: before, after: after, delimiter: character)
                        if desired != flanking(before: before, after: after, delimiter: character, cjkFriendly: false),
                           let placement = bridgePlacement(before: before, after: after,
                                                           delimiter: character, desired: desired) {
                            if placement.before { output += sentinel }
                            output.unicodeScalars.append(contentsOf: scalars[index..<end])
                            if placement.after { output += sentinel }
                            index = end
                            continue
                        }
                    }
                }
                output.unicodeScalars.append(contentsOf: scalars[index..<end])
                index = end
            }
        }
        return output
    }

    static func strip(_ string: String) -> String {
        string.replacingOccurrences(of: sentinel, with: "")
    }
}
