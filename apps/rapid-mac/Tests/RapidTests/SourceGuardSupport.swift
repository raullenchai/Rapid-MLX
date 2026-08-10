import Foundation

/// Literal-aware canonicalisation for tests that assert on Swift source text.
enum SourceGuardSupport {
    enum LiteralPolicy: Equatable { case preserve, erase }

    static func canonicalSource(_ source: String, literals policy: LiteralPolicy) -> String {
        var output = ""
        output.reserveCapacity(source.count)
        var index = source.startIndex
        while index < source.endIndex {
            if hasPrefix("//", in: source, at: index) {
                while index < source.endIndex, source[index] != "\n" { index = source.index(after: index) }
                continue
            }
            if hasPrefix("/*", in: source, at: index) {
                index = endOfBlockComment(in: source, at: index)
                continue
            }
            if canStartBareRegex(in: source, at: index),
               let literal = canonicalizedBareRegex(
                   in: source, at: index, policy: policy
               ) {
                // Keep one slash as a fail-closed marker. Text gates can
                // still inspect executable source after the regex, while a
                // brace-balancing caller refuses to guess whether remaining
                // slash syntax is regex or division.
                output += policy == .erase ? "/" + literal.text : literal.text
                index = literal.end
                continue
            }
            if let literal = canonicalizedExtendedRegex(
                in: source, at: index, policy: policy
            ) {
                output += literal.text
                index = literal.end
                continue
            }
            if let literal = canonicalizedStringLiteral(
                in: source, at: index, policy: policy
            ) {
                output += literal.text
                index = literal.end
                continue
            }
            if !source[index].isWhitespace { output.append(source[index]) }
            index = source.index(after: index)
        }
        return output
    }

    static func balancedBlock(in source: String, openingBraceAt start: String.Index) -> String? {
        guard start < source.endIndex, source[start] == "{" else { return nil }
        var depth = 0
        var index = start
        while index < source.endIndex {
            // Bare regex literals and division share `/` syntax. Guessing
            // which one this is can let a brace inside `/[}]/` terminate the
            // scan. Extended regexes have already been erased, so fail closed
            // on every unresolved slash and make the caller choose a parser
            // before balancing slash-bearing source.
            if source[index] == "/" { return nil }
            if source[index] == "{" { depth += 1 }
            if source[index] == "}" {
                depth -= 1
                if depth == 0 { return String(source[start...index]) }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func hasPrefix(_ prefix: String, in source: String, at index: String.Index) -> Bool {
        source[index...].hasPrefix(prefix)
    }

    private static func canonicalizedStringLiteral(
        in source: String,
        at start: String.Index,
        policy: LiteralPolicy
    ) -> (text: String, end: String.Index)? {
        var quote = start
        var hashes = 0
        while quote < source.endIndex, source[quote] == "#" {
            hashes += 1
            quote = source.index(after: quote)
        }
        guard quote < source.endIndex, source[quote] == "\"" else { return nil }

        var quoteCount = 1
        if hasPrefix("\"\"\"", in: source, at: quote) {
            let afterTripleQuote = source.index(quote, offsetBy: 3)
            if afterTripleQuote < source.endIndex, source[afterTripleQuote].isNewline {
                quoteCount = 3
            }
        }
        guard let end = endOfStringLiteral(in: source, at: start) else { return nil }
        let contentStart = source.index(quote, offsetBy: quoteCount)
        let contentEnd = source.index(end, offsetBy: -(quoteCount + hashes))
        return (
            canonicalizedLiteral(
                in: source,
                opening: start..<contentStart,
                content: contentStart..<contentEnd,
                closing: contentEnd..<end,
                hashCount: hashes,
                policy: policy
            ),
            end
        )
    }

    private static func canonicalizedExtendedRegex(
        in source: String,
        at start: String.Index,
        policy: LiteralPolicy
    ) -> (text: String, end: String.Index)? {
        var slash = start
        var hashes = 0
        while slash < source.endIndex, source[slash] == "#" {
            hashes += 1
            slash = source.index(after: slash)
        }
        guard hashes > 0, slash < source.endIndex, source[slash] == "/",
              let end = endOfExtendedRegex(in: source, at: start)
        else { return nil }

        let contentStart = source.index(after: slash)
        let contentEnd = source.index(end, offsetBy: -(hashes + 1))
        return (
            canonicalizedLiteral(
                in: source,
                opening: start..<contentStart,
                content: contentStart..<contentEnd,
                closing: contentEnd..<end,
                hashCount: hashes,
                policy: policy
            ),
            end
        )
    }

    private static func canonicalizedBareRegex(
        in source: String,
        at start: String.Index,
        policy: LiteralPolicy
    ) -> (text: String, end: String.Index)? {
        guard start < source.endIndex, source[start] == "/",
              !hasPrefix("//", in: source, at: start),
              !hasPrefix("/*", in: source, at: start),
              let end = endOfBareRegex(in: source, at: start)
        else { return nil }

        let contentStart = source.index(after: start)
        let contentEnd = source.index(before: end)
        return (
            canonicalizedLiteral(
                in: source,
                opening: start..<contentStart,
                content: contentStart..<contentEnd,
                closing: contentEnd..<end,
                hashCount: 0,
                policy: policy
            ),
            end
        )
    }

    private static func canStartBareRegex(
        in source: String,
        at slash: String.Index
    ) -> Bool {
        var cursor = slash
        while cursor > source.startIndex {
            let previous = source.index(before: cursor)
            if source[previous].isWhitespace {
                cursor = previous
                continue
            }
            if "=([{,:;!?&|+-*%^~<>".contains(source[previous]) { return true }
            guard source[previous].isLetter else { return false }

            var wordStart = previous
            while wordStart > source.startIndex {
                let candidate = source.index(before: wordStart)
                guard source[candidate].isLetter else { break }
                wordStart = candidate
            }
            let word = source[wordStart...previous]
            return ["return", "throw", "case", "in", "where", "try", "await"]
                .contains(String(word))
        }
        return true
    }

    private static func canonicalizedLiteral(
        in source: String,
        opening: Range<String.Index>,
        content: Range<String.Index>,
        closing: Range<String.Index>,
        hashCount: Int,
        policy: LiteralPolicy
    ) -> String {
        var output = policy == .preserve
            ? String(source[opening].filter { !$0.isWhitespace })
            : "\"\""
        var index = content.lowerBound
        while index < content.upperBound {
            if let expressionStart = interpolationExpressionStart(
                in: source, at: index, hashCount: hashCount
            ), let expressionEnd = endOfInterpolation(in: source, at: expressionStart),
               expressionEnd < content.upperBound {
                if policy == .preserve {
                    output.append(
                        contentsOf: source[index..<expressionStart].filter { !$0.isWhitespace }
                    )
                } else {
                    output.append("(")
                }
                output += canonicalSource(
                    String(source[expressionStart..<expressionEnd]),
                    literals: policy
                )
                output.append(")")
                index = source.index(after: expressionEnd)
                continue
            }
            if policy == .preserve, !source[index].isWhitespace {
                output.append(source[index])
            }
            index = source.index(after: index)
        }
        if policy == .preserve {
            output.append(contentsOf: source[closing].filter { !$0.isWhitespace })
        }
        return output
    }

    private static func interpolationExpressionStart(
        in source: String,
        at slash: String.Index,
        hashCount: Int
    ) -> String.Index? {
        guard slash < source.endIndex, source[slash] == "\\" else { return nil }
        var index = source.index(after: slash)
        for _ in 0..<hashCount {
            guard index < source.endIndex, source[index] == "#" else { return nil }
            index = source.index(after: index)
        }
        guard index < source.endIndex, source[index] == "(" else { return nil }
        return source.index(after: index)
    }

    private static func endOfInterpolation(
        in source: String,
        at expressionStart: String.Index
    ) -> String.Index? {
        var depth = 1
        var index = expressionStart
        while index < source.endIndex {
            if hasPrefix("//", in: source, at: index) {
                while index < source.endIndex, source[index] != "\n" {
                    index = source.index(after: index)
                }
                continue
            }
            if hasPrefix("/*", in: source, at: index) {
                index = endOfBlockComment(in: source, at: index)
                continue
            }
            if canStartBareRegex(in: source, at: index),
               let end = endOfBareRegex(in: source, at: index) {
                index = end
                continue
            }
            if let end = endOfExtendedRegex(in: source, at: index) {
                index = end
                continue
            }
            if let end = endOfStringLiteral(in: source, at: index) {
                index = end
                continue
            }
            if source[index] == "(" { depth += 1 }
            if source[index] == ")" {
                depth -= 1
                if depth == 0 { return index }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func endOfBlockComment(in source: String, at start: String.Index) -> String.Index {
        var depth = 0
        var index = start
        while index < source.endIndex {
            if hasPrefix("/*", in: source, at: index) {
                depth += 1
                index = source.index(index, offsetBy: 2)
            } else if hasPrefix("*/", in: source, at: index) {
                depth -= 1
                index = source.index(index, offsetBy: 2)
                if depth == 0 { return index }
            } else { index = source.index(after: index) }
        }
        return index
    }

    private static func endOfStringLiteral(in source: String, at start: String.Index) -> String.Index? {
        var quote = start
        var hashes = 0
        while quote < source.endIndex, source[quote] == "#" {
            hashes += 1
            quote = source.index(after: quote)
        }
        guard quote < source.endIndex, source[quote] == "\"" else { return nil }
        var quoteCount = 1
        if hasPrefix("\"\"\"", in: source, at: quote) {
            let afterTripleQuote = source.index(quote, offsetBy: 3)
            if afterTripleQuote < source.endIndex, source[afterTripleQuote].isNewline {
                quoteCount = 3
            }
        }
        var index = source.index(quote, offsetBy: quoteCount)
        while index < source.endIndex {
            if let expressionStart = interpolationExpressionStart(
                in: source, at: index, hashCount: hashes
            ), let expressionEnd = endOfInterpolation(in: source, at: expressionStart) {
                index = source.index(after: expressionEnd)
                continue
            }
            if source[index] == "\\", escapeUses(hashCount: hashes, in: source, at: index) {
                index = indexAfterEscape(hashCount: hashes, in: source, at: index)
                continue
            }
            if let end = closingDelimiter(quoteCount: quoteCount, hashCount: hashes, in: source, at: index) {
                return end
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func endOfExtendedRegex(in source: String, at start: String.Index) -> String.Index? {
        var slash = start
        var hashes = 0
        while slash < source.endIndex, source[slash] == "#" {
            hashes += 1
            slash = source.index(after: slash)
        }
        guard hashes > 0, slash < source.endIndex, source[slash] == "/" else { return nil }
        var index = source.index(after: slash)
        while index < source.endIndex {
            if let expressionStart = interpolationExpressionStart(
                in: source, at: index, hashCount: hashes
            ), let expressionEnd = endOfInterpolation(in: source, at: expressionStart) {
                index = source.index(after: expressionEnd)
                continue
            }
            if source[index] == "\\" {
                index = source.index(index, offsetBy: 2, limitedBy: source.endIndex) ?? source.endIndex
                continue
            }
            if source[index] == "/" {
                var end = source.index(after: index)
                var seen = 0
                while seen < hashes, end < source.endIndex, source[end] == "#" {
                    seen += 1
                    end = source.index(after: end)
                }
                if seen == hashes { return end }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func endOfBareRegex(
        in source: String,
        at start: String.Index
    ) -> String.Index? {
        guard start < source.endIndex, source[start] == "/",
              !hasPrefix("//", in: source, at: start),
              !hasPrefix("/*", in: source, at: start)
        else { return nil }
        var index = source.index(after: start)
        var insideCharacterClass = false
        while index < source.endIndex {
            if source[index].isNewline { return nil }
            if let expressionStart = interpolationExpressionStart(
                in: source, at: index, hashCount: 0
            ), let expressionEnd = endOfInterpolation(in: source, at: expressionStart) {
                index = source.index(after: expressionEnd)
                continue
            }
            if source[index] == "\\" {
                index = source.index(index, offsetBy: 2, limitedBy: source.endIndex)
                    ?? source.endIndex
                continue
            }
            if source[index] == "[" { insideCharacterClass = true }
            if source[index] == "]" { insideCharacterClass = false }
            if source[index] == "/", !insideCharacterClass {
                return source.index(after: index)
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func closingDelimiter(quoteCount: Int, hashCount: Int, in source: String, at start: String.Index) -> String.Index? {
        var index = start
        for _ in 0..<quoteCount {
            guard index < source.endIndex, source[index] == "\"" else { return nil }
            index = source.index(after: index)
        }
        for _ in 0..<hashCount {
            guard index < source.endIndex, source[index] == "#" else { return nil }
            index = source.index(after: index)
        }
        return index
    }

    private static func escapeUses(hashCount: Int, in source: String, at slash: String.Index) -> Bool {
        var index = source.index(after: slash)
        for _ in 0..<hashCount {
            guard index < source.endIndex, source[index] == "#" else { return false }
            index = source.index(after: index)
        }
        return true
    }

    private static func indexAfterEscape(hashCount: Int, in source: String, at slash: String.Index) -> String.Index {
        var index = source.index(after: slash)
        for _ in 0..<hashCount where index < source.endIndex { index = source.index(after: index) }
        return index < source.endIndex ? source.index(after: index) : index
    }
}
