import Foundation

/// Shared cleanup for text a model wrote when it was asked for a *value*
/// rather than for prose.
///
/// Both background completions — the conversation title and the follow-up
/// questions — ask for something with a shape, and both get back the shape
/// wrapped in whatever the model felt like adding: a `Title:` label, a
/// markdown fence, a numbered list, smart quotes, a leftover `<think>` block.
/// The wrapping is the same in each case, so it is stripped in one place;
/// what differs is what each caller accepts afterwards, and that stays with
/// the caller.
///
/// Everything here is `nonisolated` and total — no throws, no optionals
/// beyond "there was nothing here", no I/O. That is the point: the part of
/// this feature that decides what a reader sees can be tested exhaustively
/// without a model.
enum ModelReplyText {

    /// Text with any reasoning block removed.
    ///
    /// Both callers send `enableThinking: false`, and the kwarg is only
    /// emitted when thinking is off (``ChatStreamClient``), so this should
    /// never fire. It stays because several chat templates emit the block
    /// from the template itself regardless of the kwarg, and a `<think>`
    /// preamble would otherwise become the title.
    nonisolated static func strippingReasoning(_ text: String) -> String {
        guard let close = text.range(of: "</think>", options: .backwards) else {
            return text
        }
        return String(text[close.upperBound...])
    }

    /// Text with whole-line markdown fences removed. Only lines that are
    /// *nothing but* a fence go — a line of prose containing backticks is
    /// left for the caller to judge.
    nonisolated static func strippingFenceLines(_ text: String) -> String {
        text.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") else { return true }
                // ``` or ```swift — but not ``` some prose ```
                return trimmed.dropFirst(3).contains(where: \.isWhitespace)
            }
            .joined(separator: "\n")
    }

    /// The first line with something on it, after reasoning and fences are
    /// gone. Nil when the reply was empty or was nothing but scaffolding.
    nonisolated static func firstMeaningfulLine(_ text: String) -> String? {
        strippingFenceLines(strippingReasoning(text))
            .split(separator: "\n", omittingEmptySubsequences: false)
            .lazy
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty }
    }

    /// Every line with something on it, in order.
    nonisolated static func meaningfulLines(_ text: String) -> [String] {
        strippingFenceLines(strippingReasoning(text))
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Drops a leading `Label:` / `标题：` and the separator after it.
    /// Matching is case-insensitive and the separator may be a colon, a
    /// full-width colon, a dash or an em dash.
    nonisolated static func strippingLabel(_ line: String, labels: [String]) -> String {
        let lowered = line.lowercased()
        for label in labels where lowered.hasPrefix(label.lowercased()) {
            let rest = line.dropFirst(label.count)
                .drop { $0 == " " }
            guard let separator = rest.first,
                  ":：-—".contains(separator) else { continue }
            return String(rest.dropFirst()).trimmingCharacters(in: .whitespaces)
        }
        return line
    }

    /// Drops `1.` / `2)` / `-` / `*` / `•` and the space after it.
    nonisolated static func strippingListMarker(_ line: String) -> String {
        var rest = Substring(line)
        if let first = rest.first, "-*•".contains(first) {
            rest = rest.dropFirst()
        } else {
            let digits = rest.prefix(while: \.isNumber)
            guard !digits.isEmpty, digits.count <= 2,
                  let punctuation = rest.dropFirst(digits.count).first,
                  ".)".contains(punctuation)
            else { return line }
            rest = rest.dropFirst(digits.count + 1)
        }
        // A marker is only a marker when something follows it with a space
        // between — `-42` is a number, not a bulleted 42.
        guard let next = rest.first, next == " " else { return line }
        return String(rest).trimmingCharacters(in: .whitespaces)
    }

    /// Symmetric wrappers a model adds when it thinks it is quoting: matched
    /// quotes of several nationalities, and markdown emphasis.
    private static let wrappers: [(String, String)] = [
        ("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"),
        ("«", "»"), ("「", "」"), ("『", "』"), ("《", "》"),
        ("**", "**"), ("*", "*"), ("`", "`"), ("[", "]"), ("(", ")"),
    ]

    /// Peels matched wrappers, outermost first, until none match. Bounded by
    /// the wrapper count so a pathological reply cannot spin.
    nonisolated static func strippingWrappers(_ line: String) -> String {
        var result = line.trimmingCharacters(in: .whitespaces)
        for _ in 0..<wrappers.count {
            guard let match = wrappers.first(where: { open, close in
                result.hasPrefix(open) && result.hasSuffix(close)
                    && result.count > open.count + close.count
            }) else { break }
            result = String(
                result.dropFirst(match.0.count).dropLast(match.1.count)
            ).trimmingCharacters(in: .whitespaces)
        }
        return result
    }

    /// Drops a trailing full stop. A question mark stays — a question makes
    /// a perfectly good title, and it is load-bearing for follow-ups.
    nonisolated static func strippingTrailingStop(_ line: String) -> String {
        guard let last = line.last, last == "." || last == "。" else { return line }
        return String(line.dropLast()).trimmingCharacters(in: .whitespaces)
    }

    /// Runs of whitespace — including newlines — become single spaces.
    /// Same fold ``ConversationStore/title(from:)`` applies, so a generated
    /// title and a derived one are normalised identically.
    nonisolated static func collapsingWhitespace(_ text: String) -> String {
        text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Openings a model uses when it is talking to you instead of answering.
    private static let refusalPrefixes = [
        "i ", "i'm", "i'd", "sure", "here", "here's", "certainly", "of course",
        "as an", "okay", "ok,", "understood", "好的", "当然", "抱歉", "作为",
    ]

    nonisolated static func looksLikeRefusal(_ line: String) -> Bool {
        let lowered = line.lowercased()
        return refusalPrefixes.contains { lowered.hasPrefix($0) }
    }
}
