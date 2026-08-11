import Foundation

/// Turns bare URLs into links.
///
/// `https://example.com` typed without brackets is plain text as far as
/// CommonMark is concerned. GFM's autolink extension covers it, but
/// swift-markdown does not implement that extension, so it lands here — which
/// is also where ChatGPT puts it (`AutoDetectURLMarkdownPlugin`, a
/// post-process pass, not a parse rule).
///
/// Two things are deliberately left alone:
///
/// * **Runs that already carry a link.** `[text](url)` must keep its own
///   destination; re-scanning its display text could overwrite it.
/// * **Inline code.** A URL inside backticks is being shown, not offered.
struct AutoLinkPostProcessor: MarkdownPostProcessor {

    public init() {}

    public func process(
        _ block: MarkdownItem.TextBlock,
        context: MarkdownPostProcessContext
    ) -> MarkdownItem.TextBlock? {
        var output: [InlineRun] = []
        var changed = false

        for run in block.runs {
            guard run.link == nil, !run.isInlineCode,
                  let split = Self.split(run) else {
                output.append(run)
                continue
            }
            output.append(contentsOf: split)
            changed = true
        }

        guard changed else { return nil }
        var result = block
        result.runs = output
        return result
    }

    /// Split one run into link and non-link pieces, or nil if it has no URL.
    static func split(_ run: InlineRun) -> [InlineRun]? {
        let text = run.text
        let matches = detector.matches(
            in: text, range: NSRange(text.startIndex..., in: text)
        )
        guard !matches.isEmpty else { return nil }

        var pieces: [InlineRun] = []
        var cursor = text.startIndex

        for match in matches {
            guard let range = Range(match.range, in: text),
                  let url = match.url else { continue }
            if cursor < range.lowerBound {
                var before = run
                before.text = String(text[cursor..<range.lowerBound])
                pieces.append(before)
            }
            var linked = run
            linked.text = String(text[range])
            linked.link = url
            pieces.append(linked)
            cursor = range.upperBound
        }

        if cursor < text.endIndex {
            var tail = run
            tail.text = String(text[cursor...])
            pieces.append(tail)
        }
        return pieces.isEmpty ? nil : pieces
    }

    /// `NSDataDetector` rather than a hand-written regex.
    ///
    /// URL boundaries are genuinely hard — trailing punctuation, parentheses
    /// that may or may not belong to the URL, IDN, ports. The platform
    /// detector is the same one Mail and Messages use, so its answers match
    /// what a macOS user already expects a link to be.
    ///
    /// Restricted to `.link`; the detector can also find dates, addresses and
    /// phone numbers, none of which should silently become tappable in a chat
    /// transcript.
    private static let detector: NSDataDetector = {
        // The initialiser only throws for an invalid checking-type mask, and
        // `.link` is valid, so this cannot fail at runtime.
        try! NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
    }()
}
