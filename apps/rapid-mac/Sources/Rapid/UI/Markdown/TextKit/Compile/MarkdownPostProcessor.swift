import Foundation

/// A pass that rewrites compiled text blocks before they reach the renderer.
///
/// Some markdown features cannot be expressed as parse rules because they are
/// not markdown: a bare URL is ordinary text until something decides to make
/// it a link, and `$x^2$` is ordinary text until something recognises TeX.
/// ChatGPT keeps seven of these (`AutoDetectURLMarkdownPlugin`,
/// `LatexMarkdownPlugin`, `GroupAdjacentImagesMarkdownPlugin`, …) as a
/// post-process stage rather than folding them into the parser, and the same
/// split applies here — the parser stays a CommonMark parser, and everything
/// that is *not* CommonMark lives behind this protocol.
///
/// Processors see one block at a time and return nil to mean "unchanged",
/// which lets the caller skip rebuilding a block that nothing touched.
protocol MarkdownPostProcessor: Sendable {
    func process(
        _ block: MarkdownItem.TextBlock,
        context: MarkdownPostProcessContext
    ) -> MarkdownItem.TextBlock?
}

/// What a processor knows about the stream it is being run inside.
///
/// `isComplete` matters for anything with a delimiter: mid-stream, a lone `$`
/// may be the opening half of a formula whose closing half has not arrived, so
/// a processor should leave it alone rather than commit to an interpretation it
/// will have to undo. ChatGPT carries the same flag on
/// `MarkdownPostProcessStreamingContext`, alongside a `didRunCompletionScan`
/// bit that forces one final pass once the stream ends.
struct MarkdownPostProcessContext: Sendable {
    /// False while tokens are still arriving.
    public var isComplete: Bool

    public init(isComplete: Bool) {
        self.isComplete = isComplete
    }
}

extension MarkdownResult {
    /// Run `processors` over every text block.
    ///
    /// Deliberately a full re-scan on each call. ChatGPT's
    /// `IncrementalTextPostProcessCache` tracks a `scannedCharacterOffset` and
    /// only walks new text, with a `rewindOffset` so a delimiter split across
    /// two flushes is still seen — worth doing when it is measurably needed,
    /// but that is a cache to add against a profile, not on speculation.
    public func postProcessed(
        with processors: [MarkdownPostProcessor],
        context: MarkdownPostProcessContext
    ) -> MarkdownResult {
        guard !processors.isEmpty else { return self }
        var anyChanged = false
        let newItems = items.map { item -> MarkdownItem in
            guard case let .text(block) = item else { return item }
            var current = block
            var blockChanged = false
            for processor in processors {
                if let next = processor.process(current, context: context) {
                    current = next
                    blockChanged = true
                }
            }
            guard blockChanged else { return item }
            anyChanged = true
            return .text(current)
        }
        return anyChanged ? MarkdownResult(items: newItems, revision: revision) : self
    }
}
