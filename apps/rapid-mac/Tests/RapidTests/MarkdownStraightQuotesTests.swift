import Foundation
import Testing
@testable import Rapid

/// The transcript shows the characters the model emitted.
///
/// swift-markdown enables cmark's "smart" typography by default, which turned
/// a bare `{"city": "Tokyo"}` outside a code fence into `{ “city”: “Tokyo” }`
/// — valid-looking, invalid JSON when copied (0.14.3 dogfood, 2026-09-18).
/// ``MarkdownCompiler.parseOptions`` disables it; these pins keep it off.
@Suite("Markdown compiler keeps straight quotes and ASCII punctuation")
@MainActor
struct MarkdownStraightQuotesTests {

    private func proseText(_ source: String) -> String {
        MarkdownCompiler().compile(source).items.flatMap { item -> [InlineRun] in
            if case .text(let block) = item { return block.runs }
            return []
        }
        .map(\.text)
        .joined()
    }

    @Test("Bare JSON outside a fence keeps its straight double quotes")
    func bareJSONKeepsStraightQuotes() {
        let json = #"{"city": "Tokyo", "country": "Japan", "population": 13962000}"#
        let text = proseText(json)
        #expect(text.contains(#""city""#), "Straight quotes must survive: \(text)")
        #expect(!text.contains("“") && !text.contains("”"), "No typographic quotes may be introduced: \(text)")
    }

    @Test("Apostrophes, double hyphens and ellipses are not typeset either")
    func asciiPunctuationIsPreserved() {
        let text = proseText("It's 'quoted' -- wait... done")
        #expect(text.contains("It's") && text.contains("'quoted'"), "apostrophes rewritten: \(text)")
        #expect(text.contains("--") && text.contains("..."), "dashes or ellipsis rewritten: \(text)")
        #expect(!text.contains("’") && !text.contains("—") && !text.contains("…"), "typographic characters introduced: \(text)")
    }

    @Test("Both parse entry points share the option set")
    func optionsIncludeDisableSmartOpts() {
        #expect(MarkdownCompiler.parseOptions.contains(.disableSmartOpts))
        #expect(MarkdownCompiler.parseOptions.contains(.parseBlockDirectives))
    }
}
