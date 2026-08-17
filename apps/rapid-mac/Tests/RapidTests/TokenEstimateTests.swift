import Foundation
import Testing
@testable import Rapid

/// Contracts for ``TokenEstimate`` and the layout-noise collapsing that feeds
/// it. Both exist because prompt cost is paid in TOKENS, and the previous
/// `characters / 4` rule mispriced real documents badly enough to double the
/// time to first answer on a Chinese PDF.
///
/// Expected ratios are pinned against measurements taken with the Qwen3.5
/// tokenizer over the same text — see ``TokenEstimate`` for the calibration
/// notes. The assertions below are ranges, not equalities: the estimator is
/// deliberately approximate, and pinning exact numbers would make it
/// un-retunable.
@Suite("Token estimation")
struct TokenEstimateTests {

    // MARK: - Per-script cost

    @Test("CJK text costs far more per character than Latin text")
    func cjkCostsMoreThanLatin() {
        let chinese = String(repeating: "人工智能代理系统", count: 100)   // 800 chars
        let english = String(repeating: "artificial ", count: 73)        // ~800 chars

        let cjkTokens = TokenEstimate.tokens(in: chinese)
        let latinTokens = TokenEstimate.tokens(in: english)

        // The whole point: identical character counts, very different cost.
        #expect(abs(chinese.count - english.count) < 50)
        #expect(cjkTokens > latinTokens)
        // Measured ratio is ~1.5x; assert the direction and rough magnitude.
        #expect(Double(cjkTokens) / Double(latinTokens) > 1.3)
    }

    @Test("A Chinese document is not under-counted the way chars/4 did")
    func chineseIsNotUndercounted() {
        // The regression this whole change exists for: 24k characters of this
        // book measured 13,306 real tokens while chars/4 claimed 6,000.
        let chinese = String(repeating: "深入理解智能代理的设计原理与工程实践。", count: 600)
        let naiveEstimate = chinese.count / 4
        let estimate = TokenEstimate.tokens(in: chinese)
        #expect(estimate > naiveEstimate * 2)
    }

    @Test("An empty string costs nothing and any text costs at least one token")
    func boundaryCosts() {
        #expect(TokenEstimate.tokens(in: "") == 0)
        #expect(TokenEstimate.tokens(in: "a") >= 1)
        #expect(TokenEstimate.tokens(in: "文") >= 1)
    }

    // MARK: - Budgeted prefix

    @Test("prefix returns the whole text when it already fits")
    func prefixKeepsShortText() {
        let text = "short enough"
        #expect(TokenEstimate.prefix(text, withinTokens: 10_000) == text)
    }

    @Test("prefix respects the budget and yields fewer characters for CJK")
    func prefixIsTokenBudgetedPerScript() {
        let chinese = String(repeating: "智能代理系统设计", count: 2_000)
        let english = String(repeating: "intelligent agent systems ", count: 2_000)

        let cjkSlice = TokenEstimate.prefix(chinese, withinTokens: 500)
        let latinSlice = TokenEstimate.prefix(english, withinTokens: 500)

        #expect(TokenEstimate.tokens(in: cjkSlice) <= 500)
        #expect(TokenEstimate.tokens(in: latinSlice) <= 500)
        // Same token budget must buy FEWER Chinese characters — that is what
        // keeps the two documents costing the same prompt.
        #expect(cjkSlice.count < latinSlice.count)
    }

    @Test("prefix never splits a grapheme cluster")
    func prefixRespectsGraphemeBoundaries() {
        // A flag emoji is several scalars in one Character; slicing inside it
        // would corrupt the text and can't be expressed as a String index.
        let text = String(repeating: "🇯🇵👨‍👩‍👧‍👦", count: 500)
        for budget in [1, 3, 7, 25, 100] {
            let slice = TokenEstimate.prefix(text, withinTokens: budget)
            #expect(text.hasPrefix(slice))
        }
    }

    @Test("A zero or negative budget yields nothing")
    func nonPositiveBudgetIsEmpty() {
        #expect(TokenEstimate.prefix("content", withinTokens: 0).isEmpty)
        #expect(TokenEstimate.prefix("content", withinTokens: -5).isEmpty)
    }

    // MARK: - Layout noise

    @Test("Dot leaders in a table of contents are collapsed")
    func dotLeadersAreCollapsed() {
        // Measured: one 302-page book's contents held 9,127 of these runs, and
        // they tokenize at ~0.5 tokens/char — the worst case for a BPE vocab.
        let toc = "Introduction. . . . . . . . . . . . . . . . . . . 3\nChapter 1 . . . . . . . . . 7"
        let cleaned = ChatFileAttachment.collapsingLayoutNoise(toc)

        #expect(!cleaned.contains(". . . ."))
        // The information survives; only the padding goes.
        #expect(cleaned.contains("Introduction"))
        #expect(cleaned.contains("3"))
        #expect(cleaned.contains("Chapter 1"))
        #expect(cleaned.contains("7"))
        #expect(TokenEstimate.tokens(in: cleaned) < TokenEstimate.tokens(in: toc))
    }

    @Test("Ordinary punctuation is left alone")
    func realPunctuationSurvives() {
        // Conservative by design: only runs of four or more leaders are
        // touched, so prose, decimals, and ellipses are untouched.
        let prose = "Wait... really? The value was 3.14159 and the ratio 0.25."
        #expect(ChatFileAttachment.collapsingLayoutNoise(prose) == prose)
    }

    @Test("Rule lines are collapsed but short dashes are not")
    func ruleLinesCollapse() {
        let ruled = "Header\n------------\nBody"
        #expect(!ChatFileAttachment.collapsingLayoutNoise(ruled).contains("------------"))

        let dashed = "a well-known state-of-the-art result"
        #expect(ChatFileAttachment.collapsingLayoutNoise(dashed) == dashed)
    }

    @Test("Collapsing shrinks a realistic contents page substantially")
    func collapsingShrinksRealisticTOC() {
        let entries = (1...200).map { n in
            "Section \(n)" + String(repeating: ". ", count: 40) + "\(n * 3)"
        }
        let toc = entries.joined(separator: "\n")
        let cleaned = ChatFileAttachment.collapsingLayoutNoise(toc)

        let before = TokenEstimate.tokens(in: toc)
        let after = TokenEstimate.tokens(in: cleaned)
        // The padding is the overwhelming majority of this page's cost.
        #expect(after < before / 2)
    }
}
