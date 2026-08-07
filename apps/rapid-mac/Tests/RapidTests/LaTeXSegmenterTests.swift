import Foundation
import Testing
@testable import Rapid

/// Issue #131: pin the ``LaTeXSegmenter`` contract. Math-rendering
/// regressions are easy to miss visually (the difference between
/// "rendered fine" and "rendered as source" is only obvious if you
/// know what to look for), so we lean on a thick test that
/// hard-codes the expected segment shape for every wire-shape we
/// see from real models.
@Suite("LaTeXSegmenter — math/markdown split (issue #131)")
struct LaTeXSegmenterTests {

    // MARK: - Empty / no-math hot path

    @Test("Empty input returns zero segments")
    func emptyInput() {
        #expect(LaTeXSegmenter.segment("") == [])
    }

    @Test("Plain markdown with no math returns ONE markdown segment (hot path)")
    func plainMarkdownHotPath() {
        let body = """
        # Title

        A paragraph with **bold** and *italic*.

        - bullet one
        - bullet two
        """
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)],
                "no math markers → exactly one .markdown segment, byte-for-byte input")
    }

    // MARK: - Inline math ($...$)

    @Test("Inline math in prose splits cleanly")
    func inlineMathInProse() {
        let body = "We compute $x^2 + y^2$ to find the radius."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [
            .markdown("We compute "),
            .math(latex: "x^2 + y^2", displayMode: false),
            .markdown(" to find the radius.")
        ])
    }

    @Test("Two inline math runs on the same line")
    func twoInlineMath() {
        let body = "Let $a = 1$ and $b = 2$."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [
            .markdown("Let "),
            .math(latex: "a = 1", displayMode: false),
            .markdown(" and "),
            .math(latex: "b = 2", displayMode: false),
            .markdown(".")
        ])
    }

    @Test("Inline math at start of line emits no leading empty markdown")
    func inlineMathAtStart() {
        let body = "$y = mx + b$ is the formula."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [
            .math(latex: "y = mx + b", displayMode: false),
            .markdown(" is the formula.")
        ])
    }

    // MARK: - Display math ($$...$$)

    @Test("Display math on its own block")
    func displayMathBlock() {
        let body = """
        Integration by parts:

        $$ \\int u \\, dv = uv - \\int v \\, du $$

        Apply twice for x^2 sin(x).
        """
        let segments = LaTeXSegmenter.segment(body)
        let expectedLatex = " \\int u \\, dv = uv - \\int v \\, du "
        #expect(segments == [
            .markdown("Integration by parts:\n\n"),
            .math(latex: expectedLatex, displayMode: true),
            .markdown("\n\nApply twice for x^2 sin(x).")
        ])
    }

    @Test("Multi-line display math (the common case for derivations)")
    func multiLineDisplayMath() {
        let body = """
        $$
        f(x) = ax^2 + bx + c
        $$
        """
        let segments = LaTeXSegmenter.segment(body)
        let expectedLatex = "\nf(x) = ax^2 + bx + c\n"
        #expect(segments == [
            .math(latex: expectedLatex, displayMode: true)
        ])
    }

    // MARK: - Anti-cases (must NOT be treated as math)

    @Test("Fenced code block: $...$ inside ``` stays literal")
    func fencedCodeBlockSurvives() {
        let body = """
        Here's a shell snippet:

        ```bash
        echo "$5.00 paid"
        ```

        Plain prose after.
        """
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)],
                "fenced code body must NOT have its $ treated as math open")
    }

    @Test("Inline backtick code: `$5.00` stays literal")
    func inlineBacktickSurvives() {
        let body = "Today it cost `$5.00` to ship."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)])
    }

    @Test("Escaped dollar (\\$) stays literal")
    func escapedDollarSurvives() {
        let body = "Cost: \\$20 plus tax."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)])
    }

    @Test("Bare dollar with no closer is treated as literal prose")
    func bareDollarNoCloser() {
        let body = "Pay $20 today, please."
        let segments = LaTeXSegmenter.segment(body)
        // Inline math is single-line; the lack of a closing $ on the
        // same line means we fall back to literal markdown.
        #expect(segments == [.markdown(body)],
                "bare $ with no inline close → literal markdown, no false-positive math")
    }

    @Test("Single $ followed by newline does NOT open math")
    func dollarBeforeNewline() {
        let body = "Total: $20\nNext line"
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)])
    }

    // MARK: - Edge cases worth pinning

    @Test("Display math right after inline math, no plain between")
    func displayRightAfterInline() {
        let body = "$x$$$y$$"
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [
            .math(latex: "x", displayMode: false),
            .math(latex: "y", displayMode: true)
        ])
    }

    // MARK: - Codex r1 anti-cases (#131)

    @Test("Currency dollars on one line: $20 to $30 stays literal")
    func currencyDollarsOnOneLine() {
        let body = "Revenue rose from $20 to $30 this quarter."
        let segments = LaTeXSegmenter.segment(body)
        // Codex r1 P1 (#131): MathJax convention — $ followed by a
        // digit is currency, not math. Without the guard, the two $
        // would otherwise pair as an inline math run.
        #expect(segments == [.markdown(body)],
                "currency-dollar pairs must NOT become a math segment")
    }

    @Test("Currency followed by non-digit math: $x$ to $20 splits cleanly")
    func currencyAfterRealMath() {
        let body = "Let $x$ be the input, and $20 be the price."
        let segments = LaTeXSegmenter.segment(body)
        // First $...$ is real math; second $ is currency (digit follows).
        // With the guard, the second $ is rejected and never opens math,
        // so the tail stays a single markdown segment.
        #expect(segments == [
            .markdown("Let "),
            .math(latex: "x", displayMode: false),
            .markdown(" be the input, and $20 be the price.")
        ])
    }

    @Test("4-space-indented code block: $x$ stays literal markdown")
    func indentedCodeBlockSurvives() {
        // CommonMark treats 4+ leading spaces as a code block. The
        // segmenter must NOT scan its body for dollars.
        let body = """
        Before:

            echo "$x = 1$"
            echo "$y$$z$"

        After.
        """
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)],
                "indented code block contents must stay literal")
    }

    @Test("Tab-indented code block: $y$ stays literal markdown")
    func tabIndentedCodeSurvives() {
        let body = "Snippet:\n\n\tprintf \"$x$\\n\"\n\nDone."
        let segments = LaTeXSegmenter.segment(body)
        #expect(segments == [.markdown(body)],
                "tab-indented code block contents must stay literal")
    }

    // MARK: - LaTeXMarkdownView.displayMathOnly collapse (#131 v1)

    @Test("displayMathOnly: pure markdown passes through unchanged")
    func displayMathOnlyPassThrough() {
        let segs: [LaTeXSegment] = [.markdown("hello world")]
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == segs)
    }

    @Test("displayMathOnly: inline math is re-wrapped back into adjacent markdown")
    func displayMathOnlyInlineCollapse() {
        // The segmenter emitted three pieces around an inline run; the
        // view's v1 collapse should fold them into ONE markdown segment
        // so MarkdownUI can keep the surrounding paragraph intact.
        let segs: [LaTeXSegment] = [
            .markdown("We compute "),
            .math(latex: "x^2 + y^2", displayMode: false),
            .markdown(" to find the radius.")
        ]
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == [
            .markdown("We compute $x^2 + y^2$ to find the radius.")
        ])
    }

    @Test("displayMathOnly: display math is preserved as its own segment")
    func displayMathOnlyDisplayPreserved() {
        let segs: [LaTeXSegment] = [
            .markdown("Result:\n\n"),
            .math(latex: "f(x) = ax^2 + bx + c", displayMode: true),
            .markdown("\n\nDone.")
        ]
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == segs)
    }

    @Test("displayMathOnly: mixed inline+display in one body")
    func displayMathOnlyMixed() {
        // Inline math is folded back; display math survives. Around the
        // display segment, the markdown chunks remain separate so
        // MarkdownUI doesn't have to render the display math as source.
        let segs: [LaTeXSegment] = [
            .markdown("Inline "),
            .math(latex: "a", displayMode: false),
            .markdown(" then\n\n"),
            .math(latex: "b = c", displayMode: true),
            .markdown("\n\nand inline "),
            .math(latex: "d", displayMode: false),
            .markdown(" again.")
        ]
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == [
            .markdown("Inline $a$ then\n\n"),
            .math(latex: "b = c", displayMode: true),
            .markdown("\n\nand inline $d$ again.")
        ])
    }

    // MARK: - LaTeX bracket delimiters, \( … \) and \[ … \]
    //
    // 2026-08 dogfood regression. `bonsai-27b-2bit` (reproduced on a
    // DeepSeek model) answered a plain word problem using ONLY the
    // bracket delimiters — no `$` anywhere. The segmenter knew only
    // `$`, so the whole body went to MarkdownUI, and CommonMark's
    // backslash-escape rule ate the delimiters: `\(`, `\)`, `\[`, `\]`
    // all wrap ASCII punctuation, so they collapse to bare brackets,
    // while `\frac` / `\times` (backslash + letter, not an escape)
    // survive verbatim. Verified independently:
    //   AttributedString(markdown: #"So: \[ 0.85P = 47 \]"#)
    //     → "So: [ 0.85P = 47 ]"
    // which is byte-for-byte what the dogfood screenshots showed.

    @Test("Inline bracket math \\( … \\) splits like $ … $")
    func inlineBracketMath() {
        let segments = LaTeXSegmenter.segment(#"Let \( P \) be the price."#)
        #expect(segments == [
            .markdown("Let "),
            .math(latex: " P ", displayMode: false),
            .markdown(" be the price.")
        ])
    }

    @Test("Display bracket math \\[ … \\] splits like $$ … $$")
    func displayBracketMath() {
        let segments = LaTeXSegmenter.segment(#"So: \[ 0.85P = 47 \]"#)
        #expect(segments == [
            .markdown("So: "),
            .math(latex: " 0.85P = 47 ", displayMode: true)
        ])
    }

    @Test("Multi-line display bracket math")
    func multiLineDisplayBracketMath() {
        let segments = LaTeXSegmenter.segment("\\[\nf(x) = ax^2\n\\]")
        #expect(segments == [.math(latex: "\nf(x) = ax^2\n", displayMode: true)])
    }

    @Test("LaTeX row break \\\\ inside \\[ … \\] does not fake a closer")
    func rowBreakDoesNotClose() {
        // ``\\`` is LaTeX's row break. Read one character at a time its
        // second backslash + a following ``]`` would look like ``\]``,
        // truncating the formula. The scanner must consume both.
        let segments = LaTeXSegmenter.segment(#"\[ a \\ b \]"#)
        #expect(segments == [.math(latex: #" a \\ b "#, displayMode: true)])
    }

    @Test("\\right] inside display math does not close the run early")
    func rightBracketDoesNotClose() {
        let segments = LaTeXSegmenter.segment(#"\[ x \in \left[0,1\right] \]"#)
        #expect(segments == [.math(latex: #" x \in \left[0,1\right] "#, displayMode: true)])
    }

    @Test("Dollar and bracket delimiters mix in one body")
    func mixedDollarAndBracketDelimiters() {
        let segments = LaTeXSegmenter.segment(#"$a$ and \(b\) and $$c$$ and \[d\]"#)
        #expect(segments == [
            .math(latex: "a", displayMode: false),
            .markdown(" and "),
            .math(latex: "b", displayMode: false),
            .markdown(" and "),
            .math(latex: "c", displayMode: true),
            .markdown(" and "),
            .math(latex: "d", displayMode: true)
        ])
    }

    @Test("Inline bracket math may wrap across a single newline")
    func inlineBracketWrapsOneNewline() {
        // Unlike ``$``, ``\(`` is unambiguous, so the single-line guard
        // that protects prose dollars is not needed here — only a blank
        // line ends the search.
        let segments = LaTeXSegmenter.segment("Val \\( a +\nb \\) done.")
        #expect(segments == [
            .markdown("Val "),
            .math(latex: " a +\nb ", displayMode: false),
            .markdown(" done.")
        ])
    }

    // MARK: - Bracket anti-cases (opener scan)
    //
    // NOTE for future maintainers: every test in THIS section also
    // passes against the pre-fix segmenter, which treated bracket
    // delimiters as plain text and so could never produce a false
    // positive. They do not pin the fix — they pin that the fix stays
    // conservative. The tests above are the ones that fail on a revert.
    //
    // These four also only exercise the OPENER scan: the opener sits
    // inside the code region, so the run never starts and the closer
    // scan is never entered. The "closer scan" section below covers
    // the mirror image — opener in prose, closer inside the region —
    // which is where the interesting failures live.

    @Test("Unclosed \\( stays literal markdown")
    func unclosedBracketOpener() {
        let body = #"Use \( to open a group."#
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("\\\\( is an escaped backslash, not a math opener")
    func escapedBackslashBeforeParen() {
        let body = #"Escaped: \\(not math\\)"#
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("Inline bracket math stops at a blank line")
    func inlineBracketStopsAtParagraphBreak() {
        // An opener the model forgot to close must not swallow the
        // rest of the reply into "math".
        let body = "Start \\( x\n\nEnd \\) done."
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("Bracket math inside a fenced code block stays literal")
    func bracketMathInFenceSurvives() {
        let body = "Snippet:\n\n```tex\n\\( x \\) and \\[ y \\]\n```\n\nAfter."
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("Bracket math inside a code span stays literal")
    func bracketMathInCodeSpanSurvives() {
        let body = #"Write `\(x\)` for inline math."#
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("Bracket math inside an indented code block stays literal")
    func bracketMathInIndentedCodeSurvives() {
        let body = "Before:\n\n    echo \"\\( x \\)\"\n\nAfter."
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    // MARK: - Closer scan
    //
    // Every test here has an opener in PROSE and a candidate closer
    // somewhere the scan must not honour, so the run is actually
    // started and `findBracketClose` actually walks. That is the gap
    // the opener-side anti-cases above leave open.

    @Test("Nested \\( : the first opener is prose, the second renders")
    func nestedInlineOpenerIsProse() {
        // LaTeX cannot nest inline math, so a second \( before any
        // closer means the first one was never math. Without the bail
        // the first opener reaches the SECOND expression's closer and
        // eats the prose between them.
        let segments = LaTeXSegmenter.segment(#"Use \( to group, then \(x\)."#)
        #expect(segments == [
            .markdown(#"Use \( to group, then "#),
            .math(latex: "x", displayMode: false),
            .markdown(".")
        ])
    }

    @Test("Nested \\[ : the first opener is prose, the second renders")
    func nestedDisplayOpenerIsProse() {
        let segments = LaTeXSegmenter.segment(#"Use \[ then \[x\]."#)
        #expect(segments == [
            .markdown(#"Use \[ then "#),
            .math(latex: "x", displayMode: true),
            .markdown(".")
        ])
    }

    @Test("A nested opener of the OTHER kind does not abandon the run")
    func nestedOtherKindOpenerKeepsRun() {
        // Deliberate asymmetry: bailing here would hand the formula
        // back to CommonMark, which strips \[ \] to bare brackets —
        // the original bug. Keeping it degrades to MathView's
        // literal-source fallback instead, which shows more.
        let segments = LaTeXSegmenter.segment(#"\[ a \( b \) c \]"#)
        #expect(segments == [.math(latex: #" a \( b \) c "#, displayMode: true)])
    }

    @Test("A closer inside a fenced code block does not close the run")
    func closerInsideFenceIsNotAClose() {
        let body = "Use \\[ to open.\n\n```tex\n\\]\n```\n\nDone."
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("A closer inside a code span does not close the run")
    func closerInsideCodeSpanIsNotAClose() {
        let body = #"Use \[ to open, and `\]` to close."#
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("A closer inside an indented code block does not close the run")
    func closerInsideIndentedCodeIsNotAClose() {
        let body = "Use \\[ to open.\n\n    echo \"\\]\"\n\nDone."
        #expect(LaTeXSegmenter.segment(body) == [.markdown(body)])
    }

    @Test("An indented CONTINUATION line still closes the run")
    func indentedContinuationStillCloses() {
        // The counterweight to the three tests above. Math bodies are
        // routinely indented, and an indented line that merely follows
        // the `\[` line is not a CommonMark code block — there is no
        // blank line before it. Testing indentation alone (what the
        // opener scan does) would abandon this formula.
        let segments = LaTeXSegmenter.segment("\\[\n    P = 47 \\]")
        #expect(segments == [.math(latex: "\n    P = 47 ", displayMode: true)])
    }

    @Test("First closer wins for a stray opener in plain prose")
    func firstCloserWinsInProse() {
        // Pinning the decision, not an accident of the scan: with no
        // nested opener and no code region in between, the first
        // closer closes. Accepted because CommonMark would have
        // rendered both bracket escapes as bare brackets anyway.
        let segments = LaTeXSegmenter.segment(#"Use \[ then later \] here."#)
        #expect(segments == [
            .markdown("Use "),
            .math(latex: " then later ", displayMode: true),
            .markdown(" here.")
        ])
    }

    // MARK: - The two dogfood answers, verbatim

    /// The shirt-discount answer exactly as `bonsai-27b-2bit` emitted
    /// it. The user-visible contract is the assertion below: NOTHING
    /// that MarkdownUI will mangle may remain in a `.markdown`
    /// segment.
    private static let discountAnswer = #"""
    Let \( P \) be the original price of the shirt.
    A 15% discount means you pay \( 100\% - 15\% = 85\% \) of the original price.
    So: \[ 0.85P = 47 \]
    Solving for \( P \): \[ P = \frac{47}{0.85} \approx 55.29 \]
    Answer: The original price was $55.29.
    """#

    private static let billSplitAnswer = #"""
    Let \( B \) = Bob's payment
    Alice pays 40% more than Bob: \( A = 1.4B \)
    Substitute \( A = 1.4B \): \[ C = \frac{1}{2}(2.4B) = 1.2B \]
    """#

    /// Concatenation of everything that would be handed to MarkdownUI.
    private static func markdownHandedToRenderer(_ body: String) -> String {
        LaTeXSegmenter.segment(body).compactMap { segment -> String? in
            guard case .markdown(let text) = segment else { return nil }
            return text
        }.joined()
    }

    @Test("Discount answer: no delimiter or LaTeX command reaches MarkdownUI")
    func discountAnswerFullySegmented() {
        let markdown = Self.markdownHandedToRenderer(Self.discountAnswer)
        // Delimiters: CommonMark would strip these to bare brackets.
        #expect(!markdown.contains(#"\("#))
        #expect(!markdown.contains(#"\)"#))
        #expect(!markdown.contains(#"\["#))
        #expect(!markdown.contains(#"\]"#))
        // Commands: CommonMark leaves these as visible source.
        #expect(!markdown.contains(#"\frac"#))
        #expect(!markdown.contains(#"\approx"#))
    }

    @Test("Bill-split answer: no delimiter or LaTeX command reaches MarkdownUI")
    func billSplitAnswerFullySegmented() {
        let markdown = Self.markdownHandedToRenderer(Self.billSplitAnswer)
        #expect(!markdown.contains(#"\("#))
        #expect(!markdown.contains(#"\)"#))
        #expect(!markdown.contains(#"\["#))
        #expect(!markdown.contains(#"\]"#))
        #expect(!markdown.contains(#"\frac"#))
    }

    @Test("Dogfood answer: the two display formulas become display-math segments")
    func discountAnswerDisplayMath() {
        let display = LaTeXSegmenter.segment(Self.discountAnswer).compactMap { segment -> String? in
            guard case .math(let latex, let displayMode) = segment, displayMode else { return nil }
            return latex
        }
        #expect(display == [" 0.85P = 47 ", #" P = \frac{47}{0.85} \approx 55.29 "#])
    }

    @Test("Dogfood answer: trailing currency is NOT swallowed as math")
    func discountAnswerCurrencyUntouched() {
        // "$55.29." ends the reply. The currency guard must keep it in
        // markdown — a lone trailing `$` has no closer anyway, but this
        // pins that the bracket work did not disturb it.
        let segments = LaTeXSegmenter.segment(Self.discountAnswer)
        #expect(segments.last == LaTeXSegment.markdown("\nAnswer: The original price was $55.29."))
    }

    @Test("Round-trip: segments concatenated with delimiters re-form input")
    func roundTrip() {
        let inputs = [
            "Plain text.",
            "Inline $x^2$ in middle.",
            "$$y = mx + b$$",
            "Mixed: $a$ then $$b$$ then $c$.",
            "Code: ```\n$x$\n``` keeps source.",
        ]
        for body in inputs {
            let segments = LaTeXSegmenter.segment(body)
            let reconstructed = segments.map { seg -> String in
                switch seg {
                case .markdown(let s): return s
                case .math(let latex, let display):
                    return display ? "$$\(latex)$$" : "$\(latex)$"
                }
            }.joined()
            #expect(reconstructed == body,
                    "segmenter must be lossless: '\(body)' → '\(reconstructed)'")
        }
    }

    @Test("Round-trip: bracket delimiters normalise onto their $ form")
    func roundTripBracketNormalises() {
        // Delimiter STYLE is deliberately not preserved — `\(x\)` and
        // `$x$` mean the same thing and produce the same segment. So
        // the round-trip lands on the `$` form, not byte-for-byte
        // input. This pins that choice.
        let segments = LaTeXSegmenter.segment(#"Let \(x\) then \[y\]"#)
        let reconstructed = segments.map { seg -> String in
            switch seg {
            case .markdown(let s): return s
            case .math(let latex, let display):
                return display ? "$$\(latex)$$" : "$\(latex)$"
            }
        }.joined()
        #expect(reconstructed == "Let $x$ then $$y$$")
    }

    @Test("displayMathOnly: bracket-sourced inline math folds back as $ … $")
    func displayMathOnlyBracketInlineCollapse() {
        // v1 still does not RENDER inline math (see the deferral note
        // on ``LaTeXMarkdownView``). What changes with the bracket fix
        // is that the fold-back emits `$A = 1.4B$` rather than letting
        // CommonMark strip `\( … \)` down to a bare `( A = 1.4B )`.
        let segs = LaTeXSegmenter.segment(#"Alice pays: \( A = 1.4B \) total."#)
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == [
            .markdown("Alice pays: $ A = 1.4B $ total.")
        ])
    }

    @Test("displayMathOnly: bracket display math survives as its own segment")
    func displayMathOnlyBracketDisplaySurvives() {
        let segs = LaTeXSegmenter.segment(#"So: \[ 0.85P = 47 \] done."#)
        #expect(LaTeXMarkdownView.displayMathOnly(segs) == [
            .markdown("So: "),
            .math(latex: " 0.85P = 47 ", displayMode: true),
            .markdown(" done.")
        ])
    }
}
