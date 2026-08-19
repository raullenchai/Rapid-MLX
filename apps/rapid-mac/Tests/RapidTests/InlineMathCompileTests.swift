import Foundation
import Testing
@testable import Rapid

/// Inline math survives markdown parsing and comes out as its own run.
///
/// `LaTeXSegmenter` has always recognised `$…$` and `\(…\)`; the compiler then
/// threw the result away, re-wrapping each formula back into the prose it
/// handed the parser (`pending += "$\(latex)$"`). That was the safe choice at
/// the time — `MarkdownCompiler`'s own comment calls it "correct-but-unstyled,
/// rather than a shattered sentence" — but it left inline math rendering as
/// literal dollar signs.
///
/// Re-wrapping cannot simply stop: `$x_1$` handed back to the parser becomes
/// `x`, emphasis, `1` — the underscore is markdown syntax and the subscript is
/// gone. A sentinel carries the formula through parsing instead, and is
/// swapped back afterwards.
@Suite("Inline math compilation")
@MainActor
struct InlineMathCompileTests {

    private func runs(_ source: String) -> [InlineRun] {
        MarkdownCompiler().compile(source).items.flatMap { item -> [InlineRun] in
            if case .text(let block) = item { return block.runs }
            return []
        }
    }

    private func maths(_ source: String) -> [String] {
        runs(source).compactMap(\.math)
    }

    /// The case that forced the sentinel: an underscore inside a formula is
    /// emphasis syntax to the markdown parser.
    @Test("A subscript survives the markdown parser")
    func subscriptSurvives() {
        #expect(maths("The value $x_1$ matters.") == ["x_1"])
        #expect(maths("Sum $a_i + b_j$ here.") == ["a_i + b_j"])
    }

    /// Both spellings the segmenter accepts. The bracket form is not optional:
    /// some models emit only `\(…\)`.
    @Test("Both delimiter spellings produce math")
    func bothSpellings() {
        #expect(maths("Inline $y$ form.") == ["y"])
        #expect(maths("Inline \\(y_2\\) form.") == ["y_2"])
    }

    /// A formula inherits the styling of the run it came from, so emphasis and
    /// links do not stop at the formula's edge.
    @Test("Surrounding inline style is inherited")
    func stylingIsInherited() {
        let bold = runs("Bold **$e^{i\\pi}$** here.").first { $0.math != nil }
        #expect(bold?.math == "e^{i\\pi}")
        #expect(bold?.isStrong == true)
    }

    /// The formula keeps its source spelling in `text`, so copy, VoiceOver and
    /// search see `$x$` rather than the private-use sentinel.
    @Test("The run still carries readable text")
    func textIsStillReadable() {
        let math = runs("Value $x$ here.").first { $0.math != nil }
        #expect(math?.text == "$x$")
    }

    /// Math reaches every place runs live, not just top-level paragraphs.
    /// A formula in a table cell is the case the original #131 review named.
    @Test("Math is restored inside tables and list items")
    func mathInsideNestedBlocks() {
        let table = MarkdownCompiler()
            .compile("| a | b |\n|---|---|\n| $x^2$ | 2 |").items
        var cellMath: [String] = []
        for case .table(let block) in table {
            for row in block.rows { for cell in row { cellMath += cell.compactMap(\.math) } }
        }
        #expect(cellMath == ["x^2"])
        #expect(maths("- item with $\\frac{1}{2}$") == ["\\frac{1}{2}"])
    }

    /// The segmenter's guards must still hold end to end — these are the
    /// false positives that would turn ordinary prose into formulas.
    @Test("Prose that merely contains dollars stays prose")
    func noFalsePositives() {
        #expect(maths("Cost is $20 to $30 today.").isEmpty)
        #expect(maths("Code `$x$` stays literal.").isEmpty)
        #expect(maths("```\n$x$\n```").isEmpty)
    }

    /// Display math keeps its own block — this change is about inline only.
    @Test("Display math is untouched")
    func displayMathUnchanged() {
        let items = MarkdownCompiler().compile("Before\n\n$$x^2$$\n\nAfter").items
        let display = items.compactMap { item -> String? in
            if case .math(let block) = item { return block.latex }
            return nil
        }
        #expect(display == ["x^2"])
    }

    /// No sentinel may reach a reader. If expansion ever fails, this is the
    /// symptom: a private-use character rendered as a missing glyph.
    @Test("No sentinel leaks into rendered text")
    func noSentinelLeaks() {
        for source in [
            "The value $x_1$ matters.",
            "Bold **$e^{i\\pi}$** here.",
            "| a | $x^2$ |\n|---|---|\n| 1 | 2 |",
            "- item with $\\frac{1}{2}$",
        ] {
            for run in runs(source) {
                #expect(!run.text.contains("\u{E000}"), "sentinel leaked in: \(source)")
                #expect(!run.text.contains("\u{E001}"), "sentinel leaked in: \(source)")
            }
        }
    }
}
