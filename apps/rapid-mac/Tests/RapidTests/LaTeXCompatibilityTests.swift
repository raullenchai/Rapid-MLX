import Foundation
import Testing
import SwiftMath
@testable import Rapid

/// Guards the bridge between the LaTeX models write and the subset SwiftMath
/// parses.
///
/// Every case asserts the thing that matters to a reader — *SwiftMath accepts
/// the result* — rather than the exact string the bridge produced. Asserting
/// on the rewritten text would freeze an implementation detail: there is more
/// than one correct spelling of `\pmod{n}` in SwiftMath's dialect, and a
/// better one should not have to break a test.
@Suite("LaTeX compatibility bridge") @MainActor
struct LaTeXCompatibilityTests {

    private func parses(_ latex: String) -> Bool {
        var error: NSError?
        let list = MTMathListBuilder.build(
            fromString: LaTeXCompatibility.normalized(latex), error: &error
        )
        return list != nil && error == nil
    }

    // MARK: - The reported failure

    /// The formula that sent this bug in: a model explaining Fermat's little
    /// theorem, taken verbatim from `conversations.json`. `\mod` is the whole
    /// reason the block rendered as monospaced `$$…$$` source.
    @Test("The reported formula renders")
    func reportedFormulaRenders() {
        #expect(parses("a^{\\varphi(p)} \\equiv 1 \\mod p"))
        #expect(parses("a^{p-1} \\equiv 1 \\mod p"))
    }

    @Test("The whole mod family renders", arguments: [
        "a \\mod p",
        "a \\bmod p",
        "a \\pmod p",
        "a \\pmod{n}",
        "a \\equiv b \\pmod{n^2}",
    ])
    func modFamilyRenders(_ latex: String) {
        #expect(parses(latex))
    }

    /// The reason `\mod` is registered rather than substituted textually: a
    /// rewrite keyed on the characters `\mod` would corrupt every command
    /// that starts with them. SwiftMath's parser takes the longest run of
    /// letters, so registration cannot.
    ///
    /// Only the parse is asserted. A companion `normalized("\\dotsb") ==
    /// "\\dotsb"` was here and was removed: `normalized` does no textual
    /// substitution for *any* registered symbol, so that assertion could not
    /// fail for this implementation — it restated the design instead of
    /// testing it. What keeps the registrations honest is
    /// ``argumentCommandsRender``, which fails when they are removed.
    @Test("Registration does not shadow longer commands")
    func registrationDoesNotShadowLongerCommands() {
        #expect(parses("A \\models B"))
    }

    // MARK: - Environments

    @Test("Numbered environments render", arguments: [
        "\\begin{align} a &= b \\\\ c &= d \\end{align}",
        "\\begin{align} a = b \\\\ c = d \\end{align}",
        "\\begin{align*} a &= b \\end{align*}",
        "\\begin{equation} a = b \\end{equation}",
        "\\begin{equation*} E = mc^2 \\end{equation*}",
        "\\begin{gather} a = b \\\\ c = d \\end{gather}",
        "\\begin{array}{cc} a & b \\\\ c & d \\end{array}",
    ])
    func numberedEnvironmentsRender(_ latex: String) {
        #expect(parses(latex))
    }

    /// The row padding splits on `\\`, which also separates the rows of a
    /// nested environment — so a body holding one is left alone.
    ///
    /// This formula already parses untouched. Splitting it would append a
    /// column to the inner `matrix`'s first row and break something that
    /// worked, which is the one outcome a compatibility bridge must never
    /// produce.
    @Test("A body with a nested environment is left alone")
    func nestedEnvironmentIsNotPadded() {
        let latex = "\\begin{cases} \\begin{matrix} 1 \\\\ 2 \\end{matrix} & a "
            + "\\\\ 0 & b \\end{cases}"
        #expect(LaTeXCompatibility.normalized(latex) == latex)
        #expect(parses(latex))
    }

    /// SwiftMath demands exactly two columns; LaTeX is happy with one, and a
    /// one-column `cases` is what gets written when the branches carry no
    /// condition.
    @Test("One-column cases is padded, two-column is untouched")
    func casesColumnPadding() {
        #expect(parses("\\begin{cases} a \\\\ b \\end{cases}"))
        let twoColumn = "\\begin{cases} 1 & x > 0 \\\\ 0 & x \\le 0 \\end{cases}"
        #expect(LaTeXCompatibility.normalized(twoColumn) == twoColumn)
        #expect(parses(twoColumn))
    }

    // MARK: - Numbering and argument commands

    @Test("Numbering is stripped", arguments: [
        "a = b \\tag{1}",
        "a = b \\tag*{$\\ast$}",
        "a = b \\notag",
        "a = b \\nonumber",
        "\\begin{align} a &= b \\label{eq:one} \\end{align}",
    ])
    func numberingIsStripped(_ latex: String) {
        #expect(parses(latex))
    }

    @Test("Argument commands render", arguments: [
        "\\operatorname{lcm}(a, b)",
        "\\boxed{x = 42}",
        "\\boxed{\\frac{a}{b}}",
        "\\dots",
        "\\lVert x \\rVert",
        // `\left`/`\right` read a different table inside SwiftMath, one with
        // no extension point — so this spelling needs the substitution and
        // not the registration, and the bare form above cannot catch it.
        "\\left\\lVert x \\right\\rVert",
        "\\left\\lVert \\frac{a}{b} \\right\\rVert^2",
    ])
    func argumentCommandsRender(_ latex: String) {
        #expect(parses(latex))
    }

    /// The box is decoration; the answer inside it is not. A rewrite that
    /// dropped the body along with the rule would be worse than the
    /// raw-source fallback it replaces.
    @Test("Unwrapping a command keeps its body")
    func unwrappingKeepsBody() {
        #expect(LaTeXCompatibility.normalized("\\boxed{x = 42}").contains("x = 42"))
        #expect(LaTeXCompatibility.normalized("\\operatorname{lcm}").contains("lcm"))
    }

    /// A group containing an unmatched escaped brace must still end at its
    /// own closing brace. `\left\{ … \right.` is where this shows up in
    /// practice: the `\{` has no `\}` to pair with, so a scanner that counts
    /// it loses the group and leaves the command in place.
    @Test("Escaped braces do not close the group early")
    func escapedBracesDoNotCloseEarly() {
        let matched = LaTeXCompatibility.normalized("\\boxed{\\{1, 2\\}}")
        #expect(matched.contains("1, 2"))
        #expect(!matched.contains("boxed"))

        let unmatched = LaTeXCompatibility.normalized("\\boxed{\\left\\{ x \\right. }")
        #expect(unmatched.contains("\\left\\{ x \\right."))
        #expect(!unmatched.contains("boxed"))
    }

    /// A command name has to end where the scanner thinks it ends.
    /// `\operatornamewithlimits` is amsmath's, begins with `\operatorname`,
    /// and means something else — rewriting its prefix turns one formula
    /// into a different one silently, which is worse than not rewriting.
    @Test("A longer command is not rewritten through its prefix")
    func longerCommandIsNotRewrittenThroughItsPrefix() {
        let latex = "\\operatornamewithlimits{argmax}_x f(x)"
        #expect(LaTeXCompatibility.normalized(latex) == latex)
    }

    // MARK: - Degrading safely

    /// A body the bridge cannot read cleanly is handed on untouched, so it
    /// reaches ``MathView``'s raw-source fallback — the same place it reaches
    /// today — instead of a rewrite that means something else.
    @Test("Unreadable bodies pass through unchanged", arguments: [
        "\\boxed{x = 42",
        "\\operatorname",
        "\\tag",
    ])
    func unreadableBodiesPassThrough(_ latex: String) {
        #expect(LaTeXCompatibility.normalized(latex) == latex)
    }

    /// The overwhelmingly common case: nothing in the formula needs bridging,
    /// and the bridge is required not to disturb it.
    @Test("Bodies needing nothing are returned unchanged", arguments: [
        "E = mc^2",
        "\\frac{a}{b}",
        "\\sum_{i=1}^{n} i^2",
        "\\begin{aligned} a &= b \\end{aligned}",
        "\\int_0^1 x \\, dx",
    ])
    func untouchedBodies(_ latex: String) {
        #expect(LaTeXCompatibility.normalized(latex) == latex)
        #expect(parses(latex))
    }
}
