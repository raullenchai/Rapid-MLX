import Foundation
import SwiftMath

/// Bridges the LaTeX models write to the subset SwiftMath parses.
///
/// ``MathView`` falls back to showing the raw `$$…$$` source in monospace
/// when SwiftMath cannot parse a body — deliberately, so the reader is not
/// left with a hole. But a model writing ordinary number theory trips that
/// fallback on its first line: `\mod` is not in SwiftMath's table, and one
/// unknown command fails the whole formula, not just the token. Measured
/// against 46 commands common in chat output, 15 failed.
///
/// Two mechanisms, chosen per command:
///
/// * **Registration**, via SwiftMath's own `add(latexSymbol:)` extension
///   point, for commands that are simply missing from its table. Preferred
///   wherever it fits, because the parser reads the longest run of letters
///   after a backslash as the command name — teaching it `mod` cannot
///   misfire on `\models`, whereas a textual substitution would.
/// * **Source rewriting**, for anything registration cannot express: a
///   command that takes an argument, or an environment name.
///
/// Every rewrite is lossy in some way and each one says what it gives up.
/// The bar is that the reader sees typeset mathematics that means what the
/// model meant, not that the output is pixel-identical to LaTeX's.
///
/// The original source is *not* replaced anywhere it is shown to a person:
/// ``MathView``'s fallback text and accessibility label, and the inline
/// renderer's prose fallback, all keep what the model actually wrote.
@MainActor
enum LaTeXCompatibility {

    /// The body to hand SwiftMath. Returns `latex` unchanged when nothing
    /// applies, which is the common case.
    static func normalized(_ latex: String) -> String {
        _ = missingSymbolsRegistered
        var source = latex
        // Numbering first: `\tag` can sit inside an `align` row, and the
        // `cases` padding below counts columns in rows this has cleaned.
        source = removingNumbering(source)
        source = rewritingEnvironments(source)
        source = rewritingArgumentCommands(source)
        return source
    }

    // MARK: - Commands SwiftMath does not have

    /// Registered once, lazily, on the first formula rather than at launch —
    /// a chat that never shows mathematics never pays for it.
    ///
    /// Each atom is built fresh: `MTMathAtom` is a class and layout mutates
    /// it, so two table entries must not share one instance.
    private static let missingSymbolsRegistered: Void = {
        // `\mod` and `\bmod` differ in LaTeX only by the space before them.
        // Both become the same operator here; the spacing difference is not
        // worth a second code path in a chat transcript.
        for name in ["mod", "bmod"] {
            MTMathAtomFactory.add(
                latexSymbol: name,
                value: MTMathAtomFactory.operatorWithName("mod", limits: false)
            )
        }
        // `\dots` is amsmath's context-sensitive ellipsis — it becomes a
        // baseline or a centred row depending on what surrounds it. SwiftMath
        // has only the two explicit forms, so this aliases the baseline one,
        // which is what `\dots` resolves to in the great majority of uses.
        alias("dots", to: "ldots")
        // The `\lVert`/`\rVert` pair carries LaTeX's opening/closing spacing;
        // aliasing both to the plain double bar keeps the glyph and loses the
        // asymmetric spacing around it.
        alias("lVert", to: "|")
        alias("rVert", to: "|")
    }()

    /// Teaches SwiftMath `name` by pointing it at a symbol it already has.
    /// Silent when the target is missing: a bridge that cannot be built is a
    /// formula that falls back to its source, which is where it is today.
    private static func alias(_ name: String, to existing: String) {
        guard let atom = MTMathAtomFactory.atom(forLatexSymbol: existing) else { return }
        MTMathAtomFactory.add(latexSymbol: name, value: atom)
    }

    // MARK: - Numbering

    /// Equation numbering has no meaning in a chat transcript — there is no
    /// document to cross-reference into — and SwiftMath rejects all of it.
    private static func removingNumbering(_ source: String) -> String {
        var source = rewritingCommand(source, "tag") { _ in "" }
        source = rewritingCommand(source, "label") { _ in "" }
        source = source.replacingOccurrences(of: "\\nonumber", with: "")
        source = source.replacingOccurrences(of: "\\notag", with: "")
        return source
    }

    // MARK: - Environments

    private static func rewritingEnvironments(_ source: String) -> String {
        var source = source
        // `aligned` is SwiftMath's spelling of the same grid. `gather` maps
        // there too: it centres its rows where `aligned` does not, which
        // costs horizontal placement and keeps the line structure.
        for name in ["align*", "align", "gather*", "gather"] {
            source = source
                .replacingOccurrences(of: "\\begin{\(name)}", with: "\\begin{aligned}")
                .replacingOccurrences(of: "\\end{\(name)}", with: "\\end{aligned}")
        }
        // `equation` wraps a single body and exists only to number it. The
        // wrapper goes; the body is already display math by the time it is
        // here, because it arrived inside `$$…$$`.
        for name in ["equation*", "equation"] {
            source = source
                .replacingOccurrences(of: "\\begin{\(name)}", with: "")
                .replacingOccurrences(of: "\\end{\(name)}", with: "")
        }
        source = rewritingArrayEnvironment(source)
        source = paddingRows(source)
        return source
    }

    /// `array` carries a column specification SwiftMath's `matrix` has no
    /// place for. Dropping the spec loses per-column alignment and keeps the
    /// grid, which is the part that carries the meaning.
    private static func rewritingArrayEnvironment(_ source: String) -> String {
        var source = source
        while let opening = source.range(of: "\\begin{array}") {
            var end = opening.upperBound
            var probe = end
            while probe < source.endIndex, source[probe] == " " {
                probe = source.index(after: probe)
            }
            if probe < source.endIndex, source[probe] == "{",
               let close = balancedClose(source, from: probe) {
                end = source.index(after: close)
            }
            source.replaceSubrange(opening.lowerBound..<end, with: "\\begin{matrix}")
        }
        return source.replacingOccurrences(of: "\\end{array}", with: "\\end{matrix}")
    }

    /// SwiftMath requires `cases` and `aligned` to have exactly two columns;
    /// LaTeX is happy with one. One column is what gets written when the
    /// branches carry no condition, and — after the renames above — it is
    /// also every `gather` row and every `align` row the author did not
    /// bother to align. Padding an empty second cell renders the same rows.
    ///
    /// Runs after the renames, so it sees `gather` and `align` in their
    /// `aligned` spelling. Rows with more than two columns are left alone —
    /// no padding rescues those, and leaving them reaches the same raw-source
    /// fallback they reach today.
    private static func paddingRows(_ source: String) -> String {
        var source = source
        for environment in ["cases", "aligned"] {
            source = mapEnvironmentBody(source, environment) { body in
                // `\\` separates the rows of a *nested* environment too, so
                // splitting a body that holds one cuts in the wrong places
                // and the padding lands in the wrong grid. Such a body is
                // left alone: SwiftMath will likely reject it and fall back
                // to the raw source, which is where it stands today — better
                // than a rewrite that quietly means something else.
                guard !body.contains("\\begin{") else { return body }
                return body.components(separatedBy: "\\\\")
                    .map { $0.contains("&") ? $0 : $0 + " &" }
                    .joined(separator: "\\\\")
            }
        }
        return source
    }

    // MARK: - Commands that take an argument

    private static func rewritingArgumentCommands(_ source: String) -> String {
        // `\left`/`\right` resolve their operand through SwiftMath's separate
        // `delimiters` table, which is a `let` with no extension point — so
        // registering `\lVert` above does nothing for the `\left\lVert …
        // \right\rVert` spelling, which is the one written for any norm taller
        // than a single symbol. Substituting the plain double bar is the only
        // route, and loses the same asymmetric spacing the registration does.
        var source = source
            .replacingOccurrences(of: "\\left\\lVert", with: "\\left\\|")
            .replacingOccurrences(of: "\\right\\rVert", with: "\\right\\|")
        // `\operatorname` sets an upright multi-letter name and adds operator
        // spacing around it. `\mathrm` gives the upright letters; the spacing
        // is lost.
        source = rewritingCommand(source, "operatorname") { "\\mathrm{\($0)}" }
        // The box around a model's final answer is decoration. Losing the
        // rule keeps the answer.
        source = rewritingCommand(source, "boxed") { $0 }
        // `\pmod{n}` is spelled out rather than registered because it takes
        // the modulus as an argument and wraps it in parentheses.
        source = rewritingCommand(source, "pmod") { "\\ (\\mathrm{mod}\\ \($0))" }
        return source
    }

    // MARK: - Scanning

    /// Replaces every `\name` and its one argument with `transform(argument)`.
    ///
    /// Leaves the source untouched at any occurrence it cannot read cleanly —
    /// an unbalanced group, or a `\name` at the very end — so a malformed
    /// body degrades to the raw-source fallback rather than to a rewrite that
    /// silently means something else.
    private static func rewritingCommand(
        _ source: String, _ name: String, _ transform: (String) -> String
    ) -> String {
        let characters = Array(source)
        let token = Array("\\" + name)
        var output = ""
        var index = 0

        while index < characters.count {
            // The name must end here: `\tag` must not match inside `\tagged`.
            let matches = index + token.count <= characters.count
                && Array(characters[index..<(index + token.count)]) == token
                && !(index + token.count < characters.count
                     && characters[index + token.count].isLetter)
            guard matches else {
                output.append(characters[index])
                index += 1
                continue
            }

            var cursor = index + token.count
            if cursor < characters.count, characters[cursor] == "*" { cursor += 1 }
            while cursor < characters.count, characters[cursor] == " " { cursor += 1 }
            guard let argument = argument(in: characters, at: cursor) else {
                output.append(characters[index])
                index += 1
                continue
            }
            output += transform(argument.body)
            index = argument.next
        }
        return output
    }

    /// A command's single argument: a balanced `{…}` group, or — as LaTeX
    /// permits when the argument is one token — the next control sequence or
    /// character.
    private static func argument(
        in characters: [Character], at start: Int
    ) -> (body: String, next: Int)? {
        guard start < characters.count else { return nil }

        if characters[start] == "{" {
            var depth = 0
            var index = start
            while index < characters.count {
                // A backslash escapes whatever follows, so `\{` inside the
                // group must not move the depth.
                if characters[index] == "\\" {
                    index += 2
                    continue
                }
                if characters[index] == "{" { depth += 1 }
                if characters[index] == "}" {
                    depth -= 1
                    if depth == 0 {
                        return (String(characters[(start + 1)..<index]), index + 1)
                    }
                }
                index += 1
            }
            return nil
        }

        if characters[start] == "\\" {
            var index = start + 1
            while index < characters.count, characters[index].isLetter { index += 1 }
            // A one-character control sequence such as `\,` has no letters.
            guard index > start + 1 else {
                guard start + 1 < characters.count else { return nil }
                return (String(characters[start...(start + 1)]), start + 2)
            }
            return (String(characters[start..<index]), index)
        }

        return (String(characters[start]), start + 1)
    }

    /// The index of the `}` closing the group that opens at `start`.
    private static func balancedClose(
        _ source: String, from start: String.Index
    ) -> String.Index? {
        var depth = 0
        var index = start
        while index < source.endIndex {
            if source[index] == "\\" {
                index = source.index(index, offsetBy: 2, limitedBy: source.endIndex)
                    ?? source.endIndex
                continue
            }
            if source[index] == "{" { depth += 1 }
            if source[index] == "}" {
                depth -= 1
                if depth == 0 { return index }
            }
            index = source.index(after: index)
        }
        return nil
    }

    /// Rewrites the body of every `\begin{name}…\end{name}` in place.
    private static func mapEnvironmentBody(
        _ source: String, _ name: String, _ transform: (String) -> String
    ) -> String {
        let opening = "\\begin{\(name)}"
        let closing = "\\end{\(name)}"
        var output = ""
        var rest = Substring(source)

        while let open = rest.range(of: opening),
              let close = rest.range(of: closing, range: open.upperBound..<rest.endIndex) {
            output += rest[..<open.upperBound]
            output += transform(String(rest[open.upperBound..<close.lowerBound]))
            output += closing
            rest = rest[close.upperBound...]
        }
        return output + rest
    }
}
