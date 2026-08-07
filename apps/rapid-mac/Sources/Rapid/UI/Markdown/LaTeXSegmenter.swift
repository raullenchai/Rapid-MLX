import Foundation

/// Issue #131: split an assistant message's body into alternating
/// Markdown and LaTeX segments so the chat view can route each kind
/// to the right renderer.
///
/// Why segment in *our* code rather than write a MarkdownUI plugin:
/// MarkdownUI's renderer flattens unknown ``$...$`` runs back into
/// the surrounding paragraph, so the math expression renders as
/// literal source. By splitting first we hand MarkdownUI only
/// markdown it understands and reserve the math runs for
/// ``SwiftMath``'s typesetter.
///
/// Delimiter rules (matched against KaTeX / MathJax defaults that
/// every model in the wild emits):
///
/// * ``$$ ... $$`` — display math. Centered, larger glyph metrics.
///   Can span multiple lines.
/// * ``$ ... $`` — inline math. Sits inside a Markdown paragraph.
///   Single-line only (a literal newline ends the inline run so a
///   stray dollar sign in prose doesn't swallow the rest of the
///   reply into "math").
/// * ``\[ ... \]`` — display math, LaTeX bracket form.
/// * ``\( ... \)`` — inline math, LaTeX bracket form.
///
/// The bracket forms are NOT optional extras: they are what
/// instruction-tuned models actually emit. A 2026-08 dogfood run
/// (`bonsai-27b-2bit`, reproduced on a DeepSeek model) emitted only
/// bracket delimiters for a plain word problem — no ``$`` anywhere.
/// Missing them is not a cosmetic gap, because the raw body then
/// reaches MarkdownUI and CommonMark's backslash-escape rule eats
/// the delimiters: ``\(``, ``\)``, ``\[`` and ``\]`` all wrap ASCII
/// punctuation, so they collapse to bare ``(``, ``)``, ``[``, ``]``
/// while ``\frac`` / ``\times`` (backslash + letter, not a valid
/// escape) survive verbatim. The user sees
/// ``( P = \frac{47}{0.85} \approx 55.29 )`` — delimiters silently
/// stripped, LaTeX left as source. Verified directly:
/// ``AttributedString(markdown: #"So: \[ 0.85P = 47 \]"#)`` yields
/// ``So: [ 0.85P = 47 ]``.
///
/// Delimiter STYLE is not preserved. ``\( x \)`` and ``$x$`` both
/// produce ``.math(latex: "x", displayMode: false)`` — the choice
/// carries no meaning downstream, and normalising keeps the segment
/// enum (and every equality assertion built on it) unchanged. The
/// round-trip property below therefore holds up to that
/// normalisation.
///
/// Anti-cases we intentionally do NOT treat as math:
///
/// * **Inside fenced code blocks** — ``$x$`` written inside
///   ``\`\`\`bash`` should render as literal source, never as math.
///   The segmenter scans for ``\`\`\``-fenced blocks first and
///   leaves their contents untouched.
/// * **Indented (4-space) code blocks** — same reasoning.
/// * **Backtick-quoted inline code** — `` `$5.00` `` is literal
///   shell prose, not math. The segmenter skips ``...`` runs.
/// * **Escaped dollar** — ``\$`` stays a literal dollar sign and
///   never opens a math run.
/// * **Escaped backslash before a bracket** — ``\\(`` is CommonMark's
///   escaped backslash followed by a literal paren, so it does NOT
///   open inline math. Same for ``\\[``.
/// * **Unclosed bracket opener** — ``Use \( to group`` has no ``\)``,
///   so the opener stays literal markdown rather than swallowing the
///   rest of the reply.
/// * **Bare dollar in prose** — ``it costs $20 today`` has only one
///   ``$`` so there's no closing pair; the segmenter requires a
///   close before the next blank line / EOF before opening math.
///   Treated as plain markdown if no closer is found.
/// * **Currency-dollar pairs** (codex r1 P1, #131) — ``$20 to $30``
///   has TWO ``$`` on one line so the bare-dollar guard above
///   doesn't catch it. We also reject any ``$`` whose IMMEDIATELY
///   NEXT character is a digit, mirroring the MathJax convention
///   that ``$`` followed by ``0-9`` is currency, not math. A real
///   math opener almost always starts with a backslash control
///   sequence (``\int``, ``\frac``), a variable letter, or a
///   bracket — never a bare digit on its own.
/// * **Indented code blocks** (codex r1 P2, #131) — CommonMark
///   treats any line indented by 4+ spaces as a code block,
///   regardless of fence markers. The segmenter recognises these
///   and skips the entire indented run before scanning for
///   dollars, so ``    echo "$x"`` stays literal.
///
/// The implementation is a single linear scan over the input
/// string. We track three positions: the start of the current plain
/// run, the cursor, and the start of the candidate math run. When
/// math closes, we emit the leading plain segment + the math
/// segment and slide the plain-run start past the close. EOF emits
/// the trailing plain segment.
enum LaTeXSegment: Equatable {
    /// Markdown body — hand straight to ``MarkdownUI``.
    case markdown(String)
    /// LaTeX body (the delimiters are stripped). ``displayMode`` is
    /// true for ``$$...$$``, false for ``$...$``.
    case math(latex: String, displayMode: Bool)
}

enum LaTeXSegmenter {

    /// Split ``input`` into alternating ``.markdown`` / ``.math``
    /// segments. The returned array reconstructs ``input`` exactly
    /// when ``.markdown`` bodies are concatenated and ``.math``
    /// bodies are re-wrapped with ``$``/``$$``, so a "round-trip"
    /// sanity test can pin the contract. Bodies written with the
    /// bracket delimiters round-trip onto their ``$``-form (see the
    /// normalisation note in the type doc).
    ///
    /// Empty ``.markdown("")`` segments are NOT emitted — the caller
    /// just sees ``[.math, .math]`` if two math runs sit
    /// back-to-back. Same for a math run at the start or end of
    /// the body.
    static func segment(_ input: String) -> [LaTeXSegment] {
        guard !input.isEmpty else { return [] }
        var segments: [LaTeXSegment] = []
        var plainStart = input.startIndex
        var cursor = input.startIndex

        while cursor < input.endIndex {
            let c = input[cursor]

            // Codex r1 P2 (#131): skip CommonMark indented code
            // blocks (4+ leading spaces / tab at start of line)
            // before any dollar scanning. Without this, a snippet
            // like ``    echo '$x$'`` indented under a list item
            // would have its ``$x$`` falsely treated as math.
            if isAtLineStart(input, at: cursor),
               let codeBlockEnd = endOfIndentedCodeBlock(input, lineStart: cursor) {
                cursor = codeBlockEnd
                continue
            }

            // Skip over a fenced code block — search for the
            // matching ``` and slide cursor past it. Conservative:
            // treats ``` at column 0 OR after a newline as a fence.
            if c == "`", isFenceStart(input, at: cursor) {
                cursor = endOfFencedBlock(input, fenceStart: cursor)
                continue
            }

            // Skip over an inline backtick-quoted run — match the
            // shortest closing run of the same length. Matches
            // CommonMark behaviour for `code spans`.
            if c == "`" {
                cursor = endOfInlineCode(input, openStart: cursor)
                continue
            }

            // Backslash: either a LaTeX bracket-delimiter opener
            // (``\(`` / ``\[``) or a CommonMark escape (``\$``, ``\\``,
            // ``\_``, …). Both consume TWO characters, so a doubled
            // backslash can never be re-read as a delimiter opener:
            // ``\\(`` is an escaped backslash followed by a literal
            // paren, not math.
            if c == "\\", input.index(after: cursor) < input.endIndex {
                let markerIndex = input.index(after: cursor)
                let marker = input[markerIndex]
                if marker == "(" || marker == "[" {
                    // ``\[`` is display, ``\(`` is inline — the
                    // bracket forms carry the same meaning as
                    // ``$$``/``$`` and normalise onto the same cases.
                    let isDisplay = (marker == "[")
                    let bodyStart = input.index(after: markerIndex)
                    if let bodyEnd = findBracketClose(input, from: bodyStart, displayMode: isDisplay) {
                        // Emit any leading plain markdown.
                        if plainStart < cursor {
                            segments.append(.markdown(String(input[plainStart..<cursor])))
                        }
                        let latex = String(input[bodyStart..<bodyEnd])
                        segments.append(.math(latex: latex, displayMode: isDisplay))
                        // The closer is two characters: ``\)`` / ``\]``.
                        let closeEnd = input.index(bodyEnd, offsetBy: 2)
                        plainStart = closeEnd
                        cursor = closeEnd
                        continue
                    }
                }
                // ``\$``, an opener with no closer, or any other
                // backslash escape — literal. Skip both characters.
                cursor = input.index(after: markerIndex)
                continue
            }

            // Math open candidate.
            if c == "$" {
                let next = input.index(after: cursor)
                let isDisplay = next < input.endIndex && input[next] == "$"
                let openLen = isDisplay ? 2 : 1
                let bodyStart = input.index(cursor, offsetBy: openLen)
                // Codex r1 P1 (#131): currency-dollar guard. Reject
                // ``$N...`` where N is a digit — that's prose
                // currency ("$20"), not math. Math openers in the
                // wild are alphabetic / backslash / bracket; a
                // bare digit immediately after ``$`` is a strong
                // signal we're inside finance prose. Applies to
                // inline only; display ``$$...`` is unambiguous.
                if !isDisplay,
                   bodyStart < input.endIndex,
                   input[bodyStart].isASCII,
                   input[bodyStart].isNumber {
                    cursor = input.index(after: cursor)
                    continue
                }
                if let bodyEnd = findClose(input, from: bodyStart, displayMode: isDisplay) {
                    // Emit any leading plain markdown.
                    if plainStart < cursor {
                        segments.append(.markdown(String(input[plainStart..<cursor])))
                    }
                    let latex = String(input[bodyStart..<bodyEnd])
                    segments.append(.math(latex: latex, displayMode: isDisplay))
                    let closeEnd = input.index(bodyEnd, offsetBy: openLen)
                    plainStart = closeEnd
                    cursor = closeEnd
                    continue
                }
                // No matching close → treat the ``$`` as literal.
                cursor = input.index(after: cursor)
                continue
            }

            cursor = input.index(after: cursor)
        }

        // Trailing plain run.
        if plainStart < input.endIndex {
            segments.append(.markdown(String(input[plainStart..<input.endIndex])))
        }

        return segments
    }

    // MARK: - Internals

    /// True when ``index`` is at start of a line — start of input
    /// or just after a newline.
    private static func isAtLineStart(_ s: String, at index: String.Index) -> Bool {
        if index == s.startIndex { return true }
        let prev = s.index(before: index)
        return s[prev] == "\n"
    }

    /// Codex r1 P2 (#131): CommonMark indented code block detection.
    /// Returns the index of the FIRST line that is NOT part of the
    /// code block (or ``endIndex`` if the whole rest of input is
    /// indented). Returns ``nil`` when ``lineStart`` is not the
    /// beginning of an indented-code line.
    ///
    /// Rules:
    ///   * Line must start with at least 4 spaces OR a tab.
    ///   * The block continues across consecutive indented lines
    ///     AND across blank lines that are followed by another
    ///     indented line (CommonMark's "blank-line continuation").
    ///   * The block ends on the first non-blank line that is NOT
    ///     indented enough.
    private static func endOfIndentedCodeBlock(_ s: String, lineStart: String.Index) -> String.Index? {
        guard isIndentedCodeLine(s, lineStart: lineStart) else { return nil }
        var i = lineStart
        var lastConsumedBlockEnd = lineStart
        while i < s.endIndex {
            let thisLineStart = i
            // Advance i to the start of the next line (or endIndex).
            while i < s.endIndex && s[i] != "\n" {
                i = s.index(after: i)
            }
            let lineEnd = i  // points at "\n" or endIndex
            if i < s.endIndex { i = s.index(after: i) }
            if isIndentedCodeLine(s, lineStart: thisLineStart) {
                lastConsumedBlockEnd = i
                continue
            }
            if isBlankLine(s, lineStart: thisLineStart, lineEnd: lineEnd) {
                // Blank line continues the block only if the NEXT
                // non-blank line is also indented code; conservatively
                // peek ahead.
                if i < s.endIndex && isIndentedCodeLine(s, lineStart: i) {
                    lastConsumedBlockEnd = i
                    continue
                }
                // Blank line is NOT part of the code block — back
                // off so the segmenter scans it as regular markdown.
                return lastConsumedBlockEnd
            }
            // Non-indented, non-blank line — block ends here.
            return thisLineStart
        }
        return lastConsumedBlockEnd
    }

    /// Strict form of the indented-code test, used by the closer scan.
    /// True only where CommonMark would actually OPEN an indented code
    /// block: at a line start, indented 4+ spaces (or a tab), AND
    /// preceded by a blank line or the start of input.
    ///
    /// ``endOfIndentedCodeBlock`` alone answers "does this line look
    /// indented", which is the right question when deciding whether to
    /// scan a region for math openers and the wrong one when deciding
    /// whether to abandon a formula already in progress — see the note
    /// on ``findBracketClose``.
    private static func startsIndentedCodeBlock(_ s: String, at index: String.Index) -> Bool {
        guard isAtLineStart(s, at: index), isIndentedCodeLine(s, lineStart: index) else {
            return false
        }
        if index == s.startIndex { return true }
        // ``index`` is at a line start, so the character before it is
        // the newline that ended the previous line. Walk back over that
        // line and require it to be blank.
        let newline = s.index(before: index)
        var previousLineStart = newline
        while previousLineStart > s.startIndex {
            let before = s.index(before: previousLineStart)
            if s[before] == "\n" { break }
            previousLineStart = before
        }
        return isBlankLine(s, lineStart: previousLineStart, lineEnd: newline)
    }

    private static func isIndentedCodeLine(_ s: String, lineStart: String.Index) -> Bool {
        guard lineStart < s.endIndex else { return false }
        if s[lineStart] == "\t" { return true }
        // At least 4 leading spaces.
        var spaces = 0
        var i = lineStart
        while i < s.endIndex && s[i] == " " && spaces < 4 {
            spaces += 1
            i = s.index(after: i)
        }
        guard spaces == 4 else { return false }
        // Also reject blank lines (4 spaces then newline / EOF) —
        // a blank line is handled by ``isBlankLine`` instead.
        if i == s.endIndex || s[i] == "\n" { return false }
        return true
    }

    private static func isBlankLine(_ s: String, lineStart: String.Index, lineEnd: String.Index) -> Bool {
        var i = lineStart
        while i < lineEnd {
            let c = s[i]
            if c != " " && c != "\t" { return false }
            i = s.index(after: i)
        }
        return true
    }

    /// True when ``index`` is at column 0 (start of input or after
    /// a newline) and the next three characters are ```` ``` ````.
    private static func isFenceStart(_ s: String, at index: String.Index) -> Bool {
        // Position check: start of input OR previous char is newline.
        if index != s.startIndex {
            let prev = s.index(before: index)
            if s[prev] != "\n" { return false }
        }
        // Three-or-more backticks.
        var count = 0
        var i = index
        while i < s.endIndex && s[i] == "`" {
            count += 1
            i = s.index(after: i)
            if count >= 3 { return true }
        }
        return false
    }

    /// Scan past a fenced block. Returns the index just AFTER the
    /// closing fence (or ``endIndex`` if the block runs to EOF —
    /// the caller treats unclosed fences as eating the rest of
    /// the body, matching CommonMark's tolerant behaviour).
    private static func endOfFencedBlock(_ s: String, fenceStart: String.Index) -> String.Index {
        // Count the fence char-length at fenceStart.
        var fenceLen = 0
        var i = fenceStart
        while i < s.endIndex && s[i] == "`" {
            fenceLen += 1
            i = s.index(after: i)
        }
        // Walk until we find a line whose first non-space chars are
        // ``\`\`\`{n}`` with n >= fenceLen.
        while i < s.endIndex {
            // Advance to the next line's start.
            while i < s.endIndex && s[i] != "\n" { i = s.index(after: i) }
            if i == s.endIndex { return s.endIndex }
            i = s.index(after: i)  // past the newline
            // Skip leading spaces (up to 3 per CommonMark, but we're
            // lenient).
            var lineStart = i
            while lineStart < s.endIndex && s[lineStart] == " " {
                lineStart = s.index(after: lineStart)
            }
            var closeCount = 0
            var j = lineStart
            while j < s.endIndex && s[j] == "`" {
                closeCount += 1
                j = s.index(after: j)
            }
            if closeCount >= fenceLen {
                // Skip the rest of the close line.
                while j < s.endIndex && s[j] != "\n" { j = s.index(after: j) }
                if j < s.endIndex { j = s.index(after: j) }
                return j
            }
            i = j
        }
        return s.endIndex
    }

    /// Match a CommonMark inline-code run. ``openStart`` points at
    /// the first backtick. Returns the index just AFTER the closing
    /// run of equal length; if no close found, returns the index
    /// just after the open (so the segmenter doesn't lose ground).
    private static func endOfInlineCode(_ s: String, openStart: String.Index) -> String.Index {
        var openLen = 0
        var i = openStart
        while i < s.endIndex && s[i] == "`" {
            openLen += 1
            i = s.index(after: i)
        }
        let bodyStart = i
        // Walk looking for a run of exactly ``openLen`` backticks.
        while i < s.endIndex {
            // Don't span across a paragraph break — CommonMark says
            // inline code can't contain a blank line.
            if s[i] == "\n",
               s.index(after: i) < s.endIndex,
               s[s.index(after: i)] == "\n" {
                // Blank line — bail; treat the open as literal.
                return bodyStart
            }
            if s[i] == "`" {
                var run = 0
                var j = i
                while j < s.endIndex && s[j] == "`" {
                    run += 1
                    j = s.index(after: j)
                }
                if run == openLen {
                    return j
                }
                i = j
                continue
            }
            i = s.index(after: i)
        }
        return bodyStart  // unclosed → consume only the open
    }

    /// Find the closing ``\)`` (inline) / ``\]`` (display) for a
    /// bracket-delimited math run whose body starts at ``bodyStart``.
    /// Returns the index of the closing BACKSLASH — the body is
    /// ``bodyStart..<result`` and the closer occupies two characters —
    /// or ``nil`` when no closer is found.
    ///
    /// A backslash inside the body always consumes the character that
    /// follows it. That is what keeps LaTeX's own row break, ``\\``,
    /// from faking a closer: in ``\[ a \\] b \]`` the ``\\`` is a row
    /// break and the ``]`` right after it is a literal bracket, so the
    /// run closes at the final ``\]`` and not at the third character
    /// of ``\\]``.
    ///
    /// ## Code regions are skipped, exactly like the opener scan
    ///
    /// A closer that lives inside a fenced block, a code span or an
    /// indented code block does NOT close the run. Without this, an
    /// unclosed ``\[`` in prose reaches forward and matches the ``\]``
    /// a user wrote *inside* a code sample — swallowing the prose and
    /// the code block in between into one "formula". The opener scan
    /// is careful to skip those three regions; the closer scan has to
    /// be equally careful or the care is one-sided.
    ///
    /// The indented-code test is deliberately STRICTER here than in
    /// the opener scan, which treats any 4-space line as code. The two
    /// mistakes are not symmetric: a false skip in the opener scan
    /// only means "do not look for math here", which loses nothing,
    /// while a false skip here abandons a real formula. Math bodies
    /// are very commonly indented —
    ///
    /// ```
    /// \[
    ///     P = \frac{47}{0.85} \]
    /// ```
    ///
    /// — so this requires what CommonMark actually requires to OPEN an
    /// indented code block: a preceding blank line (or the start of
    /// input). The indented continuation of a ``\[`` line is not a
    /// code block under that rule, and its closer is still honoured.
    ///
    /// ## Nested openers: first closer wins, same-kind opener bails
    ///
    /// LaTeX cannot nest ``\( … \( … \) … \)`` — a second opener of
    /// the same kind before any closer is therefore strong evidence
    /// the FIRST opener was prose, not math. The scan gives up, the
    /// caller emits that opener as literal text and rescans from just
    /// past it, so ``Use \( to group, then \(x\).`` keeps the prose
    /// and still renders ``x``.
    ///
    /// A nested opener of the OTHER kind does not bail. Rejecting the
    /// outer run there would drop the whole formula back to CommonMark,
    /// which strips the delimiters to bare brackets — the original bug.
    /// Keeping it degrades instead to ``MathView``'s literal-source
    /// fallback, which is strictly more informative.
    ///
    /// Beyond those two rules the first closer wins. A stray ``\[`` in
    /// prose followed much later by a stray ``\]`` does become one math
    /// run; that is accepted, because CommonMark would have rendered
    /// both as bare brackets anyway, and because models do not emit
    /// lone bracket escapes in prose.
    private static func findBracketClose(_ s: String, from bodyStart: String.Index, displayMode: Bool) -> String.Index? {
        let closer: Character = displayMode ? "]" : ")"
        let opener: Character = displayMode ? "[" : "("
        var i = bodyStart
        while i < s.endIndex {
            let c = s[i]

            // Code regions — a closer inside one does not close the run.
            if startsIndentedCodeBlock(s, at: i),
               let codeBlockEnd = endOfIndentedCodeBlock(s, lineStart: i) {
                i = codeBlockEnd
                continue
            }
            if c == "`", isFenceStart(s, at: i) {
                i = endOfFencedBlock(s, fenceStart: i)
                continue
            }
            if c == "`" {
                i = endOfInlineCode(s, openStart: i)
                continue
            }

            if c == "\\" {
                let next = s.index(after: i)
                if next == s.endIndex { return nil }
                if s[next] == closer { return i }
                // Nested opener of the same kind — the first opener was
                // prose. Give up so the caller can rescan past it.
                if s[next] == opener { return nil }
                i = s.index(after: next)
                continue
            }
            // Inline-math paragraph guard: an unclosed ``\(`` must not
            // swallow the rest of the reply. Unlike ``$``, a bare
            // ``\(`` in prose is vanishingly rare (it is CommonMark's
            // escape for a literal paren, which models never emit), so
            // a SINGLE newline is tolerated — models do wrap long
            // inline runs — and only a blank line ends the search.
            // Display math keeps the lenient ``$$`` behaviour because
            // multi-line ``\[ … \]`` blocks are the norm.
            if !displayMode, c == "\n" {
                let next = s.index(after: i)
                if next == s.endIndex { return nil }
                if s[next] == "\n" { return nil }
            }
            i = s.index(after: i)
        }
        return nil
    }

    /// Find the closing ``$`` (or ``$$``) for a math run starting
    /// at ``bodyStart``. Returns ``nil`` if no close found before
    /// EOF or, for inline math, before the next blank line. Math
    /// closers preceded by ``\`` are escaped and don't count.
    private static func findClose(_ s: String, from bodyStart: String.Index, displayMode: Bool) -> String.Index? {
        var i = bodyStart
        // Inline math has a paragraph-break terminator — once we
        // see a blank line (two consecutive newlines) without
        // finding a close, the run was just a stray ``$`` in
        // prose and we give up. Display math is more lenient
        // because models often emit multi-line ``$$ ... $$`` blocks.
        while i < s.endIndex {
            let c = s[i]
            // Escaped dollar — skip.
            if c == "\\",
               s.index(after: i) < s.endIndex,
               s[s.index(after: i)] == "$" {
                i = s.index(i, offsetBy: 2)
                continue
            }
            // Inline-math line-break guard.
            if !displayMode && c == "\n" {
                // Look at the next char for the blank-line break.
                let next = s.index(after: i)
                if next == s.endIndex { return nil }
                if s[next] == "\n" { return nil }
                // Single newline inside inline math: bail. Inline
                // math is expected on one line; multi-line should
                // use ``$$ ... $$``.
                return nil
            }
            if c == "$" {
                if displayMode {
                    // Need ``$$`` to close.
                    let next = s.index(after: i)
                    if next < s.endIndex && s[next] == "$" {
                        return i  // body ends here
                    }
                    i = s.index(after: i)
                    continue
                } else {
                    return i  // single $ closes
                }
            }
            i = s.index(after: i)
        }
        return nil
    }
}
