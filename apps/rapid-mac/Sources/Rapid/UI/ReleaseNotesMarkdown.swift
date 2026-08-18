import SwiftUI

/// Minimal block-level Markdown renderer for the in-app update dialog's
/// release notes.
///
/// ## Why this exists
///
/// The notes rendered in Settings → App come straight from the
/// GitHub Release body / ``latest.json`` ``notes`` field, which is the
/// raw CHANGELOG section — full Markdown (``##`` / ``###`` headings,
/// ``-`` bullets, ``**bold**``, inline `` `code` ``). SwiftUI's
/// ``Text`` only interprets *inline* Markdown via ``AttributedString``;
/// it renders block syntax (headings, bullets, fences) as literal
/// ``##`` / ``-`` characters. The pre-v0.8.18 dialog used a bare
/// ``Text(release.notes)`` and so dumped the raw Markdown — the
/// "升级页面太丑" report.
///
/// This renderer does a line-based block parse (the only structure
/// CHANGELOG notes actually use) and renders each block with the right
/// SwiftUI styling, delegating *inline* spans (`**bold**`, `` `code` ``,
/// `[text](url)`) to ``AttributedString``'s inline-only parser so we
/// don't reimplement inline Markdown.
///
/// It is deliberately NOT a full CommonMark implementation — no nested
/// lists, tables, blockquotes, or setext headings. Those don't appear
/// in our CHANGELOG and would bloat a dialog-only helper. Anything it
/// doesn't recognise degrades to a plain paragraph, so unknown syntax
/// is shown verbatim rather than dropped.
struct ReleaseNotesMarkdown: View {
    let raw: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Self.parse(raw)) { block in
                blockView(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func blockView(_ block: Block) -> some View {
        switch block.kind {
        case .heading(let level, let text):
            Text(Self.inline(text))
                .font(Self.headingFont(level: level))
                .foregroundStyle(level >= 3 ? .secondary : .primary)
                .padding(.top, level <= 2 ? 4 : 2)
        case .bullet(let text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("•").foregroundStyle(.secondary)
                Text(Self.inline(text))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .font(.callout)
        case .ordered(let marker, let text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker).foregroundStyle(.secondary).monospacedDigit()
                Text(Self.inline(text))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .font(.callout)
        case .code(let text):
            Text(text)
                .font(.system(.callout, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
        case .paragraph(let text):
            Text(Self.inline(text))
                .font(.callout)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private static func headingFont(level: Int) -> Font {
        switch level {
        case 1:  return .title3.bold()
        case 2:  return .headline
        default: return .subheadline.bold()
        }
    }

    // MARK: - Inline

    /// Render inline Markdown (`**bold**`, `*italic*`, `` `code` ``,
    /// `[text](url)`) to an ``AttributedString``. Block syntax is left
    /// to the line parser, so we use the inline-only interpreter and
    /// preserve whitespace. Falls back to the literal string if the
    /// parser throws (it shouldn't for our inputs, but a malformed link
    /// must never blank out a line).
    static func inline(_ s: String) -> AttributedString {
        if let attr = try? AttributedString(
            markdown: s,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            return attr
        }
        return AttributedString(s)
    }

    // MARK: - Block model

    struct Block: Identifiable {
        let id: Int
        let kind: Kind
        enum Kind {
            case heading(level: Int, text: String)
            case bullet(text: String)
            case ordered(marker: String, text: String)
            case code(text: String)
            case paragraph(text: String)
        }
    }

    /// Line-based block parse. Drops a redundant leading version
    /// heading (``## [0.8.18] — date`` / ``# v0.8.18``) because the
    /// dialog header already shows "vX.Y.Z — you have vA.B.C".
    static func parse(_ raw: String) -> [Block] {
        var blocks: [Block] = []
        var nextID = 0
        func push(_ kind: Block.Kind) {
            blocks.append(Block(id: nextID, kind: kind))
            nextID += 1
        }

        let lines = raw.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var inFence = false
        var fenceBuffer: [String] = []
        var droppedLeadingVersionHeading = false
        var seenContent = false

        // The currently-open paragraph or list item. CHANGELOG notes are
        // hard-wrapped (a single bullet spans several indented lines), so
        // a non-marker line that follows a paragraph/bullet/ordered block
        // is a *soft-wrapped continuation* and must fold into it with a
        // space — otherwise wrapped bullets split into stray paragraphs
        // and inline spans like `**bold**` that straddle the wrap break.
        // Only paragraph/bullet/ordered can stay open; headings and code
        // fences are always flushed immediately. (codex MAJOR, PR review.)
        var open: Block.Kind?
        func flushOpen() {
            if let o = open { push(o) }
            open = nil
        }

        func flushFence() {
            // Drop a trailing blank line some editors leave before ```.
            let text = fenceBuffer.joined(separator: "\n")
                .trimmingCharacters(in: .newlines)
            if !text.isEmpty { push(.code(text: text)) }
            fenceBuffer = []
        }

        for rawLine in lines {
            let line = rawLine
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code block toggle (``` or ~~~).
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                if inFence { flushFence(); inFence = false }
                else { flushOpen(); inFence = true }
                continue
            }
            if inFence { fenceBuffer.append(line); continue }

            // Blank line — block separator. Closes any open block.
            if trimmed.isEmpty { flushOpen(); continue }

            // ATX heading: 1–6 leading '#', then a space.
            if let h = headingMatch(trimmed) {
                flushOpen()
                // Skip the first heading if it's just the version
                // (changelog "## [X.Y.Z] — date") — redundant with the
                // dialog header.
                if !seenContent, !droppedLeadingVersionHeading,
                   looksLikeVersionHeading(h.text) {
                    droppedLeadingVersionHeading = true
                    continue
                }
                seenContent = true
                push(.heading(level: h.level, text: h.text))
                continue
            }

            seenContent = true

            // Bullet: -, *, or + followed by a space. Starts a new open
            // list item.
            if let b = bulletMatch(trimmed) {
                flushOpen()
                open = .bullet(text: b)
                continue
            }
            // Ordered: digits, then '.' or ')', then a space.
            if let o = orderedMatch(trimmed) {
                flushOpen()
                open = .ordered(marker: o.marker, text: o.text)
                continue
            }

            // Non-marker line: a soft-wrapped continuation of the open
            // block, or the start of a new paragraph.
            switch open {
            case .paragraph(let t):
                open = .paragraph(text: t + " " + trimmed)
            case .bullet(let t):
                open = .bullet(text: t + " " + trimmed)
            case .ordered(let m, let t):
                open = .ordered(marker: m, text: t + " " + trimmed)
            case .heading, .code, .none:
                // headings/code are never left open; .none → new paragraph.
                open = .paragraph(text: trimmed)
            }
        }
        if inFence { flushFence() }   // unterminated fence — render what we have
        flushOpen()                   // flush a trailing paragraph/list item
        return blocks
    }

    // MARK: - Line matchers (no NSRegularExpression — cheap char scans)

    private static func headingMatch(_ s: String) -> (level: Int, text: String)? {
        var level = 0
        var idx = s.startIndex
        while idx < s.endIndex, s[idx] == "#", level < 6 {
            level += 1
            idx = s.index(after: idx)
        }
        guard level > 0, idx < s.endIndex, s[idx] == " " else { return nil }
        let text = String(s[idx...]).trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        return (level, text)
    }

    private static func bulletMatch(_ s: String) -> String? {
        guard let first = s.first, first == "-" || first == "*" || first == "+" else { return nil }
        let after = s.index(after: s.startIndex)
        guard after < s.endIndex, s[after] == " " else { return nil }
        return String(s[after...]).trimmingCharacters(in: .whitespaces)
    }

    private static func orderedMatch(_ s: String) -> (marker: String, text: String)? {
        var idx = s.startIndex
        var digits = ""
        while idx < s.endIndex, s[idx].isNumber {
            digits.append(s[idx])
            idx = s.index(after: idx)
        }
        guard !digits.isEmpty, idx < s.endIndex, s[idx] == "." || s[idx] == ")" else { return nil }
        let sep = s[idx]
        let afterSep = s.index(after: idx)
        guard afterSep < s.endIndex, s[afterSep] == " " else { return nil }
        let text = String(s[afterSep...]).trimmingCharacters(in: .whitespaces)
        return ("\(digits)\(sep)", text)
    }

    /// "[0.8.18] — 2026-06-29", "v0.8.18", "0.8.18 (2026-06-29)" — the
    /// shapes a CHANGELOG version header takes. We treat a heading as a
    /// version header when, after stripping a leading 'v'/'[' the first
    /// token is a dotted numeric version.
    private static func looksLikeVersionHeading(_ text: String) -> Bool {
        var t = text.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("[") { t.removeFirst() }
        if t.hasPrefix("v") || t.hasPrefix("V") { t.removeFirst() }
        // Leading run of digits/'.', collected with an explicit loop.
        // (An earlier `t.prefix { ... }` + `split(...).allSatisfy(\.isNumber)`
        // formulation over Substrings tripped a runtime trap in the
        // optimized test build, so we keep this deliberately plain.)
        var dotCount = 0
        var sawDigit = false
        var lastWasDot = true   // a leading '.' is malformed
        for ch in t {
            if ch.isNumber {
                sawDigit = true
                lastWasDot = false
            } else if ch == "." {
                if lastWasDot { return false }   // "1..2" / leading dot
                dotCount += 1
                lastWasDot = true
            } else {
                break   // end of the version token
            }
        }
        // Require at least "N.N" (one dot, digits present, no trailing dot)
        // so a heading like "5 things" doesn't match.
        return sawDigit && dotCount >= 1 && !lastWasDot
    }
}
