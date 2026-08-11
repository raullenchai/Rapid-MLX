import SwiftUI
import AppKit

/// Token colouring for fenced code blocks in assistant replies.
///
/// ## Why this exists rather than a dependency
///
/// ``MarkdownUI`` renders a fenced block as one undifferentiated
/// ``Text`` — it ships no highlighter — so before this type every code
/// block in the transcript was flat monospaced grey.
///
/// The obvious dependency, ``Highlightr`` (a Swift wrapper around
/// highlight.js), was measured and rejected on three counts. It resolves
/// `highlight.min.js` through ``Bundle.module``, which is exactly the
/// lookup that does not survive assembly into a shipped `.app` — the
/// same wall documented at length in ``MathView`` — so it would have
/// returned `nil` in production and highlighted nothing, while working
/// fine in `swift run`. It costs ~2.1 MB of resources (1.0 MB of
/// JavaScript, 1.1 MB across 271 stylesheets of which we would use one).
/// And it needs ``JavaScriptCore``: this app links no JS runtime today,
/// and adding one to colour code — evaluated per block, on the same
/// streaming path we are trying to make smoother — is not a trade worth
/// making for syntax colour.
///
/// A hand-written scanner has none of those properties. It compiles into
/// the binary (no bundle lookup can fail), adds no measurable size, and
/// runs as a single linear pass.
///
/// ## Scope
///
/// A focused set of common languages found in chat replies, ranging from
/// Swift and Python to web, systems, data, and shell formats. Five token
/// classes are emitted: comment, string, number, keyword, and type.
///
/// This is deliberately NOT a parser. It is a lexical scanner with no
/// grammar and no notion of scope, so it will colour a keyword used as
/// an identifier (`class` as a dict key in Python) as a keyword. That
/// mis-colouring is invisible in practice and the cost of avoiding it —
/// a real parser per language — is not repayable.
///
/// ## Contract
///
/// * An unrecognised or absent language returns plain unstyled text, so
///   the fallback is exactly today's rendering and a language we have
///   never heard of can never render worse than before.
/// * Only ``foregroundColor`` is set. Font, size and weight stay with
///   the caller's ``.markdownTextStyle``, so the block keeps its
///   monospaced face and Dynamic-Type scaling untouched.
/// * Colours are ``NSColor(name:dynamicProvider:)``, matching
///   ``RapidTheme``, so light/dark follows the system appearance without
///   this type observing `colorScheme` at all. This matters: an
///   `AttributedString` is a value, so a colour baked from a
///   `@Environment(\.colorScheme)` read at build time would freeze at
///   whatever the appearance was on that pass and go black-on-black
///   after a theme switch — the failure mode already recorded against
///   the inline-math image cache.
enum SyntaxHighlighter {

    /// Highlight `code` for `language`. Returns unstyled text when the
    /// language is unknown, absent, or has no keyword set.
    static func highlight(_ code: String, language: String?) -> AttributedString {
        guard let grammar = Grammar.forLanguage(language) else {
            return AttributedString(code)
        }
        return scan(code, grammar: grammar)
    }

    /// True when ``highlight`` would colour this language. Lets the
    /// caller skip the work entirely for unknown fences.
    static func supports(language: String?) -> Bool {
        Grammar.forLanguage(language) != nil
    }

    /// Reuses the attributed prefix of a code block while streaming appends
    /// new lines. The final, unterminated line is deliberately rescanned: an
    /// identifier, string, or comment at EOF can change classification when
    /// the next chunk arrives. Everything through the last newline outside a
    /// multiline construct is stable and can be retained safely.
    ///
    /// The memo also keeps the last complete result so unrelated SwiftUI
    /// updates (hover, copy feedback, geometry preferences) do no scanner work.
    /// Replacements, shrinks, and language changes fall back to a cold scan.
    /// Instances are intentionally per code-block view and are not thread-safe.
    final class Memo {
        private var languageKey: String?
        private var lastCode = ""
        private var lastResult = AttributedString()
        private var stableSourceByteCount = 0
        private var stableResult = AttributedString()

        /// Deterministic work counter used by the streaming regression test.
        /// Counts characters tokenised, excluding prefix validation.
        private(set) var scannedCharacterCount = 0

        func highlight(_ code: String, language: String?) -> AttributedString {
            guard let grammar = Grammar.forLanguage(language) else {
                resetCachedState()
                return AttributedString(code)
            }

            let nextLanguageKey = Self.normalizedLanguage(language)
            if nextLanguageKey == languageKey, code == lastCode {
                return lastResult
            }

            let extendsLastInput = nextLanguageKey == languageKey
                && !lastCode.isEmpty
                && code.hasPrefix(lastCode)
            if !extendsLastInput {
                stableSourceByteCount = 0
                stableResult = AttributedString()
            }

            let tail = String(decoding: code.utf8.dropFirst(stableSourceByteCount), as: UTF8.self)
            scannedCharacterCount += tail.count
            let scanned = scanRecordingStablePrefix(tail, grammar: grammar)

            var result = stableResult
            result += scanned.result

            if scanned.stableSourceByteCount > 0 {
                stableSourceByteCount += scanned.stableSourceByteCount
                stableResult += scanned.stableResult
            }

            languageKey = nextLanguageKey
            lastCode = code
            lastResult = result
            return result
        }

        private func resetCachedState() {
            languageKey = nil
            lastCode = ""
            lastResult = AttributedString()
            stableSourceByteCount = 0
            stableResult = AttributedString()
        }

        private static func normalizedLanguage(_ language: String?) -> String? {
            language?.lowercased().trimmingCharacters(in: .whitespaces)
        }
    }

    // MARK: - Palette

    /// Token classes the scanner can emit.
    enum TokenKind {
        case comment
        case string
        case number
        case keyword
        case type
        case plain
    }

    /// Muted, low-saturation palette. A transcript is prose first: a
    /// full-saturation editor theme inside a reply pulls the eye to the
    /// code block over the sentence explaining it. Hues are the
    /// conventional ones (green comments, red strings, purple keywords)
    /// so the colouring reads as "code" at a glance, but each is pulled
    /// toward the background.
    ///
    /// Each colour is built ONCE and reused. Two ``Color`` values wrapping
    /// separately-constructed ``NSColor``s do not compare equal, so
    /// rebuilding per call would both allocate per token and make the
    /// resulting ``AttributedString`` runs impossible to compare — which
    /// is exactly how the test suite inspects them.
    static func color(for kind: TokenKind) -> Color {
        switch kind {
        case .comment: return commentColor
        case .string: return stringColor
        case .number: return numberColor
        case .keyword: return keywordColor
        case .type: return typeColor
        case .plain: return .primary
        }
    }

    private static let commentColor = dynamic(dark: (0x7A, 0x8A, 0x78), light: (0x6A, 0x79, 0x68))
    private static let stringColor = dynamic(dark: (0xD1, 0x8A, 0x7E), light: (0xA8, 0x4B, 0x3C))
    private static let numberColor = dynamic(dark: (0xC5, 0xA3, 0x72), light: (0x99, 0x6E, 0x2E))
    private static let keywordColor = dynamic(dark: (0xB0, 0x92, 0xD0), light: (0x7A, 0x4E, 0xA8))
    private static let typeColor = dynamic(dark: (0x7B, 0xA7, 0xC9), light: (0x2E, 0x6A, 0x93))

    private static func dynamic(
        dark: (Int, Int, Int),
        light: (Int, Int, Int)
    ) -> Color {
        Color(nsColor: NSColor(name: nil, dynamicProvider: { appearance in
            // Same ``bestMatch`` test ``RapidTheme`` uses. Spelled out
            // again rather than shared because that file's `isDark`
            // helper is `private` to it, and widening a theme internal
            // for one call site is the worse trade.
            let match = appearance.bestMatch(from: [
                .aqua, .darkAqua, .accessibilityHighContrastDarkAqua
            ])
            let isDark = (match == .darkAqua || match == .accessibilityHighContrastDarkAqua)
            let c = isDark ? dark : light
            return NSColor(
                deviceRed: CGFloat(c.0) / 255.0,
                green: CGFloat(c.1) / 255.0,
                blue: CGFloat(c.2) / 255.0,
                alpha: 1.0
            )
        }))
    }

    // MARK: - Grammar

    /// The per-language facts the scanner needs. Everything else about
    /// tokenising is shared.
    struct Grammar {
        /// Line-comment openers, longest-first so `///` wins over `//`.
        let lineComment: [String]
        /// Block-comment delimiter pairs.
        let blockComment: [(open: String, close: String)]
        /// Quote characters that open a string.
        let quotes: [Character]
        /// Triple-quote openers (Python docstrings) — matched before
        /// single quotes so `"""` doesn't scan as an empty string.
        let tripleQuotes: [String]
        /// Whether a backslash escapes the next character in a string.
        let backslashEscapes: Bool
        let keywords: Set<String>
        /// Known type names. Also matched heuristically: an identifier
        /// starting uppercase is treated as a type in languages that
        /// follow that convention.
        let types: Set<String>
        let capitalisedIdentifiersAreTypes: Bool
        /// Rust-style apostrophe-prefixed identifiers (`'a`, `'static`).
        /// These must be distinguished from single-quoted character literals.
        let apostrophePrefixedIdentifiers: Bool
        /// Line comments open only at the START of a line (`i == 0` or right
        /// after a newline). This is what a unified diff needs: its `+`/`-`
        /// markers are line-prefixes, not mid-line comment openers, so an
        /// unchanged context line like `value = a - b` must not colour from
        /// the minus onward. Off everywhere else, where `//`, `#`, `--` are
        /// genuine mid-line openers.
        let lineCommentAtLineStart: Bool
        /// A line comment opens only at a WORD boundary — start of line, or
        /// right after whitespace or a command separator. Shell's `#` is a
        /// comment only at the start of a word, so `$#`, `${#arr}` and
        /// `echo a#b` are NOT comments. Off elsewhere, where a `#`/`//`
        /// mid-token still opens a comment (Python's `a#b` does).
        let lineCommentNeedsWordBoundary: Bool
        /// Block comments nest — an inner `/*` must be matched by its own
        /// `*/` before the outer one closes. Swift and Rust both allow this;
        /// most C-family languages do not. When off, the first close wins.
        let nestableBlockComments: Bool
        /// Quote characters that open a RAW, newline-spanning string with no
        /// backslash escaping — Go's backtick string. The generic string
        /// scanner stops at a newline (right for a half-typed line during
        /// streaming); a raw multiline quote overrides both that stop and
        /// escape handling so the whole `` `...` `` literal, including any
        /// `//` inside it, scans as one string across lines.
        let rawMultilineQuotes: Set<Character>
        /// Quote characters that open a newline-spanning string that STILL
        /// honours backslash escapes — a JavaScript/TypeScript backtick
        /// template. Same "don't stop at the newline" as ``rawMultilineQuotes``
        /// but `\`` still escapes, so it is a distinct set rather than a flag.
        let escapedMultilineQuotes: Set<Character>
        /// Configured keywords/types containing punctuation that the generic
        /// identifier scanner deliberately excludes. Longest-first prevents
        /// a shorter configured token from stealing a longer one.
        let punctuationTokens: [(text: String, kind: TokenKind)]

        init(
            lineComment: [String],
            blockComment: [(open: String, close: String)],
            quotes: [Character],
            tripleQuotes: [String],
            backslashEscapes: Bool,
            keywords: Set<String>,
            types: Set<String>,
            capitalisedIdentifiersAreTypes: Bool,
            apostrophePrefixedIdentifiers: Bool = false,
            lineCommentAtLineStart: Bool = false,
            lineCommentNeedsWordBoundary: Bool = false,
            nestableBlockComments: Bool = false,
            rawMultilineQuotes: Set<Character> = [],
            escapedMultilineQuotes: Set<Character> = []
        ) {
            self.lineComment = lineComment
            self.blockComment = blockComment
            self.quotes = quotes
            self.tripleQuotes = tripleQuotes
            self.backslashEscapes = backslashEscapes
            self.keywords = keywords
            self.types = types
            self.capitalisedIdentifiersAreTypes = capitalisedIdentifiersAreTypes
            self.apostrophePrefixedIdentifiers = apostrophePrefixedIdentifiers
            self.lineCommentAtLineStart = lineCommentAtLineStart
            self.lineCommentNeedsWordBoundary = lineCommentNeedsWordBoundary
            self.nestableBlockComments = nestableBlockComments
            self.rawMultilineQuotes = rawMultilineQuotes
            self.escapedMultilineQuotes = escapedMultilineQuotes
            self.punctuationTokens = (
                keywords.compactMap { token in
                    token.contains(where: { !isIdentifierChar($0) && $0 != "@" && $0 != "#" })
                        ? (token, TokenKind.keyword) : nil
                }
                + types.compactMap { token in
                    token.contains(where: { !isIdentifierChar($0) && $0 != "@" && $0 != "#" })
                        ? (token, TokenKind.type) : nil
                }
            ).sorted { lhs, rhs in lhs.0.count > rhs.0.count }
        }

        static func forLanguage(_ raw: String?) -> Grammar? {
            guard let raw else { return nil }
            let key = raw.lowercased().trimmingCharacters(in: .whitespaces)
            switch key {
            case "swift": return .swift
            case "python", "py", "python3": return .python
            case "javascript", "js", "jsx", "mjs", "cjs", "node": return .javascript
            case "typescript", "ts", "tsx": return .typescript
            case "json", "jsonc", "json5": return .json
            case "bash", "sh", "shell", "zsh", "console", "fish": return .shell
            case "go", "golang": return .go
            case "rust", "rs": return .rust
            case "c", "h": return .c
            case "cpp", "c++", "cc", "cxx", "hpp", "objc", "objective-c", "m", "mm":
                return .cpp
            case "csharp", "cs", "c#": return .csharp
            case "java": return .java
            case "kotlin", "kt", "kts": return .kotlin
            case "ruby", "rb": return .ruby
            case "php": return .php
            case "sql", "postgres", "postgresql", "mysql", "sqlite": return .sql
            case "html", "xml", "svg", "vue", "svelte": return .markup
            case "css": return .css
            case "scss", "sass", "less": return .scss
            case "yaml", "yml": return .yaml
            case "toml", "ini", "cfg", "conf": return .toml
            case "dockerfile", "docker": return .dockerfile
            case "makefile", "make", "cmake": return .makefile
            case "lua": return .lua
            case "r": return .r
            case "scala": return .scala
            case "dart": return .dart
            case "perl", "pl": return .perl
            case "haskell", "hs": return .haskell
            case "elixir", "ex", "exs": return .elixir
            case "protobuf", "proto": return .protobuf
            case "graphql", "gql": return .graphql
            case "diff", "patch": return .diff
            default: return nil
            }
        }
    }

    // MARK: - Scanner

    /// Single linear pass. At each position we try, in order: comment,
    /// string, number, identifier. Anything else is consumed as one
    /// plain character.
    ///
    /// Ordering is the whole correctness story — a `#` inside a string
    /// must not open a comment, and a quote inside a comment must not
    /// open a string. Because each branch consumes its entire construct
    /// before returning to the top, neither can happen.
    private struct ScanResult {
        let result: AttributedString
        let stableSourceByteCount: Int
        let stableResult: AttributedString
    }

    private static func scan(_ code: String, grammar: Grammar) -> AttributedString {
        scanRecordingStablePrefix(code, grammar: grammar).result
    }

    private static func scanRecordingStablePrefix(
        _ code: String,
        grammar: Grammar
    ) -> ScanResult {
        var out = AttributedString()
        let chars = Array(code)
        var i = 0
        var stableCharacterCount = 0
        // Accumulate consecutive plain characters and emit them as one
        // run rather than one attributed run per character.
        var plainBuffer = ""

        func flushPlain() {
            guard !plainBuffer.isEmpty else { return }
            out += AttributedString(plainBuffer)
            plainBuffer = ""
        }

        func emit(_ text: String, _ kind: TokenKind) {
            flushPlain()
            var piece = AttributedString(text)
            piece.foregroundColor = color(for: kind)
            out += piece
        }

        while i < chars.count {
            // --- block comment ---
            // Match this before line comments because Lua's `--[[` block
            // opener is prefixed by its `--` line-comment opener.
            if let pair = grammar.blockComment.first(where: { matches($0.open, chars, i) }) {
                let start = i
                i += pair.open.count
                // Depth tracking: for a nesting grammar (Swift, Rust) an inner
                // `/*` must be closed by its own `*/` before the outer one
                // closes. With nesting off, depth never rises above 1 and the
                // first close wins — the original single-level behaviour.
                var depth = 1
                while i < chars.count && depth > 0 {
                    if grammar.nestableBlockComments && matches(pair.open, chars, i) {
                        depth += 1
                        i += pair.open.count
                    } else if matches(pair.close, chars, i) {
                        depth -= 1
                        i += pair.close.count
                    } else {
                        i += 1
                    }
                }
                emit(String(chars[start..<i]), .comment)
                continue
            }

            // --- line comment ---
            // When the grammar anchors line comments to the line start (diff),
            // a marker only opens one at `i == 0` or immediately after a
            // newline. `chars[start..<i]` reaching a stable prefix always
            // begins a line, so this holds identically on the streamed tail.
            let atLineStart = i == 0 || chars[i - 1] == "\n"
            // Shell's `#` opens a comment only at a word boundary, so `$#`
            // and `${#x}` are parameter expansions, not comments.
            let atWordBoundary = i == 0 || isCommentWordBoundary(chars[i - 1])
            if (!grammar.lineCommentAtLineStart || atLineStart),
               (!grammar.lineCommentNeedsWordBoundary || atWordBoundary),
               let opener = grammar.lineComment.first(where: { matches($0, chars, i) }) {
                let start = i
                i += opener.count
                while i < chars.count && chars[i] != "\n" { i += 1 }
                emit(String(chars[start..<i]), .comment)
                continue
            }

            // --- triple-quoted string (before single quotes) ---
            if let opener = grammar.tripleQuotes.first(where: { matches($0, chars, i) }) {
                let start = i
                i += opener.count
                while i < chars.count && !matches(opener, chars, i) { i += 1 }
                if i < chars.count { i += opener.count }
                emit(String(chars[start..<i]), .string)
                continue
            }

            // --- apostrophe-prefixed identifier ---
            // Rust lifetimes and loop labels use the same leading apostrophe
            // as character literals. Consume the identifier as plain text
            // unless it is immediately closed (`'a'`), which remains a string.
            if grammar.apostrophePrefixedIdentifiers,
               let end = apostropheIdentifierEnd(in: chars, at: i) {
                plainBuffer += String(chars[i..<end])
                i = end
                continue
            }

            // --- string ---
            if grammar.quotes.contains(chars[i]) {
                let quote = chars[i]
                // A raw multiline quote (Go's backtick) neither escapes nor
                // stops at a newline: it runs to its closing delimiter across
                // however many lines.
                let isRawMultiline = grammar.rawMultilineQuotes.contains(quote)
                // Raw (Go backtick) and escaped-multiline (JS/TS backtick
                // template) quotes both span newlines; only the raw one skips
                // escape handling.
                let spansNewlines = isRawMultiline
                    || grammar.escapedMultilineQuotes.contains(quote)
                let start = i
                i += 1
                while i < chars.count {
                    if !isRawMultiline && grammar.backslashEscapes && chars[i] == "\\" {
                        // Escape consumes the next character, so an
                        // escaped quote cannot terminate the string.
                        i += min(2, chars.count - i)
                        continue
                    }
                    if chars[i] == quote { i += 1; break }
                    // An unterminated string stops at end of line rather
                    // than swallowing the rest of the block — during
                    // streaming, a half-arrived line is the normal case. A
                    // multiline literal (Go raw / JS template) is the
                    // exception: its newlines are part of the string.
                    if !spansNewlines && chars[i] == "\n" { break }
                    i += 1
                }
                emit(String(chars[start..<i]), .string)
                continue
            }

            // --- number ---
            // Only at a token boundary, so the `1` in `utf8_1` stays part
            // of the identifier.
            if chars[i].isNumber && (i == 0 || !isIdentifierChar(chars[i - 1])) {
                let start = i
                while i < chars.count && isNumberChar(chars[i]) {
                    let c = chars[i]
                    i += 1
                    // A scientific exponent's sign is part of the literal:
                    // `1.5e-3` is one number, not `1.5e`, `-`, `3`. Only a
                    // sign IMMEDIATELY after e/E is absorbed, so `2 - 3`
                    // outside an exponent is untouched.
                    if (c == "e" || c == "E"),
                       i < chars.count,
                       chars[i] == "+" || chars[i] == "-" {
                        i += 1
                    }
                }
                emit(String(chars[start..<i]), .number)
                continue
            }

            // --- configured punctuation-bearing keyword / type ---
            // Match before generic identifiers so `.PHONY`, `filter-out`,
            // `background-color`, and `@font-face` reach the exact token sets
            // that already declare them instead of being split at `.` / `-`.
            if let token = grammar.punctuationTokens.first(where: {
                matches($0.text, chars, i)
                    && punctuationTokenIsDelimited($0.text, chars: chars, at: i)
            }) {
                emit(token.text, token.kind)
                i += token.text.count
                continue
            }

            // --- identifier / keyword / type ---
            if isIdentifierStart(chars[i]) {
                let start = i
                // `#` and `@` are valid sigils only at the start of a
                // directive/attribute. Consume that known-valid first
                // character before applying the narrower continuation rule.
                i += 1
                while i < chars.count && isIdentifierChar(chars[i]) { i += 1 }
                let word = String(chars[start..<i])
                if grammar.keywords.contains(word) {
                    emit(word, .keyword)
                } else if grammar.types.contains(word) {
                    emit(word, .type)
                } else if grammar.capitalisedIdentifiersAreTypes,
                          let first = word.first,
                          first.isUppercase {
                    emit(word, .type)
                } else {
                    plainBuffer += word
                }
                continue
            }

            let plainCharacter = chars[i]
            plainBuffer.append(plainCharacter)
            i += 1
            if plainCharacter == "\n" {
                // A newline reached by the top-level scanner is outside a
                // multiline comment/string. All tokens through it are final;
                // only the following line can change on the next append.
                // Record just the boundary offset — snapshotting `out` here
                // instead shared its storage and forced a full copy-on-write
                // clone on the very next append, making a cold scan quadratic
                // in line count. The stable slice is taken once, below.
                flushPlain()
                stableCharacterCount = i
            }
        }

        flushPlain()
        // Slice the finished output to the stable boundary a single time.
        // `out`'s characters are exactly the source characters (highlighting
        // preserves text), so a character offset indexes it directly.
        let stableResult: AttributedString
        if stableCharacterCount > 0 {
            let end = out.index(out.startIndex, offsetByCharacters: stableCharacterCount)
            stableResult = AttributedString(out[out.startIndex..<end])
        } else {
            stableResult = AttributedString()
        }
        let stableSource = String(chars.prefix(stableCharacterCount))
        return ScanResult(
            result: out,
            stableSourceByteCount: stableSource.utf8.count,
            stableResult: stableResult
        )
    }

    /// Does `needle` occur at `index` in `chars`?
    private static func matches(_ needle: String, _ chars: [Character], _ index: Int) -> Bool {
        let n = Array(needle)
        guard index + n.count <= chars.count else { return false }
        for k in 0..<n.count where chars[index + k] != n[k] { return false }
        return true
    }

    private static func isIdentifierStart(_ c: Character) -> Bool {
        c.isLetter || c == "_" || c == "$" || c == "@" || c == "#"
    }

    /// A shell `#` comment opens only after one of these — start of line
    /// (handled by the caller), whitespace, or a command separator /
    /// redirection operator. After a `$`, `{`, or an identifier character the
    /// `#` is part of an expansion or a word, not a comment.
    private static func isCommentWordBoundary(_ c: Character) -> Bool {
        c.isWhitespace
            || c == ";" || c == "&" || c == "|"
            || c == "(" || c == ")" || c == "`"
            || c == "<" || c == ">"
    }

    private static func isIdentifierChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_" || c == "$"
    }

    private static func punctuationTokenIsDelimited(
        _ token: String,
        chars: [Character],
        at index: Int
    ) -> Bool {
        if index > 0 {
            let previous = chars[index - 1]
            guard !isIdentifierChar(previous), previous != "-", previous != "." else {
                return false
            }
        }
        let end = index + token.count
        guard end < chars.count else { return true }
        let next = chars[end]
        return !isIdentifierChar(next) && next != "-" && next != "."
    }

    private static func apostropheIdentifierEnd(
        in chars: [Character],
        at index: Int
    ) -> Int? {
        guard chars[index] == "'", index + 1 < chars.count else { return nil }
        let first = chars[index + 1]
        guard first.isLetter || first == "_" else { return nil }

        var end = index + 2
        while end < chars.count && isIdentifierChar(chars[end]) { end += 1 }
        guard end == chars.count || chars[end] != "'" else { return nil }
        return end
    }

    /// Permissive on purpose: covers `0xFF`, `1_000`, `1.5e-3`, `100n`.
    /// A malformed literal colours as a number, which is harmless.
    private static func isNumberChar(_ c: Character) -> Bool {
        c.isHexDigit || c == "." || c == "_" || c == "x" || c == "X"
            || c == "e" || c == "E" || c == "o" || c == "b" || c == "n"
    }
}

// MARK: - Language definitions

extension SyntaxHighlighter.Grammar {

    static let swift = Self(
        lineComment: ["///", "//"],
        blockComment: [("/*", "*/")],
        quotes: ["\""],
        tripleQuotes: ["\"\"\""],
        backslashEscapes: true,
        keywords: [
            "associatedtype", "class", "deinit", "enum", "extension", "fileprivate",
            "func", "import", "init", "inout", "internal", "let", "open", "operator",
            "private", "precedencegroup", "protocol", "public", "rethrows", "static",
            "struct", "subscript", "typealias", "var", "break", "case", "catch",
            "continue", "default", "defer", "do", "else", "fallthrough", "for",
            "guard", "if", "in", "repeat", "return", "throw", "switch", "where",
            "while", "as", "false", "is", "nil", "self", "Self", "super", "throws",
            "true", "try", "async", "await", "actor", "nonisolated", "some", "any",
            "lazy", "weak", "unowned", "mutating", "nonmutating", "override", "final",
            "required", "convenience", "indirect", "@escaping", "@MainActor"
        ],
        types: [
            "Int", "Double", "Float", "String", "Bool", "Character", "Array",
            "Dictionary", "Set", "Optional", "Result", "Data", "Date", "URL",
            "Error", "Void", "AnyObject", "Task", "Sendable"
        ],
        capitalisedIdentifiersAreTypes: true,
        nestableBlockComments: true
    )

    static let python = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: ["\"\"\"", "'''"],
        backslashEscapes: true,
        keywords: [
            "False", "None", "True", "and", "as", "assert", "async", "await",
            "break", "class", "continue", "def", "del", "elif", "else", "except",
            "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
            "while", "with", "yield", "match", "case", "self", "cls"
        ],
        types: [
            "int", "float", "str", "bool", "bytes", "list", "dict", "set",
            "tuple", "frozenset", "complex", "object", "type", "range",
            "Optional", "Any", "Union", "Callable", "Iterator", "Sequence"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    static let javascript = Self(
        lineComment: ["//"],
        blockComment: [("/*", "*/")],
        quotes: ["\"", "'", "`"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "async", "await", "break", "case", "catch", "class", "const",
            "continue", "debugger", "default", "delete", "do", "else", "export",
            "extends", "finally", "for", "function", "if", "import", "in",
            "instanceof", "let", "new", "of", "return", "static", "super",
            "switch", "this", "throw", "try", "typeof", "var", "void", "while",
            "with", "yield", "true", "false", "null", "undefined", "get", "set"
        ],
        types: [
            "Array", "Object", "String", "Number", "Boolean", "Promise", "Map",
            "Set", "Symbol", "BigInt", "Date", "RegExp", "Error", "JSON", "Math",
            "console", "document", "window"
        ],
        capitalisedIdentifiersAreTypes: true,
        // Template literals span lines and still process `\`` escapes.
        escapedMultilineQuotes: ["`"]
    )

    static let typescript = Self(
        lineComment: javascript.lineComment,
        blockComment: javascript.blockComment,
        quotes: javascript.quotes,
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: javascript.keywords.union([
            "interface", "type", "enum", "namespace", "declare", "abstract",
            "implements", "private", "protected", "public", "readonly", "as",
            "is", "keyof", "infer", "satisfies", "override"
        ]),
        types: javascript.types.union([
            "string", "number", "boolean", "any", "unknown", "never", "void",
            "object", "bigint", "symbol", "Record", "Partial", "Readonly",
            "Pick", "Omit", "Awaited"
        ]),
        capitalisedIdentifiersAreTypes: true,
        escapedMultilineQuotes: javascript.escapedMultilineQuotes
    )

    /// JSON has no keywords beyond the three literals; the payoff is
    /// colouring strings and numbers apart from punctuation.
    static let json = Self(
        lineComment: ["//"],
        blockComment: [("/*", "*/")],
        quotes: ["\""],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: ["true", "false", "null"],
        types: [],
        capitalisedIdentifiersAreTypes: false
    )

    static let shell = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        // Single quotes in POSIX shell are literal, but tracking that
        // per-quote needs state this scanner doesn't carry. Escaping is
        // the safer default: it keeps `\"` inside a double-quoted string
        // from ending it, and the cost is only that `'\''` colours
        // slightly wrong.
        backslashEscapes: true,
        keywords: [
            "if", "then", "else", "elif", "fi", "case", "esac", "for", "while",
            "until", "do", "done", "in", "function", "select", "time", "return",
            "break", "continue", "export", "local", "readonly", "declare",
            "source", "alias", "unset", "shift", "trap", "set", "eval", "exec"
        ],
        types: [
            "echo", "cd", "ls", "cat", "grep", "sed", "awk", "curl", "git",
            "mkdir", "rm", "cp", "mv", "chmod", "sudo", "apt", "brew", "npm",
            "pip", "python", "node", "swift", "cargo", "go", "docker", "make"
        ],
        capitalisedIdentifiersAreTypes: false,
        // `#` is a comment only at a word boundary — `$#` and `${#x}` are not.
        lineCommentNeedsWordBoundary: true
    )

    static let go = Self(
        lineComment: ["//"],
        blockComment: [("/*", "*/")],
        quotes: ["\"", "`", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "break", "case", "chan", "const", "continue", "default", "defer",
            "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
            "interface", "map", "package", "range", "return", "select", "struct",
            "switch", "type", "var", "nil", "true", "false", "iota"
        ],
        types: [
            "bool", "byte", "complex64", "complex128", "error", "float32",
            "float64", "int", "int8", "int16", "int32", "int64", "rune", "string",
            "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "any"
        ],
        capitalisedIdentifiersAreTypes: false,
        // Go's backtick string is raw and spans lines; `"` and `'` keep the
        // ordinary single-line, escape-aware behaviour.
        rawMultilineQuotes: ["`"]
    )

    static let rust = Self(
        lineComment: ["///", "//!", "//"],
        blockComment: [("/*", "*/")],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "as", "async", "await", "break", "const", "continue", "crate", "dyn",
            "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in",
            "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
            "self", "Self", "static", "struct", "super", "trait", "true", "type",
            "unsafe", "use", "where", "while", "union"
        ],
        types: [
            "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32",
            "u64", "u128", "usize", "f32", "f64", "bool", "char", "str",
            "String", "Vec", "Option", "Result", "Box", "Rc", "Arc", "HashMap"
        ],
        capitalisedIdentifiersAreTypes: true,
        apostrophePrefixedIdentifiers: true,
        nestableBlockComments: true
    )

    // MARK: C family

    /// Shared shape for the `/* */` + `//` C-descended languages. Only
    /// the word lists differ, so they are built from this rather than
    /// repeating the delimiter facts eight times.
    private static func cLike(
        lineComment: [String] = ["//"],
        quotes: [Character] = ["\"", "'"],
        tripleQuotes: [String] = [],
        keywords: Set<String>,
        types: Set<String>,
        capitalisedAreTypes: Bool,
        nestableBlockComments: Bool = false
    ) -> Self {
        Self(
            lineComment: lineComment,
            blockComment: [("/*", "*/")],
            quotes: quotes,
            tripleQuotes: tripleQuotes,
            backslashEscapes: true,
            keywords: keywords,
            types: types,
            capitalisedIdentifiersAreTypes: capitalisedAreTypes,
            nestableBlockComments: nestableBlockComments
        )
    }

    static let c = cLike(
        keywords: [
            "auto", "break", "case", "const", "continue", "default", "do",
            "else", "enum", "extern", "for", "goto", "if", "inline", "register",
            "restrict", "return", "sizeof", "static", "struct", "switch",
            "typedef", "union", "volatile", "while", "NULL", "true", "false",
            "#include", "#define", "#ifdef", "#ifndef", "#endif", "#pragma"
        ],
        types: [
            "char", "double", "float", "int", "long", "short", "signed",
            "unsigned", "void", "size_t", "ssize_t", "bool", "FILE",
            "int8_t", "int16_t", "int32_t", "int64_t",
            "uint8_t", "uint16_t", "uint32_t", "uint64_t"
        ],
        capitalisedAreTypes: false
    )

    /// Also serves Objective-C: the `@interface` / `nil` / `BOOL` tokens
    /// are folded in rather than given a near-duplicate grammar, since a
    /// lexer with no grammar cannot tell the dialects apart anyway.
    static let cpp = cLike(
        keywords: c.keywords.union([
            "class", "namespace", "template", "typename", "public", "private",
            "protected", "virtual", "override", "final", "friend", "operator",
            "new", "delete", "this", "throw", "try", "catch", "using",
            "constexpr", "consteval", "decltype", "explicit", "mutable",
            "noexcept", "nullptr", "static_cast", "dynamic_cast", "const_cast",
            "reinterpret_cast", "co_await", "co_return", "co_yield", "concept",
            "requires", "import", "module",
            "@interface", "@implementation", "@end", "@property", "@synthesize",
            "@selector", "@autoreleasepool", "nil", "YES", "NO", "self"
        ]),
        types: c.types.union([
            "string", "wstring", "vector", "map", "unordered_map", "set",
            "unordered_set", "array", "pair", "tuple", "optional", "variant",
            "shared_ptr", "unique_ptr", "weak_ptr", "function", "thread",
            "mutex", "atomic", "auto", "BOOL", "NSString", "NSArray",
            "NSDictionary", "NSObject", "id", "instancetype"
        ]),
        capitalisedAreTypes: true
    )

    static let csharp = cLike(
        lineComment: ["///", "//"],
        keywords: [
            "abstract", "as", "base", "break", "case", "catch", "checked",
            "class", "const", "continue", "default", "delegate", "do", "else",
            "enum", "event", "explicit", "extern", "false", "finally", "fixed",
            "for", "foreach", "goto", "if", "implicit", "in", "interface",
            "internal", "is", "lock", "namespace", "new", "null", "operator",
            "out", "override", "params", "private", "protected", "public",
            "readonly", "ref", "return", "sealed", "sizeof", "stackalloc",
            "static", "struct", "switch", "this", "throw", "true", "try",
            "typeof", "unchecked", "unsafe", "using", "virtual", "volatile",
            "while", "async", "await", "var", "dynamic", "yield", "nameof",
            "record", "init", "when", "where", "with", "get", "set"
        ],
        types: [
            "bool", "byte", "char", "decimal", "double", "float", "int",
            "long", "object", "sbyte", "short", "string", "uint", "ulong",
            "ushort", "void", "List", "Dictionary", "IEnumerable", "Task",
            "Action", "Func", "Nullable", "Span"
        ],
        capitalisedAreTypes: true
    )

    static let java = cLike(
        lineComment: ["///", "//"],
        keywords: [
            "abstract", "assert", "break", "case", "catch", "class", "const",
            "continue", "default", "do", "else", "enum", "extends", "final",
            "finally", "for", "goto", "if", "implements", "import",
            "instanceof", "interface", "native", "new", "package", "private",
            "protected", "public", "return", "static", "strictfp", "super",
            "switch", "synchronized", "this", "throw", "throws", "transient",
            "try", "volatile", "while", "true", "false", "null", "var",
            "record", "sealed", "permits", "yield"
        ],
        types: [
            "boolean", "byte", "char", "double", "float", "int", "long",
            "short", "void", "String", "Object", "Integer", "Long", "Double",
            "Boolean", "Character", "List", "ArrayList", "Map", "HashMap",
            "Set", "HashSet", "Optional", "Stream", "Exception"
        ],
        capitalisedAreTypes: true
    )

    static let kotlin = cLike(
        lineComment: ["///", "//"],
        tripleQuotes: ["\"\"\""],
        keywords: [
            "as", "break", "class", "continue", "do", "else", "false", "for",
            "fun", "if", "in", "interface", "is", "null", "object", "package",
            "return", "super", "this", "throw", "true", "try", "typealias",
            "typeof", "val", "var", "when", "while", "by", "catch",
            "constructor", "delegate", "dynamic", "field", "file", "finally",
            "get", "import", "init", "param", "property", "receiver", "set",
            "setparam", "where", "abstract", "actual", "annotation",
            "companion", "const", "crossinline", "data", "enum", "expect",
            "external", "final", "infix", "inline", "inner", "internal",
            "lateinit", "noinline", "open", "operator", "out", "override",
            "private", "protected", "public", "reified", "sealed", "suspend",
            "tailrec", "vararg"
        ],
        types: [
            "Int", "Long", "Short", "Byte", "Float", "Double", "Boolean",
            "Char", "String", "Array", "List", "MutableList", "Map",
            "MutableMap", "Set", "MutableSet", "Any", "Unit", "Nothing"
        ],
        capitalisedAreTypes: true
    )

    static let scala = cLike(
        lineComment: ["///", "//"],
        keywords: [
            "abstract", "case", "catch", "class", "def", "do", "else",
            "extends", "false", "final", "finally", "for", "forSome", "if",
            "implicit", "import", "lazy", "match", "new", "null", "object",
            "override", "package", "private", "protected", "return", "sealed",
            "super", "this", "throw", "trait", "try", "true", "type", "val",
            "var", "while", "with", "yield", "given", "using", "enum",
            "export", "extension", "then"
        ],
        types: [
            "Int", "Long", "Short", "Byte", "Float", "Double", "Boolean",
            "Char", "String", "Unit", "Any", "AnyRef", "AnyVal", "Nothing",
            "Option", "Some", "None", "List", "Seq", "Map", "Set", "Vector",
            "Either", "Future", "Try"
        ],
        capitalisedAreTypes: true
    )

    static let dart = cLike(
        lineComment: ["///", "//"],
        keywords: [
            "abstract", "as", "assert", "async", "await", "break", "case",
            "catch", "class", "const", "continue", "covariant", "default",
            "deferred", "do", "dynamic", "else", "enum", "export", "extends",
            "extension", "external", "factory", "false", "final", "finally",
            "for", "get", "hide", "if", "implements", "import", "in",
            "interface", "is", "late", "library", "mixin", "new", "null",
            "on", "operator", "part", "required", "rethrow", "return", "set",
            "show", "static", "super", "switch", "sync", "this", "throw",
            "true", "try", "typedef", "var", "void", "while", "with", "yield"
        ],
        types: [
            "int", "double", "num", "bool", "String", "List", "Map", "Set",
            "Object", "Future", "Stream", "Iterable", "Widget", "BuildContext",
            "StatelessWidget", "StatefulWidget"
        ],
        capitalisedAreTypes: true
    )

    // MARK: Scripting

    static let ruby = Self(
        lineComment: ["#"],
        blockComment: [("=begin", "=end")],
        quotes: ["\"", "'", "`"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "alias", "and", "begin", "break", "case", "class", "def",
            "defined?", "do", "else", "elsif", "end", "ensure", "false", "for",
            "if", "in", "module", "next", "nil", "not", "or", "redo", "rescue",
            "retry", "return", "self", "super", "then", "true", "undef",
            "unless", "until", "when", "while", "yield", "require",
            "require_relative", "attr_accessor", "attr_reader", "attr_writer",
            "private", "protected", "public", "lambda", "proc", "puts"
        ],
        types: [
            "Integer", "Float", "String", "Symbol", "Array", "Hash", "Range",
            "Proc", "Class", "Module", "Object", "Struct", "Exception", "Time"
        ],
        capitalisedIdentifiersAreTypes: true
    )

    static let php = Self(
        lineComment: ["//", "#"],
        blockComment: [("/*", "*/")],
        quotes: ["\"", "'", "`"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "abstract", "and", "array", "as", "break", "callable", "case",
            "catch", "class", "clone", "const", "continue", "declare",
            "default", "do", "echo", "else", "elseif", "empty", "enddeclare",
            "endfor", "endforeach", "endif", "endswitch", "endwhile", "enum",
            "extends", "final", "finally", "fn", "for", "foreach", "function",
            "global", "goto", "if", "implements", "include", "include_once",
            "instanceof", "insteadof", "interface", "isset", "list", "match",
            "namespace", "new", "or", "print", "private", "protected",
            "public", "readonly", "require", "require_once", "return",
            "static", "switch", "throw", "trait", "try", "unset", "use", "var",
            "while", "xor", "yield", "true", "false", "null", "$this"
        ],
        types: [
            "int", "float", "string", "bool", "object", "mixed", "void",
            "iterable", "self", "parent", "never", "Closure", "Exception",
            "ArrayObject", "Generator", "Traversable"
        ],
        capitalisedIdentifiersAreTypes: true
    )

    static let lua = Self(
        lineComment: ["--"],
        blockComment: [("--[[", "]]")],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "and", "break", "do", "else", "elseif", "end", "false", "for",
            "function", "goto", "if", "in", "local", "nil", "not", "or",
            "repeat", "return", "then", "true", "until", "while", "self"
        ],
        types: [
            "string", "table", "math", "io", "os", "coroutine", "debug",
            "package", "require", "pairs", "ipairs", "type", "tostring",
            "tonumber", "print", "pcall", "setmetatable", "getmetatable"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    static let perl = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'", "`"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "my", "our", "local", "sub", "package", "use", "no", "require",
            "if", "elsif", "else", "unless", "while", "until", "for",
            "foreach", "do", "last", "next", "redo", "return", "and", "or",
            "not", "eq", "ne", "lt", "gt", "le", "ge", "cmp", "x", "qw", "qq",
            "bless", "ref", "defined", "undef", "wantarray"
        ],
        types: [
            "print", "printf", "sprintf", "push", "pop", "shift", "unshift",
            "splice", "split", "join", "keys", "values", "each", "sort",
            "reverse", "grep", "map", "scalar", "chomp", "chop", "die", "warn"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    static let r = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'", "`"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "if", "else", "repeat", "while", "function", "for", "in", "next",
            "break", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_",
            "NA_real_", "NA_character_", "library", "require", "return"
        ],
        types: [
            "c", "vector", "list", "matrix", "array", "data.frame", "factor",
            "numeric", "character", "logical", "integer", "double", "complex",
            "apply", "lapply", "sapply", "vapply", "mapply", "print", "paste"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    static let elixir = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: ["\"\"\""],
        backslashEscapes: true,
        keywords: [
            "def", "defp", "defmodule", "defmacro", "defmacrop", "defstruct",
            "defprotocol", "defimpl", "defdelegate", "defexception",
            "defguard", "defguardp", "do", "end", "fn", "if", "unless", "else",
            "case", "cond", "with", "for", "when", "and", "or", "not", "in",
            "after", "rescue", "catch", "raise", "throw", "try", "receive",
            "alias", "import", "require", "use", "quote", "unquote", "true",
            "false", "nil"
        ],
        types: [
            "Enum", "Map", "List", "String", "Atom", "Tuple", "Keyword",
            "Stream", "Task", "Agent", "GenServer", "Supervisor", "Process",
            "IO", "Kernel", "Integer", "Float"
        ],
        capitalisedIdentifiersAreTypes: true
    )

    static let haskell = Self(
        lineComment: ["--"],
        blockComment: [("{-", "-}")],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "case", "class", "data", "default", "deriving", "do", "else",
            "foreign", "if", "import", "in", "infix", "infixl", "infixr",
            "instance", "let", "module", "newtype", "of", "then", "type",
            "where", "forall", "mdo", "pattern"
        ],
        types: [
            "Int", "Integer", "Float", "Double", "Char", "String", "Bool",
            "Maybe", "Either", "IO", "Ordering", "Word", "Rational",
            "Functor", "Applicative", "Monad", "Show", "Eq", "Ord"
        ],
        capitalisedIdentifiersAreTypes: true,
        // Haskell `{- ... -}` block comments nest.
        nestableBlockComments: true
    )

    // MARK: Data / markup / config

    /// SQL keywords are conventionally written upper-case but the lexer
    /// matches exactly, so both cases are listed. Cheaper and more
    /// predictable than case-folding every identifier during the scan.
    static let sql = Self(
        lineComment: ["--"],
        blockComment: [("/*", "*/")],
        quotes: ["'", "\"", "`"],
        tripleQuotes: [],
        backslashEscapes: false,
        keywords: Set(
            [
                "select", "from", "where", "insert", "into", "values", "update",
                "set", "delete", "create", "table", "alter", "drop", "index",
                "view", "join", "inner", "left", "right", "full", "outer",
                "cross", "on", "group", "by", "order", "having", "limit",
                "offset", "union", "all", "distinct", "as", "and", "or", "not",
                "null", "is", "in", "between", "like", "ilike", "exists",
                "case", "when", "then", "else", "end", "with", "recursive",
                "primary", "key", "foreign", "references", "unique", "default",
                "constraint", "cascade", "grant", "revoke", "begin", "commit",
                "rollback", "transaction", "returning", "using", "asc", "desc",
                "true", "false", "if", "replace", "database", "schema"
            ].flatMap { [$0, $0.uppercased()] }
        ),
        types: Set(
            [
                "int", "integer", "bigint", "smallint", "serial", "bigserial",
                "decimal", "numeric", "real", "double", "precision", "float",
                "char", "varchar", "text", "bytea", "blob", "boolean", "bool",
                "date", "time", "timestamp", "timestamptz", "interval", "uuid",
                "json", "jsonb", "array", "count", "sum", "avg", "min", "max",
                "coalesce", "nullif", "cast", "now"
            ].flatMap { [$0, $0.uppercased()] }
        ),
        capitalisedIdentifiersAreTypes: false
    )

    /// HTML / XML and the template dialects that wrap them. Tag names are
    /// not distinguished from attributes — a real markup lexer needs
    /// nesting state this scanner does not carry — but comments,
    /// quoted attribute values and entities all colour correctly, which
    /// is most of the benefit.
    static let markup = Self(
        lineComment: [],
        blockComment: [("<!--", "-->")],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: false,
        keywords: [
            "html", "head", "body", "div", "span", "a", "p", "ul", "ol", "li",
            "table", "thead", "tbody", "tr", "td", "th", "form", "input",
            "button", "select", "option", "textarea", "label", "img", "script",
            "style", "link", "meta", "title", "header", "footer", "nav",
            "section", "article", "aside", "main", "h1", "h2", "h3", "h4",
            "h5", "h6", "template", "slot", "svg", "path", "circle", "rect"
        ],
        types: [
            "class", "id", "src", "href", "type", "name", "value", "style",
            "width", "height", "alt", "title", "rel", "target", "placeholder",
            "disabled", "checked", "selected", "readonly", "required",
            "xmlns", "viewBox", "fill", "stroke", "data", "aria"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    /// Plain CSS has ONLY `/* */` block comments — no `//`. Recognising
    /// `//` here mis-fires on the `//` in an unquoted `url(https://…)`,
    /// colouring the rest of the declaration as a comment. The `//`
    /// line comment belongs to the SCSS/Less preprocessors, which get their
    /// own grammar (``scss``) that adds it back.
    private static func cssFamily(lineComment: [String]) -> Self {
        Self(
            lineComment: lineComment,
            blockComment: [("/*", "*/")],
            quotes: ["\"", "'"],
            tripleQuotes: [],
            backslashEscapes: false,
            keywords: [
                "@media", "@import", "@charset", "@font-face", "@keyframes",
                "@supports", "@namespace", "@page", "@use", "@forward", "@mixin",
                "@include", "@extend", "@function", "@return", "@if", "@else",
                "@each", "@for", "@while", "important", "inherit", "initial",
                "unset", "revert", "var", "calc", "url", "from", "to"
            ],
            types: [
                "color", "background", "background-color", "border", "margin",
                "padding", "display", "position", "top", "right", "bottom", "left",
                "width", "height", "font", "font-size", "font-family",
                "font-weight", "text-align", "line-height", "flex", "grid", "gap",
                "opacity", "overflow", "z-index", "transform", "transition",
                "animation", "content", "cursor", "visibility", "box-shadow"
            ],
            capitalisedIdentifiersAreTypes: false
        )
    }

    static let css = cssFamily(lineComment: [])
    static let scss = cssFamily(lineComment: ["//"])

    static let yaml = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "true", "false", "null", "yes", "no", "on", "off", "True",
            "False", "Null", "TRUE", "FALSE", "NULL", "Yes", "No"
        ],
        types: [],
        capitalisedIdentifiersAreTypes: false
    )

    static let toml = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: ["\"\"\"", "'''"],
        backslashEscapes: true,
        keywords: ["true", "false"],
        types: [],
        capitalisedIdentifiersAreTypes: false
    )

    static let dockerfile = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV",
            "ADD", "COPY", "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG",
            "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL", "AS",
            "from", "run", "cmd", "copy", "env", "workdir", "as"
        ],
        types: [],
        capitalisedIdentifiersAreTypes: false
    )

    static let makefile = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\"", "'"],
        tripleQuotes: [],
        backslashEscapes: true,
        keywords: [
            "ifeq", "ifneq", "ifdef", "ifndef", "else", "endif", "include",
            "define", "endef", "export", "unexport", "override", "vpath",
            ".PHONY", ".SUFFIXES", ".DEFAULT", ".PRECIOUS", ".SECONDARY",
            "set", "project", "cmake_minimum_required", "add_executable",
            "add_library", "target_link_libraries", "find_package"
        ],
        types: [
            "wildcard", "patsubst", "subst", "shell", "foreach", "filter",
            "filter-out", "sort", "dir", "notdir", "basename", "suffix",
            "addprefix", "addsuffix", "abspath", "realpath", "call", "eval"
        ],
        capitalisedIdentifiersAreTypes: false
    )

    static let protobuf = cLike(
        keywords: [
            "syntax", "package", "import", "option", "message", "enum",
            "service", "rpc", "returns", "repeated", "optional", "required",
            "oneof", "map", "reserved", "extend", "extensions", "public",
            "weak", "stream", "true", "false"
        ],
        types: [
            "double", "float", "int32", "int64", "uint32", "uint64", "sint32",
            "sint64", "fixed32", "fixed64", "sfixed32", "sfixed64", "bool",
            "string", "bytes", "Any", "Timestamp", "Duration", "Empty"
        ],
        capitalisedAreTypes: true
    )

    static let graphql = Self(
        lineComment: ["#"],
        blockComment: [],
        quotes: ["\""],
        tripleQuotes: ["\"\"\""],
        backslashEscapes: true,
        keywords: [
            "query", "mutation", "subscription", "fragment", "on", "type",
            "interface", "union", "enum", "input", "scalar", "schema",
            "directive", "extend", "implements", "repeatable", "true",
            "false", "null"
        ],
        types: [
            "Int", "Float", "String", "Boolean", "ID"
        ],
        capitalisedIdentifiersAreTypes: true
    )

    /// Unified-diff hunks. There is no lexical structure to speak of, but
    /// colouring `+`/`-` lines is the single most useful thing a
    /// highlighter can do to a patch — and the scanner's line-comment
    /// mechanism already colours to end-of-line, so `+` and `-` are
    /// registered as "comment" openers and recoloured by the palette.
    static let diff = Self(
        lineComment: ["+++", "---", "@@", "diff ", "index ", "+", "-"],
        blockComment: [],
        quotes: [],
        tripleQuotes: [],
        backslashEscapes: false,
        keywords: [],
        types: [],
        capitalisedIdentifiersAreTypes: false,
        lineCommentAtLineStart: true
    )
}
