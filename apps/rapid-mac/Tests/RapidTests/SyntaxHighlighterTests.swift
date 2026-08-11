import Testing
import SwiftUI
@testable import Rapid

/// Pins ``SyntaxHighlighter``'s contract. The scanner is a lexer, not a
/// parser, so these tests assert token BOUNDARIES and the fallback
/// guarantees — not that every construct in every language is coloured
/// the way an IDE would colour it.
@Suite("SyntaxHighlighter — fenced code block colouring")
struct SyntaxHighlighterTests {

    /// Collect the substrings that carry a given token colour.
    private func runs(
        _ code: String,
        _ language: String?,
        kind: SyntaxHighlighter.TokenKind
    ) -> [String] {
        let attributed = SyntaxHighlighter.highlight(code, language: language)
        let wanted = SyntaxHighlighter.color(for: kind)
        var found: [String] = []
        for run in attributed.runs where run.foregroundColor == wanted {
            found.append(String(attributed[run.range].characters))
        }
        return found
    }

    private func plainText(_ attributed: AttributedString) -> String {
        String(attributed.characters)
    }

    // MARK: - Fallback guarantees

    @Test("Unknown language returns unstyled text")
    func unknownLanguageIsPlain() {
        let code = "SELECT * FROM users WHERE id = 1;"
        let out = SyntaxHighlighter.highlight(code, language: "unsupported-language")
        #expect(plainText(out) == code)
        // No run carries a colour.
        for run in out.runs {
            #expect(run.foregroundColor == nil)
        }
        #expect(!SyntaxHighlighter.supports(language: "unsupported-language"))
    }

    @Test("Absent language returns unstyled text")
    func nilLanguageIsPlain() {
        let code = "some bare fenced text"
        let out = SyntaxHighlighter.highlight(code, language: nil)
        #expect(plainText(out) == code)
        #expect(!SyntaxHighlighter.supports(language: nil))
    }

    /// The property that matters most: colouring must never corrupt the
    /// code. Whatever we emit has to read back identical to the input.
    @Test("Highlighting preserves the source exactly")
    func roundTripsSource() {
        let samples: [(String, String)] = [
            ("swift", "let x: Int = 42 // note\nprint(\"hi\\n\")"),
            ("python", "def f(x):\n    \"\"\"doc\"\"\"\n    return x * 2  # cmt"),
            ("json", "{\"a\": [1, 2.5, true, null], \"b\": \"s\"}"),
            ("bash", "grep -rn 'x' . | awk '{print $1}'  # find"),
            ("rust", "fn main() { let v: Vec<u8> = vec![1]; }"),
            ("go", "func main() { s := `raw` ; _ = s }"),
            ("typescript", "const x: Record<string, number> = {};"),
            ("javascript", "const s = `tpl ${x}`; // done")
        ]
        for (lang, code) in samples {
            let out = SyntaxHighlighter.highlight(code, language: lang)
            #expect(plainText(out) == code, "round-trip failed for \(lang)")
        }
    }

    @Test("Configured punctuation-bearing tokens colour as one run")
    func compoundConfiguredTokens() {
        #expect(runs(".PHONY: all", "makefile", kind: .keyword).contains(".PHONY"))
        #expect(runs("$(filter-out %.tmp,$(FILES))", "makefile", kind: .type)
            .contains("filter-out"))
        #expect(runs("a { background-color: red; }", "css", kind: .type)
            .contains("background-color"))
        #expect(runs("@font-face { font-family: Demo; }", "css", kind: .keyword)
            .contains("@font-face"))
        #expect(!runs("not-background-colorful", "css", kind: .type)
            .contains("background-color"))
    }

    @Test("Empty input is handled")
    func emptyInput() {
        let out = SyntaxHighlighter.highlight("", language: "swift")
        #expect(plainText(out) == "")
    }

    // MARK: - Language aliases

    @Test("Common language aliases resolve")
    func aliasesResolve() {
        for alias in ["py", "python3", "js", "jsx", "ts", "tsx", "sh", "zsh", "golang", "rs"] {
            #expect(SyntaxHighlighter.supports(language: alias), "alias \(alias) unsupported")
        }
    }

    /// The mainstream set a chat reply is likely to contain. A missing
    /// entry here means those blocks silently render flat.
    @Test("Mainstream languages are all supported")
    func mainstreamLanguagesSupported() {
        let expected = [
            "swift", "python", "javascript", "typescript", "json", "bash",
            "go", "rust", "c", "cpp", "csharp", "java", "kotlin", "ruby",
            "php", "sql", "html", "css", "yaml", "toml", "dockerfile",
            "makefile", "lua", "r", "scala", "dart", "perl", "haskell",
            "elixir", "protobuf", "graphql", "diff"
        ]
        for lang in expected {
            #expect(SyntaxHighlighter.supports(language: lang), "\(lang) unsupported")
        }
    }

    @Test("Extended aliases resolve to their language")
    func extendedAliases() {
        let aliases = [
            "c++", "cc", "hpp", "objc", "m", "c#", "cs", "kt", "rb",
            "postgres", "mysql", "xml", "svg", "vue", "scss", "sass",
            "yml", "ini", "docker", "make", "cmake", "hs", "ex", "proto",
            "gql", "patch", "json5", "fish", "cjs"
        ]
        for alias in aliases {
            #expect(SyntaxHighlighter.supports(language: alias), "alias \(alias) unsupported")
        }
    }

    /// Every grammar must round-trip its own sample. A malformed
    /// delimiter list (an unbalanced block comment, say) would corrupt
    /// output for that language only, which a single-language test would
    /// miss.
    private static let languageSamples: [String: String] = [
        "c": "#include <stdio.h>\nint main(void) { /* hi */ return 0; }",
        "cpp": "template<class T> class A { public: T v = 1; };",
        "csharp": "public async Task<int> F() => await G(); // note",
        "java": "public class A { private int x = 1; /* c */ }",
        "kotlin": "fun main() { val x: Int = 1 }",
        "ruby": "def f(a)\n  puts \"x#{a}\"\nend",
        "php": "<?php function f() { return $this->x; } // c",
        "sql": "SELECT a, b FROM t WHERE c = 'x' -- note",
        "html": "<div class=\"a\"><!-- c --><p>hi</p></div>",
        "css": "/* c */ .a { color: red; font-size: 1rem; }",
        "yaml": "key: value  # comment\nlist:\n  - true",
        "toml": "[section]\nkey = \"value\"  # c",
        "dockerfile": "FROM node:20 AS build\nRUN npm ci  # c",
        "makefile": "all:\n\t$(CC) -o x x.c  # build",
        "lua": "local function f() return nil end  -- c",
        "r": "f <- function(x) { return(x + 1) }  # c",
        "scala": "def f(x: Int): Option[Int] = Some(x)",
        "dart": "void main() { final x = <int>[1]; }",
        "perl": "my $x = shift; print \"$x\\n\";  # c",
        "haskell": "f :: Int -> Int\nf x = x + 1  -- c",
        "elixir": "defmodule A do\n  def f(x), do: x\nend",
        "protobuf": "message A { optional string b = 1; }",
        "graphql": "query Q { user(id: 1) { name } }  # c",
        "diff": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new"
    ]

    @Test(
        "Every supported language round-trips a sample",
        arguments: [
            "c", "cpp", "csharp", "java", "kotlin", "ruby", "php", "sql",
            "html", "css", "yaml", "toml", "dockerfile", "makefile", "lua",
            "r", "scala", "dart", "perl", "haskell", "elixir", "protobuf",
            "graphql", "diff"
        ]
    )
    func supportedLanguageRoundTrips(_ language: String) throws {
        let code = try #require(Self.languageSamples[language])
        let out = SyntaxHighlighter.highlight(code, language: language)
        #expect(plainText(out) == code, "round-trip failed for \(language)")
    }

    @Test("SQL keywords match in both cases")
    func sqlCaseInsensitivity() {
        #expect(runs("SELECT x FROM t", "sql", kind: .keyword).contains("SELECT"))
        #expect(runs("select x from t", "sql", kind: .keyword).contains("select"))
    }

    @Test("Languages with non-slash comments classify them")
    func alternateCommentSyntax() {
        #expect(runs("-- note\nSELECT 1", "sql", kind: .comment).contains("-- note"))
        #expect(runs("-- note\nlocal x = 1", "lua", kind: .comment).contains("-- note"))
        #expect(runs("# note\nkey: 1", "yaml", kind: .comment).contains("# note"))
        #expect(runs("<!-- note -->\n<p>x</p>", "html", kind: .comment).contains("<!-- note -->"))
        #expect(runs("{- note -}\nf = 1", "haskell", kind: .comment).contains("{- note -}"))
    }

    @Test("Dockerfile instructions classify as keywords")
    func dockerfileInstructions() {
        let keywords = runs("FROM alpine\nRUN echo hi", "dockerfile", kind: .keyword)
        #expect(keywords.contains("FROM"))
        #expect(keywords.contains("RUN"))
    }

    @Test("Language matching is case and whitespace insensitive")
    func languageNormalisation() {
        #expect(SyntaxHighlighter.supports(language: "Python"))
        #expect(SyntaxHighlighter.supports(language: "  swift  "))
        #expect(SyntaxHighlighter.supports(language: "JSON"))
    }

    // MARK: - Token classification

    @Test("Hash-prefixed directives advance and classify")
    func hashPrefixedDirective() {
        let code = "#include <stdio.h>"
        let out = SyntaxHighlighter.highlight(code, language: "c")
        #expect(plainText(out) == code)
        #expect(runs(code, "c", kind: .keyword).contains("#include"))
    }

    @Test("At-prefixed attributes advance and classify")
    func atPrefixedAttribute() {
        let code = "@MainActor func run() {}"
        let out = SyntaxHighlighter.highlight(code, language: "swift")
        #expect(plainText(out) == code)
        #expect(runs(code, "swift", kind: .keyword).contains("@MainActor"))
    }

    @Test("Swift keywords and comments are classified")
    func swiftTokens() {
        let code = "// lead\nlet name = \"v\"\nfunc go() -> Int { 42 }"
        #expect(runs(code, "swift", kind: .keyword).contains("let"))
        #expect(runs(code, "swift", kind: .keyword).contains("func"))
        #expect(runs(code, "swift", kind: .comment).contains("// lead"))
        #expect(runs(code, "swift", kind: .string).contains("\"v\""))
        #expect(runs(code, "swift", kind: .number).contains("42"))
        #expect(runs(code, "swift", kind: .type).contains("Int"))
    }

    @Test("Python triple-quoted strings scan as one string")
    func pythonDocstring() {
        let code = "def f():\n    \"\"\"line one\n    line two\"\"\"\n    pass"
        let strings = runs(code, "python", kind: .string)
        #expect(strings.contains { $0.contains("line one") && $0.contains("line two") })
    }

    /// A `#` inside a string must not open a comment. This is the
    /// ordering guarantee in the scanner, and the one most likely to
    /// break if the branches are ever reordered.
    @Test("A comment marker inside a string does not start a comment")
    func hashInsideStringIsNotComment() {
        let code = "url = \"http://x/#frag\"\n"
        let comments = runs(code, "python", kind: .comment)
        #expect(comments.isEmpty)
    }

    /// And the converse: a quote inside a comment must not open a string.
    @Test("A quote inside a comment does not start a string")
    func quoteInsideCommentIsNotString() {
        let code = "// it's fine\nlet x = 1"
        let strings = runs(code, "swift", kind: .string)
        #expect(strings.isEmpty)
    }

    @Test("Escaped quotes do not terminate a string")
    func escapedQuote() {
        let code = "let s = \"a\\\"b\"\nlet t = 1"
        let strings = runs(code, "swift", kind: .string)
        #expect(strings.contains { $0.contains("a\\\"b") })
    }

    /// An unterminated string stops at the newline. During streaming a
    /// half-arrived line is normal, and swallowing the remainder of the
    /// block would make the whole tail flash as a string on every frame.
    @Test("Unterminated string stops at end of line")
    func unterminatedStringStopsAtNewline() {
        let code = "let a = \"oops\nlet b = 2"
        let strings = runs(code, "swift", kind: .string)
        #expect(strings.allSatisfy { !$0.contains("\n") })
        // The next line still classifies normally.
        #expect(runs(code, "swift", kind: .keyword).contains("let"))
    }

    @Test("Block comments span lines and close correctly")
    func blockComment() {
        let code = "/* a\n b */ let x = 1"
        let comments = runs(code, "swift", kind: .comment)
        #expect(comments.contains { $0.contains("a") && $0.contains("b") })
        #expect(runs(code, "swift", kind: .keyword).contains("let"))
    }

    @Test("Unterminated block comment consumes the remainder")
    func unterminatedBlockComment() {
        let code = "let x = 1\n/* trailing"
        let out = SyntaxHighlighter.highlight(code, language: "swift")
        #expect(plainText(out) == code)
    }

    /// Digits inside an identifier belong to the identifier, so `utf8`
    /// must not split into `utf` + number `8`.
    @Test("Digits inside identifiers are not numbers")
    func digitsInIdentifier() {
        let code = "let utf8_1 = 0"
        let numbers = runs(code, "swift", kind: .number)
        #expect(!numbers.contains("8"))
        #expect(numbers.contains("0"))
    }

    @Test("Numeric literal forms scan as a single number")
    func numericForms() {
        for literal in ["0xFF", "1_000", "1.5", "42"] {
            let numbers = runs("let v = \(literal)", "swift", kind: .number)
            #expect(numbers.contains(literal), "failed for \(literal)")
        }
    }

    @Test("JSON literals and strings are classified")
    func jsonTokens() {
        let code = "{\"k\": true, \"n\": 1.5, \"z\": null}"
        #expect(runs(code, "json", kind: .keyword).contains("true"))
        #expect(runs(code, "json", kind: .keyword).contains("null"))
        #expect(runs(code, "json", kind: .string).contains("\"k\""))
        #expect(runs(code, "json", kind: .number).contains("1.5"))
    }

    @Test("Shell comments and keywords are classified")
    func shellTokens() {
        let code = "# setup\nif [ -f x ]; then echo 'hi'; fi"
        #expect(runs(code, "bash", kind: .comment).contains("# setup"))
        #expect(runs(code, "bash", kind: .keyword).contains("if"))
        #expect(runs(code, "bash", kind: .keyword).contains("then"))
    }

    @Test("Rust doc comments scan as comments")
    func rustDocComment() {
        let code = "/// doc\nfn f() {}"
        #expect(runs(code, "rust", kind: .comment).contains("/// doc"))
        #expect(runs(code, "rust", kind: .keyword).contains("fn"))
    }

    @Test("Rust lifetimes are not strings, while character literals are")
    func rustLifetimes() {
        let code = "fn longest<'a>(x: &'a str) -> &'a str { x }\nlet c = 'x'"
        #expect(runs(code, "rust", kind: .string) == ["'x'"])
        #expect(runs(code, "rust", kind: .keyword).contains("fn"))
        #expect(runs(code, "rust", kind: .type).filter { $0 == "str" }.count == 2)
    }

    @Test("Go raw strings are supported")
    func goRawString() {
        let code = "s := `raw string`"
        #expect(runs(code, "go", kind: .string).contains("`raw string`"))
    }

    @Test("Lua block comments win over their line-comment prefix")
    func luaBlockComment() {
        let code = "--[[ first\nlocal hidden = true\nreturn hidden\n]]\nlocal visible = false"
        let comments = runs(code, "lua", kind: .comment)
        #expect(comments.count == 1)
        #expect(comments[0].contains("local hidden"))
        #expect(runs(code, "lua", kind: .keyword).filter { $0 == "local" }.count == 1)
    }

    @Test("Diff content lines are highlighted")
    func diffContentLines() {
        let code = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new"
        let comments = runs(code, "diff", kind: .comment)
        #expect(comments.contains("-old"))
        #expect(comments.contains("+new"))
    }

    @Test("TypeScript inherits JavaScript keywords and adds its own")
    func typescriptInheritance() {
        let code = "interface X { a: string }\nconst y = 1"
        #expect(runs(code, "ts", kind: .keyword).contains("interface"))
        #expect(runs(code, "ts", kind: .keyword).contains("const"))
        #expect(runs(code, "ts", kind: .type).contains("string"))
    }

    // MARK: - Incremental rendering

    @Test("Incremental highlighting is identical after every streamed append")
    func incrementalHighlightMatchesColdScan() {
        let memo = SyntaxHighlighter.Memo()
        let chunks = [
            "let", " value", " = 1", "\n",
            "/* open", "\nstill open", " */ let text = \"a", "\\\"b\"", "\n",
            "return", " value", "\n",
        ]
        var code = ""

        for chunk in chunks {
            code += chunk
            #expect(
                memo.highlight(code, language: "swift")
                    == SyntaxHighlighter.highlight(code, language: "swift"),
                "incremental result diverged after appending \(String(reflecting: chunk))"
            )
        }

        // A reused SwiftUI code-block slot may receive replacement content.
        // Shrinks and language changes must reset instead of reusing stale runs.
        let replacement = "def f():\n    return 2\n"
        #expect(
            memo.highlight(replacement, language: "python")
                == SyntaxHighlighter.highlight(replacement, language: "python")
        )
    }

    @Test("Incremental highlighting matches a cold scan at every character boundary")
    func incrementalHighlightCharacterBoundarySweep() {
        let samples = [
            ("swift", "let greeting = \"你好\"\n/* multi\n line */ return greeting\n"),
            ("python", "def f():\n    \"\"\"multi\n    line\"\"\"\n    return 1\n"),
            ("json", "{\"emoji\": \"✨\", \"ok\": true}"),
            ("rust", "fn f<'a>(x: &'a str) -> &'a str { x }\nlet c = 'x'\n"),
            ("haskell", "{- outer\ncomment -}\nf x = x + 1\n"),
            ("diff", "@@ -1 +1 @@\n-old\n+new\n"),
        ]

        for (language, source) in samples {
            let memo = SyntaxHighlighter.Memo()
            var streamed = ""
            for character in source {
                streamed.append(character)
                #expect(
                    memo.highlight(streamed, language: language)
                        == SyntaxHighlighter.highlight(streamed, language: language),
                    "incremental result diverged for \(language) at \(streamed.count) characters"
                )
            }
        }
    }

    @Test("Incremental memo resets for same-language rewrites and fallback fences")
    func incrementalHighlightResetsSafely() {
        let memo = SyntaxHighlighter.Memo()
        _ = memo.highlight("let old = 1\nreturn old\n", language: "swift")

        let rewritten = "actor Replacement {\n    func value() -> Int { 42 }\n}\n"
        #expect(
            memo.highlight(rewritten, language: "swift")
                == SyntaxHighlighter.highlight(rewritten, language: "swift")
        )

        let shortened = "let x = 2"
        #expect(
            memo.highlight(shortened, language: "swift")
                == SyntaxHighlighter.highlight(shortened, language: "swift")
        )

        let unsupported = "plain text"
        #expect(
            memo.highlight(unsupported, language: "unknown")
                == SyntaxHighlighter.highlight(unsupported, language: "unknown")
        )
    }

    @Test("Streaming complete lines scans the source approximately once")
    func incrementalHighlightAvoidsPrefixRescans() {
        let memo = SyntaxHighlighter.Memo()
        let line = "let value = compute(x: 42, y: \"text\") // trailing comment\n"
        var code = ""
        var naiveScannedCharacters = 0

        for _ in 0..<400 {
            code += line
            naiveScannedCharacters += code.count
            _ = memo.highlight(code, language: "swift")
        }

        #expect(memo.scannedCharacterCount <= code.count * 2)
        #expect(memo.scannedCharacterCount * 20 < naiveScannedCharacters)
    }

    // MARK: - Performance shape

    /// The highlighter runs on the render path, so a large block must
    /// stay linear rather than quadratic.
    @Test("A large code block highlights in reasonable time")
    func largeBlockIsLinear() {
        let line = "let value = compute(x: 42, y: \"text\") // trailing comment\n"
        let big = String(repeating: line, count: 2_000)
        let started = Date()
        let out = SyntaxHighlighter.highlight(big, language: "swift")
        let elapsed = Date().timeIntervalSince(started)
        #expect(plainText(out) == big)
        #expect(elapsed < 3.0, "took \(elapsed)s — check for quadratic behaviour")
    }

    /// What the render path cares about is the latency of ONE more token on
    /// the block so far — the per-frame cost — not the cumulative total over
    /// a whole stream. That per-append cost is a prefix check plus a
    /// last-line rescan; both are bounded by the block size, and on a large
    /// block a single append still lands in well under a frame.
    ///
    /// (The cumulative prefix-validation across a full stream is O(n²): each
    /// append re-confirms the retained prefix. That check is a deliberate
    /// correctness safeguard — it is what detects an edit/retry REPLACEMENT
    /// rather than an append, exercised by ``incrementalHighlightResetsSafely``
    /// — and its constant is small enough that the total is negligible for
    /// chat-sized blocks. It is spread across seconds of streamed output, not
    /// paid in one frame, so it never shows as a hitch.)
    @Test("A single streamed append stays cheap on a large block")
    func perAppendCostIsBounded() {
        let memo = SyntaxHighlighter.Memo()
        let token = "value += compute(x: 42) // step\n"
        var code = ""
        for _ in 0..<2_000 {
            code += token
            _ = memo.highlight(code, language: "swift")
        }
        // One more token on a ~60 KB block. The bound is deliberately loose
        // — the deterministic linearity guarantee lives in
        // ``incrementalHighlightAvoidsPrefixRescans``; this only guards
        // against a gross per-frame regression (e.g. rescanning the whole
        // block per append) without flaking under CI contention.
        let started = Date()
        code += token
        _ = memo.highlight(code, language: "swift")
        let elapsed = Date().timeIntervalSince(started)
        #expect(elapsed < 1.0, "a single append on a ~60 KB block took \(elapsed)s")
    }

    // MARK: - Line-anchored diff markers

    /// A unified diff's `+`/`-` are line prefixes, not mid-line comment
    /// openers: an unchanged context line with an infix minus must not
    /// colour from the minus onward, while a real change line still does.
    @Test("Diff change markers colour only at the line start")
    func diffMarkersAnchoredToLineStart() {
        let code = " value = a - b\n-removed\n+added\n"
        let comments = runs(code, "diff", kind: .comment)
        #expect(!comments.contains { $0.contains("- b") })
        #expect(comments.contains("-removed"))
        #expect(comments.contains("+added"))
    }

    // MARK: - Go multiline raw strings

    /// Go's backtick string is raw and spans lines; a `//` inside it is
    /// part of the string, and code after the closing backtick resumes
    /// normal classification.
    @Test("Go backtick raw strings span multiple lines")
    func goMultilineRawString() {
        let code = "s := `line one\n// still string\nend`\nreturn 1\n"
        let strings = runs(code, "go", kind: .string)
        #expect(strings.contains { $0.contains("line one") && $0.contains("end") })
        #expect(runs(code, "go", kind: .comment).isEmpty)
        #expect(runs(code, "go", kind: .keyword).contains("return"))
    }

    // MARK: - Scientific notation

    @Test("Scientific-notation literals scan as one number")
    func scientificNotation() {
        for literal in ["1.5e-3", "2E+10", "6.02e23", "1e10"] {
            let numbers = runs("let v = \(literal)", "swift", kind: .number)
            #expect(numbers.contains(literal), "failed for \(literal)")
        }
        // An infix minus outside an exponent is still its own token.
        let numbers = runs("let d = 5 - 3", "swift", kind: .number)
        #expect(numbers.contains("5"))
        #expect(numbers.contains("3"))
        #expect(!numbers.contains { $0.contains("-") })
    }

    // MARK: - CSS vs SCSS comments

    /// Plain CSS has no `//` comment, so the `//` in an unquoted URL must
    /// not comment out the rest of the declaration. SCSS/Less keep `//`.
    @Test("Plain CSS does not treat // in a URL as a comment; SCSS still does")
    func cssUrlIsNotAComment() {
        let css = ".a { background: url(https://example.com/x.png); }"
        #expect(runs(css, "css", kind: .comment).isEmpty)
        #expect(runs("// note\n.a { color: red }", "scss", kind: .comment).contains("// note"))
        #expect(SyntaxHighlighter.supports(language: "less"))
    }

    // MARK: - JavaScript multiline template literals

    /// A backtick template literal spans lines and still honours escapes.
    /// A `//` or keyword inside it is part of the string, and code after
    /// the closing backtick resumes normal classification.
    @Test("JavaScript backtick templates span multiple lines")
    func jsMultilineTemplate() {
        let code = "const q = `line one\n// still string\nconst inside`\nreturn 1\n"
        let strings = runs(code, "javascript", kind: .string)
        #expect(strings.contains { $0.contains("line one") && $0.contains("const inside") })
        #expect(runs(code, "javascript", kind: .comment).isEmpty)
        #expect(runs(code, "javascript", kind: .keyword).contains("return"))
        // An escaped backtick does not close the template.
        let escaped = "const s = `a\\`b`\nconst t = 1"
        #expect(runs(escaped, "javascript", kind: .string).contains { $0.contains("a\\`b") })
    }

    // MARK: - Nested block comments

    /// Swift and Rust nest block comments: an inner `*/` must not close the
    /// outer comment. C-family languages (no nesting) still close at the
    /// first delimiter.
    @Test("Swift and Rust nest block comments; C does not")
    func nestedBlockComments() {
        let swift = "/* outer /* inner */ still outer */\nlet x = 1"
        let comments = runs(swift, "swift", kind: .comment)
        #expect(comments.contains { $0.contains("still outer") })
        // `let` after the true close is code, not swallowed into the comment.
        #expect(runs(swift, "swift", kind: .keyword).contains("let"))

        let rust = "/* a /* b */ c */\nfn f() {}"
        #expect(runs(rust, "rust", kind: .comment).contains { $0.contains(" c ") })
        #expect(runs(rust, "rust", kind: .keyword).contains("fn"))

        // C does NOT nest: the first `*/` closes, leaving the tail as code.
        let c = "/* a /* b */ int x = 1;"
        #expect(runs(c, "c", kind: .comment).contains { $0.contains("a") && !$0.contains("int") })
        #expect(runs(c, "c", kind: .type).contains("int"))

        // Haskell `{- -}` nests too.
        let haskell = "{- a {- b -} c -}\nf x = x"
        #expect(runs(haskell, "haskell", kind: .comment).contains { $0.contains(" c ") })
    }

    // MARK: - Shell # is a comment only at a word boundary

    /// Shell's `#` opens a comment only at the start of a word, so parameter
    /// expansions like `$#` and `${#arr[@]}` are not comments, while a real
    /// trailing `# comment` still is.
    @Test("Shell # in parameter expansion is not a comment")
    func shellHashInParameterExpansion() {
        let code = "echo $# ${#arr[@]}  # real comment\n"
        let comments = runs(code, "bash", kind: .comment)
        #expect(comments == ["# real comment"])
        // A leading-of-line comment still classifies.
        #expect(runs("# top\nls", "bash", kind: .comment).contains("# top"))
        // A `#` right after a separator / redirection operator IS a comment.
        #expect(runs("(true)# c\n", "bash", kind: .comment).contains("# c"))
        #expect(runs("echo x ># c\n", "bash", kind: .comment).contains("# c"))
    }

    // MARK: - Kotlin triple-quoted strings

    @Test("Kotlin triple-quoted strings scan as one multiline string")
    func kotlinTripleQuotedString() {
        let code = "val s = \"\"\"line one\n// still string\nend\"\"\"\nval n = 1"
        let strings = runs(code, "kotlin", kind: .string)
        #expect(strings.contains { $0.contains("line one") && $0.contains("end") })
        #expect(runs(code, "kotlin", kind: .comment).isEmpty)
        #expect(runs(code, "kotlin", kind: .keyword).contains("val"))
    }
}
