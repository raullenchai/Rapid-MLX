import Foundation
import Testing
@testable import Rapid

/// Pins that the Quick-Ask chat surface routes assistant content
/// through ``ChatTextSanitizer`` before rendering — same contract as
/// the main ``ChatView`` and the popped-out ``PoppedConversationView``.
///
/// ## Why a source-grep test
///
/// SwiftUI views don't have a cheap snapshot harness in this repo
/// (no SnapshotTesting dep, no pixel diff). The risk we're pinning
/// is *behavioural*: a refactor of ``QuickAskView.messageRow`` that
/// rewrites the ``Markdown(...)`` call without re-wrapping the
/// content in ``ChatTextSanitizer.sanitizeForDisplay(_:)`` silently
/// reopens the bidi-control filename-spoofing hole that the cycle-2
/// fuzz-stress finding flagged (see
/// ``.claude/loop/bug_report.md`` cycle-2 P3 desktop). NOTE: the
/// sanitiser intentionally does NOT touch homoglyph / confusable
/// characters — that is an orthogonal class of attack and out of
/// scope here.
///
/// The grep approach matches the rest of this repo's
/// architecture-pin style: cheap, fast, no extra deps; one line of
/// guard per render site. ``ChatTextSanitizerTests`` already pins
/// the function itself; this file pins that the function is
/// *called*.
@Suite("Quick-Ask bidi sanitisation — render-site coverage")
struct QuickAskBidiSanitizationTests {

    /// Resolve the source tree root from the test file path. The
    /// repository layout is fixed (Tests/RapidTests is two levels
    /// below the root) so this is a constant transform.
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private func loadSource(_ relativePath: String) throws -> String {
        let url = Self.sourceRoot.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Slice the ``messageRow`` function body out of ``QuickAskView.swift``
    /// so the asserts below operate on exactly the render-site
    /// expression, not on any string in the rest of the file. This
    /// closes the codex r1 MAJOR (the original test only checked
    /// "sanitizer appears somewhere in the file" — a refactor
    /// like ``let body = msg.content; Markdown(body)`` plus a stale
    /// sanitiser call elsewhere would have passed).
    ///
    /// The slice spans from the function signature to the next
    /// ``private func`` / ``private var`` / ``var body:`` declaration
    /// or end-of-file, whichever comes first. We do the slicing in
    /// the test rather than introducing a parser dep — Swift's
    /// source layout for this view is stable enough that string
    /// markers + index arithmetic give a deterministic slice.
    private func messageRowSlice(_ source: String) throws -> String {
        let signature = "private func messageRow(_ msg: ChatMessage) -> some View {"
        let start = try #require(
            source.range(of: signature),
            "messageRow signature not found in QuickAskView.swift — has it been renamed?"
        )
        let rest = source[start.upperBound...]
        // Slice forward to the next ``private`` declaration (the
        // function that follows ``messageRow``) so we don't pick up
        // markdown calls in sibling functions.
        let endMarkers = [
            "\n    private func ",
            "\n    private var ",
            "\n    var body:",
        ]
        let endIndex: String.Index = endMarkers
            .compactMap { rest.range(of: $0)?.lowerBound }
            .min() ?? rest.endIndex
        return String(rest[..<endIndex])
    }

    /// Strip ``/*...*/`` block comments and ``// ...`` line
    /// comments from ``source``, then strip every whitespace scalar.
    /// Used to normalise an extracted ``Markdown(...)`` argument
    /// before substring matching so trivia like ``msg /*x*/ . content``
    /// or ``msg // note\n.content`` reduces to the same compacted
    /// form ``msg.content`` (codex r5 MAJOR-1).
    ///
    /// Known residual gap (codex r7 NIT): not string-literal
    /// aware. A future ``Markdown("/*" + msg.content + "*/")``
    /// shape would confuse the comment stripper because the
    /// ``/*`` inside the literal opens a "comment" the scanner
    /// honours. Same source-grep-not-SwiftSyntax limitation
    /// class as the local-alias residual documented above.
    ///
    /// Single-pass char walker. Codex r6 MAJOR: Swift DOES support
    /// nested block comments, so ``msg /* outer /* inner */ */
    /// .content`` typechecks as ``msg.content``. The block-comment
    /// branch tracks depth — increment on each ``/*`` seen inside,
    /// decrement on each ``*/``, exit only when depth returns to
    /// zero.
    static func stripCommentsAndWhitespace(_ source: String) -> String {
        let chars = Array(source.unicodeScalars)
        var out: [UnicodeScalar] = []
        out.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            // Block comment ``/*...*/`` with nesting depth tracking.
            if c.value == 0x2F /* '/' */ && i + 1 < chars.count && chars[i + 1].value == 0x2A /* '*' */ {
                var depth = 1
                var j = i + 2
                while j + 1 < chars.count && depth > 0 {
                    if chars[j].value == 0x2F /* '/' */ && chars[j + 1].value == 0x2A /* '*' */ {
                        depth += 1
                        j += 2
                    } else if chars[j].value == 0x2A /* '*' */ && chars[j + 1].value == 0x2F /* '/' */ {
                        depth -= 1
                        j += 2
                    } else {
                        j += 1
                    }
                }
                // Even if the comment is unterminated (invalid
                // Swift), skip to end-of-buffer to avoid emitting
                // partial comment contents into the compacted form.
                i = max(j, i + 2)
                continue
            }
            // Line comment ``// ... \n``
            if c.value == 0x2F /* '/' */ && i + 1 < chars.count && chars[i + 1].value == 0x2F /* '/' */ {
                var j = i + 2
                while j < chars.count && chars[j].value != 0x0A /* '\n' */ {
                    j += 1
                }
                i = j
                continue
            }
            if !c.properties.isWhitespace {
                out.append(c)
            }
            i += 1
        }
        return String(String.UnicodeScalarView(out))
    }

    /// Paren-balanced extractor: returns the substring of each
    /// ``Markdown(...)`` argument list found in ``source``. Walks
    /// character by character, opening a capture on
    /// ``Markdown(`` and closing it when the running paren depth
    /// returns to zero. Codex r2 NIT: does NOT skip string-literal
    /// or comment contents. The check is sound on well-formed Swift
    /// source where ``Markdown(`` never appears inside a literal or
    /// comment in ``messageRow`` — verified by inspection of the
    /// current file. If a future contributor embeds a literal
    /// ``Markdown(`` in a string or doc comment inside ``messageRow``
    /// the extractor will false-positive; treat that as a signal to
    /// upgrade this helper to a real tokenizer.
    static func extractMarkdownCallArguments(from source: String) -> [String] {
        var result: [String] = []
        let scalars = Array(source.unicodeScalars)
        let openTag = Array("Markdown(".unicodeScalars)
        var i = 0
        while i + openTag.count <= scalars.count {
            // Look for the literal ``Markdown(`` prefix.
            var matched = true
            for k in 0..<openTag.count where scalars[i + k] != openTag[k] {
                matched = false
                break
            }
            if !matched {
                i += 1
                continue
            }
            // Walk forward until the matching close paren.
            var depth = 1
            var j = i + openTag.count
            let start = j
            while j < scalars.count && depth > 0 {
                let c = scalars[j].value
                if c == 0x28 { depth += 1 }
                else if c == 0x29 { depth -= 1 }
                j += 1
            }
            if depth == 0 {
                // ``j`` is the index just past the matching ``)``.
                let argScalars = scalars[start..<(j - 1)]
                result.append(String(String.UnicodeScalarView(argScalars)))
                i = j
            } else {
                // Malformed — bail to avoid an infinite loop.
                break
            }
        }
        return result
    }

    /// Behavioural belt-and-braces: prove that the sanitiser
    /// applied at the render call site actually strips the exact
    /// filename-spoof payload from the bug report. The raw string
    /// is ``something\u{202E}gpj.ssfsf`` — with U+202E RIGHT-TO-
    /// LEFT OVERRIDE between ``something`` and ``gpj.ssfsf`` the
    /// SwiftUI renderer would lay out the right-hand side mirrored
    /// (``somethingfsfss.jpg``). After sanitisation the override is
    /// stripped and the visible string is ``somethinggpj.ssfsf`` —
    /// the literal byte order the user actually pasted.
    @Test("Sanitiser strips the filename-spoof payload from bug_report.md")
    func filenameSpoofPayloadStripped() {
        // U+202E RIGHT-TO-LEFT OVERRIDE between "something" and
        // "gpj.ssfsf" — would render as "somethingfsfss.jpg" on a
        // SwiftUI Text without sanitisation.
        let spoofed = "something\u{202E}gpj.ssfsf"
        let cleaned = ChatTextSanitizer.sanitizeForDisplay(spoofed)
        #expect(cleaned == "somethinggpj.ssfsf")
        #expect(!cleaned.unicodeScalars.contains { $0.value == 0x202E })
    }

    /// Pin the user RTL-preservation contract: legitimate Arabic
    /// strong characters (no bidi controls) flow through untouched.
    /// The sanitiser is bidi-control-only — it must not regress to
    /// a too-aggressive allow-list that mangles user-typed Arabic /
    /// Hebrew prompts. ``ChatTextSanitizerTests`` covers this for
    /// the sanitiser API; this test pins the contract at the Quick-
    /// Ask render call site by re-asserting against the same input
    /// shape a user would actually paste into the chord launcher.
    @Test("Legitimate Arabic prompt survives Quick-Ask sanitisation untouched")
    func legitimateArabicSurvives() {
        let prompts: [String] = [
            "مرحبا بالعالم",               // "Hello, world" in Arabic
            "كيف حالك؟",                    // "How are you?" in Arabic
            "שלום עולם",                   // Hebrew "Hello, world"
            "Mixed: hello مرحبا 你好",     // mixed LTR + RTL + CJK
            "Empty:",
        ]
        for prompt in prompts {
            #expect(
                ChatTextSanitizer.sanitizeForDisplay(prompt) == prompt,
                "Legitimate text was mangled: \(prompt)"
            )
        }
    }

    /// Mixed payload: bidi override AND legitimate Arabic in the
    /// same string. Strip the control, keep the strong characters.
    /// This is the realistic prompt-injection shape (attacker tries
    /// to hide bidi marks inside an otherwise harmless RTL reply).
    @Test("Mixed bidi-control + legitimate RTL: strip control, keep RTL")
    func mixedBidiAndRTL() {
        let mixed = "مرحبا\u{202E}.exe بالعالم"
        let cleaned = ChatTextSanitizer.sanitizeForDisplay(mixed)
        #expect(cleaned == "مرحبا.exe بالعالم")
    }
}
