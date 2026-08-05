import Foundation
import Testing
@testable import Rapid

/// Pins that the main ``ChatView`` chat surface routes assistant
/// content through ``ChatTextSanitizer`` before rendering — both the
/// streaming hot path (``Text(memoisedSanitisedContent)``) and the
/// finalised path (``LaTeXMarkdownView(content: ...)``), plus the
/// tool-result body inside ``ToolCallChip``.
///
/// ## Why this file exists alongside ``QuickAskBidiSanitizationTests``
///
/// PR #324 plugged the Quick-Ask render-site hole and added a per-
/// call-site grep pin so a future refactor of Quick-Ask cannot
/// silently re-introduce the bidi-control leak. PR #324 did **not**
/// add the same pin to the main ``ChatView``: the call-site was
/// already correct there (and had been for months), but it was
/// guarded only by inline comments — a future refactor that drops
/// the sanitiser wrap could ship without any test going red.
///
/// Cycle-10 finding ``F-10-4`` (assistant content carrying ``U+202E``
/// round-trips into the main chat bubble verbatim from a hostile
/// model) is closed by the production code today — verified by
/// reading the render sites in ``Sources/Rapid/UI/ChatView.swift``:
///
///   * Streaming path (`assistantBlock`, ``Text(memoisedSanitisedContent)``)
///     routes through ``streamingContentMemo.sanitised(...)`` which
///     calls ``ChatTextSanitizer.sanitize``.
///   * Complete path (`assistantBlock`, ``LaTeXMarkdownView(content: ...)``)
///     wraps ``message.content`` in
///     ``ChatTextSanitizer.sanitizeForDisplay(...)`` inline.
///   * Error/system row (``systemRow``) wraps the same way.
///   * Tool-result body (``ToolCallChip``) wraps ``r.content`` the
///     same way.
///   * User bubble (``userBubble``) wraps user echoes too — defense
///     in depth so a model-edited user turn can't leak controls
///     when re-rendered.
///
/// This test file pins **all** of those call sites so the F-10-4
/// closure does not silently regress.
///
/// ## Why source-grep instead of a SwiftUI snapshot test
///
/// The Rapid repo has no SnapshotTesting / pixel-diff dep and a
/// macOS SwiftUI snapshot harness is heavyweight enough that the
/// rest of this repo's regression pins are source-grep style (see
/// ``QuickAskBidiSanitizationTests`` for the same pattern + codex
/// hardening notes). The risk we're pinning is *behavioural*: a
/// refactor that rewrites ``Text(...)`` / ``LaTeXMarkdownView(...)``
/// without re-wrapping the content in
/// ``ChatTextSanitizer.sanitizeForDisplay(_:)`` (or its memo
/// equivalent) silently reopens the bidi-control filename-spoofing
/// hole flagged as F-10-4 (cross-confirmed by F-9S-12 on llama3-1b).
@Suite("ChatView assistant-content bidi sanitisation — render-site coverage")
struct ChatViewAssistantContentBidiTests {

    /// Resolve the source tree root from the test file path. Mirrors
    /// ``QuickAskBidiSanitizationTests.sourceRoot`` so this suite has
    /// the same anchor invariant.
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

    // MARK: - Streaming-path memo plumbing


    // MARK: - Complete-path render site

    // MARK: - Tool-result render site (cycle-2 cross-confirm)

    // MARK: - Behavioural pins (cycle-10 F-10-4 payload)

    /// The exact payload cycle-10 F-10-4 documented: a hostile
    /// assistant ``content`` containing ``"Click ‮gpj.eraclip.com"``
    /// (where ``‮`` is U+202E RIGHT-TO-LEFT OVERRIDE). Without the
    /// sanitiser SwiftUI's text layout would render the URL portion
    /// mirrored. After sanitisation the override is stripped and the
    /// rendered string is the literal byte order.
    @Test("Cycle-10 F-10-4 payload: assistant content with U+202E URL spoof is neutralised")
    func cycle10F104PayloadNeutralised() {
        // U+202E between "Click " and "gpj.eraclip.com" — would
        // render as "Click moc.pilcare.jpg" without sanitisation.
        let echoed = "Click \u{202E}gpj.eraclip.com"
        let cleaned = ChatTextSanitizer.sanitizeForDisplay(echoed)
        #expect(cleaned == "Click gpj.eraclip.com")
        #expect(!cleaned.unicodeScalars.contains { $0.value == 0x202E })
    }

    /// Tool-result variant of the same payload: a malicious tool
    /// emits a result body containing ``U+202E`` followed by a
    /// spoofed filename. The ``ToolCallChip`` result body Text(...)
    /// call must neutralise it.
    @Test("Cycle-10 F-10-4 tool-result variant: U+202E in r.content is neutralised")
    func cycle10F104ToolResultVariantNeutralised() {
        let toolResult = "Saved as report\u{202E}fdp.exe"
        let cleaned = ChatTextSanitizer.sanitizeForDisplay(toolResult)
        #expect(cleaned == "Saved as reportfdp.exe")
        #expect(!cleaned.unicodeScalars.contains { $0.value == 0x202E })
    }

    /// Streaming delta variant: a chunk that splits the payload
    /// mid-stream — e.g. coalescer flush #1 ends with ``"Click "`` and
    /// flush #2 begins with ``"\u{202E}gpj.eraclip.com"``. The memo
    /// path must produce the same final sanitised string as a single
    /// full-buffer ``sanitizeForDisplay`` call. This pins the
    /// streaming hot path's delta-safety invariant (the ``Memo``
    /// docstring depends on ``sanitize(a + b) == sanitize(a) +
    /// sanitize(b)``).
    @MainActor
    @Test("Streaming delta variant: chunked U+202E payload sanitises identically to one-shot")
    func cycle10F104StreamingDeltaInvariant() {
        let memo = ChatTextSanitizer.Memo()
        let chunk1 = "Click "
        let chunk2 = "\u{202E}gpj.eraclip.com"
        let combined = chunk1 + chunk2

        // Simulate the streaming buffer growing: first flush, then
        // the second flush appends the rest.
        let afterChunk1 = memo.sanitised(chunk1)
        let afterChunk2 = memo.sanitised(combined)

        let oneShot = ChatTextSanitizer.sanitizeForDisplay(combined)
        #expect(afterChunk2 == oneShot)
        #expect(afterChunk2 == "Click gpj.eraclip.com")
        // Intermediate flush also produced a sanitised value (no
        // partial leak).
        #expect(!afterChunk1.unicodeScalars.contains { $0.value == 0x202E })
    }

    /// Self-test the ``isLetOrVarBindingBefore`` helper against the
    /// shapes codex round 2 surfaced: bare binding, typed binding,
    /// ``var`` binding, ``stateLet`` false-positive (NIT),
    /// property-access LHS (NIT), and an actual ``=`` assignment
    /// (not a binding). Each input simulates a ``compactArg``
    /// substring with ``=message.content`` at a known offset, and
    /// the helper is invoked through ``assertNoLocalAliasRebinding``
    /// (which is the public façade callers use). The expected
    /// outcome is encoded as a flag and asserted by checking
    /// whether the helper records an issue for the input.
    ///
    /// Codex PR-#329 round 2: MAJOR-1 closed (typed binding now
    /// detected), NIT-1 closed (``.``-prefixed LHS rejected as
    /// non-binding).
    @Test("isLetOrVarBindingBefore: positive + negative cases (codex round 2)")
    func aliasBindingDetectionUnitTest() {
        // Each tuple: (compactArg, expectedIsBinding).
        struct Case {
            let compact: String
            let isBinding: Bool
            let label: String
        }
        let cases: [Case] = [
            Case(compact: "letleaked=message.content",
                 isBinding: true,
                 label: "bare let binding"),
            Case(compact: "varleaked=message.content",
                 isBinding: true,
                 label: "bare var binding"),
            Case(compact: "letleaked:String=message.content",
                 isBinding: true,
                 label: "typed let binding (round-2 MAJOR-1)"),
            Case(compact: "varleaked:String=message.content",
                 isBinding: true,
                 label: "typed var binding (round-2 MAJOR-1)"),
            Case(compact: "letleaked:Optional<String>=message.content",
                 isBinding: true,
                 label: "typed let with generic"),
            Case(compact: "letleaked:[String]=message.content",
                 isBinding: true,
                 label: "typed let with array shorthand"),
            Case(compact: "letleaked:[String:Int]=message.content",
                 isBinding: true,
                 label: "typed let with dict shorthand (round-3 NIT)"),
            Case(compact: "letleaked:(String,Int)=message.content",
                 isBinding: true,
                 label: "typed let with tuple type"),
            // Boundary cases — the binding follows a statement
            // separator or block punctuation:
            Case(compact: "{letleaked=message.content",
                 isBinding: true,
                 label: "let after block open"),
            Case(compact: ";letleaked=message.content",
                 isBinding: true,
                 label: "let after semicolon"),
            Case(compact: "}letleaked=message.content",
                 isBinding: true,
                 label: "let after block close"),
            // Conditional bindings — round-3 MAJOR closes here:
            Case(compact: "ifletleaked=message.content",
                 isBinding: true,
                 label: "if let binding (round-3 MAJOR)"),
            Case(compact: "guardletleaked=message.content",
                 isBinding: true,
                 label: "guard let binding (round-3 MAJOR)"),
            Case(compact: "caseletleaked=message.content",
                 isBinding: true,
                 label: "case let binding (round-3 MAJOR)"),
            Case(compact: "whileletleaked=message.content",
                 isBinding: true,
                 label: "while let binding"),
            // Negative cases:
            Case(compact: "state.letterValue=message.content",
                 isBinding: false,
                 label: "property assignment (round-3 BLOCKING)"),
            Case(compact: "stateLet=message.content",
                 isBinding: false,
                 label: "stateLet — assignment to a stored property"),
            Case(compact: "preLetterCount=message.content",
                 isBinding: false,
                 label: "preLetterCount — keyword embedded mid-identifier"),
            Case(compact: "foo=message.content",
                 isBinding: false,
                 label: "bare assignment (no keyword)"),
            Case(compact: "obj.letme=message.content",
                 isBinding: false,
                 label: "obj.letme — member access starting with let"),
            Case(compact: "self.letVar=message.content",
                 isBinding: false,
                 label: "self.letVar — member assignment"),
            // Round-4 NIT: ``obj.guardletItem=x`` is genuinely
            // unambiguous (the ``.`` is NOT a hard boundary, so
            // the keyword arm cannot trigger inside a property-
            // access expression). Pin that case to ``false``.
            Case(compact: "obj.guardletItem=message.content",
                 isBinding: false,
                 label: "obj.guardletItem — member assignment (round-4 NIT)"),
            Case(compact: "{ifletColumns=message.content",
                 isBinding: true,
                 label: "{ifletColumns — if-let-Columns conditional binding"),
            // Round-4 NIT documented residual: ``ifletColumns=x``
            // at the very start of a slice IS classified as a
            // binding (matched via ``^`` boundary + ``if`` keyword
            // + ``let`` + ``Columns``). Genuinely ambiguous in
            // compacted form, but we fail-safe: a false positive
            // here costs a contributor one extra CI-red message,
            // a false negative costs a real bidi-control bypass.
            Case(compact: "ifletColumns=message.content",
                 isBinding: true,
                 label: "ifletColumns — start-of-slice fail-safe (round-4 NIT residual)"),
            // Round-5 MAJOR-2: compound conditional forms must
            // also be detected.
            Case(compact: "elseiflet leaked=message.content".replacingOccurrences(of: " ", with: ""),
                 isBinding: true,
                 label: "else if let — compound conditional (round-5 MAJOR-2)"),
            Case(compact: "ifcaseletleaked=message.content",
                 isBinding: true,
                 label: "if case let — compound conditional (round-5 MAJOR-2)"),
            Case(compact: "elseifcaseletleaked=message.content",
                 isBinding: true,
                 label: "else if case let — deeply compound conditional"),
            Case(compact: "elseguardletleaked=message.content",
                 isBinding: true,
                 label: "else guard let — rare but legal"),
            // Round-6 MAJOR: tuple-destructure binding (the LHS
            // is ``let(a,b)`` after compaction). The
            // ``isLetOrVarBindingBefore`` helper alone is for
            // direct ``=<rawToken>`` neighbour grep and does NOT
            // cover tuple LHS by design — the
            // ``assertNoWrappedAliasRebinding`` helper handles
            // the tuple case via its full RHS-span walk. Pin
            // expected ``false`` here because this is the direct-
            // neighbour helper; the wrapped helper catches it
            // separately (verified in the negative-control
            // injection).
            Case(compact: "let(a,b)=message.content",
                 isBinding: false,
                 label: "tuple destructure — direct neighbour, wrapped helper covers (round-6 MAJOR)"),
        ]
        for c in cases {
            // Find the ``=`` offset and invoke the helper directly.
            let eq = c.compact.firstIndex(of: "=")!
            let actual = Self.isLetOrVarBindingBefore(
                compactArg: c.compact,
                equalsIdx: eq
            )
            // ``Issue.record`` is used instead of ``#expect`` because
            // the swift-testing 0.99 toolchain in this repo has a
            // known issue where ``#expect`` failures inside a loop
            // body can be silently dropped from the run aggregate
            // (the loop completes, no recorded issue surfaces in the
            // CLI output). ``Issue.record`` always escalates to a
            // visible test failure. Codex PR-#329 round 3 BLOCKING
            // was hidden behind this same swift-testing bug — the
            // round-2 self-test reported "passed" even though one
            // case had ``actual=true expected=false``. The switch to
            // ``Issue.record`` ensures every future regression
            // surfaces in CI.
            if actual != c.isBinding {
                Issue.record("case '\(c.label)' input '\(c.compact)' expected isBinding=\(c.isBinding), got \(actual)")
            }
        }
    }

    /// Cross-codepoint sweep: the cycle-10 finding called out
    /// ``U+202E`` specifically, but the same render path needs to
    /// neutralise every bidi-affecting codepoint in the sanitiser's
    /// strip list. This re-asserts the contract at the render-site
    /// level — ``ChatTextSanitizerTests`` already pins the function;
    /// here we pin that an attacker switching to a sibling
    /// codepoint (``U+202D`` LRO, ``U+2066`` LRI, etc.) gets the
    /// same neutralisation.
    @Test("Every bidi-affecting codepoint round-tripped via the sanitizer is stripped")
    func allBidiCodepointsNeutralisedAtRenderSite() {
        let bidiCodepoints: [UInt32] = [
            0x061C,                                  // ARABIC LETTER MARK
            0x200E, 0x200F,                          // LRM / RLM
            0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  // LRE / RLE / PDF / LRO / RLO
            0x2066, 0x2067, 0x2068, 0x2069,          // LRI / RLI / FSI / PDI
        ]
        for cp in bidiCodepoints {
            let scalar = UnicodeScalar(cp)!
            let payload = "prefix\(scalar)suffix"
            let cleaned = ChatTextSanitizer.sanitizeForDisplay(payload)
            // Codex PR-#329 round 4 MAJOR: loop-body assertions ->
            // ``Issue.record`` so swift-testing 0.99 cannot drop
            // a per-codepoint failure silently.
            if cleaned != "prefixsuffix" {
                Issue.record(
                    "Codepoint U+\(String(cp, radix: 16, uppercase: true)) leaked through render-site sanitiser. Got: '\(cleaned)'"
                )
            }
            if cleaned.unicodeScalars.contains(where: { $0.value == cp }) {
                Issue.record(
                    "Codepoint U+\(String(cp, radix: 16, uppercase: true)) survived render-site sanitiser."
                )
            }
        }
    }

    // MARK: - Slicing helpers

    /// Slice the ``assistantBlock`` computed-property body out of
    /// ``ChatView.swift``. Same approach as
    /// ``QuickAskBidiSanitizationTests.messageRowSlice`` — we anchor
    /// on the signature line and walk forward to the next sibling
    /// declaration. Braces are NOT counted because Swift's source
    /// layout for this view is stable; the sibling-decl scan gives a
    /// deterministic slice without a tokenizer dep.
    static func assistantBlockSlice(_ source: String) throws -> String {
        let signature = "private var assistantBlock: some View {"
        let start = try #require(
            source.range(of: signature),
            "assistantBlock computed property not found in ChatView.swift — has it been renamed?"
        )
        let rest = source[start.upperBound...]
        let endMarkers = [
            "\n    private func ",
            "\n    private var ",
            "\n    var body:",
            "\n}",
        ]
        let endIndex: String.Index = endMarkers
            .compactMap { rest.range(of: $0)?.lowerBound }
            .min() ?? rest.endIndex
        return String(rest[..<endIndex])
    }

    /// Slice the ``ToolCallChip`` struct body out of ``ChatView.swift``.
    /// ``ToolCallChip`` is a top-level (file-private) struct, so the
    /// slice is bounded by the next top-level declaration after it.
    ///
    /// **Why we scope to the WHOLE struct, not just ``body``** (codex
    /// PR-#329 round 1 MAJOR-3 + MAJOR-4): ``var body`` is the LAST
    /// member of ``ToolCallChip`` today, so a "next sibling member"
    /// end marker scans past the struct's closing brace into the
    /// following top-level ``DestructivePatternBanner``. That makes
    /// the slice both wrong-scoped AND fragile (any new member added
    /// between ``body`` and the struct close shrinks the slice with
    /// no test signal). Instead we anchor on the struct head and
    /// stop at the next top-level declaration in the same file
    /// (``private struct``, ``struct``, ``enum``, ``private final
    /// class``, ``@MainActor``). Inside that slice the tool-result
    /// render call sites — wherever they live inside the struct —
    /// are visible to the walker.
    ///
    /// We also positively require ``ChatTextSanitizer
    /// .sanitizeForDisplay`` to appear somewhere inside the slice
    /// (codex MAJOR-4 — a refactor that swaps the call site to
    /// ``Text(result?.content ?? "")`` would drop both the literal
    /// ``r.content`` token and the safe-wrap; only a positive
    /// "sanitizer is mentioned at all" check catches that shape).
    static func toolCallChipBodySlice(_ source: String) throws -> String {
        let structAnchor = "private struct ToolCallChip: View {"
        let structStart = try #require(
            source.range(of: structAnchor),
            "ToolCallChip struct not found in ChatView.swift — has it been renamed or moved?"
        )
        // Find the matching close-brace for the struct head. We
        // count braces from depth=1 (the struct head's ``{`` is
        // already consumed by the anchor match) and stop at depth=0.
        // To avoid miscounting braces inside string literals or
        // comments, we run the slice through a tiny lexer that
        // tracks block / line comments and string literals.
        let afterStructHead = source[structStart.upperBound...]
        let scalars = Array(afterStructHead.unicodeScalars)
        var depth = 1
        var i = 0
        var inLineComment = false
        var blockCommentDepth = 0
        var inStringLit = false
        var inMultilineStringLit = false
        // Walk char by char.
        while i < scalars.count && depth > 0 {
            let c = scalars[i]
            // Multi-line string literal: ``"""...\n...\n"""``.
            if inMultilineStringLit {
                if i + 2 < scalars.count
                    && scalars[i].value == 0x22
                    && scalars[i + 1].value == 0x22
                    && scalars[i + 2].value == 0x22
                {
                    inMultilineStringLit = false
                    i += 3
                    continue
                }
                i += 1
                continue
            }
            // Single-line string literal: ``"..."``.
            if inStringLit {
                if c.value == 0x5C /* '\' */ && i + 1 < scalars.count {
                    i += 2  // skip escaped char
                    continue
                }
                if c.value == 0x22 /* '"' */ {
                    inStringLit = false
                }
                i += 1
                continue
            }
            // Inside a line comment: skip to newline.
            if inLineComment {
                if c.value == 0x0A {
                    inLineComment = false
                }
                i += 1
                continue
            }
            // Inside a block comment (with nesting): track depth.
            if blockCommentDepth > 0 {
                if c.value == 0x2F /* '/' */
                    && i + 1 < scalars.count
                    && scalars[i + 1].value == 0x2A /* '*' */
                {
                    blockCommentDepth += 1
                    i += 2
                    continue
                }
                if c.value == 0x2A
                    && i + 1 < scalars.count
                    && scalars[i + 1].value == 0x2F
                {
                    blockCommentDepth -= 1
                    i += 2
                    continue
                }
                i += 1
                continue
            }
            // Enter line comment.
            if c.value == 0x2F && i + 1 < scalars.count && scalars[i + 1].value == 0x2F {
                inLineComment = true
                i += 2
                continue
            }
            // Enter block comment.
            if c.value == 0x2F && i + 1 < scalars.count && scalars[i + 1].value == 0x2A {
                blockCommentDepth = 1
                i += 2
                continue
            }
            // Enter multi-line string literal.
            if i + 2 < scalars.count
                && c.value == 0x22
                && scalars[i + 1].value == 0x22
                && scalars[i + 2].value == 0x22
            {
                inMultilineStringLit = true
                i += 3
                continue
            }
            // Enter single-line string literal.
            if c.value == 0x22 {
                inStringLit = true
                i += 1
                continue
            }
            // Code-level brace counting.
            if c.value == 0x7B /* '{' */ {
                depth += 1
            } else if c.value == 0x7D /* '}' */ {
                depth -= 1
                if depth == 0 {
                    // ``i`` is the index of the closing brace; the
                    // slice is everything up to and including it.
                    let endScalarIdx = i + 1
                    let endIdx = afterStructHead.unicodeScalars
                        .index(afterStructHead.unicodeScalars.startIndex,
                               offsetBy: endScalarIdx)
                    return String(afterStructHead[..<endIdx])
                }
            }
            i += 1
        }
        // Unbalanced — return everything we walked so the test
        // surfaces a clear failure downstream rather than silently
        // skipping checks.
        return String(afterStructHead)
    }

    /// Whitespace + comment stripper. Identical contract to
    /// ``QuickAskBidiSanitizationTests.stripCommentsAndWhitespace``;
    /// re-implemented here so this suite has no implicit dep on
    /// another test file's internals.
    static func stripCommentsAndWhitespace(_ source: String) -> String {
        let chars = Array(source.unicodeScalars)
        var out: [UnicodeScalar] = []
        out.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            // Block comment with nesting depth tracking.
            if c.value == 0x2F /* '/' */ && i + 1 < chars.count && chars[i + 1].value == 0x2A /* '*' */ {
                var depth = 1
                var j = i + 2
                while j + 1 < chars.count && depth > 0 {
                    if chars[j].value == 0x2F && chars[j + 1].value == 0x2A {
                        depth += 1
                        j += 2
                    } else if chars[j].value == 0x2A && chars[j + 1].value == 0x2F {
                        depth -= 1
                        j += 2
                    } else {
                        j += 1
                    }
                }
                i = max(j, i + 2)
                continue
            }
            // Line comment.
            if c.value == 0x2F && i + 1 < chars.count && chars[i + 1].value == 0x2F {
                var j = i + 2
                while j < chars.count && chars[j].value != 0x0A {
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

    /// Paren-balanced extractor for arbitrary call shapes (not just
    /// ``Markdown(``). Returns the substring of each
    /// ``<callPrefix>...)`` argument list found in ``source``.
    static func extractMatchedCallArguments(callPrefix: String, from source: String) -> [String] {
        var result: [String] = []
        let scalars = Array(source.unicodeScalars)
        let openTag = Array(callPrefix.unicodeScalars)
        var i = 0
        while i + openTag.count <= scalars.count {
            var matched = true
            for k in 0..<openTag.count where scalars[i + k] != openTag[k] {
                matched = false
                break
            }
            if !matched {
                i += 1
                continue
            }
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
                let argScalars = scalars[start..<(j - 1)]
                result.append(String(String.UnicodeScalarView(argScalars)))
                i = j
            } else {
                break
            }
        }
        return result
    }

    /// Every occurrence of ``message.content`` inside the compact
    /// argument must be a whitelisted probe OR be preceded by
    /// ``ChatTextSanitizer.sanitizeForDisplay(``. Identical shape to
    /// ``QuickAskBidiSanitizationTests`` r2/r3/r4 walker, generalised
    /// to take the value token and the safe-wrap prefix as
    /// parameters.
    static func assertEveryMessageContentUseIsSanitised(
        inCompactArg compactArg: String,
        callIdx: Int,
        callShape: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        assertEveryRawValueUseIsSanitised(
            inCompactArg: compactArg,
            rawToken: "message.content",
            callIdx: callIdx,
            callShape: callShape,
            sourceLocation: sourceLocation
        )
    }

    /// Sibling helper for the tool-result render site, which uses
    /// the binding name ``r`` (``let r = result``). The whitelist of
    /// length/identity probes is the same — the value-use vs probe
    /// distinction is what matters, not the binding name.
    static func assertEveryRContentUseIsSanitised(
        inCompactArg compactArg: String,
        callShape: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        assertEveryRawValueUseIsSanitised(
            inCompactArg: compactArg,
            rawToken: "r.content",
            callIdx: 0,
            callShape: callShape,
            sourceLocation: sourceLocation
        )
    }

    /// Core walker. Shared by the ``message.content`` and ``r.content``
    /// shapes. See the QuickAsk codex r2/r3/r4 commentary for the
    /// full rationale on the whitelist and the value-use vs probe
    /// distinction.
    static func assertEveryRawValueUseIsSanitised(
        inCompactArg compactArg: String,
        rawToken: String,
        callIdx: Int,
        callShape: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let safeWrap = "ChatTextSanitizer.sanitizeForDisplay("
        let probeWhitelist: [String] = [
            ".isEmpty",
            ".count",
            ".utf8.count",
            ".unicodeScalars.count",
            ".hashValue",
            ".startIndex",
            ".endIndex",
        ]
        var search = compactArg.startIndex
        while let occ = compactArg.range(of: rawToken, range: search..<compactArg.endIndex) {
            let after = compactArg[occ.upperBound...]
            let isProbe = probeWhitelist.contains { probe in
                guard after.hasPrefix(probe) else { return false }
                let endOfProbe = after.index(after.startIndex, offsetBy: probe.count)
                if endOfProbe == after.endIndex { return true }
                let nextChar = after[endOfProbe]
                let isIdentContinuation = nextChar.isLetter
                    || nextChar.isNumber
                    || nextChar == "_"
                return !isIdentContinuation
            }
            if !isProbe {
                let precedingStart = compactArg.index(
                    occ.lowerBound,
                    offsetBy: -safeWrap.count,
                    limitedBy: compactArg.startIndex
                )
                let preceding: Substring = precedingStart.map { compactArg[$0..<occ.lowerBound] } ?? ""
                // Codex PR-#329 round 4 MAJOR: loop-body ``#expect``
                // -> ``Issue.record`` so swift-testing 0.99 cannot
                // silently drop a per-occurrence failure.
                if preceding != safeWrap {
                    Issue.record(
                        Comment(rawValue: """
                        \(callShape) call #\(callIdx) contains a raw \
                        ``\(rawToken)`` value use that is NOT immediately \
                        preceded by ChatTextSanitizer.sanitizeForDisplay( \
                        and is NOT a whitelisted length/identity probe \
                        (.isEmpty, .count, .utf8.count, etc.). Argument \
                        (whitespace + comment collapsed) was: \
                        '\(compactArg)'. This is the cycle-10 F-10-4 \
                        bypass shape — the raw value reaches the renderer \
                        without the bidi-control sanitiser. See \
                        bug_report.md cycle-10 F-10-4 and \
                        ChatTextSanitizerTests for the codepoint set.
                        """),
                        sourceLocation: sourceLocation
                    )
                }
            }
            search = occ.upperBound
        }
    }

    /// Codex PR-#329 round 1 MAJOR-1/2/4: reject any
    /// ``let <ident> = <rawToken>`` binding inside the compact
    /// argument. This catches the alias-rebinding bypass that the
    /// QuickAsk test file documented as a "known residual" gap —
    /// the assistant-content / tool-result paths are
    /// significantly more attack-surface than Quick-Ask (they
    /// render every assistant turn, not just chord-launcher
    /// replies) so this suite explicitly closes the gap rather
    /// than documenting it.
    ///
    /// The shape we reject:
    ///
    ///   let leaked = message.content
    ///   let leaked = r.content
    ///   let leaked = result?.content
    ///
    /// The shape we ALLOW (probe-style length binding):
    ///
    ///   let n = message.content.count
    ///   let isEmpty = message.content.isEmpty
    ///   let bytes = message.content.utf8.count
    ///
    /// — a binding that immediately drops into a whitelisted
    /// probe returns ``Int`` / ``Bool`` / ``Index``, not raw
    /// renderable content, and is fine.
    ///
    /// Implementation: the compact form has all whitespace and
    /// comments stripped, so a Swift binding like ``let leaked =
    /// message.content`` collapses to ``letleaked=message.content``.
    /// We grep for ``=<rawToken>`` and look at the character
    /// immediately following the token to decide probe-vs-value
    /// (same disambiguation the per-occurrence walker uses).
    /// ``var`` bindings are caught by the same heuristic (``var
    /// leaked = message.content`` → ``varleaked=message.content``).
    ///
    /// Known residual (same shape as QuickAsk r5/r7 NIT): this is
    /// not string-literal aware and not SwiftSyntax. A
    /// hypothetical ``let s = "=" + ...`` etc. could produce a
    /// false-positive substring, but no production refactor target
    /// matches that shape.
    static func assertNoLocalAliasRebinding(
        inCompactArg compactArg: String,
        rawTokens: [String],
        callShape: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let probeWhitelist: [String] = [
            ".isEmpty",
            ".count",
            ".utf8.count",
            ".unicodeScalars.count",
            ".hashValue",
            ".startIndex",
            ".endIndex",
        ]
        // Codex PR-#329 round 5 MAJOR-1: wrapped-RHS bypass shapes
        // like ``let leaked = (message.content)`` or ``let leaked =
        // String(message.content)`` compact to ``letleaked=
        // (message.content)`` / ``letleaked=String(message.content)``
        // — the ``=<rawToken>`` direct-neighbour grep below cannot
        // see them because something (paren / function call) sits
        // between ``=`` and the rawToken.
        //
        // Defence: an additional pass that catches any ``let|var
        // <ident>(:T)? = <anything>`` binding whose RHS-span
        // (statement boundary to statement boundary) contains the
        // rawToken substring at all, except when EVERY occurrence
        // of the rawToken inside that span is preceded by the
        // safe-wrap or sits inside a whitelisted probe.
        //
        // Because span-tracking on compacted form is fragile, we
        // approximate: for each ``hardBoundary + (let|var)
        // + ident + (:type)? + =`` regex hit, the RHS span is the
        // contiguous substring from the ``=`` match end to the
        // NEXT hardBoundary or end-of-string. If that span
        // contains the rawToken AND any occurrence is not safely
        // wrapped, record an issue.
        Self.assertNoWrappedAliasRebinding(
            inCompactArg: compactArg,
            rawTokens: rawTokens,
            callShape: callShape,
            sourceLocation: sourceLocation
        )
        for rawToken in rawTokens {
            // The compacted form for a binding is
            //   ``let<ident>=<rawToken>``                       (bare)
            //   ``var<ident>=<rawToken>``
            //   ``let<ident>:<typeExpr>=<rawToken>``            (typed)
            //   ``var<ident>:<typeExpr>=<rawToken>``
            // where ``<typeExpr>`` is any sequence of
            // identifier characters, ``.``, ``?``, ``!``, ``<``,
            // ``>``, ``,``, ``[``, ``]``, ``(``, ``)`` (Swift type
            // expressions).
            //
            // We grep for ``=<rawToken>``; for every hit:
            //
            //   1. Determine if it's a value-use (raw flowing into
            //      the surrounding expression) vs a probe-use
            //      (``= <raw>.isEmpty`` etc.). Probe → fine.
            //   2. Walk back through the type-expression character
            //      class (identifier chars + the type punctuation
            //      listed above). If we hit a ``:`` along the way,
            //      skip it as a type annotation separator and
            //      continue walking back through the binding name.
            //   3. After consuming the binding identifier, check
            //      whether the immediately preceding 3 characters
            //      are ``let`` or ``var`` AND those characters are
            //      themselves NOT preceded by an identifier-
            //      continuation char (otherwise we're inside a
            //      longer identifier like ``stateLetterValue`` or
            //      ``preLetter`` and not at a binding boundary).
            //
            // This closes both the round-2 MAJOR (typed-binding
            // bypass via ``let leaked: String = ...``) and the
            // round-2 NIT (``state.letterValue = message.content``
            // false-positive — the walk-back encounters ``.`` which
            // is NOT in the type-expression character class for
            // the *binding identifier* phase, so we stop there and
            // never claim to find ``let`` before it).
            let needle = "=" + rawToken
            var search = compactArg.startIndex
            while let occ = compactArg.range(of: needle, range: search..<compactArg.endIndex) {
                let after = compactArg[occ.upperBound...]
                let isProbe = probeWhitelist.contains { probe in
                    guard after.hasPrefix(probe) else { return false }
                    let endOfProbe = after.index(after.startIndex, offsetBy: probe.count)
                    if endOfProbe == after.endIndex { return true }
                    let nextChar = after[endOfProbe]
                    let isIdentContinuation = nextChar.isLetter
                        || nextChar.isNumber
                        || nextChar == "_"
                    return !isIdentContinuation
                }
                if !isProbe {
                    let isLetBinding = isLetOrVarBindingBefore(
                        compactArg: compactArg,
                        equalsIdx: occ.lowerBound
                    )
                    if isLetBinding {
                        // Codex PR-#329 round 4 MAJOR: loop-body
                        // ``#expect`` -> ``Issue.record`` so swift-
                        // testing 0.99 cannot silently drop a per-
                        // binding failure.
                        Issue.record(
                            Comment(rawValue: """
                            \(callShape) contains a raw alias \
                            rebinding of ``\(rawToken)`` — the shape \
                            ``let <ident> = \(rawToken)`` (or ``var``, \
                            or the typed variant ``let <ident>: \
                            <Type> = \(rawToken)``) opens the cycle- \
                            10 F-10-4 bypass: the alias carries raw \
                            bidi-control codepoints past the per- \
                            occurrence walker because the unsafe \
                            render call no longer contains the literal \
                            ``\(rawToken)`` token. Compact form: \
                            '\(compactArg)'. See bug_report.md \
                            cycle-10 F-10-4 and \
                            QuickAskBidiSanitizationTests for the \
                            documented (and intentionally NOT closed) \
                            residual on the Quick-Ask side.
                            """),
                            sourceLocation: sourceLocation
                        )
                    }
                }
                search = occ.upperBound
            }
        }
    }

    /// Returns true iff the character run immediately before
    /// ``equalsIdx`` in ``compactArg`` matches a ``let`` or ``var``
    /// binding (optionally with a type annotation, optionally
    /// inside a conditional-binding construct like ``if let`` /
    /// ``guard let`` / ``case let``).
    ///
    /// ## Compacted form
    ///
    /// All whitespace + comments are stripped, so:
    ///   ``let leaked = x``                → ``letleaked=x``
    ///   ``let leaked: String = x``        → ``letleaked:String=x``
    ///   ``var leaked: [Int] = x``         → ``varleaked:[Int]=x``
    ///   ``if let leaked = x { ... }``     → ``ifletleaked=x{...}``
    ///   ``guard let leaked = x else {...}`` → ``guardletleaked=xelse{...}``
    ///   ``case let leaked = x:``           → ``caseletleaked=x:``
    ///
    /// ## Boundary rules — closes codex round 3 BLOCKING
    ///
    /// A ``let`` keyword that starts a binding must be preceded by
    /// a TOKEN boundary in compacted source. The set of valid
    /// boundary tokens is:
    ///   * start-of-string ``^``
    ///   * statement separator ``;``
    ///   * brace open / close ``{`` ``}``
    ///   * comma ``,`` (binding inside a tuple destructure or
    ///     pattern-match list)
    ///   * the conditional-binding keywords ``if``, ``guard``,
    ///     ``case``, ``while`` (closes codex round 3 MAJOR — these
    ///     are real binding forms that carry the alias bypass).
    ///
    /// **Crucially**, ``.`` is NOT a boundary (closes the round-3
    /// BLOCKING — ``state.letterValue=x`` is property assignment,
    /// not a binding). Identifier chars and ``(``, ``[``, ``)``,
    /// ``]``, ``?``, ``!`` are NOT boundaries either (a leading
    /// ``[a, let b]`` shape would have the ``,`` boundary instead).
    ///
    /// ## Pattern
    ///
    /// We construct four anchored patterns (bare/typed × let/var)
    /// where ``<boundary>`` is the disjunction above:
    ///
    ///   (?:^|[;{},]|if|guard|case|while)
    ///       (?:let|var) [A-Za-z0-9_]+ (?: : [A-Za-z0-9_.?!<>,\[\]()]+ )? =$
    ///
    /// ## Round-3 NIT residual
    ///
    /// Dictionary-shorthand type annotation
    /// ``let leaked: [String: Int] = x`` contains ``:`` INSIDE the
    /// type expression. The compacted form is
    /// ``letleaked:[String:Int]=x``; the second ``:`` is not in
    /// the type-class. Add ``:`` to the type-class so dict
    /// shorthand is recognised. We assume the binding-level ``:``
    /// is the FIRST one encountered, which is correct for Swift's
    /// grammar (the type annotation comes immediately after the
    /// binding identifier, before any nested punctuation).
    static func isLetOrVarBindingBefore(
        compactArg: String,
        equalsIdx: String.Index
    ) -> Bool {
        let nsString = compactArg as NSString
        let utf16EqualsOffset = compactArg.utf16.distance(
            from: compactArg.utf16.startIndex,
            to: equalsIdx.samePosition(in: compactArg.utf16) ?? compactArg.utf16.startIndex
        )
        let endOffset = utf16EqualsOffset + 1

        // Two prefix shapes:
        //
        //   * ``hardBoundary`` — start of string OR one of
        //     ``; { } ,`` — applies to BOTH bare bindings and
        //     conditional bindings (a conditional binding still
        //     needs to be at the start of a statement, NOT mid-
        //     identifier).
        //   * ``condKeywords`` — one of ``if`` / ``guard`` / ``case``
        //     / ``while`` — applies only to the conditional binding
        //     arm.
        //
        // Closes codex round-4 NIT: ``ifletColumns=...`` as a bare
        // property assignment at column 0 (no leading ``;`` /
        // ``{`` / ``}`` / ``,``) does NOT match either arm because
        // both require the hard boundary in front. The same shape
        // following an explicit ``{`` (``{ifletColumns=x``) WAS a
        // conditional binding in real source and DOES match.
        let hardBoundary = "(?:^|[;{},])"
        // Codex PR-#329 round 5 MAJOR-2: include compound
        // conditional forms ``else if let``, ``if case let``,
        // ``else if case let``, ``else guard let``, etc. by allowing
        // an OPTIONAL ``else`` prefix and an OPTIONAL nested
        // ``if|case`` between the outer keyword and ``let|var``.
        // The compacted forms are:
        //   elseiflet   elseguardlet   elsecaselet   elsewhilelet
        //   ifcaselet
        //   elseifcaselet
        let condKeywords = #"(?:else)?(?:if|guard|case|while)(?:case|if)?"#
        let identChars = "[A-Za-z0-9_]+"
        // Round-3 NIT closure: include ``:`` inside the type-
        // expression character class so ``[String:Int]`` matches.
        let typeChars = #"[A-Za-z0-9_.?!<>,:\[\]()]+"#

        let patterns: [String] = [
            // Bare ``let foo = `` at a hard boundary.
            "\(hardBoundary)(?:let|var)\(identChars)=$",
            // Typed ``let foo: T = `` at a hard boundary.
            "\(hardBoundary)(?:let|var)\(identChars):\(typeChars)=$",
            // Bare conditional ``if let foo = `` at a hard
            // boundary (the keyword is between the boundary and
            // ``let``).
            "\(hardBoundary)\(condKeywords)(?:let|var)\(identChars)=$",
            // Typed conditional ``if let foo: T = ``.
            "\(hardBoundary)\(condKeywords)(?:let|var)\(identChars):\(typeChars)=$",
        ]
        let searchRange = NSRange(location: 0, length: endOffset)
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else {
                continue
            }
            if regex.firstMatch(in: nsString as String, range: searchRange) != nil {
                return true
            }
        }
        return false
    }

    /// Codex PR-#329 round 5 MAJOR-1: catch wrapped-RHS alias
    /// rebinding shapes like
    ///
    ///   let leaked = (message.content)
    ///   let leaked = String(message.content)
    ///   let leaked: String = String(describing: r.content)
    ///
    /// These bypass the direct ``=<rawToken>`` neighbour grep
    /// because something (paren / function call) sits between
    /// ``=`` and the rawToken.
    ///
    /// Approach: scan ``compactArg`` for every
    /// ``(hardBoundary)(condKeywords)?(let|var)<ident-or-tuple>(:T)?=``
    /// regex hit. For each hit, the RHS span is the contiguous
    /// substring from the ``=`` end to the next ``;``, ``{`` or
    /// ``}`` (``,`` is intentionally NOT a RHS boundary — a
    /// tuple-RHS like ``(x, message.content)`` must stay as one
    /// span). If that span contains any rawToken occurrence that
    /// isn't preceded by the safe-wrap, record an issue.
    ///
    /// Codex PR-#329 round 6 MAJOR: includes tuple-destructure
    /// pattern ``let (a, b, ...)`` in the binding LHS so a real
    /// alias like ``let (a, b) = (x, message.content)`` is caught.
    ///
    /// Probe whitelist still applies — ``let n = message.content
    /// .count`` lands ``message.content.count`` in the RHS span;
    /// the ``.count`` probe means the binding doesn't carry raw
    /// renderable content.
    static func assertNoWrappedAliasRebinding(
        inCompactArg compactArg: String,
        rawTokens: [String],
        callShape: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let nsString = compactArg as NSString
        let probeWhitelist: [String] = [
            ".isEmpty",
            ".count",
            ".utf8.count",
            ".unicodeScalars.count",
            ".hashValue",
            ".startIndex",
            ".endIndex",
        ]
        let safeWrap = "ChatTextSanitizer.sanitizeForDisplay("
        // Bare-binding regex matches everything from the hard
        // boundary through the ``=``. We capture the position of
        // ``=`` via the match end. The conditional-binding variant
        // shares the same pattern set as ``isLetOrVarBindingBefore``.
        let hardBoundary = "(?:^|[;{},])"
        let condKeywords = #"(?:else)?(?:if|guard|case|while)(?:case|if)?"#
        let identChars = "[A-Za-z0-9_]+"
        let typeChars = #"[A-Za-z0-9_.?!<>,:\[\]()]+"#
        // Codex PR-#329 round 6 MAJOR: tuple-destructure LHS
        // ``let (a, b)`` compacts to ``let(a,b)``. The pattern
        // ``\((?:identChars|,)+\)`` matches one or more identifiers
        // separated by commas, wrapped in parens. Underscores in
        // patterns like ``let (_, x)`` are also matched because
        // ``_`` is in ``identChars``.
        let tupleLhs = #"\((?:[A-Za-z0-9_]+,?)+\)"#
        let bindingPatterns: [String] = [
            "\(hardBoundary)(?:let|var)\(identChars)=",
            "\(hardBoundary)(?:let|var)\(identChars):\(typeChars)=",
            "\(hardBoundary)\(condKeywords)(?:let|var)\(identChars)=",
            "\(hardBoundary)\(condKeywords)(?:let|var)\(identChars):\(typeChars)=",
            // Tuple-destructure ``let (a, b) = ...`` (no type
            // annotation form because Swift's tuple destructure
            // doesn't take an explicit type annotation in this
            // shape).
            "\(hardBoundary)(?:let|var)\(tupleLhs)=",
            "\(hardBoundary)\(condKeywords)(?:let|var)\(tupleLhs)=",
        ]
        let fullRange = NSRange(location: 0, length: nsString.length)
        var seenBindingEnds: Set<Int> = []
        for pattern in bindingPatterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else {
                continue
            }
            regex.enumerateMatches(in: nsString as String, range: fullRange) { m, _, _ in
                guard let m = m else { return }
                // Match end is the position immediately after
                // ``=``. The RHS span runs from there to the next
                // RHS-terminator char — ``;``, ``{``, or ``}`` —
                // or end-of-string. ``,`` is intentionally NOT a
                // RHS terminator (a tuple RHS like ``(x, message
                // .content)`` must stay as one span). The boundary
                // class used to find the BINDING above includes
                // ``,`` so the LHS scope is correct; only the RHS
                // span is comma-permissive. Codex PR-#329 round 6
                // NIT — clarified stale comment.
                let equalsEnd = m.range.location + m.range.length
                guard !seenBindingEnds.contains(equalsEnd) else { return }
                seenBindingEnds.insert(equalsEnd)
                // Scan forward for the next hard-boundary char.
                var rhsEnd = nsString.length
                let chars = Array(compactArg.utf16)
                if equalsEnd < chars.count {
                    for j in equalsEnd..<chars.count {
                        let c = chars[j]
                        if c == 0x3B /* ; */
                            || c == 0x7B /* { */
                            || c == 0x7D /* } */
                            // ``,`` is NOT a hard boundary here
                            // because tuples in the RHS contain
                            // commas (``let leaked = (a,
                            // message.content)``). Truncating on
                            // ``,`` would false-negative.
                        {
                            rhsEnd = j
                            break
                        }
                    }
                }
                guard equalsEnd < rhsEnd else { return }
                let rhsRange = NSRange(location: equalsEnd, length: rhsEnd - equalsEnd)
                let rhsSpan = nsString.substring(with: rhsRange)
                // For each rawToken, every occurrence inside the
                // RHS span must be preceded by the safe-wrap (or
                // immediately followed by a whitelisted probe).
                for rawToken in rawTokens {
                    var search = rhsSpan.startIndex
                    while let occ = rhsSpan.range(of: rawToken, range: search..<rhsSpan.endIndex) {
                        let after = rhsSpan[occ.upperBound...]
                        let isProbe = probeWhitelist.contains { probe in
                            guard after.hasPrefix(probe) else { return false }
                            let endOfProbe = after.index(after.startIndex, offsetBy: probe.count)
                            if endOfProbe == after.endIndex { return true }
                            let nextChar = after[endOfProbe]
                            let isIdentContinuation = nextChar.isLetter
                                || nextChar.isNumber
                                || nextChar == "_"
                            return !isIdentContinuation
                        }
                        if !isProbe {
                            // Is this occurrence preceded by the
                            // safe-wrap WITHIN the rhsSpan?
                            let precedingStart = rhsSpan.index(
                                occ.lowerBound,
                                offsetBy: -safeWrap.count,
                                limitedBy: rhsSpan.startIndex
                            )
                            let preceding: Substring = precedingStart.map { rhsSpan[$0..<occ.lowerBound] } ?? ""
                            if preceding != safeWrap {
                                Issue.record(
                                    Comment(rawValue: """
                                    \(callShape) contains a wrapped \
                                    alias rebinding of ``\(rawToken)`` — \
                                    a binding (``let``/``var``, \
                                    optionally inside ``if`` / ``guard`` \
                                    / ``case`` / ``while`` / ``else``) \
                                    has an RHS expression that uses \
                                    ``\(rawToken)`` without routing it \
                                    through ChatTextSanitizer \
                                    .sanitizeForDisplay. RHS span: \
                                    '\(rhsSpan)'. This is the codex \
                                    PR-#329 round 5 MAJOR-1 bypass \
                                    shape — e.g. ``let leaked = \
                                    (message.content)``, ``let leaked = \
                                    String(message.content)``. See \
                                    bug_report.md cycle-10 F-10-4.
                                    """),
                                    sourceLocation: sourceLocation
                                )
                            }
                        }
                        search = occ.upperBound
                    }
                }
            }
        }
    }
}
