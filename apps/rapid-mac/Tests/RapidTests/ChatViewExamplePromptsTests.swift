import Foundation
import Testing
@testable import Rapid

/// The ``ChatView`` empty-state ``examplePrompts`` helper must
/// surface ONLY pure-text prompts. The seeded example prompts are
/// model-agnostic by design: they must read well on ANY active model
/// — including a brand-new user's starter — and never tease a
/// capability tied to one specific alias. The starter is now
/// ``lfm2.5-1b-4bit`` (2026-08-05), a text-first pick that is NOT in
/// ``ToolUseCapability.known`` — so the empty-state capability chip row
/// stays hidden for it. The seeded prompts stay generic regardless, so
/// they hold up whichever way the starter goes. The chip row's tool-bias
/// hider (PR #333 / FU-9) covers the chip surface; these tests cover
/// the seeded example prompts the chip row gate does not touch.
///
/// Constraints:
///   * ``examplePrompts`` is a ``private`` computed property — the
///     contract is pinned via source-grep against the rendered
///     literal array. The grep is brittle on purpose: a partial
///     swap that re-introduces "weather" / "calculator" / "% of"
///     should fail loudly here, BEFORE the user sees an empty
///     assistant bubble on the first interactive surface.
///   * Each prompt must be ≤55 chars to fit ``.lineLimit(1)`` in
///     the ~470pt empty-state column on a 13" Mac.
///   * The doc comment above ``examplePrompts`` must state the
///     "Three pure-text prompts" intent + name the Quickstart
///     alias so a future refactor reads the rationale before
///     re-introducing a tool-using probe.
@MainActor
@Suite("ChatView.examplePrompts — F-LWT-1 pure-text invariant")
struct ChatViewExamplePromptsTests {

    private static func chatViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Extract the three string literals between the `examplePrompts:
    /// [String]` opening bracket and its closing `]`. The helper is
    /// scoped to the empty-state computed property (the FIRST
    /// `examplePrompts` declaration around line 1383), not the
    /// later one on the System-prompt picker (around line 4036)
    /// which is a different surface entirely.
    private static func extractEmptyStateExamplePrompts() throws -> [String] {
        let src = try chatViewSource()
        // Anchor on the docstring intent so we don't pick up the
        // later ``examplePrompts`` declaration on the System-prompt
        // picker (around line 4036) — different surface, different
        // contract.
        guard let intentRange = src.range(of: "Three pure-text prompts") else {
            Issue.record("ChatView.swift no longer carries the 'Three pure-text prompts' intent docstring — F-LWT-1 partial-swap?")
            return []
        }
        let after = src[intentRange.upperBound...]
        guard let openBracket = after.firstIndex(of: "[") else {
            Issue.record("Could not locate examplePrompts array opening bracket")
            return []
        }
        guard let closeBracket = after.range(of: "]", range: openBracket..<after.endIndex) else {
            Issue.record("Could not locate examplePrompts array closing bracket")
            return []
        }
        let arrayBody = after[after.index(after: openBracket)..<closeBracket.lowerBound]
        // Pull out every double-quoted literal. The Swift source
        // doesn't use escaped double-quotes inside these prompts so
        // a simple `\"...\"` scan is sufficient.
        var prompts: [String] = []
        var current = ""
        var inLiteral = false
        for ch in arrayBody {
            if ch == "\"" {
                if inLiteral {
                    prompts.append(current)
                    current = ""
                }
                inLiteral.toggle()
            } else if inLiteral {
                current.append(ch)
            }
        }
        return prompts
    }

    @Test("Empty-state examplePrompts surface exactly three prompts")
    func threePromptsExactly() throws {
        let prompts = try Self.extractEmptyStateExamplePrompts()
        #expect(prompts.count == 3,
                "Expected three pure-text prompts, got \(prompts.count): \(prompts)")
    }

    @Test("No prompt mentions a tool-bias trigger word (calculator / weather / % / sqrt / divided)")
    func noToolBiasPrompts() throws {
        let prompts = try Self.extractEmptyStateExamplePrompts()
        // Regex match — case-insensitive, word-boundary on the
        // alphabetic tokens (so "percent" inside a longer word like
        // "percentage" still fires) but the symbols ``%`` and the
        // arithmetic words match standalone.
        let pattern = #"(?i)(percent|sqrt|square root|calculator|weather|forecast|temperature in|plus|minus|times|divided|%)"#
        let regex = try Regex(pattern)
        for prompt in prompts {
            let hit = prompt.firstMatch(of: regex)
            #expect(hit == nil,
                    "Prompt '\(prompt)' contains tool-bias trigger '\(hit?.0 ?? "")'; the seeded empty-state examples stay pure-text so they never demo a tool call the active model might flub (model-agnostic first impression).")
        }
    }

    @Test("Each prompt fits .lineLimit(1) in the ~470pt column (≤55 chars)")
    func eachPromptUnder55Chars() throws {
        let prompts = try Self.extractEmptyStateExamplePrompts()
        for prompt in prompts {
            #expect(prompt.count <= 55,
                    "Prompt '\(prompt)' is \(prompt.count) chars — would wrap or truncate in the empty-state column.")
        }
    }

    @Test("Each prompt is non-empty")
    func eachPromptNonEmpty() throws {
        let prompts = try Self.extractEmptyStateExamplePrompts()
        for prompt in prompts {
            #expect(!prompt.isEmpty)
        }
    }

    @Test("Docstring above examplePrompts names the Quickstart alias for traceability")
    func docstringNamesQuickstartAlias() throws {
        let src = try Self.chatViewSource()
        // The docstring should mention the Quickstart alias so a
        // future refactor reads the receipt before re-introducing a
        // tool-using probe. Scoped to the F-LWT-1 docstring block
        // (the first occurrence near the 1383-line empty-state
        // computed property — the later examplePrompts on the
        // System-prompt picker is a different surface).
        guard let intentRange = src.range(of: "Three pure-text prompts") else {
            Issue.record("Docstring intent 'Three pure-text prompts' missing from ChatView.swift")
            return
        }
        // Look for the alias in the 600-char neighbourhood AFTER
        // the intent line.
        let scanStart = intentRange.lowerBound
        let scanEnd = src.index(scanStart, offsetBy: 600, limitedBy: src.endIndex) ?? src.endIndex
        let neighbourhood = src[scanStart..<scanEnd]
        #expect(neighbourhood.contains("lfm2.5-1b-4bit"),
                "Docstring above ChatView.examplePrompts must name the Quickstart alias 'lfm2.5-1b-4bit' so the rationale is one source-grep away.")
    }

    @Test("Docstring above examplePrompts mentions trade-up to Recommended")
    func docstringMentionsRecommendedTradeUp() throws {
        let src = try Self.chatViewSource()
        guard let intentRange = src.range(of: "Three pure-text prompts") else {
            Issue.record("Docstring intent missing")
            return
        }
        let scanStart = intentRange.lowerBound
        let scanEnd = src.index(scanStart, offsetBy: 800, limitedBy: src.endIndex) ?? src.endIndex
        let neighbourhood = src[scanStart..<scanEnd]
        #expect(neighbourhood.lowercased().contains("recommended") ||
                neighbourhood.lowercased().contains("trade up") ||
                neighbourhood.lowercased().contains("trading up"),
                "Docstring above ChatView.examplePrompts must point users at the trade-up path to Recommended Default for the tool-calling demo.")
    }
}
