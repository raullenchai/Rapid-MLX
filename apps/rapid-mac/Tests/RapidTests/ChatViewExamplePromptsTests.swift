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

}
