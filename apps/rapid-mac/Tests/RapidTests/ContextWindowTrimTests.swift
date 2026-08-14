import Foundation
import Testing
@testable import Rapid

/// v0.5.11 regression pin for ``ChatViewModel.trimMessagesForContextWindow``.
///
/// Replaces the v0.4.19 cumulative-token meter chip. ChatGPT / Claude
/// desktops never show users a token meter — they drop oldest turns
/// behind the scenes when the conversation would exceed the model's
/// context window. rapid-mlx's server doesn't enforce a window either
/// (mlx-lm RoPE-extrapolates past training context and degrades
/// quality silently), so the client takes responsibility.
///
/// Contract pinned by this suite:
///   * ``contextWindow == nil`` or estimated total ≤ budget → no-op.
///   * Otherwise: walk newest-to-oldest, keep what fits under
///     ``keepFraction * contextWindow`` (default 0.75); always keep
///     the most recent message even if it overshoots alone; drop
///     leading non-user rows after cut so the kept tail never starts
///     mid-tool-chain; preserve a leading system row if present.
@MainActor
@Suite("v0.5.11 silent context-window trim")
struct ContextWindowTrimTests {

    // MARK: - No-op cases

    @Test("Nil context window returns the input unchanged")
    func nilContextWindowNoOp() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: String(repeating: "x", count: 1_000_000)),
            ChatMessage(role: .assistant, content: String(repeating: "y", count: 1_000_000)),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: nil
        )
        #expect(trimmed.count == history.count)
    }

    @Test("Empty input returns empty")
    func emptyInputNoOp() {
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            [], contextWindow: 32_768
        )
        #expect(trimmed.isEmpty)
    }

    @Test("Total under budget returns the input unchanged")
    func underBudgetNoOp() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: "hello"),
            ChatMessage(role: .user, content: "how are you?"),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 32_768
        )
        #expect(trimmed == history)
    }

    @Test("Locally extracted document text participates in the context budget")
    func documentTextCountsTowardBudget() throws {
        let attachment = try ChatFileAttachment(
            filename: "large.csv",
            kind: .csv,
            extractedText: String(repeating: "x", count: 8_000),
            sourceByteCount: 8_000
        )
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "old question"),
            ChatMessage(role: .assistant, content: "old answer"),
            ChatMessage(role: .user, content: "analyze", fileAttachments: [attachment]),
        ]

        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history,
            contextWindow: 1_024
        )
        #expect(trimmed.count == 1)
        #expect(trimmed.first?.fileAttachments == [attachment])
    }

    // MARK: - Trimming behaviour

    @Test("Oldest turns are dropped when total exceeds the keep fraction")
    func dropsOldestTurnsOverBudget() {
        // chars/4 estimate. Budget = 4 * 0.75 = 3 tokens. Each msg is
        // 4 chars = 1 token. Last user msg must always survive.
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "AAAA"),       // 1 tok — drops
            ChatMessage(role: .assistant, content: "BBBB"),  // 1 tok — drops
            ChatMessage(role: .user, content: "CCCC"),       // 1 tok — keeps
            ChatMessage(role: .assistant, content: "DDDD"),  // 1 tok — keeps
            ChatMessage(role: .user, content: "EEEE"),       // 1 tok — keeps (always)
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.count == 3)
        #expect(trimmed.first?.content == "CCCC")
        #expect(trimmed.last?.content == "EEEE")
    }

    @Test("Most recent user turn is always kept even if it alone overshoots the budget")
    func keepsLastUserTurnEvenWhenOversized() {
        // 1-token budget, last user msg is huge — must still be kept.
        let huge = String(repeating: "x", count: 40_000)
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "old"),
            ChatMessage(role: .assistant, content: "older"),
            ChatMessage(role: .user, content: huge),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.count == 1)
        #expect(trimmed.first?.content == huge)
    }

    @Test("System message is preserved at index 0 across a trim")
    func preservesSystemMessage() {
        let sys = ChatMessage(role: .system, content: "be terse")
        let history: [ChatMessage] = [
            sys,
            ChatMessage(role: .user, content: "AAAA"),
            ChatMessage(role: .assistant, content: "BBBB"),
            ChatMessage(role: .user, content: "CCCC"),
            ChatMessage(role: .assistant, content: "DDDD"),
            ChatMessage(role: .user, content: "EEEE"),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.first?.role == .system)
        #expect(trimmed.first?.content == "be terse")
        #expect(trimmed.last?.content == "EEEE")
    }

    // MARK: - Tool-chain integrity

    @Test("Trim never starts the wire body with a leading assistant or tool row")
    func dropsLeadingNonUserRowsAfterCut() {
        // A previous turn's tool round-trip sits mid-window. If the
        // budget cuts mid-chain, the kept tail must advance to the
        // next user boundary so the wire body never starts with a
        // bare ``tool`` or ``assistant(tool_calls)`` row.
        var toolAsst = ChatMessage(role: .assistant, content: "")
        toolAsst.toolCalls = [
            ToolCall(id: "c1", name: "calc", arguments: "{}")
        ]
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "AAAA"),       // drops
            toolAsst,                                         // drops (orphan)
            ChatMessage(role: .tool, content: "BBBB"),       // drops (orphan)
            ChatMessage(role: .user, content: "CCCC"),       // keeps
            ChatMessage(role: .assistant, content: "DDDD"),  // keeps
            ChatMessage(role: .user, content: "EEEE"),       // keeps
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.first?.role == .user)
        #expect(!trimmed.contains { $0.role == .tool })
        #expect(!trimmed.contains { ($0.toolCalls?.isEmpty == false) })
    }

    @Test("All-assistant prefix collapses to just the last user message")
    func collapsesToLastUserWhenAllNonUserBeforeIt() {
        // Pathological shape: massive assistant/tool turns ahead of
        // the latest user. After dropping non-user leading rows the
        // safety net must still leave at least the latest user msg.
        let history: [ChatMessage] = [
            ChatMessage(role: .assistant, content: String(repeating: "x", count: 4_000)),
            ChatMessage(role: .tool, content: String(repeating: "y", count: 4_000)),
            ChatMessage(role: .user, content: "ping"),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.count == 1)
        #expect(trimmed.first?.role == .user)
        #expect(trimmed.first?.content == "ping")
    }
}
