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
///   * The mandatory tail's ROWS always survive, but an oversized tool
///     result inside it has its BODY elided — see
///     ``ChatViewModel/elidingOldestToolResults(_:within:cost:)``.
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
        // Each 4-char ASCII message costs ceil(4 * 0.42) = 2 estimated tokens,
        // so five messages cost 10. A window of 8 gives a budget of
        // 8 * 0.75 = 6 tokens = three messages. The last user turn always
        // survives, so the two oldest drop.
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "AAAA"),       // 2 tok — drops
            ChatMessage(role: .assistant, content: "BBBB"),  // 2 tok — drops
            ChatMessage(role: .user, content: "CCCC"),       // 2 tok — keeps
            ChatMessage(role: .assistant, content: "DDDD"),  // 2 tok — keeps
            ChatMessage(role: .user, content: "EEEE"),       // 2 tok — keeps (always)
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 8
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

    @Test("An oversized current tool round keeps its complete user-anchored chain")
    func oversizedToolResultKeepsCurrentTurnIntact() {
        var toolCall = ChatMessage(role: .assistant)
        toolCall.toolCalls = [
            ToolCall(id: "read-1", name: "read_document", arguments: "{\"offset\":0}")
        ]
        let toolResult = ChatMessage(
            role: .tool,
            content: String(repeating: "document page content ", count: 1_000),
            toolCallID: "read-1"
        )
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "old question"),
            ChatMessage(role: .assistant, content: "old answer"),
            ChatMessage(role: .user, content: "summarize the attached report"),
            toolCall,
            toolResult,
        ]

        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history,
            contextWindow: 8_000
        )

        // The chain the current turn depends on survives as a unit: neither
        // half of an assistant(tool_calls) → tool pair is valid alone.
        let chain = trimmed.suffix(3)
        #expect(chain.first?.content == "summarize the attached report")
        #expect(chain.dropFirst().first?.toolCalls?.first?.id == "read-1")
        #expect(chain.last?.role == .tool)
        #expect(chain.last?.toolCallID == "read-1")
        // …but it does not get to overshoot the window. This result is ~9.2k
        // estimated tokens against a 6k budget, so its body is elided.
        #expect(chain.last?.content == ChatViewModel.elidedToolResultBody)
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

    // MARK: - Bounding the mandatory tail
    //
    // The tail anchored at the latest user row is kept whole so a tool chain
    // is never split — but "kept whole" cannot mean "unbounded". One
    // ``read_document`` slice is ~6.3k tokens of ASCII and the loop allows
    // twelve, so a document read can pile 75k+ tokens into a single turn: an
    // 8k model overflows on the first read, and a 32k one well before the
    // budget is spent. Elision returns those tokens without breaking the
    // assistant(tool_calls) → tool pairing the wire body needs.

    @Test("A single oversized tool result is elided rather than shipped whole")
    func oversizedToolTailIsElided() {
        var toolCall = ChatMessage(role: .assistant)
        toolCall.toolCalls = [
            ToolCall(id: "read-1", name: "read_document", arguments: "{\"offset\":0}")
        ]
        // ~6.3k tokens — one full read_document slice, against an 8k window.
        let slice = String(repeating: "document page content ", count: 700)
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "summarize the attached report"),
            toolCall,
            ChatMessage(role: .tool, content: slice, toolCallID: "read-1"),
        ]

        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history,
            contextWindow: 8_192
        )

        // Every row survives: dropping either half of the chain 400s most
        // chat templates.
        #expect(trimmed.count == 3)
        #expect(trimmed[1].toolCalls?.first?.id == "read-1")
        #expect(trimmed[2].toolCallID == "read-1")
        #expect(trimmed[2].content == ChatViewModel.elidedToolResultBody)
        let total = trimmed.reduce(0) { $0 + TokenEstimate.tokens(in: $1.modelContent) }
        #expect(total <= Int(8_192 * 0.75))
    }

    @Test("Elision starts with the OLDEST reads and stops once the tail fits")
    func elisionIsOldestFirstAndStopsEarly() {
        // The newest result is what the model is about to reason over; the
        // older ones have usually been summarised into the prose after them.
        func call(_ id: String) -> ChatMessage {
            var message = ChatMessage(role: .assistant)
            message.toolCalls = [ToolCall(id: id, name: "read_document", arguments: "{}")]
            return message
        }
        let slice = String(repeating: "page ", count: 2_000)   // ~4.2k tokens
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "read the whole thing"),
            call("r1"),
            ChatMessage(role: .tool, content: slice + "FIRST", toolCallID: "r1"),
            call("r2"),
            ChatMessage(role: .tool, content: slice + "SECOND", toolCallID: "r2"),
            call("r3"),
            ChatMessage(role: .tool, content: slice + "THIRD", toolCallID: "r3"),
        ]

        // Budget fits roughly two slices, so exactly one must go.
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history,
            contextWindow: 12_500
        )

        #expect(trimmed.count == history.count)
        #expect(trimmed[2].content == ChatViewModel.elidedToolResultBody)
        #expect(trimmed[4].content.hasSuffix("SECOND"))
        #expect(trimmed[6].content.hasSuffix("THIRD"))
    }

    @Test("An elided result tells the model the evidence is gone, not empty")
    func elisionNoticeIsNotAnEmptyResult() {
        // A bare "" reads as "the tool found nothing" — the exact
        // confabulation the ambient tool preamble exists to prevent.
        let body = ChatViewModel.elidedToolResultBody
        #expect(!body.isEmpty)
        #expect(body.localizedCaseInsensitiveContains("context window"))
        #expect(body.localizedCaseInsensitiveContains("call the tool again"))
    }

    @Test("The user's own question is never elided to make room")
    func userTurnSurvivesElision() {
        var toolCall = ChatMessage(role: .assistant)
        toolCall.toolCalls = [ToolCall(id: "r1", name: "read_document", arguments: "{}")]
        let question = "what does section 4 say about indemnity?"
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: question),
            toolCall,
            ChatMessage(role: .tool, content: String(repeating: "x", count: 60_000), toolCallID: "r1"),
        ]

        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4_096
        )

        #expect(trimmed.first?.content == question)
    }

    @Test("A tail that already fits is returned untouched")
    func fittingTailIsNotElided() {
        var toolCall = ChatMessage(role: .assistant)
        toolCall.toolCalls = [ToolCall(id: "r1", name: "web_search", arguments: "{}")]
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: String(repeating: "old ", count: 4_000)),
            ChatMessage(role: .assistant, content: String(repeating: "older ", count: 4_000)),
            ChatMessage(role: .user, content: "what is the weather?"),
            toolCall,
            ChatMessage(role: .tool, content: "sunny, 21C", toolCallID: "r1"),
        ]

        // Over budget overall — the OLD turns are what must go, not the
        // current turn's evidence.
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4_096
        )

        #expect(trimmed.count == 3)
        #expect(trimmed.last?.content == "sunny, 21C")
    }
}
