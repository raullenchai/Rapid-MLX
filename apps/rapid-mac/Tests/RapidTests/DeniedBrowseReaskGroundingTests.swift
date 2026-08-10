import Foundation
import Testing
@testable import Rapid

/// Regression lock for the denied-browse → re-ask grounding path.
///
/// The field report: after tapping "Don't allow" on a browse approval, asking
/// the SAME news question again ran `web_search` but the assistant ignored its
/// own result and repeated a canned "I can't access real-time information"
/// line. The two deterministic, engine-agnostic mechanisms that decide whether
/// that turn stays grounded both live in ``ChatViewModel``, and each already
/// has a fix — but nothing pinned the *combination* the report actually hit
/// (a poisoned history from the declined turn feeding the re-ask):
///
///   A. ``forcedToolForUserTurn`` must still force `web_search` on the re-ask,
///      even though the prior turn left a declined `.tool` row and an
///      "I can't access" refusal in history (the #1694 fresh-routing path).
///   B. ``carriesToolResultForThisTurn`` must arm the anti-confabulation
///      preamble only once a *real* result lands — not off the stale declined
///      row, and not before the fresh search runs (the #1556 path).
///
/// These are pure functions over the wire history, so they reproduce the
/// decision the live turn makes without a server, a model, or a network.
@Suite("Denied-browse re-ask grounding")
@MainActor
struct DeniedBrowseReaskGroundingTests {
    private let enabled: Set<String> = ["web_search", "browse", "weather"]

    /// The history a denied-browse turn leaves behind: the model tried to
    /// browse, the user declined, and the assistant fell back to a canned
    /// "can't access real-time info" reply.
    private var deniedBrowseHistory: [ChatMessage] {
        [
            ChatMessage(role: .user, content: "What's the latest news about the AI industry this week?"),
            ChatMessage(role: .assistant, toolCalls: [ToolCall(id: "b0", name: "browse", arguments: "{\"url\":\"https://example.com\"}")]),
            ChatMessage(role: .tool, content: "browse error: the user did not approve browsing example.com", toolCallID: "b0"),
            ChatMessage(role: .assistant, content: "You didn't allow this, so there's nothing to show. I can't access real-time information."),
        ]
    }

    private let reaskPrompt = "What's the latest news about the AI industry this week?"

    @Test("A re-ask after a denied browse still forces web_search")
    func reaskStillForcesSearch() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            reaskPrompt,
            priorMessages: deniedBrowseHistory,
            enabledToolNames: enabled
        ) == "web_search")
    }

    @Test("The stale declined row does not arm the preamble on the bare re-ask")
    func staleDeclinedRowDoesNotArm() {
        // Last `.user` is the re-ask; the declined `.tool` row sits before it,
        // so it is not evidence for this turn and must not arm the preamble.
        let bareReask = deniedBrowseHistory + [
            ChatMessage(role: .user, content: reaskPrompt),
        ]
        #expect(ChatViewModel.carriesToolResultForThisTurn(bareReask) == false)
    }

    @Test("A real web_search result arms the anti-confabulation preamble")
    func realResultArmsPreamble() {
        let synthesisRound = deniedBrowseHistory + [
            ChatMessage(role: .user, content: reaskPrompt),
            ChatMessage(role: .assistant, toolCalls: [ToolCall(id: "w1", name: "web_search", arguments: "{}")]),
            ChatMessage(role: .tool, content: "Web search via DuckDuckGo: <real headlines here>", toolCallID: "w1"),
        ]
        #expect(ChatViewModel.carriesToolResultForThisTurn(synthesisRound) == true)
        let ambient = ChatViewModel.ambientSystemMessages(
            historyOpensWithSystem: false,
            toolsAdvertised: true,
            toolResultPresent: ChatViewModel.carriesToolResultForThisTurn(synthesisRound)
        )
        #expect(ambient.count == 1)
        #expect(ambient.first?.content == ChatViewModel.toolGuidancePreamble)
    }
}
