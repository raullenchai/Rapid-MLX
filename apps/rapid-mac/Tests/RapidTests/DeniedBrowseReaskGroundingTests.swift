import Foundation
import Testing
@testable import Rapid

/// Regression lock for the denied-browse → re-ask grounding path.
///
/// The field report: after tapping "Don't allow" on a browse approval, asking
/// the SAME news question again ran `web_search` but the assistant ignored its
/// own result and repeated a canned "I can't access real-time information"
/// line. The native tool loop now owns routing. Once it calls a tool,
/// ``carriesToolResultForThisTurn`` must arm the anti-confabulation preamble
/// only for a real result from this turn — not the stale declined row.
@Suite("Denied-browse re-ask grounding")
@MainActor
struct DeniedBrowseReaskGroundingTests {
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
