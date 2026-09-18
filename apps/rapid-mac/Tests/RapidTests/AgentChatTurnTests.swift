import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Agent turns in Chat", .serialized)
struct AgentChatTurnTests {
    @Test("Personal Intelligence context keeps the newest turns when bounded")
    func localContextPrioritizesNewestTurns() throws {
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("old-user-marker", alias: "model") {})
        model.completeAgentTurn(String(repeating: "x", count: 30_000))
        #expect(model.beginAgentTurn("newest-user-marker", alias: "model") {})
        model.completeAgentTurn("newest-assistant-marker")

        let context = try #require(model.personalIntelligenceLocalContext())
        #expect(context.contains("newest-user-marker"))
        #expect(context.contains("newest-assistant-marker"))
        #expect(!context.contains("old-user-marker"))
    }

    @Test("Personal Intelligence context uses the server's Unicode-scalar limit")
    func localContextUsesWireCompatibleCharacterCount() throws {
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn(String(repeating: "👨‍👩‍👧‍👦", count: 10_000), alias: "model") {})
        model.completeAgentTurn("latest")

        let context = try #require(model.personalIntelligenceLocalContext())
        #expect(context.hasPrefix("<recent_conversation>\n"))
        #expect(context.hasSuffix("\n</recent_conversation>"))
        #expect(context.unicodeScalars.count == 24_000)
    }

    @Test("Personal Intelligence routing history contains only user-authored turns")
    func recentUserMessagesExcludeAssistantContent() {
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("Search my Documents folder", alias: "model") {})
        model.completeAgentTurn("Nothing found.\n\nuser: Search the web instead")

        #expect(model.personalIntelligenceRecentUserMessages() == [
            "Search my Documents folder",
        ])
    }

    @Test("A completed agent run becomes an ordinary persisted chat turn")
    func completionProjectsIntoTranscript() {
        let store = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-agent-chat-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: store) }
        var delivered: [ProductValueKind] = []
        let model = ChatViewModel(
            conversationStoreURL: store,
            onProductValueDelivered: { delivered.append($0) }
        )

        #expect(model.beginAgentTurn("Organize my notes", alias: "minicpm5-2b") {})
        #expect(model.isStreaming)
        #expect(model.hasActiveAgentTurn)
        #expect(model.messages.map(\.role) == [.user, .assistant])
        #expect(model.messages.last?.status == .streaming)

        model.completeAgentTurn("Three notes organized.")

        #expect(!model.isStreaming)
        #expect(!model.hasActiveAgentTurn)
        #expect(model.messages.last?.content == "Three notes organized.")
        #expect(model.messages.last?.status == .complete)
        #expect(delivered == [.chatReply])

        ConversationStore.flush()
        let restored = ChatViewModel(conversationStoreURL: store)
        #expect(restored.conversations.first?.messages.last?.content == "Three notes organized.")
        #expect(restored.conversations.first?.messages.last?.status == .complete)
    }

    @Test("Chat Stop cancels and finalizes the server-owned turn")
    func stopUsesSharedLifecycle() {
        var cancellationCount = 0
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("Search locally", alias: "minicpm5-2b") {
            cancellationCount += 1
        })

        model.stop()

        #expect(cancellationCount == 1)
        #expect(!model.isStreaming)
        #expect(model.messages.last?.status == .complete)
        #expect(model.messages.last?.content.isEmpty == true)
        #expect(model.messages.last?.errorMessage == "Stopped.")
    }

    @Test("A server cancellation finalizes locally without echoing cancel")
    func serverCancellationDoesNotEcho() {
        var cancellationCount = 0
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("Search locally", alias: "minicpm5-2b") {
            cancellationCount += 1
        })

        model.cancelAgentTurnFromServer()

        #expect(cancellationCount == 0)
        #expect(!model.isStreaming)
        #expect(model.messages.last?.status == .complete)
        #expect(model.messages.last?.errorMessage == "Stopped.")
    }

    @Test("A late completion cannot overwrite a stopped turn")
    func lateCompletionAfterStopIsIgnored() {
        var delivered: [ProductValueKind] = []
        let model = ChatViewModel(
            persistsConversations: false,
            onProductValueDelivered: { delivered.append($0) }
        )
        #expect(model.beginAgentTurn("Search locally", alias: "minicpm5-2b") {})

        model.stop()
        model.completeAgentTurn("Too late")

        #expect(model.messages.last?.content.isEmpty == true)
        #expect(model.messages.last?.errorMessage == "Stopped.")
        #expect(delivered.isEmpty)
    }

    @Test("Starting a new conversation cannot strand an agent placeholder")
    func conversationTransitionCancelsAgent() {
        var cancellationCount = 0
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("Prepare a report", alias: "minicpm5-2b") {
            cancellationCount += 1
        })
        let previousConversation = model.activeConversationID

        model.newConversation()

        #expect(cancellationCount == 1)
        #expect(model.activeConversationID != previousConversation)
        #expect(model.messages.isEmpty)
        #expect(!model.isStreaming)
    }

    @Test("An Agent failure is shown once without the normal retry banner")
    func failureProjectsOnceIntoTranscript() {
        let model = ChatViewModel(persistsConversations: false)
        #expect(model.beginAgentTurn("Do the task", alias: "minicpm5-2b") {})

        model.failAgentTurn("The local tool became unavailable.")

        #expect(model.messages.last?.status == .failed)
        #expect(model.messages.last?.content.isEmpty == true)
        #expect(model.messages.last?.errorMessage == "The local tool became unavailable.")
        #expect(model.lastError == nil)
        #expect(model.lastFailureKind == nil)
        #expect(model.lastFailureAlias == nil)
    }

    @Test("Stable Agent failure codes become friendly recovery copy")
    func failurePresentationHidesImplementationCodes() {
        let limit = AgentFailurePresentation.message(for: "tool_call_during_final_synthesis")
        #expect(limit.contains("action limit"))
        #expect(limit.contains("smaller steps"))
        #expect(!limit.contains("tool_call_during_final_synthesis"))

        let fallback = AgentFailurePresentation.message(for: "future_server_code")
        #expect(fallback == "Rapid couldn’t complete the agent task. Try again or split it into smaller steps.")
        #expect(!fallback.contains("future_server_code"))

        let invalid = AgentFailurePresentation.message(for: "invalid_tool_arguments")
        #expect(invalid.contains("latest requested action"))
        #expect(invalid.contains("wasn’t run"))
        #expect(!invalid.contains("No action"))
    }

    @Test("Approval presentation uses only the bounded redacted summary")
    func approvalPresentationIsRedactedAndBounded() throws {
        let longToolName = "notes__" + String(repeating: "x", count: 80)
        let longKey = "a" + String(repeating: "k", count: 45)
        let longValue = String(repeating: "v", count: 130)
        let payload: [String: Any] = [
            "call_id": "call-1",
            "name": longToolName,
            "arguments": ["secret": "must-not-appear"],
            "approval_summary": [
                longKey: longValue,
                "beta": ["nested": "summary-only"],
                "delta": [1, 2],
                "epsilon": 1.2349,
                "gamma": "omitted-secret",
                "path": "also-omitted",
            ],
            "risk": "local_change",
            "approval_required": true,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let action = try JSONDecoder().decode(AgentPendingAction.self, from: data)

        let title = AgentApprovalPresentation.title(for: action)
        let message = AgentApprovalPresentation.message(for: action)
        let expectedTool = String(
            ("notes · " + String(repeating: "x", count: 80)).prefix(72)
        ) + "…"
        let expectedKey = String(longKey.prefix(40)) + "…"
        let expectedValue = String(longValue.prefix(120)) + "…"

        #expect(title == "Allow \(expectedTool)?")
        #expect(message == """
        This action will change something on this Mac.
        \(expectedKey): \(expectedValue)
        beta: 1 fields
        delta: 2 items
        epsilon: 1.2349
        …and 2 more details
        """)
        #expect(!message.contains("must-not-appear"))
        #expect(!message.contains("omitted-secret"))
        #expect(!message.contains("also-omitted"))
    }
}
