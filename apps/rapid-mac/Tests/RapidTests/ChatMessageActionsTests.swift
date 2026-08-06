import Testing
@testable import Rapid

@MainActor
@Suite("Chat message actions")
struct ChatMessageActionsTests {
    @Test("Editing an older user message drops the later turns and resends the edit")
    func editOlderUserMessageRewindsConversation() {
        let viewModel = ChatViewModel(persistsConversations: false)
        let firstUser = ChatMessage(role: .user, content: "original question")
        let firstAssistant = ChatMessage(role: .assistant, content: "first answer")
        let secondUser = ChatMessage(role: .user, content: "follow-up")
        let secondAssistant = ChatMessage(role: .assistant, content: "second answer")
        viewModel.devSeedMessages([firstUser, firstAssistant, secondUser, secondAssistant])

        let edited = viewModel.editUserMessage(
            id: firstUser.id,
            newContent: "  revised question  ",
            alias: "test-model"
        )
        defer { viewModel.stopAndPersist() }

        #expect(edited)
        #expect(viewModel.messages.count == 2)
        #expect(viewModel.messages[0].role == .user)
        #expect(viewModel.messages[0].content == "revised question")
        #expect(viewModel.messages[1].role == .assistant)
        #expect(viewModel.messages[1].status == .streaming)
        #expect(!viewModel.messages.contains(where: { $0.id == secondUser.id }))
        #expect(!viewModel.messages.contains(where: { $0.id == secondAssistant.id }))
    }

    @Test("Retrying an older assistant response replays its preceding user turn")
    func retryOlderAssistantRewindsToItsUserMessage() {
        let viewModel = ChatViewModel(persistsConversations: false)
        let firstUser = ChatMessage(role: .user, content: "first question")
        let firstAssistant = ChatMessage(role: .assistant, content: "first answer")
        let secondUser = ChatMessage(role: .user, content: "second question")
        let secondAssistant = ChatMessage(role: .assistant, content: "second answer")
        viewModel.devSeedMessages([firstUser, firstAssistant, secondUser, secondAssistant])

        let retried = viewModel.retryAssistantMessage(
            id: firstAssistant.id,
            alias: "test-model"
        )
        defer { viewModel.stopAndPersist() }

        #expect(retried)
        #expect(viewModel.messages.count == 2)
        #expect(viewModel.messages[0].role == .user)
        #expect(viewModel.messages[0].content == "first question")
        #expect(viewModel.messages[1].role == .assistant)
        #expect(viewModel.messages[1].status == .streaming)
        #expect(!viewModel.messages.contains(where: { $0.id == secondUser.id }))
        #expect(!viewModel.messages.contains(where: { $0.id == secondAssistant.id }))
    }
}
