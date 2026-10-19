import Testing
@testable import Rapid

@Suite("Chat agent tip")
struct ChatAgentTipTests {
    @Test @MainActor func onlyCompletedModelRepliesQualify() {
        #expect(!ChatView.hasSuccessfulReply([]))
        #expect(!ChatView.hasSuccessfulReply([
            ChatMessage(role: .assistant, content: "Hello", status: .streaming)
        ]))
        #expect(!ChatView.hasSuccessfulReply([
            ChatMessage(role: .assistant, content: "", status: .complete),
            ChatMessage(role: .assistant, content: "Error", status: .failed),
            ChatMessage(role: .assistant, content: "Welcome", wireVisibility: .transcriptOnly)
        ]))
        #expect(ChatView.hasSuccessfulReply([
            ChatMessage(role: .assistant, content: "Hello", status: .complete)
        ]))

        var stopped = ChatMessage(role: .assistant, content: "Partial", status: .streaming)
        ChatViewModel.finaliseCancellation(message: &stopped)
        #expect(!ChatView.hasSuccessfulReply([stopped]))
    }
}
