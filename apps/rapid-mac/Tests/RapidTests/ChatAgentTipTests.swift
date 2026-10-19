import Testing
@testable import Rapid

@Suite("Chat agent tip")
struct ChatAgentTipTests {
    @Test func onlyCompletedModelRepliesQualify() {
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
    }
}
