import Foundation
import Testing
@testable import Rapid

@Suite("Small model tool reminder")
struct SmallModelToolReminderTests {
    private let tools = [
        ToolDefinition(name: "web_search", description: "Search", parameters: .object([:])),
        ToolDefinition(name: "fetch_url", description: "Fetch", parameters: .object([:]))
    ]
    private var history: [ChatMessage] {
        [ChatMessage(role: .user, content: "Hello"),
         ChatMessage(role: .assistant, content: "Hi"),
         ChatMessage(role: .user, content: "Latest question")]
    }
    private func rows(_ request: ChatStreamClient.Request) throws -> [[String: Any]] {
        try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request.messages)) as? [[String: Any]])
    }

    @Test("Every user message after an assistant gets the reminder")
    func reminder() throws {
        let original = history + [ChatMessage(role: .assistant, content: "Answer"),
                                  ChatMessage(role: .user, content: "Next question")]
        let snapshot = original
        let request = ChatStreamClient.Request(alias: "lfm3-vl-2b", messages: original, tools: tools)
        let wire = try rows(request)
        #expect(wire[0]["content"] as? String == "Hello")
        #expect(wire[2]["content"] as? String == "Latest question\n\n(Tools available: web_search, fetch_url. Call a tool when this question needs current or external information, including recent events your training may not cover, instead of answering from memory; for writing, math or translation, answer directly unless the user asks you to use a tool.)")
        #expect((wire[4]["content"] as? String)?.contains("Tools available: web_search, fetch_url") == true)
        #expect(original == snapshot)
        #expect(original[2].content == "Latest question")
    }

    @Test("Consecutive requests preserve earlier wire rows")
    func consecutiveTurns() throws {
        let first = try rows(ChatStreamClient.Request(alias: "model-2b", messages: history, tools: tools))
        let extended = history + [ChatMessage(role: .assistant, content: "Answer"),
                                  ChatMessage(role: .user, content: "Next question")]
        let second = try rows(ChatStreamClient.Request(alias: "model-2b", messages: extended, tools: tools))
        #expect(first.count < second.count)
        for (earlier, later) in zip(first, second) {
            #expect(NSDictionary(dictionary: earlier).isEqual(to: later))
        }
    }

    @Test("Wire prelude never receives the reminder")
    func prelude() throws {
        var messages = history
        messages[2].wirePrelude = "Local context"
        let wire = try rows(ChatStreamClient.Request(alias: "model-2b", messages: messages, tools: tools))
        #expect(wire[2]["content"] as? String == "Local context")
        #expect((wire[3]["content"] as? String)?.contains("Tools available:") == true)
        messages.append(ChatMessage(role: .assistant, content: "Answer", wirePrelude: "Assistant prelude"))
        let trailingPrelude = try rows(ChatStreamClient.Request(alias: "model-2b", messages: messages, tools: tools))
        #expect(trailingPrelude[4]["content"] as? String == "Assistant prelude")
    }

    @Test("First turn, large or unknown models, and disabled tools are unchanged")
    func gates() throws {
        for alias in ["model-27b", "unknown-model"] {
            let wire = try rows(ChatStreamClient.Request(alias: alias, messages: history, tools: tools))
            #expect(wire[2]["content"] as? String == "Latest question")
        }
        let first = try rows(ChatStreamClient.Request(alias: "model-4b", messages: [history[2]], tools: tools))
        #expect(first[0]["content"] as? String == "Latest question")
        let disabled = try rows(ChatStreamClient.Request(alias: "model-4b", messages: history))
        #expect(disabled[2]["content"] as? String == "Latest question")
        let boundary = try rows(ChatStreamClient.Request(alias: "model-4b", messages: history, tools: tools))
        #expect((boundary[2]["content"] as? String)?.contains("Tools available:") == true)
    }

    @Test("Image parts survive and a text part carries the reminder")
    func image() throws {
        let image = try ChatImageAttachment(filename: "photo.png", mimeType: "image/png", data: Data("photo".utf8))
        var messages = history
        messages[2].imageAttachments = [image]
        let request = ChatStreamClient.Request(alias: "model-2b", messages: messages, tools: tools, supportsImageInput: true)
        let wire = try rows(request)
        let parts = try #require(wire[2]["content"] as? [[String: Any]])
        #expect(parts.count == 3)
        #expect(parts[0]["text"] as? String == "Latest question")
        #expect(parts[1]["type"] as? String == "image_url")
        #expect((parts[2]["text"] as? String)?.contains("Tools available:") == true)
        #expect(messages[2].content == "Latest question")
        #expect(messages[2].imageAttachments.count == 1)
    }
}
