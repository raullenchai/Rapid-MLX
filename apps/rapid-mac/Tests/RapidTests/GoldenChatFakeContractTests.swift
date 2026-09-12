import Foundation
import Testing

@testable import Rapid

/// Contract checks for the shared golden-journey backend itself, so a
/// journey failure can be read as an app regression rather than a broken
/// fixture: the fake streams the bash fake's wire shape (reasoning first,
/// content chunks, `[DONE]`), records the request, and logs its lifecycle.
@Suite("GoldenChatFake contract")
struct GoldenChatFakeContractTests {
    @Test("The fake streams a default reply over URLSession")
    func streamsDefaultReply() async throws {
        let fake = GoldenChatFake()
        let session = fake.session()
        var request = URLRequest(url: fake.baseURL.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": "hello probe"]]
        ])
        let (bytes, response) = try await session.bytes(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        var lines: [String] = []
        for try await line in bytes.lines {
            lines.append(line)
            if line.contains("[DONE]") { break }
        }
        #expect(lines.contains { $0.contains("reasoning_content") })
        #expect(lines.contains { $0.contains("Hello") })
        #expect(fake.recordedPrompts() == ["hello probe"])
        #expect(fake.events() == [.chatFinished(chunks: GoldenChatFake.contentChunks.count)])
    }

    @MainActor
    @Test("A mounted surface routes a send through the fake and settles")
    func mountedSurfaceSendsAndSettles() async throws {
        let surface = GoldenChatSurface.mount()
        try await surface.sendPrompt("probe prompt")
        try await surface.waitForSendIdle()
        // The mounted surface goes through the real request assembly, so the
        // recorded prompt is the WIRE text: the user's prose followed by the
        // per-turn clock trailer. Asserting both halves here makes this golden
        // flow the end-to-end proof that #2330's "the model is told the
        // current time" contract survived moving the clock off the system row
        // (see ``ChatViewModel.stampingClockContext``).
        let recorded = try #require(surface.fake.recordedPrompts().first)
        #expect(surface.fake.recordedPrompts().count == 1)
        #expect(recorded.hasPrefix("probe prompt"))
        #expect(recorded.contains("[CURRENT LOCAL TIME]"))
        #expect(recorded.contains("The current local time is "))
    }
}
