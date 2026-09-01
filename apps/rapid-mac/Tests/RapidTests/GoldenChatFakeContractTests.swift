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
        #expect(surface.fake.recordedPrompts() == ["probe prompt"])
    }
}
