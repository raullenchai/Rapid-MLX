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

    private func clockRow(_ fake: GoldenChatFake) throws -> String {
        let body = try #require(fake.recordedBodies().last)
        let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(object["messages"] as? [[String: Any]])
        return try #require(messages[1]["content"] as? String)
    }

    private func expectClock(for row: ChatMessage, in fake: GoldenChatFake) throws {
        let clock = try clockRow(fake)
        let expected = ChatViewModel.clockContext(at: row.createdAt)
        #expect(clock.hasPrefix(expected))
        let rest = String(clock.dropFirst(expected.count))
        let firstMinute = (row.createdAt.timeIntervalSinceReferenceDate / 60).rounded(.down)
        let lastMinute = (Date().timeIntervalSinceReferenceDate / 60).rounded(.down)
        let validLines = stride(from: firstMinute, through: lastMinute, by: 1).map {
            ChatViewModel.answeringNowLine(asked: row.createdAt,
                answeringAt: Date(timeIntervalSinceReferenceDate: $0 * 60))
        }
        #expect(validLines.contains(rest))
    }

    @MainActor
    @Test("A mounted surface routes a send through the fake and settles")
    func mountedSurfaceSendsAndSettles() async throws {
        let surface = GoldenChatSurface.mount()
        try await surface.sendPrompt("probe prompt")
        try await surface.waitForSendIdle()
        #expect(surface.fake.recordedPrompts() == ["probe prompt"])
        let row = try #require(surface.chat.messages.first { $0.role == .user })
        try expectClock(for: row, in: surface.fake)
    }

    @MainActor
    @Test("Regenerate reuses the user row; an edited send mints a new one")
    func resendPathsMintOrReuseTheUserRow() async throws {
        // codex asked for the REAL resend paths, not the stamper in isolation:
        // a test that constructs rows by hand stays green if
        // ``editUserMessage`` or ``regenerateAnswer`` stops minting or reusing
        // rows correctly. This drives both through the mounted surface and
        // reads the recorded wire bodies back.
        let surface = GoldenChatSurface.mount()
        try await surface.sendPrompt("first question")
        try await surface.waitForSendIdle()

        let firstRow = try #require(surface.chat.messages.first { $0.role == .user })
        let firstWire = try #require(surface.fake.recordedPrompts().last)
        try expectClock(for: firstRow, in: surface.fake)

        // Regenerate: the SAME row is re-sent, so the prelude must come back
        // byte-identical. If it moved, every later turn in this conversation
        // would lose the shared prefix.
        let beforeRegenerate = surface.fake.recordedBodies().count
        surface.chat.regenerateLast(alias: GoldenChatSurface.alias)
        try await surface.stage.wait(for: "the regenerated request to be recorded") {
            surface.fake.recordedBodies().count > beforeRegenerate
        }
        try await surface.waitForSendIdle()
        let regeneratedRow = try #require(surface.chat.messages.first { $0.role == .user })
        #expect(regeneratedRow.id == firstRow.id)
        #expect(regeneratedRow.createdAt == firstRow.createdAt)
        let regeneratedWire = try #require(surface.fake.recordedPrompts().last)
        #expect(regeneratedWire == firstWire,
                "Re-answering the same question must re-render it identically, prelude checked separately.")
        try expectClock(for: firstRow, in: surface.fake)

        // An edited send rewinds and calls `send`, which mints a fresh row —
        // so it reports the live clock. Asserted on `createdAt` rather than on
        // the rendered text, because the prelude has minute resolution and a
        // test that edits within the same minute would see the same string.
        #expect(surface.fake.recordedBodies().count == beforeRegenerate + 1,
                "Sanity check: regenerate really did send a second request.")
        let edited = surface.chat.editUserMessage(
            id: firstRow.id,
            newContent: "edited question",
            alias: GoldenChatSurface.alias
        )
        #expect(edited, "editUserMessage must not be refused on a settled surface.")
        try await surface.stage.wait(for: "the edited request to be recorded") {
            surface.fake.recordedBodies().count > beforeRegenerate + 1
        }
        try await surface.waitForSendIdle()
        let editedRow = try #require(surface.chat.messages.first { $0.role == .user })
        #expect(editedRow.id != firstRow.id, "An edited send must mint a new row.")
        #expect(editedRow.createdAt >= firstRow.createdAt)
        let editedWire = try #require(surface.fake.recordedPrompts().last)
        #expect(editedWire.hasPrefix("edited question"))
        try expectClock(for: editedRow, in: surface.fake)
    }
}
