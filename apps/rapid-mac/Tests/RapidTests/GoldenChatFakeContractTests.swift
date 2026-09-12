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
        #expect(recorded.contains("[MESSAGE SENT]"))
        #expect(recorded.contains("This message was sent "))
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
        let firstTrailer = try #require(
            firstWire.range(of: "[MESSAGE SENT]").map { String(firstWire[$0.lowerBound...]) }
        )

        // Regenerate: the SAME row is re-sent, so the trailer must come back
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
                "Re-answering the same question must re-render it identically, trailer included.")
        #expect(regeneratedWire.hasSuffix(firstTrailer))

        // An edited send rewinds and calls `send`, which mints a fresh row —
        // so it reports the live clock. Asserted on `createdAt` rather than on
        // the rendered text, because the trailer has minute resolution and a
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
        #expect(editedWire.contains(
            ChatViewModel.clockContext(at: editedRow.createdAt)
        ), "The edited row's trailer must be stamped from its OWN createdAt.")
    }
}
