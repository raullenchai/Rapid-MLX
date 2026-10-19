import Foundation
import Testing
@testable import Rapid

struct ConversationHistoryTests {
    @Test("History saves are owner-only and preserve their contents")
    func savePermissions() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-history-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("conversations.json")
        let conversation = ChatConversation(
            id: UUID(), title: "Private chat", messages: [], createdAt: Date(), updatedAt: Date()
        )
        ConversationStore.save([conversation], to: url)
        ConversationStore.flush()
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        #expect(ConversationStore.load(from: url).map(\.id) == [conversation.id])
    }
}
