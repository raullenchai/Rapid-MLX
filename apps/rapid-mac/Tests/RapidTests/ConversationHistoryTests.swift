import Foundation
import Testing
@testable import Rapid

struct ConversationHistoryTests {
    @Test("Backend fields preserve legacy JSON and round-trip")
    func backendFields() throws {
        let fixture = #"{"createdAt":0,"hasCustomTitle":false,"hasGeneratedTitle":false,"id":"00000000-0000-0000-0000-000000000001","isArchived":false,"isPinned":false,"messages":[],"title":"Legacy","updatedAt":0}"#.data(using: .utf8)!
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        var conversation = try decoder.decode(ChatConversation.self, from: fixture)
        #expect(conversation.backend == nil)
        #expect(conversation.ownerConversationID == nil)
        #expect(try encoder.encode(conversation) == fixture)
        conversation.backend = .ownerV1
        conversation.ownerConversationID = "owner-123"
        let restored = try decoder.decode(ChatConversation.self, from: encoder.encode(conversation))
        #expect(restored.backend == .ownerV1)
        #expect(restored.ownerConversationID == "owner-123")
    }

    @Test("Harness toggle maps to the persisted backend")
    @MainActor
    func harnessToggle() {
        let suite = "Settings-ChatBackend-" + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let config = ChatBackendConfig(defaults: defaults)
        #expect(!config.harnessEnabled)
        config.harnessEnabled = true
        #expect(config.chatBackend == .ownerV1)
        #expect(ChatBackendConfig(defaults: defaults).harnessEnabled)
        config.harnessEnabled = false
        #expect(config.chatBackend == .legacy)
        #expect(!ChatBackendConfig(defaults: defaults).harnessEnabled)
    }

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
