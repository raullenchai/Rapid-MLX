import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Custom instructions")
struct CustomInstructionsTests {
    private func freshDefaults() -> (UserDefaults, String) {
        let name = "rapid-custom-instructions-test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }

    @Test("Global instructions persist and clearing removes the preference")
    func globalPersistence() {
        let (defaults, name) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: name) }

        let first = CustomInstructionsConfig(defaults: defaults)
        first.global = "Use concise answers."
        #expect(CustomInstructionsConfig(defaults: defaults).global == "Use concise answers.")

        first.global = ""
        #expect(defaults.object(forKey: CustomInstructionsConfig.storageKey) == nil)
    }

    @Test("Blank instruction layers are ignored")
    func blankLayersAreIgnored() {
        let user = ChatMessage(role: .user, content: "Hello", status: .complete)
        let result = ChatViewModel.addingInstructionLayers(
            to: [user],
            ambientPreamble: nil,
            global: " \n ",
            conversation: ""
        )
        #expect(result == [user])
    }

    @Test("Ambient, existing, global, and conversation layers share one ordered system row")
    func layersMergeInOrder() {
        let existing = ChatMessage(role: .system, content: "App system", status: .complete)
        let user = ChatMessage(role: .user, content: "Hello", status: .complete)
        let result = ChatViewModel.addingInstructionLayers(
            to: [existing, user],
            ambientPreamble: "Ambient",
            global: "  Global  ",
            conversation: "Conversation\n"
        )

        #expect(result.count == 2)
        #expect(result.first?.role == .system)
        #expect(result.first?.content == "Ambient\n\nApp system\n\nGlobal\n\nConversation")
        #expect(result.filter { $0.role == .system }.count == 1)
        #expect(result.last?.id == user.id)
    }

    @Test("Removing ambient guidance preserves every user-authored layer")
    func ambientRemovalPreservesCustomLayers() {
        let merged = ChatViewModel.addingInstructionLayers(
            to: [ChatMessage(role: .user, content: "Hello", status: .complete)],
            ambientPreamble: "Ambient",
            global: "Global",
            conversation: "Conversation"
        )
        let result = ChatViewModel.removingLeadingSystemComponent("Ambient", from: merged)
        #expect(result.first?.content == "Global\n\nConversation")
    }

    @Test("Conversation instructions persist and restore with their own chat")
    func conversationPersistenceAndIsolation() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-custom-instructions-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appendingPathComponent("conversations.json")
        let (defaults, name) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: name) }

        let model = ChatViewModel(
            customInstructions: CustomInstructionsConfig(defaults: defaults),
            conversationStoreURL: store
        )
        model.setConversationInstructions("Speak like a product analyst.")
        model.send("Review this idea", alias: "test-model")
        model.stopAndPersist()
        let savedID = model.activeConversationID
        ConversationStore.flush()

        model.newConversation()
        #expect(model.conversationInstructions.isEmpty)
        model.selectConversation(savedID)
        #expect(model.conversationInstructions == "Speak like a product analyst.")

        let reloaded = ChatViewModel(
            customInstructions: CustomInstructionsConfig(defaults: defaults),
            conversationStoreURL: store
        )
        reloaded.selectConversation(savedID)
        #expect(reloaded.conversationInstructions == "Speak like a product analyst.")
    }
}
