import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Memory store")
struct MemoryStoreTests {
    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-memory-test-\(UUID().uuidString).json")
    }

    private func freshDefaults() -> (UserDefaults, String) {
        let name = "rapid-memory-test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (defaults, name)
    }

    @Test("Empty store formats to nil")
    func emptyFormattedNil() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        #expect(store.formattedForPrompt() == nil)
    }

    @Test("Disabled store formats to nil even with entries")
    func disabledFormattedNil() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        defaults.set(false, forKey: "rapid.memory.enabled")
        let url = tempURL()
        let store = MemoryStore(fileURL: url, defaults: defaults)
        store.upsert(content: "Prefers TypeScript", conversationID: UUID())
        #expect(store.formattedForPrompt() == nil)
    }

    @Test("Upsert adds entry and formats for prompt")
    func upsertAndFormat() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        defaults.set(true, forKey: "rapid.memory.enabled")
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        store.upsert(content: "Prefers TypeScript over JavaScript", conversationID: UUID())

        let formatted = store.formattedForPrompt()
        #expect(formatted != nil)
        #expect(formatted!.contains("Prefers TypeScript"))
        #expect(formatted!.contains("<memory_context>"))
    }

    @Test("Upsert deduplicates semantically identical content")
    func dedup() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        store.upsert(content: "Uses pnpm", conversationID: UUID())
        store.upsert(content: "Uses pnpm", conversationID: UUID())
        #expect(store.entries.count == 1)
        #expect(store.entries[0].evidenceCount == 2)
    }

    @Test("Remove deletes a single entry")
    func removeSingle() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        let entry = store.upsert(content: "Test fact", conversationID: UUID())
        store.remove(id: entry.id)
        #expect(store.entries.isEmpty)
    }

    @Test("Update modifies content")
    func updateContent() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        let entry = store.upsert(content: "Old content", conversationID: UUID())
        store.update(id: entry.id, content: "New content")
        #expect(store.entries.first?.content == "New content")
    }

    @Test("Persistence round-trips through disk")
    func persistence() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        defaults.set(true, forKey: "rapid.memory.enabled")
        let url = tempURL()
        let store = MemoryStore(fileURL: url, defaults: defaults)
        store.upsert(content: "Durable fact", conversationID: UUID())

        let reloaded = MemoryStore(fileURL: url, defaults: defaults)
        #expect(reloaded.entries.count == 1)
        #expect(reloaded.entries[0].content == "Durable fact")
    }

    @Test("Prune caps at maximum entries")
    func prune() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        for index in 0..<MemoryStore.maximumEntries + 10 {
            let unique = String(format: "entry-%03d", index)
            store.upsert(content: unique, conversationID: UUID())
        }
        #expect(store.entries.count == MemoryStore.maximumEntries)
    }

    @Test("Formatted output respects character budget")
    func characterBudget() {
        let (defaults, domain) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: domain) }
        defaults.set(true, forKey: "rapid.memory.enabled")
        let store = MemoryStore(fileURL: tempURL(), defaults: defaults)
        for index in 0..<200 {
            store.upsert(content: String(repeating: "x", count: 100) + " \(index)", conversationID: UUID())
        }
        let formatted = store.formattedForPrompt()
        #expect(formatted != nil)
        #expect(formatted!.count <= MemoryStore.maximumInjectedCharacters + 50)
    }
}

@Suite("Memory extractor")
struct MemoryExtractorTests {
    @Test("buildTranscript filters tool and system messages")
    func transcriptFiltering() {
        let messages: [(role: String, content: String)] = [
            ("system", "You are helpful."),
            ("user", "Hello"),
            ("tool", "tool result data"),
            ("assistant", "Hi there!")
        ]
        let transcript = MemoryExtractor.buildTranscript(from: messages)
        #expect(transcript.contains("user: Hello"))
        #expect(transcript.contains("assistant: Hi there!"))
        #expect(!transcript.contains("system"))
        #expect(!transcript.contains("tool"))
    }

    @Test("buildTranscript truncates long messages")
    func transcriptTruncation() {
        let longContent = String(repeating: "a", count: MemoryExtractor.maximumMessageCharacters + 500)
        let messages: [(role: String, content: String)] = [
            ("user", longContent)
        ]
        let transcript = MemoryExtractor.buildTranscript(from: messages)
        #expect(transcript.count < MemoryExtractor.maximumMessageCharacters + 20)
        #expect(transcript.hasSuffix("…"))
    }

    @Test("parseOperations handles well-formed JSON array")
    func parseValid() {
        let content = """
        Here is my analysis:
        [{"action": "add", "content": "Prefers Swift"}, {"action": "remove", "content": "Old pref"}]
        """
        let operations = MemoryExtractor.parseOperations(from: content)
        #expect(operations.count == 2)
        #expect(operations[0] == .add("Prefers Swift"))
        #expect(operations[1] == .remove("Old pref"))
    }

    @Test("parseOperations returns empty on malformed output")
    func parseMalformed() {
        #expect(MemoryExtractor.parseOperations(from: "no JSON here").isEmpty)
        #expect(MemoryExtractor.parseOperations(from: "[broken").isEmpty)
        #expect(MemoryExtractor.parseOperations(from: "").isEmpty)
    }

    @Test("parseOperations tolerates markdown fences")
    func parseFenced() {
        let content = """
        ```json
        [{"action": "add", "content": "Test"}]
        ```
        """
        let operations = MemoryExtractor.parseOperations(from: content)
        #expect(operations.count == 1)
        #expect(operations[0] == .add("Test"))
    }

    @Test("parseOperations ignores unknown actions")
    func parseUnknownAction() {
        let content = "[{\"action\": \"merge\", \"content\": \"test\"}]" 
        #expect(MemoryExtractor.parseOperations(from: content).isEmpty)
    }

    @Test("reviewPrompt includes transcript and rules")
    func reviewPromptShape() {
        let prompt = MemoryExtractor.reviewPrompt(transcript: "user: Hello")
        #expect(prompt.contains("user: Hello"))
        #expect(prompt.contains("Return ONLY a JSON array"))
        #expect(prompt.contains("Do NOT save one-off"))
    }
}

extension MemoryExtractorTests {
    @Test("parseOperations treats bare fact as implicit add")
    func parseBareFact() {
        let content = "[{\"fact\": \"User switched from pnpm to bun.\"}]"
        let operations = MemoryExtractor.parseOperations(from: content)
        #expect(operations.count == 1)
        #expect(operations[0] == .add("User switched from pnpm to bun."))
    }

    @Test("parseOperations treats bare memory as implicit add")
    func parseBareMemory() {
        let content = "[{\"memory\": \"Uses Neovim.\"}]"
        let operations = MemoryExtractor.parseOperations(from: content)
        #expect(operations.count == 1)
        #expect(operations[0] == .add("Uses Neovim."))
    }
}
