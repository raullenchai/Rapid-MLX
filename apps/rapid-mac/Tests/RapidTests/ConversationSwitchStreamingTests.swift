import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Conversation switch streaming", .serialized)
struct ConversationSwitchStreamingTests {
    private func isolatedStore() throws -> (root: URL, file: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-switch-streaming-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (root, root.appendingPathComponent("conversations.json"))
    }

    private func conversation(
        title: String,
        messages: [ChatMessage],
        branches: [ChatMessage] = [],
        updatedAt: Date = Date()
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(), title: title, messages: messages, branches: branches,
            createdAt: updatedAt.addingTimeInterval(-60), updatedAt: updatedAt
        )
    }

    private func waitUntil(
        _ description: String,
        timeout: Duration = .seconds(5),
        condition: @escaping @MainActor () async -> Bool
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Timed out waiting for \(description)")
    }

    @Test("Switching during a stream keeps and stops the partial reply")
    func switchDuringStreaming() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let other = conversation(
            title: "Other",
            messages: [
                ChatMessage(role: .user, content: "other question"),
                ChatMessage(role: .assistant, content: "other answer"),
            ]
        )
        ConversationStore.save([other], to: store.file)
        ConversationStore.flush()

        let fake = GoldenChatFake()
        fake.interChunkDelay = 0.02
        fake.contentRepeat = 20_000
        let surface = GoldenChatSurface.mountWithSidebar(
            fake: fake, conversationStoreURL: store.file
        )
        surface.chat.send("stream this answer", alias: GoldenChatSurface.alias)
        let streamedID = surface.chat.activeConversationID
        try await waitUntil("non-empty streamed content") {
            surface.chat.isStreaming && surface.chat.messages.last?.content.isEmpty == false
        }
        let updatedBeforeSwitch = surface.chat.conversations
            .first { $0.id == streamedID }?.updatedAt

        surface.chat.selectConversation(other.id)
        #expect(!surface.chat.isStreaming)
        #expect(surface.chat.activeConversationID == other.id)
        #expect(surface.chat.messages.last?.content == "other answer")

        let archived = try #require(surface.chat.conversations.first { $0.id == streamedID })
        #expect(archived.updatedAt == updatedBeforeSwitch)
        let stopped = try #require(archived.messages.last)
        #expect(stopped.role == .assistant)
        #expect(stopped.status == .complete)
        #expect(stopped.errorMessage == "Stopped.")
        #expect(!stopped.content.isEmpty)
        #expect(!surface.chat.conversations.flatMap(\.allMessages).contains {
            $0.status == .streaming
        })

        surface.chat.selectConversation(streamedID)
        #expect(surface.chat.messages.last?.content == stopped.content)
        #expect(surface.chat.messages.last?.errorMessage == "Stopped.")
        ConversationStore.flush()

        let reloaded = ChatViewModel(conversationStoreURL: store.file)
        #expect(!reloaded.conversations.flatMap(\.allMessages).contains {
            $0.status == .streaming
        })
        let diskReply = try #require(
            reloaded.conversations.first { $0.id == streamedID }?.messages.last
        )
        #expect(diskReply.content == stopped.content)
        #expect(diskReply.errorMessage == "Stopped.")
    }

    @Test("New Chat during a stream archives a stopped partial reply")
    func newChatDuringStreaming() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let fake = GoldenChatFake()
        fake.interChunkDelay = 0.02
        fake.contentRepeat = 20_000
        let surface = GoldenChatSurface.mountWithSidebar(
            fake: fake, conversationStoreURL: store.file
        )
        surface.chat.send("stream before new chat", alias: GoldenChatSurface.alias)
        let streamedID = surface.chat.activeConversationID
        try await waitUntil("non-empty streamed content") {
            surface.chat.isStreaming && surface.chat.messages.last?.content.isEmpty == false
        }

        surface.chat.newConversation()
        #expect(!surface.chat.isStreaming)
        #expect(surface.chat.messages.isEmpty)
        let stopped = try #require(
            surface.chat.conversations.first { $0.id == streamedID }?.messages.last
        )
        #expect(stopped.status == .complete)
        #expect(stopped.errorMessage == "Stopped.")
        #expect(!stopped.content.isEmpty)
    }

    @Test("Switching while a tool executes finalises its assistant shell")
    func switchDuringToolExecution() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let other = conversation(
            title: "Other",
            messages: [
                ChatMessage(role: .user, content: "other question"),
                ChatMessage(role: .assistant, content: "other answer"),
            ]
        )
        ConversationStore.save([other], to: store.file)
        ConversationStore.flush()

        let fake = GoldenChatFake()
        fake.nativeWebSearchFixture = true
        let gate = ToolExecutionGate()
        let registry = BuiltinToolRegistry { _, _, _ in
            await gate.run()
        }
        let surface = GoldenChatSurface.mountWithSidebar(
            fake: fake, tools: registry, conversationStoreURL: store.file
        )
        surface.chat.send("research this topic", alias: GoldenChatSurface.alias)
        let interruptedID = surface.chat.activeConversationID
        try await waitUntil("tool execution to start behind its assistant shell") {
            guard surface.chat.isStreaming,
                  await gate.hasStarted(),
                  let shell = surface.chat.messages.last
            else { return false }
            return shell.role == .assistant
                && shell.status == .complete
                && shell.toolCalls?.isEmpty == false
        }

        surface.chat.selectConversation(other.id)
        let archived = try #require(
            surface.chat.conversations.first { $0.id == interruptedID }
        )
        let stopped = try #require(archived.messages.last)
        #expect(stopped.role == .assistant)
        #expect(stopped.status == .complete)
        #expect(stopped.errorMessage == "Stopped.")
        #expect(stopped.toolCalls == nil)
        #expect(!archived.allMessages.contains { $0.status == .streaming })

        await gate.release()
        for _ in 0..<10 { await Task.yield() }
        ConversationStore.flush()
        let reloaded = ChatViewModel(conversationStoreURL: store.file)
        let diskConversation = try #require(
            reloaded.conversations.first { $0.id == interruptedID }
        )
        let diskShell = try #require(diskConversation.messages.last)
        #expect(diskShell.status == .complete)
        #expect(diskShell.errorMessage == "Stopped.")
        #expect(diskShell.toolCalls == nil)
        #expect(!diskConversation.allMessages.contains { $0.status == .streaming })
    }

    @Test("Pure repair finalises path and branch streams")
    func pureRepair() throws {
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let stale = conversation(
            title: "Stale",
            messages: [
                ChatMessage(role: .user, content: "question"),
                ChatMessage(role: .assistant, content: "path partial", status: .streaming),
            ],
            branches: [
                ChatMessage(role: .assistant, content: "branch partial", status: .streaming),
            ],
            updatedAt: timestamp
        )

        let result = ChatViewModel.repairInterruptedStreams([stale])
        #expect(result.changed)
        #expect(result.conversations[0].updatedAt == timestamp)
        for message in result.conversations[0].messages.dropFirst() + result.conversations[0].branches {
            #expect(message.status == .complete)
            #expect(message.errorMessage == "Stopped.")
            #expect(message.content.contains("partial"))
        }
        let secondPass = ChatViewModel.repairInterruptedStreams(result.conversations)
        #expect(!secondPass.changed)
        #expect(secondPass.conversations == result.conversations)
    }

    @Test("Launch repairs and persists stale path and branch streams")
    func launchRepair() throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let stale = conversation(
            title: "Stale",
            messages: [
                ChatMessage(role: .user, content: "question"),
                ChatMessage(role: .assistant, content: "path partial", status: .streaming),
            ],
            branches: [
                ChatMessage(role: .assistant, content: "branch partial", status: .streaming),
            ],
            updatedAt: timestamp
        )
        ConversationStore.save([stale], to: store.file)
        ConversationStore.flush()

        let model = ChatViewModel(conversationStoreURL: store.file)
        let repaired = try #require(model.conversations.first)
        #expect(repaired.updatedAt == timestamp)
        #expect(!repaired.allMessages.contains { $0.status == .streaming })
        #expect(repaired.messages.last?.errorMessage == "Stopped.")
        #expect(repaired.branches.last?.errorMessage == "Stopped.")
        model.selectConversation(stale.id)
        #expect(model.messages.last?.content == "path partial")
        #expect(model.messages.last?.errorMessage == "Stopped.")

        ConversationStore.flush()
        let disk = try #require(ConversationStore.load(from: store.file).first)
        #expect(disk.updatedAt == timestamp)
        #expect(!disk.allMessages.contains { $0.status == .streaming })
    }

    @Test("A title lands on its conversation after switching away")
    func titleSurvivesSwitch() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let transcript = [
            ChatMessage(role: .user, content: "explain orbital resonance"),
            ChatMessage(role: .assistant, content: "It is a periodic gravitational relationship."),
        ]
        let a = conversation(title: "explain orbital resonance", messages: transcript)
        let b = conversation(title: "B", messages: [ChatMessage(role: .user, content: "B")])
        ConversationStore.save([a, b], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: a.id, transcript: transcript) { _ in
            await gate.complete()
        }
        model.scheduleTitle(for: a.id, transcript: transcript) { _ in
            "Duplicate title"
        }
        model.selectConversation(b.id)
        model.send("another B question", alias: "test-model")
        #expect(model.hasPendingTitleTask(for: a.id))
        model.stopAndPersist()
        #expect(model.hasPendingTitleTask(for: a.id))
        await gate.release(with: "Orbital resonance")
        try await waitUntil("title to land") {
            model.conversations.first { $0.id == a.id }?.hasGeneratedTitle == true
        }

        #expect(model.conversations.first { $0.id == a.id }?.title == "Orbital resonance")
        #expect(model.conversations.first { $0.id == b.id }?.title == "B")
        #expect(!model.hasPendingTitleTask(for: a.id))
    }

    @Test("Sending in the same conversation cancels its pending title")
    func sameConversationSendCancelsTitle() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let transcript = [
            ChatMessage(role: .user, content: "first question"),
            ChatMessage(role: .assistant, content: "first answer"),
        ]
        let a = conversation(title: "first question", messages: transcript)
        ConversationStore.save([a], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.selectConversation(a.id)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: a.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.hasPendingTitleTask(for: a.id))
        model.send("second question", alias: "test-model")
        #expect(!model.hasPendingTitleTask(for: a.id))
        model.stopAndPersist()
        await gate.release(with: "Late title")
        for _ in 0..<10 { await Task.yield() }
        #expect(model.conversations.first { $0.id == a.id }?.hasGeneratedTitle == false)
    }

    @Test("Regenerating cancels the pending title for the rejected answer")
    func regenerateCancelsTitle() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let transcript = [
            ChatMessage(role: .user, content: "first question"),
            ChatMessage(role: .assistant, content: "rejected answer"),
        ]
        let conversation = conversation(title: "first question", messages: transcript)
        ConversationStore.save([conversation], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.selectConversation(conversation.id)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: conversation.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.hasPendingTitleTask(for: conversation.id))
        model.regenerateLast(alias: "test-model")
        #expect(!model.hasPendingTitleTask(for: conversation.id))
        model.stopAndPersist()

        await gate.release(with: "Rejected answer title")
        for _ in 0..<10 { await Task.yield() }
        #expect(
            model.conversations.first { $0.id == conversation.id }?.hasGeneratedTitle == false
        )
    }

    @Test("Switching branches cancels the pending title for the abandoned branch")
    func branchSwitchCancelsTitle() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let user = ChatMessage(role: .user, content: "first question")
        let original = ChatMessage(
            role: .assistant,
            content: "original answer",
            parentID: user.id,
            createdAt: Date(timeIntervalSince1970: 1)
        )
        let replacement = ChatMessage(
            role: .assistant,
            content: "replacement answer",
            parentID: user.id,
            createdAt: Date(timeIntervalSince1970: 2)
        )
        let transcript = [user, replacement]
        let branched = conversation(
            title: "first question",
            messages: transcript,
            branches: [original]
        )
        ConversationStore.save([branched], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.selectConversation(branched.id)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: branched.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.hasPendingTitleTask(for: branched.id))
        #expect(model.selectBranch(at: 0, forSiblingOf: replacement.id))
        #expect(!model.hasPendingTitleTask(for: branched.id))

        await gate.release(with: "Abandoned branch title")
        for _ in 0..<10 { await Task.yield() }
        let saved = model.conversations.first { $0.id == branched.id }
        #expect(saved?.title == "first question")
        #expect(saved?.hasGeneratedTitle == false)
    }

    @Test("Deleting a message cancels its pending title")
    func deletingMessageCancelsTitle() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "answer")
        let transcript = [user, answer]
        let conversation = conversation(title: "question", messages: transcript)
        ConversationStore.save([conversation], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.selectConversation(conversation.id)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: conversation.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.hasPendingTitleTask(for: conversation.id))
        #expect(model.deleteMessage(id: answer.id))
        #expect(!model.hasPendingTitleTask(for: conversation.id))

        await gate.release(with: "Deleted message title")
        for _ in 0..<10 { await Task.yield() }
        let saved = model.conversations.first { $0.id == conversation.id }
        #expect(saved?.title == "question")
        #expect(saved?.hasGeneratedTitle == false)
    }

    @Test("Deleting a conversation cancels its pending title")
    func deletingCancelsTitle() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let transcript = [
            ChatMessage(role: .user, content: "question"),
            ChatMessage(role: .assistant, content: "answer"),
        ]
        let a = conversation(title: "question", messages: transcript)
        ConversationStore.save([a], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: a.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.hasPendingTitleTask(for: a.id))
        model.deleteConversation(a.id)
        #expect(!model.hasPendingTitleTask(for: a.id))
        await gate.release(with: "Late title")
        for _ in 0..<10 { await Task.yield() }
        #expect(!model.conversations.contains { $0.id == a.id })
    }

    @Test("A custom title wins over a pending generated title")
    func customTitleWins() async throws {
        let store = try isolatedStore()
        defer {
            ConversationStore.flush()
            try? FileManager.default.removeItem(at: store.root)
        }
        let transcript = [
            ChatMessage(role: .user, content: "question"),
            ChatMessage(role: .assistant, content: "answer"),
        ]
        let a = conversation(title: "question", messages: transcript)
        ConversationStore.save([a], to: store.file)
        ConversationStore.flush()
        let model = ChatViewModel(conversationStoreURL: store.file)
        let gate = TitleCompletionGate()

        model.scheduleTitle(for: a.id, transcript: transcript) { _ in
            await gate.complete()
        }
        #expect(model.renameConversation(a.id, to: "My title"))
        await gate.release(with: "Generated title")
        try await waitUntil("title task cleanup") {
            !model.hasPendingTitleTask(for: a.id)
        }
        #expect(model.conversations.first { $0.id == a.id }?.title == "My title")
        #expect(model.conversations.first { $0.id == a.id }?.hasGeneratedTitle == false)
    }
}

private actor TitleCompletionGate {
    private var result: String?
    private var continuation: CheckedContinuation<String, Never>?

    func complete() async -> String {
        if let result { return result }
        return await withCheckedContinuation { continuation = $0 }
    }

    func release(with value: String) {
        result = value
        continuation?.resume(returning: value)
        continuation = nil
    }
}

private actor ToolExecutionGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func run() async -> ToolCallResult {
        started = true
        await withCheckedContinuation { continuation = $0 }
        return ToolCallResult(toolCallID: "", content: "gated search result")
    }

    func hasStarted() -> Bool {
        started
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}
