import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Temporary chats", .serialized)
struct TemporaryChatTests {
    // MARK: - Helpers

    private func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-temp-chat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeSession(
        storageRoot: URL = TemporaryChatStorage.root(),
        id: UUID = UUID(),
        protocolClass: AnyClass = TemporaryChatCaptureProtocol.self,
        makeTools: @escaping (DocumentContentCache, BrowseContentCache) -> any ToolRegistry = { _, _ in
            EmptyToolRegistry()
        }
    ) -> TemporaryChatSession {
        TemporaryChatCaptureProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [protocolClass]
        return TemporaryChatSession(
            id: id,
            storageRoot: storageRoot,
            server: nil,
            sampling: nil,
            client: ChatStreamClient(
                baseURL: URL(string: "fake://temporary-chat")!,
                session: URLSession(configuration: config)
            ),
            makeTools: makeTools
        )
    }

    private func waitForTurn(_ chat: ChatViewModel) async throws {
        for _ in 0..<300 where chat.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!chat.isStreaming)
    }

    /// Every regular file under `root`, by relative path. Empty directories are
    /// not content, so they are ignored.
    private func snapshot(of root: URL) -> [String: Data] {
        var result: [String: Data] = [:]
        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys
        ) else { return result }
        // The enumerator can report /private/var for a /var temp directory.
        let base = root.resolvingSymlinksInPath().path
        for case let url as URL in walker {
            guard (try? url.resourceValues(forKeys: Set(keys)).isRegularFile) == true else { continue }
            let relative = String(url.resolvingSymlinksInPath().path.dropFirst(base.count))
            result[relative] = try? Data(contentsOf: url)
        }
        return result
    }

    private func withHome<T>(_ home: URL, _ body: () async throws -> T) async rethrows -> T {
        let previous = ProcessInfo.processInfo.environment["HOME"]
        setenv("HOME", home.path, 1)
        defer {
            if let previous { setenv("HOME", previous, 1) } else { unsetenv("HOME") }
        }
        return try await body()
    }

    private func requestSystemPrompt() throws -> String {
        let body = try #require(TemporaryChatCaptureProtocol.lastRequestBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        return try #require(messages.first { $0["role"] as? String == "system" }?["content"] as? String)
    }

    // MARK: - No trace

    @Test("A temporary chat leaves no trace on disk once it is closed")
    func leavesNoTrace() async throws {
        let home = try scratchDirectory()
        let sources = try scratchDirectory()
        defer {
            try? FileManager.default.removeItem(at: home)
            try? FileManager.default.removeItem(at: sources)
        }
        let suite = "rapid-temp-chat-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        try await withHome(home) {
            // A populated, enabled memory file in the place the app keeps it.
            let memoryURL = ApplicationSupportLocator.applicationSupportRoot()
                .appendingPathComponent("memory-library-v1.json")
            try FileManager.default.createDirectory(
                at: memoryURL.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            let memory = MemoryStore(fileURL: memoryURL, defaults: defaults)
            memory.isEnabled = true
            _ = memory.upsert(content: "Prefers green tea", conversationID: UUID())
            let before = snapshot(of: home)
            #expect(!before.isEmpty)

            let document = sources.appendingPathComponent("notes.txt")
            try "Quarterly numbers, at some length.".write(to: document, atomically: true, encoding: .utf8)

            let session = makeSession()
            let attachment = try ChatFileAttachment(
                contentsOf: document, cache: session.documentCache
            )
            session.chat.send(
                "Summarise this", alias: "test-model", fileAttachments: [attachment]
            )
            try await waitForTurn(session.chat)
            session.chat.regenerateLast(alias: "test-model")
            try await waitForTurn(session.chat)

            let sessionDirectory = TemporaryChatStorage.directory(for: session.id)
            #expect(FileManager.default.fileExists(atPath: sessionDirectory.path))

            session.close()
            ConversationStore.flush()
            ConversationDraftStore.flush()
            ConversationFolderStore.flush()

            #expect(!FileManager.default.fileExists(atPath: sessionDirectory.path))
            #expect(snapshot(of: home) == before)
        }
    }

    @Test("Attachments go to the chat's own cache, never the shared one")
    func attachmentsStayInTheWindowCache() throws {
        let root = try scratchDirectory()
        let sources = try scratchDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: sources)
        }
        let session = makeSession(storageRoot: root)
        defer { session.close() }

        let document = sources.appendingPathComponent("notes.txt")
        try "Private text.".write(to: document, atomically: true, encoding: .utf8)
        let attachment = try ChatFileAttachment(contentsOf: document, cache: session.documentCache)

        #expect(session.documentCache.get(attachment.id) != nil)
        #expect(DocumentContentCache.shared.get(attachment.id) == nil)
        let files = snapshot(of: root).keys
        #expect(!files.isEmpty)
        #expect(
            files.allSatisfy { $0.hasPrefix("/\(session.id.uuidString)/documents/") },
            "unexpected paths: \(Array(files))"
        )
    }

    @Test("read_document and browse use the caches the registry was given")
    func registryUsesInjectedCaches() async throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        var seenDocuments: DocumentContentCache?
        var seenPages: BrowseContentCache?
        let session = makeSession(storageRoot: root) { documents, pages in
            seenDocuments = documents
            seenPages = pages
            return BuiltinToolRegistry(documentCache: documents, browseCache: pages)
        }
        defer { session.close() }
        #expect(seenDocuments === session.documentCache)
        #expect(seenPages != nil)

        let id = UUID()
        session.documentCache.put(
            id, entry: DocumentContentCache.Entry(filename: "a.txt", text: "session-private body")
        )
        let registry = BuiltinToolRegistry(documentCache: session.documentCache)
        let found = await registry.run(ToolCall(
            id: "1", name: "read_document", arguments: #"{"document_id":"\#(id.uuidString)"}"#
        ))
        #expect(found.content.contains("session-private body"))

        let shared = await BuiltinToolRegistry().run(ToolCall(
            id: "2", name: "read_document", arguments: #"{"document_id":"\#(id.uuidString)"}"#
        ))
        #expect(shared.isError)
    }

    // MARK: - Lifecycle

    @Test("The launch sweep removes leftovers from a crashed run")
    func sweepRemovesLeftovers() throws {
        let appSupport = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: appSupport) }
        let root = TemporaryChatStorage.root(applicationSupport: appSupport)
        let leftover = TemporaryChatStorage.directory(for: UUID(), in: root)
        try FileManager.default.createDirectory(at: leftover, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: leftover.appendingPathComponent("extract.json"))

        TemporaryChatStorage.removeAll(in: root)

        #expect(!FileManager.default.fileExists(atPath: root.path))
    }

    @Test("Closing one session deletes only its own directory")
    func closeIsPerWindow() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = makeSession(storageRoot: root)
        let second = makeSession(storageRoot: root)
        defer { second.close() }
        for session in [first, second] {
            let directory = TemporaryChatStorage.directory(for: session.id, in: root)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("x".utf8).write(to: directory.appendingPathComponent("f"))
        }

        first.close()
        first.close()

        #expect(!FileManager.default.fileExists(
            atPath: TemporaryChatStorage.directory(for: first.id, in: root).path))
        #expect(FileManager.default.fileExists(
            atPath: TemporaryChatStorage.directory(for: second.id, in: root).path))
    }

    @Test("Closing cancels a turn that is still streaming")
    func closeStopsInFlightWork() async throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let session = makeSession(storageRoot: root, protocolClass: TemporaryChatStallProtocol.self)
        session.chat.send("Hello", alias: "test-model")
        #expect(session.chat.isStreaming)

        session.close()
        for _ in 0..<300 where session.chat.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!session.chat.isStreaming)
    }

    // MARK: - Instructions, memory, backend

    @Test("Global instructions and memory are absent from the request; base and date remain")
    func requestCarriesNeitherInstructionsNorMemory() async throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let session = makeSession(storageRoot: root)
        defer { session.close() }
        #expect(session.chat.memoryStore == nil)
        #expect(session.chat.customInstructions.global.isEmpty)

        session.chat.send("Hello", alias: "test-model")
        try await waitForTurn(session.chat)

        let system = try requestSystemPrompt()
        #expect(system.hasPrefix(ChatViewModel.baseAssistantInstruction + "\n\n[CURRENT DATE]"))
        #expect(!system.contains("[GLOBAL USER INSTRUCTIONS]"))
        #expect(!system.contains("[CONVERSATION INSTRUCTIONS"))
    }

    @Test("The ephemeral instructions config never touches preferences")
    func ephemeralInstructionsDoNotPersist() {
        let key = CustomInstructionsConfig.storageKey
        let before = UserDefaults.standard.string(forKey: key)

        let config = CustomInstructionsConfig.ephemeral()
        #expect(config.global.isEmpty)
        config.global = "Always answer in rhyme."

        #expect(config.global == "Always answer in rhyme.")
        #expect(UserDefaults.standard.string(forKey: key) == before)
    }

    @Test("The backend is pinned to legacy whatever the shared setting says")
    func backendIsPinnedToLegacy() throws {
        let suite = "rapid-temp-chat-backend-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let shared = ChatBackendConfig(defaults: defaults)
        shared.chatBackend = .ownerV1
        let key = ChatBackendConfig.storageKey
        let before = UserDefaults.standard.string(forKey: key)

        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let session = makeSession(storageRoot: root)
        defer { session.close() }
        let pinned = ChatBackendConfig(pinnedTo: .legacy)
        pinned.chatBackend = .ownerV1

        #expect(!session.chat.isOwnerBackedChat)
        #expect(shared.chatBackend == .ownerV1)
        #expect(UserDefaults.standard.string(forKey: key) == before)
    }

    // MARK: - Toggle and exit warning

    private func makeController(root: URL) -> TemporaryChatController {
        TemporaryChatController(
            server: nil, sampling: nil, storageRoot: root,
            makeTools: { _, _ in EmptyToolRegistry() }
        )
    }

    private func use(_ controller: TemporaryChatController) {
        controller.session?.chat.devSeedMessages([
            ChatMessage(role: .user, content: "question"),
            ChatMessage(role: .assistant, content: "answer"),
        ])
    }

    @Test("The toggle turns temporary mode on and off; an unused chat leaves without a warning")
    func toggleRoundTripWithoutInteraction() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = makeController(root: root)
        #expect(!controller.isActive)

        controller.toggle()
        #expect(controller.isActive)
        #expect(!controller.hasInteraction)

        controller.toggle()
        #expect(!controller.isActive)
        #expect(!controller.isConfirmingExit)
    }

    @Test("A used chat asks before it is lost, and Cancel keeps it")
    func usedChatWarnsAndCancelKeepsIt() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = makeController(root: root)
        controller.toggle()
        use(controller)
        #expect(controller.hasInteraction)

        var ran = false
        controller.leave { ran = true }
        #expect(controller.isConfirmingExit)
        #expect(controller.isActive)
        #expect(!ran)

        controller.cancelExit()
        #expect(controller.isActive)
        #expect(!controller.isConfirmingExit)
        #expect(!ran)
    }

    @Test("Discarding deletes the chat's files and then runs the pending action")
    func discardRunsTheAction() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = makeController(root: root)
        controller.toggle()
        use(controller)
        let id = try #require(controller.session?.id)
        let directory = TemporaryChatStorage.directory(for: id, in: root)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: directory.appendingPathComponent("f"))

        var ran = false
        controller.leave { ran = true }
        controller.confirmDiscard()

        #expect(!controller.isActive)
        #expect(ran)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Leaving with nothing active just runs the action")
    func leaveWhenInactiveRunsImmediately() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = makeController(root: root)
        var ran = false
        controller.leave { ran = true }
        #expect(ran)
        #expect(!controller.isConfirmingExit)
    }

    @Test("The export snapshot carries the visible transcript")
    func exportSnapshotHasTheTranscript() throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = makeController(root: root)
        controller.toggle()
        use(controller)
        let snapshot = try #require(controller.session?.exportSnapshot())

        #expect(snapshot.messages.map(\.content) == ["question", "answer"])
        #expect(ConversationExport.markdown(snapshot).contains("answer"))
    }

    // MARK: - Ordinary chats

    @Test("An ordinary chat still persists and keeps its memory store")
    func ordinaryChatStillPersists() async throws {
        let root = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appendingPathComponent("conversations.json")
        let suite = "rapid-temp-chat-memory-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let memory = MemoryStore(fileURL: root.appendingPathComponent("memory.json"), defaults: defaults)

        TemporaryChatCaptureProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TemporaryChatCaptureProtocol.self]
        let chat = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://temporary-chat")!,
                session: URLSession(configuration: config)
            ),
            memoryStore: memory,
            conversationStoreURL: store
        )
        chat.send("Remember this chat", alias: "test-model")
        try await waitForTurn(chat)
        ConversationStore.flush()

        #expect(chat.memoryStore === memory)
        #expect(FileManager.default.fileExists(atPath: store.path))
        #expect(chat.conversations.count == 1)
    }
}

private final class TemporaryChatCaptureProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() { lastRequestBody = nil }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lastRequestBody = Self.bodyData(from: request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func bodyData(from request: URLRequest) -> Data? {
        guard let stream = request.httpBodyStream else { return request.httpBody }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}

/// Accepts the request and never answers, so the turn stays in flight.
private final class TemporaryChatStallProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {}
    override func stopLoading() {}
}
