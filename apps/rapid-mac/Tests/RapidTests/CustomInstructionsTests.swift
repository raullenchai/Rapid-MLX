import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Custom instructions")
struct CustomInstructionsTests {
    private static func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

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

    @Test("Instruction layers are bounded before persistence and wire use")
    func instructionLengthBound() {
        let overlong = String(repeating: "x", count: CustomInstructionsConfig.maximumLength + 10)
        #expect(CustomInstructionsConfig.limited(overlong).count == 4_000)
        #expect(CustomInstructionsConfig.normalized(overlong)?.count == 4_000)

        let (defaults, name) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        defaults.set(overlong, forKey: CustomInstructionsConfig.storageKey)
        #expect(CustomInstructionsConfig(defaults: defaults).global.count == 4_000)
        let config = CustomInstructionsConfig(defaults: defaults)
        config.global = overlong
        #expect(config.global.count == 4_000)
        #expect(defaults.string(forKey: CustomInstructionsConfig.storageKey)?.count == 4_000)
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
        #expect(
            result.first?.content == """
            Ambient

            App system

            [GLOBAL USER INSTRUCTIONS]
            These user preferences apply unless this conversation has a conflicting instruction:
            Global

            [CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]
            These instructions apply only to this conversation. If they conflict with the global user instructions above, follow THESE conversation instructions. They do not override earlier application, safety, or tool instructions:
            Conversation
            """
        )
        #expect(result.filter { $0.role == .system }.count == 1)
        #expect(result.last?.id == user.id)
    }

    @Test("Effective prompt preview uses the wire assembly and includes automatic context")
    func effectivePromptPreviewUsesWireAssembly() {
        let preview = ChatViewModel.effectiveSystemPrompt(
            dateContext: "[CURRENT DATE AND TIME]\nToday is Tuesday, August 25, 2026.",
            global: "Reply in plain language.",
            conversation: "Use bullet points."
        )

        #expect(preview.hasPrefix("[CURRENT DATE AND TIME]"))
        #expect(preview.contains("[GLOBAL USER INSTRUCTIONS]"))
        #expect(preview.contains("Reply in plain language."))
        #expect(preview.contains("[CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]"))
        #expect(preview.contains("Use bullet points."))
        #expect(preview.range(of: "Reply in plain language.")!.lowerBound
            < preview.range(of: "Use bullet points.")!.lowerBound)
    }

    @Test("Effective prompt preview refreshes across midnight and time-zone changes")
    func effectivePromptPreviewFollowsClockAndZone() throws {
        var utc = Calendar(identifier: .gregorian)
        utc.timeZone = try #require(TimeZone(identifier: "UTC"))
        let beforeMidnight = try #require(
            utc.date(from: DateComponents(
                year: 2026, month: 8, day: 25, hour: 23, minute: 59
            ))
        )
        let afterMidnight = beforeMidnight.addingTimeInterval(120)

        let before = EffectiveSystemPromptDisclosure.prompt(
            at: beforeMidnight,
            calendar: utc,
            global: "Global",
            conversation: "Conversation"
        )
        let after = EffectiveSystemPromptDisclosure.prompt(
            at: afterMidnight,
            calendar: utc,
            global: "Global",
            conversation: "Conversation"
        )

        #expect(before.contains("Tuesday, August 25, 2026"))
        #expect(before.contains("11:59 PM (GMT, GMT)"))
        #expect(after.contains("Wednesday, August 26, 2026"))
        #expect(after.contains("12:01 AM (GMT, GMT)"))

        var tokyo = utc
        tokyo.timeZone = try #require(TimeZone(identifier: "Asia/Tokyo"))
        let tokyoPreview = EffectiveSystemPromptDisclosure.prompt(
            at: beforeMidnight,
            calendar: tokyo,
            global: "Global",
            conversation: "Conversation"
        )
        #expect(tokyoPreview.contains("Wednesday, August 26, 2026"))
        #expect(tokyoPreview.contains("8:59 AM (GMT+9, Asia/Tokyo)"))
        #expect(tokyoPreview != before)
    }

    @Test("System prompt UI names global and conversation precedence explicitly")
    func systemPromptTerminologyAndPreviewWiring() throws {
        let settings = try Self.source("Sources/Rapid/UI/SettingsView.swift")
        #expect(settings.contains("case .instructions: return \"System Prompt\""))
        #expect(settings.contains("\"Global default\""))
        #expect(settings.contains("Conversation prompts can override it."))
        #expect(settings.contains("Settings.SystemPrompt.EffectivePreview"))

        let editor = try Self.source("Sources/Rapid/UI/InstructionTextEditor.swift")
        #expect(editor.contains("Text(\"Conversation System Prompt\")"))
        #expect(editor.contains("this prompt wins."))
        #expect(editor.contains("DisclosureGroup(\"Effective System Prompt\""))
        #expect(editor.contains("Tool and attachment context may be added when you send."))
        #expect(editor.contains("TimelineView(.periodic(from: .now, by: 60))"))
        #expect(editor.contains("at: context.date"))
        #expect(editor.contains("calendar: .autoupdatingCurrent"))
        #expect(editor.contains("ChatViewModel.effectiveSystemPrompt"))
        #expect(editor.contains("ChatView.SystemPrompt.EffectivePreview"))

        let chat = try Self.source("Sources/Rapid/UI/ChatView.swift")
        #expect(chat.contains(".accessibilityLabel(\"Conversation system prompt\")"))
        #expect(chat.contains("global: viewModel.customInstructions.global"))
    }

    @Test("Conversation instructions explicitly override conflicting global preferences")
    func conversationLayerHasExplicitPrecedence() {
        let result = ChatViewModel.addingInstructionLayers(
            to: [ChatMessage(role: .user, content: "Test", status: .complete)],
            ambientPreamble: nil,
            global: "Reply only in Simplified Chinese.",
            conversation: "Reply only in English."
        )

        let system = result.first?.content ?? ""
        #expect(system.contains("[GLOBAL USER INSTRUCTIONS]"))
        #expect(system.contains("[CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]"))
        #expect(
            system.contains(
                "If they conflict with the global user instructions above, follow THESE conversation instructions."
            )
        )
        #expect(system.contains("They do not override earlier application, safety, or tool instructions"))
        #expect(system.range(of: "Reply only in Simplified Chinese.")!.lowerBound
            < system.range(of: "Reply only in English.")!.lowerBound)
        #expect(result.filter { $0.role == .system }.count == 1)
    }

    @Test("Send puts both instruction layers in one prioritized wire system message")
    func sendIncludesPrioritizedInstructionsOnWire() async throws {
        CustomInstructionsCaptureProtocol.reset()
        let (defaults, name) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: name) }
        let config = CustomInstructionsConfig(defaults: defaults)
        config.global = "Reply only in Simplified Chinese."
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://custom-instructions")!,
                session: CustomInstructionsCaptureProtocol.session()
            ),
            customInstructions: config,
            persistsConversations: false
        )
        model.setConversationInstructions("Reply only in English.")

        model.send("Test", alias: "test-model")
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        let body = try #require(CustomInstructionsCaptureProtocol.lastRequestBody)
        let json = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        let messages = try #require(json["messages"] as? [[String: Any]])
        #expect(messages.filter { $0["role"] as? String == "system" }.count == 1)
        let system = try #require(messages.first?["content"] as? String)
        #expect(system.contains("Reply only in Simplified Chinese."))
        #expect(system.contains("Reply only in English."))
        #expect(
            system.contains(
                "If they conflict with the global user instructions above, follow THESE conversation instructions."
            )
        )
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
        #expect(result.first?.content.hasPrefix("[GLOBAL USER INSTRUCTIONS]") == true)
        #expect(result.first?.content.contains("Global") == true)
        #expect(result.first?.content.contains("[CONVERSATION INSTRUCTIONS") == true)
        #expect(result.first?.content.contains("Conversation") == true)
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

    @Test("A cleared conversation instruction can be replaced in the same chat")
    func conversationInstructionCanBeClearedAndReadded() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-custom-instructions-readd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appendingPathComponent("conversations.json")
        let (defaults, name) = freshDefaults()
        defer { defaults.removePersistentDomain(forName: name) }

        let model = ChatViewModel(
            customInstructions: CustomInstructionsConfig(defaults: defaults),
            conversationStoreURL: store
        )
        model.setConversationInstructions("First instruction")
        model.send("Create the chat", alias: "test-model")
        model.stopAndPersist()
        let id = model.activeConversationID

        model.setConversationInstructions("")
        #expect(model.conversationInstructions.isEmpty)
        #expect(model.conversations.first { $0.id == id }?.customInstructions == nil)

        model.setConversationInstructions("Replacement instruction")
        #expect(model.conversationInstructions == "Replacement instruction")
        #expect(
            model.conversations.first { $0.id == id }?.customInstructions
                == "Replacement instruction"
        )

        let wire = ChatViewModel.addingInstructionLayers(
            to: [ChatMessage(role: .user, content: "Next turn", status: .complete)],
            ambientPreamble: nil,
            global: model.customInstructions.global,
            conversation: model.conversationInstructions
        )
        #expect(wire.first?.content.contains("[CONVERSATION INSTRUCTIONS") == true)
        #expect(wire.first?.content.contains("Replacement instruction") == true)

        ConversationStore.flush()
        let reloaded = ChatViewModel(
            customInstructions: CustomInstructionsConfig(defaults: defaults),
            conversationStoreURL: store
        )
        reloaded.selectConversation(id)
        #expect(reloaded.conversationInstructions == "Replacement instruction")
    }

    @Test("The conversation instruction editor takes focus when its popover opens")
    func conversationEditorAutoFocusWiring() throws {
        let source = try Self.source("Sources/Rapid/UI/InstructionTextEditor.swift")
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)
        #expect(stripped.contains("@FocusStateprivatevarfocused:Bool"))
        #expect(stripped.contains(".focused($focused)"))
        #expect(
            stripped.contains(
                ".task{guardautoFocuselse{return}awaitTask.yield()guard!Task.isCancelledelse{return}focused=true}"
            )
        )
        #expect(
            stripped.contains(
                "accessibilityIdentifier:\"ChatView.ConversationInstructions.Editor\",autoFocus:true"
            )
        )
    }
}

private final class CustomInstructionsCaptureProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() {
        lastRequestBody = nil
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [CustomInstructionsCaptureProtocol.self]
        return URLSession(configuration: config)
    }

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
