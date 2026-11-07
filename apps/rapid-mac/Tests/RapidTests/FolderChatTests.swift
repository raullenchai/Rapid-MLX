import Foundation
import Testing
@testable import Rapid

@Suite("Explicit folder chat") @MainActor
struct FolderChatTests {
    @Test func defaultsRequireExplicitOptIn() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let config = ChatBackendConfig(defaults: defaults)
        let model = ChatViewModel(chatBackendConfig: config, persistsConversations: false)
        #expect(!model.folderChatAvailable)
        model.startFolderChat(folderID: "folder", title: "Policies")
        #expect(model.folderChatScope == nil)
    }
    @Test func scopeFollowsConversationAndNewChatClearsIt() async throws {
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = true
        let client = try await ChatViewModelOwnerRoutingTests().fake()
        await client.enableFolderChat()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.startFolderChat(folderID: "folder", title: "Policies")
        let id = model.activeConversationID
        model.send("Question", alias: "model")
        await model._testingWaitForCurrentTurn()
        let request = try #require(await client.calls.compactMap { call -> ChatTurnRequest? in if case .submit(let request, _) = call { request } else { nil } }.first)
        #expect(request.localSearch?.folderIDs == ["folder"])
        #expect(!request.actionTools)
        #expect(await client.folderScopes == [FolderChatScope(folderIDs: ["folder"])])
        model.newConversation()
        #expect(model.folderChatScope == nil)
        model.selectConversation(id)
        #expect(model.folderChatScope?.folderIDs == ["folder"])
        #expect(model.folderChatTitle == "Policies")
        model.removeFolderChatScope()
        #expect(model.folderChatScope == nil)
    }
    @Test func oldOwnerDoesNotReceiveFolderTurn() async throws {
        let client = try await ChatViewModelOwnerRoutingTests().fake()
        let session = OwnerChatSession(client: client)
        do {
            try await session.submit(text: "Question", target: .init(url: URL(string: "http://localhost:8000")!, bearer: nil, model: "m"), localSearch: .init(folderIDs: ["folder"]))
            Issue.record("Expected a capability rejection")
        } catch is ChatOwnerTurnRejection {}
        #expect(await client.calls.isEmpty)
        await session.close()
    }
    @Test func folderScopeRequestIsAdditiveAndStrict() throws {
        let ordinary = ChatTurnRequest(turnID: "t", conversationID: "c", branchID: "b", expectedRevision: 0, inferenceURL: "http://localhost:8000", model: "m")
        let object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(ordinary)) as? [String: Any])
        #expect(object["local_search"] == nil)
        for invalid in [#"{"folder_ids":[],"limit":3}"#, #"{"folder_ids":["f","f"],"limit":3}"#, #"{"folder_ids":["f"],"limit":11}"#, #"{"folder_ids":["f"],"path":"/private"}"#] {
            #expect(throws: (any Error).self) { try JSONDecoder().decode(FolderChatScope.self, from: Data(invalid.utf8)) }
        }
    }
    @Test func sourceReceiptRoundTripsWithoutRawAttachmentInTranscript() throws {
        let source = #"{"type":"source","source_id":"local-search:turn:1","call_id":"local-search:turn","title":"policy.txt · lines 3–4","attachment_id":"blob","url":null}"#
        var reducer = ChatTurnReducer(turnID: "turn")
        let envelope = "{\"turn_id\":\"turn\",\"seq\":1,\"event\":\(source)}"
        _ = try reducer.apply(JSONDecoder().decode(ChatWireEnvelope.self, from: Data(envelope.utf8)))
        let rows = ownerProjectionMessages(reducer.projection, assistantID: UUID(), createdAt: Date())
        let row = try #require(rows.first(where: { $0.ownerSource != nil }))
        #expect(row.content == "policy.txt · lines 3–4")
        #expect(row.wireVisibility == .transcriptOnly)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(row))
        #expect(restored.ownerSource?.attachmentID == "blob")
        #expect(restored == row)
    }
}

@Suite("Folder chat native interaction") @MainActor
struct FolderChatInteractionTests {
    @Test(arguments: [520.0, 720.0]) func savedSourceOpensOwnedPreviewAndScopeCanBeRemoved(width: Double) async throws {
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = true
        let fixture = OwnerChatSessionTests()
        let source = #"{"type":"source","source_id":"local-search:turn-1:1","call_id":"local-search:turn-1","title":"policy.txt · lines 3–4","attachment_id":"saved-excerpt","url":null}"#
        let client = try await fixture.fake([
            fixture.frame(source, 1),
            fixture.frame(#"{"type":"text.delta","part_id":"p","text":"The refund window is 30 days [1]."}"#, 2),
            fixture.frame(#"{"type":"turn.terminal","outcome":"completed"}"#, 3),
        ])
        await client.enableFolderChat()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.startFolderChat(folderID: "policies", title: "Policies")
        model.send("What is the refund window?", alias: "local-fixture")
        await model._testingWaitForCurrentTurn()
        let stage = GoldenStage(
            ChatView(viewModel: model, server: ServerManager(testingState: .idle), alias: .constant("local-fixture"), readiness: .ready(alias: "local-fixture"))
                .environment(DownloadManager()).environment(QuickstartCoordinator()),
            size: CGSize(width: width, height: 620)
        )
        try await stage.waitForIdentifier("Chat.FolderScope")
        let sourceID = try #require(stage.identifier(withPrefix: "Chat.SavedSource."))
        #expect(stage.treeText().contains("policy.txt"))
        #expect(!stage.treeText().contains("saved-excerpt"))
        if let directory = ProcessInfo.processInfo.environment["RAPID_FOLDER_REVIEW_DIR"] {
            try stage.writeScreenshot(to: URL(fileURLWithPath: directory).appendingPathComponent("folder-chat-\(Int(width)).png"))
        }
        try stage.press(sourceID)
        try await stage.waitForIdentifier("Chat.SavedSourcePreview")
        try await stage.waitForText("Saved evidence, not the current file.")
        if let directory = ProcessInfo.processInfo.environment["RAPID_FOLDER_REVIEW_DIR"] {
            try stage.writeScreenshot(to: URL(fileURLWithPath: directory).appendingPathComponent("folder-source-\(Int(width)).png"), presentation: true)
        }
        #expect(await client.sourcePreviewCalls.first?.0 == "c")
        #expect(await client.sourcePreviewCalls.first?.1 == "saved-excerpt")
        stage.pressEscape()
        try await stage.waitForIdentifierGone("Chat.SavedSourcePreview")
        try await stage.waitForIdentifier("Chat.RemoveFolderScope")
        try stage.press("Chat.RemoveFolderScope")
        try await stage.waitForIdentifierGone("Chat.FolderScope")
        #expect(model.folderChatScope == nil)
        #expect(model.messages.contains { $0.ownerSource != nil })
    }
}

@Suite("Folder preparation cancellation")
struct FolderChatPreparationTests {
    actor Gate {
        var entered = false
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async { entered = true; await withCheckedContinuation { continuation = $0 } }
        func release() { continuation?.resume(); continuation = nil }
    }
    @Test(arguments: [true, false]) func stopDuringCapabilityOrScopeSaveCannotSubmit(capability: Bool) async throws {
        let client = try await OwnerChatSessionTests().fake([])
        await client.enableFolderChat()
        let gate = Gate()
        await client.setFolderGate(capability: capability) { await gate.wait() }
        let owner = OwnerChatSession(client: client)
        let task = Task { try await owner.submit(text: "Question", target: .init(url: URL(string: "http://localhost:8000")!, bearer: nil, model: "m"), localSearch: .init(folderIDs: ["f"])) }
        for _ in 0..<100 { if await gate.entered { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(await gate.entered)
        try await owner.stop()
        await gate.release()
        do { try await task.value; Issue.record("Expected cancellation") } catch is CancellationError {}
        #expect(!(await client.calls.contains { if case .submit = $0 { true } else { false } }))
        #expect(await client.folderScopes.isEmpty == capability)
        await owner.close()
    }
}

/// Opt-in dogfood against an owned real inference/owner pair. The external
/// runner holds the project's GPU lock and prohibits downloads for its lifetime.
@Suite("Folder chat real local user path") @MainActor
struct FolderChatRealUserTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_FOLDER_REAL_OWNER_URL"] != nil))
    func composerToRealOwnerAndSavedPassage() async throws {
        let env = ProcessInfo.processInfo.environment
        let ownerURLString = try #require(env["RAPID_FOLDER_REAL_OWNER_URL"])
        let ownerURL = try #require(URL(string: ownerURLString))
        let portString = try #require(env["RAPID_FOLDER_REAL_INFERENCE_PORT"])
        let port = try #require(Int(portString))
        let folderID = try #require(env["RAPID_FOLDER_REAL_FOLDER_ID"])
        let directory = try #require(env["RAPID_FOLDER_REVIEW_DIR"])
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = true
        let client = try URLSessionChatTurnClient(endpoint: ownerURL, bearer: "owned-folder-test")
        let server = ServerManager(testingState: .ready(alias: "default"), activePort: port, activeBearer: "owned-inference-test", sessionDefaults: UserDefaults(suiteName: UUID().uuidString)!)
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, ownerWorkspace: URL(fileURLWithPath: directory), server: server, persistsConversations: false)
        model.startFolderChat(folderID: folderID, title: "Owned policy documents")
        let stage = GoldenStage(ChatView(viewModel: model, server: server, alias: .constant("default"), readiness: .ready(alias: "default"))
            .environment(DownloadManager()).environment(QuickstartCoordinator()), size: CGSize(width: 720, height: 680))
        try await stage.waitForIdentifier(AutosizingTextView.composeAccessibilityIdentifier)
        try stage.setValue("Can hand-dyed yarn wound into a ball be returned?", for: AutosizingTextView.composeAccessibilityIdentifier)
        try stage.submitComposer()
        await model._testingWaitForCurrentTurn()
        #expect(model.lastError == nil)
        let answer = model.messages.filter { $0.role == .assistant }.map(\.content).joined(separator: "\n")
        #expect(answer.lowercased().contains("cannot") || answer.lowercased().contains("no"))
        #expect(answer.contains("[1]"))
        try await stage.waitForText(answer)
        let source = try #require(model.messages.compactMap(\.ownerSource).first)
        #expect(source.url == nil)
        let sourceID = "Chat.SavedSource." + source.sourceID
        try await stage.waitForIdentifier(sourceID)
        try stage.writeScreenshot(to: URL(fileURLWithPath: directory).appendingPathComponent("real-folder-answer.png"))
        try stage.press(sourceID)
        try await stage.waitForIdentifier("Chat.SavedSourcePreview")
        try await stage.waitForText("cannot be returned")
        let saved = try await model.folderSourcePreview(source)
        #expect(saved.text.contains("Supporting policy context."))
        #expect(saved.text.contains("cannot be returned"))
        stage.pressEscape()
        try await stage.waitForIdentifierGone("Chat.SavedSourcePreview")
        let receipt: [String: String] = ["answer": answer, "source_id": source.sourceID, "attachment_id": source.attachmentID ?? "", "saved_text": saved.text, "owner_conversation_id": model.conversations.first?.ownerConversationID ?? ""]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: directory).appendingPathComponent("real-native-receipt.json"))
        var formats: [[String: String]] = []
        for (question, expected, filename) in [
            ("How many business days do orders take to ship?", "two", "shipping.pdf"),
            ("Do office visits require an appointment?", "yes", "office-policy.docx")
        ] {
            let oldIDs = Set(model.messages.compactMap(\.ownerSource).map(\.sourceID))
            try stage.setValue(question, for: AutosizingTextView.composeAccessibilityIdentifier)
            try stage.submitComposer()
            await model._testingWaitForCurrentTurn()
            let response = try #require(model.messages.last(where: { $0.role == .assistant })?.content)
            #expect(response.lowercased().contains(expected) || (expected == "two" && response.contains("2")))
            try await stage.waitForText(response)
            let document = try #require(model.messages.compactMap(\.ownerSource).first(where: { !oldIDs.contains($0.sourceID) && $0.title.contains(filename) }))
            try stage.press("Chat.SavedSource." + document.sourceID)
            try await stage.waitForIdentifier("Chat.SavedSourcePreview")
            let preview = try await model.folderSourcePreview(document)
            #expect(preview.text.contains(expected == "two" ? "two business days" : "appointment"))
            formats.append(["question": question, "answer": response, "source_id": document.sourceID, "title": document.title, "saved_text": preview.text])
            stage.pressEscape()
            try await stage.waitForIdentifierGone("Chat.SavedSourcePreview")
        }
        let emptyFolderID = try #require(env["RAPID_FOLDER_REAL_EMPTY_FOLDER_ID"])
        model.startFolderChat(folderID: emptyFolderID, title: "Owned empty folder")
        try stage.setValue("zqxvjkplrnop", for: AutosizingTextView.composeAccessibilityIdentifier)
        try stage.submitComposer()
        await model._testingWaitForCurrentTurn()
        #expect(model.messages.last(where: { $0.role == .assistant })?.content.contains("couldn't find supporting information") == true)
        try JSONSerialization.data(withJSONObject: formats, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: directory).appendingPathComponent("real-format-receipts.json"))
        try Data((model.conversations.first?.ownerConversationID ?? "").utf8).write(to: URL(fileURLWithPath: directory).appendingPathComponent("real-nohit-conversation.txt"))

    }
}

@Suite("Folder search entry point") @MainActor
struct FolderChatEntryPointTests {
    @Test(arguments: [false, true]) func explicitSetupOrFolderSelection(enabled: Bool) async throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let status = try decoder.decode(LocalSearchStatus.self, from: Data(#"{"enabled":true,"model":{"id":"embedding","downloaded":true,"loaded":false},"folders":[{"id":"selected","path":"/owned/Policies","state":"idle","files_indexed":1,"files_skipped":0,"chunks":1}],"indexing":{"active":false,"files_done":1,"files_total":1},"totals":{"files":1,"chunks":1,"index_bytes":0}}"#.utf8))
        let search = LocalSearchViewModel(fetchStatus: { _ in status })
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = enabled
        let chat = ChatViewModel(chatBackendConfig: config, persistsConversations: false)
        var settingsOpened = false
        let stage = GoldenStage(LocalSearchView(server: ServerManager(testingState: .ready(alias: "local")), model: search, onStartModel: { false }, onAskFolder: { folder in
            chat.startFolderChat(folderID: folder.id, title: LocalSearchPath.folderTitle(folder.path))
        }, folderChatAvailable: chat.folderChatAvailable, onFolderChatSettings: { settingsOpened = true }).environment(DownloadManager()), size: CGSize(width: 720, height: 620))
        try await stage.waitForIdentifier("LocalSearch.Folder.0.Ask")
        try stage.press("LocalSearch.Folder.0.Ask")
        #expect(settingsOpened == !enabled)
        #expect(config.harnessEnabled == enabled)
        #expect(chat.messages.isEmpty)
        #expect(chat.folderChatScope?.folderIDs == (enabled ? ["selected"] : nil))
        search.disappear()
    }
}
