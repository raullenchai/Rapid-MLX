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
