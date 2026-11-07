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
