import Foundation
@testable import Rapid

/// Deterministic owner boundary for adapter tests. Transport-free fake uses the
/// same byte decoder as the URLSession client and records explicit mutations.
actor FakeChatTurnClient: ChatTurnClient {
    enum Call: Equatable {
        case submit(ChatTurnRequest, after: Int)
        case stream(String, after: Int)
        case turn(String), cancel(String), respond(ChatPermissionResponse), operation(String)
        case context(ChatTurnRequest), compact(ChatTurnRequest, String), undo(String, String)
    }

    var deliveryError: ChatTurnClientError?
    var operationError: ChatTurnClientError?
    var responseError: ChatTurnClientError?
    var cancelError: ChatTurnClientError?
    var submitFailures = 0
    var malformedConversation = false
    func failSubmissions(_ count: Int) { submitFailures = count }
    func failCancel() { cancelError = .beforeHeaders }
    func setMalformedConversation() { malformedConversation = true }
    func setDeliveryError(_ error: ChatTurnClientError?) { deliveryError = error }
    func setResponseError(_ error: ChatTurnClientError?) { responseError = error }
    func setOperationError(_ error: ChatTurnClientError?) { operationError = error }
    var substituteTurnID = false
    func enableTurnSubstitution() { substituteTurnID = true }
    func setFrames(_ value: [Data]) { frames = value }
    func setReceipt(_ value: ChatWireJSON) { receipt = value }
    func createConversation(workspace: String) async throws -> ChatWireJSON { if malformedConversation { return .object(["id": .string("c")]) }; return .object(["id": .string("c"), "branch_id": .string("b"), "branches": .array([.object(["id": .string("b"), "revision": .integer(0)])])]) }
    func conversation(id: String) async throws -> ChatWireJSON { try await createConversation(workspace: "") }
    private(set) var calls: [Call] = []
    var frames: [Data]
    let contextValue: ChatWireContextEvent
    var receipt: ChatWireJSON = .object([:])

    init(frames: [Data], context: ChatWireContextEvent) {
        self.frames = frames
        contextValue = context
    }

    func submit(_ request: ChatTurnRequest, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws {
        calls.append(.submit(request, after: after))
        if submitFailures > 0 { submitFailures -= 1; throw ChatTurnClientError.beforeHeaders }
        try await deliver(turnID: request.turnID, after: after, onEvent: onEvent)
    }
    func stream(turnID: String, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws {
        calls.append(.stream(turnID, after: after))
        try await deliver(turnID: turnID, after: after, onEvent: onEvent)
    }
    func turn(turnID: String) async throws -> ChatWireJSON { calls.append(.turn(turnID)); return receipt }
    func cancel(turnID: String) async throws -> ChatWireJSON { calls.append(.cancel(turnID)); if let cancelError { throw cancelError }; return receipt }
    func respond(_ response: ChatPermissionResponse) async throws -> ChatWireJSON { calls.append(.respond(response)); if let responseError { throw responseError }; return receipt }
    func operation(opID: String) async throws -> ChatWireJSON { calls.append(.operation(opID)); if let operationError { throw operationError }; return receipt }
    func context(_ request: ChatTurnRequest) async throws -> ChatWireContextEvent {
        calls.append(.context(request)); return contextValue
    }
    func compact(_ request: ChatTurnRequest, instructions: String) async throws -> ChatCompactResponse {
        calls.append(.compact(request, instructions)); return ChatCompactResponse(context: contextValue, notice: "Compacted")
    }
    func undo(conversationID: String, operationID: String) async throws -> ChatWireJSON {
        calls.append(.undo(conversationID, operationID)); return receipt
    }

    private func deliver(turnID: String, after: Int, onEvent: ChatTurnEventHandler) async throws {
        if let deliveryError { throw deliveryError }
        var decoder = ChatSSEDecoder()
        for frame in frames {
            try Task.checkCancellation()
            let data = substituteTurnID ? Data(String(decoding: frame, as: UTF8.self).replacingOccurrences(of: "turn-1", with: turnID).utf8) : frame
            for event in try decoder.append(data) where event.seq > after { try await onEvent(event) }
        }
        try decoder.finish()
    }
}
