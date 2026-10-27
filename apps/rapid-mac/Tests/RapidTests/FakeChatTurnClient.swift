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
        try await deliver(after: after, onEvent: onEvent)
    }
    func stream(turnID: String, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws {
        calls.append(.stream(turnID, after: after))
        try await deliver(after: after, onEvent: onEvent)
    }
    func turn(turnID: String) async throws -> ChatWireJSON { calls.append(.turn(turnID)); return receipt }
    func cancel(turnID: String) async throws -> ChatWireJSON { calls.append(.cancel(turnID)); return receipt }
    func respond(_ response: ChatPermissionResponse) async throws -> ChatWireJSON { calls.append(.respond(response)); return receipt }
    func operation(opID: String) async throws -> ChatWireJSON { calls.append(.operation(opID)); return receipt }
    func context(_ request: ChatTurnRequest) async throws -> ChatWireContextEvent {
        calls.append(.context(request)); return contextValue
    }
    func compact(_ request: ChatTurnRequest, instructions: String) async throws -> ChatCompactResponse {
        calls.append(.compact(request, instructions)); return ChatCompactResponse(context: contextValue, notice: "Compacted")
    }
    func undo(conversationID: String, operationID: String) async throws -> ChatWireJSON {
        calls.append(.undo(conversationID, operationID)); return receipt
    }

    private func deliver(after: Int, onEvent: ChatTurnEventHandler) async throws {
        var decoder = ChatSSEDecoder()
        for frame in frames {
            try Task.checkCancellation()
            for event in try decoder.append(frame) where event.seq > after { try await onEvent(event) }
        }
        try decoder.finish()
    }
}
