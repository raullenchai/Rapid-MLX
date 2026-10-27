import Foundation

typealias ChatTurnEventHandler = @Sendable (ChatWireEnvelope) async throws -> Void

/// Streaming calls await the consumer, so the reducer controls backpressure and
/// can stop a gapped stream before any cursor is advanced. Cancelling the reader
/// only detaches; cancel(turnID:) is the explicit owner operation.
protocol ChatTurnClient: Sendable {
    func createConversation(workspace: String) async throws -> ChatWireJSON
    func conversation(id: String) async throws -> ChatWireJSON
    func submit(_ request: ChatTurnRequest, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws
    func stream(turnID: String, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws
    func turn(turnID: String) async throws -> ChatWireJSON
    func cancel(turnID: String) async throws -> ChatWireJSON
    func respond(_ response: ChatPermissionResponse) async throws -> ChatWireJSON
    func operation(opID: String) async throws -> ChatWireJSON
    func context(_ request: ChatTurnRequest) async throws -> ChatWireContextEvent
    func compact(_ request: ChatTurnRequest, instructions: String) async throws -> ChatCompactResponse
    func undo(conversationID: String, operationID: String) async throws -> ChatWireJSON
}

extension ChatTurnClient {
    func createConversation(workspace: String) async throws -> ChatWireJSON { throw ChatTurnClientError.invalidResponse }
    func conversation(id: String) async throws -> ChatWireJSON { throw ChatTurnClientError.invalidResponse }
}

struct ChatCompactResponse: Decodable, Equatable, Sendable {
    let context: ChatWireContextEvent
    let notice: String
}

enum ChatTurnClientError: Error, Equatable {
    case http(status: Int)
    case invalidResponse
    case invalidEndpoint
    case invalidCursor
    case disconnected(lastEventID: String)
    case beforeHeaders
}

/// Endpoint and bearer are injected by the future owner bootstrap layer. This
/// client neither discovers nor launches the owner and never runs tools locally.
actor URLSessionChatTurnClient: ChatTurnClient {
    private let endpoint: URL
    private let bearer: String
    private let session: URLSession
    private let redirectDelegate = ChatOwnerRedirectDelegate()

    init(endpoint: URL, bearer: String, session: URLSession? = nil) throws {
        guard ["http", "https"].contains(endpoint.scheme?.lowercased()),
              endpoint.host != nil, endpoint.user == nil, endpoint.password == nil,
              endpoint.query == nil, endpoint.fragment == nil,
              !bearer.isEmpty, !bearer.contains("\r"), !bearer.contains("\n") else {
            throw ChatTurnClientError.invalidEndpoint
        }
        self.endpoint = endpoint
        self.bearer = bearer
        if let session {
            self.session = session
        } else {
            let config = URLSessionConfiguration.ephemeral
            config.timeoutIntervalForRequest = 180
            config.timeoutIntervalForResource = 24 * 60 * 60
            config.urlCache = nil
            config.connectionProxyDictionary = [:]
            self.session = URLSession(configuration: config)
        }
    }

    func createConversation(workspace: String) async throws -> ChatWireJSON {
        try await json(["conversations"], method: "POST", body: try JSONEncoder().encode(["workspace": workspace]))
    }

    func conversation(id: String) async throws -> ChatWireJSON {
        try await json(["conversations", id])
    }

    func submit(_ intent: ChatTurnRequest, after: Int = 0, onEvent: @escaping ChatTurnEventHandler) async throws {
        let body = try validatedBody(intent)
        let request = try makeRequest(["turns"], method: "POST", body: body, turnID: intent.turnID, after: after)
        // Reconcile only a lost response before headers. Once streaming begins,
        // the caller owns the applied cursor and resumes with GET.
        let opened: (URLSession.AsyncBytes, URLResponse)
        do {
            opened = try await session.bytes(for: request, delegate: redirectDelegate)
        } catch let error as URLError where error.code != .cancelled {
            try Task.checkCancellation()
            do {
                _ = try await turn(turnID: intent.turnID)
            } catch ChatTurnClientError.http(status: 404) {
                // Admission may not have happened. Reuse this exact intent/ID.
            } catch is CancellationError { throw CancellationError() }
            catch let error as URLError where error.code == .cancelled { throw error }
            catch { throw ChatTurnClientError.beforeHeaders }
            // The unversioned turn snapshot lacks the original request. Re-POST
            // also checks the server's intent hash instead of attaching to an
            // accidentally reused ID whose conflicting POST response was lost.
            do { opened = try await session.bytes(for: request, delegate: redirectDelegate) }
            catch is CancellationError { throw CancellationError() }
            catch let error as URLError where error.code == .cancelled { throw error }
            catch { throw ChatTurnClientError.beforeHeaders }
        } catch is CancellationError { throw CancellationError() }
        catch let error as URLError where error.code == .cancelled { throw error }
        catch { throw ChatTurnClientError.beforeHeaders }
        try await consume(opened, turnID: intent.turnID, after: after, onEvent: onEvent)
    }

    func stream(turnID: String, after: Int = 0, onEvent: @escaping ChatTurnEventHandler) async throws {
        let request = try makeRequest(["turns", turnID, "events"], turnID: turnID, after: after)
        let opened = try await session.bytes(for: request, delegate: redirectDelegate)
        try await consume(opened, turnID: turnID, after: after, onEvent: onEvent)
    }

    func turn(turnID: String) async throws -> ChatWireJSON {
        try await json(["turns", turnID])
    }

    func cancel(turnID: String) async throws -> ChatWireJSON {
        try await json(["turns", turnID, "cancel"], method: "POST", body: Data("{}".utf8))
    }

    func respond(_ response: ChatPermissionResponse) async throws -> ChatWireJSON {
        let body = try JSONEncoder().encode(response)
        _ = try JSONDecoder().decode(ChatPermissionResponse.self, from: body)
        return try await json(["permission.response"], method: "POST", body: body)
    }

    func operation(opID: String) async throws -> ChatWireJSON {
        try await json(["operations", opID])
    }

    func context(_ request: ChatTurnRequest) async throws -> ChatWireContextEvent {
        try await json(["conversations", request.conversationID, "context"], method: "POST", body: validatedBody(request))
    }

    func compact(_ request: ChatTurnRequest, instructions: String = "") async throws -> ChatCompactResponse {
        struct Body: Encodable { let request: ChatTurnRequest; let instructions: String }
        _ = try validatedBody(request)
        guard instructions.unicodeScalars.count <= 8192 else { throw ChatWireError.invalidField("instructions") }
        return try await json(
            ["conversations", request.conversationID, "compact"], method: "POST",
            body: JSONEncoder().encode(Body(request: request, instructions: instructions))
        )
    }

    func undo(conversationID: String, operationID: String) async throws -> ChatWireJSON {
        struct Body: Encodable { let operation_id: String }
        guard operationID.range(of: "^[A-Za-z0-9_-]{1,80}$", options: .regularExpression) != nil else {
            throw ChatWireError.invalidField("operation_id")
        }
        return try await json(
            ["conversations", conversationID, "undo"], method: "POST",
            body: JSONEncoder().encode(Body(operation_id: operationID))
        )
    }

    private func validatedBody(_ request: ChatTurnRequest) throws -> Data {
        let body = try JSONEncoder().encode(request)
        _ = try JSONDecoder().decode(ChatTurnRequest.self, from: body)
        return body
    }

    private func makeRequest(
        _ path: [String], method: String = "GET", body: Data? = nil,
        turnID: String? = nil, after: Int = 0
    ) throws -> URLRequest {
        guard after >= 0 else { throw ChatTurnClientError.invalidCursor }
        var url = endpoint.appendingPathComponent("v1/rapid/chat")
        for component in path {
            // IDs remain opaque; disallow path syntax rather than silently
            // treating an ID as an additional route or normalized dot segment.
            guard !component.isEmpty, component != ".", component != "..",
                  !component.contains("/"), !component.contains("\\") else {
                throw ChatTurnClientError.invalidEndpoint
            }
            url.appendPathComponent(component)
        }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 180)
        request.httpMethod = method
        request.httpBody = body
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        request.setValue(turnID == nil ? "application/json" : "text/event-stream", forHTTPHeaderField: "Accept")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        if let turnID {
            guard !turnID.contains("\r"), !turnID.contains("\n") else { throw ChatTurnClientError.invalidCursor }
            request.setValue("\(turnID):\(after)", forHTTPHeaderField: "Last-Event-ID")
        }
        return request
    }

    private func validate(_ response: URLResponse, contentType: String) throws {
        guard let http = response as? HTTPURLResponse else { throw ChatTurnClientError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw ChatTurnClientError.http(status: http.statusCode) }
        guard http.mimeType?.lowercased() == contentType else { throw ChatTurnClientError.invalidResponse }
    }

    private func json<T: Decodable>(_ path: [String], method: String = "GET", body: Data? = nil) async throws -> T {
        let request = try makeRequest(path, method: method, body: body)
        let (data, response) = try await session.data(for: request, delegate: redirectDelegate)
        try validate(response, contentType: "application/json")
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func consume(
        _ opened: (URLSession.AsyncBytes, URLResponse), turnID: String, after: Int,
        onEvent: @escaping ChatTurnEventHandler
    ) async throws {
        let (bytes, response) = opened
        // Explicitly release the read on malformed/status responses, consumer
        // failure, terminal, or task cancellation. This never calls owner cancel.
        defer { bytes.task.cancel() }
        try validate(response, contentType: "text/event-stream")
        var decoder = ChatSSEDecoder()
        var cursor = after
        for try await byte in bytes {
            try Task.checkCancellation()
            if let event = try decoder.append(byte) {
                guard event.turnID == turnID else { throw ChatWireError.wrongTurn }
                try await onEvent(event)
                cursor = max(cursor, event.seq)
                if case .turnTerminal = event.event { return }
            }
        }
        try decoder.finish()
        throw ChatTurnClientError.disconnected(lastEventID: "\(turnID):\(cursor)")
    }
}

/// Owner requests are never redirected to another route/origin with a bearer.
private final class ChatOwnerRedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
