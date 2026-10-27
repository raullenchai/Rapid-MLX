import Foundation

/// Owns one in-memory projection and its inseparable applied cursor.
actor OwnerChatSession {
    struct Target: Sendable {
        let url: URL
        let bearer: String?
        let model: String
    }
    enum State: Equatable, Sendable { case idle, streaming, reconnecting, finished, failed(String) }
    enum SessionError: Error { case busy, invalidConversation, permissionUnavailable }
    struct Snapshot: Sendable {
        let projection: ChatTurnProjection?
        let state: State
        let unavailablePermissions: Set<String>
    }
    private enum Resume: Error { case gap }
    private let clientProvider: @Sendable () async throws -> any ChatTurnClient
    private var client: (any ChatTurnClient)?
    private var reducer: ChatTurnReducer?
    private var state: State = .idle
    private var unavailable: Set<String> = []
    private var observers: [UUID: AsyncStream<Snapshot>.Continuation] = [:]
    private let workspace: URL
    private var conversationID: String?
    private var branchID: String?
    private var task: Task<Void, Never>?
    private var preparing = false
    private var stoppedWhilePreparing = false
    private var resolving: Set<String> = []

    init(connection: ChatOwnerConnection, conversationID: String? = nil, workspace: URL = FileManager.default.homeDirectoryForCurrentUser) {
        self.workspace = workspace.resolvingSymlinksInPath().standardizedFileURL
        self.clientProvider = { try await connection.reconnect() }
        self.conversationID = conversationID
    }
    init(client: any ChatTurnClient, conversationID: String? = nil, workspace: URL = FileManager.default.homeDirectoryForCurrentUser) {
        self.workspace = workspace.resolvingSymlinksInPath().standardizedFileURL
        self.clientProvider = { client }
        self.conversationID = conversationID
    }
    init(clientProvider: @escaping @Sendable () async throws -> any ChatTurnClient) {
        self.workspace = FileManager.default.homeDirectoryForCurrentUser.resolvingSymlinksInPath().standardizedFileURL
        self.clientProvider = clientProvider
    }
    func close() {
        task?.cancel()
        task = nil
        for observer in observers.values { observer.finish() }
        observers.removeAll()
    }
    func ownerConversationID() -> String? { conversationID }
    func snapshot() -> Snapshot { Snapshot(projection: reducer?.projection, state: state, unavailablePermissions: unavailable) }
    func snapshots() -> AsyncStream<Snapshot> {
        let id = UUID()
        return AsyncStream { continuation in
            observers[id] = continuation
            continuation.yield(snapshot())
            continuation.onTermination = { [weak self] _ in Task { await self?.removeObserver(id) } }
        }
    }
    private func removeObserver(_ id: UUID) { observers.removeValue(forKey: id) }
    private func publish() { let value = snapshot(); for observer in observers.values { observer.yield(value) } }

    func submit(text: String, target: Target, onConversation: @Sendable (String) async -> Void = { _ in }) async throws {
        guard task == nil, !preparing else { throw SessionError.busy }
        preparing = true
        stoppedWhilePreparing = false
        defer { preparing = false }
        try checkPreparation()
        let client = try await clientProvider()
        try checkPreparation()
        self.client = client
        let conversation: ChatWireJSON
        if let conversationID { conversation = try await client.conversation(id: conversationID) }
        else { conversation = try await client.createConversation(workspace: workspace.path) }
        try checkPreparation()
        guard case .object(let object) = conversation, case .string(let id) = object["id"] else { throw SessionError.invalidConversation }
        conversationID = id
        await onConversation(id)
        try checkPreparation()
        guard case .string(let branch) = object["branch_id"],
              case .array(let branches) = object["branches"],
              let selected = branches.first(where: { if case .object(let b) = $0 { return b["id"] == .string(branch) }; return false }),
              case .object(let fields) = selected, case .integer(let revision) = fields["revision"] else { throw SessionError.invalidConversation }
        conversationID = id; branchID = branch
        let turnID = UUID().uuidString.lowercased()
        let request = ChatTurnRequest(turnID: turnID, conversationID: id, branchID: branch, expectedRevision: revision, text: text, inferenceURL: target.url.absoluteString, model: target.model, approvals: .always, inferenceAPIKey: target.bearer)
        reducer = ChatTurnReducer(turnID: turnID); unavailable = []; state = .streaming; publish()
        task = Task { await self.run(request, client: client) }
    }
    private func checkPreparation() throws {
        try Task.checkCancellation()
        if stoppedWhilePreparing { throw CancellationError() }
    }
    private func apply(_ event: ChatWireEnvelope) throws {
        guard var current = reducer else { return }
        let effect = try current.apply(event)
        reducer = current
        if effect == .resume(after: current.lastApplied) { throw Resume.gap }
        if effect == .applied { publish() }
    }
    private func run(_ request: ChatTurnRequest, client initialClient: any ChatTurnClient) async {
        var client = initialClient
        var initial = true
        var failures = 0
        while !Task.isCancelled, reducer?.projection.terminal == nil {
            let cursor = reducer?.lastApplied ?? 0
            do {
                let handler: ChatTurnEventHandler = { [weak self] event in try await self?.apply(event) }
                if initial { try await client.submit(request, after: 0, onEvent: handler); initial = false }
                else { try await client.stream(turnID: request.turnID, after: cursor, onEvent: handler) }
            } catch is CancellationError { break }
            catch {
                if case ChatTurnClientError.beforeHeaders = error {} else { initial = false }
                if Self.isPermanent(error) {
                    state = .failed(error.localizedDescription); publish(); break
                }
            }
            if reducer?.projection.terminal == nil && !Task.isCancelled {
                failures = (reducer?.lastApplied ?? 0) > cursor ? 1 : failures + 1
                if failures >= 4 {
                    state = .failed("Could not reconnect to chat owner."); publish(); break
                }
                state = .reconnecting; publish()
                while !Task.isCancelled {
                    do {
                        try await Task.sleep(for: .milliseconds(250))
                        client = try await clientProvider()
                        self.client = client
                        break
                    } catch is CancellationError { break }
                    catch {
                        if Self.isPermanent(error) { state = .failed(error.localizedDescription); publish(); break }
                        failures += 1
                        if failures >= 4 { state = .failed("Could not reconnect to chat owner."); publish(); break }
                    }
                }
                if Task.isCancelled { break }
                if case .failed = state { break }
                state = .streaming; publish()
            }
        }
        if reducer?.projection.terminal != nil { state = .finished; publish() }
        task = nil
    }
    private static func isPermanent(_ error: Error) -> Bool {
        if let error = error as? ChatTurnClientError {
            switch error {
            case .http(let status): return (400..<500).contains(status) && status != 401
            case .invalidResponse, .invalidEndpoint, .invalidCursor: return true
            case .disconnected, .beforeHeaders: return false
            }
        }
        if let wire = error as? ChatWireError { return wire != .unexpectedEOF }
        return error is DecodingError
    }
    func stop() async throws {
        if preparing { stoppedWhilePreparing = true }
        guard let client, let turnID = reducer?.projection.turnID, reducer?.projection.terminal == nil else { return }
        var current = client
        var lastError: Error?
        for attempt in 0..<3 {
            do {
                if attempt > 0 { current = try await clientProvider(); self.client = current }
                _ = try await current.cancel(turnID: turnID)
                return
            } catch { lastError = error }
        }
        throw lastError!
    }
    func resolve(opID: String, decision: ChatPermissionResponse.Decision) async throws {
        guard let client, let permission = reducer?.projection.pendingPermissions.first(where: { $0.opID == opID }), !unavailable.contains(opID), !resolving.contains(opID) else { throw SessionError.permissionUnavailable }
        resolving.insert(opID)
        defer { resolving.remove(opID) }
        do {
            let receipt = try await client.operation(opID: opID)
            guard case .object(let obj) = receipt, obj["id"] == .string(opID), obj["state"] == .string("intent"), obj["revision"] == .integer(permission.revision),
                  let expiry = Self.number(obj["expires_at"]), expiry > Date().timeIntervalSince1970,
                  permission.expiresAt > Date().timeIntervalSince1970,
                  reducer?.projection.pendingPermissions.contains(permission) == true else { throw SessionError.permissionUnavailable }
            _ = try await client.respond(ChatPermissionResponse(opID: opID, revision: permission.revision, decision: decision))
            unavailable.insert(opID); publish()
        } catch {
            if case SessionError.permissionUnavailable = error {
                unavailable.insert(opID); publish()
            } else if case ChatTurnClientError.http(let status) = error, status == 404 || status == 409 {
                unavailable.insert(opID); publish()
            }
            throw error
        }
    }
    private static func number(_ value: ChatWireJSON?) -> Double? {
        switch value { case .integer(let n): Double(n); case .number(let n): n; default: nil }
    }
}
