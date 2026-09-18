import Foundation

@MainActor
protocol AgentRuntimeTransport: Sendable {
    func create(
        goal: String,
        model: String?,
        toolNames: [String]?,
        trustedInstructions: String?,
        localContext: String?,
        recentUserMessages: [String]?,
        execution: AgentExecutionMode,
        bearerToken: String?
    ) async throws -> AgentRunView

    func get(runID: String, bearerToken: String?) async throws -> AgentRunView
    func events(runID: String, after: Int, bearerToken: String?) async throws -> AgentEventsView
    func resolveApproval(
        runID: String,
        callID: String,
        approved: Bool,
        bearerToken: String?
    ) async throws -> AgentRunView
    func submitToolResult(
        runID: String,
        callID: String,
        content: String,
        isError: Bool,
        executed: Bool,
        declined: Bool,
        bearerToken: String?
    ) async throws -> AgentRunView
    func cancel(runID: String, bearerToken: String?) async throws -> AgentRunView
}

enum PersonalIntelligenceConfig {
    static let introductionCompletedKey =
        "Rapid.personalIntelligence.introductionCompleted"
    static let preferredEnabledKey = "Rapid.personalIntelligence.preferredEnabled"
    static let conversationStatesKey = "Rapid.personalIntelligence.conversationStates"
    static let defaultPreferredEnabled = true

    /// Accept only the server's exact live model → harness binding. Tool-call
    /// support is deliberately insufficient, and a stale profile for the
    /// previously selected model must not enable this one.
    static func harnessProfile(
        for alias: String,
        serverProfile: ServerModelProfile?
    ) -> String? {
        guard let serverProfile,
              serverProfile.id == alias,
              let parser = serverProfile.toolCallParser?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !parser.isEmpty,
              let harness = serverProfile.personalIntelligenceProfile?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !harness.isEmpty,
              let qualification = serverProfile.personalIntelligenceQualification?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !qualification.isEmpty else { return nil }
        return harness
    }

    static func supportsModel(
        _ alias: String,
        serverProfile: ServerModelProfile?
    ) -> Bool {
        harnessProfile(for: alias, serverProfile: serverProfile) != nil
    }

    static func isEnabled(
        conversationEnabled: Bool,
        alias: String,
        serverProfile: ServerModelProfile?
    ) -> Bool {
        conversationEnabled && supportsModel(alias, serverProfile: serverProfile)
    }

    static func loadConversationStates(
        from defaults: UserDefaults = .standard
    ) -> [UUID: Bool] {
        guard let stored = defaults.dictionary(forKey: conversationStatesKey) else {
            return [:]
        }
        return stored.reduce(into: [:]) { result, entry in
            guard let id = UUID(uuidString: entry.key), let enabled = entry.value as? Bool else {
                return
            }
            result[id] = enabled
        }
    }

    static func saveConversationStates(
        _ states: [UUID: Bool],
        to defaults: UserDefaults = .standard
    ) {
        defaults.set(
            Dictionary(uniqueKeysWithValues: states.map { ($0.key.uuidString, $0.value) }),
            forKey: conversationStatesKey
        )
    }

    static func reconciledConversationStates(
        _ states: [UUID: Bool],
        activeConversationID: UUID,
        storedConversationIDs: Set<UUID>,
        newlyCreatedConversationIDs: Set<UUID> = [],
        introductionCompleted: Bool,
        preferredEnabled: Bool
    ) -> [UUID: Bool] {
        let retainedIDs = storedConversationIDs.union([activeConversationID])
        var result = states.filter { retainedIDs.contains($0.key) }

        // Previously saved conversations must not be retroactively opted in
        // when the user accepts the introduction. Only a genuinely new draft
        // inherits the default preference.
        for id in storedConversationIDs where result[id] == nil {
            result[id] = newlyCreatedConversationIDs.contains(id)
                ? introductionCompleted && preferredEnabled
                : false
        }
        if result[activeConversationID] == nil {
            result[activeConversationID] = storedConversationIDs.contains(activeConversationID)
                ? false
                : introductionCompleted && preferredEnabled
        }
        return result
    }
}

enum AgentRunStatus: String, Codable, Equatable, Sendable {
    case ready
    case awaitingModel = "awaiting_model"
    case awaitingApproval = "awaiting_approval"
    case awaitingToolResult = "awaiting_tool_result"
    case completed
    case failed
    case cancelled

    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled: true
        case .ready, .awaitingModel, .awaitingApproval, .awaitingToolResult: false
        }
    }
}

enum AgentToolRisk: String, Codable, Equatable, Sendable {
    case readOnly = "read_only"
    case localChange = "local_change"
    case externalSideEffect = "external_side_effect"
}

enum AgentExecutionMode: String, Codable, Equatable, Sendable {
    case server
    case client
}

struct AgentPendingAction: Codable, Equatable, Sendable {
    let callID: String
    let name: String
    let arguments: [String: CodableJSON]
    let approvalSummary: [String: CodableJSON]?
    let risk: AgentToolRisk
    let approvalRequired: Bool

    enum CodingKeys: String, CodingKey {
        case callID = "call_id"
        case name, arguments
        case approvalSummary = "approval_summary"
        case risk
        case approvalRequired = "approval_required"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        callID = try container.decode(String.self, forKey: .callID)
        name = try container.decode(String.self, forKey: .name)
        arguments = try container.decodeIfPresent(
            [String: CodableJSON].self,
            forKey: .arguments
        ) ?? [:]
        approvalSummary = try container.decodeIfPresent(
            [String: CodableJSON].self,
            forKey: .approvalSummary
        )
        risk = try container.decode(AgentToolRisk.self, forKey: .risk)
        approvalRequired = try container.decode(Bool.self, forKey: .approvalRequired)
    }
}

struct AgentRunView: Codable, Equatable, Sendable {
    let id: String
    let model: String
    let profile: String
    let personalIntelligenceQualification: String?
    let status: AgentRunStatus
    let modelTurns: Int
    let toolRounds: Int
    let finalSynthesis: Bool
    let failureCode: String?
    let output: String?
    let pendingAction: AgentPendingAction?

    enum CodingKeys: String, CodingKey {
        case id, model, profile, status
        case personalIntelligenceQualification = "personal_intelligence_qualification"
        case modelTurns = "model_turns"
        case toolRounds = "tool_rounds"
        case finalSynthesis = "final_synthesis"
        case failureCode = "failure_code"
        case output
        case pendingAction = "pending_action"
    }
}

enum AgentEventSchemaVersion: Int, Codable, Equatable, Sendable {
    case v1 = 1
}

struct AgentEvent: Codable, Equatable, Sendable {
    let schemaVersion: AgentEventSchemaVersion
    let sequence: Int
    let type: String
    let createdAt: Double
    let data: [String: CodableJSON]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sequence, type
        case createdAt = "created_at"
        case data
    }
}

struct AgentEventsView: Codable, Equatable, Sendable {
    let runID: String
    let status: AgentRunStatus
    let events: [AgentEvent]
    let nextAfter: Int

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case status, events
        case nextAfter = "next_after"
    }
}

enum AgentRuntimeClientError: Error, Equatable, LocalizedError {
    case invalidRunID
    case invalidResponse
    case http(status: Int, message: String)
    case malformedResponse
    case harnessProfileMismatch(expected: String, received: String)
    case modelBindingMismatch(expected: String, received: String)
    case qualificationMismatch(expected: String, received: String?)

    var errorDescription: String? {
        switch self {
        case .invalidRunID:
            "The Rapid Agent Runtime run ID is empty or unsafe."
        case .invalidResponse:
            "The Rapid Agent Runtime returned a non-HTTP response."
        case .http(_, let message):
            message
        case .malformedResponse:
            "The Rapid Agent Runtime returned an unreadable response."
        case .harnessProfileMismatch(let expected, let received):
            "Personal Intelligence expected the \(expected) harness, but the server returned \(received). Update Rapid-MLX and try again."
        case .modelBindingMismatch(let expected, let received):
            "Personal Intelligence expected \(expected), but the server started \(received). The mismatched run was stopped; refresh the model and try again."
        case .qualificationMismatch(let expected, let received):
            "Personal Intelligence expected qualification \(expected), but the server returned \(received ?? "none"). The mismatched run was stopped; update Rapid-MLX and try again."
        }
    }
}

/// Typed transport for the server-owned agent loop.
///
/// This deliberately contains no planning or tool policy. Desktop renders the
/// run and supplies user decisions; the Python runtime remains the sole owner
/// of budgets, call identity, approval state, and event ordering.
final class AgentRuntimeClient: Sendable {
    private struct CreateRequest: Encodable {
        let goal: String
        let model: String?
        let toolNames: [String]?
        let trustedInstructions: String?
        let localContext: String?
        let recentUserMessages: [String]?
        let execution: AgentExecutionMode

        enum CodingKeys: String, CodingKey {
            case goal, model
            case toolNames = "tool_names"
            case trustedInstructions = "trusted_instructions"
            case localContext = "local_context"
            case recentUserMessages = "recent_user_messages"
            case execution
        }
    }

    private struct ApprovalRequest: Encodable {
        let callID: String
        let approved: Bool

        enum CodingKeys: String, CodingKey {
            case callID = "call_id"
            case approved
        }
    }

    private struct ToolResultRequest: Encodable {
        let callID: String
        let content: String
        let isError: Bool
        let executed: Bool
        let declined: Bool

        enum CodingKeys: String, CodingKey {
            case callID = "call_id"
            case content
            case isError = "is_error"
            case executed
            case declined
        }

        func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(callID, forKey: .callID)
            try container.encode(content, forKey: .content)
            try container.encode(isError, forKey: .isError)
            try container.encode(executed, forKey: .executed)
            // Older servers reject unknown fields. Preserve their common path
            // while sending the new semantic marker only when it is needed.
            if declined {
                try container.encode(true, forKey: .declined)
            }
        }
    }

    private struct ErrorEnvelope: Decodable {
        let detail: CodableJSON?
    }

    private let session: URLSession
    let baseURL: URL
    let requestTimeout: TimeInterval

    init(
        baseURL: URL = ChatStreamClient.defaultBaseURL,
        session: URLSession? = nil,
        requestTimeout: TimeInterval = 300
    ) {
        self.baseURL = baseURL
        self.session = session ?? ChatStreamClient.sharedSession
        self.requestTimeout = requestTimeout
    }

    func create(
        goal: String,
        model: String? = nil,
        toolNames: [String]? = nil,
        trustedInstructions: String? = nil,
        localContext: String? = nil,
        recentUserMessages: [String]? = nil,
        execution: AgentExecutionMode,
        bearerToken: String? = nil
    ) async throws -> AgentRunView {
        let path = "v1/agent/runs"
        let request = CreateRequest(
            goal: goal,
            model: model,
            toolNames: toolNames,
            trustedInstructions: trustedInstructions,
            localContext: localContext,
            recentUserMessages: recentUserMessages?.isEmpty == true ? nil : recentUserMessages,
            execution: execution
        )
        do {
            return try await send(
                method: "POST",
                path: path,
                bearerToken: bearerToken,
                body: request
            )
        } catch AgentRuntimeClientError.http(let status, let message)
            where request.recentUserMessages != nil && status == 422
            && Self.serverRejectedUnknownField(message, field: "recent_user_messages") {
            return try await send(
                method: "POST",
                path: path,
                bearerToken: bearerToken,
                body: CreateRequest(
                goal: goal,
                model: model,
                toolNames: toolNames,
                trustedInstructions: trustedInstructions,
                localContext: localContext,
                recentUserMessages: nil,
                execution: execution
                )
            )
        }
    }

    func get(runID: String, bearerToken: String? = nil) async throws -> AgentRunView {
        try await send(
            method: "GET",
            path: "v1/agent/runs/\(try encodedPathComponent(runID))",
            bearerToken: bearerToken
        )
    }

    func events(
        runID: String,
        after: Int,
        bearerToken: String? = nil
    ) async throws -> AgentEventsView {
        try await send(
            method: "GET",
            path: "v1/agent/runs/\(try encodedPathComponent(runID))/events",
            queryItems: [URLQueryItem(name: "after", value: String(max(0, after)))],
            bearerToken: bearerToken
        )
    }

    func resolveApproval(
        runID: String,
        callID: String,
        approved: Bool,
        bearerToken: String? = nil
    ) async throws -> AgentRunView {
        try await send(
            method: "POST",
            path: "v1/agent/runs/\(try encodedPathComponent(runID))/approval",
            bearerToken: bearerToken,
            body: ApprovalRequest(callID: callID, approved: approved)
        )
    }

    func submitToolResult(
        runID: String,
        callID: String,
        content: String,
        isError: Bool,
        executed: Bool,
        declined: Bool = false,
        bearerToken: String? = nil
    ) async throws -> AgentRunView {
        let path = "v1/agent/runs/\(try encodedPathComponent(runID))/tool-result"
        do {
            return try await send(
                method: "POST",
                path: path,
                bearerToken: bearerToken,
                body: ToolResultRequest(
                    callID: callID,
                    content: content,
                    isError: isError,
                    executed: executed,
                    declined: declined
                )
            )
        } catch AgentRuntimeClientError.http(let status, let message)
            where declined && !executed && status == 422
            && Self.serverRejectedUnknownField(message, field: "declined") {
            // Servers predating the structured decline field reject unknown
            // keys. Retry the same non-executed result without that marker;
            // its content retains the legacy explanation.
            return try await send(
                method: "POST",
                path: path,
                bearerToken: bearerToken,
                body: ToolResultRequest(
                    callID: callID,
                    content: content,
                    isError: isError,
                    executed: executed,
                    declined: false
                )
            )
        }
    }

    private static func serverRejectedUnknownField(
        _ message: String,
        field: String
    ) -> Bool {
        let normalized = message.lowercased()
        return normalized.contains(field.lowercased())
            && (normalized.contains("unknown") || normalized.contains("extra"))
    }

    func cancel(runID: String, bearerToken: String? = nil) async throws -> AgentRunView {
        try await send(
            method: "POST",
            path: "v1/agent/runs/\(try encodedPathComponent(runID))/cancel",
            bearerToken: bearerToken,
            body: Optional<String>.none
        )
    }

    private func send<Response: Decodable, Body: Encodable>(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        bearerToken: String?,
        body: Body? = nil
    ) async throws -> Response {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        let basePath = components?.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/")) ?? ""
        components?.percentEncodedPath = "/" + [basePath, path]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components?.url else { throw AgentRuntimeClientError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = requestTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let bearerToken, !bearerToken.isEmpty {
            request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            request.httpBody = try encoder.encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AgentRuntimeClientError.invalidResponse
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw AgentRuntimeClientError.http(
                status: http.statusCode,
                message: Self.errorMessage(from: data, status: http.statusCode)
            )
        }
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(Response.self, from: data)
        } catch {
            throw AgentRuntimeClientError.malformedResponse
        }
    }

    private func send<Response: Decodable>(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        bearerToken: String?
    ) async throws -> Response {
        try await send(
            method: method,
            path: path,
            queryItems: queryItems,
            bearerToken: bearerToken,
            body: Optional<String>.none
        )
    }

    private func encodedPathComponent(_ value: String) throws -> String {
        let uuidHyphenOffsets: Set<Int> = [8, 13, 18, 23]
        let characters = Array(value.utf8)
        guard characters.count == 36,
              characters.enumerated().allSatisfy({ offset, byte in
                  if uuidHyphenOffsets.contains(offset) { return byte == 45 }
                  return (48 ... 57).contains(byte) || (97 ... 102).contains(byte)
              }) else {
            throw AgentRuntimeClientError.invalidRunID
        }
        return value
    }

    private static func errorMessage(from data: Data, status: Int) -> String {
        let decoder = JSONDecoder()
        if let envelope = try? decoder.decode(ErrorEnvelope.self, from: data),
           let detail = envelope.detail {
            switch detail {
            case .string(let message): return message
            default:
                if let encoded = try? JSONEncoder().encode(detail),
                   let message = String(data: encoded, encoding: .utf8) {
                    return message
                }
            }
        }
        return "Rapid Agent Runtime request failed (HTTP \(status))."
    }
}

extension AgentRuntimeClient: AgentRuntimeTransport {}
