import Foundation

struct CUAAppOption: Codable, Equatable, Identifiable, Sendable {
    var name: String?
    var bundleID: String?
    var pid: Int

    var id: Int { pid }
    var displayName: String {
        let label = name?.nilIfBlank ?? bundleID?.nilIfBlank ?? "Unknown app"
        return "\(label) — PID \(pid)"
    }
    var isBrowser: Bool {
        let bundle = (bundleID ?? "").lowercased()
        return bundle == "com.apple.safari"
            || bundle == "com.apple.safaritechnologypreview"
            || bundle.hasPrefix("com.google.chrome")
            || bundle.hasPrefix("com.microsoft.edgemac")
            || bundle.hasPrefix("org.chromium.chromium")
    }

    enum CodingKeys: String, CodingKey {
        case name, pid
        case bundleID = "bundle_id"
    }
}

struct CUAWindowOption: Codable, Equatable, Identifiable, Sendable {
    var windowID: String
    var index: Int
    var title: String
    var x: Double?
    var y: Double?
    var width: Double?
    var height: Double?

    var id: String { windowID }
    var displayTitle: String { title.nilIfBlank ?? "Untitled window" }
    var displayName: String {
        var detail = "Window \(index + 1)"
        if let width, let height, width > 0, height > 0 {
            detail += " · \(Int(width.rounded()))×\(Int(height.rounded()))"
        }
        return "\(displayTitle) — \(detail)"
    }

    enum CodingKeys: String, CodingKey {
        case index, title, x, y, width, height
        case windowID = "window_id"
    }
}

struct CUARunTarget: Codable, Equatable, Identifiable, Sendable {
    var targetID: String
    var app: String
    var pid: Int
    var windowID: String
    var allowedDomain: String
    var bundleID: String? = nil
    var processStartTime: Double? = nil

    var id: String { targetID }

    /// The create response echoes the committed run binding through the
    /// narrower server run-target schema. Process identity fields are request
    /// validation inputs and are intentionally absent from that response.
    func hasSameRunBinding(as other: CUARunTarget) -> Bool {
        targetID == other.targetID
            && app == other.app
            && pid == other.pid
            && windowID == other.windowID
            && allowedDomain == other.allowedDomain
    }

    enum CodingKeys: String, CodingKey {
        case app, pid
        case targetID = "target_id"
        case windowID = "window_id"
        case allowedDomain = "allowed_domain"
        case bundleID = "bundle_id"
        case processStartTime = "process_start_time"
    }
}

struct CUATargetResolveRequest: Codable, Equatable, Sendable {
    var goal: String
    var planner: String
    var allowRemoteAppDiscovery: Bool

    enum CodingKeys: String, CodingKey {
        case goal, planner
        case allowRemoteAppDiscovery = "allow_remote_app_discovery"
    }
}

struct CUATargetProposal: Codable, Equatable, Identifiable, Sendable {
    var targetID: String
    var app: String
    var pid: Int
    var windowID: String
    var allowedDomain: String
    var displayName: String
    var bundleID: String
    var processStartTime: Double

    var id: String { targetID }

    enum CodingKeys: String, CodingKey {
        case app, pid
        case targetID = "target_id"
        case windowID = "window_id"
        case allowedDomain = "allowed_domain"
        case displayName = "display_name"
        case bundleID = "bundle_id"
        case processStartTime = "process_start_time"
    }
}

struct CUATargetApprovalOption: Codable, Equatable, Identifiable, Sendable {
    var optionID: String
    var label: String
    var targetIDs: [String]

    var id: String { optionID }

    enum CodingKeys: String, CodingKey {
        case label
        case optionID = "option_id"
        case targetIDs = "target_ids"
    }
}

struct CUATargetApproval: Codable, Equatable, Sendable {
    var kind: String
    var prompt: String
    var options: [CUATargetApprovalOption]
}

struct CUATargetAutomationRequest: Codable, Equatable, Sendable {
    var bundleID: String
    var displayName: String

    enum CodingKeys: String, CodingKey {
        case bundleID = "bundle_id"
        case displayName = "display_name"
    }
}

struct CUATargetResolution: Codable, Equatable, Sendable {
    var status: String
    var targets: [CUATargetProposal]
    var initialTargetID: String?
    var reason: String
    var approval: CUATargetApproval?
    var automation: CUATargetAutomationRequest? = nil

    enum CodingKeys: String, CodingKey {
        case status, targets, reason, approval, automation
        case initialTargetID = "initial_target_id"
    }
}

private extension String {
    var nilIfBlank: String? {
        let value = trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }
}

/// One selectable slow-thinking planner preset from `GET /v1/cua/planners`.
struct CUAPlannerOption: Codable, Equatable, Identifiable, Sendable {
    var name: String
    var model: String
    var url: String
    var textOnly: Bool
    var note: String
    var hasApiKey: Bool
    var userCreated: Bool
    var allowRemote: Bool

    var id: String { name }

    var displayName: String {
        "\(name) — \(model)"
    }

    init(
        name: String,
        model: String,
        url: String,
        textOnly: Bool,
        note: String = "",
        hasApiKey: Bool = false,
        userCreated: Bool = false,
        allowRemote: Bool = false
    ) {
        self.name = name
        self.model = model
        self.url = url
        self.textOnly = textOnly
        self.note = note
        self.hasApiKey = hasApiKey
        self.userCreated = userCreated
        self.allowRemote = allowRemote
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        model = try container.decode(String.self, forKey: .model)
        url = try container.decode(String.self, forKey: .url)
        textOnly = try container.decodeIfPresent(Bool.self, forKey: .textOnly) ?? false
        note = try container.decodeIfPresent(String.self, forKey: .note) ?? ""
        // Tolerate older sidecars that predate the settings feature.
        hasApiKey = try container.decodeIfPresent(Bool.self, forKey: .hasApiKey) ?? false
        userCreated = try container.decodeIfPresent(Bool.self, forKey: .userCreated) ?? false
        allowRemote = try container.decodeIfPresent(Bool.self, forKey: .allowRemote) ?? false
    }

    enum CodingKeys: String, CodingKey {
        case name, model, url, note
        case textOnly = "text_only"
        case hasApiKey = "has_api_key"
        case userCreated = "user_created"
        case allowRemote = "allow_remote"
    }
}

/// Body for `POST /v1/cua/planners` (user adds a planner endpoint in settings).
struct CUAPlannerCreateRequest: Codable, Equatable, Sendable {
    var name: String
    var url: String
    var model: String
    var apiKey: String?
    var reasoningEffort: String?
    var textOnly: Bool
    var allowRemote: Bool

    enum CodingKeys: String, CodingKey {
        case name, url, model
        case apiKey = "api_key"
        case reasoningEffort = "reasoning_effort"
        case textOnly = "text_only"
        case allowRemote = "allow_remote"
    }
}

/// One numbered progress event from `GET /v1/cua/runs/{id}/events`.
struct CUAEvent: Codable, Equatable, Sendable {
    var seq: Int
    var kind: String
    var step: Int?
    var action: String?
    var stepInstruction: String?
    var outcome: String?
    var targetLabel: String?
    var status: String?
    var finalSummary: String?
    var reason: String?
    var error: String? = nil
    var app: String? = nil
    var gateID: String? = nil
    var target: String? = nil
    var targetID: String? = nil
    var fromTargetID: String? = nil

    enum CodingKeys: String, CodingKey {
        case seq, kind, step, action, outcome, status, reason, error, app, target
        case gateID = "gate_id"
        case targetID = "target_id"
        case fromTargetID = "from_target_id"
        case stepInstruction = "step_instruction"
        case targetLabel = "target_label"
        case finalSummary = "final_summary"
    }

    var isTerminal: Bool { kind == "terminal" }
    var isGate: Bool { kind == "gate" }
}

private struct CUAApprovalRequest: Codable {
    var gateID: String
    var approved: Bool

    enum CodingKeys: String, CodingKey {
        case gateID = "gate_id"
        case approved
    }
}

private struct CUAPermissionRequest: Codable {
    let permission: String
}

struct CUAPendingGate: Codable, Equatable, Sendable {
    var gateID: String
    var reason: String?
    var action: String?
    var target: String?
    var targetID: String? = nil

    enum CodingKeys: String, CodingKey {
        case gateID = "gate_id"
        case reason, action, target
        case targetID = "target_id"
    }
}

/// Full run view returned by the events endpoint.
struct CUARunView: Codable, Equatable, Sendable {
    var runID: String
    var app: String
    var goal: String
    var status: String
    var finalSummary: String
    var error: String
    var planner: String
    var eventsAfterSeq: Int
    var events: [CUAEvent]
    var pendingGate: CUAPendingGate? = nil
    var windowID: String? = nil
    var targets: [CUARunTarget]? = nil
    var activeTargetID: String? = nil

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case app, goal, status, error, planner, events
        case finalSummary = "final_summary"
        case eventsAfterSeq = "events_after_seq"
        case pendingGate = "pending_gate"
        case windowID = "window_id"
        case targets
        case activeTargetID = "active_target_id"
    }
}

struct CUARunCreated: Codable, Equatable, Sendable {
    var runID: String
    var status: String
    var windowID: String?
    var clientRequestID: String?
    var targets: [CUARunTarget]? = nil
    var activeTargetID: String? = nil

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case status
        case windowID = "window_id"
        case clientRequestID = "client_request_id"
        case targets
        case activeTargetID = "active_target_id"
    }
}

/// Create-run request body for `POST /v1/cua/runs`.
struct CUARunRequest: Codable, Equatable, Sendable {
    var app: String
    var goal: String
    var planner: String
    var openURL: String
    var allowedDomain: String
    var maxSteps: Int
    var humanLogin: Bool
    var windowID: String
    var clientRequestID: String
    var targets: [CUARunTarget]?
    var initialTargetID: String?
    var bundleID: String?
    var processStartTime: Double?

    init(
        app: String, goal: String, planner: String, openURL: String,
        allowedDomain: String, maxSteps: Int, humanLogin: Bool, windowID: String,
        clientRequestID: String = UUID().uuidString.lowercased(),
        targets: [CUARunTarget]? = nil, initialTargetID: String? = nil,
        bundleID: String? = nil, processStartTime: Double? = nil
    ) {
        self.app = app
        self.goal = goal
        self.planner = planner
        self.openURL = openURL
        self.allowedDomain = allowedDomain
        self.maxSteps = maxSteps
        self.humanLogin = humanLogin
        self.windowID = windowID
        self.clientRequestID = clientRequestID
        self.targets = targets
        self.initialTargetID = initialTargetID
        self.bundleID = bundleID
        self.processStartTime = processStartTime
    }

    enum CodingKeys: String, CodingKey {
        case app, goal, planner
        case openURL = "open_url"
        case allowedDomain = "allowed_domain"
        case maxSteps = "max_steps"
        case humanLogin = "human_login"
        case windowID = "window_id"
        case clientRequestID = "client_request_id"
        case targets
        case initialTargetID = "initial_target_id"
        case bundleID = "bundle_id"
        case processStartTime = "process_start_time"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(app, forKey: .app)
        try container.encode(goal, forKey: .goal)
        try container.encode(planner, forKey: .planner)
        try container.encode(openURL, forKey: .openURL)
        try container.encode(maxSteps, forKey: .maxSteps)
        try container.encode(humanLogin, forKey: .humanLogin)
        try container.encode(clientRequestID, forKey: .clientRequestID)
        if let targets {
            try container.encode(targets, forKey: .targets)
            try container.encode(initialTargetID, forKey: .initialTargetID)
        } else {
            try container.encode(allowedDomain, forKey: .allowedDomain)
            try container.encode(windowID, forKey: .windowID)
            try container.encodeIfPresent(bundleID, forKey: .bundleID)
            try container.encodeIfPresent(processStartTime, forKey: .processStartTime)
        }
    }
}

enum CUAClientError: LocalizedError, Equatable {
    case http(Int, String)
    case typedHTTP(Int, code: String, message: String, recovery: [String])
    case windowBinding(
        expected: String, actual: String?, runID: String, cancellationFailed: Bool
    )
    case requestBinding(
        expected: String, actual: String?, runID: String, cancellationFailed: Bool
    )
    case targetBinding(runID: String, cancellationFailed: Bool)

    var errorDescription: String? {
        switch self {
        case let .http(code, detail):
            return "CUA request failed (HTTP \(code)): \(detail)"
        case let .typedHTTP(code, errorCode, message, recovery):
            let hint = recovery.first.map { " \($0)" } ?? ""
            return "CUA request failed (HTTP \(code), \(errorCode)): \(message)\(hint)"
        case let .windowBinding(_, _, _, cancellationFailed):
            let stop = cancellationFailed
                ? " Rapid could not confirm that the rejected run stopped. Stop the local server before retrying."
                : " The rejected run was stopped."
            return "Rapid could not confirm the app selected for this task.\(stop) Start setup again."
        case let .requestBinding(_, _, _, cancellationFailed):
            let stop = cancellationFailed
                ? " The unverified task may still be executing. Stop it immediately."
                : " The unverified task was stopped."
            return "Rapid could not confirm that the task started safely.\(stop)"
        case let .targetBinding(_, cancellationFailed):
            let stop = cancellationFailed
                ? " The unverified task may still be executing. Stop it immediately."
                : " The unverified task was stopped."
            return "Rapid could not confirm the apps approved for this task.\(stop) Start setup again."
        }
    }
}

/// Loopback-only HTTP client for the server's `/v1/cua` surface.
///
/// The CUA API drives the local computer, so it must never be reachable through
/// a non-loopback host, and a bearer token is required. The app-owned server
/// inherits the app's TCC grants (Accessibility, Screen Recording,
/// Automation), which standalone processes do not have.
struct CUAClient: CUAAPI, Sendable {
    let baseURL: URL
    let bearerToken: String
    let session: URLSession

    func capabilities() async throws -> CUACapabilities {
        let (data, response) = try await send(path: "/v1/cua/capabilities", method: "GET")
        return try decode(CUACapabilities.self, from: data, response: response)
    }

    init?(
        host: String,
        port: Int,
        bearerToken: String,
        session: URLSession = .shared
    ) {
        guard host == "127.0.0.1",
              (1 ... 65_535).contains(port),
              !bearerToken.isEmpty,
              let baseURL = URL(string: "http://127.0.0.1:\(port)")
        else { return nil }
        self.baseURL = baseURL
        self.bearerToken = bearerToken
        self.session = session
    }

    func planners() async throws -> [CUAPlannerOption] {
        let (data, response) = try await send(path: "/v1/cua/planners", method: "GET")
        return try decode([CUAPlannerOption].self, from: data, response: response)
    }

    func resolveTargets(
        goal: String, planner: String, allowRemoteAppDiscovery: Bool
    ) async throws -> CUATargetResolution {
        let body = try JSONEncoder().encode(
            CUATargetResolveRequest(
                goal: goal, planner: planner,
                allowRemoteAppDiscovery: allowRemoteAppDiscovery
            )
        )
        let (data, response) = try await send(
            path: "/v1/cua/targets/resolve", method: "POST", body: body
        )
        return try decode(CUATargetResolution.self, from: data, response: response)
    }

    func apps() async throws -> [CUAAppOption] {
        let (data, response) = try await send(path: "/v1/cua/apps", method: "GET")
        return try decode([CUAAppOption].self, from: data, response: response)
    }

    func windows(app: String) async throws -> [CUAWindowOption] {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        guard let encodedApp = app.addingPercentEncoding(withAllowedCharacters: allowed),
              let url = URL(
                  string: baseURL.absoluteString + "/v1/cua/apps/\(encodedApp)/windows"
              )
        else { throw URLError(.badURL) }
        let (data, response) = try await send(url: url, method: "GET")
        return try decode([CUAWindowOption].self, from: data, response: response)
    }

    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {
        let body = try JSONEncoder().encode(request)
        let (data, response) = try await send(
            path: "/v1/cua/planners", method: "POST", body: body
        )
        try requireSuccess(data, response)
    }

    func deletePlanner(name: String) async throws {
        let (data, response) = try await send(
            path: "/v1/cua/planners/\(name)", method: "DELETE"
        )
        try requireSuccess(data, response)
    }

    func create(_ request: CUARunRequest) async throws -> String {
        let body = try JSONEncoder().encode(request)
        let (data, response) = try await send(
            path: "/v1/cua/runs", method: "POST", body: body
        )
        let created = try decode(CUARunCreated.self, from: data, response: response)
        let targetBindingMatches = request.targets.map { requestedTargets in
            created.targets?.count == requestedTargets.count
                && zip(created.targets ?? [], requestedTargets).allSatisfy {
                    $0.hasSameRunBinding(as: $1)
                }
                && created.activeTargetID == request.initialTargetID
        } ?? (created.windowID == request.windowID)
        guard targetBindingMatches, created.clientRequestID == request.clientRequestID
        else {
            var cancellationFailed = false
            do {
                try await cancel(runID: created.runID)
            } catch {
                cancellationFailed = true
            }
            if request.targets != nil && !targetBindingMatches {
                throw CUAClientError.targetBinding(
                    runID: created.runID, cancellationFailed: cancellationFailed
                )
            }
            if request.targets == nil && created.windowID != request.windowID {
                throw CUAClientError.windowBinding(
                    expected: request.windowID, actual: created.windowID,
                    runID: created.runID, cancellationFailed: cancellationFailed
                )
            }
            throw CUAClientError.requestBinding(
                expected: request.clientRequestID, actual: created.clientRequestID,
                runID: created.runID, cancellationFailed: cancellationFailed
            )
        }
        return created.runID
    }

    func run(clientRequestID: String) async throws -> CUARunCreated {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        guard let encoded = clientRequestID.addingPercentEncoding(withAllowedCharacters: allowed),
              let url = URL(
                  string: baseURL.absoluteString + "/v1/cua/runs/by-request/\(encoded)"
              )
        else { throw URLError(.badURL) }
        let (data, response) = try await send(url: url, method: "GET")
        return try decode(CUARunCreated.self, from: data, response: response)
    }

    func permissions() async throws -> CUAPermissionStatus {
        let (data, response) = try await send(path: "/v1/cua/permissions", method: "GET")
        return try decode(CUAPermissionStatus.self, from: data, response: response)
    }

    func requestPermission(
        _ permission: MacAutomationPermission
    ) async throws -> CUAPermissionRequestResult {
        let wirePermission = switch permission {
        case .accessibility: "accessibility"
        case .screenRecording: "screen_recording"
        }
        let body = try JSONEncoder().encode(
            CUAPermissionRequest(permission: wirePermission)
        )
        let (data, response) = try await send(
            path: "/v1/cua/permissions/request", method: "POST", body: body
        )
        return try decode(
            CUAPermissionRequestResult.self, from: data, response: response
        )
    }

    func events(runID: String, after: Int) async throws -> CUARunView {
        var components = URLComponents(
            url: baseURL.appendingPathComponent("/v1/cua/runs/\(runID)/events"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [URLQueryItem(name: "after", value: String(after))]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        let (data, response) = try await send(url: url, method: "GET")
        return try decode(CUARunView.self, from: data, response: response)
    }

    func approve(runID: String, gateID: String) async throws {
        let body = try JSONEncoder().encode(
            CUAApprovalRequest(gateID: gateID, approved: true)
        )
        let (data, response) = try await send(
            path: "/v1/cua/runs/\(runID)/approval", method: "POST", body: body
        )
        try requireSuccess(data, response)
    }

    func cancel(runID: String) async throws {
        let (data, response) = try await send(
            path: "/v1/cua/runs/\(runID)/cancel", method: "POST"
        )
        try requireSuccess(data, response)
    }

    private func send(
        path: String,
        method: String,
        body: Data? = nil
    ) async throws -> (Data, URLResponse) {
        guard let url = URL(string: baseURL.absoluteString + path) else {
            throw URLError(.badURL)
        }
        return try await send(url: url, method: method, body: body)
    }

    private func send(
        url: URL,
        method: String,
        body: Data? = nil
    ) async throws -> (Data, URLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }
        let (data, response) = try await session.data(for: request)
        return (data, response)
    }

    private func requireSuccess(_ data: Data, _ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200 ... 299).contains(http.statusCode) else {
            struct TypedDetail: Decodable {
                let code: String
                let message: String
                let recovery: [String]
            }
            struct TypedErrorBody: Decodable { let detail: TypedDetail }
            if let detail = try? JSONDecoder().decode(TypedErrorBody.self, from: data).detail {
                throw CUAClientError.typedHTTP(
                    http.statusCode,
                    code: detail.code,
                    message: detail.message,
                    recovery: detail.recovery
                )
            }
            struct ErrorBody: Decodable { let detail: String }
            let detail = (try? JSONDecoder().decode(ErrorBody.self, from: data).detail)
                ?? String(data: data.prefix(300), encoding: .utf8)
                ?? ""
            throw CUAClientError.http(http.statusCode, detail)
        }
    }

    private func decode<T: Decodable>(
        _ type: T.Type, from data: Data, response: URLResponse
    ) throws -> T {
        try requireSuccess(data, response)
        return try JSONDecoder().decode(type, from: data)
    }
}
