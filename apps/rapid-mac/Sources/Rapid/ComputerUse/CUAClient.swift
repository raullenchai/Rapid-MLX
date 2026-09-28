import Foundation

/// One selectable slow-thinking planner preset from `GET /v1/cua/planners`.
struct CUAPlannerOption: Codable, Equatable, Identifiable, Sendable {
    var name: String
    var model: String
    var url: String
    var textOnly: Bool
    var note: String
    var hasApiKey: Bool
    var userCreated: Bool

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
        userCreated: Bool = false
    ) {
        self.name = name
        self.model = model
        self.url = url
        self.textOnly = textOnly
        self.note = note
        self.hasApiKey = hasApiKey
        self.userCreated = userCreated
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
    }

    enum CodingKeys: String, CodingKey {
        case name, model, url, note
        case textOnly = "text_only"
        case hasApiKey = "has_api_key"
        case userCreated = "user_created"
    }
}

/// Body for `POST /v1/cua/planners` (user adds a cloud brain in settings).
/// Supplying apiKey is the user's consent to send task data to that endpoint.
struct CUAPlannerCreateRequest: Codable, Equatable, Sendable {
    var name: String
    var url: String
    var model: String
    var apiKey: String?
    var reasoningEffort: String?
    var textOnly: Bool

    enum CodingKeys: String, CodingKey {
        case name, url, model
        case apiKey = "api_key"
        case reasoningEffort = "reasoning_effort"
        case textOnly = "text_only"
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

    enum CodingKeys: String, CodingKey {
        case seq, kind, step, action, outcome, status, reason, error
        case stepInstruction = "step_instruction"
        case targetLabel = "target_label"
        case finalSummary = "final_summary"
    }

    var isTerminal: Bool { kind == "terminal" }
    var isGate: Bool { kind == "gate" }
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
    var runDir: String

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case app, goal, status, error, planner, events
        case finalSummary = "final_summary"
        case eventsAfterSeq = "events_after_seq"
        case runDir = "run_dir"
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

    enum CodingKeys: String, CodingKey {
        case app, goal, planner
        case openURL = "open_url"
        case allowedDomain = "allowed_domain"
        case maxSteps = "max_steps"
        case humanLogin = "human_login"
    }
}

enum CUAClientError: LocalizedError, Equatable {
    case http(Int, String)

    var errorDescription: String? {
        switch self {
        case let .http(code, detail):
            return "CUA request failed (HTTP \(code)): \(detail)"
        }
    }
}

/// Loopback-only HTTP client for the server's `/v1/cua` surface.
///
/// The guard mirrors `DraftPostLanguageRuntime`: the CUA API drives the local
/// computer, so it must never be reachable through a non-loopback host, and a
/// bearer token is required. The app-owned server inherits the app's TCC
/// grants (Accessibility, Screen Recording, Automation), which standalone
/// processes do not have.
struct CUAClient: CUAAPI, Sendable {
    let baseURL: URL
    let bearerToken: String
    let session: URLSession

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
        struct Created: Codable {
            var runID: String
            enum CodingKeys: String, CodingKey { case runID = "run_id" }
        }
        return try decode(Created.self, from: data, response: response).runID
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

    func approve(runID: String) async throws {
        let (data, response) = try await send(
            path: "/v1/cua/runs/\(runID)/approval", method: "POST"
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
