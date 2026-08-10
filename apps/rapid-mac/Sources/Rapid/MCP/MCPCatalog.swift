import Foundation
import Observation

/// Live view of what the engine's MCP subsystem is actually doing.
///
/// Issue #1716: the app had no way to see which servers connected, which tools
/// they exposed, or why one failed. The engine has answered all three
/// questions over HTTP for a while (`vllm_mlx/routes/mcp_routes.py`); nothing
/// asked. This polls those routes and republishes the answers as observable
/// state the Settings panel and the tool registry both read.
///
/// Deliberately a *read* model plus one action (``reload``). The engine owns
/// the connections; this owns nothing but the last thing it was told.
@MainActor
@Observable
final class MCPCatalog {
    struct ServerStatus: Identifiable, Equatable, Sendable {
        let name: String
        /// Engine-side connection state — `connected`, `error`, `disconnected`.
        let state: String
        let transport: String
        let toolsCount: Int
        let error: String?

        var id: String { name }
        var isConnected: Bool { state == "connected" }
    }

    /// Per-server connection rows, including entries the engine's config
    /// validation rejected outright (those arrive as `state == "error"`).
    private(set) var servers: [ServerStatus] = []

    /// Every tool the connected servers expose, already in the wire shape the
    /// chat request body needs. Names are engine-namespaced `server__tool`.
    private(set) var tools: [ToolDefinition] = []

    /// Which server each tool came from, keyed by tool name. Needed by the
    /// approval prompt — "run `read_file`?" is not a question a user can
    /// answer without knowing whose `read_file` it is.
    private(set) var serverForTool: [String: String] = [:]

    /// Whole-subsystem failure (MCP could not start at all), as opposed to one
    /// server failing. Comes from the engine's `error` field.
    private(set) var subsystemError: String?

    /// True once the engine knows about a config path. Distinguishes "the app
    /// never passed --mcp-config" from "config present but broken".
    private(set) var isConfigured: Bool = false

    /// Last transport-level failure talking to the engine itself. Separate
    /// from ``subsystemError``: this one means we don't know the state, rather
    /// than knowing it's bad.
    private(set) var fetchError: String?

    /// Bumped at the start of every ``refresh`` / ``reload``. Each run captures
    /// its value and, after its network awaits, commits only if still current —
    /// so a slow poll that started before a reload can't complete last and
    /// restore the tools the reload just removed. `@MainActor` makes the
    /// bump-and-capture and the compare-and-commit each atomic between awaits.
    private var mutationGeneration = 0

    private let session: URLSession
    /// Resolves the live endpoint. A closure rather than a stored host/port
    /// because both float across a restart (`ServerManager.activePort`
    /// sweeps 8000–8009 and the bearer rotates every launch), and a snapshot
    /// taken at construction would be stale by the first refresh.
    private let endpoint: @MainActor () -> (host: String, port: Int, bearer: String?)?

    init(
        session: URLSession = .shared,
        endpoint: @escaping @MainActor () -> (host: String, port: Int, bearer: String?)?
    ) {
        self.session = session
        self.endpoint = endpoint
    }

    /// Drop everything we believe about the engine's MCP state.
    ///
    /// Called when the child goes away. Keeping a stale tool list across a
    /// server stop would let the chat loop advertise — and try to execute —
    /// tools that no process is behind.
    /// Whether a namespaced tool name is a legal OpenAI function name:
    /// `[A-Za-z0-9_-]`, 1–64 characters. Names from a connector are arbitrary,
    /// so this is the gate that keeps an un-emittable name off the tool list.
    static func isLegalFunctionName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= 64 else { return false }
        return name.unicodeScalars.allSatisfy {
            ($0 >= "a" && $0 <= "z") || ($0 >= "A" && $0 <= "Z")
                || ($0 >= "0" && $0 <= "9") || $0 == "_" || $0 == "-"
        }
    }

    func clear() {
        // Invalidate any in-flight refresh/reload: without this bump a slow poll
        // that started before the clear could complete last and restore the
        // very servers and tools we are wiping because the child went away.
        mutationGeneration += 1
        servers = []
        tools = []
        serverForTool = [:]
        subsystemError = nil
        fetchError = nil
        isConfigured = false
    }

    // MARK: - Refresh

    /// Re-read `/v1/mcp/servers` and `/v1/mcp/tools`.
    @discardableResult
    func refresh() async -> Bool {
        guard let ep = endpoint() else {
            clear()
            return false
        }
        mutationGeneration += 1
        let generation = mutationGeneration
        do {
            // Fetch BOTH routes before publishing anything. Committing the
            // server rows and then failing the tools fetch would leave the two
            // inconsistent — new servers advertised alongside a stale tool list
            // that a reload may have just changed. Build locals, commit as one.
            let serversResponse: ServersResponse = try await get("/v1/mcp/servers", ep)
            let toolsResponse: ToolsResponse = try await get("/v1/mcp/tools", ep)

            // A newer refresh/reload started while we were on the network; its
            // result is the current truth. Discard ours rather than clobber it.
            guard generation == mutationGeneration else { return false }

            servers = serversResponse.servers.map {
                ServerStatus(
                    name: $0.name,
                    state: $0.state,
                    transport: $0.transport,
                    toolsCount: $0.tools_count,
                    error: $0.error
                )
            }
            subsystemError = serversResponse.error
            isConfigured = serversResponse.configured ?? false
            // Drop any tool whose namespaced `server__tool` name can't be a
            // legal OpenAI function name — `[A-Za-z0-9_-]`, at most 64 chars.
            // Capping the server half at 32 bounds neither the length nor the
            // characters of a connector's own tool names, and advertising a
            // name the model can't emit — or that 400s on the wire — reads as
            // "that tool silently does nothing". Not advertising it is honest.
            let usableTools = toolsResponse.tools.filter {
                MCPCatalog.isLegalFunctionName($0.name)
            }
            tools = usableTools.map {
                ToolDefinition(
                    name: $0.name,
                    description: $0.description,
                    parameters: $0.parameters ?? .object([:])
                )
            }
            serverForTool = Dictionary(
                usableTools.map { ($0.name, $0.server) },
                uniquingKeysWith: { first, _ in first }
            )
            fetchError = nil
            return true
        } catch {
            // Don't wipe the last-known-good list on a transient failure — a
            // dropped poll shouldn't make every connector row flicker away, and
            // nothing was committed above, so servers and tools stay in sync.
            // Say we couldn't check instead.
            fetchError = error.localizedDescription
            return false
        }
    }

    /// Ask the engine to re-read the config file and rebuild its connections,
    /// then refresh. This is what makes a Settings edit take effect without
    /// restarting the model (issue #1716 acceptance item 4).
    ///
    /// - Returns: `true` when the engine actually picked the change up.
    ///   `false` means the caller should fall back to telling the user a
    ///   restart is needed. Two distinct cases produce `false`, and both must:
    ///   the route failed or is missing (older engine build), OR it answered
    ///   but reports it has no config path — which is what happens when the
    ///   running child was spawned before connectors were switched on and so
    ///   never received `--mcp-config`. That second case returns HTTP 200, so
    ///   treating "the request succeeded" as "the change applied" would clear
    ///   the restart banner while the edit sat inert.
    @discardableResult
    func reload() async -> Bool {
        guard let ep = endpoint() else { return false }
        mutationGeneration += 1
        let generation = mutationGeneration
        do {
            let response: ServersResponse = try await post("/v1/mcp/reload", ep)
            // Superseded by a newer mutation while the reload was in flight.
            guard generation == mutationGeneration else { return false }
            servers = response.servers.map {
                ServerStatus(
                    name: $0.name,
                    state: $0.state,
                    transport: $0.transport,
                    toolsCount: $0.tools_count,
                    error: $0.error
                )
            }
            subsystemError = response.error
            // Default true for an engine build that predates the field: it
            // answered the reload route at all, so it is new enough to have
            // done the work.
            isConfigured = response.configured ?? true
            fetchError = nil
            if !isConfigured { return false }
        } catch {
            fetchError = error.localizedDescription
            return false
        }
        // Tools come from the second route; the reload response only carries
        // server rows. Report the refresh outcome rather than a hardcoded
        // `true`: if that fetch fails the tool list is the pre-reload one, and
        // a caller told "reload succeeded" would keep advertising tools a
        // reconfigure may have just removed.
        let refreshed = await refresh()
        if !refreshed, generation == mutationGeneration {
            // The engine reloaded (servers committed above) but we couldn't
            // read the new tool list. Unlike a transient poll failure, a reload
            // is a known state change — keeping the pre-reload tools would
            // advertise ones the reconfigure may have removed, so drop them and
            // let `fetchError` drive a retry rather than showing stale tools.
            tools = []
            serverForTool = [:]
        }
        return refreshed
    }

    // MARK: - Execute

    /// Run one tool through the engine and return its textual result.
    ///
    /// The approval decision happens in ``MCPToolRegistry`` before this is
    /// called — by the time a request reaches here the user has said yes.
    func execute(toolName: String, argumentsJSON: String) async throws -> ExecuteResponse {
        guard let ep = endpoint() else { throw CatalogError.serverNotRunning }
        // Arguments arrive as the model's raw JSON string. Empty means a
        // no-arg tool, which is legal.
        let trimmed = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        let argumentsValue: CodableJSON
        if trimmed.isEmpty {
            argumentsValue = .object([:])
        } else {
            guard let data = trimmed.data(using: .utf8),
                  let decoded = try? JSONDecoder().decode(CodableJSON.self, from: data),
                  case .object = decoded else {
                throw CatalogError.badArguments
            }
            argumentsValue = decoded
        }
        let body = ExecuteRequest(tool_name: toolName, arguments: argumentsValue)
        return try await post("/v1/mcp/execute", ep, body: body)
    }

    enum CatalogError: LocalizedError {
        case serverNotRunning
        case badArguments
        case http(Int)

        var errorDescription: String? {
            switch self {
            case .serverNotRunning:
                return "the local server isn't running"
            case .badArguments:
                return "tool arguments must be a JSON object"
            case .http(let code):
                return "the local server returned HTTP \(code)"
            }
        }
    }

    // MARK: - Transport

    private typealias Endpoint = (host: String, port: Int, bearer: String?)

    private func request(_ path: String, _ ep: Endpoint) throws -> URLRequest {
        guard let url = URL(string: "http://\(ep.host):\(ep.port)\(path)") else {
            throw CatalogError.serverNotRunning
        }
        var req = URLRequest(url: url)
        // Same per-launch bearer the chat stream uses (`ChatStreamClient`).
        if let bearer = ep.bearer, !bearer.isEmpty {
            req.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        req.timeoutInterval = 15
        return req
    }

    private func get<T: Decodable>(_ path: String, _ ep: Endpoint) async throws -> T {
        try await send(try request(path, ep))
    }

    private func post<T: Decodable>(
        _ path: String,
        _ ep: Endpoint,
        body: (some Encodable)? = Optional<Never>.none
    ) async throws -> T {
        var req = try request(path, ep)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body {
            req.httpBody = try JSONEncoder().encode(body)
        }
        // A tool can legitimately take a while (a query, a fetch). The engine
        // applies the per-server timeout from the config; give it room.
        req.timeoutInterval = 120
        return try await send(req)
    }

    private func send<T: Decodable>(_ req: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: req)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw CatalogError.http(http.statusCode)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: - Wire shapes (mirror vllm_mlx/api/models.py)

    private struct ServersResponse: Decodable {
        struct Server: Decodable {
            let name: String
            let state: String
            let transport: String
            let tools_count: Int
            let error: String?
        }
        let servers: [Server]
        let error: String?
        /// Optional so the app keeps working against an engine build that
        /// predates this field.
        let configured: Bool?
    }

    private struct ToolsResponse: Decodable {
        struct Tool: Decodable {
            let name: String
            let description: String
            let server: String
            let parameters: CodableJSON?
        }
        let tools: [Tool]
    }

    private struct ExecuteRequest: Encodable {
        let tool_name: String
        let arguments: CodableJSON
    }

    struct ExecuteResponse: Decodable {
        let tool_name: String
        let content: CodableJSON?
        let is_error: Bool
        let error_message: String?

        /// Flatten the engine's result into the string the model sees.
        var text: String {
            if let error_message, is_error { return error_message }
            guard let content else { return "" }
            if case .string(let s) = content { return s }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            if let data = try? encoder.encode(content),
               let s = String(data: data, encoding: .utf8) {
                return s
            }
            return ""
        }
    }
}
