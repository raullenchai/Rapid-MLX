import Darwin
import Foundation

enum ChatOwnerConnectionError: Error, Equatable, LocalizedError {
    case backendDisabled
    case invalidProfile
    case invalidRuntime
    case invalidDiscovery(String)
    case incompatibleOwner
    case ownerUnavailable
    case requestFailed(Int)

    var errorDescription: String? {
        switch self {
        case .backendDisabled: "The shared chat owner is not enabled."
        case .invalidProfile: "The chat profile name is invalid."
        case .invalidRuntime: "The bundled chat runtime is unavailable."
        case .invalidDiscovery: "The chat owner discovery record is invalid."
        case .incompatibleOwner: "The chat owner requires a Desktop update."
        case .ownerUnavailable: "The chat owner could not be started."
        case .requestFailed(let status): "The chat owner request failed (HTTP \(status))."
        }
    }
}

struct ChatOwnerStatus: Equatable, Sendable {
    let endpoint: URL
    let ownerEpoch: UUID
    let storeUUID: UUID
    let pid: Int32
    let processStartTime: TimeInterval
    let credentialFile: String
}

/// Immutable interpreter contract used to invoke Python's lock-owning bootstrap.
/// Shipping construction accepts only the authenticated bundle layout. Tests may
/// inject the supplied development venv while retaining the same hardened flags.
struct ChatOwnerRuntime: Sendable {
    let interpreter: URL
    let pythonHome: URL?
    let pythonPath: URL?
    let baseEnvironment: [String: String]

    static func bundled(
        resolution: ServerLocator.Resolution,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        authenticateBundle: (URL) -> Bool = DesktopRuntimeTrust.authenticateUncached
    ) throws -> Self {
        guard resolution.source == .bundled else {
            throw ChatOwnerConnectionError.invalidRuntime
        }
        try DesktopRuntimeTrust.verifyBeforeLaunch(
            resolution.binary,
            environment: environment,
            authenticateBundle: authenticateBundle
        )
        let root = resolution.binary.deletingLastPathComponent().deletingLastPathComponent()
        let interpreter = root.appendingPathComponent("python/bin/python3.12")
        let pythonPath = root.appendingPathComponent("site-packages")
        guard FileManager.default.isExecutableFile(atPath: interpreter.path),
              FileManager.default.fileExists(atPath: pythonPath.path) else {
            throw ChatOwnerConnectionError.invalidRuntime
        }
        return Self(
            interpreter: interpreter,
            pythonHome: root.appendingPathComponent("python"),
            pythonPath: pythonPath,
            baseEnvironment: environment
        )
    }

    init(
        interpreter: URL,
        pythonHome: URL? = nil,
        pythonPath: URL? = nil,
        baseEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.interpreter = interpreter
        self.pythonHome = pythonHome
        self.pythonPath = pythonPath
        self.baseEnvironment = baseEnvironment
    }

    fileprivate func ensureOwner(profile: String, chatHome: URL) async throws {
        guard FileManager.default.isExecutableFile(atPath: interpreter.path) else {
            throw ChatOwnerConnectionError.invalidRuntime
        }
        var environment = baseEnvironment
        environment["RAPID_MLX_CHAT_HOME"] = chatHome.path
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONSAFEPATH"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment.removeValue(forKey: "PYTHONSTARTUP")
        environment.removeValue(forKey: "RAPID_MLX_WATCHDOG_PPID")
        if let pythonHome { environment["PYTHONHOME"] = pythonHome.path }
        if let pythonPath { environment["PYTHONPATH"] = pythonPath.path }

        let process = Process()
        process.executableURL = interpreter
        process.arguments = [
            "-P", "-u", "-s", "-m", "rapid_mlx.chat_runtime.owner",
            "--profile", profile, "--ensure",
        ]
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        let box = ChatOwnerProcessBox(process)
        let status = try await Task.detached(priority: .userInitiated) {
            try box.process.run()
            box.process.waitUntilExit()
            return box.process.terminationStatus
        }.value
        guard status == 0 else { throw ChatOwnerConnectionError.ownerUnavailable }
    }
}

private final class ChatOwnerProcessBox: @unchecked Sendable {
    let process: Process
    init(_ process: Process) { self.process = process }
}

struct ChatOwnerDiscovery: Sendable {
    static let formatVersion = 1
    static let wireVersion = 1
    static let schemaVersion = 4
    static let maximumFileBytes = 65_536

    let status: ChatOwnerStatus
    fileprivate let bearer: String

    static func profileDirectory(
        profile: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        guard profile.range(
            of: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
            options: .regularExpression
        ) != nil else { throw ChatOwnerConnectionError.invalidProfile }
        if let override = environment["RAPID_MLX_CHAT_HOME"], !override.isEmpty {
            let root = URL(fileURLWithPath: override).standardizedFileURL
            guard root.path.hasPrefix("/") else { throw ChatOwnerConnectionError.invalidDiscovery("root") }
            return root.appendingPathComponent(profile, isDirectory: true)
        }
        guard let home = environment["HOME"], home.hasPrefix("/") else {
            throw ChatOwnerConnectionError.invalidDiscovery("home")
        }
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent("Library/Application Support/Rapid/chat", isDirectory: true)
            .appendingPathComponent(profile, isDirectory: true)
    }

    /// Returns nil only for absent or demonstrably stale state. Malformed,
    /// insecure, or incompatible records fail closed and never authorize a
    /// replacement launch. PID identity is checked but is never kill authority.
    static func read(
        profileDirectory: URL,
        processStartTime: (Int32) -> TimeInterval? = liveProcessStartTime
    ) throws -> Self? {
        let run = profileDirectory.appendingPathComponent("run", isDirectory: true)
        guard FileManager.default.fileExists(atPath: run.path) else { return nil }
        try validatePrivateDirectory(run)
        let discoveryURL = run.appendingPathComponent("discovery.json")
        guard FileManager.default.fileExists(atPath: discoveryURL.path) else { return nil }
        let data = try readPrivateFile(discoveryURL)
        let wire: WireRecord
        do {
            wire = try JSONDecoder().decode(WireRecord.self, from: data)
        } catch {
            throw ChatOwnerConnectionError.invalidDiscovery("json")
        }
        guard wire.formatVersion == formatVersion else {
            throw ChatOwnerConnectionError.invalidDiscovery("format")
        }
        guard wire.wireVersion == wireVersion, wire.schemaVersion == schemaVersion else {
            throw ChatOwnerConnectionError.incompatibleOwner
        }
        guard let epoch = UUID(uuidString: wire.ownerEpoch),
              epoch.uuidString.lowercased() == wire.ownerEpoch.lowercased(),
              let store = UUID(uuidString: wire.storeUUID),
              store.uuidString.lowercased() == wire.storeUUID.lowercased(),
              wire.pid > 0,
              wire.processStartTime.isFinite,
              wire.processStartTime > 0,
              wire.credentialFile == "bearer-\(wire.ownerEpoch.lowercased())",
              let endpoint = validatedEndpoint(wire.endpoint) else {
            throw ChatOwnerConnectionError.invalidDiscovery("fields")
        }
        guard let actualStart = processStartTime(wire.pid),
              abs(actualStart - wire.processStartTime) <= 0.01 else { return nil }
        let credentialURL = run.appendingPathComponent(wire.credentialFile)
        guard FileManager.default.fileExists(atPath: credentialURL.path) else { return nil }
        let credentialData = try readPrivateFile(credentialURL)
        guard let bearer = String(data: credentialData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !bearer.isEmpty,
              !bearer.contains("\r"), !bearer.contains("\n") else { return nil }
        return Self(
            status: ChatOwnerStatus(
                endpoint: endpoint,
                ownerEpoch: epoch,
                storeUUID: store,
                pid: wire.pid,
                processStartTime: wire.processStartTime,
                credentialFile: wire.credentialFile
            ),
            bearer: bearer
        )
    }

    private struct WireRecord: Decodable {
        let formatVersion: Int
        let storeUUID: String
        let ownerEpoch: String
        let pid: Int32
        let processStartTime: TimeInterval
        let endpoint: String
        let wireVersion: Int
        let schemaVersion: Int
        let credentialFile: String

        enum CodingKeys: String, CodingKey {
            case formatVersion = "format_version"
            case storeUUID = "store_uuid"
            case ownerEpoch = "owner_epoch"
            case pid
            case processStartTime = "process_start_time"
            case endpoint
            case wireVersion = "wire_version"
            case schemaVersion = "schema_version"
            case credentialFile = "credential_file"
        }
    }

    private static func validatedEndpoint(_ value: String) -> URL? {
        guard let components = URLComponents(string: value),
              components.scheme?.lowercased() == "http",
              components.host == "127.0.0.1",
              let port = components.port, (1...65_535).contains(port),
              components.user == nil, components.password == nil,
              components.path.isEmpty, components.query == nil, components.fragment == nil,
              let url = components.url else { return nil }
        return url
    }

    private static func validatePrivateDirectory(_ url: URL) throws {
        var info = stat()
        guard lstat(url.path, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFDIR,
              info.st_uid == getuid(),
              (info.st_mode & 0o077) == 0 else {
            throw ChatOwnerConnectionError.invalidDiscovery("permissions")
        }
    }

    private static func readPrivateFile(_ url: URL) throws -> Data {
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else {
            if errno == ENOENT { throw CocoaError(.fileNoSuchFile) }
            throw ChatOwnerConnectionError.invalidDiscovery("file")
        }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG,
              info.st_uid == getuid(),
              (info.st_mode & 0o077) == 0,
              info.st_size <= maximumFileBytes else {
            throw ChatOwnerConnectionError.invalidDiscovery("permissions")
        }
        var result = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            guard count >= 0 else { throw ChatOwnerConnectionError.invalidDiscovery("read") }
            if count == 0 { break }
            result.append(buffer, count: count)
            guard result.count <= maximumFileBytes else {
                throw ChatOwnerConnectionError.invalidDiscovery("size")
            }
        }
        return result
    }

    static func liveProcessStartTime(_ pid: Int32) -> TimeInterval? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.stride)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
        return TimeInterval(info.pbi_start_tvsec) + TimeInterval(info.pbi_start_tvusec) / 1_000_000
    }
}

/// Owns only the Desktop client lease. The Python owner remains detached and
/// shared with CLI clients; closing this actor never signals an owner PID.
actor ChatOwnerConnection {
    private struct LiveOwner: Sendable {
        let discovery: ChatOwnerDiscovery
        let leaseSeconds: TimeInterval
    }

    private let profile: String
    private let profileDirectory: URL
    private let launcher: @Sendable (String, URL) async throws -> Void
    private let processStartTime: @Sendable (Int32) -> TimeInterval?
    private let clientID = UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    private let transport: ChatOwnerControlTransport
    private var live: LiveOwner?
    private var heartbeatTask: Task<Void, Never>?
    private var closed = false

    init(
        profile: String = "default",
        runtime: ChatOwnerRuntime,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        session: URLSession? = nil,
        launcher: (@Sendable (String, URL) async throws -> Void)? = nil,
        processStartTime: @escaping @Sendable (Int32) -> TimeInterval? = ChatOwnerDiscovery.liveProcessStartTime
    ) throws {
        self.profile = profile
        profileDirectory = try ChatOwnerDiscovery.profileDirectory(
            profile: profile,
            environment: environment
        )
        self.launcher = launcher ?? { profile, home in
            try await runtime.ensureOwner(profile: profile, chatHome: home)
        }
        self.processStartTime = processStartTime
        transport = ChatOwnerControlTransport(session: session)
    }

    static func connectIfEnabled(
        backend: ChatBackend,
        profile: String = "default",
        resolution: ServerLocator.Resolution? = ServerLocator.locate(),
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async throws -> ChatOwnerConnection? {
        guard backend == .ownerV1 else { return nil }
        guard let resolution else { throw ChatOwnerConnectionError.invalidRuntime }
        let runtime = try ChatOwnerRuntime.bundled(
            resolution: resolution,
            environment: environment
        )
        let connection = try ChatOwnerConnection(
            profile: profile,
            runtime: runtime,
            environment: environment
        )
        _ = try await connection.turnClient()
        return connection
    }

    func turnClient() async throws -> any ChatTurnClient {
        let owner = try await establish()
        return try URLSessionChatTurnClient(
            endpoint: owner.discovery.status.endpoint,
            bearer: owner.discovery.bearer
        )
    }

    /// Explicit recovery seam for P04 after an owner interruption. The next
    /// client always uses freshly authenticated discovery and the rotated bearer.
    @discardableResult
    func reconnect() async throws -> any ChatTurnClient {
        live = nil
        return try await turnClient()
    }

    func status() async throws -> ChatOwnerStatus {
        try await establish().discovery.status
    }

    func close() async {
        guard !closed else { return }
        closed = true
        heartbeatTask?.cancel()
        heartbeatTask = nil
        if let live {
            try? await transport.detach(
                endpoint: live.discovery.status.endpoint,
                bearer: live.discovery.bearer,
                clientID: clientID
            )
        }
        live = nil
    }

    private func establish() async throws -> LiveOwner {
        guard !closed else { throw CancellationError() }
        var discovery = try ChatOwnerDiscovery.read(profileDirectory: profileDirectory, processStartTime: processStartTime)
        if let candidate = discovery {
            do {
                try await transport.validate(candidate)
            } catch ChatOwnerConnectionError.incompatibleOwner {
                throw ChatOwnerConnectionError.incompatibleOwner
            } catch {
                discovery = nil
            }
        }
        if discovery == nil {
            try await launcher(profile, profileDirectory.deletingLastPathComponent())
            discovery = try ChatOwnerDiscovery.read(profileDirectory: profileDirectory, processStartTime: processStartTime)
            guard let discovery else { throw ChatOwnerConnectionError.ownerUnavailable }
            try await transport.validate(discovery)
        }
        guard let discovery else { throw ChatOwnerConnectionError.ownerUnavailable }
        if let live, live.discovery.status == discovery.status,
           live.discovery.bearer == discovery.bearer {
            return live
        }
        let lease = try await transport.attach(
            discovery,
            clientID: clientID
        )
        let established = LiveOwner(discovery: discovery, leaseSeconds: lease)
        live = established
        startHeartbeatIfNeeded()
        return established
    }

    private func startHeartbeatIfNeeded() {
        guard heartbeatTask == nil else { return }
        heartbeatTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                let delay = await self.heartbeatDelay()
                do { try await Task.sleep(for: .seconds(delay)) } catch { return }
                await self.renewLease()
            }
        }
    }

    private func heartbeatDelay() -> TimeInterval {
        min(15, max(0.1, (live?.leaseSeconds ?? 45) / 3))
    }

    private func renewLease() async {
        guard !closed, let current = live else { return }
        do {
            let lease = try await transport.attach(current.discovery, clientID: clientID)
            live = LiveOwner(discovery: current.discovery, leaseSeconds: lease)
        } catch {
            live = nil
            _ = try? await establish()
        }
    }
}

private final class ChatOwnerNoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

private final class ChatOwnerControlTransport: @unchecked Sendable {
    private let session: URLSession
    private let redirectDelegate = ChatOwnerNoRedirectDelegate()

    init(session: URLSession? = nil) {
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = 2
            configuration.timeoutIntervalForResource = 5
            configuration.urlCache = nil
            configuration.connectionProxyDictionary = [:]
            self.session = URLSession(configuration: configuration)
        }
    }

    func validate(_ discovery: ChatOwnerDiscovery) async throws {
        let (data, response) = try await request(
            discovery,
            path: ["capabilities"],
            method: "GET"
        )
        guard (200..<300).contains(response.statusCode) else {
            throw ChatOwnerConnectionError.requestFailed(response.statusCode)
        }
        let capabilities = try JSONDecoder().decode(Capabilities.self, from: data)
        guard capabilities.wireVersion == ChatOwnerDiscovery.wireVersion,
              capabilities.schemaVersion == ChatOwnerDiscovery.schemaVersion else {
            throw ChatOwnerConnectionError.incompatibleOwner
        }
        guard capabilities.ownerEpoch == discovery.status.ownerEpoch.uuidString.lowercased(),
              capabilities.storeUUID == discovery.status.storeUUID.uuidString.lowercased() else {
            throw ChatOwnerConnectionError.invalidDiscovery("handshake")
        }
    }

    func attach(_ discovery: ChatOwnerDiscovery, clientID: String) async throws -> TimeInterval {
        let body = try JSONEncoder().encode(LeaseRequest(clientID: clientID))
        let (data, response) = try await request(
            discovery,
            path: ["clients"],
            method: "POST",
            body: body
        )
        guard (200..<300).contains(response.statusCode) else {
            throw ChatOwnerConnectionError.requestFailed(response.statusCode)
        }
        let lease = try JSONDecoder().decode(LeaseResponse.self, from: data)
        guard lease.clientID == clientID, lease.leaseSeconds.isFinite, lease.leaseSeconds > 0 else {
            throw ChatOwnerConnectionError.invalidDiscovery("lease")
        }
        return lease.leaseSeconds
    }

    func detach(endpoint: URL, bearer: String, clientID: String) async throws {
        let discovery = ChatOwnerDiscovery(
            status: ChatOwnerStatus(
                endpoint: endpoint,
                ownerEpoch: UUID(), storeUUID: UUID(), pid: 1,
                processStartTime: 1, credentialFile: "unused"
            ),
            bearer: bearer
        )
        let (_, response) = try await request(
            discovery,
            path: ["clients", clientID],
            method: "DELETE"
        )
        guard (200..<300).contains(response.statusCode) else {
            throw ChatOwnerConnectionError.requestFailed(response.statusCode)
        }
    }

    private func request(
        _ discovery: ChatOwnerDiscovery,
        path: [String],
        method: String,
        body: Data? = nil
    ) async throws -> (Data, HTTPURLResponse) {
        var url = discovery.status.endpoint.appendingPathComponent("v1/rapid/chat")
        for component in path { url.appendPathComponent(component) }
        var request = URLRequest(
            url: url,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: 2
        )
        request.httpMethod = method
        request.httpBody = body
        request.setValue("Bearer \(discovery.bearer)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, rawResponse) = try await session.data(for: request, delegate: redirectDelegate)
        guard let response = rawResponse as? HTTPURLResponse else {
            throw ChatOwnerConnectionError.ownerUnavailable
        }
        return (data, response)
    }

    private struct Capabilities: Decodable {
        let wireVersion: Int
        let schemaVersion: Int
        let storeUUID: String
        let ownerEpoch: String
        enum CodingKeys: String, CodingKey {
            case wireVersion = "wire_version"
            case schemaVersion = "schema_version"
            case storeUUID = "store_uuid"
            case ownerEpoch = "owner_epoch"
        }
    }

    private struct LeaseRequest: Encodable {
        let clientID: String
        enum CodingKeys: String, CodingKey { case clientID = "client_id" }
    }

    private struct LeaseResponse: Decodable {
        let clientID: String
        let leaseSeconds: TimeInterval
        enum CodingKeys: String, CodingKey {
            case clientID = "client_id"
            case leaseSeconds = "lease_seconds"
        }
    }
}

/// A fresh value is captured from ServerManager for every owner turn so a
/// sidecar restart cannot reuse its old port or inference bearer.
struct ChatOwnerInferenceTarget: Sendable, Equatable {
    let inferenceURL: String
    let inferenceAPIKey: String

    @MainActor
    init?(server: ServerManager) {
        guard let model = server.servingAlias,
              !model.isEmpty,
              let bearer = server.activeBearer,
              !bearer.isEmpty,
              (1...65_535).contains(server.activePort) else { return nil }
        inferenceURL = "http://127.0.0.1:\(server.activePort)/v1"
        inferenceAPIKey = bearer
    }
}
