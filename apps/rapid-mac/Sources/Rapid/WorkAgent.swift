import AppKit
import CryptoKit
import Darwin
import Foundation
import Observation

extension Notification.Name {
    static let newWorkChat = Notification.Name("Rapid.NewWorkChat")
}

struct WorkStep: Identifiable {
    let id: String
    var command: String
    var status: String
    var output: String = ""
}

struct WorkPermission: Identifiable {
    let id: String
    let rpcID: Any
    let sessionID: String
    let toolCallID: String
    let command: String
    let options: [[String: Any]]

    init(id: String, sessionID: String, toolCallID: String, command: String,
         options: [[String: Any]], rpcID: Any? = nil) {
        self.id = id
        self.rpcID = rpcID ?? id
        self.sessionID = sessionID
        self.toolCallID = toolCallID
        self.command = command
        self.options = options
    }
}

@MainActor @Observable
final class WorkAgent {
    static let shared = WorkAgent()
    static var gooseChat: Bool { Bundle.main.bundleIdentifier == "com.rapidmlx.rapid" }
    static var fullAccess: Bool { gooseChat }
    var onChatEvent: ((GooseChatEvent) -> Void)?
    var currentSessionID: String? { sessionID }
    private(set) var text = ""
    private(set) var steps: [WorkStep] = []
    private(set) var isRunning = false
    private(set) var isBusy = false
    private(set) var isPreparing = false
    private(set) var isOpeningSession = false
    private(set) var conversationRevision = 0
    var error: String?
    var permission: WorkPermission?
    private var process: Process?
    private var proxyProcess: Process?
    private var bridgeProcess: Process?
    private var port = 0
    private var proxyPort = 0
    private var bridgePort = 0
    private var secret = ""
    private var connection: String?
    private var sessionID: String? { turnOwner.sessionID }
    private var sessionHasPrompted = false
    private var pendingGooseChatSession: Task<Void, Error>?
    private var prefixWarmTask: Task<Void, Never>?
    private(set) var prefixWarmStartedAt: Date?
    private(set) var isPrefixWarming = false
    private var proxyBearerFileURL: URL?
    private var nextID = 1
    private var waiters: [String: CheckedContinuation<Data, Error>] = [:]
    private var streams: [Task<Void, Never>] = []
    private var folder: URL? { turnOwner.folder }
    private var serverIdentity: String?
    private var sandboxProfileURLs: [URL] = []
    private var pendingPermissions: [WorkPermission] = []
    private var toolIdentities: [String: String] = [:]
    private var rawToolOutputs: [String: String] = [:]
    private var generation: Int { turnOwner.generation }
    private let turnOwner = GooseTurnOwner()
    var transport = ACPTransport.live
    var lifecycleGeneration: Int { generation }

    init() {}

    func acceptSessionForTesting(_ id: String, folder: URL) {
        turnOwner.bind(sessionID: id, folder: folder)
    }

    func handleWire(_ data: Data) throws {
        guard let message = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkError.message("Invalid ACP message.")
        }
        handle(message)
    }

    func beginNewChat() {
        conversationRevision += 1
        if isPreparing || isBusy || isOpeningSession { stop() }
        else if isRunning { isOpeningSession = true }
        text = ""
        steps = []
        error = nil
    }

    /// Start the next Goose session while the user is composing a new chat.
    /// A Send joins this task, so a fast Send cannot open a second session.
    func preopenGooseChatSession() {
        guard Self.gooseChat, isRunning, !isBusy, sessionHasPrompted else { return }
        let previous = pendingGooseChatSession
        let requestedGeneration = generation
        pendingGooseChatSession = Task {
            if let previous { _ = try? await previous.value }
            try Task.checkCancellation()
            guard generation == requestedGeneration else { throw CancellationError() }
            try await newSession()
        }
    }

    func ensureNewGooseChatSession() async throws {
        if let pendingGooseChatSession {
            try await pendingGooseChatSession.value
            self.pendingGooseChatSession = nil
        } else if sessionHasPrompted {
            try await newSession()
        }
    }

    /// The model server may restart during a Goose turn and rotate its key.
    /// Keep the proxy's upstream key current without interrupting that turn.
    func updateProxyBearer(_ bearer: String) {
        guard Self.gooseChat, let proxyBearerFileURL else { return }
        do {
            try Data(bearer.utf8).write(to: proxyBearerFileURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600],
                ofItemAtPath: proxyBearerFileURL.path)
        } catch {
            self.error = "Could not refresh the local model connection."
        }
    }

    func prepare(folder: URL, rapidPort: Int, model: String, bearer: String) async throws {
        // A Send can arrive while ready-model prewarm is still opening Goose.
        for _ in 0..<200 {
            try Task.checkCancellation()
            if !isPreparing { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        try Task.checkCancellation()
        guard !isPreparing else { throw WorkError.message("Chat is already starting.") }
        let requestedRevision = conversationRevision
        let folder = folder.resolvingSymlinksInPath()
        let identity = "\(rapidPort):\(model):\(bearer)"
        if isRunning, sessionID != nil, self.folder == folder, serverIdentity == identity { return }
        isPreparing = true
        defer { isPreparing = false }
        stop()
        let prepareGeneration = generation
        var completed = false
        defer { if !completed && generation == prepareGeneration { stop() } }
        turnOwner.setFolder(folder)
        serverIdentity = identity
        error = nil
        text = ""
        steps = []
        let fileManager = FileManager.default
        let resources = Bundle.main.resourceURL!
        let goose = resources.appendingPathComponent("Helpers/goose")
        let proxy = resources.appendingPathComponent("Helpers/work-inference-proxy.py")
        guard fileManager.isExecutableFile(atPath: goose.path),
              fileManager.fileExists(atPath: proxy.path) else {
            throw WorkError.message("Chat helper is missing from this build.")
        }
        let python = resources.appendingPathComponent("rapid-mlx/python/bin")
        let pythonExecutable = python.appendingPathComponent("python3")
        let rootSupport = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(Self.gooseChat ? "Library/Application Support/Rapid/agent-goose-chat" : (Self.fullAccess ? "Library/Application Support/Rapid/agent-full" : "Library/Application Support/Rapid/agent"), isDirectory: true)
        let folderHash = SHA256.hash(data: Data(folder.path.utf8))
            .map { String(format: "%02x", $0) }.joined()
        let support = rootSupport.appendingPathComponent("workspaces/\(folderHash)", isDirectory: true)
        let temporary = support.appendingPathComponent("tmp", isDirectory: true)
        let providerDirectory = support.appendingPathComponent("config/custom_providers", isDirectory: true)
        try fileManager.createDirectory(at: rootSupport, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: rootSupport.path)
        if Self.gooseChat { _ = try GooseToolPresentation.containedDestination(support, root: rootSupport) }
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: support.path)
        if Self.gooseChat { try GoosePermissionPolicy.resetNativePolicy(in: support) }
        if Self.gooseChat {
            let config = support.appendingPathComponent("config/config.yaml")
            _ = try GooseToolPresentation.containedDestination(config, root: support)
            try fileManager.createDirectory(at: config.deletingLastPathComponent(),
                                            withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
            if !fileManager.fileExists(atPath: config.path) {
                let search = resources.appendingPathComponent("Helpers/work-web-search-mcp.py")
                func yamlString(_ value: String) throws -> String {
                    let data = try JSONSerialization.data(withJSONObject: value, options: .fragmentsAllowed)
                    return String(decoding: data, as: UTF8.self)
                }
                let platforms = ["developer", "skills", "extensionmanager", "analyze", "apps",
                                 "summon", "tom", "todo", "chatrecall", "scheduler", "summarize",
                                 "codemode", "orchestrator"]
                var lines = ["extensions:"]
                for name in platforms {
                    lines += ["  \(name):", "    type: platform",
                              "    enabled: \(["developer", "skills", "extensionmanager"].contains(name))"]
                }
                for name in ["computercontroller", "memory", "autovisualiser", "tutorial"] {
                    lines += ["  \(name):", "    type: builtin", "    enabled: false"]
                }
                lines += ["  rapidweb:", "    type: stdio", "    enabled: true",
                          "    cmd: \(try yamlString(pythonExecutable.path))",
                          "    args: [\"-I\", \"-S\", \(try yamlString(search.path))]",
                          "    env_keys: [\"WEB_SEARCH_GATE_PORT\", \"WEB_SEARCH_GATE_TOKEN\"]"]
                try (lines.joined(separator: "\n") + "\n").write(to: config, atomically: true, encoding: .utf8)
            }
        }
        for directory in [temporary, providerDirectory] {
            if Self.gooseChat { _ = try GooseToolPresentation.containedDestination(directory, root: support) }
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        }
        if Self.gooseChat {
            proxyBearerFileURL = support.appendingPathComponent("proxy-upstream-bearer")
            updateProxyBearer(bearer)
        }
        port = try Self.freePort()
        repeat { proxyPort = try Self.freePort() } while proxyPort == port
        repeat { bridgePort = try Self.freePort() } while bridgePort == port || bridgePort == proxyPort
        secret = UUID().uuidString + UUID().uuidString
        let inferenceKey = UUID().uuidString + UUID().uuidString
        let contextLimit = Self.fullAccess
            ? await Self.contextLimit(rapidPort: rapidPort, model: model, bearer: bearer)
            : 16384
        try checkPreparation(prepareGeneration)
        guard conversationRevision == requestedRevision else { throw WorkError.message("Chat start cancelled.") }
        let provider: [String: Any] = [
            "name": "custom_rapid_local", "engine": "openai", "display_name": "Rapid Local",
            "api_key_env": "RAPID_LOCAL_WORK_KEY",
            "base_url": "http://127.0.0.1:\(Self.gooseChat ? bridgePort : proxyPort)/v1/chat/completions",
            "models": [["name": model, "context_limit": contextLimit]],
            "supports_streaming": true, "requires_auth": true, "dynamic_models": false,
        ]
        let providerURL = providerDirectory.appendingPathComponent("custom_rapid_local.json")
        try JSONSerialization.data(withJSONObject: provider).write(to: providerURL, options: .atomic)
        let skillDirectory = support.appendingPathComponent(".agents/skills/local-documents", isDirectory: true)
        if Self.fullAccess {
            if fileManager.fileExists(atPath: skillDirectory.path) {
                try fileManager.removeItem(at: skillDirectory)
            }
        } else {
            try fileManager.createDirectory(at: skillDirectory, withIntermediateDirectories: true)
            let skillSource = resources.appendingPathComponent("Helpers/local-documents/SKILL.md")
            try? fileManager.removeItem(at: skillDirectory.appendingPathComponent("SKILL.md"))
            try fileManager.copyItem(at: skillSource, to: skillDirectory.appendingPathComponent("SKILL.md"))
        }
        let gooseProfileURL = temporary.appendingPathComponent("work-\(UUID().uuidString).sb")
        let proxyProfileURL = temporary.appendingPathComponent("proxy-\(UUID().uuidString).sb")
        if !Self.fullAccess {
            try Self.profile(folder: folder, support: support, temporary: temporary,
                             resources: resources, outboundPorts: [proxyPort, port])
                .write(to: gooseProfileURL, atomically: true, encoding: .utf8)
            sandboxProfileURLs.append(gooseProfileURL)
            try Self.profile(folder: folder, support: support, temporary: temporary,
                             resources: resources, outboundPorts: [rapidPort])
                .write(to: proxyProfileURL, atomically: true, encoding: .utf8)
            sandboxProfileURLs.append(proxyProfileURL)
        }
        // Start from an allowlist: the shell tool must not inherit the app's
        // unrelated credentials or proxy configuration.
        var env: [String: String] = [
            "HOME": Self.helperHome(gooseChat: Self.gooseChat, fullAccess: Self.fullAccess,
                                     userHome: fileManager.homeDirectoryForCurrentUser.path,
                                     support: support.path),
            "USER": NSUserName(),
            "LOGNAME": NSUserName(),
            "LANG": "en_US.UTF-8",
            "NO_PROXY": "127.0.0.1,localhost",
        ]
        env["GOOSE_PATH_ROOT"] = support.path
        env["WORK_HELPER_OWNERSHIP_TOKEN"] = UUID().uuidString + UUID().uuidString
        env["GOOSE_SERVER__SECRET_KEY"] = secret
        if Self.gooseChat {
            env["GOOSE_SHELL"] = resources.appendingPathComponent("Helpers/bash").path
        }
        env["GOOSE_TELEMETRY_OFF"] = "1"
        env["GOOSE_TELEMETRY_ENABLED"] = "false"
        env["GOOSE_DISABLE_KEYRING"] = "1"
        env["GOOSE_PROVIDER"] = "custom_rapid_local"
        env["GOOSE_MODEL"] = model
        env["GOOSE_MODE"] = Self.gooseChat ? "approve" : (Self.fullAccess ? "auto" : "approve")
        env["RAPID_LOCAL_WORK_KEY"] = Self.gooseChat ? inferenceKey : bearer
        if Self.gooseChat {
            let searchToken = UUID().uuidString + UUID().uuidString
            env["WEB_SEARCH_GATE_PORT"] = String(proxyPort)
            env["WEB_SEARCH_GATE_TOKEN"] = searchToken
        }
        env["PATH"] = python.path + ":/usr/bin:/bin:/usr/sbin:/sbin"
        env["PYTHONPATH"] = resources.appendingPathComponent("rapid-mlx/site-packages").path
        if Self.fullAccess {
            let venv = rootSupport.appendingPathComponent("venv")
            let pipMarker = venv.appendingPathComponent(".rapid-pip-ready")
            if !fileManager.fileExists(atPath: pipMarker.path) {
                let setup = Process()
                setup.executableURL = pythonExecutable
                setup.arguments = ["-m", "venv", "--without-pip", venv.path]
                try setup.run()
                setup.waitUntilExit()
                guard setup.terminationStatus == 0 else { throw WorkError.message("Could not create Work Python venv.") }
                let bootstrap = Process()
                bootstrap.executableURL = venv.appendingPathComponent("bin/python3")
                bootstrap.arguments = ["-c", "import pathlib,sysconfig,zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sysconfig.get_path('purelib'))",
                                       resources.appendingPathComponent("Helpers/pip.whl").path]
                try bootstrap.run()
                bootstrap.waitUntilExit()
                guard bootstrap.terminationStatus == 0 else { throw WorkError.message("Could not install Work venv pip.") }
                let pipCommand = venv.appendingPathComponent("bin/pip")
                try "#!/bin/sh\nexec \"$(dirname \"$0\")/python3\" -m pip \"$@\"\n"
                    .write(to: pipCommand, atomically: true, encoding: .utf8)
                try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: pipCommand.path)
                try Data().write(to: pipMarker)
            }
            env["PATH"] = venv.appendingPathComponent("bin").path + ":" + env["PATH", default: ""]
            env["VIRTUAL_ENV"] = venv.path
        } else {
            env["PIP_NO_INDEX"] = "1"
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = temporary.path
        var proxyEnv = env
        proxyEnv.removeValue(forKey: "GOOSE_SERVER__SECRET_KEY")
        proxyEnv.removeValue(forKey: "RAPID_LOCAL_WORK_KEY")
        proxyEnv["WORK_PROXY_BEARER"] = bearer
        if let proxyBearerFileURL { proxyEnv["WORK_PROXY_BEARER_FILE"] = proxyBearerFileURL.path }
        proxyEnv["WORK_PROXY_METRICS_KEY"] = secret
        if Self.gooseChat {
            proxyEnv["WORK_SEARCH_TOKEN"] = env["WEB_SEARCH_GATE_TOKEN"]
            if let selected = UserDefaults.standard.string(forKey: "rapid.webSearch.provider"),
               let setting = ["parallel": ("rapid.web-search.parallel", "PARALLEL_API_KEY"),
                              "tavily": ("rapid.web-search.tavily", "TAVILY_API_KEY"),
                              "brave": ("rapid.web-search.brave", "BRAVE_API_KEY")][selected],
               case .found(let key) = SystemKeychain().readWithoutUserInteraction(account: setting.0),
               !key.isEmpty {
                proxyEnv[setting.1] = key.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        try checkPreparation(prepareGeneration)
        proxyProcess = try Self.launchHelper(
            profile: Self.fullAccess ? nil : proxyProfileURL, python: pythonExecutable,
            executable: pythonExecutable,
            arguments: ["-I", "-S", proxy.path, String(proxyPort), String(rapidPort), model]
                + (Self.fullAccess ? ["--full-access"] : [])
                + (Self.gooseChat ? ["--goose-chat"] : []),
            directory: support, environment: proxyEnv)
        let startedProxy = proxyProcess!
        guard await Self.waitForHealth(process: startedProxy, port: proxyPort) else {
            throw WorkError.message("The local Chat inference gate did not start.")
        }
        try checkPreparation(prepareGeneration)
        if Self.gooseChat {
            var bridgeEnv = env
            bridgeEnv.removeValue(forKey: "GOOSE_SERVER__SECRET_KEY")
            bridgeEnv.removeValue(forKey: "RAPID_LOCAL_WORK_KEY")
            bridgeEnv["WORK_BRIDGE_SCOPED_KEY"] = inferenceKey
            bridgeEnv["WORK_BRIDGE_UPSTREAM_KEY"] = bearer
            let watchdog = resources.appendingPathComponent("Helpers/work-helper-watchdog.py")
            try checkPreparation(prepareGeneration)
            bridgeProcess = try Self.launchHelper(profile: nil, python: pythonExecutable,
                executable: pythonExecutable,
                arguments: ["-I", "-S", watchdog.path, "--credential-bridge",
                            String(bridgePort), String(proxyPort)],
                directory: support, environment: bridgeEnv)
            let startedBridge = bridgeProcess!
            guard await Self.waitForHealth(process: startedBridge, port: bridgePort) else {
                throw WorkError.message("The local inference credential bridge did not start.")
            }
            try checkPreparation(prepareGeneration)
        }
        guard conversationRevision == requestedRevision else { throw WorkError.message("Chat start cancelled.") }
        try checkPreparation(prepareGeneration)
        let gooseArguments = ["serve", "--host", "127.0.0.1", "--port", String(port),
                              "--platform", Self.fullAccess ? "cli" : "desktop"]
            + (Self.gooseChat ? [] : ["--with-builtin",
                                       Self.builtinExtensions(gooseChat: false, fullAccess: Self.fullAccess)])
        process = try Self.launchHelper(
            profile: Self.fullAccess ? nil : gooseProfileURL, python: pythonExecutable,
            executable: goose,
            arguments: gooseArguments,
            directory: folder, environment: env)
        let launchGeneration = generation
        process?.terminationHandler = { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.helperStreamEnded(generation: launchGeneration, session: self.sessionID)
            }
        }
        let startedProcess = process!
        guard await Self.waitForHealth(process: startedProcess, port: port) else {
            throw WorkError.message("The Chat helper did not start.")
        }
        try checkPreparation(prepareGeneration)
        guard conversationRevision == requestedRevision else { throw WorkError.message("Chat start cancelled.") }
        isRunning = true
        let (initResult, response) = try await sendImmediate(method: "initialize", params: [
            "protocolVersion": 1,
            "clientCapabilities": ["fs": ["readTextFile": false, "writeTextFile": false]],
            "clientInfo": ["name": "Rapid Work", "version": "0.1"],
        ])
        try checkPreparation(prepareGeneration)
        guard conversationRevision == requestedRevision else { throw WorkError.message("Chat start cancelled.") }
        guard initResult["error"] == nil,
              let id = response.value(forHTTPHeaderField: "Acp-Connection-Id") else {
            throw WorkError.message("The Chat connection failed to initialize.")
        }
        connection = id
        openStream(session: nil)
        try await newSession()
        try checkPreparation(prepareGeneration)
        guard conversationRevision == requestedRevision else { throw WorkError.message("Chat start cancelled.") }
        completed = true
        if Self.gooseChat { startPrefixWarm(model: model, bearer: bearer) }
    }

    private func startPrefixWarm(model: String, bearer: String) {
        guard prefixWarmTask == nil else { return }
        let generation = self.generation
        let port = proxyPort
        let prefix = Bundle.main.resourceURL!
            .appendingPathComponent("Helpers/goose-mini-prefix.json")
        isPrefixWarming = true
        prefixWarmStartedAt = Date()
        prefixWarmTask = Task {
            defer {
                if self.generation == generation {
                    self.isPrefixWarming = false
                    self.prefixWarmTask = nil
                }
            }
            guard let data = try? Data(contentsOf: prefix),
                  var body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else { return }
            body["model"] = model
            body["max_tokens"] = 1
            body["temperature"] = 0
            body["stream"] = true
            guard let payload = try? JSONSerialization.data(withJSONObject: body) else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.httpBody = payload
            request.timeoutInterval = 180
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    /// A fast Send takes priority over speculative prefix prefill.
    func waitForPrefixWarm() async {
        prefixWarmTask?.cancel()
    }

    func checkPreparation(_ expected: Int) throws {
        guard generation == expected, !Task.isCancelled else { throw CancellationError() }
    }

    func newSession() async throws {
        try Task.checkCancellation()
        if !isRunning {
            isOpeningSession = false
            text = ""
            steps = []
            error = nil
            return
        }
        guard let folder else { return }
        guard !isBusy else { throw WorkError.message("Stop the current task before starting a new session.") }
        let sessionGeneration = generation
        isOpeningSession = true
        defer { if generation == sessionGeneration { isOpeningSession = false } }
        if let old = sessionID {
            turnOwner.clearSession()
            try? await notify(method: "session/cancel", params: ["sessionId": old], session: old)
            guard generation == sessionGeneration else { throw WorkError.message("Chat stopped.") }
            streams.last?.cancel()
        }
        text = ""
        steps = []
        error = nil
        permission = nil
        pendingPermissions = []
        toolIdentities = [:]
        rawToolOutputs = [:]
        let result = try await call(method: "session/new", params: [
            "cwd": folder.path, "mcpServers": [],
            "_meta": ["client": "Rapid Work"],
        ])
        guard generation == sessionGeneration else { throw WorkError.message("Chat stopped.") }
        guard let sid = result["sessionId"] as? String else {
            throw WorkError.message("The Chat session could not be created.")
        }
        turnOwner.bind(sessionID: sid, folder: folder)
        sessionHasPrompted = false
        onChatEvent?(.session(sid))
        openStream(session: sid)
        try await Task.sleep(for: .milliseconds(300))
        guard generation == sessionGeneration else { throw WorkError.message("Chat stopped.") }
    }

    func prompt(_ message: String) async {
        guard let sessionID else { return }
        sessionHasPrompted = true
        let promptGeneration = generation
        let metricSequence = Self.gooseChat ? await inferenceMetric()?.sequence : nil
        let previousTextCount = text.count
        let previousStepCount = steps.count
        isBusy = true
        error = nil
        let metricPoll = Task { [weak self] in
            guard let self, Self.gooseChat else { return }
            while !Task.isCancelled && generation == promptGeneration {
                if let metric = await inferenceMetric(),
                   metric.sequence > (metricSequence ?? 0),
                   let rate = metric.rate, rate > 0,
                   !Task.isCancelled, generation == promptGeneration {
                    onChatEvent?(.throughput(rate))
                }
                try? await Task.sleep(for: .milliseconds(350))
            }
        }
        defer { metricPoll.cancel() }
        do {
            let instruction = Self.fullAccess ? message : "Use the local-documents skill. Stay inside the selected folder. " +
                "Read PDF with pypdf, CSV with csv.DictReader, XLSX with openpyxl. " +
                "Use Decimal for money, verify totals, format currency with thousands separators and two decimal places, and write results in the folder. Put every total label and its amount on the same line. " +
                "Do not install packages, use the network, or start background or detached processes. User request: " + message
            let result = try await call(method: "session/prompt", params: [
                "sessionId": sessionID,
                "prompt": [["type": "text", "text": instruction]],
            ], session: sessionID, timeout: 900)
            if Self.fullAccess, generation == promptGeneration,
               Self.shouldContinue(gooseChat: Self.gooseChat, result: result,
                                   newText: String(text.dropFirst(previousTextCount)),
                                   newSteps: Array(steps.dropFirst(previousStepCount))) {
                _ = try await call(method: "session/prompt", params: [
                    "sessionId": sessionID,
                    "prompt": [["type": "text", "text":
                        Self.gooseChat
                            ? "Continue the unfinished response without repeating completed work."
                            : "Continue from the existing files. Review failed tool calls and verify every requested input was processed. Repair incomplete outputs and finish the remaining checks or tests without repeating completed work."]],
                ], session: sessionID, timeout: 900)
            }
            if Self.gooseChat, generation == promptGeneration,
               let metric = await inferenceMetric(), metric.sequence > (metricSequence ?? 0),
               let rate = metric.rate, rate > 0 {
                onChatEvent?(.throughput(rate))
            }
        } catch {
            if generation == promptGeneration {
                self.error = redact(error.localizedDescription)
                onChatEvent?(.error(redact(error.localizedDescription)))
            }
        }
        if generation == promptGeneration { isBusy = false }
    }

    private func inferenceMetric() async -> (sequence: Int, rate: Double?)? {
        guard let url = URL(string: "http://127.0.0.1:\(proxyPort)/metrics") else { return nil }
        var request = URLRequest(url: url)
        request.setValue(secret, forHTTPHeaderField: "X-Metrics-Key")
        request.timeoutInterval = 2
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let metric = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sequence = metric["sequence"] as? Int else { return nil }
        return (sequence, metric["tokensPerSecond"] as? Double)
    }

    static func helperHome(gooseChat: Bool, fullAccess: Bool,
                           userHome: String, support: String) -> String {
        // Goose Chat must not import the user's global agent skills into every
        // conversation. They can add thousands of tokens and change tool use.
        gooseChat ? support : (fullAccess ? userHome : support)
    }

    static func builtinExtensions(gooseChat: Bool, fullAccess: Bool) -> String {
        gooseChat ? "developer,computercontroller,memory,autovisualiser,tutorial" : (fullAccess
            ? "developer,computercontroller,memory,autovisualiser,tutorial" : "developer")
    }

    func cancel() {
        guard let sessionID else { return }
        let cancelledGeneration = generation
        onChatEvent?(.cancelled)
        Task {
            guard generation == cancelledGeneration else { return }
            try? await notify(method: "session/cancel", params: ["sessionId": sessionID], session: sessionID)
            if generation == cancelledGeneration { stop() }
        }
    }

    func answerPermission(_ request: WorkPermission, allow: Bool) {
        guard turnOwner.consume(request) else { return }
        let kind = allow ? "allow_once" : "reject_once"
        guard let option = request.options.first(where: { $0["kind"] as? String == kind }),
              let optionID = option["optionId"] as? String else { return }
        if permission?.id == request.id {
            permission = pendingPermissions.isEmpty ? nil : pendingPermissions.removeFirst()
        }
        let answeredGeneration = generation
        Task {
            guard generation == answeredGeneration,
                  sessionID == request.sessionID else { return }
            try? await post(["jsonrpc": "2.0", "id": request.rpcID,
                             "result": ["outcome": ["outcome": "selected", "optionId": optionID]]],
                            session: request.sessionID)
        }
    }

    func stop() {
        turnOwner.reset()
        prefixWarmTask?.cancel()
        prefixWarmTask = nil
        isPrefixWarming = false
        prefixWarmStartedAt = nil
        pendingGooseChatSession?.cancel()
        pendingGooseChatSession = nil
        sessionHasPrompted = false
        proxyBearerFileURL = nil
        streams.forEach { $0.cancel() }
        streams = []
        let children = [process, proxyProcess, bridgeProcess].compactMap { $0 }
        for child in children {
            let group = child.processIdentifier
            if group > 0 { _ = Darwin.kill(-group, SIGTERM) }
        }
        if !children.isEmpty { usleep(800_000) }
        for child in children {
            let group = child.processIdentifier
            if group > 0 { _ = Darwin.kill(-group, SIGKILL) }
        }
        process = nil
        proxyProcess = nil
        bridgeProcess = nil
        for profile in sandboxProfileURLs { try? FileManager.default.removeItem(at: profile) }
        sandboxProfileURLs = []
        isRunning = false
        isBusy = false
        isOpeningSession = false
        permission = nil
        pendingPermissions = []
        toolIdentities = [:]
        rawToolOutputs = [:]
        connection = nil
        secret = ""
        for (_, continuation) in waiters {
            continuation.resume(throwing: WorkError.message("Chat stopped."))
        }
        waiters = [:]
    }

    static func resetStoredNativePermissions() throws {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Rapid/agent-goose-chat/workspaces")
        guard FileManager.default.fileExists(atPath: root.path) else { return }
        for support in try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
            _ = try GooseToolPresentation.containedDestination(support, root: root)
            try GoosePermissionPolicy.resetNativePolicy(in: support)
        }
    }

    private func redact(_ value: String) -> String {
        Self.redacted(value, secrets: [secret, serverIdentity?.components(separatedBy: ":").last ?? ""])
    }

    static func redacted(_ value: String, secrets: [String]) -> String {
        var safe = value
        for secret in secrets where secret.count >= 8 {
            safe = safe.replacingOccurrences(of: secret, with: "[redacted]")
            // ACP can split output into arbitrary chunks. Hide a trailing
            // prefix until the following update establishes the full value.
            for length in stride(from: min(secret.count - 1, safe.count), through: 4, by: -1) {
                if safe.hasSuffix(String(secret.prefix(length))) {
                    safe.replaceSubrange(safe.index(safe.endIndex, offsetBy: -length)..., with: "[redacted]")
                    break
                }
            }
        }
        return safe
    }

    private func openStream(session: String?) {
        let streamGeneration = generation
        let connection = self.connection
        let port = self.port
        let secret = self.secret
        streams.append(Task {
            guard let url = URL(string: "http://127.0.0.1:\(port)/acp") else { return }
            var request = URLRequest(url: url)
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            request.setValue(secret, forHTTPHeaderField: "X-Secret-Key")
            request.setValue(connection, forHTTPHeaderField: "Acp-Connection-Id")
            request.setValue(session, forHTTPHeaderField: "Acp-Session-Id")
            do {
                try await transport.stream(request) { line in
                    guard !Task.isCancelled else { return }
                    guard line.hasPrefix("data:"),
                          let data = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces).data(using: .utf8) else { return }
                    guard self.generation == streamGeneration else { return }
                    try? self.handleWire(data)
                }
                if !Task.isCancelled { helperStreamEnded(generation: streamGeneration, session: session) }
            } catch {
                if !Task.isCancelled { helperStreamEnded(generation: streamGeneration, session: session) }
            }
        })
    }

    func helperStreamEnded(generation expected: Int, session: String?) {
        guard generation == expected, session == sessionID else { return }
        let failure = "The Chat helper stopped unexpectedly."
        let activeTurn = isBusy || isPreparing || isOpeningSession || !waiters.isEmpty
        error = failure
        if activeTurn { onChatEvent?(.error(failure)) }
        stop()
    }

    private func handle(_ message: [String: Any]) {
        if let id = message["id"], message["method"] == nil,
           let continuation = waiters.removeValue(forKey: String(describing: id)) {
            if let failure = message["error"] {
                continuation.resume(throwing: WorkError.message("ACP error: \(redact(String(describing: failure)))"))
            } else {
                let result = message["result"] as? [String: Any] ?? [:]
                if let data = try? JSONSerialization.data(withJSONObject: result) {
                    continuation.resume(returning: data)
                } else {
                    continuation.resume(throwing: WorkError.message("Invalid Chat response."))
                }
            }
            return
        }
        if message["method"] as? String == "session/request_permission",
           let params = message["params"] as? [String: Any],
           let id = message["id"],
           let sid = params["sessionId"] as? String,
           let tool = params["toolCall"] as? [String: Any] {
            guard turnOwner.accepts(sid, generation: generation) else { return }
            let input = tool["rawInput"] as? [String: Any]
            let request = WorkPermission(id: String(describing: id), sessionID: sid,
                toolCallID: tool["toolCallId"] as? String ?? String(describing: id),
                command: input?["command"] as? String ?? tool["title"] as? String ?? "Unknown command",
                options: params["options"] as? [[String: Any]] ?? [], rpcID: id)
            guard turnOwner.register(request) else { return }
            if Self.gooseChat {
                let name = toolIdentities[request.toolCallID]
                    ?? GoosePermissionPolicy.toolIdentity(tool) ?? "unknown"
                let decision = GoosePermissionPolicy.decide(tool: name)
                onChatEvent?(.permission(request, name, decision.cardState))
                if decision.answer == .allow { answerPermission(request, allow: true) }
                else if decision.answer == .decline { answerPermission(request, allow: false) }
                else if permission == nil { permission = request }
                else { pendingPermissions.append(request) }
            } else if Self.fullAccess {
                answerPermission(request, allow: true)
            } else if permission == nil {
                permission = request
            } else {
                pendingPermissions.append(request)
            }
            return
        }
        guard message["method"] as? String == "session/update",
              let params = message["params"] as? [String: Any],
              let updateSession = params["sessionId"] as? String,
              turnOwner.accepts(updateSession, generation: generation),
              let update = params["update"] as? [String: Any],
              let kind = update["sessionUpdate"] as? String else { return }
        switch kind {
        case "agent_message_chunk":
            if let content = update["content"] as? [String: Any],
               content["type"] as? String == "text", let chunk = content["text"] as? String {
                let safe = redact(chunk)
                text += safe
                onChatEvent?(.text(safe))
            }
        case "agent_thought_chunk":
            if let content = update["content"] as? [String: Any],
               let chunk = content["text"] as? String { onChatEvent?(.thinking(redact(chunk))) }
        case "tool_call":
            let id = update["toolCallId"] as? String ?? UUID().uuidString
            if let identity = GoosePermissionPolicy.toolIdentity(update) { toolIdentities[id] = identity }
            let input = update["rawInput"] as? [String: Any]
            if !steps.contains(where: { $0.id == id }) {
                steps.append(WorkStep(id: id, command: redact(input?["command"] as? String ?? update["title"] as? String ?? "Tool"), status: "running"))
            }
            onChatEvent?(.tool(id: id, name: redact(update["title"] as? String ?? "Tool"),
                               input: redact(input.map { Self.printJSON($0) } ?? ""), status: steps.first(where: { $0.id == id })?.status ?? "running", output: ""))
        case "tool_call_update":
            guard let id = update["toolCallId"] as? String else { return }
            if !steps.contains(where: { $0.id == id }) {
                steps.append(WorkStep(id: id, command: "Tool", status: "running"))
            }
            guard let index = steps.firstIndex(where: { $0.id == id }) else { return }
            if let next = update["status"] as? String,
               !Self.isTerminalToolStatus(steps[index].status) { steps[index].status = next }
            if let raw = update["rawOutput"] as? [String: Any] {
                rawToolOutputs[id] = (raw["stdout"] as? String ?? "") + (raw["stderr"] as? String ?? "")
            }
            if let content = update["content"] as? [[String: Any]] {
                let text = content.compactMap { item -> String? in
                    if let nested = item["content"] as? [String: Any],
                       nested["type"] as? String == "text" { return nested["text"] as? String }
                    if item["type"] as? String == "text" { return item["text"] as? String }
                    return nil
                }.joined(separator: "\n")
                if !text.isEmpty { rawToolOutputs[id] = text }
            }
            if let meta = update["_meta"] as? [String: Any],
               let notification = meta["toolNotification"] as? [String: Any],
               let params = notification["params"] as? [String: Any],
               let chunks = params["chunks"] as? [[String: Any]] {
                rawToolOutputs[id, default: ""] += chunks.compactMap { $0["output"] as? String }.joined()
            }
            steps[index].output = redact(rawToolOutputs[id] ?? "")
            let changedInput = (update["rawInput"] as? [String: Any]).map { Self.printJSON($0) } ?? ""
            onChatEvent?(.tool(id: id, name: redact(update["title"] as? String ?? "Tool"),
                               input: redact(changedInput), status: steps[index].status, output: steps[index].output))
        default: break
        }
    }

    static func isTerminalToolStatus(_ status: String) -> Bool {
        ["completed", "failed", "cancelled", "canceled", "error"].contains(status.lowercased())
    }

    private static func printJSON(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: data, encoding: .utf8) else { return "" }
        return text
    }

    func call(method: String, params: [String: Any], session: String? = nil,
                      timeout: TimeInterval = 90) async throws -> [String: Any] {
        let id = nextID
        let requestGeneration = generation
        nextID += 1
        let data: Data = try await withCheckedThrowingContinuation { continuation in
            waiters[String(id)] = continuation
            Task {
                guard generation == requestGeneration else {
                    waiters.removeValue(forKey: String(id))?.resume(throwing: CancellationError())
                    return
                }
                do {
                    try await post(["jsonrpc": "2.0", "id": id, "method": method, "params": params], session: session)
                } catch {
                    waiters.removeValue(forKey: String(id))?.resume(throwing: error)
                }
            }
            Task {
                try? await Task.sleep(for: .seconds(timeout))
                waiters.removeValue(forKey: String(id))?.resume(
                    throwing: WorkError.message("Chat request timed out: \(method)"))
            }
        }
        guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkError.message("Invalid Chat response.")
        }
        return result
    }

    private func notify(method: String, params: [String: Any], session: String?) async throws {
        try await post(["jsonrpc": "2.0", "method": method, "params": params], session: session)
    }

    private func sendImmediate(method: String, params: [String: Any]) async throws -> ([String: Any], HTTPURLResponse) {
        let id = nextID
        nextID += 1
        let (data, response) = try await postData(["jsonrpc": "2.0", "id": id, "method": method, "params": params], session: nil)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkError.message("Invalid Chat initialization response.")
        }
        return (object, response)
    }

    private func post(_ body: [String: Any], session: String?) async throws {
        _ = try await postData(body, session: session)
    }

    private func postData(_ body: [String: Any], session: String?) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/acp")!)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(secret, forHTTPHeaderField: "X-Secret-Key")
        request.setValue(connection, forHTTPHeaderField: "Acp-Connection-Id")
        request.setValue(session, forHTTPHeaderField: "Acp-Session-Id")
        let (data, response) = try await transport.send(request)
        guard let response = response as? HTTPURLResponse, response.statusCode < 400 else {
            throw WorkError.message("Chat transport rejected a request.")
        }
        return (data, response)
    }

    static func contextLimit(from data: Data, model: String) -> Int? {
        guard let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = payload["data"] as? [[String: Any]],
              let entry = models.first(where: { $0["id"] as? String == model }),
              let limit = entry["context_window"] as? Int,
              limit > 0 else { return nil }
        return limit
    }

    static func needsContinuation(result: [String: Any], newText: String) -> Bool {
        result["stopReason"] as? String == "max_tokens" ||
            newText.contains("Response stopped because the model reached its output-token limit.")
    }

    static func needsFollowUp(result: [String: Any], newText: String, newSteps: [WorkStep]) -> Bool {
        needsContinuation(result: result, newText: newText) ||
            newSteps.contains { $0.status.lowercased() == "failed" }
    }

    static func shouldContinue(gooseChat: Bool, result: [String: Any], newText: String,
                               newSteps: [WorkStep]) -> Bool {
        gooseChat ? needsContinuation(result: result, newText: newText)
            : needsFollowUp(result: result, newText: newText, newSteps: newSteps)
    }

    private static func contextLimit(rapidPort: Int, model: String, bearer: String) async -> Int {
        let fallback = 32768  // Matches `rapid-mlx agents goose --setup` when metadata is unavailable.
        guard let url = URL(string: "http://127.0.0.1:\(rapidPort)/v1/models") else { return fallback }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 5
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { return fallback }
            return contextLimit(from: data, model: model) ?? fallback
        } catch {
            return fallback
        }
    }

    private static func freePort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw WorkError.message("Could not allocate a local port.") }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: in_addr_t(0x0100007f))
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard result == 0 else { throw WorkError.message("Could not bind a local port.") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let found = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        guard found == 0 else { throw WorkError.message("Could not read a local port.") }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private static func launchHelper(profile: URL?, python: URL, executable: URL,
                                        arguments: [String], directory: URL,
                                        environment: [String: String]) throws -> Process {
        let child = Process()
        child.executableURL = python
        child.arguments = ["-I", "-S", Bundle.main.resourceURL!.appendingPathComponent("Helpers/work-helper-watchdog.py").path,
                           String(getpid())] + (profile.map { ["/usr/bin/sandbox-exec", "-f", $0.path] } ?? [])
                           + [executable.path] + arguments
        child.currentDirectoryURL = directory
        child.environment = environment
        child.standardOutput = FileHandle.nullDevice
        child.standardError = FileHandle.nullDevice
        try child.run()
        for _ in 0..<100 {
            if getpgid(child.processIdentifier) == child.processIdentifier { return child }
            if !child.isRunning { break }
            usleep(10_000)
        }
        if child.isRunning { child.terminate() }
        throw WorkError.message("Could not isolate the Chat helper process group.")
    }

    private static func waitForHealth(process: Process, port: Int) async -> Bool {
        for _ in 0..<60 {
            if !process.isRunning { return false }
            if let url = URL(string: "http://127.0.0.1:\(port)/health"),
               let (_, response) = try? await URLSession.shared.data(from: url),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                return true
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        return false
    }

    private static func profile(folder: URL, support: URL, temporary: URL,
                                resources: URL, outboundPorts: [Int]) -> String {
        func quoted(_ value: String) -> String {
            "\"" + value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
        }
        // SQLite resolves every ancestor before opening sessions.db. Permit
        // metadata only on those ancestors; contents remain unreadable.
        var ancestors = Set<String>()
        for url in [folder, support, resources] {
            var parent = url.deletingLastPathComponent()
            while parent.path != "/" {
                ancestors.insert(parent.path)
                parent.deleteLastPathComponent()
            }
        }
        let ancestorRules = ancestors.sorted().map { "(literal \(quoted($0)))" }.joined(separator: " ")
        let networkRules = outboundPorts.map {
            "(allow network-outbound (remote ip \"localhost:\($0)\"))"
        }.joined(separator: "\n")
        return """
        (version 1)
        (deny default)
        (allow process*)
        (allow mach-lookup)
        (allow sysctl-read)
        (allow file-read* (literal "/") (literal "/private")
            (subpath "/System") (subpath "/usr") (subpath "/bin")
            (subpath "/sbin") (subpath "/Library")
            (subpath "/private/etc") (subpath "/dev"))
        (allow file-read-metadata \(ancestorRules))
        (allow file-read* (subpath \(quoted(resources.path)))
            (subpath \(quoted(support.path))) (subpath \(quoted(folder.path))))
        (allow file-write* (literal "/dev/null") (subpath \(quoted(folder.path)))
            (subpath \(quoted(support.path))) (subpath \(quoted(temporary.path))))
        (allow network-inbound (local ip "localhost:*"))
        \(networkRules)
        """
    }
}

enum WorkError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        if case .message(let value) = self { return value }
        return nil
    }
}
