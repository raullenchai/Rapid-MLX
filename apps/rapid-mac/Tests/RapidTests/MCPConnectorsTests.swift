import Foundation
import Testing
@testable import Rapid

/// Issue #1716 — MCP went from engine-complete-and-app-invisible to a real
/// desktop surface. These pin the parts that are load-bearing for safety:
///
///   * the config the app writes is the shape the engine reads, and a name
///     that would produce an uncallable tool never reaches it;
///   * a disabled tool is refused at dispatch, not merely omitted from the
///     request body;
///   * a tool that was never approved does not execute;
///   * built-in tools keep their own switch and cannot be shadowed by a
///     connector claiming their name.
@MainActor
@Suite("MCP connectors (issue #1716)")
final class MCPConnectorsTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    nonisolated(unsafe) private var tempDirs: [URL] = []

    deinit {
        TestDefaultsScope.cleanup(suiteNames: createdSuiteNames)
        for dir in tempDirs { try? FileManager.default.removeItem(at: dir) }
    }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-mcp-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    /// Fresh defaults with the connectors master switch already on. The
    /// registry gates advertise + dispatch on that switch, so a test about the
    /// per-tool or approval gates has to start from "connectors on" or it would
    /// only ever exercise the master gate.
    private func enabledDefaults() -> UserDefaults {
        let d = freshDefaults()
        d.set(true, forKey: MCPConfigStore.enabledKey)
        return d
    }

    private func tempConfigURL() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-mcp-\(UUID().uuidString)", isDirectory: true)
        tempDirs.append(dir)
        return dir.appendingPathComponent("mcp.json")
    }

    // MARK: - Server name validation

    @Test("A server name that would produce an uncallable tool name is rejected")
    func nameValidation() {
        // The engine namespaces tools as `server__tool`, and that composite
        // travels as an OpenAI function name — `[a-zA-Z0-9_-]`, max 64. A
        // space here means the model is handed a name it can never emit, and
        // the symptom is "that connector's tools are silently ignored".
        #expect(MCPServerConfig.isValidName("filesystem"))
        #expect(MCPServerConfig.isValidName("my-server_2"))
        #expect(!MCPServerConfig.isValidName("my server"))
        #expect(!MCPServerConfig.isValidName("emoji🙂"))
        #expect(!MCPServerConfig.isValidName("dots.are.out"))
        #expect(!MCPServerConfig.isValidName(""))
        #expect(!MCPServerConfig.isValidName(String(repeating: "a", count: 33)))
        // `__` is the namespace separator; a name containing it splits wrong
        // and the server's tools never dispatch. A single `_` is fine.
        #expect(!MCPServerConfig.isValidName("my__server"))
        #expect(MCPServerConfig.isValidName("my_server"))
    }

    @Test("A stdio entry needs a command and an SSE entry needs a valid URL")
    func transportValidation() {
        var stdio = MCPServerConfig(name: "ok", transport: .stdio, command: nil)
        #expect(stdio.validationError != nil)
        stdio.command = "uvx"
        #expect(stdio.validationError == nil)

        var sse = MCPServerConfig(name: "ok", transport: .sse, url: nil)
        #expect(sse.validationError != nil)
        sse.url = "ftp://example.com"
        #expect(sse.validationError != nil, "only http/https should pass")
        sse.url = "https://example.com/mcp"
        #expect(sse.validationError == nil)
    }

    // MARK: - Config file round-trip

    @Test("Config round-trips through the ecosystem-standard mcpServers shape")
    func configRoundTrip() throws {
        let store = MCPConfigStore(defaults: freshDefaults(), fileURL: tempConfigURL())
        try store.upsert(MCPServerConfig(
            name: "time",
            transport: .stdio,
            command: "uvx",
            args: ["mcp-server-time", "--local-timezone=UTC"],
            env: ["TZ": "UTC"]
        ))
        try store.upsert(MCPServerConfig(
            name: "remote",
            transport: .sse,
            url: "https://example.com/mcp"
        ))

        let data = try MCPConfigStore.encode(store.servers)
        // The engine reads `mcpServers` (and so do Claude Desktop / VS Code),
        // so a config authored here must be liftable into any of them.
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let map = root?["mcpServers"] as? [String: Any]
        #expect(map?.count == 2)
        #expect(map?["time"] != nil)

        let decoded = try MCPConfigStore.decode(data)
        #expect(decoded.map(\.name) == ["remote", "time"], "sorted for a stable list")
        let time = try #require(decoded.first { $0.name == "time" })
        #expect(time.command == "uvx")
        #expect(time.args == ["mcp-server-time", "--local-timezone=UTC"])
        #expect(time.env == ["TZ": "UTC"])
        let remote = try #require(decoded.first { $0.name == "remote" })
        #expect(remote.transport == .sse)
        #expect(remote.url == "https://example.com/mcp")
        // A URL entry must not carry a stale command from an earlier edit.
        #expect(remote.command == nil)
    }

    @Test("An imported config with an uncallable server name surfaces a load error")
    func importedInvalidNameSurfacesOnLoad() throws {
        // Validation is enforced on the write path, but an imported config —
        // the whole reason for the ecosystem-standard shape — never went
        // through it. `my.server` is a legal JSON key yet an illegal tool-name
        // namespace, so the engine would forward it and the model would
        // silently never call it. The load has to say so.
        let url = tempConfigURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data(#"{"mcpServers":{"my.server":{"command":"npx"}}}"#.utf8).write(to: url)

        let store = MCPConfigStore(defaults: freshDefaults(), fileURL: url)
        #expect(store.loadError != nil)
        #expect(store.loadError?.contains("my.server") == true)
    }

    @Test("A hand-written entry with no transport decodes as stdio rather than failing")
    func decodeToleratesMissingTransport() throws {
        // Refusing to decode would drop the entry from the UI entirely, which
        // is the exact "my connectors vanished" failure this feature fixes.
        let json = #"{"mcpServers":{"fs":{"command":"npx","args":["-y","pkg"]}}}"#
        let decoded = try MCPConfigStore.decode(Data(json.utf8))
        #expect(decoded.count == 1)
        #expect(decoded[0].name == "fs")
        #expect(decoded[0].transport == .stdio)
        #expect(decoded[0].enabled, "absent `enabled` means enabled")
    }

    @Test("The engine's historical `servers` key still loads")
    func decodeAcceptsLegacyKey() throws {
        let json = #"{"servers":{"fs":{"transport":"stdio","command":"npx"}}}"#
        let decoded = try MCPConfigStore.decode(Data(json.utf8))
        #expect(decoded.map(\.name) == ["fs"])
    }

    @Test("A duplicate name is refused rather than silently overwriting")
    func duplicateNameRefused() throws {
        let store = MCPConfigStore(defaults: freshDefaults(), fileURL: tempConfigURL())
        try store.upsert(MCPServerConfig(name: "fs", command: "npx"))
        #expect(throws: MCPConfigStore.SaveError.self) {
            try store.upsert(MCPServerConfig(name: "fs", command: "uvx"))
        }
        // Editing that same entry must still work — the duplicate check is
        // against OTHER entries, not the one being replaced.
        try store.upsert(MCPServerConfig(name: "fs", command: "uvx"), replacing: "fs")
        #expect(store.servers.first?.command == "uvx")
    }

    @Test("The config file is written 0600")
    func configFilePermissions() throws {
        let url = tempConfigURL()
        let store = MCPConfigStore(defaults: freshDefaults(), fileURL: url)
        try store.upsert(MCPServerConfig(name: "fs", command: "npx"))
        // The file names local commands that the engine will execute. Another
        // account on a shared Mac must not be able to read — let alone edit —
        // it. `.atomic` writes via rename and does NOT carry permissions, so
        // this pins the explicit chmod after the rename.
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let perms = try #require(attrs[.posixPermissions] as? NSNumber)
        #expect(perms.intValue == 0o600)
    }

    // MARK: - Launch flag

    @Test("--mcp-config is passed only when connectors are on AND a server is enabled")
    func launchConfigPathGate() throws {
        let store = MCPConfigStore(defaults: freshDefaults(), fileURL: tempConfigURL())

        // Off by default: connectors run arbitrary local programs, so the
        // subsystem stays entirely absent until asked for.
        #expect(!store.isEnabled)
        #expect(store.launchConfigPath == nil)

        store.isEnabled = true
        #expect(store.launchConfigPath == nil, "enabled but no servers is still nothing to start")

        try store.upsert(MCPServerConfig(name: "fs", command: "npx"))
        #expect(store.launchConfigPath != nil)

        try store.setServerEnabled("fs", false)
        #expect(store.launchConfigPath == nil, "every server switched off is nothing to start")

        try store.setServerEnabled("fs", true)
        store.isEnabled = false
        #expect(store.launchConfigPath == nil, "master switch wins")
    }

    @Test("serveArguments appends --mcp-config last, after the cors-origins nargs list")
    func serveArgumentsCarriesMCPConfig() throws {
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000,
            mcpConfigPath: "/tmp/mcp.json"
        )
        // ``--cors-origins`` is argparse ``nargs="+"``; only a following
        // ``--``-prefixed flag terminates its collection. If ``--mcp-config``
        // ever stopped starting with ``--`` (or a bare value were appended
        // after it) the path would be swallowed as a third CORS origin.
        let idx = try #require(argv.firstIndex(of: "--mcp-config"))
        #expect(argv[idx + 1] == "/tmp/mcp.json")
        #expect(idx == argv.count - 2, "the path is the final argv element")
        let corsIdx = try #require(argv.firstIndex(of: "--cors-origins"))
        #expect(corsIdx < idx)
    }

    @Test("serveArguments is unchanged when no MCP config path is supplied")
    func serveArgumentsOmitsMCPConfigByDefault() {
        // A user who never turns connectors on must get the exact pre-#1716
        // argv — no new flag, no behaviour change.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        #expect(!argv.contains("--mcp-config"))
        #expect(argv == [
            "serve",
            "qwen3.5-4b-4bit",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--cors-origins", "http://127.0.0.1", "http://localhost",
        ])
    }

    // MARK: - Approval gate

    private func makeApproval() -> MCPToolApprovalStore {
        MCPToolApprovalStore(defaults: freshDefaults())
    }

    /// Wait until the approval sheet is up, deterministically. A fixed sleep
    /// flakes on a loaded worker that hasn't scheduled the requesting task yet;
    /// this returns the instant `pendingRequest` populates, and gives up after
    /// a generous bound so a genuine hang fails loudly instead of spinning.
    private func waitForPending(_ store: MCPToolApprovalStore) async {
        for _ in 0..<400 {
            if store.pendingRequest != nil { return }
            try? await Task.sleep(nanoseconds: 5_000_000)  // 5ms; ~2s ceiling
        }
    }

    @Test("Always-allow is remembered for that tool and no other")
    func grantIsPerTool() async {
        let store = makeApproval()
        #expect(!store.isGranted("fs__read_file"))

        async let decision = store.requestApproval(
            toolName: "fs__read_file",
            serverName: "fs",
            argumentsJSON: #"{"path":"/etc/hosts"}"#
        )
        await waitForPending(store)
        store.answer(.alwaysAllowTool)
        let answered = await decision
        #expect(answered == .alwaysAllowTool)

        #expect(store.isGranted("fs__read_file"))
        // The point of per-tool consent: approving one tool must not hand out
        // the rest of that server's surface.
        #expect(!store.isGranted("fs__write_file"))
        #expect(!store.isGranted("shell__run"))

        // A granted tool no longer prompts.
        let second = await store.requestApproval(
            toolName: "fs__read_file", serverName: "fs", argumentsJSON: "{}"
        )
        #expect(second == .allowOnce)
    }

    @Test("Allow-once does not persist")
    func allowOnceIsNotRemembered() async {
        let store = makeApproval()
        async let decision = store.requestApproval(
            toolName: "fs__read_file", serverName: "fs", argumentsJSON: "{}"
        )
        await waitForPending(store)
        store.answer(.allowOnce)
        _ = await decision
        #expect(!store.isGranted("fs__read_file"))
    }

    @Test("Reset revokes every remembered grant but leaves the auto-approve mode alone")
    func resetGrants() async {
        let defaults = freshDefaults()
        let store = MCPToolApprovalStore(defaults: defaults)
        store.mode = .autoApproveAll
        defaults.set(true, forKey: MCPToolApprovalStore.grantKey("fs__read_file"))

        let reloaded = MCPToolApprovalStore(defaults: defaults)
        #expect(reloaded.grantedTools.contains("fs__read_file"), "grants survive a relaunch")
        reloaded.resetGrants()
        #expect(reloaded.grantedTools.isEmpty)
        #expect(defaults.object(forKey: MCPToolApprovalStore.grantKey("fs__read_file")) == nil)
        // Resetting individual grants is not a request to change the global
        // posture — that's a separate switch.
        #expect(reloaded.mode == .autoApproveAll)
    }

    @Test("A cancelled turn reports unavailable, not a user decline")
    func cancellationIsNotADecline() async {
        // The distinction is user-visible: a decline is an outcome the user
        // chose; a cancellation is not something to attribute to them.
        let store = makeApproval()
        let task = Task {
            await store.requestApproval(
                toolName: "fs__read_file", serverName: "fs", argumentsJSON: "{}"
            )
        }
        await waitForPending(store)
        task.cancel()
        let outcome = await task.value
        #expect(outcome == .unavailable)
        #expect(store.pendingRequest == nil, "the sheet is torn down with the turn")
    }

    @Test("A second concurrent prompt is refused rather than left hanging")
    func reentrancyIsRefused() async {
        let store = makeApproval()
        async let first = store.requestApproval(
            toolName: "fs__a", serverName: "fs", argumentsJSON: "{}"
        )
        await waitForPending(store)
        let second = await store.requestApproval(
            toolName: "fs__b", serverName: "fs", argumentsJSON: "{}"
        )
        #expect(second == .unavailable)
        store.answer(.deny)
        let firstOutcome = await first
        #expect(firstOutcome == .deny)
    }

    @Test("Repointing a connector at different code revokes its remembered grants")
    func reconfiguringAConnectorRevokesGrants() throws {
        let defaults = freshDefaults()
        // A remembered "always allow" for a tool on the "fs" connector.
        defaults.set(true, forKey: MCPToolApprovalStore.grantKey("fs__read_file"))
        let approval = MCPToolApprovalStore(defaults: defaults)
        #expect(approval.isGranted("fs__read_file"))

        let url = tempConfigURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let store = MCPConfigStore(defaults: defaults, fileURL: url)
        store.onServerReconfigured = { approval.revokeGrants(forServer: $0) }

        try store.upsert(MCPServerConfig(name: "fs", command: "npx"))
        #expect(approval.isGranted("fs__read_file"), "adding a connector is not a reconfiguration")

        // Point the SAME name at a different command. The grant would otherwise
        // silently authorize the new program — it must be dropped instead.
        try store.upsert(MCPServerConfig(name: "fs", command: "evil"), replacing: "fs")
        #expect(!approval.isGranted("fs__read_file"))
    }

    @Test("A non-code edit keeps the grant, but removing the connector drops it")
    func benignEditKeepsGrantRemovalDropsIt() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: MCPToolApprovalStore.grantKey("fs__read_file"))
        let approval = MCPToolApprovalStore(defaults: defaults)

        let url = tempConfigURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let store = MCPConfigStore(defaults: defaults, fileURL: url)
        store.onServerReconfigured = { approval.revokeGrants(forServer: $0) }
        try store.upsert(MCPServerConfig(name: "fs", command: "npx"))

        // Bumping the timeout doesn't change what code runs — the grant stays.
        try store.upsert(MCPServerConfig(name: "fs", command: "npx", timeout: 60), replacing: "fs")
        #expect(approval.isGranted("fs__read_file"))

        // Removing the connector drops the grant so re-adding starts unapproved.
        try store.remove(named: "fs")
        #expect(!approval.isGranted("fs__read_file"))
    }

    @Test("A hand-edited command is caught by the launch reconcile and revokes the grant")
    func handEditRevokesGrantOnReconcile() {
        let defaults = freshDefaults()
        defaults.set(true, forKey: MCPToolApprovalStore.grantKey("fs__read_file"))
        let approval = MCPToolApprovalStore(defaults: defaults)

        // First launch establishes the baseline fingerprint for "fs" = npx.
        let original = MCPServerConfig(name: "fs", command: "npx")
        approval.reconcileGrants(against: ["fs": original.executionFingerprint])
        #expect(approval.isGranted("fs__read_file"), "unchanged code keeps the grant")

        // A relaunch with the same fingerprint keeps it too.
        approval.reconcileGrants(against: ["fs": original.executionFingerprint])
        #expect(approval.isGranted("fs__read_file"))

        // Next launch after the config file was hand-edited to a new command.
        let edited = MCPServerConfig(name: "fs", command: "evil")
        approval.reconcileGrants(against: ["fs": edited.executionFingerprint])
        #expect(!approval.isGranted("fs__read_file"))
    }

    @Test("Auto-approve mode skips the prompt entirely")
    func autoApproveSkipsPrompt() async {
        let store = makeApproval()
        store.mode = .autoApproveAll
        let decision = await store.requestApproval(
            toolName: "anything__at_all", serverName: "x", argumentsJSON: "{}"
        )
        #expect(decision == .allowOnce)
        #expect(store.pendingRequest == nil)
    }

    @Test("The tool half of a namespaced name is split on the first separator")
    func shortToolName() {
        #expect(MCPToolApprovalStore.shortToolName("fs__read_file") == "read_file")
        // The server half never contains `__`, so a tool whose own name does
        // keeps it.
        #expect(MCPToolApprovalStore.shortToolName("fs__odd__name") == "odd__name")
        #expect(MCPToolApprovalStore.shortToolName("no_separator") == "no_separator")
    }

    // MARK: - Registry dispatch

    private func makeCatalog() -> MCPCatalog {
        // No endpoint: every network path short-circuits, which is what we
        // want for the gate tests — a call that reaches the transport has
        // already passed the gates under test.
        MCPCatalog(endpoint: { nil })
    }

    @Test("A disabled connector tool is refused at dispatch, not merely unadvertised")
    func disabledToolIsRefusedAtDispatch() async {
        // Omitting a tool from the request body does NOT stop a malformed
        // model emitting a call for it. The dispatch-side refusal is the
        // load-bearing gate.
        let registry = MCPToolRegistry(
            catalog: makeCatalog(),
            approval: makeApproval(),
            defaults: enabledDefaults()
        )
        registry.setToolEnabled("fs__read_file", false)
        #expect(!registry.isToolEnabled("fs__read_file"))

        let result = await registry.run(
            ToolCall(id: "c1", name: "fs__read_file", arguments: "{}")
        )
        #expect(result.isError)
        #expect(result.failureKind == .userDeclined)
        #expect(result.content.contains("turned off"))
    }

    @Test("A tool switched off stays off across a relaunch")
    func disabledToolPersists() {
        let defaults = freshDefaults()
        let catalog = makeCatalog()
        let approval = makeApproval()
        MCPToolRegistry(catalog: catalog, approval: approval, defaults: defaults)
            .setToolEnabled("fs__read_file", false)
        let reloaded = MCPToolRegistry(catalog: catalog, approval: approval, defaults: defaults)
        #expect(reloaded.disabledTools.contains("fs__read_file"))
    }

    @Test("A denied tool call does not execute")
    func deniedToolDoesNotRun() async {
        let approval = makeApproval()
        let registry = MCPToolRegistry(
            catalog: makeCatalog(),
            approval: approval,
            defaults: enabledDefaults()
        )
        async let result = registry.run(
            ToolCall(id: "c1", name: "fs__read_file", arguments: #"{"path":"/etc/passwd"}"#)
        )
        await waitForPending(approval)
        #expect(approval.pendingRequest != nil, "the user is asked before anything runs")
        approval.answer(.deny)

        let r = await result
        #expect(r.isError)
        #expect(r.failureKind == .userDeclined)
        #expect(r.content.contains("declined"))
    }

    @Test("The approval prompt names the server and shows the arguments display-safe")
    func promptCarriesContext() async {
        let approval = makeApproval()
        // A model can hide the real target behind bidi / zero-width scalars.
        // The user has to see what the engine will actually receive.
        async let decision = approval.requestApproval(
            toolName: "fs__read_file",
            serverName: "fs",
            argumentsJSON: "{\"path\":\"/etc/\u{202E}gnp.txt\"}"
        )
        await waitForPending(approval)
        let pending = approval.pendingRequest
        #expect(pending?.serverName == "fs")
        #expect(pending?.shortName == "read_file")
        #expect(pending?.argumentsPreview.contains("\\u{202E}") == true)
        #expect(pending?.argumentsPreview.contains("\u{202E}") == false)
        approval.answer(.deny)
        _ = await decision
    }

    @Test("The prompt shows the whole argument payload, not a capped preview")
    func promptShowsFullArguments() async {
        let approval = makeApproval()
        // The consent gate executes the complete JSON, so it has to SHOW the
        // complete JSON. A model can push the dangerous part past any cap; the
        // sheet scrolls, so there is no reason to hide it.
        let tail = "rm -rf /important"
        let bigArgs = "{\"cmd\":\"" + String(repeating: "a", count: 600) + tail + "\"}"
        async let decision = approval.requestApproval(
            toolName: "sh__run", serverName: "sh", argumentsJSON: bigArgs
        )
        await waitForPending(approval)
        let preview = approval.pendingRequest?.argumentsPreview ?? ""
        #expect(preview.contains(tail), "content past the old 400-char cap must be visible")
        approval.answer(.deny)
        _ = await decision
    }

    @Test("The master switch gates advertise + dispatch even with a populated catalog")
    func masterSwitchGatesConnectorTools() async {
        // The running child keeps its connectors loaded until it restarts, so a
        // later `/healthz` ready transition can repopulate the catalog after
        // the user switched connectors off. The load-bearing gate is on the
        // registry the chat loop reads — not the catalog it can refill from.
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"fs","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"fs__read_file","description":"d","server":"fs","parameters":{}}],"count":1}
        """
        await catalog.refresh()
        #expect(!catalog.tools.isEmpty)

        let defaults = enabledDefaults()
        let registry = MCPToolRegistry(
            catalog: catalog, approval: makeApproval(), defaults: defaults
        )
        // Connectors on: the populated tool is advertised.
        #expect(registry.definitions.map { $0.function.name } == ["fs__read_file"])

        // Flip the master switch off. The catalog is still populated (no
        // restart yet), but the tool must vanish from what the model sees AND
        // be refused if a call slips through anyway.
        defaults.set(false, forKey: MCPConfigStore.enabledKey)
        #expect(registry.definitions.isEmpty)

        let result = await registry.run(
            ToolCall(id: "c1", name: "fs__read_file", arguments: "{}")
        )
        #expect(result.isError)
        #expect(result.failureKind == .userDeclined)
        #expect(result.content.contains("Connectors are turned off"))
    }

    // MARK: - Composite registry

    private func makeComposite() -> CompositeToolRegistry {
        CompositeToolRegistry(
            builtin: BuiltinToolRegistry(
                browseApproval: BrowseApprovalStore(defaults: freshDefaults()),
                webSearch: WebSearchConfig(defaults: freshDefaults(), keychain: NullKeychain())
            ),
            mcp: MCPToolRegistry(
                catalog: makeCatalog(),
                approval: makeApproval(),
                defaults: freshDefaults()
            )
        )
    }

    @Test("With no connectors up the composite is exactly the built-in surface")
    func compositeIsBuiltinWhenNoConnectors() {
        // A user who never turns connectors on must see no change at all.
        let names = makeComposite().definitions.map { $0.function.name }
        #expect(names == ["web_search", "browse", "weather"])
    }

    @Test("A built-in tool still dispatches to the built-in registry")
    func compositeRoutesBuiltins() async {
        let result = await makeComposite().run(
            ToolCall(id: "c1", name: "weather", arguments: "{}")
        )
        // Reached the real tool (which then complains about its arguments)
        // rather than falling through to the unknown-tool branch.
        #expect(!result.content.contains("unknown tool"))
    }

    @Test("An unknown name is refused with the names that ARE available")
    func compositeRefusesUnknown() async {
        let result = await makeComposite().run(
            ToolCall(id: "c1", name: "definitely__not_a_tool", arguments: "{}")
        )
        #expect(result.isError)
        #expect(result.content.contains("unknown tool"))
        #expect(result.content.contains("web_search"))
    }

    @Test("Golden path: a live connector tool joins the built-in surface, dispatches, and the master switch collapses it")
    func compositeConnectorGoldenPath() async {
        // The end-to-end surface the chat loop actually reads: built-ins plus a
        // connected MCP tool, one registry, gated by the master switch. Covers
        // in one flow what the unit tests check in pieces — advertise, route,
        // and the connectors-off collapse.
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"time","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"time__now","description":"d","server":"time","parameters":{}}],"count":1}
        """
        await catalog.refresh()

        let defaults = enabledDefaults()
        let composite = CompositeToolRegistry(
            builtin: BuiltinToolRegistry(
                browseApproval: BrowseApprovalStore(defaults: freshDefaults()),
                webSearch: WebSearchConfig(defaults: freshDefaults(), keychain: NullKeychain())
            ),
            mcp: MCPToolRegistry(catalog: catalog, approval: makeApproval(), defaults: defaults)
        )

        // Connectors on: the connector tool sits alongside the built-in three.
        #expect(composite.definitions.map { $0.function.name }
            == ["web_search", "browse", "weather", "time__now"])
        // And a call for it routes to the MCP side (reaching the approval gate),
        // not the unknown-tool branch.
        async let dispatched = composite.run(
            ToolCall(id: "c1", name: "time__now", arguments: "{}")
        )
        await waitForPending(composite.mcp.approval)
        composite.mcp.approval.answer(.deny)
        let routed = await dispatched
        #expect(!routed.content.contains("unknown tool"))

        // Master switch off collapses the surface back to the built-ins.
        defaults.set(false, forKey: MCPConfigStore.enabledKey)
        #expect(composite.definitions.map { $0.function.name }
            == ["web_search", "browse", "weather"])
    }

    // MARK: - Catalog / hot reload

    /// Hands out a port nobody else in this run is using.
    ///
    /// swift-testing runs tests in PARALLEL. The stub's response table is
    /// necessarily static (URLProtocol instances are created by URLSession,
    /// so there is nowhere else to put it), and an earlier version had every
    /// test write to the same keys and call a global `reset()`. Tests then
    /// wiped each other's fixtures mid-run: one reload test failed outright,
    /// and — worse — the other passed for the wrong reason, because it
    /// asserted `!applied` and a stubbed-out request fails too. Keying every
    /// fixture by port keeps the tests independent without serialising them.
    private static var nextStubPort = 9000

    /// Fixture key: port scopes it to one test, path selects the route.
    private func key(_ port: Int, _ path: String) -> String { "\(port)\(path)" }

    private func stubbedCatalog() -> (catalog: MCPCatalog, port: Int) {
        let port = Self.nextStubPort
        Self.nextStubPort += 1
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [MCPStubProtocol.self]
        let catalog = MCPCatalog(
            session: URLSession(configuration: cfg),
            endpoint: { (host: "127.0.0.1", port: port, bearer: "secret") }
        )
        return (catalog, port)
    }

    @Test("Refresh publishes the engine's server rows, tools, and tool→server map")
    func refreshPublishesEngineState() async {
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"fs","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"fs__read_file","description":"Read a file","server":"fs",
                   "parameters":{"type":"object"}}],"count":1}
        """
        await catalog.refresh()

        #expect(catalog.servers.map(\.name) == ["fs"])
        #expect(catalog.servers.first?.isConnected == true)
        #expect(catalog.tools.map { $0.function.name } == ["fs__read_file"])
        // The approval prompt can't ask a useful question without this map —
        // "run read_file?" is unanswerable without knowing whose.
        #expect(catalog.serverForTool["fs__read_file"] == "fs")
        #expect(catalog.isConfigured)
    }

    @Test("A reload the engine answers but has no config for reports failure")
    func reloadWithoutEngineConfigIsNotSuccess() async {
        // The trap this pins: the child was spawned BEFORE connectors were
        // switched on, so it never received --mcp-config. The reload route
        // still answers 200 with an error string. Treating "the request
        // succeeded" as "the change applied" would clear the restart banner
        // while the user's edit sat inert — the exact silently-ignored-edit
        // failure issue #1716 calls out.
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/reload")] = """
        {"servers":[],"error":"No MCP config path known — start the server with --mcp-config",
         "configured":false}
        """
        let applied = await catalog.reload()
        #expect(!applied)
        #expect(catalog.subsystemError != nil)
        // The panel's restart banner is DERIVED from this flag rather than
        // stored in view state — an earlier version kept it in `@State`, which
        // meant switching Settings tabs silently reset it and left the user
        // with only the engine's "start the server with --mcp-config"
        // complaint. It has to stay false here for the banner to survive.
        #expect(!catalog.isConfigured)
    }

    @Test("A successful reload against a configured engine clears the restart condition")
    func reloadAgainstConfiguredEngineSucceeds() async {
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/reload")] = """
        {"servers":[{"name":"time","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"time","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"time__get_current_time","description":"d","server":"time","parameters":{}}],"count":1}
        """
        let applied = await catalog.reload()
        #expect(applied)
        #expect(catalog.isConfigured)
        // A reload has to leave the tool list current too — the response only
        // carries server rows, so it must follow up with the tools route or a
        // newly-added connector's tools would never reach the model.
        #expect(catalog.tools.map { $0.function.name } == ["time__get_current_time"])
    }

    @Test("A tool whose namespaced name isn't a legal function name is dropped")
    func illegalToolNameIsNotAdvertised() async {
        let (catalog, port) = stubbedCatalog()
        let longName = "srv__" + String(repeating: "x", count: 70)  // 75 > 64
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] =
            #"{"servers":[],"error":null,"configured":true}"#
        // Three tools: legal, too long, and with a space (both illegal in an
        // OpenAI function name). Only the legal one survives.
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"srv__ok","description":"d","server":"srv","parameters":{}},
                  {"name":"\(longName)","description":"d","server":"srv","parameters":{}},
                  {"name":"srv__has space","description":"d","server":"srv","parameters":{}}],"count":3}
        """
        await catalog.refresh()
        // The illegal ones can't be emitted by the model, so they are not
        // offered rather than advertised as tools it can never call.
        #expect(catalog.tools.map { $0.function.name } == ["srv__ok"])
        #expect(catalog.serverForTool[longName] == nil)
        #expect(catalog.serverForTool["srv__has space"] == nil)
    }

    @Test("A reload whose tool re-fetch fails reports failure, not success")
    func reloadReportsToolRefetchFailure() async {
        // The reload route answered and reconfigured the engine, but the
        // follow-up tools fetch failed. Reporting success here would leave the
        // caller believing the catalog reflects the new config while it still
        // holds the pre-reload tool list — the stale-advertise trap.
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/reload")] = """
        {"servers":[{"name":"time","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"time","state":"connected","transport":"stdio","tools_count":1,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.statusCodes[key(port, "/v1/mcp/tools")] = 503
        let applied = await catalog.reload()
        #expect(!applied, "a reload that couldn't refresh the tool list is not a success")
        #expect(catalog.fetchError != nil)
    }

    @Test("A reload against an engine with no reload route reports failure")
    func reloadAgainstOlderEngineIsNotSuccess() async {
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.statusCodes[key(port, "/v1/mcp/reload")] = 404
        let applied = await catalog.reload()
        #expect(!applied, "an older engine build must fall back to the restart banner")
        // Pin WHY it failed. `!applied` alone also holds when the request
        // never reached the stub at all, which is how the parallel-fixture bug
        // let this test pass while its sibling failed.
        #expect(catalog.fetchError?.contains("404") == true)
    }

    @Test("A transient fetch failure keeps the last-known-good list")
    func refreshFailureDoesNotWipeState() async {
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = """
        {"servers":[{"name":"fs","state":"connected","transport":"stdio","tools_count":0,"error":null}],
         "error":null,"configured":true}
        """
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = #"{"tools":[],"count":0}"#
        await catalog.refresh()
        #expect(catalog.servers.count == 1)

        // A dropped poll must not make every connector row flicker away.
        MCPStubProtocol.statusCodes[key(port, "/v1/mcp/servers")] = 503
        await catalog.refresh()
        #expect(catalog.servers.count == 1)
        #expect(catalog.fetchError != nil, "…but it must say we couldn't check")
    }

    @Test("Clearing the catalog drops every tool")
    func clearDropsTools() async {
        let (catalog, port) = stubbedCatalog()
        MCPStubProtocol.responses[key(port, "/v1/mcp/servers")] = #"{"servers":[],"error":null,"configured":true}"#
        MCPStubProtocol.responses[key(port, "/v1/mcp/tools")] = """
        {"tools":[{"name":"fs__read_file","description":"d","server":"fs","parameters":{}}],"count":1}
        """
        await catalog.refresh()
        #expect(!catalog.tools.isEmpty)

        // Called when the child goes away, and when the user switches
        // connectors off. Keeping the list would let the chat loop advertise
        // tools with no process behind them.
        catalog.clear()
        #expect(catalog.tools.isEmpty)
        #expect(catalog.serverForTool.isEmpty)

        // Connectors ON so the empty result is attributable to `clear()`, not
        // to the master gate — otherwise the assertion would hold vacuously.
        let registry = MCPToolRegistry(
            catalog: catalog, approval: makeApproval(), defaults: enabledDefaults()
        )
        #expect(registry.definitions.isEmpty)
    }
}

/// Stubs the loopback MCP routes so the catalog can be driven without an
/// engine.
///
/// Fixtures are keyed `"<port><path>"`, not by path alone. URLSession
/// instantiates the protocol itself, so the table has to be static — and
/// swift-testing runs tests in parallel, so a path-only key had concurrent
/// tests overwriting and clearing each other's fixtures. There is
/// deliberately no `reset()`: a global wipe is precisely what broke them.
/// A lock-guarded string-keyed map with the same subscript API as a
/// `Dictionary`. The fixture tables are read from ``URLProtocol/startLoading``
/// — a URLSession background thread — while the `@MainActor` tests write them,
/// and a bare `Dictionary` is undefined under that concurrent access (it
/// crashed the test process). Same `[key]` shape, so call sites are unchanged.
final class LockedFixtureMap<Value>: @unchecked Sendable {
    private var store: [String: Value] = [:]
    private let lock = NSLock()
    subscript(key: String) -> Value? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return store[key]
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            store[key] = newValue
        }
    }
}

final class MCPStubProtocol: URLProtocol, @unchecked Sendable {
    static let responses = LockedFixtureMap<String>()
    static let statusCodes = LockedFixtureMap<Int>()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url
        let key = "\(url?.port ?? 0)\(url?.path ?? "")"
        let status = Self.statusCodes[key] ?? 200
        let body = Self.responses[key] ?? "{}"
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Keychain double so constructing a ``WebSearchConfig`` in these tests never
/// touches the real login keychain (which would prompt, and would leak between
/// runs). ``BuiltinToolsTests`` has its own file-private equivalent; these
/// tests never exercise a key path, so this one just answers "nothing stored".
private struct NullKeychain: KeychainStoring, Sendable {
    func read(account: String) -> String? { nil }
    @discardableResult func write(account: String, secret: String) -> Bool { true }
    @discardableResult func delete(account: String) -> Bool { true }
}
