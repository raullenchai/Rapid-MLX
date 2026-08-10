import Foundation
import Observation

/// Owns `~/.config/rapid-mlx/mcp.json` — the file the engine reads when the
/// app passes `--mcp-config` at spawn.
///
/// Issue #1716: before this, a desktop user who wanted MCP tools had to
/// hand-author that file and know it existed. The app never wrote it and never
/// pointed the engine at it. This store is the write half; ``MCPCatalog`` is
/// the read-back half.
///
/// The file is deliberately the ecosystem-standard shape (`mcpServers` at the
/// root, the same key Claude Desktop and VS Code use), so a config the user
/// already has can be dropped in, and one authored here can be lifted out. The
/// engine accepts both that key and its own historical `servers`
/// (`vllm_mlx/mcp/config.py: select_server_map`).
@MainActor
@Observable
final class MCPConfigStore {
    /// Master opt-in. Off means the app does not pass `--mcp-config` at all,
    /// so the engine starts with no MCP subsystem — not merely with zero
    /// servers. Connectors run arbitrary local commands; that is an explicit
    /// choice, not a default.
    static let enabledKey = "rapid.mcp.connectors.enabled.v1"

    private let defaults: UserDefaults
    private let fileURL: URL

    /// Servers in file order. Order is preserved on save so a user who opens
    /// the JSON by hand sees a stable file.
    private(set) var servers: [MCPServerConfig] = []

    /// Non-nil when the file exists but could not be read or parsed. Surfaced
    /// in the panel — silently showing an empty list over a broken file is the
    /// exact "my connectors vanished" failure this feature is fixing.
    private(set) var loadError: String?

    var isEnabled: Bool {
        didSet {
            guard isEnabled != oldValue else { return }
            defaults.set(isEnabled, forKey: Self.enabledKey)
        }
    }

    /// Invoked with a server's name when that server's execution identity
    /// changes — a command/URL/args/env edit, a rename, or a removal. Wired to
    /// ``MCPToolApprovalStore/revokeGrants(forServer:)`` so a remembered
    /// "always allow" can't silently transfer to code the user did not approve.
    var onServerReconfigured: ((String) -> Void)?

    init(defaults: UserDefaults = .standard, fileURL: URL? = nil) {
        self.defaults = defaults
        self.fileURL = fileURL ?? Self.defaultFileURL
        self.isEnabled = defaults.bool(forKey: Self.enabledKey)
        load()
    }

    /// `~/.config/rapid-mlx/mcp.json` — the first entry of the engine's own
    /// search path (`vllm_mlx/mcp/config.py: CONFIG_SEARCH_PATHS`). Resolved
    /// from the real home directory rather than `NSHomeDirectory()`, which a
    /// sandboxed context would redirect into a container the engine child
    /// would never look in.
    static var defaultFileURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("rapid-mlx", isDirectory: true)
            .appendingPathComponent("mcp.json", isDirectory: false)
    }

    /// Path handed to the engine as `--mcp-config`, or `nil` when there is
    /// nothing worth starting the MCP subsystem for.
    ///
    /// Returning `nil` for "enabled but no servers" is deliberate: passing a
    /// config with an empty `mcpServers` map makes the engine stand up a
    /// manager, connect to nothing, and report an MCP subsystem the user has
    /// no reason to see.
    var launchConfigPath: String? {
        guard isEnabled else { return nil }
        guard servers.contains(where: { $0.enabled }) else { return nil }
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        return fileURL.path
    }

    // MARK: - Read

    func load() {
        loadError = nil
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            servers = []
            return
        }
        do {
            let data = try Data(contentsOf: fileURL)
            servers = try Self.decode(data)
        } catch {
            servers = []
            loadError = "Couldn't read \(fileURL.path): \(error.localizedDescription)"
            return
        }
        // Validation is enforced on the write path (``upsert``), but an
        // imported config — the whole point of the ecosystem-standard shape —
        // never went through it. The engine accepts any dictionary key as a
        // name, so an illegal one is forwarded verbatim, becomes `bad name__tool`,
        // and the model silently can't call it. Surface it here instead of
        // letting it fail invisibly downstream.
        let invalid = servers.filter { !MCPServerConfig.isValidName($0.name) }
        if !invalid.isEmpty {
            let names = invalid.map { "“\($0.name)”" }.joined(separator: ", ")
            loadError =
                "\(invalid.count == 1 ? "Connector" : "Connectors") \(names) "
                + "\(invalid.count == 1 ? "has" : "have") an invalid name — use up to "
                + "\(MCPServerConfig.maxNameLength) letters, numbers, dashes or "
                + "underscores. Its tools won't be callable until renamed."
        }
    }

    /// Parse the `mcpServers` map into an ordered array, reattaching each map
    /// key as the entry's `name`.
    ///
    /// `static` + `Data`-in so the round-trip can be tested without touching
    /// the user's real config directory.
    static func decode(_ data: Data) throws -> [MCPServerConfig] {
        let root = try JSONDecoder().decode(Root.self, from: data)
        let map = root.mcpServers ?? root.servers ?? [:]
        return map
            .map { name, entry -> MCPServerConfig in
                var entry = entry
                entry.name = name
                return entry
            }
            // The JSON object is unordered, so sort by name for a stable list.
            // Without this the Settings rows reshuffle on every load.
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    static func encode(_ servers: [MCPServerConfig]) throws -> Data {
        var map: [String: MCPServerConfig] = [:]
        for server in servers { map[server.name] = server }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(Root(mcpServers: map, servers: nil))
    }

    private struct Root: Codable {
        var mcpServers: [String: MCPServerConfig]?
        /// The engine's historical key. Read for back-compat with a config a
        /// user wrote against an older guide; never written.
        var servers: [String: MCPServerConfig]?
    }

    // MARK: - Write

    enum SaveError: LocalizedError {
        case invalid(String)
        case duplicateName(String)
        case io(String)

        var errorDescription: String? {
            switch self {
            case .invalid(let why):      return why
            case .duplicateName(let n):  return "A connector named “\(n)” already exists."
            case .io(let why):           return why
            }
        }
    }

    /// Insert or replace one server and persist.
    ///
    /// - Parameter replacing: the name being edited, when this is an edit
    ///   rather than an add. Passing it lets a rename land without tripping
    ///   the duplicate check against the entry's own old name.
    func upsert(_ server: MCPServerConfig, replacing originalName: String? = nil) throws {
        if let why = server.validationError { throw SaveError.invalid(why) }
        if servers.contains(where: { $0.name == server.name && $0.name != originalName }) {
            throw SaveError.duplicateName(server.name)
        }
        // Consent invalidation. If an edit changes what code this connector
        // runs — a new command/URL/args/env, or a rename — any "always allow"
        // remembered against the old name must be dropped, or it would silently
        // authorize the replacement. Enable/timeout edits don't change the code
        // and keep their grants. Decide here, but revoke only AFTER the write
        // is durable: a failed persist must not strand a connector with its
        // grants deleted.
        var next = servers
        var reconfiguredServer: String?
        if let originalName, let idx = next.firstIndex(where: { $0.name == originalName }) {
            if next[idx].runsDifferentCode(from: server) || originalName != server.name {
                reconfiguredServer = originalName
            }
            next[idx] = server
        } else {
            next.append(server)
        }
        try persist(next)
        if let reconfiguredServer { onServerReconfigured?(reconfiguredServer) }
    }

    func remove(named name: String) throws {
        // Persist first: a failed removal must not reset consent for a
        // connector that is still installed. A removed server's grants are dead
        // keys, so drop them once the removal is durable.
        try persist(servers.filter { $0.name != name })
        onServerReconfigured?(name)
    }

    func setServerEnabled(_ name: String, _ enabled: Bool) throws {
        var next = servers
        guard let idx = next.firstIndex(where: { $0.name == name }) else { return }
        next[idx].enabled = enabled
        try persist(next)
    }

    private func persist(_ next: [MCPServerConfig]) throws {
        let sorted = next.sorted {
            $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
        do {
            let data = try Self.encode(sorted)
            let dir = fileURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true,
                // The directory holds a file naming local commands to run.
                // 0700 keeps it out of reach of other accounts on a shared Mac.
                attributes: [.posixPermissions: 0o700]
            )
            // createDirectory does NOT tighten an already-existing directory,
            // so a `~/.config/rapid-mlx` created earlier at the umask default
            // (e.g. 0755) would leave the 0600-window argument below false.
            // Set it explicitly, every time — this is what shields the brief
            // post-rename window where the file still carries the umask mode.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: dir.path
            )
            // Atomic so a crash mid-write can't leave the engine reading a
            // truncated config on next launch.
            try data.write(to: fileURL, options: [.atomic])
            // `.atomic` writes via a temp file and renames, which does NOT
            // carry permissions across — set them after the rename, every
            // time, or the mode silently reverts to the default umask.
            //
            // There is a brief window between the rename and this chmod where
            // the file carries the umask default (typically 0644). It is not
            // reachable: `.atomic`'s temp file is created inside `dir`, and
            // `dir` is 0700, so no other account can traverse into it to read
            // the file during that window regardless of the file's own mode.
            // The 0600 is defence in depth on top of the 0700 directory.
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: fileURL.path
            )
            servers = sorted
            loadError = nil
        } catch {
            throw SaveError.io("Couldn't save \(fileURL.path): \(error.localizedDescription)")
        }
    }
}
