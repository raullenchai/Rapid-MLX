import Foundation

/// One MCP server entry, as it round-trips through `~/.config/rapid-mlx/mcp.json`.
///
/// Mirrors the engine's `MCPServerConfig` (`vllm_mlx/mcp/types.py`) field for
/// field. The engine is the one that actually spawns these processes and
/// validates them (`vllm_mlx/mcp/security.py`); this type exists so the app can
/// author the file the engine reads, and so the editor sheet has something
/// typed to bind to.
struct MCPServerConfig: Codable, Equatable, Hashable, Sendable, Identifiable {
    enum Transport: String, Codable, CaseIterable, Sendable {
        case stdio
        case sse

        var displayName: String {
            switch self {
            case .stdio: return "Command (stdio)"
            case .sse:   return "URL (SSE)"
            }
        }
    }

    /// The map key in `mcpServers`. Also the namespace half of every tool name
    /// this server exposes — see ``Self/isValidName(_:)``.
    var name: String
    var transport: Transport
    /// stdio only. Must be on the engine's command allowlist
    /// (`security.py: ALLOWED_COMMANDS`) or the engine rejects the entry.
    var command: String?
    var args: [String]
    var env: [String: String]
    /// SSE only.
    var url: String?
    var enabled: Bool
    var timeout: Double

    var id: String { name }

    init(
        name: String,
        transport: Transport = .stdio,
        command: String? = nil,
        args: [String] = [],
        env: [String: String] = [:],
        url: String? = nil,
        enabled: Bool = true,
        timeout: Double = 30
    ) {
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args
        self.env = env
        self.url = url
        self.enabled = enabled
        self.timeout = timeout
    }

    // MARK: - Validation

    /// Longest server name we accept.
    ///
    /// The engine namespaces every tool as `server__tool`
    /// (`vllm_mlx/mcp/types.py: MCPTool.full_name`), and that composite string
    /// travels as an OpenAI function name — which the spec caps at 64
    /// characters from `[a-zA-Z0-9_-]`. Capping the server half at 32 leaves
    /// room for a realistic tool name plus the two-underscore separator, and
    /// rejecting here means the user finds out in the editor sheet rather than
    /// through a model that mysteriously never calls one server's tools.
    static let maxNameLength = 32

    /// True when `name` can safely form the namespace half of a tool name.
    ///
    /// Deliberately stricter than the engine, which accepts any dictionary
    /// key: a server called `my server` produces `my server__read_file`, and
    /// a space is not a legal OpenAI function-name character.
    static func isValidName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= maxNameLength else { return false }
        // `__` is the namespace separator: the engine builds `server__tool` and
        // both sides split on the FIRST `__`. A name that contains one (e.g.
        // `my__server`) would be parsed as server `my`, tool `server__…`, and
        // never dispatch. A single `_` is fine.
        if name.contains("__") { return false }
        for ch in name.unicodeScalars {
            let ok = (ch >= "a" && ch <= "z")
                || (ch >= "A" && ch <= "Z")
                || (ch >= "0" && ch <= "9")
                || ch == "_" || ch == "-"
            if !ok { return false }
        }
        return true
    }

    /// A stable string of this connector's execution identity — transport,
    /// command, arguments, environment, URL. Two configs with the same
    /// fingerprint run the same code; a change means consent must be
    /// re-established. Used to catch hand-edits to the config file, which never
    /// pass through the in-app edit path. Not cryptographic — only needs to
    /// change when the execution identity does.
    var executionFingerprint: String {
        let envPart = env.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ",")
        return [
            transport.rawValue,
            command ?? "",
            args.joined(separator: "\u{1}"),
            envPart,
            url ?? "",
        ].joined(separator: "\u{2}")
    }

    /// True when `other` would launch or reach a different program than `self`
    /// — a changed transport, command, arguments, environment, or URL.
    ///
    /// Deliberately ignores `name`, `enabled`, and `timeout`: those don't
    /// change what code runs, so they don't invalidate a consent grant. Drives
    /// ``MCPConfigStore``'s grant invalidation on edit.
    func runsDifferentCode(from other: MCPServerConfig) -> Bool {
        transport != other.transport
            || command != other.command
            || args != other.args
            || env != other.env
            || url != other.url
    }

    /// Human-readable reason this entry can't be saved, or `nil` when it can.
    /// Drives the editor sheet's inline error and its disabled Save button.
    var validationError: String? {
        if name.isEmpty {
            return "Give this connector a name."
        }
        if !Self.isValidName(name) {
            return "Use up to \(Self.maxNameLength) letters, numbers, dashes or underscores — the name becomes part of every tool name."
        }
        switch transport {
        case .stdio:
            if (command ?? "").trimmingCharacters(in: .whitespaces).isEmpty {
                return "A command connector needs a command to run."
            }
        case .sse:
            let raw = (url ?? "").trimmingCharacters(in: .whitespaces)
            if raw.isEmpty {
                return "A URL connector needs a URL."
            }
            guard let parsed = URL(string: raw), let scheme = parsed.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else {
                return "Enter an http:// or https:// URL."
            }
        }
        if timeout <= 0 {
            return "Timeout must be greater than zero."
        }
        return nil
    }

    // MARK: - Wire shape

    /// The engine's JSON keys. `name` is the map key, not a field, so it is
    /// excluded — ``MCPConfigStore`` reattaches it on load.
    private enum CodingKeys: String, CodingKey {
        case transport, command, args, env, url, enabled, timeout
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // A hand-written config (or one pasted from another tool) routinely
        // omits `transport` and just gives a command. Default rather than
        // fail: refusing to decode would drop the entry from the UI entirely,
        // which is the "my connectors vanished" failure this feature exists to
        // avoid.
        self.transport = try c.decodeIfPresent(Transport.self, forKey: .transport) ?? .stdio
        self.command = try c.decodeIfPresent(String.self, forKey: .command)
        self.args = try c.decodeIfPresent([String].self, forKey: .args) ?? []
        self.env = try c.decodeIfPresent([String: String].self, forKey: .env) ?? [:]
        self.url = try c.decodeIfPresent(String.self, forKey: .url)
        self.enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        self.timeout = try c.decodeIfPresent(Double.self, forKey: .timeout) ?? 30
        self.name = ""  // reattached by MCPConfigStore from the map key
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(transport, forKey: .transport)
        // Only write the fields this transport actually uses. A stdio entry
        // carrying a stale `url` (left behind by a user who switched transport
        // mid-edit) reads as ambiguous in a file the user may open by hand.
        switch transport {
        case .stdio:
            try c.encodeIfPresent(command, forKey: .command)
            if !args.isEmpty { try c.encode(args, forKey: .args) }
            if !env.isEmpty { try c.encode(env, forKey: .env) }
        case .sse:
            try c.encodeIfPresent(url, forKey: .url)
        }
        try c.encode(enabled, forKey: .enabled)
        try c.encode(timeout, forKey: .timeout)
    }
}
