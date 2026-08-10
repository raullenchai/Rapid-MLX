import Foundation
import Observation

/// ``ToolRegistry`` over the MCP tools the engine has connected.
///
/// Issue #1716. The engine never injects MCP tools into `/v1/chat/completions`
/// on its own (`MCPClientManager.get_merged_tools` exists but nothing calls
/// it), so the tool loop stays entirely on this side — which is what lets the
/// consent gate live in the UI where it belongs. The flow per call is:
///
///   model emits `server__tool`
///     → ``ChatViewModel`` refuses it unless it was advertised this round
///     → ``run(_:)`` → ``MCPToolApprovalStore`` (user says yes)
///     → `POST /v1/mcp/execute` → result back to the model
///
/// Two independent gates precede execution and both are load-bearing: a tool
/// the user switched off is never advertised AND is refused at dispatch, and a
/// tool that was never approved does not run even if it was advertised.
@MainActor
@Observable
final class MCPToolRegistry: ToolRegistry {
    let catalog: MCPCatalog
    let approval: MCPToolApprovalStore

    /// Per-tool off switches from the Connectors panel, keyed by namespaced
    /// tool name. Kept here rather than in ``ChatViewModel/disabledTools``
    /// because these names come and go with the connected servers, while that
    /// set is seeded once from a fixed built-in list.
    private static func enabledKey(_ toolName: String) -> String {
        "rapid.mcp.tool.enabled.\(toolName).v1"
    }

    private let defaults: UserDefaults
    private(set) var disabledTools: Set<String>

    init(
        catalog: MCPCatalog,
        approval: MCPToolApprovalStore,
        defaults: UserDefaults = .standard
    ) {
        self.catalog = catalog
        self.approval = approval
        self.defaults = defaults
        var disabled = Set<String>()
        for (key, value) in defaults.dictionaryRepresentation() {
            guard key.hasPrefix("rapid.mcp.tool.enabled."),
                  key.hasSuffix(".v1"),
                  (value as? Bool) == false else { continue }
            let name = String(
                key.dropFirst("rapid.mcp.tool.enabled.".count).dropLast(".v1".count)
            )
            if !name.isEmpty { disabled.insert(name) }
        }
        self.disabledTools = disabled
    }

    func setToolEnabled(_ toolName: String, _ enabled: Bool) {
        defaults.set(enabled, forKey: Self.enabledKey(toolName))
        if enabled {
            disabledTools.remove(toolName)
        } else {
            disabledTools.insert(toolName)
        }
    }

    func isToolEnabled(_ toolName: String) -> Bool {
        !disabledTools.contains(toolName)
    }

    /// The master switch, read from the same defaults ``MCPConfigStore`` writes
    /// it to. The authoritative connectors-off gate lives HERE, not in the
    /// catalog: turning the switch off clears the catalog, but the running
    /// child still has its connectors loaded, so a later ``/healthz`` ready
    /// transition (`ContentView`) can repopulate the catalog from that live
    /// child. Gating advertise + dispatch on the switch means "connectors off"
    /// holds even across that repopulation — the model is never handed a
    /// connector tool the user turned off, whatever the catalog currently says.
    private var connectorsEnabled: Bool {
        defaults.bool(forKey: MCPConfigStore.enabledKey)
    }

    /// Everything the catalog reports, minus what the user switched off — and
    /// nothing at all while the master switch is off.
    var definitions: [ToolDefinition] {
        guard connectorsEnabled else { return [] }
        return catalog.tools.filter { !disabledTools.contains($0.function.name) }
    }

    /// Tools the catalog knows about regardless of the user's switches — the
    /// Settings list needs the off ones too, or a disabled tool would vanish
    /// and leave no way to re-enable it.
    var allKnownTools: [ToolDefinition] { catalog.tools }

    func run(_ call: ToolCall) async -> ToolCallResult {
        let name = call.function.name

        // Master switch. The child may still have connectors loaded (they
        // unload on restart), so refuse execution here rather than trust that
        // the catalog was cleared — same reasoning as ``definitions``.
        guard connectorsEnabled else {
            return ToolCallResult(
                toolCallID: call.id,
                content: "Connectors are turned off in Settings → Connectors; '\(name)' was not run.",
                isError: true,
                failureKind: .userDeclined
            )
        }

        // Defence in depth. ``ChatViewModel`` already refuses anything not
        // advertised this round, but this registry is reachable from any
        // future caller and a disabled connector tool must never execute.
        guard !disabledTools.contains(name) else {
            return ToolCallResult(
                toolCallID: call.id,
                content: "tool '\(name)' is turned off in Settings → Connectors and was not run.",
                isError: true,
                failureKind: .userDeclined
            )
        }

        let server = catalog.serverForTool[name] ?? "unknown"

        switch await approval.requestApproval(
            toolName: name,
            serverName: server,
            argumentsJSON: call.function.arguments
        ) {
        case .allowOnce, .alwaysAllowTool:
            break
        case .deny:
            return ToolCallResult(
                toolCallID: call.id,
                content: "The user declined to run '\(name)'. Continue without it.",
                isError: true,
                failureKind: .userDeclined
            )
        case .unavailable:
            return ToolCallResult(
                toolCallID: call.id,
                content: "'\(name)' was not run — the request was cancelled before it could be approved.",
                isError: true,
                failureKind: .userDeclined
            )
        }

        do {
            let response = try await catalog.execute(
                toolName: name,
                argumentsJSON: call.function.arguments
            )
            let text = response.text
            return ToolCallResult(
                toolCallID: call.id,
                // An empty body from a successful call is not an error, but
                // handing the model "" invites it to invent the answer. Say
                // plainly that the tool returned nothing.
                content: text.isEmpty && !response.is_error
                    ? "(the tool returned no content)"
                    : text,
                isError: response.is_error
            )
        } catch {
            return ToolCallResult(
                toolCallID: call.id,
                content: "'\(name)' could not run — \(error.localizedDescription).",
                isError: true
            )
        }
    }
}
