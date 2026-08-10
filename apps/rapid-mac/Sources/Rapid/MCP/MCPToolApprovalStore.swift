import Foundation
import Observation

/// Per-tool consent gate for MCP connector tools.
///
/// Issue #1716 states the principle: an MCP server is an arbitrary local
/// process the model can invoke, so "should this model be allowed to call this
/// tool" is a **user** decision and belongs in a UI, not in a JSON file.
///
/// The shape is deliberately ``BrowseApprovalStore``'s — same `Mode`, same
/// `Decision` vocabulary (including the load-bearing `unavailable` vs `deny`
/// distinction), same continuation/cancellation dance. Two approval prompts
/// that behave subtly differently is a worse outcome than a little structural
/// repetition, and the browse store's cancellation handling was already
/// hard-won.
///
/// What's new here is that the grant is **remembered per tool**: "Always
/// allow" on `time__get_current_time` must not silently also grant
/// `shell__run`. `BrowseApprovalStore` can flip one global mode because it
/// gates one tool; this gates an open-ended set.
@MainActor
@Observable
final class MCPToolApprovalStore {
    enum Mode: String {
        case ask
        /// Blanket auto-approve. Off by default and never set implicitly —
        /// only the explicit Settings switch turns it on, so "Always allow"
        /// on a single tool can't widen into everything.
        case autoApproveAll
    }

    enum Decision: Equatable {
        case allowOnce
        /// Approve and remember for this tool specifically.
        case alwaysAllowTool
        /// The USER said no.
        case deny
        /// Nobody decided: the turn was cancelled, or a prompt was already up.
        /// Reported as ``FailureDiagnosis.Kind.userDeclined``'s sibling case,
        /// not as a decline — see ``BrowseApprovalStore/Decision/unavailable``.
        case unavailable
    }

    struct PendingApproval: Equatable {
        /// Namespaced tool name as the engine knows it (`server__tool`).
        let toolName: String
        /// The server half, shown on its own. "Run read_file?" is unanswerable
        /// without knowing whose `read_file`.
        let serverName: String
        /// Short, display-safe tool name (the part after `server__`).
        let shortName: String
        /// Capped, display-safe preview of the arguments the model chose.
        let argumentsPreview: String
    }

    static let modeKey = "rapid.mcp.approval.mode.v1"
    /// Per-tool grant. Keyed on the namespaced name so two servers exposing a
    /// tool of the same name are granted separately.
    static func grantKey(_ toolName: String) -> String {
        "rapid.mcp.approval.tool.\(toolName).v1"
    }

    private let defaults: UserDefaults

    var mode: Mode {
        didSet {
            guard mode != oldValue else { return }
            defaults.set(mode.rawValue, forKey: Self.modeKey)
        }
    }

    private(set) var pendingRequest: PendingApproval?
    private var pendingContinuation: CheckedContinuation<Decision, Never>?

    /// Tools with a remembered "always allow". Mirrored in memory so the
    /// Settings list can render (and revoke) them without scanning defaults.
    private(set) var grantedTools: Set<String>

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.mode = Mode(rawValue: defaults.string(forKey: Self.modeKey) ?? "") ?? .ask
        var granted = Set<String>()
        for (key, value) in defaults.dictionaryRepresentation() {
            guard key.hasPrefix("rapid.mcp.approval.tool."),
                  key.hasSuffix(".v1"),
                  (value as? Bool) == true else { continue }
            let name = String(key.dropFirst("rapid.mcp.approval.tool.".count).dropLast(".v1".count))
            if !name.isEmpty { granted.insert(name) }
        }
        self.grantedTools = granted
    }

    // MARK: - Queries

    func isGranted(_ toolName: String) -> Bool {
        mode == .autoApproveAll || grantedTools.contains(toolName)
    }

    /// Forget every remembered grant. The blanket auto-approve mode is a
    /// separate switch and is left alone — a user resetting individual grants
    /// has not asked to change the global posture.
    func resetGrants() {
        for name in grantedTools {
            defaults.removeObject(forKey: Self.grantKey(name))
        }
        grantedTools = []
    }

    // MARK: - Gate

    /// Gate one tool call. Returns immediately when already approved,
    /// otherwise suspends until the UI answers.
    func requestApproval(
        toolName: String,
        serverName: String,
        argumentsJSON: String
    ) async -> Decision {
        if isGranted(toolName) { return .allowOnce }
        // Re-entrancy guard — tool execution is serial, so a second pending
        // request means something is wrong; refuse rather than hang. NOT a
        // decline: the user was never shown this call.
        if pendingRequest != nil { return .unavailable }

        let shortName = Self.shortToolName(toolName)
        let preview = BrowseApprovalStore.displaySafe(
            BrowseApprovalStore.previewLine(argumentsJSON, cap: 400)
        )

        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Decision, Never>) in
                // Cancellation can land between the re-entrancy check and
                // here; re-check inside the body so we bail rather than
                // install a sheet nobody can ever answer.
                if Task.isCancelled {
                    continuation.resume(returning: .unavailable)
                    return
                }
                self.pendingContinuation = continuation
                self.pendingRequest = PendingApproval(
                    toolName: toolName,
                    serverName: BrowseApprovalStore.displaySafe(serverName),
                    shortName: BrowseApprovalStore.displaySafe(shortName),
                    argumentsPreview: preview
                )
            }
        } onCancel: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let cont = self.pendingContinuation {
                    self.pendingContinuation = nil
                    self.pendingRequest = nil
                    cont.resume(returning: .unavailable)
                }
            }
        }
    }

    /// Called by the SwiftUI dialog with the user's choice; resumes the tool.
    func answer(_ decision: Decision) {
        guard let pending = pendingRequest else { return }
        if decision == .alwaysAllowTool {
            defaults.set(true, forKey: Self.grantKey(pending.toolName))
            grantedTools.insert(pending.toolName)
        }
        pendingRequest = nil
        pendingContinuation?.resume(returning: decision)
        pendingContinuation = nil
    }

    /// The tool half of an engine-namespaced `server__tool` name.
    ///
    /// Splits on the FIRST `__` — a tool whose own name contains a double
    /// underscore keeps it, which is right: the server half is what the engine
    /// prefixed, and it never contains one.
    static func shortToolName(_ fullName: String) -> String {
        guard let range = fullName.range(of: "__") else { return fullName }
        return String(fullName[range.upperBound...])
    }
}
