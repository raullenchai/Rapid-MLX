import Foundation
import Observation

/// Consent gate for actions performed by Rapid's built-in local-workspace tools.
///
/// Read-only actions may be remembered per tool for this app session. Mutating actions deliberately
/// set `allowsPersistentGrant` to false and therefore require approval every
/// time, even when the user previously approved a read or search.
@MainActor
@Observable
final class LocalToolApprovalStore {
    enum Decision: Equatable {
        case allowOnce
        case alwaysAllowTool
        case deny
        case unavailable
    }

    struct PendingApproval: Equatable {
        let toolName: String
        let title: String
        let argumentsPreview: String
        let allowsPersistentGrant: Bool
    }

    private var sessionGrants: Set<String> = []
    private(set) var pendingRequest: PendingApproval?
    private var pendingContinuation: CheckedContinuation<Decision, Never>?
    private var pendingToken = 0

    init(defaults: UserDefaults = .standard) {
        // Kept injectable for symmetry with the other approval stores and
        // deterministic tests. Local grants intentionally never persist.
        _ = defaults
    }

    func isGranted(_ toolName: String) -> Bool {
        sessionGrants.contains(toolName)
    }

    func requestApproval(
        toolName: String,
        title: String,
        argumentsJSON: String,
        allowsPersistentGrant: Bool
    ) async -> Decision {
        if allowsPersistentGrant, isGranted(toolName) { return .allowOnce }
        if pendingRequest != nil { return .unavailable }

        let token = pendingToken &+ 1
        pendingToken = token
        let preview = BrowseApprovalStore.displaySafe(argumentsJSON)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if Task.isCancelled {
                    continuation.resume(returning: .unavailable)
                    return
                }
                pendingContinuation = continuation
                pendingRequest = PendingApproval(
                    toolName: toolName,
                    title: BrowseApprovalStore.displaySafe(title),
                    argumentsPreview: preview,
                    allowsPersistentGrant: allowsPersistentGrant
                )
            }
        } onCancel: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self, self.pendingToken == token else { return }
                let continuation = self.pendingContinuation
                self.pendingContinuation = nil
                self.pendingRequest = nil
                continuation?.resume(returning: .unavailable)
            }
        }
    }

    func answer(_ decision: Decision) {
        guard let pending = pendingRequest else { return }
        if decision == .alwaysAllowTool, pending.allowsPersistentGrant {
            sessionGrants.insert(pending.toolName)
        }
        pendingRequest = nil
        pendingContinuation?.resume(returning: decision)
        pendingContinuation = nil
    }
}
