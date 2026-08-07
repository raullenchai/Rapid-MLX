import Foundation

/// A user-facing explanation for a known failure mode. Raw process,
/// transport, and tool output stays in logs/model context; views render only
/// this stable copy and, when available, one concrete recovery action.
struct FailureDiagnosis: Equatable, Sendable {
    enum Kind: String, CaseIterable, Codable, Equatable, Hashable, Sendable {
        case modelOutOfMemory
        case modelLoadFailed
        case engineNotRunning
        case webSearchOffline
        case webSearchUnavailable
        /// The free DuckDuckGo backend throttled this machine. Distinct from
        /// ``webSearchUnavailable`` because the remedy is different: nothing in
        /// Settings is misconfigured, so "check its settings" sends the user to
        /// a dead end. The only real fix is a different backend.
        case webSearchRateLimited
        case commandPermissionDenied
        case commandFailed
        case fileNotFound
        case filePermissionDenied
        case toolFailed
        case downloadFailed
        case downloadSourceUnavailable
        case requestFailed
    }

    enum Action: String, Equatable, Sendable {
        case retry
        case restart
        case openModelManagement
        case switchDownloadSource
        case openPermissions
        /// Deep-link to Settings → Tools, where the web-search backend is
        /// chosen and its key is pasted. Routed through ``SettingsRouter``
        /// like the other "open Settings on THIS tab" actions.
        case openWebSearchSettings

        var title: String {
            switch self {
            case .retry: return "Retry"
            case .restart: return "Restart"
            case .openModelManagement: return "Open Model Management"
            case .switchDownloadSource: return "Switch source"
            case .openPermissions: return "Open Permissions"
            case .openWebSearchSettings: return "Open Web Search Settings"
            }
        }

        var systemImage: String {
            switch self {
            case .retry, .restart: return "arrow.clockwise"
            case .openModelManagement: return "square.stack.3d.up"
            case .switchDownloadSource: return "arrow.triangle.2.circlepath"
            case .openPermissions: return "hand.raised"
            case .openWebSearchSettings: return "magnifyingglass"
            }
        }
    }

    let kind: Kind
    let message: String
    let action: Action?

    /// The recovery action a tool card may render inline, or nil for "render
    /// no button". Two gates, both load-bearing:
    ///
    ///   * **Settings deep-links only.** ``.retry`` would have to rewind the
    ///     whole chat turn; the assistant row above the card already owns that
    ///     affordance. "Open Settings on the right tab" has nowhere else to
    ///     live, so it is the one action the card offers.
    ///   * **Only when the deep-link can actually run.** The card resolves
    ///     ``SettingsRouter`` optionally so it still renders in a host that
    ///     never injected one (previews, the snapshot harness). Those hosts
    ///     have no Settings window to open either — and a visible button that
    ///     does nothing is precisely the failure this diagnosis exists to
    ///     remove, so the button must be absent rather than inert.
    ///
    /// Pure + static because the view that calls it is `private` inside
    /// ChatView and a SwiftUI body can't be exercised from the test suite.
    static func inlineToolCardAction(
        for diagnosis: FailureDiagnosis?,
        canRouteToSettings: Bool
    ) -> Action? {
        guard canRouteToSettings else { return nil }
        switch diagnosis?.action {
        case .openWebSearchSettings: return .openWebSearchSettings
        default: return nil
        }
    }
}

/// Rule-based classification for common failures. Matching deliberately uses
/// raw details only as input; none of those details are returned for display.
enum FailureDiagnoser {
    nonisolated static func diagnosis(
        for kind: FailureDiagnosis.Kind,
        modelAlias: String? = nil
    ) -> FailureDiagnosis {
        let message: String
        let action: FailureDiagnosis.Action?

        switch kind {
        case .modelOutOfMemory:
            if let modelAlias,
               ModelSizing.estimate(alias: modelAlias).paramsBillions != nil {
                let required = ModelSizing.estimate(alias: modelAlias).totalGB
                message = "This model needs about \(formatGB(required)) GB free. Free up memory or choose a smaller model."
            } else {
                message = "This model needs more free memory. Free up memory or choose a smaller model."
            }
            action = .openModelManagement
        case .modelLoadFailed:
            message = "This model couldn't load. Check the model files or choose another model."
            action = .openModelManagement
        case .engineNotRunning:
            message = "The local engine isn't running. Restart it, then try again."
            action = .restart
        case .webSearchOffline:
            message = "Web search couldn't connect. Turn on network access, then try again."
            action = .retry
        case .webSearchUnavailable:
            message = "Web search couldn't finish. Check its settings, then try again."
            action = .retry
        case .webSearchRateLimited:
            // Deliberately NOT "check its settings": everything in Settings is
            // already correct when this fires. DuckDuckGo rate-limits the free
            // endpoint per IP within a handful of searches, so the honest
            // remedy is a different backend, and the message has to say which
            // ones and where. Kept to one sentence of situation + one of
            // remedy so it still reads inside the tool card.
            message = "DuckDuckGo is rate-limiting web searches from this Mac. Switch to Brave Search or Tavily in Settings → Tools and add a free key."
            action = .openWebSearchSettings
        case .commandPermissionDenied:
            message = "The command tried to change a protected location. Allow that folder, then try again."
            action = .openPermissions
        case .commandFailed:
            message = "The command didn't finish successfully. Check the command, then try again."
            action = .retry
        case .fileNotFound:
            message = "That file isn't there. Check its name or location, then try again."
            action = .retry
        case .filePermissionDenied:
            message = "Rapid doesn't have access to that file. Allow access, then try again."
            action = .openPermissions
        case .toolFailed:
            message = "The tool couldn't finish. Check its input, then try again."
            action = .retry
        case .downloadFailed:
            message = "The model download didn't finish. Check your connection, then try again."
            action = .retry
        case .downloadSourceUnavailable:
            message = "The current download source couldn't be reached. Switch source and try again."
            action = .switchDownloadSource
        case .requestFailed:
            message = "Rapid couldn't finish that request. Try again, or restart the model."
            action = .retry
        }
        return FailureDiagnosis(kind: kind, message: message, action: action)
    }

    /// Classifies a completed tool result. A non-nil return means the result
    /// should be styled as failed even if the tool itself returned structured
    /// output with `isError == false` (notably `run_command` exit failures).
    nonisolated static func toolFailureKind(
        toolName: String,
        content: String,
        isError: Bool
    ) -> FailureDiagnosis.Kind? {
        let raw = content.lowercased()

        if toolName == "run_command", let command = commandResult(from: content), command.exitCode != 0 {
            if containsAny(command.stderr.lowercased(), permissionSignals) {
                return .commandPermissionDenied
            }
            if containsAny(command.stderr.lowercased(), missingFileSignals) {
                return .fileNotFound
            }
            return .commandFailed
        }

        guard isError else { return nil }

        if toolName == "web_search" {
            // ``WebSearchTool`` stamps ``.webSearchRateLimited`` on the result
            // directly, so this branch only matters for rows restored from an
            // older transcript (no stored kind) — hence the narrow, DDG-anchored
            // signals. A Brave/Tavily quota error must NOT land here: telling a
            // Brave user to "switch to Brave" is the same dead end this fix is
            // removing.
            if raw.contains("duckduckgo"), containsAny(raw, duckDuckGoThrottleSignals) {
                return .webSearchRateLimited
            }
            return containsAny(raw, offlineSignals) ? .webSearchOffline : .webSearchUnavailable
        }
        if ["read_file", "list_directory", "write_file", "edit_file"].contains(toolName) {
            if containsAny(raw, missingFileSignals) { return .fileNotFound }
            if containsAny(raw, permissionSignals) { return .filePermissionDenied }
        }
        if toolName == "run_command" {
            if containsAny(raw, permissionSignals) { return .commandPermissionDenied }
            if containsAny(raw, missingFileSignals) { return .fileNotFound }
            return .commandFailed
        }
        return .toolFailed
    }

    nonisolated static func downloadFailureKind(
        raw: String,
        usingMirror: Bool
    ) -> FailureDiagnosis.Kind {
        guard usingMirror else { return .downloadFailed }
        let value = raw.lowercased()
        let mirrorSignals = offlineSignals + [
            "mirror", "models.rapidmlx.com", "cloudflare", "bad gateway",
            "service unavailable", "gateway timeout", "status 502", "status 503", "status 504",
        ]
        return containsAny(value, mirrorSignals) ? .downloadSourceUnavailable : .downloadFailed
    }

    nonisolated static func modelLoadFailureKind(raw: String) -> FailureDiagnosis.Kind {
        let value = raw.lowercased()
        return containsAny(value, memorySignals) ? .modelOutOfMemory : .modelLoadFailed
    }

    nonisolated static func engineFailureKind(raw: String) -> FailureDiagnosis.Kind {
        let value = raw.lowercased()
        if containsAny(value, memorySignals) { return .modelOutOfMemory }
        if containsAny(value, modelLoadSignals) { return .modelLoadFailed }
        return .engineNotRunning
    }

    nonisolated static func chatFailureKind(raw: String) -> FailureDiagnosis.Kind {
        let value = raw.lowercased()
        if containsAny(value, ["out of memory", "more memory", "memory than your mac"]) {
            return .modelOutOfMemory
        }
        if containsAny(value, [
            "can't reach the model", "couldn't reach the model", "disconnected",
            "connection refused", "local engine isn't running",
        ]) {
            return .engineNotRunning
        }
        return .requestFailed
    }

    nonisolated static func chatFailureKind(error: Error) -> FailureDiagnosis.Kind {
        if let chat = error as? ChatStreamError {
            switch chat {
            case .streamTruncated:
                return .engineNotRunning
            case .httpStatus(_, let body), .transport(let body):
                if modelLoadFailureKind(raw: body) == .modelOutOfMemory {
                    return .modelOutOfMemory
                }
                return .requestFailed
            }
        }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorCannotConnectToHost, NSURLErrorCannotFindHost,
                 NSURLErrorNetworkConnectionLost:
                return .engineNotRunning
            default:
                return .requestFailed
            }
        }
        return .requestFailed
    }

    private struct CommandResult {
        let exitCode: Int
        let stderr: String
    }

    nonisolated private static func commandResult(from content: String) -> CommandResult? {
        guard let data = content.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let exitCode = object["exit_code"] as? Int else {
            return nil
        }
        return CommandResult(exitCode: exitCode, stderr: object["stderr"] as? String ?? "")
    }

    nonisolated private static func containsAny(_ value: String, _ signals: [String]) -> Bool {
        signals.contains(where: value.contains)
    }

    nonisolated private static let offlineSignals = [
        "not connected to the internet", "network is unreachable", "network connection was lost",
        "could not resolve host", "cannot find host", "cannot connect", "connection refused",
        "connection reset", "dns", "offline", "timed out", "timeout",
    ]

    /// Throttle wording DuckDuckGo results have carried across releases.
    /// Only consulted together with a ``duckduckgo`` mention (see
    /// ``toolFailureKind``).
    nonisolated private static let duckDuckGoThrottleSignals = [
        "throttl", "rate limit", "rate-limit", "anti-bot", "blocked this request",
    ]

    nonisolated private static let missingFileSignals = [
        "no such file", "file not found", "does not exist", "is missing", "missing or not a directory",
    ]

    nonisolated private static let permissionSignals = [
        "operation not permitted", "permission denied", "access is blocked", "access denied",
        "user denied", "sandbox denial", "deny file-write", "read-only file system",
    ]

    nonisolated private static let memorySignals = [
        "out of memory", "insufficient memory", "memory pressure", "metal-cap",
        "gpu_memory_utilization", "projected kv", "metal active",
    ]

    nonisolated private static let modelLoadSignals = [
        "couldn't start the model", "could not start the model",
        "couldn't load the model", "could not load the model", "failed to load the model",
        "model name isn't valid", "model files",
    ]

    nonisolated private static func formatGB(_ value: Double) -> String {
        let rounded = ceil(value * 10) / 10
        if rounded.rounded() == rounded { return String(Int(rounded)) }
        return String(format: "%.1f", rounded)
    }
}
