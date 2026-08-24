import Foundation

/// A user-facing explanation for a known outcome that stopped short of a
/// result. Raw process, transport, and tool output stays in logs/model
/// context; views render only this stable copy and, when available, one
/// concrete recovery action.
///
/// Most kinds describe a genuine fault. One does not: ``Kind/userDeclined``
/// is the user answering "no" to a permission prompt, which is an ordinary
/// outcome and must never be dressed as a malfunction — see ``Severity``.
struct FailureDiagnosis: Equatable, Sendable {
    enum Kind: String, CaseIterable, Codable, Equatable, Hashable, Sendable {
        case modelOutOfMemory
        case modelLoadFailed
        case engineNotRunning
        case webSearchOffline
        case webSearchUnavailable
        /// A user-supplied Keenable key was rejected. Keep keyed account
        /// failures distinct from provider availability: the network is up,
        /// retrying the same request cannot repair the credential, and the
        /// useful destination is the key field in Settings.
        case webSearchKeyRejected
        /// The stored Keenable key has exhausted its monthly credit allowance.
        case webSearchKeyQuotaExceeded
        /// The stored Keenable key hit its account/organisation rate cap.
        case webSearchKeyRateLimited
        /// The free DuckDuckGo backend throttled this machine. Distinct from
        /// ``webSearchUnavailable`` because the remedy is different: nothing in
        /// Settings is misconfigured, so "check its settings" sends the user to
        /// a dead end. The only real fix is a different backend.
        case webSearchRateLimited
        /// A browse fetch crossed the bounded response cap. The agent can
        /// normally recover by searching or choosing a narrower page.
        case browsePageTooLarge
        case commandPermissionDenied
        case commandFailed
        case fileNotFound
        case filePermissionDenied
        case toolFailed
        case userDeclined
        case downloadFailed
        /// The user stopped the pull themselves. Distinct from
        /// ``downloadFailed`` because the two differ in the only thing the
        /// copy is for: nothing went wrong, and "check your connection" is
        /// advice about a fault that did not happen. Paper 05.1 flags the
        /// collapsed reading explicitly ("For a user-initiated stop that
        /// connection advice is wrong … the fix belongs in FailureDiagnoser,
        /// not in the design"), so the split lives here rather than in a
        /// screen deciding to say something different about the same kind.
        case downloadCancelled
        case downloadSourceUnavailable
        case requestFailed

        /// Forward-tolerant decode, matching ``ChatMessage/Role`` and
        /// ``ChatMessage/Status``: a raw value this build doesn't know (a
        /// transcript written by a NEWER build, then a downgrade) degrades to
        /// the generic ``.toolFailed`` rather than throwing. A throw here is
        /// not local — ``ConversationStore/load`` treats one undecodable
        /// message as a corrupt library and sides the whole history file, so
        /// the user's sidebar reads as wiped.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Kind(rawValue: raw) ?? .toolFailed
        }

        /// The nearest kind that EVERY shipped build already knows, used when
        /// writing to a key older builds decode strictly. Cases present since
        /// the first release map to themselves; anything added later names its
        /// closest ancestor here. See ``ChatMessage/encode(to:)``.
        var legacyPersistedKind: Kind {
            switch self {
            case .userDeclined:
                return .toolFailed
            // Added in #1580, after the same release ``userDeclined`` post-dates,
            // so it carries the same hazard: an older build decoding this raw
            // value throws, and one throw sides the ENTIRE history file.
            // ``webSearchUnavailable`` is the honest ancestor — it is the kind
            // this very condition used to land on, so an older build shows the
            // copy it always showed for a throttled search.
            case .webSearchRateLimited, .webSearchKeyRejected,
                 .webSearchKeyQuotaExceeded, .webSearchKeyRateLimited:
                return .webSearchUnavailable
            case .browsePageTooLarge:
                return .toolFailed
            // Added after the same release, and carrying the same hazard. A
            // download outcome is never written into a transcript today, but
            // the ancestor is named anyway so that stays true by construction
            // rather than by nobody having tried it.
            case .downloadCancelled:
                return .downloadFailed
            case .modelOutOfMemory, .modelLoadFailed, .engineNotRunning,
                 .webSearchOffline, .webSearchUnavailable,
                 .commandPermissionDenied, .commandFailed,
                 .fileNotFound, .filePermissionDenied, .toolFailed,
                 .downloadFailed, .downloadSourceUnavailable, .requestFailed:
                return self
            }
        }
    }

    /// How a diagnosis should be PRESENTED, independent of what it says.
    ///
    /// Something the app (or the network, or the model) got wrong is an
    /// ``error``: red, alarming, worth interrupting for. Something the USER
    /// chose is a ``notice``: it explains why nothing came back and then gets
    /// out of the way. Painting a deliberate "Don't allow" red tells the user
    /// their own decision was a malfunction, and — because the copy on that
    /// lane is written for faults — blames their input for it.
    enum Severity: String, Equatable, Hashable, Sendable {
        case error
        case notice
    }

    /// A recovery affordance a failure card may offer. Every case must be
    /// something a view can actually carry out in THIS app — an action with no
    /// destination is the defect this type exists to prevent, and it is worse
    /// than no button because it fires at the moment the user has already been
    /// let down once.
    ///
    /// ``CaseIterable`` on purpose: it lets the routing table
    /// (``SettingsRouter/settingsCategory(for:)``) be pinned exhaustively by a
    /// test, so a new case cannot be added without a stated destination.
    ///
    /// **Removed:** `openPermissions`. It titled a button "Open Permissions"
    /// and had nowhere to send it — this app's ``SettingsView/Category`` set
    /// is `models, modelManagement, tools, appearance, privacy, app`, and none
    /// of those holds a folder-grant or file-access control (Settings →
    /// Privacy is telemetry consent). Nor were the conditions that produced it
    /// reachable: it was emitted only for ``Kind/commandPermissionDenied`` and
    /// ``Kind/filePermissionDenied``, which ``FailureDiagnoser/toolFailureKind``
    /// derives only for the tool names `run_command`, `read_file`,
    /// `list_directory`, `write_file`, and `edit_file` — none of which this
    /// build ships (see ``BuiltinToolRegistry``: filesystem and shell tools are
    /// deliberately absent because there is no sandbox manager to gate them).
    /// Routing it to the "closest" tab would only have moved the dead end and
    /// kept a label that lies about what the user will find there, so the two
    /// kinds now carry no action at all. The ``Kind`` cases stay — they are
    /// persisted in transcripts and must keep decoding.
    enum Action: String, CaseIterable, Equatable, Sendable {
        case retry
        case restart
        case openModelManagement
        case switchDownloadSource
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
            case .openWebSearchSettings: return "Open Web Search Settings"
            }
        }

        var systemImage: String {
            switch self {
            case .retry, .restart: return "arrow.clockwise"
            case .openModelManagement: return "square.stack.3d.up"
            case .switchDownloadSource: return "arrow.triangle.2.circlepath"
            case .openWebSearchSettings: return "magnifyingglass"
            }
        }
    }

    let kind: Kind
    let message: String
    let action: Action?

    var severity: Severity { kind.severity }

    /// The recovery action a tool card may render inline, or nil for "render
    /// no button". Three gates, all load-bearing:
    ///
    ///   * **Errors only.** A ``Severity/notice`` is an outcome the USER chose
    ///     (they answered "no" to a permission prompt), so there is nothing for
    ///     the app to recover and no button to offer. Gating on severity rather
    ///     than on `action == nil` keeps the card's rule in one place: a notice
    ///     that ever gains an action still must not sprout a button here.
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
    /// The action switch is exhaustive with no `default` for the same reason
    /// ``Kind/severity`` is: a newly added action must state whether the tool
    /// card offers it, rather than being silently swallowed.
    ///
    /// Pure + static because the view that calls it is `private` inside
    /// ChatView and a SwiftUI body can't be exercised from the test suite.
    static func inlineToolCardAction(
        for diagnosis: FailureDiagnosis?,
        canRouteToSettings: Bool
    ) -> Action? {
        guard canRouteToSettings, let diagnosis, diagnosis.severity == .error else { return nil }
        switch diagnosis.action {
        case .openWebSearchSettings:
            return .openWebSearchSettings
        case .retry, .restart, .openModelManagement, .switchDownloadSource, .none:
            return nil
        }
    }
}

extension FailureDiagnosis.Kind {
    /// Deliberately an exhaustive switch with no `default`: a new kind must
    /// state whether it is a fault the app is reporting or an outcome the user
    /// chose, rather than inheriting the alarming lane by omission.
    var severity: FailureDiagnosis.Severity {
        switch self {
        // Both of these are the user's own answer, not a malfunction: one
        // declines a permission prompt, the other stops a transfer already
        // running. Painting either red tells somebody their own decision
        // broke something.
        case .userDeclined, .downloadCancelled:
            return .notice
        // A throttled backend is something that went wrong out in the world,
        // not something the user picked — it stays on the error lane, and its
        // copy and deep-link do the recovering.
        case .modelOutOfMemory, .modelLoadFailed, .engineNotRunning,
             .webSearchOffline, .webSearchUnavailable, .webSearchRateLimited,
             .webSearchKeyRejected, .webSearchKeyQuotaExceeded,
             .webSearchKeyRateLimited,
             .browsePageTooLarge,
             .commandPermissionDenied, .commandFailed,
             .fileNotFound, .filePermissionDenied, .toolFailed,
             .downloadFailed, .downloadSourceUnavailable, .requestFailed:
            return .error
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
        case .webSearchKeyRejected:
            message = "Keenable rejected this API key. Re-paste it in Settings → Tools, or clear it to use keyless search."
            action = .openWebSearchSettings
        case .webSearchKeyQuotaExceeded:
            message = "This Keenable key has used its monthly credits. Check its plan, or clear the key in Settings → Tools to use keyless search."
            action = .openWebSearchSettings
        case .webSearchKeyRateLimited:
            message = "This Keenable key is rate-limited. Wait a moment, check its plan, or clear the key in Settings → Tools to use keyless search."
            action = .openWebSearchSettings
        case .webSearchRateLimited:
            // Deliberately NOT "check its settings": everything in Settings is
            // already correct when this fires. DuckDuckGo rate-limits the free
            // endpoint per IP within a handful of searches, so the honest
            // remedy is a different backend, and the message has to say which
            // ones and where. Kept to one sentence of situation + one of
            // remedy so it still reads inside the tool card. Not Brave any
            // more (#2043): Brave requires a card on file now, so pitching it
            // as the free fix would steer the user into surprise billing.
            message = "DuckDuckGo is rate-limiting web searches from this Mac. Switch to Keenable (no key) or add a free Parallel or Tavily key in Settings → Tools."
            action = .openWebSearchSettings
        case .browsePageTooLarge:
            message = "This page is too large for Rapid to read at once. Search it or open a smaller page instead."
            action = nil
        case .commandPermissionDenied:
            // No action, and no "allow that folder, then try again" — this app
            // ships no shell tool and no place to grant a folder, so the old
            // copy asked for a control that does not exist and the old button
            // led nowhere. See ``FailureDiagnosis/Action``. What is left states
            // the outcome and stops.
            message = "The command tried to change a protected location, so it was blocked."
            action = nil
        case .commandFailed:
            message = "The command didn't finish successfully. Check the command, then try again."
            action = .retry
        case .fileNotFound:
            message = "That file isn't there. Check its name or location, then try again."
            action = .retry
        case .filePermissionDenied:
            // Same reasoning as ``commandPermissionDenied``: no file tools ship
            // here, so there is no access to grant and nothing to open.
            message = "Rapid doesn't have access to that file."
            action = nil
        case .toolFailed:
            message = "The tool couldn't finish. Check its input, then try again."
            action = .retry
        case .userDeclined:
            // Nothing went wrong, so the copy states the outcome and stops.
            // No action button: the app has nothing to fix and nothing to
            // recover, and a prominent "Retry" would offer to re-run the exact
            // thing the user just refused — one stray click away from the
            // permission prompt existing for nothing. A user who declined by
            // mistake simply asks again, and the transcript's own per-message
            // Retry is still right there for a mis-click.
            // "nothing to show" rather than "nothing happened": a redirect can
            // be declined after the approved page has already been fetched, so
            // the honest claim is about the result, not the whole exchange.
            message = "You didn't allow this, so there's nothing to show. Ask again if you change your mind."
            action = nil
        case .downloadFailed:
            message = "The model download didn't finish. Check your connection, then try again."
            action = .retry
        case .downloadCancelled:
            // States what the user did, what it left behind, and what the one
            // button will do — and stops. No connection advice (nothing was
            // wrong with the connection), and deliberately silent about
            // whether a fresh pull reuses bytes already on disk: that is a
            // property of the downloader, not a promise this app is in a
            // position to make. See Paper 05.1 state 10 — "No Pause and no
            // resume."
            message = "You stopped this download, so the model isn't installed. Download it again, or choose a different model."
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
    ///
    /// This is a FALLBACK for tools that report nothing but text. A tool that
    /// knows what happened says so directly by setting
    /// ``ToolCallResult/failureKind``, and the dispatch boundary
    /// (``BuiltinToolRegistry/run``) prefers that over anything inferred here.
    /// ``FailureDiagnosis/Kind/userDeclined`` in particular is ONLY ever
    /// producer-set: "the user said no" is a fact the approval gate holds, not
    /// something to re-derive by pattern-matching an English sentence that any
    /// tool could word differently tomorrow (or that a fetched page could
    /// contain verbatim).
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

        if toolName == "browse",
           raw.contains("page exceeded"),
           raw.contains("mb cap") {
            return .browsePageTooLarge
        }

        if toolName == "web_search" {
            // Keenable deliberately returns account failures as errors instead
            // of silently falling back to the shared keyless pool. Preserve
            // that precise cause at the UI boundary; collapsing all three to
            // ``webSearchUnavailable`` hid the only useful recovery action.
            if raw.contains("keenable rejected the api key") {
                return .webSearchKeyRejected
            }
            if raw.contains("keenable key's monthly credit allowance is used up") {
                return .webSearchKeyQuotaExceeded
            }
            if raw.contains("keenable rate limit hit for this api key") {
                return .webSearchKeyRateLimited
            }
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
