import Foundation

/// The single authoritative answer to two questions the app was
/// previously answering in four different places, inconsistently:
///
///   1. Can the user send a message right now?
///   2. If not — what is happening, and what should they do about it?
///
/// ## Why this type exists
///
/// Before this, readiness was re-derived per surface and the surfaces
/// disagreed. ``ChatView`` decided the empty-state subtitle from
/// ``ServerState`` alone; the composer's Send button consulted only
/// whether the draft was non-empty; ``ConnectToolsView`` rendered *two*
/// readiness sentences at once with two different verbs ("start a chat"
/// in the header, "Start a model" in the body); and the model picker
/// showed a cache glyph that said nothing about whether anything was
/// running. A user could see "Choose a model", a bright enabled Send
/// button, and — on pressing it — a transcript row reading
/// "Couldn't start ." with an empty model name.
///
/// ``ModelReadiness`` collapses all of that into one value derived once
/// per render from one set of inputs. Every surface that talks about
/// readiness reads its copy off this type, so they cannot drift.
///
/// ## The lifecycle it models
///
///     choose  →  download (if needed)  →  start  →  ready  →  send
///
/// The cases below are exactly the user-visible steps of that sequence,
/// plus the two ways out of it (``failed``, ``engineMissing``). Note
/// that ``needsDownload`` and ``needsStart`` are distinct: "chosen but
/// not on disk" and "on disk but not running" are different situations
/// with different costs (minutes vs seconds) and different copy, and
/// conflating them is most of why users could not tell choosing from
/// downloading from starting.
///
/// ## Sending is gated on ``ready``
///
/// ``sendAllowed`` is true only in ``ready``. This replaces the previous
/// implicit contract where pressing Send on a cold model silently
/// triggered a multi-gigabyte download behind an indeterminate spinner.
/// The trade is deliberate: one extra click (the banner's Start action)
/// buys the user a visible, cancellable, explicable startup. The draft
/// is never consumed by a gated send — see ``ChatView.send``.
///
/// Pure and free of SwiftUI so the whole truth table is unit-testable
/// without standing up a view host or a live subprocess.
enum ModelReadiness: Equatable {
    /// The engine binary is missing — setup never finished. Terminal
    /// until the user reinstalls; ``ContentView`` owns this screen.
    case engineMissing
    /// No model chosen yet (empty alias, or an internal placeholder
    /// that ``ModelDisplayName`` refuses to treat as a name).
    case noModel
    /// A real alias is chosen but its weights are not on disk.
    case needsDownload(alias: String, sizeText: String?)
    /// A real alias is chosen and cached, but nothing is serving it.
    case needsStart(alias: String)
    /// Weights are being pulled. ``detail`` carries bytes/speed/ETA when
    /// the byte monitor has a signal; ``fraction`` drives a determinate bar.
    case downloading(alias: String, detail: String?, fraction: Double?)
    /// Weights are on disk and the child is loading them into Metal.
    case starting(alias: String, detail: String?)
    /// Serving. The only state in which ``sendAllowed`` is true.
    case ready(alias: String)
    /// The last start or turn failed. ``action`` is the recovery step.
    case failed(alias: String?, message: String, action: Action?)

    // MARK: - Recovery / next-step actions

    /// The one concrete thing the user can do from a given state.
    ///
    /// ``chooseModel`` deliberately carries no button at the call site:
    /// the model picker sits ~40pt away in the composer and is already
    /// labelled "Choose a model", so rendering a second control with the
    /// same words would be a duplicate action. The case exists so the
    /// copy and the VoiceOver announcement can still name the step.
    enum Action: Equatable {
        case chooseModel
        case downloadAndStart(alias: String)
        case start(alias: String)
        case retry(alias: String)

        var title: String {
            switch self {
            case .chooseModel:      return "Choose a model"
            case .downloadAndStart: return "Download & start"
            case .start:            return "Start"
            case .retry:            return "Retry"
            }
        }

        var systemImage: String {
            switch self {
            case .chooseModel:      return "square.stack.3d.up"
            case .downloadAndStart: return "icloud.and.arrow.down"
            case .start:            return "play.fill"
            case .retry:            return "arrow.clockwise"
            }
        }

        /// True when the action should render as a real button. See the
        /// ``chooseModel`` note above.
        var isRenderable: Bool {
            if case .chooseModel = self { return false }
            return true
        }

        /// The alias this action operates on, when it has one.
        var alias: String? {
            switch self {
            case .chooseModel:
                return nil
            case .downloadAndStart(let a), .start(let a), .retry(let a):
                return a
            }
        }
    }

    /// Which status token the surface should paint. Mirrors the four
    /// roles ``ServerStatusPill`` already maps ``ServerState`` onto, so
    /// "amber means working, green means ready, red means broken" stays
    /// one fact about the app rather than a per-view convention.
    enum StatusRole: Equatable {
        case idle
        case working
        case ready
        case error
    }

    // MARK: - Inputs

    /// A pure snapshot of ``DownloadProgress`` taken at render time.
    /// Passing a snapshot rather than the `@Observable` object keeps
    /// ``resolve`` callable from tests and off the main actor.
    struct ProgressSnapshot: Equatable {
        var activity: DownloadProgress.StartupActivity
        var subtitle: String?
        var fraction: Double?

        init(
            activity: DownloadProgress.StartupActivity,
            subtitle: String? = nil,
            fraction: Double? = nil
        ) {
            self.activity = activity
            self.subtitle = subtitle
            self.fraction = fraction
        }
    }

    /// A chat-level failure worth surfacing as a readiness problem.
    /// Only consulted when the server is not itself in flight — an
    /// in-progress start always outranks a stale error from the turn
    /// that triggered it.
    struct Failure: Equatable {
        var message: String
        var kind: FailureDiagnosis.Kind?
        var alias: String?

        init(message: String, kind: FailureDiagnosis.Kind? = nil, alias: String? = nil) {
            self.message = message
            self.kind = kind
            self.alias = alias
        }
    }

    // MARK: - Resolution

    /// Derive readiness from live state.
    ///
    /// Precedence, highest first — the ordering is the contract:
    ///
    ///   1. ``engineMissing`` — nothing else is meaningful without a binary.
    ///   2. In-flight start (``downloading`` / ``starting``). Beats a
    ///      stale failure: the user pressed Retry and it is working.
    ///   3. ``ready``.
    ///   4. A crashed child, which carries its own diagnostic message.
    ///   5. A chat-level failure.
    ///   6. No model chosen.
    ///   7. Chosen but not cached → ``needsDownload``.
    ///   8. Otherwise → ``needsStart``.
    ///
    /// - Parameter isAliasCached: `nil` means the catalog has not loaded
    ///   yet. We resolve to ``needsStart`` in that case rather than
    ///   ``needsDownload``: claiming a download is required when we do
    ///   not know is the more misleading of the two errors, and the
    ///   Start action behaves identically either way (``ServerManager``
    ///   pulls on demand).
    static func resolve(
        serverState: ServerState,
        alias: String,
        isAliasCached: Bool?,
        sizeText: String? = nil,
        progress: ProgressSnapshot? = nil,
        failure: Failure? = nil
    ) -> ModelReadiness {
        if case .missing = serverState { return .engineMissing }

        if case .starting(let starting) = serverState {
            // Defensive fallback rather than `?? starting`: if BOTH the
            // serving alias and the picker's are placeholders, echoing
            // the raw string would render "Starting Loading" or, worse,
            // "Starting " with a trailing space. ``ModelDisplayName``
            // already uses this phrase for the same situation.
            let name = displayable(starting) ?? displayable(alias) ?? "your local model"
            let activity = progress?.activity ?? .starting
            if case .downloading = activity {
                return .downloading(
                    alias: name,
                    detail: progress?.subtitle,
                    fraction: progress?.fraction
                )
            }
            return .starting(alias: name, detail: startingDetail(for: activity, subtitle: progress?.subtitle))
        }

        if case .ready(let serving) = serverState, let name = displayable(serving) {
            return .ready(alias: name)
        }

        // A failure belongs to the model that FAILED, not to whatever the
        // picker holds now.
        //
        // Both branches below used to fire unconditionally, so once
        // `kimi-k2.6` crashed the banner was pinned to it: choosing
        // `bonsai-1.7b-2bit` — or anything else — kept rendering Kimi's
        // failure, its Retry, and its name in the placeholder and the
        // tooltip. The chat-level branch was worse than stale, it was
        // wrong: it read `failure.alias ?? alias`, so a failure with no
        // recorded alias got re-attributed to the newly chosen model and
        // accused it of an error it never had.
        //
        // Neither branch mutates ``ServerManager`` or drops the failed
        // turn from the transcript. The child really did crash and the
        // user really should still see that message in their history;
        // what changes is only whether the failure is still THIS
        // selection's problem.
        let selected = displayable(alias)

        if case .crashed(let crashed, let message) = serverState {
            let name = displayable(crashed)
            if failureApplies(failedAlias: name, selectedAlias: selected) {
                return .failed(
                    alias: name,
                    message: crashMessage(raw: message, alias: name),
                    action: name.map { ModelReadiness.Action.retry(alias: $0) }
                )
            }
        }

        if let failure {
            let name = displayable(failure.alias)
            if failureApplies(failedAlias: name, selectedAlias: selected) {
                return .failed(
                    alias: name,
                    message: failureMessage(failure),
                    action: name.map { ModelReadiness.Action.retry(alias: $0) }
                )
            }
        }

        guard let name = selected else { return .noModel }

        if isAliasCached == false {
            return .needsDownload(alias: name, sizeText: normalizedSize(sizeText))
        }
        return .needsStart(alias: name)
    }

    // MARK: - Derived presentation

    /// True only when a message can actually be sent right now.
    var sendAllowed: Bool {
        if case .ready = self { return true }
        return false
    }

    var isReady: Bool { sendAllowed }

    /// True when the state is a fault the user has to act on, as opposed
    /// to a step they have not taken yet or work already in flight.
    var isFailure: Bool {
        switch self {
        case .failed, .engineMissing: return true
        default:                      return false
        }
    }

    var statusRole: StatusRole {
        switch self {
        case .engineMissing, .failed:      return .error
        case .noModel, .needsDownload,
             .needsStart:                  return .idle
        case .downloading, .starting:      return .working
        case .ready:                       return .ready
        }
    }

    /// True while work is in flight, so a status dot knows to pulse.
    var isWorking: Bool {
        switch self {
        case .downloading, .starting: return true
        default:                      return false
        }
    }

    /// The alias this state is about, when it has one.
    var alias: String? {
        switch self {
        case .engineMissing, .noModel:
            return nil
        case .needsDownload(let a, _), .needsStart(let a),
             .downloading(let a, _, _), .starting(let a, _), .ready(let a):
            return a
        case .failed(let a, _, _):
            return a
        }
    }

    /// Determinate progress, when a fraction is genuinely known.
    var progressFraction: Double? {
        if case .downloading(_, _, let fraction) = self { return fraction }
        return nil
    }

    var action: Action? {
        switch self {
        case .engineMissing:                    return nil
        case .noModel:                          return .chooseModel
        case .needsDownload(let a, _):          return .downloadAndStart(alias: a)
        case .needsStart(let a):                return .start(alias: a)
        case .downloading, .starting, .ready:   return nil
        case .failed(_, _, let action):         return action
        }
    }

    // MARK: - Copy
    //
    // One vocabulary, used by every surface (item 4 of the Phase 1
    // brief): you CHOOSE a model, DOWNLOAD it if needed, START it, and
    // then it is READY. No surface may invent a fifth verb — the old
    // "start a chat to generate the key" in Connect Tools is exactly the
    // drift this section exists to prevent.

    /// Short status line — the bold half of the readiness banner.
    var headline: String {
        switch self {
        case .engineMissing:
            return "Setup didn't finish"
        case .noModel:
            return "No model chosen"
        case .needsDownload(let a, _):
            return "\(a) isn't downloaded yet"
        case .needsStart(let a):
            return "\(a) isn't running"
        case .downloading(let a, _, _):
            return "Downloading \(a)"
        case .starting(let a, _):
            return "Starting \(a)"
        case .ready(let a):
            return "Ready — \(a)"
        case .failed(let a, _, _):
            return a.map { "Couldn't start \($0)" } ?? "Something went wrong"
        }
    }

    /// The explanation under the headline. This is the "clearly explain
    /// why sending is unavailable" half of the contract.
    var detail: String? {
        switch self {
        case .engineMissing:
            return "Rapid-MLX can't find its engine. Reopen the app to run setup again."
        case .noModel:
            // Points at the picker rather than duplicating it as a button.
            return "Choose a model in the box below to get started."
        case .needsDownload(_, let sizeText):
            if let sizeText {
                return "It downloads once (\(sizeText)), then starts in seconds."
            }
            return "It downloads once, then starts in seconds."
        case .needsStart:
            return "It's already downloaded — starting takes a few seconds."
        case .downloading(_, let detail, _):
            return detail ?? "Starting the download…"
        case .starting(_, let detail):
            return detail ?? "Loading the model into memory…"
        case .ready:
            return nil
        case .failed(_, let message, _):
            return message
        }
    }

    /// Placeholder for the compose field. Terse — it names the blocking
    /// step rather than repeating the banner's full sentence, so the two
    /// are complementary instead of redundant.
    var composerPlaceholder: String {
        switch self {
        case .engineMissing:            return "Setup didn't finish"
        case .noModel:                  return "Choose a model first"
        case .needsDownload(let a, _):  return "Download \(a) first"
        case .needsStart(let a):        return "Start \(a) first"
        case .downloading(let a, _, _): return "Downloading \(a)…"
        case .starting(let a, _):       return "Starting \(a)…"
        case .ready:                    return "Send a message…"
        case .failed:                   return "Retry to continue"
        }
    }

    /// Send-button tooltip. Doubles as the VoiceOver announcement when a
    /// gated send is attempted, so both channels say the same thing.
    var sendTooltip: String {
        switch self {
        case .engineMissing:
            return "Rapid-MLX can't find its engine yet."
        case .noModel:
            return "Choose a model before sending."
        case .needsDownload(let a, _):
            return "Download \(a) before sending."
        case .needsStart(let a):
            return "Start \(a) before sending."
        case .downloading(let a, _, _):
            return "\(a) is still downloading."
        case .starting(let a, _):
            return "\(a) is still starting."
        case .ready:
            return "Send"
        case .failed(let a, _, _):
            return a.map { "\($0) isn't running — retry to continue." }
                ?? "Not ready to send yet."
        }
    }

    /// The line under "Ask anything" on the chat hero. Preserves the two
    /// strings the approved empty state already shipped ("Choose a model
    /// to start", "Chatting with <alias>") and extends the same voice to
    /// the states that previously had none.
    var emptyStateSubtitle: String {
        switch self {
        case .engineMissing:
            return "Setup didn't finish"
        case .noModel:
            return "Choose a model to start"
        case .needsDownload(let a, _):
            return "Download \(a) to start"
        case .needsStart(let a):
            return "Start \(a) to begin"
        case .downloading:
            return "Downloading your local model…"
        case .starting:
            return "Preparing your local model…"
        case .ready(let a):
            return "Chatting with \(a)"
        case .failed(let a, _, _):
            return a.map { "Couldn't start \($0)" } ?? "Something went wrong"
        }
    }

    /// The quieter third line on the hero. Nil whenever the subtitle
    /// already says everything — an empty state should not stack three
    /// sentences that mean the same thing.
    var emptyStateHint: String? {
        switch self {
        case .needsDownload(_, let sizeText):
            guard let sizeText else { return nil }
            return "First download is about \(sizeText)."
        case .downloading(_, let detail, _):
            return detail
        case .failed(_, let message, _):
            return message
        default:
            return nil
        }
    }

    /// Composed VoiceOver label for the readiness banner. Comma-joined
    /// tokens are how AppKit consumes a composed label.
    var accessibilityLabel: String {
        var parts = [headline]
        if let detail { parts.append(detail) }
        return parts.joined(separator: ", ")
    }

    /// Is a recorded failure still the CURRENT selection's problem?
    ///
    /// Three cases, and the asymmetry between the last two is the point:
    ///
    ///   * Nothing chosen — show the failure. It is the most useful thing
    ///     on screen, and there is no other model to describe instead.
    ///   * A model is chosen and the failure names a model — show it only
    ///     when they are the same model.
    ///   * A model is chosen and the failure names nothing — suppress it.
    ///     We cannot prove the failure is about this model, and blaming
    ///     the user's fresh pick for an unattributable error is the worse
    ///     of the two mistakes. The selection's own state is shown
    ///     instead, which is always true.
    ///
    /// ``static`` so the rule can be pinned directly by tests rather than
    /// only through the eight-case resolve matrix.
    static func failureApplies(failedAlias: String?, selectedAlias: String?) -> Bool {
        guard let selectedAlias else { return true }
        guard let failedAlias else { return false }
        return failedAlias == selectedAlias
    }

    // MARK: - Pure helpers

    /// A name we are willing to show a user, or `nil`. Routed through
    /// ``ModelDisplayName`` so an internal placeholder ("Loading",
    /// "Starting", "") can never be interpolated into copy as if it
    /// were a model — the defect behind "Couldn't start .".
    private static func displayable(_ alias: String?) -> String? {
        guard let alias else { return nil }
        return ModelDisplayName.configValue(alias: alias)
    }

    /// Treat an empty size string (``ModelSizing`` has no estimate for
    /// this alias) as absent rather than rendering "(  )".
    private static func normalizedSize(_ sizeText: String?) -> String? {
        guard let sizeText, !sizeText.trimmingCharacters(in: .whitespaces).isEmpty else {
            return nil
        }
        return sizeText
    }

    /// Copy for the ``starting`` detail line, keyed off the same
    /// ``StartupActivity`` the picker's own progress subtitle uses so
    /// the two can never claim different phases.
    private static func startingDetail(
        for activity: DownloadProgress.StartupActivity,
        subtitle: String?
    ) -> String? {
        switch activity {
        case .warmingUp:  return "Warming up…"
        case .loading:    return subtitle ?? "Loading the model into memory…"
        case .starting:   return subtitle ?? "Starting the model…"
        case .downloading: return subtitle
        }
    }

    /// A crashed child's raw message is engine output. Prefer the
    /// classified sentence; fall back to the raw text only when it is
    /// short enough to already read as prose.
    private static func crashMessage(raw: String, alias: String?) -> String {
        let kind = FailureDiagnoser.modelLoadFailureKind(raw: raw)
        return FailureDiagnoser.diagnosis(for: kind, modelAlias: alias).message
    }

    /// Prefer the structured diagnosis over the display string. This is
    /// what puts ``ChatViewModel.lastFailureKind`` to work — it was
    /// computed and discarded before.
    private static func failureMessage(_ failure: Failure) -> String {
        if let kind = failure.kind {
            return FailureDiagnoser.diagnosis(for: kind, modelAlias: failure.alias).message
        }
        return failure.message
    }
}
