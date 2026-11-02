import Foundation
import Observation

/// Owns every ledger read, so the races between a request in flight and a key
/// that changed underneath it have exactly one place to be handled.
///
/// ## Why this is not in the view
///
/// The first implementation did the load inline in `ShareComputeView`, and that
/// shape cannot be made correct. Three races follow from it directly:
///
/// * A slow request started under key A lands after the user saved key B, and
///   its 401 deletes key B — the key that was never rejected.
/// * A request in flight when the user removes their key lands afterwards and
///   re-renders the ledger for a key that is gone.
/// * Two refreshes overlap and the slower one wins, showing older data.
///
/// All three are the same bug: a result being applied without asking whether
/// the world it was requested in still exists. The fix is a **generation
/// counter** bumped on every credential change, plus **single-flight** so
/// concurrent refreshes cannot exist in the first place.
///
/// ## The rules, in one place
///
/// 1. Every request captures `generation` at issue. Its result is applied only
///    if `generation` is still current. Anything else is dropped silently.
/// 2. A credential change bumps `generation` and cancels the in-flight task,
///    so a late result from the old world can never win.
/// 3. A rejected key is never auto-removed — it is this Mac's single
///    QuickSilver credential, managed from Settings → Experimental → Share
///    Compute; silently clearing it would also strand node registration, and
///    the user already has a place to replace it. See ``apply(error:)``.
/// 4. Manual refresh, the poll loop, and the credential-changed path all go
///    through ``refresh(reason:)``. There is no second path.
///
/// ## Credential resolution — ONE key
///
/// Owner decision: the user-facing QuickSilver credential is the `qsppk-`
/// provider key, entered once in Settings → Experimental → Share Compute,
/// and it is the ONLY credential this coordinator ever reads.
///
/// * Found → use it.
/// * Missing → the "add your key" onboarding state (``ShareComputeLedgerState/noKey``).
/// * Corrupted or unavailable → that error.
///
/// Rapid briefly kept a read-only `qsprk-` key working as a migration
/// fallback; the owner decision was to drop it outright rather than keep
/// maintaining its edge cases — see ``ShareComputeLegacyReadKeyCleanup``,
/// which this coordinator uses once at construction to clear out whatever is
/// left of it, and never reads again.

@MainActor
@Observable
final class ShareComputeLedgerCoordinator {

    /// Why a refresh was asked for. Decides only whether the gates below
    /// apply — never what gets requested.
    enum RefreshReason: Equatable, Sendable {
        /// Credits became visible.
        case appear
        /// The user pressed Refresh.
        case manual
        /// The slow background loop.
        case poll
        /// The provider key was just saved, replaced, or removed in
        /// Settings → Experimental → Share Compute; this is its first
        /// validation.
        case keyChanged

        /// Credential changes skip the five-minute INTERVAL: the user is
        /// waiting to find out whether the key they just entered works, and
        /// making them wait for that answer would be absurd.
        ///
        /// This does NOT skip the 429 cooldown for a REPLACEMENT — QuickSilver
        /// rate-limits by IP, not by credential, so a freshly pasted key still
        /// cannot buy a request; see ``refresh(reason:)``. A REMOVAL is the
        /// one case that bypasses everything, because it needs no request at
        /// all to answer.
        var bypassesInterval: Bool { self == .keyChanged }
    }

    // MARK: Published state

    private(set) var state: ShareComputeLedgerState = .loading
    /// `qsppk-…7f24`. Never the key.
    private(set) var savedKeyLabel: String?
    /// When the most recent request was ISSUED. Drives the interval gate.
    private(set) var lastRefreshStartedAt: Date?
    /// Set when the endpoint returns 429. Until it passes, Refresh is refused.
    private(set) var rateLimitedUntil: Date?

    // MARK: Collaborators

    /// The ONLY user-facing key: the `qsppk-` provider key.
    private let providerStore: ShareComputeProviderKeyStore
    private let load: @Sendable (ShareComputeProviderKey) async throws -> ShareComputeLedgerAccount
    private let now: @Sendable () -> Date
    /// How the wake task waits. Injected so tests can drive a gate expiry
    /// without sleeping five real minutes.
    private let sleep: @Sendable (TimeInterval) async -> Void
    /// A golden-mode fixture, resolved once at construction. When present no
    /// Keychain read and no request ever happens.
    private let fixture: ShareComputeLedgerFixture?

    // MARK: Race control

    /// Bumped on every credential change. A request whose captured value no
    /// longer matches is from a world that no longer exists.
    private var generation = 0
    /// Bumped on every issued request, so an out-of-order completion loses to
    /// the newer one even within a single generation.
    private var issue = 0
    /// The single in-flight walk, if any.
    private var inFlight: Task<Void, Never>?

    /// Why Refresh is unavailable, or ``open`` when it is.
    ///
    /// A STORED property, recomputed at every event that can change it. The
    /// first version derived this from `now()` inside a computed property, and
    /// that cannot work: `@Observable` sees mutations, and a wall-clock
    /// threshold passing is not one. SwiftUI would never re-evaluate the
    /// button, so it stayed greyed out until some unrelated redraw happened to
    /// rescue it. Storing the decision — and having the wake task write it —
    /// is what makes the control re-enable itself.
    enum RefreshGate: Equatable, Sendable {
        case open
        /// A walk is running.
        case busy
        /// Inside the minimum interval; lifts at this instant.
        case interval(until: Date)
        /// QuickSilver returned 429; lifts at this instant.
        case rateLimited(until: Date)
    }

    private(set) var refreshGate: RefreshGate = .open

    /// A REPLACEMENT provider key was resolved while the 429 cooldown was
    /// still running, so loading it is owed once the cooldown lifts. Never
    /// set for a removal — `.missing` needs no request to show `.noKey`, so
    /// there is nothing to defer and this is explicitly cleared instead; see
    /// ``refresh(reason:)``.
    private var pendingKeyValidation = false
    /// The key changed while Credits was hidden: the next ``appear`` must
    /// re-resolve the credential instead of honouring the refresh interval.
    private var keyChangedWhileHidden = false

    /// Cancellable holder for the single wake task.
    ///
    /// A box rather than a stored `Task` so `deinit` — which is nonisolated —
    /// can cancel it without hopping to the main actor.
    private final class WakeBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Void, Never>?
        private(set) var liveCount = 0

        func replace(with new: Task<Void, Never>?) {
            lock.lock()
            task?.cancel()
            if task != nil { liveCount -= 1 }
            task = new
            if new != nil { liveCount += 1 }
            lock.unlock()
        }

        func cancel() { replace(with: nil) }

        /// Releases the slot WITHOUT cancelling — for the task that is itself
        /// finishing. Cancelling the task you are currently running inside is
        /// a good way to poison work it goes on to start.
        func release() {
            lock.lock()
            if task != nil { liveCount -= 1 }
            task = nil
            lock.unlock()
        }

        var isScheduled: Bool {
            lock.lock(); defer { lock.unlock() }
            return task != nil
        }
    }

    private let wake = WakeBox()

    /// `keychain` backs the provider-key store, matching production: the
    /// real app has exactly one system Keychain. A test injects an
    /// `InMemoryKeychain`.
    init(
        keychain: any KeychainStoring = SystemKeychain(),
        load: (@Sendable (ShareComputeProviderKey) async throws -> ShareComputeLedgerAccount)? = nil,
        now: @escaping @Sendable () -> Date = { Date() },
        sleep: (@Sendable (TimeInterval) async -> Void)? = nil,
        fixture: ShareComputeLedgerFixture? = ShareComputeLedgerFixture.requested()
    ) {
        self.providerStore = ShareComputeProviderKeyStore(keychain: keychain)
        self.load = load ?? { try await ShareComputeLedgerClient().account(key: $0) }
        self.now = now
        self.sleep = sleep ?? { seconds in
            try? await Task.sleep(for: .seconds(max(0, seconds)))
        }
        self.fixture = fixture
        if let fixture {
            state = fixture.state
            savedKeyLabel = fixture.savedKeyLabel
        } else {
            // One-shot, best-effort: golden mode never touches the Keychain
            // at all (see the `fixture` branch above and in `refresh`), so
            // this only runs on the real path. The result is deliberately
            // ignored — see ``ShareComputeLegacyReadKeyCleanup``.
            ShareComputeLegacyReadKeyCleanup.run(keychain: keychain)
        }
    }

    deinit {
        // Nonisolated, so the box is what makes this legal.
        wake.cancel()
    }

    // MARK: - Gates

    /// Whether a walk is running right now. One at a time: a second Refresh
    /// click must not start a second account walk (which would be N more
    /// requests against a 120/hour budget for identical data).
    var isBusy: Bool { inFlight != nil }

    /// Read-only completion probe for tests that control an in-flight loader.
    var refreshCompletion: Task<Void, Never>? { inFlight }

    /// Whether the endpoint has rate-limited us and the cooldown is unexpired.
    var isRateLimited: Bool {
        guard let rateLimitedUntil else { return false }
        return now() < rateLimitedUntil
    }

    /// Whether the Refresh control should be enabled.
    ///
    /// False while a walk runs, while rate-limited, and inside the minimum
    /// interval — so the button cannot be clicked into a burst.
    /// Reads the STORED gate — which is what makes a wake-task write visible
    /// to `@Observable` — and then checks the deadline against the clock.
    ///
    /// Both halves are needed. Storing alone would leave the button wrong if a
    /// wake were ever late or missed (app suspended, timer cancelled while
    /// Credits was hidden). Computing alone would be invisible to SwiftUI,
    /// which is the bug this whole mechanism exists to fix. Reading
    /// `refreshGate` registers the dependency; comparing to `now()` keeps the
    /// answer honest.
    var allowsManualRefresh: Bool {
        switch refreshGate {
        case .open: return true
        case .busy: return false
        case .interval(let until), .rateLimited(let until): return now() >= until
        }
    }

    private var intervalElapsed: Bool {
        guard let lastRefreshStartedAt else { return true }
        return now().timeIntervalSince(lastRefreshStartedAt)
            >= ShareComputeLedgerRefresh.minimumInterval
    }

    /// True only until the FIRST request is issued.
    ///
    /// Deliberately keyed on ``lastRefreshStartedAt``, not on whether there is
    /// data on screen. The `state.account == nil` version was wrong in a way
    /// that mattered: a first attempt that 429'd, 503'd, returned a malformed
    /// body, or was cancelled leaves no account — so every subsequent tab
    /// switch looked like a fresh first entry and re-requested immediately,
    /// which is precisely the burst the limiter exists to stop.
    private var isFirstAttempt: Bool { lastRefreshStartedAt == nil }

    /// Whether a wake task is currently armed. For tests.
    var hasScheduledWake: Bool { wake.isScheduled }

    /// How many wake tasks are alive. Must never exceed 1.
    var liveWakeTaskCount: Int { wake.liveCount }

    /// A quiet line for a Refresh that is currently refused, so a disabled
    /// button is never unexplained.
    var refreshHoldNote: String? {
        // Same rule as `allowsManualRefresh`: an expired deadline holds
        // nothing, even if the wake has not run yet.
        guard !allowsManualRefresh else { return nil }
        switch refreshGate {
        case .open, .busy:
            return nil
        case .rateLimited:
            return String(localized: "QuickSilver is rate-limiting ledger checks. Refresh is paused for a few minutes.")
        case .interval:
            return String(localized: "Recently refreshed. The ledger updates again shortly.")
        }
    }

    /// Recomputes the stored gate from the clock and the current bookkeeping.
    ///
    /// Called at every event that can change it: request start, request
    /// completion, a wake firing, a removal. Assignment is what publishes.
    private func updateGate() {
        if inFlight != nil {
            refreshGate = .busy
            return
        }
        if let rateLimitedUntil, now() < rateLimitedUntil {
            refreshGate = .rateLimited(until: rateLimitedUntil)
            return
        }
        if let lastRefreshStartedAt {
            let next = lastRefreshStartedAt
                .addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval)
            if now() < next {
                refreshGate = .interval(until: next)
                return
            }
        }
        refreshGate = .open
    }

    // MARK: - Entry points

    /// Credits became visible.
    ///
    /// Immediate when there is nothing on screen — the first entry must not sit
    /// blank for five minutes. Once data exists, toggling tabs is subject to
    /// the interval like any other refresh, so tab-flipping cannot be used to
    /// bypass it.
    func appear() {
        if keyChangedWhileHidden {
            keyChangedWhileHidden = false
            refresh(reason: .keyChanged)
            return
        }
        refresh(reason: .appear)
    }

    func manualRefresh() {
        refresh(reason: .manual)
    }

    /// The slow loop. Returns when the enclosing task is cancelled, and
    /// cancels any walk still running as it goes.
    func poll() async {
        defer { cancelInFlight() }
        while !Task.isCancelled {
            refresh(reason: .poll)
            do {
                try await Task.sleep(
                    for: .seconds(ShareComputeLedgerRefresh.recommendedInterval)
                )
            } catch {
                return
            }
        }
    }

    /// Cancels the in-flight walk AND the wake timer.
    ///
    /// Called when Credits stops being visible: an invisible tab has no button
    /// to re-enable, so keeping a timer alive would be a wakeup for nobody.
    /// Re-entering re-arms it through ``appear``.
    func cancelInFlight() {
        inFlight?.cancel()
        inFlight = nil
        wake.cancel()
        updateGate()
    }

    // MARK: - Credential changes

    /// The ONE entry point the UI calls: the provider key in Settings →
    /// Experimental → Share Compute changed (saved, replaced, or removed).
    /// Re-resolves the credential immediately, bypassing the five-minute
    /// interval — the user is waiting to learn whether their change worked.
    ///
    /// * REMOVAL (`.missing`) bypasses the 429 cooldown too: it needs no
    ///   request at all, so it resolves straight to
    ///   ``ShareComputeLedgerState/noKey``, dropping whatever ledger and
    ///   label were on screen, and cancels any cooldown wake that was armed.
    /// * REPLACEMENT (`.found`) still respects an active 429 cooldown —
    ///   QuickSilver rate-limits by IP, not by credential — and is deferred
    ///   until it lifts; see ``refresh(reason:)`` and ``gateLifted()``.
    ///
    /// ``ShareComputeManager`` owns the actual Keychain write; this just
    /// tells the ledger its world changed. The generation bump happens
    /// before anything else, so a result still travelling from the OLD
    /// credential — in flight, or a late 401 — can never land after this.
    ///
    /// When Credits is not on screen (`isVisible == false`) nothing is
    /// requested: the old world is still invalidated, and the next
    /// ``appear`` performs the key-change load instead.
    func providerKeyChanged(isVisible: Bool = true) {
        invalidate()
        guard isVisible else {
            keyChangedWhileHidden = true
            return
        }
        keyChangedWhileHidden = false
        refresh(reason: .keyChanged)
    }

    /// Ends the current world: nothing issued before this point may be applied.
    private func invalidate() {
        generation &+= 1
        cancelInFlight()
    }

    // MARK: - Wake timer

    /// The next moment a time gate lifts, or `nil` when none is holding.
    private var nextGateExpiry: Date? {
        let instant = now()
        var candidates: [Date] = []
        if let rateLimitedUntil, instant < rateLimitedUntil {
            candidates.append(rateLimitedUntil)
        }
        if let lastRefreshStartedAt {
            let next = lastRefreshStartedAt
                .addingTimeInterval(ShareComputeLedgerRefresh.minimumInterval)
            if instant < next { candidates.append(next) }
        }
        return candidates.min()
    }

    /// Arms a single timer for the nearest gate expiry.
    ///
    /// Always replaces any existing timer rather than adding one, so repeated
    /// tab switches and repeated saves cannot accumulate a pile of wakeups all
    /// firing at once.
    private func scheduleWake() {
        guard let deadline = nextGateExpiry else {
            wake.cancel()
            return
        }
        let delay = max(0, deadline.timeIntervalSince(now()))
        let sleep = self.sleep
        wake.replace(with: Task { [weak self] in
            await sleep(delay)
            guard !Task.isCancelled else { return }
            await self?.gateLifted()
        })
    }

    /// A time gate expired. Publishes an observable change so the Refresh
    /// button re-enables itself, then performs a REPLACEMENT load that was
    /// owed to a key resolved during the cooldown — see ``refresh(reason:)``.
    /// Another gate may still be holding (e.g. the cooldown lifted but the
    /// interval has not) — re-arm for that one.
    private func gateLifted() {
        // `release`, not `cancel`: this runs INSIDE the wake task, and
        // cancelling it here would poison a refresh it might start next.
        wake.release()
        // The observable write. This is the whole reason the timer exists.
        updateGate()

        if pendingKeyValidation, !isRateLimited {
            pendingKeyValidation = false
            refresh(reason: .keyChanged)
            return
        }
        scheduleWake()
    }

    // MARK: - The one request path

    private func refresh(reason: RefreshReason) {
        if let fixture {
            // Golden mode: no Keychain, no request, no network.
            state = fixture.state
            savedKeyLabel = fixture.savedKeyLabel
            return
        }

        // Single-flight. A second Refresh while a walk runs is a no-op, not a
        // second walk.
        guard inFlight == nil else { return }

        // GATE 1 — the 429 cooldown. Blocks appear/manual/poll outright.
        // `keyChanged` is let through here regardless: a REMOVAL needs no
        // request at all to answer (see `.missing` below), and whether a
        // REPLACEMENT must wait for the cooldown is decided in the `.found`
        // case below it, once the credential is actually known.
        if reason != .keyChanged, isRateLimited {
            scheduleWake()
            return
        }

        // GATE 2 — the minimum interval. `keyChanged` bypasses it, and so
        // does the genuine first attempt (nothing has ever been requested,
        // so a blank tab must not sit for five minutes).
        if !reason.bypassesInterval && !isFirstAttempt {
            guard intervalElapsed else {
                scheduleWake()
                return
            }
        }

        // Credential resolution — ONE key, see the type-level note.
        switch providerStore.load() {
        case .found(let key):
            savedKeyLabel = key.redactedLabel
            if reason == .keyChanged, isRateLimited {
                // A replacement resolved while the 429 cooldown is still
                // running: QuickSilver rate-limits by IP, not by credential,
                // so spending another request right now buys nothing. Drop the
                // previous key's account so its credits never sit under the
                // new key's label, show the cooldown, and let `gateLifted()`
                // load this key once the cooldown actually lifts.
                state = .rateLimited
                pendingKeyValidation = true
                scheduleWake()
                return
            }
            // Either not rate-limited, or this call IS the owed validation
            // itself — either way nothing is left pending.
            pendingKeyValidation = false
            // A new credential starts from a blank ledger: the previous key's
            // rows must never sit under this key's label, even if it fails.
            if reason == .keyChanged { state = .loading }
            start(key: key)
        case .missing:
            savedKeyLabel = nil
            state = .noKey
            // Nothing is owed to a credential that is gone — drop any
            // cooldown wake a pending REPLACEMENT (or an ordinary 429) had
            // armed, so it never fires for a key that no longer exists.
            pendingKeyValidation = false
            wake.cancel()
        case .unavailable:
            state = .unavailable(.keychainUnavailable)
            pendingKeyValidation = false
        case .corrupted:
            // There IS a saved key — it just doesn't decode as one. Claiming
            // `.noKey` here would mask a broken credential behind the wrong
            // onboarding message.
            savedKeyLabel = nil
            state = .unavailable(.providerKeyCorrupted)
            pendingKeyValidation = false
        }
    }

    private func start(key: ShareComputeProviderKey) {
        generationSafeStart(key: key, generation: generation)
    }

    private func generationSafeStart(
        key: ShareComputeProviderKey,
        generation captured: Int
    ) {
        issue &+= 1
        let captuedIssue = issue
        lastRefreshStartedAt = now()
        updateGate()
        state = state.beginningLoad()

        inFlight = Task { [weak self] in
            guard let self else { return }
            do {
                let account = try await self.load(key)
                await self.finish(captured, captuedIssue) { coordinator in
                    coordinator.state = .loadedState(account)
                }
            } catch is CancellationError {
                await self.clearInFlight(captuedIssue)
            } catch let error as ShareComputeLedgerError {
                await self.finish(captured, captuedIssue) { coordinator in
                    coordinator.apply(error: error)
                }
            } catch {
                await self.finish(captured, captuedIssue) { coordinator in
                    coordinator.state = coordinator.state
                        .failing(.unreachable(error.localizedDescription))
                }
            }
        }
    }

    /// Whether a completed request may be applied.
    ///
    /// Extracted as a pure function because it is THE rule this type exists to
    /// enforce, and an inline `guard` chain inside a detached task is close to
    /// untestable. Three independent ways to lose:
    ///
    /// * **cancelled** — Credits stopped being visible, or a credential change
    ///   tore the request down. In practice this is the barrier that fires
    ///   most often, because ``invalidate`` cancels before it bumps anything.
    /// * **stale generation** — a key was saved or removed since the request
    ///   was issued, so its result describes a credential that is no longer in
    ///   play. Defence in depth behind cancellation, and the guard that would
    ///   still hold if a future path mutated the key without cancelling.
    /// * **superseded issue** — a newer request exists. Ordering, not
    ///   credentials: a slow response must never overwrite a faster newer one.
    static func acceptsResult(
        capturedGeneration: Int,
        currentGeneration: Int,
        capturedIssue: Int,
        currentIssue: Int,
        isCancelled: Bool
    ) -> Bool {
        !isCancelled
            && capturedGeneration == currentGeneration
            && capturedIssue == currentIssue
    }

    /// Applies a result only if the world it was requested in is still current.
    private func finish(
        _ captured: Int,
        _ capturedIssue: Int,
        _ apply: (ShareComputeLedgerCoordinator) -> Void
    ) {
        defer { clearInFlight(capturedIssue) }
        guard Self.acceptsResult(
            capturedGeneration: captured,
            currentGeneration: generation,
            capturedIssue: capturedIssue,
            currentIssue: issue,
            isCancelled: Task.isCancelled
        ) else { return }
        apply(self)
        // A request just completed, so the interval gate (and possibly a fresh
        // 429 cooldown) is now holding. Publish it and arm the timer that will
        // lift it — otherwise Refresh stays greyed out until something
        // unrelated redraws the view.
        updateGate()
        scheduleWake()
    }

    /// Releases the single-flight slot, but only if it still belongs to this
    /// request — a newer one may already own it.
    private func clearInFlight(_ capturedIssue: Int) {
        guard capturedIssue == issue else { return }
        inFlight = nil
        updateGate()
    }

    private func apply(error: ShareComputeLedgerError) {
        if error == .rateLimited {
            // Hold the Refresh control down for a cooldown, and keep whatever
            // is on screen: a 429 says nothing about the data's validity.
            rateLimitedUntil = now().addingTimeInterval(
                ShareComputeLedgerRefresh.rateLimitCooldown
            )
        }
        if error == .unauthorized {
            // Never auto-removed: it is this Mac's single QuickSilver
            // credential, managed from Settings → Experimental → Share
            // Compute; silently clearing it would also strand node
            // registration, and the user already has a place to replace it.
            savedKeyLabel = nil
            state = .unauthorized
            return
        }
        state = state.failing(error)
    }
}
