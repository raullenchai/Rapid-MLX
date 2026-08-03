import Foundation
import Observation
import SwiftUI

/// First-launch single-button onboarding for brand-new users who have
/// no model on disk yet.
///
/// ## Why this surface exists
///
/// Production DMG ships with ``BUNDLE_MODEL=0`` (see ``scripts/build.sh``)
/// so the .app envelope contains zero model weights — the v0.7.1
/// bundled-snapshot path in ``BundledModel`` only fires for airgapped
/// builds. A brand-new user launches Rapid-MLX Desktop, lands on the
/// chat surface, and has nothing to chat with: the picker is a haystack,
/// every entry triggers a 1-80 GB cold download, the chat composer is
/// inert until something finishes loading. That is the worst possible
/// first-touch shape for an inference app.
///
/// Quickstart collapses the cold-start into one click. The card surfaces
/// when (a) the user has no last-served alias persisted AND (b) the
/// quickstart flag in UserDefaults hasn't been set yet AND (c) the
/// server isn't already busy with something else. Clicking "Get started"
/// triggers a ``rapid-mlx pull bonsai-1.7b-2bit`` (~0.5 GB) via the
/// existing ``DownloadManager``, then auto-spawns
/// ``rapid-mlx serve bonsai-1.7b-2bit`` once the pull is done, then
/// drops the user into chat with a single seeded assistant message
/// introducing the model.
///
/// ## Why bonsai-1.7b-2bit
///
/// Starter = Ternary-Bonsai 1.7B (mlx 2-bit; rapid-mlx alias
/// ``bonsai-1.7b-2bit``, PR #1092). This supersedes the earlier
/// ``qwen3-0.6b-4bit`` starter. History: the original ``qwen3.5-4b-4bit``
/// (~2.3 GB) cold-installed in ~11 minutes at the user's observed
/// 4.4 MB/s — an atrocious first-impression — so F-LWT-1 dropped to a
/// tiny ~400 MB 0.6B model purely for install latency, at the cost of
/// the tool-calls demo (0.6B can't reliably emit ``tool_calls``).
///
/// Bonsai removes that tradeoff. A ternary (1.58-bit) 1.7B checkpoint
/// weighs only ~0.5 GB — still a ~1-minute cold install on the same
/// link — yet it is a real Qwen3 that holds coherent multi-turn chat
/// AND emits clean ``tool_calls`` (6/6 in eval, hermes parser). The
/// brand-new user gets both halves on the very first model: "the app
/// works" AND "it can actually do things."
///
/// ### What this means for the empty state
///
///   1. Tool calls ARE reachable on the starter. ``ToolUseCapability``
///      lists ``bonsai-`` as ``.known`` (≥1.5 B floor), so the
///      empty-state capability chip row renders instead of being
///      hidden by the tool-bias gate (PR #333 + FU-9).
///   2. The ``ChatView`` empty-state ``examplePrompts`` stay
///      model-agnostic pure-text by design — they must read well on
///      ANY starter, not tease a capability tied to one alias — so
///      they are unchanged by this swap.
///   3. Users who want more depth trade up via the picker's
///      **Recommended Default** (``RAMBucketedDefault``, RAM-aware —
///      e.g. ``qwen3.5-9b-4bit`` on an 18 GB Mac), and the
///      ``UpgradeBanner`` nudges them there after a few turns.
///
/// ### What we keep
///
///   * One-click install + chat for the brand-new user, ~1-minute
///     cold install.
///   * Coherent pure-text chat + working tool calls on the starter.
///   * A dedicated "Quickstart" picker section (RAM-blind, persists
///     post-dismiss) so a user who skipped Quickstart can still
///     one-click install the demo model from the picker.
///
/// The alias resolves in ``vllm_mlx/aliases.json`` (rapid-mlx submodule)
/// so the value is pinned, not derived. Bumping it is a deliberate
/// product decision — change the constant + re-run the model
/// recommendation tests, and keep ``BundledModel.bundledAlias`` in
/// lock-step (the upgrade nudge keys on it).
///
/// ## Why a separate surface (not folded into ``ModelPickerBar``)
///
/// The picker is the long-tail browse-and-trade-up affordance. It
/// presents tens of aliases, makes the user reason about size/quant/
/// context, and is exactly the friction we want to spare first-touch
/// users. Quickstart is a parallel surface that replaces ONLY the
/// chat-area frame — the picker bar above stays visible so a user who
/// reads "or browse all models →" has the picker already in sight. The
/// card never returns once the user successfully Quickstarts or
/// manually picks something else from the picker (eligibility falls).
///
/// ## Why ``@Observable`` + a coordinator (not pure-view state)
///
/// The state machine outlives the view: a Quickstart download in
/// flight should survive a SwiftUI re-mount (e.g. main window
/// briefly hidden), and the persisted flag must be writable from
/// outside SwiftUI (tests, future Settings → "Reset onboarding"
/// affordance). Lifting the state into an ``@Observable`` coordinator
/// makes both shapes natural.

/// One selectable model in the Quickstart wizard's "choose your first
/// model" step (#1524). The wizard defaults to — and recommends — the
/// tiny 0.6B starter (the locked "0.6B first-run" decision), but lets the
/// user trade up to a bigger model before the first download.
///
/// ``hfRepo`` is pinned only for the starter (it wires the precise
/// bytes-on-disk monitor for the first-impression cold install — see the
/// ``kickoffDownload`` rationale). The bigger options pass ``nil`` and
/// fall back to tqdm file-count progress; both drive an identical
/// download → serve → seed pipeline.
struct QuickstartModelChoice: Equatable, Identifiable, Sendable {
    var id: String { alias }
    /// Canonical alias resolved in ``vllm_mlx/aliases.json``.
    let alias: String
    /// Prose label for onboarding copy ("Bonsai · 1.7B"). Hand-picked
    /// rather than catalog-derived so the copy never reads a raw alias.
    let displayName: String
    /// HF repo backing the byte monitor. Pinned for the starter; ``nil``
    /// for bigger options (tqdm-fallback progress is acceptable there).
    let hfRepo: String?
    /// One-line blurb shown under the name in the chooser.
    let blurb: String
    /// True for the recommended starter — the default selection, the
    /// "START HERE" badge, and the qualitative (meter-free) card.
    let isStarter: Bool
}

/// Persistent state owner + state machine for the Quickstart surface.
@MainActor
@Observable
final class QuickstartCoordinator {
    /// Phases the Quickstart UI walks through.
    enum Phase: Equatable {
        /// Initial state — the hero card is showing or the surface
        /// is hidden (we use ``QuickstartView.shouldShow`` to gate).
        case idle
        /// User clicked Get started, the pre-flight disk probe came
        /// back below ``DiskSpaceProbe.quickstartRequiredBytes``, and
        /// the card is showing the non-blocking low-disk warning with
        /// Continue + Cancel. Continue → ``.downloading``; Cancel →
        /// back to ``.idle``. See FU-4 / PR #338 review.
        case lowDiskWarning(freeBytes: Int64, requiredBytes: Int64)
        /// ``DownloadManager`` is pulling ``coordinator.selection.alias``
        /// in the background. The card swaps to an inline progress
        /// view; ``progressView`` reads ``DownloadManager.job(for:)``.
        case downloading
        /// Download finished, ``ServerManager.start`` is in flight.
        case starting
        /// Server is serving ``alias`` and the seeded assistant message
        /// has been appended to the active session. Quickstart hands off
        /// to the normal chat surface.
        case ready
        /// Download or serve failed. ``message`` is a single-line
        /// human-readable summary suitable for inline display.
        /// "Retry" is offered; the persistent done-flag is NOT set.
        case failed(message: String)
    }

    /// The default + recommended starter — the first-run decision.
    /// Pinned, not derived (F-LWT-1: ~11 min cold install of the old 4B
    /// pick was the wrong first-impression tradeoff; a small starter
    /// wins). 2026-07-10: swapped from ``qwen3-0.6b-4bit`` to the
    /// Ternary Bonsai 1.7B (rapid-mlx alias ``bonsai-1.7b-2bit``, PR
    /// #1092) — a ~0.5 GB ternary (1.58-bit) checkpoint that keeps the
    /// small-first-download win but, unlike the 0.6B, is multi-turn
    /// coherent AND emits clean ``tool_calls`` (6/6 on the eval
    /// harness). It therefore clears ``ToolUseCapability.known``, so the
    /// empty-state capability chips surface and the starter no longer
    /// has to hide its agentic surface. Users still trade up to a
    /// bigger model in the wizard or later via the picker.
    static let defaultChoice = QuickstartModelChoice(
        alias: "bonsai-1.7b-2bit",
        displayName: "Bonsai · 1.7B",
        hfRepo: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
        blurb: "Small download (~0.5 GB), runs on any Mac. Surprisingly capable — real chat and tool calls. Upgrade anytime for more depth.",
        isStarter: true
    )

    /// The curated onboarding ladder: the starter first (default
    /// selection), then a couple of bigger trade-ups. Deliberately a
    /// SHORT fixed list, not the RAM-bucketed recommendations — the
    /// wizard's job is "start small, one download"; the full RAM-aware
    /// catalog lives one tap away behind "Browse all models". The bigger
    /// options carry ``hfRepo: nil`` (tqdm-fallback progress is fine off
    /// the first-impression path); size + benchmark meters resolve from
    /// ``ModelSizing`` / ``BenchScoresCatalog`` at render.
    static let onboardingChoices: [QuickstartModelChoice] = [
        defaultChoice,
        QuickstartModelChoice(
            alias: "qwen3.5-4b-4bit",
            displayName: "Qwen 3.5 · 4B",
            hfRepo: nil,
            blurb: "Better everyday quality. Still light on disk.",
            isStarter: false
        ),
        QuickstartModelChoice(
            alias: "qwen3.5-9b-4bit",
            displayName: "Qwen 3.5 · 9B",
            hfRepo: nil,
            blurb: "Strong all-rounder if you have the RAM to spare.",
            isStarter: false
        ),
    ]

    /// UserDefaults key for the persistent "Quickstart already
    /// completed" flag. Once set, the surface NEVER returns — not
    /// even after the user deletes every model. Versioned so a
    /// future Quickstart refresh (e.g. introducing a different
    /// default model) can re-show without clobbering the v1 flag.
    static let storageKey: String = "rapid.quickstart.v1.done"

    /// Welcome message seeded into the active session after the sidecar
    /// comes online, so the user always lands in chat with a friendly
    /// intro rather than an empty transcript. Interpolates the chosen
    /// model's display name.
    ///
    /// The starter (bonsai-1.7b-2bit) is intentionally the smallest
    /// pick, so its copy keeps the "start in about a minute, trade up
    /// any time" framing. A bigger pick gets a plainer intro without
    /// the "smallest model" framing (it earned the trade-up, so don't
    /// undersell it).
    var seedMessage: String {
        if selection.isStarter {
            return """
You're chatting with \(selection.displayName) — our smallest model, picked so \
you can start chatting in about a minute. Open the picker any time to trade up \
to a larger model (the Recommended row is matched to your Mac's RAM).
"""
        }
        return """
You're chatting with \(selection.displayName), running entirely on your Mac. \
Open the picker any time to switch models.
"""
    }

    /// Current phase of the state machine.
    private(set) var phase: Phase = .idle

    /// Which model the wizard's "choose your first model" step has
    /// selected. Defaults to (and recommends) the starter; the chooser
    /// reassigns it via ``select(_:)`` before the download kicks off.
    /// Everything downstream (download, serve, seed, progress copy, the
    /// ContentView visibility gate's alias check) reads this instead of
    /// a pinned constant.
    private(set) var selection: QuickstartModelChoice = QuickstartCoordinator.defaultChoice

    /// Which pre-download wizard screen shows while ``phase`` is
    /// ``.idle``. Once the download kicks off, ``phase`` leaves ``.idle``
    /// and the download / starting / failed cards take over regardless of
    /// ``stage``. Orthogonal to the download-lifecycle machine.
    enum Stage: Equatable {
        /// The centered hero — brand, tagline, "Get started".
        case welcome
        /// The "choose your first model" step.
        case chooseModel
    }
    private(set) var stage: Stage = .welcome

    /// Advance from the hero to the model chooser ("Get started").
    func advanceToChooseModel() { stage = .chooseModel }

    /// Back out of the chooser to the hero ("Back").
    func backToWelcome() { stage = .welcome }

    /// Set the model the wizard will download. No-op once a download is
    /// in flight (``phase != .idle``) so a late tap can't retarget an
    /// active pull.
    func select(_ choice: QuickstartModelChoice) {
        guard case .idle = phase else { return }
        selection = choice
    }

    /// True once ``markDone`` has been called. Read on every eligibility
    /// check so the surface never returns. Mirrors UserDefaults.
    private(set) var done: Bool

    /// True once the seeded assistant message has been appended to the
    /// active session. Stops ``markReady`` from double-seeding when the
    /// observation pipeline fires multiple ``.ready`` transitions for
    /// the same start (auto-respawn cycle, scheduler tick, …).
    private(set) var hasSeededWelcome: Bool = false

    /// Codex r4 MAJOR: provenance flag for the deferred-seed retry path.
    /// Set ONLY when ``markReady`` was called from inside a real
    /// Quickstart flow whose seed returned ``false`` (no active session
    /// yet). The parent view's ``.onChange(of: store.activeID)``
    /// observer consults this flag before retrying — without it, the
    /// observer could fire a stray Quickstart welcome into a normal
    /// chat for a user who dismissed Quickstart and later picked
    /// gemma3-1b-qat-4bit manually.
    ///
    /// Cleared on:
    ///   * successful seed (markReady seed -> true)
    ///   * user-initiated revoke (releaseInFlight)
    ///   * a fresh Quickstart click (enterDownloading)
    ///   * server moves to a different alias (ContentView observer)
    ///   * test reset
    ///
    /// Codex r5 MAJOR: persisted to UserDefaults so a deferred-welcome
    /// flow survives quit-mid-flow. Without persistence, a user who
    /// reached Quickstart ``.ready`` but quit before ``activeID``
    /// landed would re-launch with ``ServerManager.lastServedAlias``
    /// already set to gemma3-1b-qat-4bit (so Quickstart eligibility
    /// falls), in-memory flag lost, welcome permanently skipped.
    private(set) var awaitingWelcomeSeed: Bool {
        didSet {
            UserDefaults.standard.set(awaitingWelcomeSeed, forKey: Self.awaitingSeedKey)
            // #1524: pin the alias the deferred seed is waiting on. Before
            // #1524 every comparison used the single pinned static, so a
            // quit-mid-flow relaunch trivially matched. Now the live
            // ``selection`` drives the seed target, but ``selection`` is
            // NOT persisted — a fresh ``QuickstartCoordinator()`` re-inits
            // it to ``defaultChoice`` (0.6B). Persisting the target alias
            // here (and restoring ``selection`` from it in ``init``) keeps
            // the ContentView seed observers comparing the served alias
            // against the model that was actually in flight, so a
            // non-default pick's welcome message survives the relaunch.
            if awaitingWelcomeSeed {
                UserDefaults.standard.set(selection.alias, forKey: Self.awaitingSeedAliasKey)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.awaitingSeedAliasKey)
            }
        }
    }

    /// UserDefaults key for the persistent ``awaitingWelcomeSeed``
    /// flag. Versioned alongside ``storageKey``.
    static let awaitingSeedKey: String = "rapid.quickstart.v1.awaitingSeed"

    /// UserDefaults key for the alias a persisted deferred seed is
    /// waiting on (#1524). Only meaningful while ``awaitingSeedKey`` is
    /// true; cleared in lockstep by the ``awaitingWelcomeSeed`` didSet.
    static let awaitingSeedAliasKey: String = "rapid.quickstart.v1.awaitingSeedAlias"

    init() {
        self.done = UserDefaults.standard.bool(forKey: Self.storageKey)
        // Codex r5: read the persisted awaiting-seed flag so a
        // quit-mid-deferred-flow relaunch can resume the welcome
        // injection once an active session lands. (Assigning a stored
        // property in ``init`` does NOT trigger the didSet, so this read
        // can't clobber the persisted alias below.)
        self.awaitingWelcomeSeed = UserDefaults.standard.bool(forKey: Self.awaitingSeedKey)
        // #1524: if a deferred seed survived a quit, restore the model it
        // was waiting on so the seed observers match the served alias and
        // the welcome copy names the right model (not the reset default).
        if self.awaitingWelcomeSeed,
           let alias = UserDefaults.standard.string(forKey: Self.awaitingSeedAliasKey) {
            self.selection = Self.choice(forAlias: alias)
        }
    }

    /// Resolve a wizard choice from a persisted alias — used to restore
    /// ``selection`` after a quit-mid-deferred-seed relaunch (#1524). The
    /// seed target is always one of ``onboardingChoices`` in the common
    /// (same-version) case; falls back to a minimal choice that still
    /// carries the alias so the seed comparison matches even if the
    /// onboarding ladder changed between the version that persisted and
    /// the version that restores.
    static func choice(forAlias alias: String) -> QuickstartModelChoice {
        if let match = onboardingChoices.first(where: { $0.alias == alias }) {
            return match
        }
        return QuickstartModelChoice(
            alias: alias,
            displayName: alias,
            hfRepo: nil,
            blurb: "",
            isStarter: alias == defaultChoice.alias
        )
    }

    /// Persist the "Quickstart already completed" flag and trip the
    /// in-memory mirror. Idempotent.
    func markDone() {
        done = true
        UserDefaults.standard.set(true, forKey: Self.storageKey)
    }

    /// Test-only reset so the suite can drive the state machine from
    /// scratch in every case without leaking flag state across runs.
    /// NOT exposed in any production UI; the design contract is that
    /// Quickstart is one-shot per Mac.
    internal func _testingReset() {
        done = false
        phase = .idle
        stage = .welcome
        selection = Self.defaultChoice
        hasSeededWelcome = false
        awaitingWelcomeSeed = false
        UserDefaults.standard.removeObject(forKey: Self.storageKey)
        UserDefaults.standard.removeObject(forKey: Self.awaitingSeedKey)
        UserDefaults.standard.removeObject(forKey: Self.awaitingSeedAliasKey)
    }

    /// External clearer for the awaiting-seed provenance flag. Called
    /// from ContentView's ``.onChange(of: server.state)`` observer
    /// when the server moves to a foreign alias while a deferred-seed
    /// is pending — without this, a user who reached the deferred-seed
    /// state then switched away then later switched back would get a
    /// stale welcome injected (codex r5 MODERATE).
    func clearPendingSeed() {
        awaitingWelcomeSeed = false
    }

    /// Mark the Quickstart download as in-flight. The card swaps to
    /// the progress view that reads ``DownloadManager`` directly.
    /// Clears any stale ``awaitingWelcomeSeed`` flag — a fresh user-
    /// initiated Quickstart click invalidates any pending-seed state
    /// from a prior aborted flow.
    func enterDownloading() {
        phase = .downloading
        awaitingWelcomeSeed = false
    }

    /// Surface the non-blocking low-disk warning between the hero card
    /// and the download kickoff. The user owns the "continue anyway"
    /// decision (per ``feedback_copy_mature_competitors`` — LM Studio /
    /// Ollama warn but never block). FU-4 / PR #338 review.
    func enterLowDiskWarning(freeBytes: Int64, requiredBytes: Int64) {
        phase = .lowDiskWarning(freeBytes: freeBytes, requiredBytes: requiredBytes)
    }

    /// User chose Cancel on the low-disk warning — return to the hero
    /// card so they can either close the window or click Get started
    /// again after freeing space. Distinct from ``enterFailed`` because
    /// this isn't a failure shape — the download never started.
    func cancelLowDiskWarning() {
        phase = .idle
    }

    /// Mark the serve transition (called once the pull lands and we
    /// hand off to ``ServerManager.start``).
    func enterStarting() {
        phase = .starting
    }

    /// Record a terminal failure. Does NOT flip ``done`` so the next
    /// surface render shows Quickstart again (with "Retry" if the
    /// failure was the download, plain "Get started" otherwise).
    func enterFailed(message: String) {
        phase = .failed(message: message)
    }

    /// Release the in-flight phase WITHOUT seeding the welcome or
    /// flipping ``done``. Used when the user has revised their intent
    /// mid-flow — clicked a DIFFERENT model in the still-visible
    /// picker, server lands at ``.ready(other-alias)``. We don't treat
    /// this as a failure (the user got what they wanted), but we also
    /// don't pretend Quickstart finished (they never saw the welcome).
    /// Phase flips to ``.ready`` so the visibility predicate's
    /// in-flight gate releases and ChatView takes the frame.
    ///
    /// Also clears any stale ``awaitingWelcomeSeed`` flag so the
    /// parent's ``.onChange(of: activeID)`` retry observer doesn't
    /// fire a stray welcome message after the user revised intent.
    func releaseInFlight() {
        phase = .ready
        awaitingWelcomeSeed = false
    }

    /// Drop into ready state, seed the welcome assistant message into
    /// the active session exactly once, and persist the done flag.
    ///
    /// Idempotent on the second call: multiple ``.ready`` notifications
    /// (auto-respawn cycle, scheduler tick) flip ``done`` once and seed
    /// the message once. Returns ``true`` when the seed actually
    /// landed.
    ///
    /// Codex r2 MAJOR: ``seed`` returns ``true`` only when the welcome
    /// message actually landed in a session. The parent's seed closure
    /// short-circuits on ``store.activeID == nil`` (no active session
    /// available — possible during the brief window before
    /// ``SessionStore.awaitInitialLoad`` lands or if the user deleted
    /// every session mid-Quickstart). We must NOT mark the flow as
    /// done in that case: ``hasSeededWelcome`` would lock out the
    /// retry on the next ``.ready`` fire, and the persistent done flag
    /// would silently skip the welcome message forever.
    @discardableResult
    func markReady(seed: () -> Bool) -> Bool {
        phase = .ready
        if hasSeededWelcome {
            // Already done on a prior tick — re-affirm ``done`` so a
            // restored coordinator that lost the flag still persists
            // it, but don't try to seed again.
            markDone()
            awaitingWelcomeSeed = false
            return false
        }
        let seeded = seed()
        guard seeded else {
            // Welcome couldn't land — leave the door open for the
            // next ``.ready`` tick (or a retry after the user creates
            // a session). Phase still flips to ``.ready`` so the
            // visibility predicate's "in-flight" guard releases. Flag
            // the deferred-seed provenance (codex r4 MAJOR) so the
            // parent's activeID retry observer knows this is a real
            // pending seed (not a user who manually picked the
            // Quickstart alias from the picker after dismissing).
            awaitingWelcomeSeed = true
            return false
        }
        hasSeededWelcome = true
        markDone()
        awaitingWelcomeSeed = false
        return true
    }

    /// Pure eligibility predicate so the contract test can pin it
    /// without standing up SwiftUI environment or ``ServerManager``
    /// state in full. Returns ``true`` when the Quickstart card
    /// should render in place of the normal chat-or-overlay tree.
    ///
    /// "First run" is decided from state THIS app owns — never from
    /// the shared Hugging Face cache. #298 originally added a
    /// ``hasAnyCachedAlias`` gate that scanned ``~/.cache/huggingface
    /// /hub`` and suppressed Quickstart whenever ANY ``models--*``
    /// directory existed. That over-reached: the HF cache is shared
    /// across the whole MLX / transformers ecosystem, so a genuinely
    /// new user who merely had a Whisper / VAD / forced-aligner model
    /// from some other tool was denied onboarding and dumped into the
    /// raw picker — exactly the worst first-touch Quickstart exists to
    /// avoid. The gate is now app-owned only.
    ///
    /// Three gates, all must hold:
    ///   1. ``done == false`` — the persistent one-shot guard
    ///      (``rapid.quickstart.v1.done``). Set once the user completes
    ///      OR dismisses Quickstart, so the card never returns.
    ///   2. ``lastServedAlias == nil`` — our own "has this app ever
    ///      served a model?" signal (``rapid.serve.lastAlias``, written
    ///      by ``ServerManager`` on a successful serve). A user who
    ///      ever reached a running model — via Quickstart OR the picker
    ///      — is no longer new, so the card stays down.
    ///   3. ``serverState`` is ``.idle`` or ``.stopped`` — anything
    ///      else means a model is already engaged (``.ready`` /
    ///      ``.starting`` / ``.crashed``) or the install overlay is
    ///      already in charge (``.missing``).
    ///
    /// Both persisted signals live in ``UserDefaults`` and survive
    /// relaunch, reinstall, and Migration Assistant — the only way to
    /// re-trigger onboarding is to clear them (a deliberate developer
    /// ``defaults delete``), which is the correct semantics for "reset
    /// first-run", not something inferred from disk contents.
    static func isEligible(
        done: Bool,
        lastServedAlias: String?,
        serverState: ServerState
    ) -> Bool {
        guard !done else { return false }
        guard lastServedAlias == nil else { return false }
        switch serverState {
        case .idle, .stopped:
            return true
        case .ready, .starting, .crashed, .missing:
            return false
        }
    }
}

/// Hero card + post-click progress / failure states. Centered in the
/// main area; the parent view replaces ``mainArea`` with this when
/// ``QuickstartCoordinator`` reports the surface should show.
struct QuickstartView: View {
    @Environment(SettingsRouter.self) private var settingsRouter
    @Environment(\.openSettings) private var openSettings
    @Bindable var coordinator: QuickstartCoordinator
    @Bindable var downloads: DownloadManager
    @Bindable var server: ServerManager

    /// Callback the parent supplies for the "or browse all models →"
    /// link. The parent dismisses the Quickstart surface for the
    /// current session (without flipping the persisted flag) so the
    /// existing picker becomes visible. Lifted out as a closure so
    /// this view can stay agnostic of how the parent toggles its own
    /// state.
    var onBrowseAll: () -> Void

    /// Callback the parent supplies for seeding the welcome message
    /// into the active session. Closing over ``SessionStore`` /
    /// ``ChatViewModel`` from outside keeps Quickstart from importing
    /// the entire chat module surface. Returns ``true`` when the
    /// message actually landed (an active session existed and the
    /// append succeeded) so the coordinator can defer ``markDone``
    /// until the welcome has reached the user (codex r2 MAJOR).
    var onSeedWelcome: () -> Bool

    /// Test seam: override the pre-flight free-bytes probe so the
    /// unit / integration test suite can drive the low-disk-warning
    /// transition without touching real free space. Defaults to the
    /// production ``DiskSpaceProbe.freeBytesForHFCache`` helper.
    ///
    /// Returns the free bytes the caller should compare against
    /// ``DiskSpaceProbe.quickstartRequiredBytes``. ``nil`` means "probe
    /// failed / no signal" and degrades to ``Decision.ok`` (no warning).
    var freeBytesProbe: () -> Int64? = { DiskSpaceProbe.freeBytesForHFCache() }

    var body: some View {
        content
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(RapidTheme.canvas)
            // Observe serve transitions so we can flip to ``.ready`` (and
            // seed the welcome message) as soon as the sidecar comes
            // online. ``.task(id:)`` re-fires on every ``server.state``
            // change — the serve-side handoff is the only thing we need
            // to react to here; the download side is driven by the
            // explicit "Download & start" tap below.
            .task(id: server.state) {
                handleServerStateChange()
            }
            // Observe download-job transitions so the failed branch lights
            // up the inline "Retry" card and the completed branch hands off
            // to ``server.start``. ``.task(id:)`` re-fires when the job's
            // status enum changes — exactly the trigger shape we want.
            .task(id: downloadJobStatusKey) {
                handleDownloadStatusChange()
            }
    }

    /// Top-level router. While ``phase`` is ``.idle`` the wizard shows
    /// the ``stage``-driven welcome / choose-model steps (full pane);
    /// once a download is in flight the download-lifecycle cards take
    /// the frame (centered). The lifecycle machine itself is unchanged —
    /// only its idle branch now fans out to two wizard screens.
    @ViewBuilder
    private var content: some View {
        switch coordinator.phase {
        case .idle:
            switch coordinator.stage {
            case .welcome:     welcomeStep
            case .chooseModel: chooseModelStep
            }
        case .lowDiskWarning(let freeBytes, let requiredBytes):
            centeredCard(progressStep: nil) {
                lowDiskCard(freeBytes: freeBytes, requiredBytes: requiredBytes)
            }
        case .downloading:
            centeredCard(progressStep: 2) { downloadingCard }
        case .starting, .ready:
            // .ready is transitional — the parent swaps to ChatView, but
            // a one-frame race can land here; the starting copy is a calm
            // fallback so the user never sees a blank pane.
            centeredCard(progressStep: 2) { startingCard }
        case .failed(let message):
            centeredCard(progressStep: nil) { failedCard(message: message) }
        }
    }

    /// The centered card chrome shared by the download-lifecycle states.
    /// ``progressStep`` (0-indexed) shows the top progress bar on the
    /// happy path (download / starting); ``nil`` omits it for the
    /// low-disk / failed interstitials, where a "progress" bar would
    /// misread.
    ///
    /// The card caps at 460pt but SHRINKS on a narrow detail pane rather
    /// than overflowing — ``QuickstartView`` lives in the split view's
    /// detail column, so at the 640pt window floor with the sidebar
    /// visible the pane is only ~360pt (memory #459/#464: NavigationSplit
    /// detail clips instead of scrolling on macOS 14/15). ``maxWidth`` +
    /// the outer horizontal inset keeps the chrome inside the column.
    @ViewBuilder
    private func centeredCard<Content: View>(
        progressStep: Int?,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 0) {
            if let progressStep {
                OnboardingTopBar(step: progressStep)
                    .padding(.top, 22)
            }
            Spacer()
            VStack(spacing: 20) { content() }
                .padding(28)
                .frame(maxWidth: 460)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                        .fill(RapidTheme.card)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                        .stroke(RapidTheme.hairline, lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.06), radius: 18, x: 0, y: 8)
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Quickstart")
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
    }

    /// Stable key for ``.task(id:)`` so SwiftUI re-fires the handler on
    /// every job status transition. ``DownloadManager.Job.status`` is
    /// ``Equatable``; flattening it to a tag string keeps the task id
    /// ``Hashable`` without leaning on case payloads.
    private var downloadJobStatusKey: String {
        guard let job = downloads.job(for: coordinator.selection.alias) else {
            return "absent"
        }
        switch job.status {
        case .running:   return "running"
        case .completed: return "completed"
        case .cancelled: return "cancelled"
        case .failed(let message): return "failed:\(message)"
        }
    }

    // MARK: - Wizard steps (phase == .idle)

    /// Step 1 — the centered brand hero. "Get started" advances to the
    /// model chooser (it no longer kicks off a download directly; the
    /// download starts from the chooser once a model is picked).
    @ViewBuilder
    private var welcomeStep: some View {
        VStack(spacing: 0) {
            Spacer()
            OnboardingBrandMark(size: 78).padding(.bottom, 22)

            Text("WELCOME TO RAPID-MLX")
                .scaledSystemFont(11, weight: .semibold)
                .tracking(2.2)
                .foregroundStyle(.secondary)
                .padding(.bottom, 14)

            (Text("Fast, free AI\n")
             + Text("that runs on your Mac.").italic().foregroundColor(RapidTheme.brand))
                .scaledSystemFont(38, relativeTo: .largeTitle, weight: .bold)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, 16)

            VStack(spacing: 5) {
                Text("Local models on Apple Silicon — no account, no subscription.")
                Text("Download one model and start chatting in minutes.")
            }
            .scaledSystemFont(14)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)

            Spacer()

            Button {
                coordinator.advanceToChooseModel()
            } label: {
                Text("Get started")
                    .scaledSystemFont(15, weight: .semibold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 34).padding(.vertical, 12)
                    .background(Capsule().fill(RapidTheme.amber))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.defaultAction)
            .accessibilityIdentifier("Quickstart.GetStarted")
            .accessibilityLabel("Get started — choose your first model")
            .padding(.bottom, 10)

            // #549 (§16 wayfinding): the hero must answer "how do I get
            // out?" — before this the only exit was the "Browse all
            // models" link on step 2, trapping a first-run user sitting
            // on step 1. A low-emphasis Skip drops straight into the app
            // (same `onBrowseAll` dismiss path the chooser uses), and
            // `.cancelAction` makes Esc leave onboarding — mirroring the
            // Skip control OnboardingTour already ships.
            Button("Skip for now") {
                onBrowseAll()
            }
            .buttonStyle(.plain)
            .scaledSystemFont(12, weight: .medium)
            .foregroundStyle(.secondary)
            .keyboardShortcut(.cancelAction)
            .accessibilityIdentifier("Quickstart.Skip")
            .accessibilityLabel("Skip onboarding and go to the app")
            .padding(.bottom, 18)

            OnboardingStepDots(current: 0, total: 3).padding(.bottom, 34)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 44)
    }

    /// Step 2 — the model chooser: recommended starter (default
    /// selection) + bigger trade-ups + "Browse all models". The primary
    /// footer button kicks off the download for the current selection.
    @ViewBuilder
    private var chooseModelStep: some View {
        let choices = QuickstartCoordinator.onboardingChoices
        VStack(alignment: .leading, spacing: 0) {
            OnboardingTopBar(step: 1).padding(.top, 22)

            Text("Choose your first model")
                .scaledSystemFont(24, relativeTo: .title, weight: .bold)
                .padding(.top, 22)
            Text("Start small — you can download bigger models anytime in Settings.")
                .scaledSystemFont(13).foregroundStyle(.secondary)
                .padding(.top, 4).padding(.bottom, 18)

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(choices.filter { $0.isStarter }) { choice in
                        QuickstartRecommendedCard(
                            choice: choice,
                            selected: coordinator.selection.alias == choice.alias,
                            sizeText: Self.sizeText(for: choice.alias)
                        ) { coordinator.select(choice) }
                        .padding(.bottom, 16)
                    }

                    Text("OR PICK A BIGGER ONE")
                        .scaledSystemFont(10, weight: .semibold).tracking(1)
                        .foregroundStyle(.tertiary)
                        .padding(.bottom, 9)

                    VStack(spacing: 9) {
                        ForEach(choices.filter { !$0.isStarter }) { choice in
                            QuickstartCompactCard(
                                choice: choice,
                                selected: coordinator.selection.alias == choice.alias,
                                sizeText: Self.sizeText(for: choice.alias)
                            ) { coordinator.select(choice) }
                        }
                    }

                    Button {
                        onBrowseAll()
                    } label: {
                        Text("Browse all models →")
                            .scaledSystemFont(12, weight: .medium)
                            .foregroundStyle(RapidTheme.brand)
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 14)
                    .accessibilityIdentifier("Quickstart.BrowseAll")
                    .accessibilityLabel("Browse all models")
                }
            }

            OnboardingWizardFooter(
                primaryTitle: "Download & start",
                onBack: { coordinator.backToWelcome() },
                onPrimary: { startQuickstart() }
            )
            .padding(.top, 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Tighter than the welcome hero's 44pt inset: the chooser holds
        // rigid-column cards, and this step lives in the split-view detail
        // pane (~360pt at the 640pt window floor with the sidebar shown).
        // 24pt keeps the trade-up cards' names readable instead of
        // squeezed to a sliver (memory #459/#464).
        .padding(.horizontal, 24)
        .padding(.bottom, 26)
    }

    /// Human-readable download size for a choice card. MB under 1 GB
    /// (so the 0.6B reads "~370 MB", not "0.4 GB"), one-decimal GB above.
    /// Returns "" when ``ModelSizing`` has no estimate.
    static func sizeText(for alias: String) -> String {
        let gb = ModelSizing.estimate(alias: alias).weightsGB
        guard gb > 0 else { return "" }
        if gb < 1 {
            return "~\(Int((gb * 1024).rounded())) MB"
        }
        return String(format: "%.1f GB", gb)
    }

    // MARK: - Subviews

    @ViewBuilder
    private var downloadingCard: some View {
        let job = downloads.job(for: coordinator.selection.alias)
        let fraction = job?.progress.progressFraction
        let subtitle = QuickstartView.progressSubtitle(
            job: job,
            displayName: coordinator.selection.displayName
        )
        let eta = QuickstartView.etaCaption(job: job)

        if let fraction {
            ProgressView(value: fraction, total: 1.0)
                .progressViewStyle(.linear)
                .frame(maxWidth: 360)
        } else {
            ProgressView()
                .controlSize(.large)
        }
        Text("Downloading \(coordinator.selection.displayName)")
            .font(.title3.weight(.semibold))
        Text(subtitle)
            .font(.callout.monospacedDigit())
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
        if let eta {
            Text(eta)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        Text("You can keep this window open — we'll start chat as soon as the download finishes.")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .multilineTextAlignment(.center)
            .padding(.top, 4)
    }

    @ViewBuilder
    private var startingCard: some View {
        ProgressView()
            .controlSize(.large)
        Text("Starting \(coordinator.selection.displayName)…")
            .font(.title3.weight(.semibold))
        Text("Loading the model into Metal. Usually 5–15 seconds.")
            .font(.callout)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }

    /// Low-disk warning card. Non-blocking — the user can still
    /// proceed with Continue (per LM Studio / Ollama UX) or Cancel
    /// back to the hero card. Visual language matches ``failedCard``
    /// (amber tint + warning glyph) so the user reads it as a caution,
    /// not an error.
    @ViewBuilder
    private func lowDiskCard(freeBytes: Int64, requiredBytes: Int64) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(RapidTheme.amberTint)
                .frame(width: 60, height: 60)
            Image(systemName: "externaldrive.badge.exclamationmark")
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(RapidTheme.amberDeep)
        }
        .accessibilityHidden(true)

        VStack(spacing: 8) {
            Text("Low disk space")
                .font(.title3.weight(.semibold))
            Text(QuickstartView.lowDiskBannerBody(
                freeBytes: freeBytes,
                requiredBytes: requiredBytes,
                displayName: coordinator.selection.displayName
            ))
            .font(.callout)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(QuickstartView.lowDiskAccessibilityLabel(
            freeBytes: freeBytes,
            requiredBytes: requiredBytes,
            displayName: coordinator.selection.displayName
        ))

        Button {
            kickoffDownload()
        } label: {
            Text("Continue anyway")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 2)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .keyboardShortcut(.defaultAction)
        .accessibilityIdentifier("Quickstart.LowDisk.Continue")
        .accessibilityLabel("Continue download despite low disk space")

        Button {
            coordinator.cancelLowDiskWarning()
        } label: {
            Text("Cancel")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.large)
        .keyboardShortcut(.cancelAction)
        .accessibilityIdentifier("Quickstart.LowDisk.Cancel")
        .accessibilityLabel("Cancel — return to Quickstart without downloading")
    }

    @ViewBuilder
    private func failedCard(message: String) -> some View {
        Text("Quickstart didn't finish")
            .font(.title3.weight(.semibold))

        let job = downloads.job(for: coordinator.selection.alias)
        let kind: FailureDiagnosis.Kind = {
            if case .crashed(let alias, let serverMessage) = server.state,
               alias == coordinator.selection.alias {
                return FailureDiagnoser.modelLoadFailureKind(raw: serverMessage)
            }
            return job?.failureKind ?? FailureDiagnoser.downloadFailureKind(
                raw: message,
                usingMirror: job?.source != .huggingFace
            )
        }()
        let diagnosis = FailureDiagnoser.diagnosis(for: kind)
        FailureDiagnosisView(
            diagnosis: diagnosis,
            onAction: handleQuickstartFailureAction,
            isActionDisabled: server.isOperating,
            actionAccessibilityIdentifier: quickstartActionIdentifier(for: diagnosis.action)
        )

        Button {
            onBrowseAll()
        } label: {
            Text("or browse all models →")
                .font(.callout)
        }
        .buttonStyle(.borderless)
    }

    private func handleQuickstartFailureAction(_ action: FailureDiagnosis.Action) {
        switch action {
        case .switchDownloadSource:
            coordinator.enterDownloading()
            if downloads.job(for: coordinator.selection.alias) != nil {
                _ = downloads.retryDownload(
                    alias: coordinator.selection.alias,
                    source: .huggingFace
                )
            } else {
                _ = downloads.startDownload(
                    alias: coordinator.selection.alias,
                    hfPath: coordinator.selection.hfRepo,
                    source: .huggingFace
                )
            }
        case .retry:
            if downloads.job(for: coordinator.selection.alias) != nil {
                coordinator.enterDownloading()
                _ = downloads.retryDownload(alias: coordinator.selection.alias)
            } else {
                startQuickstart()
            }
        case .restart:
            coordinator.enterStarting()
            Task { await server.start(alias: coordinator.selection.alias) }
        case .openModelManagement:
            settingsRouter.requestedCategory = .modelManagement
            openSettings()
        case .openPermissions:
            // The minimal app has no Permissions tab; land on Models.
            settingsRouter.requestedCategory = .models
            openSettings()
        }
    }

    private func quickstartActionIdentifier(
        for action: FailureDiagnosis.Action?
    ) -> String? {
        switch action {
        case .retry: return "Quickstart.Retry"
        case .restart: return "Quickstart.Restart"
        case .openModelManagement: return "Quickstart.OpenModelManagement"
        case .switchDownloadSource: return "Quickstart.SwitchSource"
        case .openPermissions: return "Quickstart.OpenPermissions"
        case nil: return nil
        }
    }

    // MARK: - Actions

    /// Entry point bound to the hero card's "Get started" button AND
    /// the failed card's "Retry" button. Splits into a pre-flight
    /// disk probe (FU-4) and the real kickoff:
    ///
    ///   * Probe ``freeBytesProbe`` (defaults to the HF cache volume).
    ///   * Run ``DiskSpaceProbe.decide`` against
    ///     ``DiskSpaceProbe.quickstartRequiredBytes``.
    ///   * ``.ok`` → fire ``kickoffDownload`` directly.
    ///   * ``.warn`` → flip the coordinator to ``.lowDiskWarning`` and
    ///     let the user choose Continue / Cancel from the rendered
    ///     banner.
    ///
    /// Warn-only by design — see ``DiskSpaceProbe`` rationale and the
    /// ``feedback_copy_mature_competitors`` note.
    private func startQuickstart() {
        QuickstartView.applyPreflightDecision(
            decision: DiskSpaceProbe.decide(
                freeBytes: freeBytesProbe(),
                requiredBytes: DiskSpaceProbe.quickstartRequiredBytes
            ),
            coordinator: coordinator,
            onKickoff: { kickoffDownload() }
        )
    }

    /// Pure adapter mapping a ``DiskSpaceProbe.Decision`` onto the
    /// Quickstart coordinator + kickoff closure. Lifted out of
    /// ``startQuickstart`` so the unit suite can pin the
    /// "Continue must bypass the probe" + "warn flips into warning
    /// phase" contracts without standing up SwiftUI or
    /// ``DownloadManager`` (codex r1 MINOR — regression to
    /// re-probing inside Continue would silently reintroduce the
    /// warning loop the comment at ``kickoffDownload`` cautions
    /// against).
    @MainActor
    static func applyPreflightDecision(
        decision: DiskSpaceProbe.Decision,
        coordinator: QuickstartCoordinator,
        onKickoff: () -> Void
    ) {
        switch decision {
        case .ok:
            onKickoff()
        case .warn(let freeBytes, let requiredBytes):
            coordinator.enterLowDiskWarning(
                freeBytes: freeBytes,
                requiredBytes: requiredBytes
            )
        }
    }

    /// Actually fire the download. Split from ``startQuickstart`` so
    /// the low-disk warning card's "Continue anyway" button can
    /// bypass the probe (the user has already seen + accepted the
    /// warning — re-running the probe would either be a no-op or, in
    /// the unlikely race where the disk filled further in the few
    /// seconds the banner was on screen, trap them in a warning loop).
    private func kickoffDownload() {
        coordinator.enterDownloading()
        // ``hfPath`` wires the cache-directory byte monitor so the
        // progress card reads true bytes-on-disk, not just tqdm file
        // counts. Without this the bar could sit at "0/1 files" for
        // the entire 700 MB pull (HF tqdm counts files, not bytes).
        let started = downloads.startDownload(
            alias: coordinator.selection.alias,
            hfPath: coordinator.selection.hfRepo
        )
        // ``startDownload`` returns ``false`` either because the
        // binary is missing (the synthetic ``.failed`` job already
        // landed and our ``.task(id:)`` observer will pick it up) or
        // because a running job already exists for the alias. The
        // second case is benign — we just stay in ``.downloading``
        // and let the existing job finish.
        _ = started
    }

    private func handleDownloadStatusChange() {
        guard case .downloading = coordinator.phase else { return }
        guard let job = downloads.job(for: coordinator.selection.alias) else { return }
        switch job.status {
        case .running:
            return
        case .completed:
            // Codex r2 BLOCKING: if the server is already engaged with
            // a DIFFERENT alias (user used the still-visible picker
            // mid-download), don't fire ``server.start(gemma...)`` —
            // ``ServerManager.start`` would early-return on
            // ``child == nil`` failing and leave the coordinator stuck
            // in ``.starting`` forever, masking the chat surface. The
            // user's revised intent wins; release the in-flight phase
            // and let the parent's visibility predicate drop us.
            if case .ready(let alias) = server.state,
               alias != coordinator.selection.alias {
                coordinator.releaseInFlight()
                return
            }
            if case .starting(let alias) = server.state,
               alias != coordinator.selection.alias {
                coordinator.releaseInFlight()
                return
            }
            // Hand off to the serve side. ``server.start`` is async
            // and re-enters main actor; we kick it via a Task because
            // the .task(id:) closure is already main-actor bound.
            coordinator.enterStarting()
            Task { @MainActor in
                await server.start(
                    alias: coordinator.selection.alias,
                    hfPath: coordinator.selection.hfRepo
                )
            }
        case .cancelled:
            coordinator.enterFailed(message: "Download was cancelled.")
        case .failed(let message):
            coordinator.enterFailed(message: QuickstartView.friendlyFailureMessage(raw: message))
        }
    }

    private func handleServerStateChange() {
        // The serve transition can race the download observer: the
        // user could click Get started, downloads finishes mid-flight,
        // ``server.start`` lands at ``.ready`` BEFORE the
        // download-status observer fired. Guard on the live state so
        // both ordering paths converge on ``markReady``.
        if case .ready(let alias) = server.state,
           alias == coordinator.selection.alias {
            // Codex r3 MINOR + r4 MINOR: the completed Quickstart
            // download job is dismissed inside the parent's
            // ``seedQuickstartWelcome`` closure on the seed-landed
            // branch. Centralising the dismiss inside the seed
            // closure (rather than here on the seeded==true path)
            // also covers the deferred-seed retry path: a later
            // ``markReady(seed: seedQuickstartWelcome)`` invocation
            // from the parent's ``store.activeID`` / ``server.state``
            // observers lands the welcome AFTER this view has
            // unmounted, so the view-site dismiss wouldn't fire.
            _ = coordinator.markReady {
                onSeedWelcome()
            }
            return
        }
        // Codex r2 BLOCKING: server moved on to a DIFFERENT alias
        // while we were mid-flow (user clicked something in the
        // still-visible picker). Don't fire ``server.start`` from
        // ``handleDownloadStatusChange`` against that state; just
        // release the in-flight phase so the parent's visibility
        // predicate drops us. Falling back to ``.ready`` instead of
        // ``.failed`` because the user's revised intent is not an
        // error — they actively chose a different model.
        if case .ready(let alias) = server.state,
           alias != coordinator.selection.alias,
           case .starting = coordinator.phase {
            // Don't seed (their chosen model is what they want to chat
            // with) and don't flip the persistent done flag — they
            // never finished Quickstart, so a fresh install on a
            // different Mac should still see it.
            coordinator.releaseInFlight()
            return
        }
        if case .crashed(let alias, let message) = server.state,
           alias == coordinator.selection.alias,
           case .starting = coordinator.phase {
            coordinator.enterFailed(message: QuickstartView.friendlyFailureMessage(raw: message))
        }
    }

    // MARK: - Pure helpers (test seam)

    /// Format a byte count for the low-disk banner copy. Pure so the
    /// banner string can be pinned by a unit test.
    ///
    /// Unit cutoff:
    ///   * `< 1 GB` → "N MB" (no decimals), e.g. ``99 MB``
    ///   * `≥ 1 GB` → "N.N GB" (one decimal),  e.g. ``1.5 GB``
    ///
    /// Issue #357: the previous one-decimal-GB formatter rendered a
    /// 99 MB volume as ``0.1 GB`` — both rounds UP and uses the wrong
    /// unit. LM Studio / Ollama (the precedents cited by PR #353) both
    /// switch to MB under 1 GB, so we match.
    ///
    /// Negative inputs clamp to ``0`` — the formatter should never
    /// produce a negative display, even if a future caller passes a
    /// degenerate value.
    static func formatBytesForBanner(_ bytes: Int64) -> String {
        let clamped = max(bytes, 0)
        let mbDivisor: Int64 = 1024 * 1024            // 1 MiB
        let gbDivisor: Int64 = 1024 * 1024 * 1024     // 1 GiB
        if clamped < gbDivisor {
            // Codex r1 MINOR: floor (integer division) instead of
            // `String(format: "%.0f MB", Double / mbDivisor)`. `%.0f`
            // rounds, so `1 GiB - 1` byte would render as `1024 MB`
            // — the very rounding-up pathology the GB branch avoids.
            // Floored MB never crosses the 1 GB cutoff visually.
            let mb = clamped / mbDivisor
            return "\(mb) MB"
        }
        let gb = Double(clamped) / Double(gbDivisor)
        return String(format: "%.1f GB", gb)
    }

    /// Body copy for the low-disk warning banner. Pure helper so the
    /// unit test can pin the copy + numeric rendering without standing
    /// up SwiftUI. Matches the FU-4 spec text shape but with both
    /// numbers filled in from the actual probe.
    static func lowDiskBannerBody(freeBytes: Int64, requiredBytes: Int64, displayName: String) -> String {
        let free = formatBytesForBanner(freeBytes)
        let need = formatBytesForBanner(requiredBytes)
        return "\(free) free on the volume that holds your Hugging Face cache. " +
               "This download needs ~\(need) (\(displayName) " +
               "weights + safety margin). Continue anyway?"
    }

    /// VoiceOver label for the warning card. The banner body is repeated
    /// near-verbatim so screen-reader users get the same numbers as
    /// sighted users; the trailing prompt is rephrased to read as one
    /// sentence rather than a question fragment.
    static func lowDiskAccessibilityLabel(freeBytes: Int64, requiredBytes: Int64, displayName: String) -> String {
        let free = formatBytesForBanner(freeBytes)
        let need = formatBytesForBanner(requiredBytes)
        return "Low disk space warning. \(free) free on the volume that holds " +
               "your Hugging Face cache; this download needs about \(need) for " +
               "\(displayName) weights plus a safety margin. " +
               "Choose Continue anyway to start the download, or Cancel to return."
    }

    /// Build the progress subtitle the downloading card shows. Pure
    /// function so the unit test can pin the byte/percent rendering
    /// without standing up a real ``DownloadManager.Job``.
    ///
    /// Preference order matches the rest of the app (see
    /// ``ContentView.startingOverlay``): structured byte progress
    /// when the byte monitor has observed real disk growth; tqdm
    /// fractions otherwise; bare "Downloading…" when nothing
    /// observable has landed yet.
    static func progressSubtitle(
        job: DownloadManager.Job?,
        displayName: String
    ) -> String {
        guard let job else {
            return "Connecting to mirror…"
        }
        if let subtitle = job.progress.progressSubtitle {
            return subtitle
        }
        return "Connecting to mirror…"
    }

    /// "ETA mm:ss" caption when tqdm has stabilised one, else nil.
    /// Pulled out alongside ``progressSubtitle`` so both pieces of
    /// the progress card stay testable as plain functions.
    static func etaCaption(job: DownloadManager.Job?) -> String? {
        guard let job else { return nil }
        switch job.progress.phase {
        case .downloading(_, _, _, _, _, let eta):
            guard let eta else { return nil }
            return "ETA \(eta)"
        case .idle, .preparing, .fetching, .warmingUp:
            return nil
        }
    }

    /// Compatibility helper retained for coordinator tests and older call
    /// sites. Unknown details deliberately use a safe diagnosis rather than
    /// falling through to raw subprocess output.
    ///
    /// Whitespace-only input (``"\n\n\n"``, ``"   "``, etc.) is treated
    /// the same as empty so the failure card never renders a visually
    /// blank bubble — pre-fix, those strings landed in the verbatim
    /// fall-through (#290). Trimming happens AFTER the keyword
    /// classifier so a message like "  network down  " still classifies;
    /// only the bare-whitespace case takes the empty fallback.
    static func friendlyFailureMessage(raw: String) -> String {
        let lowered = raw.lowercased()
        if lowered.contains("429") || lowered.contains("rate limit") {
            return "Hugging Face is rate-limiting downloads right now. Try again in a minute."
        }
        if lowered.contains("network") || lowered.contains("connection") || lowered.contains("dns")
            || lowered.contains("timeout") || lowered.contains("timed out") {
            return "Network error during download. Check your connection and retry."
        }
        if lowered.contains("no space") || lowered.contains("disk full") {
            return "Not enough disk space to download the model. Free ~3 GB and retry."
        }
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Download didn't finish. Retry to try again."
        }
        return FailureDiagnoser.diagnosis(for: .downloadFailed).message
    }
}
