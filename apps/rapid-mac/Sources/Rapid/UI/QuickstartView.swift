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
/// when (a) the user has no last-served alias persisted — or has one that
/// is a ``retiredStarters`` entry, i.e. a model we stranded them on —
/// AND (b) the quickstart flag in UserDefaults hasn't been set yet AND
/// (c) the server isn't already busy with something else. Clicking
/// "Get started"
/// triggers a ``rapid-mlx pull lfm2.5-1b-4bit`` (~0.6 GB) via the
/// existing ``DownloadManager``, then auto-spawns
/// ``rapid-mlx serve lfm2.5-1b-4bit`` once the pull is done, then
/// drops the user into chat with a single seeded assistant message
/// introducing the model.
///
/// ## Why lfm2.5-1b-4bit
///
/// Starter = LFM2.5 1.2B Instruct (mlx 4-bit; rapid-mlx alias
/// ``lfm2.5-1b-4bit``). This supersedes the ``bonsai-1.7b-2bit``
/// starter, which degenerated 4/4 on a plain-chat word problem — see
/// ``defaultChoice`` for the measurements.
///
/// History: the original ``qwen3.5-4b-4bit`` (~2.3 GB) cold-installed in
/// ~11 minutes at the user's observed 4.4 MB/s — an atrocious first
/// impression — so F-LWT-1 dropped to a tiny ~400 MB 0.6B purely for
/// install latency. #1092 then moved to Bonsai on the strength of a
/// tool-call eval. Both swaps optimised something other than "does the
/// first answer come out right", which is what the user actually sees.
///
/// ### The selection criterion this slot actually has
///
/// A starter is judged on the first plain-chat reply, not on capability
/// breadth. Concretely, in priority order:
///
///   1. **It must terminate and be coherent.** Non-negotiable, and the
///      one thing neither prior pick was measured on. Note the guard
///      cannot cover for a weak model here: the loop breaker and the
///      streaming hard-stop are both gated on ``request.has_tools``,
///      and onboarding is plain chat.
///   2. **It must answer immediately.** A reasoning model is
///      disqualified regardless of quality — hidden thinking means a
///      blank screen on the one interaction that forms the impression.
///      This is why the stronger ``lfm2.5-2.6b-4bit`` is not the
///      starter even though it is the 8-15 GB tier pick.
///   3. **Download small enough not to lose the user.** Real, but
///      weaker than it looks: 637 MB pulled in 21 s, *faster* than
///      Bonsai's 484 MB in 24 s. Shard parallelism dominates at this
///      size, so a few hundred MB is not the deciding axis.
///
/// Tool calling is deliberately absent from that list. It is the right
/// bar for a *recommended* model, not for the first 60 seconds.
///
/// ### What this means for the empty state
///
///   1. The starter is text-first. If ``ToolUseCapability`` does not
///      list ``lfm2.5-`` as ``.known``, the empty-state capability chip
///      row stays hidden by the tool-bias gate (PR #333 + FU-9) — which
///      is the correct, honest surface for it.
///   2. The ``ChatView`` empty-state prompts stay model-agnostic pure
///      text by design — they must read well on ANY starter, not tease
///      a capability tied to one alias — so they are unchanged.
///   3. Users who want more depth trade up via the picker's
///      **Recommended Default** (``RAMBucketedDefault``, RAM-aware —
///      e.g. ``qwen3.5-9b-4bit`` on an 18 GB Mac), and the
///      ``UpgradeBanner`` nudges them there after a few turns.
///
/// ### What we keep
///
///   * One-click install + chat for the brand-new user, well under a
///     minute of cold install.
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
/// small starter (see ``QuickstartCoordinator.defaultChoice``), but lets
/// the user trade up to a bigger model before the first download.
///
/// ``hfRepo`` is pinned only for the starter (it wires the precise
/// bytes-on-disk monitor for the first-impression cold install — see the
/// ``kickoffDownload`` rationale). The bigger options pass ``nil`` and
/// fall back to tqdm file-count progress; both drive an identical
/// download → serve → seed pipeline.
struct QuickstartModelChoice: Equatable, Identifiable, Sendable {
    enum Tier: Equatable, Sendable {
        case starter
        case lowMemory
        case tradeUp
    }

    var id: String { alias }
    /// Canonical alias resolved in ``vllm_mlx/aliases.json``.
    let alias: String
    /// Prose label for onboarding copy ("LFM2.5 · 1.2B"). Hand-picked
    /// rather than catalog-derived so the copy never reads a raw alias.
    let displayName: String
    /// HF repo backing the byte monitor. Pinned for the starter; ``nil``
    /// for bigger options (tqdm-fallback progress is acceptable there).
    let hfRepo: String?
    /// Curated download size for choices whose alias rounds away a meaningful
    /// parameter fraction. The starter alias says `1b` for a 1.2B repository;
    /// using its alias estimate under-reported both the chooser and progress
    /// denominator. Other choices continue to use `ModelSizing` estimates.
    let downloadBytes: Int64?
    /// One-line blurb shown under the name in the chooser.
    let blurb: String
    /// Where this choice belongs in the deliberately short onboarding
    /// ladder. Sub-1B models stay hidden from the normal picker because
    /// they are materially less capable, but ``lowMemory`` gives a user
    /// who cannot safely load the starter an honest escape hatch.
    let tier: Tier

    init(
        alias: String,
        displayName: String,
        hfRepo: String?,
        downloadBytes: Int64? = nil,
        blurb: String,
        tier: Tier
    ) {
        self.alias = alias
        self.displayName = displayName
        self.hfRepo = hfRepo
        self.downloadBytes = downloadBytes
        self.blurb = blurb
        self.tier = tier
    }

    var isStarter: Bool { tier == .starter }
    var isLowMemory: Bool { tier == .lowMemory }
}

/// Persistent state owner + state machine for the Quickstart surface.
@MainActor
@Observable
final class QuickstartCoordinator {
    /// The four PUBLIC onboarding steps (Paper 05.1.G — "Four public
    /// steps, and Ready is confirmed").
    ///
    /// Everything the user can be doing during setup collapses onto one of
    /// these four. Micro-states are NOT steps: hardware detection,
    /// recommendation loading, choosing a recommended / cached /
    /// alternative model and reviewing a model all live inside
    /// ``chooseModel``; preparing, offline, insufficient disk, an
    /// interrupted download, a download failure and its retry all live
    /// inside ``download``; starting, the pre-load memory confirmation and
    /// Ready all live inside ``start``.
    ///
    /// A failure never becomes a fifth step — it keeps the macro step that
    /// owns it (see ``FailureOrigin``), so the rail does not jump when
    /// something goes wrong.
    enum Step: Int, CaseIterable, Equatable, Sendable {
        case welcome = 0
        case chooseModel = 1
        case download = 2
        case start = 3

        /// The one place the public step count is stated. Onboarding V3
        /// moves the production progress model from three steps to four;
        /// every "Step N of M" label reads M from here.
        static let total: Int = Step.allCases.count

        /// 1-based number as spoken and displayed ("Step 3 of 4").
        var displayNumber: Int { rawValue + 1 }
    }

    /// Which macro step owns a terminal failure. Carried on ``Phase/failed``
    /// so the progress rail keeps reporting the step the user was actually
    /// in when it broke — a download failure is still Step 3, a load failure
    /// is still Step 4.
    enum FailureOrigin: Equatable, Sendable {
        /// The pull did not finish (network, mirror, disk, cancellation).
        case download
        /// The weights are on disk but the serve did not come up.
        case start
    }

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
        /// The selected model is serving and onboarding is STOPPED here,
        /// waiting for the user.
        ///
        /// This is the Onboarding V3 change of meaning (Paper 05.1.G —
        /// "Readiness does not dismiss setup"). Before, readiness itself
        /// completed the flow and handed off to chat; the user was never
        /// asked and never confirmed. Now readiness only moves us here: the
        /// full-window surface stays up, nothing is persisted, and the flow
        /// ends only when the user activates Start chatting
        /// (``confirmStartChatting(seedWelcome:)``).
        case ready
        /// Terminal. The onboarding surface has released the frame, either
        /// because the user confirmed Ready or because they revised their
        /// intent mid-flow (``releaseInFlight``). Whether onboarding was
        /// actually COMPLETED is ``done``'s business, not this phase's —
        /// only the confirmed path writes it.
        case dismissed
        /// Download or serve failed. ``message`` is a single-line
        /// human-readable summary suitable for inline display; ``origin``
        /// pins the macro step that owns the failure so the rail stays put.
        /// "Retry" is offered; the persistent done-flag is NOT set.
        case failed(message: String, origin: FailureOrigin)
    }

    /// The default + recommended starter — the first-run decision.
    /// Pinned, not derived (F-LWT-1: ~11 min cold install of the old 4B
    /// pick was the wrong first-impression tradeoff; a small starter
    /// wins).
    ///
    /// ## History
    ///
    /// - 2026-07-10 (#1092): ``qwen3-0.6b-4bit`` → ``bonsai-1.7b-2bit``.
    /// - 2026-08-05: ``bonsai-1.7b-2bit`` → ``lfm2.5-1b-4bit``, because
    ///   the Bonsai starter does not survive an ordinary chat question.
    ///
    /// ## Why the Bonsai starter had to go
    ///
    /// A community report showed the starter collapsing on a basic
    /// multi-step word problem. Reproduced on an M2 Pro against engine
    /// 0.12.4, one plain-chat request (no tools), 4 attempts at two
    /// different token budgets: it degenerated **4/4** and terminated
    /// **0/4**. Output doubles words within the first line ("for for",
    /// "the the the the"), then collapses into an unbounded loop
    /// (``\text{1} \text{1} …``, ``1 + 9 = 1 + 9 = …``) that only ends
    /// when it hits ``max_tokens``.
    ///
    /// Two things made this the worst possible default. It is *fast*
    /// while being wrong, so it reads as "this app is broken" rather
    /// than "this Mac is slow". And the runaway-generation guard cannot
    /// save it: both the logits-level loop breaker and the streaming
    /// hard-stop are gated on ``request.has_tools``, and onboarding is
    /// plain chat — so nothing intervenes.
    ///
    /// The prior "6/6 clean ``tool_calls``" evidence is not contradicted;
    /// it just measured the wrong thing for this slot. Emitting
    /// well-formed tool calls says nothing about staying coherent in the
    /// free-form chat every new user actually types first.
    ///
    /// ## Why the 1.2B and not the 2.6B
    ///
    /// Measured on the same M2 Pro, same prompt. The download worry that
    /// motivated a ~0.5 GB pick does not survive measurement: 637 MB
    /// pulled in 21 s, *faster* than Bonsai's 484 MB in 24 s (HF shard
    /// parallelism dominates at this size).
    ///
    /// ``lfm2.5-2.6b-4bit`` is the stronger model and stays the 8-15 GB
    /// tier recommendation — but it is the wrong *starter*. It routes
    /// through the ``qwen3`` reasoning parser, so ~2/3 of its output is
    /// hidden thinking: 3.6 s to a first answer, most of it a blank
    /// screen. The 1.2B has no reasoning phase — 1.1 s, 170 tok/s, and it
    /// answered correctly and terminated on **16/16** recorded runs
    /// (12/12 of them in one controlled repro; quote the 16, it is the
    /// whole sample). For a first impression, "instant and right" beats
    /// "smarter but silent first".
    ///
    /// Users still trade up in the wizard or later via the picker.
    static let defaultChoice = QuickstartModelChoice(
        alias: "lfm2.5-1b-4bit",
        displayName: "LFM2.5 · 1.2B",
        hfRepo: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
        downloadBytes: 663_397_140,
        blurb: "Small download (~0.6 GB), runs on any Mac. Answers instantly and follows instructions well. Upgrade anytime for more depth.",
        tier: .starter
    )

    /// Deliberately weaker than the starter. The normal model picker hides
    /// sub-1B models to protect users from accidentally choosing quality
    /// below the product floor; onboarding surfaces this one explicitly as
    /// a memory-first fallback and names the trade-off instead of pretending
    /// it is an equivalent recommendation.
    static let lowMemoryChoice = QuickstartModelChoice(
        alias: "qwen3-0.6b-4bit",
        displayName: "Qwen 3 · 0.6B",
        hfRepo: "mlx-community/Qwen3-0.6B-4bit",
        blurb: "Lowest memory and fastest startup. Good for basic chat, but less accurate and not recommended for tools.",
        tier: .lowMemory
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
        lowMemoryChoice,
        QuickstartModelChoice(
            alias: "qwen3.5-4b-4bit",
            displayName: "Qwen 3.5 · 4B",
            hfRepo: nil,
            blurb: "Better everyday quality. Still light on disk.",
            tier: .tradeUp
        ),
        QuickstartModelChoice(
            alias: "qwen3.5-9b-4bit",
            displayName: "Qwen 3.5 · 9B",
            hfRepo: nil,
            blurb: "Strong all-rounder if you have the RAM to spare.",
            tier: .tradeUp
        ),
    ]

    /// UserDefaults key for the persistent "Quickstart already
    /// completed" flag. Once set, the surface NEVER returns — not
    /// even after the user deletes every model. Versioned so a
    /// Quickstart refresh can re-show without clobbering the older flag.
    ///
    /// Moved to v2 on 2026-08-05 for the retired-starter swap. The bump
    /// alone is not the migration: ``isEligible`` still honours
    /// ``legacyStorageKey`` so a v1 dismissal is not silently undone.
    static let storageKey: String = "rapid.quickstart.v2.done"

    /// Pre-2026-08-05 completion flag. Read-only — nothing writes it any
    /// more; it exists so a user who dismissed under v1 stays dismissed.
    static let legacyStorageKey: String = "rapid.quickstart.v1.done"

    /// Welcome message seeded into the active session after the sidecar
    /// comes online, so the user always lands in chat with a friendly
    /// intro rather than an empty transcript. Interpolates the chosen
    /// model's display name.
    ///
    /// The starter (lfm2.5-1b-4bit) is intentionally the smallest
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

    /// Where inside **Step 2 · Choose a model** the user is (Paper 05.2.B —
    /// "Five micro-stages inside one macro step").
    ///
    /// These are branches, not steps. Every one of them reports
    /// ``Step/chooseModel``, so the rail reads `Step 2 of 4` throughout and
    /// never gains a fifth row for the catalogue or for review. The public
    /// four-step model introduced by PR #1917 is untouched: ``stage`` still
    /// decides the macro step, and this enum is deliberately not consulted by
    /// ``step(phase:stage:)`` at all — which is what makes "a micro-stage
    /// cannot become a step" true by construction rather than by review.
    ///
    /// Ordering matches the user's path through Step 2, not a progress value;
    /// nothing sub-numbers the kicker (`STEP 2.3 OF 4` is forbidden).
    enum Step2Stage: String, CaseIterable, Equatable, Sendable {
        /// 2a — reading this Mac's chip and unified memory.
        ///
        /// Real detection, never a simulated scan: ``MacHardware/detect()``
        /// and ``MemoryProbe/snapshot(...)`` are synchronous sysctl reads, so
        /// in production this resolves within the same render pass and is not
        /// observably on screen. It is modelled anyway so the hardware read
        /// has a named home inside Step 2 rather than being smuggled in
        /// somewhere that could later claim its own step — and so nothing is
        /// tempted to add a delay to make a stage "visible".
        case checkingHardware
        /// 2b — matching models to this Mac.
        ///
        /// The genuinely asynchronous one: the shortlist cannot say which
        /// models are already on disk, and therefore cannot derive its footer
        /// verb, until ``ModelCatalog/load(binary:hubCacheOverride:)`` has
        /// answered. Indeterminate by nature — neither subprocess reports
        /// progress, so nothing here may draw a determinate bar.
        case findingFit
        /// 2c — the recommended shortlist (cached rows, starter, low-memory
        /// fallback, trade-ups, and a catalogue pick carried back as YOUR PICK).
        case choosing
        /// 2d — in-window Browse all models. The real catalogue, on the setup
        /// canvas: no Settings window, no second window, no sheet.
        case browsing
        /// 2e — Review download: name the cost before spending it.
        case reviewing
    }

    /// Which list a Review download was opened from, so Back can return to it
    /// (Paper 05.2.J · S2 — the old "Secondary Back → Welcome" note is
    /// superseded; Review returns to its origin, never to the hero).
    enum ReviewOrigin: Equatable, Sendable {
        case shortlist
        case catalogue
    }

    private(set) var step2Stage: Step2Stage = .choosing

    /// The list a Review download was entered from. Only meaningful while
    /// ``step2Stage`` is ``Step2Stage/reviewing``; retained afterwards so the
    /// value is stable for the duration of the Back that reads it.
    private(set) var reviewOrigin: ReviewOrigin = .shortlist

    // MARK: - Browse all models state (Paper 05.2.H — what must survive Back)
    //
    // All of it lives here rather than in `@State` for the same reason
    // ``selection`` does: it has to survive a SwiftUI re-mount, and Back out of
    // Review has to be able to restore a list the view may have torn down.
    //
    // None of it is persisted to UserDefaults. A relaunch starts Step 2 clean.

    /// Catalogue search text, verbatim. Matched against alias AND Hugging Face
    /// repo by ``ModelCacheActions/filter(_:by:query:)``.
    var catalogQuery: String = ""

    /// Catalogue filter segment. ``ModelCacheActions/FilterMode`` reused as-is.
    var catalogFilter: ModelCacheActions.FilterMode = .all

    /// Catalogue sort order. ``ModelCacheActions/SortOrder`` reused as-is.
    var catalogSort: ModelCacheActions.SortOrder = .familyThenSize

    /// Scroll anchor for the catalogue, as an **alias** rather than a pixel
    /// offset — a pixel offset points at a different row after a filter or
    /// sort change, which is exactly when restoring it matters.
    var catalogScrollID: String?

    /// Enter in-window Browse all models (Paper 05.2.H · T1).
    ///
    /// Carries the selection in and leaves query / filter / sort / scroll
    /// exactly as the user last left them, so re-entering the catalogue is a
    /// return rather than a reset.
    func beginBrowsingCatalog() {
        guard case .idle = phase else { return }
        stage = .chooseModel
        step2Stage = .browsing
    }

    /// Leave the catalogue for the recommended shortlist (Paper 05.2.H · T2).
    ///
    /// Retains every piece of catalogue state. The selection is not touched:
    /// if it is a model the shortlist does not natively list, the shortlist
    /// shows it as YOUR PICK (approved default D2) rather than silently
    /// disagreeing with the footer.
    func backToRecommendedModels() {
        guard case .idle = phase else { return }
        stage = .chooseModel
        step2Stage = .choosing
    }

    /// Open Review download for the current selection (Paper 05.2.H · T3).
    ///
    /// ``origin`` decides the Back label and destination. Review is never the
    /// origin of another Review, so a call made while already reviewing keeps
    /// the original origin rather than pinning Review to itself.
    func beginReviewDownload(origin: ReviewOrigin) {
        guard case .idle = phase else { return }
        guard step2Stage != .reviewing else { return }
        stage = .chooseModel
        reviewOrigin = origin
        step2Stage = .reviewing
    }

    /// Back out of Review download to the list it was opened from.
    ///
    /// The caller re-derives the footer *after* this returns — the list is
    /// rebuilt first, the selection revalidated second, the primary derived
    /// third (Paper 05.2.G — "Return from Review restores origin").
    func backFromReviewDownload() {
        guard case .idle = phase else { return }
        stage = .chooseModel
        switch reviewOrigin {
        case .shortlist: step2Stage = .choosing
        case .catalogue: step2Stage = .browsing
        }
    }

    /// Record the row the catalogue should be anchored on when it is restored.
    func rememberCatalogAnchor(_ alias: String?) {
        catalogScrollID = alias
    }

    /// Move one level closer to the shortlist, if the user is inside a Step 2
    /// sub-stage. Returns `true` when it handled the request.
    ///
    /// This is the backstop for Paper 05.2.G's invariant — *"while the user is
    /// inside Browse all models or Review download, Escape can only move them
    /// one level closer to the shortlist; it can never leave setup from
    /// there"*.
    ///
    /// The footer's `.cancelAction` Back normally consumes Escape before
    /// anything else sees it. But onboarding is presented in a `.sheet`, and a
    /// sheet dismissal is also reachable by swipe-down and by any future host
    /// that decides Escape means "close this". Routing that request through
    /// here first means it resolves to the SAME destination as the visible Back
    /// control rather than skipping setup from two levels deep — so the
    /// invariant holds no matter which layer wins the key.
    @discardableResult
    func retreatWithinStep2() -> Bool {
        guard case .idle = phase, stage == .chooseModel else { return false }
        switch step2Stage {
        case .reviewing:
            backFromReviewDownload()
            return true
        case .browsing:
            backToRecommendedModels()
            return true
        case .checkingHardware, .findingFit, .choosing:
            // The Step 2 root. Onboarding's own Skip/Back meaning resumes here
            // and only here (Escape priority 4).
            return false
        }
    }

    /// The public macro step the current (phase, stage) pair belongs to.
    ///
    /// Pure and static so the four-step mapping can be pinned exhaustively
    /// without a SwiftUI host, and so every rendered rail reads the SAME
    /// function rather than hard-coding an ordinal per screen — which is
    /// how the old model ended up with two screens both claiming step 3.
    static func step(phase: Phase, stage: Stage) -> Step {
        switch phase {
        case .idle:
            // The pre-download wizard screens are the only place ``stage``
            // is load-bearing; once a download is in flight the lifecycle
            // machine owns the step.
            switch stage {
            case .welcome:     return .welcome
            case .chooseModel: return .chooseModel
            }
        case .lowDiskWarning, .downloading:
            // Insufficient disk is a download-time interstitial, not a step
            // of its own — the user is being asked about the pull they just
            // authorised.
            return .download
        case .starting, .ready, .dismissed:
            return .start
        case .failed(_, let origin):
            switch origin {
            case .download: return .download
            case .start:    return .start
            }
        }
    }

    /// Live macro step for the current state.
    var step: Step { Self.step(phase: phase, stage: stage) }

    /// Advance from the hero to the model chooser ("Get started").
    ///
    /// Always enters at the top of Step 2. Catalogue query / filter / sort
    /// survive — re-entering Step 2 should not silently retype the user's
    /// search — but the surface they see is the one Step 2 opens on.
    ///
    /// Enters ``Step2Stage/checkingHardware`` rather than
    /// ``Step2Stage/choosing`` because the shortlist genuinely cannot be drawn
    /// truthfully yet: which of its models are already on disk, and therefore
    /// what the footer verb is, comes from the catalogue snapshot.
    /// ``resolveRecommendationLoading(catalogLoaded:)`` moves it on.
    func advanceToChooseModel() {
        stage = .chooseModel
        step2Stage = .checkingHardware
        // From here on, a launch that finds setup unfinished knows the user
        // has been here before. Written on entry rather than on the first
        // download because entering Step 2 is already the point at which
        // "Get started" stops being a truthful thing to offer them next time.
        setupBegun = true
    }

    /// Settle the two pre-shortlist micro-stages against real signals.
    ///
    /// Hardware detection is synchronous, so ``Step2Stage/checkingHardware``
    /// leaves as soon as anything asks; the wait that actually exists is the
    /// catalogue load behind ``Step2Stage/findingFit``. Nothing here invents a
    /// duration — if the snapshot is already in hand on entry, Step 2 opens
    /// straight onto the shortlist and neither loading surface is ever drawn.
    ///
    /// Navigational micro-stages are left alone: a catalogue that re-loads
    /// under the user must not yank them out of Browse all models or Review.
    func resolveRecommendationLoading(catalogLoaded: Bool) {
        guard case .idle = phase, stage == .chooseModel else { return }
        switch step2Stage {
        case .checkingHardware, .findingFit:
            step2Stage = catalogLoaded ? .choosing : .findingFit
        case .choosing:
            // The snapshot can be invalidated after the fact (a download
            // completes and bumps the cache generation). Report that honestly
            // rather than leaving a shortlist whose cached column is stale.
            if !catalogLoaded { step2Stage = .findingFit }
        case .browsing, .reviewing:
            break
        }
    }

    /// Back out of the chooser to the hero ("Back").
    ///
    /// Only reachable from the Step 2 root: Browse all models and Review
    /// download own the Back control while they are showing, so this cannot be
    /// how a user leaves either of them.
    func backToWelcome() {
        stage = .welcome
        step2Stage = .choosing
    }

    /// Drop a serve that the pre-load memory guard declined back to the
    /// model chooser (#1503). The handoff to ``ServerManager.start`` parks
    /// on ``pendingMemoryWarning`` and returns WITHOUT changing
    /// ``server.state``, so nothing ever moves ``phase`` out of
    /// ``.starting`` on its own — the sheet would sit on "Starting…"
    /// forever. When the user declines the risky load we return here:
    /// NOT ``.failed`` (the download succeeded — only loading was refused),
    /// and NOT ``.idle``/``.welcome`` (they already chose a model), but the
    /// chooser, where they can free memory and retry, pick a smaller model,
    /// or browse all. Leaving ``.starting`` is what releases the sheet.
    ///
    /// ``step2Stage`` is deliberately NOT reset: it still holds the micro-stage
    /// the user was on when they authorised the load, so declining returns them
    /// to the shortlist, the catalogue or Review download — whichever they
    /// actually left. Paper 05.2.J · S3 supersedes the old "Cancel lands on the
    /// model chooser" note, which was only ever true while the chooser was
    /// Step 2's single surface.
    func returnToChooser() {
        phase = .idle
        stage = .chooseModel
    }

    /// Set the model the wizard will download. No-op once a download is
    /// in flight (``phase != .idle``) so a late tap can't retarget an
    /// active pull.
    ///
    /// Moving to a different alias invalidates any pending-Ready
    /// provenance: the flow that reached Ready was about the OLD model, and
    /// keeping the record would let a later relaunch offer to confirm a
    /// model the user has since walked away from.
    func select(_ choice: QuickstartModelChoice) {
        guard case .idle = phase else { return }
        selection = choice
        if let pending = pendingReadyAlias, pending != choice.alias {
            clearPendingReady()
        }
    }

    /// True once ``markDone`` has been called. Read on every eligibility
    /// check so the surface never returns. Mirrors UserDefaults.
    private(set) var done: Bool

    /// Snapshot of ``legacyStorageKey`` taken at init. Never written.
    let legacyDone: Bool

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
            defaults.set(awaitingWelcomeSeed, forKey: Self.awaitingSeedKey)
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
                defaults.set(selection.alias, forKey: Self.awaitingSeedAliasKey)
            } else {
                defaults.removeObject(forKey: Self.awaitingSeedAliasKey)
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

    /// Provenance for an onboarding flow that reached ``Phase/ready`` but
    /// has NOT been confirmed with Start chatting (Paper 05.1.G —
    /// "Completion is what persists, not readiness").
    ///
    /// ## Why a second key rather than reusing ``storageKey``
    ///
    /// ``storageKey`` answers "is onboarding finished?" and must stay
    /// truthful: it is written only by the user's confirmation. But
    /// "finished" and "never started" are not the only two states any more
    /// — a user can quit while the Ready screen is on screen, and on the
    /// next launch we owe them that same screen rather than either the
    /// normal shell (which would silently swallow the flow) or the welcome
    /// hero (which would pretend nothing happened). This key records
    /// exactly that third state, and names the alias it is about so the
    /// claim can be re-verified instead of trusted.
    ///
    /// ## What it is NOT
    ///
    /// It is not a readiness cache. A stored alias alone never re-enters
    /// Ready — ``QuickstartView.handleServerStateChange`` re-enters it only
    /// when ``ServerManager`` genuinely reports ``.ready`` for that alias
    /// on this launch. If the model is no longer ready the user lands back
    /// on the ordinary chooser with their pick preselected, and nothing
    /// claims a download or a selection was "resumed".
    ///
    /// Cleared on: confirmation, ``releaseInFlight``, a fresh
    /// ``enterDownloading``, ``skipForNow``, selecting a different alias,
    /// and ``_testingReset``.
    static let pendingReadyAliasKey: String = "rapid.quickstart.v1.pendingReadyAlias"

    /// Provenance for "this install has been inside setup before and never
    /// finished it" (Paper 05.1 state 18 — "Relaunch, setup incomplete").
    ///
    /// ## Why a third key
    ///
    /// The two existing persisted signals answer different questions.
    /// ``storageKey`` says setup was COMPLETED; ``pendingReadyAliasKey`` says
    /// a specific model reached Ready and is owed a confirmation. Between them
    /// sits the common interruption: somebody opened the app, walked into
    /// Step 2, maybe started a download, and quit. Nothing recorded that, so
    /// the next launch greeted them with "Get started" — a first-run
    /// invitation offered to somebody who is not on their first run.
    ///
    /// ## What it deliberately does NOT do
    ///
    /// It carries nothing forward. No selection, no job record, no partial-
    /// download bookkeeping — Paper is explicit that the app holds none of
    /// that across a relaunch, and that whether the underlying pull reuses
    /// bytes already in the Hugging Face cache is a property of the
    /// downloader that Rapid-MLX cannot promise. So this flag changes what
    /// the welcome screen CALLS its primary and nothing else: the model is
    /// still chosen from scratch and the download still starts as a fresh
    /// pull. It is a fact about history, never a restored transfer.
    ///
    /// Written when the user first enters Step 2. Cleared only by completion
    /// and by ``_testingReset()`` — a Skip does not clear it, because setup is
    /// still owed and "Continue setup" is still the truthful label.
    static let setupBegunKey: String = "rapid.quickstart.v1.setupBegun"

    private(set) var setupBegun: Bool {
        didSet { defaults.set(setupBegun, forKey: Self.setupBegunKey) }
    }

    /// Whether the welcome screen is greeting a returning, unfinished setup
    /// rather than a first run.
    ///
    /// Not persisted separately — it is the question the two persisted flags
    /// already answer together, asked in one place so the screen cannot get
    /// the conjunction wrong.
    var isResumingIncompleteSetup: Bool { setupBegun && !done }

    /// Alias of an unconfirmed Ready flow, or ``nil`` when there is none.
    private(set) var pendingReadyAlias: String? {
        didSet {
            if let pendingReadyAlias {
                defaults.set(pendingReadyAlias, forKey: Self.pendingReadyAliasKey)
            } else {
                defaults.removeObject(forKey: Self.pendingReadyAliasKey)
            }
        }
    }

    /// True while an unconfirmed Ready flow is on the books.
    var hasPendingReady: Bool { pendingReadyAlias != nil }

    /// Injectable so tests validate erasure against a scratch suite instead
    /// of mutating the developer's real ``defaults`` when they
    /// run the suite (#1973). Defaults to ``.standard`` — production is
    /// unchanged.
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.done = defaults.bool(forKey: Self.storageKey)
        self.legacyDone = defaults.bool(forKey: Self.legacyStorageKey)
        // History only. Nothing below reconstructs a phase, a selection or a
        // job from it — a relaunch always starts at ``.idle``, which is what
        // makes "never restore a fake active transfer" true by construction
        // rather than by remembering to avoid it.
        self.setupBegun = defaults.bool(forKey: Self.setupBegunKey)
        // Codex r5: read the persisted awaiting-seed flag so a
        // quit-mid-deferred-flow relaunch can resume the welcome
        // injection once an active session lands. (Assigning a stored
        // property in ``init`` does NOT trigger the didSet, so this read
        // can't clobber the persisted alias below.)
        self.awaitingWelcomeSeed = defaults.bool(forKey: Self.awaitingSeedKey)
        self.pendingReadyAlias = defaults.string(forKey: Self.pendingReadyAliasKey)
        // #1524: if a deferred seed survived a quit, restore the model it
        // was waiting on so the seed observers match the served alias and
        // the welcome copy names the right model (not the reset default).
        if self.awaitingWelcomeSeed,
           let alias = defaults.string(forKey: Self.awaitingSeedAliasKey) {
            self.selection = Self.choice(forAlias: alias)
        }
        // An unconfirmed Ready flow restores its model and drops the user
        // back at the chooser rather than the welcome hero — they already
        // made this choice, and re-asking "would you like to get started?"
        // of somebody who downloaded and loaded a model reads as amnesia.
        //
        // Deliberately NOT ``phase = .ready``: at init nothing has verified
        // the model is actually up on this launch. The Ready screen is
        // re-entered by the live server observer or not at all.
        if let alias = self.pendingReadyAlias {
            self.selection = Self.choice(forAlias: alias)
            self.stage = .chooseModel
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
            tier: alias == defaultChoice.alias ? .starter : .tradeUp
        )
    }

    /// Persist the "Quickstart already completed" flag and trip the
    /// in-memory mirror. Idempotent.
    func markDone() {
        done = true
        defaults.set(true, forKey: Self.storageKey)
        // Setup is finished, so there is no unfinished setup to resume.
        // Retired rather than left set: ``isResumingIncompleteSetup`` already
        // guards on ``done``, but a stale true here would come back to life if
        // a future ``storageKey`` bump ever re-opened onboarding, and offer to
        // "continue" a run that completed on an older version.
        setupBegun = false
    }

    /// Put the wizard back to the state a Mac has before it has ever run.
    ///
    /// Quickstart is one-shot per Mac by design, and no shipping UI offers a
    /// way back — the only callers are the test suite, ``DevSnapshot``, and
    /// the debug-only Settings → Developer panel, which exists so that flow
    /// can be rehearsed without faking `$HOME`.
    ///
    /// This does NOT clear `rapid.serve.lastAlias`, which gates the wizard
    /// independently of these flags (see ``isEligible``). Callers wanting a
    /// true first-run state must stop the server too; ``ReonboardingReset``
    /// does exactly that.
    internal func resetForReonboarding() {
        done = false
        phase = .idle
        stage = .welcome
        step2Stage = .choosing
        reviewOrigin = .shortlist
        catalogQuery = ""
        catalogFilter = .all
        catalogSort = .familyThenSize
        catalogScrollID = nil
        selection = Self.defaultChoice
        hasSeededWelcome = false
        awaitingWelcomeSeed = false
        pendingReadyAlias = nil
        // Union of both sides of the #1946 merge, not either one: each
        // cleared a flag the other did not, and dropping either leaves the
        // reset silently incomplete.
        setupBegun = false
        // #1946's resume marker. Without this the relaunch this reset
        // triggers greets the user with "Continue setup" — the exact
        // untruthful state that PR exists to remove.
        defaults.removeObject(forKey: Self.setupBegunKey)
        defaults.removeObject(forKey: Self.storageKey)
        // Clear the legacy v1 flag too. A user upgraded from a build that
        // wrote ``rapid.quickstart.v1.done`` would otherwise relaunch reading
        // ``legacyDone == true``, which ``onboardingOwed`` treats as "already
        // onboarded" and suppresses Quickstart — silently defeating the reset
        // unless "erase all settings" was also chosen (#1973). ``legacyDone``
        // is a launch-time ``let``, so removing the key is what takes effect
        // on the relaunch this reset triggers.
        defaults.removeObject(forKey: Self.legacyStorageKey)
        defaults.removeObject(forKey: Self.awaitingSeedKey)
        defaults.removeObject(forKey: Self.awaitingSeedAliasKey)
        defaults.removeObject(forKey: Self.pendingReadyAliasKey)
    }

    /// The name 44 call sites in the suite and ``DevSnapshot`` already use.
    /// Kept as an alias rather than renamed at every site, so this change
    /// stays reviewable as "the reset grew a second caller".
    internal func _testingReset() { resetForReonboarding() }

    /// Drop the record of an unconfirmed Ready flow. Idempotent.
    func clearPendingReady() {
        pendingReadyAlias = nil
    }

    /// The user asked to leave setup for now ("Skip for now", Esc, or a
    /// swipe-down on the sheet).
    ///
    /// Skip keeps its existing semantics — it does NOT write the completion
    /// flag, so onboarding is still owed on a later launch — but it does
    /// retire any pending-Ready record. Someone who deliberately walked away
    /// from the Ready screen has answered the question it was asking; coming
    /// back to it on the next launch would be re-asking.
    func skipForNow() {
        clearPendingReady()
        awaitingWelcomeSeed = false
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
        // A fresh pull is a fresh flow: whatever reached Ready before is no
        // longer the thing being confirmed.
        clearPendingReady()
    }

    /// Surface the non-blocking low-disk warning between the hero card
    /// and the download kickoff. The user owns the "continue anyway"
    /// decision (per ``feedback_copy_mature_competitors`` — LM Studio /
    /// Ollama warn but never block). FU-4 / PR #338 review.
    func enterLowDiskWarning(freeBytes: Int64, requiredBytes: Int64) {
        phase = .lowDiskWarning(freeBytes: freeBytes, requiredBytes: requiredBytes)
    }

    /// User chose Cancel on the low-disk warning.
    ///
    /// Returns to the Step 2 micro-stage the pull was authorised from —
    /// shortlist, catalogue or Review download — because ``stage`` and
    /// ``step2Stage`` were never touched on the way in and leaving ``phase``
    /// is the whole of the way out. Paper 05.2.D states the destination
    /// explicitly: "Cancel returns here, not to Welcome."
    ///
    /// Distinct from ``enterFailed(message:origin:)`` because this isn't a
    /// failure shape — the download never started, and nothing about the
    /// user's selection has been invalidated.
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
    ///
    /// ``origin`` is the macro step that owns the failure. It exists so a
    /// failure never reads as its own step: a broken pull still reports
    /// Step 3, a serve that would not come up still reports Step 4.
    func enterFailed(message: String, origin: FailureOrigin) {
        phase = .failed(message: message, origin: origin)
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
        phase = .dismissed
        awaitingWelcomeSeed = false
        clearPendingReady()
    }

    /// Readiness landed for the selected model: park onboarding on the
    /// Ready screen and record that a confirmation is outstanding.
    ///
    /// This is deliberately the WHOLE of what readiness does. Before
    /// Onboarding V3 this method also seeded the welcome message and wrote
    /// the completion flag, so the app decided on the user's behalf that
    /// setup was finished the instant a subprocess reported a port was
    /// listening. Paper 05.1.G retires that ending explicitly ("Kept for the
    /// record, not for build … must not be re-introduced"): readiness is
    /// something to state, and completion is something to confirm.
    ///
    /// So nothing is persisted here except the provenance saying a
    /// confirmation is owed, and the surface stays up.
    ///
    /// Idempotent, because readiness is not a single event: an auto-respawn
    /// cycle, a residency refresh or a scheduler tick can all republish
    /// ``.ready`` for the same serve. Repeat calls re-affirm the same state
    /// and change nothing. A flow that has already been confirmed or
    /// released is never dragged back onto the Ready screen.
    func enterReady() {
        guard !done else { return }
        guard phase != .dismissed else { return }
        phase = .ready
        pendingReadyAlias = selection.alias
    }

    /// The user activated **Start chatting** — the single completion
    /// transaction for onboarding.
    ///
    /// Runs, in order: seed the welcome message exactly once, persist the
    /// completion flag, retire the pending-Ready provenance, and release the
    /// surface. Everything outside this object's ownership — routing to
    /// Chat, announcing completion, moving keyboard focus — is the caller's
    /// half of the transaction and runs only when this returns ``true``.
    ///
    /// Idempotent by construction: the guard is the phase itself, so a
    /// double-click, a repeated key activation, or a stray re-entry after
    /// completion all return ``false`` without seeding a second welcome,
    /// re-writing the flag, or re-running the caller's transition.
    ///
    /// - Parameter seedWelcome: appends the welcome assistant message to the
    ///   intended chat session, returning ``true`` when it actually landed.
    ///   A ``false`` return does NOT block completion — the user asked to
    ///   start chatting and must not be stranded on a screen they already
    ///   dismissed — but it does leave ``awaitingWelcomeSeed`` set so the
    ///   parent's retry observer can land the message once a session exists.
    /// - Returns: ``true`` when this call performed the transaction.
    @discardableResult
    func confirmStartChatting(seedWelcome: () -> Bool) -> Bool {
        guard case .ready = phase else { return false }
        if !hasSeededWelcome {
            if seedWelcome() {
                hasSeededWelcome = true
                awaitingWelcomeSeed = false
            } else {
                awaitingWelcomeSeed = true
            }
        }
        markDone()
        clearPendingReady()
        phase = .dismissed
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
    ///      (``rapid.quickstart.v2.done``). Set once the user completes
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
    /// Aliases retired because they do not survive an ordinary chat, not
    /// because something better came along.
    ///
    /// Gate 2 below treats "has served a model" as "is not a new user".
    /// That inference breaks for the one cohort this list exists for: a
    /// user whose only model is ``bonsai-1.7b-2bit`` did reach a running
    /// model, so the gate calls them onboarded — but what they onboarded
    /// onto degenerates 4/4 on a plain-chat question (see
    /// ``defaultChoice``). Bumping ``storageKey`` to v2 alone does not
    /// reach them: their ``rapid.serve.lastAlias`` is set, so gate 2 keeps
    /// the card down and they stay stranded on the broken starter.
    ///
    /// Membership here is a strong claim — it re-opens onboarding for
    /// someone already using the app. Add an alias only when it is
    /// effectively unusable, never merely superseded.
    ///
    /// Scope, precisely: ``rapid.serve.lastAlias`` is the *most recent*
    /// serve, not a history. So the carve-out fires for anyone whose
    /// **current** model is retired — including a user who traded up and
    /// later went back to it deliberately, who is arguably not stranded.
    /// That is accepted rather than fixed: the alternative is persisting
    /// an onboarding history, which is more state to keep correct than
    /// the four-line gate it would protect, and the cost of a false
    /// positive is bounded — the card appears once on an idle server and
    /// dismissing it sets ``done`` permanently. What the carve-out will
    /// never do is reach a user whose current model is anything else.
    static let retiredStarters: Set<String> = ["bonsai-1.7b-2bit"]

    /// Whether the persisted alias is one we retired for being unusable.
    static func isStranded(_ lastServedAlias: String?) -> Bool {
        guard let alias = lastServedAlias else { return false }
        return retiredStarters.contains(alias)
    }

    /// Gates 1 + 2 of ``isEligible`` on their own: does this install still
    /// owe the user onboarding? Persisted state only — no ``ServerState``,
    /// no session flags.
    ///
    /// ## Why this is split out (issue #1589)
    ///
    /// Two code paths need the SAME answer at moments when they see
    /// different ``ServerState``, so the server-state gate cannot be part
    /// of the shared question:
    ///
    /// * ``ContentView.quickstartVisible`` asks on every render, by which
    ///   point a server may legitimately be engaged.
    /// * ``ContentView.runLaunchAutoStart`` must ask *before* it engages
    ///   one — it is the thing that would move the state.
    ///
    /// Pre-fix, auto-start asked neither and simply started a model on any
    /// Mac with something in the HF cache. That flipped ``serverState`` to
    /// ``.starting`` before the sheet's predicate ever ran, and BOTH of
    /// gate 3 here and ``ContentView.serverEngagedWithDifferentAlias`` then
    /// read the app's own self-inflicted state as "this is not a new user".
    /// The wizard became unreachable for everyone except users with a
    /// completely empty cache. Routing both callers through this one
    /// predicate is what stops the two halves drifting apart again — see
    /// ``LaunchOnboardingOrderingTests``.
    ///
    /// - Parameter legacyDone: the pre-v2 completion flag
    ///   (``legacyStorageKey``). Bumping ``storageKey`` to v2 is what
    ///   re-opens onboarding, but on its own it re-opens it for *everyone*
    ///   who had not completed under v2 — including a user who deliberately
    ///   dismissed the card under v1 and never served anything, whose
    ///   ``done`` and ``lastServedAlias`` both read empty. That would break
    ///   the documented "the card never returns" contract for people the
    ///   version bump was never about. A v1 dismissal is therefore still
    ///   honoured; it is overridden only for the stranded cohort, which is
    ///   the entire reason the key moved.
    static func onboardingOwed(
        done: Bool,
        legacyDone: Bool = false,
        lastServedAlias: String?
    ) -> Bool {
        guard !done else { return false }
        let stranded = isStranded(lastServedAlias)
        guard !(legacyDone && !stranded) else { return false }
        // Gate 2, with the retired-starter carve-out. `nil` is the
        // genuinely-new user; a retired starter is a user we stranded.
        if lastServedAlias != nil, !stranded {
            return false
        }
        return true
    }

    /// ``onboardingOwed`` plus gate 3 — the presentation-time question.
    /// Kept as the sheet's entry point so existing callers and tests read
    /// unchanged; the persisted half now lives in one place.
    static func isEligible(
        done: Bool,
        legacyDone: Bool = false,
        lastServedAlias: String?,
        serverState: ServerState
    ) -> Bool {
        guard onboardingOwed(
            done: done,
            legacyDone: legacyDone,
            lastServedAlias: lastServedAlias
        ) else { return false }
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

    /// The ONLY mechanism that opens this app's Settings. It declares a real
    /// ``Window("Settings", id: "settings")`` and no SwiftUI ``Settings``
    /// scene, so ``@Environment(\.openSettings)`` — which this view used to
    /// hold — targets a scene that does not exist and does nothing at all.
    /// That is the worst place for a dead button: the failure card is on
    /// screen precisely because the user's first download or start already
    /// failed. See ``SettingsRouter`` for the ordering rule.
    @Environment(\.openWindow) private var openWindow
    @Bindable var coordinator: QuickstartCoordinator
    @Bindable var downloads: DownloadManager
    @Bindable var server: ServerManager

    /// The shared catalogue snapshot — every alias the engine knows, with its
    /// cached flag and size-on-disk. Named for its original single use (#1793:
    /// spotting a model already present) and kept for call-site stability; it
    /// has always carried the WHOLE catalogue, which is what lets in-window
    /// Browse all models read the real thing without a second load.
    var cachedModels: [ModelEntry] = []

    /// Whether that snapshot has actually landed.
    ///
    /// Load-bearing, not cosmetic: ``ModelCatalog/load(binary:hubCacheOverride:)``
    /// returns `[]` on failure, so an empty array is ambiguous on its own —
    /// "still loading" and "the subprocess failed" are different claims and
    /// neither is evidence that a model is absent. Together with the array this
    /// resolves ``catalogState``, which gates every Step 2 primary.
    ///
    /// Defaults to `true` so the many call sites that hand over a ready-made
    /// fixture keep rendering a settled list; ContentView passes the real flag.
    var catalogLoaded: Bool = true

    /// This Mac, read once per view lifetime. Same pattern and same source as
    /// ``ModelPickerBar`` and ``SettingsModelManagementPanel``, so onboarding's
    /// "won't fit" decision is the one the rest of the app already makes.
    @State private var hardware: MacHardware = .detect()

    /// The alias a Step 3 cancellation has already been requested for.
    ///
    /// Keyed by alias rather than held as a `Bool` so it cannot leak across
    /// models: a user who cancels one download, goes back, picks a different
    /// model and starts again must get a live Cancel control on the new pull,
    /// not a control suppressed by the previous one's request.
    ///
    /// View state, not coordinator state, on purpose — it is about one
    /// on-screen control's press, and it must NOT survive a re-mount the way
    /// the download itself does. The authoritative record of what happened to
    /// the transfer is the job's own status.
    @State private var cancelRequestedAlias: String?

    /// Callback the parent supplies for "Skip for now". The parent
    /// dismisses the Quickstart surface for the current session (without
    /// flipping the persisted flag) so the existing picker becomes visible.
    /// Lifted out as a closure so this view can stay agnostic of how the
    /// parent toggles its own state.
    ///
    /// This is ONLY the skip path. "Browse all models" used to share it, on
    /// the theory that both mean "let me look around first" — but they differ
    /// in exactly the thing that matters: skipping accepts whatever the app
    /// picks, browsing is a request to choose. Sharing the closure made the
    /// link a dismiss button that dropped the user's selection and left them
    /// on the alphabetical fallback (#1653). Browsing is handled in this view
    /// now, by ``browseAllModels()``.
    var onSkip: () -> Void

    /// Callback the parent supplies for seeding the welcome message
    /// into the intended chat session. Closing over ``ChatViewModel``
    /// from outside keeps Quickstart from importing the entire chat
    /// module surface. Returns ``true`` when the message actually landed
    /// (a session existed and the append succeeded) so the coordinator
    /// can tell "seeded" from "still owed" (codex r2 MAJOR).
    ///
    /// Called from exactly one place — the Start chatting transaction —
    /// so the welcome is a consequence of the user finishing setup rather
    /// than of a subprocess reporting a listening port.
    var onSeedWelcome: () -> Bool

    /// The parent's half of the Start chatting transaction, run ONLY after
    /// ``QuickstartCoordinator/confirmStartChatting(seedWelcome:)`` reports
    /// it performed the state change.
    ///
    /// Split this way so the two halves cannot disagree about whether
    /// completion happened: the coordinator owns seeding, persistence and
    /// dismissal; the parent owns routing to Chat, the accessibility
    /// announcement and composer focus — none of which this view has the
    /// environment to do, and all of which must fire exactly once.
    var onCompleted: () -> Void = {}

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

    /// The Direction D two-plane shell (Paper 05.1.A).
    ///
    /// Every state is a rail beside a canvas. The rail says where you are and
    /// what this Mac is; the canvas says what is being decided and what the
    /// action will do. Which rail depends on whether the user can act: D1 —
    /// the mineral setup rail — asks, and D2 — the graphite subject rail —
    /// waits. Nothing here decides *what* the state is; that is still
    /// ``coordinator.phase`` and ``coordinator.stage``, unchanged.
    private var content: some View {
        GeometryReader { proxy in
            let layout = OnboardingLayout.resolve(width: proxy.size.width)
            Group {
                if layout.isCompact {
                    // Paper 05.1.E. The rail rotates rather than shrinking:
                    // a 300pt column beside a 720pt window leaves the canvas
                    // around 400pt, which the model cards cannot use.
                    VStack(spacing: 0) {
                        compactRailPlane
                        canvasPlane
                    }
                } else {
                    HStack(spacing: 0) {
                        railPlane
                        canvasPlane
                    }
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .topLeading)
            .environment(\.onboardingLayout, layout)
        }
        // The surface fills whatever the window gives it. There is no maximum:
        // setup IS the window now, not a panel inside one.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The rotated rail, D2 band included.
    @ViewBuilder
    private var compactRailPlane: some View {
        if let lifecycle = subjectLifecycle {
            let job = downloads.job(for: coordinator.selection.alias)
            @Bindable var progress = job?.progress ?? DownloadProgress()
            OnboardingCompactSubjectBand(
                lifecycle: coordinator.phase == .downloading
                    ? Self.downloadLifecycleName(progress: progress)
                    : lifecycle,
                identity: coordinator.selection.alias,
                fraction: coordinator.phase == .downloading ? progress.progressFraction : nil,
                bytesLine: Self.subjectBytesLine(job: job)
            )
        } else {
            OnboardingCompactRail(step: coordinator.step)
        }
    }

    /// D2 for the blocking lifecycle states, D1 for everything else.
    ///
    /// A memory warning parked on top of ``Phase/starting`` is a DECISION, not
    /// a wait, so it keeps the D1 rail — the user has something to answer and
    /// the rail must keep showing them the step and the machine facts that
    /// answer it with.
    @ViewBuilder
    private var railPlane: some View {
        if let lifecycle = subjectLifecycle {
            let job = downloads.job(for: coordinator.selection.alias)
            @Bindable var progress = job?.progress ?? DownloadProgress()
            OnboardingSubjectRail(
                lifecycle: coordinator.phase == .downloading
                    ? Self.downloadLifecycleName(progress: progress)
                    : lifecycle,
                identity: coordinator.selection.alias,
                fraction: coordinator.phase == .downloading ? progress.progressFraction : nil,
                bytesLine: Self.subjectBytesLine(job: job),
                rateLine: Self.subjectRateLine(job: job),
                stepLabel: "STEP \(coordinator.step.displayNumber) OF \(QuickstartCoordinator.Step.total)",
                stepName: coordinator.step.railTitle.localizedUppercase
            )
        } else {
            OnboardingSetupRail(
                step: coordinator.step,
                hardware: hardware,
                // Free space joins the rail from Step 2 onward, where it
                // starts bearing on the decision (Paper 05.1 frames 03/04).
                freeSpace: coordinator.step == .welcome
                    ? nil
                    : freeBytesProbe().map { Self.wholeGB(Double($0) / Double(1 << 30)) }
            )
        }
    }

    /// The lifecycle name for the D2 rail, or `nil` when this state is a D1
    /// decision. Derived from the real download phase so "PREPARING" is only
    /// claimed while the pull genuinely has nothing to report yet.
    private var subjectLifecycle: String? {
        switch coordinator.phase {
        case .downloading:
            let job = downloads.job(for: coordinator.selection.alias)
            return Self.downloadLifecycleName(phase: job?.progress.phase)
        case .starting:
            guard QuickstartView.memoryWarningToPresent(
                phase: coordinator.phase,
                pending: server.pendingMemoryWarning,
                selectionAlias: coordinator.selection.alias
            ) == nil else { return nil }
            return "STARTING"
        case .idle, .lowDiskWarning, .ready, .dismissed, .failed:
            return nil
        }
    }

    /// Which of Paper's two download lifecycles a job is in.
    ///
    /// Pure so "a pull with nothing observed yet must not be called
    /// DOWNLOADING" can be pinned. Paper draws these as separate states (09
    /// and 14) and the difference is real: one has bytes to report, the other
    /// has a request that landed and no transfer yet.
    static func downloadLifecycleName(phase: DownloadProgress.Phase?) -> String {
        switch phase {
        case .downloading, .fetching:
            return "DOWNLOADING"
        case .idle, .preparing, .warmingUp, nil:
            return "PREPARING"
        }
    }

    /// The mirror's aggregate byte heartbeat is a stronger signal than its
    /// coarse phase. It deliberately does not mutate ``phase`` (the next file
    /// event owns that state), so a rail that looked only at the enum could
    /// claim PREPARING while measured bytes were already moving.
    static func downloadLifecycleName(progress: DownloadProgress) -> String {
        progress.hasDiskObservation ? "DOWNLOADING" : downloadLifecycleName(phase: progress.phase)
    }

    /// "271 MB / 633 MB" — only once the byte monitor has observed real disk
    /// growth against a known total. Never a synthesised denominator.
    static func subjectBytesLine(job: DownloadManager.Job?) -> String? {
        guard let progress = job?.progress,
              progress.hasDiskObservation,
              let bytes = progress.bytesDownloaded
        else { return nil }
        if let total = progress.totalBytes, total > 0 {
            return "\(DownloadProgress.formatBytes(bytes)) / \(DownloadProgress.formatBytes(total))"
        }
        // A measured total is discarded when the mirror proves it wrong
        // (done > total). Keep the truthful numerator visible rather than
        // regressing to an indeterminate track while bytes are flowing.
        return "\(DownloadProgress.formatBytes(bytes)) downloaded"
    }

    /// "4.4 MB/s · 2 min left" — rate first, and the ETA appended only when
    /// ``DownloadProgress`` has measured one. Paper 05.1.A forbids an ETA
    /// before bytes move, so there is deliberately no pre-download branch.
    static func subjectRateLine(job: DownloadManager.Job?) -> String? {
        guard let progress = job?.progress, let speed = progress.bytesPerSecond, speed > 0
        else { return nil }
        var parts = [DownloadProgress.formatSpeed(bytesPerSecond: speed)]
        if let eta = progress.etaText { parts.append(eta) }
        return parts.joined(separator: " · ")
    }

    /// The right-hand plane: one composition per state.
    @ViewBuilder
    private var canvasPlane: some View {
        switch coordinator.phase {
        case .idle:
            switch coordinator.stage {
            case .welcome:     welcomeStep
            case .chooseModel: chooseModelStep
            }
        case .lowDiskWarning(let freeBytes, let requiredBytes):
            OnboardingCenteredCanvas {
                lowDiskCard(freeBytes: freeBytes, requiredBytes: requiredBytes)
            }
        case .downloading:
            OnboardingCenteredCanvas(trailing: 72) { downloadingCard }
        case .ready:
            // Onboarding V3: readiness is a destination, not a hand-off.
            // The surface stays here until the user confirms.
            OnboardingCenteredCanvas { readyCard }
        case .dismissed:
            // Terminal — the parent's visibility predicate has already
            // dropped this surface. A one-frame race can still render here,
            // so paint nothing rather than a step that is no longer true.
            Color.clear
        case .starting:
            // #1503: a serve handed off from Quickstart funnels through
            // ServerManager's pre-load memory guard. On a Mac under heavy
            // memory pressure the guard PARKS the load on
            // ``server.pendingMemoryWarning`` and returns WITHOUT changing
            // ``server.state``. The shared confirmation ``.alert`` is
            // anchored on ContentView — BEHIND this full-window onboarding
            // sheet — so it can never present: the sheet waits for a serve
            // that will never arrive, and the guard waits for an answer the
            // user can't reach. A hard deadlock the user sees as a permanent
            // "Starting…". Surface the SAME decision inside the sheet, where
            // it is reachable. (ContentView suppresses its covered alert for
            // exactly this case via the same predicate.)
            if let warning = QuickstartView.memoryWarningToPresent(
                phase: coordinator.phase,
                pending: server.pendingMemoryWarning,
                selectionAlias: coordinator.selection.alias
            ) {
                OnboardingCenteredCanvas { memoryWarningCard(warning) }
            } else {
                // .ready is transitional — the parent swaps to ChatView, but
                // a one-frame race can land here; the starting copy is a calm
                // fallback so the user never sees a blank pane.
                OnboardingCenteredCanvas(trailing: 72) { startingCard }
            }
        case .failed(let message, _):
            OnboardingCenteredCanvas { failedCard(message: message) }
        }
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
    /// Step 1 — the privacy claim, at display scale (Paper 05.1 state 01).
    ///
    /// The headline is the product's one substantive promise rather than a
    /// marketing line, and the mascot carries the brand so the type does not
    /// have to. Left-aligned against the canvas's deep leading margin: the
    /// asymmetry is the composition, and a centred hero would read as a
    /// landing page rather than a setup screen.
    @ViewBuilder
    private var welcomeStep: some View {
        OnboardingCenteredCanvas {
            VStack(alignment: .leading, spacing: 0) {
                CheetahLogo(size: 156)
                    .padding(.bottom, 30)

                OnboardingDisplayTitle(text: "Nothing you type\nleaves this Mac.", size: 52)

                VStack(alignment: .leading, spacing: 9) {
                    Text("Models run locally on Apple Silicon. You download one model "
                         + "once — after that it works with no network at all.")
                    Text("No account, no subscription.")
                }
                .scaledSystemFont(17, relativeTo: .title3)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
                .padding(.top, 26)

                // Setup was entered on an earlier launch and never finished.
                // Say so, and say what is and is not carried over — nothing
                // is, and a screen that stayed silent about it while offering
                // to "continue" would let the user assume a download picked up
                // where it left off. Paper 05.1 state 18: "The copy promises a
                // fresh download, never a resume."
                if coordinator.isResumingIncompleteSetup {
                    Text("Setup didn't finish last time. Nothing was carried over — "
                         + "choose a model and it downloads from here.")
                        .scaledSystemFont(13)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
                        .padding(.top, 18)
                        .accessibilityIdentifier("Quickstart.ResumeNotice")
                }

                OnboardingActionLane {
                    Button {
                        coordinator.advanceToChooseModel()
                    } label: {
                        Text(Self.welcomePrimaryTitle(resuming: coordinator.isResumingIncompleteSetup))
                    }
                    .buttonStyle(.onboardingPrimary)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("Quickstart.GetStarted")
                    .accessibilityLabel(
                        coordinator.isResumingIncompleteSetup
                            ? "Continue setup — choose your first model"
                            : "Get started — choose your first model"
                    )

                    // #549 (§16 wayfinding): the hero must answer "how do I
                    // get out?" — before this the only exit was the "Browse
                    // all models" link on step 2, trapping a first-run user
                    // sitting on step 1. A low-emphasis Skip drops straight
                    // into the app, and `.cancelAction` makes Esc leave
                    // onboarding.
                    //
                    // This is the app's one genuine "dismiss onboarding"
                    // control. "Browse all models" on step 2 shared it until
                    // #1653; it does not any more, because a user asking to
                    // see the catalogue has not asked to leave setup.
                    Button("Skip for now") {
                        onSkip()
                    }
                    .buttonStyle(.onboardingQuiet)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("Quickstart.Skip")
                    .accessibilityLabel("Skip onboarding and go to the app")
                }
                .padding(.top, 44)
            }
        }
    }

    /// The welcome primary's verb.
    ///
    /// Both labels lead to exactly the same place — the model chooser — and
    /// that is the point: the difference is what the app is willing to CLAIM
    /// about the user, not what the button does. "Get started" told a
    /// returning, half-finished user they had not started, which is the one
    /// thing the screen knows to be false. Pure so the pairing can be pinned
    /// without a SwiftUI host (Paper 05.1 state 18 — "Primary Continue setup →
    /// the model chooser").
    static func welcomePrimaryTitle(resuming: Bool) -> String {
        resuming ? "Continue setup" : "Get started"
    }

    // MARK: - Step 2 · Choose a model

    /// Step 2's router over its five micro-stages (Paper 05.2.B).
    ///
    /// Every branch renders through the one shared ``step2Scaffold``, so the
    /// rail cannot drift off `Step 2 of 4` by a screen forgetting which step it
    /// belongs to — the mistake the old three-step model made twice. The rail
    /// itself is drawn once, by ``railPlane``, from ``coordinator.step``.
    @ViewBuilder
    private var chooseModelStep: some View {
        Group {
            switch coordinator.step2Stage {
            case .checkingHardware:
                checkingHardwareStep
            case .findingFit:
                findingFitStep
            case .choosing:
                recommendedShortlistStep
            case .browsing:
                browseAllStep
            case .reviewing:
                reviewDownloadStep
            }
        }
        // Settle the two pre-shortlist micro-stages against the real catalogue
        // signal rather than a timer. Re-fires when the snapshot lands.
        .task(id: catalogLoaded) {
            coordinator.resolveRecommendationLoading(catalogLoaded: catalogLoaded)
        }
    }

    /// The canvas + footer lane every Step 2 micro-stage shares.
    ///
    /// One canvas, and exactly one footer lane holding at most one Back and one
    /// primary. No breadcrumb: a trail would imply a depth this flow does not
    /// have. The kicker and title live in the body, because Direction D places
    /// them differently for the column layouts (beside the content) and for the
    /// catalogue (above it).
    @ViewBuilder
    private func step2Scaffold<Body: View, Footer: View>(
        trailing: CGFloat = OnboardingD.canvasTrailing,
        @ViewBuilder body: @escaping () -> Body,
        @ViewBuilder footer: @escaping () -> Footer
    ) -> some View {
        OnboardingCanvas(trailing: trailing) {
            // One geometry for every state: the principal group fills the
            // canvas and centres in it, the action lane is anchored beneath.
            OnboardingCanvasLayout(principal: body) {
                footer().padding(.top, 28)
            }
        }
    }

    /// Paper's Step 2 body: a fixed heading column beside the live list.
    ///
    /// The asymmetry is load-bearing. The heading column is where the branch
    /// names itself and — on Review download — where the cost is stated, while
    /// the list stays in exactly the same place across all three micro-stages
    /// so switching between them never moves the rows under the pointer.
    private func step2Columns<Aside: View, Content: View>(
        kicker: String,
        title: String,
        subtitle: String?,
        @ViewBuilder aside: @escaping () -> Aside,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        OnboardingStepColumns(
            kicker: Self.microStageKicker(kicker),
            title: title,
            subtitle: subtitle,
            aside: aside,
            content: content
        )
    }

    /// `STEP 2 OF 4 · <MICRO-STAGE>`. Pure so a test can pin the format —
    /// specifically that the count comes from ``QuickstartCoordinator/Step/total``
    /// and that nothing sub-numbers it.
    static func microStageKicker(_ stageName: String) -> String {
        "STEP \(QuickstartCoordinator.Step.chooseModel.displayNumber) "
            + "OF \(QuickstartCoordinator.Step.total) · \(stageName)"
    }

    // MARK: - 2a / 2b — hardware detection and recommendation loading

    /// 2a — reading this Mac (Paper 05.1 state 02).
    ///
    /// The chip and the unified memory are synchronous sysctl reads and are
    /// therefore already known when this draws; the free-space probe is the
    /// one value that genuinely arrives later, so it is the only one shown as
    /// a placeholder. Nothing here is a scanning animation with an invented
    /// duration — Paper forbids that explicitly, and the honest rendering of a
    /// read that has already happened is the answer.
    @ViewBuilder
    private var checkingHardwareStep: some View {
        transientStep(
            kicker: "CHECKING THIS MAC",
            title: "Reading this Mac…",
            subtitle: "Rapid-MLX checks the chip, the unified memory and the free "
                + "space on the volume that holds your Hugging Face cache. "
                + "Nothing is uploaded — the read is local.",
            identifier: "Quickstart.Step2.CheckingHardware"
        ) {
            VStack(spacing: 0) {
                transientFact("Chip") {
                    Text(hardware.brandString)
                        .scaledSystemFont(13, design: .monospaced)
                        .foregroundStyle(RapidTheme.textPrimary)
                }
                transientFact("Unified memory") {
                    Text(Self.wholeGB(hardware.physicalRAMGB))
                        .scaledSystemFont(13, design: .monospaced)
                        .foregroundStyle(RapidTheme.textPrimary)
                }
                transientFact("Free on cache volume") {
                    if let free = freeBytesProbe() {
                        Text(Self.formatBytesForBanner(free))
                            .scaledSystemFont(13, design: .monospaced)
                            .foregroundStyle(RapidTheme.textPrimary)
                    } else {
                        OnboardingSkeleton(width: 84)
                    }
                }
            }
            .frame(maxWidth: 460, alignment: .leading)
        }
    }

    /// 2b — matching models to this Mac (Paper 05.1 state 03).
    ///
    /// The genuinely asynchronous one: the shortlist cannot say which models
    /// are already on disk until ``ModelCatalog/load`` answers. Only the
    /// "already on this Mac" group is unknown, so only it is drawn as
    /// placeholder rows — the ladder below it is a fixed list and is never
    /// pretended to be computed.
    @ViewBuilder
    private var findingFitStep: some View {
        transientStep(
            kicker: "FINDING THE BEST FIT",
            title: "Matching models to \(Self.wholeGB(hardware.physicalRAMGB))…",
            subtitle: "Rapid-MLX is reading the model catalogue to see which models "
                + "are already on this Mac and how much each one would download. "
                + "The short list below is fixed; only the “already downloaded” "
                + "part depends on this read.",
            identifier: "Quickstart.Step2.FindingFit"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                OnboardingGroupLabel(text: "ALREADY ON THIS MAC")
                ForEach(0..<2, id: \.self) { _ in
                    HStack(spacing: 14) {
                        OnboardingSkeleton(width: 32, height: 32)
                        VStack(alignment: .leading, spacing: 6) {
                            OnboardingSkeleton(width: 168, height: 10)
                            OnboardingSkeleton(width: 116, height: 9)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 18)
                    .frame(height: OnboardingD.rowHeight)
                    .background(
                        RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                            .fill(RapidTheme.surfaceRaised)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                            .strokeBorder(RapidTheme.hairline, lineWidth: 1)
                    )
                }
            }
        }
    }

    /// The chrome the two pre-shortlist micro-stages share. The footer is
    /// present but disabled — the user can still go Back, and there is nothing
    /// yet to progress to.
    @ViewBuilder
    private func transientStep<Content: View>(
        kicker: String,
        title: String,
        subtitle: String,
        identifier: String,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        step2Scaffold {
            step2Columns(
                kicker: kicker,
                title: title,
                subtitle: subtitle
            ) {
                EmptyView()
            } content: {
                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier(identifier)
            }
        } footer: {
            OnboardingStepFooter(
                primaryTitle: OnboardingModelSelection.disabledPrimary.title,
                primaryEnabled: false,
                onBack: { coordinator.backToWelcome() },
                onPrimary: {}
            )
        }
    }

    @ViewBuilder
    private func transientFact<Value: View>(
        _ label: String,
        @ViewBuilder value: () -> Value
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .scaledSystemFont(13)
                .foregroundStyle(RapidTheme.textSecondary)
            Spacer(minLength: 12)
            value()
        }
        .padding(.vertical, 11)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: - 2c — the recommended shortlist

    /// What the Step 2 heading column says about the current selection.
    ///
    /// Paper draws the chooser three ways, and which one shows is a pure
    /// function of the pick (05.1 states 04, 05, 06). The list on the right is
    /// identical in all three — only the narrative beside it changes, because
    /// what the user needs explained differs completely between "here is the
    /// recommendation", "you have chosen something bigger, here is the cost"
    /// and "this one is already downloaded".
    enum SelectionNarrative: Equatable {
        /// State 04 — the default. The starter, the low-memory fallback, or
        /// anything the other two cases do not claim.
        case chooseFirst
        /// State 05 — a bigger model is picked, so the cost is compared.
        case biggerAndCost
        /// State 06 — the pick is already in the shared cache.
        case alreadyHere

        var title: String {
            switch self {
            case .chooseFirst:   return "Choose your\nfirst model"
            case .biggerAndCost: return "Bigger, and\nwhat it costs"
            case .alreadyHere:   return "One is already\nhere."
            }
        }
    }

    /// Resolve the narrative from the selection alone.
    ///
    /// Pure, and ordered: cached-ness wins over size, because "you already
    /// have this" is the more useful thing to say about a 9B that happens to
    /// be on disk than "here is what it costs to download". A trade-up only
    /// takes the comparison branch when there is something to compare it
    /// WITH — a lone trade-up would produce a one-column table, which is a
    /// statement dressed as a comparison.
    static func selectionNarrative(
        alias: String,
        cachedModels: [ModelEntry],
        comparableTradeUps: [QuickstartModelChoice]
    ) -> SelectionNarrative {
        if canStartWithoutDownload(alias: alias, cachedModels: cachedModels) {
            return .alreadyHere
        }
        let isTradeUp = comparableTradeUps.contains { $0.alias == alias }
        if isTradeUp, comparableTradeUps.count >= 2 {
            return .biggerAndCost
        }
        return .chooseFirst
    }

    /// The subtitle under the narrative title.
    static func selectionSubtitle(
        _ narrative: SelectionNarrative,
        hardware: MacHardware
    ) -> String {
        switch narrative {
        case .chooseFirst:
            return "Start small — you can download bigger models anytime in Settings."
        case .biggerAndCost:
            return "The difference is download size and how much memory the model "
                + "holds while it runs, against this Mac's \(wholeGB(hardware.physicalRAMGB))."
        case .alreadyHere:
            return "Another MLX app already downloaded this model into the shared "
                + "Hugging Face cache. Picking it skips the download entirely."
        }
    }

    /// The comparison columns for ``SelectionNarrative/biggerAndCost``.
    ///
    /// Every figure is an existing reading: ``sizeText`` for the download,
    /// ``ModelSizing/estimate(alias:)`` for the memory, and
    /// ``ModelSizing/classify(_:on:)`` — the same classification the primary is
    /// gated on — for the fit. No benchmark, no quality claim.
    static func comparisonColumns(
        selection: String,
        tradeUps: [QuickstartModelChoice],
        hardware: MacHardware
    ) -> [OnboardingComparisonTable.Column] {
        tradeUps.map { choice in
            let footprint = ModelSizing.estimate(alias: choice.alias)
            let fit = ModelSizing.classify(footprint, on: hardware)
            return OnboardingComparisonTable.Column(
                title: comparisonColumnTitle(for: choice),
                isPicked: choice.alias == selection,
                download: sizeText(for: choice),
                memory: footprint.totalGB > 0 ? "≈ \(preciseGB(footprint.totalGB))" : "Unknown",
                fit: fitText(fit),
                fitIsWarning: fit != .recommended
            )
        }
    }

    /// The short column header — the parameter count, which is the axis the
    /// user is actually comparing. Falls back to the display name when the
    /// alias carries no size token.
    static func comparisonColumnTitle(for choice: QuickstartModelChoice) -> String {
        if let params = ModelSizing.estimate(alias: choice.alias).paramsBillions {
            let whole = params.rounded()
            let text = abs(params - whole) < 0.05
                ? String(Int(whole))
                : String(format: "%.1f", params)
            return "\(text)B"
        }
        return choice.displayName
    }

    /// Plain words for a ``ModelSizing/Fit``. Read from the classification, so
    /// the table and the disabled primary can never disagree.
    static func fitText(_ fit: ModelSizing.Fit) -> String {
        switch fit {
        case .recommended: return "Comfortable"
        case .borderline:  return "Tight"
        case .tooBig:      return "Won't fit"
        }
    }

    /// The recommended shortlist: models already on this Mac, the starter, an
    /// honest low-memory fallback, bigger trade-ups, a catalogue pick carried
    /// back as YOUR PICK, and the link into the in-window catalogue.
    @ViewBuilder
    private var recommendedShortlistStep: some View {
        let list = shortlist
        let primary = primary(for: .shortlist)
        let narrative = Self.selectionNarrative(
            alias: coordinator.selection.alias,
            cachedModels: cachedModels,
            comparableTradeUps: list.tradeUps
        )
        step2Scaffold {
            step2Columns(
                kicker: "CHOOSE A MODEL",
                title: narrative.title,
                subtitle: Self.selectionSubtitle(narrative, hardware: hardware)
            ) {
                // Paper 05.1 state 05: the comparison is part of the heading
                // column, not a second panel — it is the explanation of the
                // pick, and it sits where every other explanation sits.
                if narrative == .biggerAndCost {
                    OnboardingComparisonTable(
                        columns: Self.comparisonColumns(
                            selection: coordinator.selection.alias,
                            tradeUps: list.tradeUps,
                            hardware: hardware
                        ),
                        fitLabel: "Fit on \(Self.wholeGB(hardware.physicalRAMGB))"
                    )
                    .padding(.top, 26)
                }
            } content: {
            OnboardingIntrinsicColumn {
                VStack(alignment: .leading, spacing: 10) {
                    if !list.cached.isEmpty {
                        OnboardingGroupLabel(text: "ALREADY ON THIS MAC")
                            .padding(.top, 14)
                        ForEach(list.cached) { entry in
                            let choice = Self.choice(forCachedModel: entry)
                            QuickstartCompactCard(
                                choice: choice,
                                selected: coordinator.selection.alias == entry.alias,
                                sizeText: entry.sizeOnDisk ?? "",
                                isCached: true,
                                onActivate: { activatePrimary(in: .shortlist) }
                            ) { coordinator.select(choice) }
                            .accessibilityIdentifier("Quickstart.CachedModel.\(entry.alias)")
                        }
                    }

                    ForEach(list.starters) { choice in
                        QuickstartRecommendedCard(
                            choice: choice,
                            selected: coordinator.selection.alias == choice.alias,
                            sizeText: Self.sizeText(for: choice),
                            onActivate: { activatePrimary(in: .shortlist) }
                        ) { coordinator.select(choice) }
                    }

                    if !list.lowMemory.isEmpty {
                        OnboardingGroupLabel(text: "NEED THE LIGHTEST OPTION?")
                            .padding(.top, 14)
                        ForEach(list.lowMemory) { choice in
                            QuickstartLowMemoryCard(
                                choice: choice,
                                selected: coordinator.selection.alias == choice.alias,
                                sizeText: Self.sizeText(for: choice),
                                onActivate: { activatePrimary(in: .shortlist) }
                            ) { coordinator.select(choice) }
                        }
                    }

                    if !list.tradeUps.isEmpty {
                        OnboardingGroupLabel(text: "OR PICK A BIGGER ONE")
                            .padding(.top, 14)
                        ForEach(list.tradeUps) { choice in
                            let cached = Self.cachedModel(
                                alias: choice.alias,
                                cachedModels: cachedModels
                            )
                            QuickstartCompactCard(
                                choice: choice,
                                selected: coordinator.selection.alias == choice.alias,
                                sizeText: cached?.sizeOnDisk ?? Self.sizeText(for: choice),
                                isCached: cached != nil,
                                onActivate: { activatePrimary(in: .shortlist) }
                            ) { coordinator.select(choice) }
                        }
                    }

                    // Approved default D2. A model chosen in the catalogue that
                    // the shortlist does not natively list comes back with the
                    // user rather than vanishing — otherwise Back lands them on
                    // a list that visibly disagrees with the footer, which
                    // reads as "my choice was ignored".
                    if let pick = list.yourPick {
                        OnboardingGroupLabel(text: "YOUR PICK")
                            .padding(.top, 14)
                        let choice = Self.choice(forCatalogEntry: pick)
                        QuickstartCompactCard(
                            choice: choice,
                            selected: coordinator.selection.alias == pick.alias,
                            sizeText: Self.rowSizeText(for: pick),
                            isCached: pick.cached,
                            onActivate: { activatePrimary(in: .shortlist) }
                        ) { coordinator.select(choice) }
                        .accessibilityIdentifier("Quickstart.YourPick.\(pick.alias)")
                    }

                    Button("Browse all models →") {
                        browseAllModels()
                    }
                    .buttonStyle(.onboardingLink)
                    .padding(.top, 2)
                    .accessibilityIdentifier("Quickstart.BrowseAll")
                    .accessibilityLabel("Browse all models")
                }
                .padding(.trailing, 2)
            }
            }
        } footer: {
            OnboardingStepFooter(
                primaryTitle: primary.title,
                primaryEnabled: primary.isEnabled,
                onBack: { coordinator.backToWelcome() },
                onPrimary: { activatePrimary(in: .shortlist) }
            )
        }
    }

    // MARK: - 2d — Browse all models, in window

    /// The real catalogue on the setup canvas (Paper 05.2.C).
    ///
    /// It does not open Settings, does not open a second window, does not
    /// present a sheet and does not dismiss onboarding — the whole point of
    /// 05.2 is that browsing is a move inside Step 2, not a way out of it.
    @ViewBuilder
    private var browseAllStep: some View {
        let entries = visibleCatalogEntries
        let heading = ModelCacheActions.listHeading(
            filter: coordinator.catalogFilter,
            query: coordinator.catalogQuery,
            visibleCount: entries.count,
            totalCount: Self.onboardingCatalogModels(cachedModels).count
        )
        let primary = primary(for: .catalogue)
        // The catalogue is the one Step 2 stage that does NOT use the fixed
        // heading column: 175 rows carrying an alias, a repo, badges and a size
        // need the full canvas width, so the heading becomes a band across the
        // top instead (Paper 05.2.C).
        step2Scaffold(trailing: OnboardingD.canvasTrailing) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .bottom, spacing: OnboardingD.columnGap) {
                    VStack(alignment: .leading, spacing: 0) {
                        OnboardingKicker(text: Self.microStageKicker("BROWSE ALL MODELS"))
                            .accessibilityIdentifier("Quickstart.Step2.Kicker")
                            .padding(.bottom, 16)
                        OnboardingDisplayTitle(text: "All models")
                    }
                    Spacer(minLength: 24)
                    Text("Everything rapid-mlx can serve on this Mac. "
                         + "Your shortlist pick stays selected.")
                        .scaledSystemFont(15, relativeTo: .callout)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(width: 330, alignment: .leading)
                }

                catalogToolbar(heading: heading)
                    .padding(.top, 26)

                catalogBody(entries: entries)
                    .padding(.top, 20)
            }
        } footer: {
            OnboardingStepFooter(
                primaryTitle: primary.title,
                primaryEnabled: primary.isEnabled,
                backTitle: "← Back to recommended models",
                backAccessibilityLabel: "Back to recommended models",
                onBack: { returnToRecommendedModels() },
                onPrimary: { activatePrimary(in: .catalogue) }
            )
        }
    }

    /// Search, sort and filter. All three write straight to the coordinator so
    /// they survive Review download and a SwiftUI re-mount.
    @ViewBuilder
    private func catalogToolbar(heading: ModelCacheActions.ListHeading) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                HStack(spacing: 9) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 13))
                        .foregroundStyle(RapidTheme.textSecondary)
                        .accessibilityHidden(true)
                    TextField(
                        "Search models or Hugging Face repo",
                        text: $coordinator.catalogQuery
                    )
                    .textFieldStyle(.plain)
                    .scaledSystemFont(13)
                    // Escape priority 1. A search field holding text owns the
                    // key and clears itself; empty, it declines so the event
                    // reaches the footer's Back at priority 3. Without this,
                    // one Escape would leave the catalogue with the user's
                    // query still on screen behind them.
                    .onKeyPress(.escape) {
                        guard !coordinator.catalogQuery.isEmpty else { return .ignored }
                        coordinator.catalogQuery = ""
                        return .handled
                    }
                    .accessibilityIdentifier("Quickstart.BrowseAll.Search")
                    .accessibilityLabel("Search models")
                }
                .padding(.horizontal, 13)
                .frame(height: 36)
                .background(
                    RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                        .fill(RapidTheme.surfaceRaised)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                        .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
                )

                RapidSegmentedControl(
                    selection: $coordinator.catalogFilter,
                    options: ModelCacheActions.FilterMode.allCases.map {
                        .init(value: $0, title: $0.displayLabel)
                    },
                    accessibilityLabel: "Filter"
                )
                .fixedSize()
                .accessibilityIdentifier("Quickstart.BrowseAll.Filter")

                Menu {
                    ForEach(ModelCacheActions.SortOrder.allCases) { order in
                        Button {
                            coordinator.catalogSort = order
                        } label: {
                            if coordinator.catalogSort == order {
                                Label(order.displayLabel, systemImage: "checkmark")
                            } else {
                                Text(order.displayLabel)
                            }
                        }
                        // Each order is its own control: the golden-flow
                        // harness reaches menu items by identifier, so without
                        // one per row it can open the menu but never choose.
                        .accessibilityIdentifier("Quickstart.BrowseAll.Sort.\(order.rawValue)")
                    }
                } label: {
                    Text(coordinator.catalogSort.displayLabel)
                        .scaledSystemFont(12)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .padding(.horizontal, 13)
                .frame(height: 36)
                .background(
                    RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                        .fill(RapidTheme.surfaceRaised)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                        .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
                )
                // The scene tints app-wide amber and a borderless Menu's label
                // reads the TINT, not the foreground style — without this the
                // utility control renders as the page's primary action.
                .tint(nil)
                .foregroundStyle(RapidTheme.textPrimary)
                .accessibilityIdentifier("Quickstart.BrowseAll.SortMenu")
                .accessibilityLabel("Sort")
            }

            // The list's own header lane. Column captions live here rather
            // than on every row, and the trailing spacer reserves the
            // selection glyph's width so SIZE lands over the size column.
            HStack(spacing: 14) {
                Text(heading.countText.localizedUppercase)
                    .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                    .tracking(OnboardingD.Tracking.groupLabel)
                    .monospacedDigit()
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityIdentifier("Quickstart.BrowseAll.Count")
                    .accessibilityLabel(heading.accessibilityLabel)
                Spacer(minLength: 8)
                Text("SIZE")
                    .scaledSystemFont(10, relativeTo: .caption2, design: .monospaced)
                    .tracking(OnboardingD.Tracking.badge)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .frame(width: OnboardingD.rowSizeSlot, alignment: .trailing)
                    .accessibilityHidden(true)
                Color.clear.frame(width: OnboardingD.selectionGlyph, height: 1)
            }
            .padding(.horizontal, 18)
        }
    }

    /// Which of the catalogue's five bodies to draw. The order matters: a list
    /// that has not spoken cannot be reported empty, and an empty CACHE is a
    /// different fact from a search that matched nothing.
    @ViewBuilder
    private func catalogBody(entries: [ModelEntry]) -> some View {
        switch catalogState {
        case .loading:
            catalogNotice(
                symbol: nil,
                title: "Loading models…",
                body: "Reading the catalogue from the engine.",
                identifier: "Quickstart.BrowseAll.Loading"
            )
        case .failed:
            catalogNotice(
                symbol: "exclamationmark.triangle",
                title: "Couldn't load the model catalogue",
                body: "The engine didn't return a model list. "
                    + "You can still start with a recommended model.",
                identifier: "Quickstart.BrowseAll.Error"
            )
        case .ready:
            if entries.isEmpty {
                if coordinator.catalogFilter == .cached
                    && coordinator.catalogQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    catalogNotice(
                        symbol: "internaldrive",
                        title: "No models on this Mac yet",
                        body: "Nothing has been downloaded. "
                            + "Switch to All to choose your first model.",
                        identifier: "Quickstart.BrowseAll.EmptyCache"
                    )
                } else {
                    catalogNotice(
                        symbol: "magnifyingglass",
                        title: "No models match",
                        body: "Try a different search, or clear it to see everything.",
                        identifier: "Quickstart.BrowseAll.NoResults"
                    )
                }
            } else {
                catalogList(entries: entries)
            }
        }
    }

    @ViewBuilder
    private func catalogNotice(
        symbol: String?,
        title: String,
        body: String,
        identifier: String
    ) -> some View {
        VStack(spacing: 10) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: 24, weight: .regular))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
            } else {
                ProgressView().controlSize(.small)
            }
            Text(title)
                .scaledSystemFont(15, relativeTo: .callout, weight: .semibold)
                .foregroundStyle(RapidTheme.textPrimary)
            Text(body)
                .scaledSystemFont(13)
                .foregroundStyle(RapidTheme.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.vertical, 48)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier)
        .accessibilityLabel("\(title). \(body)")
    }

    /// One flat scroller. No pagination, no lazy-load spinner, no nested
    /// scrollers — the catalogue is a few hundred rows at most.
    private var catalogScrollPosition: Binding<String?> {
        Binding(
            get: { coordinator.catalogScrollID },
            set: { alias in
                // SwiftUI may publish nil while a search/filter temporarily
                // removes the anchored row. Keep the last real alias so
                // clearing that filter can restore the user's position.
                if let alias { coordinator.rememberCatalogAnchor(alias) }
            }
        )
    }

    @ViewBuilder
    private func catalogList(entries: [ModelEntry]) -> some View {
        ScrollView {
            LazyVStack(spacing: 6) {
                ForEach(entries) { entry in
                    catalogRow(entry).id(entry.alias)
                }
            }
            .padding(.vertical, 2)
            .scrollTargetLayout()
        }
        // This is the actual visible scroll anchor, not merely the selected
        // row. It updates as the user scrolls and lives on the coordinator, so
        // Review/remount can restore it by stable alias rather than by pixels.
        .scrollPosition(id: catalogScrollPosition, anchor: .center)
        .accessibilityIdentifier("Quickstart.BrowseAll.List")
    }

    @ViewBuilder
    private func catalogRow(_ entry: ModelEntry) -> some View {
        let choice = Self.choice(forCatalogEntry: entry)
        let available = OnboardingModelSelection.isAvailable(alias: entry.alias, hardware: hardware)
        OnboardingCatalogRow(
            alias: entry.alias,
            subtitle: Self.catalogRowSubtitle(
                entry: entry,
                available: available,
                hardware: hardware
            ),
            sizeText: Self.rowSizeText(for: entry),
            selected: coordinator.selection.alias == entry.alias,
            isAvailable: available,
            badges: Self.catalogRowBadges(entry: entry, available: available),
            onActivate: { activatePrimary(in: .catalogue) }
        ) {
            coordinator.select(choice)
            coordinator.rememberCatalogAnchor(entry.alias)
        }
        // No `.disabled(!available)`. A WON'T FIT row is a live control that
        // selects, and whose Review is reachable and read-only (Paper 05.2.D):
        // "refusing to answer would be worse than answering". What used to be
        // enforced here — by making the row take no click at all — is now
        // enforced where it belongs, in the one derivation that decides what
        // the primary does: ``OnboardingModelSelection.primary`` hands back
        // ``Action/reviewIncompatible`` from a list and a DISABLED commit from
        // Review, so no path from this row reaches a download or a start.
        //
        // The muted treatment is unchanged and still driven by `isAvailable:`.
    }

    /// The second line of a catalogue row.
    ///
    /// Normally the Hugging Face repo — the fact that disambiguates two aliases
    /// of the same family. For a row this Mac cannot run, the repo is replaced
    /// by the reason, because at that point the reason is the more useful fact
    /// and the row is not a candidate anyway. Both come from data the catalogue
    /// and ``ModelSizing`` already hold; nothing here is a new claim.
    static func catalogRowSubtitle(
        entry: ModelEntry,
        available: Bool,
        hardware: MacHardware
    ) -> String {
        guard available else {
            let needed = ModelSizing.estimate(alias: entry.alias).totalGB
            guard needed > 0 else { return "Needs more memory than this Mac has" }
            return "Needs ≈ \(Self.wholeGB(needed)) · this Mac has \(Self.wholeGB(hardware.physicalRAMGB))"
        }
        let repo = entry.hfRepo ?? ""
        return repo.isEmpty ? entry.alias : repo
    }

    /// The badges a catalogue row carries.
    ///
    /// A fixed, closed set drawn from facts already on screen elsewhere:
    /// cached-ness from the catalogue snapshot, and runnability from the same
    /// ``ModelSizing`` classification the primary is already gated on. No new
    /// capability, benchmark or compatibility claim is introduced.
    static func catalogRowBadges(
        entry: ModelEntry,
        available: Bool
    ) -> [OnboardingCatalogRow.Badge] {
        var badges: [OnboardingCatalogRow.Badge] = []
        if entry.alias == QuickstartCoordinator.defaultChoice.alias {
            badges.append(.init(text: "RECOMMENDED", tone: .amber))
        }
        if !available {
            badges.append(.init(text: "WON'T FIT", tone: .error))
        } else if entry.cached {
            badges.append(.init(text: "ON THIS MAC", tone: .ready))
        }
        return badges
    }

    /// Whole gibibytes — for a MACHINE figure: this Mac's memory, free space,
    /// the fit threshold. One place so the catalogue row, the rail and the
    /// Review table cannot round the same number differently.
    static func wholeGB(_ value: Double) -> String {
        "\(Int(value.rounded())) GB"
    }

    /// One decimal — for a MODEL figure: an estimated footprint.
    ///
    /// The split is deliberate and matches Paper. A machine has 32 GB, flatly;
    /// an estimate of 8.7 GB rounded to "9 GB" loses the precision that makes
    /// two trade-ups distinguishable, and 5.9 vs 6 is exactly the comparison
    /// the user is being shown.
    static func preciseGB(_ value: Double) -> String {
        String(format: "%.1f GB", value)
    }

    /// Leave the catalogue, remembering where the user was.
    private func returnToRecommendedModels() {
        coordinator.backToRecommendedModels()
    }

    // MARK: - 2e — Review download

    /// Name the cost before spending it (Paper 05.2.D).
    ///
    /// Shows only what the product can truthfully state: identity, the size
    /// estimate the rest of the app quotes, whether it is already on disk, the
    /// memory it will occupy against this Mac's, the free space the pre-flight
    /// probe actually measured, and where the files land. No ETA, no benchmark
    /// claim, no invented compatibility verdict.
    ///
    /// The shortlist stays live on the right: picking a different row
    /// re-renders this detail in place, so the user can compare without
    /// leaving Step 2.
    @ViewBuilder
    private var reviewDownloadStep: some View {
        let alias = coordinator.selection.alias
        let cached = Self.cachedModel(alias: alias, cachedModels: cachedModels)
        let primary = primary(for: .review)
        // The same availability seam every other surface reads. Note it is
        // asked here, on arrival, rather than passed in by whatever opened the
        // screen: Review's companion list can change the selection in place,
        // and this must follow it.
        let runsHere = OnboardingModelSelection.isAvailable(alias: alias, hardware: hardware)
        step2Scaffold {
            step2Columns(
                kicker: "REVIEW DOWNLOAD",
                title: coordinator.selection.displayName,
                subtitle: Self.reviewSubtitle(cached: cached, runsHere: runsHere)
            ) {
                VStack(alignment: .leading, spacing: 0) {
                    if !runsHere {
                        OnboardingInlineNote(
                            text: Self.incompatibilityNote(alias: alias, hardware: hardware),
                            identifier: "Quickstart.Review.Incompatible"
                        )
                        .padding(.top, 20)
                    }

                    OnboardingFactTable(rows: Self.reviewFacts(
                        alias: alias,
                        cached: cached,
                        cachedModels: cachedModels,
                        hardware: hardware,
                        freeBytes: freeBytesProbe(),
                        runsHere: runsHere
                    ))
                    .padding(.top, 26)

                    // A download footnote frames a cost about to be paid, so it
                    // has nothing to say about a download that will not happen;
                    // the memory note takes the slot instead. One line under
                    // the table either way — the composition does not change.
                    if let footnote = runsHere
                        ? Self.reviewFootnote(alias: alias, cached: cached)
                        : Self.memoryHeadroomFootnote(hardware: hardware) {
                        Text(footnote)
                            .scaledSystemFont(12)
                            .foregroundStyle(RapidTheme.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, 18)
                            .accessibilityIdentifier("Quickstart.Review.Footnote")
                    }
                }
            } content: {
                reviewCompanionList
            }
        } footer: {
            OnboardingStepFooter(
                primaryTitle: primary.title,
                primaryEnabled: primary.isEnabled,
                backTitle: coordinator.reviewOrigin == .catalogue
                    ? "← Back to all models"
                    : "← Back to recommended models",
                backAccessibilityLabel: coordinator.reviewOrigin == .catalogue
                    ? "Back to all models"
                    : "Back to recommended models",
                primaryAccessibilityHint: runsHere
                    ? nil
                    : Self.incompatiblePrimaryHint(alias: alias, hardware: hardware),
                onBack: { coordinator.backFromReviewDownload() },
                onPrimary: { activatePrimary(in: .review) }
            )
        }
    }

    /// The shortlist, rendered beside Review download so a comparison never
    /// costs a navigation. Selecting here re-renders the fact table in place —
    /// the same ``coordinator.select`` every other list calls, so the
    /// micro-stage does not change and Back still returns to the origin.
    @ViewBuilder
    private var reviewCompanionList: some View {
        let list = shortlist
        OnboardingIntrinsicColumn {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(list.starters) { choice in
                    QuickstartRecommendedCard(
                        choice: choice,
                        selected: coordinator.selection.alias == choice.alias,
                        sizeText: Self.sizeText(for: choice),
                        onActivate: { activatePrimary(in: .review) }
                    ) { coordinator.select(choice) }
                }
                if !list.lowMemory.isEmpty {
                    OnboardingGroupLabel(text: "NEED THE LIGHTEST OPTION?")
                        .padding(.top, 14)
                    ForEach(list.lowMemory) { choice in
                        QuickstartLowMemoryCard(
                            choice: choice,
                            selected: coordinator.selection.alias == choice.alias,
                            sizeText: Self.sizeText(for: choice),
                            onActivate: { activatePrimary(in: .review) }
                        ) { coordinator.select(choice) }
                    }
                }
                if !list.tradeUps.isEmpty {
                    OnboardingGroupLabel(text: "OR PICK A BIGGER ONE")
                        .padding(.top, 14)
                    ForEach(list.tradeUps) { choice in
                        let cached = Self.cachedModel(alias: choice.alias, cachedModels: cachedModels)
                        QuickstartCompactCard(
                            choice: choice,
                            selected: coordinator.selection.alias == choice.alias,
                            sizeText: cached?.sizeOnDisk ?? Self.sizeText(for: choice),
                            isCached: cached != nil,
                            onActivate: { activatePrimary(in: .review) }
                        ) { coordinator.select(choice) }
                    }
                }
                if let pick = list.yourPick {
                    OnboardingGroupLabel(text: "YOUR PICK")
                        .padding(.top, 14)
                    let choice = Self.choice(forCatalogEntry: pick)
                    QuickstartCompactCard(
                        choice: choice,
                        selected: coordinator.selection.alias == pick.alias,
                        sizeText: Self.rowSizeText(for: pick),
                        isCached: pick.cached,
                        onActivate: { activatePrimary(in: .review) }
                    ) { coordinator.select(choice) }
                    .accessibilityIdentifier("Quickstart.YourPick.\(pick.alias)")
                }
            }
            .padding(.trailing, 2)
        }
    }

    /// Review's fact rows, in Paper's order (05.2.D).
    ///
    /// Pure, and every value comes from a source the app already reads —
    /// ``sizeText`` for the download, ``ModelEntry/cached`` for presence,
    /// ``ModelSizing`` against ``MacHardware`` for memory, the pre-flight probe
    /// for free space. A row whose source has nothing to say is omitted rather
    /// than printed with a placeholder.
    /// - Parameter runsHere: false when ``ModelSizing`` classifies the alias
    ///   ``ModelSizing/Fit/tooBig`` on this Mac. It adds one row and recolours
    ///   another; it removes nothing, because Paper 05.2.D keeps the shape of
    ///   the screen identical between a model that can start and one that
    ///   cannot — the user compared them from the same list a moment ago.
    static func reviewFacts(
        alias: String,
        cached: ModelEntry?,
        cachedModels: [ModelEntry],
        hardware: MacHardware,
        freeBytes: Int64?,
        runsHere: Bool = true
    ) -> [OnboardingFactRow] {
        var rows: [OnboardingFactRow] = [
            // The alias, first. Paper's fact list starts at the download size,
            // but the identity row predates it and the golden-flow harness
            // addresses it by identifier — and naming the exact alias about to
            // be pulled is the least this screen can do.
            OnboardingFactRow(
                "Model",
                alias,
                identifier: "Quickstart.Review.Alias"
            ),
            OnboardingFactRow(
                cached == nil ? "Download" : "Size on disk",
                reviewDownloadText(alias: alias, cached: cached),
                identifier: "Quickstart.Review.Size"
            ),
            OnboardingFactRow(
                "On this Mac",
                cached == nil ? "Not downloaded yet" : "Already downloaded",
                isStrong: cached != nil,
                identifier: "Quickstart.Review.CachedStatus"
            ),
        ]

        let footprint = ModelSizing.estimate(alias: alias).totalGB
        if footprint > 0, hardware.physicalRAMGB > 0 {
            rows.append(OnboardingFactRow(
                "Memory when loaded",
                "≈ \(preciseGB(footprint)) of \(wholeGB(hardware.physicalRAMGB))",
                isAlert: !runsHere,
                identifier: "Quickstart.Review.Memory"
            ))
            // Only shown when it is the reason for something. On a model that
            // fits, the usable pool is trivia; on one that does not, it is the
            // fact the verdict rests on, and printing it lets the user check
            // the arithmetic rather than take the refusal on trust.
            if !runsHere, hardware.usableRAMGB > 0 {
                rows.append(OnboardingFactRow(
                    "Usable for a model",
                    "\(preciseGB(hardware.usableRAMGB)) of \(wholeGB(hardware.physicalRAMGB))",
                    isStrong: false,
                    identifier: "Quickstart.Review.UsableMemory"
                ))
            }
        }

        if let freeBytes {
            rows.append(OnboardingFactRow(
                "Free space",
                "\(wholeGB(Double(freeBytes) / Double(1 << 30))) available",
                identifier: "Quickstart.Review.FreeSpace"
            ))
        }

        if let repo = reviewRepo(alias: alias, cached: cached, cachedModels: cachedModels) {
            rows.append(OnboardingFactRow(
                "Hugging Face",
                repo,
                isStrong: false,
                identifier: "Quickstart.Review.Repo"
            ))
        }
        return rows
    }

    /// The download size Review quotes.
    ///
    /// Prefers the choice's PINNED byte count when the alias is one of the
    /// onboarding ladder — the same number its card shows. Without this the
    /// starter read "~633 MB" on the card and "~563 MB" two clicks later,
    /// because the card uses the pinned figure and the generic path falls back
    /// to the parameter-derived estimate. One model, one number.
    static func reviewDownloadText(alias: String, cached: ModelEntry?) -> String {
        if cached == nil,
           let choice = QuickstartCoordinator.onboardingChoices.first(where: { $0.alias == alias }),
           choice.downloadBytes != nil {
            return sizeText(for: choice)
        }
        return reviewSizeText(alias: alias, cached: cached)
    }

    /// The line under the fact table. Only offered for a download that has not
    /// happened yet — for a cached model there is no cost to frame.
    static func reviewFootnote(alias: String, cached: ModelEntry?) -> String? {
        guard cached == nil else { return nil }
        let size = reviewDownloadText(alias: alias, cached: nil)
        guard size != "Unknown" else {
            return "One download, once. After that this model starts in seconds and needs no network."
        }
        return "One \(size) pull, once. After that this model starts in seconds and needs no network."
    }

    // MARK: - 2e — Review download · incompatible memory (Paper 05.2.D)

    /// The sentence under the model's name on Review download.
    ///
    /// Three states, one slot. Paper writes the incompatible one as a flat
    /// statement — "It cannot run on this Mac" — with no apology and no offer,
    /// because the screen's whole job at that point is to answer a question
    /// the user already asked.
    static func reviewSubtitle(cached: ModelEntry?, runsHere: Bool) -> String {
        guard runsHere else { return "This model cannot run on this Mac." }
        return cached == nil
            ? "This downloads once and then runs entirely on your Mac."
            : "Already on this Mac — nothing will be downloaded."
    }

    /// Paper 05.2.D's callout: what it needs, what this Mac has, and how much
    /// of that a model may actually use.
    ///
    /// Every figure is an existing reading — ``ModelSizing/estimate(alias:)``
    /// and ``MacHardware/usableRAMGB`` — rendered through the same two helpers
    /// the fact table uses, so the number in the callout and the number in the
    /// table below it are the same number, spelled the same way.
    static func incompatibilityNote(alias: String, hardware: MacHardware) -> String {
        let needed = ModelSizing.estimate(alias: alias).totalGB
        guard needed > 0, hardware.physicalRAMGB > 0 else {
            return "This model needs more memory than this Mac has."
        }
        return "Needs ≈ \(preciseGB(needed)) of memory. This Mac has "
            + "\(wholeGB(hardware.physicalRAMGB)), of which roughly "
            + "\(preciseGB(hardware.usableRAMGB)) is usable for a model."
    }

    /// The line under the fact table on an incompatible Review, in place of
    /// the download footnote — there is no download to frame.
    ///
    /// Paper's sentence is the first half. The second half is added because
    /// without it the screen is misleading at the margin: the ceiling is 75%
    /// of the usable pool, not the pool, so a 21 GB model on a 32 GB Mac is
    /// refused while the callout says 25.6 GB is usable. Naming the limit
    /// turns an apparent contradiction into an arithmetic the user can follow.
    /// The number comes from ``ModelSizing/largestFittingGB(on:)`` so it
    /// cannot drift from the band that produced the verdict.
    static func memoryHeadroomFootnote(hardware: MacHardware) -> String? {
        guard hardware.physicalRAMGB > 0, hardware.usableRAMGB > 0 else { return nil }
        let ceiling = ModelSizing.largestFittingGB(on: hardware)
        return "macOS keeps about a fifth of unified memory for itself, so a "
            + "\(wholeGB(hardware.physicalRAMGB)) Mac has roughly "
            + "\(preciseGB(hardware.usableRAMGB)) to give a model. Rapid keeps "
            + "some of that free for your conversation, so it offers models "
            + "needing up to about \(preciseGB(ceiling))."
    }

    /// What VoiceOver is told about the greyed primary on an incompatible
    /// Review. macOS announces a disabled control as "dimmed" and stops there.
    static func incompatiblePrimaryHint(alias: String, hardware: MacHardware) -> String {
        "Unavailable. \(incompatibilityNote(alias: alias, hardware: hardware))"
    }

    // MARK: - Step 2 derivation (pure seams)

    /// The recommended shortlist exactly as it renders, in render order.
    struct Shortlist: Equatable {
        var cached: [ModelEntry] = []
        var starters: [QuickstartModelChoice] = []
        var lowMemory: [QuickstartModelChoice] = []
        var tradeUps: [QuickstartModelChoice] = []
        /// Approved default D2 — a catalogue pick carried back by Back.
        var yourPick: ModelEntry?

        /// Every alias the user can currently see and click, in render order.
        /// This is the "visible" half of `selection ∩ visible rows`.
        var visibleAliases: [String] {
            cached.map(\.alias)
                + starters.map(\.alias)
                + lowMemory.map(\.alias)
                + tradeUps.map(\.alias)
                + (yourPick.map { [$0.alias] } ?? [])
        }
    }

    /// Build the shortlist. Static and pure so the YOUR PICK rule and the
    /// visible-alias set can be pinned without a SwiftUI host.
    static func shortlist(catalog: [ModelEntry], selection: String) -> Shortlist {
        let choices = QuickstartCoordinator.onboardingChoices
        let existing = Array(quickstartCachedModels(catalog).prefix(6))
        let existingAliases = Set(existing.map(\.alias))
        var native = existingAliases
        native.formUnion(choices.map(\.alias))
        return Shortlist(
            cached: existing,
            starters: choices.filter { $0.isStarter && !existingAliases.contains($0.alias) },
            lowMemory: choices.filter { $0.isLowMemory && !existingAliases.contains($0.alias) },
            tradeUps: choices.filter { $0.tier == .tradeUp && !existingAliases.contains($0.alias) },
            yourPick: native.contains(selection)
                ? nil
                : onboardingCatalogModels(catalog).first { $0.alias == selection }
        )
    }

    private var shortlist: Shortlist {
        Self.shortlist(catalog: cachedModels, selection: coordinator.selection.alias)
    }

    /// The catalogue slice onboarding may offer (approved default D4): chat
    /// models only. Image, audio and video models are managed in Settings, not
    /// chosen during first-run setup. Scoped to onboarding — Settings → Models
    /// is deliberately unaffected.
    static func onboardingCatalogModels(_ entries: [ModelEntry]) -> [ModelEntry] {
        entries.filter { $0.kind == .chat }
    }

    /// The catalogue as the user currently sees it: chat-only, searched,
    /// filtered and sorted, through the same primitives Settings → Models uses.
    static func visibleCatalogEntries(
        catalog: [ModelEntry],
        query: String,
        filter: ModelCacheActions.FilterMode,
        sort: ModelCacheActions.SortOrder
    ) -> [ModelEntry] {
        let scoped = onboardingCatalogModels(catalog)
        return ModelCacheActions.sorted(
            ModelCacheActions.filter(scoped, by: filter, query: query),
            order: sort
        )
    }

    private var visibleCatalogEntries: [ModelEntry] {
        Self.visibleCatalogEntries(
            catalog: cachedModels,
            query: coordinator.catalogQuery,
            filter: coordinator.catalogFilter,
            sort: coordinator.catalogSort
        )
    }

    /// Resolve loading / failed / ready from the snapshot plus its landed flag.
    static func catalogState(
        catalog: [ModelEntry],
        loaded: Bool
    ) -> OnboardingModelSelection.CatalogState {
        guard loaded else { return .loading }
        // ``ModelCatalog.load`` returns `[]` when its subprocess failed, so an
        // empty catalogue is that sentinel — NOT a Mac with nothing downloaded.
        // An empty cache still lists every downloadable alias.
        return onboardingCatalogModels(catalog).isEmpty ? .failed : .ready
    }

    private var catalogState: OnboardingModelSelection.CatalogState {
        Self.catalogState(catalog: cachedModels, loaded: catalogLoaded)
    }

    /// The visible rows of one list context, as the CTA contract needs them.
    private func visibleRows(
        for context: OnboardingModelSelection.ListContext
    ) -> [OnboardingModelSelection.Row] {
        switch context {
        case .shortlist:
            let cachedAliases = Set(Self.quickstartCachedModels(cachedModels).map(\.alias))
            return shortlist.visibleAliases.map { alias in
                OnboardingModelSelection.Row(
                    alias: alias,
                    isCached: cachedAliases.contains(alias),
                    isAvailable: OnboardingModelSelection.isAvailable(alias: alias, hardware: hardware)
                )
            }
        case .catalogue:
            return OnboardingModelSelection.rows(for: visibleCatalogEntries, hardware: hardware)
        case .review:
            // Review shows exactly one model: the selection. Its cached-ness
            // comes from the catalogue snapshot, never from the copy above it.
            let alias = coordinator.selection.alias
            guard !alias.isEmpty else { return [] }
            return [OnboardingModelSelection.Row(
                alias: alias,
                isCached: Self.canStartWithoutDownload(alias: alias, cachedModels: cachedModels),
                isAvailable: OnboardingModelSelection.isAvailable(alias: alias, hardware: hardware)
            )]
        }
    }

    /// The footer primary for a list context. Re-derived on every render.
    private func primary(
        for context: OnboardingModelSelection.ListContext
    ) -> OnboardingModelSelection.Primary {
        OnboardingModelSelection.primary(
            selection: coordinator.selection.alias,
            visibleRows: visibleRows(for: context),
            catalogState: catalogState,
            context: context
        )
    }

    /// The single activation path (Paper 05.2.G — "One action, three inputs").
    ///
    /// The footer primary, Return (via the footer's `.defaultAction`) and a
    /// double-click on a row all land here, so no input can reach an action
    /// the user cannot see. A disabled primary makes every one of them inert.
    private func activatePrimary(in context: OnboardingModelSelection.ListContext) {
        let primary = primary(for: context)
        guard primary.isEnabled else { return }
        switch primary.action {
        case .reviewDownload, .reviewIncompatible:
            // Both open the same micro-stage. What differs is what it says
            // once it is there, and that is re-derived from the selection on
            // arrival rather than carried across as a flag — so a user who
            // switches to a runnable model on Review's live companion list
            // gets a working primary without navigating anywhere.
            coordinator.beginReviewDownload(
                origin: context == .catalogue ? .catalogue : .shortlist
            )
        case .startExisting, .downloadAndStart:
            // One production route for both. ``startQuickstart`` already
            // branches on the same cached truth: a cached alias skips the
            // download machinery entirely and hands straight to
            // ``ServerManager.start`` (Step 4), an uncached one runs the disk
            // pre-flight and then the pull (Step 3).
            startQuickstart()
        }
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

    static func sizeText(for choice: QuickstartModelChoice) -> String {
        guard let bytes = choice.downloadBytes else {
            return sizeText(for: choice.alias)
        }
        let mib = Double(bytes) / Double(1 << 20)
        if mib < 1024 {
            return "~\(Int(mib.rounded())) MB"
        }
        return String(format: "%.1f GB", mib / 1024)
    }

    /// Stable, bounded presentation for models already on disk. The catalogue
    /// supplied here is the chat catalogue; retain the kind check defensively
    /// so a future combined snapshot cannot leak image/video aliases into the
    /// first-chat path. This returns the complete eligible set because lookup
    /// correctness must not depend on the UI's six-row presentation bound.
    static func quickstartCachedModels(_ entries: [ModelEntry]) -> [ModelEntry] {
        entries.filter { $0.cached && $0.kind == .chat }
    }

    static func choice(forCachedModel entry: ModelEntry) -> QuickstartModelChoice {
        QuickstartModelChoice(
            alias: entry.alias,
            displayName: entry.alias,
            hfRepo: entry.hfRepo,
            blurb: entry.isExternal ? "Already downloaded by another MLX app." : "Already downloaded and ready to start.",
            tier: .tradeUp
        )
    }

    /// A wizard choice for any catalogue row, cached or not.
    ///
    /// The alias is the identity on both branches — never the display name,
    /// never a curated label — so the same model picked from the shortlist and
    /// from the catalogue is one selection, and `select` on either is the same
    /// act. Uncached rows carry no blurb: the catalogue has no curated prose
    /// for them and inventing one would be a claim we cannot support.
    static func choice(forCatalogEntry entry: ModelEntry) -> QuickstartModelChoice {
        guard !entry.cached else { return choice(forCachedModel: entry) }
        return QuickstartModelChoice(
            alias: entry.alias,
            displayName: entry.alias,
            hfRepo: entry.hfRepo,
            blurb: "",
            tier: .tradeUp
        )
    }

    /// The size a catalogue row shows: what it occupies if it is here, what it
    /// would cost if it is not.
    static func rowSizeText(for entry: ModelEntry) -> String {
        if entry.cached, let onDisk = entry.sizeOnDisk, !onDisk.isEmpty {
            return onDisk
        }
        return sizeText(for: entry.alias)
    }

    // MARK: - Review download facts (pure)

    /// The Hugging Face repo to quote on Review, when the catalogue knows one.
    /// Prefers the cached entry (it came from `rapid-mlx ls`, which resolved
    /// the repo) and falls back to the catalogue row.
    static func reviewRepo(
        alias: String,
        cached: ModelEntry?,
        cachedModels: [ModelEntry]
    ) -> String? {
        if let repo = cached?.hfRepo, !repo.isEmpty { return repo }
        let repo = onboardingCatalogModels(cachedModels)
            .first { $0.alias == alias }?
            .hfRepo
        guard let repo, !repo.isEmpty else { return nil }
        return repo
    }

    /// Size for the Review screen. A cached model reports what it actually
    /// occupies; an uncached one reports the same ``ModelSizing`` estimate the
    /// rest of the app quotes, so no two surfaces name different numbers for
    /// the same model. Returns an explicit "Unknown" rather than an empty row
    /// when there is no estimate — a blank would read as "free".
    static func reviewSizeText(alias: String, cached: ModelEntry?) -> String {
        if let cached, let onDisk = cached.sizeOnDisk, !onDisk.isEmpty {
            return onDisk
        }
        let estimate = sizeText(for: alias)
        return estimate.isEmpty ? "Unknown" : estimate
    }

    /// Free space on the volume that holds the Hugging Face cache — the same
    /// probe the download pre-flight runs, quoted before the commit rather
    /// than only after it. `nil` when the probe has no signal, in which case
    /// the row is omitted instead of claiming a number.
    static func reviewFreeSpaceText(probe: () -> Int64?) -> String? {
        guard let free = probe() else { return nil }
        return "\(formatBytesForBanner(free)) available"
    }

    static func canStartWithoutDownload(alias: String, cachedModels: [ModelEntry]) -> Bool {
        cachedModel(alias: alias, cachedModels: cachedModels) != nil
    }

    static func cachedModel(alias: String, cachedModels: [ModelEntry]) -> ModelEntry? {
        quickstartCachedModels(cachedModels).first { $0.alias == alias }
    }

    // MARK: - Subviews

    /// Step 3's canvas (Paper 05.1 state 09).
    ///
    /// The rail owns the numbers — percentage, bytes, rate, ETA — so the canvas
    /// deliberately repeats none of them. What it owns is meaning: what the
    /// transfer IS, and what leaving, quitting or stopping will do. Everything
    /// here is a plain statement of consequence; nothing is a progress read-out
    /// wearing prose.
    @ViewBuilder
    private var downloadingCard: some View {
        let job = downloads.job(for: coordinator.selection.alias)

        VStack(alignment: .leading, spacing: 0) {
            OnboardingDisplayTitle(text: "One download,\nthen it's yours.")

            Text("The model files are being written into your Hugging Face cache. "
                 + "This is a plain file transfer from the model mirror — nothing "
                 + "about you is sent with it.")
                .scaledSystemFont(16, relativeTo: .title3)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
                .padding(.top, 18)

            VStack(spacing: 0) {
                downloadFact(
                    "LEAVING",
                    "Keep this window open. Setup finishes on its own and opens chat "
                        + "as soon as the files land.",
                    isFirst: true
                )
                downloadFact(
                    "QUITTING",
                    "Quitting Rapid-MLX stops the transfer. Setup is not marked "
                        + "finished, so it runs again on the next launch."
                )
                // Paper's third row said "NO CANCEL — cancelling a download
                // lives in Settings → Models, which is behind this window until
                // setup ends". That is superseded: the merged recovery PR added
                // a Cancel to this screen precisely because that arrangement
                // left a part-finished download with no exit but quitting. The
                // row now describes the control that exists.
                downloadFact(
                    "STOPPING",
                    "Cancel download stops the transfer. The model is not installed, "
                        + "and you can retry it or choose a different one.",
                    isLast: true
                )
            }
            .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
            .padding(.top, 34)

            // The way out of Step 3.
            //
            // Onboarding is a full-window sheet, so ``DownloadStrip`` — the
            // app's ordinary cancel affordance — is behind it and unreachable
            // for the entire pull. Without this control the only exits from a
            // download the user no longer wants are quitting the app or waiting
            // it out, and a multi-gigabyte trade-up makes "wait it out" a very
            // long time to be stuck. The cancellation RECOVERY path already
            // existed and was reachable (an app quit reaches it); what did not
            // exist was any way to ask for it from the screen that is on top.
            if let cancelAlias = Self.downloadCancelTarget(
                jobStatus: job?.status,
                selectionAlias: coordinator.selection.alias,
                alreadyRequested: cancelRequestedAlias == coordinator.selection.alias
            ) {
                Button("Cancel download") {
                    cancelRequestedAlias = cancelAlias
                    downloads.cancelDownload(alias: cancelAlias)
                }
                .buttonStyle(.onboardingQuiet)
                .padding(.top, 26)
            // Deliberately NO keyboard shortcut.
            //
            // `.defaultAction` would put a destructive action on Return, on a
            // screen whose whole job is waiting — the single most likely
            // stray keypress here. `.cancelAction` is no better: Escape
            // already has a meaning inside this sheet (retreat within Step 2,
            // else leave setup, see ``ContentView.quickstartSheetPresented``),
            // and quietly redefining it to "destroy the running transfer"
            // would make one key do two very different things depending on a
            // phase the user cannot see. Click, Tab-then-Space, and VoiceOver
            // all reach it; nothing needs a shortcut to be reachable.
            .accessibilityIdentifier("Quickstart.Download.Cancel")
            .accessibilityLabel("Cancel download of \(coordinator.selection.displayName)")
            // Says what is lost, without claiming anything about bytes
            // already on disk — see ``FailureDiagnosis/Kind/downloadCancelled``.
            .accessibilityHint("Stops the download. The model will not be installed.")
            }
        }
        .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
    }

    /// One `LABEL · consequence` row on the Step 3 canvas. Hairline-separated
    /// facts rather than a card: the canvas states consequences, and boxing
    /// each one would make three equal-weight cards out of three sentences.
    @ViewBuilder
    private func downloadFact(
        _ label: String,
        _ body: String,
        isFirst: Bool = false,
        isLast: Bool = false
    ) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(label)
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.groupLabel)
                .foregroundStyle(RapidTheme.textTertiary)
                .frame(width: 92, alignment: .leading)
                .padding(.top, 3)
            Text(body)
                .scaledSystemFont(14)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 16)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .overlay(alignment: .bottom) {
            if isLast { Rectangle().fill(RapidTheme.hairline).frame(height: 1) }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label). \(body)")
    }

    /// Which alias, if any, the Step 3 card should offer to cancel.
    ///
    /// Pure so "an active download exposes a cancel action, and a settled one
    /// does not" can be pinned without a SwiftUI host — the exact property a
    /// rendered-only control cannot be tested for, and the one that was
    /// missing.
    ///
    /// Returns `nil` — meaning draw no control at all — in four cases, each of
    /// which would otherwise put a button on screen that does nothing:
    ///
    ///   * **No job.** Nothing has started, or the record is already gone.
    ///   * **Not running.** Completed, failed, or already cancelled.
    ///     ``DownloadManager/cancelDownload(alias:)`` is a no-op against all
    ///     three, and offering to stop something that already stopped is the
    ///     "looks actionable while doing nothing" defect in miniature.
    ///   * **Already requested.** The optimistic flip to ``Status/cancelled``
    ///     lands on the same run-loop turn, but a second click in the same
    ///     frame would still re-signal a process that is mid-SIGTERM and start
    ///     a second hard-kill timer. One request per job, enforced here.
    ///   * **No selection.** Defensive; there is nothing to name.
    static func downloadCancelTarget(
        jobStatus: DownloadManager.Job.Status?,
        selectionAlias: String,
        alreadyRequested: Bool
    ) -> String? {
        guard !selectionAlias.isEmpty else { return nil }
        guard !alreadyRequested else { return nil }
        guard let jobStatus else { return nil }
        guard case .running = jobStatus else { return nil }
        return selectionAlias
    }

    /// Step 4's canvas while the serve comes up (Paper 05.1 state 15).
    ///
    /// The rail carries STARTING, the identity and the indeterminate track, so
    /// the canvas states only what loading means and roughly how long it takes.
    /// No spinner here: a second progress indicator beside the rail's would be
    /// two things reporting one wait.
    @ViewBuilder
    private var startingCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingDisplayTitle(text: "Loading into memory.")

            Text("The weights are on disk. They are being loaded into Metal and the "
                 + "model is warming up — usually 5 to 15 seconds, longer the first "
                 + "time a model runs.")
                .scaledSystemFont(16, relativeTo: .title3)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 18)
        }
        .frame(maxWidth: OnboardingD.proseWidth, alignment: .leading)
    }

    /// The Ready confirmation screen — the end of Step 4 and the only
    /// thing that ends onboarding.
    ///
    /// No spinner, no progress, no countdown: the model IS ready, and
    /// dressing the wait for a click as work would be a lie about what the
    /// app is doing. The green tile is the one place setup uses the ready
    /// colour, and the amber primary is this screen's single strong moment.
    @ViewBuilder
    private var readyCard: some View {
        OnboardingOutcomeBlock(
            glyph: "checkmark",
            tone: .ready,
            kicker: "SETUP COMPLETE",
            title: "\(coordinator.selection.displayName) is ready.",
            message: "It is loaded on this Mac and answering locally. "
                + "Nothing you type from here leaves the machine."
        ) {
            OnboardingActionLane {
                Button("Start chatting") {
                    completeOnboarding()
                }
                .buttonStyle(.onboardingPrimary)
                // The native default action, not custom key handling: Return
                // activates it and Space activates it while focused, both via
                // AppKit's ordinary button semantics.
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("Quickstart.Ready.StartChatting")
                .accessibilityLabel("Start chatting with \(coordinator.selection.displayName)")
            }
        }
    }

    /// Run the Start chatting transaction.
    ///
    /// The coordinator half is authoritative and idempotent — it decides
    /// whether this activation is the one that completes setup. The parent
    /// half (route to Chat, announce, focus the composer) runs only on that
    /// verdict, so a double activation cannot fire a second transition.
    private func completeOnboarding() {
        guard coordinator.confirmStartChatting(seedWelcome: onSeedWelcome) else { return }
        onCompleted()
    }

    /// In-sheet twin of ContentView's memory-warning ``.alert`` (#1503).
    /// That alert is unreachable while this full-window onboarding sheet is
    /// up, so a Quickstart serve that trips the pre-load memory guard would
    /// otherwise strand the user on a permanent "Starting…". Same copy, same
    /// two ``ServerManager`` actions — presented where the user is looking.
    ///
    /// "Load anyway" carries no ``.defaultAction`` shortcut on purpose: the
    /// risky choice must not be what Return triggers. Cancel owns
    /// ``.cancelAction`` so Esc DECLINES the load (the safe default) rather
    /// than dismissing the sheet out from under the decision.
    @ViewBuilder
    private func memoryWarningCard(_ warning: ModelSizing.MemoryWarning) -> some View {
        let fallback = Self.lowMemoryRecoveryChoice(for: warning)
        OnboardingOutcomeBlock(
            glyph: "exclamationmark.triangle",
            tone: .amber,
            kicker: "STEP \(QuickstartCoordinator.Step.start.displayNumber) "
                + "OF \(QuickstartCoordinator.Step.total) · BEFORE LOADING",
            title: warning.title,
            message: warning.message
        ) {
            // The two measured numbers behind the guard's decision, stated as
            // figures rather than buried in the sentence above.
            HStack(alignment: .top, spacing: 44) {
                OnboardingStat(
                    label: "NEEDS",
                    value: Self.wholeGB(warning.footprintGB),
                    tone: RapidTheme.statusError
                )
                OnboardingStat(
                    label: "FREE NOW",
                    value: Self.wholeGB(warning.freeGB)
                )
                if warning.totalGB > 0 {
                    OnboardingStat(
                        label: "THIS MAC",
                        value: Self.wholeGB(warning.totalGB)
                    )
                }
            }
            .padding(.top, 30)
        } actions: {
            OnboardingActionLane {
                if let fallback {
                    Button("Switch to \(fallback.displayName)") {
                        server.cancelPendingMemoryLoad(warning)
                        coordinator.returnToChooser()
                        coordinator.select(fallback)
                        startQuickstart()
                    }
                    .buttonStyle(.onboardingPrimary)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("Quickstart.Memory.SwitchToLowMemory")
                    .accessibilityLabel("Switch to \(fallback.displayName), the lowest-memory option")
                }

                Button(warning.confirmTitle) {
                    // Re-enters ``start`` with the guard bypassed. We stay in
                    // ``.starting``; ``handleServerStateChange`` seeds the
                    // welcome and dismisses the sheet once the child reaches
                    // ``.ready``.
                    server.confirmPendingMemoryLoad(warning)
                }
                .buttonStyle(.onboardingOutline)
                .accessibilityIdentifier("Quickstart.Memory.LoadAnyway")

                Button("Cancel") {
                    // Drop the parked load and leave ``.starting`` for the
                    // chooser so the sheet stops waiting on a serve that will
                    // never come.
                    server.cancelPendingMemoryLoad(warning)
                    coordinator.returnToChooser()
                }
                .buttonStyle(.onboardingQuiet)
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("Quickstart.Memory.Cancel")
            }
        }
    }

    /// Return the curated low-memory fallback only when the same snapshot
    /// that blocked the original load says the replacement falls below the
    /// 85% danger line. This prevents a reassuring "Switch" button from
    /// merely leading to a second warning. If the snapshot is unavailable,
    /// Cancel still returns to the chooser and the fallback remains visible,
    /// but the warning does not claim it is safe.
    static func lowMemoryRecoveryChoice(
        for warning: ModelSizing.MemoryWarning
    ) -> QuickstartModelChoice? {
        let fallback = QuickstartCoordinator.lowMemoryChoice
        guard warning.alias != fallback.alias, warning.totalGB > 0 else { return nil }
        let footprint = ModelSizing.estimate(alias: fallback.alias)
        guard footprint.totalGB < warning.footprintGB else { return nil }
        let gib = Double(1 << 30)
        let usedGB = max(0, warning.totalGB - warning.freeGB)
        let safety = ModelSizing.memorySafety(
            footprint: footprint,
            usedBytes: UInt64((usedGB * gib).rounded()),
            totalBytes: UInt64((warning.totalGB * gib).rounded())
        )
        return safety == .unsafe ? nil : fallback
    }

    /// Which memory warning, if any, the Quickstart sheet must present
    /// itself rather than delegate to ContentView's covered ``.alert``
    /// (#1503). Returns the pending warning ONLY while we are actively
    /// driving a serve (``phase == .starting``) AND the parked load is for
    /// OUR selection — a warning carrying a different alias belongs to some
    /// other start path and is not ours to resolve inside onboarding. Pure
    /// so the deadlock scenario can be pinned without a SwiftUI host, and so
    /// ContentView can gate its alert on the exact same condition.
    static func memoryWarningToPresent(
        phase: QuickstartCoordinator.Phase,
        pending: ModelSizing.MemoryWarning?,
        selectionAlias: String
    ) -> ModelSizing.MemoryWarning? {
        guard case .starting = phase else { return nil }
        guard let pending, pending.alias == selectionAlias else { return nil }
        return pending
    }

    /// Low-disk warning card. Non-blocking — the user can still
    /// proceed with Continue (per LM Studio / Ollama UX) or Cancel
    /// back to the hero card. Visual language matches ``failedCard``
    /// (amber tint + warning glyph) so the user reads it as a caution,
    /// not an error.
    @ViewBuilder
    private func lowDiskCard(freeBytes: Int64, requiredBytes: Int64) -> some View {
        OnboardingOutcomeBlock(
            glyph: "externaldrive.badge.exclamationmark",
            tone: .amber,
            kicker: "STEP \(QuickstartCoordinator.Step.download.displayNumber) "
                + "OF \(QuickstartCoordinator.Step.total) · BEFORE THE DOWNLOAD",
            title: "Low disk space",
            message: QuickstartView.lowDiskBannerBody(
                freeBytes: freeBytes,
                requiredBytes: requiredBytes,
                displayName: coordinator.selection.displayName
            )
        ) {
            HStack(alignment: .top, spacing: 44) {
                OnboardingStat(
                    label: "FREE NOW",
                    value: Self.formatBytesForBanner(freeBytes),
                    tone: RapidTheme.statusError
                )
                OnboardingStat(
                    label: "NEEDED",
                    value: Self.formatBytesForBanner(requiredBytes)
                )
            }
            .padding(.top, 30)
            // The prose above already carries both numbers for VoiceOver, so
            // the figures repeat them visually only.
            .accessibilityHidden(true)
        } actions: {
            OnboardingActionLane {
                Button("Continue anyway") {
                    kickoffDownload()
                }
                .buttonStyle(.onboardingPrimary)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("Quickstart.LowDisk.Continue")
                .accessibilityLabel("Continue download despite low disk space")

                Button("Cancel") {
                    coordinator.cancelLowDiskWarning()
                }
                .buttonStyle(.onboardingQuiet)
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("Quickstart.LowDisk.Cancel")
                .accessibilityLabel("Cancel — go back to choosing a model without downloading")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(QuickstartView.lowDiskAccessibilityLabel(
            freeBytes: freeBytes,
            requiredBytes: requiredBytes,
            displayName: coordinator.selection.displayName
        ))
    }

    @ViewBuilder
    private func failedCard(message: String) -> some View {
        let job = downloads.job(for: coordinator.selection.alias)
        let kind = Self.failureKind(
            jobFailureKind: job?.failureKind,
            jobUsesMirror: job?.source != .huggingFace,
            serverState: server.state,
            selectionAlias: coordinator.selection.alias,
            message: message
        )
        let diagnosis = FailureDiagnoser.diagnosis(for: kind)

        OnboardingOutcomeBlock(
            glyph: Self.failureGlyph(for: kind),
            tone: kind.severity == .notice ? .amber : .error,
            kicker: Self.failureKicker(for: kind, origin: coordinator.step),
            title: Self.failureTitle(for: kind),
            message: diagnosis.message
        ) {
            OnboardingActionLane {
                if let action = diagnosis.action {
                    Button(action.title) {
                        handleQuickstartFailureAction(action)
                    }
                    .buttonStyle(.onboardingPrimary)
                    .disabled(server.isOperating)
                    .accessibilityIdentifier(quickstartActionIdentifier(for: action) ?? "")
                }

                // The way back to choosing. Every failure and every
                // cancellation is one model's problem, so the user must be
                // able to go pick a different one — and land where they
                // actually were, not on a catalogue they may never have
                // opened. Stays inside onboarding: it does not dismiss setup
                // and it does not open Settings.
                Button(Self.failureBackTitle(for: coordinator.step2Stage)) {
                    returnToModelSelection()
                }
                .buttonStyle(.onboardingLink)
                .accessibilityIdentifier("Quickstart.Failure.BackToModelSelection")
                .accessibilityLabel(Self.failureBackAccessibilityLabel(for: coordinator.step2Stage))
            }
        }
    }

    /// The glyph over a failure headline.
    ///
    /// A cancellation is the user's own stop, so it gets a stop mark on the
    /// amber (caution) lane rather than a red fault symbol — the same
    /// distinction ``FailureDiagnosis/Kind/downloadCancelled``'s `.notice`
    /// severity makes in the diagnosis itself.
    static func failureGlyph(for kind: FailureDiagnosis.Kind) -> String {
        switch kind {
        case .downloadCancelled: return "stop.circle"
        case .downloadSourceUnavailable: return "wifi.exclamationmark"
        case .modelOutOfMemory: return "memorychip"
        default: return "exclamationmark.triangle"
        }
    }

    /// `STEP 3 OF 4 · DOWNLOAD STOPPED`. The kicker names the macro step the
    /// failure belongs to — which is the step the user was actually in, never
    /// a step of the failure's own — and then what happened.
    static func failureKicker(
        for kind: FailureDiagnosis.Kind,
        origin: QuickstartCoordinator.Step
    ) -> String {
        let what: String
        switch kind {
        case .downloadCancelled:         what = "DOWNLOAD STOPPED"
        case .downloadSourceUnavailable: what = "SOURCE UNAVAILABLE"
        case .modelOutOfMemory:          what = "NOT ENOUGH MEMORY"
        case .modelLoadFailed:           what = "COULDN'T LOAD"
        default:                         what = "DIDN'T FINISH"
        }
        return "STEP \(origin.displayNumber) OF \(QuickstartCoordinator.Step.total) · \(what)"
    }

    /// Classify a Quickstart failure. Pure so the one inference that matters —
    /// a cancelled download must NOT read as a network fault — can be pinned
    /// without a SwiftUI host.
    ///
    /// Order is the contract. A crashed serve for OUR alias is a load failure
    /// whatever the download did, because the weights are already on disk.
    /// Otherwise the job's own recorded kind wins: ``DownloadManager`` knows
    /// whether it was cancelled or broke, and that knowledge must never be
    /// re-derived from prose. Only when there is no job left to ask does this
    /// fall back to classifying the message.
    static func failureKind(
        jobFailureKind: FailureDiagnosis.Kind?,
        jobUsesMirror: Bool,
        serverState: ServerState,
        selectionAlias: String,
        message: String
    ) -> FailureDiagnosis.Kind {
        if case .crashed(let alias, let serverMessage) = serverState,
           alias == selectionAlias {
            return FailureDiagnoser.modelLoadFailureKind(raw: serverMessage)
        }
        if let jobFailureKind { return jobFailureKind }
        // The job was reaped (a relaunch, a dismissal) but the phase survived.
        // The cancellation message is one this app wrote, so recognising it
        // here is reading our own record rather than parsing subprocess prose.
        if message == FailureDiagnoser.diagnosis(for: .downloadCancelled).message {
            return .downloadCancelled
        }
        return FailureDiagnoser.downloadFailureKind(raw: message, usingMirror: jobUsesMirror)
    }

    /// The failure card's heading.
    ///
    /// "Quickstart didn't finish" is a fault report, and a cancellation is not
    /// a fault — the user is the one who stopped it. Everything else keeps the
    /// shipped title unchanged.
    static func failureTitle(for kind: FailureDiagnosis.Kind) -> String {
        kind == .downloadCancelled ? "Download stopped" : "Quickstart didn't finish"
    }

    /// Name the destination the way every other Step 2 Back does, so the
    /// control says where it goes rather than only that it goes back.
    static func failureBackTitle(for stage: QuickstartCoordinator.Step2Stage) -> String {
        switch stage {
        case .browsing:  return "← Back to all models"
        case .reviewing: return "← Back to review download"
        case .checkingHardware, .findingFit, .choosing:
            return "← Back to recommended models"
        }
    }

    /// The same destination, spoken.
    ///
    /// Derived from ``failureBackTitle(for:)`` rather than written out a
    /// second time, so the two can never name different destinations — but
    /// with the leading arrow removed. VoiceOver reads U+2190 aloud as
    /// "left-pointing arrow", which turns a control whose whole job is to
    /// state where it goes into one that opens with a glyph name.
    static func failureBackAccessibilityLabel(
        for stage: QuickstartCoordinator.Step2Stage
    ) -> String {
        failureBackTitle(for: stage)
            .replacingOccurrences(of: "←", with: "")
            .trimmingCharacters(in: .whitespaces)
    }

    /// What VoiceOver says when onboarding lands on a recovery screen.
    ///
    /// ## Why an announcement is required here at all
    ///
    /// Both ways into this screen are silent for a VoiceOver user otherwise.
    /// A cancellation replaces the control the user just pressed, so focus has
    /// nowhere to return to and the press reads as having done nothing — the
    /// same defect the Recheck button had. A genuine failure is worse: it
    /// arrives asynchronously, with no interaction at that instant, so the
    /// screen changes under a user who receives no signal that it did.
    ///
    /// macOS SwiftUI has no live region (see ``VoiceOverAnnouncer``), so the
    /// AppKit announcement is the only reliable path.
    ///
    /// Composed from the SAME `kind` the card renders from, so what is spoken
    /// and what is drawn cannot drift: heading, the diagnosis message, and the
    /// one action offered.
    static func recoveryAnnouncement(for kind: FailureDiagnosis.Kind) -> String {
        let diagnosis = FailureDiagnoser.diagnosis(for: kind)
        var parts = [failureTitle(for: kind), diagnosis.message]
        if let action = diagnosis.action {
            parts.append("Action: \(action.title).")
        }
        return parts.joined(separator: " ")
    }

    /// Leave a failure for the Step 2 micro-stage the user actually left.
    ///
    /// ``QuickstartCoordinator/returnToChooser()`` deliberately does not touch
    /// ``QuickstartCoordinator/step2Stage``, so the shortlist, the catalogue
    /// (with its query, filter, sort and scroll anchor) or Review download all
    /// come back exactly as they were, with the selection still made.
    ///
    /// The ``dismissTerminalState`` call is what keeps the move INSIDE
    /// onboarding after a load failure. Returning to Step 2 puts the phase
    /// back to ``QuickstartCoordinator/Phase/idle``, and at that point the
    /// parent's visibility predicate is the only thing holding the surface up
    /// — but ``QuickstartCoordinator/isEligible(done:legacyDone:lastServedAlias:serverState:)``
    /// reports ineligible while ``ServerState/crashed`` is live, so the sheet
    /// would close and drop the user into the shell mid-setup. Clearing the
    /// terminal state is also the honest reading of the click: the user has
    /// seen the crash and is going to choose something else, which is exactly
    /// what ``ServerManager/dismissTerminalState()`` is for — it additionally
    /// cancels the pending auto-respawn that would otherwise reload the very
    /// model they are walking away from.
    private func returnToModelSelection() {
        server.dismissTerminalState()
        coordinator.returnToChooser()
    }

    /// Enter in-window Browse all models — the ONE destination for every
    /// "browse" affordance in onboarding.
    ///
    /// ## What this replaces
    ///
    /// Paper 05.2.J · S1 supersedes the shipped behaviour, which staged a
    /// Settings tab, ended the wizard's modal session, waited out an AppKit
    /// race and opened a second window. That was already the second attempt:
    /// #1653 fixed a version where browsing simply dismissed the wizard and
    /// discarded the pick. Both share a root cause — the catalogue lived
    /// somewhere onboarding was not — and the fix is to stop leaving.
    ///
    /// So: no ``SettingsRouter``, no ``dismiss()``, no ``openWindow``, no
    /// second window and no reset of the public step. The selection is carried
    /// in, and the catalogue's own query / filter / sort / scroll anchor are
    /// exactly where the user last left them.
    ///
    /// ``returnToChooser()`` runs first because ``beginBrowsingCatalog()``
    /// correctly refuses any phase but ``Phase/idle``. The failure card used
    /// to be the other caller that needed it; it now returns to the
    /// micro-stage the user actually left instead of forcing everyone into the
    /// catalogue — see ``returnToModelSelection()``.
    private func browseAllModels() {
        coordinator.returnToChooser()
        coordinator.beginBrowsingCatalog()
    }

    /// Enter Step 3, and re-arm the Cancel control.
    ///
    /// Every route into a download goes through here — first attempt, Retry,
    /// and Switch source. Clearing ``cancelRequestedAlias`` is what makes the
    /// second pull of the SAME alias cancellable: without it, a user who
    /// cancelled `lfm2.5-1b-4bit` and then retried it would get a live
    /// download with its Cancel control suppressed by the previous attempt's
    /// request, which is the original trap wearing a different hat.
    private func beginDownloadPhase() {
        cancelRequestedAlias = nil
        coordinator.enterDownloading()
    }

    private func handleQuickstartFailureAction(_ action: FailureDiagnosis.Action) {
        switch action {
        case .switchDownloadSource:
            beginDownloadPhase()
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
                beginDownloadPhase()
                _ = downloads.retryDownload(alias: coordinator.selection.alias)
            } else {
                startQuickstart()
            }
        case .restart:
            coordinator.enterStarting()
            Task { await server.start(alias: coordinator.selection.alias) }
        case .openModelManagement, .openWebSearchSettings:
            // One path for every Settings deep-link. ``route`` stages the
            // target tab and only then runs the open — ``SettingsView`` reads
            // the router from ``.onAppear``, so the assignment has to land
            // first, and passing the open as a closure means this call site
            // cannot get that order wrong.
            //
            // ``openWindow(id: "settings")``, NOT ``openSettings()`` — see the
            // ``openWindow`` property above.
            settingsRouter.route(action) { openWindow(id: "settings") }
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
        case .openWebSearchSettings: return "Quickstart.OpenWebSearchSettings"
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
        if let cached = Self.cachedModel(
            alias: coordinator.selection.alias,
            cachedModels: cachedModels
        ) {
            startCachedModel(cached)
            return
        }
        QuickstartView.applyPreflightDecision(
            decision: DiskSpaceProbe.decide(
                freeBytes: freeBytesProbe(),
                requiredBytes: DiskSpaceProbe.quickstartRequiredBytes
            ),
            coordinator: coordinator,
            onKickoff: { kickoffDownload() }
        )
    }

    /// Cached models skip both the disk-space warning and DownloadManager.
    /// `ServerManager.start` still owns cache validation, memory guarding and
    /// the normal ready/failure transitions, so this is a shorter route into
    /// the same serving lifecycle rather than a second implementation.
    private func startCachedModel(_ cached: ModelEntry) {
        coordinator.enterStarting()
        Task { @MainActor in
            await server.start(
                alias: cached.alias,
                hfPath: cached.hfRepo
            )
        }
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
        beginDownloadPhase()
        // ``hfPath`` wires the cache-directory byte monitor so the
        // progress card reads true bytes-on-disk, not just tqdm file
        // counts. Without this the bar could sit at "0/1 files" for
        // the entire 700 MB pull (HF tqdm counts files, not bytes).
        let started = downloads.startDownload(
            alias: coordinator.selection.alias,
            hfPath: coordinator.selection.hfRepo,
            totalBytes: coordinator.selection.downloadBytes
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
            // The user stopped it. Still ``.failed`` — that is the phase the
            // recovery screen lives on and it keeps the rail on Step 3 where
            // the user actually is — but the message comes from the
            // cancellation diagnosis, so nothing downstream has to infer the
            // cause from a string. Paper 05.1 state 10.
            enterRecovery(
                kind: .downloadCancelled,
                message: FailureDiagnoser.diagnosis(for: .downloadCancelled).message,
                origin: .download
            )
        case .failed(let message):
            enterRecovery(
                kind: job.failureKind ?? FailureDiagnoser.downloadFailureKind(
                    raw: message,
                    usingMirror: job.source != .huggingFace
                ),
                message: QuickstartView.friendlyFailureMessage(raw: message),
                origin: .download
            )
        }
    }

    /// Move onto a recovery screen and say so.
    ///
    /// One call site shape for every route in, because the announcement is
    /// exactly as load-bearing as the phase change: the surface swaps under
    /// the user, and on macOS SwiftUI nothing else tells a VoiceOver user
    /// that it did.
    ///
    /// Naturally fires once per arrival — every caller sits behind a phase
    /// guard that the ``enterFailed`` below invalidates, so a re-mount or a
    /// republished status lands on an early return rather than a second
    /// announcement.
    private func enterRecovery(
        kind: FailureDiagnosis.Kind,
        message: String,
        origin: QuickstartCoordinator.FailureOrigin
    ) {
        coordinator.enterFailed(message: message, origin: origin)
        VoiceOverAnnouncer.announce(Self.recoveryAnnouncement(for: kind))
    }

    private func handleServerStateChange() {
        // The serve transition can race the download observer: the
        // user could click Get started, downloads finishes mid-flight,
        // ``server.start`` lands at ``.ready`` BEFORE the
        // download-status observer fired. Guard on the live state so
        // both ordering paths converge on ``enterReady``.
        if case .ready(let alias) = server.state,
           alias == coordinator.selection.alias {
            // Onboarding V3: this is the WHOLE readiness effect. Nothing is
            // seeded, nothing is persisted and nothing is dismissed here —
            // the user does that from the Ready screen. Repeat notifications
            // (auto-respawn, residency tick) land on an idempotent no-op.
            coordinator.enterReady()
            return
        }
        // Relaunch into an unconfirmed Ready flow: the launch auto-start is
        // bringing up the very model that flow was waiting on. Report Step 4
        // truthfully while it loads instead of either fabricating Ready from
        // the stored alias or leaving the user parked on the chooser while
        // the app visibly works. If the load never lands, the crashed branch
        // below and the ordinary chooser both remain reachable.
        if case .starting(let alias) = server.state,
           alias == coordinator.selection.alias,
           coordinator.hasPendingReady,
           case .idle = coordinator.phase {
            coordinator.enterStarting()
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
            // The weights are on disk; it is the load that failed. Keeping
            // the origin means the rail still reads Step 4 rather than
            // sending the user back through the download.
            enterRecovery(
                kind: FailureDiagnoser.modelLoadFailureKind(raw: message),
                message: QuickstartView.friendlyFailureMessage(raw: message),
                origin: .start
            )
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
    /// The number in this copy is ``DiskSpaceProbe/quickstartRequiredBytes`` —
    /// a flat pre-flight floor that is the same for every model. The shipped
    /// wording attributed it to "\<model\> weights + safety margin", which
    /// reads as a per-model measurement and is wrong by a wide margin at both
    /// ends: the starter is ~0.6 GB against a 2 GiB floor, and a large
    /// trade-up needs far more than the floor. Paper 05.1 state 12 states the
    /// rule directly — "Needed is the flat 2 GiB pre-flight floor, not the
    /// model size — the copy says so."
    ///
    /// ``displayName`` is still named, because the contrast is the point: the
    /// user is being asked about *this* download and needs to know the number
    /// is not about it.
    static func lowDiskBannerBody(freeBytes: Int64, requiredBytes: Int64, displayName: String) -> String {
        let free = formatBytesForBanner(freeBytes)
        let need = formatBytesForBanner(requiredBytes)
        return "\(free) free on the volume that holds your Hugging Face cache. " +
               "Setup asks for at least \(need) free before any download — " +
               "a flat floor, not the size of \(displayName). Continue anyway?"
    }

    /// VoiceOver label for the warning card. The banner body is repeated
    /// near-verbatim so screen-reader users get the same numbers as
    /// sighted users; the trailing prompt is rephrased to read as one
    /// sentence rather than a question fragment.
    static func lowDiskAccessibilityLabel(freeBytes: Int64, requiredBytes: Int64, displayName: String) -> String {
        let free = formatBytesForBanner(freeBytes)
        let need = formatBytesForBanner(requiredBytes)
        return "Low disk space warning. \(free) free on the volume that holds " +
               "your Hugging Face cache; setup asks for at least \(need) free " +
               "before any download, which is a flat floor rather than the size " +
               "of \(displayName). " +
               "Choose Continue anyway to start the download, or Cancel to go back."
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
