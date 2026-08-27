import Foundation

/// FU-1: persisted user preference governing the launch-time
/// auto-start path. Lives next to ``AutoStartDecision`` so the key
/// constant + default value are one grep away from the gate that
/// consumes them. UI binds via ``@AppStorage(storageKey)`` in the
/// Settings → Models panel; the launch hook in ``ContentView``
/// reads the same key and threads ``userOptedIn`` through to
/// ``AutoStartDecision.decide``.
///
/// Default is ``true`` so existing users see no behavior change
/// after upgrade. The "opt-out" framing comes from the new
/// ``Settings`` toggle: users who flip it OFF will skip auto-start
/// on the NEXT launch.
enum AutoStartPreference {
    /// UserDefaults key — mirrors the ``rapid.*.v1`` keyspace
    /// convention used by ``ModelPickerVisibility.showAllStorageKey``,
    /// ``AppearanceConfig``, and the sidebar collapsed-section flags.
    /// The ``v1`` suffix lets a future shape change opt into a new
    /// key without inheriting the old one's value silently.
    static let storageKey: String = "rapid.server.auto_start_on_launch.v1"

    /// Default value when the user has never touched the toggle.
    /// ``true`` preserves the v0.7.x behavior (auto-start when the
    /// 3 gates pass) — only users who opt out will see a change.
    static let defaultValue: Bool = true
}

/// Pure launch-time decision: should the desktop auto-spawn the
/// bundled rapid-mlx sidecar on app launch, or surface a manual-Start
/// CTA, or leave the state untouched?
///
/// ## Why this exists (issue #223)
///
/// Pre-v0.7.19 the launch path in ``ContentView`` only auto-started
/// when one of two narrow conditions held: a persisted
/// ``lastServedAlias`` from a prior ``.ready`` session, OR a bundled
/// snapshot was present on disk (only true for dev builds with
/// ``BUNDLE_MODEL=1``). Production DMGs ship with ``BUNDLE_MODEL=0``
/// (see ``scripts/build.sh``), so a brand-new shipped user with no
/// last-served alias falls through and lands on a chat surface whose
/// composer is gated behind the manual "Start" button — exactly the
/// "first-touch broken" symptom the issue documents.
///
/// The fix: a 3-condition gate that the desktop checks once on
/// ``ContentView.task``:
///
/// 1. **Alias resolvable** — pick the persisted last-served alias if
///    present; else the bundled-snapshot alias if available; else any
///    cached alias on disk (the upgrade-with-``defaults delete`` shape
///    from #298). The chosen alias MUST belong to the catalog the
///    sidecar can serve; we never auto-start something we can't
///    confirm.
/// 2. **Binary reachable** — ``ServerLocator.find() != nil``. Covers
///    the bundled-sidecar path (since v0.6.6 the .app ships
///    ``Contents/Resources/rapid-mlx/bin/rapid-mlx``), the PATH/brew
///    install, the runtime-override slot, and the ``RAPID_BIN`` test
///    override.
/// 3. **Model cached** — the alias from gate 1 has a snapshot
///    directory under the user's HF cache. Without this gate a "Start"
///    decision would silently kick off a multi-GB download with no
///    user consent — exactly the v0.7.0 footgun ``BundledModel`` was
///    introduced to fix.
///
/// When all three hold → ``.start(alias)``. When alias + binary hold
/// but the model isn't cached → ``.promptDownload(alias)`` so the
/// chat empty state can render an actionable "Click Start to download
/// <alias>" CTA instead of the generic "Idle". Anything else →
/// ``.skip(reason)`` so the auto-start path is a no-op and the
/// existing manual lifecycle stays the user's affordance.
///
/// ## Architectural choice: eager (not lazy)
///
/// The issue's open question about battery for "read existing chats
/// only" users was resolved in favour of eager (auto-start on
/// ``didFinishLaunching``). The competitive baselines (Ollama, LM
/// Studio) both auto-start on launch; deferring until first-compose
/// would re-introduce the very "click did nothing" silence the issue
/// pins as the dominant friction. Battery impact is bounded by the
/// idle decode-path's near-zero cost on Apple Silicon; users who
/// genuinely want chat-archive-only browsing can Stop the sidecar
/// from the picker and the choice survives via ``lastServedAlias``
/// clear (Stop's documented contract).
///
/// ## Idempotency
///
/// The helper is gated on ``serverState`` — only ``.idle`` produces a
/// non-skip decision. The launch hook is ``.task`` (fires once per
/// view appearance), and ``ServerManager.start`` itself guards on
/// ``!isOperating`` + ``child == nil`` — three layers of defence
/// against double-spawn under tabswitch / scene re-mount.
///
/// ## Orthogonality to ``HideDockChoice.hideAlways`` (v0.8.2 dogfood
/// finding #9, ``01-launch-fuzz.md``)
///
/// The dogfood report noted that in ``.hideAlways`` mode the SwiftUI
/// ``WindowGroup`` is still instantiated (``window count == 1``,
/// ``visible == false``) — SwiftUI does not lazily skip a window's
/// body when its activation policy is ``.accessory``. That means the
/// ``ContentView``'s launch-time ``.task`` (which calls this helper)
/// fires even for users who picked "Hide Dock icon + Don't ask again".
///
/// **By design, ``decide`` does NOT consult ``HideDockChoice``.** The
/// two preferences are scoped to different questions:
///
/// * ``HideDockChoice`` — Dock-icon visibility on close. A UI-surface
///   question. Does NOT govern launch-time sidecar warm-up or any
///   other background activity.
/// * ``AutoStartPreference`` — launch-time sidecar / model auto-start
///   only. Users who want to suppress sidecar warm-up at launch flip
///   ``auto_start_on_launch.v1 = false`` in Settings → Models; that
///   override beats every other gate (see ``SkipReason.userOptedOut``).
///   The toggle does NOT suppress every launch-time task: telemetry
///   has its own opt-in (``TelemetryClient.optedIn``), Quick Ask
///   hotkey is user-disableable by unbinding the chord in
///   Settings → Quick Ask, and the update poll currently always runs
///   while the app is open (no user-facing opt-out today). See the
///   ``RapidApp`` ``WindowGroup`` comment for the per-surface map.
///
/// Coupling auto-start to ``hideAlways`` would surprise users who
/// picked "Hide Dock icon" for a quieter menu-bar UX but still expect
/// the sidecar to be ready when they invoke Quick Ask (⌥+Space) or
/// open the window from the tray. Matches Ollama / LM Studio shape:
/// both auto-start in menu-bar-only mode unless explicitly disabled.
///
/// ``HideAlwaysOrthogonalToAutoStartTests`` pins the orthogonality
/// contract at the behavior level: ``userOptedIn: false`` short-
/// circuits every other gate, the default stays ``true`` (Ollama-
/// shape), and the two preferences live in non-overlapping
/// UserDefaults namespaces (``rapid.server.*`` vs ``rapid.window.*``).
enum AutoStartDecision: Equatable {
    /// All three gates hold. The launch hook should immediately call
    /// ``ServerManager.start(alias:)`` for this alias.
    case start(alias: String)

    /// Alias resolvable AND binary reachable, but the chosen alias's
    /// weights aren't on disk. Auto-start would silently kick off a
    /// multi-GB download — surface a download-aware CTA in the empty
    /// state instead. ``alias`` is the alias the user would download
    /// if they click Start.
    case promptDownload(alias: String)

    /// No auto-start action — server is already engaged, no alias
    /// resolves, or the binary is missing entirely. The existing
    /// install / manual-Start affordances own the frame.
    case skip(reason: SkipReason)

    /// Why a decision came back as ``.skip``. Surfaced as an
    /// associated value so the contract test can pin the precedence
    /// (user-opt-out beats onboarding-pending
    /// beats server-busy beats binary-missing beats no-alias) without
    /// inspecting log output, and so a future
    /// telemetry counter can bucket without re-deriving the gate.
    /// ``CaseIterable`` is the load-bearing conformance for the
    /// cardinality contract in ``AutoStartDecisionTests`` (issue
    /// #356). It lets a single ``allCases.count`` assertion fail
    /// loudly when a new ``SkipReason`` is added — forcing whoever
    /// adds the case to also update the precedence ladder above and
    /// audit every consumer (today only `==`-comparisons in tests;
    /// tomorrow potentially a telemetry-bucketing ``switch``).
    enum SkipReason: String, Equatable, CaseIterable {
        /// FU-1: the user explicitly turned OFF "Auto-start model on
        /// launch" in Settings → Models. Highest precedence so the
        /// opt-out wins over every other gate — when the user said
        /// "don't load a model" we don't even spend the catalog probe
        /// or surface a ``promptDownload`` caption that would suggest
        /// they should click Start.
        case userOptedOut
        /// #1589: Quickstart still owes this user onboarding
        /// (``QuickstartCoordinator.onboardingOwed``). Auto-start exists
        /// to restore *your last-used model*; a user who has never
        /// finished onboarding has no last-used model, so there is
        /// nothing to restore and inventing one both contradicts the
        /// Settings copy and — because starting moves ``serverState`` off
        /// ``.idle`` — suppresses the very wizard that was supposed to
        /// choose the first model. Onboarding wins the race.
        case onboardingPending
        /// The alias we would resume is a ``QuickstartCoordinator``
        /// retired starter — a model withdrawn for being unusable, not
        /// merely superseded. Auto-starting it would put the user back
        /// in the broken chat AND move ``serverState`` off ``.idle``,
        /// which suppresses the Quickstart rescue card that exists to
        /// get them off it. Skipping leaves the frame to that card.
        ///
        /// Narrower than ``onboardingPending`` and still reachable behind
        /// it: a user who dismissed Quickstart under v1 (``legacyDone``)
        /// with no ``lastServedAlias`` is not owed onboarding, yet the
        /// alphabetical cached-fallback can still land on the retired
        /// alias. This is the gate that catches them.
        case retiredStarter
        /// ``serverState`` is not ``.idle`` — either a previous
        /// ``.task`` already kicked an auto-start (re-entry), the
        /// user manually clicked Start before this helper fired, or
        /// the server already crashed and is awaiting user input.
        case serverNotIdle
        /// ``ServerLocator.find()`` returned nil. The
        /// ``missingOverlay`` in ``ContentView`` owns the frame —
        /// auto-start has nothing to launch.
        case binaryMissing
        /// No alias resolved from any of the three sources
        /// (last-served / bundled / first-cached). Fresh install with
        /// no bundled snapshot AND no prior downloads — Quickstart
        /// owns this frame.
        case noResolvableAlias
    }

    /// Pure decision function. All inputs are values, no side effects,
    /// no FS reads — the caller computes ``binaryReachable`` /
    /// ``cachedAliases`` from real probes once and threads the
    /// snapshot through.
    ///
    /// ``cachedAliases`` is the set of aliases the sidecar can serve
    /// today without a network round-trip (the cached half of
    /// ``ModelCatalog.load``). If the chosen alias is in the set we
    /// know cond-3 holds; if not, we still report it as
    /// ``.promptDownload`` so the empty-state caption can name the
    /// pending download in a user-actionable way.
    ///
    /// ``bundledFallbackAlias`` is normally
    /// ``BundledModel.firstLaunchAlias`` — the v0.7.1 instant-on
    /// alias when the snapshot is staged inside the .app. Caller may
    /// pass any string here for tests; the helper doesn't validate
    /// the alias against any catalog (the cache-membership check
    /// against ``cachedAliases`` is the de-facto "is this a real
    /// alias" probe — an alias the sidecar would refuse to serve is
    /// also never going to land in the user's HF cache).
    ///
    /// Precedence for alias resolution:
    /// 1. ``lastServedAlias`` — the user already picked something on
    ///    a prior launch; honour it even if it isn't cached (we'll
    ///    return ``.promptDownload``, the user's intent stays
    ///    represented). Gated by ``storedAliasIsServable`` when a
    ///    ``catalogAliases`` snapshot is supplied: a stored alias that
    ///    is not a catalog member, or that ``rejectsAlias`` refuses,
    ///    falls through to the tiers below rather than being restored
    ///    into a guaranteed serve failure (#1706).
    /// 2. ``bundledFallbackAlias`` — the BUNDLE_MODEL=1 dev build
    ///    instant-on path.
    /// 3. First entry in ``cachedAliases`` sorted alphabetically —
    ///    the upgrade-with-``defaults delete`` shape from #298. We
    ///    pick deterministically so the same Mac re-launching twice
    ///    in a row converges on the same alias without flapping.
    ///    A sorted pick beats "smallest by size" because we don't
    ///    have a footprint table here (caller didn't probe sizes);
    ///    and "smallest by alias length" / "most recent by mtime"
    ///    both add FS-stat overhead without changing the user
    ///    outcome — the user will swap via the picker the moment
    ///    they want something else.
    static func decide(
        lastServedAlias: String?,
        bundledFallbackAlias: String?,
        binaryReachable: Bool,
        cachedAliases: Set<String>,
        serverState: ServerState,
        catalogAliases: Set<String>? = nil,
        rejectsAlias: (String) -> Bool = { _ in false },
        userOptedIn: Bool = true,
        onboardingPending: Bool = false,
        isRetiredStarter: (String) -> Bool = { _ in false }
    ) -> AutoStartDecision {
        // FU-1 precedence #0 (highest): if the user has turned the
        // auto-start preference OFF in Settings → Models, never
        // spawn at launch — even if all 3 condition gates would pass.
        // The user can still trigger a manual start from the model
        // picker / Start CTA; this gate exclusively governs the
        // launch-time spawn. Placed above every other branch so we
        // don't surface a ``.promptDownload`` caption to a user who
        // explicitly said "don't try to load a model on launch."
        //
        // Defaulted to ``true`` at the parameter so all existing
        // call sites + tests retain their current contract — only the
        // launch hook in ``ContentView`` opts into the new gate by
        // threading the live ``@AppStorage`` value through.
        if !userOptedIn {
            return .skip(reason: .userOptedOut)
        }

        // #1589 precedence #0.5: the first-run surfaces own the launch
        // before auto-start gets a turn. Both gates sit ABOVE the
        // ``serverState`` switch deliberately — the whole defect was that
        // auto-start moved the state first and every downstream predicate
        // then read that self-inflicted ``.starting`` as evidence the user
        // was not new. A gate placed below the switch could only observe
        // the damage, never prevent it.
        //
        if onboardingPending {
            return .skip(reason: .onboardingPending)
        }

        // Idempotency precedence #1: only fire when state is `.idle`.
        // Anything else means a previous launch tick already acted,
        // the user is mid-flow, or the install overlay owns the
        // frame. Skip without picking an alias.
        switch serverState {
        case .idle:
            break
        case .missing:
            // Be explicit: binary missing = different skip reason than
            // "we never reached the helper". Lets the caller / tests
            // distinguish "install overlay" from "no auto-start".
            return .skip(reason: .binaryMissing)
        case .starting, .ready, .stopped, .crashed:
            return .skip(reason: .serverNotIdle)
        }

        // Idempotency precedence #2: even at `.idle`, an unreachable
        // binary means there's nothing to spawn. ContentView's
        // `.missing` overlay owns the user's next click.
        guard binaryReachable else {
            return .skip(reason: .binaryMissing)
        }

        // Alias resolution — strict precedence per the doc comment.
        // ``rejectsAlias`` always guards the CACHED-FALLBACK tier
        // (codex r1 MAJOR): the alphabetically-first cached entry is a
        // best-effort heuristic that must respect the same
        // ``ModelSizing.tooBig`` guard the picker's Start CTA enforces
        // (otherwise a cleared-defaults user with one large cached
        // model on disk gets auto-OOM'd on launch). Since #1706 it also
        // guards the STORED lastServed tier — together with a catalog-
        // membership check — whenever ``catalogAliases`` is supplied, so
        // a stale/oversized/renamed stored alias declines to the picker
        // instead of being restored into a failed serve. bundled stays
        // product-curated and unconditionally honoured.
        let resolved = resolveAlias(
            lastServedAlias: lastServedAlias,
            bundledFallbackAlias: bundledFallbackAlias,
            cachedAliases: cachedAliases,
            catalogAliases: catalogAliases,
            rejectsAlias: rejectsAlias
        )
        guard let alias = resolved else {
            return .skip(reason: .noResolvableAlias)
        }

        // A retired starter must not be resumed. Auto-start defaults to
        // ON, so without this the rescue is decorative: the stranded user
        // launches, we restart the broken model, ``serverState`` leaves
        // ``.idle``, and Quickstart's third gate hides the card that was
        // supposed to reach them. Placed after resolution so the reason is
        // specific rather than folded into ``noResolvableAlias``.
        if isRetiredStarter(alias) {
            return .skip(reason: .retiredStarter)
        }

        // Cond-3 gate — model on disk?
        if cachedAliases.contains(alias) {
            return .start(alias: alias)
        }
        return .promptDownload(alias: alias)
    }

    /// Empty-state caption when ``decide`` returned
    /// ``.promptDownload``. Surfaces the alias the user would
    /// download if they click Start, and (when supplied) a
    /// human-readable size — defending against the v0.7.0 footgun
    /// where "Idle / Start" hid a multi-GB pull behind a one-word
    /// button label.
    ///
    /// ``sizeText`` is the formatted footprint (e.g. "~17 GB"). When
    /// nil — the caller couldn't resolve a footprint — the copy
    /// degrades gracefully to a no-size variant the user can still
    /// act on.
    ///
    /// Pure for the same reason ``ChatView.emptyStatePoweredByCopy``
    /// is: pinning the copy template is a single ``#expect`` line in
    /// the contract test, no SwiftUI environment to stand up.
    static func promptDownloadCaption(alias: String, sizeText: String?) -> String {
        if let sizeText, !sizeText.isEmpty {
            return "Click Start to download \(alias) (\(sizeText))."
        }
        return "Click Start to download \(alias)."
    }

    // MARK: - Helpers

    /// Pure resolver — extracted so the precedence rules can be
    /// pinned without standing up the wider ``decide`` machinery.
    ///
    /// ``rejectsAlias`` is always consulted for the cached-fallback
    /// tier, and — when ``catalogAliases`` is supplied — for the
    /// stored ``lastServedAlias`` tier as well (see
    /// ``storedAliasIsServable`` and #1706). The bundled tier is
    /// product-curated and unconditionally honoured.
    private static func resolveAlias(
        lastServedAlias: String?,
        bundledFallbackAlias: String?,
        cachedAliases: Set<String>,
        catalogAliases: Set<String>?,
        rejectsAlias: (String) -> Bool
    ) -> String? {
        if let stored = trimmed(lastServedAlias),
           storedAliasIsServable(
               stored, catalogAliases: catalogAliases, rejectsAlias: rejectsAlias
           ) {
            return stored
        }
        if let bundled = trimmed(bundledFallbackAlias) {
            return bundled
        }
        // Sorted alphabetically so the launch decision converges on
        // the same alias across consecutive re-launches with the same
        // disk state. ``localizedStandardCompare`` mirrors the rule
        // ``ModelCatalog.load`` uses to sort the picker, so the auto-
        // start pick matches the row a user would scan first if they
        // were doing it by eye.
        //
        // ``rejectsAlias`` filters out picks the caller refuses (the
        // production caller uses ``ModelSizing.classify`` to reject
        // ``.tooBig`` — would-OOM-on-this-Mac aliases). When every
        // cached entry is rejected the resolver falls through to nil,
        // and ``decide`` reports ``.noResolvableAlias``; the user
        // sees the existing "Pick a model from the top bar" copy
        // rather than getting auto-OOM'd. Picking the first
        // ACCEPTABLE alias in alphabetical order keeps the
        // determinism guarantee for the common case.
        let sorted = cachedAliases
            .sorted(by: { $0.localizedStandardCompare($1) == .orderedAscending })
        for candidate in sorted {
            let trimmedCandidate = trimmed(candidate)
            guard let final = trimmedCandidate else { continue }
            if !rejectsAlias(final) {
                return final
            }
        }
        return nil
    }

    /// Whether a stored ``lastServedAlias`` may be resumed as-is.
    ///
    /// ## Why (issue #1706)
    ///
    /// Before this gate the stored alias was returned unconditionally.
    /// That let the desktop try to restore an alias the bundled engine
    /// cannot serve *now* — the reported case was an image-generation
    /// alias (``flux2-klein-4b``) left in ``rapid.serve.lastAlias`` by a
    /// different checkout's build, which is absent from the chat catalog
    /// entirely. The stored alias also stops being servable when it is
    /// renamed/dropped across engine versions, when the engine is
    /// downgraded, or when free memory has shrunk since the last run
    /// (the ``.tooBig`` classification is a per-launch fact, not a
    /// last-time guarantee). In every case the old path spent a model
    /// load on a request it could have known would fail, and greeted the
    /// returning user with a failure banner instead of the picker.
    ///
    /// Two membership-scoped checks, mirroring how ``CacheAwareDefault``
    /// validates the RAM-bucketed default:
    ///
    /// 1. **Catalog membership.** The alias must appear in the catalog
    ///    the sidecar can serve. A member that simply is not downloaded
    ///    yet still qualifies here — cached-vs-uncached is a separate
    ///    axis that ``decide`` uses to choose ``.start`` vs
    ///    ``.promptDownload``, so honouring an uncached-but-real alias
    ///    (the user's prior intent) is preserved.
    /// 2. **OOM guard.** The alias must pass ``rejectsAlias`` — the same
    ///    ``ModelSizing.classify(...) == .tooBig`` predicate the
    ///    cached-fallback tier and the picker's Start CTA enforce, so a
    ///    "worked last time" alias that no longer fits this Mac's free
    ///    memory declines to the picker rather than auto-OOM-ing.
    ///
    /// When either fails, ``resolveAlias`` falls through to the bundled
    /// / cached-scan ladder — "restore what I was using" stays the
    /// dominant behaviour and only yields when the alias genuinely
    /// cannot be served now.
    ///
    /// ``catalogAliases == nil`` means the caller did not supply an
    /// authoritative catalog snapshot (legacy call sites / unit tests
    /// that predate #1706). There we keep the historical unconditional-
    /// honour contract so those callers are unaffected; the production
    /// launch hook in ``ContentView`` always supplies the real set.
    private static func storedAliasIsServable(
        _ alias: String,
        catalogAliases: Set<String>?,
        rejectsAlias: (String) -> Bool
    ) -> Bool {
        guard let catalogAliases else { return true }
        guard catalogAliases.contains(alias) else { return false }
        return !rejectsAlias(alias)
    }

    private static func trimmed(_ s: String?) -> String? {
        guard let s else { return nil }
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }
}
