import Foundation
import Testing
@testable import Rapid

/// Issue #1589 — the launch auto-start raced the Quickstart onboarding
/// wizard and always won, so on any Mac with something already in the
/// shared Hugging Face cache the entire first-run wizard was unreachable.
///
/// ## The defect, precisely
///
/// ``ContentView.runLaunchAutoStart`` fired on launch and started the
/// alphabetically-first cached alias. That moved ``ServerManager.state``
/// to ``.starting`` *before* ``ContentView.quickstartVisible`` was ever
/// evaluated, and the sheet's predicate then failed twice over on state
/// the app had inflicted on itself:
///
///   * ``ContentView.serverEngagedWithDifferentAlias`` — true for
///     ``.starting(alias)`` whenever the alias differs from the Quickstart
///     starter, which it always did;
///   * ``QuickstartCoordinator.isEligible`` gate 3 — false for
///     ``.starting`` and ``.ready``.
///
/// Verified empirically on the reporter's Mac: identical fresh state,
/// flipping only ``rapid.server.auto_start_on_launch.v1`` decided whether
/// the wizard appeared at all.
///
/// ## Why these tests are shaped like this
///
/// The failure was a race between two code paths that never referenced
/// each other, so neither path's own tests could catch it — each was
/// individually correct. What was missing was a test of the ORDER. Every
/// case below therefore asserts on BOTH sides of the race at once: what
/// the launch gate decides, and what the wizard predicate says about the
/// same user. A future change that desynchronises them fails here rather
/// than shipping.
// ``QuickstartCoordinator`` is ``@MainActor`` (it backs a SwiftUI view),
// so the suite adopts the same isolation — matching ``QuickstartViewTests``.
@MainActor
@Suite("#1589 — launch auto-start must not outrace first-run onboarding")
struct LaunchOnboardingOrderingTests {

    @Test("Quickstart cannot interrupt Audio, Images, or Launch")
    func quickstartOnlyPresentsOnChat() {
        #expect(ContentView.quickstartCanPresent(in: .chat))
        #expect(!ContentView.quickstartCanPresent(in: .audio))
        #expect(!ContentView.quickstartCanPresent(in: .images))
        #expect(!ContentView.quickstartCanPresent(in: .launch))
    }

    /// Convenience: the launch gate exactly as ``ContentView`` calls it,
    /// for a user sitting at the pre-auto-start ``.idle`` state.
    private func launchDecision(
        lastServedAlias: String?,
        cachedAliases: Set<String>,
        done: Bool = false,
        legacyDone: Bool = false,
        serverState: ServerState = .idle
    ) -> AutoStartDecision {
        AutoStartDecision.decide(
            lastServedAlias: lastServedAlias,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: cachedAliases,
            serverState: serverState,
            userOptedIn: true,
            onboardingPending: QuickstartCoordinator.onboardingOwed(
                done: done,
                legacyDone: legacyDone,
                lastServedAlias: lastServedAlias
            ),
            isRetiredStarter: { alias in
                !done && QuickstartCoordinator.retiredStarters.contains(alias)
            }
        )
    }

    // MARK: - Repro

    /// The exact reported shape: never-onboarded user, empty app state,
    /// but a populated shared HF cache (CLI user, reinstall, upgrade, or
    /// a model pulled by any other route). Pre-fix this returned
    /// ``.start(alias: "bonsai-27b-2bit")`` — an 8.4 GB 2-bit 27B nobody
    /// chose — and the wizard never rendered.
    @Test("#1589 repro: never-onboarded user with a cached model does NOT auto-start, and DOES see the wizard")
    func neverOnboardedWithCachedModelDefersToWizard() {
        let cached: Set<String> = ["bonsai-27b-2bit", "qwen3.5-4b-4bit"]

        let decision = launchDecision(lastServedAlias: nil, cachedAliases: cached)
        #expect(
            decision == .skip(reason: .onboardingPending),
            "auto-start must stand down for a user onboarding is still owed to"
        )

        // The other half of the race: because nothing started, the server
        // is still `.idle` and the wizard's own predicate now passes.
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: nil,
            serverState: .idle
        ))
        #expect(!ContentView.serverEngagedWithDifferentAlias(
            state: .idle,
            quickstartAlias: QuickstartCoordinator.defaultChoice.alias
        ))
    }

    @Test("legacy audio ownership cannot suppress first-run onboarding")
    func legacyAudioAliasStillOwesOnboarding() {
        let audio = ModelEntry(
            alias: "speech-input",
            hfRepo: "example/speech-input",
            sizeOnDisk: "500 MB",
            cached: true,
            kind: .audio,
            audioCapability: .transcription
        )
        let launchPlan = SessionModelRestore.launchPlan(
            legacyLastAlias: audio.alias,
            dictationAlias: audio.alias,
            speechAlias: nil,
            catalog: [audio],
            autoStartEnabled: false
        )
        let restored = launchPlan.models

        #expect(restored.chatAlias == nil)
        #expect(launchPlan.chatAliasResolved)
        #expect(!launchPlan.shouldAutoStart)
        #expect(QuickstartCoordinator.onboardingOwed(
            done: false,
            legacyDone: false,
            lastServedAlias: restored.chatAlias
        ))
        #expect(launchDecision(
            lastServedAlias: restored.chatAlias,
            cachedAliases: [audio.alias]
        ) == .skip(reason: .onboardingPending))
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            legacyDone: false,
            lastServedAlias: restored.chatAlias,
            serverState: .idle
        ))
    }

    /// The pre-fix state, asserted from the wizard's side, so the test
    /// file documents *why* the gate has to sit where it does. Had
    /// auto-start been allowed to run, this is what the sheet would have
    /// seen — and both predicates say "not a new user" purely because of
    /// the app's own action.
    @Test("#1589 mechanism: a self-inflicted .starting state defeats BOTH wizard predicates")
    func selfInflictedStartingStateDefeatsTheWizard() {
        let autoStarted = ServerState.starting(alias: "bonsai-27b-2bit")
        #expect(!QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: nil,
            serverState: autoStarted
        ), "gate 3 reads .starting as 'a model is already engaged'")
        #expect(ContentView.serverEngagedWithDifferentAlias(
            state: autoStarted,
            quickstartAlias: QuickstartCoordinator.defaultChoice.alias
        ), "and the alias differs from the starter, so the sheet cedes too")
        // Which is why the gate is placed ABOVE the serverState switch in
        // `decide`: a check that runs after the start can only observe the
        // damage, never prevent it.
    }

    // MARK: - The case auto-start exists for must not regress

    @Test("Returning user with a lastServedAlias still auto-starts exactly as before")
    func returningUserStillAutoStarts() {
        let decision = launchDecision(
            lastServedAlias: "qwen3.5-4b-4bit",
            cachedAliases: ["bonsai-27b-2bit", "qwen3.5-4b-4bit"]
        )
        #expect(decision == .start(alias: "qwen3.5-4b-4bit"))
        // And the wizard stays away for them.
        #expect(!QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: "qwen3.5-4b-4bit",
            serverState: .idle
        ))
    }

    /// A returning user whose model is no longer on disk keeps the
    /// download-prompt CTA — the new gates must not swallow it.
    @Test("Returning user whose lastServedAlias is not cached still gets promptDownload")
    func returningUserUncachedStillPromptsDownload() {
        let decision = launchDecision(
            lastServedAlias: "qwen3.5-4b-4bit",
            cachedAliases: []
        )
        #expect(decision == .promptDownload(alias: "qwen3.5-4b-4bit"))
    }

    /// A user who completed onboarding but later cleared defaults (#298)
    /// has no lastServedAlias, yet `done` is set — auto-start's cached
    /// fallback is exactly what that shape needs and must survive.
    @Test("Onboarded user with cleared lastServedAlias still auto-starts from the cached fallback")
    func onboardedUserWithClearedAliasStillAutoStarts() {
        let decision = launchDecision(
            lastServedAlias: nil,
            cachedAliases: ["qwen3.5-4b-4bit", "gemma3-1b-qat-4bit"],
            done: true
        )
        #expect(decision == .start(alias: "gemma3-1b-qat-4bit"))
    }

    // MARK: - Stranded-starter carve-out (must keep working)

    /// A user whose recorded starter was retired for being unusable is
    /// deliberately treated as new. Auto-start must not resume the broken
    /// model — doing so is what strands them, because the resulting
    /// `.starting` suppresses the rescue card.
    @Test("Stranded-starter user is treated as new: no auto-start, wizard eligible")
    func strandedStarterTreatedAsNew() {
        let retired = QuickstartCoordinator.retiredStarters.first!
        #expect(QuickstartCoordinator.onboardingOwed(
            done: false,
            lastServedAlias: retired
        ), "the retired-starter carve-out must survive the extraction")

        let decision = launchDecision(
            lastServedAlias: retired,
            cachedAliases: [retired, "qwen3.5-4b-4bit"]
        )
        #expect(decision == .skip(reason: .onboardingPending))
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: retired,
            serverState: .idle
        ))
    }

    /// `.retiredStarter` is narrower than `.onboardingPending` but still
    /// reachable behind it — a v1 dismisser with no lastServedAlias is not
    /// owed onboarding, yet the alphabetical cached fallback can still
    /// land on the retired alias. Pins that the older gate did not become
    /// dead code.
    @Test("retiredStarter remains reachable behind onboardingPending (v1 dismisser, cached fallback)")
    func retiredStarterStillReachable() {
        let retired = QuickstartCoordinator.retiredStarters.first!
        // legacyDone + no lastServedAlias → not owed onboarding …
        #expect(!QuickstartCoordinator.onboardingOwed(
            done: false,
            legacyDone: true,
            lastServedAlias: nil
        ))
        // … but the cached fallback resolves to the retired alias.
        let decision = launchDecision(
            lastServedAlias: nil,
            cachedAliases: [retired, "zzz-some-other-model"],
            legacyDone: true
        )
        #expect(decision == .skip(reason: .retiredStarter))
    }

    // MARK: - Precedence ladder

    /// The skip reasons are a diagnostic surface, so their order has to be
    /// stable and meaningful: the user's explicit opt-out outranks
    /// everything, then onboarding, then mechanical state.
    @Test("Skip precedence: userOptedOut > onboardingPending > serverNotIdle")
    func skipPrecedenceLadder() {
        // Opt-out beats both new gates.
        #expect(AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .idle,
            userOptedIn: false,
            onboardingPending: true
        ) == .skip(reason: .userOptedOut))

        // Onboarding beats the mechanical serverState skip — the gate has
        // to sit above the switch or it cannot prevent the race.
        #expect(AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .starting(alias: "bonsai-27b-2bit"),
            onboardingPending: true
        ) == .skip(reason: .onboardingPending))
    }

    /// Back-compat anchor, same shape as the existing `userOptedIn`
    /// default pin: the new parameter defaults to "not pending" so every
    /// pre-#1589 call site and test keeps its contract.
    @Test("The onboarding gate defaults to false — pre-#1589 callers are unchanged")
    func onboardingGateDefaultsToFalse() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-4b-4bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .idle
        )
        #expect(decision == .start(alias: "qwen3.5-4b-4bit"))
    }

    // MARK: - The desynchronisation guard

    /// The invariant the whole fix exists to hold, over the full cross
    /// product of the persisted first-run state: **auto-start never starts
    /// a model for a user the wizard would still be offered to.**
    ///
    /// This is the assertion that would have failed before the fix, and it
    /// is deliberately expressed in terms of the two public predicates
    /// rather than their internals — so it keeps holding whichever side a
    /// future change touches, and fails the moment they disagree.
    @Test("Invariant: decide() never returns .start while the wizard is still eligible")
    func autoStartAndWizardNeverDisagree() {
        let aliasOptions: [String?] = [
            nil,
            "qwen3.5-4b-4bit",
            QuickstartCoordinator.retiredStarters.first!,
        ]
        for done in [false, true] {
            for legacyDone in [false, true] {
                for lastServed in aliasOptions {
                    let decision = launchDecision(
                        lastServedAlias: lastServed,
                        cachedAliases: [
                            "bonsai-27b-2bit",
                            "qwen3.5-4b-4bit",
                            QuickstartCoordinator.retiredStarters.first!,
                        ],
                        done: done,
                        legacyDone: legacyDone
                    )
                    let wizardEligible = QuickstartCoordinator.isEligible(
                        done: done,
                        legacyDone: legacyDone,
                        lastServedAlias: lastServed,
                        // `.idle` is what the wizard WOULD see if auto-start
                        // keeps its hands off — the whole point of the fix.
                        serverState: .idle
                    )
                    if wizardEligible {
                        #expect(
                            decision == .skip(reason: .onboardingPending),
                            """
                            wizard eligible but auto-start decided \(decision) \
                            for done=\(done) legacyDone=\(legacyDone) \
                            lastServed=\(lastServed ?? "nil")
                            """
                        )
                    }
                }
            }
        }
    }

    /// `onboardingOwed` must be exactly `isEligible` minus gate 3 — the
    /// extraction is only safe if it did not quietly change the answer.
    @Test("Extraction is behaviour-preserving: isEligible == onboardingOwed && serverState in {.idle, .stopped}")
    func extractionPreservesEligibility() {
        let aliasOptions: [String?] = [
            nil,
            "qwen3.5-4b-4bit",
            QuickstartCoordinator.retiredStarters.first!,
        ]
        let states: [ServerState] = [
            .idle,
            .stopped,
            .missing,
            .ready(alias: "qwen3.5-4b-4bit"),
            .starting(alias: "qwen3.5-4b-4bit"),
            .crashed(alias: "qwen3.5-4b-4bit", message: "boom"),
        ]
        for done in [false, true] {
            for legacyDone in [false, true] {
                for lastServed in aliasOptions {
                    let owed = QuickstartCoordinator.onboardingOwed(
                        done: done,
                        legacyDone: legacyDone,
                        lastServedAlias: lastServed
                    )
                    for state in states {
                        let stateAllows: Bool
                        switch state {
                        case .idle, .stopped: stateAllows = true
                        case .ready, .starting, .crashed, .missing: stateAllows = false
                        }
                        #expect(QuickstartCoordinator.isEligible(
                            done: done,
                            legacyDone: legacyDone,
                            lastServedAlias: lastServed,
                            serverState: state
                        ) == (owed && stateAllows))
                    }
                }
            }
        }
    }
}
