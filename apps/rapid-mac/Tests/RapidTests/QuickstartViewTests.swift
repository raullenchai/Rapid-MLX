import Foundation
import Testing
@testable import Rapid

/// Contract for the first-launch Quickstart surface.
///
/// Pins:
/// - the pinned alias / repo / display copy that drive both the UI and
///   the value bundled into ``aliases.json``
/// - eligibility predicate truth table — Quickstart shows ONLY when
///   (no persisted done flag) AND (no last-served alias) AND
///   (server in .idle / .stopped)
/// - state machine transitions idle → downloading → starting → ready,
///   with the seeded welcome message firing exactly once per flow
/// - failure path stays in .failed without flipping the persistent
///   done bit so the next launch re-offers the card
/// - friendlyFailureMessage classifier handles the common cold-start
///   error shapes (HF rate limit, network, disk full)
/// - progressSubtitle / etaCaption pure helpers render the right copy
///   under the tqdm phase shapes the downloading card renders against
@MainActor
@Suite("Quickstart — first-launch single-button onboarding")
struct QuickstartViewTests {

    // MARK: - Helpers

    /// Build a coordinator in a known-clean state. ``QuickstartCoordinator``
    /// reads the persistent ``rapid.quickstart.v2.done`` flag at init,
    /// so an earlier test could in principle have flipped it via
    /// production code; the explicit reset pins the starting condition.
    private func makeCoordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    // MARK: - Pinned identifiers

    @Test("Seeded welcome copy names the model and points at the picker")
    func seedMessageReadsWell() {
        let copy = makeCoordinator().seedMessage
        #expect(copy.contains(QuickstartCoordinator.defaultChoice.displayName))
        #expect(copy.lowercased().contains("picker"))
    }

    @Test("F-LWT-1: welcome message does NOT promise tool calling on 0.6B")
    func seedMessageDropsToolCallPromise() {
        // The pre-F-LWT-1 copy promised "It handles tool calls
        // (calculator, web search) reliably"; the swap explicitly
        // removes that promise because 0.6B cannot keep it. Users
        // who want the tool-calling demo trade up via the picker's
        // Recommended Default.
        let lowered = makeCoordinator().seedMessage.lowercased()
        #expect(!lowered.contains("tool call"))
        #expect(!lowered.contains("tool_call"))
        #expect(!lowered.contains("calculator"))
        #expect(!lowered.contains("web search"))
    }

    /// F-LWT-1 belt-and-suspenders: explicitly forbid the previous
    /// 4B pick. The principled checks above pin the new identifiers;
    /// this guard catches an accidental "revert to 4B for capability
    /// reasons" regression even if a future refactor accidentally
    /// re-introduces the literal.
    @Test("F-LWT-1: alias must NOT regress to ``qwen3.5-4b-4bit``")
    func aliasIsNotPreviousFourB() {
        #expect(
            QuickstartCoordinator.defaultChoice.alias != "qwen3.5-4b-4bit",
            "Quickstart alias must not regress to qwen3.5-4b-4bit — the starter is lfm2.5-1b-4bit, which keeps the small-download win while adding real tool calls."
        )
    }

    @Test("Persistent flag storage key is v1-versioned")
    func storageKeyVersioned() {
        // The OnboardingState pattern: versioned key so a future
        // Quickstart refresh (e.g. swapping the default model)
        // doesn't clobber the v1 flag and forces the new flow on
        // existing users. v1 is the only shipping version today.
        #expect(QuickstartCoordinator.storageKey == "rapid.quickstart.v2.done")
    }

    // MARK: - Eligibility predicate (the critical contract)

    @Test("Eligible: fresh install — no done, no last-served, idle server")
    func eligibleOnFreshInstall() {
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: nil,
            serverState: .idle
        ))
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: nil,
            serverState: .stopped
        ))
    }

    @Test("Not eligible: persistent done flag set (one-shot guard)")
    func notEligibleWhenDone() {
        // Even on a fresh server state, the done flag must hold —
        // a user who completed Quickstart on this Mac last week and
        // later deleted the model should NOT see Quickstart return.
        #expect(!QuickstartCoordinator.isEligible(
            done: true,
            lastServedAlias: nil,
            serverState: .idle
        ))
    }

    @Test("Not eligible: a last-served alias already exists")
    func notEligibleWhenLastServedAliasPresent() {
        // The user has previously served SOMETHING (didn't go
        // through Quickstart; manually picked from the picker).
        // We don't want to interrupt their next launch by stepping
        // in front of the chat surface.
        #expect(!QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: "qwen3.6-27b",
            serverState: .idle
        ))
    }

    @Test("Not eligible: server already engaged (ready / starting / crashed / missing)")
    func notEligibleWhenServerEngaged() {
        for state: ServerState in [
            .ready(alias: "gemma3-1b-qat-4bit"),
            .starting(alias: "gemma3-1b-qat-4bit"),
            .crashed(alias: "gemma3-1b-qat-4bit", message: "boom"),
            .missing,
        ] {
            #expect(!QuickstartCoordinator.isEligible(
                done: false,
                lastServedAlias: nil,
                serverState: state
            ), "Expected not-eligible for state \(state)")
        }
    }

    @Test("Regression: unrelated HF-cached models no longer suppress Quickstart")
    func eligibleEvenWithUnrelatedHFCache() {
        // Regression for the over-broad #298 gate: the eligibility
        // predicate must depend ONLY on app-owned state (done +
        // lastServedAlias + serverState), never on the shared HF
        // cache. A brand-new user whose ~/.cache/huggingface/hub holds
        // Whisper / VAD / forced-aligner models from some OTHER tool
        // has never used THIS app, so onboarding must still fire.
        // There is deliberately no cache parameter to pass anymore —
        // that this compiles and returns true IS the guarantee.
        #expect(QuickstartCoordinator.isEligible(
            done: false,
            lastServedAlias: nil,
            serverState: .idle
        ))
    }

    // MARK: - State machine

    @Test("State machine: idle → downloading → starting → ready seeds welcome once")
    func stateMachineHappyPath() {
        let coord = makeCoordinator()
        #expect(coord.phase == .idle)
        #expect(!coord.done)

        coord.enterDownloading()
        #expect(coord.phase == .downloading)
        #expect(!coord.done, "done flag must NOT flip mid-flow")

        coord.enterStarting()
        #expect(coord.phase == .starting)
        #expect(!coord.done)

        var seedCount = 0
        let landed = coord.markReady { seedCount += 1; return true }
        #expect(landed)
        #expect(coord.phase == .ready)
        #expect(coord.done, "markReady must flip the persistent done flag on success")
        #expect(seedCount == 1, "seed closure must fire exactly once")

        // A second .ready notification (auto-respawn cycle, scheduler
        // tick, etc.) must NOT re-seed the welcome message.
        let landedAgain = coord.markReady { seedCount += 1; return true }
        #expect(!landedAgain)
        #expect(seedCount == 1, "seed closure must not double-fire")
        #expect(coord.done)
    }

    @Test("Codex r2 MAJOR: seed-returns-false must NOT flip done or hasSeededWelcome")
    func seedFailureKeepsRetryDoorOpen() {
        let coord = makeCoordinator()
        coord.enterStarting()
        // Simulate "no active session yet" — onSeedWelcome returns
        // false; markReady must respect that and leave both gates
        // open so a later .ready tick (after the user creates a
        // session, or auto-restart finishes) can finish the seed.
        let landed = coord.markReady { false }
        #expect(!landed, "markReady must report failure when seed returned false")
        #expect(!coord.done, "done must NOT flip when seed reported failure")
        #expect(!coord.hasSeededWelcome, "hasSeededWelcome must NOT flip when seed reported failure")
        #expect(coord.awaitingWelcomeSeed, "awaitingWelcomeSeed must flip so the parent's retry observer can fire")
        // Phase still progresses so the visibility predicate's
        // in-flight gate releases — the alternative (keep .starting)
        // would lock the user out of chat indefinitely.
        #expect(coord.phase == .ready)

        // A later retry tick with a seed that succeeds completes the
        // flow properly. Codex r3 MAJOR closure: ContentView's
        // ``.onChange(of: store.activeID)`` observer fires this re-try
        // when an active session finally lands — without that observer
        // the user would be permanently denied the welcome message.
        var retrySeedCount = 0
        let retried = coord.markReady { retrySeedCount += 1; return true }
        #expect(retried)
        #expect(retrySeedCount == 1)
        #expect(coord.done)
        #expect(coord.hasSeededWelcome)
        #expect(!coord.awaitingWelcomeSeed, "awaitingWelcomeSeed must clear on successful seed")
    }

    @Test("Codex r4 MAJOR: awaitingWelcomeSeed clears on revoke / fresh-Quickstart click")
    func awaitingWelcomeSeedClearsOnIntentChange() {
        // Provenance scenario codex flagged: without the flag, a user
        // who dismissed Quickstart and later manually started
        // gemma3-1b-qat-4bit from the picker could see a stray welcome
        // injected into their first chat. Pin that:
        //   1. releaseInFlight (foreign-alias revoke) clears the flag
        //   2. enterDownloading (fresh Quickstart click) clears any
        //      stale flag from a prior aborted flow
        let coord = makeCoordinator()
        coord.enterStarting()
        _ = coord.markReady { false }
        #expect(coord.awaitingWelcomeSeed)

        // Scenario 1: user clicks "or browse all models" or picks
        // another alias from the picker → releaseInFlight.
        coord.releaseInFlight()
        #expect(!coord.awaitingWelcomeSeed)

        // Reset and set it again to test the other clear path.
        coord._testingReset()
        coord.enterStarting()
        _ = coord.markReady { false }
        #expect(coord.awaitingWelcomeSeed)

        // Scenario 2: user clicks Get started AGAIN (kicks a fresh
        // flow). enterDownloading clears the stale provenance so the
        // new flow's own seed cycle controls the flag.
        coord.enterDownloading()
        #expect(!coord.awaitingWelcomeSeed)
    }

    @Test("Codex r5 MAJOR: awaitingWelcomeSeed persists across coordinator init")
    func awaitingSeedSurvivesRelaunch() {
        // Quit-mid-flow scenario codex flagged: server reaches .ready,
        // ServerManager persists lastServedAlias; user quits before
        // activeID lands. Without persistence of awaitingWelcomeSeed,
        // the next launch sees Quickstart ineligible (lastServedAlias
        // == quickstart-alias) AND the in-memory flag is lost, so the
        // welcome is permanently skipped. Persistence closes that gap.
        let coord = makeCoordinator()
        coord.enterStarting()
        _ = coord.markReady { false }
        #expect(coord.awaitingWelcomeSeed)

        // Simulate process restart by constructing a fresh coordinator
        // — same shape as ``donePersists`` test above. The persisted
        // flag must round-trip.
        let next = QuickstartCoordinator()
        #expect(next.awaitingWelcomeSeed, "awaitingWelcomeSeed must survive across coordinator init")
        next._testingReset()
    }

    @Test("#1524: a non-default wizard pick's deferred seed restores its selection on relaunch")
    func awaitingSeedRestoresNonDefaultSelectionAcrossRelaunch() {
        // #1524 regression: the deferred-seed path persists a flag but
        // ``selection`` is NOT persisted — a fresh coordinator re-inits it
        // to ``defaultChoice`` (0.6B). Before the alias-persist fix, a
        // user who picked a BIGGER model, reached ``.ready``, then quit
        // before ``activeID`` landed would relaunch with ``selection`` at
        // 0.6B; ContentView's seed observers compare the served alias
        // (the bigger model) against ``selection.alias`` (0.6B), mismatch,
        // and CLEAR the pending seed → welcome message permanently lost.
        let bigger = QuickstartCoordinator.onboardingChoices.first { !$0.isStarter }
        #expect(bigger != nil, "onboarding ladder must offer a non-starter trade-up")
        guard let bigger else { return }

        let coord = makeCoordinator()
        coord.select(bigger)
        #expect(coord.selection == bigger, "select() must retarget while idle")
        coord.enterStarting()
        _ = coord.markReady { false }
        #expect(coord.awaitingWelcomeSeed)

        // Simulate process restart: the fresh coordinator must restore
        // ``selection`` to the bigger pick (from the persisted alias), not
        // fall back to the 0.6B default — otherwise the seed observers
        // would drop the welcome for exactly this non-default flow.
        let next = QuickstartCoordinator()
        #expect(next.awaitingWelcomeSeed, "flag must survive relaunch")
        #expect(next.selection.alias == bigger.alias,
                "selection must be restored to the in-flight non-default pick, not reset to defaultChoice")
        #expect(next.selection == bigger, "the full restored choice must round-trip (name + isStarter drive the seed copy)")
        #expect(!next.seedMessage.contains(QuickstartCoordinator.defaultChoice.displayName),
                "the welcome copy must name the restored model, not the 0.6B default")
        next._testingReset()
    }

    @Test("Codex r5 MODERATE: clearPendingSeed clears flag (used by ContentView server-state observer)")
    func clearPendingSeedExposesExternalClear() {
        // The ContentView observer for ``server.state`` needs an
        // external entry point to drop the provenance flag when the
        // user switches to a different model after entering the
        // deferred-seed state. Pin the public surface.
        let coord = makeCoordinator()
        coord.enterStarting()
        _ = coord.markReady { false }
        #expect(coord.awaitingWelcomeSeed)
        coord.clearPendingSeed()
        #expect(!coord.awaitingWelcomeSeed)
        // Idempotent — clearing twice is a no-op.
        coord.clearPendingSeed()
        #expect(!coord.awaitingWelcomeSeed)
    }

    @Test("Codex r4 MAJOR: successful seed does NOT set awaitingWelcomeSeed (steady-state)")
    func successfulSeedNeverSetsAwaiting() {
        // The flag is supposed to be a provenance signal for the
        // DEFERRED path only. A happy-path Quickstart whose seed lands
        // on the first call must never raise the flag — otherwise the
        // observer in ContentView would fire spuriously on the next
        // activeID change (e.g. user creates a new chat after the
        // flow completes).
        let coord = makeCoordinator()
        coord.enterStarting()
        _ = coord.markReady { true }
        #expect(coord.done)
        #expect(coord.hasSeededWelcome)
        #expect(!coord.awaitingWelcomeSeed)
    }

    @Test("releaseInFlight: phase flips to ready WITHOUT seed or done")
    func releaseInFlightDoesNotCommit() {
        // Codex r2 BLOCKING: user clicks Get started, then picks a
        // DIFFERENT model from the still-visible picker. The
        // ``handleDownloadStatusChange`` /
        // ``handleServerStateChange`` paths short-circuit through
        // ``releaseInFlight`` to release Quickstart visibility
        // WITHOUT marking the flow as done — they never saw the
        // welcome, so a fresh install on a different Mac should
        // still see Quickstart.
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.releaseInFlight()
        #expect(coord.phase == .ready)
        #expect(!coord.done, "releaseInFlight must NOT flip the persistent done flag")
        #expect(!coord.hasSeededWelcome)
    }

    @Test("serverEngagedWithDifferentAlias predicate truth table")
    func serverEngagedPredicate() {
        let quickstartAlias = QuickstartCoordinator.defaultChoice.alias
        // Engaged with a different alias → true (Quickstart cedes).
        #expect(ContentView.serverEngagedWithDifferentAlias(
            state: .ready(alias: "qwen3.6-27b"),
            quickstartAlias: quickstartAlias
        ))
        #expect(ContentView.serverEngagedWithDifferentAlias(
            state: .starting(alias: "qwen3.6-27b"),
            quickstartAlias: quickstartAlias
        ))
        #expect(ContentView.serverEngagedWithDifferentAlias(
            state: .crashed(alias: "qwen3.6-27b", message: "boom"),
            quickstartAlias: quickstartAlias
        ))
        // Engaged with our OWN alias → false (we're in the happy path).
        #expect(!ContentView.serverEngagedWithDifferentAlias(
            state: .ready(alias: quickstartAlias),
            quickstartAlias: quickstartAlias
        ))
        // Not engaged at all → false (eligibility decides).
        for state: ServerState in [.idle, .stopped, .missing] {
            #expect(!ContentView.serverEngagedWithDifferentAlias(
                state: state,
                quickstartAlias: quickstartAlias
            ))
        }
    }

    @Test("Failure path: enterFailed records message and does NOT flip done")
    func failurePathPreservesEligibility() {
        let coord = makeCoordinator()
        coord.enterDownloading()
        coord.enterFailed(message: "network unreachable")
        if case .failed(let message) = coord.phase {
            #expect(message == "network unreachable")
        } else {
            Issue.record("Expected .failed phase, got \(coord.phase)")
        }
        #expect(!coord.done, "Failed Quickstart must NOT flip the done flag — next launch re-offers the card")
        // Eligibility should still report TRUE on the next launch
        // (idle server, no last-served alias) so the user gets a
        // retry on relaunch even if they Cmd+Q'd between attempts.
        #expect(QuickstartCoordinator.isEligible(
            done: coord.done,
            lastServedAlias: nil,
            serverState: .idle
        ))
    }

    @Test("Persistent done flag survives via UserDefaults round-trip")
    func donePersists() {
        let coord = makeCoordinator()
        coord.markDone()
        #expect(coord.done)
        // Fresh coordinator picks up the flag the previous one left
        // — same shape as ``OnboardingState.hasSeen``.
        let next = QuickstartCoordinator()
        #expect(next.done, "fresh QuickstartCoordinator must read the persisted flag")
        // Clean up so we don't leak the flag into other tests.
        next._testingReset()
    }

    @Test("Reset clears the done flag and resets phase to idle")
    func resetClearsEverything() {
        let coord = makeCoordinator()
        coord.markDone()
        coord.enterFailed(message: "boom")
        coord._testingReset()
        #expect(!coord.done)
        #expect(coord.phase == .idle)
        #expect(!coord.hasSeededWelcome)
    }

    // MARK: - Failure classifier (friendlyFailureMessage)

    @Test("Failure classifier recognises HF 429 / rate-limit")
    func friendlyFailureRateLimit() {
        let msg = QuickstartView.friendlyFailureMessage(
            raw: "HTTPError: 429 Too Many Requests"
        )
        #expect(msg.contains("rate-limiting") || msg.lowercased().contains("rate"))
        #expect(msg.lowercased().contains("try again"))
    }

    @Test("Failure classifier recognises network errors")
    func friendlyFailureNetwork() {
        let msg = QuickstartView.friendlyFailureMessage(
            raw: "ConnectionResetError: peer reset"
        )
        #expect(msg.lowercased().contains("network") || msg.lowercased().contains("connection"))
        #expect(msg.lowercased().contains("retry"))
    }

    @Test("Failure classifier recognises disk-full errors")
    func friendlyFailureDiskFull() {
        let msg = QuickstartView.friendlyFailureMessage(
            raw: "OSError: [Errno 28] No space left on device"
        )
        #expect(msg.lowercased().contains("disk space"))
    }

    @Test("Failure classifier replaces unrecognised internals with safe fallback copy")
    func friendlyFailureUsesSafeFallback() {
        let raw = "ValueError: alias resolution failed for foo"
        let msg = QuickstartView.friendlyFailureMessage(raw: raw)
        #expect(msg != raw)
        #expect(!msg.contains("ValueError"))
        #expect(msg.lowercased().contains("try again"))
    }

    @Test("Failure classifier handles empty stderr tails")
    func friendlyFailureEmptyInput() {
        let msg = QuickstartView.friendlyFailureMessage(raw: "")
        #expect(!msg.isEmpty, "must not surface an empty bubble to the user")
    }

    // MARK: - Progress helpers

    @Test("progressSubtitle falls back to friendly copy when no job exists")
    func progressSubtitleNoJob() {
        let copy = QuickstartView.progressSubtitle(
            job: nil,
            displayName: "Gemma 3 1B QAT"
        )
        #expect(!copy.isEmpty)
        // Must not surface "nil" or stack-trace style debug output —
        // this is the moment between the user clicking Get started
        // and HF resolving the first byte.
        #expect(!copy.lowercased().contains("nil"))
    }

    @Test("etaCaption returns nil when tqdm hasn't stabilised an ETA")
    func etaCaptionNilEarly() {
        let mgr = DownloadManager()
        let job = mgr._testingSeedJob(alias: QuickstartCoordinator.defaultChoice.alias)
        // ``.idle`` phase — no ETA available yet.
        #expect(QuickstartView.etaCaption(job: job) == nil)
    }

    // MARK: - Regression: post-success handoff to ChatView

    @Test("Codex r1 BLOCKING regression: .ready phase + done flag → eligibility predicate returns false")
    func readyPhaseAfterSuccessReportsNotEligible() {
        // The integration bug codex flagged: after ``markReady`` fires
        // ``done = true``, the parent view must stop rendering the
        // Quickstart card and hand the frame off to ChatView. The
        // eligibility predicate is the single gate the parent consults,
        // so pin it: ``done == true`` immediately makes the predicate
        // return ``false`` regardless of phase or other inputs.
        let coord = makeCoordinator()
        coord.enterDownloading()
        _ = coord.markReady { true }
        #expect(coord.done)
        #expect(coord.phase == .ready)
        // Even with ``.ready`` phase still set, an isEligible read on
        // a freshly idle server must report not-eligible — proving the
        // persistent done flag short-circuits ``isEligible`` and that
        // the ``.ready`` phase is NOT relied on as a sticky "keep card
        // up" signal.
        #expect(!QuickstartCoordinator.isEligible(
            done: coord.done,
            lastServedAlias: nil,
            serverState: .idle
        ))
        // And of course the post-handoff server state (.ready with the
        // Quickstart alias) also pins not-eligible.
        #expect(!QuickstartCoordinator.isEligible(
            done: coord.done,
            lastServedAlias: QuickstartCoordinator.defaultChoice.alias,
            serverState: .ready(alias: QuickstartCoordinator.defaultChoice.alias)
        ))
    }
}

/// F-LWT-1 source-grep tripwire: catch a partial swap that leaves
/// the file mixing old 4B constants with the new 0.6B identifiers.
/// SwiftUI files often duplicate string literals across docstrings,
/// computed properties, and accessibility labels; a one-line drift
/// is exactly the kind of bug the F-LWT-1 review explicitly called
/// out.
@MainActor
@Suite("Quickstart source-grep tripwires — F-LWT-1 partial-swap guard")
struct QuickstartViewSourceGrepTests {

    // #1524: the old blanket source-grep for the ``qwen3.5-4b-4bit``
    // literal is retired — 4B is now a legitimate *bigger trade-up*
    // option in ``QuickstartCoordinator.onboardingChoices`` — so a
    // whole-file grep would fire on an intentional line. The intent
    // (the STARTER must stay the small lfm2.5-1b-4bit pick, never
    // the old 4B one) is preserved as direct value assertions on
    // ``defaultChoice``.

    @Test("Starter alias stays the lfm2.5-1b-4bit pick, never regresses to the old 4B")
    func starterAliasNotOld4B() {
        #expect(QuickstartCoordinator.defaultChoice.alias == "lfm2.5-1b-4bit")
        #expect(QuickstartCoordinator.defaultChoice.alias != "qwen3.5-4b-4bit")
    }

    @Test("Starter hfRepo is the bonsai repo, not the old 4B repo")
    func starterHFRepoNotOld4B() {
        #expect(QuickstartCoordinator.defaultChoice.hfRepo == "mlx-community/LFM2.5-1.2B-Instruct-4bit")
        #expect(QuickstartCoordinator.defaultChoice.hfRepo?.contains("4B") != true)
    }
}
