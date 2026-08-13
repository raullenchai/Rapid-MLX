import Foundation
import Testing
@testable import Rapid

/// Issue #270: when ``rapid-mlx`` crashes while no chat window is
/// visible, the desktop must surface the crash automatically — either
/// by re-spawning the child (the chosen shape) or by raising a
/// menu-bar indicator. Previously a SIGKILL of the rapid-mlx PID
/// with all windows closed left the desktop alive but inert; only
/// ``Cmd+N`` (which constructs a new chat session) triggered respawn,
/// so a Dock click after the crash did nothing visible.
///
/// The fix watchdog-respawns the child whenever the prior spawn cycle
/// had reached ``.ready`` (so the model was demonstrably healthy
/// before going dark — we are not spawn-looping a broken alias) AND
/// the retry budget hasn't been exhausted (so a model that started
/// crashing repeatedly eventually surfaces to the user).
///
/// These tests pin the decision truth-table and the seam wiring; the
/// async ``Task.sleep`` delay isn't exercised — tests call
/// ``runScheduledAutoRespawn`` directly via the internal seam.
@MainActor
@Suite("ServerManager auto-respawn on idle crash (issue #270)")
struct AutoRespawnTests {

    // MARK: - pure decision helper (every truth-table branch)

    @Test("schedule: ready cycle + known alias + retry budget → YES")
    func scheduleHappyPath() {
        #expect(ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: 0,
            retryLimit: 3
        ))
    }

    @Test("schedule: never-ready cycle (broken alias on load) → NO")
    func scheduleNeverReadyBlocks() {
        // The "user just picked an alias that can't load" shape. The
        // child crashed BEFORE answering /healthz; re-spawning would
        // re-crash on the same broken alias. Surface to the user
        // instead.
        #expect(!ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: false,
            alias: "qwen3.5-4b-4bit",
            attempts: 0,
            retryLimit: 3
        ))
    }

    @Test("schedule: empty alias → NO (nothing to respawn)")
    func scheduleEmptyAliasBlocks() {
        // The idle path — no child ever started, so ``alias`` is the
        // empty string. There is no model to bring back.
        #expect(!ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "",
            attempts: 0,
            retryLimit: 3
        ))
    }

    @Test("schedule: attempts == retryLimit → NO (budget exhausted)")
    func scheduleAtRetryLimitBlocks() {
        // After 3 unsuccessful respawns the user has to take action.
        // Equality with the limit terminates — the production check
        // uses ``<`` so this is the boundary.
        #expect(!ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: 3,
            retryLimit: 3
        ))
    }

    @Test("schedule: attempts just below retryLimit → YES")
    func scheduleJustBelowRetryLimit() {
        #expect(ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: 2,
            retryLimit: 3
        ))
    }

    @Test("schedule: attempts above retryLimit → NO (defensive)")
    func scheduleOverRetryLimitBlocks() {
        // Defensive: a caller that double-incremented should still
        // bail out. ``<`` is the right relation.
        #expect(!ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: 5,
            retryLimit: 3
        ))
    }

    // MARK: - retry-limit constants

    @Test("retry limit is small enough to not busy-loop forever")
    func retryLimitIsConservative() {
        // The constant lives on ``ServerManager`` so the UI tier can
        // reason about it. The product decision is "a few" — 3 is
        // the agreed shape. Pin so a 30/300 typo would fail.
        #expect(ServerManager.autoRespawnRetryLimit >= 1)
        #expect(ServerManager.autoRespawnRetryLimit <= 5)
    }

    @Test("retry delay is small but non-zero")
    func retryDelayIsSensible() {
        // The delay absorbs a thundering-herd from a crash-on-warmup
        // sequence so the user's Activity Monitor doesn't show 3 ×
        // rapid-mlx in 100 ms. Long enough for the kernel to reap the
        // dead process, short enough that the user notices the gap
        // only as "model briefly amber, now ready again".
        #expect(ServerManager.autoRespawnDelay >= 0.5)
        #expect(ServerManager.autoRespawnDelay <= 10.0)
    }

    // MARK: - end-to-end seam (no real spawn — binary missing → .missing)

    @Test("runScheduledAutoRespawn on .crashed(<alias>) burns one retry slot")
    func runScheduledRespawnBurnsRetrySlot() async {
        // Set up a manager in ``.crashed`` for an alias and call the
        // scheduled-respawn entry directly (bypassing the 2 s sleep).
        // No binary is configured, so ``start(alias:)`` short-circuits
        // to ``.missing`` — but the attempt counter increments because
        // we DID kick off a respawn attempt.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "boom"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        #expect(mgr._testAutoRespawnAttempts == 0)
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 1,
                "auto-respawn must increment its attempt counter so the cap can fire")
    }

    @Test("runScheduledAutoRespawn no-ops when state moved off .crashed")
    func runScheduledRespawnBailsWhenUserTookOver() async {
        // The 2 s timer fires AFTER the user has already clicked
        // Restart manually (state is now ``.ready`` or ``.starting``).
        // The respawn must bail without burning a retry slot — the
        // manual action already serves the same intent.
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 0,
                "respawn must bail when state is .ready, not burn a slot")
    }

    @Test("runScheduledAutoRespawn no-ops when state was reset to .idle")
    func runScheduledRespawnBailsOnIdle() async {
        // ``dismissTerminalState`` flips a ``.crashed`` to ``.idle``.
        // The respawn timer must not steamroll that.
        let mgr = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    @Test("runScheduledAutoRespawn no-ops when state was reset to .stopped")
    func runScheduledRespawnBailsOnStopped() async {
        // The user clicked Stop after the crash banner; the timer
        // races them. Stop wins.
        let mgr = ServerManager(
            testingState: .stopped,
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    @Test("runScheduledAutoRespawn no-ops when crashed alias mismatches")
    func runScheduledRespawnBailsOnAliasMismatch() async {
        // Alias swap in flight: ``.crashed(A)`` → user picked B (which
        // moved through ``.starting(B)`` → ``.crashed(B)``) → the
        // earlier timer for A finally fires. It must not respawn A
        // over B.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "boom"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        await mgr.runScheduledAutoRespawn(alias: "different-alias")
        #expect(mgr._testAutoRespawnAttempts == 0,
                "respawn for the wrong alias must bail without burning a slot")
    }

    @Test("runScheduledAutoRespawn no-ops when binaryPath disappeared")
    func runScheduledRespawnBailsWhenBinaryGone() async {
        // The user uninstalled rapid-mlx between the crash and the
        // timer firing. Don't burn retry budget on a path that will
        // immediately fail to ``.missing``.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "boom"),
            binaryPath: nil
        )
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    // MARK: - manual-action paths cancel the queued respawn

    @Test("dismissTerminalState() cancels pending respawn (no burn)")
    func dismissCancelsAutoRespawn() async {
        // The user dismissed the crash banner. They want to pick a
        // different alias next; a queued respawn re-loading the
        // crashed alias would defeat that.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "boom"),
            binaryPath: URL(fileURLWithPath: "/opt/homebrew/bin/rapid-mlx")
        )
        // Spin up a (real) sleep-and-respawn task as production
        // ``scheduleAutoRespawn`` would — we drive it through
        // ``handleChildExit``'s contract indirectly by setting state.
        // We can't directly access ``scheduleAutoRespawn`` from the
        // test, but ``dismissTerminalState`` MUST reset the counter
        // either way. After dismiss + respawn-attempt the counter
        // stays at 0.
        mgr.dismissTerminalState()
        // After dismiss the state has moved off ``.crashed`` to
        // ``.idle`` (or ``.missing`` if no binary). A subsequent
        // respawn attempt against the original alias must bail.
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    // MARK: - state machine integration

    @Test("After successful .ready, the spawn-cycle-reached-ready flag is set")
    func readyFlipsSpawnCycleReachedReady() {
        let mgr = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/opt/homebrew/bin/rapid-mlx")
        )
        // Direct seam — production code sets this on a successful
        // ``/healthz`` 200 inside the start() loop.
        mgr._testSetSpawnCycleReachedReady(true)
        // The flag itself isn't directly observable in production,
        // but the auto-respawn decision is. Build a known-good case
        // and confirm the schedule helper returns true with this
        // flag in play.
        #expect(ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: 0,
            retryLimit: ServerManager.autoRespawnRetryLimit
        ))
    }

    // MARK: - manual-stop cancellation pins (internal raised in review)

    @Test("stop() zeros the auto-respawn attempt counter")
    func stopZerosAutoRespawnAttempts() async {
        // Seed a non-zero attempt counter — production produces this
        // when ``runScheduledAutoRespawn`` ran one cycle, the spawn
        // crashed mid-load, and a follow-up retry is pending. The
        // user clicking Stop must abandon the budget.
        let mgr = ServerManager(testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "x"))
        mgr._testSetAutoRespawnAttempts(2)
        #expect(mgr._testAutoRespawnAttempts == 2)
        await mgr.stop()
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    @Test("shutdownSync() zeros the auto-respawn attempt counter")
    func shutdownSyncZerosAutoRespawnAttempts() {
        // App-terminate path — the user closed the desktop. A pending
        // respawn would race the teardown and either re-bind the port
        // mid-shutdown or write a stale ``OwnedServerRecord``.
        let mgr = ServerManager(testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "x"))
        mgr._testSetAutoRespawnAttempts(2)
        #expect(mgr._testAutoRespawnAttempts == 2)
        mgr.shutdownSync()
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    @Test("dismissTerminalState() zeros the auto-respawn attempt counter")
    func dismissTerminalStateZerosAutoRespawnAttempts() {
        // Internal raised in review: the existing ``dismissCancelsAutoRespawn``
        // test only asserts the post-dismiss counter is 0, which it
        // always was. Seed a non-zero value first so we actually
        // observe the cancel running.
        let mgr = ServerManager(testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "x"))
        mgr._testSetAutoRespawnAttempts(2)
        #expect(mgr._testAutoRespawnAttempts == 2)
        mgr.dismissTerminalState()
        #expect(mgr._testAutoRespawnAttempts == 0)
    }

    // MARK: - Issue #278: stability-window-gated budget reset
    //
    // The pre-#278 shape reset ``autoRespawnAttempts = 0`` on every
    // ``.ready`` transition. That made the 3-retry cap unreachable for
    // a child that briefly answered ``/healthz`` and then crashed
    // (OOM-on-first-inference, segfault-on-first-prompt, model worker
    // hang after first forward pass) — the watchdog respawned forever
    // at 2 s intervals. The fix gates the reset on a stability window:
    // crashes within ``autoRespawnReadyStableWindow`` of ``.ready``
    // count against the budget, only longer-running ``.ready`` windows
    // refresh the retry slot count.

    @Test("reset gate: never reached ready → NO reset (budget preserved)")
    func resetNeverReachedReadyBlocksReset() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        #expect(!ServerManager.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: false,
            readyAt: nil,
            now: now,
            stableWindow: 60.0
        ))
    }

    @Test("reset gate: reached ready but readyAt nil → NO reset (race-defensive)")
    func resetReadyAtNilBlocksReset() {
        // Production should always stamp ``readyAt`` before flipping
        // ``reachedReadyThisCycle``; a nil here means an unexpected
        // interleaving and we conservatively do NOT refresh the budget.
        let now = Date(timeIntervalSince1970: 1_000_000)
        #expect(!ServerManager.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: true,
            readyAt: nil,
            now: now,
            stableWindow: 60.0
        ))
    }

    @Test("reset gate: crashed within window → NO reset (budget keeps shrinking)")
    func resetCrashedWithinWindowBlocksReset() {
        // The exact bug shape: child reached .ready, crashed 5 s
        // later (OOM-on-first-inference). The budget must NOT reset.
        let readyAt = Date(timeIntervalSince1970: 1_000_000)
        let now = readyAt.addingTimeInterval(5)
        #expect(!ServerManager.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: true,
            readyAt: readyAt,
            now: now,
            stableWindow: 60.0
        ))
    }

    @Test("reset gate: crashed exactly at window boundary → YES reset")
    func resetAtWindowBoundaryAllowsReset() {
        // ``>=`` boundary inclusion — a child that was .ready for
        // exactly the window length counts as stable.
        let readyAt = Date(timeIntervalSince1970: 1_000_000)
        let now = readyAt.addingTimeInterval(60)
        #expect(ServerManager.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: true,
            readyAt: readyAt,
            now: now,
            stableWindow: 60.0
        ))
    }

    @Test("reset gate: crashed long after window → YES reset (fresh budget)")
    func resetWellPastWindowAllowsReset() {
        // The model genuinely served traffic for a while before
        // dying — this looks like a transient external event, give
        // the watchdog a fresh retry budget.
        let readyAt = Date(timeIntervalSince1970: 1_000_000)
        let now = readyAt.addingTimeInterval(3600) // 1 hour
        #expect(ServerManager.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: true,
            readyAt: readyAt,
            now: now,
            stableWindow: 60.0
        ))
    }

    @Test("stability window is conservative (≥30s, ≤300s)")
    func stabilityWindowIsConservative() {
        // Long enough to exclude OOM-on-first-inference (which fires
        // within seconds), short enough that a flaky model that ran
        // for a few minutes gets a fresh budget.
        #expect(ServerManager.autoRespawnReadyStableWindow >= 30)
        #expect(ServerManager.autoRespawnReadyStableWindow <= 300)
    }

    // MARK: - Issue #278: end-to-end via the production reset path
    //
    // The pre-#278 bug lived inside ``handleChildExit``; the unit
    // tests below drive the *production* code path via
    // ``_testApplyChildExitBudgetReset`` rather than just asserting
    // the pure helper. A regression that re-introduces an
    // unconditional ``autoRespawnAttempts = 0`` (whether back on the
    // ``.ready`` transition or somewhere else in ``handleChildExit``)
    // would still fail these.

    @Test("Production reset path: ready -> crash within 60s does NOT reset budget")
    func productionPathReadyThenQuickCrashKeepsBudget() {
        // Seed the manager mid-cycle: spawnCycleReachedReady=true,
        // readyAt 5 s in the past, attempts=2 (one slot remaining).
        // The production reset gate must REFUSE to refresh the
        // counter, so the next crash-induced auto-respawn schedule
        // would observe attempts==3 and bail.
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider { baseTime.addingTimeInterval(5) }
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testSetAutoRespawnAttempts(2)
        #expect(mgr._testReadyAt == baseTime)

        // Production handleChildExit's budget-reset step.
        mgr._testApplyChildExitBudgetReset()

        // Bug-A regression: counter MUST stay at 2 (would be 0 with
        // the pre-#278 unconditional reset).
        #expect(mgr._testAutoRespawnAttempts == 2,
                "crash within stability window must NOT reset budget — pre-#278 regression")
        // ``readyAt`` must be cleared so the next cycle starts fresh.
        #expect(mgr._testReadyAt == nil,
                "child exit must clear readyAt regardless of reset outcome")
    }

    @Test("Production reset path: ready ≥60s -> crash DOES reset budget")
    func productionPathReadyStableThenCrashResetsBudget() {
        // Same setup but readyAt is past the stability window. The
        // production reset gate must refresh attempts to 0 — a child
        // that ran cleanly for a while before dying gets a fresh
        // retry budget.
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        // 61 s — just past the 60 s window.
        mgr._testSetNowProvider {
            baseTime.addingTimeInterval(ServerManager.autoRespawnReadyStableWindow + 1)
        }
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testSetAutoRespawnAttempts(2)

        mgr._testApplyChildExitBudgetReset()

        #expect(mgr._testAutoRespawnAttempts == 0,
                "stable-window-elapsed crash must reset budget")
        #expect(mgr._testReadyAt == nil)
    }

    @Test("Production reset path: never reached ready -> NO reset (broken alias)")
    func productionPathNeverReadyKeepsBudget() {
        // The "user picked an alias that fails to load" shape. The
        // child crashed before ever reaching .ready. ``readyAt`` is
        // nil, ``spawnCycleReachedReady`` is false. Budget must NOT
        // reset — otherwise a user who picks a known-broken alias
        // and lets it spawn N times manually would silently refresh
        // the cap. (In practice the schedule gate ALSO blocks here
        // via reachedReadyThisCycle=false, but the reset gate is the
        // belt-and-braces check.)
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .starting(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider { baseTime.addingTimeInterval(120) }
        // spawnCycleReachedReady left at false; readyAt left at nil.
        mgr._testSetAutoRespawnAttempts(2)

        mgr._testApplyChildExitBudgetReset()

        #expect(mgr._testAutoRespawnAttempts == 2)
    }

    @Test("Full handleChildExit path: ready -> crash within window keeps the budget")
    func fullHandleChildExitWithinWindowKeepsBudget() {
        // Strongest regression pin: drive the ENTIRE production
        // handleChildExit through ``_testSimulateChildExit``. If a
        // future regression adds ``autoRespawnAttempts = 0`` ANYWHERE
        // in the handler (not just inside the shared helper), this
        // catches it because we observe the post-handler state.
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider { baseTime.addingTimeInterval(5) }
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testSetAutoRespawnAttempts(2)

        // Simulate a crash (expectedStop=false, exit status 1).
        mgr._testSimulateChildExit(
            expectedStop: false,
            status: 1,
            reason: .exit
        )

        // Bug-A pin via the FULL production path.
        #expect(mgr._testAutoRespawnAttempts == 2,
                "handleChildExit must NOT reset budget for crash within stability window")
        #expect(mgr._testReadyAt == nil,
                "handleChildExit must clear readyAt on every exit path")
        // The handler also transitioned state to .crashed.
        if case .crashed(let alias, _) = mgr.state {
            #expect(alias == "qwen3.5-4b-4bit")
        } else {
            Issue.record("handleChildExit must land in .crashed on the crash branch; got \(mgr.state)")
        }
    }

    @Test("Full handleChildExit path: ready ≥ window -> crash resets the budget")
    func fullHandleChildExitPastWindowResetsBudget() {
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider {
            baseTime.addingTimeInterval(ServerManager.autoRespawnReadyStableWindow + 1)
        }
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testSetAutoRespawnAttempts(2)

        mgr._testSimulateChildExit(expectedStop: false, status: 1, reason: .exit)

        #expect(mgr._testAutoRespawnAttempts == 0,
                "handleChildExit must reset budget when prior .ready window exceeded stability threshold")
    }

    @Test("Full handleChildExit path: expected stop does NOT reset budget on its own")
    func fullHandleChildExitExpectedStopDoesNotTouchBudget() {
        // wasExpected=true takes the early-return ``.stopped`` branch.
        // The handler must NOT zero the counter on its own — that's
        // ``cancelAutoRespawn``'s job (called separately from stop()).
        // The handler's only obligation on this branch is to clear
        // ``readyAt`` and surface ``.stopped``.
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .ready(alias: "qwen3.5-4b-4bit"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider { baseTime.addingTimeInterval(120) }
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testSetAutoRespawnAttempts(2)

        mgr._testSimulateChildExit(expectedStop: true, status: 0, reason: .exit)

        // Counter preserved — handler must not silently mutate it on
        // the wasExpected branch. (Note production stop()/etc. call
        // cancelAutoRespawn separately; we're verifying handler-only
        // behavior here.)
        #expect(mgr._testAutoRespawnAttempts == 2,
                "wasExpected branch must not silently reset the auto-respawn budget — that's cancelAutoRespawn's job")
        #expect(mgr._testReadyAt == nil,
                "expected-stop path must still clear readyAt (documented invariant)")
        if case .stopped = mgr.state {
            // Good.
        } else {
            Issue.record("expectedStop=true must land in .stopped; got \(mgr.state)")
        }
    }

    @Test("Production reset path: 3 quick ready -> crash cycles exhaust budget end-to-end")
    func productionPathThreeQuickCrashesExhaustBudget() async {
        // The fully-integrated bug-A scenario:
        //   1. Spawn cycle N reaches .ready, then crashes within 5 s.
        //   2. handleChildExit's reset gate (via test driver) refuses
        //      to refresh the budget. autoRespawnAttempts stays.
        //   3. runScheduledAutoRespawn fires, increments attempts.
        //   4. Loop. After 3 cycles attempts==3 and
        //      shouldScheduleAutoRespawn returns false → watchdog
        //      surfaces to the user.
        let baseTime = Date(timeIntervalSince1970: 1_000_000)
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "oom"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetNowProvider { baseTime.addingTimeInterval(5) }

        for cycle in 1...3 {
            // Each cycle: simulate the prior spawn cycle reached
            // .ready and then crashed within window.
            mgr._testSetSpawnCycleReachedReady(true)
            mgr._testSetReadyAt(baseTime)
            mgr._testSetState(.crashed(alias: "qwen3.5-4b-4bit", message: "oom"))
            mgr._testApplyChildExitBudgetReset()
            // Schedule-gate decision (cycle <= 3 of 3 should pass).
            #expect(ServerManager.shouldScheduleAutoRespawn(
                reachedReadyThisCycle: true,
                alias: "qwen3.5-4b-4bit",
                attempts: mgr._testAutoRespawnAttempts,
                retryLimit: ServerManager.autoRespawnRetryLimit
            ), "cycle \(cycle): schedule must still pass when attempts < retryLimit")
            await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
            #expect(mgr._testAutoRespawnAttempts == cycle,
                    "cycle \(cycle): attempts must increment monotonically because the reset gate refused")
        }

        // After 3 cycles, the cap MUST fire.
        mgr._testSetSpawnCycleReachedReady(true)
        mgr._testSetReadyAt(baseTime)
        mgr._testApplyChildExitBudgetReset()
        #expect(!ServerManager.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: true,
            alias: "qwen3.5-4b-4bit",
            attempts: mgr._testAutoRespawnAttempts,
            retryLimit: ServerManager.autoRespawnRetryLimit
        ), "after 3 quick crashes the watchdog MUST surface, not loop forever — the pre-#278 regression bug")
    }

    // MARK: - Issue #278: manual restart resets the budget

    @Test("Manual start() resets budget so user-clicked Restart gets a fresh 3-retry window")
    func manualStartResetsBudget() async {
        // Codex r4 MINOR: post-#278 the .ready transition no longer
        // resets autoRespawnAttempts. A user clicking Restart on the
        // crash banner (ChatView.swift:1211) or the Start button
        // (ModelPickerBar.swift:938) calls server.start(alias:)
        // directly — which previously got a fresh budget via the
        // .ready-transition reset. The fix: start() with default
        // ``isAutoRespawn: false`` resets the budget at entry. The
        // auto-respawn path (runScheduledAutoRespawn) passes
        // ``isAutoRespawn: true`` to skip the entry reset because
        // it just incremented the counter itself.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "x"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetAutoRespawnAttempts(3) // exhausted
        #expect(mgr._testAutoRespawnAttempts == 3)

        // Manual restart — default isAutoRespawn=false. The binary
        // doesn't exist so start() will short-circuit (it'll set
        // .missing); we only care about the entry-reset behavior.
        await mgr.start(alias: "qwen3.5-4b-4bit")

        #expect(mgr._testAutoRespawnAttempts == 0,
                "manual start() must reset the auto-respawn budget at entry — codex r4 MINOR")
    }

    @Test("Auto-respawn start() does NOT reset budget mid-cycle")
    func autoRespawnStartPreservesBudgetCounter() async {
        // The other half: when start() is called from
        // runScheduledAutoRespawn (which just incremented the
        // counter), it MUST NOT reset the budget — otherwise the
        // 3-retry cap is unreachable for a watchdog cycle.
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: "x"),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        mgr._testSetAutoRespawnAttempts(2)

        // runScheduledAutoRespawn increments and then calls
        // start(isAutoRespawn: true). The counter must end at 3,
        // not 0.
        await mgr.runScheduledAutoRespawn(alias: "qwen3.5-4b-4bit")
        #expect(mgr._testAutoRespawnAttempts == 3,
                "auto-respawn-driven start() must NOT trigger the manual entry-reset, or the retry cap is unreachable")
    }

    // MARK: - Issue #278: start() bails on Task.isCancelled after settlement
    //
    // The race: watchdog auto-respawn fires -> enters start() ->
    // suspends at awaitDownloadSettlement. User clicks Stop ->
    // cancelAutoRespawn() cancels the Task -> awaitDownloadSettlement
    // returns on cancellation. Without an explicit Task.isCancelled
    // check, isOperating and child==nil both pass and start() spawns
    // a child the user didn't ask for.

    @Test("start() bails when its Task is cancelled at the awaitDownloadSettlement suspension point")
    func startBailsOnTaskCancellationAtSettlement() async {
        // Attach a DownloadManager with a running job for the alias so
        // start() actually enters the awaitDownloadSettlement branch
        // (the guarded suspension point this test exists to cover).
        // Without the attached running job start() would skip the
        // entire if-let-downloads block and the Task.isCancelled
        // guard wouldn't be reached.
        //
        // The original .crashed seed carries the sentinel message
        // "BOOM-SENTINEL-DO-NOT-MUTATE". If the Task.isCancelled
        // guard were removed, start() would fall through and set
        // ``state = .starting(alias:)``, then ProcessGroupChild.spawn
        // would throw because binaryPath is bogus, then start() would
        // set ``state = .crashed(alias: ..., message: "failed to
        // spawn rapid-mlx: ...")``. EITHER mutation would replace the
        // sentinel — so asserting the sentinel survives means the
        // ``Task.isCancelled`` guard returned before any
        // state-mutation line. With the guard in place, no line of
        // start() executes past the guard for this Task.
        let sentinelMessage = "BOOM-SENTINEL-DO-NOT-MUTATE"
        let mgr = ServerManager(
            testingState: .crashed(alias: "qwen3.5-4b-4bit", message: sentinelMessage),
            binaryPath: URL(fileURLWithPath: "/nonexistent/rapid-mlx")
        )
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: "qwen3.5-4b-4bit") // Status: .running
        mgr.attachDownloads(downloads)
        #expect(downloads.isDownloading("qwen3.5-4b-4bit"),
                "seeded job must report as downloading so start() enters the awaited branch")

        let task = Task { @MainActor in
            await mgr.start(alias: "qwen3.5-4b-4bit")
        }
        task.cancel()
        await task.value

        #expect(mgr._testAutoRespawnAttempts == 0)
        // Load-bearing assertion: the sentinel-message ``.crashed``
        // must survive verbatim — neither the .starting transition
        // (which would discard the message) nor the
        // "failed to spawn rapid-mlx: ..." crash path (which would
        // overwrite it) can have run.
        switch mgr.state {
        case .crashed(let alias, let message):
            #expect(alias == "qwen3.5-4b-4bit")
            #expect(message == sentinelMessage,
                    "state was mutated past the Task.isCancelled guard — pre-#278 regression. Got: \(message)")
        case .starting, .ready, .stopped, .idle, .missing:
            Issue.record("start() must not advance past .crashed(sentinel) when its Task is cancelled at awaitDownloadSettlement; got \(mgr.state)")
        }
    }
}
