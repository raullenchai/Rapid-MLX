import Darwin
import Foundation
import Observation

/// Lifecycle phases of the embedded `rapid-mlx serve` child process.
///
/// Mirrors the `ServerState` enum in the v0.1 Tauri reference. The Swift
/// version flattens the binary-path payload onto the manager itself
/// because SwiftUI reads it directly; only state-discriminating data is
/// kept in the enum.
enum ServerState: Equatable {
    /// The `rapid-mlx` CLI was not found on the host at all.
    case missing
    /// CLI is present, no child process is running.
    case idle
    /// Child has been spawned, `/healthz` has not yet returned 200.
    case starting(alias: String)
    /// `/healthz` answered 200 — the chat surface (when it lands) can
    /// talk to the backend.
    case ready(alias: String)
    /// Child exited or crashed unexpectedly.
    case crashed(alias: String, message: String)
    /// User clicked Stop; child terminated by request.
    case stopped
}

/// Owns the embedded `rapid-mlx serve` subprocess. SwiftUI views read the
/// `@Observable` properties directly; mutating methods must be called on
/// the main actor (the `@MainActor` annotation on the class enforces
/// that at compile time).
///
/// Design notes:
///   - `@MainActor` because every state mutation feeds SwiftUI; pushing
///     them through `await MainActor.run { ... }` from a background
///     actor adds latency for no isolation benefit (Process callbacks
///     already hop threads, so we sink everything to main here).
///   - The bounded ring buffer of log lines is capped at
///     `logBufferCapacity` so a runaway server can't blow up memory
///     during long downloads.
///   - The start/stop critical section is serialized by `isOperating`;
///     two concurrent clicks cannot race because SwiftUI buttons run
///     on the main actor and we hold the actor across the entire
///     async start (which awaits `Task.sleep` between polls).
@MainActor
@Observable
final class ServerManager {
    // MARK: - Public state (read by SwiftUI)

    /// Current lifecycle phase. Drives all UI state controls.
    private(set) var state: ServerState

    /// Models currently held by the one sidecar process, plus total process
    /// memory against its configured ceiling.
    private(set) var residency: ModelResidencySnapshot = .empty

    @ObservationIgnored
    private var residencyClient = ServerResidencyClient()

    /// Set when ``start`` declines a load because the model's footprint
    /// on top of live memory use would risk exhausting RAM (#324). A
    /// top-level view binds a confirmation alert to this; the user
    /// either acknowledges the risk (``confirmPendingMemoryLoad``) or
    /// backs out (``cancelPendingMemoryLoad``). ``nil`` when no load is
    /// being held for confirmation.
    var pendingMemoryWarning: ModelSizing.MemoryWarning?

    /// How the user answered the most recent memory prompt. Read by
    /// ``awaitMemoryDecision`` so ``ensureServing`` can tell "user backed
    /// out" (an honest startup failure) from "user chose Load anyway"
    /// (wait for the re-entered launch instead of reporting failure).
    private var memoryLoadConfirmed: Bool = false

    /// The re-entered launch spawned by "Load anyway". ``ensureServing``
    /// awaits this instead of polling for a state change: ``start`` can
    /// legitimately sit in ``awaitDownloadSettlement`` (a background pull
    /// still fetching the model) with ``state`` still idle for far longer
    /// than any fixed bound, and timing out there would drop the user's
    /// message even though the model does come up.
    private var memoryConfirmTask: Task<Void, Never>?

    /// Confirmed launches still running, by sequence number. Polled by
    /// ``awaitConfirmedLaunch`` instead of awaiting the task's ``value``:
    /// awaiting a non-throwing Task is NOT cancellation-aware, so a caller
    /// that gets cancelled mid-wait (chat Stop, switching conversations)
    /// would stay suspended until a possibly-stalled download finished.
    ///
    /// Keyed per launch rather than a single flag: two confirmations can
    /// overlap (confirm A while its pull settles, then park and CANCEL B),
    /// and a shared flag would let B's cancel tell A's waiter that A had
    /// finished — dropping A's chat turn while A was still coming up.
    private var memoryConfirmRunning: Set<Int> = []

    /// Monotonic id for the most recent confirmed launch.
    private var memoryConfirmSeq = 0

    /// Alias the child is currently serving once `/healthz` answered 200,
    /// else `nil`. Authoritative source of truth for which model id any
    /// outgoing chat request should put in `model:` — picker bar state
    /// can lag the spawn cycle, so request shape derives from this.
    var servingAlias: String? {
        if case .ready(let alias) = state { return alias }
        return nil
    }

    func isModelResident(_ alias: String) -> Bool {
        guard case .ready = state else { return false }
        return residency.contains(alias) || servingAlias == alias
    }

    /// Feed the existing readiness resolver an alias-specific view of a
    /// process that may now hold several ready engines.
    func readinessState(for alias: String) -> ServerState {
        isModelResident(alias) ? .ready(alias: alias) : state
    }

    func refreshResidency() async {
        guard case .ready = state else {
            residency = .empty
            return
        }
        guard let snapshot = await residencyClient.fetch(
            port: activePort,
            bearer: activeBearer
        ) else { return }
        residency = snapshot
    }

    /// Alias of the child that currently owns the runtime — for BOTH
    /// ``.starting`` and ``.ready``. Unlike ``servingAlias`` (which is
    /// ``.ready``-only) this is true the moment a child is spawned, so a
    /// config change made *during* startup is correctly flagged as needing
    /// a restart: the child already received its one-time ``--mcp-config``
    /// at launch and won't see the new file until it is replaced.
    var launchedChildAlias: String? {
        switch state {
        case .starting(let alias), .ready(let alias): return alias
        default: return nil
        }
    }

    /// Absolute path to `rapid-mlx` if it was found at construction, else
    /// `nil`. Surfaced in the UI as a small caption when state is
    /// `.missing` so the user knows what we looked for.
    private(set) var binaryPath: URL?

    /// Provenance captured by the same decision that selected `binaryPath`.
    /// A resolved path alone is insufficient when two managed launchers point
    /// at the same target; About uses this to report the actual winning slot.
    private(set) var binaryResolution: ServerLocator.Resolution?

    /// Tail of stdout/stderr lines from the live child, oldest first.
    /// Bounded to `logBufferCapacity` entries.
    private(set) var logLines: [String] = []

    /// True while a start or stop is in flight. The UI disables both
    /// buttons during this window so a second click cannot race the
    /// first into spawning a duplicate child.
    private(set) var isOperating: Bool = false

    /// Live download / load progress derived from the child's stderr
    /// tqdm output. ``.idle`` until the first tqdm line lands; flips to
    /// ``.fetching`` / ``.downloading`` while HuggingFace pulls files,
    /// then to ``.loading`` once the engine starts compiling Metal
    /// shaders. Surfaced by ``ContentView`` as a top-bar pill so the
    /// user knows what's happening during a 5-30 GB cold start instead
    /// of staring at the log tail.
    let downloadProgress: DownloadProgress = DownloadProgress()

    /// Live handle to the HF cache-directory byte monitor for the
    /// current ``.starting`` cycle (if one is running). Stopped + cleared
    /// whenever ``downloadProgress.reset()`` fires and whenever the
    /// child exits, so the polling task never outlives the start cycle
    /// it was bound to. ``nil`` when no in-flight start, or when the
    /// caller didn't pass an ``hfPath`` (alias not in the catalog) /
    /// when the HF cache root couldn't be resolved.
    private var startupByteMonitor: HFCacheByteMonitor.Handle?

    /// v0.4.36: wall-clock of when the current ``.starting`` cycle
    /// began. Used by the UI to surface an elapsed-time clock during
    /// ``.idle`` / ``.fetching`` / ``.downloading`` / ``.loading``
    /// phases so the user has feedback that the app isn't hung even
    /// when the tqdm parser hasn't produced a structured signal yet
    /// (cached models skip the download phase entirely and tqdm
    /// sometimes outputs in a format the regex can't catch). Nil
    /// whenever the server isn't currently starting.
    private(set) var startedAt: Date?

    // MARK: - Tunables

    /// Fixed host/port pair. v0.1 had no reason to configure this and v0.2
    /// inherits that choice; the chat surface derives its default from
    /// ``PortSweep.defaultPort`` via ``ChatStreamClient.loopbackURL(port:)``
    /// and re-targets onto ``activePort`` at first request.
    private let host: String = "127.0.0.1"

    /// The port the most recent ``start()`` actually bound rapid-mlx
    /// to. Initialised to ``PortSweep.defaultPort`` (8000) so callers
    /// reading this before the first spawn don't have to special-case
    /// "no port yet". When ``PortAllocator`` falls back to 8001+ the
    /// value is republished so ChatViewModel re-targets the chat
    /// client URL.
    private(set) var activePort: Int = PortSweep.defaultPort

    /// Issue #17 desktop-half: per-launch bearer secret. Generated
    /// fresh by ``start()`` and handed to the child via the
    /// ``RAPID_MLX_API_KEY`` env (NOT argv); cleared by
    /// ``handleChildExit`` / ``terminateChild`` so a stale value
    /// can't survive into the next launch.
    ///
    /// Chat clients read this and add
    /// ``Authorization: Bearer <secret>`` to every request so an
    /// unrelated local process can't drive inference against our
    /// loopback-bound server. ``nil`` means "server not running" —
    /// or in the (rare) RNG-failure case "we refused to start
    /// without auth", which surfaces as ``.crashed`` to the user.
    private(set) var activeBearer: String?

    /// Health-check budget — interpreted as a **stall window** since
    /// v0.7.13, not a wall-clock-from-launch cap. The deadline slides
    /// forward every time ``downloadProgress`` reports forward motion
    /// (a heartbeat tick, a per-file completion, a phase transition).
    /// The loop only terminates if no progress AND no successful
    /// ``/healthz`` is observed for the full window.
    ///
    /// Why the change: the old shape — a fixed 30 min deadline from
    /// launch — silently killed multi-hour downloads on slow links.
    /// A 10 GB model at 683 KB/s takes ~4 hours; the deadline fired
    /// at 30 min, ``terminateChild`` SIGKILL'd the child mid-pull,
    /// the partial download was orphaned, and the user's next attempt
    /// started from zero. Reported in the wild during v0.7.12
    /// dogfooding.
    ///
    /// Why a sliding window vs. an outright removal of the deadline:
    /// genuinely-wedged children (rapid-mlx hangs on a dead Python
    /// thread, network drops mid-download with no recovery) need to
    /// surface as ``.crashed`` instead of the UI sitting forever on
    /// "Downloading model files". 30 min of NO progress is a
    /// reasonable enough threshold that real silent-wedge regressions
    /// still register, and it leaves a comfortable margin past the
    /// longest plausible warmup (Metal-shader compile on a giant MoE
    /// is typically < 5 min).
    private let healthStallWindow: TimeInterval = 30 * 60

    /// Interval between `/healthz` probes once the child is up.
    private let healthPollInterval: TimeInterval = 0.5

    /// v0.6 audit P1 (ServerManager — silent-crash detection):
    /// once the child has reported ready, continue polling /healthz
    /// at a slower cadence so a silent rapid-mlx crash (Python OOM,
    /// model-load deadlock, segfault in the inference worker)
    /// surfaces in the UI within seconds of going dark instead of
    /// only on the user's next chat send. 30 s is the eyeball
    /// budget between "everything looks fine" and "I notice the
    /// status pill went amber" — short enough to feel responsive,
    /// long enough that VPN flaps / Mac sleep cycles don't trip it.
    private let runtimeHealthInterval: TimeInterval = 30.0

    /// Number of consecutive runtime probes that must fail before
    /// we transition the state to ``.crashed``. Three at 30 s = a
    /// ~90 s grace window, which absorbs the most common false
    /// positives (VPN reconnect, brief network blip, large batch
    /// inference hogging the event loop) without making the user
    /// wait minutes to notice a real crash.
    private let runtimeHealthFailureThreshold: Int = 3

    /// Wall-clock budget given to ``rapid-mlx`` between SIGTERM and
    /// SIGKILL during ``terminateChild`` (alias-switch, user Stop,
    /// runtime-health timeout). Was 5 s; the v0.7.6 bump to 30 s
    /// follows from a real-user trace:
    ///
    /// rapid-mlx's FastAPI lifespan ``shutdown`` hook serialises
    /// the in-memory prefix-cache to disk one safetensors file per
    /// KV-cache entry — each entry is 200–260 MB on a 27 B/4-bit
    /// model and the writer holds the asyncio event loop while it
    /// does so. With the 5 s grace, an 18-entry / ~4.4 GB flush
    /// got SIGKILL'd ~mid-write, leaving ``prefix_cache/<rev>.new/``
    /// with a partial set of files. The atomic
    /// ``.new/`` → final rename never happened, so the next launch
    /// re-prefills the whole prompt instead of replaying the cache.
    /// 30 s covers the steady-state flush on the largest aliases we
    /// ship; the upstream rapid-mlx fix (background-persist /
    /// interruptible flush) lands separately and will bring the
    /// flush time back to near-zero, but we want correct behaviour
    /// against today's released sidecar too.
    ///
    /// Trade-off: a genuinely wedged sidecar now takes up to 30 s to
    /// SIGKILL after the user clicks Stop. That's still bounded and
    /// the UI immediately reflects ``.stopped`` once
    /// ``terminateChild`` returns; the user just doesn't see a fresh
    /// alias load until SIGKILL fires.
    ///
    /// ``shutdownSync`` (Cmd-Q / applicationWillTerminate) keeps the
    /// 5 s grace intentionally — AppKit's terminate handler runs on
    /// the main thread with a finite budget before macOS force-kills
    /// the host app, so we can't safely block it for 30 s. Cmd-Q
    /// mid-flush will still truncate; the upstream rapid-mlx fix is
    /// the proper resolution there.
    ///
    /// Exposed as ``internal`` so the test suite can pin the value
    /// against accidental regressions — see ``SigtermGracePeriodTests``.
    internal let sigtermGracePeriod: TimeInterval = 30.0

    /// Cap on the in-memory log tail. The UI displays the last ~10 lines,
    /// but we keep more so a future "copy logs" affordance has enough
    /// context.
    private let logBufferCapacity: Int = 200

    /// Issue #270 (idle-state silent crash): when the embedded child
    /// exits unexpectedly AFTER it had been ``.ready`` for the current
    /// alias, attempt to bring it back automatically so the next
    /// window-open / Dock-click finds a warm server. Capped so a
    /// genuinely-broken alias (model file corrupted, OOM on this
    /// machine, parser bug crashing on warmup) doesn't busy-loop
    /// spawning-and-crashing for the rest of the app's lifetime.
    ///
    /// Three retries with a fixed 2 s gap. Matches the conservative
    /// upper bound on a clean ``rapid-mlx serve`` cold-start without
    /// model download (~1.5 s from spawn to ``/healthz`` 200 on the
    /// bundled bonsai-1.7b-2bit). On a download-required cold-start the
    /// retry is harmless because the byte monitor sees real progress
    /// and the stall window doesn't fire.
    nonisolated internal static let autoRespawnRetryLimit: Int = 3
    nonisolated internal static let autoRespawnDelay: TimeInterval = 2.0

    /// Tracks the number of consecutive auto-respawn attempts for the
    /// current alias. Issue #278: reset to 0 inside ``handleChildExit``
    /// ONLY when the prior ``.ready`` window stayed up for at least
    /// ``autoRespawnReadyStableWindow`` — crashes within that window
    /// count against the cap so a pathological "ready -> crash" loop
    /// terminates. Manual user actions (``stop()``, ``shutdownSync()``,
    /// ``dismissTerminalState()``) also zero the counter unconditionally
    /// because the user has taken over the lifecycle and a fresh
    /// click-driven Start gets a fresh budget. A fresh ``start(alias:)``
    /// does NOT cancel the queued respawn directly —
    /// ``runScheduledAutoRespawn``'s state-recheck observes the new
    /// ``.starting``/``.ready`` state and bails without burning a retry
    /// slot, so the indirection is harmless.
    @ObservationIgnored
    private var autoRespawnAttempts: Int = 0

    /// In-flight auto-respawn task, if any. Cancelled by every code
    /// path that takes manual control of the lifecycle (``stop()``,
    /// ``shutdownSync()``, ``dismissTerminalState()``) so a queued
    /// respawn can't race the user's intent. A fresh ``start(alias:)``
    /// does not cancel here — see ``autoRespawnAttempts`` for why.
    @ObservationIgnored
    private var autoRespawnTask: Task<Void, Never>?

    // MARK: - Private process bookkeeping

    /// The running process-group leader, if any. `rapid-mlx serve`
    /// forks Python workers; signalling only the parent can orphan
    /// descendants that keep the port / GPU memory alive. We spawn the
    /// child into its own process group and always signal `-pgid`.
    private var child: ProcessGroupChild?

    /// True while we are deliberately stopping the child. The
    /// `terminationHandler` checks this to decide whether to surface the
    /// exit as `.stopped` (expected) or `.crashed` (unexpected).
    /// Latches once the terminal SIGTERM has been delivered so a second
    /// ``beginShutdown`` (the reap phase calls it again) cannot re-signal a
    /// child that is already shutting down gracefully. Reset when a new
    /// child is spawned so a restart is not permanently un-signallable.
    private var didSignalShutdown = false

    private var expectedStop: Bool = false

    /// Issue #270: the current spawn cycle observed at least one
    /// ``.ready`` transition. Drives the auto-respawn decision in
    /// ``handleChildExit`` — a child that crashed BEFORE it ever
    /// answered ``/healthz`` 200 likely has a broken alias / missing
    /// model / OOM-on-load and respawning would just busy-loop. Only
    /// auto-respawn cases where the model was demonstrably healthy
    /// before going dark.
    ///
    /// Reset to ``false`` at the top of every ``start()``, flipped to
    /// ``true`` the moment the start loop transitions to ``.ready``.
    private var spawnCycleReachedReady: Bool = false

    /// Issue #278: wall-clock moment the current spawn cycle transitioned
    /// to ``.ready``. ``handleChildExit`` consults this to decide whether
    /// the prior ``.ready`` window was long enough to be considered
    /// "demonstrably stable" — only stable readys reset
    /// ``autoRespawnAttempts`` to 0. A child that briefly reached
    /// ``.ready`` and then crashed (OOM-on-first-inference, segfault on
    /// a particular prompt) within ``autoRespawnReadyStableWindow``
    /// counts the crash against the retry budget so a pathological
    /// "ready -> crash -> respawn" loop terminates.
    ///
    /// Cleared at the top of every ``start()`` and on every child exit
    /// (so a never-ready cycle never seeds a stale timestamp).
    private var readyAt: Date?

    /// Issue #278: deterministic ``Date`` seam so unit tests can pin the
    /// "crashed within window of .ready" decision without sleeping for
    /// 60 s of wall time. Production wires ``Date.init`` so the manager
    /// reads real wall clock; ``_testSetNowProvider`` swaps in a
    /// controllable clock for tests. ``@ObservationIgnored`` keeps the
    /// closure off SwiftUI's tracking tree.
    @ObservationIgnored
    private var nowProvider: () -> Date = Date.init

    /// Issue #278: minimum time the child must stay ``.ready`` before a
    /// subsequent crash is considered "transient enough that the model
    /// is fine, the world wobbled" and the auto-respawn budget is
    /// allowed to reset. Crashes within this window count against the
    /// retry budget so a ready -> crash loop eventually terminates.
    ///
    /// 60 s is a deliberately conservative middle ground: long enough
    /// to exclude OOM-on-first-inference / segfault-on-first-prompt
    /// (both fire within seconds of the user sending the first chat
    /// after ``.ready``), short enough that a child that genuinely
    /// served traffic for a few minutes before dying gets a fresh
    /// retry budget for the new crash.
    nonisolated internal static let autoRespawnReadyStableWindow: TimeInterval = 60.0

    /// v0.6 audit P1 (silent-crash detection): runtime /healthz
    /// monitor that runs while ``state == .ready`` and flips the
    /// state to ``.crashed`` after ``runtimeHealthFailureThreshold``
    /// consecutive probe failures. Owned by ``startRuntimeHealthMonitor``
    /// / ``cancelRuntimeHealthMonitor`` so the loop is bounded to
    /// exactly one launch's worth of ``.ready`` time.
    @ObservationIgnored
    private var runtimeHealthTask: Task<Void, Never>?

    /// codex r2 BLOCKING: while ``terminateChild`` is running, the
    /// SIGTERM it just sent can cause the child's
    /// ``terminationHandler`` to fire → ``handleChildExit`` → which
    /// unconditionally cancelled ``runtimeHealthTask`` → which
    /// happens to be the task currently executing ``terminateChild``
    /// → its remaining ``Task.sleep`` grace windows immediately throw
    /// ``CancellationError``, collapsing the SIGTERM/SIGKILL grace
    /// loops to zero. This flag gates that indirect path: while it
    /// is true, ``handleChildExit`` skips the runtime-monitor cancel
    /// because ``terminateChild`` already owns the lifecycle of
    /// both the child and the monitor for this teardown.
    private var isInsideTerminateChild: Bool = false

    /// Pipes whose readability handlers stay live until the child exits.
    /// We retain them so ARC doesn't tear down the handler closures
    /// mid-stream.
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    /// Reused ephemeral session for the ``/healthz`` poll loop. The
    /// previous shape constructed (and immediately invalidated) a
    /// fresh ``URLSession`` per probe, which churns an ICU/CFNetwork
    /// thread every 500 ms for the entire 30-minute health budget.
    /// One ephemeral session has no on-disk cache, no cookie store,
    /// and short-lived URLProtocol stacks — safe to keep across the
    /// loop. ``@ObservationIgnored`` keeps the ``@Observable`` macro
    /// from synthesizing a tracking accessor for it (URLSession is
    /// not Equatable / no SwiftUI surface reads it).
    /// [codex audit r1 ServerManager.swift:577]
    @ObservationIgnored
    private let healthSession: URLSession = {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 1.5
        cfg.timeoutIntervalForResource = 1.5
        return URLSession(configuration: cfg)
    }()

    /// Optional back-reference to the app's ``DownloadManager``. Wired
    /// from ``RapidApp.init`` after both singletons are constructed.
    /// ``start(alias:)`` consults this to stagger the serve spawn behind
    /// any in-flight background pull for the same alias — without the
    /// stagger, ``rapid-mlx pull`` and ``rapid-mlx serve`` would both
    /// write the same HF shards (pull via R2, serve via HF-fallback
    /// because the snapshot dir is already occupied), doubling disk +
    /// bandwidth and leaving an orphan blob behind. See rapid-desktop
    /// issue #253. ``nil`` in headless test harnesses that build a
    /// ``ServerManager`` directly without an app shell — the start path
    /// simply skips the stagger in that case.
    @ObservationIgnored
    private weak var downloads: DownloadManager?

    // MARK: - Construction

    init() {
        let resolution = ServerLocator.locate()
        self.binaryResolution = resolution
        self.binaryPath = resolution?.binary
        self.state = (resolution == nil) ? .missing : .idle
    }

    /// Wire the app's ``DownloadManager`` so ``start(alias:)`` can
    /// stagger behind any in-flight background pull for the same
    /// alias. Called once from ``RapidApp.init`` after both singletons
    /// land. Held weakly to avoid pinning ``DownloadManager`` past app
    /// shutdown if the harness ever tears it down independently.
    func attachDownloads(_ downloads: DownloadManager) {
        self.downloads = downloads
    }

    /// Internal test seam — lets ``RapidTests`` drive the view tree
    /// through every ``ServerState`` branch without spawning a real
    /// child or relying on whatever ``rapid-mlx`` happens to be on
    /// disk. Not part of any production code path; the underscore
    /// prefix mirrors the Swift Standard Library convention for
    /// "API kept around for testing only."
    internal init(
        testingState: ServerState,
        binaryPath: URL? = nil,
        residency: ModelResidencySnapshot = .empty
    ) {
        self.state = testingState
        self.binaryPath = binaryPath
        self.residency = residency
        self.binaryResolution = binaryPath.map {
            ServerLocator.Resolution(binary: $0, source: .unknown, version: nil)
        }
    }

    /// codex r1 BLOCKING #3 test seam — install a stub
    /// ``ProcessGroupChild`` so ``runRuntimeHealthLoop``'s identity
    /// guard (``self.child === process``) sees the same instance
    /// the test passes in as ``process``. Production code goes
    /// through ``start()`` which installs the real spawned child.
    internal func _testInstallChild(_ process: ProcessGroupChild) {
        self.child = process
    }

    /// codex r1 BLOCKING #3 test seam — clear the installed stub
    /// child. Mirrors what production ``handleChildExit`` does on
    /// process death.
    internal func _testClearChild() {
        self.child = nil
    }

    /// codex r1 BLOCKING #3 test seam — drive the ``state`` field
    /// directly. Lets the runtime-monitor tests simulate a manual
    /// stop landing mid-loop to pin the state-drift guard.
    internal func _testSetState(_ newState: ServerState) {
        self.state = newState
    }

    /// Issue #270 test seam — observe the current auto-respawn attempt
    /// counter so the auto-respawn-retry-cap test can assert how many
    /// times ``runScheduledAutoRespawn`` actually incremented it.
    internal var _testAutoRespawnAttempts: Int { autoRespawnAttempts }

    /// Issue #270 test seam — drive the spawn-cycle-reached-ready flag
    /// directly. ``runRuntimeHealthLoop`` / the start() polling loop
    /// flip this on a successful ``.ready`` transition; tests that
    /// simulate "we were healthy, then crashed" need to set it.
    internal func _testSetSpawnCycleReachedReady(_ value: Bool) {
        self.spawnCycleReachedReady = value
    }

    /// Issue #270 test seam — seed the auto-respawn attempt counter so
    /// cancellation tests can verify the counter went from non-zero to
    /// zero after a manual stop / shutdown / dismiss path. Production
    /// callers must not touch this; the counter is owned by
    /// ``runScheduledAutoRespawn`` and ``cancelAutoRespawn``.
    internal func _testSetAutoRespawnAttempts(_ value: Int) {
        self.autoRespawnAttempts = value
    }

    /// Issue #278 test seam — swap in a controllable ``() -> Date`` so
    /// the "crashed within stability window of .ready" decision can be
    /// pinned deterministically. Production code uses ``Date.init``.
    internal func _testSetNowProvider(_ provider: @escaping () -> Date) {
        self.nowProvider = provider
    }

    /// Issue #278 test seam — drive ``readyAt`` directly to simulate
    /// "the spawn cycle reached .ready at time T". Production code
    /// sets this inside the start() polling loop on a successful
    /// ``/healthz`` 200 transition.
    internal func _testSetReadyAt(_ value: Date?) {
        self.readyAt = value
    }

    /// Issue #278 test seam — observe the current ``readyAt``
    /// timestamp so the ready-window-reset tests can assert the field
    /// was/wasn't set across a spawn cycle.
    internal var _testReadyAt: Date? { readyAt }

    /// Issue #278 test seam — drive the budget-reset half of
    /// ``handleChildExit`` against the current ``readyAt`` /
    /// ``spawnCycleReachedReady`` / ``nowProvider`` configuration
    /// WITHOUT requiring a real spawn + child-exit. Delegates to the
    /// SAME private ``applyChildExitBudgetReset`` that production
    /// ``handleChildExit`` calls, so the test path cannot drift from
    /// the real behavior — any future regression that re-introduces
    /// an unconditional reset somewhere in the real handler still
    /// gets caught by these tests via the shared helper.
    internal func _testApplyChildExitBudgetReset() {
        applyChildExitBudgetReset(reachedReadyThisCycle: spawnCycleReachedReady)
    }

    /// Issue #278 test seam — drive the FULL production
    /// ``handleChildExit`` path with a stub ``ProcessGroupChild``.
    /// Lets unit tests exercise every line of the handler (not just
    /// the shared budget-reset helper) so a regression that
    /// introduces an extraneous ``autoRespawnAttempts = 0`` ANYWHERE
    /// in the handler still gets caught. Production callers must not
    /// touch this; ``terminationHandler`` is the only legitimate
    /// caller and it owns the real ``ProcessGroupChild``.
    ///
    /// ``expectedStop`` mirrors the real flag the handler reads;
    /// callers pass ``true`` to simulate a user-driven stop and
    /// ``false`` to simulate a crash. The status / reason pair is
    /// surfaced into the ``.crashed`` message exactly as production
    /// would.
    internal func _testSimulateChildExit(
        expectedStop: Bool,
        status: Int32,
        reason: Process.TerminationReason
    ) {
        let stubChild = ProcessGroupChild.testStub()
        self.child = stubChild
        self.expectedStop = expectedStop
        handleChildExit(process: stubChild, status: status, reason: reason)
    }

    // MARK: - Persisted "last served" alias (v0.5.3 auto-restart)

    /// UserDefaults key holding the alias of the model the user most
    /// recently asked us to serve. Written on every transition into
    /// ``.ready(alias:)`` and cleared on user-initiated ``stop()``.
    /// ``RapidApp`` reads this on launch to decide whether to
    /// auto-resume the previous session's model (LM Studio shape:
    /// the loaded model survives an app restart).
    nonisolated fileprivate static let lastServedAliasKey = "rapid.serve.lastAlias"

    /// Currently persisted last-served alias, if any. ``nil`` after a
    /// user-initiated Stop or a fresh install. Exposed as a static
    /// method so ``RapidApp.init`` can probe it before constructing
    /// the manager — the auto-restart decision happens on the main
    /// scene's ``.task``, not inside the init.
    nonisolated static func lastServedAlias() -> String? {
        guard let raw = UserDefaults.standard.string(forKey: lastServedAliasKey) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Bring the embedded child to ``.ready(alias)`` regardless of
    /// where the state machine currently sits.
    ///
    /// Idempotent in the happy case: if we're already serving the
    /// requested alias, return ``true`` immediately — useful for the
    /// regenerate-with-different-alias chevron, which would otherwise
    /// race the picker / restart confirm flow and fire its request
    /// against the OLD resident model (see PR for the original bug).
    ///
    /// Returns ``true`` if state lands in ``.ready(alias)``,
    /// ``false`` on any failure terminal state. Callers should
    /// surface a UI error on false; we don't throw because the
    /// SwiftUI call sites are easier to express as `let ok = await …`.
    /// - Parameter hfPath: the alias's Hugging Face repo when the
    ///   caller knows it. Forwarded to ``start`` so a cold pull
    ///   installs the bytes-on-disk progress monitor; without it the
    ///   user watches a featureless spinner for the whole download.
    func ensureServing(alias: String, hfPath: String? = nil) async -> Bool {
        await ensureServing(
            alias: alias,
            hfPath: hfPath,
            estimatedMemoryGB: nil,
            replacementGroup: nil
        )
    }

    func ensureServing(
        alias: String,
        hfPath: String?,
        estimatedMemoryGB: Double?,
        replacementGroup: ResidentModelReplacementGroup? = nil
    ) async -> Bool {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        // Replacement policy matters only when loading a different model.
        // The requested alias is already the active assistant, so asking the
        // residency endpoint to replace it is redundant and breaks legacy
        // sidecars: their 404 fallback stops and restarts the model on every
        // chat send before the request can leave the app.
        if case .ready(let current) = state, current == trimmed {
            return true
        }
        if replacementGroup == nil, isModelResident(trimmed) { return true }

        // A healthy sidecar can admit another engine without replacing the
        // process. Only a 404/405 from an older bundled server falls back to
        // the legacy stop/start path; capacity and load failures stay failures
        // so we never hide a rejected ceiling by unloading the primary model.
        if case .ready = state, child != nil {
            let estimate = estimatedMemoryGB ?? ModelSizing.estimate(alias: trimmed).totalGB
            let result = await residencyClient.load(
                alias: trimmed,
                hfPath: hfPath,
                estimatedSizeGB: estimate,
                replaceGroup: replacementGroup,
                port: activePort,
                bearer: activeBearer
            )
            switch result {
            case .loaded(let status):
                if !residency.contains(status.id) {
                    residency = ModelResidencySnapshot(
                        memoryLimitBytes: residency.memoryLimitBytes,
                        memoryUsedBytes: residency.memoryUsedBytes,
                        memoryAvailableBytes: residency.memoryAvailableBytes,
                        idleTTLSeconds: residency.idleTTLSeconds,
                        loadsTotal: residency.loadsTotal + 1,
                        evictionsTotal: residency.evictionsTotal,
                        models: residency.models + [status]
                    )
                }
                await refreshResidency()
                if replacementGroup != nil {
                    state = .ready(alias: trimmed)
                }
                return true
            case .unsupported:
                break
            case .rejected(let message):
                appendLogLines(["Resident model load declined: \(message)"])
                return false
            }
        }
        // Someone else — the picker's Start CTA, auto-start on launch,
        // Quickstart — may already be bringing up the very alias we
        // want. Tearing that launch down to restart it from zero would
        // discard whatever progress it has made, and on a cold pull
        // would re-enter a multi-gigabyte download. Wait for it.
        if case .starting(let current) = state, current == trimmed {
            await awaitStartupSettled(alias: trimmed)
            return isServing(trimmed)
        }
        // Tear down whatever's there (a DIFFERENT alias, ready or
        // mid-``.starting``). ``stop()`` is a noop if child is nil, so
        // the idle/stopped/missing cases just fall through to
        // ``start(alias:)``.
        if child != nil {
            await stop()
        }
        await start(alias: trimmed, hfPath: hfPath)
        // ``start`` also returns without spawning when the pre-load
        // memory guard parks the load on a confirmation prompt. Reading
        // ``isServing`` now would report "couldn't start the model"
        // while the user is still looking at the dialog — and the action
        // that triggered the load (typically a first chat message) would
        // be marked failed and then silently dropped the moment the user
        // picks "Load anyway". Wait for the answer instead.
        if pendingMemoryWarning != nil {
            let confirmed = await awaitMemoryDecision()
            if confirmed {
                // Wait out the actual re-entered launch — no fixed bound,
                // because it may legitimately sit on a background download.
                // Bind to THIS confirmation so an unrelated later cancel
                // cannot end the wait early.
                await awaitConfirmedLaunch(memoryConfirmSeq)
            }
        }
        // ``start`` returns silently when another caller already owns
        // the launch (``isOperating`` set, or ``child`` non-nil), which
        // leaves us sitting in ``.starting``. Reporting failure there
        // would tell the user "couldn't start the model" while the
        // model is in fact loading perfectly well — so wait it out
        // instead, exactly as in the short-circuit above.
        if case .starting(let current) = state, current == trimmed {
            await awaitStartupSettled(alias: trimmed)
        }
        return isServing(trimmed)
    }

    /// True when the child is serving exactly this alias.
    private func isServing(_ alias: String) -> Bool {
        if case .ready(let current) = state, current == alias { return true }
        return false
    }

    /// Suspend until the state machine leaves ``.starting`` for this
    /// alias — i.e. until the launch someone else owns either reaches
    /// ``.ready`` or fails.
    ///
    /// Polls rather than observing because ``state`` is read from a
    /// dozen SwiftUI surfaces and adding an observation seam here would
    /// be a much larger change; the poll only runs while a start is
    /// genuinely in flight, and 200 ms is far below human perception
    /// against a 15-60 s load. Returns early on cancellation so a
    /// caller that gives up doesn't pin this task.
    /// Blocks while a load is parked on the memory-confirmation prompt.
    /// Returns whether the user chose to load anyway. Polls rather than
    /// awaiting a continuation so a dismissed-without-answer alert (or a
    /// build with no view bound to ``pendingMemoryWarning``) resolves via
    /// ``cancelPendingMemoryLoad`` instead of stranding the caller.
    private func awaitMemoryDecision() async -> Bool {
        while pendingMemoryWarning != nil {
            do {
                try await Task.sleep(nanoseconds: 200_000_000)
            } catch {
                return false
            }
        }
        return memoryLoadConfirmed
    }


    /// Waits for a confirmed memory-risky launch to finish. Unbounded (a
    /// pre-spawn download can take minutes) but cancellation-aware: the
    /// ``Task.sleep`` throws when the CALLER is cancelled, so we stop waiting
    /// and leave the launch itself running — the user did ask for it.
    private func awaitConfirmedLaunch(_ seq: Int) async {
        while memoryConfirmRunning.contains(seq) {
            do {
                try await Task.sleep(nanoseconds: 200_000_000)
            } catch {
                return
            }
        }
    }

    private func awaitStartupSettled(alias: String) async {
        while true {
            guard case .starting(let current) = state, current == alias else { return }
            do {
                try await Task.sleep(nanoseconds: 200_000_000)
            } catch {
                return
            }
        }
    }

    // MARK: - Public API

    /// Refreshes `binaryPath` and resets to `.idle` / `.missing`. Called
    /// from the app's launch hook after the orphan sweep so the UI shows
    /// the correct initial state even if the user installed rapid-mlx
    /// just before launching Rapid.
    func refreshBinary() {
        let resolution = ServerLocator.locate()
        self.binaryResolution = resolution
        self.binaryPath = resolution?.binary
        if resolution == nil {
            state = .missing
        } else if case .missing = state {
            state = .idle
        }
    }

    /// Transitions a ``.crashed`` or ``.stopped`` state back to
    /// ``.idle`` when the user dismisses the error overlay
    /// (typically because they want to pick a different alias). Does
    /// NOT touch ``.starting`` or ``.ready`` — those represent live
    /// children that need a real ``stop()`` to wind down. Idempotent
    /// when no terminal state is set.
    func dismissTerminalState() {
        switch state {
        case .crashed, .stopped:
            // Issue #270: dismissing the crash banner is the user
            // saying "I've seen this, I want to move on" — almost
            // always followed by a fresh alias pick. A pending
            // auto-respawn racing them would re-load the very model
            // they just abandoned, against their intent. Cancel + reset
            // the retry budget here.
            cancelAutoRespawn()
            state = (binaryPath == nil) ? .missing : .idle
        default:
            return
        }
    }

    /// Spawn `rapid-mlx serve <alias> --host 127.0.0.1 --port 8000`,
    /// stream its stdout/stderr into `logLines`, and poll `/healthz`
    /// until it returns 200. On success the state transitions
    /// `.idle/.stopped -> .starting -> .ready`; on failure the child is
    /// torn down and state moves to `.crashed` with a human message.
    ///
    /// Issue #278: the optional ``isAutoRespawn`` flag tells the
    /// method whether the call is being driven by the watchdog
    /// auto-respawn timer (``runScheduledAutoRespawn`` only) or by a
    /// human action (Dock click, picker button, crash-banner Restart,
    /// auto-resume on app launch). Human-driven calls reset
    /// ``autoRespawnAttempts`` to 0 so a fresh "I'm taking over"
    /// click gets the documented 3-retry budget — previously this
    /// happened via the unconditional reset on the ``.ready``
    /// transition (the bug Bug A removed), which incidentally also
    /// covered manual restarts. The default is ``false`` to make the
    /// behavior obvious at every public call site.
    /// The user acknowledged the memory warning and wants to load
    /// anyway. Takes the ``warning`` by value (not off ``pendingMemory-
    /// Warning``) because the alert's dismissal clears that property on
    /// the same run-loop turn the button fires — reading it here would
    /// race to nil and silently drop the load. Re-enters ``start`` with
    /// the guard bypassed.
    func confirmPendingMemoryLoad(_ warning: ModelSizing.MemoryWarning) {
        memoryLoadConfirmed = true
        pendingMemoryWarning = nil
        // Publish the task BEFORE clearing nothing else — both this and the
        // waiter run on the MainActor, so a waiter that observes
        // ``pendingMemoryWarning == nil`` is guaranteed to see this task.
        memoryConfirmSeq += 1
        let seq = memoryConfirmSeq
        memoryConfirmRunning.insert(seq)
        memoryConfirmTask = Task { [weak self] in
            await self?.start(
                alias: warning.alias,
                hfPath: warning.hfPath,
                isAutoRespawn: warning.isAutoRespawn,
                bypassMemoryGuard: true
            )
            self?.memoryConfirmRunning.remove(seq)
        }
    }

    /// The user backed out of a memory-risky load. Just drops the
    /// held request; ``state`` is untouched (it never left idle/stopped
    /// because ``start`` returned before spawning).
    func cancelPendingMemoryLoad() {
        // Deliberately leaves ``memoryConfirmRunning`` alone: this cancels a
        // load that was never started, so any launch still in flight belongs
        // to an EARLIER confirmation and its waiter must not be told it
        // finished.
        memoryLoadConfirmed = false
        memoryConfirmTask = nil
        pendingMemoryWarning = nil
    }

    func start(
        alias: String,
        hfPath: String? = nil,
        isAutoRespawn: Bool = false,
        bypassMemoryGuard: Bool = false
    ) async {
        // Issue #278: a manual restart is the user taking over the
        // lifecycle — reset the budget at entry so a previously
        // exhausted counter doesn't make a quick post-restart crash
        // surface immediately instead of allowing the documented
        // 3-retry auto-respawn budget. Auto-respawn calls skip this
        // (they're already inside the budget bookkeeping).
        if !isAutoRespawn {
            autoRespawnAttempts = 0
        }
        guard !isOperating else { return }
        guard child == nil else { return }
        guard let binary = binaryPath else {
            state = .missing
            return
        }
        let trimmedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedAlias.isEmpty else { return }
        // Reject anything that could be misread as an extra argv flag
        // or that would inject newlines / control bytes into the log
        // stream. Alias grammar in vllm_mlx/aliases.json is
        // ``[a-z0-9._-]`` and the longest registered entry is ~32
        // chars; cap conservatively at 128 so a typo doesn't generate
        // a giant child argv. [codex audit r1 ServerManager.swift:308]
        guard Self.isValidAlias(trimmedAlias) else {
            state = .crashed(
                alias: trimmedAlias,
                message: "That model name isn't valid. Pick a model from the bar at the top."
            )
            return
        }

        // Pre-load memory guard (#324). Loading a model whose footprint,
        // stacked on top of what is ALREADY resident, would push unified
        // memory past ~85% of total can freeze or kernel-panic the whole
        // Mac — a far worse outcome than declining to load. This is the
        // single choke point every start path funnels through (picker,
        // first message, auto-restart, quickstart), so the check + the
        // confirmation prompt live here once instead of at each call site.
        // Unlike the picker's static ``ModelSizing.classify`` (bands vs
        // 80% of TOTAL), this projects the footprint onto LIVE used memory,
        // catching the reported near-crash where a "fits the Mac" model
        // still exhausts the RAM other apps left free. ``bypassMemoryGuard``
        // is set only by the explicit "Load anyway" path, after the user
        // acknowledged the risk in the prompt.
        //
        // Auto-respawn is exempt (``!isAutoRespawn``): the watchdog re-fires
        // ``start(isAutoRespawn: true)`` on a fixed schedule, so returning
        // early here would spin — set-warning → return → watchdog re-fires →
        // repeat, forever, with a modal stuck on screen and no model serving.
        // Respawn is also recovering a model that ALREADY fit when it first
        // started; a genuine free-RAM drop is bounded by the respawn-attempt
        // budget, and the user's manual restart still routes through the guard.
        if !bypassMemoryGuard, !isAutoRespawn, let snapshot = MemoryProbe.snapshot() {
            let footprint = ModelSizing.estimate(alias: trimmedAlias)
            let safety = ModelSizing.memorySafety(
                footprint: footprint,
                usedBytes: snapshot.usedBytes,
                totalBytes: snapshot.totalBytes
            )
            // Only ``.unsafe`` (>= 85%, the panic line) blocks. ``.tight``
            // is defined as "will load but risks swap / stalls" — holding
            // it behind a modal would fire on ordinary loads (13 GiB used
            // on a 32 GiB Mac + an 11.8 GiB model projects to 77.5%) and
            // train the user to click through the one prompt that matters.
            // Every mature local-model app draws the same line: refuse
            // only what is genuinely dangerous, and surface "tight"
            // passively — the picker's static sizing bands already do.
            if safety == .unsafe {
                pendingMemoryWarning = ModelSizing.MemoryWarning(
                    alias: trimmedAlias,
                    hfPath: hfPath,
                    isAutoRespawn: isAutoRespawn,
                    severity: safety,
                    footprintGB: footprint.totalGB,
                    freeGB: Double(snapshot.freeBytes) / Double(1 << 30),
                    totalGB: Double(snapshot.totalBytes) / Double(1 << 30)
                )
                memoryLoadConfirmed = false
                // The user is now the decision-maker for this alias, so a
                // queued auto-respawn must not answer for them. Parking a
                // load leaves ``state`` untouched — still ``.crashed`` when
                // the user hit Restart after a crash — so
                // ``runScheduledAutoRespawn``'s state recheck would PASS and
                // fire ``start(isAutoRespawn: true)``, which bypasses this
                // guard and loads the very model the user may be about to
                // decline. Cancel it; a confirm re-enters ``start`` explicitly.
                cancelAutoRespawn()
                return
            }
        }

        // Issue #253: if a background ``rapid-mlx pull`` is already
        // fetching this alias (kicked off from the picker's right-click
        // "Download in background", Settings → Model Management
        // Download, or the chat upgrade banner), stagger the serve
        // spawn behind it. Serve's own ``_ensure_model_downloaded``
        // will then cache-hit instead of racing the pull onto the
        // same HF shards — empirically that's the difference between
        // 2× disk + 2× bandwidth (with an orphan blob left behind)
        // and a clean ~15 s cache-hit start. ``state`` stays at
        // whatever it was on entry (``.idle`` / ``.stopped``) during
        // the wait; the user's escape hatch is the picker's Cancel
        // affordance on the pull job itself (``stop()`` would no-op
        // because no child has been spawned yet).
        if let downloads, downloads.isDownloading(trimmedAlias) {
            await downloads.awaitDownloadSettlement(alias: trimmedAlias)
            // Issue #278: ``awaitDownloadSettlement`` returns early on
            // Task cancellation (see DownloadManager.swift:182-188 —
            // it catches the cancellation thrown by ``Task.sleep`` and
            // returns rather than busy-looping). That cancellation is
            // exactly what ``stop()``/``shutdownSync()``/``dismiss-
            // TerminalState`` produce via ``cancelAutoRespawn`` when
            // this ``start(alias:)`` was reached through the watchdog
            // auto-respawn path. Without an explicit cancellation
            // check the existing ``!isOperating`` + ``child == nil``
            // guards both pass (the spawn critical section hasn't
            // been entered yet so ``isOperating`` was never set true
            // by this call, and ``stop`` already nilled ``child``),
            // and ``start`` would happily spawn a NEW child the user
            // just clicked Stop on. Bail before doing any further
            // work — every later check is gated on the same Task.
            if Task.isCancelled { return }
            // codex r1 BLOCKING: the await above is a MainActor
            // suspension point and ``start(alias:)`` is reentrant. A
            // second ``start()`` call landing on the actor while we're
            // parked here (e.g. user clicks Restart on the crash
            // banner while a previous start is waiting on its pull)
            // would have ALREADY passed the ``!isOperating`` /
            // ``child == nil`` guards above — they were evaluated
            // before this suspension point. Without a post-await
            // recheck the second call could spawn a serve child,
            // ``self.child = process`` succeeds, and then THIS
            // resumption would overwrite ``self.child`` with a SECOND
            // spawn against a different port, leaving the first child
            // orphaned with its bearer + pipes + ownership record
            // stranded. Re-check both guards and bail if anything
            // moved on during the wait.
            guard !isOperating else { return }
            guard child == nil else { return }
        }

        // Codex round 1-4 finding (all 4 rounds): the previous shape
        // held ``isOperating = true`` for the entire health/download
        // window (up to 30 minutes for a first-time large model
        // download). The UI disabled Stop while ``isOperating`` was
        // true and ``stop()`` also no-op'd, leaving the user with no
        // way to cancel a hung first-time pull.
        //
        // Split the operation into TWO phases:
        //   * "spawn critical section" — atomic process setup +
        //     ``process.run()``. ``isOperating`` is only true here,
        //     for at most a few hundred ms.
        //   * "health wait" — the long polling loop. ``isOperating``
        //     is false here; ``stop()`` can preempt by terminating
        //     ``child`` and the polling loop notices ``child == nil``
        //     and returns.
        didSignalShutdown = false
        isOperating = true

        // Clear the log tail from any previous run so the user only
        // sees output relevant to the current process.
        logLines.removeAll(keepingCapacity: true)
        downloadProgress.reset()
        // Stop any leftover byte monitor from a previous .starting
        // cycle before kicking a new one — defensive in case the
        // previous cycle exited without going through ``handleChildExit``
        // (e.g. spawn-thrown failure path).
        startupByteMonitor?.stop()
        startupByteMonitor = nil
        startedAt = Date()
        state = .starting(alias: trimmedAlias)
        expectedStop = false
        // Issue #270: clear the "this spawn ever became ready" flag —
        // the auto-respawn path in ``handleChildExit`` consults it to
        // decide whether the exit is worth retrying (was-healthy →
        // retry) vs. silently-broken-on-load (don't loop).
        spawnCycleReachedReady = false
        // Issue #278: clear the prior cycle's ready timestamp so
        // ``handleChildExit`` cannot mis-read a stale value from a
        // previous cycle as "this cycle was stable for ages, reset
        // the budget".
        readyAt = nil

        // Wire the on-disk byte monitor for this start cycle. HF's
        // outer "Fetching N files" tqdm bar counts FILES, not bytes —
        // on a 6.8 GB / 11-shard model the bar sits at "0/9 files (0%)"
        // for many minutes while the first shard streams silently. The
        // monitor polls ``~/.cache/huggingface/hub/models--<owner>--<repo>/``
        // every 3 s so the UI can render real bytes-on-disk progress
        // independent of tqdm cadence. Unknown hfPath / unresolvable HF
        // cache root leave the byte channel at ``nil``; the existing
        // tqdm-derived copy stays in charge.
        installStartupByteMonitor(alias: trimmedAlias, hfPath: hfPath)

        // Resolve a free port BEFORE starting the child. The previous
        // shape hard-coded :8000 and surfaced rapid-mlx's own
        // "Port 8000 is already in use" stderr as a generic crash —
        // common when the user runs vite / jupyter / fastapi on the
        // same machine. ``PortAllocator`` walks 8000..8009 and picks
        // the first port not held by a foreign process.
        //
        // Run OFF the main actor. ``allocate()`` is blocking work by
        // nature: per candidate port it forks ``lsof`` (+ one ``ps``
        // per pid it finds) and, when it finds a rapid-owned orphan,
        // waits out a SIGTERM grace before probing the bind. Calling
        // that synchronously from this @MainActor method froze the UI
        // for the whole walk — with several occupied candidates the
        // freeze ran into multiple seconds of unresponsive window and
        // spinning beachball, right at the moment the user clicked a
        // model. ``PortAllocator``/``PortSweep`` are plain nonisolated
        // enums over process + socket syscalls with no shared mutable
        // state, so hopping them onto a detached task is safe.
        //
        // The surrounding ``isOperating = true`` (set above, cleared
        // on every exit path below) is what makes this new suspension
        // point safe against ``start``'s documented reentrancy: a
        // second call landing on the actor while we're parked here
        // bails at the ``guard !isOperating`` at the top rather than
        // racing a second spawn onto a different port.
        // Wait out the opportunistic launch sweep before allocating. It runs
        // detached so launch is not blocked, but it reaps anything on the
        // candidate port that LOOKS like rapid-mlx — including a server this
        // launch just spawned, if an auto-start beat the sweep's `lsof` to the
        // port. Awaiting it here keeps launch responsive while making a spawn
        // and a sweep mutually exclusive.
        await PortSweep.awaitLaunchSweep()
        let allocated = await Task.detached(priority: .userInitiated) {
            PortAllocator.allocate()
        }.value
        // A quit can land while we were parked on the allocator above. Spawning
        // now would put a child on the far side of `beginShutdown`, so nothing
        // ever reaps it — an orphan holding the port for the next launch.
        if didSignalShutdown {
            isOperating = false
            return
        }
        guard let resolvedPort = allocated else {
            isOperating = false
            state = .crashed(
                alias: trimmedAlias,
                message: "Couldn't start the model — another app may already be using what Rapid needs to run. Quit other local AI apps (LM Studio, Ollama) or development servers, then click Restart."
            )
            return
        }
        activePort = resolvedPort

        // Issue #17 desktop-half: generate a per-launch bearer
        // secret, hand it to the child via RAPID_MLX_API_KEY, and
        // pin it on self so ChatStreamClient can add the matching
        // Authorization header. SecRandomCopyBytes failing is
        // pathological (kernel-level RNG starvation); we surface as
        // .crashed rather than silently spawning unauthenticated.
        guard let bearer = BearerSecret.generate() else {
            isOperating = false
            state = .crashed(
                alias: trimmedAlias,
                message: "Couldn't start the model securely. Restart Rapid-MLX; if this keeps happening, please file a bug."
            )
            return
        }
        // Codex r1 P3 (#17): hold the bearer in a local until the
        // spawn succeeds, then publish to ``activeBearer``. The
        // previous shape published BEFORE the spawn, so a thrown
        // spawn left ``activeBearer`` non-nil in the ``.crashed``
        // state and a follow-up chat attempt would send the stale
        // secret to whatever later bound ``activePort``.

        // Issue #271: there is exactly ONE spawn shape (cold start,
        // post-crash respawn, alias switch, auto-respawn all share it).
        // No ``--api-key`` / ``--listen-fd`` divergence — the bearer
        // travels through ``RAPID_MLX_API_KEY`` env so ``ps -ax`` from
        // an unprivileged local process can't read it, and the port is
        // passed as a plain ``--port <int>`` so the child binds via
        // the standard ``uvicorn`` shape.
        // Per-recommendation launch flags, derived HERE so every start
        // path funnels through one place — the composer's cold-start path,
        // explicit crash recovery, and auto-respawn all reach `start`
        // but none of them thread flags, so computing at the call sites
        // (as an earlier revision did) silently dropped them. RAM-gated:
        // `launchFlags` returns the flags only when this alias is the pick
        // for this Mac's RAM tier (e.g. the --no-mllm + kv-cache trio for
        // the 24 GB gemma-4-26b), so a hand-picked model that isn't the
        // recommendation gets none.
        let hardware = MacHardware.detect()
        var extraFlags = RAMBucketedDefault.launchFlags(
            forAlias: trimmedAlias,
            physicalRAMGB: hardware.physicalRAMGB
        )
        extraFlags.append(contentsOf: [
            "--resident-memory-limit-gb",
            String(format: "%.0f", ModelSizing.residentMemoryCeilingGB(on: hardware)),
            "--resident-model-idle-ttl", "1800",
        ])
        let arguments = Self.serveArguments(
            alias: trimmedAlias,
            host: host,
            port: activePort,
            extraFlags: extraFlags
        )

        // Issue #503: resolve the user's "Models folder" preference for
        // this launch. ``validatedOverrideURL`` returns the folder only
        // when it's set AND currently a reachable directory; a set-but-
        // unavailable folder (external drive unplugged) resolves to nil
        // so we fall back to the default location without failing the
        // load. Surface that fallback as a calm, non-fatal note in the
        // log tail so a user who unplugged their drive understands why
        // downloads went to the internal disk this time.
        let modelsFolderOverride = ModelsFolderPreference.validatedOverrideURL()?.path
        if modelsFolderOverride == nil, ModelsFolderPreference.hasCustomFolder() {
            appendLogLines([
                "Your chosen models folder isn't available right now — Rapid is using its default location until it's back."
            ])
        }
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        // Capture readability handlers. We append to `logLines` on the
        // main actor because SwiftUI reads it; the closures hop threads
        // off the Process IO queue, so we always wrap the append in a
        // Task pinned to MainActor.
        //
        // Crash-safe + non-blocking drain. `availableData` raises an
        // uncatchable NSException on a bad descriptor (SIGABRTs the
        // process) when the child's pipe FD races teardown, and
        // `read(upToCount:)` blocks until 64 KiB fills — which would
        // freeze the log tail at startup until that much stderr
        // accumulates. A per-pipe ``PipeDrainer`` OWNS its read handle
        // (keeping the FD valid for any late handler firing) and drains it
        // non-blocking; each is constructed here while the handle is live.
        let stdoutDrainer = PipeDrainer(stdoutPipe.fileHandleForReading)
        let stderrDrainer = PipeDrainer(stderrPipe.fileHandleForReading)
        let makeChunkHandler: (PipeDrainer) -> @Sendable (FileHandle) -> Void = { drainer in
            { [weak self] _ in
                let data = drainer.drain().data
                guard !data.isEmpty else { return }
                guard let text = String(data: data, encoding: .utf8) else { return }
                // rapid-mlx's HuggingFace tqdm output uses '\r' to refresh
                // in place when stderr is not a TTY. Treat both as
                // separators so progress refreshes show up as discrete
                // log lines.
                let lines = text.split(whereSeparator: { $0 == "\r" || $0 == "\n" })
                    .map(String.init)
                    .filter { !$0.isEmpty }
                guard !lines.isEmpty else { return }
                Task { @MainActor [weak self] in
                    self?.appendLogLines(lines)
                }
            }
        }
        stdoutPipe.fileHandleForReading.readabilityHandler = makeChunkHandler(stdoutDrainer)
        stderrPipe.fileHandleForReading.readabilityHandler = makeChunkHandler(stderrDrainer)

        // Termination handler fires on a background queue — must hop
        // back to MainActor before touching state.
        //
        // Codex round-4 finding: previously ``handleChildExit`` ran
        // unconditionally, blindly clearing ``child`` and pipes. If
        // an OLD child's termination handler fired AFTER a NEW
        // ``start()`` had already installed a NEW child, the stale
        // exit would tear down the new pipes and flip state to
        // crashed/stopped. Capture the spawning Process by reference
        // and ignore the callback when ``self.child !== proc`` —
        // the stale exit belongs to a process we already replaced.
        let process: ProcessGroupChild
        do {
            process = try ProcessGroupChild.spawn(
                executableURL: binary,
                arguments: arguments,
                standardInput: .nullDevice,
                standardOutput: stdoutPipe,
                standardError: stderrPipe,
                // Issue #272: ``replaceEnvironment: true`` + the
                // allowlist-filtered env from ``serveEnvironmentAdditions``
                // means the child runs with ONLY the bearer, our HF
                // pinning, and a small allowlisted subset of the
                // launcher's env. Third-party secrets a user may have
                // exported in their shell (ANTHROPIC_API_KEY etc.)
                // never enter the sidecar's address space.
                environmentAdditions: Self.serveEnvironmentAdditions(
                    bearer: bearer,
                    ambient: ProcessInfo.processInfo.environment,
                    // Issue #1412: the engine's server-oriented default may
                    // retain up to 20% of RAM in prefix-cache entries. The
                    // desktop shares unified memory with foreground apps, so
                    // give its sidecar a smaller, hardware-scaled ceiling.
                    physicalRAMBytes: MacHardware.detect().physicalRAMBytes,
                    availableRAMBytes: MemoryProbe.snapshot()?.freeBytes ?? 0,
                    // Issue #449: stamp the launcher's PID so the
                    // bundled rapid-mlx (>=PR #942) self-terminates
                    // when this process dies under SIGKILL. The
                    // sidecar polls ``os.getppid()`` against this
                    // value every 2 s and exits the moment the live
                    // PPID stops matching, instead of running on as
                    // an orphan reparented to launchd.
                    supervisorPID: ProcessInfo.processInfo.processIdentifier,
                    // Issue #503: honour the user's chosen models
                    // folder. ``validatedOverrideURL`` returns nil when
                    // no folder is set OR the folder isn't a reachable
                    // directory right now (external drive unplugged), so
                    // this transparently falls back to the default
                    // location — the model still loads, no crash.
                    modelsFolderOverride: modelsFolderOverride
                ),
                replaceEnvironment: true,
                startMonitorImmediately: false
            ) { [weak self] proc in
                let status = proc.terminationStatus
                let reason = proc.terminationReason
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.child === proc else {
                        // Stale termination from a replaced child — drop.
                        return
                    }
                    self.handleChildExit(process: proc, status: status, reason: reason)
                }
            }
        } catch {
            startedAt = nil
            // Spawn failed: the pipes were never handed to `self`, so
            // `teardownPipes()` can't reach them. Detach the readability
            // handlers we installed above so the FileHandle → handler →
            // PipeDrainer → FileHandle cycle is broken and both read FDs
            // are released. Without this, every failed launch leaks a pair
            // of descriptors (codex round-5 finding).
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            // Log the raw spawn error for support; the user only sees a
            // clean, actionable line. localizedDescription here can leak
            // a path, POSIX wording, or an error domain — engine internals
            // (principle: error copy must be human + actionable).
            print("[server] failed to start the model: \(error.localizedDescription)")
            state = .crashed(alias: trimmedAlias, message: "Couldn't start the model. Restart Rapid-MLX and try again.")
            isOperating = false
            return
        }

        self.child = process
        // Codex r1 P3 (#17): only publish the bearer after the spawn
        // has succeeded — see comment at the bearer guard above.
        self.activeBearer = bearer
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe
        // #20: persist ownership before startMonitor() so a crash
        // during startup still leaves a record the next-launch
        // PortSweep can use to clean up precisely (instead of
        // falling back to the basename heuristic).
        OwnedServerRecord(
            pid: process.processIdentifier,
            pgid: process.processGroupID,
            port: activePort,
            alias: trimmedAlias,
            writtenAt: Date()
        ).persist()
        process.startMonitor()

        // SPAWN CRITICAL SECTION ENDS HERE. From this point on
        // ``stop()`` is allowed to preempt the health wait — the
        // polling loop checks ``self.child`` on every iteration and
        // returns the moment it becomes nil (which ``stop()`` does
        // via ``terminateChild``).
        isOperating = false

        // Poll /healthz until 200 or the hard deadline expires. The
        // child's terminationHandler updates `state` to .crashed if the
        // process exits, so we also bail when `child` is nil.
        //
        // Codex audit r1 finding (ServerManager.swift:388): the prior
        // shape only checked ``self.child == nil`` — if a second
        // ``start()`` had spawned a NEW child while this loop was
        // sleeping, the loop would happily probe ``/healthz`` for the
        // NEW process and flip ``state = .ready(alias: trimmedAlias)``
        // with the OLD (stale) alias. Capture the spawning Process by
        // reference and bail when it no longer matches ``self.child``
        // — the new launch's loop owns the transition.
        // v0.7.13: the deadline is now a sliding window relative to
        // the most recently observed forward-progress signal, not a
        // fixed instant from launch. Initial reference is "now" so
        // a cold-start whose ``/healthz`` never answers AND whose
        // child never emits a recognisable progress line still gets
        // terminated within ``healthStallWindow`` of launch.
        var lastProgressAt = Date()
        while Self.shouldKeepWaitingForHealth(
            now: Date(),
            lastProgressAt: lastProgressAt,
            stallWindow: healthStallWindow
        ) {
            if self.child !== process {
                // Either terminationHandler nilled it (.crashed /
                // .stopped path) or a replace-start swapped in a
                // new process. Either way, this loop's launch is done.
                return
            }
            // Slide the deadline forward whenever the parser has
            // observed a fresh signal — heartbeat, R2 completion,
            // tqdm tick, or phase transition all advance
            // ``downloadProgress.lastTickAt``. Bytes-on-disk also
            // count: ``applyDiskObservation`` updates ``lastTickAt``
            // so HFCacheByteMonitor's disk polls count too. The net
            // effect: a download that's actively moving will never
            // hit the stall-window cap, no matter how slow.
            //
            // Invariant: ``downloadProgress.reset()`` is called by
            // ``start()`` before this loop runs, which sets
            // ``lastTickAt = .distantPast``. The ``tick >
            // lastProgressAt`` guard discards that sentinel (since
            // ``.distantPast < Date()``), so a child that NEVER
            // emits a recognised signal AND never answers /healthz
            // still hits the original 30-min hard cap measured from
            // launch — the safety invariant survives.
            let tick = downloadProgress.lastTickAt
            if tick > lastProgressAt {
                lastProgressAt = tick
            }
            if await probeHealth() {
                // PR #26 codex meta-review finding 4 (P2): re-check
                // child identity AFTER the await. ``start()`` is
                // main-actor reentrant across the ``probeHealth``
                // suspension point — a user rapidly clicking
                // Stop then Start can have substituted ``self.child``
                // with a NEW process while we were waiting on
                // /healthz. Flipping ``state = .ready(alias: old)``
                // here would overwrite the new launch's ``.starting``
                // with the stale alias, and worse, persist the wrong
                // alias as ``lastServedAlias`` for the next app
                // launch's resume logic.
                guard self.child === process else {
                    return
                }
                startedAt = nil
                state = .ready(alias: trimmedAlias)
                // Issue #270: mark the spawn cycle as "demonstrably
                // healthy" so a subsequent ``handleChildExit`` knows
                // an auto-respawn is worth attempting.
                spawnCycleReachedReady = true
                // Issue #278: stamp the moment we reached .ready so
                // ``handleChildExit`` can decide whether the prior
                // .ready window was long enough to be considered
                // demonstrably stable. The previous shape reset
                // ``autoRespawnAttempts = 0`` unconditionally here,
                // which meant a child that briefly answered /healthz
                // and then crashed within seconds (OOM-on-first-
                // inference, segfault-on-first-prompt) cleared the
                // 3-retry cap on EVERY cycle and the watchdog looped
                // forever at 2 s intervals. The reset now happens in
                // ``handleChildExit`` gated on the stability window.
                readyAt = nowProvider()
                // v0.5.3: remember this alias so the next app launch
                // can auto-restart it (LM Studio shape). Persist only
                // on the happy-path ``.ready`` — failures don't
                // overwrite the previous good-known value, so a
                // crashed launch attempt doesn't strand the resume
                // logic on a model the user can't actually load.
                UserDefaults.standard.set(trimmedAlias, forKey: Self.lastServedAliasKey)
                await refreshResidency()
                // v0.6 audit P1 (silent-crash detection): now that
                // the child is ready, start the runtime health
                // monitor so a subsequent silent crash surfaces
                // within ~90 s instead of "the next time the user
                // tries to send a chat".
                startRuntimeHealthMonitor(process: process, alias: trimmedAlias)
                return
            }
            // `try?` here would swallow CancellationError and leave
            // this loop spinning on the main actor at full speed —
            // firing already-cancelled probeHealth calls until the
            // 30-minute stall window lapsed. Return instead: a
            // cancelled caller wants us gone, and tearing the child
            // down is `stop()`'s job, not ours.
            do {
                try await Task.sleep(nanoseconds: UInt64(healthPollInterval * 1_000_000_000))
            } catch {
                return
            }
        }
        // Stall window elapsed with no progress AND no successful
        // /healthz. Tear the child down and report — phrased as a
        // stall rather than a wall-clock timeout so the user
        // understands the failure mode (not "you ran out of time"
        // but "we stopped seeing any signs of life from rapid-mlx").
        await terminateChild(
            reason: "The model stopped responding for \(Int(healthStallWindow / 60)) minutes."
        )
    }

    /// Pure decision helper for ``start``'s health-wait loop —
    /// extracted so unit tests can pin the stall-window math without
    /// having to stand up a real ``ServerManager`` + child process.
    /// Returns ``true`` while the loop should keep polling, ``false``
    /// once the stall window has lapsed since the last observed
    /// progress signal.
    static func shouldKeepWaitingForHealth(
        now: Date,
        lastProgressAt: Date,
        stallWindow: TimeInterval
    ) -> Bool {
        let idle = now.timeIntervalSince(lastProgressAt)
        return idle < stallWindow
    }

    /// Stop the running child. SIGTERM with a 2 s grace window, then
    /// SIGKILL if still alive. State transitions to `.stopped` on
    /// success.
    func stop() async {
        // Issue #270: the user clicked Stop. A pending auto-respawn
        // racing them would defeat the click — cancel it AND reset
        // the retry budget so a subsequent user-driven Start gets a
        // fresh count. Runs BEFORE the child guard because the user's
        // Stop intent applies even from ``.crashed`` (no live child but
        // a queued respawn would still race a subsequent state change).
        cancelAutoRespawn()
        guard !isOperating else { return }
        guard child != nil else { return }
        isOperating = true
        defer { isOperating = false }
        await terminateChild(reason: nil)
    }

    /// Synchronous, fire-and-forget teardown used by app shutdown
    /// (`applicationWillTerminate`). Cannot await because AppKit's
    /// terminate handler runs the main loop one more spin before
    /// returning control to the OS. We send SIGTERM, wait a short
    /// blocking window, then SIGKILL — same pattern as `stop()` but
    /// without the async hops.
    ///
    /// Split into ``beginShutdown`` (signal, non-blocking) and this
    /// reaping half so ``AppDelegate.runTerminationSequence`` can
    /// signal BOTH this child and the download children before any
    /// grace window starts, letting the two waits overlap instead of
    /// summing. The grace budget itself is unchanged.
    ///
    /// Returns immediately when there is nothing to reap so the quit
    /// path costs nothing in the common "server never started" case.
    func beginShutdown() {
        // Issue #270: app is quitting. Even if no child is currently
        // alive, a pending auto-respawn task could fire AFTER
        // ``applicationWillTerminate`` returned to AppKit but BEFORE
        // the process actually exits, spawning a child that gets
        // orphaned. Cancel unconditionally here, before the child guard.
        cancelAutoRespawn()
        // Actually idempotent, not merely "cheap to call twice".
        // ``shutdownSync`` calls this again at the start of the reap
        // phase, so without this latch the child receives a SECOND
        // SIGTERM part-way through its first graceful shutdown —
        // uvicorn reads that as "stop waiting, exit now" and abandons
        // the prefix-cache serialisation that the 5 s grace below exists
        // to protect, leaving the partial ``prefix_cache/<rev>.new/``
        // this teardown is specifically written to avoid.
        guard !didSignalShutdown else { return }
        guard let process = child else { return }
        guard process.isRunning || process.isProcessGroupAlive else { return }
        didSignalShutdown = true
        expectedStop = true
        process.signalProcessGroup(SIGTERM)
    }

    func shutdownSync() {
        // ``beginShutdown`` is idempotent and cheap, so calling it here
        // keeps this method correct as a standalone teardown even
        // though the production quit path calls it separately first.
        beginShutdown()
        guard let process = child else { return }
        guard process.isRunning || process.isProcessGroupAlive else { return }
        // NOTE: the 5 s SIGTERM grace is deliberate and load-bearing,
        // not slack to be trimmed — rapid-mlx's FastAPI shutdown hook
        // serialises the in-memory prefix cache to disk here, and
        // SIGKILLing mid-write leaves a partial ``prefix_cache/<rev>.new/``
        // that forces a full re-prefill on the next launch. See the
        // ``sigtermGracePeriod`` doc comment for the full rationale.
        // What this path DOES avoid is burning the grace when the
        // child is already gone: the poll below exits as soon as the
        // process group dies.
        let deadline = Date().addingTimeInterval(5.0)
        while Date() < deadline && process.isProcessGroupAlive {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if process.isProcessGroupAlive {
            process.signalProcessGroup(SIGKILL)
            // Give the kernel a beat to reap so the port is free on
            // next launch even if the user immediately reopens Rapid.
            let killGrace = Date().addingTimeInterval(0.5)
            while Date() < killGrace && process.isProcessGroupAlive {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        teardownPipes()
        child = nil
        // #17: clear the bearer the moment the child is gone so a
        // post-stop chat request can't slip through with a stale
        // secret targeting whatever happens to bind the port next.
        // (#1035: the nil transition evicts cached MCP tools via didSet.)
        activeBearer = nil
        // Issue #278: honour the "readyAt cleared on every child
        // exit" invariant in shutdownSync too (parallel to the
        // terminateChild defensive teardown). App-termination only,
        // so this is belt-and-braces against a hypothetical
        // teardown-during-restart race.
        readyAt = nil
        // codex r1 (#20 PR #142): the synchronous shutdown path
        // (applicationWillTerminate) didn't clear the persisted
        // ownership record. A normal Cmd-Q with the server running
        // left the file on disk; if the kernel later re-used the
        // recorded PID for an unrelated process, the next launch's
        // sweep would PGID-kill it. Best-effort clear here.
        OwnedServerRecord.clear()
    }

    // MARK: - Internals

    /// Common SIGTERM-then-SIGKILL teardown shared by `stop()` and the
    /// health-deadline path. `reason` is non-nil when called from the
    /// timeout branch; in that case the resulting state is `.crashed`
    /// rather than `.stopped`.
    ///
    /// codex r1 BLOCKING (#2): when ``terminateChild`` is invoked
    /// from inside ``runRuntimeHealthLoop``, the default
    /// ``cancelMonitor: true`` would cancel the currently-executing
    /// task — and the very next ``await Task.sleep`` inside this
    /// function would throw ``CancellationError``, collapsing both
    /// the 5 s SIGTERM grace and the 1 s SIGKILL grace to 0. The
    /// runtime loop now calls ``terminateChild(reason:cancelMonitor:false)``
    /// so its own teardown can sleep through the grace windows;
    /// the loop's natural ``return`` after the call clears the
    /// ``runtimeHealthTask`` handle via ``handleChildExit``'s
    /// cancel-on-exit branch.
    private func terminateChild(reason: String?, cancelMonitor: Bool = true) async {
        guard let process = child else { return }
        let alias: String
        switch state {
        case .starting(let a), .ready(let a), .crashed(let a, _):
            alias = a
        default:
            alias = ""
        }
        // v0.6 audit P1 (silent-crash detection): we are about to
        // tear the child down. Whatever the runtime monitor would
        // observe next is no longer relevant — it must NOT race the
        // teardown to flip state to ``.crashed`` after our caller
        // has set ``.stopped``. Skip when the caller IS the monitor
        // (see codex r1 BLOCKING #2 above).
        if cancelMonitor {
            cancelRuntimeHealthMonitor()
        }
        // codex r2 BLOCKING: gate the indirect cancel path.
        // ``handleChildExit`` (fired by the SIGTERM about to happen)
        // would otherwise call ``cancelRuntimeHealthMonitor`` and
        // cancel us mid-teardown, collapsing the grace windows
        // below. Reset on the way out.
        let priorInsideFlag = isInsideTerminateChild
        isInsideTerminateChild = true
        defer { isInsideTerminateChild = priorInsideFlag }
        expectedStop = (reason == nil)
        process.signalProcessGroup(SIGTERM)
        // Budget controlled by ``sigtermGracePeriod`` — see its
        // doc-comment for the prefix-cache-flush rationale that
        // drove the v0.7.6 5 s → 30 s bump.
        let graceDeadline = Date().addingTimeInterval(sigtermGracePeriod)
        while Date() < graceDeadline && process.isProcessGroupAlive {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        if process.isProcessGroupAlive {
            process.signalProcessGroup(SIGKILL)
            let killDeadline = Date().addingTimeInterval(1.0)
            while Date() < killDeadline && process.isProcessGroupAlive {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        // `terminationHandler` will fire and call `handleChildExit`,
        // which clears `child` and moves state to `.stopped` /
        // `.crashed`. But if for some reason the handler doesn't run
        // promptly (we've never seen this, but defensive), nil it out
        // here so a subsequent start() can proceed.
        if !process.isProcessGroupAlive {
            teardownPipes()
            child = nil
            // #17: see shutdownSync — bearer is dead the moment the
            // child is.
            activeBearer = nil
            startedAt = nil
            // Issue #278: defensive teardown also has to honour the
            // "readyAt cleared on every child exit" invariant in
            // case the terminationHandler is starved (app teardown,
            // signal-handler reentrancy). Without this, a hard
            // ``terminateChild`` followed by a fresh ``start()`` that
            // somehow stalls before reaching its own ``readyAt = nil``
            // line could read a stale value. Belt-and-braces.
            readyAt = nil
            // #20: we own this child's full lifecycle; the next
            // launch must not see a stale record pointing at a PID
            // that's already been reaped. Clear unconditionally —
            // ``handleChildExit`` may not get a chance to run if
            // the termination handler is starved by app teardown.
            OwnedServerRecord.clear()
            if let message = reason {
                state = .crashed(alias: alias, message: message)
            } else {
                state = .stopped
            }
        }
    }

    /// Runs on the main actor when the child terminates. Decides whether
    /// the exit was user-requested (`.stopped`) or unexpected
    /// (`.crashed`).
    private func handleChildExit(
        process: ProcessGroupChild,
        status: Int32,
        reason: Process.TerminationReason
    ) {
        let alias: String
        switch state {
        case .starting(let a), .ready(let a), .crashed(let a, _):
            alias = a
        default:
            alias = ""
        }
        teardownPipes()
        // The cache-dir byte monitor is bound to the ``.starting`` →
        // ``.ready`` window; the child is gone so the poller has no
        // more useful observations to make. ``Handle.stop`` is
        // idempotent — safe to call when no monitor was ever started.
        startupByteMonitor?.stop()
        startupByteMonitor = nil
        // v0.6 audit P1 (silent-crash detection): cancel the runtime
        // monitor on every child-exit path so a zombie probe loop
        // doesn't outlive its process — it would either keep
        // hammering a port we no longer own (worst case: a fresh
        // unrelated process bound to the same port answers and we
        // mis-report "ready") or it would flip ``state = .crashed``
        // after the user already saw ``.stopped``.
        //
        // codex r2 BLOCKING: skip when ``terminateChild`` is on the
        // stack — the SIGTERM it sent caused this handler to fire,
        // and cancelling now would cut off the very task driving
        // ``terminateChild``'s grace-window sleeps. ``terminateChild``
        // owns monitor-cleanup for its caller via the ``cancelMonitor``
        // parameter; we hand that responsibility back to it.
        if !isInsideTerminateChild {
            cancelRuntimeHealthMonitor()
        }
        let wasExpected = expectedStop
        expectedStop = false
        child = nil
        // #17: the child owns the secret; the secret is meaningless
        // (and a leak vector) once the child is gone.
        activeBearer = nil
        startedAt = nil
        // Issue #278: snapshot + window-gate the auto-respawn budget
        // reset, then clear ``readyAt``. The wasExpected-stop branch
        // below returns early; calling the helper here means the
        // documented "readyAt cleared on every child exit" invariant
        // holds for both paths, AND the same code services the
        // .crashed branch's reset gate so there is exactly one
        // copy of the decision logic.
        //
        // ``wasExpected == true`` corresponds to user-initiated stop
        // or shutdown — those paths separately zero the counter via
        // ``cancelAutoRespawn`` (called from ``stop()`` /
        // ``shutdownSync()`` / ``dismissTerminalState()``) so passing
        // ``reachedReadyThisCycle = false`` here just clears
        // ``readyAt`` without touching ``autoRespawnAttempts``.
        let reachedReadyThisCycle = spawnCycleReachedReady
        applyChildExitBudgetReset(reachedReadyThisCycle: !wasExpected && reachedReadyThisCycle)
        // #20: the child is gone (clean exit or crash). The next
        // launch must not pick up a record pointing at this PID,
        // which the kernel will re-use for an unrelated process
        // before the user even notices. Best-effort clear — if
        // ``terminateChild`` already cleared it, ``clear`` is a no-op.
        OwnedServerRecord.clear()
        if wasExpected {
            state = .stopped
            // v0.5.3: an explicit Stop is the user telling us they
            // don't want this model loaded anymore — clear the
            // persisted alias so the next launch doesn't auto-resume
            // it. Crash-paths fall through this branch (handled
            // below) and INTENTIONALLY keep the persisted value so
            // the user can hit Restart against the last-known-good
            // alias without picker re-selection.
            UserDefaults.standard.removeObject(forKey: Self.lastServedAliasKey)
            return
        }
        if process.isProcessGroupAlive {
            process.signalProcessGroup(SIGTERM)
            ProcessGroupChild.reapProcessGroupInBackground(processGroupID: process.processGroupID)
        }
        let message: String
        switch reason {
        case .exit:
            message = status == 0
                ? "The model stopped on its own (no restart was requested)."
                : "The model stopped unexpectedly."
        case .uncaughtSignal:
            // SIGKILL (9) on a model process is almost always the macOS
            // memory pressure killer — surface an OOM-aware, actionable
            // message instead of a raw signal number.
            message = status == 9
                ? "The model ran out of memory and was stopped. Try a smaller model, or close other apps to free up memory."
                : "The model stopped unexpectedly."
        @unknown default:
            message = "The model stopped unexpectedly."
        }
        state = .crashed(alias: alias, message: message)
        // Issue #270: silent idle-state crash. The user closed every
        // chat window via Cmd+W and then rapid-mlx died (OOM, SIGSEGV,
        // model worker hang). Previously the desktop stayed alive but
        // did nothing — no tray badge, no notification, and Dock click
        // didn't trigger a respawn either (only Cmd+N did). The next
        // window-open hit a stale ``.crashed`` banner instead of a warm
        // model.
        //
        // Watchdog-respawn unconditionally so the next Dock click /
        // window open / chat send finds a model ready to answer. We
        // gate on ``spawnCycleReachedReady`` so a "user just clicked an
        // alias that doesn't load" failure can't busy-loop spawning
        // a broken model. The retry counter (capped at
        // ``autoRespawnRetryLimit``) catches the case where the model
        // was healthy then started crashing repeatedly — eventually
        // the user has to take action.
        if Self.shouldScheduleAutoRespawn(
            reachedReadyThisCycle: reachedReadyThisCycle,
            alias: alias,
            attempts: autoRespawnAttempts,
            retryLimit: Self.autoRespawnRetryLimit
        ) {
            scheduleAutoRespawn(alias: alias)
        }
    }

    /// Pure decision helper for the auto-respawn budget reset gate in
    /// ``handleChildExit``. Returns ``true`` when the prior spawn cycle
    /// was demonstrably stable — reached ``.ready`` AND stayed there
    /// for at least ``stableWindow`` — and the retry budget can be
    /// refreshed to zero. Returns ``false`` for never-ready cycles and
    /// for cycles that crashed within the window (the latter is the
    /// pathological "ready -> crash" loop the retry cap is supposed to
    /// catch). Exposed ``internal static`` for unit tests.
    ///
    /// ``readyAt == nil`` returns ``false``: production should always
    /// have stamped the timestamp before flipping
    /// ``reachedReadyThisCycle = true``; a nil here is a bug-equivalent
    /// race and we conservatively don't refresh the budget.
    nonisolated internal static func shouldResetAutoRespawnBudget(
        reachedReadyThisCycle: Bool,
        readyAt: Date?,
        now: Date,
        stableWindow: TimeInterval
    ) -> Bool {
        guard reachedReadyThisCycle else { return false }
        guard let readyAt else { return false }
        return now.timeIntervalSince(readyAt) >= stableWindow
    }

    /// Pure decision helper for the auto-respawn gate in
    /// ``handleChildExit``. Three independent conditions, all of which
    /// must hold:
    ///
    ///   * ``reachedReadyThisCycle`` — the now-dead spawn cycle had
    ///     answered ``/healthz`` 200 at least once. A child that died
    ///     before reaching ``.ready`` (alias not in cache, model file
    ///     corrupted, Metal-shader compile fails on this GPU) is the
    ///     "broken alias" shape and respawning would busy-loop. The
    ///     user picked something that can't run; respawning won't
    ///     change that.
    ///   * ``!alias.isEmpty`` — we know which alias to respawn. The
    ///     idle path (no child ever started) has alias == "" and we
    ///     have nothing to bring back.
    ///   * ``attempts < retryLimit`` — we haven't exhausted the budget
    ///     yet. A model that was healthy then started crashing on
    ///     every respawn (memory pressure, runaway prefix-cache flush
    ///     hitting OOM) eventually has to surface to the user.
    ///
    /// Exposed ``internal static`` so unit tests can pin every branch
    /// of the truth table without standing up a real spawn.
    nonisolated internal static func shouldScheduleAutoRespawn(
        reachedReadyThisCycle: Bool,
        alias: String,
        attempts: Int,
        retryLimit: Int
    ) -> Bool {
        guard reachedReadyThisCycle else { return false }
        guard !alias.isEmpty else { return false }
        guard attempts < retryLimit else { return false }
        return true
    }

    /// Issue #278: shared "this child has exited, apply the
    /// stability-window-gated reset" step. Snapshots ``readyAt``,
    /// clears it (honouring the documented "cleared on every child
    /// exit" invariant), and conditionally zeros
    /// ``autoRespawnAttempts``. Called by ``handleChildExit`` on the
    /// real exit path AND by the ``_testApplyChildExitBudgetReset``
    /// seam — keeping one copy of the decision logic means the
    /// production path can't drift from what unit tests cover.
    ///
    /// ``reachedReadyThisCycle`` is parameterised (not read directly
    /// from ``spawnCycleReachedReady``) so the production caller can
    /// pass ``false`` on the expected-stop branch — that branch
    /// separately zeros the counter via ``cancelAutoRespawn`` so all
    /// this needs to do is clear ``readyAt``.
    private func applyChildExitBudgetReset(reachedReadyThisCycle: Bool) {
        let priorReadyAt = readyAt
        readyAt = nil
        if Self.shouldResetAutoRespawnBudget(
            reachedReadyThisCycle: reachedReadyThisCycle,
            readyAt: priorReadyAt,
            now: nowProvider(),
            stableWindow: Self.autoRespawnReadyStableWindow
        ) {
            autoRespawnAttempts = 0
        }
    }

    /// Schedule an auto-respawn for ``alias`` after ``autoRespawnDelay``.
    /// Idempotent — cancels any pending respawn before re-arming so a
    /// rapid double-crash doesn't queue two ``start()`` tasks racing the
    /// same actor. Bumps the attempt counter inside ``runScheduledAutoRespawn``
    /// so a cancellation BEFORE the timer fires (user took manual action
    /// before the 2 s window elapsed) doesn't burn a retry slot.
    ///
    /// Tests bypass the ``Task.sleep`` delay by calling
    /// ``runScheduledAutoRespawn(alias:)`` directly via the internal seam.
    private func scheduleAutoRespawn(alias: String) {
        autoRespawnTask?.cancel()
        let nanos = UInt64(Self.autoRespawnDelay * 1_000_000_000)
        autoRespawnTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: nanos)
            guard let self else { return }
            if Task.isCancelled { return }
            await self.runScheduledAutoRespawn(alias: alias)
        }
    }

    /// Body of the auto-respawn timer. Pulled out so the test seam
    /// can drive it directly without waiting on a real ``Task.sleep``.
    /// Re-checks every precondition that could have flipped while the
    /// timer was sleeping (user clicked Stop, user clicked Restart,
    /// user picked a different alias, the binary disappeared).
    internal func runScheduledAutoRespawn(alias: String) async {
        // User took manual action between the crash and the timer
        // firing — bail without burning a retry slot. ``.crashed``
        // for our alias is the ONLY state where auto-respawn is
        // appropriate; everything else means a human is driving.
        switch state {
        case .crashed(let a, _) where a == alias:
            break
        default:
            return
        }
        // Binary went away (uninstall, runtime-override revoked) —
        // surface as missing instead of looping on a non-existent
        // file. ``start(alias:)`` would flip to ``.missing`` anyway,
        // but doing it here also burns a retry slot for nothing.
        guard binaryPath != nil else {
            return
        }
        autoRespawnAttempts += 1
        // Issue #278: pass ``isAutoRespawn: true`` so ``start()``'s
        // entry-reset (added for manual-restart parity) doesn't
        // wipe the attempt counter we just incremented.
        await start(alias: alias, isAutoRespawn: true)
    }

    /// Cancel any pending auto-respawn and clear the attempt counter.
    /// Invoked by every code path where the user takes manual control
    /// of the lifecycle so a queued respawn can't race their intent
    /// (``stop()``, ``dismissTerminalState()``, ``shutdownSync()``).
    /// A fresh ``start(alias:)`` does not call here — the queued task's
    /// own state-recheck (``runScheduledAutoRespawn``) bails when state
    /// is no longer ``.crashed`` for the matching alias.
    private func cancelAutoRespawn() {
        autoRespawnTask?.cancel()
        autoRespawnTask = nil
        autoRespawnAttempts = 0
    }

    /// Tear down stdout/stderr readability handlers so ARC can free the
    /// pipes. Failing to nil these out leaks the file descriptors for
    /// the lifetime of the app.
    private func teardownPipes() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
        stderrPipe = nil
    }

    /// Append log lines and trim to capacity. Public for tests; called
    /// from the stdout/stderr readability handlers via MainActor hops.
    ///
    /// Tqdm progress lines are dispatched into ``downloadProgress`` so
    /// the top-bar progress pill can re-render — they are still kept in
    /// the log tail because a power user might want to scroll the raw
    /// HuggingFace output too (a future "filter progress" toggle could
    /// drop them).
    /// Resolve the per-alias HF cache directory and kick off a
    /// ``HFCacheByteMonitor`` polling task for the current start
    /// cycle. Wires ``downloadProgress.setTotalBytes`` from the
    /// ``ModelSizing`` weight estimate so the UI can render
    /// "X / Y GB · Z%" the moment the first observation lands.
    ///
    /// All failure modes (no hfPath supplied, sanitisation reject,
    /// HF cache root unresolvable) leave the byte channel at ``nil``
    /// without affecting the spawn — the UI falls through to the
    /// existing tqdm-derived copy.
    ///
    /// The monitor task captures the @MainActor-isolated
    /// ``downloadProgress`` instance and updates it via
    /// ``MainActor.run`` from a detached low-priority task — the
    /// observable property writes are serialised through the actor,
    /// so SwiftUI sees a coherent (bytes, total) view at all times.
    private func installStartupByteMonitor(alias: String, hfPath: String?) {
        let totalBytes = DownloadManager.estimateTotalBytes(for: alias)
        downloadProgress.setTotalBytes(totalBytes)
        guard let hfPath, !hfPath.isEmpty else {
            return
        }
        guard let hubCacheRoot = BundledModel.userHFCacheURL(
            environment: ProcessInfo.processInfo.environment,
            // Issue #503: watch the SAME directory the engine is about
            // to download into when a custom models folder is set, so
            // the bytes-on-disk overlay reflects real progress instead
            // of watching an empty default cache at 0%.
            preferredOverride: ModelsFolderPreference.validatedOverrideURL()
        ) else {
            return
        }
        guard let cacheDir = HFCacheByteMonitor.cacheDirectoryURL(
            hubCacheRoot: hubCacheRoot,
            hfPath: hfPath
        ) else {
            return
        }
        let progress = downloadProgress
        startupByteMonitor = HFCacheByteMonitor.start(
            cacheDir: cacheDir,
            progress: progress
        )
    }

    private func appendLogLines(_ lines: [String]) {
        // Pass the RAW line to `downloadProgress.ingest` — the
        // progress parser scans for tqdm tokens (`%`, `[…<…]`,
        // `it/s`) that the scrubber's URL/header patterns don't
        // touch, but keeping the contract that the parser sees
        // the original text avoids surprising future patterns
        // (e.g. a "Downloading https://hf.co/…?token=…" line
        // whose URL the scrubber will redact, while the progress
        // parser only cares about the trailing `[X/Y]`).
        //
        // v0.7.11 review NIT #3: the R2 puller fires its aggregate
        // ``[bytes] D/T`` heartbeat at ~2 Hz × 60-120 s, which
        // would otherwise evict every legitimate startup log /
        // warning from the 200-line ``logLines`` ring buffer well
        // before the user opened the log drawer. Suppress those
        // exact lines from the user-visible log tail — they're
        // already consumed by the progress overlay above and
        // carry no signal a human would want to read.
        var displayable: [String] = []
        displayable.reserveCapacity(lines.count)
        for line in lines {
            downloadProgress.ingest(line)
            if !DownloadProgress.isHeartbeatLogLine(line) {
                displayable.append(line)
            }
        }
        // Scrub BEFORE storage so a memory dump or "Copy Logs"
        // CTA also surfaces the redacted text — audit P1.
        let scrubbed = displayable.map(LogScrubber.scrub)
        logLines.append(contentsOf: scrubbed)
        if logLines.count > logBufferCapacity {
            logLines.removeFirst(logLines.count - logBufferCapacity)
        }
    }

    /// Best-effort GET on `/healthz`. We use `URLSession` rather than a
    /// raw TCP write because Foundation already ships a robust HTTP
    /// client and v0.2 has no binary-size constraint to justify
    /// reinventing it. A 1.5 s per-request timeout keeps the poll
    /// loop responsive.
    private func probeHealth() async -> Bool {
        guard let url = URL(string: "http://\(host):\(activePort)/healthz") else { return false }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 1.5
        do {
            let (_, response) = try await healthSession.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200..<300).contains(http.statusCode)
        } catch {
            return false
        }
    }

    /// v0.6 audit P1 (silent-crash detection): spawn the runtime
    /// /healthz monitor. Bound to one launch by capturing the
    /// ``process`` reference at spawn time — a later ``start()``
    /// that swaps ``self.child`` causes the loop to bail rather
    /// than mis-attribute a probe failure to the new launch. Idempotent:
    /// cancels any prior monitor first so a fast .ready → terminate →
    /// .ready cycle doesn't leak overlapping loops.
    private func startRuntimeHealthMonitor(process: ProcessGroupChild, alias: String) {
        cancelRuntimeHealthMonitor()
        runtimeHealthTask = Task { @MainActor [weak self] in
            await self?.runRuntimeHealthLoop(process: process, alias: alias)
        }
    }

    /// Cancel + nil the runtime monitor. Safe to call when nothing is
    /// running. Called from ``terminateChild``, ``handleChildExit``,
    /// and the top of ``startRuntimeHealthMonitor`` itself.
    private func cancelRuntimeHealthMonitor() {
        runtimeHealthTask?.cancel()
        runtimeHealthTask = nil
    }

    /// Body of the runtime health loop. Polls ``probe`` every
    /// ``interval`` seconds; after ``threshold`` consecutive
    /// failures, flips the state to ``.crashed`` and triggers a
    /// clean teardown.
    ///
    /// Bail conditions (all return without state change):
    ///   * ``Task.isCancelled`` — set by ``cancelRuntimeHealthMonitor``.
    ///   * ``self.child !== process`` — a replace-start swapped in
    ///     a new child; the new launch's monitor owns the transition.
    ///   * ``state`` is no longer ``.ready`` for our alias —
    ///     somebody else has already moved on.
    ///
    /// Parameters are taken with defaults so the production path
    /// (``startRuntimeHealthMonitor``) remains a no-arg call while
    /// tests inject a tight loop + deterministic probe to pin the
    /// contract without spinning up a real HTTP server.
    /// [codex r1 BLOCKING #3]
    internal func runRuntimeHealthLoop(
        process: ProcessGroupChild,
        alias: String,
        interval: TimeInterval? = nil,
        threshold: Int? = nil,
        probe: (() async -> Bool)? = nil
    ) async {
        let actualInterval = interval ?? runtimeHealthInterval
        let actualThreshold = threshold ?? runtimeHealthFailureThreshold
        var consecutiveFailures = 0
        // Sleep first so the first runtime probe is offset one
        // interval past the startup loop's final successful probe
        // — back-to-back probes against a still-warming worker can
        // spuriously fail.
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: UInt64(actualInterval * 1_000_000_000))
            if Task.isCancelled { return }
            // Identity check BEFORE the probe — a replace-start may
            // have happened during the sleep.
            guard self.child === process else { return }
            // Skip probing if state drifted away from .ready for
            // our alias. A manual stop / restart can be in flight;
            // let ``terminateChild`` own the transition.
            switch state {
            case .ready(let a) where a == alias:
                break
            default:
                return
            }
            let ok: Bool
            if let probe = probe {
                ok = await probe()
            } else {
                ok = await probeHealth()
            }
            // codex r1 BLOCKING #1: re-check cancellation, identity,
            // AND state after the probe await — the loop may have
            // been cancelled mid-probe by a stop()/teardown that
            // already moved state to ``.stopped``. Incrementing the
            // failure counter past that point and flipping to
            // ``.crashed`` would silently un-do the user's Stop.
            if Task.isCancelled { return }
            guard self.child === process else { return }
            switch state {
            case .ready(let a) where a == alias:
                break
            default:
                return
            }
            if ok {
                consecutiveFailures = 0
                continue
            }
            consecutiveFailures += 1
            if consecutiveFailures >= actualThreshold {
                // Lock in the crash BEFORE tearing the child down so
                // a deadlocked-but-alive Python worker (where
                // ``terminationHandler`` never fires) still surfaces
                // an amber UI pill.
                state = .crashed(
                    alias: alias,
                    message: "The model stopped responding for \(Int(actualInterval) * actualThreshold) seconds."
                )
                // codex r1 BLOCKING #2: pass cancelMonitor=false so
                // ``terminateChild``'s own ``Task.sleep`` grace
                // windows (5 s SIGTERM, 1 s SIGKILL) aren't
                // collapsed to zero by us-cancelling-ourselves.
                await terminateChild(reason: "The model stopped responding.", cancelMonitor: false)
                return
            }
        }
    }

    /// Validate that ``alias`` is safe to pass as argv[1] to
    /// ``rapid-mlx serve``. The two security-critical invariants:
    ///
    ///   * No leading ``-`` — otherwise the value is parsed as a CLI
    ///     flag and an attacker who controls the alias picker (e.g.
    ///     deep-link, URL-scheme handler, future scripting bridge)
    ///     could smuggle ``--config /etc/passwd`` or similar.
    ///   * No control characters (anything < 0x20 or 0x7F) — keeps
    ///     log-tail rendering safe and prevents terminal-escape
    ///     injection into the stderr stream we surface in the UI.
    ///   * No whitespace, no shell meta — even though we never go
    ///     through ``sh``, this defends against a future change that
    ///     accidentally adds an ``sh -c`` indirection.
    ///
    /// Cap length at 128 — registered aliases top out at ~32 chars;
    /// 128 absorbs any plausible hf_path the user might paste.
    /// Exposed ``internal`` so the test suite can pin the contract.
    /// ``nonisolated`` because this is a pure grammar check with no
    /// shared state — also lets ``serveArguments`` invoke it from a
    /// nonisolated context for the defense-in-depth assert.
    /// [codex audit r1 ServerManager.swift:308]
    nonisolated static func isValidAlias(_ alias: String) -> Bool {
        guard !alias.isEmpty, alias.count <= 128 else { return false }
        guard !alias.hasPrefix("-") else { return false }
        for scalar in alias.unicodeScalars {
            let v = scalar.value
            // Reject ASCII control + DEL.
            if v < 0x20 || v == 0x7F { return false }
            // Allow A-Z, a-z, 0-9.
            if (v >= 0x30 && v <= 0x39)
                || (v >= 0x41 && v <= 0x5A)
                || (v >= 0x61 && v <= 0x7A) {
                continue
            }
            // Allow the small punctuation set the alias grammar uses
            // plus ``/`` and ``:`` so an hf_path-shaped value works.
            switch v {
            case 0x2E, 0x5F, 0x2D, 0x2F, 0x3A: continue  // . _ - / :
            default: return false
            }
        }
        return true
    }

    // MARK: - Unified spawn shape (issue #271)

    /// Pure builder for the ``rapid-mlx serve`` argv. Issue #271 pins
    /// exactly ONE spawn shape regardless of launch trigger (cold
    /// start, post-crash respawn, alias switch, auto-respawn):
    ///
    ///   * No ``--api-key`` flag — the per-launch bearer travels via
    ///     ``RAPID_MLX_API_KEY`` env (see ``serveEnvironmentAdditions``)
    ///     so an unprivileged local process can't ``ps -ax | grep``
    ///     it out.
    ///   * No ``--listen-fd`` — we hand the child a port number and let
    ///     it bind via the standard ``uvicorn`` shape. ``--listen-fd``
    ///     would imply an FD-inheritance dance that has no reason to
    ///     exist here (we own port allocation in ``PortAllocator``).
    ///   * Alias passed as a positional argv[1]. ``isValidAlias`` has
    ///     already rejected leading-dash and control-character shapes
    ///     before we build this list, so it can't be misread as a flag.
    ///   * Explicit ``--cors-origins http://127.0.0.1 http://localhost``
    ///     (issue #306). Without this flag the sidecar defaults to
    ///     ``["*"]`` (``vllm_mlx/cli.py:899``); a wildcard CORS
    ///     allowlist combined with #303 (bearer env not yet enforced
    ///     as 401) would let any drive-by webpage drive the user's
    ///     local model via ``fetch``. Today's bundled build (v0.7.37)
    ///     happens to ship the middleware inert (preflight returns
    ///     405), but pinning the policy here keeps the desktop safe
    ///     across every future bundle bump.
    ///
    ///     The desktop's own client is the in-process SwiftUI app
    ///     calling the loopback HTTP API from native code — no Origin
    ///     header, so CORS does not apply to it regardless of the
    ///     allowlist. The allowlist therefore exists solely to defend
    ///     against drive-by webpages. The two values are the
    ///     ``scheme://host`` shapes a browser sends as ``Origin``
    ///     when the page itself is served from default-port
    ///     loopback (``http://localhost/`` or ``http://127.0.0.1/``);
    ///     Starlette ``CORSMiddleware.allow_origins`` is exact-match
    ///     (no wildcard subdomain / port semantics), so a browser
    ///     tool served from ``http://localhost:3000`` (Open WebUI's
    ///     default) sends ``Origin: http://localhost:3000`` and is
    ///     intentionally rejected by this minimal allowlist. That is
    ///     the SECURE default — third-party browser tools on
    ///     non-default ports are NOT a supported integration; the
    ///     desktop's own SwiftUI client does not need them. If/when
    ///     in-browser tooling becomes a product use case, this
    ///     allowlist (or a sidecar ``--cors-origin-regex`` flag) is
    ///     the right place to extend.
    ///
    /// Exposed ``internal static`` so ``SpawnArgumentsTests`` can pin
    /// the contract without standing up a real spawn.
    nonisolated internal static func serveArguments(
        alias: String,
        host: String,
        port: Int,
        extraFlags: [String] = []
    ) -> [String] {
        // Defense in depth: ``start(alias:)`` already calls
        // ``isValidAlias`` before reaching here, but a future caller
        // bypassing that gate would re-introduce the leading-dash
        // injection risk (alias parsed as ``--port`` etc.). Catch
        // misuse in debug builds; production trusts the gate.
        assert(isValidAlias(alias), "serveArguments requires an alias that already passed isValidAlias(_:)")
        var args = [
            "serve",
            alias,
            "--host", host,
            "--port", String(port),
            // Issue #306: pin an explicit loopback-only CORS allowlist
            // so a future rapid-mlx bundle bump that wires the CORS
            // middleware can't silently re-enable wildcard
            // (``Access-Control-Allow-Origin: *``) on the user's
            // local-only LLM. ``--cors-origins`` uses ``nargs="+"`` so
            // the two URL values trail the flag and consume up to the
            // next ``--``-prefixed flag.
            "--cors-origins", "http://127.0.0.1", "http://localhost",
        ]
        // Per-recommendation launch flags (e.g. the gemma-4-26b KV-cache
        // trio on the 24 GB tier). Appended AFTER ``--cors-origins`` so the
        // leading ``--`` of the first flag terminates that flag's
        // ``nargs="+"`` collection. ``start(alias:)`` derives ``extraFlags``
        // centrally from ``RAMBucketedDefault.launchFlags`` — only non-empty
        // when the alias is the recommended pick for THIS Mac's RAM — so a
        // hand-picked model on a larger Mac keeps its full capabilities.
        args.append(contentsOf: extraFlags)
        return args
    }

    /// Issue #272: the env we hand the bundled ``rapid-mlx`` child is
    /// constructed from a small explicit allowlist of ambient vars +
    /// our own desktop-injected overrides. Anything not on this list
    /// (``ANTHROPIC_API_KEY``, ``BRAVE_API_KEY``, ``OPENAI_*``,
    /// ``GH_TOKEN`` etc. exported in the launching shell) is DROPPED so
    /// it can't surface in ``ps eww`` against the sidecar PID or leak
    /// into crash-reporter / telemetry snapshots of the child's env.
    ///
    /// Allowlist (not a denylist — denylists miss the next new
    /// third-party secret var):
    ///
    ///   * POSIX baseline that any well-behaved CLI needs to find its
    ///     resolver/locale/tmpdir: ``PATH``, ``HOME``, ``USER``,
    ///     ``LOGNAME``, ``LANG``, ``LC_ALL``, ``LC_CTYPE``, ``TMPDIR``,
    ///     ``TZ``.
    ///   * Python launcher pointers so the bundled interpreter finds
    ///     its stdlib + site-packages: ``PYTHONHOME``, ``PYTHONPATH``.
    ///   * CA bundle pointers so ``huggingface_hub`` outbound requests
    ///     can verify TLS: ``SSL_CERT_FILE``, ``SSL_CERT_DIR``.
    ///   * macOS-injected bookkeeping that some Foundation paths
    ///     consult and that's already public via ``ps``:
    ///     ``__CFBundleIdentifier``, ``XPC_SERVICE_NAME``.
    ///   * HF cache-root pointers so the child resolves the same
    ///     directory the launcher byte monitor watches:
    ///     ``HF_HOME``, ``HF_HUB_CACHE``, ``XDG_CACHE_HOME``.
    ///     These are PATH config (not secrets) — the launcher's own
    ///     ``BundledModel.userHFCacheURL`` reads them from ambient
    ///     when picking which directory to watch, so dropping them
    ///     from the child's env splits the cache: the launcher
    ///     watches ``HF_HOME/hub``, while the child falls back to
    ///     ``~/.cache/huggingface/hub`` and downloads there. The
    ///     bytes-on-disk progress overlay sees 0% while the model
    ///     actually downloads twice on a re-launch (codex #275
    ///     post-merge audit; issue #277).
    ///   * HF Hub behavior knobs the user can already configure
    ///     ambiently and that ``DownloadManager.augmentedEnv``
    ///     already forwards to ``rapid-mlx pull`` (full ambient
    ///     passthrough). Without these on the serve-side allowlist,
    ///     ``rapid-mlx pull`` and the in-band ``rapid-mlx serve``
    ///     cold-download disagree on offline mode / private mirror
    ///     / telemetry / hf_transfer — e.g. a user with
    ///     ``HF_HUB_OFFLINE=1`` sees the background pull respect
    ///     offline mode but a serve-side cold path tries the network
    ///     and fails opaquely. Forwarding these closes that
    ///     asymmetry (codex #279 r1):
    ///     ``HF_ENDPOINT`` (private mirror URL),
    ///     ``HF_HUB_OFFLINE`` (offline mode),
    ///     ``HF_HUB_DISABLE_TELEMETRY`` (privacy),
    ///     ``HF_HUB_ENABLE_HF_TRANSFER`` (perf).
    ///     All four are non-secret functional config; the same
    ///     ambient-tampering threat applies to any allowlist
    ///     entry and is materially the same risk class as
    ///     ``HF_HUB_DISABLE_XET`` already on the override channel.
    ///
    /// Desktop-injected (always added, override the allowlist):
    /// ``RAPID_MLX_API_KEY`` (bearer; argv stays clean per #271),
    /// ``PYTHONUNBUFFERED`` (so tqdm reaches our log tail),
    /// ``HF_HUB_DISABLE_PROGRESS_BARS`` (force bars on), plus
    /// ``HF_HUB_DISABLE_XET`` / ``HF_HUB_DOWNLOAD_TIMEOUT`` (with
    /// ambient pass-through so the power-user override channel
    /// survives the allowlist).
    nonisolated internal static let serveEnvironmentAllowlist: Set<String> = [
        "PATH", "HOME", "USER", "LOGNAME",
        "LANG", "LC_ALL", "LC_CTYPE",
        "TMPDIR", "TZ",
        "PYTHONHOME", "PYTHONPATH",
        "SSL_CERT_FILE", "SSL_CERT_DIR",
        "__CFBundleIdentifier", "XPC_SERVICE_NAME",
        "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME",
        "HF_ENDPOINT", "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_ENABLE_HF_TRANSFER",
    ]

    /// Subset of the allowlist whose semantics demand "unset" and
    /// "empty string" be treated identically. ``huggingface_hub``'s
    /// own resolver (and ``BundledModel.userHFCacheURL`` which mirrors
    /// it) falls through to the next precedence tier on empty values.
    /// If the spawn helper forwarded ``HF_HOME=""`` verbatim, the
    /// child would still take the empty-string branch as "set" and
    /// skip the same fallback the launcher used — re-introducing the
    /// #277 desync from a different angle (codex #279 r1).
    ///
    /// Limited to the three cache-root keys because that is where the
    /// asymmetry is provable from the launcher source. Behavior knobs
    /// like ``HF_ENDPOINT`` / ``HF_HUB_OFFLINE`` are forwarded
    /// verbatim to match ``DownloadManager.augmentedEnv``'s full
    /// passthrough shape; the child decides how to interpret an
    /// empty value there.
    nonisolated internal static let serveEnvironmentDropIfEmpty: Set<String> = [
        "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME",
    ]

    /// Pure builder for the FULL env handed to ``ProcessGroupChild.spawn``
    /// for the ``rapid-mlx serve`` child. Despite the historical
    /// "additions" name (#271 landed the helper, #272 turned it into
    /// the SSOT) the result is the COMPLETE env: the spawn call site
    /// uses ``replaceEnvironment: true`` so nothing else is inherited.
    ///
    /// Three layers, applied in order (later layers override earlier):
    ///   1. Allowlisted subset of ``ambient`` (see
    ///      ``serveEnvironmentAllowlist``).
    ///   2. Desktop-injected ``RAPID_MLX_API_KEY`` + ``PYTHONUNBUFFERED``
    ///      + ``HF_HUB_DISABLE_PROGRESS_BARS`` +
    ///      ``RAPID_MLX_WATCHDOG_PPID`` (issue #449).
    ///   3. ``HF_HUB_DISABLE_XET`` / ``HF_HUB_DOWNLOAD_TIMEOUT``
    ///      with ambient pass-through (operator override channel).
    ///
    /// Empty ``bearer`` is treated as "no bearer to inject" — we don't
    /// ship a sentinel value the child could misread as a real key.
    /// ``ambient`` is an explicit parameter (rather than read inside
    /// the helper) so tests can drive the allowlist + override
    /// pass-through deterministically without mutating
    /// ``ProcessInfo.processInfo.environment``.
    ///
    /// ``supervisorPID`` is the PID stamped into
    /// ``RAPID_MLX_WATCHDOG_PPID`` so the sidecar's parent-PID
    /// watchdog (vllm-mlx PR #942) can self-terminate the moment
    /// the desktop process dies under SIGKILL (the kernel cannot
    /// run our atexit handler in that path). Issue #449: Persona 3
    /// (Sam) saw a 32 GB RSS orphan still serving on port 8001 after
    /// a forced ``kill -9`` on the Rapid binary; the watchdog inside
    /// the bundled rapid-mlx checks ``os.getppid()`` every 2 s and
    /// shuts down cleanly when the live PPID stops matching this
    /// stamp. Passed as a parameter (rather than read inside the
    /// helper) so tests can pin the env contract deterministically.
    /// ``-1`` (the default) is the "do not stamp" sentinel — kept as
    /// the default so existing allowlist / env-pinning tests don't
    /// have to thread the watchdog PID through every call site, while
    /// the production call site below explicitly passes the launcher's
    /// PID. A separate source-pinned contract test
    /// (``WatchdogSpawnEnvTests``) asserts the prod call uses
    /// ``ProcessInfo.processInfo.processIdentifier`` so a future
    /// refactor that drops the explicit pass cannot silently
    /// re-introduce the orphan-sidecar bug.
    ///
    /// ``modelsFolderOverride`` is the desktop "Models folder"
    /// preference (issue #503): the absolute path of the folder the
    /// user pointed Rapid at, or ``nil`` for the default location. When
    /// non-nil/non-empty it is injected as ``HF_HUB_CACHE`` in Layer 2
    /// so the engine downloads/loads there — OVERRIDING any ambient
    /// cache-root var (the desktop owns this choice; a stray
    /// ``HF_HUB_CACHE`` from the launch shell must not win over the
    /// user's explicit Settings pick). Passed as a parameter (rather
    /// than read from ``UserDefaults`` inside this nonisolated static)
    /// so the injection contract stays deterministically testable and
    /// the existence check lives at the call site, which falls the
    /// value back to ``nil`` when the drive is unplugged.
    nonisolated internal static func serveEnvironmentAdditions(
        bearer: String,
        ambient: [String: String],
        physicalRAMBytes: UInt64 = 0,
        availableRAMBytes: UInt64 = 0,
        supervisorPID: Int32 = -1,
        modelsFolderOverride: String? = nil
    ) -> [String: String] {
        // Layer 1: allowlisted ambient. The cache-root keys
        // (``HF_HOME`` / ``HF_HUB_CACHE`` / ``XDG_CACHE_HOME``)
        // additionally drop empty values so the child mirrors the
        // launcher's ``BundledModel.userHFCacheURL`` precedence-
        // fallthrough on ``HF_HOME=""`` (codex #279 r1).
        var env: [String: String] = [:]
        for key in serveEnvironmentAllowlist {
            guard let value = ambient[key] else { continue }
            if value.isEmpty, serveEnvironmentDropIfEmpty.contains(key) {
                continue
            }
            env[key] = value
        }

        // Layer 2: desktop-injected, always.
        if !bearer.isEmpty {
            env["RAPID_MLX_API_KEY"] = bearer
        }
        // Issue #1412: rapid-mlx defaults the prefix cache to 20% of
        // available RAM, which is appropriate for a dedicated server but
        // can push a 16/32 GB desktop Mac into swap while the user is also
        // running a browser or IDE. Reserve at most 8% of physical RAM for the
        // desktop sidecar, never exceed 20% of currently available RAM, and
        // cap large machines at 4 GiB. This is a direct Layer-2 write: an
        // ambient shell export must not defeat the app's memory-pressure
        // policy for the child it owns.
        if let prefixCacheMaxBytes = desktopPrefixCacheMaxBytes(
            physicalRAMBytes: physicalRAMBytes,
            availableRAMBytes: availableRAMBytes
        ) {
            env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"] = String(prefixCacheMaxBytes)
        }
        // Issue #449: stamp the launcher's PID into
        // ``RAPID_MLX_WATCHDOG_PPID`` so the sidecar's parent-PID
        // watchdog (vllm-mlx PR #942) can detect a SIGKILL of the
        // desktop process and self-terminate. Without this, a forced
        // ``kill -9`` on Rapid (or an OS-level OOM kill / panic)
        // would leave the bundled rapid-mlx running indefinitely
        // under launchd (PID 1), holding 20-30 GB of model weights
        // and the loopback port the NEXT launch needs to bind.
        // PortSweep (PR #170) reaps detected orphans only on the
        // NEXT launch — the watchdog is what makes the reap happen
        // mid-session.
        //
        // Direct write (``env[...] = ...``) — never ``setdefault``-
        // equivalent. The rapid-mlx side intentionally lets the env
        // var override the CLI flag when both are present, so a
        // stale launcher PID inherited from a developer-set shell
        // export would mis-target the watchdog if we let it through.
        // The launcher OWNS the parent-child relationship; whatever
        // the operator's shell had set is irrelevant to the sidecar
        // we just spawned.
        //
        // ``supervisorPID > 0`` gate matches the rapid-mlx helper's
        // ``ppid <= 1`` early-out (PID 1 / 0 / negative is the
        // sentinel for "no real parent to watch"). Tests pass ``-1``
        // when they want to assert the legacy non-stamping shape.
        if supervisorPID > 1 {
            env["RAPID_MLX_WATCHDOG_PPID"] = String(supervisorPID)
        }
        // Force Python to flush stdout/stderr line-by-line so
        // huggingface_hub's tqdm progress bars reach our readability
        // handler without sitting in the libc block-buffer until the
        // next 4 KiB flush. Without this the first-time-download
        // overlay can sit on the same "Downloading N/M files" line
        // for minutes while bytes are actually moving (#150).
        env["PYTHONUNBUFFERED"] = "1"
        // Disable HF Hub's "I'm in a non-TTY, hide bars" code path.
        // Defensive — the env var defaults to "0" (bars enabled), but
        // a stray ambient export from the user's shell could otherwise
        // silently mute the entire download UX. The allowlist already
        // drops it from ambient, but pin it explicitly so future
        // allowlist edits can't accidentally re-expose the bug.
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

        // Issue #503: the desktop "Models folder" preference. When the
        // user has pointed Rapid at an explicit folder (already
        // validated to exist by the call site), inject it as
        // ``HF_HUB_CACHE`` so the engine downloads + loads there. Direct
        // write in Layer 2 — never a ``setdefault`` — so it OVERRIDES a
        // stray ambient ``HF_HUB_CACHE`` that survived the allowlist:
        // the user's Settings choice is authoritative over whatever the
        // launch shell happened to export. ``nil`` / empty leaves the
        // env untouched so the engine keeps its default location (and
        // the drop-if-empty behaviour of the allowlist is preserved).
        if let modelsFolderOverride, !modelsFolderOverride.isEmpty {
            env["HF_HUB_CACHE"] = modelsFolderOverride
        }

        // Layer 3: HF Hub overrides with ambient pass-through.
        //
        // huggingface_hub 1.19 selects the hf_xet 1.5.1
        // chunked-download client by default. Diagnosed
        // 2026-06-16 against a v0.7.0 install on a 2.3 MB/s
        // residential link: raw curl through the Xet bridge
        // CDNs pulled 200 MB steadily; the Python client on
        // the same socket stopped at ~6 MB transferred and
        // sent 0 bytes over the next 60 s with no error,
        // leaving the chat composer dead at "Downloading
        // 0/N files" forever. Setting HF_HUB_DISABLE_XET=1
        // routes the downloader back to plain HTTPS range-
        // GETs against the same CDN, which completes. The
        // ${VAR:-default} pattern preserves the operator
        // override: a user with a working Xet config can
        // launchctl-setenv HF_HUB_DISABLE_XET=0 and we'll
        // forward that through. HF_HUB_DOWNLOAD_TIMEOUT=300
        // is cheap insurance for slow links Xet historically
        // masked behind chunked retries.
        env["HF_HUB_DISABLE_XET"] = ambient["HF_HUB_DISABLE_XET"] ?? "1"
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = ambient["HF_HUB_DOWNLOAD_TIMEOUT"] ?? "300"

        return env
    }

    /// Desktop-specific prefix-cache budget: the smallest of 8% physical RAM,
    /// 20% currently available RAM, and 4 GiB. The available-RAM clamp keeps
    /// this override from raising the engine's ordinary 20%-of-available
    /// budget on a machine that is already under pressure. If either probe
    /// fails, omit the override and retain the engine's live available-memory
    /// fallback instead of manufacturing an unsafe fixed ceiling. Kept pure so
    /// the spawn contract is deterministic in tests.
    nonisolated internal static func desktopPrefixCacheMaxBytes(
        physicalRAMBytes: UInt64,
        availableRAMBytes: UInt64 = 0
    ) -> UInt64? {
        guard physicalRAMBytes > 0, availableRAMBytes > 0 else { return nil }
        let fourGiB = UInt64(4) << 30
        let eightPercent = (physicalRAMBytes / 100) * 8
        return min(min(eightPercent, availableRAMBytes / 5), fourGiB)
    }
}

/// Minimal POSIX-spawn wrapper for children that must own a process
/// group. Foundation `Process` does not expose `posix_spawnattr`, so
/// `ServerManager` uses this for `rapid-mlx serve` and keeps
/// `DownloadManager` on `Process` because downloads do not fork a live
/// server tree.
final class ProcessGroupChild: @unchecked Sendable {
    let processIdentifier: pid_t
    let processGroupID: pid_t

    /// codex r2 NIT: ``testStub()`` previously used a high PID (99 999)
    /// hoping it would not collide with a real process group on the
    /// test host. High-PID-density CI containers can violate that
    /// assumption — ``kill(-PID, 0)`` returns 0, ``isProcessGroupAlive``
    /// returns true, and tests that fall into ``terminateChild`` end
    /// up signalling an unrelated process group and blocking on the
    /// 5 s SIGTERM + 1 s SIGKILL grace windows. This flag short-
    /// circuits both surfaces — stubs are always "not alive" and any
    /// ``signalProcessGroup`` call becomes a no-op.
    let isStub: Bool

    private let lock = NSLock()
    private var running: Bool = true
    private var monitorStarted: Bool = false
    private var rawTerminationStatus: Int32 = 0
    private var rawTerminationReason: Process.TerminationReason = .exit
    private var terminationHandler: (@Sendable (ProcessGroupChild) -> Void)?

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }

    var terminationStatus: Int32 {
        lock.lock()
        defer { lock.unlock() }
        return rawTerminationStatus
    }

    var terminationReason: Process.TerminationReason {
        lock.lock()
        defer { lock.unlock() }
        return rawTerminationReason
    }

    var isProcessGroupAlive: Bool {
        if isStub { return false }
        if kill(-processGroupID, 0) == 0 { return true }
        return errno == EPERM
    }

    private init(
        processIdentifier: pid_t,
        terminationHandler: (@Sendable (ProcessGroupChild) -> Void)?,
        isStub: Bool = false
    ) {
        self.processIdentifier = processIdentifier
        self.processGroupID = processIdentifier
        self.terminationHandler = terminationHandler
        self.isStub = isStub
    }

    /// codex r1 BLOCKING #3 + codex r2 NIT: ``runRuntimeHealthLoop``
    /// needs an identity-comparable ``ProcessGroupChild`` instance
    /// to drive its replace-start guard. ``init`` is private so
    /// production code can't accidentally bypass ``spawn``; this
    /// internal factory exists for unit tests only.
    ///
    /// ``isStub: true`` short-circuits ``isProcessGroupAlive``
    /// (returns ``false``) AND any ``signalProcessGroup(_:)`` call
    /// (no-op). High-PID-density CI containers can have a real
    /// process group at any PID; we sidestep that hazard entirely
    /// by gating both surfaces explicitly rather than hoping the
    /// PID doesn't collide.
    internal static func testStub() -> ProcessGroupChild {
        ProcessGroupChild(processIdentifier: 1, terminationHandler: nil, isStub: true)
    }

    @discardableResult
    static func spawn(
        executableURL: URL,
        arguments: [String],
        standardInput: FileHandle,
        standardOutput: Pipe,
        standardError: Pipe,
        environmentAdditions: [String: String] = [:],
        replaceEnvironment: Bool = false,
        startMonitorImmediately: Bool = true,
        terminationHandler: (@Sendable (ProcessGroupChild) -> Void)? = nil
    ) throws -> ProcessGroupChild {
        var attributes: posix_spawnattr_t?
        var fileActions: posix_spawn_file_actions_t?
        try check(posix_spawnattr_init(&attributes), operation: "posix_spawnattr_init")
        defer { posix_spawnattr_destroy(&attributes) }
        try check(posix_spawn_file_actions_init(&fileActions), operation: "posix_spawn_file_actions_init")
        defer { posix_spawn_file_actions_destroy(&fileActions) }

        try check(
            posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP)),
            operation: "posix_spawnattr_setflags"
        )
        try check(
            posix_spawnattr_setpgroup(&attributes, 0),
            operation: "posix_spawnattr_setpgroup"
        )

        let providedStdinFD = standardInput.fileDescriptor
        let openedNullFD: Int32?
        let stdinFD: Int32
        if providedStdinFD >= 0 {
            openedNullFD = nil
            stdinFD = providedStdinFD
        } else {
            let fd = open("/dev/null", O_RDONLY)
            guard fd >= 0 else {
                throw ProcessGroupSpawnError(operation: "open /dev/null", code: errno)
            }
            openedNullFD = fd
            stdinFD = fd
        }
        defer {
            if let openedNullFD {
                close(openedNullFD)
            }
        }
        let stdoutReadFD = standardOutput.fileHandleForReading.fileDescriptor
        let stdoutWriteFD = standardOutput.fileHandleForWriting.fileDescriptor
        let stderrReadFD = standardError.fileHandleForReading.fileDescriptor
        let stderrWriteFD = standardError.fileHandleForWriting.fileDescriptor

        try dup(fd: stdinFD, to: STDIN_FILENO, actions: &fileActions)
        try dup(fd: stdoutWriteFD, to: STDOUT_FILENO, actions: &fileActions)
        try dup(fd: stderrWriteFD, to: STDERR_FILENO, actions: &fileActions)
        try closeInChild(fd: stdoutReadFD, actions: &fileActions)
        try closeInChild(fd: stderrReadFD, actions: &fileActions)
        try closeInChild(fd: stdinFD, actions: &fileActions)
        try closeInChild(fd: stdoutWriteFD, actions: &fileActions)
        try closeInChild(fd: stderrWriteFD, actions: &fileActions)

        let argv = [executableURL.path] + arguments
        // Build the child's env. ``replaceEnvironment: true`` (issue
        // #272) starts from an EMPTY base and writes only the supplied
        // additions, so the launcher's full env — including
        // third-party secrets a user may have exported in the shell
        // they launched the desktop from (``ANTHROPIC_API_KEY``,
        // ``BRAVE_API_KEY``, ``GH_TOKEN`` etc.) — does NOT reach the
        // child. The ``rapid-mlx serve`` spawn site uses this mode
        // and pre-filters the desired ambient subset via
        // ``ServerManager.serveEnvironmentAdditions``.
        //
        // Default (``replaceEnvironment: false``) preserves the
        // historical merge-over-inherited shape for any caller that
        // genuinely needs full env passthrough (issue #17: bearer
        // injection still works because the addition wins last).
        var merged = replaceEnvironment ? [:] : ProcessInfo.processInfo.environment
        for (k, v) in environmentAdditions {
            merged[k] = v
        }
        let envp = merged
            .map { "\($0.key)=\($0.value)" }
            .sorted()

        var pid: pid_t = 0
        let result = argv.withCStringArray { argvPointer in
            envp.withCStringArray { envPointer in
                executableURL.path.withCString { executablePath in
                    posix_spawn(
                        &pid,
                        executablePath,
                        &fileActions,
                        &attributes,
                        argvPointer,
                        envPointer
                    )
                }
            }
        }
        try check(result, operation: "posix_spawn")

        standardOutput.fileHandleForWriting.closeFile()
        standardError.fileHandleForWriting.closeFile()

        let child = ProcessGroupChild(
            processIdentifier: pid,
            terminationHandler: terminationHandler
        )
        if startMonitorImmediately {
            child.startMonitor()
        }
        return child
    }

    func signalProcessGroup(_ signal: Int32) {
        if isStub { return }
        if kill(-processGroupID, signal) != 0 && errno != ESRCH {
            // Best effort: callers still wait / escalate based on
            // `isProcessGroupAlive`, so there is nothing actionable to
            // throw from a shutdown path.
        }
    }

    static func reapProcessGroupInBackground(processGroupID: pid_t) {
        DispatchQueue.global(qos: .utility).async {
            let deadline = Date().addingTimeInterval(5.0)
            while Date() < deadline {
                if kill(-processGroupID, 0) != 0 && errno == ESRCH {
                    return
                }
                Thread.sleep(forTimeInterval: 0.1)
            }
            if kill(-processGroupID, SIGKILL) != 0 && errno != ESRCH {
                return
            }
        }
    }

    func startMonitor() {
        lock.lock()
        guard !monitorStarted else {
            lock.unlock()
            return
        }
        monitorStarted = true
        lock.unlock()
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            var waitStatus: Int32 = 0
            let waited = waitpid(self.processIdentifier, &waitStatus, 0)
            guard waited == self.processIdentifier else { return }
            let decoded = Self.decode(waitStatus: waitStatus)
            self.lock.lock()
            self.running = false
            self.rawTerminationStatus = decoded.status
            self.rawTerminationReason = decoded.reason
            let handler = self.terminationHandler
            self.lock.unlock()
            handler?(self)
        }
    }

    private static func dup(
        fd: Int32,
        to target: Int32,
        actions: inout posix_spawn_file_actions_t?
    ) throws {
        guard fd != target else { return }
        try check(
            posix_spawn_file_actions_adddup2(&actions, fd, target),
            operation: "posix_spawn_file_actions_adddup2"
        )
    }

    private static func closeInChild(
        fd: Int32,
        actions: inout posix_spawn_file_actions_t?
    ) throws {
        guard fd > STDERR_FILENO else { return }
        try check(
            posix_spawn_file_actions_addclose(&actions, fd),
            operation: "posix_spawn_file_actions_addclose"
        )
    }

    private static func decode(waitStatus: Int32) -> (
        reason: Process.TerminationReason,
        status: Int32
    ) {
        let statusCode = waitStatus & 0x7f
        if statusCode == 0 {
            return (.exit, (waitStatus >> 8) & 0xff)
        }
        if statusCode != 0x7f {
            return (.uncaughtSignal, statusCode)
        }
        return (.exit, waitStatus)
    }

    private static func check(_ result: Int32, operation: String) throws {
        guard result == 0 else {
            throw ProcessGroupSpawnError(operation: operation, code: result)
        }
    }
}

private struct ProcessGroupSpawnError: LocalizedError {
    let operation: String
    let code: Int32

    var errorDescription: String? {
        let message = String(cString: strerror(code))
        return "\(operation) failed: \(message)"
    }
}

private extension Array where Element == String {
    func withCStringArray<Result>(
        _ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) throws -> Result
    ) rethrows -> Result {
        let cStrings = map { strdup($0) }
        defer {
            for pointer in cStrings {
                free(pointer)
            }
        }

        let pointer = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: count + 1)
        defer { pointer.deallocate() }
        for index in indices {
            pointer[index] = cStrings[index]
        }
        pointer[count] = nil
        return try body(pointer)
    }
}
