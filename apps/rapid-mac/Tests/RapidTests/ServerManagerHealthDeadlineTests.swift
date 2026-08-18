import Foundation
import Testing
@testable import Rapid

/// v0.7.13 fix: ``ServerManager`` previously hard-killed the rapid-mlx
/// child after 30 minutes of wall-clock from launch, regardless of
/// whether a download was making forward progress. On slow user links
/// a 10 GB model takes ~4 hours at 683 KB/s — the deadline fired
/// mid-pull, the partial download was orphaned, and the user's next
/// attempt restarted from zero.
///
/// The new shape: the 30-minute budget is a **stall window** measured
/// against the most recently observed forward-progress signal. A
/// download that is moving — heartbeats, R2 completions, tqdm ticks,
/// disk observations — keeps the loop alive indefinitely. A genuinely
/// stuck child (no progress, no /healthz) still surfaces as
/// ``.crashed`` within the same 30-minute idle window.
///
/// The polling loop itself is intertwined with network I/O, so we
/// pin the decision via the pure helper ``ServerManager
/// .shouldKeepWaitingForHealth(now:lastProgressAt:stallWindow:)``.
@MainActor
@Suite("ServerManager health-deadline stall window (v0.7.13)")
struct ServerManagerHealthDeadlineTests {
    private let window: TimeInterval = 30 * 60

    @Test("Loopback health probes never inherit system or PAC proxies")
    func loopbackHealthProbeIsDirect() {
        let configuration = ServerManager.loopbackHealthSessionConfiguration()
        #expect(configuration.connectionProxyDictionary?.isEmpty == true)
        #expect(configuration.timeoutIntervalForRequest == 1.5)
        #expect(configuration.timeoutIntervalForResource == 1.5)
    }

    /// Boundary 1 — at zero idle time the loop must keep waiting.
    /// ``now == lastProgressAt`` is the launch-instant case, and the
    /// most recently observed-progress case after a heartbeat tick
    /// lands. Either way we are not stalled.
    @Test("Fresh progress observation → keep waiting")
    func keepWaitingOnFreshProgress() {
        let t = Date(timeIntervalSince1970: 1_000_000)
        #expect(ServerManager.shouldKeepWaitingForHealth(
            now: t,
            lastProgressAt: t,
            stallWindow: window
        ))
    }

    /// 29 minutes into a stall is still within the budget. Loop
    /// continues.
    @Test("Idle for 29 min (< stall window) → keep waiting")
    func keepWaitingBelowStallWindow() {
        let t0 = Date(timeIntervalSince1970: 2_000_000)
        let now = t0.addingTimeInterval(29 * 60)
        #expect(ServerManager.shouldKeepWaitingForHealth(
            now: now,
            lastProgressAt: t0,
            stallWindow: window
        ))
    }

    /// At exactly the stall window the loop EXITS — the polling loop
    /// uses ``< stallWindow``, so equality terminates. Asserting the
    /// boundary precisely so a future ``<=`` regression would fail.
    @Test("Idle for exactly 30 min (== stall window) → exit")
    func exitAtStallWindowBoundary() {
        let t0 = Date(timeIntervalSince1970: 3_000_000)
        let now = t0.addingTimeInterval(window)
        #expect(!ServerManager.shouldKeepWaitingForHealth(
            now: now,
            lastProgressAt: t0,
            stallWindow: window
        ))
    }

    /// 31 minutes idle is past the budget; loop must exit so the
    /// caller fires ``terminateChild``.
    @Test("Idle for 31 min (> stall window) → exit")
    func exitPastStallWindow() {
        let t0 = Date(timeIntervalSince1970: 4_000_000)
        let now = t0.addingTimeInterval(31 * 60)
        #expect(!ServerManager.shouldKeepWaitingForHealth(
            now: now,
            lastProgressAt: t0,
            stallWindow: window
        ))
    }

    /// The headline scenario this fix exists to support: a 4-hour
    /// download that ticks every 500 ms keeps the loop alive.
    /// Simulated by advancing ``lastProgressAt`` along with ``now``,
    /// always staying within the window.
    @Test("4-hour slow download with heartbeats every 500 ms → never times out")
    func longDownloadWithProgressNeverTimesOut() {
        var now = Date(timeIntervalSince1970: 5_000_000)
        let target = now.addingTimeInterval(4 * 3600)
        var lastProgressAt = now
        while now < target {
            #expect(ServerManager.shouldKeepWaitingForHealth(
                now: now,
                lastProgressAt: lastProgressAt,
                stallWindow: window
            ))
            now = now.addingTimeInterval(0.5)
            // Heartbeat coincident with the tick — actual production
            // path through ``downloadProgress.lastTickAt``.
            lastProgressAt = now
        }
    }

    /// What if heartbeats stop mid-download? The loop should keep
    /// waiting up to the stall window since the LAST observed
    /// heartbeat — not since launch. Simulate: 3 hours of healthy
    /// ticks, then 29 minutes of silence (still within window) →
    /// keep waiting; then 31 minutes of silence → exit.
    @Test("Heartbeats stop after 3 h → loop tolerates 30 min idle, not more")
    func toleratesIdleAfterLongProgress() {
        let launch = Date(timeIntervalSince1970: 6_000_000)
        let lastHeartbeat = launch.addingTimeInterval(3 * 3600)
        // Still within stall window past the last heartbeat → wait.
        #expect(ServerManager.shouldKeepWaitingForHealth(
            now: lastHeartbeat.addingTimeInterval(29 * 60),
            lastProgressAt: lastHeartbeat,
            stallWindow: window
        ))
        // Past stall window → exit.
        #expect(!ServerManager.shouldKeepWaitingForHealth(
            now: lastHeartbeat.addingTimeInterval(31 * 60),
            lastProgressAt: lastHeartbeat,
            stallWindow: window
        ))
    }
}
