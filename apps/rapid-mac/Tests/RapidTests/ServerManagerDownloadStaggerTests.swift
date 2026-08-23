import Foundation
import Testing
@testable import Rapid

/// rapid-desktop issue #253 — the desktop GUI could spawn
/// ``rapid-mlx pull <alias>`` (via ``DownloadManager``) and
/// ``rapid-mlx serve <alias>`` (via ``ServerManager``) concurrently
/// for the same alias. With Rapid-MLX 0.7.27+ both subprocesses run
/// their own mirror code, double-writing the HF cache (snapshot dir +
/// orphan blob) and burning 2× disk + 2× bandwidth + 272 s extra on
/// the cold start the user is waiting on.
///
/// The fix wires ``ServerManager`` to ``DownloadManager`` and gates
/// ``start(alias:)`` on ``awaitDownloadSettlement`` so the serve spawn
/// staggers behind any in-flight background pull for the same alias.
/// These tests pin both halves of that contract.
@MainActor
@Suite("ServerManager — #253 stagger behind in-flight DownloadManager pull")
struct ServerManagerDownloadStaggerTests {
    private let alias = "qwen3.6-27b-4bit"

    private func waitUntil(
        deadline: Date,
        predicate: () -> Bool
    ) async -> Bool {
        while Date() < deadline {
            if predicate() { return true }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        return predicate()
    }

    @Test("awaitDownloadSettlement returns immediately when no job exists")
    func settlementNoOpsWithoutJob() async {
        let probe = SettlementSleepProbe()
        let downloads = DownloadManager(settlementSleep: probe.sleep)
        await downloads.awaitDownloadSettlement(alias: alias)
        #expect(probe.callCount == 0, "a settled alias must not enter the polling sleep")
        #expect(!downloads.isDownloading(alias))
    }

    @Test("awaitDownloadSettlement returns immediately when job already finished")
    func settlementNoOpsAfterTerminalStatus() async {
        let probe = SettlementSleepProbe()
        let downloads = DownloadManager(settlementSleep: probe.sleep)
        _ = downloads._testingSeedJob(alias: alias)
        downloads._testingFinish(alias: alias, status: 0, reason: .exit)
        await downloads.awaitDownloadSettlement(alias: alias)
        #expect(probe.callCount == 0, "a terminal job must not enter the polling sleep")
        #expect(!downloads.isDownloading(alias))
    }

    @Test("awaitDownloadSettlement suspends while running, returns after .completed")
    func settlementSuspendsWhileRunning() async {
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: alias)
        #expect(downloads.isDownloading(alias))

        async let waiter: Void = downloads.awaitDownloadSettlement(alias: alias)

        // Give the waiter a couple of polling cycles before settling.
        try? await Task.sleep(nanoseconds: 400_000_000)
        #expect(downloads.isDownloading(alias))

        downloads._testingFinish(alias: alias, status: 0, reason: .exit)
        let settled = await waitUntil(deadline: Date().addingTimeInterval(2)) {
            !downloads.isDownloading(alias)
        }
        #expect(settled)
        await waiter
    }

    @Test("awaitDownloadSettlement also unblocks on .cancelled")
    func settlementUnblocksOnCancel() async {
        let downloads = DownloadManager()
        _ = downloads._testingSeedJob(alias: alias)
        async let waiter: Void = downloads.awaitDownloadSettlement(alias: alias)
        try? await Task.sleep(nanoseconds: 350_000_000)
        downloads._testingFinish(
            alias: alias,
            status: 9,
            reason: .uncaughtSignal,
            wasCancelling: true
        )
        await waiter
        #expect(!downloads.isDownloading(alias))
    }

    @Test("attached DownloadManager is held weakly — release post-attach drops the ref")
    func attachedDownloadsHeldWeakly() {
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/opt/homebrew/bin/rapid-mlx")
        )
        weak var weakDownloads: DownloadManager?
        do {
            let downloads = DownloadManager()
            weakDownloads = downloads
            server.attachDownloads(downloads)
            #expect(weakDownloads != nil)
        }
        // The DownloadManager's only strong reference went out of
        // scope; ServerManager's weak handle must not pin it alive.
        // Pinning would tie the manager's lifetime to the
        // ServerManager singleton, which outlives any reasonable
        // teardown harness and breaks ARC-based test cleanup.
        #expect(weakDownloads == nil)
    }

    @Test("awaitDownloadSettlement returns promptly on Task cancellation (no busy-loop)")
    func settlementHonorsTaskCancellation() async {
        // codex r1 BLOCKING: the previous shape used
        // ``try? await Task.sleep(...)`` which swallows
        // ``CancellationError`` and re-enters the ``while
        // isDownloading`` check immediately. On cancellation the loop
        // becomes a tight MainActor busy-poll that freezes the UI
        // until the pull settles. The fix returns out of the loop on
        // cancellation so the start ``Task`` can unwind cleanly.
        let probe = SettlementSleepProbe()
        let downloads = DownloadManager(settlementSleep: probe.sleep)
        _ = downloads._testingSeedJob(alias: alias)
        #expect(downloads.isDownloading(alias))

        let waiter = Task { @MainActor in
            await downloads.awaitDownloadSettlement(alias: alias)
        }
        // Synchronize on the exact polling boundary instead of guessing when
        // a loaded runner has scheduled 350 ms of wall time.
        await probe.waitUntilEntered()
        waiter.cancel()
        await waiter.value
        // The deterministic signal is that the wait returned while the job
        // is STILL running: it exited via cancellation, not settlement. A
        // sub-second wall-clock bound only measures parallel runner load.
        #expect(probe.callCount == 1, "cancellation must not re-enter the polling loop")
        #expect(downloads.isDownloading(alias))
    }
}

@MainActor
private final class SettlementSleepProbe {
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private(set) var callCount = 0

    func waitUntilEntered() async {
        if callCount > 0 { return }
        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func sleep() async throws {
        callCount += 1
        let waiters = enteredWaiters
        enteredWaiters.removeAll()
        waiters.forEach { $0.resume() }
        try await Task.sleep(for: .seconds(60))
    }
}
