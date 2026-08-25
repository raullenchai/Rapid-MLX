import Foundation
import Observation
import Testing

@testable import Rapid

@Suite("Launch auto-start memory guard")
struct LaunchAutoStartMemoryTests {
    @MainActor
    @Test("unsafe launch resume defers silently, explicit Start still warns")
    func launchResumeDoesNotPresentUnsolicitedWarning() async {
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = {
            MemoryProbe.Snapshot(
                totalBytes: 16 * 1_073_741_824,
                usedBytes: 15 * 1_073_741_824
            )
        }

        let alias = "qwen3-235b-4bit"
        await server.start(alias: alias, isLaunchAutoStart: true)

        #expect(server.state == .idle)
        #expect(server.pendingMemoryWarning == nil)
        #expect(server.servingAlias == nil)

        await server.start(alias: alias)

        #expect(server.pendingMemoryWarning?.alias == alias)
        #expect(server.pendingMemoryWarning?.severity == .unsafe)
        #expect(server.servingAlias == nil)
    }

    @MainActor
    @Test("parked warning re-samples the injected probe in both directions")
    func parkedWarningRefreshesFromInjectedProbe() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = LockedMemorySnapshots(
            .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.current }

        await server.start(alias: "qwen3.5-9b-4bit")
        let originalID = try #require(server.pendingMemoryWarning?.id)
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 16 * gib)
        let becameTight = await server.refreshPendingMemoryWarning()
        #expect(becameTight?.old == .unsafe)
        #expect(becameTight?.new == .tight)
        #expect(server.pendingMemoryWarning?.confirmTitle == "Load model")

        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        let becameSafe = await server.refreshPendingMemoryWarning()
        #expect(becameSafe?.old == .tight)
        #expect(becameSafe?.new == .safe)
        #expect(server.pendingMemoryWarning?.id == originalID)
        #expect(server.pendingMemoryWarning?.confirmTitle == "Load model")

        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        let becameUnsafe = await server.refreshPendingMemoryWarning()
        #expect(becameUnsafe?.old == .safe)
        #expect(becameUnsafe?.new == .unsafe)
        #expect(server.pendingMemoryWarning?.id == originalID)
    }

    @MainActor
    @Test("a newly-safe Load action rechecks memory at activation")
    func safeActionDoesNotBecomeAStaleBypass() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = LockedMemorySnapshots(
            .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.current }

        await server.start(alias: "qwen3.5-9b-4bit")
        let originalID = try #require(server.pendingMemoryWarning?.id)
        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        _ = await server.refreshPendingMemoryWarning()
        let safeWarning = try #require(server.pendingMemoryWarning)
        #expect(safeWarning.severity == .safe)

        // Pressure returns after the last visible sample but before the user
        // clicks. The ordinary Load action must not carry a stale waiver.
        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        server.confirmPendingMemoryLoad(safeWarning)
        for _ in 0 ..< 100 where server.pendingMemoryWarning?.severity != .unsafe {
            try await Task.sleep(for: .milliseconds(10))
        }
        let rechecked = try #require(server.pendingMemoryWarning)
        #expect(rechecked.severity == .unsafe)
        #expect(rechecked.id == originalID)
        #expect(server.state == .idle)
    }

    @MainActor
    @Test("an older overlapping sample cannot overwrite a newer result")
    func overlappingRefreshesKeepNewestResult() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = OrderedMemorySnapshots(
            initial: .init(totalBytes: 32 * gib, usedBytes: 30 * gib),
            delayed: .init(totalBytes: 32 * gib, usedBytes: 16 * gib),
            newest: .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.next() }

        await server.start(alias: "qwen3.5-9b-4bit")
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        let older = Task { await server.refreshPendingMemoryWarning() }
        await snapshots.waitForDelayedSample()
        let newer = Task { await server.refreshPendingMemoryWarning() }
        let newerTransition = await newer.value
        snapshots.releaseDelayedSample()
        let olderTransition = await older.value

        #expect(newerTransition?.new == .safe)
        #expect(olderTransition == nil)
        #expect(server.pendingMemoryWarning?.severity == .safe)
    }

    @MainActor
    @Test("activation invalidates an older periodic memory sample")
    func activationInvalidatesOlderPeriodicSample() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = ActivationRaceMemorySnapshots(
            unsafe: .init(totalBytes: 32 * gib, usedBytes: 30 * gib),
            safe: .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.next() }

        await server.start(alias: "qwen3.5-9b-4bit")
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        _ = await server.refreshPendingMemoryWarning()
        let safeWarning = try #require(server.pendingMemoryWarning)
        #expect(safeWarning.severity == .safe)

        let olderPeriodicRefresh = Task {
            await server.refreshPendingMemoryWarning()
        }
        await snapshots.waitForPeriodicSample()

        server.confirmPendingMemoryLoad(safeWarning)
        for _ in 0 ..< 100 where server.pendingMemoryWarning?.severity != .unsafe {
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        snapshots.releasePeriodicSample()
        let olderTransition = await olderPeriodicRefresh.value

        #expect(olderTransition == nil)
        #expect(server.pendingMemoryWarning?.id == safeWarning.id)
        #expect(server.pendingMemoryWarning?.severity == .unsafe)
        #expect(server.state == .idle)
    }

    @MainActor
    @Test("a cancelled view-bound refresh cannot update the parked warning")
    func cancelledRefreshDoesNotApplyItsSample() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = OrderedMemorySnapshots(
            initial: .init(totalBytes: 32 * gib, usedBytes: 30 * gib),
            delayed: .init(totalBytes: 32 * gib, usedBytes: 2 * gib),
            newest: .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.next() }

        await server.start(alias: "qwen3.5-9b-4bit")
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        let refresh = Task { await server.refreshPendingMemoryWarning() }
        await snapshots.waitForDelayedSample()
        refresh.cancel()
        snapshots.releaseDelayedSample()

        #expect(await refresh.value == nil)
        #expect(server.pendingMemoryWarning?.severity == .unsafe)
    }

    @MainActor
    @Test("confirm survives SwiftUI's same-turn alert dismissal")
    func confirmedWarningIsNotCancelledByAlertDismissal() async throws {
        let gib = UInt64(1_073_741_824)
        let snapshots = BlockingActivationSnapshots(
            .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        let completion = LockedCompletionFlag()
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.next() }

        let load = Task {
            _ = await server.ensureServing(alias: "qwen3.5-9b-4bit")
            completion.markComplete()
        }
        while server.pendingMemoryWarning == nil {
            await Task.yield()
        }
        let warning = try #require(server.pendingMemoryWarning)

        server.confirmPendingMemoryLoad(warning)
        await snapshots.waitForActivationSample()
        // SwiftUI writes false to the alert Binding after its button action.
        // That same-turn dismissal must not reverse an accepted decision.
        server.cancelPendingMemoryLoad(warning)
        try await Task.sleep(for: .milliseconds(350))
        #expect(!completion.isComplete)

        snapshots.releaseActivationSample()
        _ = await load.value
    }

    @MainActor
    @Test("a live memory refresh invalidates the view-bound pendingMemoryWarning (observation fires)")
    func liveRefreshInvalidatesPendingMemoryWarningObservation() async throws {
        // ONBOARD-MEM-LIVE regression pin. The queue used to be a plain value type
        // stored in the @Observable ServerManager, so refreshCurrentWarning's in-place
        // `pending[0].warning` replacement never fired withMutation — the "Before
        // loading" verdict stayed frozen on its original snapshot even as free memory
        // changed. With MemoryLoadConfirmationQueue now an @Observable reference type,
        // replacing the head warning MUST invalidate the exact property the card binds:
        // server.pendingMemoryWarning. That is what makes red-to-green (and green-to-red)
        // actually re-render rather than update an invisible value.
        let gib = UInt64(1_073_741_824)
        let snapshots = LockedMemorySnapshots(
            .init(totalBytes: 32 * gib, usedBytes: 30 * gib)
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.memorySnapshotProvider = { snapshots.current }

        await server.start(alias: "qwen3.5-9b-4bit")
        #expect(server.pendingMemoryWarning?.severity == .unsafe)

        let changed = LockedCompletionFlag()
        withObservationTracking {
            _ = server.pendingMemoryWarning
        } onChange: {
            changed.markComplete()
        }

        snapshots.current = .init(totalBytes: 32 * gib, usedBytes: 2 * gib)
        _ = await server.refreshPendingMemoryWarning()
        #expect(server.pendingMemoryWarning?.severity == .safe)
        #expect(
            changed.isComplete,
            "replacing the parked warning must invalidate server.pendingMemoryWarning so the onboarding verdict re-renders live"
        )
    }
}

private final class LockedMemorySnapshots: @unchecked Sendable {
    private let lock = NSLock()
    private var value: MemoryProbe.Snapshot

    init(_ value: MemoryProbe.Snapshot) {
        self.value = value
    }

    var current: MemoryProbe.Snapshot {
        get { lock.withLock { value } }
        set { lock.withLock { value = newValue } }
    }
}

private final class OrderedMemorySnapshots: @unchecked Sendable {
    private let lock = NSLock()
    private let delayedRelease = DispatchSemaphore(value: 0)
    private let initial: MemoryProbe.Snapshot
    private let delayed: MemoryProbe.Snapshot
    private let newest: MemoryProbe.Snapshot
    private var callCount = 0
    private var delayedSampleStarted = false

    init(
        initial: MemoryProbe.Snapshot,
        delayed: MemoryProbe.Snapshot,
        newest: MemoryProbe.Snapshot
    ) {
        self.initial = initial
        self.delayed = delayed
        self.newest = newest
    }

    func next() -> MemoryProbe.Snapshot {
        let call = lock.withLock {
            defer { callCount += 1 }
            if callCount == 1 {
                delayedSampleStarted = true
            }
            return callCount
        }
        switch call {
        case 0:
            return initial
        case 1:
            delayedRelease.wait()
            return delayed
        default:
            return newest
        }
    }

    func waitForDelayedSample() async {
        while !lock.withLock({ delayedSampleStarted }) {
            await Task.yield()
        }
    }

    func releaseDelayedSample() {
        delayedRelease.signal()
    }
}

private final class ActivationRaceMemorySnapshots: @unchecked Sendable {
    private let lock = NSLock()
    private let periodicRelease = DispatchSemaphore(value: 0)
    private let unsafe: MemoryProbe.Snapshot
    private let safe: MemoryProbe.Snapshot
    private var callCount = 0
    private var periodicSampleStarted = false

    init(unsafe: MemoryProbe.Snapshot, safe: MemoryProbe.Snapshot) {
        self.unsafe = unsafe
        self.safe = safe
    }

    func next() -> MemoryProbe.Snapshot {
        let call = lock.withLock {
            defer { callCount += 1 }
            if callCount == 2 { periodicSampleStarted = true }
            return callCount
        }
        switch call {
        case 0, 3:
            return unsafe
        case 2:
            periodicRelease.wait()
            return safe
        default:
            return safe
        }
    }

    func waitForPeriodicSample() async {
        while !lock.withLock({ periodicSampleStarted }) {
            await Task.yield()
        }
    }

    func releasePeriodicSample() {
        periodicRelease.signal()
    }
}

private final class BlockingActivationSnapshots: @unchecked Sendable {
    private let lock = NSLock()
    private let activationRelease = DispatchSemaphore(value: 0)
    private let value: MemoryProbe.Snapshot
    private var callCount = 0
    private var activationStarted = false

    init(_ value: MemoryProbe.Snapshot) {
        self.value = value
    }

    func next() -> MemoryProbe.Snapshot {
        let call = lock.withLock {
            defer { callCount += 1 }
            if callCount == 1 { activationStarted = true }
            return callCount
        }
        if call == 1 { activationRelease.wait() }
        return value
    }

    func waitForActivationSample() async {
        while !lock.withLock({ activationStarted }) {
            await Task.yield()
        }
    }

    func releaseActivationSample() {
        activationRelease.signal()
    }
}

private final class LockedCompletionFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var complete = false

    var isComplete: Bool { lock.withLock { complete } }
    func markComplete() { lock.withLock { complete = true } }
}
