import Foundation
import Testing
@testable import Rapid

@MainActor
final class UpdateTestClock {
    var date = Date(timeIntervalSince1970: 1_000)
    func advance(_ seconds: TimeInterval) { date.addTimeInterval(seconds) }
}

actor UpdateTestFeed {
    var version = "0.16.5"
    var fails = false
    private(set) var calls = 0
    func set(version: String = "0.16.6", fails: Bool = false) {
        self.version = version
        self.fails = fails
    }
    func fetch() throws -> UpdateChecker.Release {
        calls += 1
        if fails { throw URLError(.notConnectedToInternet) }
        return UpdateChecker.Release(
            schemaVersion: 1, version: version, tagName: "v\(version)",
            htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "", dmgURL: nil
        )
    }
}

private actor UpdateTestSleeper {
    private struct SleepState {
        var cancelled = false
        var continuation: CheckedContinuation<Void, Error>?
    }
    private var pending: [UUID: SleepState] = [:]
    private var waits: [(Int, CheckedContinuation<TimeInterval, Never>)] = []
    private var delays: [TimeInterval] = []

    func sleep(_ seconds: TimeInterval) async throws {
        let id = UUID()
        pending[id] = SleepState()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                // Actor isolation makes registration atomic with cancelSleep.
                // Retain early cancellation until this generation registers.
                guard pending[id]?.cancelled == false, !Task.isCancelled else {
                    pending[id] = nil
                    continuation.resume(throwing: CancellationError())
                    return
                }
                pending[id]?.continuation = continuation
                delays.append(seconds)
                let ready = waits.filter { $0.0 <= delays.count }
                waits.removeAll { $0.0 <= delays.count }
                for (number, waiter) in ready { waiter.resume(returning: delays[number - 1]) }
            }
        } onCancel: {
            Task { await self.cancelSleep(id) }
        }
    }

    private func cancelSleep(_ id: UUID) {
        guard let state = pending[id] else { return }
        if let continuation = state.continuation {
            pending[id] = nil
            continuation.resume(throwing: CancellationError())
        } else {
            pending[id]?.cancelled = true
        }
    }

    func waiting(_ number: Int) async -> TimeInterval {
        if delays.count >= number { return delays[number - 1] }
        return await withCheckedContinuation { waits.append((number, $0)) }
    }

    func wake() {
        for (id, state) in pending {
            if let continuation = state.continuation {
                pending[id] = nil
                continuation.resume()
            }
        }
    }
}

@MainActor
@Suite("Update refresh policy")
struct UpdateRefreshPolicyTests {
    @Test("Scheduled refresh discovers a release without relaunch; starts coalesce")
    func scheduledRefresh() async {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(currentVersion: "0.16.5", fetcher: { try await feed.fetch() },
                                    checksEnabled: { true }, now: { clock.date },
                                    sleep: { try await sleeper.sleep($0) })
        async let first: Void = checker.startRefreshing()
        async let second: Void = checker.startRefreshing()
        _ = await (first, second)
        #expect(await sleeper.waiting(1) == UpdateRefreshPolicy.refreshInterval)
        #expect(await feed.calls == 1)
        #expect(checker.availableUpdate == nil)
        await feed.set()
        clock.advance(UpdateRefreshPolicy.refreshInterval + 1)
        await sleeper.wake()
        #expect(await sleeper.waiting(2) == UpdateRefreshPolicy.refreshInterval)
        #expect(checker.availableUpdate?.version == "0.16.6")
        #expect(await feed.calls == 2)
        checker.stopRefreshing()
    }

    @Test("Activation skips fresh checks; forced check fetches; stale activation discovers updates")
    func activationAndForcedCheck() async {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        let checker = UpdateChecker(currentVersion: "0.16.5", fetcher: { try await feed.fetch() },
                                    checksEnabled: { true }, now: { clock.date })
        await checker.check()
        await checker.refreshIfNeeded()
        #expect(await feed.calls == 1)
        await checker.check()
        #expect(await feed.calls == 2)
        await feed.set()
        clock.advance(UpdateRefreshPolicy.refreshInterval + 1)
        await checker.refreshIfNeeded()
        #expect(checker.availableUpdate?.version == "0.16.6")
        #expect(checker.refreshPolicy.lastSuccessAt == clock.date)
    }

    @Test("Initial offline failure retries on schedule and resets cadence after recovery")
    func offlineRecovery() async {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        await feed.set(fails: true)
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(currentVersion: "0.16.5", fetcher: { try await feed.fetch() },
                                    checksEnabled: { true }, now: { clock.date },
                                    sleep: { try await sleeper.sleep($0) })
        checker.startRefreshing()
        #expect(await sleeper.waiting(1) == 60)
        #expect(checker.refreshPolicy.lastSuccessAt == nil)
        clock.advance(59)
        await checker.refreshIfNeeded()
        #expect(await feed.calls == 1)
        clock.advance(1)
        await sleeper.wake()
        #expect(await sleeper.waiting(2) == 120)
        await feed.set()
        clock.advance(120)
        await sleeper.wake()
        #expect(await sleeper.waiting(3) == UpdateRefreshPolicy.refreshInterval)
        #expect(checker.availableUpdate?.version == "0.16.6")
        #expect(checker.refreshPolicy.consecutiveFailures == 0)
        checker.stopRefreshing()
    }

    @Test("Retry doubles and caps at one hour; failure preserves last successful timestamp")
    func retryCap() {
        var policy = UpdateRefreshPolicy()
        let date = Date(timeIntervalSince1970: 0)
        policy.succeeded(at: date)
        for expected in [60.0, 120, 240, 480, 960, 1920, 3600, 3600, 3600] {
            policy.failed(at: date)
            #expect(policy.delay(at: date) == expected)
            #expect(policy.lastSuccessAt == date)
        }
    }

    @Test("Cancellation after stale success retries in one minute without a failure")
    func cancelledStaleSuccess() {
        var policy = UpdateRefreshPolicy()
        let success = Date(timeIntervalSince1970: 0)
        let cancelled = success.addingTimeInterval(UpdateRefreshPolicy.refreshInterval + 1)
        policy.succeeded(at: success)
        policy.cancelled(at: cancelled)
        #expect(policy.lastSuccessAt == success)
        #expect(policy.consecutiveFailures == 0)
        #expect(policy.delay(at: cancelled) == 60)
        #expect(policy.delay(at: cancelled.addingTimeInterval(59)) == 1)
        #expect(policy.delay(at: cancelled.addingTimeInterval(60)) == 0)
        policy.succeeded(at: cancelled.addingTimeInterval(60))
        #expect(policy.lastCancelledAt == nil)
        #expect(policy.delay(at: cancelled.addingTimeInterval(60)) == UpdateRefreshPolicy.refreshInterval)
    }

    @Test("Cancellation before any success retries in one minute without a failure")
    func cancelledBeforeSuccess() {
        var policy = UpdateRefreshPolicy()
        let date = Date(timeIntervalSince1970: 0)
        policy.cancelled(at: date)
        #expect(policy.lastSuccessAt == nil)
        #expect(policy.consecutiveFailures == 0)
        #expect(policy.delay(at: date) == 60)
        #expect(policy.delay(at: date.addingTimeInterval(60)) == 0)
    }

    @Test("Cancellation preserves failure count and the next failure resumes backoff")
    func cancellationDuringBackoff() {
        var policy = UpdateRefreshPolicy()
        let date = Date(timeIntervalSince1970: 0)
        policy.failed(at: date)
        policy.failed(at: date)
        policy.cancelled(at: date)
        #expect(policy.consecutiveFailures == 2)
        #expect(policy.delay(at: date) == 60)
        policy.failed(at: date.addingTimeInterval(60))
        #expect(policy.lastCancelledAt == nil)
        #expect(policy.consecutiveFailures == 3)
        #expect(policy.delay(at: date.addingTimeInterval(60)) == 240)
    }

    @Test("Opt-out skips scheduled, activation and forced fetches and waits six hours")
    func optedOutScheduler() async {
        let feed = UpdateTestFeed()
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(fetcher: { try await feed.fetch() }, checksEnabled: { false },
                                    sleep: { try await sleeper.sleep($0) })
        checker.startRefreshing()
        #expect(await sleeper.waiting(1) == UpdateRefreshPolicy.refreshInterval)
        await checker.refreshIfNeeded()
        await checker.check()
        await sleeper.wake()
        #expect(await sleeper.waiting(2) == UpdateRefreshPolicy.refreshInterval)
        #expect(await feed.calls == 0)
        checker.stopRefreshing()
    }

    @Test("Forced and activation checks coalesce with an in-flight fetch")
    func inFlightCoalescing() async {
        let feed = UpdateTestFeed()
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(fetcher: {
            try await sleeper.sleep(1)
            return try await feed.fetch()
        }, checksEnabled: { true })
        let first = Task { await checker.check() }
        #expect(await sleeper.waiting(1) == 1)
        #expect(checker.checking)
        await checker.check()
        await checker.refreshIfNeeded()
        await sleeper.wake()
        _ = await first.value
        #expect(await feed.calls == 1)
        #expect(!checker.checking)
    }

    @Test("Cancellation preserves release, error and successful timestamp without failure backoff")
    func cancellationPreservesState() async {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(fetcher: {
            try await sleeper.sleep(1)
            try Task.checkCancellation()
            return try await feed.fetch()
        }, checksEnabled: { true }, now: { clock.date })
        let first = Task { await checker.check() }
        _ = await sleeper.waiting(1)
        await sleeper.wake()
        _ = await first.value
        let success = checker.refreshPolicy.lastSuccessAt
        clock.advance(1)
        let cancelled = Task { await checker.check() }
        _ = await sleeper.waiting(2)
        cancelled.cancel()
        _ = await cancelled.value
        #expect(checker.latest?.version == "0.16.5")
        #expect(checker.lastError == nil)
        #expect(checker.refreshPolicy.lastSuccessAt == success)
        #expect(checker.refreshPolicy.consecutiveFailures == 0)
        #expect(!checker.checking)
    }

    @Test("A manual failure interrupts the normal wait to schedule a retry")
    func manualFailureWakesScheduler() async {
        let clock = UpdateTestClock()
        let feed = UpdateTestFeed()
        let sleeper = UpdateTestSleeper()
        let checker = UpdateChecker(fetcher: { try await feed.fetch() }, checksEnabled: { true },
                                    now: { clock.date }, sleep: { try await sleeper.sleep($0) })
        checker.startRefreshing()
        #expect(await sleeper.waiting(1) == UpdateRefreshPolicy.refreshInterval)
        await feed.set(fails: true)
        await checker.check()
        #expect(await sleeper.waiting(2) == 60)
        #expect(checker.latest?.version == "0.16.5")
        checker.stopRefreshing()
    }
}
