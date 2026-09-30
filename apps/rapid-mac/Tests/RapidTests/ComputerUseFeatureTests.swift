import Foundation
import os
import Testing
@testable import Rapid

@Suite("Experimental Computer Use")
struct ComputerUseFeatureTests {
    @Test("Computer Use is opt-in")
    func defaultsOff() throws {
        let suite = "rapid.computer-use-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(!ComputerUseFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: ComputerUseFeatureConfig.enabledKey)
        #expect(ComputerUseFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Disabling while Computer Use is active returns to Chat")
    func disablingRecoversNavigation() {
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .computerUse,
            enabled: false
        ) == .chat)
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .computerUse,
            enabled: true
        ) == .computerUse)
        #expect(ContentView.sectionAfterComputerUseGateChange(
            current: .images,
            enabled: false
        ) == .images)
    }

    /// ViewInspector is not available in this target, so the behavioral
    /// transition above is paired with the repository's established wiring
    /// guard pattern. This fails if ContentView stops observing the stored
    /// gate or stops applying the tested transition to its live selection.
    @Test("The stored Computer Use gate drives live navigation recovery")
    func gateChangeIsWiredToNavigation() throws {
        let contentView = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let source = try String(contentsOf: contentView, encoding: .utf8)
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .preserve)

        #expect(canonical.contains("@AppStorage(ComputerUseFeatureConfig.enabledKey)privatevarcomputerUseEnabled"))
        #expect(canonical.contains(".onChange(of:experimentalDestinationState,initial:true){_,statein"))
        #expect(canonical.contains("section=Self.sectionAfterComputerUseGateChange(current:section,enabled:state.computerUseEnabled)"))
    }

    @Test("Startup model link maintenance never blocks MainActor")
    @MainActor
    func startupModelLinkMaintenanceIsDetached() async {
        let coordinator = StartupModelLinkMaintenanceCoordinator()
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        var startedIterator = started.makeAsyncIterator()
        let release = DispatchSemaphore(value: 0)
        let ranOnMainThread = OSAllocatedUnfairLock<Bool?>(initialState: nil)
        let maintenance = Task { @MainActor in
            await coordinator.run(generation: 1) {
                ranOnMainThread.withLock { $0 = Thread.isMainThread }
                startedContinuation.yield()
                release.wait()
            }
        }
        _ = await startedIterator.next()

        // Reaching this assertion on MainActor while the maintenance closure
        // is deliberately blocked proves the window/CUA actor remains free.
        release.signal()
        await maintenance.value
        #expect(ranOnMainThread.withLock { $0 } == false)
    }

    @Test("Repeated restores share one model link maintenance operation")
    func startupModelLinkMaintenanceIsSingleFlight() async {
        let coordinator = StartupModelLinkMaintenanceCoordinator()
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        var startedIterator = started.makeAsyncIterator()
        let release = DispatchSemaphore(value: 0)
        let count = OSAllocatedUnfairLock(initialState: 0)
        let catalogReads = OSAllocatedUnfairLock(initialState: 0)
        let operation: @Sendable () -> Void = {
            count.withLock { $0 += 1 }
            startedContinuation.yield()
            release.wait()
        }

        let first = Task { await coordinator.run(generation: 1, operation) }
        _ = await startedIterator.next()
        let second = Task {
            await coordinator.run(generation: 1, operation)
            catalogReads.withLock { $0 += 1 }
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
        #expect(count.withLock { $0 } == 1)
        #expect(catalogReads.withLock { $0 } == 0)
        release.signal()
        await first.value
        await second.value

        #expect(count.withLock { $0 } == 1)
        #expect(catalogReads.withLock { $0 } == 1)

        release.signal()
        await coordinator.run(generation: 2, operation)
        #expect(count.withLock { $0 } == 2)
    }

    @Test("Newer model link maintenance satisfies older waiting generations")
    func startupModelLinkMaintenanceNeverRegressesGeneration() async {
        let coordinator = StartupModelLinkMaintenanceCoordinator()
        let count = OSAllocatedUnfairLock(initialState: 0)
        await coordinator.run(generation: 1) {
            count.withLock { $0 += 1 }
        }

        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        var startedIterator = started.makeAsyncIterator()
        let release = DispatchSemaphore(value: 0)
        let newest = Task {
            await coordinator.run(generation: 3) {
                count.withLock { $0 += 1 }
                startedContinuation.yield()
                release.wait()
            }
        }
        _ = await startedIterator.next()
        let stale = Task {
            await coordinator.run(generation: 2) {
                count.withLock { $0 += 1 }
            }
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
        stale.cancel()
        release.signal()
        await newest.value
        await stale.value

        await coordinator.run(generation: 2) { count.withLock { $0 += 1 } }
        await coordinator.run(generation: 3) { count.withLock { $0 += 1 } }
        #expect(count.withLock { $0 } == 2)
    }
}
