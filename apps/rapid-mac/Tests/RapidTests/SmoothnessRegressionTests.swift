import Foundation
import os
import Testing
import ViewInspector
@testable import Rapid

@Suite("SwiftUI smoothness regressions")
struct SmoothnessRegressionTests {
    @Test("CPUProbe.snapshot executes outside the main thread")
    @MainActor
    func cpuProbeSnapshotLeavesMainThread() async {
        let sampledOnMain = OSAllocatedUnfairLock<Bool?>(initialState: nil)

        _ = await SystemProbeSampler.sample {
            CPUProbe.snapshot { isMainThread in
                sampledOnMain.withLock { $0 = isMainThread }
            }
        }

        #expect(sampledOnMain.withLock { $0 } == false)
    }

    @Test("MemoryProbe.snapshot executes outside the main thread")
    @MainActor
    func memoryProbeSnapshotLeavesMainThread() async {
        let sampledOnMain = OSAllocatedUnfairLock<Bool?>(initialState: nil)

        _ = await SystemProbeSampler.sample {
            MemoryProbe.snapshot { isMainThread in
                sampledOnMain.withLock { $0 = isMainThread }
            }
        }

        #expect(sampledOnMain.withLock { $0 } == false)
    }

    @Test("GPUProbe.snapshot executes outside the main thread")
    @MainActor
    func gpuProbeSnapshotLeavesMainThread() async {
        let sampledOnMain = OSAllocatedUnfairLock<Bool?>(initialState: nil)

        _ = await SystemProbeSampler.sample {
            GPUProbe.snapshot { isMainThread in
                sampledOnMain.withLock { $0 = isMainThread }
            }
        }

        #expect(sampledOnMain.withLock { $0 } == false)
    }

    @Test("Memory pill seeds a real snapshot before its first render")
    @MainActor
    func memoryPillDoesNotFlashFailurePlaceholder() throws {
        let snapshot = MemoryProbe.Snapshot(
            totalBytes: 32 * UInt64(1 << 30),
            usedBytes: 8 * UInt64(1 << 30)
        )
        let firstLabel = try MemoryPill(sample: { snapshot })
            .inspect()
            .find(ViewType.Text.self)
            .string()

        #expect(firstLabel == "8.0 GB / 32 GB")
    }

    @Test("Memory pill publishes sampled state on the main thread")
    @MainActor
    func memoryPillPublishesOnMainThread() async throws {
        let publication = OSAllocatedUnfairLock<(count: Int, onMain: Bool?)>(
            initialState: (0, nil)
        )
        let snapshot = MemoryProbe.Snapshot(totalBytes: 100, usedBytes: 50)
        let sut = MemoryPill(
            refreshInterval: 1,
            sample: { snapshot },
            snapshotDidPublish: { _ in
                publication.withLock {
                    $0.count += 1
                    $0.onMain = Thread.isMainThread
                }
            }
        )

        try await ViewHosting.host(sut) {
            let didPublish = try await waitUntil {
                publication.withLock { $0.count > 0 }
            }
            #expect(didPublish)
        }

        #expect(publication.withLock { $0.onMain } == true)
    }

    @Test("Memory pill stops its sampling loop when removed")
    @MainActor
    func memoryPillCancelsSamplingOnDisappear() async throws {
        let sampleCount = OSAllocatedUnfairLock<Int>(initialState: 0)
        let snapshot = MemoryProbe.Snapshot(totalBytes: 100, usedBytes: 50)
        let sut = MemoryPill(
            refreshInterval: 0.1,
            sample: {
                sampleCount.withLock { $0 += 1 }
                return snapshot
            }
        )

        try await ViewHosting.host(sut) {
            let sampledAfterMount = try await waitUntil {
                sampleCount.withLock { $0 >= 2 }
            }
            #expect(sampledAfterMount)
        }

        let countAtRemoval = sampleCount.withLock { $0 }
        try await Task.sleep(for: .milliseconds(250))
        #expect(sampleCount.withLock { $0 } == countAtRemoval)
    }

    private func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    @MainActor
    private func waitUntil(
        _ condition: @MainActor () -> Bool
    ) async throws -> Bool {
        for _ in 0..<100 {
            if condition() { return true }
            try await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }
}
