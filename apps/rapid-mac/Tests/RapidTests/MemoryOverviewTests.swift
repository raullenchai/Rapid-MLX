import Darwin
import Foundation
import Testing
@testable import Rapid

struct MemoryOverviewTests {
    private let start = Date(timeIntervalSince1970: 1_000)
    private let session = MemoryOverviewSession(sessionID: UUID(), port: 7660, bearer: "test-only", model: "first")

    @Test func ringWrapKeepsNewestSamplesInOrder() {
        var history = MemoryOverviewHistory()
        for index in 0..<250 {
            history.append(date: start.addingTimeInterval(Double(index)), system: nil,
                           metal: MetalMemorySnapshot(active: Double(index), cache: nil, peak: nil), session: session)
        }
        #expect(history.samples.count == MemoryOverviewHistory.capacity)
        #expect(history.samples.first?.metalActiveGB == 149)
        #expect(history.samples.last?.metalActiveGB == 249)
        #expect(history.samples.map(\.date) == (149..<250).map { start.addingTimeInterval(Double($0)) })
    }

    @Test func fiveMinuteWindowIncludesEndpointsAndExpiresOldReadings() {
        var history = MemoryOverviewHistory()
        for index in 0...100 {
            history.append(date: start.addingTimeInterval(Double(index * 3)), system: nil, metal: nil, session: session)
        }
        #expect(history.samples.count == 101)
        #expect(history.samples.first?.date == start)
        history.append(date: start.addingTimeInterval(303), system: nil, metal: nil, session: session)
        #expect(history.samples.count == 101)
        #expect(history.samples.first?.date == start.addingTimeInterval(3))
        history.append(date: start.addingTimeInterval(1_000), system: nil, metal: nil, session: session)
        #expect(history.samples.count == 1)
        // A backwards wall-clock adjustment cannot retain future readings.
        history.append(date: start, system: nil, metal: nil, session: session)
        #expect(history.samples.count == 1)
    }

    @Test func everySessionBoundaryClearsHistory() {
        let changes = [
            MemoryOverviewSession(sessionID: UUID(), port: session.port, bearer: session.bearer, model: session.model),
            MemoryOverviewSession(sessionID: session.sessionID, port: session.port, bearer: "rotated", model: session.model),
            MemoryOverviewSession(sessionID: session.sessionID, port: session.port, bearer: session.bearer, model: "second"),
            MemoryOverviewSession(sessionID: session.sessionID, port: 7661, bearer: session.bearer, model: session.model),
            MemoryOverviewSession(sessionID: nil, port: session.port, bearer: nil, model: nil)
        ]
        for changed in changes {
            var history = MemoryOverviewHistory()
            history.append(date: start, system: nil, metal: nil, session: session)
            history.append(date: start.addingTimeInterval(3), system: nil, metal: nil, session: session)
            #expect(history.samples.count == 2)
            history.append(date: start.addingTimeInterval(6), system: nil, metal: nil, session: changed)
            #expect(history.session == changed)
            #expect(history.samples.count == 1)
            #expect(history.samples.first?.date == start.addingTimeInterval(6))
            history.reset(for: changed)
            #expect(history.samples.isEmpty)
        }
    }

    @Test func missingReadingsStayIndependentAndNeverBecomeZero() {
        var history = MemoryOverviewHistory()
        let mac = MemoryProbe.Snapshot(totalBytes: 16_000_000_000, usedBytes: 4_000_000_000)
        history.append(date: start, system: mac, metal: nil, session: session)
        #expect(history.samples.last?.systemUsedGB == 4)
        #expect(history.samples.last?.metalActiveGB == nil)
        for (index, active) in [Double.nan, .infinity, -1].enumerated() {
            history.append(date: start.addingTimeInterval(Double(index + 1) * 3), system: nil,
                           metal: MetalMemorySnapshot(active: active, cache: nil, peak: nil), session: session)
            #expect(history.samples.last?.systemUsedGB == nil)
            #expect(history.samples.last?.metalActiveGB == nil)
        }
        history.append(date: start.addingTimeInterval(15), system: nil,
                       metal: MetalMemorySnapshot(active: 0, cache: nil, peak: nil), session: session)
        #expect(history.samples.last?.metalActiveGB == 0)
    }

    @Test func formattingAndShareUseDecimalGBAndUnavailableDash() {
        #expect(MemoryOverviewFormat.bytes(1_250_000_000) == "1.25 GB")
        #expect(MemoryOverviewFormat.bytes(0) == "0.00 GB")
        #expect(MemoryOverviewFormat.bytes(nil) == "—")
        #expect(MemoryOverviewFormat.percent(nil) == "—")
        #expect(MemoryOverviewFormat.percent(.nan) == "—")
        #expect(MemoryOverviewFormat.percent(.infinity) == "—")
        #expect(MemoryOverviewFormat.percent(-1) == "—")
        #expect(MemoryOverviewFormat.engineShare(activeGB: 4, totalBytes: 16_000_000_000) == 0.25)
        #expect(MemoryOverviewFormat.engineShare(activeGB: 0, totalBytes: 16_000_000_000) == 0)
        #expect(MemoryOverviewFormat.engineShare(activeGB: .greatestFiniteMagnitude, totalBytes: 1) == nil)
        #expect(MemoryOverviewFormat.engineShare(activeGB: 1, totalBytes: 0) == nil)
        #expect(MemoryOverviewFormat.engineShare(activeGB: 1, totalBytes: nil) == nil)
        #expect(MemoryOverviewFormat.engineShare(activeGB: nil, totalBytes: 16_000_000_000) == nil)
        for invalid in [-1, Double.nan, .infinity] {
            #expect(MemoryOverviewFormat.engineShare(activeGB: invalid, totalBytes: 16_000_000_000) == nil)
        }
        // Keep an above-total measurement honest; only the drawn bar is capped.
        #expect(MemoryOverviewFormat.engineShare(activeGB: 20, totalBytes: 16_000_000_000) == 1.25)
    }

    @Test func breakdownSumsToUsedAndTotal() throws {
        var vm = vm_statistics64()
        vm.internal_page_count = 400
        vm.purgeable_count = 50
        vm.wire_count = 100
        vm.compressor_page_count = 25
        vm.active_count = 900 // File-backed pages must not inflate app memory.
        let snapshot = MemoryProbe.snapshot(totalBytes: 10_000, pageSize: 10, vm: vm)
        #expect(snapshot.appBytes == 3_500)
        #expect(snapshot.wiredBytes == 1_000)
        #expect(snapshot.compressedBytes == 250)
        let app = try #require(snapshot.appBytes)
        let wired = try #require(snapshot.wiredBytes)
        let compressed = try #require(snapshot.compressedBytes)
        let sum = app + wired + compressed
        #expect(sum == snapshot.usedBytes)
        #expect(sum + snapshot.freeBytes == snapshot.totalBytes)
    }

    @Test func breakdownClampsWithoutUnderflowOrOverflow() {
        var vm = vm_statistics64()
        vm.internal_page_count = 10
        vm.purgeable_count = 20
        vm.wire_count = 8
        vm.compressor_page_count = 8
        let capped = MemoryProbe.snapshot(totalBytes: 100, pageSize: 10, vm: vm)
        #expect(capped.appBytes == 0)
        #expect(capped.wiredBytes == 80)
        #expect(capped.compressedBytes == 20)
        #expect(capped.usedBytes == 100 && capped.freeBytes == 0)
        let overflow = MemoryProbe.snapshot(totalBytes: 100, pageSize: .max, vm: vm)
        #expect(overflow.wiredBytes == 100 && overflow.compressedBytes == 0)
        let zero = MemoryProbe.snapshot(totalBytes: 0, pageSize: 10, vm: vm)
        #expect(zero.usedBytes == 0 && zero.freeBytes == 0 && zero.usedRatio == 0)
    }
}
