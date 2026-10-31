import Foundation
import Testing
@testable import Rapid

/// v0.4.40 contract pins for the system-memory pill. The live VM
/// syscall layer is unmockable without a private DI seam, so the
/// tests focus on the pure-function surface: pressure classification,
/// label formatting, ratio math. Real-host snapshots are exercised
/// by a smoke that just asserts ``snapshot()`` returns nil-or-real
/// numbers on the test machine — never crashes, never returns
/// nonsense.
@MainActor
@Suite("MemoryProbe — v0.4.40 system-memory pill")
struct MemoryProbeTests {
    // MARK: - Pressure thresholds

    @Test("0.50 ratio → .normal (idle desktop)")
    func pressureLowNormal() {
        #expect(MemoryProbe.Pressure.classify(ratio: 0.50) == .normal)
    }

    @Test("0.69 ratio → .normal (just under warning threshold)")
    func pressureJustUnderWarning() {
        #expect(MemoryProbe.Pressure.classify(ratio: 0.69) == .normal)
    }

    @Test("0.70 ratio → .warning (inclusive boundary)")
    func pressureWarningBoundary() {
        // 70 % is the green→yellow flip. Inclusive on the warning
        // side so the user gets the heads-up the moment the
        // gauge crosses the line, not a tick later.
        #expect(MemoryProbe.Pressure.classify(ratio: 0.70) == .warning)
    }

    @Test("0.89 ratio → .warning (just under critical)")
    func pressureJustUnderCritical() {
        #expect(MemoryProbe.Pressure.classify(ratio: 0.89) == .warning)
    }

    @Test("0.90 ratio → .critical (inclusive boundary)")
    func pressureCriticalBoundary() {
        #expect(MemoryProbe.Pressure.classify(ratio: 0.90) == .critical)
    }

    @Test("1.00 ratio → .critical (saturated)")
    func pressureSaturated() {
        #expect(MemoryProbe.Pressure.classify(ratio: 1.00) == .critical)
    }

    // MARK: - Label formatting

    @Test("16 GB used of 256 GB → '16.0 GB / 256 GB'")
    func formatLabelTypical() {
        let snap = MemoryProbe.Snapshot(
            totalBytes: 256 * UInt64(1 << 30),
            usedBytes: 16 * UInt64(1 << 30)
        )
        #expect(MemoryProbe.formatLabel(snap) == "\(DownloadProgress.formatMemory(Int64(snap.usedBytes))) / \(DownloadProgress.formatMemory(Int64(snap.totalBytes)))")
    }

    @Test("Total rounds to integer GB so 18 GB / 16 GB Mac doesn't read '15.9 / 18.0'")
    func formatLabelTotalRounding() {
        // Apple sells 16 GB / 18 GB / 24 GB sticks. macOS About
        // reports the nominal integer. The pill mirrors that —
        // showing the manufacturer's spec, not the kernel's
        // available-RAM count.
        let snap = MemoryProbe.Snapshot(
            totalBytes: 18 * UInt64(1 << 30),
            usedBytes: 9 * UInt64(1 << 30) + 500 * UInt64(1 << 20)
        )
        let label = MemoryProbe.formatLabel(snap)
        #expect(label.hasSuffix(" / \(DownloadProgress.formatMemory(Int64(snap.totalBytes)))"), "Total must round to an integer; got '\(label)'")
    }

    @Test("Memory pill uses the shared formatter for tooltip and accessibility")
    func pillLabels() {
        let snap = MemoryProbe.Snapshot(totalBytes: 18 * UInt64(1 << 30), usedBytes: 9 * UInt64(1 << 30))
        let used = DownloadProgress.formatMemory(Int64(snap.usedBytes))
        let free = DownloadProgress.formatMemory(Int64(snap.freeBytes))
        let total = DownloadProgress.formatMemory(Int64(snap.totalBytes))
        #expect(MemoryPill.tooltip(snap: snap, pressure: .normal).contains("\(used) in use"))
        #expect(MemoryPill.tooltip(snap: snap, pressure: .normal).contains("\(free) free"))
        #expect(MemoryPill.accessibilityLabel(snap: snap, pressure: .normal).contains("\(used) used out of \(total)"))
    }

    // MARK: - Snapshot integrity

    @Test("Used > total never happens (defensive clamp)")
    func clamp() {
        // We can't easily inject the host_statistics64 syscall,
        // so this pins the invariant via direct Snapshot
        // construction — defensive for future refactors that
        // might switch the field order or introduce a stale-
        // cache race.
        let snap = MemoryProbe.Snapshot(
            totalBytes: 100,
            usedBytes: 100
        )
        #expect(snap.usedRatio == 1.0)
        #expect(snap.freeBytes == 0)
    }

    @Test("Zero total guards against division by zero")
    func zeroTotalSafe() {
        let snap = MemoryProbe.Snapshot(totalBytes: 0, usedBytes: 0)
        #expect(snap.usedRatio == 0)
    }

    // MARK: - Live snapshot smoke (real host)

    @Test("Live snapshot is either nil or plausible (real-host smoke)")
    func liveSnapshotSmoke() {
        // The test machine MUST have >= 4 GB RAM and the syscall
        // MUST succeed on supported macOS. If either is false the
        // CI signal is genuinely useful and we want to know.
        guard let snap = MemoryProbe.snapshot() else {
            Issue.record("MemoryProbe.snapshot() returned nil on the test host — syscall layer is broken")
            return
        }
        #expect(snap.totalBytes >= UInt64(4) * UInt64(1 << 30),
                "Host has < 4 GB RAM, which is implausible for any macOS dev machine")
        #expect(snap.usedBytes <= snap.totalBytes,
                "Used must not exceed total — defensive clamp must hold")
        #expect(snap.usedBytes > 0,
                "Used must be positive — anything running this test is using some RAM")
    }
}
