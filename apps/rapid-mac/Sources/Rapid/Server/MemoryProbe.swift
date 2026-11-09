import Darwin
import Foundation

/// Live system-memory snapshot for the bottom-status pill. Surfaces
/// the "how tight is RAM right now" signal that desktop users
/// running large local models routinely care about — the failure
/// mode the user flagged on v0.4.39 is "I can't tell if my Mac is
/// about to swap or kernel-panic and Activity Monitor isn't open."
///
/// We deliberately model this as a free-standing probe rather than
/// a class because every read is fully self-contained: ``host_statistics64``
/// + ``hw.memsize`` are stateless syscalls. The pill view holds a
/// timer and calls ``MemoryProbe.snapshot()`` every refresh; no
/// inter-call state to leak.
enum MemoryProbe {
    /// One probe of the host's memory state. Bytes throughout — the
    /// pill view does the GB conversion so the test suite can pin
    /// raw counts without floating-point fuzz.
    struct Snapshot: Equatable, Sendable {
        let totalBytes: UInt64
        let usedBytes: UInt64
        let appBytes: UInt64?
        let wiredBytes: UInt64?
        let compressedBytes: UInt64?

        init(totalBytes: UInt64, usedBytes: UInt64, appBytes: UInt64? = nil,
             wiredBytes: UInt64? = nil, compressedBytes: UInt64? = nil) {
            self.totalBytes = totalBytes
            self.usedBytes = min(usedBytes, totalBytes)
            self.appBytes = appBytes
            self.wiredBytes = wiredBytes
            self.compressedBytes = compressedBytes
        }

        var freeBytes: UInt64 { totalBytes - usedBytes }
        var usedRatio: Double {
            guard totalBytes > 0 else { return 0 }
            return Double(usedBytes) / Double(totalBytes)
        }
    }

    /// Activity-Monitor-style pressure bucket. Drives the colour dot
    /// next to the GB readout so the user can read state at a
    /// glance without parsing numbers.
    ///
    /// Thresholds picked to match Apple's own "Memory Pressure"
    /// gauge in Activity Monitor: green up to ~70%, yellow through
    /// ~90%, red past that. The exact ratios Apple uses aren't
    /// public, but this matches the qualitative behaviour observed
    /// on M-series machines during heavy model loads.
    enum Pressure: Equatable, Sendable {
        case normal
        case warning
        case critical

        static func classify(ratio: Double) -> Pressure {
            if ratio >= 0.90 { return .critical }
            if ratio >= 0.70 { return .warning }
            return .normal
        }
    }

    /// Read live VM stats. Returns nil if either syscall fails —
    /// the pill renders a hyphen in that case rather than zeros so
    /// the user can tell "probe broken" from "lots of free memory."
    static func snapshot(
        executionObserver: (@Sendable (Bool) -> Void)? = nil
    ) -> Snapshot? {
        executionObserver?(Thread.isMainThread)
        guard let total = sysctlUInt64("hw.memsize") else { return nil }
        guard let vm = readVMStats() else { return nil }
        // Read the page size via sysctl rather than the concurrency-unsafe C global.
        let pageSize = sysctlUInt64("hw.pagesize") ?? 16384
        return snapshot(totalBytes: total, pageSize: pageSize, vm: vm)
    }

    /// Activity Monitor's Memory Used: app memory + wired + compressed.
    /// Internal pages minus purgeable pages measure app memory. Active pages
    /// also contain reclaimable file-backed cache and must not be added.
    static func snapshot(totalBytes: UInt64, pageSize: UInt64, vm: vm_statistics64) -> Snapshot {
        let internalPages = UInt64(vm.internal_page_count)
        let purgeablePages = UInt64(vm.purgeable_count)
        let appPages = internalPages - min(internalPages, purgeablePages)
        // Cap each component against the remaining physical RAM. During a
        // transient over-count the segments still sum to used, without overflow.
        func bytes(_ pages: UInt64, remaining: UInt64) -> UInt64 {
            let (value, overflow) = pages.multipliedReportingOverflow(by: pageSize)
            return overflow ? remaining : min(value, remaining)
        }
        let app = bytes(appPages, remaining: totalBytes)
        let wired = bytes(UInt64(vm.wire_count), remaining: totalBytes - app)
        let compressed = bytes(UInt64(vm.compressor_page_count), remaining: totalBytes - app - wired)
        return Snapshot(totalBytes: totalBytes, usedBytes: app + wired + compressed,
                        appBytes: app, wiredBytes: wired, compressedBytes: compressed)
    }

    /// Format the pill body: "12.3 GB used · 256 GB total". Kept
    /// out of the view so the contract can be pinned by tests —
    /// the labelling style is part of the UI promise, not an
    /// incidental detail.
    static func formatTotal(_ bytes: UInt64) -> String {
        let gib = Double(1 << 30)
        return DownloadProgress.formatMemory(DownloadProgress.bytes(fromDouble: (Double(bytes) / gib).rounded() * gib))
    }

    static func formatLabel(_ snapshot: Snapshot) -> String {
        return "\(DownloadProgress.formatMemory(Int64(clamping: snapshot.usedBytes))) / \(formatTotal(snapshot.totalBytes))"
    }

    // MARK: - Syscall plumbing

    private static func readVMStats() -> vm_statistics64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return stats
    }

    private static func sysctlUInt64(_ key: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        if sysctlbyname(key, &value, &size, nil, 0) != 0 { return nil }
        return value
    }
}
