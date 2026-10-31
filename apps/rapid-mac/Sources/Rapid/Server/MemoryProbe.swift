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
        var freeBytes: UInt64 { totalBytes &- usedBytes }
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
        // Activity Monitor reports "Memory Used" as
        //   app_memory + wired + compressed.
        //
        // VM stats give us active + wired + compressed directly,
        // which is functionally the same thing for our purposes.
        // We exclude inactive + speculative + free because those
        // are reclaimable on demand — counting them as "used"
        // would make every machine look memory-starved at idle.
        //
        // ``hw.pagesize`` (sysctl) gives the same value as the
        // ``vm_kernel_page_size`` global but doesn't trip Swift 6
        // strict-concurrency on the global C var.
        let pageSize = sysctlUInt64("hw.pagesize") ?? 16384
        let active = UInt64(vm.active_count) * pageSize
        let wired = UInt64(vm.wire_count) * pageSize
        let compressed = UInt64(vm.compressor_page_count) * pageSize
        let used = active &+ wired &+ compressed
        // Defensive clamp: pathological systems where the sum
        // overruns hw.memsize would otherwise produce a >100%
        // ratio. Clip so the colour-dot threshold can't underflow
        // into red on a healthy machine due to a brief race.
        return Snapshot(totalBytes: total, usedBytes: min(used, total))
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
