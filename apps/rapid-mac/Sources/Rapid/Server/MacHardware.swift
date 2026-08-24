import Foundation

/// Snapshot of the host Mac's relevant hardware capabilities for
/// model-fit decisions. Read once at app launch; the values can't
/// change without a reboot (RAM) or a hardware swap (chip).
///
/// We follow whichllm's convention of reserving 20% of RAM for the
/// OS, browser, IDE, etc. and treating the remaining 80% as the
/// usable pool for the model + KV cache + the rapid-mlx Python
/// process. Apple Silicon's unified memory means the GPU draws
/// from the same pool, so a 16 GB Mac genuinely has ~12 GB to
/// give a model — there is no separate VRAM headroom.
struct MacHardware: Sendable, Equatable {
    /// Apple Silicon chip family — only the dimensions we actually
    /// branch on. We deliberately don't model the Intel-Mac era
    /// because Rapid is mlx-only and mlx requires Apple Silicon.
    enum ChipFamily: String, Sendable {
        case m1, m2, m3, m4, mUnknown
    }

    enum ChipTier: String, Sendable {
        case base, pro, max, ultra, unknown
    }

    /// Raw brand string as returned by ``machdep.cpu.brand_string``;
    /// kept for display ("M3 Ultra"). Display always uses this; the
    /// family/tier parsing is only for capability lookups.
    let brandString: String
    let family: ChipFamily
    let tier: ChipTier
    /// Total physical RAM in bytes.
    let physicalRAMBytes: UInt64
    /// Unified-memory bandwidth in GB/s, sourced from the whichllm
    /// reference table. Used to label the speed expectation in the
    /// picker — a model that fits on an M3 Ultra at 800 GB/s feels
    /// very different from the same model on an M1 base at 68 GB/s.
    let memoryBandwidthGBs: Double

    /// Total RAM in gibibytes (binary GB), rounded to one decimal —
    /// matches how macOS About reports it.
    var physicalRAMGB: Double {
        Double(physicalRAMBytes) / Double(1 << 30)
    }

    /// Usable share of RAM for a model and KV cache. Follows
    /// whichllm's "80% rule" (20% reserved for OS + everything else).
    var usableRAMGB: Double {
        physicalRAMGB * 0.80
    }

    // MARK: - Probe

    /// Read live sysctl values. Returns a probe even if the chip
    /// can't be classified (older / forward-compatible chips fall
    /// through to ``.mUnknown / .unknown`` rather than crashing the
    /// picker).
    ///
    /// ``RAPID_HARDWARE_RAM_GB`` and ``RAPID_HARDWARE_BRAND`` override
    /// the corresponding probes only with ``RAPID_GUI_HARDWARE_FIXTURE=1``.
    /// Production launches never set that gate; golden flows pin the complete
    /// identity so AX output is host-independent.
    static func detect() -> MacHardware {
        let environment = ProcessInfo.processInfo.environment
        let brand = Self.brandString(environment: environment)
        let mem = Self.physicalRAMBytes(environment: environment)
        let (family, tier) = Self.classify(brand)
        let bw = Self.bandwidthGBs(family: family, tier: tier)
        return MacHardware(
            brandString: brand,
            family: family,
            tier: tier,
            physicalRAMBytes: mem,
            memoryBandwidthGBs: bw
        )
    }

    /// ``hw.memsize`` in bytes, unless ``RAPID_HARDWARE_RAM_GB`` pins
    /// it. Pure so the golden-flow pin is unit-testable without mutating the
    /// test process's global environment.
    static func physicalRAMBytes(environment: [String: String]) -> UInt64 {
        physicalRAMBytes(environment: environment) {
            sysctlUInt64("hw.memsize")
        }
    }

    static func physicalRAMBytes(
        environment: [String: String],
        fallback: () -> UInt64?
    ) -> UInt64 {
        if environment["RAPID_GUI_HARDWARE_FIXTURE"] == "1",
           let override = environment["RAPID_HARDWARE_RAM_GB"],
           let gb = Double(override), gb.isFinite, gb > 0, gb <= 1024 {
            let bytes = UInt64((gb * Double(1 << 30)).rounded())
            if bytes > 0 { return bytes }
        }
        return fallback() ?? 0
    }

    /// The golden harness pins the displayed chip together with RAM so AX
    /// output remains identical across CI and release-validation Macs.
    static func brandString(environment: [String: String]) -> String {
        brandString(environment: environment) {
            sysctlString("machdep.cpu.brand_string")
        }
    }

    static func brandString(
        environment: [String: String],
        fallback: () -> String?
    ) -> String {
        if environment["RAPID_GUI_HARDWARE_FIXTURE"] == "1",
           let override = environment["RAPID_HARDWARE_BRAND"],
           !override.isEmpty, override.utf8.count <= 128 {
            return override
        }
        return fallback() ?? "Apple Silicon"
    }

    // MARK: - Display helpers

    /// Short human-readable label, e.g. "18 GB · M3 Pro". Drives the
    /// "Recommended for your Mac" header in the picker so the user
    /// knows what the recommendation is anchored to.
    var shortDescription: String {
        let ramGB = Int(physicalRAMGB.rounded())
        // Brand strings come back like "Apple M3 Pro" — strip the
        // "Apple " prefix because every chip has it.
        let chip = brandString.replacingOccurrences(of: "Apple ", with: "")
        if chip.isEmpty {
            return "\(ramGB) GB"
        }
        return "\(ramGB) GB · \(chip)"
    }

    // MARK: - Internals

    /// Classify ``Apple M3 Pro`` → (.m3, .pro). Falls through to
    /// ``.mUnknown / .unknown`` for anything we don't recognise so
    /// the picker stays functional on forward-compat chips.
    static func classify(_ brand: String) -> (ChipFamily, ChipTier) {
        let lower = brand.lowercased()
        let family: ChipFamily = {
            if lower.contains(" m1") || lower.hasSuffix(" m1") { return .m1 }
            if lower.contains(" m2") || lower.hasSuffix(" m2") { return .m2 }
            if lower.contains(" m3") || lower.hasSuffix(" m3") { return .m3 }
            if lower.contains(" m4") || lower.hasSuffix(" m4") { return .m4 }
            return .mUnknown
        }()
        let tier: ChipTier = {
            if lower.contains("ultra") { return .ultra }
            if lower.contains("max") { return .max }
            if lower.contains("pro") { return .pro }
            // No suffix → base chip. Empty family also falls here.
            if family != .mUnknown { return .base }
            return .unknown
        }()
        return (family, tier)
    }

    /// Unified-memory bandwidth in GB/s. Numbers vendored from
    /// whichllm's gpu.py — these are theoretical peak from Apple's
    /// own marketing materials, not benchmarked throughput, but
    /// they're directionally correct for our "is this chip fast
    /// enough to matter at this size?" labelling.
    static func bandwidthGBs(family: ChipFamily, tier: ChipTier) -> Double {
        switch (family, tier) {
        case (.m1, .ultra): return 800
        case (.m1, .max):   return 400
        case (.m1, .pro):   return 200
        case (.m1, .base):  return 68
        case (.m2, .ultra): return 800
        case (.m2, .max):   return 400
        case (.m2, .pro):   return 200
        case (.m2, .base):  return 100
        case (.m3, .ultra): return 800
        case (.m3, .max):   return 400
        case (.m3, .pro):   return 150
        case (.m3, .base):  return 100
        case (.m4, .ultra): return 819
        case (.m4, .max):   return 546
        case (.m4, .pro):   return 273
        case (.m4, .base):  return 120
        default:            return 100  // conservative floor
        }
    }

    // MARK: - sysctl primitives

    static func sysctlString(_ key: String) -> String? {
        var size: size_t = 0
        if sysctlbyname(key, nil, &size, nil, 0) != 0 { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        if sysctlbyname(key, &buffer, &size, nil, 0) != 0 { return nil }
        let utf8 = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: utf8, as: UTF8.self)
    }

    static func sysctlUInt64(_ key: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        if sysctlbyname(key, &value, &size, nil, 0) != 0 { return nil }
        return value
    }
}
