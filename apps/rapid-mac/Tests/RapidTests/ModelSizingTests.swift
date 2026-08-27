import Foundation
import Testing
@testable import Rapid

/// Contract for the RAM-aware picker: the right model fits on the
/// right Mac, the wrong one doesn't, the boundary cases follow
/// whichllm's "80% of RAM" rule.
@Suite("ModelSizing parsing + classification")
struct ModelSizingTests {
    // MARK: - Param parsing

    @Test("Parses billion-scale param suffix from common alias shapes")
    func parsesParams() {
        #expect(ModelSizing.parseParamsBillions("qwen3.5-4b") == 4)
        #expect(ModelSizing.parseParamsBillions("gemma-4-12b") == 12)
        #expect(ModelSizing.parseParamsBillions("llama-3.1-8b-8bit") == 8)
        #expect(ModelSizing.parseParamsBillions("smollm3-3b") == 3)
        #expect(ModelSizing.parseParamsBillions("qwen3.5-122b-mxfp4") == 122)
    }

    @Test("Picks the larger param when alias carries multiple Bs (total + active)")
    func picksLargerParam() {
        // "qwen3-coder-next-80b-a3b" — 80B total weights, 3B active. The
        // picker cares about TOTAL weights for RAM headroom, not active
        // params, so we pick 80, not 3.
        let v = ModelSizing.parseParamsBillions("qwen3-coder-next-80b-a3b")
        #expect(v == 80)
    }

    @Test("Returns nil when no recognisable param count is present")
    func nilWhenUnknown() {
        #expect(ModelSizing.parseParamsBillions("custom-alias-no-size") == nil)
        #expect(ModelSizing.parseParamsBillions("") == nil)
    }

    // MARK: - Quantization parsing

    @Test("Defaults to 4-bit for mlx-community style aliases")
    func defaultQuantIs4Bit() {
        #expect(ModelSizing.parseBitsPerWeight("qwen3.5-4b") == 4)
        #expect(ModelSizing.parseBitsPerWeight("gemma-4-12b") == 4)
    }

    @Test("Picks up 8-bit suffix variants")
    func picksUp8Bit() {
        #expect(ModelSizing.parseBitsPerWeight("qwen3.6-27b-8bit") == 8)
        #expect(ModelSizing.parseBitsPerWeight("llama-3.1-70b-8bit") == 8)
    }

    @Test("Picks up bf16/fp16 markers as 16-bit")
    func picksUp16Bit() {
        #expect(ModelSizing.parseBitsPerWeight("qwen-test-bf16") == 16)
    }

    @Test("Picks up sub-4-bit and ternary quant tags (#520)")
    func picksUpLowBitQuants() {
        // The Quickstart starter is a 2-bit ternary build; before #520
        // the parser rounded every non-8/16 tag to 4-bit.
        #expect(ModelSizing.parseBitsPerWeight("bonsai-1.7b-2bit") == 2)
        #expect(ModelSizing.parseBitsPerWeight("some-model-3bit") == 3)
        #expect(ModelSizing.parseBitsPerWeight("some-model-6bit") == 6)
        #expect(ModelSizing.parseBitsPerWeight("some-model-ternary") == 2)
        // Delimiter-bounded parse: a wider tag must not be read as a
        // narrower substring. "16-bit" contains "6-bit" and "1.58bit"
        // contains "8bit"; both must resolve to the correct width.
        #expect(ModelSizing.parseBitsPerWeight("qwen-test-16bit") == 16)
        #expect(ModelSizing.parseBitsPerWeight("qwen-test-16-bit") == 16)
        #expect(ModelSizing.parseBitsPerWeight("prism-1.58bit-ternary") == 2)
        // mxfp4 carries no "bit"/"ternary" token → safe 4-bit default.
        #expect(ModelSizing.parseBitsPerWeight("qwen3.6-122b-mxfp4") == 4)
    }

    // MARK: - Footprint

    @Test("4-bit weights footprint is roughly 0.55 GB per billion params")
    func weightsFootprint4Bit() {
        let f = ModelSizing.estimate(alias: "qwen3-8b")
        #expect(f.paramsBillions == 8)
        // 8B × 0.55 = 4.4 GB
        #expect(abs(f.weightsGB - 4.4) < 0.05)
    }

    @Test("8-bit weights footprint is roughly 1.05 GB per billion params")
    func weightsFootprint8Bit() {
        let f = ModelSizing.estimate(alias: "qwen3.6-27b-8bit")
        #expect(f.paramsBillions == 27)
        // 27B × 1.05 = 28.35 GB
        #expect(abs(f.weightsGB - 28.35) < 0.1)
    }

    @Test("2-bit weights footprint matches the real bonsai model, not a 4-bit guess (#520)")
    func weightsFootprint2Bit() {
        let f = ModelSizing.estimate(alias: "bonsai-1.7b-2bit")
        #expect(f.paramsBillions == 1.7)
        #expect(f.bitsPerWeight == 2)
        // 1.7B × 0.28 = 0.476 GB (~488 MB), close to the real ~484 MB of
        // weights. Before #520 this returned 1.7 × 0.55 = 0.935 GB
        // (~957 MB) — the ~2x-inflated "957 MB" the download bar showed.
        #expect(abs(f.weightsGB - 0.476) < 0.02)
        #expect(f.weightsGB < 0.6, "must not fall back to the ~0.94 GB 4-bit estimate")
    }

    // MARK: - Fit classification

    @Test("4B model is recommended on an 18 GB Mac")
    func qwen4BFitsOn18GB() {
        let hw = mockMac(ramGB: 18)
        let f = ModelSizing.estimate(alias: "qwen3.5-4b")
        #expect(ModelSizing.classify(f, on: hw) == .recommended)
    }

    @Test("12B model is too big on an 18 GB Mac")
    func gemma12BTooBigOn18GB() {
        // User-reported: gemma-4-12b crashes their 18 GB MacBook. The
        // classifier MUST flag this as ``.tooBig`` so the picker can
        // mark it red and warn before they Start.
        let hw = mockMac(ramGB: 18)
        let f = ModelSizing.estimate(alias: "gemma-4-12b")
        let fit = ModelSizing.classify(f, on: hw)
        #expect(fit == .tooBig, "expected .tooBig on 18 GB, got \(fit)")
    }

    @Test("12B model is recommended on a 64 GB Mac")
    func gemma12BFitsOn64GB() {
        let hw = mockMac(ramGB: 64)
        let f = ModelSizing.estimate(alias: "gemma-4-12b")
        #expect(ModelSizing.classify(f, on: hw) == .recommended)
    }

    @Test("122B model is too big on a 96 GB Mac")
    func qwen122BTooBigOn96GB() {
        let hw = mockMac(ramGB: 96)
        let f = ModelSizing.estimate(alias: "qwen3.5-122b")
        #expect(ModelSizing.classify(f, on: hw) == .tooBig)
    }

    @Test("122B model fits on an M3 Ultra 256 GB Mac")
    func qwen122BFitsOnM3Ultra() {
        let hw = mockMac(ramGB: 256)
        let f = ModelSizing.estimate(alias: "qwen3.5-122b")
        #expect(ModelSizing.classify(f, on: hw) == .recommended)
    }

    @Test("Unknown alias defaults to borderline, not blocked")
    func unknownAliasIsBorderline() {
        // A custom alias with no recognisable size shouldn't be auto-
        // rejected — the user might know what they're doing.
        let hw = mockMac(ramGB: 18)
        let f = ModelSizing.estimate(alias: "custom-unknown-alias")
        #expect(ModelSizing.classify(f, on: hw) == .borderline)
    }

    // MARK: - Lineage

    @Test("Newer family generations outrank older")
    func lineageMonotonic() {
        #expect(ModelSizing.lineageScore("qwen3.6-8b") > ModelSizing.lineageScore("qwen3.5-8b"))
        #expect(ModelSizing.lineageScore("qwen3.5-8b") > ModelSizing.lineageScore("qwen3-8b"))
        #expect(ModelSizing.lineageScore("qwen3-8b") > ModelSizing.lineageScore("qwen2.5-8b"))
        #expect(ModelSizing.lineageScore("llama-3.3-8b") > ModelSizing.lineageScore("llama-3.1-8b"))
    }

    // MARK: - Live memory safety (pre-load guard)

    @Test("memorySafety: a model that fits TOTAL but not FREE is unsafe")
    func liveGuardCatchesLowFreeMemory() {
        let g = ModelSizing.estimate(alias: "gemma-4-12b") // ~11.8 GB
        let gib: UInt64 = 1 << 30
        // 64 GB Mac with 50 GB already used projects to ~96%: advisory,
        // but not a blocking confirmation while compression/swap can absorb it.
        #expect(ModelSizing.memorySafety(footprint: g, usedBytes: 50 * gib, totalBytes: 64 * gib) == .tight)
        // Beyond physical RAM requires an explicit decision.
        #expect(ModelSizing.memorySafety(footprint: g, usedBytes: 55 * gib, totalBytes: 64 * gib) == .unsafe)
        // Same Mac idle → comfortable.
        #expect(ModelSizing.memorySafety(footprint: g, usedBytes: 10 * gib, totalBytes: 64 * gib) == .safe)
        // 32 GB Mac, 14 GB used → ~80% projected → ordinary load.
        #expect(ModelSizing.memorySafety(footprint: g, usedBytes: 14 * gib, totalBytes: 32 * gib) == .safe)
    }

    @Test("memorySafety: 95% is advisory and only beyond 100% blocks")
    func liveGuardBlocksOnlyAtDangerLine() {
        let gib: UInt64 = 1 << 30
        #expect(ModelSizing.memorySafety(
            footprintGB: 1, usedBytes: 94 * gib, totalBytes: 100 * gib
        ) == .tight)
        #expect(ModelSizing.memorySafety(
            footprintGB: 0, usedBytes: 100 * gib, totalBytes: 100 * gib
        ) == .tight)
        #expect(ModelSizing.memorySafety(
            footprintGB: 1, usedBytes: 100 * gib, totalBytes: 100 * gib
        ) == .unsafe)
        #expect(!ModelSizing.requiresMemoryConfirmation(.tight))
        #expect(ModelSizing.requiresMemoryConfirmation(.unsafe))
    }

    @Test("memorySafety: fails open on unknown params or unreadable probe")
    func liveGuardFailsOpen() {
        let gib: UInt64 = 1 << 30
        // Custom alias with no parseable param count → never block.
        let unknown = ModelSizing.estimate(alias: "some-private-checkpoint")
        #expect(unknown.paramsBillions == nil)
        #expect(ModelSizing.memorySafety(footprint: unknown, usedBytes: 60 * gib, totalBytes: 64 * gib) == .safe)
        // Probe failure (total 0) → safe, so a broken syscall never blocks a load.
        let g = ModelSizing.estimate(alias: "gemma-4-12b")
        #expect(ModelSizing.memorySafety(footprint: g, usedBytes: 0, totalBytes: 0) == .safe)
    }

    @Test("MemoryWarning explains the projected utilisation and actionable headroom")
    func memoryWarningCopy() {
        let w = ModelSizing.MemoryWarning(
            alias: "gemma-4-12b", hfPath: nil, isAutoRespawn: false,
            severity: .unsafe, footprintGB: 5.9, freeGB: 0.8, totalGB: 24.0,
            plannedReleaseGB: 6.3
        )
        #expect(w.title.contains("gemma-4-12b"))
        #expect(w.message.contains("121%"))
        #expect(w.message.contains("accounts for about 6 GB released"))
        #expect(w.message.contains("more memory than this Mac has"))
        #expect(w.message.contains("6 GB"))
        #expect(!w.message.contains("only about 9 GB is free"))
        #expect(w.confirmTitle.localizedCaseInsensitiveContains("anyway"))
    }

    // MARK: - Helpers

    private func mockMac(ramGB: Int) -> MacHardware {
        MacHardware(
            brandString: "Apple M3 Pro",
            family: .m3,
            tier: .pro,
            physicalRAMBytes: UInt64(ramGB) * UInt64(1 << 30),
            memoryBandwidthGBs: 150
        )
    }
}
