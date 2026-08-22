import Foundation
import Testing
@testable import Rapid

/// Contract for v0.4.18 model-info popover catalog. Pins:
///   - context-window mapping for popular alias families is stable
///   - unknown families return `nil` context (we render "—", not "32k")
///   - params / quant / RAM flow through from `ModelSizing`
///   - label formatters render cleanly for whole-billion, sub-billion,
///     and unknown cases
@Suite("ModelInfoCatalog — v0.4.18")
struct ModelInfoCatalogTests {
    // MARK: - Family + context

    @Test("Qwen 3.6 / 3.5 / 3 land in their own family slots with 32k context")
    func qwenFamilies() {
        let i36 = ModelInfoCatalog.info(for: "qwen3.6-27b", hfRepo: nil)
        #expect(i36.family == "Qwen 3.6")
        #expect(i36.contextWindow == 32_768)

        let i35 = ModelInfoCatalog.info(for: "qwen3.5-4b", hfRepo: nil)
        #expect(i35.family == "Qwen 3.5")
        #expect(i35.contextWindow == 32_768)

        let i3 = ModelInfoCatalog.info(for: "qwen3-8b", hfRepo: nil)
        #expect(i3.family == "Qwen 3")
        #expect(i3.contextWindow == 32_768)
    }

    @Test("Llama 3.1+ reports 131072 (128k label); bare Llama 3 reports 8192")
    func llamaContextSplit() {
        let l31 = ModelInfoCatalog.info(for: "llama-3.1-8b", hfRepo: nil)
        #expect(l31.contextWindow == 131_072)
        #expect(l31.contextLabel == "128k")

        let l3 = ModelInfoCatalog.info(for: "llama-3-8b", hfRepo: nil)
        #expect(l3.contextWindow == 8_192)
        #expect(l3.contextLabel == "8k")
    }

    @Test("Gemma 3 reports the stock 8k (long-context variant ships as a separate repo)")
    func gemmaContext() {
        let g3 = ModelInfoCatalog.info(for: "gemma-3-12b", hfRepo: nil)
        #expect(g3.family == "Gemma 3")
        #expect(g3.contextWindow == 8_192)
    }

    @Test("Ornith 1.5 reports the 256k window from both official model configs")
    func ornithContext() {
        for alias in ["ornith-1.5-9b-bf16", "ornith-1.5-35b-a3b-bf16"] {
            let info = ModelInfoCatalog.info(for: alias, hfRepo: nil)
            #expect(info.family == "Ornith 1.5")
            #expect(info.contextWindow == 262_144)
            #expect(info.contextLabel == "256k")
        }
        #expect(ModelInfoCatalog.info(for: "ornithology-9b", hfRepo: nil).family == "Unknown")
    }

    @Test("Unknown alias families return nil context — meter shows '—', not a guess")
    func unknownFamily() {
        let mystery = ModelInfoCatalog.info(for: "totally-made-up-7b", hfRepo: nil)
        #expect(mystery.family == "Unknown")
        #expect(mystery.contextWindow == nil)
        #expect(mystery.contextLabel == "—")
    }

    // MARK: - Sizing flow-through

    @Test("ModelSizing params + quant + RAM all flow through")
    func sizingFlowThrough() {
        let info = ModelInfoCatalog.info(for: "qwen3.6-27b-8bit", hfRepo: "x/y")
        #expect(info.paramsBillions == 27)
        #expect(info.bitsPerWeight == 8)
        // 27B × 1.05 + 1.2 base + 6 KV (params ≥ 25 band) ≈ 35.55
        #expect(info.approxRAMGB > 34 && info.approxRAMGB < 37)
    }

    @Test("Sub-1B params render as e.g. '0.6B', not '0B'")
    func subBillionLabel() {
        let info = ModelInfoCatalog.info(for: "bonsai-0.6b", hfRepo: nil)
        #expect(info.paramsLabel == "0.6B")
    }

    @Test("Whole-billion params render without a trailing '.0'")
    func wholeBillionLabel() {
        let info = ModelInfoCatalog.info(for: "qwen3.5-4b", hfRepo: nil)
        #expect(info.paramsLabel == "4B")
    }

    @Test("Unknown-params alias still surfaces with '—' labels")
    func unknownParamsLabel() {
        let info = ModelInfoCatalog.info(for: "totally-mystery", hfRepo: nil)
        #expect(info.paramsLabel == "—")
        #expect(info.ramLabel == "—")
    }

    // MARK: - Format helpers

    @Test("contextLabel renders multiples of 1024 as '<n>k'; odd values stay literal")
    func contextLabelFormatting() {
        let qwen = ModelInfoCatalog.info(for: "qwen3.5-4b", hfRepo: nil)
        #expect(qwen.contextLabel == "32k")
        let llama = ModelInfoCatalog.info(for: "llama-3.1-8b", hfRepo: nil)
        #expect(llama.contextLabel == "128k")  // 131072 / 1024 = 128
        let phi3 = ModelInfoCatalog.info(for: "phi-3-mini", hfRepo: nil)
        #expect(phi3.contextLabel == "4k")     // 4096 / 1024 = 4
    }

    @Test("quantLabel renders bits as '<n>-bit'")
    func quantLabelFormatting() {
        let four = ModelInfoCatalog.info(for: "qwen3.5-4b", hfRepo: nil)
        #expect(four.quantLabel == "4-bit")
        let eight = ModelInfoCatalog.info(for: "qwen3.5-27b-8bit", hfRepo: nil)
        #expect(eight.quantLabel == "8-bit")
    }

    // MARK: - Issue #363: serverContextWindow override

    @Test("serverContextWindow wins over family heuristic when present + positive")
    func serverWindowWinsOverFamily() {
        // Qwen 3.5 family heuristic is 32k; the rapid-mlx engine
        // exposes the real `max_position_embeddings` which can be
        // larger (Qwen 3.5 4B is 40960 per the upstream config).
        // The server value MUST win so the user's max-tokens slider
        // auto-scales to the real cap instead of the stale heuristic.
        let info = ModelInfoCatalog.info(
            for: "qwen3.5-4b",
            hfRepo: nil,
            serverContextWindow: 40_960
        )
        #expect(info.contextWindow == 40_960)
        #expect(info.family == "Qwen 3.5")
    }

    @Test("serverContextWindow nil falls back to family heuristic (older sidecar)")
    func nilServerWindowFallsBackToFamily() {
        // Older rapid-mlx (< 0.8.4) doesn't emit `context_window` —
        // the decoded `ServerModelProfile.contextWindow` is `nil`.
        // The catalog MUST fall through to the per-family heuristic
        // so the popover + trim logic still have a useful number.
        let info = ModelInfoCatalog.info(
            for: "qwen3.5-4b",
            hfRepo: nil,
            serverContextWindow: nil
        )
        #expect(info.contextWindow == 32_768)
    }

    @Test("serverContextWindow 0 / negative falls back to family heuristic")
    func nonPositiveServerWindowFallsBack() {
        // Defense against a future server-side regression that ever
        // emits 0 / negative on the wire. Render the family
        // heuristic rather than "0k" or "-1".
        let zero = ModelInfoCatalog.info(
            for: "qwen3.5-4b",
            hfRepo: nil,
            serverContextWindow: 0
        )
        #expect(zero.contextWindow == 32_768)
        let neg = ModelInfoCatalog.info(
            for: "qwen3.5-4b",
            hfRepo: nil,
            serverContextWindow: -1
        )
        #expect(neg.contextWindow == 32_768)
    }

    @Test("Unknown-family alias still benefits from serverContextWindow when present")
    func unknownFamilyHonoursServerWindow() {
        // The family heuristic returns nil for unknown aliases (we'd
        // rather render "—" than guess 4096), but when the server
        // does emit a value we surface it — the alias may be a brand-
        // new family the catalog hasn't learned about yet.
        let info = ModelInfoCatalog.info(
            for: "brand-new-family-9b",
            hfRepo: nil,
            serverContextWindow: 65_536
        )
        #expect(info.family == "Unknown")
        #expect(info.contextWindow == 65_536)
    }

    @Test("Unknown-family alias + no server window stays nil (popover shows '—')")
    func unknownFamilyNilServerStaysNil() {
        let info = ModelInfoCatalog.info(
            for: "brand-new-family-9b",
            hfRepo: nil,
            serverContextWindow: nil
        )
        #expect(info.contextWindow == nil)
        #expect(info.contextLabel == "—")
    }

    @Test("contextWindowFallback returns the family-table value (or nil) for an alias")
    func fallbackHelperMatchesFamilyTable() {
        #expect(ModelInfoCatalog.contextWindowFallback(forAlias: "qwen3.6-27b") == 32_768)
        #expect(ModelInfoCatalog.contextWindowFallback(forAlias: "llama-3.1-8b") == 131_072)
        #expect(ModelInfoCatalog.contextWindowFallback(forAlias: "gemma-4-12b") == 8_192)
        #expect(ModelInfoCatalog.contextWindowFallback(forAlias: "phi-4-mini") == 16_384)
        #expect(ModelInfoCatalog.contextWindowFallback(forAlias: "unknown-blah") == nil)
    }
}
