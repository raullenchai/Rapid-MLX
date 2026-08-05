import Foundation
import Testing
@testable import Rapid

/// v0.7.16 — pin the per-alias benchmark JSON sidecar that drives the
/// picker hover tooltip. The JSON must:
///
///   1. Decode cleanly into ``BenchScores`` for every alias the
///      curated ``RAMBucketedDefault/tiers`` table references.
///   2. Honour the spec-locked General-&-Reasoning merge rule
///      (`mean(mmlu_pro, gpqa_diamond)` when both present, single
///      bench otherwise, ``nil`` when neither).
///   3. Round-trip a handful of sample alias scores so a stray edit
///      to the JSON (typo in a key, accidental string-vs-number) is
///      caught at CI time.
///   4. Never fabricate a value to fill a gap — gaps stay ``nil``.
///
/// The five-axis order (General & Reasoning → Code → Tool →
/// Instruction Following → Speed) is the user-signed-off spec; a
/// future "Speed at the top" reshuffle should fail the
/// ``axisOrderMatchesSpec`` test.
@Suite("BenchScoresCatalog — JSON sidecar + per-alias score loader")
struct BenchScoresTests {

    // MARK: - JSON load

    @Test("benchmark-scores.json loads cleanly and yields at least one alias row")
    func jsonLoadsAndDecodes() {
        let aliases = BenchScoresCatalog.allAliases
        #expect(!aliases.isEmpty, "Catalog returned empty — JSON load probably failed")
    }

    // MARK: - Sample alias spot-checks

    @Test("qwen3.5-9b-4bit decodes with expected scores")
    func sampleQwen35_9b() {
        let s = BenchScoresCatalog.lookup(alias: "qwen3.5-9b-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        // General-&-Reasoning = mean(82.5, 81.7) = 82.1
        #expect(closeEnough(s.generalReasoning, 82.1))
        #expect(closeEnough(s.mmluPro, 82.5))
        #expect(closeEnough(s.gpqaDiamond, 81.7))
        #expect(s.generalReasoningSource == "mean(mmlu_pro, gpqa_diamond)")
        #expect(closeEnough(s.code, 65.6))
        #expect(closeEnough(s.tool, 66.1))
        #expect(closeEnough(s.ifeval, 91.5))
        #expect(closeEnough(s.speedTps, 106.4))
    }

    @Test("gemma3-1b-qat-4bit bench row: fast (262 t/s), near-floor reasoning, code is an honest gap")
    func sampleGemma3_1bQat() {
        // v0.8.18: gemma3-1b-qat-4bit is no longer a recommended
        // speed pick (reasoning ≈17.0 — incoherent in chat), but it
        // stays in the bench catalog + "All aliases" list, so its
        // lookup must still resolve.
        let s = BenchScoresCatalog.lookup(alias: "gemma3-1b-qat-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        #expect(closeEnough(s.speedTps, 262.0))
        #expect(closeEnough(s.ifeval, 80.2))
        // Code is now an honest gap: Gemma 3 1B publishes no coding
        // benchmark (the prior 1.9 was unsourced). 2026-06-30 nullify.
        #expect(s.code == nil)
        // Tool is null — Google does not publish BFCL for Gemma.
        #expect(s.tool == nil)
        // General-&-Reasoning is mean(14.7, 19.2) = 16.95 ≈ 17.0.
        #expect(closeEnough(s.generalReasoning, 17.0, tolerance: 0.1))
    }

    @Test("phi-4-mini-4bit has a 70.3 Tool score (BFCL published)")
    func samplePhi4Mini() {
        let s = BenchScoresCatalog.lookup(alias: "phi-4-mini-4bit")
        #expect(s != nil)
        #expect(closeEnough(s?.tool, 70.3))
        #expect(closeEnough(s?.speedTps, 159.4))
    }

    @Test("devstral-v2-24b-4bit: code preserved; reasoning + IFEval are honest gaps for the 2512 variant")
    func sampleDevstralV2() {
        let s = BenchScoresCatalog.lookup(alias: "devstral-v2-24b-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        // Code stays (a real SWE-bench-family number).
        #expect(closeEnough(s.code, 65.9))
        // 2026-06-30 nullify: Devstral Small 2 (2512) publishes only
        // SWE/Terminal-bench — no MMLU-Pro/GPQA, no IFEval. The prior
        // 77.8 / 83.3 weren't sourced to the 2512 card.
        #expect(s.generalReasoning == nil)
        #expect(s.ifeval == nil)
        // Tool is null — Mistral does not publish BFCL for Devstral.
        #expect(s.tool == nil)
    }

    @Test("qwen3-coder-30b-4bit: reasoning + code + IFEval are honest gaps (no incomparable substitution)")
    func sampleQwen3Coder30bGaps() {
        // #468 model. Qwen publishes no MMLU-Pro/GPQA/IFEval for the
        // Coder variant; the prior 62.0 / 84.7 were borrowed from the
        // base Qwen3-30B-A3B general model. 2026-06-30 nullify.
        let s = BenchScoresCatalog.lookup(alias: "qwen3-coder-30b-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        #expect(s.generalReasoning == nil)
        #expect(s.generalReasoningSource == nil)
        #expect(s.ifeval == nil)
        // #468: code is an honest gap too. The only available number was
        // Artificial Analysis's Coding *Index* (29.0) — a normalized
        // composite on a different scale than the pass@1 family the Code
        // axis renders — which made this dedicated coder show the LOWEST
        // code bar next to its bucket-mates ("recommending a broken
        // model"). Per the no-incomparable-substitution policy it is left
        // null (renders as a dashed track), not shown at a cross-scale
        // value.
        #expect(s.code == nil)
    }

    @Test("qwen3.5-122b-8bit carries the published Qwen3.5-122B-A10B numbers (ground-truth fill)")
    func sample122b_8bit() {
        let s = BenchScoresCatalog.lookup(alias: "qwen3.5-122b-8bit")
        #expect(s != nil)
        guard let s = s else { return }
        // 2026-06-30 ground-truth fill: previously all-null, now carries
        // the official Qwen3.5-122B-A10B card numbers (med-confidence).
        // General-&-Reasoning = mean(86.7, 86.6) = 86.65.
        #expect(closeEnough(s.generalReasoning, 86.65))
        #expect(closeEnough(s.mmluPro, 86.7))
        #expect(closeEnough(s.gpqaDiamond, 86.6))
        #expect(s.generalReasoningSource == "mean(mmlu_pro, gpqa_diamond)")
        #expect(closeEnough(s.code, 78.9))
        #expect(closeEnough(s.tool, 72.2))
        #expect(closeEnough(s.ifeval, 93.4))
        #expect(closeEnough(s.speedTps, 42.7))
    }

    @Test("qwen3.5-4b-4bit mismap fixed: now the Qwen3.5-4B card numbers, not Qwen3-4B-2507")
    func sampleQwen35_4bMismapFixed() {
        let s = BenchScoresCatalog.lookup(alias: "qwen3.5-4b-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        // General-&-Reasoning = mean(79.1, 76.2) = 77.65.
        #expect(closeEnough(s.generalReasoning, 77.65))
        #expect(closeEnough(s.mmluPro, 79.1))
        #expect(closeEnough(s.gpqaDiamond, 76.2))
        #expect(closeEnough(s.code, 55.8))
        #expect(closeEnough(s.tool, 50.3))
        #expect(closeEnough(s.ifeval, 89.8))
        // speed_tps is locally measured and out of scope for the
        // ground-truth reconciliation — unchanged.
        #expect(closeEnough(s.speedTps, 157.6))
    }

    @Test("gemma-4-12b-4bit mismap fixed: real Gemma 4 12B numbers, not the Gemma-3-12B floor")
    func sampleGemma4_12bMismapFixed() {
        let s = BenchScoresCatalog.lookup(alias: "gemma-4-12b-4bit")
        #expect(s != nil)
        guard let s = s else { return }
        // General-&-Reasoning = mean(77.2, 78.8) = 78.0.
        #expect(closeEnough(s.generalReasoning, 78.0))
        #expect(closeEnough(s.code, 72.0))
        #expect(closeEnough(s.tool, 69.0))
        // Gemma 4 12B IT does not publish IFEval — the old 88.9 was a
        // Gemma-3-12B floor placeholder and is now an honest gap.
        #expect(s.ifeval == nil)
    }

    @Test("Single-basis general-reasoning fills carry the canonical source string")
    func sampleSingleBasisSources() {
        // deepseek-coder fill: MMLU-Pro only (no GPQA published).
        let ds = BenchScoresCatalog.lookup(alias: "deepseek-coder-v2-lite-16b-4bit")
        #expect(ds?.generalReasoningSource == "mmlu_pro only")
        #expect(closeEnough(ds?.generalReasoning, 41.57))
        // llama3-3b fill: GPQA-Diamond only (vanilla MMLU dropped to keep
        // the axis comparable — MMLU is far easier than MMLU-Pro/GPQA).
        let ll = BenchScoresCatalog.lookup(alias: "llama3-3b-4bit")
        #expect(ll?.generalReasoningSource == "gpqa_diamond only")
        #expect(closeEnough(ll?.generalReasoning, 32.8))
        #expect(closeEnough(ll?.tool, 67.0))
        #expect(closeEnough(ll?.ifeval, 77.4))
    }

    @Test("Unknown alias returns nil (graceful fallback for uncatalogued rows)")
    func unknownAliasReturnsNil() {
        let s = BenchScoresCatalog.lookup(alias: "nonexistent-1.2.3-4bit")
        #expect(s == nil)
    }

    // MARK: - Merge rule

    @Test("mergeGeneralReasoning: mean of both present scores")
    func mergeBothPresent() {
        let (value, source) = BenchScoresCatalog.mergeGeneralReasoning(
            mmluPro: 80.0, gpqaDiamond: 70.0
        )
        #expect(closeEnough(value, 75.0))
        #expect(source == "mean(mmlu_pro, gpqa_diamond)")
    }

    @Test("mergeGeneralReasoning: single MMLU-Pro fallback")
    func mergeMMLUOnly() {
        let (value, source) = BenchScoresCatalog.mergeGeneralReasoning(
            mmluPro: 60.0, gpqaDiamond: nil
        )
        #expect(closeEnough(value, 60.0))
        #expect(source == "mmlu_pro only")
    }

    @Test("mergeGeneralReasoning: single GPQA fallback")
    func mergeGPQAOnly() {
        let (value, source) = BenchScoresCatalog.mergeGeneralReasoning(
            mmluPro: nil, gpqaDiamond: 45.0
        )
        #expect(closeEnough(value, 45.0))
        #expect(source == "gpqa_diamond only")
    }

    @Test("mergeGeneralReasoning: both nil → nil score AND nil source")
    func mergeBothNil() {
        let (value, source) = BenchScoresCatalog.mergeGeneralReasoning(
            mmluPro: nil, gpqaDiamond: nil
        )
        #expect(value == nil)
        #expect(source == nil)
    }

    // MARK: - Threshold table

    @Test("Threshold table matches benchmarks-locked.md")
    func thresholdTableMatchesSpec() {
        // Pinned per `/tmp/benchmarks-locked.md` + the merged
        // General-&-Reasoning rule from the task spec (50 / 75).
        let cases: [(BenchScores.Axis, Double, Double, Double, String)] = [
            (.generalReasoning, 50,  75,  100, ""),
            (.code,             30,  65,  100, ""),
            (.tool,             50,  70,  100, ""),
            (.ifeval,           75,  88,  100, ""),
            (.speed,            80, 180,  300, " t/s"),
        ]
        for (axis, good, great, normalizer, suffix) in cases {
            let t = axis.thresholds
            #expect(t.good == good, "\(axis) good")
            #expect(t.great == great, "\(axis) great")
            #expect(t.normalizer == normalizer, "\(axis) normalizer")
            #expect(t.suffix == suffix, "\(axis) suffix")
        }
    }

    @Test("Axis order matches the user-signed-off spec — Speed LAST, not first")
    func axisOrderMatchesSpec() {
        // The task spec is explicit: General & Reasoning → Code →
        // Tool → Instruction Following → Speed. Speed sits LAST
        // even though it's a key purchase signal — it gets its own
        // bar because it matters, but the ordering communicates
        // quality-first.
        let expected: [BenchScores.Axis] = [
            .generalReasoning,
            .code,
            .tool,
            .ifeval,
            .speed,
        ]
        #expect(BenchScores.Axis.allCases == expected)
    }

    @Test("Axis labels are non-empty and the General-&-Reasoning Chinese label is locked")
    func axisLabelsLocked() {
        for axis in BenchScores.Axis.allCases {
            #expect(!axis.label.isEmpty)
            #expect(!axis.bilingualLabel.isEmpty)
        }
        // Lock the canonical Chinese name for the merged G&R bar so
        // a future copy edit can't silently drop it.
        #expect(BenchScores.Axis.generalReasoning.bilingualLabel.contains("通识和推理"))
    }

    @Test("Every alias referenced by RAMBucketedDefault has a benchmark-scores.json row")
    func everyRecommendedAliasHasScoreRow() {
        // Gather every distinct alias the curated bucket table can
        // surface. The picker can recommend any of these on hover,
        // so the JSON must carry a row (even if every axis is null
        // for VL / Llama-3 rows that don't publish bench numbers).
        var distinct: Set<String> = []
        for tier in RAMBucketedDefault.tiers {
            for pick in tier.picks {
                distinct.insert(pick.alias)
            }
        }
        // Curated picks that legitimately publish NO standard benchmark —
        // their capability / speed come from the maintainer's own eval and
        // surface in the recommendation stats line, not the bench meters.
        // The picker degrades gracefully for these (no bar block, no card
        // meters), so a bench JSON row is not required.
        // lfm2.5-2.6b-4bit joined 2026-08-04 as the 8-15 GB pick. It is
        // genuinely unscored — not on Artificial Analysis, no published
        // standard bench — so it belongs here rather than getting a
        // fabricated row. The anti-fabrication check below is what keeps
        // that honest.
        let noStandardBench: Set<String> = [
            "bonsai-27b-2bit", "lfm2.5-8b-a1b-4bit", "lfm2.5-2.6b-4bit",
        ]
        let known = Set(BenchScoresCatalog.allAliases)
        let missing = distinct.subtracting(known).subtracting(noStandardBench)
        #expect(
            missing.isEmpty,
            "Aliases recommended by RAMBucketedDefault but missing a bench JSON row: \(missing.sorted())"
        )
        // Anti-fabrication: the allowlisted picks publish no standard
        // benchmark, so they must NOT carry a bench-scores.json row —
        // otherwise re-introducing fabricated maintainer-eval numbers
        // into the standard-bench columns would silently pass this test.
        let fabricated = noStandardBench.intersection(known)
        #expect(
            fabricated.isEmpty,
            "Allowlisted no-standard-bench aliases must not have a bench JSON row (fabrication risk): \(fabricated.sorted())"
        )
    }
}

// MARK: - Helpers

private func closeEnough(_ a: Double?, _ b: Double, tolerance: Double = 0.05) -> Bool {
    guard let a = a else { return false }
    return abs(a - b) < tolerance
}
