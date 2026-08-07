import Foundation

/// Industry-standard benchmark scores for a model alias, plus measured
/// decode tok/s. When a model has no compatible published row, a fully
/// recorded Rapid-MLX release eval may fill the gap; its per-row provenance
/// names that local suite and the raw counts live under `docs/benchmarks/`.
/// All four LLM bars sit on a 0–100 % scale.
///
/// The picker hover tooltip renders five bars top → bottom:
///
///   1. General & Reasoning (`通识和推理`) — arithmetic mean of
///      MMLU-Pro + GPQA Diamond (when both are present). When only
///      one of the two is published, ``generalReasoning`` carries
///      that single number and ``generalReasoningSource`` records
///      which bench it came from so the tooltip footer can be
///      honest about the basis.
///   2. Code — LiveCodeBench v6 pass@1.
///   3. Tool — BFCL v3 composite (v4 numbers are treated as
///      comparable magnitude; per the audit's gap-honesty policy
///      the UI label stays version-agnostic).
///   4. Instruction Following — IFEval prompt-strict accuracy.
///   5. Speed — measured long-context decode tok/s; hardware is recorded in
///      the row's provenance (community rows use M3 Ultra, the release sweep
///      uses M2 Pro).
///
/// ``nil`` on any axis means neither a compatible published number nor a
/// recorded local release eval exists for that axis.
/// The UI must render the dashed track + em-dash value in that row.
/// **Never** fabricate a value to fill a gap; the explicit user
/// policy (recorded in `model-recs-audit.md` §1.5) is honest gaps
/// over plausible fakes.
struct BenchScores: Equatable, Sendable {
    /// Merged General-&-Reasoning score on a 0–100 scale.
    /// `mean(mmluPro, gpqaDiamond)` when both are published;
    /// otherwise the single published bench (see `generalReasoningSource`).
    let generalReasoning: Double?
    /// Human-readable label describing how `generalReasoning` was
    /// derived. Examples:
    ///   * `"mean(mmlu_pro, gpqa_diamond)"`
    ///   * `"mmlu_pro only"`
    ///   * `"gpqa_diamond only"`
    /// ``nil`` when the merged value itself is ``nil``.
    let generalReasoningSource: String?
    /// Original per-bench breakdown when both inputs are available.
    /// Surfaced as the "82.5 / 81.7" caption under the merged value
    /// in the tooltip when present.
    let mmluPro: Double?
    let gpqaDiamond: Double?

    /// LiveCodeBench v6 pass@1, 0–100.
    let code: Double?
    /// BFCL composite (v3/v4), 0–100.
    let tool: Double?
    /// IFEval prompt-strict accuracy, 0–100.
    let ifeval: Double?
    /// Decode tok/s; see the JSON row's `speed_source` for hardware.
    let speedTps: Double?

    /// Axes rendered in the tooltip, top → bottom. The order is the
    /// user-signed-off spec: General-&-Reasoning, Code, Tool,
    /// Instruction Following, Speed.
    enum Axis: String, CaseIterable, Sendable {
        case generalReasoning
        case code
        case tool
        case ifeval
        case speed

        /// English label used in the tooltip (the canonical Chinese
        /// label `通识和推理` is documented in ``localizedLabel`` for
        /// future bilingual surfaces but the rendered tooltip stays
        /// English to match the rest of the picker copy).
        var label: String {
            switch self {
            case .generalReasoning: return "General & Reasoning"
            case .code:             return "Code"
            case .tool:             return "Tool"
            case .ifeval:           return "Instruction Following"
            case .speed:            return "Speed"
            }
        }

        /// Bilingual label `"General & Reasoning / 通识和推理"`. Reserved
        /// for the future Chinese-locale render; never used in the
        /// shipped picker today but kept beside ``label`` so a future
        /// bilingual switch is a one-line edit.
        var bilingualLabel: String {
            switch self {
            case .generalReasoning: return "General & Reasoning / 通识和推理"
            case .code:             return "Code / 代码"
            case .tool:             return "Tool / 工具调用"
            case .ifeval:           return "Instruction Following / 指令遵循"
            case .speed:            return "Speed / 速度"
            }
        }

        /// `(good, great, normalizer, valueSuffix)` for the axis.
        ///
        /// Thresholds were locked in `/tmp/benchmarks-locked.md`:
        /// MMLU-Pro 60/80 + GPQA Diamond 40/70 mean → 50/75 for the
        /// merged General-&-Reasoning bar.
        var thresholds: (good: Double, great: Double, normalizer: Double, suffix: String) {
            switch self {
            case .generalReasoning: return (50,  75,  100, "")
            case .code:             return (30,  65,  100, "")
            case .tool:             return (50,  70,  100, "")
            case .ifeval:           return (75,  88,  100, "")
            case .speed:            return (80, 180,  300, " t/s")
            }
        }

        /// Footer copy describing the source bench. Drives a
        /// speed methodology note and lets the tooltip surface benchmark
        /// version pedantry
        /// once at the bottom rather than on every row.
        var sourceDescription: String {
            switch self {
            case .generalReasoning: return "MMLU-Pro + GPQA Diamond (mean)"
            case .code:             return "LiveCodeBench v6"
            case .tool:             return "BFCL v3 / v4"
            case .ifeval:           return "IFEval (prompt-strict)"
            case .speed:            return "Measured long-context decode (hardware per row)"
            }
        }
    }

    /// Read the value for one axis. Returns ``nil`` for gaps; the
    /// renderer treats nil as a dashed track + em-dash value.
    func value(for axis: Axis) -> Double? {
        switch axis {
        case .generalReasoning: return generalReasoning
        case .code:             return code
        case .tool:             return tool
        case .ifeval:           return ifeval
        case .speed:            return speedTps
        }
    }
}

// MARK: - Catalog loader

/// Lazy single-shot loader that decodes `benchmark-scores.json`
/// from the SPM resource bundle (dev / test) or `Bundle.main`
/// (shipped .app). Mirrors the `CheetahLogo` / `Localizable.xcstrings`
/// dual-lookup pattern so a missing resource degrades to an empty
/// dictionary (every alias renders the dashed-bar tooltip) rather
/// than crashing the app on first picker hover.
enum BenchScoresCatalog {
    /// Single canonical map, populated once. An empty dictionary on
    /// load failure means every `lookup(alias:)` returns ``nil`` and
    /// the tooltip shows five dashed bars — survivable, never a crash.
    private static let cache: [String: BenchScores] = loadAll()

    /// Resource lookup for the JSON sidecar. Mirrors `CheetahLogo`:
    ///   1. ``Bundle.main`` (production .app — build.sh flattens the
    ///      JSON into `Contents/Resources/`).
    ///   2. SPM resource bundle next to the test runner / `swift run`
    ///      executable, probed via ``Bundle(url:)`` so a miss returns
    ///      `nil` instead of crashing.
    ///   3. The Swift Testing host (Xcode) sometimes copies SPM
    ///      resources beside the test bundle's executable — also
    ///      handled by the BundleFinder walk.
    static func lookup(alias: String) -> BenchScores? { cache[alias] }

    /// Every alias that has at least one published bench score —
    /// useful for unit tests that want to assert "the JSON parses
    /// and contains > 0 entries" without locking the count.
    static var allAliases: [String] { Array(cache.keys) }

    private static func loadAll() -> [String: BenchScores] {
        guard let url = resourceURL() else { return [:] }
        guard let data = try? Data(contentsOf: url) else { return [:] }
        guard let payload = try? JSONDecoder().decode(Payload.self, from: data) else {
            return [:]
        }
        return payload.models.mapValues { $0.toBenchScores() }
    }

    private static func resourceURL() -> URL? {
        if let url = Bundle.main.url(forResource: "benchmark-scores", withExtension: "json") {
            return url
        }
        let anchor = Bundle(for: BenchScoresBundleFinder.self).bundleURL.deletingLastPathComponent()
        let bundleURL = anchor.appendingPathComponent("Rapid_Rapid.bundle")
        if let bundle = Bundle(url: bundleURL),
           let url = bundle.url(forResource: "benchmark-scores", withExtension: "json") {
            return url
        }
        // Last resort: SPM's synthesised resource bundle next to the
        // test executable. ``Bundle.module`` would assert-crash on
        // miss, so we manually walk to it instead.
        let finderBundle = Bundle(for: BenchScoresBundleFinder.self)
        if let url = finderBundle.url(forResource: "benchmark-scores", withExtension: "json") {
            return url
        }
        return nil
    }

    /// Internal payload shape — matches the JSON sidecar's top-level
    /// object. Only the `models` map is needed at runtime; the
    /// `axes` block exists in the JSON for human readers but is
    /// duplicated in `BenchScores.Axis.thresholds`. The test suite
    /// pins both to catch drift.
    private struct Payload: Decodable {
        let models: [String: RawScores]
    }

    private struct RawScores: Decodable {
        let general_reasoning: Double?
        let general_reasoning_source: String?
        let general_reasoning_basis: Basis?
        let code: Double?
        let tool: Double?
        let ifeval: Double?
        let speed_tps: Double?

        struct Basis: Decodable {
            let mmlu_pro: Double?
            let gpqa_diamond: Double?
        }

        func toBenchScores() -> BenchScores {
            BenchScores(
                generalReasoning: general_reasoning,
                generalReasoningSource: general_reasoning_source,
                mmluPro: general_reasoning_basis?.mmlu_pro,
                gpqaDiamond: general_reasoning_basis?.gpqa_diamond,
                code: code,
                tool: tool,
                ifeval: ifeval,
                speedTps: speed_tps
            )
        }
    }

    /// Pure helper for tests + future regenerators: merge MMLU-Pro
    /// + GPQA Diamond into the single General-&-Reasoning value
    /// using the spec-locked rule (`arithmetic mean` when both are
    /// present; the single bench otherwise; `nil` when neither
    /// exists). Returns the merged value AND the source string the
    /// UI surfaces in the tooltip footer.
    static func mergeGeneralReasoning(
        mmluPro: Double?,
        gpqaDiamond: Double?
    ) -> (value: Double?, source: String?) {
        switch (mmluPro, gpqaDiamond) {
        case let (m?, g?):
            return ((m + g) / 2.0, "mean(mmlu_pro, gpqa_diamond)")
        case let (m?, nil):
            return (m, "mmlu_pro only")
        case let (nil, g?):
            return (g, "gpqa_diamond only")
        case (nil, nil):
            return (nil, nil)
        }
    }
}

/// Bundle anchor — identical pattern to ``CheetahLogo`` so the
/// resource lookup behaves the same way whether the caller is a
/// SwiftPM CLI test or the production .app.
private final class BenchScoresBundleFinder {}
