import Foundation

/// RAM footprint estimator + fit classifier for a rapid-mlx alias on
/// a given Mac. The picker uses this to push a "Recommended" section
/// to the top of the menu and grey out anything that won't fit.
///
/// We follow whichllm's compatibility model: estimate the model's
/// weight bytes from its parameter count and quantization, add a
/// fixed runtime overhead (Python + uvicorn + mlx framework), then
/// a KV-cache reserve. A model fits when total < usable RAM.
enum ModelSizing {
    /// Per-Mac classification of "will this even run?". The picker
    /// renders ``.recommended`` first, then ``.cached`` second, marks
    /// ``.borderline`` with a yellow warning, and disables ``.tooBig``.
    enum Fit: String, Sendable, Equatable {
        /// Comfortable headroom for KV cache + chat context.
        case recommended
        /// Will run but the KV cache budget is tight; flag with a
        /// yellow icon and tooltip rather than block.
        case borderline
        /// Estimated footprint exceeds 80% of RAM — will swap or
        /// OOM. Picker disables and surfaces "Needs X GB, your Mac
        /// has Y GB".
        case tooBig
    }

    /// Estimate of how many gibibytes the alias needs at run time.
    /// ``weightsGB`` is the dominant term; the rest is fixed runtime
    /// floor (rapid-mlx + Python + framework) plus a KV reserve.
    struct Footprint: Sendable, Equatable {
        let alias: String
        /// Parameters in billions parsed from the alias name; ``nil``
        /// if we couldn't find a number (e.g. an unfamiliar custom
        /// alias). Treated as "unknown" — picker leaves it ungated.
        let paramsBillions: Double?
        /// Effective bits per weight (2, 3, 4, 6, 8, or 16). Default 4
        /// for any mlx-community alias since they ship 4-bit unless the
        /// name explicitly says otherwise. Sub-4-bit values cover the
        /// low-bit / ternary MLX builds (e.g. ``bonsai-1.7b-2bit``),
        /// which the estimator previously rounded up to 4-bit and so
        /// over-stated ~2x. #520.
        let bitsPerWeight: Int
        /// Estimated weight-tensor footprint at the given bit-width.
        let weightsGB: Double
        /// Floor: Python + rapid-mlx + mlx runtime + tokenizer +
        /// adapters. Roughly constant across aliases.
        let baseOverheadGB: Double
        /// Reserve we want left over for KV cache + the prompt the
        /// user actually wants to send. 8 GB at 4096 ctx for mid-
        /// size models; we keep this conservative because OOM
        /// during generation is a worse UX than declining to load.
        let kvReserveGB: Double

        /// Total RAM needed for a comfortable run.
        var totalGB: Double {
            weightsGB + baseOverheadGB + kvReserveGB
        }
    }

    // MARK: - Estimate

    /// Compute a footprint estimate from the alias name.
    ///
    /// ``alias`` is parsed for two pieces of information:
    ///   * Parameter count: the largest ``\d+(\.\d+)?B?`` run that
    ///     looks like a size, e.g. ``qwen3.6-27b`` → 27, ``gemma-4-12b`` → 12.
    ///   * Quantization: reads an explicit ``16/8/6/4/3/2bit`` (or
    ///     ``ternary``) tag; defaults to 4-bit if not specified.
    ///     mlx-community ships overwhelmingly in 4-bit so the default
    ///     is safe.
    static func estimate(alias: String) -> Footprint {
        let params = parseParamsBillions(alias)
        let bits = parseBitsPerWeight(alias)
        // Weight tensors: params × bytes-per-weight. Each figure folds in
        // the mlx group-quant scale/zero-point overhead (a per-group
        // scale+bias whose relative cost grows as the bit width shrinks).
        //  16-bit: 2.0  byte/param
        //   8-bit: 1.05 (1.0  + ~5%)
        //   6-bit: 0.80 (0.75 + ~7%)
        //   4-bit: 0.55 (0.5  + ~10%)
        //   3-bit: 0.42 (0.375 + ~12%)
        //   2-bit / ternary: 0.28 — measured against bonsai-1.7b-2bit
        //     (484 MB weights / 1.7 B params). Ternary (1.58-bit) MLX
        //     builds pack to 2-bit storage, so they land here too.
        let bytesPerParam: Double
        switch bits {
        case 16: bytesPerParam = 2.0
        case 8: bytesPerParam = 1.05
        case 6: bytesPerParam = 0.80
        case 4: bytesPerParam = 0.55
        case 3: bytesPerParam = 0.42
        case 2: bytesPerParam = 0.28
        default: bytesPerParam = 0.55
        }
        let weightsGB = (params ?? 0) * bytesPerParam
        return Footprint(
            alias: alias,
            paramsBillions: params,
            bitsPerWeight: bits,
            weightsGB: weightsGB,
            baseOverheadGB: 1.2,
            kvReserveGB: kvReserve(forParams: params)
        )
    }

    /// Default budget for all engines held by the desktop sidecar. Reuses the
    /// same 80%-of-physical usable pool as picker classification so startup
    /// warnings and runtime eviction do not disagree about available memory.
    static func residentMemoryCeilingGB(on hardware: MacHardware) -> Double {
        max(4, floor(hardware.usableRAMGB))
    }

    /// Residency charge sent to the server. Downloaded bytes are better
    /// evidence for image aliases whose names carry no parameter count; add a
    /// modest runtime margin and keep the ordinary model estimate as a floor.
    static func residentEstimateGB(alias: String, sizeText: String? = nil) -> Double {
        let heuristic = estimate(alias: alias).totalGB
        guard let bytes = ModelCacheActions.parseSizeBytes(sizeText), bytes > 0
        else { return heuristic }
        let diskGiB = Double(bytes) / Double(UInt64(1) << 30)
        return max(heuristic, diskGiB * 1.25 + 0.5)
    }

    /// Pick a KV-reserve target proportional to model size — bigger
    /// models have bigger per-token cache cost. The picker is mostly
    /// concerned with order-of-magnitude, not precision.
    static func kvReserve(forParams params: Double?) -> Double {
        guard let p = params else { return 2.0 }
        if p < 4 { return 1.5 }
        if p < 10 { return 2.5 }
        if p < 25 { return 4.0 }
        return 6.0
    }

    // MARK: - Fit

    /// Classify a footprint against a host's usable RAM pool.
    ///
    /// Bands chosen against ``hardware.usableRAMGB`` (80% of total),
    /// calibrated by the empirical "gemma-4-12b crashes my 18 GB
    /// MacBook" report:
    ///   * ≤ 50% of usable → ``.recommended`` (comfortable KV budget)
    ///   * 50–75% of usable → ``.borderline`` (loads, but tight under
    ///     long contexts — MLX Metal allocation_limit defaults to 90 %
    ///     of physical RAM, so the loading-time spike eats into the
    ///     KV budget hard)
    ///   * > 75% of usable → ``.tooBig`` (the model + loading spike
    ///     + Python overhead routinely exceeds the OS jetsam ceiling
    ///     on smaller Macs)
    static func classify(_ footprint: Footprint, on hardware: MacHardware) -> Fit {
        // Unknown params → leave it ungated so a user typing a custom
        // alias doesn't get a false "won't fit" warning.
        guard footprint.paramsBillions != nil else { return .borderline }
        let needed = footprint.totalGB
        let pool = hardware.usableRAMGB
        if pool <= 0 { return .borderline }
        let ratio = needed / pool
        if ratio <= 0.50 { return .recommended }
        if ratio <= 0.75 { return .borderline }
        return .tooBig
    }

    /// The largest total footprint that still classifies as something other
    /// than ``Fit/tooBig`` on this host — i.e. the actual ceiling ``classify``
    /// enforces, in GB.
    ///
    /// Exists so a screen explaining a WON'T FIT verdict can quote the real
    /// limit instead of implying it is ``MacHardware/usableRAMGB``. Those are
    /// not the same number, and the gap is where a misleading explanation
    /// lives: on a 32 GB Mac the usable pool is 25.6 GB, but a 21 GB model is
    /// still refused, so "needs 21 GB, 25.6 GB usable" reads as a
    /// contradiction unless the 75% headroom is named. Derived from the band
    /// above rather than restated, so the two cannot drift apart.
    static func largestFittingGB(on hardware: MacHardware) -> Double {
        max(0, hardware.usableRAMGB * 0.75)
    }

    // MARK: - Live memory safety (pre-load guard)

    /// Verdict for loading a model RIGHT NOW given live memory use.
    ///
    /// Unlike ``classify`` — which bands a footprint against a STATIC
    /// 80%-of-total estimate and so only knows "does this model fit
    /// this Mac at all" — this projects the footprint on top of what
    /// is *actually in use this second*. A model that "fits the Mac"
    /// is still flagged when other apps — or a model already loaded in
    /// this one — have eaten the free RAM. That gap
    /// is the reported near-crash: gemma-4-12b classifies ``.fits`` on
    /// a larger Mac, yet loading it with little free RAM pushed unified
    /// memory past the danger line.
    ///
    /// #324: unified memory past ~85% of total can trip the iBoot
    /// AMCC async-abort firmware path and **kernel-panic the whole
    /// machine** rather than raise a userspace OOM — so ``.unsafe`` is
    /// pinned to that ~85% danger line and is a must-confirm block,
    /// never a silent proceed.
    enum MemorySafety: String, Sendable, Equatable {
        /// Projected use < 75% of total — comfortable.
        case safe
        /// 75-85% — will load but risks swap / stalls; warn.
        case tight
        /// ≥ 85% — at/over the kernel-panic danger line; block + confirm.
        case unsafe
    }

    /// Project ``footprint`` onto ``usedBytes`` and bucket the result.
    /// Returns ``.safe`` when the numbers are unreadable or the param
    /// count is unknown — never block a load on missing data (the
    /// loader still surfaces genuine failures downstream).
    static func memorySafety(
        footprint: Footprint,
        usedBytes: UInt64,
        totalBytes: UInt64
    ) -> MemorySafety {
        guard totalBytes > 0, footprint.paramsBillions != nil else { return .safe }
        let gib = Double(1 << 30)
        let footprintBytes = footprint.totalGB * gib
        let projected = (Double(usedBytes) + footprintBytes) / Double(totalBytes)
        if projected >= 0.85 { return .unsafe }
        if projected >= 0.75 { return .tight }
        return .safe
    }

    /// A pending "this load may exhaust memory" prompt. Held on
    /// ``ServerManager`` as observable state so any model-start path
    /// (picker, first message, auto-restart, quickstart) surfaces the
    /// SAME confirmation without each call site re-implementing it.
    /// Copy lives here (not in a view) so ``ModelSizingTests`` can pin
    /// it without a SwiftUI host — same pattern as the tooBig alert.
    struct MemoryWarning: Equatable, Sendable, Identifiable {
        let id = UUID()
        let alias: String
        let hfPath: String?
        let isAutoRespawn: Bool
        let severity: MemorySafety
        /// Estimated GB the model needs.
        let footprintGB: Double
        /// GB free at the moment the load was attempted.
        let freeGB: Double
        /// Total unified memory. Needed because the guard is based on
        /// projected utilisation, not on footprint-versus-free alone.
        var totalGB: Double = 0

        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.alias == rhs.alias
                && lhs.hfPath == rhs.hfPath
                && lhs.isAutoRespawn == rhs.isAutoRespawn
                && lhs.severity == rhs.severity
                && lhs.footprintGB == rhs.footprintGB
                && lhs.freeGB == rhs.freeGB
                && lhs.totalGB == rhs.totalGB
        }

        var title: String {
            switch severity {
            case .unsafe:
                return "\(alias) may crash your Mac right now"
            case .tight, .safe:
                return "\(alias) is a tight fit right now"
            }
        }

        var message: String {
            let need = max(1, Int(footprintGB.rounded()))
            let free = max(0, Int(freeGB.rounded()))
            guard totalGB > 0 else {
                let facts = "\(alias) needs about \(need) GB, and about \(free) GB is free right now."
                return facts + " Close some apps or pick a smaller model before loading it."
            }
            let usedGB = max(0, totalGB - freeGB)
            let projectedPercent = Int(((usedGB + footprintGB) / totalGB * 100).rounded())
            let threshold = severity == .unsafe ? 85.0 : 75.0
            let toFree = max(0, usedGB + footprintGB - threshold / 100 * totalGB)
            let freeAction = max(1, Int(toFree.rounded(.up)))
            let facts = "Loading it would put memory use at about \(projectedPercent)% of \(Int(totalGB.rounded())) GB."
            switch severity {
            case .unsafe:
                return facts + " Past about 85%, macOS may freeze or restart. Free about \(freeAction) GB by closing some apps, or pick a smaller model."
            case .tight, .safe:
                return facts + " It should load but may stall under longer chats. Consider closing some apps or picking a smaller model."
            }
        }

        /// The confirm button title — worded to match the risk.
        var confirmTitle: String {
            severity == .unsafe ? "Load anyway (risky)" : "Load anyway"
        }
    }

    // MARK: - Lineage ranking

    /// Lineage score borrowed from whichllm's ``MODEL_LINEAGE_VERSIONS``.
    /// Higher score = newer generation of its family; the picker uses
    /// this to surface the *newest* fitting model from each family
    /// first in the Recommended section. Without it, ``qwen2.5`` and
    /// ``qwen3`` jostle for the same slot at the top.
    static func lineageScore(_ alias: String) -> Int {
        let a = alias.lowercased()
        // Qwen family (most populated alias group in rapid-mlx)
        if a.contains("qwen3.6") { return 70 }
        if a.contains("qwen3.5") { return 60 }
        if a.contains("qwen3-coder-next") { return 60 }
        if a.contains("qwen3-vl") { return 55 }
        if a.contains("qwen3") { return 50 }
        if a.contains("qwq") { return 40 }
        if a.contains("qwen2.5") { return 30 }
        if a.contains("qwen2") { return 20 }
        // Llama family
        if a.contains("llama-4.5") || a.contains("llama4.5") { return 50 }
        if a.contains("llama-4") || a.contains("llama4") { return 40 }
        if a.contains("llama-3.3") || a.contains("llama3.3") { return 35 }
        if a.contains("llama-3.2") || a.contains("llama3.2") { return 32 }
        if a.contains("llama-3.1") || a.contains("llama3.1") { return 30 }
        if a.contains("llama-3") || a.contains("llama3") { return 25 }
        // Gemma family
        if a.contains("gemma-4") || a.contains("gemma4") { return 40 }
        if a.contains("gemma-3n") || a.contains("gemma3n") { return 35 }
        if a.contains("gemma-3") || a.contains("gemma3") { return 30 }
        if a.contains("gemma-2") || a.contains("gemma2") { return 20 }
        // Mistral family
        if a.contains("mistral-3") { return 30 }
        if a.contains("mistral") { return 25 }
        // Other notable families
        if a.contains("smollm3") { return 30 }
        if a.contains("hermes-3") || a.contains("hermes3") { return 30 }
        if a.contains("deepseek-v4") { return 40 }
        if a.contains("deepseek-v3") { return 35 }
        if a.contains("glm-5") { return 40 }
        if a.contains("glm-4.7") || a.contains("glm47") { return 35 }
        if a.contains("phi-4") { return 35 }
        if a.contains("phi-3") { return 30 }
        return 10
    }

    // MARK: - Alias parsers

    /// Extract the parameter count in billions from an alias.
    /// Handles ``qwen3.6-27b``, ``llama-3.1-8b-8bit``, ``smollm3-3b``,
    /// ``gemma-4-12b-qat`` — pulls the largest number followed by ``b``.
    /// Compiled once. ``NSRegularExpression(pattern:)`` runs a full ICU
    /// pattern compile on every call, and this is reached once per alias per
    /// SwiftUI body pass (``ModelPickerBar`` rebuilds its ``Menu`` content
    /// eagerly, even closed). Profiling a 1920-character stream put 48% of
    /// the main thread inside ``uregex_open`` — the pattern is a literal, so
    /// compiling it per call bought nothing.
    private static let paramsBillionsRegex = try? NSRegularExpression(
        pattern: #"(\d+(?:\.\d+)?)\s*[bB]\b"#
    )

    private static let bitsPerWeightRegex = try? NSRegularExpression(
        pattern: #"(?<![0-9.])(\d{1,2})-?bit\b"#
    )

    static func parseParamsBillions(_ alias: String) -> Double? {
        // Match patterns like "27b", "8.5b", "0.6b", "27B" — case
        // insensitive. We pick the LARGEST such match because aliases
        // like ``qwen3-coder-next-80b-a3b`` carry both the full-weight
        // size (80B — the one that matters for RAM) and the active
        // params (3B — what the model uses per token).
        guard let regex = paramsBillionsRegex else { return nil }
        let nsAlias = alias as NSString
        let matches = regex.matches(in: alias, range: NSRange(location: 0, length: nsAlias.length))
        var best: Double? = nil
        for m in matches {
            guard m.numberOfRanges >= 2 else { continue }
            let captured = nsAlias.substring(with: m.range(at: 1))
            guard let v = Double(captured), v >= 0.1, v <= 1000 else { continue }
            if best == nil || v > (best ?? 0) { best = v }
        }
        return best
    }

    /// Parse the quantization bit width from an alias. 4 is the
    /// default for mlx-community models; 8 only when explicitly
    /// labelled; 16 reserved for hypothetical full-precision aliases;
    /// 6 / 3 / 2 for the sub-4-bit and ternary MLX builds we now ship
    /// (e.g. ``bonsai-1.7b-2bit``). Order matters: the wider tags are
    /// tested first because a narrower substring can appear inside a
    /// wider one ("16bit" contains "6bit"). #520.
    static func parseBitsPerWeight(_ alias: String) -> Int {
        let lower = alias.lowercased()
        // Full-precision float markers.
        if lower.contains("bf16") || lower.contains("fp16") { return 16 }
        // Ternary (1.58-bit) MLX builds pack to 2-bit storage. Match the
        // word up front so "1.58bit-ternary" resolves to ternary instead
        // of being read as 8-bit by the embedded "…8bit" run.
        if lower.contains("ternary") { return 2 }
        // Explicit "<N>bit" / "<N>-bit" token, delimiter-bounded so a
        // wider tag can't be read as a narrower substring: plain
        // ``contains`` parsed "16-bit" as 6-bit ("16-bit" contains
        // "6-bit") and "1.58bit" as 8-bit. The leading lookbehind
        // rejects a digit/dot immediately before the width; the trailing
        // ``\b`` closes the token. #520.
        //
        // Compiled once — see ``paramsBillionsRegex``.
        if let regex = Self.bitsPerWeightRegex {
            let ns = lower as NSString
            if let match = regex.firstMatch(in: lower, range: NSRange(location: 0, length: ns.length)),
               match.numberOfRanges >= 2,
               let width = Int(ns.substring(with: match.range(at: 1))) {
                switch width {
                case 2, 3, 4, 6, 8, 16: return width
                default: return 4  // unrecognised width → safe 4-bit default
                }
            }
        }
        return 4
    }
}
