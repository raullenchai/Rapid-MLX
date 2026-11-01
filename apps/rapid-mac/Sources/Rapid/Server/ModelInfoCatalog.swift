import Foundation

/// Compact descriptor for one alias surfaced in the "model info" popover
/// next to the picker. Combines what we already mine from the alias name
/// (`ModelSizing` for params / quant / weights) with a family-derived
/// context window so a user clicking the (i) button gets the full
/// picture without leaving the chat surface.
///
/// `contextWindow` is `nil` when we don't have a confident value for an
/// alias's family — the popover renders "—" rather than guessing 32k for
/// every unknown row. The cumulative-token meter (#928) treats `nil` as
/// "no meter, just the running count."
struct ModelInfo: Equatable, Sendable {
    let alias: String
    let hfRepo: String?
    let paramsBillions: Double?
    let bitsPerWeight: Int
    /// Approximate weight + runtime footprint in gibibytes, mirrors
    /// `ModelSizing.Footprint.totalGB`.
    let approxRAMGB: Double
    /// Context window in tokens, or `nil` if the family is unknown to
    /// the catalog. We deliberately don't fall back to a generic 4096
    /// because that would lie to the user about long-context Qwen3.5
    /// (32k) and Llama3.1 (128k) aliases.
    let contextWindow: Int?
    /// Short family label for the popover header, e.g. "Qwen 3.6".
    /// Inferred from the alias name; "Unknown" for anything not in
    /// the family table.
    let family: String

    /// Pretty-print params as e.g. "27B" or "0.6B" or "—" if unknown.
    var paramsLabel: String {
        guard let p = paramsBillions else { return "—" }
        if p >= 1 {
            // 27 → "27B", 27.5 → "27.5B"
            return p.truncatingRemainder(dividingBy: 1) == 0
                ? "\(Int(p))B"
                : String(format: "%.1fB", p)
        }
        // Sub-1B → "0.6B"
        return String(format: "%.1fB", p)
    }

    /// "4-bit" / "8-bit" / "16-bit".
    var quantLabel: String { "\(bitsPerWeight)-bit" }

    /// "32k" / "128k" / "—".
    var contextLabel: String {
        guard let ctx = contextWindow else { return "—" }
        if ctx >= 1024, ctx % 1024 == 0 {
            return "\(ctx / 1024)k"
        }
        return "\(ctx)"
    }

    /// "~3.4 GB" / "—" if params unknown.
    var ramLabel: String {
        guard paramsBillions != nil else { return "—" }
        return "~\(DownloadProgress.formatMemory(DownloadProgress.bytes(fromDouble: approxRAMGB * Double(1 << 30))))"
    }
}

/// Static lookup of context windows by alias family. The catalog is
/// intentionally hand-curated — there's no API to query an mlx-community
/// repo's `config.json` from inside the desktop app without spawning
/// rapid-mlx, and Llama 3.1's 128k vs. Llama 3's 8k is too important to
/// guess wrong. New families add a line here.
///
/// Sources:
/// - Qwen2.5 / 3 / 3.5 / 3.6: HF model cards (32k / 32k / 32k / 32k)
/// - Llama 3 / 3.1 / 3.2 / 3.3: 8k → 128k upgrade with 3.1
/// - Gemma 2 / 3: 8k → 8k (NOT 128k — the long-context variant is a
///   separate repo)
/// - GLM-4.5 / 4.7 / 5: 128k
/// - DeepSeek V3: 128k
/// - Phi-3 / 4: 4k / 16k
/// - Mistral 3: 32k
/// - Hermes 3: 128k
/// - Ornith 1.5: 256k
/// - SmolLM3: 8k
enum ModelInfoCatalog {
    /// Resolve an alias to a `ModelInfo`. `hfRepo` is optional and just
    /// flows through to the struct; the lookup is keyed off the alias
    /// string (lowercased) only.
    ///
    /// `serverContextWindow` is the value emitted by rapid-mlx ≥ 0.8.4's
    /// `/v1/models` `context_window` field — when present, it wins over
    /// the local family-table heuristic (the server reads the loaded
    /// engine's `max_position_embeddings`, the canonical source of
    /// truth). Issue #363. Older sidecars that don't emit the field
    /// pass `nil` here and the family table provides a defense-in-depth
    /// fallback so the popover / max-tokens slider still has a value.
    /// Pass `nil` from call sites that don't yet plumb the server
    /// profile through.
    static func info(
        for alias: String,
        hfRepo: String?,
        serverContextWindow: Int? = nil
    ) -> ModelInfo {
        let footprint = ModelSizing.estimate(alias: alias)
        let (family, fallbackCtx) = familyAndContext(for: alias)
        // Server value wins when present + positive. The "positive"
        // gate guards against a future server-side regression that
        // ever emits 0 / negative — we'd rather fall back to the
        // family heuristic than render "0k" in the popover.
        let ctx: Int?
        if let serverCtx = serverContextWindow, serverCtx > 0 {
            ctx = serverCtx
        } else {
            ctx = fallbackCtx
        }
        return ModelInfo(
            alias: alias,
            hfRepo: ModelCatalog.sanitizedHuggingFaceRepo(hfRepo),
            paramsBillions: footprint.paramsBillions,
            bitsPerWeight: footprint.bitsPerWeight,
            approxRAMGB: footprint.totalGB,
            contextWindow: ctx,
            family: family
        )
    }

    /// Issue #363 — per-family context-window fallback for the desktop
    /// when the rapid-mlx sidecar predates the cross-repo fix and
    /// doesn't emit `context_window` on `/v1/models`. Returns `nil`
    /// for families the catalog has never heard of; the popover then
    /// renders "—" rather than guessing 4096 for a model that might
    /// be 128 KB long-context.
    ///
    /// Thin wrapper over `familyAndContext` so call sites that only
    /// need the integer don't have to discard the family label tuple.
    /// Kept separate so a future refactor that wants to split the
    /// fallback table from the display label can do so without
    /// touching every caller.
    static func contextWindowFallback(forAlias alias: String) -> Int? {
        familyAndContext(for: alias).contextWindow
    }

    /// Best-effort family display + context window. Returns
    /// `("Unknown", nil)` if no pattern matches.
    static func familyAndContext(for alias: String) -> (family: String, contextWindow: Int?) {
        let boundedAlias = alias.utf8.count > ModelCatalog.maxAliasBytes
            ? String(alias.prefix(ModelCatalog.maxAliasBytes))
            : alias
        let a = boundedAlias.lowercased()
        // Numeric version keys must not match the start of a longer number.
        func has(_ key: String) -> Bool {
            guard key.last?.isNumber == true else { return a.contains(key) }
            var start = a.startIndex
            while let range = a.range(of: key, range: start..<a.endIndex) {
                if range.upperBound == a.endIndex || !a[range.upperBound].isNumber {
                    return true
                }
                start = a.index(after: range.lowerBound)
            }
            return false
        }
        // Qwen family — order matters (qwen3.8 / qwen3.6 before qwen3)
        if has("qwen3.8") { return ("Qwen 3.8", 32_768) }
        if has("qwen3.6") { return ("Qwen 3.6", 32_768) }
        if has("qwen3.5") { return ("Qwen 3.5", 32_768) }
        if has("qwen3-coder-next") { return ("Qwen 3 Coder Next", 32_768) }
        if has("qwen3-vl") { return ("Qwen 3 VL", 32_768) }
        if has("qwen3") { return ("Qwen 3", 32_768) }
        if has("qwq") { return ("QwQ", 32_768) }
        if has("qwen2.5") { return ("Qwen 2.5", 32_768) }
        if has("qwen2") { return ("Qwen 2", 32_768) }
        // K2 Horizon dense checkpoints advertise a 524k context window. The
        // loaded server's memory-aware effective limit still wins when present.
        if has("k2-horizon") { return ("K2 Horizon", 524_288) }
        // Llama
        if has("llama-4.5") || has("llama4.5") { return ("Llama 4.5", 131_072) }
        if has("llama-4") || has("llama4") { return ("Llama 4", 131_072) }
        if has("llama-3.3") || has("llama3.3") { return ("Llama 3.3", 131_072) }
        if has("llama-3.2") || has("llama3.2") { return ("Llama 3.2", 131_072) }
        if has("llama-3.1") || has("llama3.1") { return ("Llama 3.1", 131_072) }
        if has("llama-3") || has("llama3") { return ("Llama 3", 8_192) }
        // Gemma — long-context variants ship as separate repos and we
        // surface them as the same family display; we only assert the
        // stock window here.
        if has("gemma-4") || has("gemma4") { return ("Gemma 4", 8_192) }
        if has("gemma-3n") || has("gemma3n") { return ("Gemma 3n", 8_192) }
        if has("gemma-3") || has("gemma3") { return ("Gemma 3", 8_192) }
        if has("gemma-2") || has("gemma2") { return ("Gemma 2", 8_192) }
        // Mistral
        if has("mistral-3") { return ("Mistral 3", 32_768) }
        if has("mistral") { return ("Mistral", 32_768) }
        // GLM
        if has("glm-5") { return ("GLM 5", 131_072) }
        if has("glm-4.7") || has("glm47") { return ("GLM 4.7", 131_072) }
        if has("glm-4") || has("glm4") { return ("GLM 4", 131_072) }
        // DeepSeek
        if has("deepseek-v4.1") || has("deepseek-v41") { return ("DeepSeek V4.1", 131_072) }
        if has("deepseek-v4") { return ("DeepSeek V4", 131_072) }
        if has("deepseek-v3") { return ("DeepSeek V3", 131_072) }
        if has("deepseek") { return ("DeepSeek", 32_768) }
        // Phi
        if has("phi-4") || has("phi4") { return ("Phi 4", 16_384) }
        if has("phi-3") || has("phi3") { return ("Phi 3", 4_096) }
        // SmolLM
        if has("smollm3") { return ("SmolLM 3", 8_192) }
        if has("smollm") { return ("SmolLM", 2_048) }
        // Hermes
        if has("hermes-3") || has("hermes3") { return ("Hermes 3", 131_072) }
        if has("hermes") { return ("Hermes", 32_768) }
        // Ornith-1.5 official MLX checkpoints advertise 262144 through
        // ``text_config.max_position_embeddings`` in both the 9B dense and
        // 35B-A3B MoE configs. The live server value still wins when present.
        if a.hasPrefix("ornith-1.5-") { return ("Ornith 1.5", 262_144) }
        // Live server context overrides these family labels when available.
        if a == "lfm3-vl-2b-8bit" { return ("LFM3 VL", 128_000) }
        if a.hasPrefix("lfm3-vl-") { return ("LFM3 VL", nil) }
        if a == "lfm3-1.7b-8bit" { return ("LFM3", 262_144) }
        if a.hasPrefix("lfm3-") { return ("LFM3", nil) }
        if a.hasPrefix("lfm2.5-") { return ("LFM2.5", nil) }
        // Bonsai (small / experimental)
        if has("bonsai2") { return ("Bonsai 2", 4_096) }
        if has("bonsai") { return ("Bonsai", 4_096) }
        return ("Unknown", nil)
    }
}
