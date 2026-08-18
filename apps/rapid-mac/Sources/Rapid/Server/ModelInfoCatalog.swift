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
        return String(format: "~%.1f GB", approxRAMGB)
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
        // Qwen family — order matters (qwen3.8 / qwen3.6 before qwen3)
        if a.contains("qwen3.8") { return ("Qwen 3.8", 32_768) }
        if a.contains("qwen3.6") { return ("Qwen 3.6", 32_768) }
        if a.contains("qwen3.5") { return ("Qwen 3.5", 32_768) }
        if a.contains("qwen3-coder-next") { return ("Qwen 3 Coder Next", 32_768) }
        if a.contains("qwen3-vl") { return ("Qwen 3 VL", 32_768) }
        if a.contains("qwen3") { return ("Qwen 3", 32_768) }
        if a.contains("qwq") { return ("QwQ", 32_768) }
        if a.contains("qwen2.5") { return ("Qwen 2.5", 32_768) }
        if a.contains("qwen2") { return ("Qwen 2", 32_768) }
        // Llama
        if a.contains("llama-4.5") || a.contains("llama4.5") { return ("Llama 4.5", 131_072) }
        if a.contains("llama-4") || a.contains("llama4") { return ("Llama 4", 131_072) }
        if a.contains("llama-3.3") || a.contains("llama3.3") { return ("Llama 3.3", 131_072) }
        if a.contains("llama-3.2") || a.contains("llama3.2") { return ("Llama 3.2", 131_072) }
        if a.contains("llama-3.1") || a.contains("llama3.1") { return ("Llama 3.1", 131_072) }
        if a.contains("llama-3") || a.contains("llama3") { return ("Llama 3", 8_192) }
        // Gemma — long-context variants ship as separate repos and we
        // surface them as the same family display; we only assert the
        // stock window here.
        if a.contains("gemma-4") || a.contains("gemma4") { return ("Gemma 4", 8_192) }
        if a.contains("gemma-3n") || a.contains("gemma3n") { return ("Gemma 3n", 8_192) }
        if a.contains("gemma-3") || a.contains("gemma3") { return ("Gemma 3", 8_192) }
        if a.contains("gemma-2") || a.contains("gemma2") { return ("Gemma 2", 8_192) }
        // Mistral
        if a.contains("mistral-3") { return ("Mistral 3", 32_768) }
        if a.contains("mistral") { return ("Mistral", 32_768) }
        // GLM
        if a.contains("glm-5") { return ("GLM 5", 131_072) }
        if a.contains("glm-4.7") || a.contains("glm47") { return ("GLM 4.7", 131_072) }
        if a.contains("glm-4") || a.contains("glm4") { return ("GLM 4", 131_072) }
        // DeepSeek
        if a.contains("deepseek-v4") { return ("DeepSeek V4", 131_072) }
        if a.contains("deepseek-v3") { return ("DeepSeek V3", 131_072) }
        if a.contains("deepseek") { return ("DeepSeek", 32_768) }
        // Phi
        if a.contains("phi-4") || a.contains("phi4") { return ("Phi 4", 16_384) }
        if a.contains("phi-3") || a.contains("phi3") { return ("Phi 3", 4_096) }
        // SmolLM
        if a.contains("smollm3") { return ("SmolLM 3", 8_192) }
        if a.contains("smollm") { return ("SmolLM", 2_048) }
        // Hermes
        if a.contains("hermes-3") || a.contains("hermes3") { return ("Hermes 3", 131_072) }
        if a.contains("hermes") { return ("Hermes", 32_768) }
        // Bonsai (small / experimental)
        if a.contains("bonsai") { return ("Bonsai", 4_096) }
        return ("Unknown", nil)
    }
}
