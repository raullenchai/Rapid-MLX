import Foundation

/// Pure classification of a rapid-mlx alias into a **brand** (for the
/// coloured monogram tile in the Models surface) and a **model type**
/// (chat vs vision, for the row's type glyph). Issue #507.
///
/// Why this doesn't reuse ``ModelInfoCatalog.familyAndContext``: that
/// table is tuned for the *context-window* lookup and returns
/// `"Unknown"` for aliases whose family name isn't a substring of the
/// alias — most notably `gpt-oss-*` (no "gpt" branch) and `devstral-*`
/// (doesn't contain "mistral"). The brand tile must still colour those
/// correctly, so brand detection is its own alias-keyed matcher here.
/// Kept free of SwiftUI so every branch is unit-testable without a view
/// host; the colour + gradient mapping lives in ``BrandIcon`` (the view
/// layer) keyed off the ``ModelBrand`` case this returns.
enum ModelType: String, Equatable, Sendable {
    /// Text-in / text-out chat model.
    case chat
    /// Accepts image input (a `-vl` / vision-language alias).
    case vision
}

/// The brand families we paint a distinct monogram tile for. `other`
/// is the honest catch-all — it renders a neutral tile with a
/// 2-letter fallback monogram derived from the alias, never a wrong
/// brand colour.
enum ModelBrand: String, CaseIterable, Equatable, Sendable {
    case qwen
    case llama
    case gemma
    case gptOss
    case deepseek
    case mistral
    case phi
    case glm
    case smollm
    case hermes
    case ornith
    case other

    /// The 2-character tile monogram for the recognised brands. `other`
    /// has no fixed monogram — callers use ``ModelBrandStyle.monogram``
    /// which derives a per-alias fallback.
    var monogram: String {
        switch self {
        case .qwen:     return "Qw"
        case .llama:    return "Ll"
        case .gemma:    return "Ge"
        case .gptOss:   return "GO"
        case .deepseek: return "DS"
        case .mistral:  return "Mi"
        case .phi:      return "Ph"
        case .glm:      return "GL"
        case .smollm:   return "Sm"
        case .hermes:   return "He"
        case .ornith:   return "Or"
        case .other:    return "?"
        }
    }
}

enum ModelBrandStyle {
    /// Classify an alias into its brand. Order matters where one token
    /// is a substring of another (`devstral`/`ministral` before the
    /// bare `mistral` check is unnecessary since we match the specific
    /// tokens, but `gpt-oss` is checked before any generic fallthrough).
    static func brand(forAlias alias: String) -> ModelBrand {
        let a = alias.lowercased()
        if a.contains("qwen") || a.contains("qwq") { return .qwen }
        if a.contains("gpt-oss") || a.contains("gptoss") { return .gptOss }
        if a.contains("llama") { return .llama }
        if a.contains("gemma") { return .gemma }
        if a.contains("deepseek") { return .deepseek }
        // Mistral house: Mistral, Devstral, Ministral, Magistral, Codestral.
        if a.contains("mistral") || a.contains("devstral")
            || a.contains("ministral") || a.contains("magistral")
            || a.contains("codestral") { return .mistral }
        if a.contains("phi-") || a.contains("phi4") || a.contains("phi3") { return .phi }
        if a.contains("glm") { return .glm }
        if a.contains("smollm") { return .smollm }
        if a.contains("hermes") { return .hermes }
        if a.hasPrefix("ornith-1.5-") { return .ornith }
        return .other
    }

    /// The monogram to paint on the tile. Recognised brands use their
    /// fixed 2-letter mark; `other` derives the first two alphabetic
    /// characters of the alias, upper-cased (e.g. `bonsai-…` → "BO"),
    /// so an unlisted family still gets a legible, non-misleading tile.
    static func monogram(forAlias alias: String) -> String {
        let brand = brand(forAlias: alias)
        if brand != .other { return brand.monogram }
        let letters = alias.filter { $0.isLetter }
        guard let first = letters.first else { return "?" }
        if letters.count >= 2 {
            let second = letters[letters.index(letters.startIndex, offsetBy: 1)]
            return String([first, second]).uppercased()
        }
        return String(first).uppercased()
    }

    /// Chat vs vision, from the curated alias family. Desktop launches every
    /// alias classified as vision with ``--mllm`` so image input is ready when
    /// the composer appears; this classification is therefore a launch
    /// contract, not merely a decorative glyph.
    ///
    /// Keep the explicit text-only exceptions aligned with server alias pins.
    /// In particular, the 4-bit Qwen3.5 4B alias currently carries
    /// ``is_text_only: true`` in ``aliases.json`` and the server intentionally
    /// rejects ``--mllm`` for it. Its 6/8-bit siblings are not pinned and keep
    /// their checkpoint's vision tower.
    static func modelType(forAlias alias: String) -> ModelType {
        let a = alias.lowercased()
        if a == "qwen3.5-4b-4bit"
            || a == "qwen3.5-122b-mxfp4"
            || a.contains("gemma3-1b") { return .chat }
        if a.contains("-vl-") || a.hasSuffix("-vl") || a.contains("vl-")
            || a.contains("-vision") || a.contains("qwen3.5-")
            || a.contains("qwen3.6-") || a.contains("qwen3.8-")
            || a.contains("gemma3-") || a.contains("gemma-3n-")
            || a.contains("gemma-4-") { return .vision }
        return .chat
    }

    static func supportsImageInput(forAlias alias: String) -> Bool {
        modelType(forAlias: alias) == .vision
    }

    /// A human-readable family name for the row's meta line. Prefers
    /// ``ModelInfoCatalog.familyAndContext`` (rich version-aware names
    /// like "Qwen 3.6") but overrides the two aliases it labels
    /// "Unknown" — `gpt-oss` and `devstral` — with a correct family so
    /// the caption never reads "Unknown · 8-bit" for a shipped model.
    static func displayFamily(forAlias alias: String) -> String {
        let a = alias.lowercased()
        if a.contains("gpt-oss") || a.contains("gptoss") { return "GPT-OSS" }
        if a.contains("devstral") { return "Devstral" }
        let family = ModelInfoCatalog.familyAndContext(for: alias).family
        return family == "Unknown" ? "Model" : family
    }
}
