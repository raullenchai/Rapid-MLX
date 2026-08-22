import Foundation

/// Script-aware token estimation.
///
/// ## Why not `characters / 4`
///
/// That rule of thumb is OpenAI's, published for ENGLISH, and it is close
/// enough there. It is badly wrong for CJK text, where a single character
/// routinely costs a whole token — so the same 24,000 characters that cost
/// ~6,000 tokens of English cost ~13,000 tokens of Chinese.
///
/// Under-counting is the dangerous direction. Every consumer of an estimate
/// here is deciding whether something FITS: the context trim decides what to
/// drop, and the attachment preview decides how much document to ship. An
/// estimate 2x low means the trim keeps an over-window body (the server then
/// rejects it, or the model RoPE-extrapolates and quietly degrades) and the
/// preview silently sends twice the intended prompt, which is paid for in
/// prefill time on every turn.
///
/// ## Calibration
///
/// Constants were fitted by least squares against the Qwen3.5 tokenizer
/// (`tokenizer.json` from `mlx-community/Qwen3.5-4B-MLX-4bit`) over 4,000-char
/// chunks of two real extracted documents — a 302-page Chinese technical book
/// and an English quarterly report. The fit gave 0.613 tokens/char for CJK and
/// 0.400 for everything else, with 5.6% mean error.
///
/// Note how far 0.400 is from the familiar `chars / 4` (= 0.25). That rule is
/// quoted for clean English PROSE, and real extracted documents are not that:
/// they carry numbers, punctuation, headings, table cells and page markers,
/// all of which tokenize far worse than running text. Using 0.25 on real
/// documents under-counted by ~1.6x even for pure ASCII.
///
/// The shipped constants round the fit UP. A deliberate bias: this estimator
/// decides what FITS, so over-counting costs a little headroom while
/// under-counting costs a rejected request or a silently doubled prefill.
///
/// Exact counts would need the model's own tokenizer, which the app does not
/// ship and which differs per model. Being right per-script to within tens of
/// percent is the goal — the previous single global constant was not.
enum TokenEstimate {
    /// Han ideographs, kana, and CJK punctuation. Fitted 0.613.
    static let cjkTokensPerCharacter = 0.65
    /// Hangul syllables. Measured 0.363 in isolation; the document fit had no
    /// Korean to contribute, so this keeps the isolated measurement plus margin.
    static let hangulTokensPerCharacter = 0.45
    /// Everything else — Latin, digits, whitespace, symbols. Fitted 0.400.
    static let defaultTokensPerCharacter = 0.42

    /// Estimated tokens in `text`, summed per character class.
    ///
    /// Iterates unicode scalars rather than Characters: a grapheme cluster can
    /// hold several scalars (an emoji with modifiers, a decomposed syllable)
    /// and classifying only its first would misprice the rest.
    static func tokens(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var cjk = 0
        var hangul = 0
        var other = 0
        for scalar in text.unicodeScalars {
            if isCJK(scalar) {
                cjk += 1
            } else if isHangul(scalar) {
                hangul += 1
            } else {
                other += 1
            }
        }
        let total = Double(cjk) * cjkTokensPerCharacter
            + Double(hangul) * hangulTokensPerCharacter
            + Double(other) * defaultTokensPerCharacter
        // Any non-empty text costs at least one token.
        return max(1, Int(total.rounded(.up)))
    }

    /// Longest prefix of `text` estimated to fit within `tokenBudget`.
    ///
    /// Walks scalar by scalar and stops at the budget, so the cut respects the
    /// same per-script weights as ``tokens(in:)`` — a Chinese document yields
    /// proportionally fewer characters than an English one for the same token
    /// budget, which is the entire point.
    ///
    /// The cut is snapped back to a Character boundary: slicing mid-grapheme
    /// would corrupt the text (and can't be expressed as a String index the
    /// caller could reuse).
    static func prefix(_ text: String, withinTokens tokenBudget: Int) -> String {
        guard tokenBudget > 0 else { return "" }
        guard tokens(in: text) > tokenBudget else { return text }

        var spent = 0.0
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            var cost = 0.0
            for scalar in character.unicodeScalars {
                if isCJK(scalar) {
                    cost += cjkTokensPerCharacter
                } else if isHangul(scalar) {
                    cost += hangulTokensPerCharacter
                } else {
                    cost += defaultTokensPerCharacter
                }
            }
            if spent + cost > Double(tokenBudget) { break }
            spent += cost
            index = text.index(after: index)
        }
        return String(text[text.startIndex..<index])
    }

    // MARK: - Classification

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3000...0x303F,      // CJK symbols and punctuation
             0x3040...0x309F,      // Hiragana
             0x30A0...0x30FF,      // Katakana
             0x3400...0x4DBF,      // CJK ext A
             0x4E00...0x9FFF,      // CJK unified ideographs
             0xF900...0xFAFF,      // CJK compatibility ideographs
             0xFF00...0xFFEF,      // Halfwidth and fullwidth forms
             0x20000...0x2FA1F:    // CJK ext B–F, compatibility supplement
            return true
        default:
            return false
        }
    }

    private static func isHangul(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x1100...0x11FF,      // Hangul jamo
             0x3130...0x318F,      // Hangul compatibility jamo
             0xA960...0xA97F,      // Hangul jamo extended-A
             0xAC00...0xD7AF:      // Hangul syllables
            return true
        default:
            return false
        }
    }
}
