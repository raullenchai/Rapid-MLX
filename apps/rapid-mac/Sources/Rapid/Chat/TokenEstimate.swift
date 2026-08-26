import Foundation

/// Conservative script-aware token estimates for prompt budgeting.
enum TokenEstimate {
    static let cjkTokensPerCharacter = 0.65
    static let hangulTokensPerCharacter = 0.45
    static let defaultTokensPerCharacter = 0.42

    static func tokens(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var total = 0.0
        for scalar in text.unicodeScalars {
            total += tokenCost(of: scalar)
        }
        return max(1, Int(total.rounded(.up)))
    }

    /// Returns the longest budgeted prefix without splitting a grapheme cluster.
    static func prefix(_ text: String, withinTokens tokenBudget: Int) -> String {
        guard tokenBudget > 0 else { return "" }
        guard tokens(in: text) > tokenBudget else { return text }

        var spent = 0.0
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            let cost = character.unicodeScalars.reduce(0.0) { $0 + tokenCost(of: $1) }
            if spent + cost > Double(tokenBudget) { break }
            spent += cost
            index = text.index(after: index)
        }
        return String(text[text.startIndex..<index])
    }

    // MARK: - Classification

    private static func tokenCost(of scalar: Unicode.Scalar) -> Double {
        if isCJK(scalar) { return cjkTokensPerCharacter }
        if isHangul(scalar) { return hangulTokensPerCharacter }
        return defaultTokensPerCharacter
    }

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
