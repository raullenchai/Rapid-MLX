import Foundation

/// Shared CJK-majority signal for background titles and questions.
enum ReplyScript {
    nonisolated static func usesCJK(_ text: String) -> Bool {
        let (cjkLetters, otherLetters) = letterCounts(text)
        return cjkLetters > otherLetters
    }

    /// At least one CJK letter, whatever the Latin around it.
    nonisolated static func containsCJK(_ text: String) -> Bool {
        letterCounts(text).cjk > 0
    }

    private nonisolated static func letterCounts(_ text: String) -> (cjk: Int, other: Int) {
        var cjkLetters = 0
        var otherLetters = 0
        for scalar in text.unicodeScalars where CharacterSet.letters.contains(scalar) {
            if isCJKScalar(scalar) {
                cjkLetters += 1
            } else {
                otherLetters += 1
            }
        }
        return (cjkLetters, otherLetters)
    }

    /// Is the user writing in a CJK language, rather than quoting CJK words
    /// in another one?
    ///
    /// Counts CJK characters against Latin *words*, not letters. A Chinese
    /// question about code ("请解释 Python list comprehension": 3 against 3)
    /// is CJK writing, while an English question about a Chinese word
    /// ("What does 你好 mean?": 2 against 3) is not. A CJK character carries
    /// roughly a word, so letter counts would hand both to English.
    nonisolated static func writesInCJK(_ text: String) -> Bool {
        var cjk = 0
        var words = 0
        var inWord = false
        for scalar in text.unicodeScalars {
            let isLetter = CharacterSet.letters.contains(scalar)
            if isLetter && isCJKScalar(scalar) {
                cjk += 1
                inWord = false
            } else if isLetter {
                if !inWord { words += 1 }
                inWord = true
            } else {
                inWord = false
            }
        }
        return cjk > 0 && cjk >= words
    }

    /// CJK writing with Han and no kana or hangul: the text that gets the
    /// Han-script title and chip cues. Ordinary Japanese always carries kana,
    /// so it is excluded. Short Han-only Japanese ("東京天気") cannot be told
    /// apart, which is why those cues are written in Han script but ask for
    /// "the user's language", never for Chinese by name.
    nonisolated static func isChinese(_ text: String) -> Bool {
        guard writesInCJK(text) else { return false }
        return !text.unicodeScalars.contains(where: isKanaOrHangul)
    }

    private nonisolated static func isCJKScalar(_ scalar: Unicode.Scalar) -> Bool {
        isHan(scalar) || isKanaOrHangul(scalar)
    }

    private nonisolated static func isHan(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF,       // unified ideographs
             0x3400...0x4DBF,       // extension A
             0xF900...0xFAFF,       // compatibility ideographs
             0x20000...0x323AF:     // supplementary extensions
            return true
        default:
            return false
        }
    }

    private nonisolated static func isKanaOrHangul(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF,       // hiragana, katakana
             0x31F0...0x31FF,       // katakana phonetic extensions
             0xFF66...0xFF9F,       // half-width katakana
             0x1100...0x11FF,       // hangul jamo
             0x3130...0x318F,       // hangul compatibility jamo
             0xAC00...0xD7AF:       // hangul syllables
            return true
        default:
            return false
        }
    }
}
