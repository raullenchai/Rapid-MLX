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
        return !text.unicodeScalars.contains {
            (0x3040...0x30FF).contains($0.value) || (0xAC00...0xD7AF).contains($0.value)
        }
    }

    private nonisolated static func isCJKScalar(_ scalar: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(scalar.value)         // unified ideographs
            || (0x3040...0x30FF).contains(scalar.value)  // kana
            || (0xAC00...0xD7AF).contains(scalar.value)  // hangul
    }
}
