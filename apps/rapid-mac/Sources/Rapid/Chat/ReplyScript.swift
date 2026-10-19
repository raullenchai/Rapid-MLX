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
            if (0x4E00...0x9FFF).contains(scalar.value)         // unified ideographs
                || (0x3040...0x30FF).contains(scalar.value)     // kana
                || (0xAC00...0xD7AF).contains(scalar.value)     // hangul
            {
                cjkLetters += 1
            } else {
                otherLetters += 1
            }
        }
        return (cjkLetters, otherLetters)
    }

    /// Han text with no kana or hangul: the text that gets the Han-script
    /// title and chip cues. Presence, not majority: a Chinese question about
    /// code ("请解释 Python list comprehension") has more Latin letters than
    /// Han and is still written by a Chinese reader. Ordinary Japanese always
    /// carries kana, so it is excluded. Short Han-only Japanese ("東京天気")
    /// cannot be told apart, which is why those cues are written in Han
    /// script but ask for "the user's language", never for Chinese by name.
    nonisolated static func isChinese(_ text: String) -> Bool {
        var han = 0
        var kanaAndHangul = 0
        for scalar in text.unicodeScalars where CharacterSet.letters.contains(scalar) {
            if (0x4E00...0x9FFF).contains(scalar.value) {
                han += 1
            } else if (0x3040...0x30FF).contains(scalar.value)
                || (0xAC00...0xD7AF).contains(scalar.value)
            {
                kanaAndHangul += 1
            }
        }
        return han > 0 && kanaAndHangul == 0
    }
}
