import Foundation

/// Shared CJK-majority signal for background titles and questions.
enum ReplyScript {
    nonisolated static func usesCJK(_ text: String) -> Bool {
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
        return cjkLetters > otherLetters
    }

    /// Han must outnumber kana and hangul before choosing a Chinese cue.
    nonisolated static func isChinese(_ text: String) -> Bool {
        guard usesCJK(text) else { return false }
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
        return han > kanaAndHangul
    }
}
