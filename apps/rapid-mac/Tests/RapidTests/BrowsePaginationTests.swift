import Foundation
import Testing
@testable import Rapid

@Suite("Browse pagination")
struct BrowsePaginationTests {
    @Test("sparse indexes preserve Unicode character offsets")
    func sparseIndexesPreserveCharacterOffsets() {
        let text = String(repeating: "a", count: 4_095) + "👩🏽‍💻\n后续"
        let entry = BrowseContentCache.Entry(
            title: nil,
            markdown: text,
            finalURL: "https://example.com"
        )

        #expect(entry.count == 4_099)
        let emoji = entry.index(atCharacterOffset: 4_095)
        let afterEmoji = entry.index(atCharacterOffset: 4_096)
        #expect(String(text[emoji..<afterEmoji]) == "👩🏽‍💻")
        #expect(entry.index(atCharacterOffset: Int.max) == text.endIndex)
    }

    @Test("derived indexes survive cache encoding")
    func derivedIndexesSurviveEncoding() throws {
        let original = BrowseContentCache.Entry(
            title: "title",
            markdown: String(repeating: "字", count: 8_300),
            finalURL: "https://example.com"
        )
        let decoded = try JSONDecoder().decode(
            BrowseContentCache.Entry.self,
            from: JSONEncoder().encode(original)
        )

        #expect(decoded.count == 8_300)
        #expect(decoded.index(atCharacterOffset: 8_299) < decoded.markdown.endIndex)
        #expect(decoded.index(atCharacterOffset: 8_300) == decoded.markdown.endIndex)
    }
}
