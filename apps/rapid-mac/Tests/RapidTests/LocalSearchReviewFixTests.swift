import Foundation
import Testing
@testable import Rapid

@Suite struct LocalSearchReviewFixTests {
    private let knownReasons = [
        "Unsupported file type", "Too large", "No readable text", "Scanned PDF without text",
        "Password-protected", "Could not be read", "Private or system file", "Audio is not supported yet"
    ]

    @Test func reasonsMapToDistinctCatalogEntriesWithNeighbouringLanguages() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings"))
        let catalog = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let strings = try #require(catalog["strings"] as? [String: [String: Any]])
        let languages = Set(try #require(strings["Why files were not indexed"]?["localizations"] as? [String: Any]).keys)
        #expect(Set(knownReasons.map(LocalSearchSkipReason.localized)).count == knownReasons.count)
        #expect(LocalSearchSkipReason.localized("future reason") == String(localized: "File contents could not be indexed"))
        #expect(LocalSearchSkipReason.localized("") == LocalSearchSkipReason.localized("future reason"))
        for reason in knownReasons {
            #expect(LocalSearchSkipReason.localized(reason) == String(localized: String.LocalizationValue(reason)))
            let folder = LocalSearchFolder(id: "f", path: "/files", state: .idle, filesIndexed: 0,
                filesSkipped: 2, chunks: 0, lastIndexedAt: nil, error: nil, skipReasons: [reason: 2])
            #expect(folder.skippedExplanation == "\(LocalSearchSkipReason.localized(reason)): 2")
        }
        let keys = knownReasons + ["File contents could not be indexed", "Matched by file name",
            "Matched by file name — %@", "Indexing paused — it resumes automatically",
            "Indexing paused while a chat response is generating — it resumes automatically"]
        var chineseReasons = Set<String>()
        for key in keys {
            let localizations = try #require(strings[key]?["localizations"] as? [String: [String: Any]])
            #expect(Set(localizations.keys) == languages)
            let unit = try #require(localizations["zh-Hans"]?["stringUnit"] as? [String: String])
            #expect(unit["state"] == "translated")
            let value = try #require(unit["value"])
            #expect(value != key)
            if knownReasons.contains(key) { chineseReasons.insert(value) }
        }
        #expect(chineseReasons.count == knownReasons.count)
    }

    @Test func pausedReasonDecodesOptionallyAndKeepsPolling() throws {
        for (extra, reason) in [("", nil), (#", "paused_reason":null"#, nil),
            (#", "paused_reason":"chat""#, "chat"), (#", "paused_reason":"future""#, "future")] as [(String, String?)] {
            let json = #"{"enabled":true,"model":{"id":"m","downloaded":true,"loaded":true},"folders":[],"indexing":{"active":true,"files_done":2,"files_total":10"#
                + extra + #"},"totals":{"files":2,"chunks":2,"index_bytes":0}}"#
            let status = try LocalSearchClient.decode(LocalSearchStatus.self, from: Data(json.utf8))
            #expect(status.indexing.pausedReason == reason)
            #expect(status.needsPolling)
            let expected: String
            switch reason {
            case nil: expected = String(localized: "Indexing: \(2) of \(10) files")
            case "chat": expected = String(localized: "Indexing paused while a chat response is generating — it resumes automatically")
            default: expected = String(localized: "Indexing paused — it resumes automatically")
            }
            #expect(status.indexing.progressLabel == expected)
        }
        let scanning = LocalSearchStatus.Indexing(active: true, folderId: nil, currentPath: nil, filesDone: 2, filesTotal: 0)
        #expect(scanning.progressLabel == String(localized: "Scanning and indexing: \(2) files processed"))
    }

    @Test func contentIndexingDefaultsAndNameOnlyRowPresentation() throws {
        let base: [String: Any] = ["path": "/files/scan.pdf", "folder_id": "f", "title": "scan",
            "snippet": "Readable excerpt", "highlights": [], "score": 1, "matched": [], "modified_at": 0, "size": 12]
        for field in [nil, NSNull(), true, false] as [Any?] {
            var json = base
            json["content_indexed"] = field
            json["skip_reason"] = "Scanned PDF without text"
            let result = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            let indexed = (field as? Bool) ?? true
            #expect(result.contentIndexed == indexed)
            #expect(result.skipReason == "Scanned PDF without text")
            if indexed {
                #expect(result.nameOnlyExplanation == nil)
                #expect(result.snippetLabel == "Readable excerpt")
            } else {
                #expect(result.snippetLabel == String(localized: "Matched by file name — \(LocalSearchSkipReason.localized("Scanned PDF without text"))"))
            }
        }
        for reason in [nil, NSNull(), "future reason"] as [Any?] {
            var json = base
            json["content_indexed"] = false
            json["snippet"] = ""
            json["skip_reason"] = reason
            let result = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(result.skipReason == reason as? String)
            #expect(result.snippetLabel == (reason is String
                ? String(localized: "Matched by file name — \(LocalSearchSkipReason.localized("future reason"))")
                : String(localized: "Matched by file name")))
        }
    }
}
