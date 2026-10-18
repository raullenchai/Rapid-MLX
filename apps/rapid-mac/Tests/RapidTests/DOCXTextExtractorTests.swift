import Compression
import Foundation
import Testing
@testable import Rapid

@Suite("Bounded DOCX extraction", .serialized)
struct DOCXTextExtractorTests {
    private let ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    private func xml(_ body: String) -> Data {
        Data("<w:document xmlns:w=\"\(ns)\"><w:body>\(body)</w:body></w:document>".utf8)
    }
    private struct Entry {
        var name = "word/document.xml"
        var data: Data
        var method = 0
        var flags = 0
        var declared: Int?
        var crc: UInt32?
        var precompressed: Data?
    }
    private func zip(_ entries: [Entry]) throws -> Data {
        var output = Data(), central = Data()
        func put(_ value: Int, _ width: Int, into data: inout Data) {
            for shift in 0..<width { data.append(UInt8(truncatingIfNeeded: value >> (8 * shift))) }
        }
        for entry in entries {
            let name = Data(entry.name.utf8), offset = output.count
            var payload = entry.precompressed ?? entry.data
            if entry.method == 8, entry.precompressed == nil {
                payload = try (entry.data as NSData).compressed(using: .zlib) as Data
            }
            let crc = Int(entry.crc ?? DOCXTextExtractor.crc32(entry.data))
            for (value, width) in [(0x04034b50,4),(20,2),(entry.flags,2),(entry.method,2),(0,2),(0,2),(crc,4),(payload.count,4),(entry.declared ?? entry.data.count,4),(name.count,2),(0,2)] {
                put(value,width,into:&output)
            }
            output.append(name); output.append(payload)
            for (value,width) in [(0x02014b50,4),(20,2),(20,2),(entry.flags,2),(entry.method,2),(0,2),(0,2),(crc,4),(payload.count,4),(entry.declared ?? entry.data.count,4),(name.count,2),(0,2),(0,2),(0,2),(0,2),(0,4),(offset,4)] {
                put(value,width,into:&central)
            }
            central.append(name)
        }
        let offset = output.count
        output.append(central)
        for (value,width) in [(0x06054b50,4),(0,2),(0,2),(entries.count,2),(entries.count,2),(central.count,4),(offset,4),(0,2)] { put(value,width,into:&output) }
        return output
    }
    private func expectFailure(_ data: Data, _ failure: DOCXTextExtractor.Failure,
                               limits: DOCXTextExtractor.Limits = .default) {
        let start = Date()
        #expect(throws: failure) { try DOCXTextExtractor.extract(data, limits: limits) }
        #expect(Date().timeIntervalSince(start) < 2)
    }
    private func attachment(_ data: Data, cache: DocumentContentCache) throws -> ChatFileAttachment {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("docx-\(UUID()).DOCX")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        return try ChatFileAttachment(contentsOf: url, cache: cache)
    }

    @Test func normalStoredAndDeflated() throws {
        let source = xml("<w:p><w:r><w:t>Hello</w:t><w:tab/><w:t>world</w:t><w:br/><w:t>line</w:t><w:cr/><w:delText>deleted</w:delText><w:instrText>instruction</w:instrText></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>cell</w:t></w:r></w:p></w:tc><w:tc><w:t>bare</w:t></w:tc></w:tr></w:tbl><x xmlns=\"other\">ignored</x>")
        let expected = "Hello\tworld\nline\n\ncell\nbare\t"
        for method in [0,8] {
            let archive = try zip([Entry(data: source, method: method)])
            let result = try DOCXTextExtractor.extract(archive)
            #expect(result.text == expected)
            #expect(!result.reachedCharacterCap)
            let cache = DocumentContentCache(diskDirectory: nil)
            let item = try attachment(archive, cache: cache)
            #expect(item.kind == .docx)
            #expect(item.kind.displayName == "DOCX")
            #expect(item.kind.systemImage == "doc.richtext")
            #expect(cache.get(item.id)?.text == expected.trimmingCharacters(in: .whitespacesAndNewlines))
            #expect(!item.wasTruncated)
        }
        #expect(ChatFileAttachment.recognizesDocument(at: URL(fileURLWithPath: "/tmp/test.DOCX")))
        #expect(try JSONDecoder().decode(ChatFileAttachment.Kind.self, from: Data("\"future\"".utf8)) == .unknown)
    }

    @Test func strictNamespace() throws {
        let source = String(decoding: xml("<w:p><w:t>Strict</w:t><w:tab/><w:t>OOXML</w:t><w:br/><w:t>text</w:t></w:p>"), as: UTF8.self)
            .replacingOccurrences(of: ns, with: "http://purl.oclc.org/ooxml/wordprocessingml/main")
        let data = try zip([Entry(data: Data(source.utf8))])
        #expect(try DOCXTextExtractor.extract(data).text == "Strict\tOOXML\ntext\n")
    }

    @Test func largePreviewAndCap() throws {
        let text = String(repeating: "paragraph text ", count: 150_000)
        let data = try zip([Entry(data: xml("<w:p><w:t>\(text)</w:t></w:p>"))])
        let cache = DocumentContentCache(diskDirectory: nil)
        let item = try attachment(data, cache: cache)
        #expect(item.hasUnshownContent)
        #expect(item.promptText.contains("read_document"))
        #expect(cache.get(item.id)?.text.count == text.count - 1)
        #expect(TokenEstimate.tokens(in: item.extractedText) <= ChatFileAttachment.maxCombinedTokens)
        var limits = DOCXTextExtractor.Limits.default
        limits.maxCharacters = 5
        let result = try DOCXTextExtractor.extract(data, limits: limits)
        #expect(result.text == "parag")
        #expect(result.reachedCharacterCap)
    }

    @Test func injectedLimitEdgesAndDataSlice() throws {
        let good = try zip([Entry(data: xml("<w:t>ok</w:t>"), method: 8)])
        let slice = (Data([0]) + good).dropFirst()
        #expect(try DOCXTextExtractor.extract(slice).text == "ok")
        var limits = DOCXTextExtractor.Limits.default
        limits.maxCompressionRatio = Int.max
        #expect(try DOCXTextExtractor.extract(good, limits: limits).text == "ok")
        limits.maxCompressionRatio = -1
        expectFailure(good, .tooLarge, limits: limits)
        limits = .default
        limits.maxCharacters = 0
        let result = try DOCXTextExtractor.extract(good, limits: limits)
        #expect(result.text.isEmpty)
        #expect(result.reachedCharacterCap)
    }

    @Test func budgets() throws {
        let real = Data(repeating: 65, count: 8 * 1024 * 1024)
        let compressed = try zip([Entry(data: real, method: 8)])
        var limits = DOCXTextExtractor.Limits.default
        limits.maxCompressionRatio = 2
        expectFailure(compressed, .tooLarge, limits: limits)
        limits.maxCompressionRatio = 1000
        limits.maxDocumentXMLBytes = 1024 * 1024
        expectFailure(try zip([Entry(data: real, method: 8, declared: 100)]), .tooLarge, limits: limits)
        expectFailure(try zip([Entry(data: real)]), .tooLarge, limits: limits)
        let small = try zip([Entry(data: xml("<w:p/>"))])
        limits = .default; limits.maxArchiveBytes = small.count - 1
        expectFailure(small, .tooLarge, limits: limits)
        limits = .default; limits.maxEntryCount = 0
        expectFailure(small, .tooLarge, limits: limits)
        limits = .default; limits.maxDocumentXMLBytes = 1
        expectFailure(small, .tooLarge, limits: limits)
    }

    @Test func defaultLimitZipBombs() throws {
        let limits = DOCXTextExtractor.Limits.default
        let ratioBomb: Data
        do {
            let source = Data(repeating: 0, count: 100 * 1024 * 1024)
            let payload = try (source as NSData).compressed(using: .zlib) as Data
            #expect(source.count > payload.count * limits.maxCompressionRatio)
            ratioBomb = try zip([Entry(data: source, method: 8, precompressed: payload)])
        }
        // Truthful metadata must reject this bomb before decompression.
        expectFailure(ratioBomb, .tooLarge, limits: limits)

        let lyingSizeBomb: Data
        do {
            let source = Data(repeating: 0, count: 80 * 1024 * 1024)
            let payload = try (source as NSData).compressed(using: .zlib) as Data
            let declared = payload.count * 10
            #expect(declared <= limits.maxDocumentXMLBytes)
            #expect(declared <= payload.count * limits.maxCompressionRatio)
            #expect(source.count > limits.maxDocumentXMLBytes)
            lyingSizeBomb = try zip([Entry(data: source, method: 8, declared: declared,
                                           precompressed: payload)])
        }
        // Metadata passes preflight; the actual output must hit the streaming budget.
        expectFailure(lyingSizeBomb, .tooLarge, limits: limits)
    }

    @Test func defaultLimitEntryCount() throws {
        let tiny = Data([0])
        do {
            let entries = (0..<4097).map { Entry(name: "f\($0)", data: tiny) }
            expectFailure(try zip(entries), .tooLarge)
        }
        do {
            var entries = (0..<4095).map { Entry(name: "f\($0)", data: tiny) }
            entries.append(Entry(data: xml("<w:t>ok</w:t>")))
            let archive = try zip(entries)
            #expect(try DOCXTextExtractor.extract(archive).text == "ok")
        }
    }

    @Test func hostileArchives() throws {
        let good = try zip([Entry(data: xml("<w:p><w:t>ok</w:t></w:p>"))])
        expectFailure(Data("random".utf8), .notAZip)
        expectFailure(Data([0x50,0x4b]), .malformed)
        expectFailure(Data(good.prefix(good.count / 2)), .malformed)
        expectFailure(try zip([Entry(data: xml(""), crc: 1)]), .malformed)
        expectFailure(try zip([Entry(data: Data("<!DOCTYPE x [<!ENTITY a 'boom'>]><x>&a;</x>".utf8))]), .malformed)
        expectFailure(try zip([Entry(data: Data("<!doctype x><x/>".utf8))]), .malformed)
        expectFailure(try zip([Entry(data: Data("<!EnTiTy x 'value'><x/>".utf8))]), .malformed)
        expectFailure(try zip([Entry(data: Data("<broken>".utf8))]), .malformed)
        expectFailure(try zip([Entry(name: "other", data: Data())]), .missingDocument)
        expectFailure(try zip([Entry(data: xml(""), method: 12)]), .unsupportedArchive)
        expectFailure(try zip([Entry(data: xml(""), flags: 1)]), .encrypted)
        expectFailure(try zip([Entry(name: "other", data: Data(), flags: 64), Entry(data: xml(""))]), .encrypted)
        expectFailure(Data([0xd0,0xcf,0x11,0xe0,0xa1,0xb1,0x1a,0xe1]), .encrypted)
        expectFailure(try zip([Entry(data: xml("")), Entry(data: xml(""))]), .malformed)
        expectFailure(try zip([Entry(data: xml(""), declared: 0xffffffff)]), .unsupportedArchive)
        var sentinel = good
        sentinel[sentinel.count - 12] = 255; sentinel[sentinel.count - 11] = 255
        expectFailure(sentinel, .unsupportedArchive)
        let deep = Data((String(repeating: "<a>", count: 257) + String(repeating: "</a>", count: 257)).utf8)
        expectFailure(try zip([Entry(data: deep)]), .malformed)
        let utf16 = "<!DOCTYPE x><x/>".data(using: .utf16LittleEndian) ?? Data()
        expectFailure(try zip([Entry(data: utf16)]), .malformed)
    }

    @Test func productionCharacterCap() throws {
        let paragraph = "<w:p><w:t>" + String(repeating: "x", count: 1000) + "</w:t></w:p>"
        let text = String(repeating: paragraph, count: ChatFileAttachment.maxExtractedCharacters / 1000 + 1)
        let data = try zip([Entry(data: xml(text))])
        let cache = DocumentContentCache(diskDirectory: nil)
        let item = try attachment(data, cache: cache)
        #expect(item.wasTruncated)
        #expect(item.totalCharacterCount == ChatFileAttachment.maxExtractedCharacters)
        #expect(cache.get(item.id)?.hitSizeCeiling == true)
        #expect(cache.get(item.id)?.isComplete == false)
    }

    @Test func structuralValidation() throws {
        let source = xml("<w:t>ok</w:t>")
        let good = try zip([Entry(data: source)])
        let cd = 30 + "word/document.xml".utf8.count + source.count
        let end = good.count - 22
        func patched(_ offset: Int, _ value: Int, _ width: Int = 4) -> Data {
            var bytes = good
            for shift in 0..<width { bytes[offset + shift] = UInt8(truncatingIfNeeded: value >> (shift * 8)) }
            return bytes
        }
        for bytes in [patched(end, 0), patched(end + 16, good.count), patched(end + 12, good.count),
                      patched(cd, 0), patched(cd + 28, 65535, 2), patched(0, 0x11114b50),
                      patched(26, 65535, 2), patched(cd + 20, good.count),
                      patched(cd + 42, cd), patched(cd + 24, source.count - 1),
                      patched(end + 12, good.count - cd - 23)] {
            expectFailure(bytes, .malformed)
        }
        for bytes in [patched(end + 4, 1, 2), patched(end + 6, 1, 2),
                      patched(end + 8, 2, 2), patched(end + 16, 0xffffffff),
                      patched(end + 12, 0xffffffff), patched(cd + 20, 0xffffffff),
                      patched(cd + 42, 0xffffffff), patched(cd + 34, 1, 2)] {
            expectFailure(bytes, .unsupportedArchive)
        }
        expectFailure(patched(6, 64, 2), .encrypted)
        var locator = good
        locator.insert(contentsOf: [0x50,0x4b,0x06,0x07] + Array(repeating: 0, count: 16), at: end)
        expectFailure(locator, .unsupportedArchive)
        var localSizeLie = good
        // Stored bytes are checked against the actual budget independently of the declaration.
        for shift in 0..<4 { localSizeLie[cd + 24 + shift] = 0 }
        var limits = DOCXTextExtractor.Limits.default
        limits.maxDocumentXMLBytes = 1
        expectFailure(localSizeLie, .tooLarge, limits: limits)
        var brokenDeflate = try zip([Entry(data: source, method: 8)])
        brokenDeflate[47] = 0xff
        expectFailure(brokenDeflate, .malformed)
        #expect(DOCXTextExtractor.crc32(Data("123456789".utf8)) == 0xcbf43926)
    }

    @Test func parserDefenseCallbacks() {
        let start = Date()
        let parser = XMLParser(data: Data())
        let callbacks: [(DOCXTextExtractor.TextDelegate) -> Void] = [
            { $0.parser(parser, foundInternalEntityDeclarationWithName: "a", value: "b") },
            { $0.parser(parser, foundExternalEntityDeclarationWithName: "a", publicID: nil, systemID: "file:///tmp/a") },
            { $0.parser(parser, foundUnparsedEntityDeclarationWithName: "a", publicID: nil, systemID: nil, notationName: "n") },
            { $0.parser(parser, foundElementDeclarationWithName: "a", model: "ANY") },
            { #expect($0.parser(parser, resolveExternalEntityName: "a", systemID: nil) == nil) }
        ]
        for callback in callbacks {
            let delegate = DOCXTextExtractor.TextDelegate(limit: 20, deadline: start.addingTimeInterval(20), now: { start })
            callback(delegate)
            #expect(delegate.failure == .malformed)
        }
        let timed = DOCXTextExtractor.TextDelegate(limit: 20, deadline: start, now: { start })
        for _ in 0..<1024 { timed.parser(parser, foundCharacters: "ignored") }
        #expect(timed.failure == .timedOut)
        #expect(Date().timeIntervalSince(start) < 2)
    }

    @Test func crcDeadline() throws {
        // A bad checksum prevents a later checkpoint from masking an uninterrupted CRC pass.
        let data = try zip([Entry(data: Data(repeating: 65, count: 3 * 1024 * 1024), crc: 0)])
        let start = Date()
        var calls = 0
        // Initial clock, central entry, stored copy, then the first CRC chunk.
        // Expire at the second chunk, after CRC processing has started.
        #expect(throws: DOCXTextExtractor.Failure.timedOut) {
            try DOCXTextExtractor.extract(data, now: {
                calls += 1
                return start.addingTimeInterval(calls < 5 ? 0 : 21)
            })
        }
        #expect(calls == 5)
        #expect(Date().timeIntervalSince(start) < 2)
    }

    @Test func parsingDeadline() throws {
        let data = try zip([Entry(data: xml(String(repeating: "<w:p><w:t>x</w:t></w:p>", count: 1000)))])
        let start = Date()
        var calls = 0
        // Initial clock, central entry, stored copy, CRC chunk/checkpoint and declaration scan precede parsing.
        #expect(throws: DOCXTextExtractor.Failure.timedOut) {
            try DOCXTextExtractor.extract(data, now: {
                calls += 1
                return start.addingTimeInterval(calls < 7 ? 0 : 21)
            })
        }
        #expect(Date().timeIntervalSince(start) < 2)
    }

    @Test func timeout() throws {
        let data = try zip([Entry(data: xml("<w:t>ok</w:t>"), method: 8)])
        let start = Date()
        for jumpAt in [2, 3] {
            var calls = 0
            #expect(throws: DOCXTextExtractor.Failure.timedOut) {
                try DOCXTextExtractor.extract(data, now: {
                    calls += 1
                    return start.addingTimeInterval(calls < jumpAt ? 0 : 21)
                })
            }
        }
        #expect(Date().timeIntervalSince(start) < 2)
    }

    @Test func errorCopy() throws {
        let cache = DocumentContentCache(diskDirectory: nil)
        for (data, error) in [
            (Data("bad".utf8), ChatFileAttachment.ValidationError.invalidDOCX),
            (Data([0xd0,0xcf,0x11,0xe0,0xa1,0xb1,0x1a,0xe1]), .docxEncrypted),
            (try zip([Entry(data: xml(""), declared: 65 * 1024 * 1024)]), .docxTooComplex),
            (try zip([Entry(data: xml(""))]), .noExtractableText(.docx))
        ] {
            let start = Date()
            #expect(throws: error) { try attachment(data, cache: cache) }
            #expect(Date().timeIntervalSince(start) < 2)
        }
        #expect(ChatFileAttachment.ValidationError.docxTooComplex.errorDescription == "This Word document is too large or complex to read safely. Save it as PDF or TXT, then attach that file.")
        #expect(ChatFileAttachment.ValidationError.docxEncrypted.errorDescription == "This Word document is password-protected or in an older format. Remove the password or save it as PDF or TXT, then attach that file.")
        #expect(ChatFileAttachment.ValidationError.invalidDOCX.errorDescription == "This Word document couldn't be read. Save it as PDF or TXT, then attach that file.")
        #expect(ChatFileAttachment.ValidationError.noExtractableText(.docx).errorDescription == "This Word document contains no readable text.")
        #expect(ChatFileAttachment.ValidationError.tooLarge.errorDescription == "PDF, CSV, TXT, and DOCX files must be 100 MB or smaller.")
        #expect(ChatFileAttachment.ValidationError.unsupportedType.errorDescription == "Choose a PDF, CSV, TXT, or DOCX file.")
    }
}
