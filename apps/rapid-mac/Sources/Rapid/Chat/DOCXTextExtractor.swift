import Compression
import Foundation

/// Reads only the main document part; never expands the rest of the package.
struct DOCXTextExtractor: Sendable {
    struct Limits: Sendable {
        var maxArchiveBytes = ChatFileAttachment.maxSourceBytes
        var maxEntryCount = 4096
        var maxDocumentXMLBytes = 64 * 1024 * 1024
        var maxCompressionRatio = 1000
        var maxCharacters = ChatFileAttachment.maxExtractedCharacters
        var timeout: TimeInterval = 20
        static let `default` = Limits()
    }
    struct Result: Sendable { let text: String; let reachedCharacterCap: Bool }
    enum Failure: Error, Equatable {
        case notAZip, encrypted, unsupportedArchive, missingDocument, tooLarge, malformed, timedOut
    }

    static func extract(_ data: Data, limits: Limits = .default, now: () -> Date = Date.init) throws -> Result {
        let deadline = now().addingTimeInterval(limits.timeout)
        func checkTime() throws { if now() >= deadline { throw Failure.timedOut } }
        guard data.count <= limits.maxArchiveBytes else { throw Failure.tooLarge }
        if data.starts(with: [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]) { throw Failure.encrypted }
        guard data.starts(with: [0x50, 0x4b]) else { throw Failure.notAZip }
        // Work in zero-based offsets even when the caller supplied a Data slice.
        let bytes = Data(data)
        func number(_ offset: Int, _ width: Int) throws -> Int {
            guard offset >= 0, width <= bytes.count, offset <= bytes.count - width else { throw Failure.malformed }
            return (0..<width).reduce(0) { $0 | Int(bytes[offset + $1]) << ($1 * 8) }
        }
        func u16(_ offset: Int) throws -> Int { try number(offset, 2) }
        func u32(_ offset: Int) throws -> Int { try number(offset, 4) }
        guard bytes.count >= 22 else { throw Failure.malformed }
        var end: Int?
        for offset in stride(from: bytes.count - 22, through: max(0, bytes.count - 65_557), by: -1) {
            if try u32(offset) == 0x06054b50, try u16(offset + 20) == bytes.count - offset - 22 {
                end = offset
                break
            }
        }
        guard let end else { throw Failure.malformed }
        if end >= 20, try u32(end - 20) == 0x07064b50 { throw Failure.unsupportedArchive }
        let disk = try u16(end + 4), cdDisk = try u16(end + 6)
        let diskCount = try u16(end + 8), count = try u16(end + 10)
        let cdSize = try u32(end + 12), cd = try u32(end + 16)
        guard disk == 0, cdDisk == 0, diskCount == count,
              count != 0xffff, diskCount != 0xffff, cdSize != 0xffffffff, cd != 0xffffffff
        else { throw Failure.unsupportedArchive }
        guard count <= limits.maxEntryCount else { throw Failure.tooLarge }
        guard cd <= end, cdSize <= end - cd else { throw Failure.malformed }
        let cdEnd = cd + cdSize
        var cursor = cd
        var document: (method: Int, compressed: Int, expanded: Int, crc: UInt32, local: Int)?
        for _ in 0..<count {
            try checkTime()
            guard cursor <= cdEnd, cdEnd - cursor >= 46, try u32(cursor) == 0x02014b50 else { throw Failure.malformed }
            let flags = try u16(cursor + 8)
            if flags & 0x41 != 0 { throw Failure.encrypted }
            let nameLength = try u16(cursor + 28), extra = try u16(cursor + 30), comment = try u16(cursor + 32)
            let length = 46 + nameLength + extra + comment
            guard length <= cdEnd - cursor else { throw Failure.malformed }
            if bytes[(cursor + 46)..<(cursor + 46 + nameLength)].elementsEqual("word/document.xml".utf8) {
                guard document == nil else { throw Failure.malformed }
                let compressed = try u32(cursor + 20), expanded = try u32(cursor + 24), local = try u32(cursor + 42)
                guard compressed != 0xffffffff, expanded != 0xffffffff, local != 0xffffffff,
                      try u16(cursor + 34) == 0 else { throw Failure.unsupportedArchive }
                let method = try u16(cursor + 10)
                guard method == 0 || method == 8 else { throw Failure.unsupportedArchive }
                guard expanded <= limits.maxDocumentXMLBytes, limits.maxCompressionRatio >= 0 else { throw Failure.tooLarge }
                let (ratioBudget, overflow) = max(1, compressed).multipliedReportingOverflow(by: limits.maxCompressionRatio)
                guard overflow || expanded <= ratioBudget else { throw Failure.tooLarge }
                document = (method, compressed, expanded, UInt32(try u32(cursor + 16)), local)
            }
            cursor += length
        }
        guard cursor == cdEnd else { throw Failure.malformed }
        guard let document else { throw Failure.missingDocument }
        let local = document.local
        guard local < cd, cd - local >= 30, try u32(local) == 0x04034b50 else { throw Failure.malformed }
        if try u16(local + 6) & 0x41 != 0 { throw Failure.encrypted }
        let headerLength = 30 + (try u16(local + 26)) + (try u16(local + 28))
        guard headerLength <= cd - local else { throw Failure.malformed }
        let start = local + headerLength
        guard document.compressed <= cd - start else { throw Failure.malformed }
        var xml = Data()
        if document.method == 0 {
            try checkTime()
            guard document.compressed <= limits.maxDocumentXMLBytes else { throw Failure.tooLarge }
            xml = bytes.subdata(in: start..<(start + document.compressed))
        } else {
            try bytes.withUnsafeBytes { raw in
                // The validated local header guarantees a nonempty backing allocation.
                let source = raw.bindMemory(to: UInt8.self)
                guard let base = source.baseAddress else { throw Failure.malformed }
                let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 65_536)
                defer { buffer.deallocate() }
                var stream = compression_stream(dst_ptr: buffer, dst_size: 0,
                                                src_ptr: base + start, src_size: document.compressed, state: nil)
                // ZIP method 8 is raw DEFLATE; COMPRESSION_ZLIB is Apple's raw-DEFLATE decoder.
                guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { throw Failure.malformed }
                defer { compression_stream_destroy(&stream) }
                stream.src_ptr = base + start
                stream.src_size = document.compressed
                while true {
                    try checkTime()
                    stream.dst_ptr = buffer
                    stream.dst_size = 65_536
                    let status = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                    let produced = 65_536 - stream.dst_size
                    guard produced <= limits.maxDocumentXMLBytes - xml.count else { throw Failure.tooLarge }
                    xml.append(buffer, count: produced)
                    if status == COMPRESSION_STATUS_END {
                        guard stream.src_size == 0 else { throw Failure.malformed }
                        break
                    }
                    guard status == COMPRESSION_STATUS_OK, produced > 0 else { throw Failure.malformed }
                }
            }
        }
        guard xml.count == document.expanded else { throw Failure.malformed }
        let crc = try xml.withUnsafeBytes { raw in
            var crc = UInt32.max
            let chunkSize = 1024 * 1024
            for offset in stride(from: 0, to: raw.count, by: chunkSize) {
                try checkTime()
                let end = min(offset + chunkSize, raw.count)
                crc = updateCRC32(crc, with: UnsafeRawBufferPointer(rebasing: raw[offset..<end]))
            }
            return crc ^ UInt32.max
        }
        guard crc == document.crc else { throw Failure.malformed }
        try checkTime()
        // NUL removal also detects declarations in UTF-16/32 documents.
        let markers = [Array("<!DOCTYPE".utf8), Array("<!ENTITY".utf8)]
        try xml.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for index in bytes.indices {
                if index % 65_536 == 0 { try checkTime() }
                guard bytes[index] == 60 else { continue }
                for marker in markers {
                    var cursor = index
                    var matched = 0
                    while cursor < bytes.count, matched < marker.count {
                        let byte = bytes[cursor]
                        cursor += 1
                        if byte == 0 { continue }
                        let upper = (97...122).contains(byte) ? byte - 32 : byte
                        if upper != marker[matched] { break }
                        matched += 1
                    }
                    if matched == marker.count { throw Failure.malformed }
                }
            }
        }
        return try withoutActuallyEscaping(now) { clock in
            let delegate = TextDelegate(limit: max(0, limits.maxCharacters), deadline: deadline, now: clock)
            let parser = XMLParser(data: xml)
            parser.shouldProcessNamespaces = true
            parser.shouldResolveExternalEntities = false
            parser.shouldReportNamespacePrefixes = false
            parser.delegate = delegate
            let success = parser.parse()
            if let failure = delegate.failure { throw failure }
            guard success || delegate.capped else { throw Failure.malformed }
            try checkTime()
            return Result(text: delegate.text, reachedCharacterCap: delegate.capped)
        }
    }

    private static let crcTable: [UInt32] = (0..<256).map { value in
        (0..<8).reduce(UInt32(value)) { crc, _ in (crc >> 1) ^ (crc & 1 == 1 ? 0xedb88320 : 0) }
    }
    static func crc32(_ bytes: Data) -> UInt32 {
        bytes.withUnsafeBytes { updateCRC32(UInt32.max, with: $0) } ^ UInt32.max
    }
    static func updateCRC32(_ crc: UInt32, with bytes: UnsafeRawBufferPointer) -> UInt32 {
        bytes.reduce(crc) { ($0 >> 8) ^ crcTable[Int(($0 ^ UInt32($1)) & 255)] }
    }

    // Internal so declaration rejection can be tested separately from preflight.
    final class TextDelegate: NSObject, XMLParserDelegate {
        let limit: Int
        let deadline: Date
        let now: () -> Date
        var text = ""
        var count = 0
        var depth = 0
        var textDepth: Int?
        var ignoredDepth: Int?
        var callbacks = 0
        var capped = false
        var failure: Failure?
        let namespaces = ["http://schemas.openxmlformats.org/wordprocessingml/2006/main",
                          "http://purl.oclc.org/ooxml/wordprocessingml/main"]
        init(limit: Int, deadline: Date, now: @escaping () -> Date) {
            self.limit = limit; self.deadline = deadline; self.now = now
        }
        func abort(_ parser: XMLParser, _ reason: Failure = .malformed) { failure = reason; parser.abortParsing() }
        func tick(_ parser: XMLParser) {
            callbacks += 1
            if callbacks % 1024 == 0, now() >= deadline { abort(parser, .timedOut) }
        }
        func append(_ value: String, _ parser: XMLParser) {
            let piece = value.prefix(max(0, limit - count))
            text.append(contentsOf: piece)
            count += piece.count
            if !value[piece.endIndex...].allSatisfy(\.isWhitespace) { capped = true; parser.abortParsing() }
        }
        func parser(_ parser: XMLParser, didStartElement name: String, namespaceURI: String?, qualifiedName: String?, attributes: [String: String]) {
            tick(parser)
            depth += 1
            guard depth <= 256 else { abort(parser); return }
            guard namespaces.contains(namespaceURI ?? "") else { return }
            if name == "delText" || name == "instrText" { ignoredDepth = depth }
            guard ignoredDepth == nil else { return }
            if name == "t" { textDepth = depth }
            if name == "tab" { append("\t", parser) }
            if name == "br" || name == "cr" { append("\n", parser) }
        }
        func parser(_ parser: XMLParser, foundCharacters string: String) {
            tick(parser)
            if textDepth != nil, ignoredDepth == nil { append(string, parser) }
        }
        func parser(_ parser: XMLParser, didEndElement name: String, namespaceURI: String?, qualifiedName: String?) {
            tick(parser)
            if textDepth == depth { textDepth = nil }
            if ignoredDepth == depth { ignoredDepth = nil }
            if namespaces.contains(namespaceURI ?? "") {
                if name == "p" { append("\n", parser) }
                if name == "tc", text.last != "\n" { append("\t", parser) }
            }
            depth -= 1
        }
        func parser(_ parser: XMLParser, foundInternalEntityDeclarationWithName: String, value: String?) { abort(parser) }
        func parser(_ parser: XMLParser, foundExternalEntityDeclarationWithName: String, publicID: String?, systemID: String?) { abort(parser) }
        func parser(_ parser: XMLParser, foundUnparsedEntityDeclarationWithName: String, publicID: String?, systemID: String?, notationName: String?) { abort(parser) }
        func parser(_ parser: XMLParser, foundElementDeclarationWithName: String, model: String) { abort(parser) }
        func parser(_ parser: XMLParser, resolveExternalEntityName: String, systemID: String?) -> Data? { abort(parser); return nil }
    }
}
