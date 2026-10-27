import Foundation

/// Incremental wire-v1 SSE framing. UTF-8 is decoded only after a complete line;
/// a network read may end inside any code point, field, or CRLF delimiter.
struct ChatSSEDecoder {
    static let maxLineBytes = 1 << 20
    static let maxEventBytes = 2 << 20

    private var line: [UInt8] = []
    private var skipLF = false
    private var firstLine = true
    private var eventBytes = 0
    private var dataLines: [String] = []
    private var eventID: String?
    private var eventType: String?

    mutating func append(_ data: Data) throws -> [ChatWireEnvelope] {
        var events: [ChatWireEnvelope] = []
        for byte in data {
            if let event = try append(byte) { events.append(event) }
        }
        return events
    }

    mutating func append(_ byte: UInt8) throws -> ChatWireEnvelope? {
        if skipLF {
            skipLF = false
            if byte == 10 { return nil }
        }
        if byte == 10 || byte == 13 {
            skipLF = byte == 13
            return try completeLine()
        }
        guard line.count < Self.maxLineBytes,
              eventBytes + line.count < Self.maxEventBytes else {
            throw ChatWireError.frameTooLarge
        }
        line.append(byte)
        return nil
    }

    /// An incomplete frame is never dispatched at EOF. The caller reconnects
    /// from the last *applied* event, including when EOF falls between frames.
    func finish() throws {
        guard line.isEmpty, dataLines.isEmpty, eventID == nil, eventType == nil else {
            throw ChatWireError.unexpectedEOF
        }
    }

    private mutating func completeLine() throws -> ChatWireEnvelope? {
        guard var text = String(bytes: line, encoding: .utf8) else {
            throw ChatWireError.invalidFrame
        }
        eventBytes += line.count + 1
        line.removeAll(keepingCapacity: true)
        guard eventBytes <= Self.maxEventBytes else { throw ChatWireError.frameTooLarge }
        if firstLine {
            firstLine = false
            if text.first == "\u{feff}" { text.removeFirst() }
        }
        if text.isEmpty {
            defer {
                dataLines.removeAll(keepingCapacity: true)
                eventID = nil
                eventType = nil
                eventBytes = 0
            }
            guard !dataLines.isEmpty else { return nil }
            let envelope = try JSONDecoder().decode(
                ChatWireEnvelope.self, from: Data(dataLines.joined(separator: "\n").utf8)
            )
            guard eventID == envelope.eventID, eventType == envelope.event.type else {
                throw ChatWireError.invalidFrame
            }
            return envelope
        }
        if text.hasPrefix(":") { return nil }
        let pieces = text.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let field = String(pieces[0])
        var value = pieces.count == 2 ? String(pieces[1]) : ""
        if value.hasPrefix(" ") { value.removeFirst() }
        switch field {
        case "data": dataLines.append(value)
        case "event": eventType = value
        case "id":
            guard !value.contains("\0") else { throw ChatWireError.invalidFrame }
            eventID = value
        default: break // SSE extensions (including retry) do not change wire events.
        }
        return nil
    }
}
