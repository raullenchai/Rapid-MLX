import Foundation
import UniformTypeIdentifiers

/// Renders a saved conversation into a file the user can keep.
///
/// Pure — no panels, no AppKit, no ambient `Date()` in any output-shaping
/// path. Everything the format depends on (the clock, the time zone) is a
/// parameter, so the whole surface is unit-testable without a UI and without
/// the tests being time-zone dependent. ``ConversationExportPanel`` owns the
/// AppKit half.
enum ConversationExport {
    enum Format: String, CaseIterable, Identifiable {
        case markdown
        case json

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .markdown: return "Markdown"
            case .json: return "JSON"
            }
        }

        var fileExtension: String {
            switch self {
            case .markdown: return "md"
            case .json: return "json"
            }
        }

        var contentType: UTType {
            switch self {
            // The save panel uses this declaration to choose and validate the
            // extension. Advertising plain text here can rewrite the proposed
            // `.md` filename to `.txt`; editor ownership is irrelevant to that
            // contract because Markdown already conforms to plain text.
            case .markdown:
                return UTType(filenameExtension: "md", conformingTo: .plainText)
                    ?? .plainText
            case .json: return .json
            }
        }
    }

    // MARK: - Markdown

    /// The conversation as it reads on screen: a question-and-answer
    /// document someone else can open and follow.
    ///
    /// **This is deliberately not a complete transcript.** It carries the two
    /// roles the chat surface actually shows as conversation turns — the
    /// user's prompts and the assistant's replies — and nothing else:
    ///
    ///   * `.tool` rows are skipped because ``ChatView`` skips them too
    ///     (`message.role != .tool`); they surface as results attached to the
    ///     assistant row that called for them, never as their own turn.
    ///   * `.system` rows are wire-only — instruction layers are merged into
    ///     the outbound body at send time, not into the visible transcript.
    ///   * `.unknown` is the cross-version sentinel (#477), not something a
    ///     reader of this document has any use for.
    ///   * Reasoning is left out entirely. It is on screen behind a
    ///     disclosure, but the whole point of exporting is handing the result
    ///     to someone, and a model's thinking is noise to that reader.
    ///
    /// What survives besides the prose — attachment names, error notices,
    /// truncation markers — is there because dropping it makes the remaining
    /// text misleading rather than merely shorter: a prompt saying "summarise
    /// this" with no sign of a file reads as nonsense, and a failed turn with
    /// its error removed reads as the model having answered with silence.
    ///
    /// ``json(_:)`` is the complete, lossless record. Splitting the jobs this
    /// way is what lets each format be good at one of them.
    ///
    /// Attachments are named, never inlined: a single 20MB image
    /// (``ChatImageAttachment/maxBytes``) becomes ~27MB of base64, which would
    /// make the common "export this chat to send to someone" case produce a
    /// file no editor will open.
    static func markdown(
        _ conversation: ChatConversation,
        timeZone: TimeZone = .current
    ) -> String {
        var out = "# \(sanitizedLine(conversation.title))\n\n"

        let stamp = timestampFormatter(timeZone)
        out += "*Created \(stamp.string(from: conversation.createdAt))"
        out += " · Updated \(stamp.string(from: conversation.updatedAt))*\n"

        // A conversation with no turns still exports — an empty file with a
        // title beats a save that silently produces nothing.
        for message in conversation.messages where isConversationTurn(message.role) {
            out += "\n---\n\n## \(heading(for: message.role))\n\n"
            out += body(of: message)
        }

        return out
    }

    /// Whether a role is one the chat surface presents as a turn of the
    /// conversation. Spelled as an exhaustive switch so a future role has to
    /// be classified here rather than defaulting into (or out of) the export.
    static func isConversationTurn(_ role: ChatMessage.Role) -> Bool {
        switch role {
        case .user, .assistant: return true
        case .system, .tool, .unknown: return false
        }
    }

    private static func heading(for role: ChatMessage.Role) -> String {
        switch role {
        case .user: return "You"
        case .assistant: return "Assistant"
        // Unreachable — filtered by ``isConversationTurn(_:)`` before we get
        // here. Spelled out rather than defaulted so adding a role that IS a
        // turn can't silently ship with a wrong label.
        case .system, .tool, .unknown: return "Note"
        }
    }

    private static func body(of message: ChatMessage) -> String {
        var out = ""

        let content = ChatTextSanitizer.sanitizeForPasteboard(message.content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !content.isEmpty {
            out += content + "\n"
            if message.contentTruncated {
                out += "\n*(response truncated)*\n"
            }
        }

        for image in message.imageAttachments {
            out += "\n> 🖼 \(sanitizedLine(image.filename)) (\(image.mimeType))\n"
        }
        for file in message.fileAttachments {
            out += "\n> 📎 \(sanitizedLine(file.filename)) (\(file.detailText))\n"
        }

        // A failed turn that exported as an empty section would read as if the
        // model had answered with nothing.
        if let error = message.errorMessage,
           !error.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out += "\n> ⚠️ \(sanitizedLine(error))\n"
        }

        if out.isEmpty { out = "*(empty)*\n" }
        return out
    }

    // MARK: - JSON

    /// Complete single-conversation export.
    ///
    /// Encodes ``ChatConversation`` as-is, attachments included, because the
    /// reason to pick JSON over Markdown is precisely that it round-trips.
    /// Stripping the attachment bytes to save space would quietly turn the
    /// archival format into a lossy one.
    ///
    /// Everything ``markdown(_:timeZone:)`` leaves out — reasoning, tool
    /// rows, instruction layers — is preserved here. Timestamps are
    /// millisecond-accurate rather than bit-exact; see ``encoder()``.
    static func json(_ conversation: ChatConversation) throws -> Data {
        try encoder().encode(conversation)
    }

    /// The whole library in one file: conversations plus the folder list, so
    /// the filing survives the round trip and not just the transcripts.
    struct Archive: Codable, Equatable {
        /// Bumped only on a breaking layout change, so a future importer can
        /// tell a v1 file from whatever replaces it without guessing.
        var schemaVersion: Int = 1
        var exportedAt: Date
        var folders: [ChatFolder]
        var conversations: [ChatConversation]
    }

    static func allChats(
        conversations: [ChatConversation],
        folders: [ChatFolder],
        exportedAt: Date
    ) throws -> Data {
        try encoder().encode(
            Archive(
                exportedAt: exportedAt,
                folders: ChatFolder.displayOrder(folders),
                conversations: conversations
            )
        )
    }

    /// Fractional-seconds ISO8601, used for every date in an exported file.
    ///
    /// A `FormatStyle` value rather than an `ISO8601DateFormatter`: it is a
    /// `Sendable` struct, so it can be a shared constant without the
    /// thread-safety caveat the old formatter class carries.
    static let iso8601Style = Date.ISO8601FormatStyle(includingFractionalSeconds: true)

    /// Shared encoder for both JSON exports.
    ///
    /// Dates are ISO8601 strings (`2026-08-13T02:02:54.277Z`), NOT Swift's
    /// default numeric encoding. The default is `timeIntervalSinceReferenceDate`
    /// — seconds since **2001-01-01**, which is Apple's epoch and nobody
    /// else's. Written into a file meant to be read and scripted against, that
    /// number is a trap: parsed as a Unix timestamp, which is what any reader
    /// will reasonably assume, it silently yields a date 31 years off. Not an
    /// error, not a crash — just a wrong date that looks entirely plausible.
    ///
    /// The cost is real and accepted: ISO8601 caps at milliseconds, so a
    /// round trip rounds sub-millisecond precision away. For message
    /// timestamps that is worth nothing, and being unambiguously readable is
    /// worth a lot. (This is why the format is described as a faithful
    /// *record* rather than a bit-exact one — the transcript, attachments and
    /// filing all survive exactly; the clock is millisecond-accurate.)
    ///
    /// Note this deliberately diverges from ``ConversationStore``'s on-disk
    /// encoding, which keeps the native numeric form. The store is private
    /// app state that nothing outside the app ever reads; an export is the
    /// opposite, and optimising both for their own audience beats making one
    /// file format serve two of them.
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        // Sorted keys so two exports of unchanged history diff cleanly — the
        // point of a backup you can keep in version control.
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(date.formatted(iso8601Style))
        }
        return encoder
    }

    /// The matching decoder. Exposed so anything reading an exported file
    /// (tests today, an importer later) can't drift from ``encoder()``.
    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let raw = try container.decode(String.self)
            guard let date = try? Date(raw, strategy: iso8601Style) else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Expected an ISO8601 date, got “\(raw)”"
                )
            }
            return date
        }
        return decoder
    }

    // MARK: - Filenames

    static func defaultFilename(
        for conversation: ChatConversation,
        format: Format,
        at date: Date,
        timeZone: TimeZone = .current
    ) -> String {
        let slug = filenameSlug(conversation.title)
        return "\(slug)-\(fileStamp(date, timeZone: timeZone)).\(format.fileExtension)"
    }

    static func defaultArchiveFilename(
        at date: Date,
        timeZone: TimeZone = .current
    ) -> String {
        "rapid-mlx-chats-\(fileStamp(date, timeZone: timeZone)).json"
    }

    /// Title → a filename component that is safe on every filesystem the app
    /// can save to, including the case-insensitive, `:`-hostile ones.
    static func filenameSlug(_ title: String, maxLength: Int = 48) -> String {
        let mapped = title.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(mapped)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
            .lowercased()
        if collapsed.isEmpty { return "chat" }
        return String(collapsed.prefix(maxLength))
    }

    private static func fileStamp(_ date: Date, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd-HHmm"
        return formatter.string(from: date)
    }

    private static func timestampFormatter(_ timeZone: TimeZone) -> DateFormatter {
        let formatter = DateFormatter()
        // POSIX locale + explicit format: the header is a machine-stable
        // record, and a localised one would make the file's content depend on
        // the exporting machine's region settings.
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }

    /// Collapse a value into a single line so it can't break out of the
    /// heading / blockquote it is being interpolated into.
    private static func sanitizedLine(_ raw: String) -> String {
        ChatTextSanitizer.sanitizeForPasteboard(raw)
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
