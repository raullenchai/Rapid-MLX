import Foundation
import Testing
@testable import Rapid

/// Conversation export: the Markdown renderer and the JSON archive.
///
/// All of ``ConversationExport`` is pure, so these drive it directly rather
/// than through a save panel — the AppKit half (``ConversationExportPanel``)
/// only picks a URL and writes the bytes these functions return.
@Suite("Conversation export")
struct ConversationExportTests {
    @Test("Export formats advertise their real file types")
    func exportContentTypesMatchExtensions() {
        #expect(
            ConversationExport.Format.markdown.contentType.preferredFilenameExtension
                == "md"
        )
        #expect(ConversationExport.Format.markdown.contentType.conforms(to: .plainText))
        #expect(ConversationExport.Format.json.contentType == .json)
    }

    /// Fixed instant + fixed zone: the header is formatted with an explicit
    /// POSIX format, and a test that read the machine's zone would pass in
    /// one timezone and fail in another.
    private let utc = TimeZone(identifier: "UTC")!
    private var epoch: Date { Date(timeIntervalSince1970: 1_770_000_000) }

    private func conversation(
        title: String = "How do I pin a chat",
        messages: [ChatMessage] = [],
        customInstructions: String? = nil
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(),
            title: title,
            messages: messages,
            createdAt: epoch,
            updatedAt: epoch.addingTimeInterval(3600),
            customInstructions: customInstructions
        )
    }

    // MARK: - Markdown

    @Test("Renders the title and a stable, zone-explicit timestamp header")
    func markdownHeader() {
        let md = ConversationExport.markdown(conversation(), timeZone: utc)
        #expect(md.hasPrefix("# How do I pin a chat\n"))
        #expect(md.contains("*Created 2026-02-02 02:40"))
        #expect(md.contains("Updated 2026-02-02 03:40*"))
    }

    @Test("Exports the question-and-answer turns and only those")
    func markdownIsQuestionAndAnswerOnly() {
        let md = ConversationExport.markdown(
            conversation(messages: [
                ChatMessage(role: .user, content: "hi"),
                ChatMessage(role: .assistant, content: "hello"),
                ChatMessage(role: .system, content: "be brief"),
                ChatMessage(role: .tool, content: "{\"ok\":true}"),
                ChatMessage(role: .unknown, content: "from a newer build"),
            ]),
            timeZone: utc
        )
        #expect(md.contains("## You"))
        #expect(md.contains("## Assistant"))
        // The document has to read like what the chat surface shows. ChatView
        // renders `role != .tool` and instruction rows never reach the visible
        // transcript at all, so none of these belong in a file meant to be
        // handed to someone.
        #expect(!md.contains("be brief"))
        #expect(!md.contains("{\"ok\":true}"))
        #expect(!md.contains("from a newer build"))
        #expect(!md.contains("## Note"))
    }

    @Test("Roles are classified explicitly, not by falling through a default")
    func conversationTurnClassification() {
        #expect(ConversationExport.isConversationTurn(.user))
        #expect(ConversationExport.isConversationTurn(.assistant))
        #expect(!ConversationExport.isConversationTurn(.system))
        #expect(!ConversationExport.isConversationTurn(.tool))
        #expect(!ConversationExport.isConversationTurn(.unknown))
    }

    @Test("Reasoning is left out — the reader wants the answer, not the thinking")
    func markdownOmitsReasoning() {
        let md = ConversationExport.markdown(
            conversation(messages: [
                ChatMessage(
                    role: .assistant,
                    content: "42",
                    reasoning: "let me think about this at length"
                )
            ]),
            timeZone: utc
        )
        #expect(md.contains("42"))
        #expect(!md.contains("let me think about this at length"))
        #expect(!md.contains("<details>"))
    }

    @Test("A truncated answer says so rather than looking like the whole answer")
    func markdownTruncationMarkers() {
        let md = ConversationExport.markdown(
            conversation(messages: [
                ChatMessage(
                    role: .assistant,
                    content: "partial",
                    reasoning: "cut",
                    reasoningTruncated: true,
                    contentTruncated: true
                )
            ]),
            timeZone: utc
        )
        #expect(md.contains("*(response truncated)*"))
        // Reasoning isn't exported, so a marker about its truncation would be
        // a note about something the reader can't see.
        #expect(!md.contains("*(reasoning truncated)*"))
    }

    @Test("Attachments are listed by name, never inlined as base64")
    func markdownListsAttachmentsWithoutBytes() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        let image = try ChatImageAttachment(
            filename: "diagram.png",
            mimeType: "image/png",
            data: png
        )
        let md = ConversationExport.markdown(
            conversation(messages: [
                ChatMessage(role: .user, content: "look", imageAttachments: [image])
            ]),
            timeZone: utc
        )
        #expect(md.contains("diagram.png"))
        #expect(md.contains("image/png"))
        // The whole point of the format choice: a 20MB image must not become
        // 27MB of base64 in a file meant to be read.
        #expect(!md.contains(png.base64EncodedString()))
    }

    @Test("A failed turn exports its error instead of an empty section")
    func markdownSurfacesErrors() {
        let md = ConversationExport.markdown(
            conversation(messages: [
                ChatMessage(
                    role: .assistant,
                    content: "",
                    status: .failed,
                    errorMessage: "the model stopped responding"
                )
            ]),
            timeZone: utc
        )
        #expect(md.contains("the model stopped responding"))
        #expect(!md.contains("*(empty)*"))
    }

    @Test("An empty conversation still exports a titled file")
    func markdownEmptyConversation() {
        let md = ConversationExport.markdown(conversation(messages: []), timeZone: utc)
        #expect(md.hasPrefix("# How do I pin a chat"))
    }

    @Test("A multi-line title can't break out of the heading")
    func markdownCollapsesTitleNewlines() {
        let md = ConversationExport.markdown(
            conversation(title: "line one\nline two"),
            timeZone: utc
        )
        #expect(md.hasPrefix("# line one line two\n"))
    }

    @Test("Custom instructions stay out of the readable export")
    func markdownOmitsCustomInstructions() {
        let md = ConversationExport.markdown(
            conversation(customInstructions: "answer in French"),
            timeZone: utc
        )
        // Configuration, not conversation — it is never shown as a turn on
        // screen either. The JSON export keeps it (see `jsonRoundTrips`).
        #expect(!md.contains("answer in French"))
    }

    // MARK: - JSON

    @Test("Single-conversation JSON round-trips")
    func jsonRoundTrips() throws {
        // Everything Markdown deliberately drops — reasoning, tool rows,
        // instructions — must survive here, or the two formats have no
        // division of labour and the archive isn't one.
        //
        // Message timestamps are pinned to whole seconds so this can assert
        // exact equality; the millisecond-rounding tradeoff has its own test
        // below rather than being blurred into this one.
        let original = conversation(
            messages: [
                ChatMessage(role: .user, content: "hi", createdAt: epoch),
                ChatMessage(
                    role: .assistant,
                    content: "hello",
                    reasoning: "think",
                    createdAt: epoch
                ),
                ChatMessage(
                    role: .tool,
                    content: "{\"ok\":true}",
                    toolCallID: "call-1",
                    createdAt: epoch
                ),
            ],
            customInstructions: "answer in French"
        )
        let data = try ConversationExport.json(original)
        let restored = try ConversationExport.decoder().decode(
            ChatConversation.self, from: data
        )
        #expect(restored == original)
    }

    @Test("Dates are readable ISO8601, not a 2001-epoch number")
    func jsonDatesAreISO8601() throws {
        let data = try ConversationExport.json(
            conversation(messages: [ChatMessage(role: .user, content: "hi", createdAt: epoch)])
        )
        let text = try #require(String(data: data, encoding: .utf8))

        // Quoted: a string, not a bare number.
        #expect(text.contains("\"2026-02-02T02:40:00"))
        // The default Swift encoding would write 791692800 here — seconds
        // since 2001 — which any reader parses as a Unix timestamp and
        // silently lands 31 years out. Pinning the exact digits means nobody
        // can "simplify" the encoder back to the default without this failing.
        #expect(!text.contains("791692800"))
    }

    @Test("Sub-millisecond precision is the accepted cost of a readable date")
    func jsonDatesRoundToMilliseconds() throws {
        let precise = Date(timeIntervalSince1970: 1_770_000_000.123_456_7)
        let original = conversation(
            messages: [ChatMessage(role: .user, content: "hi", createdAt: precise)]
        )
        let restored = try ConversationExport.decoder().decode(
            ChatConversation.self,
            from: try ConversationExport.json(original)
        )
        let delta = abs(
            try #require(restored.messages.first).createdAt.timeIntervalSince(precise)
        )
        // Documented, deliberate: accurate to the millisecond, not the
        // microsecond. Anything worse would mean the format changed.
        #expect(delta < 0.001)
        #expect(restored.messages.first?.content == "hi")
    }

    @Test("The archive carries folders as well as transcripts")
    func archiveIncludesFolders() throws {
        let folder = ChatFolder(name: "Work")
        var filed = conversation(title: "Filed")
        filed.folderID = folder.id

        let data = try ConversationExport.allChats(
            conversations: [filed, conversation(title: "Loose")],
            folders: [folder],
            exportedAt: epoch
        )
        let archive = try ConversationExport.decoder().decode(
            ConversationExport.Archive.self, from: data
        )

        #expect(archive.schemaVersion == 1)
        #expect(archive.folders.map(\.name) == ["Work"])
        #expect(archive.conversations.count == 2)
        // Filing survives the round trip, not just the transcripts.
        #expect(archive.conversations.first(where: { $0.title == "Filed" })?.folderID == folder.id)
    }

    // MARK: - Filenames

    @Test("Filenames are filesystem-safe and carry a sortable stamp")
    func filenameSlugAndStamp() {
        let name = ConversationExport.defaultFilename(
            for: conversation(title: "Q3: budget / review"),
            format: .markdown,
            at: epoch,
            timeZone: utc
        )
        #expect(name == "q3-budget-review-2026-02-02-0240.md")
        #expect(!name.contains("/"))
        #expect(!name.contains(":"))
    }

    @Test("A title with nothing usable in it still yields a filename")
    func filenameFallback() {
        #expect(ConversationExport.filenameSlug("···") == "chat")
        #expect(ConversationExport.filenameSlug("") == "chat")
    }

    @Test("Archive filename is stamped and JSON-suffixed")
    func archiveFilename() {
        let name = ConversationExport.defaultArchiveFilename(at: epoch, timeZone: utc)
        #expect(name == "rapid-mlx-chats-2026-02-02-0240.json")
    }
}
