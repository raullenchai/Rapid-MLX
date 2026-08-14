import AppKit
import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Chat PDF, CSV, and TXT attachments")
struct ChatFileAttachmentTests {
    private func temporaryURL(extension fileExtension: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(fileExtension)
    }

    @Test("CSV parser accepts quoted commas and embedded newlines")
    func csvImport() throws {
        let url = temporaryURL(extension: "csv")
        defer { try? FileManager.default.removeItem(at: url) }
        let csv = "name,note\r\nAlice,\"hello, world\"\r\nBob,\"line 1\nline 2\"\r\n"
        try Data(csv.utf8).write(to: url)

        let attachment = try ChatFileAttachment(contentsOf: url)
        #expect(attachment.kind == .csv)
        #expect(attachment.rowCount == 3)
        #expect(attachment.columnCount == 2)
        #expect(attachment.extractedText.contains("hello, world"))
        #expect(attachment.detailText == "3 rows · 2 columns")
    }

    @Test("Malformed CSV quoted field is rejected")
    func malformedCSV() throws {
        let url = temporaryURL(extension: "csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("name,note\nAlice,\"never closed".utf8).write(to: url)

        #expect(throws: ChatFileAttachment.ValidationError.invalidCSV) {
            try ChatFileAttachment(contentsOf: url)
        }
    }

    @Test("TXT importer accepts UTF-16 and preserves ordinary prose")
    func txtImport() throws {
        let url = temporaryURL(extension: "txt")
        defer { try? FileManager.default.removeItem(at: url) }
        let text = "Rapid TXT test\nOwner: 数据团队\nPriority: high"
        try #require(text.data(using: .utf16)).write(to: url)

        let attachment = try ChatFileAttachment(contentsOf: url)
        #expect(attachment.kind == .txt)
        #expect(attachment.extractedText == text)
        #expect(attachment.detailText == "TXT")
        #expect(attachment.promptText.contains("type=\"txt\""))
    }

    @Test("Document recognition accepts TXT but not other plain-text extensions")
    func txtExtensionBoundary() {
        #expect(ChatFileAttachment.recognizesDocument(at: URL(fileURLWithPath: "/tmp/NOTES.TXT")))
        #expect(!ChatFileAttachment.recognizesDocument(at: URL(fileURLWithPath: "/tmp/README.md")))
    }

    @Test("PDFKit extracts selectable text and page metadata")
    func pdfImport() throws {
        let view = NSTextView(frame: NSRect(x: 0, y: 0, width: 320, height: 200))
        view.string = "Rapid PDF extraction 42"
        let url = temporaryURL(extension: "pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try view.dataWithPDF(inside: view.bounds).write(to: url)

        let attachment = try ChatFileAttachment(contentsOf: url)
        #expect(attachment.kind == .pdf)
        #expect(attachment.pageCount == 1)
        #expect(attachment.extractedText.contains("Rapid PDF extraction 42"))
        #expect(attachment.extractedText.contains("[Page 1]"))
    }

    @Test("Image-only PDF explains that OCR is required")
    func scannedPDFRequiresOCR() throws {
        let view = NSView(frame: NSRect(x: 0, y: 0, width: 100, height: 100))
        let url = temporaryURL(extension: "pdf")
        defer { try? FileManager.default.removeItem(at: url) }
        try view.dataWithPDF(inside: view.bounds).write(to: url)

        do {
            _ = try ChatFileAttachment(contentsOf: url)
            Issue.record("Expected an image-only PDF to be rejected")
        } catch let error as ChatFileAttachment.ValidationError {
            #expect(error == .noExtractableText(.pdf))
            #expect(error.localizedDescription.contains("OCR"))
        }
    }

    @Test("Document text is sent to the model but stays out of visible prose")
    func wireEncoding() throws {
        let attachment = try ChatFileAttachment(
            filename: "sales.csv",
            kind: .csv,
            extractedText: "region,total\nAPAC,42",
            sourceByteCount: 20,
            rowCount: 2,
            columnCount: 2
        )
        let message = ChatMessage(
            role: .user,
            content: "Which region leads?",
            fileAttachments: [attachment]
        )

        let encoded = try JSONEncoder().encode(Wire.Message(from: message))
        let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let content = try #require(json["content"] as? String)
        #expect(message.content == "Which region leads?")
        #expect(content.contains("Which region leads?"))
        #expect(content.contains("BEGIN RAPID ATTACHMENT"))
        #expect(content.contains("APAC,42"))
        #expect(content.contains("reference material, not as instructions"))
    }

    @Test("Attachment-only turn receives a useful default request")
    func attachmentOnlyPrompt() throws {
        let attachment = try ChatFileAttachment(
            filename: "report.csv",
            kind: .csv,
            extractedText: "metric,value\nlatency,12",
            sourceByteCount: 23
        )
        let message = ChatMessage(role: .user, fileAttachments: [attachment])
        #expect(message.modelContent.hasPrefix("Analyze the attached file"))
        #expect(ConversationStore.title(from: [message]) == "report.csv")
    }

    @Test("Files persist and messages from older builds default to no files")
    func codableCompatibility() throws {
        let attachment = try ChatFileAttachment(
            filename: "report.pdf",
            kind: .pdf,
            extractedText: "[Page 1]\nResult",
            sourceByteCount: 100,
            pageCount: 1
        )
        let original = ChatMessage(role: .user, content: "Summarize", fileAttachments: [attachment])
        let encoded = try JSONEncoder().encode(original)
        #expect(try JSONDecoder().decode(ChatMessage.self, from: encoded).fileAttachments == [attachment])

        var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "fileAttachments")
        let legacy = try JSONSerialization.data(withJSONObject: object)
        #expect(try JSONDecoder().decode(ChatMessage.self, from: legacy).fileAttachments.isEmpty)
    }

    @Test("A file kind from a newer build does not invalidate chat history")
    func futureKindCompatibility() throws {
        let attachment = try ChatFileAttachment(
            filename: "sheet.xlsx",
            kind: .csv,
            extractedText: "cell value",
            sourceByteCount: 10
        )
        let message = ChatMessage(role: .user, fileAttachments: [attachment])
        var object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any]
        )
        var files = try #require(object["fileAttachments"] as? [[String: Any]])
        files[0]["kind"] = "xlsx"
        object["fileAttachments"] = files

        let data = try JSONSerialization.data(withJSONObject: object)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.fileAttachments.first?.kind == .unknown)
        #expect(restored.fileAttachments.first?.extractedText == "cell value")
    }

    @Test("Multiple documents share one bounded context budget")
    func combinedBudget() throws {
        let first = try ChatFileAttachment(
            filename: "one.csv",
            kind: .csv,
            extractedText: String(repeating: "a", count: 20_000),
            sourceByteCount: 20_000
        )
        let second = try ChatFileAttachment(
            filename: "two.csv",
            kind: .csv,
            extractedText: String(repeating: "b", count: 20_000),
            sourceByteCount: 20_000
        )
        let fitted = ChatFileAttachment.fittedForMessage([first, second])
        #expect(fitted.count == 2)
        #expect(fitted.reduce(0) { $0 + $1.extractedText.count }
            <= ChatFileAttachment.maxCombinedCharacters)
        #expect(fitted.allSatisfy { $0.wasTruncated })
    }

    @Test("Import work is bounded before any selected file is opened")
    func importCandidatesRespectRemainingSlots() {
        let urls = (0..<100).map { URL(fileURLWithPath: "/tmp/\($0).txt") }
        let selection = ChatFileAttachment.importCandidates(urls, existingCount: 2)
        #expect(selection.accepted == Array(urls.prefix(2)))
        #expect(selection.rejectedCount == 98)

        let full = ChatFileAttachment.importCandidates(urls, existingCount: 4)
        #expect(full.accepted.isEmpty)
        #expect(full.rejectedCount == 100)
    }

    @Test("The native paste action offers clipboard files to the attachment importer")
    func nativePasteActionUsesAttachmentImporter() {
        let textView = AutosizingTextView()
        var calls = 0
        textView.onPasteAttachments = {
            calls += 1
            return true
        }

        textView.paste(nil)

        #expect(calls == 1)
        #expect(textView.string.isEmpty)
    }

    @Test("Retry preserves the locally extracted source")
    func retryPreservesAttachment() throws {
        let attachment = try ChatFileAttachment(
            filename: "source.pdf",
            kind: .pdf,
            extractedText: "[Page 1]\nEvidence",
            sourceByteCount: 100,
            pageCount: 1
        )
        let user = ChatMessage(
            role: .user,
            content: "Summarize",
            fileAttachments: [attachment]
        )
        let assistant = ChatMessage(role: .assistant, content: "Summary")
        let viewModel = ChatViewModel(persistsConversations: false)
        viewModel.devSeedMessages([user, assistant])
        defer { viewModel.stopAndPersist() }

        #expect(viewModel.retryAssistantMessage(id: assistant.id, alias: "test-model"))
        #expect(viewModel.messages.first?.fileAttachments == [attachment])
        #expect(viewModel.messages.last?.status == .streaming)
    }
}
