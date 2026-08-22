import Foundation
import Testing
@testable import Rapid

@Suite("Chat image attachments")
struct ChatImageAttachmentTests {
    @Test("multimodal user message encodes text followed by image_url data URI")
    func wireEncoding() throws {
        let attachment = try ChatImageAttachment(
            filename: "photo.png",
            mimeType: "image/png",
            data: Data([0x89, 0x50, 0x4e, 0x47])
        )
        let message = ChatMessage(
            role: .user,
            content: "What is this?",
            imageAttachments: [attachment]
        )
        let encoded = try JSONEncoder().encode(Wire.Message(from: message))
        let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let parts = try #require(json["content"] as? [[String: Any]])
        #expect(parts.count == 2)
        #expect(parts[0]["type"] as? String == "text")
        #expect(parts[0]["text"] as? String == "What is this?")
        #expect(parts[1]["type"] as? String == "image_url")
        let imageURL = try #require(parts[1]["image_url"] as? [String: Any])
        #expect((imageURL["url"] as? String)?.hasPrefix("data:image/png;base64,") == true)
    }

    @Test("text-only aliases omit images from existing conversation history")
    func textOnlyAliasOmitsHistoricalImages() throws {
        let attachment = try ChatImageAttachment(
            filename: "photo.png",
            mimeType: "image/png",
            data: Data([0x89, 0x50, 0x4e, 0x47])
        )
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-4b-4bit",
            messages: [
                ChatMessage(
                    role: .user,
                    content: "What is this?",
                    imageAttachments: [attachment]
                )
            ]
        )

        let encoded = try JSONEncoder().encode(request.messages[0])
        let json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(json["content"] as? String == "What is this?")
        #expect(String(data: encoded, encoding: .utf8)?.contains("data:image/") == false)
    }

    @Test("authoritative false capability strips images even from a vision-looking alias")
    func customVisionLookingAliasOmitsImages() throws {
        let attachment = try ChatImageAttachment(
            filename: "photo.png", mimeType: "image/png", data: Data("image".utf8)
        )
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-company-tuned",
            messages: [ChatMessage(role: .user, content: "Explain", imageAttachments: [attachment])],
            supportsImageInput: false
        )
        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains("data:image/"))
    }

    @Test("a new image replaces older image payloads on the wire")
    func newImageBecomesAttachmentFocus() throws {
        let first = try ChatImageAttachment(
            filename: "race.png", mimeType: "image/png", data: Data("race".utf8)
        )
        let second = try ChatImageAttachment(
            filename: "cheetah.png", mimeType: "image/png", data: Data("cheetah".utf8)
        )
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-9b-4bit",
            messages: [
                ChatMessage(role: .user, content: "What is this?", imageAttachments: [first]),
                ChatMessage(role: .assistant, content: "A race."),
                ChatMessage(role: .user, content: "What is this?", imageAttachments: [second]),
            ]
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains(first.data.base64EncodedString()))
        #expect(wire.contains(second.data.base64EncodedString()))
    }

    @Test("a new document does not resend an older image")
    func documentClearsHistoricalImageFocus() throws {
        let image = try ChatImageAttachment(
            filename: "race.png", mimeType: "image/png", data: Data("race".utf8)
        )
        let document = try ChatFileAttachment(
            filename: "statement.pdf",
            kind: .pdf,
            extractedText: "Current statement content",
            sourceByteCount: 25,
            pageCount: 1
        )
        let request = ChatStreamClient.Request(
            alias: "gemma-4-e2b-4bit",
            messages: [
                ChatMessage(role: .user, content: "What is this?", imageAttachments: [image]),
                ChatMessage(role: .assistant, content: "A race."),
                ChatMessage(role: .user, content: "Review this", fileAttachments: [document]),
            ]
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains("data:image/"))
        #expect(wire.contains("Current statement content"))
    }

    @Test("plain-text follow-up keeps only the most recent image")
    func followUpKeepsLatestImageFocus() throws {
        let first = try ChatImageAttachment(
            filename: "race.png", mimeType: "image/png", data: Data("race".utf8)
        )
        let second = try ChatImageAttachment(
            filename: "cheetah.png", mimeType: "image/png", data: Data("cheetah".utf8)
        )
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-9b-4bit",
            messages: [
                ChatMessage(role: .user, content: "First", imageAttachments: [first]),
                ChatMessage(role: .assistant, content: "A race."),
                ChatMessage(role: .user, content: "Second", imageAttachments: [second]),
                ChatMessage(role: .assistant, content: "A cheetah."),
                ChatMessage(role: .user, content: "What color is it?"),
            ]
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains(first.data.base64EncodedString()))
        #expect(wire.contains(second.data.base64EncodedString()))
    }

    @Test("plain-text follow-up after a document does not resurrect an older image")
    func documentFollowUpKeepsDocumentFocus() throws {
        let image = try ChatImageAttachment(
            filename: "race.png", mimeType: "image/png", data: Data("race".utf8)
        )
        let document = try ChatFileAttachment(
            filename: "statement.pdf",
            kind: .pdf,
            extractedText: "Total: 42",
            sourceByteCount: 9,
            pageCount: 1
        )
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-9b-4bit",
            messages: [
                ChatMessage(role: .user, content: "First", imageAttachments: [image]),
                ChatMessage(role: .assistant, content: "A race."),
                ChatMessage(role: .user, content: "Review", fileAttachments: [document]),
                ChatMessage(role: .assistant, content: "The total is 42."),
                ChatMessage(role: .user, content: "What is the total?"),
            ]
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains("data:image/"))
        #expect(wire.contains("Total: 42"))
    }

    @Test("attachments survive conversation persistence and old messages default empty")
    func codableCompatibility() throws {
        let attachment = try ChatImageAttachment(
            filename: "photo.jpg", mimeType: "image/jpeg", data: Data([1, 2, 3])
        )
        let original = ChatMessage(role: .user, content: "look", imageAttachments: [attachment])
        let restored = try JSONDecoder().decode(
            ChatMessage.self, from: JSONEncoder().encode(original)
        )
        #expect(restored.imageAttachments == [attachment])

        var object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as? [String: Any]
        )
        object.removeValue(forKey: "imageAttachments")
        let legacyData = try JSONSerialization.data(withJSONObject: object)
        #expect(try JSONDecoder().decode(ChatMessage.self, from: legacyData).imageAttachments.isEmpty)
    }

    @Test("20 MB cap and accepted MIME types are enforced")
    func validation() throws {
        #expect(throws: ChatImageAttachment.ValidationError.self) {
            try ChatImageAttachment(
                filename: "huge.png",
                mimeType: "image/png",
                data: Data(count: ChatImageAttachment.maxBytes + 1)
            )
        }
        #expect(throws: ChatImageAttachment.ValidationError.self) {
            try ChatImageAttachment(filename: "x.webp", mimeType: "image/webp", data: Data())
        }
    }
}
