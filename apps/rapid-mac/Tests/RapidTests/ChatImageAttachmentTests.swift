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
