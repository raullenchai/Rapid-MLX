import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import Rapid

@Suite("Chat image attachments")
struct ChatImageAttachmentTests {
    @Test("decoded image dimensions are bounded before bitmap creation")
    func decodedDimensionBudget() {
        #expect(ChatImageAttachment.dimensionsFit(width: 5_000, height: 4_000))
        #expect(ChatImageAttachment.dimensionsFit(width: 8_000, height: 8_000))
        #expect(!ChatImageAttachment.dimensionsFit(width: 8_001, height: 8_000))
        #expect(!ChatImageAttachment.dimensionsFit(width: 20_000, height: 1))
        #expect(!ChatImageAttachment.dimensionsFit(width: 0, height: 4_000))
    }

    @Test("vision-bound dimensions preserve aspect ratio with a 2048-pixel long edge")
    func visionDimensionBudget() {
        #expect(ChatImageAttachment.normalizedPixelSize(width: 5_000, height: 4_000) == (2_048, 1_638))
        #expect(ChatImageAttachment.normalizedPixelSize(width: 3_840, height: 2_160) == (2_048, 1_152))
        #expect(ChatImageAttachment.normalizedPixelSize(width: 140, height: 140) == (140, 140))
    }

    @Test("large JPEG is normalized at the attachment boundary while a small PNG stays unchanged")
    func largeStandardImageNormalizes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-image-normalization-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let jpeg = root.appendingPathComponent("photo.jpg")
        try Self.writeSolidImage(to: jpeg, width: 5_000, height: 4_000, type: .jpeg)
        let normalized = try ChatImageAttachment(contentsOf: jpeg)
        let normalizedSource = try #require(
            CGImageSourceCreateWithData(normalized.data as CFData, nil)
        )
        let normalizedProperties = try #require(
            CGImageSourceCopyPropertiesAtIndex(normalizedSource, 0, nil) as? [CFString: Any]
        )
        #expect((normalizedProperties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue == 2_048)
        #expect((normalizedProperties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue == 1_638)

        let png = root.appendingPathComponent("small.png")
        try Self.writeSolidImage(to: png, width: 140, height: 140, type: .png)
        let originalPNG = try Data(contentsOf: png)
        let unchanged = try ChatImageAttachment(contentsOf: png)
        #expect(unchanged.filename == "small.png")
        #expect(unchanged.mimeType == "image/png")
        #expect(unchanged.data == originalPNG)
    }

    @Test("real HEIC fixture normalizes to truthful JPEG attachment bytes")
    func heicNormalizesAtAttachmentBoundary() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("__Snapshots__/cheetah-logo-96.heic")
        let sourceType = try fixture.resourceValues(forKeys: [.contentTypeKey]).contentType
        #expect(sourceType?.conforms(to: .heic) == true)

        let attachment = try ChatImageAttachment(contentsOf: fixture)

        #expect(attachment.filename == "cheetah-logo-96.jpg")
        #expect(attachment.mimeType == "image/jpeg")
        #expect(attachment.data.count <= ChatImageAttachment.maxBytes)
        let normalizedSource = try #require(
            CGImageSourceCreateWithData(attachment.data as CFData, nil)
        )
        #expect(CGImageSourceGetType(normalizedSource) as String? == UTType.jpeg.identifier)
        #expect(CGImageSourceGetCount(normalizedSource) == 1)
    }

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

    @Test("plain-text follow-up never inherits an image rejected by an earlier request")
    @MainActor
    func followUpDoesNotInheritRejectedImage() throws {
        let image = try ChatImageAttachment(
            filename: "photo.png", mimeType: "image/png", data: Data("photo".utf8)
        )
        let history = ChatViewModel.filterEmptyAssistantsForWire([
            ChatMessage(
                role: .user,
                content: "Describe this",
                imageAttachments: [image],
                imageDeliveryStatus: .rejected
            ),
            ChatMessage(
                role: .assistant,
                status: .failed,
                errorMessage: "This model is running text-only."
            ),
            ChatMessage(role: .user, content: "hi"),
        ])
        let request = ChatStreamClient.Request(
            alias: "vision-model",
            messages: history,
            supportsImageInput: true
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(!wire.contains("data:image/"))
        #expect(wire.contains("\"hi\""))
    }

    @Test("retrying a rejected image turn sends that image again")
    func directRetryIncludesRejectedImage() throws {
        let image = try ChatImageAttachment(
            filename: "photo.png", mimeType: "image/png", data: Data("photo".utf8)
        )
        let request = ChatStreamClient.Request(
            alias: "vision-model",
            messages: [
                ChatMessage(
                    role: .user,
                    content: "Describe this",
                    imageAttachments: [image],
                    imageDeliveryStatus: .rejected
                )
            ],
            supportsImageInput: true
        )

        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        #expect(wire.contains(image.data.base64EncodedString()))
    }

    @Test("request outcome updates only its exact persisted image turn")
    @MainActor
    func deliveryOutcomeUsesMessageIdentity() throws {
        let firstID = UUID()
        let secondID = UUID()
        var messages = [
            ChatMessage(
                id: firstID,
                role: .user,
                imageDeliveryStatus: .pending
            ),
            ChatMessage(
                id: secondID,
                role: .user,
                imageDeliveryStatus: .pending
            ),
        ]

        ChatViewModel.applyImageDeliveryEvent(
            in: &messages,
            messageID: firstID,
            event: .terminalRejection
        )

        #expect(messages[0].imageDeliveryStatus == .rejected)
        #expect(messages[1].imageDeliveryStatus == .pending)
    }

    @Test("accepted image delivery cannot regress after a later stream failure")
    @MainActor
    func acceptedDeliveryIsMonotonic() {
        let id = UUID()
        var messages = [
            ChatMessage(id: id, role: .user, imageDeliveryStatus: .pending)
        ]

        ChatViewModel.applyImageDeliveryEvent(
            in: &messages, messageID: id, event: .accepted
        )
        ChatViewModel.applyImageDeliveryEvent(
            in: &messages, messageID: id, event: .terminalRejection
        )
        #expect(messages[0].imageDeliveryStatus == .accepted)

        messages[0].imageDeliveryStatus = .rejected
        ChatViewModel.applyImageDeliveryEvent(
            in: &messages, messageID: id, event: .accepted
        )
        #expect(messages[0].imageDeliveryStatus == .accepted)
    }

    @Test("first transient failure leaves one bounded image retry")
    @MainActor
    func firstTransientFailureAllowsOneRetry() {
        let id = UUID()
        var messages = [
            ChatMessage(id: id, role: .user, imageDeliveryStatus: .pending)
        ]

        ChatViewModel.applyImageDeliveryEvent(
            in: &messages, messageID: id, event: .transientFailure
        )

        #expect(messages[0].imageDeliveryStatus == .retryable)
    }

    @Test("cancellation does not spend an image retry")
    @MainActor
    func cancellationPreservesRetryableDelivery() {
        let id = UUID()
        var messages = [
            ChatMessage(id: id, role: .user, imageDeliveryStatus: .retryable)
        ]

        ChatViewModel.applyImageDeliveryEvent(
            in: &messages, messageID: id, event: .abandoned
        )

        #expect(messages[0].imageDeliveryStatus == .retryable)
    }

    @Test("stopping an in-flight image request restores its unused retry")
    @MainActor
    func streamCancellationClearsPendingDelivery() async throws {
        ImageCancellationProtocol.reset()
        let image = try ChatImageAttachment(
            filename: "photo.png",
            mimeType: "image/png",
            data: Data("cancelled-photo".utf8)
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://image-cancellation")!,
                session: ImageCancellationProtocol.session()
            ),
            persistsConversations: false
        )

        model.send(
            "Describe this",
            alias: "vision-model",
            supportsImageInput: true,
            imageAttachments: [image]
        )
        try await Self.waitUntil { ImageCancellationProtocol.didStart }
        #expect(model.messages.first?.imageDeliveryStatus == .pending)

        model.stop()
        try await Self.waitForStream(model)

        #expect(ImageCancellationProtocol.didStop)
        #expect(model.messages.first?.imageDeliveryStatus == nil)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("a second terminal image failure quarantines only that image turn")
    @MainActor
    func secondTerminalFailureQuarantinesImage() async throws {
        ImageFailureQuarantineProtocol.reset()
        let image = try ChatImageAttachment(
            filename: "photo.png",
            mimeType: "image/png",
            data: Data("persistent-failure-photo".utf8)
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://image-failure")!,
                session: ImageFailureQuarantineProtocol.session()
            ),
            persistsConversations: false
        )

        model.send(
            "Describe this",
            alias: "vision-model",
            supportsImageInput: true,
            imageAttachments: [image]
        )
        try await Self.waitForStream(model)
        #expect(model.messages.first?.imageDeliveryStatus != .rejected)

        model.send(
            "Try that image once more",
            alias: "vision-model",
            supportsImageInput: true
        )
        try await Self.waitForStream(model)
        #expect(model.messages.first?.imageDeliveryStatus == .rejected)
        let secondFailure = try #require(model.messages.last)
        #expect(secondFailure.status == .failed)
        #expect(!(secondFailure.errorMessage ?? "").isEmpty)
        #expect(!(secondFailure.errorMessage ?? "").contains("temporary runtime failure"))
        #expect(!(secondFailure.errorMessage ?? "").contains("HTTP 500"))

        model.send(
            "Continue without the image",
            alias: "vision-model",
            supportsImageInput: true
        )
        try await Self.waitForStream(model)

        let requestBodies = ImageFailureQuarantineProtocol.capturedBodies()
        try #require(requestBodies.count == 3)
        let encodedImage = image.data.base64EncodedString()
        #expect(String(decoding: requestBodies[0], as: UTF8.self).contains(encodedImage))
        #expect(String(decoding: requestBodies[1], as: UTF8.self).contains(encodedImage))
        #expect(!String(decoding: requestBodies[2], as: UTF8.self).contains(encodedImage))
        #expect(model.messages.last?.content == "recovered")
        #expect(model.messages.last?.status == .complete)
    }

    @Test("model-start failure clears pending image delivery before persistence")
    @MainActor
    func startupFailureClearsPendingDelivery() throws {
        let imageID = UUID()
        let model = ChatViewModel(persistsConversations: false)
        model.devSeedMessages([
            ChatMessage(id: imageID, role: .user, imageDeliveryStatus: .pending),
            ChatMessage(role: .assistant, status: .streaming),
        ])

        model.finishWithStartupFailure(
            placeholderIndex: 1,
            alias: "model",
            imageMessageID: imageID
        )

        #expect(model.messages[0].imageDeliveryStatus == nil)
        #expect(model.messages[1].status == .failed)
    }

    @Test("cancelled model start clears pending image delivery")
    @MainActor
    func startupCancellationClearsPendingDelivery() {
        let imageID = UUID()
        let model = ChatViewModel(persistsConversations: false)
        model.devSeedMessages([
            ChatMessage(id: imageID, role: .user, imageDeliveryStatus: .pending),
            ChatMessage(role: .assistant, status: .streaming),
        ])

        model.finishStartupCancellation(
            placeholderIndex: 1,
            imageMessageID: imageID
        )

        #expect(model.messages[0].imageDeliveryStatus == nil)
        #expect(model.messages[1].status == .complete)
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

    @Test("image delivery outcome persists and legacy turns remain follow-up compatible")
    func deliveryOutcomeCodableCompatibility() throws {
        let attachment = try ChatImageAttachment(
            filename: "photo.jpg", mimeType: "image/jpeg", data: Data([1, 2, 3])
        )
        let rejected = ChatMessage(
            role: .user,
            content: "look",
            imageAttachments: [attachment],
            imageDeliveryStatus: .rejected
        )
        let restored = try JSONDecoder().decode(
            ChatMessage.self,
            from: JSONEncoder().encode(rejected)
        )
        #expect(restored.imageDeliveryStatus == .rejected)

        let retryable = ChatMessage(
            role: .user,
            content: "look",
            imageAttachments: [attachment],
            imageDeliveryStatus: .retryable
        )
        #expect(
            try JSONDecoder().decode(
                ChatMessage.self,
                from: JSONEncoder().encode(retryable)
            ).imageDeliveryStatus == .retryable
        )

        var legacyObject = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(rejected)) as? [String: Any]
        )
        legacyObject.removeValue(forKey: "imageDeliveryStatus")
        let legacy = try JSONDecoder().decode(
            ChatMessage.self,
            from: JSONSerialization.data(withJSONObject: legacyObject)
        )
        #expect(legacy.imageDeliveryStatus == nil)

        legacyObject["imageDeliveryStatus"] = "future-outcome"
        let forward = try JSONDecoder().decode(
            ChatMessage.self,
            from: JSONSerialization.data(withJSONObject: legacyObject)
        )
        #expect(forward.imageDeliveryStatus == .rejected)
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

    private func tempDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-img-budget-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// Creates real files with the given byte sizes so
    /// `importCandidates` (which reads `fileSizeKey`) sees real sizes.
    private func materialize(_ files: [(name: String, size: Int)], in root: URL) throws -> [URL] {
        try files.map { entry in
            let url = root.appendingPathComponent(entry.name)
            try Data(repeating: 0, count: entry.size).write(to: url)
            return url
        }
    }

    /// A raw byte count whose encoded data-URL size is *at most* `encoded`.
    /// Lets a test express "this file encodes to the image budget" without
    /// hard-coding base64 expansion in every assertion.
    private func rawCountForEncoded(_ encoded: Int, mimeType: String = "image/png") -> Int {
        let prefix = ChatImageAttachment.encodedDataURLByteCount(mimeType: mimeType, rawBytes: 0)
        let base64 = max(0, encoded - prefix)
        return (base64 / 4) * 3
    }

    @Test("importCandidates enforces the count and aggregate-byte boundaries")
    func importCandidatesBoundaries() throws {
        let root = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        // Count boundary: 6 files, 4-slot budget => 4 accepted, 2 rejected,
        // and the binding limit is the count, not bytes.
        let six = try materialize([
            ("a.png", 100), ("b.png", 100), ("c.png", 100),
            ("d.png", 100), ("e.png", 100), ("f.png", 100),
        ], in: root)
        let countSelection = ChatImageAttachment.importCandidates(
            six, existingCount: 0, existingBytes: 0
        )
        #expect(countSelection.accepted == Array(six.prefix(ChatImageAttachment.maxImagesPerMessage)))
        #expect(countSelection.rejectedCount == 6 - ChatImageAttachment.maxImagesPerMessage)
        #expect(countSelection.limit == .count)

        // Aggregate-byte boundary: an image whose *encoded* form is over the
        // remaining budget is skipped (its per-file 20 MB check still happens
        // at read time), while a following image that fits is still admitted,
        // and the binding limit is bytes.
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        // one raw byte more than encodes to the full budget -> encoded > budget
        let bigRaw = rawCountForEncoded(budget) + 1
        let halfRaw = rawCountForEncoded(budget / 2)
        let big = try materialize([
            ("big.png", bigRaw), ("small.png", 100),
        ], in: root)
        let bigSelection = ChatImageAttachment.importCandidates(
            big, existingCount: 0, existingBytes: 0
        )
        #expect(bigSelection.accepted == [big[1]])
        #expect(bigSelection.rejectedCount == 1)
        #expect(bigSelection.limit == .bytes)

        let thirds = try materialize([
            ("one.png", halfRaw), ("two.png", halfRaw), ("three.png", 100),
        ], in: root)
        let thirdsSelection = ChatImageAttachment.importCandidates(
            thirds, existingCount: 0, existingBytes: 0
        )
        // Two encoded-half-budget images fit; the third's encoded bytes push
        // the set over the combined budget.
        #expect(thirdsSelection.accepted == Array(thirds.prefix(2)))
        #expect(thirdsSelection.rejectedCount == 1)

        // Existing budget/count are honored: no count slots remain.
        let withExisting = ChatImageAttachment.importCandidates(
            [thirds[2]], existingCount: 4, existingBytes: 0
        )
        #expect(withExisting.accepted.isEmpty)
        #expect(withExisting.rejectedCount == 1)
    }

    @Test("importCandidates admits a mixed accepted/rejected batch in selection order")
    func importCandidatesMixedBatch() throws {
        let root = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        let oversizeRaw = rawCountForEncoded(budget) + 1
        // First fits, second is over the aggregate budget, third fits again
        // only if the second was skipped and bytes remain.
        let urls = try materialize([
            ("keep.png", 10), ("oversize.png", oversizeRaw), ("keep2.png", 10),
        ], in: root)
        let selection = ChatImageAttachment.importCandidates(
            urls, existingCount: 0, existingBytes: 0
        )
        #expect(selection.accepted == [urls[0], urls[2]])
        #expect(selection.rejectedCount == 1)
    }

    @Test("fittedForMessage caps count and keeps only images that fit the combined budget")
    func fittedForMessageBoundaries() throws {
        let small = try ChatImageAttachment(
            filename: "a.png", mimeType: "image/png", data: Data(repeating: 1, count: 10)
        )
        let big = try ChatImageAttachment(
            filename: "b.png", mimeType: "image/png",
            data: Data(repeating: 2, count: rawCountForEncoded(
                ChatImageAttachment.maxCombinedEncodedImageBytes
            ))
        )
        let many = try ChatImageAttachment(
            filename: "c.png", mimeType: "image/png", data: Data(repeating: 3, count: 10)
        )
        // A single image that fills the whole budget leaves room for nothing else.
        #expect(ChatImageAttachment.fittedForMessage([big, small, many]) == [big])
        // When the big image is not first, it cannot coexist with the two small
        // images that already used some budget; the small ones are kept.
        #expect(ChatImageAttachment.fittedForMessage([small, big, many]) == [small, many])
        // Count cap: 6 small images fit bytes but not count.
        let sixSmall = try (0..<6).map {
            try ChatImageAttachment(
                filename: "\($0).png", mimeType: "image/png", data: Data(repeating: 1, count: 10)
            )
        }
        #expect(ChatImageAttachment.fittedForMessage(sixSmall).count
            == ChatImageAttachment.maxImagesPerMessage)
    }

    @Test("wire request never exceeds the accepted, bounded image set")
    func wireRequestStaysWithinImageBudget() throws {
        // Distinct byte payloads keep each base64 encoding unique so the wire
        // can be checked for exactly which images were admitted.
        let many = try (0..<ChatImageAttachment.maxImagesPerMessage + 5).map { index in
            try ChatImageAttachment(
                filename: "\(index).png",
                mimeType: "image/png",
                data: Data(repeating: UInt8(64 + index), count: 10)
            )
        }
        let admitted = ChatImageAttachment.fittedForMessage(many)
        #expect(admitted.count == ChatImageAttachment.maxImagesPerMessage)
        let request = ChatStreamClient.Request(
            alias: "qwen3.5-9b-4bit",
            messages: [
                ChatMessage(role: .user, content: "Describe", imageAttachments: admitted)
            ]
        )
        let wire = String(decoding: try JSONEncoder().encode(request.messages), as: UTF8.self)
        for attachment in admitted {
            #expect(wire.contains(attachment.data.base64EncodedString()))
        }
        let rejected = Array(many.dropFirst(ChatImageAttachment.maxImagesPerMessage))
        for rejectedImage in rejected {
            #expect(!wire.contains(rejectedImage.data.base64EncodedString()))
        }
    }

    @Test("wire construction inherently bounds an over-limit message (restore/upgrade path)")
    func wireConstructionBoundsOverLimitMessage() throws {
        // An upgraded or restored profile can carry a message that predates the
        // budget, with more images than the wire may carry. Wire.Message.init
        // must re-apply the budget itself, not trust the message to be pre-fit.
        // First, more images than the count cap.
        let tooMany = try (0..<ChatImageAttachment.maxImagesPerMessage + 5).map {
            try ChatImageAttachment(
                filename: "\($0).png", mimeType: "image/png",
                data: Data(repeating: UInt8(64 + $0), count: 10)
            )
        }
        let countMessage = ChatMessage(
            role: .user, content: "Describe", imageAttachments: tooMany
        )
        let countWire = String(
            decoding: try JSONEncoder().encode(
                ChatStreamClient.Request(
                    alias: "qwen3.5-9b-4bit",
                    messages: [countMessage]
                ).messages
            ),
            as: UTF8.self
        )
        let expectedKeep = ChatImageAttachment.fittedForMessage(tooMany)
        #expect(expectedKeep.count == ChatImageAttachment.maxImagesPerMessage)
        for kept in expectedKeep {
            #expect(countWire.contains(kept.data.base64EncodedString()))
        }
        for dropped in tooMany.dropFirst(ChatImageAttachment.maxImagesPerMessage) {
            #expect(!countWire.contains(dropped.data.base64EncodedString()))
        }

        // Second, fewer images whose combined *encoded* bytes exceed the wire
        // budget. A message with several near-budget images must have all but
        // the fitting subset stripped at the wire, even though each is under
        // the 20 MB per-file cap. Distinct byte payloads keep each base64
        // encoding unique so the wire reveals exactly which image survived.
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        let heavyRaw = rawCountForEncoded(budget / 2) + 1  // encodes to > budget/2
        let heavies = try (0..<3).map {
            try ChatImageAttachment(
                filename: "heavy\($0).png", mimeType: "image/png",
                data: Data(repeating: UInt8(7 + $0), count: heavyRaw)
            )
        }
        // Two of these cannot co-fit (each is > half the encoded budget).
        #expect(ChatImageAttachment.fittedForMessage(heavies).count == 1)
        let heavyMessage = ChatMessage(
            role: .user,
            content: "Describe",
            imageAttachments: heavies
        )
        let heavyWire = String(
            decoding: try JSONEncoder().encode(
                ChatStreamClient.Request(
                    alias: "qwen3.5-9b-4bit",
                    messages: [heavyMessage]
                ).messages
            ),
            as: UTF8.self
        )
        // The wired message carries only the one image that fits, not all three
        // the over-limit message held.
        #expect(heavyWire.contains(heavies[0].data.base64EncodedString()))
        #expect(!heavyWire.contains(heavies[1].data.base64EncodedString()))
        #expect(!heavyWire.contains(heavies[2].data.base64EncodedString()))
    }

    private static func writeSolidImage(
        to url: URL,
        width: Int,
        height: Int,
        type: UTType
    ) throws {
        let context = try #require(CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.setFillColor(red: 0.9, green: 0.4, blue: 0.1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(context.makeImage())
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(
            output,
            type.identifier as CFString,
            1,
            nil
        ))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        try (output as Data).write(to: url)
    }

    @MainActor
    private static func waitForStream(_ model: ChatViewModel) async throws {
        // The full Swift suite runs hundreds of MainActor tests in parallel.
        // Keep a hard bound, but leave enough room for actor scheduling under
        // that load so this transport test measures behavior, not queue delay.
        let deadline = ContinuousClock.now.advanced(by: .seconds(30))
        while model.isStreaming, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        try #require(!model.isStreaming, "the canned chat request must finish")
    }

    @MainActor
    private static func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(30))
        while !condition(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        try #require(condition(), "the canned transport event must arrive")
    }
}

private final class ImageCancellationProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var started = false
    nonisolated(unsafe) private static var stopped = false

    static var didStart: Bool {
        lock.lock()
        defer { lock.unlock() }
        return started
    }

    static var didStop: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    static func reset() {
        lock.lock()
        started = false
        stopped = false
        lock.unlock()
    }

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ImageCancellationProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.lock()
        Self.started = true
        Self.lock.unlock()
        // Intentionally never answer. Cancelling ChatViewModel's in-flight
        // task must cancel URLSession and route through the real catch path.
    }

    override func stopLoading() {
        Self.lock.lock()
        Self.stopped = true
        Self.lock.unlock()
    }
}

private final class ImageFailureQuarantineProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var bodies: [Data] = []

    static func reset() {
        lock.lock()
        bodies = []
        lock.unlock()
    }

    static func capturedBodies() -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        return bodies
    }

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ImageFailureQuarantineProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body = Self.requestBody(from: request)
        Self.lock.lock()
        Self.bodies.append(body)
        let requestNumber = Self.bodies.count
        Self.lock.unlock()

        let statusCode = requestNumber <= 2 ? 500 : 200
        let contentType = requestNumber <= 2 ? "application/json" : "text/event-stream"
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": contentType]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let responseBody = requestNumber <= 2
            ? Data(#"{"error":{"message":"temporary runtime failure","type":"server_error"}}"#.utf8)
            : Data("""
                data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}

                data: [DONE]

                """.utf8)
        client?.urlProtocol(self, didLoad: responseBody)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func requestBody(from request: URLRequest) -> Data {
        guard let stream = request.httpBodyStream else { return request.httpBody ?? Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4_096)
        while true {
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return Data() }
        }
    }
}
