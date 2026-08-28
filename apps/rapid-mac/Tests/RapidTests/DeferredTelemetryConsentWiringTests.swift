import Foundation
import Testing
@testable import Rapid

@Suite("Deferred telemetry consent delivery wiring")
struct DeferredTelemetryConsentWiringTests {
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func source(_ path: String) throws -> String {
        try String(contentsOf: packageRoot.appendingPathComponent(path), encoding: .utf8)
    }

    @Test("The app routes all three completed-value signals to one coordinator")
    func appOwnsTheSignalFanIn() throws {
        let app = try Self.source("Sources/Rapid/RapidApp.swift")

        #expect(app.contains("let consentCoordinator = DeferredTelemetryConsentCoordinator()"))
        #expect(app.components(separatedBy: "consentCoordinator?.productValueDelivered(kind)").count - 1 == 3)
    }

    @Test("Chat signals only a nonempty completed final turn")
    func chatSignalsFinalDelivery() throws {
        let chat = try Self.source("Sources/Rapid/Chat/ChatViewModel.swift")
        let loop = try #require(chat.range(of: "private func runToolLoop("))
        let body = String(chat[loop.lowerBound...])

        #expect(body.contains("if !Task.isCancelled,"))
        #expect(body.contains("delivered.status == .complete"))
        #expect(body.contains("!delivered.content.trimmingCharacters"))
        #expect(body.contains("onProductValueDelivered(.chatReply)"))
    }

    @Test("A successful vision turn does not claim the text-only chat milestone")
    @MainActor
    func visionReplyDoesNotSignalChatReply() async throws {
        let image = try ChatImageAttachment(
            filename: "photo.png",
            mimeType: "image/png",
            data: Data("image".utf8)
        )
        var deliveredKinds: [ProductValueKind] = []
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: ActivationVisionReplyProtocol.session()
        )
        let model = ChatViewModel(
            client: client,
            persistsConversations: false,
            onProductValueDelivered: { deliveredKinds.append($0) }
        )

        model.send(
            "What is in this photo?",
            alias: "vision-model",
            supportsImageInput: true,
            imageAttachments: [image]
        )
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while model.isStreaming, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming, "the canned successful stream must finish")
        #expect(deliveredKinds.isEmpty)
        #expect(model.messages.last?.content == "ok")
    }

    @Test("Dictation signals after transcript delivery and history persistence")
    func dictationSignalsDeliveredTranscript() throws {
        let dictation = try Self.source("Sources/Rapid/Dictation/DictationController.swift")
        let signal = try #require(dictation.range(of: "onProductValueDelivered(.dictationTranscript)"))
        let history = try #require(dictation.range(of: "history.record(", options: .backwards,
                                                   range: dictation.startIndex..<signal.lowerBound))
        let delivery = try #require(dictation.range(of: "DictationInjector.deliver(", options: .backwards,
                                                    range: dictation.startIndex..<signal.lowerBound))

        #expect(delivery.lowerBound < history.lowerBound)
        #expect(history.lowerBound < signal.lowerBound)
        #expect(dictation.components(separatedBy: "onProductValueDelivered(.dictationTranscript)").count - 1 == 1)
    }

    @Test("Only a newly generated image signals product value")
    func imageSignalsGenerationNotEdit() throws {
        let image = try Self.source("Sources/Rapid/Images/ImageGenViewModel.swift")
        let generateStart = try #require(image.range(of: "private func runGenerate("))
        let editStart = try #require(image.range(of: "private func runEdit("))
        let generation = String(image[generateStart.lowerBound..<editStart.lowerBound])
        let editing = String(image[editStart.lowerBound...])

        #expect(generation.contains("if let first = images.first"))
        #expect(generation.contains("onProductValueDelivered(.generatedImage)"))
        #expect(!editing.contains("onProductValueDelivered(.generatedImage)"))
    }

    @Test("The invitation is non-modal, focus-neutral, and fully addressable")
    func bannerInteractionContract() throws {
        let banner = try Self.source("Sources/Rapid/UI/TelemetryConsentView.swift")

        #expect(banner.contains("Help improve Rapid by sharing anonymous usage data?"))
        #expect(banner.contains("Change this anytime in Settings → Privacy."))
        for identifier in ["Banner", "Share", "Decline", "Close"] {
            #expect(banner.contains("TelemetryConsent.PostValue\(identifier == "Banner" ? "" : ".")\(identifier)"))
        }
        #expect(!banner.contains(".isModal"))
        #expect(banner.contains("Button(\"No thanks\", role: .cancel) { consent.decline() }"))
        #expect(!banner.contains(".keyboardShortcut(.cancelAction)"),
                "Escape belongs to the active app interaction; only an explicit click may decline")
        #expect(!banner.contains("@FocusState"))
    }

    @Test("Every consent surface discloses the Desktop activation shape and country derivation")
    func activationDisclosureContract() throws {
        let banner = try Self.source("Sources/Rapid/UI/TelemetryConsentView.swift")
        let settings = try Self.source("Sources/Rapid/UI/SettingsView.swift")
        let privacy = try Self.source("PRIVACY.md")
        let normalizedPrivacy = privacy.split(whereSeparator: \Character.isWhitespace).joined(separator: " ")

        for surface in [banner, settings] {
            #expect(surface.contains("first successful text chat reply, dictation, or generated image"))
            #expect(surface.contains("does not send a vision-reply milestone"))
            #expect(surface.contains("only the milestone name and “Desktop”"))
            #expect(surface.contains("derives a country code but never stores your IP"))
        }
        #expect(privacy.contains("`activation` — once per install"))
        #expect(privacy.contains("vision-reply milestone"))
        #expect(privacy.contains("is reserved in the event schema but is not sent by this version"))
        #expect(privacy.contains("`surface: desktop`"))
        #expect(privacy.contains("two-letter country code"))
        #expect(normalizedPrivacy.contains("the IP address is never persisted"))
    }
}

private final class ActivationVisionReplyProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ActivationVisionReplyProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
