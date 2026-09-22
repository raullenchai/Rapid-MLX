import Foundation
import Testing
@testable import Rapid

@Suite("Telemetry launch notice wiring")
struct TelemetryNoticeWiringTests {
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func source(_ path: String) throws -> String {
        try String(contentsOf: packageRoot.appendingPathComponent(path), encoding: .utf8)
    }

    @Test("The app owns one launch notice and preserves all activation signals")
    func appOwnsTheSignalFanIn() throws {
        let app = try Self.source("Sources/Rapid/RapidApp.swift")

        #expect(app.contains("let noticeCoordinator = TelemetryNoticeCoordinator()"))
        #expect(app.contains("let starPromptCoordinator = GitHubStarPromptCoordinator()"))
        #expect(app.components(separatedBy: "noticeCoordinator?.productValueDelivered(kind)").count - 1 == 3)
        #expect(app.components(separatedBy: "starPromptCoordinator?.productValueDelivered(kind)").count - 1 == 3)
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
        await model._testingWaitForCurrentTurn()

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

    @Test("The notice is non-modal, single-action, and fully addressable")
    func bannerInteractionContract() throws {
        let banner = try Self.source("Sources/Rapid/UI/TelemetryNoticeView.swift")

        #expect(banner.contains("Anonymous telemetry is now on by default"))
        #expect(banner.contains("Button(\"Got it\")"))
        #expect(banner.contains("TelemetryNotice.Banner"))
        #expect(banner.contains("TelemetryNotice.Acknowledge"))
        #expect(!banner.contains("No thanks"))
        #expect(!banner.contains("Share"))
        #expect(!banner.contains(".isModal"))
        #expect(!banner.contains("@FocusState"))
        #expect(banner.contains(".onAppear { notice.noticeDidAppear() }"))
    }

    @Test("The launch notice contains the complete default-on disclosure")
    func disclosureContract() throws {
        let banner = try Self.source("Sources/Rapid/UI/TelemetryNoticeView.swift")
        let settings = try Self.source("Sources/Rapid/UI/SettingsView.swift")
        for phrase in [
            "on by default", "previously turned telemetry off", "metadata-only",
            "the app's to rapidmlx.com's telemetry service", "the bundled engine's to PostHog Cloud (US)",
            "never your IP or a per-person profile; the app's collector keeps only a coarse country code",
            "never sends prompts, responses, file paths, or API key values",
            "Nothing is sent before this notice appears",
            "rapid-mlx telemetry off", "RAPID_MLX_TELEMETRY=0", "DO_NOT_TRACK=1",
            "https://rapidmlx.com/docs/telemetry",
        ] {
            #expect(banner.contains(phrase), "missing disclosure phrase: \(phrase)")
        }
        for phrase in [
            "the app's to rapidmlx.com's telemetry service",
            "the bundled engine's to PostHog Cloud (US)",
            "never your IP or a per-person profile; the app's collector keeps only a coarse country code",
        ] {
            #expect(settings.contains(phrase), "Settings missing disclosure phrase: \(phrase)")
        }
        #expect(settings.contains("telemetry is on by default"))
        #expect(settings.contains("https://rapidmlx.com/docs/telemetry"))
    }

    @Test("The shipped privacy policy states the default-on notice contract")
    func privacyDisclosureContract() throws {
        let banner = try Self.source("Sources/Rapid/UI/TelemetryNoticeView.swift")
        let privacy = try Self.source("PRIVACY.md")
        let normalized = privacy.replacingOccurrences(of: "**", with: "")
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
        let command = "rapid-mlx telemetry off"
        let processors = ["rapidmlx.com's telemetry service", "PostHog Cloud (US)"]
        for phrase in [
            "Starting in 0.15.0, it is on by default after a one-time in-app acknowledgement notice.",
            "This includes installs that turned telemetry off before 0.15.0, which are told about the change in that notice.",
            "A refusal recorded in 0.15.0 or later is never reversed.",
            "Settings → Privacy", command,
            "RAPID_MLX_TELEMETRY=0", "DO_NOT_TRACK=1",
        ] {
            #expect(normalized.contains(phrase), "missing privacy phrase: \(phrase)")
        }
        #expect(banner.contains(command), "banner and privacy policy must use the same CLI command")
        for processor in processors {
            #expect(banner.contains(processor), "banner missing telemetry processor: \(processor)")
            #expect(normalized.contains(processor), "privacy policy missing telemetry processor: \(processor)")
        }
        #expect(!privacy.contains("Default: **off until you make an"))
        #expect(!privacy.contains("only after the same opt-in"))
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
