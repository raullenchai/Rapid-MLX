import Foundation
import Testing

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
        for identifier in ["Banner", "Share", "Decline", "Close"] {
            #expect(banner.contains("TelemetryConsent.PostValue\(identifier == "Banner" ? "" : ".")\(identifier)"))
        }
        #expect(!banner.contains(".isModal"))
        #expect(banner.contains(".keyboardShortcut(.cancelAction)"))
        #expect(!banner.contains("@FocusState"))
    }
}
