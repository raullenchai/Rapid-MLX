import AppKit
import CryptoKit
import XCTest

@MainActor
final class ChatAttachmentJourneyTests: XCTestCase {
    func testPickerAttachmentsStayWithTheirConversationAndWirePayload() throws {
        continueAfterFailure = false
        let harness = try RapidUITestHarness(
            testName: name,
            fakeSettings: ["FAKE_VISION_CHAT": "1"]
        )
        defer { harness.shutDown() }
        harness.launch()
        harness.startModel()

        let first = harness.rapidMacRoot
            .appendingPathComponent("Tests/RapidTests/__Snapshots__/cheetah-logo-28.png")
        let second = harness.rapidMacRoot
            .appendingPathComponent("Tests/RapidTests/__Snapshots__/cheetah-logo-96.png")
        let document = harness.rapidMacRoot
            .appendingPathComponent("Tests/GUIGoldenFlows/Fixtures/chat-document.txt")

        harness.chooseFile(first, actionIdentifier: "ChatView.Attachments.UploadPhoto")
        let firstChip = harness.element("ChatView.Attachment.Remove.\(first.lastPathComponent)")
        XCTAssertTrue(firstChip.waitForExistence(timeout: 10))
        harness.send("Create conversation A", expectedRequestCount: 1)
        let conversationA = harness.conversationRows().firstMatch
        XCTAssertTrue(conversationA.waitForExistence(timeout: 10))
        let conversationAIdentifier = conversationA.identifier

        // Leave an unsent attachment in A, then prove B does not inherit it.
        harness.chooseFile(first, actionIdentifier: "ChatView.Attachments.UploadPhoto")
        XCTAssertTrue(firstChip.waitForExistence(timeout: 10))
        harness.element("Sidebar.NewChat").click()
        XCTAssertTrue(harness.waitUntil(timeout: 10) { !firstChip.exists })

        harness.chooseFile(second, actionIdentifier: "ChatView.Attachments.UploadPhoto")
        let secondChip = harness.element("ChatView.Attachment.Remove.\(second.lastPathComponent)")
        XCTAssertTrue(secondChip.waitForExistence(timeout: 10))
        harness.send("Create conversation B", expectedRequestCount: 2)

        harness.element(conversationAIdentifier).click()
        XCTAssertTrue(firstChip.waitForExistence(timeout: 10))
        XCTAssertFalse(secondChip.exists)
        harness.send("Send A's pending image", expectedRequestCount: 3)

        harness.chooseFile(document, actionIdentifier: "ChatView.Attachments.UploadFile")
        let documentChip = harness.element("ChatView.Attachment.Remove.\(document.lastPathComponent)")
        XCTAssertTrue(documentChip.waitForExistence(timeout: 10))
        harness.send("Send A's document", expectedRequestCount: 4)

        let requests = harness.chatRequests()
        XCTAssertEqual(imageHashes(in: requests[0]), [try dataURLHash(first)])
        XCTAssertEqual(imageHashes(in: requests[1]), [try dataURLHash(second)])
        XCTAssertEqual(imageHashes(in: requests[2]), [try dataURLHash(first)])
        XCTAssertTrue(text(in: requests[3]).contains("Revenue: 42"))
        XCTAssertTrue(text(in: requests[3]).contains("Region: APAC"))
        XCTAssertTrue(imageHashes(in: requests[3]).isEmpty)
    }

    func testDragPasteAndRemovalPreserveWireIdentity() throws {
        continueAfterFailure = false
        let harness = try RapidUITestHarness(
            testName: name,
            fakeSettings: ["FAKE_VISION_CHAT": "1"]
        )
        defer { harness.shutDown() }
        harness.launch()
        harness.startModel()

        let image = harness.rapidMacRoot
            .appendingPathComponent("Tests/RapidTests/__Snapshots__/cheetah-logo-96.png")
        let document = harness.rapidMacRoot
            .appendingPathComponent("Tests/GUIGoldenFlows/Fixtures/chat-document.txt")

        harness.dragFile(image)
        let imageChip = harness.element("ChatView.Attachment.Remove.\(image.lastPathComponent)")
        XCTAssertTrue(imageChip.waitForExistence(timeout: 15))
        harness.send("Dragged photo", expectedRequestCount: 1)

        harness.dragFile(document)
        let documentChip = harness.element("ChatView.Attachment.Remove.\(document.lastPathComponent)")
        XCTAssertTrue(documentChip.waitForExistence(timeout: 15))
        harness.send("Dragged document", expectedRequestCount: 2)

        try harness.pasteImage(image)
        let pastedChip = harness.element("ChatView.Attachment.Remove.Pasted image.png")
        XCTAssertTrue(pastedChip.waitForExistence(timeout: 15))
        pastedChip.click()
        XCTAssertTrue(harness.waitUntil(timeout: 10) { !pastedChip.exists })
        harness.send("Removed before send", expectedRequestCount: 3)

        try harness.pasteImage(image)
        XCTAssertTrue(pastedChip.waitForExistence(timeout: 15))
        harness.send("Pasted photo", expectedRequestCount: 4)

        let requests = harness.chatRequests()
        XCTAssertEqual(imageHashes(in: requests[0]), [try dataURLHash(image)])
        XCTAssertTrue(text(in: requests[1]).contains("Revenue: 42"))
        XCTAssertTrue(text(in: requests[1]).contains("Region: APAC"))
        XCTAssertTrue(imageHashes(in: requests[1]).isEmpty)
        XCTAssertTrue(imageHashes(in: requests[2]).isEmpty)
        XCTAssertEqual(imageHashes(in: requests[3]), [try pastedImageHash(image)])
    }

    private func imageHashes(in request: [String: Any]) -> [String] {
        guard let payloads = request["user_payloads"] as? [[String: Any]],
              let latest = payloads.last else { return [] }
        return latest["image_url_sha256"] as? [String] ?? []
    }

    private func text(in request: [String: Any]) -> String {
        guard let payloads = request["user_payloads"] as? [[String: Any]],
              let latest = payloads.last else { return "" }
        return latest["text"] as? String ?? ""
    }

    private func dataURLHash(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        let encoded = "data:image/png;base64," + data.base64EncodedString()
        return SHA256.hash(data: Data(encoded.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private func pastedImageHash(_ url: URL) throws -> String {
        let source = try Data(contentsOf: url)
        guard let image = NSImage(data: source),
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            XCTFail("Could not reproduce the app's pasted-image encoding")
            return ""
        }
        let encoded = "data:image/png;base64," + png.base64EncodedString()
        return SHA256.hash(data: Data(encoded.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
