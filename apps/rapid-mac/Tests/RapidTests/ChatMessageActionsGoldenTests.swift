import AppKit
import Foundation
import SwiftUI
import Testing

@testable import Rapid

/// The `message-actions` golden journey, sunk from the AX bash harness
/// (`gui-golden-flows.sh`) into `swift test`: every inline message action
/// must produce its advertised result. The real ``ChatView`` is mounted on
/// a ``GoldenStage`` and driven through the same accessibility identifiers
/// the out-of-process flow used; the streaming backend is the shared
/// ``GoldenChatFake``, so the "a request actually left the process"
/// assertions keep an independent witness.
///
/// Journey inventory: `Tests/GUIGoldenFlows/journeys.yaml` lists this as
/// `driver: swift`; the GUI coverage contract maps that driver to this
/// suite's presence instead of a bash dispatcher case.
@MainActor
@Suite("Golden journey: message-actions", .serialized)
struct ChatMessageActionsGoldenTests {

    static func waitForSettledReply(on surface: GoldenChatSurface) async throws {
        try await surface.stage.waitForText("deterministic content")
        // The reply streams in many chunks; settle on the final words so
        // later assertions see the finished turn, not a partial one.
        try await surface.stage.waitForText("to assert on.")
        try await surface.waitForSendIdle()
    }

    // MARK: - The journey

    @Test("Every inline message action produces its advertised result")
    func messageActionsJourney() async throws {
        let surface = GoldenChatSurface.mount()
        let stage = surface.stage

        try await surface.sendPrompt("original message action prompt")
        try await Self.waitForSettledReply(on: surface)

        // Discover the per-message action identifiers from the live tree,
        // exactly as the bash flow did — no test-only entry points.
        guard let copyID = stage.identifier(withPrefix: "ChatView.Message.Copy.", last: true) else {
            Issue.record("completed turn exposes no assistant Copy action: \(stage.identifiers())")
            return
        }
        let selectID = copyID.replacingOccurrences(of: "Message.Copy.", with: "Message.SelectText.")
        let retryID = copyID.replacingOccurrences(of: "Message.Copy.", with: "Message.Retry.")
        guard let editID = stage.identifier(withPrefix: "ChatView.Message.Edit.") else {
            Issue.record("completed turn exposes no user Edit action: \(stage.identifiers())")
            return
        }

        // Copy response fills the pasteboard. The general pasteboard belongs
        // to whoever runs this suite, so snapshot it and put their contents
        // back afterwards.
        let savedPasteboardItems = (NSPasteboard.general.pasteboardItems ?? []).map {
            item -> NSPasteboardItem in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) {
                    copy.setData(data, forType: type)
                }
            }
            return copy
        }
        defer {
            NSPasteboard.general.clearContents()
            if !savedPasteboardItems.isEmpty {
                NSPasteboard.general.writeObjects(savedPasteboardItems)
            }
        }
        NSPasteboard.general.clearContents()
        try stage.press(copyID)
        try await stage.wait(for: "pasteboard content after Copy") {
            (NSPasteboard.general.string(forType: .string) ?? "").isEmpty == false
        }
        #expect(
            NSPasteboard.general.string(forType: .string)?.contains("deterministic content") == true,
            "Copy response did not place the assistant reply on the pasteboard"
        )

        // Select text opens its sheet; Done dismisses it.
        try stage.press(selectID)
        try await stage.waitForIdentifier("SelectText.Done")
        #expect(stage.treeText().contains("Selection here crosses paragraphs"))
        try stage.press("SelectText.Done")
        try await stage.waitForIdentifierGone("SelectText.Done")

        // Edit, then cancel: the draft must never be sent.
        let editSuffix = String(editID.dropFirst("ChatView.Message.Edit.".count))
        try stage.press(editID)
        try await stage.waitForIdentifier("ChatView.Message.EditField.\(editSuffix)")
        try stage.setValue(
            "cancelled edit must not send",
            for: "ChatView.Message.EditField.\(editSuffix)"
        )
        try stage.press("ChatView.Message.CancelEdit.\(editSuffix)")
        try await stage.waitForIdentifier(editID)
        // A wrongly-scheduled asynchronous send would leave the process a
        // beat after Cancel; give it that beat so the assertion can catch it.
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(
            !surface.fake.recordedPrompts().contains { $0.contains("cancelled edit must not send") },
            "cancelling a message edit sent the draft"
        )

        // Retry sends a replacement request.
        let requestsBeforeRetry = surface.fake.recordedBodies().count
        try stage.press(retryID)
        try await stage.wait(for: "replacement request after Retry") {
            surface.fake.recordedBodies().count > requestsBeforeRetry
        }
        try await Self.waitForSettledReply(on: surface)

        // Edit, then save: the edited prompt replaces the turn and sends.
        guard let editAfterRetry = stage.identifier(withPrefix: "ChatView.Message.Edit.") else {
            Issue.record("retried turn exposes no Edit action: \(stage.identifiers())")
            return
        }
        let saveSuffix = String(editAfterRetry.dropFirst("ChatView.Message.Edit.".count))
        try stage.press(editAfterRetry)
        try await stage.waitForIdentifier("ChatView.Message.EditField.\(saveSuffix)")
        try stage.setValue(
            "saved edited message prompt",
            for: "ChatView.Message.EditField.\(saveSuffix)"
        )
        try stage.press("ChatView.Message.SaveEdit.\(saveSuffix)")
        try await stage.wait(for: "the edited prompt to be sent") {
            surface.fake.recordedPrompts().contains("saved edited message prompt")
        }
        try await Self.waitForSettledReply(on: surface)
        #expect(stage.treeText().contains("saved edited message prompt"))
    }

    @Test("Jump to latest returns a settled transcript to its tail")
    func jumpToLatestOnSettledTranscript() async throws {
        let surface = GoldenChatSurface.mount()
        let stage = surface.stage

        // #1904 regression shape: a long, fully settled answer, moved away
        // from its tail. Jump to latest used to re-pin the follow flag but
        // leave the transcript physically scrolled up, because nothing was
        // streaming and no document-frame change would ever fire again.
        try await surface.sendPrompt("shape:long finished answer for jump-to-bottom")
        try await stage.waitForText("END-OF-LONG-ANSWER")
        try await surface.waitForSendIdle()
        try await stage.wait(for: "the settled transcript to rest at its tail") {
            (stage.scrollFraction() ?? 0) > 0.97
        }
        let bottom = try #require(stage.scrollFraction())

        try stage.setScrollFraction(0)
        try await stage.waitForIdentifier("Transcript.JumpToBottom")
        let scrolled = try #require(stage.scrollFraction())
        #expect(
            scrolled < bottom - 0.02,
            "scroll fixture did not move away from the settled transcript tail"
        )

        try stage.press("Transcript.JumpToBottom")
        try await stage.wait(for: "the transcript to return to its tail") {
            (stage.scrollFraction() ?? 0) > scrolled + 0.02
        }
        try await stage.waitForIdentifierGone("Transcript.JumpToBottom")
    }

    @Test("Model info opens from its anchor, dismisses, and can reopen")
    func modelInfoPopoverRoundTrip() async throws {
        let surface = GoldenChatSurface.mount()
        let stage = surface.stage

        try stage.press("ModelPickerBar.ModelInfo")
        try await stage.waitForText("Parameters")
        #expect(stage.treeText().contains(GoldenChatSurface.alias))
        // Esc is the user's dismissal gesture for a transient popover.
        // Dismiss-then-reopen proves it is not a one-way overlay that traps
        // the rest of the composer, and that the anchor stays live — the
        // bash journey only ever asserted the anchor's second press
        // succeeded, so this is a strictly stronger contract.
        stage.pressEscape()
        try await stage.wait(for: "model info popover to close") {
            !stage.treeText().contains("Parameters")
        }
        try stage.press("ModelPickerBar.ModelInfo")
        try await stage.waitForText("Parameters")
        stage.pressEscape()
        try await stage.wait(for: "model info popover to close again") {
            !stage.treeText().contains("Parameters")
        }
    }
}
