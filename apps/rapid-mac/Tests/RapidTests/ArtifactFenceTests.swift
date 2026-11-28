import AppKit
import Foundation
import Testing
@testable import Rapid

@Suite("Unclosed code fences")
struct ArtifactFenceTests {

    private func closedFlags(_ source: String) -> [Bool] {
        MarkdownCompiler().compile(source).items.compactMap { item in
            if case .code(let block) = item { return block.isClosed }
            return nil
        }
    }

    @Test("Only a fence that never closes is reported as unclosed", arguments: [
        ("```html\n<p>x</p>", [false]),
        ("```html\n<p>x</p>\n```", [true]),
        ("```html\n<p>x</p>\n```\n\ntail", [true]),
        ("~~~html\n<p>x</p>", [false]),
        ("~~~html\n<p>x</p>\n~~~", [true]),
        ("````html\n<p>x</p>\n```\n", [false]),
        ("````html\n<p>x</p>\n`````", [true]),
        ("```html\n<p>x</p>\n``` trailing\n", [false]),
        ("   ```html\n<p>x</p>\n   ```", [true]),
        ("```html\n<p>x</p>\n```\n\n```python\nx = 1", [true, false]),
        ("```html\n<p>x</p>\n```\n\n```python\nx = 1\n```", [true, true]),
        ("`code` and ``` not a fence ``` here", []),
        ("just text", []),
    ] as [(String, [Bool])])
    func flags(_ source: String, _ expected: [Bool]) {
        #expect(closedFlags(source) == expected, "\(source.debugDescription)")
    }

    @Test("A cut-off page is not offered, a cut-off diagram still is")
    func extraction() {
        let page = "<!DOCTYPE html><html><body>x</body></html>"
        #expect(ArtifactPanelModel.blocks(in: "```html\n\(page)").isEmpty)
        #expect(ArtifactPanelModel.blocks(in: "```html\n\(page)\n```").map(\.kind) == [.html])
        #expect(ArtifactPanelModel.blocks(in: "```mermaid\ngraph TD\n  A --> B").map(\.kind) == [.mermaid])
    }
}

@MainActor
@Suite("Unclosed fences in the transcript and library")
struct ArtifactFenceViewTests {

    private let page = "<!DOCTYPE html><html><head><title>Counter</title></head><body></body></html>"

    private func block(_ code: String, isFinal: Bool, isClosed: Bool) -> MarkdownCodeBlockView {
        let view = MarkdownCodeBlockView(options: MarkdownOptions())
        view.frame = NSRect(x: 0, y: 0, width: 400, height: 200)
        view.configure(
            code: code, language: "html", options: MarkdownOptions(),
            isFinal: isFinal, isClosed: isClosed
        )
        view.layoutSubtreeIfNeeded()
        return view
    }

    private func previewButton(in view: NSView) -> NSButton? {
        view.subviews.compactMap { $0 as? NSButton }
            .first { $0.accessibilityIdentifier() == "CodeBlock.Preview" }
    }

    private func card(in view: NSView) -> ArtifactCardView? {
        view.subviews.compactMap { $0 as? ArtifactCardView }.first
    }

    @Test("A settled block whose fence never closed shows source, not a card")
    func stoppedStreamIsNotACard() {
        let cut = block(page, isFinal: true, isClosed: false)
        #expect(previewButton(in: cut)?.isHidden == true)
        #expect(card(in: cut)?.isHidden == true)

        let whole = block(page, isFinal: true, isClosed: true)
        #expect(previewButton(in: whole)?.isHidden == false)
        #expect(card(in: whole)?.isHidden == false)
    }

    @Test("The library skips a cancelled reply's unclosed page")
    func libraryExcludesCutOffPage() async {
        let images = FileManager.default.temporaryDirectory.appendingPathComponent("fence-img-\(UUID())")
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent("fence-aud-\(UUID())")
        defer {
            try? FileManager.default.removeItem(at: images)
            try? FileManager.default.removeItem(at: audio)
        }
        let library = ArtifactLibrary(
            imageStore: ImageArtifactStore(directory: images), audioStore: AudioArtifactStore(directory: audio)
        )
        let q = ChatMessage(role: .user, content: "q", createdAt: Date(timeIntervalSince1970: 1))
        let cut = ChatMessage(
            role: .assistant, content: "```html\n\(page)", status: .complete, parentID: q.id,
            createdAt: Date(timeIntervalSince1970: 2)
        )
        let conversation = ChatConversation(
            id: UUID(), title: "Cancelled", messages: [q, cut],
            createdAt: Date(timeIntervalSince1970: 1), updatedAt: Date(timeIntervalSince1970: 2)
        )
        await library.refresh(conversations: [conversation])
        #expect(library.items.isEmpty)
    }
}
