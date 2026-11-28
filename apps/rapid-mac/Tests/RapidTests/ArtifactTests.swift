import AppKit
import Foundation
import Testing
@testable import Rapid

@Suite("Artifact source detection")
struct ArtifactSourceTests {

    private let page = "<!DOCTYPE html><html><head><title>Counter</title></head><body></body></html>"

    @Test("A fence tagged html is a page, whatever its case")
    func taggedHTML() {
        #expect(ArtifactHTMLSource.looksLikeHTML(code: "<p>hi</p>", language: "html"))
        #expect(ArtifactHTMLSource.looksLikeHTML(code: "<p>hi</p>", language: "HTML"))
    }

    @Test("An untagged fence is a page only when it opens like one", arguments: [
        ("<!DOCTYPE html>\n<html></html>", true),
        ("  \n<!doctype HTML>", true),
        ("<html lang=\"en\">", true),
        ("<HTML>", true),
        ("<html>", true),
        ("<htmlish>", false),
        ("<div>hi</div>", false),
        ("<!DOCTYPE htmlx>", false),
        ("let x = \"<html>\"", false),
        ("", false),
        ("   \n  ", false),
    ])
    func untagged(_ code: String, _ expected: Bool) {
        #expect(ArtifactHTMLSource.looksLikeHTML(code: code, language: nil) == expected)
    }

    @Test("Another language tag is never a page")
    func otherTags() {
        #expect(!ArtifactHTMLSource.looksLikeHTML(code: page, language: "swift"))
        #expect(!ArtifactHTMLSource.looksLikeHTML(code: page, language: "xml"))
    }

    @Test("Empty tagged content is not a page")
    func emptyTagged() {
        #expect(!ArtifactHTMLSource.looksLikeHTML(code: " \n", language: "html"))
    }

    @Test("Source over the cap stays code")
    func overCap() {
        let filler = String(repeating: "a", count: ArtifactHTMLSource.maximumSourceBytes)
        #expect(!ArtifactHTMLSource.looksLikeHTML(code: "<html>" + filler, language: nil))
        let fits = String(repeating: "a", count: ArtifactHTMLSource.maximumSourceBytes - 6)
        #expect(ArtifactHTMLSource.looksLikeHTML(code: "<html>" + fits, language: nil))
        #expect(ArtifactHTMLSource.maximumSourceBytes == 256 * 1024)
    }

    @Test("SVG and Mermaid keep priority over html")
    func kindPriority() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\"><rect width=\"10\" height=\"10\"/></svg>"
        #expect(ArtifactItem.kind(code: svg, language: "html") == .svg)
        #expect(ArtifactItem.kind(code: "graph TD\n  A --> B", language: nil) == .mermaid)
        #expect(ArtifactItem.kind(code: "graph TD\n  A --> B", language: "mermaid") == .mermaid)
        #expect(ArtifactItem.kind(code: page, language: nil) == .html)
        #expect(ArtifactItem.kind(code: "print(1)", language: "python") == nil)
    }

    @Test("The title comes from <title>, collapsed and bounded")
    func titles() {
        #expect(ArtifactHTMLSource.title(of: page) == "Counter")
        #expect(ArtifactHTMLSource.title(of: "<TITLE>\n  A   b \n</TITLE>") == "A b")
        #expect(ArtifactHTMLSource.title(of: "<html></html>") == nil)
        #expect(ArtifactHTMLSource.title(of: "<title>  </title>") == nil)
        let long = "<title>\(String(repeating: "x", count: 200))</title>"
        #expect(ArtifactHTMLSource.title(of: long)?.count == 81)
        let item = ArtifactItem(id: "a", kind: .html, source: "<html></html>")
        #expect(item.title == "HTML artifact")
    }

    @Test("Remote scripts and styles are noticed", arguments: [
        (#"<script src="https://cdn.example/x.js"></script>"#, true),
        (#"<SCRIPT type="module" SRC='http://cdn.example/x.js'>"#, true),
        (#"<script src="//cdn.example/x.js">"#, true),
        (#"<link rel="stylesheet" href="https://cdn.example/x.css">"#, true),
        (#"<style>@import url("https://fonts.example/x.css");</style>"#, true),
        (#"<style>@import "//fonts.example/x.css";</style>"#, true),
        ("<script>let a = 1</script>", false),
        (#"<script src="app.js"></script>"#, false),
        (#"<img src="https://example.com/a.png">"#, false),
        (#"<a href="https://example.com">x</a>"#, false),
    ])
    func remote(_ source: String, _ expected: Bool) {
        #expect(ArtifactHTMLSource.referencesRemoteResources(source) == expected)
        #expect(
            ArtifactItem(id: "a", kind: .html, source: source).referencesRemoteResources == expected
        )
    }
}

@MainActor
@Suite("HTML artifact in a code block")
struct ArtifactCodeBlockTests {

    private let page = """
        <!DOCTYPE html><html><head><title>Counter</title></head>
        <body><button id="b">+</button></body></html>
        """

    private func block(
        _ code: String, _ language: String?, isFinal: Bool = true,
        opener: ArtifactOpener? = nil
    ) -> MarkdownCodeBlockView {
        let view = MarkdownCodeBlockView(options: MarkdownOptions())
        view.frame = NSRect(x: 0, y: 0, width: 400, height: 200)
        view.artifactOpener = opener
        view.configure(code: code, language: language, options: MarkdownOptions(), isFinal: isFinal)
        view.layoutSubtreeIfNeeded()
        return view
    }

    private func button(_ identifier: String, in view: NSView) -> NSButton? {
        for sub in view.subviews {
            if let found = sub as? NSButton, found.accessibilityIdentifier() == identifier {
                return found
            }
            if let found = button(identifier, in: sub) { return found }
        }
        return nil
    }

    private func card(in view: NSView) -> ArtifactCardView? {
        view.subviews.compactMap { $0 as? ArtifactCardView }.first
    }

    private func press(_ button: NSButton) {
        _ = button.target?.perform(button.action, with: button)
    }

    @Test("A half-streamed page is source only; the card appears when the block closes")
    func finalityGate() {
        let partial = block(page, "html", isFinal: false)
        #expect(button("CodeBlock.Preview", in: partial)?.isHidden == true)
        #expect(card(in: partial)?.isHidden == true)

        // The same view, settled: no text delta, only finality changes.
        partial.configure(code: page, language: "html", options: MarkdownOptions(), isFinal: true)
        #expect(button("CodeBlock.Preview", in: partial)?.isHidden == false)
        #expect(card(in: partial)?.isHidden == false)
    }

    @Test("An untagged doctype block gets the card too, with header room")
    func untagged() {
        let tagged = block(page, "html")
        let untagged = block(page, nil)
        #expect(card(in: untagged)?.isHidden == false)
        #expect(untagged.height(forWidth: 400) == tagged.height(forWidth: 400))
    }

    @Test("The card is a fixed height however long the source is")
    func cardHeight() {
        let long = page.replacingOccurrences(
            of: "<body>", with: "<body>" + String(repeating: "<p>x</p>\n", count: 80)
        )
        #expect(block(page, "html").height(forWidth: 400) == block(long, "html").height(forWidth: 400))
    }

    @Test("The Preview/Code toggle swaps the card for the source and back")
    func toggle() throws {
        let view = block(page, "html")
        let toggle = try #require(button("CodeBlock.Preview", in: view))
        #expect(toggle.title == "Code")
        let cardHeight = view.height(forWidth: 400)

        press(toggle)
        #expect(toggle.title == "Preview")
        #expect(card(in: view)?.isHidden == true)
        #expect(view.height(forWidth: 400) != cardHeight)

        // A reconfigure does not reopen what the reader closed.
        view.configure(code: page, language: "html", options: MarkdownOptions())
        #expect(card(in: view)?.isHidden == true)

        press(toggle)
        #expect(card(in: view)?.isHidden == false)
        #expect(view.height(forWidth: 400) == cardHeight)
    }

    @Test("Over the cap, a page stays code only")
    func overCap() {
        let big = "<!DOCTYPE html>" + String(repeating: "a", count: ArtifactHTMLSource.maximumSourceBytes)
        let view = block(big, "html")
        #expect(button("CodeBlock.Preview", in: view)?.isHidden == true)
        #expect(card(in: view)?.isHidden == true)
    }

    @Test("Other languages are untouched")
    func otherLanguages() {
        let view = block("print(1)", "python")
        #expect(button("CodeBlock.Preview", in: view)?.isHidden == true)
        #expect(card(in: view)?.isHidden == true)
    }

    @Test("Without an opener the card has no Open button")
    func noOpener() throws {
        let view = block(page, "html")
        let cardView = try #require(card(in: view))
        let open = try #require(button("Artifact.Card.Open", in: cardView))
        #expect(open.isHidden)
    }

    @Test("The card's Open button opens the panel on this page")
    func cardOpens() throws {
        let model = ArtifactPanelModel()
        let view = block(page, "html", opener: model.opener)
        let cardView = try #require(card(in: view))
        let open = try #require(button("Artifact.Card.Open", in: cardView))
        #expect(!open.isHidden)
        press(open)
        #expect(model.isOpen)
        #expect(model.selected?.kind == .html)
        #expect(model.selected?.source == page)
        #expect(model.selected?.title == "Counter")
    }

    @Test("An SVG block gets a header Open button when there is an opener")
    func svgOpens() throws {
        let svg = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" width="100" height="50">
              <rect width="100" height="50" fill="teal"/>
            </svg>
            """
        let without = block(svg, "svg")
        #expect(without.subviews.compactMap { $0 as? NSButton }
            .first { $0.accessibilityIdentifier() == "Artifact.Card.Open" }?.isHidden == true)

        let model = ArtifactPanelModel()
        let view = block(svg, "svg", opener: model.opener)
        let open = try #require(view.subviews.compactMap { $0 as? NSButton }
            .first { $0.accessibilityIdentifier() == "Artifact.Card.Open" })
        #expect(!open.isHidden)
        press(open)
        #expect(model.selected?.kind == .svg)
    }
}

@MainActor
@Suite("Artifact panel model")
struct ArtifactPanelModelTests {

    private func fence(_ language: String, _ body: String) -> String {
        "```\(language)\n\(body)\n```"
    }

    private let counter =
        "<!DOCTYPE html><html><head><title>Counter</title></head><body><button>+</button></body></html>"
    private let clock =
        "<!DOCTYPE html><html><head><title>Clock</title></head><body>12:00</body></html>"
    private let svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>
        """

    private func message(
        _ role: ChatMessage.Role, _ content: String, parent: UUID? = nil,
        at seconds: TimeInterval = 0, status: ChatMessage.Status = .complete
    ) -> ChatMessage {
        ChatMessage(
            role: role, content: content, status: status, parentID: parent,
            createdAt: Date(timeIntervalSince1970: seconds)
        )
    }

    @Test("Artifacts are listed in message order, then block order")
    func order() {
        let q = message(.user, "make things")
        let a = message(
            .assistant,
            "One:\n\n\(fence("html", counter))\n\nTwo:\n\n\(fence("svg", svg))\n\n\(fence("python", "print(1)"))",
            parent: q.id, at: 1
        )
        let q2 = message(.user, "more", parent: a.id, at: 2)
        let a2 = message(.assistant, fence("html", clock), parent: q2.id, at: 3)

        let model = ArtifactPanelModel()
        model.messagesDidChange([q, a, q2, a2])
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))

        #expect(model.items.map(\.kind) == [.html, .svg, .html])
        #expect(model.items.map(\.title) == ["Counter", "SVG image", "Clock"])
        #expect(model.selected?.id == model.items.first?.id)
    }

    @Test("Switching to a sibling branch swaps the list")
    func branchSwitch() {
        let q = message(.user, "make a thing")
        let first = message(.assistant, fence("html", counter), parent: q.id, at: 1)
        let regenerated = message(.assistant, fence("html", clock), parent: q.id, at: 2)
        let tree = [q, first, regenerated]

        let model = ArtifactPanelModel()
        model.messagesDidChange(MessageTree.activePath(in: tree, activeLeafID: first.id))
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        #expect(model.items.map(\.title) == ["Counter"])
        #expect(model.selected?.title == "Counter")

        model.messagesDidChange(MessageTree.activePath(in: tree, activeLeafID: regenerated.id))
        #expect(model.items.map(\.title) == ["Clock"])
        #expect(model.selected?.title == "Clock")

        model.messagesDidChange(MessageTree.activePath(in: tree, activeLeafID: first.id))
        #expect(model.items.map(\.title) == ["Counter"])
    }

    @Test("A branch with no artifacts leaves an empty list, not a stale one")
    func emptyBranch() {
        let q = message(.user, "q")
        let withPage = message(.assistant, fence("html", counter), parent: q.id, at: 1)
        let plain = message(.assistant, "just text", parent: q.id, at: 2)
        let tree = [q, withPage, plain]

        let model = ArtifactPanelModel()
        model.messagesDidChange(MessageTree.activePath(in: tree, activeLeafID: withPage.id))
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        model.messagesDidChange(MessageTree.activePath(in: tree, activeLeafID: plain.id))
        #expect(model.items.isEmpty)
        #expect(model.selected == nil)
        #expect(model.isOpen)
    }

    @Test("Closed, the transcript costs nothing until the panel opens")
    func lazy() {
        let q = message(.user, "q")
        let a = message(.assistant, fence("html", counter), parent: q.id, at: 1)
        let model = ArtifactPanelModel()
        model.messagesDidChange([q, a])
        #expect(model.items.isEmpty)
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        #expect(model.items.count == 1)
    }

    @Test("A block of a streaming reply opens detached, then joins the list")
    func streamingReply() {
        let q = message(.user, "q")
        let streaming = message(
            .assistant, fence("html", counter) + "\n\nand more text", parent: q.id, at: 1,
            status: .streaming
        )
        let model = ArtifactPanelModel()
        model.messagesDidChange([q, streaming])
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        #expect(model.items.isEmpty)
        #expect(model.selected?.source == counter + "\n")

        var done = streaming
        done.status = .complete
        model.messagesDidChange([q, done])
        #expect(model.items.count == 1)
        #expect(model.selected?.id == model.items.first?.id)
        #expect(model.detached == nil)
    }

    @Test("A detached artifact is dropped once nothing is streaming")
    func detachedDropped() {
        let model = ArtifactPanelModel()
        model.messagesDidChange([])
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        #expect(model.selected != nil)
        model.messagesDidChange([message(.user, "unrelated")])
        #expect(model.selected == nil)
    }

    @Test("Only assistant messages contribute")
    func assistantOnly() {
        let user = message(.user, fence("html", counter))
        let model = ArtifactPanelModel()
        model.messagesDidChange([user])
        model.open(ArtifactItem(id: "x", kind: .html, source: "<html></html>"))
        #expect(model.items.isEmpty)
    }

    @Test("Select, close, full screen and width clamping")
    func intents() {
        let q = message(.user, "q")
        let a = message(.assistant, fence("html", counter) + "\n\n" + fence("html", clock), parent: q.id, at: 1)
        let model = ArtifactPanelModel()
        model.messagesDidChange([q, a])
        model.open(ArtifactItem(id: "x", kind: .html, source: counter + "\n"))
        let second = model.items[1].id
        model.select(second)
        #expect(model.selected?.title == "Clock")
        model.select("nope")
        #expect(model.selected?.title == "Clock")

        model.toggleFullScreen()
        #expect(model.isFullScreen)
        model.close()
        #expect(!model.isOpen)
        #expect(!model.isFullScreen, "closing must bring the chat back")

        model.setPanelWidth(10)
        #expect(model.panelWidth == ArtifactPanelModel.widthRange.lowerBound)
        model.setPanelWidth(10_000)
        #expect(model.panelWidth == ArtifactPanelModel.widthRange.upperBound)
    }

    @Test("Fingerprints change with status and length, not with identity of the array")
    func fingerprint() {
        var m = message(.assistant, "abc")
        let before = ArtifactMessageFingerprint.of([m])
        #expect(before == ArtifactMessageFingerprint.of([m]))
        m.content = "abcd"
        #expect(before != ArtifactMessageFingerprint.of([m]))
        m.content = "abc"
        m.status = .streaming
        #expect(before != ArtifactMessageFingerprint.of([m]))
    }
}

@MainActor
@Suite("Artifact panel actions")
struct ArtifactPanelActionsTests {

    @Test("File names are safe and keep the right extension")
    func fileNames() {
        let item = ArtifactItem(id: "a", kind: .html, source: "<title>a/b: c*</title>")
        #expect(ArtifactPanelActions.suggestedFileName(for: item) == "a-b- c-.html")
        let svg = ArtifactItem(id: "a", kind: .svg, source: "<svg/>")
        #expect(ArtifactPanelActions.suggestedFileName(for: svg) == "SVG image.svg")
        let mermaid = ArtifactItem(id: "a", kind: .mermaid, source: "graph TD")
        #expect(ArtifactPanelActions.suggestedFileName(for: mermaid) == "Mermaid diagram.mmd")
        #expect(ArtifactPanelActions.suggestedFileName(for: mermaid, format: .png) == "Mermaid diagram.png")
        let dots = ArtifactItem(id: "a", kind: .html, source: "<title>...</title>")
        #expect(ArtifactPanelActions.suggestedFileName(for: dots) == "artifact.html")
    }

    @Test("The temporary page is private, exact, and old leftovers are removed")
    func temporaryFile() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("artifact-actions-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let item = ArtifactItem(id: "a", kind: .html, source: "<html>é</html>")

        let first = try ArtifactPanelActions.writeTemporaryHTML(item, in: base)
        #expect(try String(contentsOf: first, encoding: .utf8) == "<html>é</html>")
        #expect(first.pathExtension == "html")
        let attributes = try FileManager.default.attributesOfItem(atPath: first.path)
        #expect((attributes[.posixPermissions] as? Int) == 0o600)

        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-3 * 24 * 60 * 60)],
            ofItemAtPath: first.path
        )
        let second = try ArtifactPanelActions.writeTemporaryHTML(item, in: base)
        #expect(!FileManager.default.fileExists(atPath: first.path))
        #expect(FileManager.default.fileExists(atPath: second.path))
    }

    @Test("A rendered diagram becomes PNG bytes")
    func png() throws {
        let image = NSImage(size: NSSize(width: 4, height: 4), flipped: false) { rect in
            NSColor.red.setFill()
            rect.fill()
            return true
        }
        let data = try #require(ArtifactPanelActions.pngData(for: image))
        #expect(data.starts(with: [0x89, 0x50, 0x4E, 0x47]))
    }

    @Test("The image fit never blows a bitmap past 2x")
    func fitScale() {
        let size = CGSize(width: 100, height: 50)
        let panel = CGSize(width: 2_000, height: 2_000)
        #expect(ArtifactPanelFit.scale(image: size, available: panel, vector: false) == 2)
        #expect(ArtifactPanelFit.scale(image: size, available: panel, vector: true) == 8)
        #expect(ArtifactPanelFit.scale(image: size, available: CGSize(width: 132, height: 500), vector: true) == 1)
    }
}

@Suite("Artifact UI wiring")
struct ArtifactWiringTests {

    private static var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOf: Self.root.appendingPathComponent("Sources/Rapid/\(path)"), encoding: .utf8)
    }

    @Test("The stable accessibility identifiers are declared")
    func identifiers() throws {
        let panel = try source("UI/Artifacts/ArtifactPanelView.swift")
            + source("UI/Artifacts/ArtifactViewer.swift")
            + source("UI/Artifacts/ArtifactPanelHost.swift")
        for id in [
            "Artifact.Panel", "Artifact.Panel.List", "Artifact.Panel.Copy", "Artifact.Panel.Save",
            "Artifact.Panel.Reload", "Artifact.Panel.OpenInBrowser", "Artifact.Panel.Close",
            "Artifact.Panel.FullScreen", "Artifact.Panel.Resize",
        ] {
            #expect(panel.contains("\"\(id)\""), "\(id)")
        }
        let card = try source("UI/Artifacts/ArtifactCardView.swift")
            + source("UI/Markdown/TextKit/Render/MarkdownCodeBlockView.swift")
        #expect(card.contains("\"Artifact.Card.Open\""))
        let library = try source("UI/Artifacts/ArtifactLibraryView.swift")
        for id in ["Artifact.Library", "Artifact.Library.Table", "Artifact.Library.Back", "Artifact.Library.Detail"] {
            #expect(library.contains("\"\(id)\""), "\(id)")
        }
        #expect(try source("UI/SidebarView.swift").contains("\"Sidebar.Artifacts\""))
    }

    @Test("The chat view only hosts the panel")
    func hook() throws {
        let chat = try source("UI/ChatView.swift")
        #expect(chat.contains(".artifactPanel(model: artifactPanel, messages: messages)"))
    }
}
