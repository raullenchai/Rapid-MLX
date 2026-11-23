import AppKit
import Foundation
import Testing
@testable import Rapid

@Suite("Audio artifact store")
struct AudioArtifactStoreTests {

    private func directory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-artifacts-\(UUID().uuidString)", isDirectory: true)
    }

    private func insert(
        _ store: AudioArtifactStore, _ bytes: [UInt8], text: String
    ) async -> String? {
        await withCheckedContinuation { continuation in
            store.insert(
                SynthesizedAudio(data: Data(bytes), contentType: "audio/wav"),
                text: text, voice: "Vivian", modelAlias: "tts"
            ) { continuation.resume(returning: $0) }
        }
    }

    @Test("Speech round-trips, newest first, with private permissions")
    func roundTrip() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AudioArtifactStore(directory: root)
        #expect(try await store.entries().isEmpty)

        #expect(await insert(store, [1, 2, 3], text: "first") == nil)
        try await Task.sleep(for: .milliseconds(20))
        #expect(await insert(store, [4, 5], text: "second") == nil)

        let entries = try await AudioArtifactStore(directory: root).entries()
        #expect(entries.map(\.text) == ["second", "first"])
        #expect(entries.map(\.byteCount) == [2, 3])
        #expect(entries.allSatisfy { $0.fileExtension == "wav" && $0.voice == "Vivian" })
        #expect(try Data(contentsOf: store.fileURL(for: entries[1])) == Data([1, 2, 3]))
        for url in [root, root.appendingPathComponent("index.json"), store.fileURL(for: entries[0])] {
            let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
            #expect(mode == (url == root ? 0o700 : 0o600), "\(url.lastPathComponent)")
        }
    }

    @Test("Beyond the limit the oldest files are removed")
    func limit() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AudioArtifactStore(directory: root, limit: 2)
        for index in 0..<3 {
            #expect(await insert(store, [UInt8(index)], text: "n\(index)") == nil)
            try await Task.sleep(for: .milliseconds(20))
        }
        let entries = try await store.entries()
        #expect(entries.map(\.text) == ["n2", "n1"])
        let files = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasSuffix(".wav") }
        #expect(files.count == 2)
    }

    @Test("Listing never writes")
    func listingIsReadOnly() async throws {
        let root = directory()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try await AudioArtifactStore(directory: root).entries()
        #expect(!FileManager.default.fileExists(atPath: root.path))
    }
}

@MainActor
@Suite("Artifact library")
struct ArtifactLibraryTests {

    private let page = "<!DOCTYPE html><html><head><title>Counter</title></head><body></body></html>"

    private func directory(_ name: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
    }

    private func conversation(
        _ title: String, messages: [ChatMessage], at seconds: TimeInterval
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(), title: title, messages: messages,
            createdAt: Date(timeIntervalSince1970: seconds),
            updatedAt: Date(timeIntervalSince1970: seconds)
        )
    }

    private func chat(
        _ title: String, content: String, at seconds: TimeInterval,
        status: ChatMessage.Status = .complete
    ) -> ChatConversation {
        let q = ChatMessage(role: .user, content: "q", createdAt: Date(timeIntervalSince1970: seconds))
        let a = ChatMessage(
            role: .assistant, content: content, status: status, parentID: q.id,
            createdAt: Date(timeIntervalSince1970: seconds + 1)
        )
        return conversation(title, messages: [q, a], at: seconds)
    }

    private func makeLibrary() -> (ArtifactLibrary, ImageArtifactStore, AudioArtifactStore, [URL]) {
        let images = directory("lib-images")
        let audio = directory("lib-audio")
        let imageStore = ImageArtifactStore(directory: images)
        let audioStore = AudioArtifactStore(directory: audio)
        return (ArtifactLibrary(imageStore: imageStore, audioStore: audioStore), imageStore, audioStore, [images, audio])
    }

    @Test("Code blocks from every stored chat are listed with their chat as the source")
    func chatsContribute() async {
        let (library, _, _, dirs) = makeLibrary()
        defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let first = chat("Counter chat", content: "```html\n\(page)\n```", at: 100)
        let second = chat("Diagram chat", content: "```mermaid\ngraph TD\n  A --> B\n```\n\n```python\nprint(1)\n```", at: 200)
        let plain = chat("Just text", content: "no fences here", at: 300)

        await library.refresh(conversations: [first, second, plain])

        #expect(library.items.map(\.originLabel) == ["Diagram chat", "Counter chat"])
        #expect(library.items.map(\.kind) == [.diagram, .page])
        #expect(library.items.map(\.title) == ["Mermaid diagram", "Counter"])
        if case .conversation(let id, let title)? = library.items.first?.origin {
            #expect(id == second.id)
            #expect(title == "Diagram chat")
        } else {
            Issue.record("the origin should be the conversation")
        }
    }

    @Test("A reply that is still streaming is not listed")
    func streamingExcluded() async {
        let (library, _, _, dirs) = makeLibrary()
        defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        await library.refresh(conversations: [
            chat("Live", content: "```html\n\(page)\n```", at: 1, status: .streaming),
        ])
        #expect(library.items.isEmpty)
    }

    @Test("Only the visible branch of a chat is listed")
    func visibleBranchOnly() async {
        let (library, _, _, dirs) = makeLibrary()
        defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let q = ChatMessage(role: .user, content: "q", createdAt: Date(timeIntervalSince1970: 1))
        let shown = ChatMessage(
            role: .assistant, content: "```html\n\(page)\n```", parentID: q.id,
            createdAt: Date(timeIntervalSince1970: 2)
        )
        let hidden = ChatMessage(
            role: .assistant, content: "```html\n<html><title>Old</title></html>\n```", parentID: q.id,
            createdAt: Date(timeIntervalSince1970: 3)
        )
        var conversation = conversation("Branchy", messages: [q, shown], at: 1)
        conversation.branches = [hidden]
        await library.refresh(conversations: [conversation])
        #expect(library.items.map(\.title) == ["Counter"])
    }

    @Test("Generated images and speech join the library")
    func mediaContributes() async throws {
        let (library, imageStore, audioStore, dirs) = makeLibrary()
        defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }

        let png = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3])
        let image = GeneratedImage(
            pngData: png, prompt: "a red fox  in\nsnow", isEdit: false,
            modelAlias: "flux", size: "512x512", createdAt: Date(timeIntervalSince1970: 500)
        )
        await withCheckedContinuation { continuation in
            imageStore.insert([image]) { _ in continuation.resume() }
        }
        await withCheckedContinuation { continuation in
            audioStore.insert(
                SynthesizedAudio(data: Data([1, 2, 3, 4]), contentType: "audio/mpeg"),
                text: "Hello there", voice: "Vivian", modelAlias: "tts"
            ) { _ in continuation.resume() }
        }

        await library.refresh(conversations: [chat("Counter chat", content: "```html\n\(page)\n```", at: 100)])

        let byKind = Dictionary(uniqueKeysWithValues: library.items.map { ($0.kind, $0) })
        #expect(Set(byKind.keys) == [.page, .image, .audio])

        let img = try #require(byKind[.image])
        #expect(img.title == "a red fox in snow")
        #expect(img.originLabel == "Image generation · flux")
        #expect(img.byteCount == png.count)
        #expect(library.imageURL(for: image.id) == imageStore.imageURL(image.id))

        let speech = try #require(byKind[.audio])
        #expect(speech.title == "Hello there")
        #expect(speech.originLabel == "Speech · tts")
        #expect(speech.byteCount == 4)
        if case .audio(let entry) = speech.payload {
            #expect(library.audioURL(for: entry).pathExtension == "mp3")
            #expect(FileManager.default.fileExists(atPath: library.audioURL(for: entry).path))
        } else {
            Issue.record("expected an audio payload")
        }
        // Newest first: the speech was just made, the image is dated 1970.
        #expect(library.items.first?.kind == .audio)
    }

    @Test("A refresh after a chat changes picks up the new content")
    func refreshFollowsChanges() async {
        let (library, _, _, dirs) = makeLibrary()
        defer { dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        var conversation = chat("Chat", content: "plain", at: 1)
        await library.refresh(conversations: [conversation])
        #expect(library.items.isEmpty)

        conversation.messages[1].content = "```html\n\(page)\n```"
        await library.refresh(conversations: [conversation])
        #expect(library.items.count == 1)

        await library.refresh(conversations: [])
        #expect(library.items.isEmpty)
    }

    @Test("The image store lists metadata without reading pixels or writing")
    func imageMetadata() async throws {
        let root = directory("meta-images")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ImageArtifactStore(directory: root)
        #expect(try await store.metadata().isEmpty)
        let image = GeneratedImage(
            pngData: Data(repeating: 7, count: 9), prompt: "p", isEdit: true, modelAlias: "m",
            size: "1x1", createdAt: Date(timeIntervalSince1970: 5)
        )
        await withCheckedContinuation { continuation in
            store.insert([image]) { _ in continuation.resume() }
        }
        let meta = try await store.metadata()
        #expect(meta.map(\.id) == [image.id])
        #expect(meta.first?.byteCount == 9)
        #expect(meta.first?.isEdit == true)
    }
}

@MainActor
@Suite("Artifact panel behaviour")
struct ArtifactPanelBehaviourTests {

    @Test("The panel opens in the side position and toggles full screen")
    func fullScreen() {
        let model = ArtifactPanelModel()
        model.open(ArtifactItem(id: "x", kind: .html, source: "<html></html>"))
        #expect(model.isOpen)
        #expect(!model.isFullScreen)
        model.toggleFullScreen()
        #expect(model.isFullScreen)
        model.toggleFullScreen()
        #expect(!model.isFullScreen)
    }

    @Test("Closing from full screen restores the chat")
    func closeLeavesFullScreen() {
        let model = ArtifactPanelModel()
        model.open(ArtifactItem(id: "x", kind: .html, source: "<html></html>"))
        model.toggleFullScreen()
        model.close()
        #expect(!model.isOpen)
        #expect(!model.isFullScreen)
        model.open(ArtifactItem(id: "x", kind: .html, source: "<html></html>"))
        #expect(!model.isFullScreen, "a reopened panel starts beside the chat")
    }

    @Test("The panel starts wide and can be dragged much wider")
    func widths() {
        let model = ArtifactPanelModel()
        #expect(model.panelWidth >= 520)
        model.setPanelWidth(1_200)
        #expect(model.panelWidth == 1_200)
    }

    private func event(_ type: NSEvent.EventType, x: CGFloat) -> NSEvent {
        NSEvent.mouseEvent(
            with: type, location: NSPoint(x: x, y: 10), modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 0, clickCount: 1, pressure: 1
        )!
    }

    @Test("Dragging the seam left widens the panel, right narrows it")
    func seamDrag() {
        let model = ArtifactPanelModel()
        let seam = ArtifactResizeSeam()
        seam.model = model
        seam.currentWidth = 500

        seam.mouseDown(with: event(.leftMouseDown, x: 800))
        seam.mouseDragged(with: event(.leftMouseDragged, x: 650))
        #expect(model.panelWidth == 650)
        seam.mouseDragged(with: event(.leftMouseDragged, x: 900))
        #expect(model.panelWidth == 400)
        #expect(!seam.mouseDownCanMoveWindow)
        #expect(seam.accessibilityIdentifier() == "Artifact.Panel.Resize")
    }
}

@Suite("Artifacts navigation wiring")
struct ArtifactNavigationWiringTests {

    private static var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOf: Self.root.appendingPathComponent("Sources/Rapid/\(path)"), encoding: .utf8)
    }

    @Test("The sidebar has an Artifacts section that opens the library page")
    func sidebarSection() throws {
        let sidebar = try source("UI/SidebarView.swift")
        #expect(sidebar.contains("case artifacts"))
        #expect(sidebar.contains("selection == .artifacts"))
        let content = try source("UI/ContentView.swift")
        #expect(content.contains("case .artifacts:"))
        #expect(content.contains("ArtifactLibraryView("))
    }

    @Test("The side panel is only reachable from an inline Open button")
    func panelOnlyFromTranscript() throws {
        // No sidebar or chat-view control may open the panel directly.
        #expect(!(try source("UI/SidebarView.swift")).contains("artifactPanel"))
        let chat = try source("UI/ChatView.swift")
        #expect(!chat.contains("artifactPanel.show"))
        #expect(!chat.contains("artifactPanel.open"))
    }

    @Test("Speech is stored when it is synthesized")
    func speechIsStored() throws {
        let audio = try source("Audio/AudioViewModel.swift")
        #expect(audio.contains("artifactStore.insert("))
    }
}
