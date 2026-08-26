import Foundation
import Testing
@testable import Rapid

@Suite("Image result deletion")
struct ImageResultDeletionTests {
    @Test("The native Keep action remains pressable through AppKit re-hosting")
    func keepActionUsesAnOrdinaryDefaultButton() throws {
        let source = try String(
            contentsOf: packageRoot.appendingPathComponent("Sources/Rapid/UI/ImagesView.swift"),
            encoding: .utf8
        )

        #expect(source.contains("Button(\"Keep\")"))
        #expect(source.contains(".keyboardShortcut(.defaultAction)"))
        #expect(source.contains(".accessibilityIdentifier(\"Images.Result.Delete.Keep\")"))
        #expect(!source.contains("Button(\"Keep\", role: .cancel)"))
    }

    @Test("Deleting the active result selects the adjacent older image")
    @MainActor
    func deletingActiveResultSelectsAdjacentImage() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        let newest = image("newest")
        let selected = image("selected")
        let oldest = image("oldest")
        viewModel.results = [newest, selected, oldest]
        viewModel.activeID = selected.id

        viewModel.delete(selected)

        #expect(viewModel.results == [newest, oldest])
        #expect(viewModel.activeImage?.id == oldest.id)
    }

    @Test("Deleting the last gallery image restores the empty stage")
    @MainActor
    func deletingOnlyResultClearsSelection() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        let only = image("only")
        viewModel.results = [only]
        viewModel.activeID = only.id

        viewModel.delete(only)

        #expect(viewModel.results.isEmpty)
        #expect(viewModel.activeID == nil)
        #expect(viewModel.activeImage == nil)
    }

    @Test("Deleting an edit source exits editing without leaving stale state")
    @MainActor
    func deletingEditSourceExitsEditing() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        viewModel.imageModels = [
            ModelEntry(
                alias: "image-model", hfRepo: "example/image", sizeOnDisk: nil,
                cached: true, kind: .image, imageCapability: .generationAndEditing
            ),
        ]
        viewModel.selectedAlias = "image-model"
        let source = image("source")
        let remaining = image("remaining")
        viewModel.results = [source, remaining]
        viewModel.beginEdit(source)

        viewModel.delete(source)

        #expect(!viewModel.isEditing)
        #expect(viewModel.results == [remaining])
        #expect(viewModel.activeImage?.id == remaining.id)
        #expect(viewModel.selectedAlias == "image-model")
    }

    @Test("Deleting an imported edit source preserves gallery results")
    @MainActor
    func deletingImportedSourcePreservesGallery() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        let galleryImage = image("gallery")
        let imported = image("imported")
        viewModel.results = [galleryImage]
        viewModel.beginEdit(imported)

        viewModel.delete(imported)

        #expect(!viewModel.isEditing)
        #expect(viewModel.results == [galleryImage])
        #expect(viewModel.activeImage?.id == galleryImage.id)
    }

    private func image(_ prompt: String) -> GeneratedImage {
        GeneratedImage(pngData: Data(prompt.utf8), prompt: prompt, isEdit: false)
    }

    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
