import Foundation
import Testing
@testable import Rapid

@Suite("Image generation resolution")
struct ImageGenerationResolutionTests {
    @Test("Every aspect and resolution maps to the expected API size")
    func outputSizes() {
        let expected: [ImageGenViewModel.Resolution: [ImageGenViewModel.Aspect: String]] = [
            .compact: [
                .square: "512x512",
                .portrait: "384x512",
                .landscape: "512x384",
            ],
            .balanced: [
                .square: "768x768",
                .portrait: "576x768",
                .landscape: "768x576",
            ],
            .detailed: [
                .square: "1024x1024",
                .portrait: "768x1024",
                .landscape: "1024x768",
            ],
            .large: [
                .square: "1280x1280",
                .portrait: "960x1280",
                .landscape: "1280x960",
            ],
            .high: [
                .square: "1536x1536",
                .portrait: "1152x1536",
                .landscape: "1536x1152",
            ],
            .maximum: [
                .square: "2048x2048",
                .portrait: "1536x2048",
                .landscape: "2048x1536",
            ],
        ]

        for resolution in ImageGenViewModel.Resolution.allCases {
            for aspect in ImageGenViewModel.Aspect.allCases {
                #expect(aspect.size(for: resolution) == expected[resolution]?[aspect])
            }
        }
    }

    @Test("A fresh view model still defaults to the pre-existing 1024x1024")
    @MainActor
    func defaultOutputSize() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        #expect(viewModel.outputSize == "1024x1024")
    }

    @Test("Generated dimensions satisfy the image API contract")
    func dimensionsAreSupported() {
        for resolution in ImageGenViewModel.Resolution.allCases {
            for aspect in ImageGenViewModel.Aspect.allCases {
                let dimensions = aspect.dimensions(for: resolution)
                #expect((256...2048).contains(dimensions.width))
                #expect((256...2048).contains(dimensions.height))
                #expect(dimensions.width.isMultiple(of: 16))
                #expect(dimensions.height.isMultiple(of: 16))
            }
        }
    }

    @Test("Editing switches to an edit model and exiting restores generation")
    @MainActor
    func editModeModelSelection() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        viewModel.imageModels = [
            ModelEntry(
                alias: "flux2-klein-4b", hfRepo: "example/generate",
                sizeOnDisk: nil, cached: true, kind: .image,
                imageCapability: .generationAndEditing
            ),
        ]
        viewModel.selectedAlias = "flux2-klein-4b"
        let source = GeneratedImage(pngData: Data([1, 2, 3]), prompt: "source", isEdit: false)

        viewModel.beginEdit(source)

        #expect(viewModel.isEditing)
        #expect(viewModel.activeImage?.id == source.id)
        #expect(viewModel.selectedAlias == "flux2-klein-4b")
        #expect(viewModel.selectableModels.map(\.alias) == ["flux2-klein-4b"])

        viewModel.cancelEdit()

        #expect(!viewModel.isEditing)
        #expect(viewModel.selectedAlias == "flux2-klein-4b")
        #expect(viewModel.selectableModels.map(\.alias) == ["flux2-klein-4b"])
    }

    @Test("Dual-capability models appear in both pickers")
    @MainActor
    func dualCapabilityModelSelection() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        viewModel.imageModels = [
            ModelEntry(
                alias: "flux2-klein-4b", hfRepo: "example/both",
                sizeOnDisk: "4.3 GiB", cached: true, kind: .image,
                imageCapability: .generationAndEditing
            ),
            ModelEntry(
                alias: "z-image-turbo", hfRepo: "example/generate",
                sizeOnDisk: "5.5 GiB", cached: true, kind: .image,
                imageCapability: .generation
            ),
        ]

        #expect(viewModel.generationModels.map(\.alias) == [
            "flux2-klein-4b", "z-image-turbo",
        ])
        #expect(viewModel.editModels.map(\.alias) == ["flux2-klein-4b"])
    }

    @Test("FLUX.2 editing uses the distilled 4-step estimate")
    @MainActor
    func editStepEstimate() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        #expect(viewModel.estimatedSteps == 4)
        viewModel.editSource = GeneratedImage(
            pngData: Data([1]), prompt: "source", isEdit: false
        )
        #expect(viewModel.estimatedSteps == 4)
    }

    @Test("A submitted image request keeps an immutable model target")
    @MainActor
    func requestTargetSnapshot() throws {
        let viewModel = ImageGenViewModel(server: ServerManager())
        viewModel.imageModels = [
            ModelEntry(
                alias: "flux2-klein-4b", hfRepo: "example/flux",
                sizeOnDisk: "4.3 GiB", cached: true, kind: .image,
                imageCapability: .generationAndEditing
            ),
            ModelEntry(
                alias: "z-image-turbo", hfRepo: "example/z",
                sizeOnDisk: "5.5 GiB", cached: true, kind: .image,
                imageCapability: .generation
            ),
        ]
        viewModel.selectedAlias = "flux2-klein-4b"
        let target = try #require(viewModel.makeRequestTarget())

        viewModel.selectedAlias = "z-image-turbo"

        #expect(target.alias == "flux2-klein-4b")
        #expect(target.hfPath == "example/flux")
    }
}
