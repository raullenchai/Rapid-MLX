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
}
