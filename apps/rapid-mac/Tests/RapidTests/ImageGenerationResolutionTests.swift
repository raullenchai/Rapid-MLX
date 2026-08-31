import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Image generation resolution")
struct ImageGenerationResolutionTests {
    @MainActor
    private final class ControlledCatalogLoader {
        private var continuations: [CheckedContinuation<[ModelEntry], Never>] = []
        var requestCount: Int { continuations.count }

        func load(_: URL) async -> [ModelEntry] {
            await withCheckedContinuation { continuations.append($0) }
        }

        func finish(_ index: Int, with entries: [ModelEntry]) {
            continuations[index].resume(returning: entries)
        }
    }

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func source(_ path: String) throws -> String {
        try String(contentsOf: packageRoot.appendingPathComponent(path), encoding: .utf8)
    }

    @Test("Images refreshes cache state live and modal engines bypass residency")
    func liveDownloadAndModalLaunchWiring() throws {
        let view = try Self.source("Sources/Rapid/UI/ImagesView.swift")
        let model = try Self.source("Sources/Rapid/Images/ImageGenViewModel.swift")

        #expect(view.contains("ImageCatalogRefreshKey(cacheGeneration: downloads.cacheGeneration)"))
        #expect(view.contains("residencyEligible: false"))
        #expect(model.components(separatedBy: "residencyEligible: false").count - 1 == 2,
                "Both generation and editing must use the modal process-swap path.")
    }

    @Test("A completed image pull changes the catalog key and view-model readiness to Start")
    @MainActor
    func completedPullInvalidatesCatalogReadiness() async throws {
        let binary = URL(fileURLWithPath: "/tmp/rapid-test-sidecar")
        let server = ServerManager(testingState: .idle, binaryPath: binary)
        let downloads = DownloadManager()
        var cached = false
        let entry: (Bool) -> ModelEntry = { isCached in
            ModelEntry(
                alias: "image-model", hfRepo: "example/image", sizeOnDisk: "1 GiB",
                cached: isCached, kind: .image, imageCapability: .generation
            )
        }
        let viewModel = ImageGenViewModel(server: server) { _ in [entry(cached)] }
        let host = NSHostingView(
            rootView: ImagesView(viewModel: viewModel, server: server)
                .environment(SettingsRouter())
                .environment(downloads)
                .environment(server)
        )
        host.layoutSubtreeIfNeeded()

        let beforeKey = ImageCatalogRefreshKey(cacheGeneration: downloads.cacheGeneration)
        for _ in 0..<100 where viewModel.imageModels.isEmpty {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(viewModel.imageModels.count == 1)
        let before = ModelReadiness.resolve(
            serverState: .idle,
            alias: "image-model",
            cacheState: viewModel.imageModels[0].cached ? .onDisk : .notOnDisk,
            sizeText: "1 GiB"
        )
        #expect(before.action == .download(alias: "image-model"))

        _ = downloads._testingSeedJob(alias: "image-model")
        cached = true
        downloads._testingFinish(alias: "image-model", status: 0, reason: .exit)
        let afterKey = ImageCatalogRefreshKey(cacheGeneration: downloads.cacheGeneration)
        #expect(afterKey != beforeKey, "A successful pull must restart the view's keyed task.")

        // The mounted ImagesView must observe cacheGeneration and refresh its
        // catalog without a test-side call to refreshCatalog().
        for _ in 0..<100 where viewModel.imageModels.first?.cached != true {
            try await Task.sleep(for: .milliseconds(20))
        }
        let after = ModelReadiness.resolve(
            serverState: .idle,
            alias: "image-model",
            cacheState: viewModel.imageModels[0].cached ? .onDisk : .notOnDisk,
            sizeText: "1 GiB"
        )
        #expect(after.action == .start(alias: "image-model"))
        _ = host
    }

    @Test("A cancelled older catalog refresh cannot overwrite the newest result")
    @MainActor
    func overlappingCatalogRefreshesKeepNewestResult() async {
        let binary = URL(fileURLWithPath: "/tmp/rapid-test-sidecar")
        let server = ServerManager(testingState: .idle, binaryPath: binary)
        let loader = ControlledCatalogLoader()
        let viewModel = ImageGenViewModel(server: server, catalogLoader: loader.load)
        let fresh = ModelEntry(
            alias: "fresh-image", hfRepo: "example/fresh", sizeOnDisk: "1 GiB",
            cached: true, kind: .image, imageCapability: .generation
        )

        let older = Task { await viewModel.refreshCatalog() }
        while loader.requestCount < 1 { await Task.yield() }
        older.cancel()
        let newer = Task { await viewModel.refreshCatalog() }
        while loader.requestCount < 2 { await Task.yield() }

        loader.finish(1, with: [fresh])
        await newer.value
        #expect(viewModel.imageModels.map(\.alias) == ["fresh-image"])

        // The cancelled subprocess may still unwind later with an empty or
        // partial result. It must not commit after the newer generation.
        loader.finish(0, with: [])
        await older.value
        #expect(viewModel.imageModels.map(\.alias) == ["fresh-image"])
        #expect(viewModel.catalogLoaded)
    }

    @Test("Image starters wrap instead of clipping in a horizontal rail")
    func startersRemainReadable() throws {
        let view = try Self.source("Sources/Rapid/UI/ImagesView.swift")
        #expect(view.contains("private var starters: some View {\n        LazyVGrid("))
        #expect(!view.contains("private var starters: some View {\n        ScrollView(.horizontal"))
    }

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

    @Test("A fresh view model defaults to the lowest resolution preset")
    @MainActor
    func defaultOutputSize() {
        let viewModel = ImageGenViewModel(server: ServerManager())
        #expect(viewModel.outputSize == "512x512")
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
