import Foundation
import Observation

/// State + orchestration for the Images tab. Mirrors ``ChatViewModel``:
/// an ``@Observable`` store the view binds to, owning the image client and
/// the results, and reading ``ServerManager.activePort`` / ``activeBearer``
/// at request time (never caching — they change across a reload).
@MainActor
@Observable
final class ImageGenViewModel {
    /// Aspect ratio stays independent from output resolution so changing one
    /// never silently resets the other.
    enum Aspect: String, CaseIterable, Identifiable {
        case square, portrait, landscape
        var id: String { rawValue }
        var label: String {
            switch self {
            case .square: return "1:1"
            case .portrait: return "3:4"
            case .landscape: return "4:3"
            }
        }
        func dimensions(for resolution: Resolution) -> (width: Int, height: Int) {
            switch self {
            case .square: return (resolution.longEdge, resolution.longEdge)
            case .portrait: return (resolution.longEdge * 3 / 4, resolution.longEdge)
            case .landscape: return (resolution.longEdge, resolution.longEdge * 3 / 4)
            }
        }

        func size(for resolution: Resolution) -> String {
            let dimensions = dimensions(for: resolution)
            return "\(dimensions.width)x\(dimensions.height)"
        }
    }

    /// Long-edge output presets. Every aspect maps these values to dimensions
    /// accepted by the server (256...2048 and a multiple of 16).
    enum Resolution: Int, CaseIterable, Identifiable {
        case compact = 512
        case balanced = 768
        case detailed = 1024
        case large = 1280
        case high = 1536
        case maximum = 2048

        var id: Int { rawValue }
        var longEdge: Int { rawValue }
    }

    /// The two phases of a render, shown very differently: a reassuring
    /// (indeterminate) cold-load, then a determinate denoise.
    enum Phase: Equatable { case preparing, denoising }

    /// A few one-tap prompt starters to beat the blank page.
    static let starters: [String] = [
        "A cozy ramen shop at night in the rain, neon, steam, 35mm",
        "Studio portrait of an elderly fisherman, dramatic side light",
        "A minimalist product shot of a ceramic mug on linen",
        "A whale drifting through clouds above a city at dusk",
    ]

    // MARK: - Composed input
    var prompt: String = ""
    var aspect: Aspect = .square
    var resolution: Resolution = .detailed

    var outputSize: String {
        aspect.size(for: resolution)
    }

    var outputSizeLabel: String {
        outputSize.replacingOccurrences(of: "x", with: " × ")
    }

    // MARK: - Catalog
    /// Every installed/available image model (the ``[image:gen]`` rows). The
    /// picker lists these directly — one dropdown that scales to N models,
    /// same shape as the chat picker, rather than a fixed set of boxes.
    var imageModels: [ModelEntry] = []
    var catalogLoaded: Bool = false
    /// The alias the picker points at. Settable directly by the dropdown.
    var selectedAlias: String = ""

    // MARK: - Results
    /// Cap on the in-memory session gallery. Each result holds a full-resolution
    /// PNG (multiple MB), so an unbounded list would grow app memory without
    /// limit across a long session; older results roll off the end.
    static let maxResults = 30
    /// Newest-first session gallery (the filmstrip).
    var results: [GeneratedImage] = []
    /// The focal image the stage shows; nil ⇒ newest, or empty state.
    var activeID: GeneratedImage.ID?
    var activeImage: GeneratedImage? {
        if let activeID, let hit = results.first(where: { $0.id == activeID }) { return hit }
        return results.first
    }

    // MARK: - Run state
    var isGenerating: Bool = false
    var phase: Phase = .preparing
    var progress: ImageClient.ImageProgress?
    var errorMessage: String?
    /// True only for the window between "Cancel pressed" and the run ending.
    private(set) var cancelling: Bool = false
    /// When the current run started — drives a live elapsed clock in the HUD
    /// that keeps moving even during the cold model-load phase.
    private(set) var genStartedAt: Date?
    /// When denoising actually began (first `running` step). ETA is computed
    /// from THIS, not ``genStartedAt`` — otherwise minutes of cold model load
    /// inflate the per-step estimate.
    private(set) var denoiseStartedAt: Date?

    /// Steps the bar should assume before the server reports a live total.
    /// Derived from the selected model family (turbo Z-Image wants ~8, the
    /// distilled Klein/schnell 4) so the bar is sensibly scaled from step one.
    var estimatedSteps: Int {
        progress?.total ?? Self.seedSteps(for: selectedAlias)
    }

    static func seedSteps(for alias: String) -> Int {
        alias.localizedCaseInsensitiveContains("z-image") ? 8 : 4
    }

    /// A readable name for the selected model, shown in the cold-load HUD.
    var selectedDisplayName: String {
        selectedAlias.isEmpty ? "the model" : selectedAlias
    }

    // MARK: - Edit (parked lane, kept for later)
    var editSource: GeneratedImage?

    private let client = ImageClient()
    private let server: ServerManager

    init(server: ServerManager) {
        self.server = server
    }

    var canSubmit: Bool {
        !isGenerating
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedAlias.isEmpty
    }

    func use(starter: String) {
        prompt = starter
    }

    func select(_ image: GeneratedImage) {
        activeID = image.id
        prompt = image.prompt
    }

    /// Load the image-gen alias catalog (safe to call repeatedly).
    func refreshCatalog() async {
        guard let binary = server.binaryPath else { return }
        imageModels = await ModelCatalog.imageEntries(binary: binary)
        catalogLoaded = true
        resolveAlias()
    }

    /// Keep ``selectedAlias`` valid: default to a cached model (so the first
    /// run doesn't force a pull), else the first image model. Only overrides
    /// when the current selection is empty or no longer in the catalog, so a
    /// user's explicit pick survives a refresh.
    private func resolveAlias() {
        let stillValid = imageModels.contains { $0.alias == selectedAlias }
        guard selectedAlias.isEmpty || !stillValid else { return }
        selectedAlias = (imageModels.first { $0.cached } ?? imageModels.first)?.alias ?? ""
    }

    // MARK: - Generate

    func submit() async {
        // Claim the run synchronously (on the MainActor, before any await) so
        // two rapid submits can't both slip past the gate and launch concurrent
        // renders. ``withRequest`` clears it when the run ends.
        guard !isGenerating else { return }
        isGenerating = true
        if let source = editSource {
            await runEdit(source: source)
        } else {
            await runGenerate()
        }
    }

    /// The in-flight cancel POST, tracked so the next render can wait for it to
    /// land — otherwise a delayed cancel could arrive after this generation
    /// ended and stop the *following* one.
    private var cancelTask: Task<Void, Never>?

    func cancel() {
        guard isGenerating, !cancelling else { return }
        cancelling = true
        let port = server.activePort
        let bearer = server.activeBearer
        let model = selectedAlias
        cancelTask = Task { await client.cancel(model: model, port: port, bearer: bearer) }
    }

    private func runGenerate() async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // Snapshot at submission: the composer stays enabled through the
        // (possibly minutes-long) warm-up await, so a later aspect/resolution
        // change must not retarget the in-flight request.
        let size = outputSize
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let selected = self.imageModels.first { $0.alias == self.selectedAlias }
            let hf = selected?.hfRepo
            let estimatedGB = ModelSizing.residentEstimateGB(
                alias: self.selectedAlias,
                sizeText: selected?.sizeOnDisk
            )
            guard await self.server.ensureServing(
                alias: self.selectedAlias,
                hfPath: hf,
                estimatedMemoryGB: estimatedGB
            ) else {
                throw ImageClientError.notReady
            }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(model: self.selectedAlias, port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.generate(
                prompt: trimmed, model: self.selectedAlias, size: size,
                count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                if self.results.count > Self.maxResults {
                    self.results.removeLast(self.results.count - Self.maxResults)
                }
                self.activeID = first.id
            }
            // Empty (cancelled before the first image) leaves the gallery as-is.
        }
    }

    private func runEdit(source: GeneratedImage) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // Same snapshot rule as ``runGenerate``.
        let size = outputSize
        guard !trimmed.isEmpty, !selectedAlias.isEmpty else { return }
        await withRequest {
            let selected = self.imageModels.first { $0.alias == self.selectedAlias }
            let hf = selected?.hfRepo
            let estimatedGB = ModelSizing.residentEstimateGB(
                alias: self.selectedAlias,
                sizeText: selected?.sizeOnDisk
            )
            guard await self.server.ensureServing(
                alias: self.selectedAlias,
                hfPath: hf,
                estimatedMemoryGB: estimatedGB
            ) else {
                throw ImageClientError.notReady
            }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(model: self.selectedAlias, port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.edit(
                imagePNG: source.pngData, prompt: trimmed, model: self.selectedAlias,
                size: size, count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                if self.results.count > Self.maxResults {
                    self.results.removeLast(self.results.count - Self.maxResults)
                }
                self.activeID = first.id
            }
        }
    }

    /// Poll the server's live denoise progress ~3×/second and mirror it into
    /// ``progress`` / ``phase`` so the stage shows a true step bar and ETA.
    private func startPolling(model: String, port: Int, bearer: String?) -> Task<Void, Never> {
        Task { [weak self] in
            while !Task.isCancelled {
                if let snap = await self?.client.fetchProgress(
                    model: model,
                    port: port,
                    bearer: bearer
                ) {
                    self?.progress = snap
                    self?.phase = snap.running ? .denoising : .preparing
                    if snap.running, self?.denoiseStartedAt == nil {
                        self?.denoiseStartedAt = Date()
                    }
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
    }

    func beginEdit(_ image: GeneratedImage) {
        editSource = image
        prompt = ""
        errorMessage = nil
    }

    func cancelEdit() { editSource = nil }

    /// Shared request wrapper: flips run state, resets progress, and funnels
    /// every failure into ``errorMessage``.
    private func withRequest(_ body: @escaping () async throws -> Void) async {
        // Wait for any prior cancel POST to land before starting, so a delayed
        // cancel can never stop this fresh render.
        await cancelTask?.value
        cancelTask = nil
        isGenerating = true
        cancelling = false
        phase = .preparing
        progress = nil
        genStartedAt = Date()
        denoiseStartedAt = nil
        errorMessage = nil
        defer {
            isGenerating = false
            cancelling = false
            progress = nil
            genStartedAt = nil
            denoiseStartedAt = nil
        }
        do {
            try await body()
            editSource = nil
            prompt = ""
        } catch let error as ImageClientError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
