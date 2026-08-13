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

    /// User-visible phases of a render. A completed denoise is not a completed
    /// request: VAE decode, image encoding, transport, and client decode still
    /// happen after the final sampling step.
    enum Phase: Equatable { case preparing, denoising, finalizing }

    static func nextPhase(from current: Phase, progress: ImageClient.ImageProgress) -> Phase {
        if progress.total > 0, progress.step >= progress.total {
            return .finalizing
        }
        if progress.running { return .denoising }
        return .preparing
    }

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
    /// Every installed/available image model (all image capability rows). The
    /// picker lists these directly — one dropdown that scales to N models,
    /// same shape as the chat picker, rather than a fixed set of boxes.
    var imageModels: [ModelEntry] = []
    var catalogLoaded: Bool = false
    /// The alias the picker points at. Settable directly by the dropdown.
    var selectedAlias: String = ""

    var generationModels: [ModelEntry] {
        imageModels.filter { $0.imageCapability?.supportsGeneration == true }
    }

    var editModels: [ModelEntry] {
        imageModels.filter { $0.imageCapability?.supportsEditing == true }
    }

    var selectableModels: [ModelEntry] {
        isEditing ? editModels : generationModels
    }

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
        if let editSource { return editSource }
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
    /// Immutable request target used by progress, status copy, and Cancel while
    /// the picker may still be bound to a different catalog selection.
    private(set) var inFlightAlias: String?
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
        progress?.total ?? Self.seedSteps(for: inFlightAlias ?? selectedAlias)
    }

    static func seedSteps(for alias: String) -> Int {
        if alias.localizedCaseInsensitiveContains("qwen-image-edit") { return 20 }
        return alias.localizedCaseInsensitiveContains("z-image") ? 8 : 4
    }

    /// A readable name for the selected model, shown in the cold-load HUD.
    var selectedDisplayName: String {
        let alias = inFlightAlias ?? selectedAlias
        return alias.isEmpty ? "the model" : alias
    }

    // MARK: - Edit
    var editSource: GeneratedImage?
    var isEditing: Bool { editSource != nil }
    private var previousGenerationAlias: String?

    private let client = ImageClient()
    private let server: ServerManager

    init(server: ServerManager) {
        self.server = server
    }

    var canSubmit: Bool {
        !isGenerating
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedAlias.isEmpty
            && selectableModels.contains { $0.alias == selectedAlias }
    }

    func use(starter: String) {
        prompt = starter
    }

    func select(_ image: GeneratedImage) {
        activeID = image.id
        if isEditing {
            editSource = image
            prompt = ""
        } else {
            prompt = image.prompt
        }
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
        let candidates = selectableModels
        let stillValid = candidates.contains { $0.alias == selectedAlias }
        guard selectedAlias.isEmpty || !stillValid else { return }
        selectedAlias = (candidates.first { $0.cached } ?? candidates.first)?.alias ?? ""
    }

    // MARK: - Generate

    func submit() async {
        // Claim the run synchronously (on the MainActor, before any await) so
        // two rapid submits can't both slip past the gate and launch concurrent
        // renders. ``withRequest`` clears it when the run ends.
        guard canSubmit, let target = makeRequestTarget() else { return }
        isGenerating = true
        inFlightAlias = target.alias
        if let source = editSource {
            await runEdit(source: source, target: target)
        } else {
            await runGenerate(target: target)
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
        guard let model = inFlightAlias else { return }
        cancelTask = Task { await client.cancel(model: model, port: port, bearer: bearer) }
    }

    private func runGenerate(target: RequestTarget) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // Snapshot at submission: the composer stays enabled through the
        // (possibly minutes-long) warm-up await, so a later aspect/resolution
        // change must not retarget the in-flight request.
        let size = outputSize
        guard !trimmed.isEmpty else { return }
        await withRequest {
            guard await self.server.ensureServing(
                alias: target.alias,
                hfPath: target.hfPath,
                estimatedMemoryGB: target.estimatedMemoryGB,
                imageMode: .generation
            ) else {
                throw ImageClientError.notReady
            }
            guard !self.cancelling else { throw CancellationError() }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(model: target.alias, port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.generate(
                prompt: trimmed, model: target.alias, size: size,
                count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                if self.results.count > Self.maxResults {
                    self.results.removeLast(self.results.count - Self.maxResults)
                }
                self.activeID = first.id
            }
            self.prompt = ""
            // Empty (cancelled before the first image) leaves the gallery as-is.
        }
    }

    private func runEdit(source: GeneratedImage, target: RequestTarget) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        await withRequest {
            guard await self.server.ensureServing(
                alias: target.alias,
                hfPath: target.hfPath,
                estimatedMemoryGB: target.estimatedMemoryGB,
                imageMode: .editing
            ) else {
                throw ImageClientError.notReady
            }
            guard !self.cancelling else { throw CancellationError() }
            let port = self.server.activePort
            let bearer = self.server.activeBearer
            let poll = self.startPolling(model: target.alias, port: port, bearer: bearer)
            defer { poll.cancel() }
            let images = try await self.client.edit(
                imagePNG: source.pngData, prompt: trimmed, model: target.alias,
                count: 1, seed: nil, port: port, bearer: bearer
            )
            if let first = images.first {
                self.results.insert(contentsOf: images, at: 0)
                if self.results.count > Self.maxResults {
                    self.results.removeLast(self.results.count - Self.maxResults)
                }
                self.activeID = first.id
                // Continue from the newest result so iterative edits never
                // accidentally reapply to the original source.
                self.editSource = first
            }
            self.prompt = ""
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
                    guard let self else { return }
                    self.progress = snap
                    self.phase = Self.nextPhase(from: self.phase, progress: snap)
                    if snap.running, self.denoiseStartedAt == nil {
                        self.denoiseStartedAt = Date()
                    }
                }
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
    }

    func beginEdit(_ image: GeneratedImage) {
        if !isEditing { previousGenerationAlias = selectedAlias }
        editSource = image
        activeID = image.id
        prompt = ""
        errorMessage = nil
        if !editModels.contains(where: { $0.alias == selectedAlias }) {
            selectedAlias = (editModels.first { $0.cached } ?? editModels.first)?.alias ?? ""
        }
    }

    func cancelEdit() {
        editSource = nil
        prompt = ""
        if let previousGenerationAlias,
           generationModels.contains(where: { $0.alias == previousGenerationAlias }) {
            selectedAlias = previousGenerationAlias
        } else {
            selectedAlias = (
                generationModels.first { $0.cached } ?? generationModels.first
            )?.alias ?? ""
        }
        previousGenerationAlias = nil
    }

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
            inFlightAlias = nil
            progress = nil
            genStartedAt = nil
            denoiseStartedAt = nil
        }
        do {
            try await body()
        } catch is CancellationError {
            // Cancel during residency loading has no image engine to signal yet;
            // once loading returns, stop locally before sending the render.
        } catch let error as ImageClientError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func makeRequestTarget() -> RequestTarget? {
        guard let selected = selectableModels.first(where: { $0.alias == selectedAlias })
        else { return nil }
        return RequestTarget(
            alias: selected.alias,
            hfPath: selected.hfRepo,
            estimatedMemoryGB: ModelSizing.residentEstimateGB(
                alias: selected.alias,
                sizeText: selected.sizeOnDisk
            )
        )
    }

    struct RequestTarget: Sendable {
        let alias: String
        let hfPath: String?
        let estimatedMemoryGB: Double
    }
}
