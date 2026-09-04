import Foundation
import Observation

@MainActor
@Observable
final class VideoGenViewModel {
    private struct ServerRequestContext: Equatable {
        let alias: String
        let port: Int
        let bearer: String
    }

    enum Mode: String, CaseIterable, Identifiable {
        case text
        case image

        var id: String { rawValue }
        var title: String { self == .text ? "Text" : "Image" }
        var capability: VideoModelCapability {
            self == .text ? .textToVideo : .imageToVideo
        }
    }

    struct ReferenceImage: Equatable {
        let data: Data
        let fileName: String
        let mimeType: String
    }

    var videoModels: [ModelEntry] = []
    var catalogLoaded = false
    var selectedAlias = ""
    var mode: Mode = .text

    var prompt = ""
    /// Zero is an internal "capabilities not loaded" sentinel. The first
    /// successful capabilities response replaces it with the shortest safe
    /// preset before the control or Generate action becomes available.
    var seconds = 0
    var size = ""
    var seed = Int.random(in: 1...999_999)
    var referenceImage: ReferenceImage?

    var capabilities: VideoCapabilities?
    var jobs: [VideoJob] = []
    var selectedJobID: String?
    var previewURL: URL?
    var isPreparing = false
    var isSubmitting = false
    var isRefreshing = false
    var isLoadingPreview = false
    var jobsAreReconciled = false
    var errorMessage: String?

    private let server: ServerManager
    private let physicalRAMGB: Double
    @ObservationIgnored private let client: any VideoClientProtocol
    @ObservationIgnored private let catalogLoader: (URL) async -> [ModelEntry]
    @ObservationIgnored private var catalogRefreshGeneration: UInt = 0
    @ObservationIgnored private var serverContextGeneration: UInt = 0
    @ObservationIgnored private var serverRefreshGeneration: UInt = 0
    @ObservationIgnored private var loadedServerContext: ServerRequestContext?
    @ObservationIgnored private var jobsServerContext: ServerRequestContext?
    @ObservationIgnored private var previewGeneration: UInt = 0
    @ObservationIgnored private var pollingGeneration: UInt = 0
    @ObservationIgnored private var pollingTask: Task<Void, Never>?
    @ObservationIgnored private let pollingInterval: Duration
    @ObservationIgnored private var missingActivePollCounts: [String: Int] = [:]
    private var pendingCacheCleanupJobIDs = Set<String>()
    private static let maximumMissingActivePolls = 5

    init(
        server: ServerManager,
        client: any VideoClientProtocol = VideoClient(),
        physicalRAMGB: Double = MacHardware.detect().physicalRAMGB,
        pollingInterval: Duration = .seconds(1),
        catalogLoader: @escaping (URL) async -> [ModelEntry] = {
            await ModelCatalog.videoEntries(binary: $0)
        }
    ) {
        self.server = server
        self.client = client
        self.physicalRAMGB = physicalRAMGB
        self.pollingInterval = pollingInterval
        self.catalogLoader = catalogLoader
    }

    var selectedModel: ModelEntry? {
        videoModels.first { $0.alias == selectedAlias }
    }

    var selectedJob: VideoJob? {
        if let selectedJobID { return jobs.first { $0.id == selectedJobID } }
        return jobs.first
    }

    var supportedModes: [Mode] {
        guard let model = selectedModel else { return [] }
        return Mode.allCases.filter { candidate in
            guard model.videoCapabilities.contains(candidate.capability) else { return false }
            if candidate == .image {
                return capabilities?.supportsImageInput == true
            }
            return capabilities?.modes.contains(candidate.capability) ?? true
        }
    }

    var sizePresets: [String] { capabilities?.sizePresets ?? [] }
    var durationPresets: [Int] { capabilities?.durationPresets(for: size) ?? [] }
    var referenceMaximumBytes: Int { capabilities?.referenceMaximumBytes ?? 0 }
    var referenceMaximumPixels: Int? { capabilities?.referenceMaximumPixels }
    var acceptedReferenceMIMETypes: Set<String> {
        capabilities?.acceptedReferenceMIMETypes ?? []
    }

    var isSelectedModelEligible: Bool {
        selectedModel.map(isModelEligible) ?? false
    }

    var memoryRequirementText: String? {
        guard let minimum = selectedModel?.minimumMemoryGB else { return nil }
        return "Needs at least \(Int(minimum.rounded())) GB unified memory; this Mac has \(Int(physicalRAMGB.rounded())) GB."
    }

    var isServerReady: Bool {
        currentServerContext != nil
    }

    var canSubmit: Bool {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return isServerReady
            && capabilities != nil
            && isSelectedModelEligible
            && supportedModes.contains(mode)
            && !trimmed.isEmpty
            && sizePresets.contains(size)
            && durationPresets.contains(seconds)
            && !isSubmitting
            && (mode == .text || referenceImage != nil)
    }

    var hasActiveJobs: Bool {
        jobs.contains {
            ($0.status == .queued || $0.status == .inProgress)
                && !pendingCacheCleanupJobIDs.contains($0.id)
        }
    }

    /// Only a live matching server can prove that an active job is still
    /// progressing. Stale history from a stopped or switched process must not
    /// block app shutdown or update coordination indefinitely.
    var hasLiveActiveJobs: Bool {
        currentServerContext != nil
            && jobsServerContext == currentServerContext
            && hasActiveJobs
    }

    var canSwitchModels: Bool {
        !isSubmitting
            && !isPreparing
            && !hasLiveActiveJobs
            && pendingCacheCleanupJobIDs.isEmpty
            && (!isServerReady || jobsAreReconciled)
    }

    var needsServerRefresh: Bool {
        isServerReady && (capabilities == nil || !jobsAreReconciled)
    }

    func isModelEligible(_ model: ModelEntry) -> Bool {
        guard let minimum = model.minimumMemoryGB,
              minimum.isFinite, minimum > 0, physicalRAMGB > 0 else { return false }
        return physicalRAMGB >= minimum
    }

    func refreshCatalog() async {
        catalogRefreshGeneration &+= 1
        let generation = catalogRefreshGeneration
        guard let binary = server.binaryPath else {
            catalogLoaded = true
            videoModels = []
            selectedAlias = ""
            selectedModelDidChange()
            return
        }
        let loaded = await catalogLoader(binary)
        guard !Task.isCancelled, generation == catalogRefreshGeneration else { return }
        let previousModel = selectedModel
        var filtered = loaded.filter {
            $0.kind == .video && !$0.videoCapabilities.isEmpty
        }
        if let previousModel,
           !canSwitchModels,
           !filtered.contains(where: { $0.alias == previousModel.alias }) {
            // A cache/catalog refresh must not orphan live or unreconciled work.
            filtered.append(previousModel)
        }
        let previousAlias = selectedAlias
        videoModels = filtered
        catalogLoaded = true
        let stillValid = filtered.contains { $0.alias == selectedAlias }
        if selectedAlias.isEmpty || !stillValid {
            selectedAlias = (filtered.first { $0.cached && isModelEligible($0) }
                ?? filtered.first(where: isModelEligible)
                ?? filtered.first)?.alias ?? ""
        }
        if previousAlias != selectedAlias {
            selectedModelDidChange()
        } else if !supportedModes.contains(mode) {
            mode = supportedModes.first ?? .text
        }
    }

    func selectModel(_ alias: String) {
        guard selectedAlias != alias,
              canSwitchModels,
              videoModels.contains(where: { $0.alias == alias }) else { return }
        selectedAlias = alias
        selectedModelDidChange()
    }

    func selectMode(_ next: Mode) {
        guard supportedModes.contains(next) else { return }
        mode = next
        if next == .text { referenceImage = nil }
    }

    func setReference(_ reference: ReferenceImage?) {
        referenceImage = reference
    }

    func selectSize(_ value: String) {
        guard size != value else { return }
        size = value
        if !durationPresets.contains(seconds) {
            seconds = durationPresets.first ?? 0
        }
    }

    func prepareSelectedModel() async {
        guard !isPreparing, let model = selectedModel, isSelectedModelEligible else { return }
        isPreparing = true
        errorMessage = nil
        defer { isPreparing = false }
        let ready = await server.ensureVideoServing(
            alias: model.alias,
            hfPath: model.hfRepo,
            minimumMemoryGB: model.minimumMemoryGB
        )
        guard selectedAlias == model.alias else { return }
        guard ready else {
            errorMessage = "Rapid couldn't start this video model. Check the memory notice or server log, then try again."
            return
        }
        await refreshServerData()
    }

    func serverStateDidChange() async {
        guard let context = currentServerContext else {
            invalidateServerContext()
            capabilities = nil
            return
        }
        if let loadedServerContext, loadedServerContext != context {
            invalidateServerContext()
            capabilities = nil
        }
        if capabilities == nil || loadedServerContext != context {
            await refreshServerData()
        }
    }

    func refreshServerData() async {
        guard let context = currentServerContext else { return }
        serverRefreshGeneration &+= 1
        let refreshGeneration = serverRefreshGeneration
        let contextGeneration = serverContextGeneration
        jobsAreReconciled = false
        isRefreshing = true
        defer {
            if refreshGeneration == serverRefreshGeneration { isRefreshing = false }
        }
        do {
            let newCapabilities = try await client.capabilities(
                port: context.port, bearer: context.bearer
            )
            guard requestIsCurrent(
                context,
                contextGeneration: contextGeneration,
                refreshGeneration: refreshGeneration
            ) else { return }
            capabilities = newCapabilities
            loadedServerContext = context
            reconcileControls()
            errorMessage = nil
            do {
                let newJobs = try await client.list(
                    port: context.port, bearer: context.bearer, limit: 30
                )
                guard requestIsCurrent(
                    context,
                    contextGeneration: contextGeneration,
                    refreshGeneration: refreshGeneration
                ) else { return }
                jobs = reconciledJobs(from: newJobs, context: context)
                jobsServerContext = context
                jobsAreReconciled = true
                reconcileSelection()
                reconcileJobPolling()
            } catch {
                guard requestIsCurrent(
                    context,
                    contextGeneration: contextGeneration,
                    refreshGeneration: refreshGeneration
                ) else { return }
                // Controls remain usable when history alone is unavailable.
                jobsAreReconciled = false
                errorMessage = "Video controls are ready, but recent videos couldn't be loaded."
            }
        } catch {
            guard requestIsCurrent(
                context,
                contextGeneration: contextGeneration,
                refreshGeneration: refreshGeneration
            ) else { return }
            jobsAreReconciled = false
            errorMessage = error.localizedDescription
        }
    }

    func pollJobs() async {
        guard let context = currentServerContext, !isRefreshing else { return }
        let contextGeneration = serverContextGeneration
        do {
            let previous = selectedJob?.status
            let newJobs = try await client.list(
                port: context.port, bearer: context.bearer, limit: 30
            )
            guard requestIsCurrent(
                context, contextGeneration: contextGeneration
            ) else { return }
            jobs = reconciledJobs(from: newJobs, context: context)
            jobsServerContext = context
            jobsAreReconciled = true
            reconcileSelection()
            if selectedJob?.status == .completed, previous != .completed {
                await loadSelectedPreview()
            }
            // Stop the owner task only after a newly-completed preview has
            // finished downloading; cancelling it earlier would cancel that
            // URLSession request along with the polling loop.
            reconcileJobPolling()
        } catch {
            // Poll failures are transient during a model stop/restart. The
            // explicit refresh/start actions surface actionable errors.
            jobsAreReconciled = false
        }
    }

    func submit() async {
        guard canSubmit, let context = currentServerContext else { return }
        let contextGeneration = serverContextGeneration
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let reference = mode == .image ? referenceImage : nil
        isSubmitting = true
        errorMessage = nil
        defer { isSubmitting = false }
        do {
            let job = try await client.create(
                VideoCreateRequest(
                    prompt: trimmed,
                    model: selectedAlias,
                    seconds: seconds,
                    size: size,
                    seed: seed,
                    reference: reference?.data,
                    referenceFileName: reference?.fileName,
                    referenceMIMEType: reference?.mimeType
                ),
                port: context.port,
                bearer: context.bearer
            )
            guard requestIsCurrent(
                context, contextGeneration: contextGeneration
            ) else { return }
            jobs.removeAll { $0.id == job.id }
            jobs.insert(job, at: 0)
            jobsServerContext = context
            missingActivePollCounts[job.id] = 0
            pendingCacheCleanupJobIDs.remove(job.id)
            selectedJobID = job.id
            previewURL = nil
            prompt = ""
            seed = Int.random(in: 1...999_999)
            reconcileJobPolling()
        } catch {
            guard requestIsCurrent(
                context, contextGeneration: contextGeneration
            ) else { return }
            errorMessage = error.localizedDescription
        }
    }

    func selectJob(_ id: String) async {
        guard jobs.contains(where: { $0.id == id }) else { return }
        selectedJobID = id
        previewURL = nil
        await loadSelectedPreview()
    }

    func loadSelectedPreview() async {
        guard let job = selectedJob, job.status == .completed,
              let context = currentServerContext,
              jobsServerContext == context else {
            previewURL = nil
            return
        }
        let contextGeneration = serverContextGeneration
        previewGeneration &+= 1
        let generation = previewGeneration
        isLoadingPreview = true
        defer { if generation == previewGeneration { isLoadingPreview = false } }
        do {
            let url = try await client.content(
                id: job.id, port: context.port, bearer: context.bearer
            )
            guard generation == previewGeneration,
                  requestIsCurrent(
                    context, contextGeneration: contextGeneration
                  ),
                  selectedJobID == job.id,
                  jobs.contains(where: { $0.id == job.id }) else { return }
            previewURL = url
        } catch {
            guard generation == previewGeneration,
                  requestIsCurrent(
                    context, contextGeneration: contextGeneration
                  ),
                  selectedJobID == job.id else { return }
            errorMessage = error.localizedDescription
        }
    }

    func delete(_ job: VideoJob) async {
        guard let context = currentServerContext,
              (jobsServerContext == context || pendingCacheCleanupJobIDs.contains(job.id)),
              job.status != .inProgress else { return }
        let contextGeneration = serverContextGeneration
        do {
            try await client.delete(
                id: job.id, port: context.port, bearer: context.bearer
            )
            guard requestIsCurrent(
                context, contextGeneration: contextGeneration
            ) else { return }
            pendingCacheCleanupJobIDs.remove(job.id)
            missingActivePollCounts.removeValue(forKey: job.id)
            jobs.removeAll { $0.id == job.id }
            reconcileJobPolling()
            if selectedJobID == job.id {
                selectedJobID = jobs.first?.id
                previewURL = nil
                await loadSelectedPreview()
            }
        } catch {
            guard requestIsCurrent(
                context, contextGeneration: contextGeneration
            ) else { return }
            if error as? VideoClientError == .cacheRemoval {
                pendingCacheCleanupJobIDs.insert(job.id)
                reconcileJobPolling()
            }
            errorMessage = error.localizedDescription
        }
    }

    private func selectedModelDidChange() {
        invalidateServerContext()
        capabilities = nil
        jobs = []
        jobsServerContext = nil
        missingActivePollCounts = [:]
        pendingCacheCleanupJobIDs = []
        jobsAreReconciled = false
        reconcileJobPolling()
        selectedJobID = nil
        previewURL = nil
        referenceImage = nil
        errorMessage = nil
        if !supportedModes.contains(mode) { mode = supportedModes.first ?? .text }
    }

    private func reconcileControls() {
        let modes = supportedModes.filter {
            capabilities?.modes.contains($0.capability) == true
        }
        if !modes.contains(mode) {
            mode = modes.first ?? supportedModes.first ?? .text
            if mode == .text { referenceImage = nil }
        }
        if !sizePresets.contains(size) { size = sizePresets.first ?? "" }
        if !durationPresets.contains(seconds) {
            seconds = durationPresets.first ?? 0
        }
    }

    private func reconcileSelection() {
        guard !jobs.isEmpty else {
            previewGeneration &+= 1
            selectedJobID = nil
            previewURL = nil
            return
        }
        if selectedJobID == nil || !jobs.contains(where: { $0.id == selectedJobID }) {
            previewGeneration &+= 1
            selectedJobID = jobs.first?.id
            previewURL = nil
        }
    }

    private func reconciledJobs(
        from serverJobs: [VideoJob],
        context: ServerRequestContext
    ) -> [VideoJob] {
        let reportedIDs = Set(serverJobs.map(\.id))
        for id in reportedIDs {
            missingActivePollCounts.removeValue(forKey: id)
        }
        guard jobsServerContext == context else {
            missingActivePollCounts = [:]
            let pendingCleanup = jobs.filter {
                pendingCacheCleanupJobIDs.contains($0.id)
                    && !reportedIDs.contains($0.id)
            }
            return serverJobs + pendingCleanup
        }
        var retained: [VideoJob] = []
        for job in jobs where !reportedIDs.contains(job.id) {
            if pendingCacheCleanupJobIDs.contains(job.id) {
                retained.append(job)
            } else if job.status == .queued || job.status == .inProgress {
                let misses = (missingActivePollCounts[job.id] ?? 0) + 1
                if misses <= Self.maximumMissingActivePolls {
                    missingActivePollCounts[job.id] = misses
                    retained.append(job)
                } else {
                    missingActivePollCounts.removeValue(forKey: job.id)
                }
            } else {
                missingActivePollCounts.removeValue(forKey: job.id)
            }
        }
        return serverJobs + retained
    }

    private var currentServerContext: ServerRequestContext? {
        guard !selectedAlias.isEmpty,
              server.servingAlias == selectedAlias,
              let bearer = server.activeBearer, !bearer.isEmpty else { return nil }
        return ServerRequestContext(
            alias: selectedAlias,
            port: server.activePort,
            bearer: bearer
        )
    }

    private func invalidateServerContext() {
        serverContextGeneration &+= 1
        serverRefreshGeneration &+= 1
        previewGeneration &+= 1
        loadedServerContext = nil
        jobsAreReconciled = false
        isRefreshing = false
        isLoadingPreview = false
        reconcileJobPolling()
    }

    /// Polling belongs to the long-lived view model, not the Video view. Jobs
    /// therefore keep progressing when the user visits another tab or hides
    /// the experimental surface. A server switch cancels the loop and leaves
    /// history intact for reconciliation if that video model is started again.
    private func reconcileJobPolling() {
        guard hasLiveActiveJobs else {
            pollingGeneration &+= 1
            pollingTask?.cancel()
            pollingTask = nil
            return
        }
        guard pollingTask == nil else { return }
        pollingGeneration &+= 1
        let generation = pollingGeneration
        let interval = pollingInterval
        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: interval)
                } catch {
                    break
                }
                guard let self else { break }
                await self.pollJobs()
                guard self.hasLiveActiveJobs else { break }
            }
            guard let self, self.pollingGeneration == generation else { return }
            self.pollingTask = nil
        }
    }

    private func requestIsCurrent(
        _ context: ServerRequestContext,
        contextGeneration: UInt,
        refreshGeneration: UInt? = nil
    ) -> Bool {
        guard contextGeneration == serverContextGeneration,
              currentServerContext == context else { return false }
        return refreshGeneration.map { $0 == serverRefreshGeneration } ?? true
    }
}
