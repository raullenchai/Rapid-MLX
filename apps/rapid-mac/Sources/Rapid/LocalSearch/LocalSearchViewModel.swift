import Foundation
import Observation

@MainActor
@Observable
final class LocalSearchViewModel {
    enum State: Equatable {
        case engineStopped, engineStarting, unsupported, loading, noFolders, modelMissing, ready, indexing, searching, noResults, results, error(String)
    }
    private(set) var state: State = .engineStopped
    private(set) var status: LocalSearchStatus?
    private(set) var results: [LocalSearchResult] = []
    private(set) var groups: [LocalSearchGroup]?
    var sections: [LocalSearchSection]? {
        guard kindFilter == .all, let groups else { return nil }
        return LocalSearchSection.previews(groups)
    }
    var displayedResults: [LocalSearchResult] { sections?.flatMap(\.results) ?? results }
    private(set) var keywordMode = false
    var query = ""
    var kindFilter: LocalSearchKindFilter = .all
    var visionAvailable: Bool { status?.model.vision == true }
    var visibleKindFilters: [LocalSearchKindFilter] { LocalSearchKindFilter.visible(vision: status?.model.vision) }
    func kindFilterChanged() { searchNow() }
    var focusRequest = 0
    var selection: String?
    var pendingRemoval: LocalSearchFolder?
    private(set) var isMutating = false
    var isIndexing: Bool { status?.needsPolling == true }
    var emptyResultsAction: EmptyResultsAction {
        if kindFilter != .all { return .clearFilter }
        return isIndexing ? .wait : .addFolder
    }
    enum EmptyResultsAction { case clearFilter, wait, addFolder }
    var searchedScope: String {
        let folders = status?.folders.map { LocalSearchPath.folderTitle($0.path) }.joined(separator: ", ") ?? ""
        return String(localized: "Searched \(folders) · \(kindFilter.title)")
    }
    private(set) var isStartingEngine = false
    func startEngine(using start: @MainActor () async -> Bool) async {
        guard !isStartingEngine else { return }
        isStartingEngine = true
        state = .engineStarting
        let succeeded = await start()
        isStartingEngine = false
        if !succeeded, endpoint == nil {
            if case .error = state { return }
            state = .error(String(localized: "The engine could not start. Try again, or check your model and engine installation in Settings."))
        }
    }
    func engineStateChanged(_ serverState: ServerState) {
        guard endpoint == nil else { return }
        if case .starting = serverState { state = .engineStarting }
        else if case .crashed(_, let message) = serverState { state = .error(message) }
        else if case .missing = serverState {
            state = .error(String(localized: "Search needs the local engine. Complete engine setup in Settings, then try again."))
        }
        else { state = isStartingEngine ? .engineStarting : .engineStopped }
    }
    func requestRemoval(_ folder: LocalSearchFolder) { pendingRemoval = folder }
    @discardableResult
    func confirmRemoval() -> Task<Void, Never>? {
        guard let folder = pendingRemoval else { return nil }
        // Capture synchronously: alert dismissal clears its presentation binding
        // before an asynchronously scheduled button action can read it.
        pendingRemoval = nil
        return Task { await remove(folder) }
    }
    func clearKindFilter() { kindFilter = .all; searchNow() }
    // Session-only disclosure choice survives view recreation and status refreshes.
    var foldersDisclosure = LocalSearchFoldersDisclosure()
    private(set) var visible = false
    private var endpoint: LocalSearchClient.Endpoint?
    private var searchTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var generation = 0
    private var sessionGeneration = 0
    private var refreshGeneration = 0
    private var handledDownload: UUID?
    private var mutationTask: Task<Bool, Never>?
    private var operationError = false
    #if DEBUG
    private var handledHarnessFolder = false

    /// Like the existing golden fixtures, the folder key is read only after
    /// the explicit harness gate passes. Release builds have no folder hook.
    static func harnessFolder(environment: [String: String] = ProcessInfo.processInfo.environment) -> String? {
        guard environment["RAPID_GUI_GOLDEN_MODE"] == "1",
              let path = environment["RAPID_GUI_LOCAL_SEARCH_FOLDER"],
              (path as NSString).isAbsolutePath else { return nil }
        return path
    }

    func addHarnessFolderIfNeeded(environment: [String: String] = ProcessInfo.processInfo.environment) async {
        guard visible, endpoint != nil, !handledHarnessFolder,
              let path = Self.harnessFolder(environment: environment) else { return }
        handledHarnessFolder = true
        await add(paths: [path])
    }
    #endif
    private var retryAction: (@MainActor () async throws -> Void)?
    private let client: LocalSearchClient
    private let debounce: Duration
    private let fetchStatus: @MainActor (LocalSearchClient.Endpoint) async throws -> LocalSearchStatus
    private let fetchSearch: @MainActor (String, [LocalSearchKind]?, LocalSearchClient.Endpoint) async throws -> LocalSearchResponse

    init(client: LocalSearchClient = .init(), debounce: Duration = .milliseconds(250),
         fetchStatus: (@MainActor (LocalSearchClient.Endpoint) async throws -> LocalSearchStatus)? = nil,
         fetchSearch: (@MainActor (String, LocalSearchClient.Endpoint) async throws -> LocalSearchResponse)? = nil,
         fetchMediaSearch: (@MainActor (String, [LocalSearchKind]?, LocalSearchClient.Endpoint) async throws -> LocalSearchResponse)? = nil) {
        self.client = client
        self.debounce = debounce
        self.fetchStatus = fetchStatus ?? { try await client.status($0) }
        self.fetchSearch = fetchMediaSearch ?? { query, kinds, endpoint in
            if let fetchSearch { return try await fetchSearch(query, endpoint) }
            return try await client.search(query, kinds: kinds, endpoint: endpoint)
        }
    }
    func appear(endpoint: LocalSearchClient.Endpoint?) async {
        visible = true
        await connect(endpoint)
    }
    func connect(_ endpoint: LocalSearchClient.Endpoint?) async {
        sessionGeneration += 1
        cancelMutations()
        cancelSearch()
        pollTask?.cancel()
        self.endpoint = endpoint
        status = nil
        results = []
        groups = nil
        selection = nil
        keywordMode = false
        retryAction = nil
        guard endpoint != nil else { state = isStartingEngine ? .engineStarting : .engineStopped; return }
        state = .loading
        await refresh()
        if !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { searchNow() }
    }
    func disappear() {
        visible = false
        endpoint = nil
        sessionGeneration += 1
        cancelMutations()
        cancelSearch()
        pollTask?.cancel()
        pollTask = nil
    }
    private func cancelMutations() {
        mutationTask?.cancel()
        mutationTask = nil
        isMutating = false
        retryAction = nil
        pendingRemoval = nil
        operationError = false
    }
    private func validateMutation(_ token: Int) throws {
        try Task.checkCancellation()
        guard visible, endpoint != nil, sessionGeneration == token else { throw CancellationError() }
    }
    private func cancelSearch() {
        generation += 1
        searchTask?.cancel()
        searchTask = nil
    }
    func queryChanged() { scheduleSearch(immediate: false) }
    func searchNow() { scheduleSearch(immediate: true) }
    private func scheduleSearch(immediate: Bool) {
        cancelSearch()
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            results = []
            groups = nil
            selection = nil
            keywordMode = false
        }
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard visible, let endpoint, status?.enabled == true else { return }
        guard !text.isEmpty else { deriveState(); return }
        let token = generation
        let kinds = kindFilter.kinds
        operationError = false
        state = .searching
        searchTask = Task { [weak self] in
            guard let self else { return }
            do {
                if !immediate { try await Task.sleep(for: debounce) }
                try Task.checkCancellation()
                let response = try await fetchSearch(text, kinds, endpoint)
                guard !Task.isCancelled, generation == token, visible else { return }
                let hadSelection = selection != nil
                results = response.results
                groups = response.groups
                keywordMode = response.mode == "keyword"
                if !displayedResults.contains(where: { $0.id == selection }) { selection = hadSelection ? nil : displayedResults.first?.id }
                searchTask = nil
                deriveState()
                // Discover external indexing and model changes even if the
                // previous snapshot was idle. Do not feed status changes back
                // into another search from this response.
                await refresh(searchIfChanged: false)
            } catch {
                guard !Task.isCancelled, generation == token, visible else { return }
                searchTask = nil
                report(error)
            }
        }
    }
    func refresh(searchIfChanged: Bool = true) async {
        guard visible, let endpoint else { return }
        let token = sessionGeneration
        refreshGeneration += 1
        let refreshToken = refreshGeneration
        do {
            let updated = try await fetchStatus(endpoint)
            guard visible, token == sessionGeneration, refreshToken == refreshGeneration, !Task.isCancelled else { return }
            let changed = status.map {
                $0.totals != updated.totals || $0.model != updated.model || $0.folders != updated.folders
                    || ($0.indexing.active && !updated.indexing.active)
            } ?? false
            status = updated
            let resetFilter = !visibleKindFilters.contains(kindFilter)
            if resetFilter {
                kindFilter = .all
                if !operationError { scheduleSearch(immediate: true) }
            }
            if !updated.enabled {
                cancelSearch()
                pollTask?.cancel()
                results = []
                groups = nil
                selection = nil
                state = .unsupported
                return
            }
            if searchTask == nil, !operationError {
                deriveState()
                if searchIfChanged, changed, !query.isEmpty { scheduleSearch(immediate: true) }
            }
            schedulePolling()
        } catch {
            guard visible, token == sessionGeneration, refreshToken == refreshGeneration, !Task.isCancelled else { return }
            if !operationError { report(error, operation: false) }
        }
    }
    private func schedulePolling() {
        pollTask?.cancel()
        guard visible, status?.needsPolling == true else { pollTask = nil; return }
        pollTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(1)) } catch { return }
            guard !Task.isCancelled else { return }
            await self?.refresh()
        }
    }
    private func deriveState() {
        guard let status else { return }
        if !status.enabled { state = .unsupported }
        else if status.folders.isEmpty { state = .noFolders }
        else if !displayedResults.isEmpty { state = .results }
        else if status.needsPolling, query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { state = .indexing }
        else if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { state = status.model.downloaded ? .ready : .modelMissing }
        else { state = .noResults }
    }
    func fileOpenFailed() {
        operationError = true
        state = .error(String(localized: "This file could not be opened. It may have moved or its drive may be disconnected. Reconnect the drive, or search again to refresh results."))
    }
    private func report(_ error: Error, operation: Bool = true) {
        operationError = operation
        if case LocalSearchClient.Failure.unsupported = error { state = .unsupported }
        else { state = .error(error.localizedDescription) }
    }
    func retry() async {
        guard visible, endpoint != nil else { return }
        if let action = retryAction { await mutate(action) }
        else {
            operationError = false
            await refresh()
            if case .error = state { return }
            if status != nil { searchNow() }
        }
    }
    @discardableResult
    private func mutate(_ action: @escaping @MainActor () async throws -> Void) async -> Bool {
        guard visible, endpoint != nil, !isMutating, !Task.isCancelled else { return false }
        isMutating = true
        let token = sessionGeneration
        let task = Task { @MainActor in
            defer {
                if token == self.sessionGeneration {
                    self.isMutating = false
                    self.mutationTask = nil
                }
            }
            do {
                try self.validateMutation(token)
                try await action()
                try self.validateMutation(token)
                self.retryAction = nil
                self.operationError = false
                await self.refresh()
                if case .error = self.state { return true }
                self.searchNow()
                return true
            } catch {
                guard !Task.isCancelled, self.visible, token == self.sessionGeneration else { return false }
                self.retryAction = action
                self.report(error)
                return false
            }
        }
        mutationTask = task
        return await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
    }
    func add(paths: [String]) async {
        guard visible, let endpoint else { return }
        let token = sessionGeneration
        var nextIndex = 0
        await mutate { [self, client] in
            // The retained retry resumes at the first unsuccessful folder.
            while nextIndex < paths.count {
                try validateMutation(token)
                try await client.add(path: paths[nextIndex], endpoint: endpoint)
                nextIndex += 1
            }
        }
    }
    func remove(_ folder: LocalSearchFolder) async {
        guard visible, let endpoint else { return }
        let token = sessionGeneration
        await mutate { [self, client] in
            try validateMutation(token)
            try await client.remove(id: folder.id, endpoint: endpoint)
        }
    }
    func downloadCompleted(_ jobID: UUID) async {
        guard visible, endpoint != nil, handledDownload != jobID else { return }
        let token = sessionGeneration
        // Completion waits behind admitted work without consuming the ID. A
        // canceled session or failed reindex can deliver the completion again.
        while let active = mutationTask {
            _ = await active.value
            guard visible, token == sessionGeneration, !Task.isCancelled else { return }
        }
        guard handledDownload != jobID, let endpoint else { return }
        await mutate { [self, client] in
            try validateMutation(token)
            try await client.reindex(id: nil, endpoint: endpoint)
            try validateMutation(token)
            handledDownload = jobID
        }
    }
    func reindex(_ id: String? = nil) async {
        guard visible, let endpoint else { return }
        let token = sessionGeneration
        await mutate { [self, client] in
            try validateMutation(token)
            try await client.reindex(id: id, endpoint: endpoint)
        }
    }
    func moveSelection(_ delta: Int) {
        let displayed = displayedResults
        guard !displayed.isEmpty else { return }
        let current = displayed.firstIndex { $0.id == selection } ?? (delta > 0 ? -1 : displayed.count)
        selection = displayed[max(0, min(displayed.count - 1, current + delta))].id
    }
    enum Direction { case up, down, left, right }
    @discardableResult
    func moveSelection(_ direction: Direction, columns: Int = 1) -> Bool {
        guard let sections else {
            if direction == .left || direction == .right {
                guard kindFilter.isGrid else { return false }
            }
            moveSelection(direction == .up ? -columns : direction == .down ? columns : direction == .left ? -1 : 1)
            return true
        }
        guard !sections.isEmpty else { return false }
        guard let sectionIndex = sections.firstIndex(where: { $0.results.contains { $0.id == selection } }),
              let itemIndex = sections[sectionIndex].results.firstIndex(where: { $0.id == selection }) else {
            selection = direction == .up ? displayedResults.last?.id : displayedResults.first?.id
            return true
        }
        let section = sections[sectionIndex]
        switch direction {
        case .left, .right:
            guard section.kind != .text else { return false }
            let offset = direction == .left ? -1 : 1
            selection = section.results[max(0, min(section.results.count - 1, itemIndex + offset))].id
        case .up, .down:
            let offset = direction == .up ? -1 : 1
            if section.kind == .text, section.results.indices.contains(itemIndex + offset) {
                selection = section.results[itemIndex + offset].id
            } else if sections.indices.contains(sectionIndex + offset) {
                let destination = sections[sectionIndex + offset]
                let index = destination.kind == .text ? (offset > 0 ? 0 : destination.results.count - 1)
                    : (section.kind == .text ? 0 : min(itemIndex, destination.results.count - 1))
                selection = destination.results[index].id
            }
        }
        return true
    }
    var selectedResult: LocalSearchResult? { displayedResults.first { $0.id == selection } }
}
