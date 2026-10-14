import AppKit
import QuickLook
import SwiftUI

struct LocalSearchView: View {
    @Bindable var server: ServerManager
    @Bindable var model: LocalSearchViewModel
    let onStartModel: @MainActor () async -> Bool
    var onOpenSettings: () -> Void = {}
    @Environment(DownloadManager.self) private var downloads
    @State private var selectedShortcuts: Set<LocalSearchFolderShortcut> = [.desktop, .documents, .downloads]
    @State private var confirmingRestart = false
    @State private var previewURL: URL?
    @ScaledMetric(relativeTo: .caption) private var indexLineHeight: CGFloat = 24
    @State private var gridWidth: CGFloat = 0
    @State private var folderContentHeight: CGFloat = 0
    @FocusState private var searchFocused: Bool
    @FocusState private var resultsFocused: Bool
    private static let searchAlias = "embeddinggemma-2-4bit"

    private func beginEngineStart() {
        Task { await model.startEngine(using: onStartModel) }
    }

    private var endpoint: LocalSearchClient.Endpoint? {
        guard server.servingAlias != nil else { return nil }
        return .init(port: server.activePort, bearer: server.activeBearer)
    }
    private struct ConnectionKey: Equatable {
        let state: ServerState
        let endpoint: LocalSearchClient.Endpoint?
    }
    private var connectionKey: ConnectionKey { .init(state: server.state, endpoint: endpoint) }
    private var job: DownloadManager.Job? { downloads.job(for: Self.searchAlias) }
    private var foldersExpanded: Bool {
        model.foldersDisclosure.isExpanded(query: model.query, hasResults: !model.displayedResults.isEmpty)
    }

    private var completedDownloadID: UUID? {
        guard model.status != nil, let job, job.status == .completed else { return nil }
        return job.instanceID
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(model.visionAvailable ? String(localized: "Search documents, photos and videos") : String(localized: "Search your files by what is in them"), text: $model.query)
                    .textFieldStyle(.plain)
                    .focused($searchFocused)
                    .accessibilityLabel(String(localized: "Search local files"))
                    .accessibilityIdentifier("LocalSearch.QueryField")
                    .onChange(of: model.query) { _, _ in model.queryChanged() }
                    .onSubmit { model.searchNow() }
                    .onKeyPress(keys: [.return], phases: .down) { press in
                        guard press.modifiers.contains(.command), let result = model.selectedResult else { return .ignored }
                        reveal(result); return .handled
                    }
                    .onKeyPress(.downArrow) {
                        guard !model.displayedResults.isEmpty else { return .ignored }
                        model.selection = model.displayedResults.first?.id; searchFocused = false; resultsFocused = true
                        return .handled
                    }
                    .onKeyPress(.upArrow) {
                        guard !model.displayedResults.isEmpty else { return .ignored }
                        model.selection = model.displayedResults.last?.id; searchFocused = false; resultsFocused = true
                        return .handled
                    }
                if model.state == .searching { ProgressView().controlSize(.small).accessibilityLabel(String(localized: "Searching")) }
            }
            .padding(RapidTheme.Space.md)
            .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
            .disabled(endpoint == nil || model.state == .unsupported)

            if let status = model.status, status.enabled, !status.folders.isEmpty {
                RapidSegmentedControl(selection: $model.kindFilter,
                    options: model.visibleKindFilters.map { .init(value: $0, title: $0.title, identifier: $0.identifier) },
                    accessibilityLabel: String(localized: "Filter local search by file kind"))
                    .accessibilityIdentifier("LocalSearch.KindFilter")
                    .onChange(of: model.kindFilter) { _, _ in model.kindFilterChanged() }
                foldersHeader(status)
                if !status.model.downloaded { modelCard }
            }
            if let status = model.status, !status.folders.isEmpty { indexingLine(status) }
            if model.keywordMode {
                Text(String(localized: "Keyword results · Download the search model to also find files by meaning."))
                    .font(.caption).foregroundStyle(.secondary)
            }
            if case .error(let message) = model.state {
                VStack(alignment: .leading, spacing: 8) {
                    Text(message).foregroundStyle(.secondary)
                    Button(endpoint == nil ? String(localized: "Retry starting engine") : String(localized: "Retry")) {
                        if endpoint == nil { beginEngineStart() } else { Task { await model.retry() } }
                    }.buttonStyle(.rapidSecondary)
                        .accessibilityLabel(endpoint == nil ? String(localized: "Retry starting engine") : String(localized: "Retry local search request"))
                        .accessibilityIdentifier("LocalSearch.Retry")
                }
            }
            if !model.displayedResults.isEmpty {
                ScrollViewReader { proxy in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(String(localized: "Results may be incomplete while indexing"))
                            .font(.caption).foregroundStyle(.secondary)
                            .opacity(model.isIndexing ? 1 : 0)
                            .accessibilityHidden(!model.isIndexing)
                            .accessibilityIdentifier("LocalSearch.IncompleteResults")
                        resultsContent
                    }
                    .multilineTextAlignment(.leading)
                    .onChange(of: model.selection) { _, id in
                        if let id {
                            if let section = model.sections?.first(where: { $0.kind != .text && $0.results.contains { $0.id == id } }) {
                                proxy.scrollTo("LocalSearch.Section." + section.id)
                            } else { proxy.scrollTo(id) }
                        }
                    }
                }
            }
            else {
                ScrollView {
                    emptyState.frame(maxWidth: .infinity).padding(.vertical, 12)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(RapidTheme.Space.xl)
        .background(RapidTheme.surfaceCanvas)
        .rapidChatDynamicTypeClamp()
        .quickLookPreview($previewURL)
        .task(id: connectionKey) {
            await model.appear(endpoint: endpoint)
            model.engineStateChanged(server.state)
            #if DEBUG
            await model.addHarnessFolderIfNeeded()
            #endif
        }
        .onAppear { searchFocused = true }
        .onChange(of: endpoint) { _, endpoint in
            if endpoint != nil { resultsFocused = false; searchFocused = true }
        }
        .onDisappear { model.disappear(); previewURL = nil }
        .onChange(of: model.focusRequest) { _, _ in resultsFocused = false; searchFocused = true }
        .alert(String(localized: "Restart model to enable Search?"), isPresented: $confirmingRestart) {
            Button(String(localized: "Restart and enable Search")) {
                if let alias = server.servingAlias { Task { _ = await server.restartForSpawnSettings(alias: alias) } }
            }
            .accessibilityIdentifier("LocalSearch.Restart.Confirm")
            Button(String(localized: "Cancel"), role: .cancel) {}
                .accessibilityIdentifier("LocalSearch.Restart.Cancel")
        } message: {
            Text(String(localized: "The running model will restart. Other resident models will unload. Your conversations and files stay untouched."))
        }
        .task(id: completedDownloadID) {
            if let id = completedDownloadID { await model.downloadCompleted(id) }
        }
        .alert(String(localized: "Remove folder from search?"), isPresented: Binding(
            get: { model.pendingRemoval != nil }, set: { if !$0 { model.pendingRemoval = nil } }
        )) {
            Button(String(localized: "Cancel"), role: .cancel) { model.pendingRemoval = nil }
                .accessibilityLabel(String(localized: "Cancel removing folder"))
                .accessibilityIdentifier("LocalSearch.Remove.Cancel")
            Button(String(localized: "Remove"), role: .destructive) {
                model.confirmRemoval()
            }.accessibilityLabel(String(localized: "Confirm removing folder from search"))
            .accessibilityIdentifier("LocalSearch.Remove.Confirm")
        } message: {
            Text(String(localized: "This removes the folder's search index. Your files stay on your Mac."))
        }
    }

    private func foldersHeader(_ status: LocalSearchStatus) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button {
                    model.foldersDisclosure.toggle(query: model.query, hasResults: !model.displayedResults.isEmpty)
                } label: {
                    Label(String(localized: "Folders (\(status.folders.count))"), systemImage: foldersExpanded ? "chevron.down" : "chevron.right")
                }
                .buttonStyle(.rapidTertiary)
                .accessibilityLabel(String(localized: "Show or hide indexed folders"))
                .accessibilityIdentifier("LocalSearch.FoldersToggle")
                Spacer()
                addFolderMenu
                    .buttonStyle(.rapidSecondary)
                    .accessibilityLabel(String(localized: "Add folders to local search"))
                    .accessibilityIdentifier("LocalSearch.AddFolder")
                    .disabled(model.isMutating)
                Button(String(localized: "Reindex")) { Task { await model.reindex() } }
                    .buttonStyle(.rapidSecondary).disabled(status.folders.isEmpty || model.isMutating)
                    .accessibilityLabel(String(localized: "Reindex all folders"))
                    .accessibilityIdentifier("LocalSearch.Reindex")
            }
            if foldersExpanded, !status.folders.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(status.folders.enumerated()), id: \.element.id) { index, folder in
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(LocalSearchPath.folderTitle(folder.path)).lineLimit(1).truncationMode(.middle)
                                        .help(folder.path).accessibilityValue(folder.path)
                                    Text(folder.countLabel)
                                        .font(.caption).foregroundStyle(.secondary)
                                    if let counts = folder.filesByKind { Text(counts.label).font(.caption).foregroundStyle(.secondary) }
                                    Text(folder.state.title).font(.caption).foregroundStyle(.secondary)
                                    if let explanation = folder.skippedExplanation {
                                        DisclosureGroup(String(localized: "Why files were not indexed")) {
                                            Text(explanation)
                                                .font(.caption).foregroundStyle(.secondary)
                                                .multilineTextAlignment(.leading)
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                        }.accessibilityIdentifier("LocalSearch.Folder.\(index).SkipReasons")
                                    }
                                    if let title = folder.recoveryTitle {
                                        Button(title) {
                                            switch folder.state {
                                            case .missing: addFolders()
                                            case .waitingForModel: downloadSearchModel()
                                            default: Task { await model.reindex(folder.id) }
                                            }
                                        }.buttonStyle(.rapidSecondary)
                                        .accessibilityIdentifier("LocalSearch.Folder.\(index).Recover")
                                    }
                                    if let error = folder.error { Text(error).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
                                }
                                .multilineTextAlignment(.leading)
                                Spacer()
                                Button { Task { await model.reindex(folder.id) } } label: { Image(systemName: "arrow.clockwise") }
                                    .buttonStyle(.rapidTertiary)
                                    .accessibilityLabel(String(localized: "Reindex \(LocalSearchPath.folderTitle(folder.path))"))
                                    .accessibilityIdentifier("LocalSearch.Folder.\(index).Reindex")
                                Button { model.requestRemoval(folder) } label: { Image(systemName: "minus.circle") }
                                    .buttonStyle(.rapidTertiary)
                                    .accessibilityLabel(String(localized: "Remove \(LocalSearchPath.folderTitle(folder.path)) from search"))
                                    .accessibilityIdentifier("LocalSearch.Folder.\(index).Remove")
                            }
                            .accessibilityElement(children: .contain)
                            .accessibilityIdentifier("LocalSearch.Folder.\(index)")
                            .disabled(model.isMutating)
                        }
                    }
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { folderContentHeight = $0 }
                }
                // Roughly four standard rows; shorter lists reserve only their content height.
                .frame(height: min(folderContentHeight, 200))
            }
        }
    }

    private func indexingLine(_ status: LocalSearchStatus) -> some View {
        HStack(spacing: 8) {
            if status.indexing.active {
                if status.indexing.pausedReason != nil {
                    Image(systemName: "pause.circle").accessibilityHidden(true)
                } else if status.indexing.filesTotal > 0 {
                    ProgressView(value: Double(status.indexing.filesDone), total: Double(max(status.indexing.filesTotal, status.indexing.filesDone)))
                        .frame(width: 70).accessibilityLabel(String(localized: "Folder indexing progress"))
                        .accessibilityIdentifier("LocalSearch.IndexProgress")
                } else {
                    ProgressView().controlSize(.small).accessibilityLabel(String(localized: "Scanning folder"))
                        .accessibilityIdentifier("LocalSearch.IndexProgress")
                }
                Text(status.indexing.progressLabel).help(status.indexing.progressLabel)
                if status.indexing.pausedReason == nil, let folder = status.folders.first(where: { $0.id == status.indexing.folderId }) {
                    Text(LocalSearchPath.folderTitle(folder.path)).lineLimit(1).truncationMode(.middle).help(folder.path)
                } else if status.indexing.pausedReason == nil, let path = status.indexing.currentPath {
                    Text(abbreviated(URL(fileURLWithPath: path).deletingLastPathComponent().path))
                        .lineLimit(1).truncationMode(.middle).help(path)
                }
            } else {
                Text(status.totals.indexedSummary)
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .lineLimit(status.indexing.active && status.indexing.pausedReason != nil ? nil : 1)
        .frame(minHeight: indexLineHeight, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("LocalSearch.IndexStatus")
    }

    private func downloadSearchModel() {
        guard job?.status != .running else { return }
        downloads.startDownload(alias: Self.searchAlias, totalBytes: 1_100_000_000)
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(localized: "Find files by meaning")).font(.headline)
            Text(String(localized: "Keyword search is available now. Download the search model to find related ideas even when the words differ."))
                .font(.callout).foregroundStyle(.secondary)
            if let job, job.status == .running {
                ProgressView(value: job.progress.progressFraction)
                    .accessibilityLabel(String(localized: "Search model download progress"))
                    .accessibilityIdentifier("LocalSearch.DownloadProgress")
                if let subtitle = job.progress.progressSubtitle { Text(subtitle).font(.caption).foregroundStyle(.secondary) }
                Button(String(localized: "Cancel Download")) { downloads.cancelDownload(alias: Self.searchAlias) }
                    .buttonStyle(.rapidSecondary).accessibilityLabel(String(localized: "Cancel search model download"))
                    .accessibilityIdentifier("LocalSearch.CancelDownload")
            } else {
                if let job, case .failed(let message) = job.status {
                    Text(message).font(.caption).foregroundStyle(.secondary)
                }
                Button(String(localized: "Download search model (1.1 GB)")) {
                    downloadSearchModel()
                }
                .buttonStyle(.rapidPrimary)
                .accessibilityLabel(String(localized: "Download search model (1.1 GB)"))
                .accessibilityIdentifier("LocalSearch.DownloadModel")
            }
        }
        .padding(RapidTheme.Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
    }

    private var resultsList: some View {
        List(selection: $model.selection) {
            ForEach(Array(model.results.enumerated()), id: \.element.id) { index, result in
                resultRow(result, index: index)
            }
        }
        .listStyle(.plain)
    }

    private func resultRow(_ result: LocalSearchResult, index: Int) -> some View {
        HStack(alignment: .top, spacing: 12) {
            if result.kind == .text {
                Image(nsImage: NSWorkspace.shared.icon(forFile: result.path))
                    .resizable().frame(width: 32, height: 32).accessibilityHidden(true)
            } else {
                LocalSearchThumbnail(result: result).frame(width: 88, height: 66)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(URL(fileURLWithPath: result.path).lastPathComponent).font(.headline)
                    Spacer()
                    timestampTag(result)
                    if let page = result.page { Text(String(localized: "page \(page)")).font(.caption).foregroundStyle(.secondary) }
                    if let line = result.line { Text(String(localized: "line \(line)")).font(.caption).foregroundStyle(.secondary) }
                    Text(Date(timeIntervalSince1970: result.modifiedAt).formatted(date: .abbreviated, time: .omitted))
                        .font(.caption).foregroundStyle(.secondary)
                        .accessibilityLabel(String(localized: "Modified date"))
                }
                Text(LocalSearchPath.resultFolder(result, folders: model.status?.folders ?? [])).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle).help(result.path).accessibilityValue(result.path)
                if let explanation = result.nameOnlyExplanation {
                    Text(explanation).font(.callout).foregroundStyle(.secondary).lineLimit(4)
                } else if result.kind == .text { highlighted(result).font(.callout).lineLimit(4) }
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(resultAccessibilityLabel(result))
        .accessibilityAddTraits(model.selection == result.id ? [.isSelected, .isButton] : .isButton)
        .accessibilityAction { select(result); open(result) }
        .accessibilityIdentifier("LocalSearch.Result.\(index)")
        .tag(result.id)
        .id(result.id)
        .onTapGesture(count: 2) { select(result); open(result) }
        .contextMenu { resultMenu(result) }
    }

    @ViewBuilder private var resultsContent: some View {
        Group {
            if model.sections != nil { groupedResults }
            else if model.kindFilter.isGrid { resultsGrid } else { resultsList }
        }
        .focused($resultsFocused)
        .accessibilityLabel(String(localized: "Local search results"))
        .accessibilityIdentifier("LocalSearch.ResultsList")
        .onKeyPress(.upArrow) { model.moveSelection(.up, columns: model.kindFilter.isGrid ? gridColumns : 1); return .handled }
        .onKeyPress(.downArrow) { model.moveSelection(.down, columns: model.kindFilter.isGrid ? gridColumns : 1); return .handled }
        .onKeyPress(.leftArrow) {
            model.moveSelection(.left) ? .handled : .ignored
        }
        .onKeyPress(.rightArrow) {
            model.moveSelection(.right) ? .handled : .ignored
        }
        .onKeyPress(keys: [.return], phases: .down) { press in
            guard let result = model.selectedResult else { return .ignored }
            if press.modifiers.contains(.command) { reveal(result) } else { open(result) }
            return .handled
        }
        .onKeyPress(characters: CharacterSet(charactersIn: "c"), phases: .down) { press in
            guard press.modifiers.contains(.command), let result = model.selectedResult else { return .ignored }
            copyPath(result); return .handled
        }
        .onKeyPress(.space) {
            guard let result = model.selectedResult else { return .ignored }
            togglePreview(result); return .handled
        }
        .onChange(of: model.selection) { _, _ in
            if previewURL != nil { previewURL = model.selectedResult.map { URL(fileURLWithPath: $0.path) } }
        }
    }
    private var gridColumns: Int { max(1, Int((gridWidth + 16) / 196)) }

    private var resultsGrid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
                ForEach(Array(model.results.enumerated()), id: \.element.id) { index, result in
                    mediaCell(result, index: index)
                }
            }
            .padding(2)
        }
        .focusable()
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { gridWidth = $0 - 4 }
    }

    private func mediaCell(_ result: LocalSearchResult, index: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            LocalSearchThumbnail(result: result).frame(height: 132)
            Text(URL(fileURLWithPath: result.path).lastPathComponent)
                .font(.headline).lineLimit(2)
            timestampTag(result)
            if let explanation = result.nameOnlyExplanation {
                Text(explanation).font(.caption).foregroundStyle(.secondary).lineLimit(4)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(model.selection == result.id ? RapidTheme.hoverFill : Color.clear,
                    in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8)
            .strokeBorder(model.selection == result.id ? RapidTheme.brand : Color.clear, lineWidth: 2))
        .contentShape(Rectangle())
        .id(result.id)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(result.nameOnlyExplanation == nil
            ? URL(fileURLWithPath: result.path).lastPathComponent : resultAccessibilityLabel(result))
        .accessibilityValue(timestampLabel(result))
        .accessibilityIdentifier("LocalSearch.Result.\(index)")
        .accessibilityAddTraits(model.selection == result.id ? [.isSelected, .isButton] : .isButton)
        .accessibilityAction { model.selection = result.id; open(result) }
        .onTapGesture(count: 2) { model.selection = result.id; resultsFocused = true; open(result) }
        .onTapGesture { model.selection = result.id; searchFocused = false; resultsFocused = true }
        .contextMenu { resultMenu(result) }
    }

    private var groupedResults: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 20) {
                ForEach(model.sections ?? []) { section in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(String(localized: "\(section.title) (\(section.total.formatted()))"))
                                .font(.headline)
                                .accessibilityAddTraits(.isHeader)
                                .accessibilityIdentifier("LocalSearch.Section.\(section.id).Header")
                            Spacer()
                            Button(String(localized: "Show all")) { model.kindFilter = section.filter }
                                .accessibilityLabel(String(localized: "Show all \(section.title)"))
                                .accessibilityIdentifier("LocalSearch.Section.\(section.id).ShowAll")
                        }
                        if section.kind == .text {
                            ForEach(section.results) { result in
                                resultRow(result, index: resultIndex(result))
                                    .padding(.horizontal, 8)
                                    .background(model.selection == result.id ? RapidTheme.hoverFill : Color.clear,
                                                in: RoundedRectangle(cornerRadius: 8))
                                    .contentShape(Rectangle())
                                    .onTapGesture { select(result) }
                            }
                        } else {
                            ScrollViewReader { proxy in
                                ScrollView(.horizontal) {
                                    LazyHStack(alignment: .top, spacing: 16) {
                                        ForEach(section.results) { result in
                                            mediaCell(result, index: resultIndex(result)).frame(width: 180)
                                        }
                                    }.padding(2)
                                }.onChange(of: model.selection) { _, id in
                                    if let id, section.results.contains(where: { $0.id == id }) {
                                        proxy.scrollTo(id)
                                    }
                                }
                            }
                        }
                    }
                    .id("LocalSearch.Section." + section.id)
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("LocalSearch.Section.\(section.id)")
                }
            }.padding(2)
        }.focusable()
    }

    private func resultIndex(_ result: LocalSearchResult) -> Int {
        model.displayedResults.firstIndex { $0.id == result.id } ?? 0
    }
    private func select(_ result: LocalSearchResult) {
        model.selection = result.id; searchFocused = false; resultsFocused = true
    }

    private func timestampLabel(_ result: LocalSearchResult) -> String {
        guard result.kind == .video, let timestamp = result.timestamp else { return "" }
        return String(localized: "at \(LocalSearchTimestamp.formatted(timestamp))")
    }

    @ViewBuilder private func timestampTag(_ result: LocalSearchResult) -> some View {
        if result.kind == .video, let timestamp = result.timestamp {
            Text(String(localized: "at \(LocalSearchTimestamp.formatted(timestamp))"))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func resultMenu(_ result: LocalSearchResult) -> some View {
        Button(String(localized: "Open")) { open(result) }.accessibilityLabel(String(localized: "Open file"))
            .accessibilityIdentifier("LocalSearch.Result.Open")
        Button(String(localized: "Quick Look")) { togglePreview(result) }.accessibilityLabel(String(localized: "Preview file with Quick Look"))
            .accessibilityIdentifier("LocalSearch.Result.QuickLook")
        Button(String(localized: "Reveal in Finder")) { reveal(result) }.accessibilityLabel(String(localized: "Reveal file in Finder"))
            .accessibilityIdentifier("LocalSearch.Result.Reveal")
        Button(String(localized: "Copy Path")) { copyPath(result) }
            .accessibilityIdentifier("LocalSearch.Result.CopyPath")
        if result.kind == .video, let timestamp = result.timestamp {
            Button(String(localized: "Copy Timestamp")) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(LocalSearchTimestamp.formatted(timestamp), forType: .string)
            }
            .accessibilityIdentifier("LocalSearch.Result.CopyTimestamp")
        }
    }

    private func highlighted(_ result: LocalSearchResult) -> Text {
        LocalSearchHighlights.segments(result.snippet, ranges: result.highlights).reduce(Text("")) {
            $0 + ($1.bold ? Text(verbatim: $1.text).bold() : Text(verbatim: $1.text))
        }
    }
    @ViewBuilder private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "magnifyingglass").scaledSystemFont(32, relativeTo: .title).foregroundStyle(RapidTheme.brand).accessibilityHidden(true)
            switch model.state {
            case .engineStopped:
                Text(String(localized: "Search your own files")).font(.headline)
                Text(String(localized: "Start Search with your current model. Everything stays on your Mac."))
                Button(String(localized: "Start Search"), action: beginEngineStart).buttonStyle(.rapidPrimary)
                    .accessibilityLabel(String(localized: "Start engine for local search"))
                    .accessibilityIdentifier("LocalSearch.StartModel")
            case .engineStarting:
                ProgressView().accessibilityLabel(String(localized: "Starting search engine"))
                    .accessibilityIdentifier("LocalSearch.StartProgress")
                Text(String(localized: "Starting the engine…")).font(.headline)
                Text(String(localized: "Preparing your model. Search will be ready here."))
                if let detail = server.downloadProgress.progressSubtitle { Text(detail).font(.caption) }
            case .unsupported:
                Text(String(localized: "Get Search ready")).font(.headline)
                if server.localSearchEnabledAtSpawn || server.localSearchSupportedAtSpawn == false {
                    Text(String(localized: "This engine does not support local search. Update the engine and restart a model."))
                } else {
                    Text(String(localized: "Search needs an engine started with Local Search enabled."))
                    Button(String(localized: "Restart model for Search")) { confirmingRestart = true }
                        .buttonStyle(.rapidPrimary).accessibilityIdentifier("LocalSearch.RestartEngine")
                }
                Button(String(localized: "Open Settings"), action: onOpenSettings)
                    .buttonStyle(.rapidSecondary).accessibilityIdentifier("LocalSearch.OpenSettings")
            case .noFolders:
                Text(model.visionAvailable ? String(localized: "Search documents, photos and videos on your Mac") : String(localized: "Search your Mac by what is in your files")).font(.headline)
                Text(String(localized: "Files never leave your Mac. Only folders you add are read."))
                firstFolderChoices
                Text(String(localized: "PDF text and Word (.docx) files are searchable. Scanned PDFs need readable text; older .doc files need conversion."))
                    .font(.caption).accessibilityIdentifier("LocalSearch.DocumentFormats")
                if model.status?.model.downloaded == false { modelCard }
            case .indexing:
                Text(String(localized: "Building your search index")).font(.headline)
                Text(String(localized: "Results will appear as files are indexed. You can search while indexing continues."))
            case .noResults:
                Text(String(localized: "No files found")).font(.headline)
                Text(model.searchedScope).accessibilityIdentifier("LocalSearch.SearchedScope")
                switch model.emptyResultsAction {
                case .clearFilter:
                    Button(String(localized: "Search all file kinds")) { model.clearKindFilter() }
                        .buttonStyle(.rapidPrimary).accessibilityIdentifier("LocalSearch.ClearKindFilter")
                case .wait:
                    Text(String(localized: "More results may appear as indexing continues. You can try different words now."))
                case .addFolder:
                    Button(String(localized: "Add another folder…"), action: addFolders)
                        .buttonStyle(.rapidPrimary).accessibilityIdentifier("LocalSearch.AddFolder.NoResults")
                }
            case .loading, .searching:
                ProgressView().accessibilityLabel(String(localized: "Loading local search"))
            case .error:
                Text(String(localized: "Search is temporarily unavailable"))
                if endpoint == nil {
                    Button(String(localized: "Open Settings"), action: onOpenSettings)
                        .buttonStyle(.rapidSecondary).accessibilityIdentifier("LocalSearch.OpenSettings")
                }
            case .ready, .results, .modelMissing:
                Text(String(localized: "What are you looking for?")).font(.headline)
                Text(model.visionAvailable ? String(localized: "Describe an idea, a photo or a video, or search for words in your files.") : String(localized: "Describe an idea, or search for words in your files."))
            }
        }
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: 480)
    }
    private var firstFolderChoices: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                Text(String(localized: "Standard folders")).font(.headline)
                ForEach([LocalSearchFolderShortcut.desktop, .documents, .downloads], id: \.self) { shortcut in
                    Toggle(shortcut.title, isOn: Binding(
                        get: { selectedShortcuts.contains(shortcut) },
                        set: { if $0 { selectedShortcuts.insert(shortcut) } else { selectedShortcuts.remove(shortcut) } }
                    )).toggleStyle(.checkbox).padding(.vertical, 4)
                        .accessibilityIdentifier(shortcut.identifier)
                }
                Button(String(localized: "Search selected folders")) {
                    Task { await model.add(paths: LocalSearchFolderShortcut.allCases.filter { selectedShortcuts.contains($0) }.compactMap { $0.url?.path }) }
                }.buttonStyle(.rapidPrimary).disabled(selectedShortcuts.isEmpty || model.isMutating)
                    .accessibilityIdentifier("LocalSearch.AddFolder.Standard")
            }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 8))
            Button {
                Task { await model.add(paths: ["/"]) }
            } label: {
                Label(String(localized: "Whole Mac"), systemImage: "desktopcomputer")
                    .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
                    .padding(.horizontal, 8)
            }.buttonStyle(.rapidSecondary).disabled(model.isMutating)
                .accessibilityIdentifier("LocalSearch.AddFolder.WholeMac")
            Text(String(localized: "Whole Mac: the first index can take a long time. You can search while it runs. System files and app internals are skipped."))
                .font(.caption).foregroundStyle(.secondary)
            Button(action: addFolders) {
                Label(String(localized: "Choose folders…"), systemImage: "folder.badge.plus")
                    .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
                    .padding(.horizontal, 8)
            }.buttonStyle(.rapidSecondary).disabled(model.isMutating)
                .accessibilityIdentifier("LocalSearch.AddFolder.Choose")
        }.accessibilityElement(children: .contain)
            .accessibilityIdentifier("LocalSearch.AddFolder.EmptyState")
    }

    private func resultAccessibilityLabel(_ result: LocalSearchResult) -> String {
        result.accessibilityLabel(folders: model.status?.folders ?? [])
    }
    private func copyPath(_ result: LocalSearchResult) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(result.path, forType: .string)
    }

    private var addFolderMenu: some View {
        Menu {
            ForEach([LocalSearchFolderShortcut.desktop, .documents, .downloads], id: \.self) { shortcut in
                folderShortcut(shortcut)
            }
            Divider()
            folderShortcut(.wholeMac)
                .help(String(localized: "System files and app internals are skipped. First indexing can take a long time."))
            Text(String(localized: "System files and app internals are skipped. First indexing can take a long time."))
                .font(.caption)
            Divider()
            Button(String(localized: "Choose Folder…"), action: addFolders)
                .accessibilityIdentifier("LocalSearch.AddFolder.Choose")
        } label: {
            Label(String(localized: "Add Folder…"), systemImage: "chevron.down")
        }
        .menuStyle(.button)
        .menuIndicator(.hidden)
        .accessibilityIdentifier("LocalSearch.AddFolder.Menu")
    }

    private func folderShortcut(_ shortcut: LocalSearchFolderShortcut) -> some View {
        let url = shortcut.url
        let added = url.map { LocalSearchPath.isCovered($0, folders: model.status?.folders ?? []) } ?? false
        return Button {
            if let url { Task { await model.add(paths: [url.path]) } }
        } label: {
            Label(shortcut.title, systemImage: added ? "checkmark" : "folder")
        }
        .disabled(added || url == nil || model.isMutating)
        .accessibilityIdentifier(shortcut.identifier)
    }

    private func addFolders() {
        let panel = NSOpenPanel()
        panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Add Folder")
        panel.begin { response in
            guard response == .OK else { return }
            let paths = panel.urls.map(\.path)
            Task { @MainActor in await model.add(paths: paths) }
        }
    }
    private func abbreviated(_ path: String) -> String { LocalSearchPath.abbreviated(path) }
    private func open(_ result: LocalSearchResult) {
        let url = URL(fileURLWithPath: result.path)
        let classified = LocalSearchFileAction.classification(for: url)
        let action = LocalSearchFileAction.perform(for: url, classified: classified,
            open: { NSWorkspace.shared.open($0) },
            reveal: { NSWorkspace.shared.activateFileViewerSelecting([$0]) })
        if action == .unavailable {
            model.fileOpenFailed()
        }
    }
    private func reveal(_ result: LocalSearchResult) { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.path)]) }
    private func togglePreview(_ result: LocalSearchResult) {
        let url = URL(fileURLWithPath: result.path)
        previewURL = previewURL == url ? nil : url
    }
}
