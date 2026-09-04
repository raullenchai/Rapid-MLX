import AppKit
import AVKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

enum VideoReferenceLoaderError: Error, Equatable {
    case notRegularFile
    case tooLarge
    case unsupportedFormat
}

enum VideoReferenceLoader {
    static func load(
        from url: URL,
        fileManager: FileManager = .default,
        maximumBytes: Int = VideoClient.maxReferenceBytes,
        maximumPixels: Int? = nil
    ) throws -> Data {
        let (readLimit, overflowed) = maximumBytes.addingReportingOverflow(1)
        guard maximumBytes >= 0, !overflowed else {
            throw VideoReferenceLoaderError.tooLarge
        }
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular else {
            throw VideoReferenceLoaderError.notRegularFile
        }
        guard let byteCount = (attributes[.size] as? NSNumber)?.uint64Value,
              byteCount <= UInt64(maximumBytes) else {
            throw VideoReferenceLoaderError.tooLarge
        }

        // Read at most one byte past the contract even if the file grows
        // between metadata inspection and the read.
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: readLimit) ?? Data()
        guard data.count <= maximumBytes else {
            throw VideoReferenceLoaderError.tooLarge
        }
        if let maximumPixels {
            guard maximumPixels > 0 else { throw VideoReferenceLoaderError.tooLarge }
            let options = [kCGImageSourceShouldCache: false] as CFDictionary
            guard let source = CGImageSourceCreateWithData(data as CFData, options),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options)
                    as? [CFString: Any],
                  let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
                  let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
                  width > 0, height > 0 else {
                throw VideoReferenceLoaderError.unsupportedFormat
            }
            let (pixelCount, overflowed) = width.multipliedReportingOverflow(by: height)
            guard !overflowed, pixelCount <= maximumPixels else {
                throw VideoReferenceLoaderError.tooLarge
            }
        }
        return data
    }

    static func mimeType(for data: Data) -> String? {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options),
              let identifier = CGImageSourceGetType(source) as String?,
              let type = UTType(identifier) else { return nil }
        if type.conforms(to: .jpeg) { return "image/jpeg" }
        if type.conforms(to: .png) { return "image/png" }
        if type.conforms(to: .webP) { return "image/webp" }
        return nil
    }
}

enum VideoPreviewSaver {
    static func save(
        source: URL,
        destination: URL,
        fileManager: FileManager = .default
    ) throws {
        let directory = destination.deletingLastPathComponent()
        let staging = directory.appendingPathComponent(
            ".rapid-video-save-\(UUID().uuidString).mp4"
        )
        defer { try? fileManager.removeItem(at: staging) }
        try fileManager.copyItem(at: source, to: staging)
        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(
                destination,
                withItemAt: staging,
                backupItemName: nil,
                options: []
            )
        } else {
            try fileManager.moveItem(at: staging, to: destination)
        }
    }
}

private struct VideoCatalogRefreshKey: Hashable {
    let cacheGeneration: UInt
}

struct VideoView: View {
    @Bindable var viewModel: VideoGenViewModel
    @Bindable var server: ServerManager
    @Environment(DownloadManager.self) private var downloads
    @Environment(SettingsRouter.self) private var settingsRouter
    @Environment(\.openWindow) private var openWindow

    @State private var showingReferenceImporter = false
    @State private var pendingDeletion: VideoJob?
    @State private var isSavingPreview = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: RapidTheme.Space.lg) {
                    header
                    stage
                    if !viewModel.jobs.isEmpty { history }
                }
                .frame(maxWidth: RapidTheme.Layout.contentMaxWidth)
                .padding(RapidTheme.Space.xl)
                .frame(maxWidth: .infinity)
            }
            composer
        }
        .background(RapidTheme.surfaceCanvas)
        .task(id: VideoCatalogRefreshKey(cacheGeneration: downloads.cacheGeneration)) {
            await viewModel.refreshCatalog()
            await viewModel.serverStateDidChange()
        }
        .onChange(of: server.state) { _, _ in
            Task { await viewModel.serverStateDidChange() }
        }
        .fileImporter(
            isPresented: $showingReferenceImporter,
            allowedContentTypes: acceptedReferenceContentTypes,
            allowsMultipleSelection: false,
            onCompletion: importReference
        )
        .sheet(item: $pendingDeletion) { job in
            VideoDeletionSheet(
                job: job,
                onKeep: { pendingDeletion = nil },
                onDelete: {
                    pendingDeletion = nil
                    Task { await viewModel.delete(job) }
                }
            )
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Video")
                    .font(RapidFont.pageTitle)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Generate short videos locally. Start small—video models are large and generation can take several minutes.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            Spacer()
            Text("EXPERIMENTAL")
                .font(RapidFont.groupLabel)
                .foregroundStyle(RapidTheme.brandPrimaryDeep)
                .padding(.horizontal, RapidTheme.Space.sm)
                .padding(.vertical, RapidTheme.Space.xs)
                .background(RapidTheme.brandPrimaryTint, in: Capsule())
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Video.Header")
    }

    @ViewBuilder
    private var stage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: RapidTheme.Radius.panel, style: .continuous)
                .fill(RapidTheme.surfaceRaised)
                .overlay {
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.panel, style: .continuous)
                        .stroke(RapidTheme.hairline, lineWidth: 1)
                }

            if let url = viewModel.previewURL {
                VideoPlaybackView(url: url)
                    .id(url)
                    .clipShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.panel, style: .continuous))
                    .overlay(alignment: .topTrailing) { resultActions }
            } else if let job = viewModel.selectedJob {
                jobStage(job)
            } else {
                emptyStage
            }
        }
        .aspectRatio(16 / 9, contentMode: .fit)
        .frame(maxHeight: 520)
        .accessibilityIdentifier("Video.Stage")
    }

    private var emptyStage: some View {
        VStack(spacing: RapidTheme.Space.md) {
            Image(systemName: "film.stack")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(RapidTheme.textTertiary)
            Text(emptyStageTitle)
                .font(RapidFont.sectionTitle)
                .foregroundStyle(RapidTheme.textPrimary)
            Text(emptyStageMessage)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .padding(RapidTheme.Space.xl)
        .accessibilityIdentifier("Video.EmptyState")
    }

    private var emptyStageTitle: String {
        if !viewModel.catalogLoaded { return "Finding video models…" }
        if viewModel.videoModels.isEmpty { return "No supported video models" }
        if !viewModel.isSelectedModelEligible { return "This model doesn't fit this Mac" }
        if !viewModel.isServerReady { return "Start a video model" }
        return "Create your first video"
    }

    private var emptyStageMessage: String {
        if !viewModel.catalogLoaded { return "Rapid is reading the local model catalog." }
        if viewModel.videoModels.isEmpty {
            return "The signed engine does not currently advertise a compatible video model."
        }
        if !viewModel.isSelectedModelEligible {
            return viewModel.memoryRequirementText ?? "This model's memory requirement couldn't be verified."
        }
        if !viewModel.isServerReady {
            return "Starting is explicit so opening this tab never replaces your current model or allocates memory by surprise."
        }
        return "Describe a short scene below. The first generation is safest at the smallest size and duration."
    }

    @ViewBuilder
    private var readinessAction: some View {
        if viewModel.videoModels.isEmpty, viewModel.catalogLoaded {
            Button("Manage Models") {
                settingsRouter.route(to: .modelManagement) { openWindow(id: "settings") }
            }
            .buttonStyle(.rapidSecondary)
            .accessibilityIdentifier("Video.ManageModels")
        } else if let model = viewModel.selectedModel {
            if downloads.isDownloading(model.alias) {
                HStack(spacing: RapidTheme.Space.sm) {
                    ProgressView().controlSize(.small)
                    Button("Cancel Download") { downloads.cancelDownload(alias: model.alias) }
                        .buttonStyle(.rapidSecondary)
                        .accessibilityIdentifier("Video.CancelDownload")
                }
            } else if !model.cached {
                Button("Download \(model.alias)") {
                    _ = downloads.startDownload(alias: model.alias, hfPath: model.hfRepo)
                }
                .buttonStyle(.rapidPrimary)
                .disabled(!viewModel.isSelectedModelEligible)
                .accessibilityIdentifier("Video.DownloadModel")
            } else if !viewModel.isServerReady {
                Button {
                    Task { await viewModel.prepareSelectedModel() }
                } label: {
                    if viewModel.isPreparing {
                        HStack(spacing: RapidTheme.Space.sm) {
                            ProgressView().controlSize(.small)
                            Text("Starting…")
                        }
                    } else {
                        Label("Start Video Model", systemImage: "play.fill")
                    }
                }
                .buttonStyle(.rapidPrimary)
                .disabled(viewModel.isPreparing || !viewModel.isSelectedModelEligible)
                .accessibilityIdentifier("Video.StartModel")
            }
        }
    }

    private func jobStage(_ job: VideoJob) -> some View {
        VStack(spacing: RapidTheme.Space.md) {
            if job.status == .queued || job.status == .inProgress || viewModel.isLoadingPreview {
                ProgressView(value: Double(job.progress), total: 100)
                    .frame(maxWidth: 300)
            } else {
                Image(systemName: job.status == .failed ? "exclamationmark.triangle" : "film")
                    .font(.system(size: 34, weight: .light))
                    .foregroundStyle(job.status == .failed ? RapidTheme.statusError : RapidTheme.textTertiary)
            }
            Text(jobStatusTitle(job))
                .font(RapidFont.sectionTitle)
                .foregroundStyle(RapidTheme.textPrimary)
            Text(job.error?.message ?? job.prompt)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .frame(maxWidth: 480)
            if job.status == .queued {
                Button("Cancel Queued Video", role: .destructive) {
                    Task { await viewModel.delete(job) }
                }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("Video.Job.Cancel")
            } else if job.status == .inProgress {
                Text("Generation can't be interrupted safely once Metal work begins.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textTertiary)
            } else if job.status == .completed && viewModel.previewURL == nil {
                HStack {
                    Button("Load Preview") { Task { await viewModel.loadSelectedPreview() } }
                        .buttonStyle(.rapidSecondary)
                        .accessibilityIdentifier("Video.LoadPreview")
                    Button("Delete", role: .destructive) { pendingDeletion = job }
                        .buttonStyle(.rapidDestructive)
                        .accessibilityIdentifier("Video.Job.Delete")
                }
            } else if job.status == .failed {
                Button("Delete", role: .destructive) { pendingDeletion = job }
                    .buttonStyle(.rapidDestructive)
                    .accessibilityIdentifier("Video.Job.Delete")
            }
        }
        .padding(RapidTheme.Space.xl)
    }

    private func jobStatusTitle(_ job: VideoJob) -> String {
        switch job.status {
        case .queued: return "Waiting to generate"
        case .inProgress: return "Generating · \(job.progress)%"
        case .completed: return viewModel.isLoadingPreview ? "Loading preview…" : "Video ready"
        case .failed: return "Generation failed"
        }
    }

    private var resultActions: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Button { savePreview() } label: { Image(systemName: "square.and.arrow.down") }
                .buttonStyle(.rapidTertiary)
                .disabled(isSavingPreview)
                .help("Save video")
                .accessibilityLabel("Save video")
                .accessibilityIdentifier("Video.Job.Save")
            if let job = viewModel.selectedJob {
                Button(role: .destructive) { pendingDeletion = job } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.rapidTertiary)
                .help("Delete video")
                .accessibilityLabel("Delete video")
                .accessibilityIdentifier("Video.Job.Delete")
            }
        }
        .padding(RapidTheme.Space.md)
    }

    private var history: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            Text("Recent videos")
                .font(RapidFont.sectionTitle)
                .foregroundStyle(RapidTheme.textPrimary)
            ScrollView(.horizontal) {
                HStack(spacing: RapidTheme.Space.sm) {
                    ForEach(viewModel.jobs) { job in
                        Button {
                            Task { await viewModel.selectJob(job.id) }
                        } label: {
                            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                                HStack {
                                    Image(systemName: statusSymbol(job.status))
                                    Text(jobStatusLabel(job.status))
                                    Spacer()
                                    Text("\(job.seconds)s")
                                }
                                .font(RapidFont.groupLabel)
                                Text(job.prompt)
                                    .font(RapidFont.caption)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.leading)
                            }
                            .foregroundStyle(RapidTheme.textPrimary)
                            .padding(RapidTheme.Space.md)
                            .frame(width: 210, height: 82, alignment: .leading)
                            .background(
                                viewModel.selectedJobID == job.id
                                    ? RapidTheme.brandPrimaryTint : RapidTheme.surfaceRaised,
                                in: RoundedRectangle(cornerRadius: RapidTheme.Radius.row)
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("\(jobStatusLabel(job.status)): \(job.prompt)")
                        .accessibilityIdentifier("Video.History.\(job.id)")
                    }
                }
            }
            .scrollIndicators(.never)
            .accessibilityIdentifier("Video.History")
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            if let error = viewModel.errorMessage {
                HStack(spacing: RapidTheme.Space.sm) {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.statusError)
                        .accessibilityIdentifier("Video.Error")
                    if viewModel.needsServerRefresh {
                        Button("Retry") { Task { await viewModel.refreshServerData() } }
                            .buttonStyle(.rapidSecondary)
                            .disabled(viewModel.isRefreshing)
                            .accessibilityIdentifier("Video.RetryServerData")
                    }
                }
            }
            if viewModel.supportedModes.count > 1 {
                Picker("Source", selection: modeBinding) {
                    ForEach(viewModel.supportedModes) { mode in Text(mode.title).tag(mode) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)
                .accessibilityIdentifier("Video.ModePicker")
            }
            if viewModel.mode == .image { referenceControls }
            ZStack(alignment: .topLeading) {
                if viewModel.prompt.isEmpty {
                    Text("Describe the motion, subject, camera, and mood…")
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .padding(.horizontal, RapidTheme.Space.md)
                        .padding(.vertical, RapidTheme.Space.md + 1)
                        .accessibilityHidden(true)
                }
                TextEditor(text: $viewModel.prompt)
                    .font(RapidFont.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 54, idealHeight: 64, maxHeight: 72)
                    .padding(RapidTheme.Space.sm)
                    .accessibilityLabel("Video prompt")
                    .accessibilityIdentifier("Video.Prompt")
            }
            .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.input))
            .overlay {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: RapidTheme.Space.md) {
                    modelMenu
                    parameterPickers
                    Spacer()
                    composerAction
                }
                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    modelMenu
                    HStack(spacing: RapidTheme.Space.md) {
                        parameterPickers
                        Spacer()
                        composerAction
                    }
                }
            }
        }
        .padding(RapidTheme.Space.lg)
        .background(RapidTheme.surfaceCanvas)
        .overlay(alignment: .top) { Divider() }
        .accessibilityIdentifier("Video.Composer")
    }

    @ViewBuilder
    private var parameterPickers: some View {
        if !viewModel.sizePresets.isEmpty {
            Picker("Size", selection: sizeBinding) {
                ForEach(viewModel.sizePresets, id: \.self) {
                    Text($0.replacingOccurrences(of: "x", with: " × ")).tag($0)
                }
            }
            .labelsHidden()
            .accessibilityLabel("Video size")
            .accessibilityIdentifier("Video.Size")
        }
        if !viewModel.durationPresets.isEmpty {
            Picker("Duration", selection: $viewModel.seconds) {
                ForEach(viewModel.durationPresets, id: \.self) {
                    Text("\($0)s").tag($0)
                }
            }
            .labelsHidden()
            .accessibilityLabel("Video duration")
            .accessibilityIdentifier("Video.Duration")
        }
    }

    @ViewBuilder
    private var composerAction: some View {
        if viewModel.isServerReady {
            Button {
                Task { await viewModel.submit() }
            } label: {
                if viewModel.isSubmitting {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Generate", systemImage: "sparkles")
                }
            }
            .buttonStyle(.rapidPrimary)
            .disabled(!viewModel.canSubmit)
            .keyboardShortcut(.return, modifiers: [.command])
            .fixedSize(horizontal: true, vertical: false)
            .accessibilityIdentifier("Video.Generate")
        } else {
            readinessAction
                .fixedSize(horizontal: true, vertical: false)
        }
    }

    private var modelMenu: some View {
        Picker("Model", selection: modelBinding) {
            if viewModel.videoModels.isEmpty { Text("No video models").tag("") }
            ForEach(viewModel.videoModels) { model in
                Text(modelPickerLabel(model))
                    .tag(model.alias)
                    .disabled(!viewModel.isModelEligible(model))
            }
        }
        .labelsHidden()
        .frame(maxWidth: 220)
        .disabled(!viewModel.canSwitchModels)
        .accessibilityLabel("Video model")
        .accessibilityIdentifier("Video.ModelMenu")
    }

    private func modelPickerLabel(_ model: ModelEntry) -> String {
        guard let minimum = model.minimumMemoryGB else { return model.alias }
        return "\(model.alias) · \(Int(minimum.rounded())) GB"
    }

    private var referenceControls: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            if let reference = viewModel.referenceImage {
                Label(reference.fileName, systemImage: "photo.fill")
                    .font(RapidFont.caption)
                    .lineLimit(1)
                Button("Remove") { viewModel.setReference(nil) }
                    .buttonStyle(.plain)
                    .foregroundStyle(RapidTheme.statusError)
                    .accessibilityIdentifier("Video.Reference.Remove")
            } else {
                Button { showingReferenceImporter = true } label: {
                    Label("Add reference image", systemImage: "photo.badge.plus")
                }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("Video.Reference.Add")
                Text("JPEG, PNG, or WebP · up to \(referenceLimitText)")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textTertiary)
            }
        }
    }

    private var modelBinding: Binding<String> {
        Binding(get: { viewModel.selectedAlias }, set: { viewModel.selectModel($0) })
    }

    private var modeBinding: Binding<VideoGenViewModel.Mode> {
        Binding(get: { viewModel.mode }, set: { viewModel.selectMode($0) })
    }

    private var sizeBinding: Binding<String> {
        Binding(get: { viewModel.size }, set: { viewModel.selectSize($0) })
    }

    private var referenceLimitText: String {
        ByteCountFormatter.string(
            fromByteCount: Int64(viewModel.referenceMaximumBytes),
            countStyle: .file
        )
    }

    private var acceptedReferenceContentTypes: [UTType] {
        [
            ("image/jpeg", UTType.jpeg),
            ("image/png", UTType.png),
            ("image/webp", UTType.webP),
        ].compactMap { mime, type in
            viewModel.acceptedReferenceMIMETypes.contains(mime) ? type : nil
        }
    }

    private func importReference(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            let scoped = url.startAccessingSecurityScopedResource()
            Task {
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                await loadReference(url)
            }
        } catch {
            viewModel.errorMessage = "Rapid couldn't open that reference image."
        }
    }

    private func loadReference(_ url: URL) async {
        do {
            let maximumBytes = viewModel.referenceMaximumBytes
            let maximumPixels = viewModel.referenceMaximumPixels
            let acceptedMIMETypes = viewModel.acceptedReferenceMIMETypes
            let (data, mime) = try await Task.detached(priority: .userInitiated) {
                let data = try VideoReferenceLoader.load(
                    from: url,
                    maximumBytes: maximumBytes,
                    maximumPixels: maximumPixels
                )
                guard let mime = VideoReferenceLoader.mimeType(for: data) else {
                    throw VideoReferenceLoaderError.unsupportedFormat
                }
                return (data, mime)
            }.value
            guard maximumBytes == viewModel.referenceMaximumBytes,
                  maximumPixels == viewModel.referenceMaximumPixels,
                  acceptedMIMETypes == viewModel.acceptedReferenceMIMETypes else { return }
            guard acceptedMIMETypes.contains(mime) else {
                viewModel.errorMessage = "Choose a valid JPEG, PNG, or WebP image."
                return
            }
            viewModel.setReference(.init(data: data, fileName: url.lastPathComponent, mimeType: mime))
            viewModel.errorMessage = nil
        } catch VideoReferenceLoaderError.tooLarge {
            viewModel.errorMessage = "That reference image exceeds this model's size limit."
        } catch VideoReferenceLoaderError.unsupportedFormat {
            viewModel.errorMessage = "Choose a valid JPEG, PNG, or WebP image."
        } catch {
            viewModel.errorMessage = "Rapid couldn't open that reference image."
        }
    }

    private func savePreview() {
        guard let source = viewModel.previewURL else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = "\(viewModel.selectedJob?.id ?? "rapid-video").mp4"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        isSavingPreview = true
        Task {
            defer { isSavingPreview = false }
            do {
                try await Task.detached(priority: .userInitiated) {
                    try VideoPreviewSaver.save(source: source, destination: destination)
                }.value
            } catch {
                viewModel.errorMessage = "Rapid couldn't save the video to that location."
            }
        }
    }

    private func statusSymbol(_ status: VideoJobStatus) -> String {
        switch status {
        case .queued: return "clock"
        case .inProgress: return "sparkles"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private func jobStatusLabel(_ status: VideoJobStatus) -> String {
        switch status {
        case .queued: return "Queued"
        case .inProgress: return "Generating"
        case .completed: return "Ready"
        case .failed: return "Failed"
        }
    }
}

private struct VideoPlaybackView: View {
    let url: URL
    @State private var player: AVPlayer

    init(url: URL) {
        self.url = url
        _player = State(initialValue: AVPlayer(url: url))
    }

    var body: some View {
        VideoPlayer(player: player)
            .onDisappear { player.pause() }
            .accessibilityLabel("Generated video preview")
    }
}

private struct VideoDeletionSheet: View {
    let job: VideoJob
    let onKeep: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            Text(job.status == .queued ? "Cancel queued video?" : "Delete this video?")
                .font(RapidFont.sectionTitle)
            Text(deletionMessage)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
            HStack {
                Spacer()
                Button("Keep", action: onKeep)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("Video.Job.Delete.Keep")
                Button(job.status == .queued ? "Cancel Video" : "Delete", role: .destructive, action: onDelete)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("Video.Job.Delete.Confirm")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 420)
    }

    private var deletionMessage: String {
        switch job.status {
        case .queued:
            return "The queued request will be removed before generation begins."
        case .failed:
            return "This failed request will be removed from recent videos."
        case .completed:
            return "The generated file will be removed from this Mac. This can't be undone."
        case .inProgress:
            return "Generation can't be deleted while it is in progress."
        }
    }
}
