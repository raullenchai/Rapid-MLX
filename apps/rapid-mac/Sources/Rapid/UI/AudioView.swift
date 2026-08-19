import AppKit
import AVFoundation
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// Audio workflows backed by the local OpenAI-compatible routes. The page
/// deliberately starts no model on appearance: with today's single-model
/// runtime, a model is swapped only when the user transcribes, loads voices,
/// or synthesizes speech.
struct AudioView: View {
    @Bindable var viewModel: AudioViewModel
    @Bindable var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @Environment(SettingsRouter.self) private var settingsRouter
    @Environment(DownloadManager.self) private var downloads
    @Environment(DictationController.self) private var dictation

    @State private var copied = false
    @State private var playback = AudioPlaybackController()
    @State private var showVoicePicker = false
    @State private var playingPreviewVoice: String?
    @State private var voicePreviewTask: Task<Void, Never>?
    @State private var voicePreviewRequestID: UUID?
    @State private var modelLoadsInFlight: Set<String> = []

    private let contentMaxWidth = RapidTheme.Layout.contentMaxWidth
    private let controlLabelWidth: CGFloat = 80
    private let controlFieldWidth: CGFloat = 320

    private var selectedAlias: String {
        viewModel.mode == .speech
            ? viewModel.selectedSpeechAlias
            : viewModel.selectedTranscriptionAlias
    }

    private var selectedEntry: ModelEntry? {
        viewModel.audioModels.first { $0.alias == selectedAlias }
    }

    /// Audio uses the same lifecycle SSOT and CTA semantics as Chat and
    /// Images: choose → Download & start / Start → ready.
    private var readiness: ModelReadiness {
        // Audio-only `serve` processes intentionally report healthy before
        // loading their lazy STT/TTS engine. For an uncached model that
        // process-level signal is not readiness: the first audio request would
        // still begin the weight download. The explicit DownloadManager job is
        // authoritative until the catalog confirms the weights are on disk.
        if let selectedEntry,
           let downloadReadiness = Self.audioDownloadReadiness(
               alias: selectedAlias,
               cached: selectedEntry.cached,
               sizeText: selectedEntry.sizeOnDisk,
               job: downloads.job(for: selectedAlias),
               activationInFlight: modelLoadsInFlight.contains(selectedAlias)
           ) {
            return downloadReadiness
        }
        if server.isResidentLoadInFlight(selectedAlias) {
            return .starting(alias: selectedAlias, detail: "Downloading or loading the audio model…")
        }
        let cacheState: ModelReadiness.CacheState
        if selectedAlias.isEmpty || !viewModel.catalogLoaded {
            cacheState = .catalogPending
        } else if let selectedEntry {
            cacheState = selectedEntry.cached ? .onDisk : .notOnDisk
        } else {
            cacheState = .notInCatalog
        }
        let progress: ModelReadiness.ProgressSnapshot? = if case .starting = server.state {
            .init(
                activity: server.downloadProgress.startupActivity,
                subtitle: server.downloadProgress.progressSubtitle,
                fraction: server.downloadProgress.progressFraction
            )
        } else { nil }
        return ModelReadiness.resolve(
            serverState: server.readinessState(for: selectedAlias),
            alias: selectedAlias,
            cacheState: cacheState,
            sizeText: selectedEntry?.sizeOnDisk,
            progress: progress,
            failure: server.residentLoadFailure(for: selectedAlias).map {
                .init(message: $0.message, alias: $0.alias)
            },
            downloadInFlight: downloads.isDownloading(selectedAlias)
        )
    }

    @MainActor
    static func audioDownloadReadiness(
        alias: String,
        cached: Bool,
        sizeText: String?,
        job: DownloadManager.Job?,
        activationInFlight: Bool
    ) -> ModelReadiness? {
        guard !alias.isEmpty, !cached else { return nil }
        if let job {
            switch job.status {
            case .running:
                return .downloading(
                    alias: alias,
                    detail: job.progress.progressSubtitle,
                    fraction: job.progress.progressFraction
                )
            case .failed(let message):
                return .failed(alias: alias, message: message, action: .retry(alias: alias))
            case .completed:
                if activationInFlight {
                    return .starting(alias: alias, detail: "Finishing the download…")
                }
            case .cancelled:
                break
            }
        }
        if activationInFlight {
            return .downloading(alias: alias, detail: "Starting the download…", fraction: nil)
        }
        return .needsDownload(alias: alias, sizeText: sizeText)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(RapidTheme.hairline)
            content
        }
        .background(RapidTheme.surfaceCanvas)
        // Settings is a separate window, so this view can remain mounted
        // while an audio pull finishes. Refresh on the shared disk-cache
        // generation instead of keeping the pre-download catalog snapshot.
        .task(id: downloads.cacheGeneration) {
            await viewModel.refreshCatalog()
        }
        .onChange(of: viewModel.mode) { _, _ in cancelVoicePreview() }
        .onDisappear { cancelVoicePreview() }
    }

    private var header: some View {
        HStack(spacing: RapidTheme.Space.lg) {
            Spacer(minLength: 0)
            // The one segmented treatment, shared with Settings. Named
            // explicitly in the UI-1 review as one of the oversized
            // controls: at `.pickerStyle(.segmented)` the selected
            // segment was amber with WHITE text. This is a component
            // swap only — the binding, the modes, and everything around
            // this control are untouched.
            RapidSegmentedControl(
                selection: $viewModel.mode,
                options: AudioViewModel.Mode.allCases.map {
                    .init(
                        value: $0,
                        title: $0.label,
                        identifier: "Audio.Mode.\($0.axName)"
                    )
                },
                accessibilityLabel: "Audio mode"
            )
            .accessibilityIdentifier("Audio.Mode")
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.vertical, RapidTheme.Space.lg)
    }

    @ViewBuilder
    private var content: some View {
        if !viewModel.catalogLoaded {
            ProgressView("Loading audio models...")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            switch viewModel.mode {
            case .dictation:
                if viewModel.transcriptionModels.isEmpty {
                    unavailableState(operation: "dictation")
                } else {
                    DictationView(
                        controller: dictation,
                        viewModel: viewModel,
                        server: server
                    )
                }
            case .transcription:
                if viewModel.transcriptionModels.isEmpty {
                    unavailableState(operation: "transcription")
                } else {
                    transcriptionSurface
                }
            case .speech:
                if viewModel.speechModels.isEmpty {
                    unavailableState(operation: "speech")
                } else {
                    speechSurface
                }
            }
        }
    }

    private func unavailableState(operation: String) -> some View {
        EmptyState(
            symbol: "waveform",
            title: "Audio unavailable",
            message: server.binaryPath == nil
                ? "The bundled engine could not be found. The rest of Rapid-MLX remains available."
                : "No model in this build supports \(operation). Audio models can be managed in Settings."
        ) {
            Button("Open Model Management", systemImage: "square.stack.3d.up") {
                openModelManagement()
            }
            .accessibilityIdentifier("Audio.EmptyState.OpenModelManagement")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityIdentifier("Audio.EmptyState")
    }

    private var transcriptionSurface: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                SectionHeader(
                    "Transcription",
                    subtitle: "Turn a local recording into text without uploading it."
                )
                filePicker
                modelPicker(
                    title: "Model",
                    selection: $viewModel.selectedTranscriptionAlias,
                    entries: viewModel.transcriptionModels,
                    identifier: "Audio.Transcription.ModelPicker"
                )
                ReadinessBanner(readiness: readiness, onAction: handleReadinessAction)
                swapNotice(alias: viewModel.selectedTranscriptionAlias)
                operationNotice
                HStack(spacing: RapidTheme.Space.md) {
                    if viewModel.isTranscribing {
                        ProgressView()
                            .controlSize(.small)
                        Text("Loading model and transcribing...")
                            .font(RapidFont.secondary)
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: RapidTheme.Space.md)
                    Button("Transcribe", systemImage: "text.badge.checkmark") {
                        playback.stop()
                        Task { await viewModel.transcribe() }
                    }
                    .buttonStyle(.rapidPrimary)
                    .disabled(
                        viewModel.selectedFileURL == nil
                            || viewModel.selectedTranscriptionAlias.isEmpty
                            || !readiness.sendAllowed
                            || viewModel.isBusy
                    )
                    .accessibilityIdentifier("Audio.Transcription.Run")
                }

                if let result = viewModel.transcription {
                    transcriptionResult(result)
                }
            }
            .frame(maxWidth: contentMaxWidth, alignment: .leading)
            .frame(maxWidth: .infinity)
            .padding(RapidTheme.Space.xl)
        }
    }

    private var filePicker: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(spacing: RapidTheme.Space.md) {
                Image(systemName: "waveform")
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                    .frame(width: 34)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                    Text(viewModel.selectedFileURL?.lastPathComponent ?? "Audio file")
                        .font(RapidFont.body)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(fileCaption)
                        .font(RapidFont.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: RapidTheme.Space.md)
                Button("Choose File", systemImage: "folder") { chooseAudioFile() }
                    .buttonStyle(.rapidSecondaryCompactUtility)
                    .accessibilityIdentifier("Audio.Transcription.FilePicker")
            }
            .padding(RapidTheme.Space.lg)
            .frame(maxWidth: .infinity, minHeight: 84)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .strokeBorder(RapidTheme.hairlineStrong, style: StrokeStyle(lineWidth: 1, dash: [5]))
            )
            .contentShape(Rectangle())
            .dropDestination(for: URL.self) { urls, _ in
                guard let url = urls.first(where: isAudioFile) else { return false }
                viewModel.selectFile(url)
                return true
            }
        }
    }

    private var fileCaption: String {
        guard let url = viewModel.selectedFileURL else {
            return "WAV, MP3, M4A, AAC, FLAC, MP4, AIFF, or CAF - up to 25 MB"
        }
        if let bytes = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
        }
        return url.pathExtension.uppercased()
    }

    private func transcriptionResult(_ result: AudioTranscriptionResult) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Result") {
                HStack(spacing: RapidTheme.Space.xs) {
                    QuietIconButton(
                        symbol: copied ? "checkmark" : "doc.on.doc",
                        label: copied ? "Copied" : "Copy transcription",
                        tint: copied ? RapidTheme.statusReady : nil
                    ) {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(result.text, forType: .string)
                        copied = true
                        Task {
                            try? await Task.sleep(for: .seconds(1.5))
                            copied = false
                        }
                    }
                    .accessibilityIdentifier("Audio.Transcription.Copy")
                    QuietIconButton(
                        symbol: "square.and.arrow.down",
                        label: "Save transcription"
                    ) { saveTranscription(result.text) }
                    .accessibilityIdentifier("Audio.Transcription.Save")
                }
            }

            ScrollView {
                Text(result.text)
                    .font(RapidFont.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(RapidTheme.Space.lg)
            }
            .frame(minHeight: 130, maxHeight: 300)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .strokeBorder(RapidTheme.hairline, lineWidth: 1)
            )
            .accessibilityIdentifier("Audio.Transcription.Result")

            if result.language != nil || result.duration != nil {
                Text(resultMetadata(result))
                    .font(RapidFont.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var speechSurface: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                SectionHeader(
                    "Speech",
                    subtitle: "Create spoken audio with a model and one of its built-in voices."
                )

                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    SectionHeader("Text")
                    TextEditor(text: $viewModel.speechText)
                        .font(RapidFont.body)
                        .scrollContentBackground(.hidden)
                        .padding(RapidTheme.Space.sm)
                        .frame(minHeight: 150)
                        .background(
                            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                                .fill(RapidTheme.surfaceRaised)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
                        )
                        .accessibilityLabel("Text to speak")
                        .accessibilityIdentifier("Audio.Speech.Text")
                }

                speechControls
                ReadinessBanner(readiness: readiness, onAction: handleReadinessAction)
                swapNotice(alias: viewModel.selectedSpeechAlias)
                operationNotice

                HStack(spacing: RapidTheme.Space.md) {
                    if viewModel.isSynthesizing {
                        ProgressView().controlSize(.small)
                        Text("Generating audio...")
                            .font(RapidFont.secondary)
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: RapidTheme.Space.md)
                    Button("Generate Speech", systemImage: "waveform.badge.plus") {
                        playback.stop()
                        Task { await viewModel.synthesize() }
                    }
                    .buttonStyle(.rapidPrimary)
                    .disabled(
                        viewModel.speechText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || viewModel.selectedSpeechAlias.isEmpty
                            || !readiness.sendAllowed
                            || viewModel.isBusy
                    )
                    .accessibilityIdentifier("Audio.Speech.Generate")
                }

                if let audio = viewModel.synthesizedAudio {
                    speechResult(audio)
                }
            }
            .frame(maxWidth: contentMaxWidth, alignment: .leading)
            .frame(maxWidth: .infinity)
            .padding(RapidTheme.Space.xl)
        }
    }

    private var speechModelSelection: Binding<String> {
        Binding(
            get: { viewModel.selectedSpeechAlias },
            set: { viewModel.selectSpeechModel($0) }
        )
    }

    private var speechControls: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: RapidTheme.Space.md) {
                controlLabel("Model")
                modelPickerControl(
                    title: "Model",
                    selection: speechModelSelection,
                    entries: viewModel.speechModels,
                    identifier: "Audio.Speech.ModelPicker"
                )
                Spacer(minLength: RapidTheme.Space.md)
            }

            HStack(spacing: RapidTheme.Space.md) {
                controlLabel("Voice")
                voicePickerControl
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Load Voices", systemImage: "arrow.clockwise") {
                        playback.stop()
                        Task { _ = await viewModel.loadVoices() }
                    }
                    .buttonStyle(.rapidSecondaryCompactUtility)
                    .disabled(
                        viewModel.selectedSpeechAlias.isEmpty
                            || !readiness.sendAllowed
                            || viewModel.isBusy
                    )
                    .accessibilityIdentifier("Audio.Speech.LoadVoices")
                    if viewModel.isLoadingVoices {
                        ProgressView().controlSize(.small)
                    }
                }
                Spacer(minLength: RapidTheme.Space.md)
            }

            HStack(spacing: RapidTheme.Space.md) {
                controlLabel("Speed")
                HStack(spacing: RapidTheme.Space.md) {
                    Slider(value: $viewModel.speed, in: 0.5...2, step: 0.05)
                        .accessibilityIdentifier("Audio.Speech.Speed")
                    Text(viewModel.speed.formatted(.number.precision(.fractionLength(2))) + "x")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .monospacedDigit()
                        .frame(width: 48, alignment: .trailing)
                }
                .frame(width: controlFieldWidth)
                .accessibilityElement(children: .contain)
                Spacer(minLength: RapidTheme.Space.md)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func speechResult(_ audio: SynthesizedAudio) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 28))
                .foregroundStyle(RapidTheme.brandPrimaryDeep)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Speech ready")
                    .font(RapidFont.body)
                Text(ByteCountFormatter.string(fromByteCount: Int64(audio.data.count), countStyle: .file))
                    .font(RapidFont.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: RapidTheme.Space.md)
            QuietIconButton(
                symbol: playback.isPlaying ? "stop.fill" : "play.fill",
                label: playback.isPlaying ? "Stop playback" : "Play speech"
            ) {
                do {
                    try playback.toggle(audio.data)
                } catch {
                    viewModel.errorMessage = "Couldn't play the audio: \(error.localizedDescription)"
                }
            }
            .accessibilityIdentifier("Audio.Speech.Play")
            QuietIconButton(
                symbol: "square.and.arrow.down",
                label: "Save speech"
            ) { saveSpeech(audio) }
            .accessibilityIdentifier("Audio.Speech.Save")
        }
        .padding(RapidTheme.Space.lg)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .fill(RapidTheme.surfaceRaised)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .strokeBorder(RapidTheme.hairline, lineWidth: 1)
        )
    }

    private func modelPicker(
        title: String,
        selection: Binding<String>,
        entries: [ModelEntry],
        identifier: String
    ) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            controlLabel(title)
            modelPickerControl(
                title: title,
                selection: selection,
                entries: entries,
                identifier: identifier
            )
            Spacer(minLength: 0)
        }
    }

    private func modelPickerControl(
        title: String,
        selection: Binding<String>,
        entries: [ModelEntry],
        identifier: String
    ) -> some View {
        Menu {
            ForEach(entries) { entry in
                Button {
                    selection.wrappedValue = entry.alias
                } label: {
                    Label(
                        entry.alias,
                        systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
                    )
                }
                .accessibilityIdentifier("\(identifier).\(entry.alias)")
                .accessibilityLabel(
                    "\(entry.alias), \(entry.cached ? "Downloaded" : "Not downloaded")"
                )
                .accessibilityAddTraits(
                    selection.wrappedValue == entry.alias ? .isSelected : []
                )
            }
        } label: {
            popupControlLabel(
                entries.first(where: { $0.alias == selection.wrappedValue })
                    .map(\.alias) ?? "Choose a model"
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .disabled(viewModel.isBusy)
        .accessibilityLabel(title)
        .accessibilityValue(selection.wrappedValue)
        .accessibilityIdentifier(identifier)
    }

    private var voicePickerControl: some View {
        Button {
            showVoicePicker.toggle()
        } label: {
            popupControlLabel(
                viewModel.selectedVoice.isEmpty ? "No voices loaded" : viewModel.selectedVoice
            )
        }
        .buttonStyle(.plain)
        .disabled(
            viewModel.voices.isEmpty
                || (viewModel.isBusy && viewModel.previewingVoice == nil)
        )
        .accessibilityLabel("Voice")
        .accessibilityValue(viewModel.selectedVoice)
        .accessibilityIdentifier("Audio.Speech.VoicePicker")
        .popover(isPresented: $showVoicePicker, arrowEdge: .top) {
            ScrollView {
                LazyVStack(spacing: RapidTheme.Space.xs) {
                    ForEach(viewModel.voices, id: \.self) { voice in
                        VoiceOptionRow(
                            voice: voice,
                            details: AudioViewModel.voiceDetails(for: voice),
                            isSelected: viewModel.selectedVoice == voice,
                            isPreviewing: viewModel.previewingVoice == voice,
                            isPlaying: playback.isPlaying && playingPreviewVoice == voice,
                            isEnabled: !viewModel.isBusy,
                            select: {
                                cancelVoicePreview()
                                viewModel.selectedVoice = voice
                                showVoicePicker = false
                            },
                            preview: { toggleVoicePreview(voice) }
                        )
                    }
                }
                .padding(RapidTheme.Space.xs)
            }
            .frame(width: controlFieldWidth, height: voicePopoverHeight)
        }
    }

    private var voicePopoverHeight: CGFloat {
        min(max(CGFloat(viewModel.voices.count) * 34 + RapidTheme.Space.sm, 42), 320)
    }

    private func popupControlLabel(_ value: String) -> some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(value)
                .font(RapidFont.body)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: RapidTheme.Space.sm)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .frame(width: controlFieldWidth, height: RapidTheme.ControlHeight.small)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                .fill(RapidTheme.surfaceCode)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous))
    }

    private func controlLabel(_ title: String) -> some View {
        Text(title)
            .font(RapidFont.secondary)
            .frame(width: controlLabelWidth, alignment: .leading)
    }

    private func handleReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .chooseModel:
            break
        case .download(let alias):
            // Download-only: fetch the weights, don't load. The banner flips
            // to "Start" once cached (see ModelReadiness two-step).
            guard let entry = viewModel.audioModels.first(where: { $0.alias == alias }),
                  !downloads.isDownloading(alias) else { break }
            _ = downloads.startDownload(
                alias: alias,
                hfPath: entry.hfRepo,
                totalBytes: ModelCacheActions.parseSizeBytes(entry.sizeOnDisk)
            )
        case .start(let alias), .retry(let alias):
            Task { await loadAudioModel(alias) }
        case .restart(let alias):
            Task {
                await server.stop()
                await loadAudioModel(alias)
            }
        case .openModelManagement:
            openModelManagement()
        }
    }

    private func loadAudioModel(_ alias: String) async {
        guard !modelLoadsInFlight.contains(alias),
              let initialEntry = viewModel.audioModels.first(where: { $0.alias == alias }) else {
            return
        }
        modelLoadsInFlight.insert(alias)
        defer { modelLoadsInFlight.remove(alias) }
        viewModel.errorMessage = nil

        // `Start` may have been rendered from a catalog snapshot taken before
        // an interrupted pull left only some numbered weight shards behind.
        // Re-probe before trusting cached=true; the engine's cache listing
        // validates that every shard is present and turns a partial back into
        // Download & start.
        if initialEntry.cached {
            await viewModel.refreshCatalog()
            guard !Task.isCancelled else { return }
        }
        guard let currentEntry = viewModel.audioModels.first(where: { $0.alias == alias }) else {
            return
        }

        if !currentEntry.cached {
            // A completed job may have landed while this view's catalog
            // snapshot was stale. Refresh before deciding to pull it again.
            if downloads.job(for: alias)?.status == .completed {
                await viewModel.refreshCatalog()
            }

            if viewModel.audioModels.first(where: { $0.alias == alias })?.cached != true {
                if let job = downloads.job(for: alias), job.status != .running {
                    downloads.dismissJob(alias: alias)
                }
                if !downloads.isDownloading(alias) {
                    _ = downloads.startDownload(
                        alias: alias,
                        hfPath: currentEntry.hfRepo,
                        totalBytes: ModelCacheActions.parseSizeBytes(currentEntry.sizeOnDisk)
                    )
                }
                await downloads.awaitDownloadSettlement(alias: alias)
                guard !Task.isCancelled else { return }
                guard downloads.job(for: alias)?.status == .completed else { return }
                await viewModel.refreshCatalog()
            }

            // `rapid-mlx pull` exiting successfully is necessary, but the
            // catalog is the final proof that the concrete HF snapshot is
            // usable. Never turn the audio server's lazy health response into
            // a false Ready state when that proof is absent.
            guard viewModel.audioModels.first(where: { $0.alias == alias })?.cached == true else {
                viewModel.errorMessage = "The download finished, but Rapid couldn't find the model on disk. Try downloading it again."
                return
            }
        }

        // A download may finish after the user selects a different audio
        // model. Keep the completed cache, but do not start the stale choice.
        guard selectedAlias == alias else { return }
        let entry = viewModel.audioModels.first { $0.alias == alias }
        _ = await server.ensureServing(
            alias: alias,
            hfPath: entry?.hfRepo,
            estimatedMemoryGB: ModelSizing.residentEstimateGB(
                alias: alias,
                sizeText: entry?.sizeOnDisk
            ),
            residencyEligible: false
        )
        await viewModel.refreshCatalog()
    }

    @ViewBuilder
    private func swapNotice(alias: String) -> some View {
        if let running = viewModel.wouldReplaceServingModel(alias: alias) {
            InlineNotice(
                message: "Starting \(alias) will stop \(running). This version runs one model at a time.",
                tone: .info
            )
        }
    }

    @ViewBuilder
    private var operationNotice: some View {
        if let message = viewModel.errorMessage {
            InlineNotice(message: message, tone: .error)
        }
    }

    private func chooseAudioFile() {
        if ProcessInfo.processInfo.environment["RAPID_GUI_GOLDEN_MODE"] == "1",
           let simulated = ProcessInfo.processInfo.environment["RAPID_SIMULATED_AUDIO_PATH"],
           !simulated.isEmpty
        {
            viewModel.selectFile(URL(fileURLWithPath: simulated))
            return
        }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        viewModel.selectFile(url)
    }

    private func isAudioFile(_ url: URL) -> Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .audio) == true
    }

    private func saveTranscription(_ text: String) {
        if ProcessInfo.processInfo.environment["RAPID_GUI_GOLDEN_MODE"] == "1",
           let simulated = ProcessInfo.processInfo.environment["RAPID_SIMULATED_TRANSCRIPTION_SAVE_PATH"],
           !simulated.isEmpty
        {
            do {
                try text.write(to: URL(fileURLWithPath: simulated), atomically: true, encoding: .utf8)
            } catch {
                viewModel.errorMessage = "Couldn't save the transcription: \(error.localizedDescription)"
            }
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "transcription.txt"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            viewModel.errorMessage = "Couldn't save the transcription: \(error.localizedDescription)"
        }
    }

    private func saveSpeech(_ audio: SynthesizedAudio) {
        if ProcessInfo.processInfo.environment["RAPID_GUI_GOLDEN_MODE"] == "1",
           let simulated = ProcessInfo.processInfo.environment["RAPID_SIMULATED_SPEECH_SAVE_PATH"],
           !simulated.isEmpty
        {
            do {
                try audio.data.write(to: URL(fileURLWithPath: simulated), options: .atomic)
            } catch {
                viewModel.errorMessage = "Couldn't save the audio: \(error.localizedDescription)"
            }
            return
        }
        let panel = NSSavePanel()
        if let type = UTType(filenameExtension: audio.fileExtension) {
            panel.allowedContentTypes = [type]
        }
        panel.nameFieldStringValue = "rapid-speech.\(audio.fileExtension)"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try audio.data.write(to: url, options: .atomic)
        } catch {
            viewModel.errorMessage = "Couldn't save the audio: \(error.localizedDescription)"
        }
    }

    private func resultMetadata(_ result: AudioTranscriptionResult) -> String {
        var parts: [String] = []
        if let language = result.language, !language.isEmpty {
            parts.append("Language: \(language)")
        }
        if let duration = result.duration {
            parts.append("Duration: \(duration.formatted(.number.precision(.fractionLength(1)))) s")
        }
        return parts.joined(separator: "  |  ")
    }

    private func openModelManagement() {
        settingsRouter.route(.openModelManagement) { openWindow(id: "settings") }
    }

    private func toggleVoicePreview(_ voice: String) {
        if playback.isPlaying, playingPreviewVoice == voice {
            playback.stop()
            playingPreviewVoice = nil
            return
        }

        voicePreviewTask?.cancel()
        playback.stop()
        playingPreviewVoice = nil

        let requestID = UUID()
        voicePreviewRequestID = requestID
        voicePreviewTask = Task {
            defer {
                if voicePreviewRequestID == requestID {
                    voicePreviewTask = nil
                    voicePreviewRequestID = nil
                }
            }
            guard let audio = await viewModel.previewVoice(voice),
                  !Task.isCancelled,
                  voicePreviewRequestID == requestID else { return }
            do {
                try playback.play(audio.data)
                playingPreviewVoice = voice
            } catch {
                viewModel.errorMessage = "Couldn't play the voice preview: \(error.localizedDescription)"
            }
        }
    }

    private func cancelVoicePreview() {
        voicePreviewTask?.cancel()
        voicePreviewTask = nil
        voicePreviewRequestID = nil
        playback.stop()
        playingPreviewVoice = nil
    }
}

private struct VoiceOptionRow: View {
    let voice: String
    let details: String
    let isSelected: Bool
    let isPreviewing: Bool
    let isPlaying: Bool
    let isEnabled: Bool
    let select: () -> Void
    let preview: () -> Void

    @State private var hovering = false

    var body: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            Button(action: select) {
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .semibold))
                        .opacity(isSelected ? 1 : 0)
                        .frame(width: 14)
                    Text(voice)
                        .font(RapidFont.body)
                        .lineLimit(1)
                    Text(details)
                        .font(RapidFont.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer(minLength: RapidTheme.Space.sm)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!isEnabled)
            .accessibilityLabel("Select \(voice), \(details)")
            .accessibilityIdentifier("Audio.Speech.VoiceOption.\(voice)")

            if isPreviewing {
                ProgressView()
                    .controlSize(.small)
                    .frame(
                        width: RapidTheme.ControlHeight.small,
                        height: RapidTheme.ControlHeight.small
                    )
                    .accessibilityLabel("Generating \(voice) preview")
            } else {
                QuietIconButton(
                    symbol: isPlaying ? "stop.circle.fill" : "play.circle.fill",
                    label: isPlaying ? "Stop \(voice) preview" : "Preview \(voice)",
                    symbolSize: 16,
                    action: preview
                )
                .disabled(!isEnabled && !isPlaying)
                .accessibilityIdentifier("Audio.Speech.PreviewVoice.\(voice)")
            }
        }
        .padding(.leading, RapidTheme.Space.sm)
        .padding(.trailing, RapidTheme.Space.xs)
        .frame(height: 30)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                .fill(isSelected || hovering ? RapidTheme.hoverFill : .clear)
        )
        .contentShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous))
        .onHover { hovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: hovering)
    }
}

@MainActor
@Observable
private final class AudioPlaybackController {
    private var player: AVAudioPlayer?
    private var monitor: Task<Void, Never>?
    private(set) var isPlaying = false

    func toggle(_ data: Data) throws {
        if isPlaying {
            stop()
            return
        }
        try play(data)
    }

    func play(_ data: Data) throws {
        stop()
        let player = try AVAudioPlayer(data: data)
        guard player.prepareToPlay(), player.play() else {
            throw AudioPlaybackError.couldNotStart
        }
        self.player = player
        isPlaying = true
        monitor = Task { [weak self, weak player] in
            while !Task.isCancelled, player?.isPlaying == true {
                try? await Task.sleep(for: .milliseconds(100))
            }
            guard !Task.isCancelled else { return }
            self?.isPlaying = false
        }
    }

    func stop() {
        monitor?.cancel()
        monitor = nil
        player?.stop()
        player = nil
        isPlaying = false
    }
}

private enum AudioPlaybackError: LocalizedError {
    case couldNotStart

    var errorDescription: String? { "Playback could not start." }
}
