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

    @State private var playback = AudioPlaybackController()
    @State private var showVoicePicker = false
    @State private var playingPreviewVoice: String?
    @State private var voicePreviewTask: Task<Void, Never>?
    @State private var voicePreviewRequestID: UUID?
    @State private var modelLoadsInFlight: Set<String> = []

    private let contentMaxWidth = RapidTheme.Layout.contentMaxWidth
    /// One control width across the Audio tabs — same as the Dictation
    /// setup rows, so the trailing line holds from tab to tab.
    private let controlFieldWidth: CGFloat = 260
    /// The voice popover keeps its own, wider measure: its rows carry a
    /// name, a detail line, and a preview button.
    private let voicePopoverWidth: CGFloat = 320

    /// The Text to Speech tab's model. Speech to Text owns its own
    /// selection and readiness inside ``DictationView``.
    private var selectedAlias: String {
        viewModel.selectedSpeechAlias
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

    private var speechSurface: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                SectionHeader(
                    "Text to Speech",
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

                speechSetupCard
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
        // Voices need the model process, so fetch them the moment it turns
        // ready: picking a voice is the user's job, loading the list is ours.
        // The row's refresh button stays for re-fetching after a hiccup.
        .task(id: "\(selectedAlias)#\(readiness.sendAllowed)") {
            guard readiness.sendAllowed, viewModel.voices.isEmpty, !viewModel.isBusy else { return }
            _ = await viewModel.loadVoices()
        }
    }

    private var speechModelSelection: Binding<String> {
        Binding(
            get: { viewModel.selectedSpeechAlias },
            set: { viewModel.selectSpeechModel($0) }
        )
    }

    /// The same setup-card grammar as Speech to Text: label and caption on
    /// the left, the control on the shared trailing line, the readiness
    /// banner inset directly under the Model row it describes. One design
    /// language across the Audio tabs — nothing on this page invents its own
    /// alignment.
    private var speechSetupCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            speechRow(
                label: "Model",
                caption: "Voices and generated audio come from this model."
            ) {
                modelPickerControl(
                    title: "Model",
                    selection: speechModelSelection,
                    entries: viewModel.speechModels,
                    identifier: "Audio.Speech.ModelPicker"
                )
            }
            // Hidden once the model is ready: in steady state the card is
            // settings, not status.
            if !readiness.sendAllowed {
                ReadinessBanner(readiness: readiness, onAction: handleReadinessAction)
                    // xs outside + the banner's own md inside = lg: text on
                    // the rows' content line, action on their trailing line.
                    .padding(.horizontal, RapidTheme.Space.xs)
                    .padding(.bottom, RapidTheme.Space.lg)
            }

            Divider().overlay(RapidTheme.hairline)

            speechRow(
                label: "Voice",
                caption: "Each model ships its own set — preview before choosing."
            ) {
                HStack(spacing: RapidTheme.Space.sm) {
                    if viewModel.isLoadingVoices {
                        ProgressView().controlSize(.small)
                    }
                    QuietIconButton(
                        symbol: "arrow.clockwise",
                        label: "Load voices",
                        help: "Reload the voice list from the running model."
                    ) {
                        playback.stop()
                        Task { _ = await viewModel.loadVoices() }
                    }
                    .disabled(
                        viewModel.selectedSpeechAlias.isEmpty
                            || !readiness.sendAllowed
                            || viewModel.isBusy
                    )
                    .accessibilityIdentifier("Audio.Speech.LoadVoices")
                    voicePickerControl
                }
            }

            Divider().overlay(RapidTheme.hairline)

            speechRow(
                label: "Speed",
                caption: "Applied when the audio is generated."
            ) {
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
            }
        }
        .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                .strokeBorder(RapidTheme.hairline)
        )
    }

    /// ``DictationView/setupRow`` minus the readiness circle: these are
    /// settings, not checklist steps, but the typography and metrics match
    /// so the two tabs read as one surface.
    private func speechRow<Control: View>(
        label: String,
        caption: String,
        @ViewBuilder control: () -> Control
    ) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(label).font(.subheadline.weight(.medium))
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.md)
            control()
        }
        .padding(RapidTheme.Space.lg)
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
                viewModel.selectedVoice.isEmpty ? voicePlaceholder : viewModel.selectedVoice
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
            .frame(width: voicePopoverWidth, height: voicePopoverHeight)
        }
    }

    /// State description for an empty voice popup — what the control is
    /// waiting on, not internal mechanics.
    private var voicePlaceholder: String {
        if viewModel.isLoadingVoices { return "Loading voices…" }
        guard readiness.sendAllowed else { return "Start the model to list voices" }
        return "Choose a voice"
    }

    private var voicePopoverHeight: CGFloat {
        min(max(CGFloat(viewModel.voices.count) * 34 + RapidTheme.Space.sm, 42), 320)
    }

    private func popupControlLabel(_ value: String) -> some View {
        PopupControlChrome(title: value, width: controlFieldWidth)
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
