import AppKit
import SwiftUI

/// Configuration surface for dictation.
///
/// Unlike Speech and Transcription this page is not where the feature is used —
/// dictation happens in whatever app the user is typing in. What lives here is
/// setup, the vocabulary that keeps proper nouns right, and the history that
/// turns mistakes into vocabulary.
struct DictationView: View {
    @Bindable var controller: DictationController
    @Bindable var viewModel: AudioViewModel
    @Bindable var server: ServerManager
    @Environment(DownloadManager.self) private var downloads

    @State private var newTerm = ""
    @State private var fixTarget: DictationHistory.Entry?
    @State private var showModelPicker = false
    @State private var handledCompletedDownloads: Set<String> = []
    @State private var notifiedSelectedCompletion: String?

    private let contentMaxWidth = RapidTheme.Layout.contentMaxWidth

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                intro
                // One layout in every state. Swapping between a "setup" card
                // and a "ready" banner hid the model and hotkey the moment
                // dictation was switched on — changing either meant turning the
                // whole feature off first.
                statusCard
                errorRow
                vocabularySection
                historySection
            }
            .frame(maxWidth: contentMaxWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(RapidTheme.Space.xl)
        }
        .task {
            controller.refreshReadiness()
            controller.revalidate()
            if controller.modelAlias.isEmpty {
                controller.modelAlias = viewModel.selectedTranscriptionAlias
            } else {
                // The alias setter refreshes cache state itself; when it was
                // already set, fetch here so the on-disk checkmark and the
                // Download button reflect reality the moment the pane opens.
                await controller.refreshModelCacheState()
            }
            if controller.vocabulary.suggestions.isEmpty {
                await controller.vocabulary.scanForSuggestions()
            }
        }
        .task(id: modelDownloadStatusKey) {
            let aliases = viewModel.transcriptionModels.map(\.alias)
            let completed = aliases.filter {
                if case .completed = downloads.job(for: $0)?.status { return true }
                return false
            }
            handledCompletedDownloads.formIntersection(completed)
            let pending = completed.filter { !handledCompletedDownloads.contains($0) }
            let selected = controller.modelAlias
            let selectedCompleted: Bool
            if case .completed = downloads.job(for: selected)?.status {
                selectedCompleted = true
            } else {
                selectedCompleted = false
            }
            if !selectedCompleted { notifiedSelectedCompletion = nil }
            let shouldNotify = DictationDownloadReplay.shouldNotifySelected(
                selected: selected, selectedCompleted: selectedCompleted,
                lastNotified: notifiedSelectedCompletion
            )
            guard !pending.isEmpty || shouldNotify else { return }
            if !pending.isEmpty {
                await viewModel.refreshCatalog()
            }
            guard !Task.isCancelled else { return }
            // Selecting an already-handled model must still update the active session.
            if shouldNotify {
                await controller.modelDownloadDidFinish(alias: selected)
                notifiedSelectedCompletion = selected
            }
            for alias in pending {
                guard !Task.isCancelled else { return }
                if viewModel.audioModels.first(where: { $0.alias == alias })?.cached != true {
                    downloads.dismissJob(alias: alias)
                }
                handledCompletedDownloads.insert(alias)
            }
        }
        // TCC grants happen outside the app and emit no notification, so the
        // only reliable moment to re-check is when the window comes back.
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSApplication.didBecomeActiveNotification
            )
        ) { _ in
            controller.refreshReadiness()
            controller.revalidate()
        }
        .sheet(item: $fixTarget) { entry in
            DictationFixSheet(controller: controller, entry: entry)
        }
    }

    // MARK: - Intro

    private var intro: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
            Text("Speech to Text")
                .font(.rapid(.headline))
            Text("Press a hotkey in any app, speak, and your words appear at the cursor. Audio is transcribed on this Mac.")
                .font(.rapid(.subheadline))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Setup

    /// The three prerequisites are shown side by side rather than as a wizard:
    /// a returning user is usually missing exactly one of them and should not
    /// have to walk the whole flow again.
    /// Status and the switch live in one row that never moves. The dot and the
    /// sentence describe what is actually true right now; the switch is the
    /// only control that changes it.
    private var enableRow: some View {
        HStack(alignment: .center, spacing: RapidTheme.Space.md) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(statusHeadline)
                    .font(.rapid(.subheadline).weight(.medium))
                    .accessibilityIdentifier("Dictation.Status")
                Text(statusDetail)
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.md)
            // Gate turning it ON, never turning it OFF, so a session whose
            // permissions lapsed can always be switched back.
            Toggle("Dictation", isOn: $controller.isEnabled)
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(!controller.readinessSnapshot.isReady && !controller.isEnabled)
                .accessibilityIdentifier("Dictation.Enable")
        }
        .padding(RapidTheme.Space.lg)
    }

    private var statusColor: Color {
        guard controller.isEnabled else { return .secondary }
        switch audioReadinessState {
        case .ready, .active:
            return RapidTheme.green
        default:
            return .orange
        }
    }

    private var statusHeadline: String {
        guard controller.isEnabled else { return String(localized: "Dictation is off") }
        if controller.phase == .preparingModel {
            return String(localized: "Loading \(ModelDisplayName.name(for: controller.modelAlias)) into memory…")
        }
        if controller.phase == .off {
            return controller.isHotkeyArmed
                ? String(localized: "Listening paused — press \(controller.trigger.label) to reconnect")
                : String(localized: "Not listening — the hotkey isn't armed")
        }
        return String(localized: "Listening — press \(controller.trigger.label) in any app")
    }

    private var statusDetail: String {
        guard controller.isEnabled else {
            return controller.readinessSnapshot.isReady
                ? String(localized: "Turn it on to dictate into any app.")
                : blockingReason
        }
        if controller.phase == .preparingModel {
            return String(localized: "The local model is warming up. Recording starts when it’s ready.")
        }
        if controller.phase == .off {
            return controller.lastError
                ?? String(localized: "Rapid will load \(ModelDisplayName.name(for: controller.modelAlias)) when you next use dictation.")
        }
        var parts = [controller.modelAlias]
        if let latency = controller.lastLatency {
            parts.append(String(localized: "\(String(format: "%.2f", latency)) s last"))
        }
        // "why was that one slow" — present only when model bring-up ate
        // noticeable time, so the common warm line stays short.
        if let detail = controller.lastLatencyDetail {
            parts.append(detail)
        }
        if let warning = controller.lastWarmupWarning {
            parts.append(warning)
        }
        return parts.filter { !$0.isEmpty }.joined(separator: " · ")
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            enableRow
            Divider().overlay(RapidTheme.hairline)
            setupRow(
                label: "Model",
                done: controller.readinessSnapshot.modelSelected
                    && controller.readinessSnapshot.modelOnDisk
            ) {
                Button {
                    showModelPicker.toggle()
                } label: {
                    PopupControlChrome(
                        title: selectedModelDetails?.displayName ?? "Choose…",
                        width: 260
                    )
                }
                .buttonStyle(.plain)
                .disabled(!audioReadinessState.allowsModelSelection)
                .accessibilityLabel("Model")
                .accessibilityValue(ModelDisplayName.name(for: controller.modelAlias))
                .accessibilityIdentifier("Dictation.Model")
                .popover(isPresented: $showModelPicker, arrowEdge: .top) {
                    transcriptionModelPicker
                }
            } detail: {
                Text(modelDetail)
            }

            // The one rendering of download state, shared with Chat, Images
            // and the sibling Audio tabs: headline + bytes/speed/ETA detail,
            // a determinate bar, and the single next action. Nothing here is
            // hand-rolled, so it cannot drift out of alignment with the rest
            // of the app's model management.
            if let readiness = modelReadiness {
                ReadinessBanner(readiness: readiness, modelDisplayName: selectedModelDetails?.displayName, onAction: handleModelReadinessAction)
                    // xs outside + the banner's own md inside = lg: the
                    // banner's text sits on the rows' content line and its
                    // action button on the rows' trailing control line.
                    .padding(.horizontal, RapidTheme.Space.xs)
                    .padding(.bottom, RapidTheme.Space.lg)
            }

            Divider().overlay(RapidTheme.hairline)

            setupRow(
                label: "Microphone",
                done: controller.readinessSnapshot.microphone
            ) {
                if !controller.readinessSnapshot.microphone {
                    Button("Allow…") {
                        Task { await controller.requestMicrophone() }
                    }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Dictation.GrantMicrophone")
                }
            } detail: {
                Text("Recording runs only while a dictation session is open.")
            }

            Divider().overlay(RapidTheme.hairline)

            setupRow(
                label: "Accessibility",
                done: controller.readinessSnapshot.accessibility
            ) {
                if !controller.readinessSnapshot.accessibility {
                    Button("Grant…") { controller.requestAccessibility() }
                        .buttonStyle(.rapidSecondary)
                        .accessibilityIdentifier("Dictation.GrantAccessibility")
                }
            } detail: {
                // macOS reads this permission when a process launches, so
                // allowing it while Rapid is running leaves the live process
                // still seeing "denied". Saying so up front beats adding a
                // second control for the one case it applies to.
                Text("Needed to watch for the hotkey and to type into other apps. macOS applies it at launch — quit and reopen Rapid after allowing.")
            }

            Divider().overlay(RapidTheme.hairline)

            setupRow(
                label: "Hotkey",
                done: !controller.isEnabled || controller.isHotkeyArmed
            ) {
                Menu {
                    Picker("Hotkey", selection: $controller.trigger) {
                        ForEach(DictationHotkey.Trigger.allCases) { trigger in
                            Text(trigger.label).tag(trigger)
                        }
                    }
                    .accessibilityIdentifier("Dictation.Hotkey.Options")
                    .pickerStyle(.inline)
                    .labelsHidden()
                } label: {
                    PopupControlChrome(title: controller.trigger.label, width: 140)
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)
                .accessibilityLabel("Hotkey")
                .accessibilityValue(controller.trigger.label)
                .accessibilityIdentifier("Dictation.Hotkey")
            } detail: {
                // Left ⌘ is absent by design: it rides along with ⌘C, ⌘V and
                // ⌘Tab dozens of times an hour, so "tapped on its own" cannot be
                // detected reliably enough to arm a microphone.
                Text("Tap once to start, once more to stop. Only right-hand modifiers are offered — the left ones collide with everyday shortcuts.")
            }

        }
        .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                .strokeBorder(RapidTheme.hairline)
        )
    }

    /// Names the one thing still missing. A disabled control with no stated
    /// reason is the worst version of this screen.
    private var blockingReason: String {
        let missing = controller.readinessSnapshot
        if missing.modelSelected == false { return "Choose a model first." }
        if missing.modelOnDisk == false {
            if case .running = downloads.job(for: controller.modelAlias)?.status {
                return "Downloading the model — dictation can turn on when it finishes."
            }
            return String(localized: "Download the model below to turn dictation on.")
        }
        if missing.microphone == false { return "Microphone access is still needed." }
        if missing.accessibility == false { return "Accessibility access is still needed." }
        return ""
    }

    private var selectedModelEntry: ModelEntry? {
        viewModel.audioModels.first {
            $0.alias == controller.modelAlias
                && $0.audioCapability?.supportsTranscription == true
        }
    }

    private var audioReadinessState: AudioReadinessState {
        let alias = controller.modelAlias
        let activeAlias = controller.activeModelAlias
        let activity: AudioReadinessState.Activity? = switch controller.phase {
        case .starting: .startingCapture
        case .recording: .recording
        case .transcribing: .transcribing
        case .off, .preparingModel, .idle: nil
        }
        let isRuntimeReady: Bool = switch controller.phase {
        case .idle, .starting, .recording, .transcribing: true
        case .off, .preparingModel: false
        }

        return .resolve(.init(
            alias: alias,
            catalogLoaded: viewModel.catalogLoaded,
            cached: selectedModelEntry?.cached,
            sizeText: selectedModelEntry?.displayedModelSize?.displayText,
            download: AudioReadinessState.downloadSnapshot(
                alias: alias,
                job: downloads.job(for: alias)
            ),
            loading: controller.phase == .preparingModel
                ? activeAlias.map {
                    .init(alias: $0, detail: "The local model is warming up…")
                }
                : nil,
            readyAlias: isRuntimeReady ? activeAlias : nil,
            activity: activity.flatMap { activity in
                activeAlias.map { .init(alias: $0, activity: activity) }
            }
        ))
    }

    /// Compatibility aliases hidden by the picker still resolve to the same
    /// canonical row. Existing users keep a truthful selected state without a
    /// forced model restart; choosing that row later migrates the stored alias.
    private func isSelectedModel(_ entry: ModelEntry) -> Bool {
        guard let selectedModelEntry else { return false }
        if selectedModelEntry.alias == entry.alias { return true }
        guard let selectedRepo = selectedModelEntry.hfRepo,
              let entryRepo = entry.hfRepo else { return false }
        return selectedRepo.caseInsensitiveCompare(entryRepo) == .orderedSame
    }

    private var selectedModelDetails: AudioViewModel.TranscriptionModelDetails? {
        guard let entry = selectedModelEntry else { return nil }
        return AudioViewModel.transcriptionDetails(
            alias: entry.alias,
            family: entry.audioFamily
        )
    }

    private var transcriptionModelPicker: some View {
        ScrollView {
            LazyVStack(spacing: RapidTheme.Space.xs) {
                ForEach(viewModel.transcriptionModels, id: \.alias) { entry in
                    TranscriptionModelOptionRow(
                        entry: entry,
                        details: AudioViewModel.transcriptionDetails(
                            alias: entry.alias,
                            family: entry.audioFamily
                        ),
                        isSelected: isSelectedModel(entry),
                        downloadState: .resolve(cached: entry.cached, jobStatus: downloads.job(for: entry.alias)?.status,
                                                fraction: downloads.job(for: entry.alias)?.progress.progressFraction),
                        onDownload: { startModelDownload(alias: entry.alias) },
                        onCancel: { downloads.cancelDownload(alias: entry.alias) }
                    ) {
                        controller.modelAlias = entry.alias
                        showModelPicker = false
                    }
                }
            }
            .padding(RapidTheme.Space.sm)
            .accessibilityIdentifier("Dictation.Model.Options")
        }
        .frame(width: 410, height: transcriptionModelPopoverHeight)
    }

    private var transcriptionModelPopoverHeight: CGFloat {
        min(max(CGFloat(viewModel.transcriptionModels.count) * 76 + 16, 92), 460)
    }

    /// Alias + job status folded into one value so `.task(id:)` re-fires on
    /// any transition (same pattern as Quickstart's download watcher).
    private var modelDownloadStatusKey: String {
        DictationDownloadReplay.downloadStatusKey(
            selected: controller.modelAlias,
            pickerAliases: viewModel.transcriptionModels.map(\.alias),
            status: { downloads.job(for: $0)?.status }
        )
    }

    /// Download state for the selected model, resolved through the same
    /// truth table the sibling Audio tabs use. `nil` (chosen and on disk, or
    /// nothing chosen) renders no banner at all.
    private var modelReadiness: ModelReadiness? {
        guard selectedModelEntry?.cached == false else { return nil }
        return audioReadinessState.modelReadinessOverride
    }

    /// Dictation never loads-on-start: both Download and Retry only fetch
    /// weights, and prewarm picks the model up from the catalog watcher once
    /// the pull lands.
    private func handleModelReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .download(let alias), .retry(let alias):
            startModelDownload(alias: alias)
        case .chooseModel, .start, .restart, .openStartupLog, .openModelManagement:
            break
        }
    }

    private func startModelDownload(alias: String) {
        guard let entry = viewModel.audioModels.first(where: { $0.alias == alias }),
              !downloads.isDownloading(alias) else { return }
        if case .failed = downloads.job(for: alias)?.status {
            downloads.dismissJob(alias: alias)
        }
        _ = downloads.startDownload(
            alias: alias, hfPath: entry.hfRepo,
            totalBytes: entry.displayedModelSize?.byteCount
        )
    }

    private var modelDetail: String {
        guard !controller.modelAlias.isEmpty else {
            // Only name models the catalog can actually offer. The engine's STT
            // side is whisper/parakeet/sensevoice today; recommending anything
            // else here would point at a picker entry that does not exist.
            return String(localized: "whisper-large-v3-turbo is the usual pick — near large-v3 accuracy at a fraction of the latency.")
        }
        if let entry = selectedModelEntry, !entry.cached {
            if case .running = downloads.job(for: controller.modelAlias)?.status {
                return String(localized: "Downloading — dictation can turn on when it finishes.")
            }
            return selectedModelDetails?.summary ?? ModelDisplayName.name(for: entry.alias)
        }
        var detail = String(localized: "Ready on disk.")
        if let serving = server.servingAlias, !serving.isEmpty, serving != controller.modelAlias {
            // Same honesty as the readiness banners elsewhere: one model at a
            // time, and the swap has a real cost the user should hear about
            // before the hotkey, not during it.
            detail += String(localized: " First dictation briefly switches the running model.")
        }
        return detail
    }

    private func setupRow<Control: View, Detail: View>(
        label: String,
        done: Bool,
        @ViewBuilder control: () -> Control,
        @ViewBuilder detail: () -> Detail
    ) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(done ? RapidTheme.green : Color.secondary)
                .font(.system(size: 14))
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(label).font(.rapid(.subheadline).weight(.medium))
                detail()
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.md)
            control()
        }
        .padding(RapidTheme.Space.lg)
    }

    @ViewBuilder
    private var errorRow: some View {
        if let error = controller.lastError,
           DictationEnablePolicy.showsErrorBanner(lastError: error, modelAlias: controller.modelAlias,
                                                  modelDownloadRowVisible: modelReadiness != nil) {
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.system(size: 12))
                Text(error)
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, RapidTheme.Space.md)
            .padding(.vertical, RapidTheme.Space.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                Color.orange.opacity(0.10),
                in: RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
            )
            .accessibilityIdentifier("Dictation.Error")
        }
    }

    /// Steady-state only. An error is not a subtitle for the word "Ready" —
    /// it gets its own row below, where it reads as a problem rather than as a
    /// description of a working feature.
    private var readyDetail: String {
        var parts = [controller.modelAlias]
        if let latency = controller.lastLatency {
            parts.append(String(localized: "\(String(format: "%.2f", latency)) s last"))
        }
        if let detail = controller.lastLatencyDetail {
            parts.append(detail)
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Vocabulary

    private var vocabularySection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.md) {
                Text("Vocabulary").font(.rapid(.subheadline).weight(.semibold))
                budgetMeter
                Spacer()
            }

            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                if controller.vocabulary.terms.isEmpty {
                    Text("No terms yet. Add the names Rapid keeps getting wrong — project names, people, product names.")
                        .font(.rapid(.caption))
                        .foregroundStyle(.secondary)
                } else {
                    FlowLayout(spacing: RapidTheme.Space.sm) {
                        ForEach(controller.vocabulary.terms) { term in
                            termChip(term)
                        }
                    }
                }

                // The cap is the whole design constraint, not a nicety —
                // measured accuracy falls off past ~20 terms.
                Text("Accuracy drops when more than \(DictationVocabulary.activeLimit) terms are sent at once, so keep this list to the names that actually get missed.")
                    .font(.rapid(.caption2))
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: RapidTheme.Space.sm) {
                    TextField("Add a name…", text: $newTerm)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 220)
                        .onSubmit(addTerm)
                        .accessibilityIdentifier("Dictation.NewTerm")
                    Button("Add", action: addTerm)
                        .buttonStyle(.rapidSecondary)
                        .disabled(newTerm.trimmingCharacters(in: .whitespaces).isEmpty)
                        .accessibilityIdentifier("Dictation.AddTerm")
                    Spacer()
                }

                if !controller.vocabulary.suggestions.isEmpty {
                    Divider().overlay(RapidTheme.hairline)
                    VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                        Text("Found on this Mac")
                            .font(.rapid(.caption).weight(.medium))
                            .foregroundStyle(.secondary)
                        FlowLayout(spacing: RapidTheme.Space.sm) {
                            ForEach(controller.vocabulary.suggestions.prefix(12), id: \.self) { name in
                                Button {
                                    controller.vocabulary.add(name)
                                } label: {
                                    Label(name, systemImage: "plus")
                                        .font(.rapid(.caption).monospaced())
                                        .padding(.horizontal, RapidTheme.Space.sm)
                                        .padding(.vertical, RapidTheme.Space.xs)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 5)
                                                .strokeBorder(
                                                    RapidTheme.hairline,
                                                    style: StrokeStyle(lineWidth: 1, dash: [3, 2])
                                                )
                                        )
                                }
                                .buttonStyle(.plain)
                                .accessibilityIdentifier("Dictation.Suggestion.\(name)")
                            }
                        }
                    }
                }
            }
            .padding(RapidTheme.Space.lg)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                    .strokeBorder(RapidTheme.hairline)
            )
        }
    }

    private var budgetMeter: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            GeometryReader { proxy in
                let fraction = min(
                    1,
                    Double(controller.vocabulary.activeCount)
                        / Double(DictationVocabulary.activeLimit)
                )
                ZStack(alignment: .leading) {
                    Capsule().fill(RapidTheme.hairline)
                    Capsule()
                        .fill(controller.vocabulary.isOverBudget ? Color.orange : RapidTheme.green)
                        .frame(width: proxy.size.width * fraction)
                }
            }
            .frame(width: 80, height: 5)
            Text("\(controller.vocabulary.activeCount) of \(DictationVocabulary.activeLimit) active")
                .font(.rapid(.caption))
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    private func termChip(_ term: DictationVocabulary.Term) -> some View {
        HStack(spacing: RapidTheme.Space.xs) {
            Text(term.text)
                .font(.rapid(.caption).monospaced())
                .foregroundStyle(term.isActive ? Color.primary : Color.secondary)
            Button {
                controller.vocabulary.remove(term.text)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(String(localized: "Remove \(term.text)"))
            .accessibilityIdentifier("Dictation.RemoveTerm.\(term.text)")
        }
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.vertical, RapidTheme.Space.xs)
        .background(
            term.isActive ? RapidTheme.brandAmberTint : RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: 5)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 5).strokeBorder(RapidTheme.hairline)
        )
        .onTapGesture {
            controller.vocabulary.setActive(term.text, !term.isActive)
        }
        .help(term.isActive ? String(localized: "Sent with each dictation. Click to park.") : String(localized: "Parked. Click to activate."))
    }

    private func addTerm() {
        let trimmed = newTerm.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        controller.vocabulary.add(trimmed)
        newTerm = ""
    }

    // MARK: - History

    private var historySection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack {
                Text("Recent").font(.rapid(.subheadline).weight(.semibold))
                Spacer()
                Toggle("Keep recordings", isOn: $controller.archiveAudio)
                    .toggleStyle(.checkbox)
                    .font(.rapid(.caption))
                    .help("Off by default. Keeping recordings lets a correction be verified against the original audio.")
                    .accessibilityIdentifier("Dictation.ArchiveAudio")
                if !controller.history.entries.isEmpty {
                    Button("Clear") { controller.history.clear() }
                        .buttonStyle(.rapidTertiary)
                        .accessibilityIdentifier("Dictation.ClearHistory")
                }
            }

            if controller.history.entries.isEmpty {
                Text("Dictations you make will show up here.")
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
                    .padding(RapidTheme.Space.lg)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RapidTheme.card,
                        in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                            .strokeBorder(RapidTheme.hairline)
                    )
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(controller.history.entries.prefix(12).enumerated()), id: \.element.id) { index, entry in
                        if index > 0 { Divider().overlay(RapidTheme.hairline) }
                        historyRow(entry)
                    }
                }
                .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: RapidTheme.cardRadius))
                .overlay(
                    RoundedRectangle(cornerRadius: RapidTheme.cardRadius)
                        .strokeBorder(RapidTheme.hairline)
                )
            }
        }
    }

    private func historyRow(_ entry: DictationHistory.Entry) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            Text(entry.date, style: .time)
                .font(.rapid(.caption).monospaced())
                .foregroundStyle(.tertiary)
                .frame(width: UIFontSettings.shared.scaled(62), alignment: .leading)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(entry.text)
                    .font(.rapid(.callout))
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                HStack(spacing: RapidTheme.Space.md) {
                    if let app = entry.appName { Text(app) }
                    Text(String(localized: "\(String(format: "%.1f", entry.duration))s"))
                    Text(String(localized: "\(String(format: "%.2f", entry.latency))s"))
                }
                .font(.rapid(.caption2))
                .foregroundStyle(.tertiary)
            }
            Spacer(minLength: RapidTheme.Space.sm)
            HStack(spacing: RapidTheme.Space.xs) {
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(entry.text, forType: .string)
                }
                .buttonStyle(.rapidTertiary)
                .accessibilityIdentifier("Dictation.CopyTranscript")
                if entry.audioFile != nil {
                    Button("Fix…") { fixTarget = entry }
                        .buttonStyle(.rapidTertiary)
                        .accessibilityIdentifier("Dictation.Fix")
                }
            }
        }
        .padding(RapidTheme.Space.lg)
    }
}

enum TranscriptionDownloadControlState: Equatable {
    case none, download, downloading(fraction: Double?), retry

    static func resolve(cached: Bool, jobStatus: DownloadManager.Job.Status?, fraction: Double?) -> Self {
        if cached { return .none }
        switch jobStatus {
        case .running: return .downloading(fraction: fraction)
        case .failed: return .retry
        case .completed: return .none
        case .cancelled, nil: return .download
        }
    }
}

private struct TranscriptionModelOptionRow: View {
    let entry: ModelEntry
    let details: AudioViewModel.TranscriptionModelDetails
    let isSelected: Bool
    let downloadState: TranscriptionDownloadControlState
    let onDownload: () -> Void
    let onCancel: () -> Void
    let select: () -> Void

    @State private var hovering = false

    var body: some View {
        HStack(alignment: .center, spacing: RapidTheme.Space.xs) {
            Button(action: select) {
                HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .semibold))
                        .opacity(isSelected ? 1 : 0)
                        .frame(width: 14, height: 18)

                    VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.xs) {
                            FlowLayout(spacing: RapidTheme.Space.xs) {
                                Text(details.displayName).font(RapidFont.body)
                                pickerBadge(details.badge)
                                if details.isRecommended {
                                    pickerBadge("recommended", emphasized: true)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            if let size = entry.displayedModelSize?.displayText {
                                Text(size)
                                    .font(RapidFont.caption)
                                    .foregroundStyle(.tertiary)
                                    .fixedSize()
                            }
                            if entry.cached {
                                Image(systemName: ModelPickerBar.cacheGlyph(cached: true))
                                    .font(.caption)
                                    .foregroundStyle(RapidTheme.green)
                                    .accessibilityHidden(true)
                            }
                        }
                        Text(details.summary)
                            .font(RapidFont.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .multilineTextAlignment(.leading)
                    }
                }
                .padding(.horizontal, RapidTheme.Space.sm)
                .padding(.vertical, RapidTheme.Space.xs)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(isSelected || hovering ? RapidTheme.hoverFill : .clear)
            )
            .contentShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous))
            .onHover { hovering = $0 }
            .rapidAnimation(RapidMotion.quick, value: hovering)
            .accessibilityLabel(accessibilityLabel)
            .accessibilityAddTraits(isSelected ? .isSelected : [])
            .accessibilityIdentifier("Dictation.Model.Option.\(entry.alias)")
            downloadControl
        }
    }

    @ViewBuilder
    private var downloadControl: some View {
        switch downloadState {
        case .none:
            EmptyView()
        case .download:
            Button(action: onDownload) { Image(systemName: "arrow.down.circle") }
                .buttonStyle(.plain)
                .accessibilityLabel(String(localized: "Download \(details.displayName)"))
                .help(String(localized: "Download"))
                .accessibilityIdentifier("Dictation.Model.Download.\(entry.alias)")
        case .retry:
            Button("Retry", action: onDownload)
                .accessibilityIdentifier("Dictation.Model.Retry.\(entry.alias)")
        case .downloading(let fraction):
            HStack(spacing: RapidTheme.Space.xs) {
                if let fraction {
                    ProgressView(value: fraction).frame(width: 42)
                        .accessibilityLabel("Dictation model download progress")
                    Text(verbatim: "100%")
                        .font(RapidFont.caption)
                        .monospacedDigit()
                        .hidden()
                        .overlay(alignment: .trailing) {
                            Text(verbatim: "\(Int((fraction * 100).rounded()))%")
                                .font(RapidFont.caption)
                                .monospacedDigit()
                        }
                } else {
                    ProgressView().controlSize(.small)
                }
                Button(action: onCancel) { Image(systemName: "xmark") }
                    .buttonStyle(.plain)
                    .accessibilityLabel(String(localized: "Cancel"))
                    .accessibilityIdentifier("Dictation.Model.Cancel.\(entry.alias)")
            }
        }
    }

    private func pickerBadge(_ text: String, emphasized: Bool = false) -> some View {
        Text(text)
            .font(.rapid(size: 9, weight: .medium))
            .foregroundStyle(emphasized ? RapidTheme.green : Color.secondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(emphasized ? RapidTheme.green.opacity(0.12) : RapidTheme.hoverFill)
            )
            .fixedSize()
    }

    private var accessibilityLabel: String {
        var parts = [details.displayName, details.badge]
        if details.isRecommended { parts.append(String(localized: "Recommended")) }
        if case .downloading(let fraction) = downloadState {
            if let fraction {
                parts.append(String(localized: "Downloading, \(Int((fraction * 100).rounded())) percent"))
            } else {
                parts.append(String(localized: "Downloading"))
            }
        } else {
            parts.append(entry.cached ? String(localized: "Downloaded") : String(localized: "Not downloaded"))
        }
        if let size = entry.displayedModelSize?.displayText { parts.append(size) }
        parts.append(details.summary)
        return parts.joined(separator: ", ")
    }
}

/// Correcting a transcript is the main way the vocabulary grows: the user is the
/// only one who knows the word was wrong, and the correction is worthless unless
/// it also teaches the model.
private struct DictationFixSheet: View {
    @Bindable var controller: DictationController
    let entry: DictationHistory.Entry

    @Environment(\.dismiss) private var dismiss
    @State private var heard = ""
    @State private var correction = ""
    @State private var verifying = false
    @State private var verdict: String?

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            Text("Fix transcription").font(.rapid(.headline))

            Text(entry.text)
                .font(.rapid(.callout))
                .padding(RapidTheme.Space.md)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RapidTheme.surfaceRaised,
                    in: RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
                )

            HStack(spacing: RapidTheme.Space.md) {
                Text("Heard").frame(width: UIFontSettings.shared.scaled(68), alignment: .leading)
                TextField("Header", text: $heard)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("Dictation.Fix.Heard")
            }
            HStack(spacing: RapidTheme.Space.md) {
                Text("Should be").frame(width: UIFontSettings.shared.scaled(68), alignment: .leading)
                TextField("herdr", text: $correction)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("Dictation.Fix.Correction")
            }

            if let verdict {
                Text(verdict)
                    .font(.rapid(.caption))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Dictation.Fix.Cancel")
                Spacer()
                Button(verifying ? String(localized: "Checking…") : String(localized: "Fix & remember")) { apply() }
                    .buttonStyle(.rapidPrimary)
                    .disabled(verifying || correction.trimmingCharacters(in: .whitespaces).isEmpty)
                    .accessibilityIdentifier("Dictation.Fix.Apply")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 440)
    }

    /// Saves the term, then re-runs the original audio through the model to
    /// confirm the hint actually helps. Adding a term can regress a different
    /// one, so a vocabulary edit that is never verified quietly rots.
    private func apply() {
        let fixed = correction.trimmingCharacters(in: .whitespaces)
        guard !fixed.isEmpty else { return }
        verifying = true
        controller.vocabulary.noteCorrection(to: fixed)

        Task {
            let rerun = await controller.retranscribe(entry)
            verifying = false
            guard let rerun else {
                applyTextEdit(fixed)
                dismiss()
                return
            }
            if rerun.localizedCaseInsensitiveContains(fixed) {
                controller.history.updateText(rerun, for: entry.id)
                dismiss()
            } else {
                // Kept in the vocabulary regardless — the user's correction is
                // ground truth even when one hint is not enough to recover it.
                verdict = String(localized: "Saved “\(fixed)”, but re-running this recording still produced: \(rerun)")
                applyTextEdit(fixed)
            }
        }
    }

    private func applyTextEdit(_ fixed: String) {
        guard !heard.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let updated = entry.text.replacingOccurrences(
            of: heard.trimmingCharacters(in: .whitespaces),
            with: fixed
        )
        controller.history.updateText(updated, for: entry.id)
    }
}

enum DictationDownloadReplay {
    static func downloadStatusKey(
        selected: String,
        pickerAliases: [String],
        status: (String) -> DownloadManager.Job.Status?
    ) -> String {
        var aliases = Set(pickerAliases)
        if !selected.isEmpty { aliases.insert(selected) }
        let statuses = aliases.sorted().map { alias in
            let label: String
            switch status(alias) {
            case .running: label = "running"
            case .completed: label = "completed"
            case .failed: label = "failed"
            case .cancelled: label = "cancelled"
            case nil: label = "none"
            }
            return "\(alias)#\(label)"
        }
        return ([selected] + statuses).joined(separator: "|")
    }

    static func shouldNotifySelected(selected: String, selectedCompleted: Bool, lastNotified: String?) -> Bool {
        selectedCompleted && lastNotified != selected
    }
}

/// Minimal wrapping layout for chips. `LazyVGrid` cannot do variable-width
/// items, and a plain `HStack` overflows once a few long names are added.
struct FlowLayout: Layout {
    var spacing: CGFloat

    static func frames(
        maxWidth: CGFloat,
        spacing: CGFloat,
        measure: [(ProposedViewSize) -> CGSize]
    ) -> (frames: [CGRect], size: CGSize) {
        var frames: [CGRect] = []
        var origin = CGPoint.zero
        var lineHeight: CGFloat = 0
        var width: CGFloat = 0
        var lineIsEmpty = true

        for measureSubview in measure {
            var size = measureSubview(.unspecified)
            if maxWidth.isFinite, size.width > maxWidth {
                size = measureSubview(ProposedViewSize(width: maxWidth, height: nil))
                size.width = min(size.width, maxWidth)
            }
            if !lineIsEmpty, origin.x + spacing + size.width > maxWidth {
                origin.x = 0
                origin.y += lineHeight + spacing
                lineHeight = 0
                lineIsEmpty = true
            }
            if !lineIsEmpty { origin.x += spacing }
            frames.append(CGRect(origin: origin, size: size))
            origin.x += size.width
            lineHeight = max(lineHeight, size.height)
            width = max(width, origin.x)
            lineIsEmpty = false
        }
        return (frames, CGSize(width: width, height: origin.y + lineHeight))
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        Self.frames(maxWidth: proposal.width ?? .infinity, spacing: spacing,
                    measure: subviews.map { subview in { subview.sizeThatFits($0) } }).size
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let frames = Self.frames(maxWidth: bounds.width, spacing: spacing,
                                 measure: subviews.map { subview in { subview.sizeThatFits($0) } }).frames
        for (subview, frame) in zip(subviews, frames) {
            subview.place(at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY),
                          proposal: ProposedViewSize(width: frame.width, height: frame.height))
        }
    }
}
