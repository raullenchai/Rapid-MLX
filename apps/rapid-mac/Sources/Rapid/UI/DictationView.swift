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
        // Runs when the selected alias's pull changes state; on completion,
        // re-read the catalog so the row flips to "Ready on disk" and the
        // model warms without another visit to the pane.
        .task(id: modelDownloadStatusKey) {
            guard case .completed = downloads.job(for: controller.modelAlias)?.status else { return }
            await controller.modelDownloadDidFinish()
            await viewModel.refreshCatalog()
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
                .font(.headline)
            Text("Press a hotkey in any app, speak, and your words appear at the cursor. Audio is transcribed on this Mac.")
                .font(.subheadline)
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
                    .font(.subheadline.weight(.medium))
                    .accessibilityIdentifier("Dictation.Status")
                Text(statusDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.md)
            if controller.isEnabled && controller.phase == .off
                && controller.readinessSnapshot.isReady {
                Button("Arm now") { Task { await controller.enable() } }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Dictation.Arm")
            }
            // Gate turning it ON, never turning it OFF, so a session whose
            // permissions lapsed can always be switched back.
            Toggle("", isOn: $controller.isEnabled)
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(!controller.readinessSnapshot.isReady && !controller.isEnabled)
                .accessibilityIdentifier("Dictation.Enable")
        }
        .padding(RapidTheme.Space.lg)
    }

    private var statusColor: Color {
        guard controller.isEnabled else { return .secondary }
        if controller.phase == .preparingModel { return .orange }
        return controller.phase == .off ? .orange : RapidTheme.green
    }

    private var statusHeadline: String {
        guard controller.isEnabled else { return "Dictation is off" }
        if controller.phase == .preparingModel {
            return "Loading \(controller.modelAlias) into memory…"
        }
        return controller.phase == .off
            ? "Not listening — the hotkey isn't armed"
            : "Ready — press \(controller.trigger.label) in any app"
    }

    private var statusDetail: String {
        guard controller.isEnabled else {
            return controller.readinessSnapshot.isReady
                ? "Turn it on to dictate into any app."
                : blockingReason
        }
        if controller.phase == .preparingModel {
            return "The local model is warming up. Recording starts when it’s ready."
        }
        var parts = [controller.modelAlias]
        if let latency = controller.lastLatency {
            parts.append(String(format: "%.2f s last", latency))
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
                .disabled(controller.phase != .off && controller.phase != .idle)
                .accessibilityLabel("Model")
                .accessibilityValue(controller.modelAlias)
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
                ReadinessBanner(readiness: readiness, onAction: handleModelReadinessAction)
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

            setupRow(label: "Hotkey", done: true) {
                Menu {
                    Picker("", selection: $controller.trigger) {
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
            return "Download the model first."
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
                        isSelected: isSelectedModel(entry)
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
        let status: String
        switch downloads.job(for: controller.modelAlias)?.status {
        case .running: status = "running"
        case .completed: status = "completed"
        case .failed: status = "failed"
        case .cancelled: status = "cancelled"
        case nil: status = "none"
        }
        return "\(controller.modelAlias)#\(status)"
    }

    /// Download state for the selected model, resolved through the same
    /// truth table the sibling Audio tabs use. `nil` (chosen and on disk, or
    /// nothing chosen) renders no banner at all.
    private var modelReadiness: ModelReadiness? {
        let alias = controller.modelAlias
        guard !alias.isEmpty, let entry = selectedModelEntry, !entry.cached else { return nil }
        let job = downloads.job(for: alias)
        if case .completed = job?.status {
            // The catalog refresh that flips `entry.cached` is in flight;
            // don't flash the Download action back in the meantime.
            return .starting(alias: alias, detail: "Finishing the download…")
        }
        return AudioView.audioDownloadReadiness(
            alias: alias,
            cached: entry.cached,
            sizeText: entry.sizeOnDisk,
            job: job,
            activationInFlight: false
        )
    }

    /// Dictation never loads-on-start: both Download and Retry only fetch
    /// weights, and prewarm picks the model up from the catalog watcher once
    /// the pull lands.
    private func handleModelReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .download(let alias), .retry(let alias):
            guard let entry = viewModel.audioModels.first(where: { $0.alias == alias }),
                  !downloads.isDownloading(alias) else { return }
            if case .failed = downloads.job(for: alias)?.status {
                downloads.dismissJob(alias: alias)
            }
            _ = downloads.startDownload(
                alias: alias,
                hfPath: entry.hfRepo,
                totalBytes: ModelCacheActions.parseSizeBytes(entry.sizeOnDisk)
            )
        case .chooseModel, .start, .restart, .openModelManagement:
            break
        }
    }

    private var modelDetail: String {
        guard !controller.modelAlias.isEmpty else {
            // Only name models the catalog can actually offer. The engine's STT
            // side is whisper/parakeet/sensevoice today; recommending anything
            // else here would point at a picker entry that does not exist.
            return "whisper-large-v3-turbo is the usual pick — near large-v3 accuracy at a fraction of the latency."
        }
        if let entry = selectedModelEntry, !entry.cached {
            if case .running = downloads.job(for: controller.modelAlias)?.status {
                return "Downloading — dictation can turn on when it finishes."
            }
            return "Not downloaded yet — dictation can turn on once it's on disk."
        }
        var detail = "Ready on disk."
        if let serving = server.servingAlias, !serving.isEmpty, serving != controller.modelAlias {
            // Same honesty as the readiness banners elsewhere: one model at a
            // time, and the swap has a real cost the user should hear about
            // before the hotkey, not during it.
            detail += " First dictation briefly switches the running model."
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
                Text(label).font(.subheadline.weight(.medium))
                detail()
                    .font(.caption)
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
        if let error = controller.lastError {
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.system(size: 12))
                Text(error)
                    .font(.caption)
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
            parts.append(String(format: "%.2f s last", latency))
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
                Text("Vocabulary").font(.subheadline.weight(.semibold))
                budgetMeter
                Spacer()
            }

            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                if controller.vocabulary.terms.isEmpty {
                    Text("No terms yet. Add the names Rapid keeps getting wrong — project names, people, product names.")
                        .font(.caption)
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
                    .font(.caption2)
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
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                        FlowLayout(spacing: RapidTheme.Space.sm) {
                            ForEach(controller.vocabulary.suggestions.prefix(12), id: \.self) { name in
                                Button {
                                    controller.vocabulary.add(name)
                                } label: {
                                    Label(name, systemImage: "plus")
                                        .font(.caption.monospaced())
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
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    private func termChip(_ term: DictationVocabulary.Term) -> some View {
        HStack(spacing: RapidTheme.Space.xs) {
            Text(term.text)
                .font(.caption.monospaced())
                .foregroundStyle(term.isActive ? Color.primary : Color.secondary)
            Button {
                controller.vocabulary.remove(term.text)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(term.text)")
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
        .help(term.isActive ? "Sent with each dictation. Click to park." : "Parked. Click to activate.")
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
                Text("Recent").font(.subheadline.weight(.semibold))
                Spacer()
                Toggle("Keep recordings", isOn: $controller.archiveAudio)
                    .toggleStyle(.checkbox)
                    .font(.caption)
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
                    .font(.caption)
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
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .frame(width: 62, alignment: .leading)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(entry.text)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                HStack(spacing: RapidTheme.Space.md) {
                    if let app = entry.appName { Text(app) }
                    Text(String(format: "%.1fs", entry.duration))
                    Text(String(format: "%.2fs", entry.latency))
                }
                .font(.caption2)
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

private struct TranscriptionModelOptionRow: View {
    let entry: ModelEntry
    let details: AudioViewModel.TranscriptionModelDetails
    let isSelected: Bool
    let select: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: select) {
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Image(systemName: "checkmark")
                    .font(.system(size: 10, weight: .semibold))
                    .opacity(isSelected ? 1 : 0)
                    .frame(width: 14, height: 18)

                VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                    HStack(spacing: RapidTheme.Space.xs) {
                        Text(details.displayName)
                            .font(RapidFont.body)
                            .lineLimit(1)
                        pickerBadge("offline")
                        pickerBadge(details.badge)
                        if details.isRecommended {
                            pickerBadge("recommended", emphasized: true)
                        }
                        Spacer(minLength: RapidTheme.Space.xs)
                        if let size = entry.sizeOnDisk {
                            Text(size)
                                .font(RapidFont.caption)
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                        }
                        Image(systemName: ModelPickerBar.cacheGlyph(cached: entry.cached))
                            .font(.caption)
                            .foregroundStyle(entry.cached ? RapidTheme.green : .secondary)
                            .accessibilityHidden(true)
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
    }

    private func pickerBadge(_ text: String, emphasized: Bool = false) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(emphasized ? RapidTheme.green : Color.secondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(emphasized ? RapidTheme.green.opacity(0.12) : RapidTheme.hoverFill)
            )
            .lineLimit(1)
    }

    private var accessibilityLabel: String {
        var parts = [details.displayName, details.badge]
        if details.isRecommended { parts.append("Recommended") }
        parts.append(entry.cached ? "Downloaded" : "Not downloaded")
        if let size = entry.sizeOnDisk { parts.append(size) }
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
            Text("Fix transcription").font(.headline)

            Text(entry.text)
                .font(.callout)
                .padding(RapidTheme.Space.md)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RapidTheme.surfaceRaised,
                    in: RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
                )

            HStack(spacing: RapidTheme.Space.md) {
                Text("Heard").frame(width: 68, alignment: .leading)
                TextField("Header", text: $heard)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("Dictation.Fix.Heard")
            }
            HStack(spacing: RapidTheme.Space.md) {
                Text("Should be").frame(width: 68, alignment: .leading)
                TextField("herdr", text: $correction)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("Dictation.Fix.Correction")
            }

            if let verdict {
                Text(verdict)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("Dictation.Fix.Cancel")
                Spacer()
                Button(verifying ? "Checking…" : "Fix & remember") { apply() }
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
                verdict = "Saved “\(fixed)”, but re-running this recording still produced: \(rerun)"
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

/// Minimal wrapping layout for chips. `LazyVGrid` cannot do variable-width
/// items, and a plain `HStack` overflows once a few long names are added.
private struct FlowLayout: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var origin = CGPoint.zero
        var lineHeight: CGFloat = 0
        var total = CGSize.zero

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if origin.x + size.width > maxWidth, origin.x > 0 {
                origin.x = 0
                origin.y += lineHeight + spacing
                lineHeight = 0
            }
            origin.x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            total.width = max(total.width, min(origin.x - spacing, maxWidth))
        }
        total.height = origin.y + lineHeight
        return total
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        var origin = bounds.origin
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if origin.x + size.width > bounds.maxX, origin.x > bounds.minX {
                origin.x = bounds.minX
                origin.y += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: origin, proposal: ProposedViewSize(size))
            origin.x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
