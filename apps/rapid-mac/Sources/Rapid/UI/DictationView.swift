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

    @State private var newTerm = ""
    @State private var fixTarget: DictationHistory.Entry?

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
            }
            if controller.vocabulary.suggestions.isEmpty {
                await controller.vocabulary.scanForSuggestions()
            }
            // Swap the model in while the user is still reading this page,
            // rather than on the first hotkey press when they are mid-sentence.
            await controller.prewarmModel()
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
        return controller.phase == .off ? .orange : RapidTheme.green
    }

    private var statusHeadline: String {
        guard controller.isEnabled else { return "Dictation is off" }
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
        var parts = [controller.modelAlias]
        if let latency = controller.lastLatency {
            parts.append(String(format: "%.2f s last", latency))
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
            ) {
                Picker("", selection: $controller.modelAlias) {
                    Text("Choose…").tag("")
                    ForEach(viewModel.transcriptionModels, id: \.alias) { entry in
                        Text(entry.alias).tag(entry.alias)
                    }
                }
                .labelsHidden()
                .frame(width: 260)
                .accessibilityIdentifier("Dictation.Model")
            } detail: {
                Text(modelDetail)
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
                Picker("", selection: $controller.trigger) {
                    ForEach(DictationHotkey.Trigger.allCases) { trigger in
                        Text(trigger.label).tag(trigger)
                    }
                }
                .labelsHidden()
                .frame(width: 140)
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
        if missing.microphone == false { return "Microphone access is still needed." }
        if missing.accessibility == false { return "Accessibility access is still needed." }
        return ""
    }

    private var modelDetail: String {
        guard !controller.modelAlias.isEmpty else {
            // Only name models the catalog can actually offer. The engine's STT
            // side is whisper/parakeet/sensevoice today; recommending anything
            // else here would point at a picker entry that does not exist.
            return "whisper-large-v3-turbo is the usual pick — near large-v3 accuracy at a fraction of the latency."
        }
        let entry = viewModel.transcriptionModels.first { $0.alias == controller.modelAlias }
        if let entry, !entry.cached {
            return "Not downloaded yet — the first dictation will fetch \(entry.sizeOnDisk ?? "the weights")."
        }
        return "Ready on disk."
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
