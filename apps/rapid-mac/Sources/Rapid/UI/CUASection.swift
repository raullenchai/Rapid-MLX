import SwiftUI

struct CUAResultPresentation: Equatable {
    let answer: String
    let evidence: String?
    let detailLabel: String
    let completionNote: String?

    var displayedAnswer: String {
        answer.isEmpty ? "The task completed without a result summary." : answer
    }

    init(summary: String) {
        let normalized = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        let separators = [
            " — evidenced by ",
            " Supporting evidence: ",
            " Evidence: ",
            "\nSupporting evidence: ",
            "\nEvidence: ",
        ]
        let match = separators.compactMap { separator -> Range<String.Index>? in
            normalized.range(of: separator, options: .caseInsensitive)
        }.min { $0.lowerBound < $1.lowerBound }

        if let match {
            let leading = normalized[..<match.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let trailing = normalized[match.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !leading.isEmpty, !trailing.isEmpty {
                if leading.count > Self.answerLimit {
                    answer = Self.boundedAnswer(leading)
                    evidence = normalized
                    detailLabel = "Full result"
                    completionNote = Self.completionNote(in: normalized, answer: answer)
                } else {
                    answer = leading
                    evidence = trailing
                    detailLabel = "Supporting evidence"
                    completionNote = nil
                }
                return
            }
        }
        if normalized.count > Self.answerLimit {
            answer = Self.boundedAnswer(normalized)
            evidence = normalized
            detailLabel = "Full result"
            completionNote = Self.completionNote(in: normalized, answer: answer)
        } else {
            answer = normalized
            evidence = nil
            detailLabel = "Supporting evidence"
            completionNote = nil
        }
    }

    private static let answerLimit = 220
    private static let completionNoteLimit = 180

    private static func boundedAnswer(_ value: String) -> String {
        let limit = value.index(value.startIndex, offsetBy: answerLimit)
        let candidate = value[..<limit]
        if let sentenceEnd = candidate.lastIndex(where: { ".!?".contains($0) }),
           value.distance(from: value.startIndex, to: sentenceEnd) >= 60 {
            return String(candidate[...sentenceEnd])
        }
        let ellipsisLimit = value.index(value.startIndex, offsetBy: answerLimit - 1)
        return value[..<ellipsisLimit]
            .trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    private static func completionNote(in value: String, answer: String) -> String? {
        var searchEnd = value.endIndex
        if let last = value.last, ".!?".contains(last) {
            searchEnd = value.index(before: searchEnd)
        }
        let prefix = value[..<searchEnd]
        guard let boundary = prefix.lastIndex(where: { ".!?\n".contains($0) }) else {
            return nil
        }
        let start = value.index(after: boundary)
        let sentence = value[start...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sentence.isEmpty, !answer.contains(sentence) else { return nil }
        guard sentence.count > completionNoteLimit else { return sentence }
        let suffixStart = sentence.index(
            sentence.endIndex, offsetBy: -(completionNoteLimit - 1)
        )
        return "…" + sentence[suffixStart...]
    }
}

struct CUAFailurePresentation: Equatable {
    let summary: String
    let technicalDetails: String
    let changeWarning: String?

    init(message: String, hasExecutedActions: Bool = false) {
        let normalized = message.trimmingCharacters(in: .whitespacesAndNewlines)
        technicalDetails = normalized.isEmpty ? "No failure details were provided." : normalized
        changeWarning = hasExecutedActions
            ? "Some changes may have been made; check the target app."
            : nil

        let payloadMarkers = [": {", ": [", "\n{", "\n["]
        let payloadStart = payloadMarkers.compactMap { marker in
            normalized.range(of: marker)?.lowerBound
        }.min()
        if let payloadStart {
            let prefix = normalized[..<payloadStart]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            if !prefix.isEmpty {
                summary = Self.boundedSummary(prefix)
                return
            }
        }

        if normalized.isEmpty {
            summary = "The task could not be completed."
        } else {
            summary = Self.boundedSummary(normalized)
        }
    }

    private static func boundedSummary(_ value: String) -> String {
        guard value.count > 240 else { return value }
        let limit = value.index(value.startIndex, offsetBy: 237)
        let candidate = value[..<limit]
        if let sentenceEnd = candidate.lastIndex(where: { ".!?".contains($0) }) {
            return String(candidate[...sentenceEnd])
        }
        return candidate.trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    static func hasPotentialSideEffects(in events: [CUAEvent]) -> Bool {
        let actionKinds = Set(["click", "fill", "press", "scroll", "save"])
        return events.contains { event in
            event.kind == "executed" && event.action.map(actionKinds.contains) == true
        }
    }
}

/// "Run an agent task" — the goal-driven computer-use section at the top of
/// the (experimental) Computer Use panel. Talks to the app-owned local
/// server's `/v1/cua` API; consent gates are enforced server-side, and the
/// only client-side action is approving a sign-in gate.
struct CUASection: View {
    @ObservedObject var viewModel: CUAViewModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var permissionSnapshot = MacAutomationPermissions.snapshot()
    @State private var brainDraftName = ""
    @State private var brainDraftURL = ""
    @State private var brainDraftModel = ""
    @State private var brainDraftAPIKey = ""
    @State private var brainDraftTextOnly = false
    @State private var brainDraftAllowRemote = false

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: 8) {
                Text("RUN AN AGENT TASK")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.phase.isBusy {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if viewModel.executorPermissions?.isReady != true
                || !permissionSnapshot.isReadyForComputerUse
            {
                permissionReadiness
            }

            if let context = viewModel.runContext {
                runContextCard(context)
            } else {
            VStack(alignment: .leading, spacing: 4) {
                Text("Task")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                ZStack(alignment: .topLeading) {
                    if viewModel.goal.isEmpty {
                        Text("Describe what you want Rapid to do…")
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .allowsHitTesting(false)
                            .accessibilityHidden(true)
                    }
                    TextEditor(text: $viewModel.goal)
                        .frame(minHeight: 52, maxHeight: 96)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .accessibilityLabel("Task goal")
                        .accessibilityIdentifier("ComputerUse.Agent.Goal")
                }
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(.secondary.opacity(0.2)))
            }

            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Brain").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    HStack(spacing: 6) {
                        Picker("Brain", selection: $viewModel.plannerName) {
                            ForEach(viewModel.plannerOptions) { option in
                                Text(option.displayName).tag(option.name)
                            }
                            if viewModel.plannerOptions.isEmpty {
                                Text(viewModel.plannerName).tag(viewModel.plannerName)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 340)
                        .accessibilityIdentifier("ComputerUse.Agent.Planner")
                        Button {
                            viewModel.showAddBrain = true
                        } label: {
                            Image(systemName: "plus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Add an OpenAI-compatible brain")
                        .accessibilityIdentifier("ComputerUse.Agent.AddBrain")
                    }
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Max steps").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Stepper("\(viewModel.maxSteps)", value: $viewModel.maxSteps, in: 1 ... 40)
                        .accessibilityIdentifier("ComputerUse.Agent.MaxSteps")
                }
            }

            targetPicker

            Label(viewModel.plannerDisclosure, systemImage: "desktopcomputer")
                .font(.caption)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("ComputerUse.Agent.BrainDisclosure")

            if !viewModel.phase.isBusy {
                Button("Start") {
                    Task { await viewModel.start() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    !viewModel.canStart || viewModel.executorPermissions?.isReady == false
                )
                .accessibilityIdentifier("ComputerUse.Agent.Start")
            }
            }

            if viewModel.phase == .starting || viewModel.phase == .running {
                activeProgress
            }

            if viewModel.phase == .awaitingApproval {
                approvalCard
            }

            if case let .finished(summary) = viewModel.phase {
                resultCard(summary: summary)
            }
            if case let .failed(message) = viewModel.phase {
                failureCard(message: message)
            }
            if viewModel.runContext != nil, !viewModel.phase.isBusy {
                Button("New Task") { viewModel.newTask() }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isSessionDetached)
                    .accessibilityIdentifier("ComputerUse.Agent.NewTask")
            }
            if let actionError = viewModel.actionError {
                Label(actionError, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.ActionError")
            }

            if !viewModel.events.isEmpty, !viewModel.phase.hasTerminalCard {
                runDetails(evidence: nil)
            }
        }
        .padding(16)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.secondary.opacity(0.2)))
        .frame(maxWidth: 984, alignment: .leading)
        .task {
            guard !viewModel.isSessionDetached else { return }
            await viewModel.loadPlanners()
            await viewModel.loadPermissions()
            await viewModel.loadTargets()
            permissionSnapshot = MacAutomationPermissions.snapshot()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                permissionSnapshot = MacAutomationPermissions.snapshot()
                if !viewModel.isSessionDetached {
                    Task { await viewModel.loadPermissions() }
                }
            }
        }
        .sheet(isPresented: $viewModel.showAddBrain) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Add a Brain").font(.headline)
                Text(
                    "Rapid always executes actions on this Mac. Choose where the planning model runs and exactly what it may receive."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Form {
                    TextField("Name (e.g. deepseek)", text: $brainDraftName)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainName")
                    TextField("Model (e.g. deepseek-reasoner)", text: $brainDraftModel)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainModel")
                    TextField("Endpoint URL", text: $brainDraftURL)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainURL")
                        .onChange(of: brainDraftURL) { _, _ in
                            brainDraftAllowRemote = false
                        }
                    SecureField("API key (optional)", text: $brainDraftAPIKey)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainKey")
                    Toggle("Text only — do not send screenshots", isOn: $brainDraftTextOnly)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainTextOnly")
                    if !brainDraftURL.isEmpty && !CUAViewModel.isLoopbackEndpoint(brainDraftURL) {
                        Toggle(
                            "Allow this endpoint to receive the task goal, Accessibility snapshot\(brainDraftTextOnly ? "" : ", and screenshot")",
                            isOn: $brainDraftAllowRemote
                        )
                        .accessibilityIdentifier("ComputerUse.Agent.BrainRemoteConsent")
                        Text("External endpoints must use HTTPS. LAN addresses are also treated as external.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if !brainDraftURL.isEmpty {
                        Text("This loopback endpoint runs on this Mac and does not require an API key.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let error = viewModel.brainError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                HStack {
                    Spacer()
                    Button("Cancel") { viewModel.showAddBrain = false }
                        .accessibilityIdentifier("ComputerUse.Agent.BrainCancel")
                    Button("Save Brain") {
                        viewModel.newBrainName = brainDraftName
                        viewModel.newBrainURL = brainDraftURL
                        viewModel.newBrainModel = brainDraftModel
                        viewModel.newBrainAPIKey = brainDraftAPIKey
                        viewModel.newBrainTextOnly = brainDraftTextOnly
                        viewModel.newBrainAllowRemote = brainDraftAllowRemote
                        Task { await viewModel.saveBrain() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(brainDraftName.isEmpty || brainDraftURL.isEmpty || brainDraftModel.isEmpty)
                    .accessibilityIdentifier("ComputerUse.Agent.BrainSave")
                }
            }
            .padding(20)
            .frame(width: 460)
            .onAppear {
                brainDraftName = ""
                brainDraftURL = ""
                brainDraftModel = ""
                brainDraftAPIKey = ""
                brainDraftTextOnly = false
                brainDraftAllowRemote = false
                viewModel.brainError = nil
            }
            .onDisappear {
                // Never reuse a stale draft (an old key must not ride along
                // into a different endpoint).
                brainDraftName = ""
                brainDraftURL = ""
                brainDraftModel = ""
                brainDraftAPIKey = ""
                brainDraftTextOnly = false
                brainDraftAllowRemote = false
            }
        }
    }

    private var targetPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Target window")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.isLoadingApps || viewModel.isLoadingWindows {
                    ProgressView().controlSize(.small)
                }
                Button("Refresh") {
                    Task { await viewModel.loadTargets() }
                }
                .buttonStyle(.borderless)
                .disabled(viewModel.phase.isBusy || viewModel.isLoadingApps)
                .accessibilityIdentifier("ComputerUse.Agent.Target.Refresh")
            }

            HStack(spacing: 10) {
                Picker(
                    "Process",
                    selection: Binding(
                        get: { viewModel.selectedPID },
                        set: { pid in Task { await viewModel.selectApp(pid: pid) } }
                    )
                ) {
                    Text("Choose an app process…").tag(Int?.none)
                    ForEach(viewModel.appOptions) { app in
                        Text(app.displayName).tag(Optional(app.pid))
                    }
                }
                .disabled(viewModel.phase.isBusy || viewModel.isLoadingApps)
                .accessibilityIdentifier("ComputerUse.Agent.Target.Process")

                Picker("Window", selection: $viewModel.selectedWindowID) {
                    Text("Choose a window…").tag(String?.none)
                    ForEach(viewModel.windowOptions) { window in
                        Text(window.displayName).tag(Optional(window.windowID))
                    }
                }
                .disabled(
                    viewModel.phase.isBusy || viewModel.selectedPID == nil
                        || viewModel.isLoadingWindows || viewModel.windowOptions.isEmpty
                )
                .accessibilityIdentifier("ComputerUse.Agent.Target.Window")
            }

            TextField("Browser domain (required for browser targets)", text: $viewModel.allowedDomain)
                .textFieldStyle(.roundedBorder)
                .disabled(viewModel.phase.isBusy || viewModel.selectedApp?.isBrowser != true)
                .accessibilityIdentifier("ComputerUse.Agent.Target.Domain")
            if let domainError = viewModel.selectedBrowserDomainError {
                Text(domainError)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.Target.DomainRequired")
            }

            if !viewModel.selectedTargets.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Authorized windows")
                        .font(.caption.weight(.semibold))
                    ForEach(viewModel.selectedTargets) { target in
                        HStack(alignment: .firstTextBaseline) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(target.displayName).font(.caption)
                                Text(
                                    target.allowedDomain.isEmpty
                                        ? "Domain: not restricted"
                                        : "Domain: \(target.allowedDomain)"
                                )
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Remove") { viewModel.removeSelectedTarget(id: target.id) }
                                .buttonStyle(.borderless)
                                .disabled(viewModel.phase.isBusy)
                                .accessibilityLabel("Remove \(target.displayName)")
                        }
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("ComputerUse.Agent.TargetSet.Item")
                    }
                }
                .accessibilityIdentifier("ComputerUse.Agent.TargetSet")
            }

            Button(viewModel.selectedTargets.isEmpty ? "Authorize selected window" : "Add selected window") {
                viewModel.addSelectedTarget()
            }
            .buttonStyle(.bordered)
            .disabled(!viewModel.canAddSelectedTarget)
            .accessibilityIdentifier("ComputerUse.Agent.Target.Add")

            if viewModel.appOptions.isEmpty, !viewModel.isLoadingApps,
               viewModel.targetError == nil
            {
                Text("No controllable app processes were found. Open an app, then refresh.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("ComputerUse.Agent.Target.Empty")
            }
            if let summary = viewModel.targetSummary {
                Label(summary, systemImage: "macwindow")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("ComputerUse.Agent.Target.Selection")
            }
            if let error = viewModel.targetError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.Target.Error")
            }
        }
        .padding(10)
        .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.Target")
    }

    private var permissionReadiness: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Check Mac permissions", systemImage: "lock.shield")
                .font(.callout.weight(.semibold))
            Text(permissionReadinessMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                ForEach(permissionSettingsLinks, id: \.rawValue) { permission in
                    Button("Open \(permission.title) Settings") {
                        MacAutomationPermissions.openSystemPrivacyPane(for: permission)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier(
                        "ComputerUse.Agent.Permission.\(permission.rawValue)"
                    )
                }
                Button("Refresh") {
                    permissionSnapshot = MacAutomationPermissions.snapshot()
                    Task { await viewModel.loadPermissions() }
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier("ComputerUse.Agent.Permission.Refresh")
            }
        }
        .padding(10)
        .background(.blue.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.PermissionReadiness")
    }

    private var permissionReadinessMessage: String {
        if let permissions = viewModel.executorPermissions {
            if permissions.isReady {
                return "The local executor has Screen Recording and Accessibility access. The app check below may differ because macOS grants access per process."
            }
            return "The local executor needs Screen Recording and Accessibility access. Open System Settings, then refresh the server check."
        }
        return "The server could not report its permission status. The settings links use this app's status as a guide; the executor will check its own access when the task starts."
    }

    private var permissionSettingsLinks: [MacAutomationPermission] {
        if viewModel.executorPermissions?.isReady != true {
            return MacAutomationPermission.allCases
        }
        return permissionSnapshot.missingForComputerUse
    }

    private var activeProgress: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(
                    viewModel.phase == .starting ? "Starting task" : "Task in progress",
                    systemImage: viewModel.phase == .starting ? "hourglass" : "gearshape.2"
                )
                .font(.headline)
                Spacer()
                Text(viewModel.phase == .starting ? "STARTING" : "RUNNING")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.blue.opacity(0.1), in: Capsule())
                Button(
                    viewModel.isRecoveringCreate
                        ? (viewModel.isStopping ? "Recovering…" : "Retry Recovery")
                        : (viewModel.isStopping ? "Stopping…" : "Stop")
                ) {
                    Task { await viewModel.cancel() }
                }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isStopping)
                    .accessibilityIdentifier("ComputerUse.Agent.Stop")
            }
            if let progress = viewModel.activeProgress {
                Text("Step \(progress.step) · limit \(progress.maxSteps): \(progress.instruction)")
                    .font(.callout)
                    .accessibilityIdentifier("ComputerUse.Agent.ActiveStep")
                HStack(spacing: 8) {
                    if let action = progress.action, !action.isEmpty {
                        Text(action).font(.caption.weight(.semibold))
                    }
                    if let target = progress.target, !target.isEmpty {
                        Text(target).font(.caption).foregroundStyle(.secondary)
                    }
                    if let outcome = progress.outcomeLabel {
                        Text(outcome)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(progress.outcome == "success" ? .green : .orange)
                            .accessibilityIdentifier("ComputerUse.Agent.StepOutcome")
                    }
                }
            } else {
                ProgressView().controlSize(.small)
                Text("Preparing the first step…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.blue.opacity(0.2)))
        .accessibilityIdentifier("ComputerUse.Agent.ActiveProgress")
    }

    private func runContextCard(_ context: CUARunContext) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(runContextTitle, systemImage: "scope")
                .font(.callout.weight(.semibold))
            Text(context.goal)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .accessibilityIdentifier("ComputerUse.Agent.RunContext.Goal")
            HStack(spacing: 14) {
                Label(
                    context.targets.isEmpty
                        ? context.targetDisplayName
                        : "Active: \(viewModel.activeTargetDisplayName ?? context.targetDisplayName)",
                    systemImage: "macwindow"
                )
                Label(context.plannerDisplayName, systemImage: "brain")
                Label("Up to \(context.maxSteps) steps", systemImage: "list.number")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if context.targets.count > 1 {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Authorized windows").font(.caption.weight(.semibold))
                    ForEach(context.targets) { target in
                        HStack {
                            Text(target.displayName)
                            Spacer()
                            Text(target.allowedDomain.isEmpty ? "No domain restriction" : target.allowedDomain)
                        }
                        .font(.caption)
                        .foregroundStyle(
                            target.targetID == viewModel.activeTargetID ? .primary : .secondary
                        )
                    }
                }
                .accessibilityIdentifier("ComputerUse.Agent.RunContext.Targets")
            }
        }
        .padding(12)
        .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.secondary.opacity(0.16)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.RunContext")
    }

    private var runContextTitle: String {
        switch viewModel.phase {
        case .starting, .running: "Current task"
        case .awaitingApproval: "Task awaiting approval"
        case .idle: "Stopped task"
        case .finished, .failed: "Task context"
        }
    }

    private func resultCard(summary: String) -> some View {
        let result = CUAResultPresentation(summary: summary)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Label("Task complete", systemImage: "checkmark.circle.fill")
                    .font(.headline)
                    .foregroundStyle(.green)
                Spacer()
                Text("COMPLETED")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.green)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.green.opacity(0.1), in: Capsule())
            }
            Text(result.displayedAnswer)
                .font(.body)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("ComputerUse.Agent.Summary.Answer")
            if let completionNote = result.completionNote {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Completion note")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(completionNote)
                        .font(.callout)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("ComputerUse.Agent.Summary.CompletionNote")
            }
            if result.evidence != nil || !viewModel.events.isEmpty {
                runDetails(evidence: result.evidence, evidenceLabel: result.detailLabel)
            }
        }
        .padding(14)
        .background(.green.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.green.opacity(0.22)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.Summary")
    }

    private func failureCard(message: String) -> some View {
        let failure = CUAFailurePresentation(
            message: message,
            hasExecutedActions: viewModel.wasSessionInterrupted
                || CUAFailurePresentation.hasPotentialSideEffects(in: viewModel.events)
        )
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Label("Task failed", systemImage: "exclamationmark.triangle.fill")
                    .font(.headline)
                    .foregroundStyle(.red)
                Spacer()
                Text("FAILED")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.red.opacity(0.1), in: Capsule())
            }
            Text(failure.summary)
                .font(.body)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("ComputerUse.Agent.Failure.Summary")
            if let changeWarning = failure.changeWarning {
                Label(changeWarning, systemImage: "exclamationmark.circle")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.Failure.ChangeWarning")
            }
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 10) {
                    Text(failure.technicalDetails)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("ComputerUse.Agent.Failure.Details")
                    if !viewModel.events.isEmpty {
                        CUAEventList(
                            events: Array(viewModel.events.reversed()),
                            targetNames: viewModel.runTargetNames
                        )
                    }
                }
                .padding(.top, 6)
            } label: {
                Text(
                    viewModel.events.isEmpty
                        ? "Technical details"
                        : "Technical details and run history (\(viewModel.events.count) events)"
                )
                .font(.caption.weight(.semibold))
            }
            .tint(.secondary)
            .accessibilityIdentifier("ComputerUse.Agent.History")
        }
        .padding(14)
        .background(.red.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.red.opacity(0.2)))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.Failure")
    }

    private func runDetails(
        evidence: String?, evidenceLabel: String = "Supporting evidence"
    ) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 10) {
                if let evidence, !evidence.isEmpty {
                    Text(evidence)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityLabel(evidenceLabel)
                        .accessibilityIdentifier("ComputerUse.Agent.Evidence")
                }
                if !viewModel.events.isEmpty {
                    CUAEventList(
                        events: Array(viewModel.events.reversed()),
                        targetNames: viewModel.runTargetNames
                    )
                }
            }
            .padding(.top, 6)
        } label: {
            Text(detailsLabel(hasEvidence: evidence != nil, evidenceLabel: evidenceLabel))
                .font(.caption.weight(.semibold))
        }
        .tint(.secondary)
        .accessibilityIdentifier("ComputerUse.Agent.History")
    }

    private func detailsLabel(hasEvidence: Bool, evidenceLabel: String) -> String {
        if viewModel.events.isEmpty { return evidenceLabel }
        if hasEvidence { return "\(evidenceLabel) and run history (\(viewModel.events.count) events)" }
        return "Run history (\(viewModel.events.count) events)"
    }

    private var approvalCard: some View {
        let approval = viewModel.pendingApproval ?? CUAPendingApproval(
            gateID: nil,
            app: viewModel.appName,
            action: nil,
            target: nil,
            reason: viewModel.pendingGateReason ?? "Approval is required before Rapid continues."
        )
        return VStack(alignment: .leading, spacing: 8) {
            Label("Approval needed", systemImage: "hand.raised.fill")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.orange)
            LabeledContent("App", value: approval.app)
            if let targetID = approval.targetID,
               let targetName = viewModel.runTargetNames[targetID]
            {
                LabeledContent("Authorized window", value: targetName)
                    .accessibilityIdentifier("ComputerUse.Agent.Approval.TargetWindow")
            }
            if let action = approval.action, !action.isEmpty {
                LabeledContent("Action", value: action)
            }
            if let target = approval.target, !target.isEmpty {
                LabeledContent("Target", value: target)
            }
            LabeledContent("Reason", value: approval.reason)
            Text("Rapid is paused. Approve only after checking the app and requested action.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Button("Approve and Continue") { Task { await viewModel.approve() } }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(!viewModel.canApprove)
                    .accessibilityIdentifier("ComputerUse.Agent.Approve")
                Button(viewModel.isStopping ? "Stopping…" : "Stop Task") {
                    Task { await viewModel.cancel() }
                }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isStopping)
                    .accessibilityIdentifier("ComputerUse.Agent.StopAtApproval")
            }
            if let approvalUnavailableMessage = viewModel.approvalUnavailableMessage {
                Text(approvalUnavailableMessage)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.ApprovalUnavailable")
            }
        }
        .padding(10)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.ApprovalCard")
    }
}

private extension CUAPhase {
    var hasTerminalCard: Bool {
        switch self {
        case .finished, .failed: true
        default: false
        }
    }
}

/// Compact, newest-first step feed (step number, action, instruction, outcome).
struct CUAEventList: View {
    let events: [CUAEvent]
    var targetNames: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(events, id: \.seq) { event in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(eventLabel(for: event))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer()
                    if let outcome = event.outcome {
                        Text(outcomeLabel(outcome))
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(outcome == "success" ? .green : .orange)
                    }
                }
                .accessibilityElement(children: .combine)
            }
        }
        .padding(10)
        .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityIdentifier("ComputerUse.Agent.Events")
    }

    private func eventLabel(for event: CUAEvent) -> String {
        switch event.kind {
        case "plan":
            let instruction = event.stepInstruction ?? event.action ?? "plan"
            return "Step \(event.step ?? 0): \(instruction)"
        case "executed":
            return "Step \(event.step ?? 0) finished"
        case "gate":
            return "Waiting for your approval at a sign-in page"
        case "gate_resolved":
            return "Approval resolved"
        case "target_switched":
            let destination = event.targetID.flatMap { targetNames[$0] }
                ?? event.targetID ?? "authorized window"
            return "Switched to \(destination)"
        case "terminal":
            return "Run \(event.status ?? "ended")"
        default:
            return event.kind
        }
    }

    private func outcomeLabel(_ outcome: String) -> String {
        switch outcome {
        case "success": "Observed expected change"
        case "no_effect": "No effect observed"
        case "wrong_effect": "Unexpected result"
        case "uncertain": "Could not verify"
        case "unavailable": "Verification unavailable"
        default: "Outcome unknown"
        }
    }
}
