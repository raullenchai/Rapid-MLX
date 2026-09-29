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
    @State private var brainDraftName = ""
    @State private var brainDraftURL = ""
    @State private var brainDraftModel = ""
    @State private var brainDraftAPIKey = ""
    @State private var brainDraftTextOnly = false
    @State private var brainDraftAllowRemote = false
  @State private var showManageModels = false
  @State private var modelPendingDeletion: CUAPlannerOption?
  @State private var showAdvanced = false
  @State private var addModelAfterManaging = false

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

            if viewModel.executorPermissions?.isReady != true {
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

        modelConfiguration

            Label(viewModel.plannerDisclosure, systemImage: "desktopcomputer")
                .font(.caption)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("ComputerUse.Agent.ModelDisclosure")

            if viewModel.selectedPlannerIsRemote {
                Toggle(
                    "Allow this model to receive this task and the names of open apps to choose where to work",
                    isOn: $viewModel.allowRemoteAppDiscovery
                )
                .font(.caption)
                .accessibilityIdentifier("ComputerUse.Agent.RemoteAppDiscoveryConsent")
            }

        DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
          Stepper("Maximum steps: \(viewModel.maxSteps)", value: $viewModel.maxSteps, in: 1...40)
            .padding(.top, 6)
            .accessibilityIdentifier("ComputerUse.Agent.MaxSteps")
        }
        .font(.caption)
        .accessibilityIdentifier("ComputerUse.Agent.Advanced")

        if let resolution = viewModel.targetResolutionApproval,
          let approval = resolution.approval
        {
          targetResolutionApproval(approval)
            } else if let error = viewModel.targetError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.ScopeError")
                if viewModel.browserAutomationRecoveryRequired {
                    Button("Open Automation Settings") {
                        MacAutomationPermissions.openAutomationSettings()
                    }
                    .accessibilityIdentifier("ComputerUse.Agent.Scope.OpenAutomationSettings")
                }
            }

        if !viewModel.phase.isBusy, viewModel.targetResolutionApproval == nil {
          Button(viewModel.isResolvingTargets ? "Finding the right apps…" : "Start") {
            Task { await viewModel.resolveAndStart() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(
            !viewModel.canResolveTask || viewModel.executorPermissions?.isReady == false
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
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                if !viewModel.isSessionDetached {
                    Task { await viewModel.loadPermissions() }
                }
            }
        }
        .onChange(of: viewModel.goal) { _, _ in
            viewModel.allowRemoteAppDiscovery = false
      if viewModel.targetResolutionApproval != nil {
        viewModel.cancelTargetResolution()
      }
    }
        .onChange(of: viewModel.plannerName) { _, _ in
            viewModel.allowRemoteAppDiscovery = false
      if viewModel.targetResolutionApproval != nil {
        viewModel.cancelTargetResolution()
      }
    }
        .sheet(isPresented: $viewModel.showAddBrain) {
            addModelSheet
        }
    .sheet(isPresented: $showManageModels, onDismiss: {
      if addModelAfterManaging {
        addModelAfterManaging = false
        viewModel.showAddBrain = true
      }
    }) {
      manageModelsSheet
    }
    }

  private var addModelSheet: some View {
    VStack(spacing: 0) {
      ScrollView {
        VStack(alignment: .leading, spacing: 18) {
          VStack(alignment: .leading, spacing: 6) {
            Text("Add Model")
              .font(.title3.weight(.semibold))
            Text(
              "Rapid always executes actions on this Mac. Choose where the planning model runs and exactly what it may receive."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
          }

          VStack(alignment: .leading, spacing: 14) {
            modelField("Name", help: "For example, My Local Model") {
              TextField("My Local Model", text: $brainDraftName)
                .accessibilityIdentifier("ComputerUse.Agent.ModelName")
            }
            modelField("Model name", help: "The model identifier expected by this endpoint") {
              TextField("Model identifier", text: $brainDraftModel)
                .accessibilityIdentifier("ComputerUse.Agent.ModelNameAtEndpoint")
            }
            modelField("Endpoint URL", help: "A loopback URL on this Mac or an HTTPS endpoint") {
              TextField("https://example.com/v1", text: $brainDraftURL)
                .accessibilityIdentifier("ComputerUse.Agent.ModelURL")
                .onChange(of: brainDraftURL) { _, _ in
                  brainDraftAllowRemote = false
                }
            }
            modelField("API key", help: "Optional. Stored only in your local Computer Use configuration.") {
              SecureField("Optional", text: $brainDraftAPIKey)
                .accessibilityIdentifier("ComputerUse.Agent.ModelKey")
            }
          }

          VStack(alignment: .leading, spacing: 12) {
            Toggle("Text only — do not send screenshots", isOn: $brainDraftTextOnly)
              .fixedSize(horizontal: false, vertical: true)
              .accessibilityIdentifier("ComputerUse.Agent.ModelTextOnly")

            if !brainDraftURL.isEmpty && !CUAViewModel.isLoopbackEndpoint(brainDraftURL) {
              Toggle(isOn: $brainDraftAllowRemote) {
                Text(
                  "Allow this endpoint to receive the task goal and, during execution, the Accessibility snapshot\(brainDraftTextOnly ? "" : ", and screenshot")"
                )
                .fixedSize(horizontal: false, vertical: true)
              }
              .accessibilityIdentifier("ComputerUse.Agent.ModelRemoteConsent")
              Text("External endpoints must use HTTPS. LAN addresses are also treated as external.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            } else if !brainDraftURL.isEmpty {
              Text("This loopback endpoint runs on this Mac and does not require an API key.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
          }

          if let error = viewModel.brainError {
            Text(error)
              .font(.caption)
              .foregroundStyle(.red)
              .fixedSize(horizontal: false, vertical: true)
          }
        }
        .padding(24)
      }

      Divider()
      HStack(spacing: 10) {
        Spacer()
        Button("Cancel") { viewModel.showAddBrain = false }
          .keyboardShortcut(.cancelAction)
          .accessibilityIdentifier("ComputerUse.Agent.ModelCancel")
        Button("Save Model") {
          viewModel.newBrainName = brainDraftName
          viewModel.newBrainURL = brainDraftURL
          viewModel.newBrainModel = brainDraftModel
          viewModel.newBrainAPIKey = brainDraftAPIKey
          viewModel.newBrainTextOnly = brainDraftTextOnly
          viewModel.newBrainAllowRemote = brainDraftAllowRemote
          Task { await viewModel.saveBrain() }
        }
        .buttonStyle(.borderedProminent)
        .keyboardShortcut(.defaultAction)
        .disabled(brainDraftName.isEmpty || brainDraftURL.isEmpty || brainDraftModel.isEmpty)
        .accessibilityIdentifier("ComputerUse.Agent.ModelSave")
      }
      .padding(.horizontal, 24)
      .padding(.vertical, 14)
    }
    .frame(minWidth: 420, idealWidth: 520, maxWidth: 620, minHeight: 460, idealHeight: 520)
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
      brainDraftName = ""
      brainDraftURL = ""
      brainDraftModel = ""
      brainDraftAPIKey = ""
      brainDraftTextOnly = false
      brainDraftAllowRemote = false
    }
  }

  private func modelField<Content: View>(
    _ label: String, help: String, @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 5) {
      Text(label)
        .font(.callout.weight(.medium))
      content()
        .textFieldStyle(.roundedBorder)
      Text(help)
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }

  private func targetResolutionApproval(_ approval: CUATargetApproval) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Label("Confirm app access", systemImage: "macwindow.on.rectangle")
        .font(.callout.weight(.semibold))
      Text(approval.prompt)
        .font(.callout)
            HStack {
        Button("Cancel") { viewModel.cancelTargetResolution() }
          .accessibilityIdentifier("ComputerUse.Agent.ScopeCancel")
                Spacer()
        ForEach(approval.options) { option in
          Button(option.label) {
            Task { await viewModel.approveResolvedTargets(optionID: option.optionID) }
                }
          .buttonStyle(.borderedProminent)
          .accessibilityIdentifier("ComputerUse.Agent.ScopeOption")
                }
      }
    }
    .padding(12)
    .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: 10))
    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.secondary.opacity(0.2)))
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("ComputerUse.Agent.ScopeApproval")
            }

  private var modelConfiguration: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text("Model")
        .font(.caption.weight(.semibold))
        .foregroundStyle(.secondary)
      HStack(spacing: 8) {
        Picker("Model", selection: $viewModel.plannerName) {
          ForEach(viewModel.plannerOptions) { option in
            Text(option.displayName).tag(option.name)
          }
          if viewModel.plannerOptions.isEmpty {
            Text("Add a model to continue").tag("")
                    }
                }
        .labelsHidden()
        .frame(maxWidth: 360)
        .disabled(viewModel.plannerOptions.isEmpty)
        .accessibilityIdentifier("ComputerUse.Agent.Model")

        Button("Add Model…") { viewModel.showAddBrain = true }
          .accessibilityIdentifier("ComputerUse.Agent.AddModel")
        Button("Manage Models…") { showManageModels = true }
          .disabled(viewModel.plannerOptions.isEmpty)
          .accessibilityIdentifier("ComputerUse.Agent.ManageModels")
                    }
                }
            }

  private var manageModelsSheet: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Manage Models").font(.headline)
      Text("Choose the local or cloud OpenAI-compatible endpoint Rapid uses to plan this task.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
      List(viewModel.plannerOptions) { model in
                        HStack(alignment: .firstTextBaseline) {
          VStack(alignment: .leading, spacing: 3) {
            Text(model.displayName)
            Text(model.url)
              .font(.caption)
                                .foregroundStyle(.secondary)
              .lineLimit(1)
                            }
                            Spacer()
          if model.userCreated {
            Button("Remove…", role: .destructive) {
              modelPendingDeletion = model
                                }
            .accessibilityLabel("Remove \(model.displayName)")
            .accessibilityIdentifier("ComputerUse.Agent.ModelRemove")
          } else {
            Text("Built in")
              .font(.caption)
              .foregroundStyle(.secondary)
                    }
                }
            }
      if let error = viewModel.brainError {
        Text(error).font(.caption).foregroundStyle(.red)
            }
      HStack {
        Button("Add Model…") {
          addModelAfterManaging = true
          showManageModels = false
        }
        .accessibilityIdentifier("ComputerUse.Agent.ModelsAdd")
        Spacer()
        Button("Done") { showManageModels = false }
          .keyboardShortcut(.defaultAction)
          .accessibilityIdentifier("ComputerUse.Agent.ModelsDone")
            }
            }
    .padding(20)
    .frame(width: 520, height: 360)
    .accessibilityIdentifier("ComputerUse.Agent.Models")
    .confirmationDialog(
      "Remove this model?",
      isPresented: Binding(
        get: { modelPendingDeletion != nil },
        set: { if !$0 { modelPendingDeletion = nil } }
      ),
      presenting: modelPendingDeletion
    ) { model in
      Button("Remove \(model.name)", role: .destructive) {
        Task {
          await viewModel.deleteModel(named: model.name)
          modelPendingDeletion = nil
        }
      }
      .accessibilityIdentifier("ComputerUse.Agent.ModelRemoveConfirm")
      Button("Cancel", role: .cancel) { modelPendingDeletion = nil }
        .accessibilityIdentifier("ComputerUse.Agent.ModelRemoveCancel")
    } message: { model in
      Text(
        "Rapid will remove the saved endpoint for \(model.displayName). You can add it again later."
      )
        }
    }

    private var permissionReadiness: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Check Mac permissions", systemImage: "lock.shield")
                .font(.callout.weight(.semibold))
            Text(permissionReadinessMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                ForEach(permissionSettingsLinks, id: \.rawValue) { permission in
                    if viewModel.canRequestPermission(permission) {
                        Button(
                            viewModel.permissionRequestInFlight == permission
                                ? "Requesting…" : "Allow \(permission.title)…"
                        ) {
                            Task { await viewModel.requestPermission(permission) }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.permissionRequestInFlight != nil)
                        .accessibilityIdentifier(
                            "ComputerUse.Agent.Permission.Request.\(permission.rawValue)"
                        )
                    }
                    Button("Open Settings") {
                        MacAutomationPermissions.openSystemPrivacyPane(for: permission)
                    }
                    .buttonStyle(.bordered)
                    .help("Open \(permission.title) in System Settings")
                    .accessibilityIdentifier(
                        "ComputerUse.Agent.Permission.Settings.\(permission.rawValue)"
                    )
                }
                Button("Refresh") {
                    Task { await viewModel.loadPermissions() }
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier("ComputerUse.Agent.Permission.Refresh")
            }
            if let message = viewModel.permissionRequestMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("ComputerUse.Agent.Permission.Message")
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
                return "Rapid-MLX Desktop has Screen Recording access, and its bundled helper has Accessibility access for Computer Use."
            }
            if viewModel.supportsPermissionRequest {
                return "Computer Use needs Screen Recording for Rapid-MLX Desktop and Accessibility for its bundled helper. Each Allow button asks macOS for the app that uses that permission."
            }
            return "Computer Use needs Screen Recording for Rapid-MLX Desktop and Accessibility for its bundled helper. Allow each app in System Settings, then refresh."
        }
        return "Rapid Computer Use could not report its permission status. Review Rapid-MLX Desktop in System Settings, then refresh."
    }

    private var permissionSettingsLinks: [MacAutomationPermission] {
        guard let permissions = viewModel.executorPermissions else {
            return MacAutomationPermission.allCases
        }
        return MacAutomationPermission.allCases.filter {
            !permissions.isGranted($0)
        }
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
                Label(context.plannerDisplayName, systemImage: "cpu")
                Label("Up to \(context.maxSteps) steps", systemImage: "list.number")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if context.targets.isEmpty, !context.allowedDomain.isEmpty {
                Label("Reviewed domain: \(context.allowedDomain)", systemImage: "network")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .accessibilityIdentifier("ComputerUse.Agent.RunContext.Domain")
            }
            if context.targets.count > 1 {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Apps used").font(.caption.weight(.semibold))
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
            if viewModel.browserAutomationRecoveryRequired {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Browser access is not allowed")
                        .font(.callout.weight(.semibold))
                    Text(
                        "The task stopped because Rapid could not verify the selected browser's domain. Allow Rapid to control that browser in System Settings, then choose New Task and try again."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    Button("Open Automation Settings") {
                        MacAutomationPermissions.openAutomationSettings()
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("ComputerUse.Agent.Failure.OpenAutomationSettings")
                }
                .padding(10)
                .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("ComputerUse.Agent.Failure.AutomationRecovery")
            }
            if viewModel.canRetrySetup {
                VStack(alignment: .leading, spacing: 6) {
                    Button("Retry setup") {
                        Task { await viewModel.retrySetup() }
                    }
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("ComputerUse.Agent.Failure.RetrySetup")
                    Text(
                        "Restores this task as a draft. Rapid will choose the apps again and ask before accessing a website."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("ComputerUse.Agent.Failure.RetrySetupHelp")
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
            if let context = viewModel.runContext, !context.allowedDomain.isEmpty {
                LabeledContent("Reviewed domain", value: context.allowedDomain)
                    .accessibilityIdentifier("ComputerUse.Agent.Approval.Domain")
            }
            if let targetID = approval.targetID,
               let targetName = viewModel.runTargetNames[targetID]
            {
                LabeledContent("App in use", value: targetName)
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
                ?? "another app"
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
