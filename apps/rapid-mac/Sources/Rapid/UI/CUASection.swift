import SwiftUI

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

            TextEditor(text: $viewModel.goal)
                .frame(minHeight: 52, maxHeight: 96)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(.secondary.opacity(0.2)))
                .accessibilityIdentifier("ComputerUse.Agent.Goal")

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
                    Text("App").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    TextField("Google Chrome", text: $viewModel.appName)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 220)
                        .accessibilityIdentifier("ComputerUse.Agent.App")
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Max steps").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Stepper("\(viewModel.maxSteps)", value: $viewModel.maxSteps, in: 1 ... 40)
                        .accessibilityIdentifier("ComputerUse.Agent.MaxSteps")
                }
            }

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

            if viewModel.phase == .starting || viewModel.phase == .running {
                activeProgress
            }

            if viewModel.phase == .awaitingApproval {
                approvalCard
            }

            if case let .finished(summary) = viewModel.phase, !summary.isEmpty {
                Label {
                    Text("Task ended: \(summary)").font(.callout)
                } icon: {
                    Image(systemName: "flag.checkered").foregroundStyle(.secondary)
                }
                .accessibilityIdentifier("ComputerUse.Agent.Summary")
            }
            if case let .failed(message) = viewModel.phase {
                Label {
                    Text(message).font(.callout)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
                .accessibilityIdentifier("ComputerUse.Agent.Failure")
            }
            if let actionError = viewModel.actionError {
                Label(actionError, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("ComputerUse.Agent.ActionError")
            }

            if !viewModel.events.isEmpty {
                CUAEventList(events: Array(viewModel.events.suffix(8).reversed()))
                if viewModel.events.count > 8 {
                    DisclosureGroup("Full run history (\(viewModel.events.count) events)") {
                        CUAEventList(events: Array(viewModel.events.reversed()))
                            .padding(.top, 4)
                    }
                    .font(.caption)
                    .accessibilityIdentifier("ComputerUse.Agent.History")
                }
            }
        }
        .padding(16)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.secondary.opacity(0.2)))
        .frame(maxWidth: 984, alignment: .leading)
        .task {
            await viewModel.loadPlanners()
            await viewModel.loadPermissions()
            permissionSnapshot = MacAutomationPermissions.snapshot()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                permissionSnapshot = MacAutomationPermissions.snapshot()
                Task { await viewModel.loadPermissions() }
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
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(
                    viewModel.phase == .starting ? "Starting task" : "Task in progress",
                    systemImage: "gearshape.2"
                )
                .font(.callout.weight(.semibold))
                Spacer()
                Button("Stop") { Task { await viewModel.cancel() } }
                    .buttonStyle(.bordered)
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
        .padding(10)
        .background(.blue.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityIdentifier("ComputerUse.Agent.ActiveProgress")
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
                    .accessibilityIdentifier("ComputerUse.Agent.Approve")
                Button("Stop Task") { Task { await viewModel.cancel() } }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("ComputerUse.Agent.StopAtApproval")
            }
        }
        .padding(10)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Agent.ApprovalCard")
    }
}

/// Compact, newest-first step feed (step number, action, instruction, outcome).
struct CUAEventList: View {
    let events: [CUAEvent]

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
