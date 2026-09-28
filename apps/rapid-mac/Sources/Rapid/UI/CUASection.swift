import SwiftUI

/// "Run an agent task" — the goal-driven computer-use section at the top of
/// the (experimental) Computer Use panel. Talks to the app-owned local
/// server's `/v1/cua` API; consent gates are enforced server-side, and the
/// only client-side action is approving a sign-in gate.
struct CUASection: View {
    @ObservedObject var viewModel: CUAViewModel
    @State private var brainDraftName = ""
    @State private var brainDraftURL = ""
    @State private var brainDraftModel = ""
    @State private var brainDraftAPIKey = ""
    @State private var brainDraftTextOnly = false

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
                        .help("Add a cloud brain (OpenAI-compatible endpoint)")
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

            HStack(spacing: 12) {
                Button(viewModel.phase.isBusy ? "Stop" : "Start") {
                    Task { await viewModel.toggle() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canStart && !viewModel.phase.isBusy)
                .accessibilityIdentifier("ComputerUse.Agent.StartStop")

                if viewModel.phase == .awaitingApproval {
                    VStack(alignment: .leading, spacing: 6) {
                        Label {
                            Text("Approval needed: \(viewModel.pendingGateReason ?? "sign-in")")
                                .font(.callout.weight(.semibold))
                        } icon: {
                            Image(systemName: "hand.raised.fill").foregroundStyle(.orange)
                        }
                        Text("Rapid paused for your approval. Act only after you verified what is being asked.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Button("Approve") {
                            Task { await viewModel.approve() }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                        .accessibilityIdentifier("ComputerUse.Agent.Approve")
                    }
                    .padding(10)
                    .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                    .accessibilityIdentifier("ComputerUse.Agent.ApprovalCard")
                }
            }

            if case let .finished(summary) = viewModel.phase, !summary.isEmpty {
                Label {
                    Text(summary).font(.callout)
                } icon: {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
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
        }
        .sheet(isPresented: $viewModel.showAddBrain) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Add a cloud brain").font(.headline)
                Text(
                    "Your task goal and a compact screen snapshot are sent to this OpenAI-compatible endpoint. Use HTTPS; the API key is stored locally (0600)."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Form {
                    TextField("Name (e.g. deepseek)", text: $brainDraftName)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainName")
                    TextField("Model (e.g. deepseek-reasoner)", text: $brainDraftModel)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainModel")
                    TextField("Base URL", text: $brainDraftURL)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainURL")
                    SecureField("API key", text: $brainDraftAPIKey)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainKey")
                    Toggle("Text-only (no screenshots)", isOn: $brainDraftTextOnly)
                        .accessibilityIdentifier("ComputerUse.Agent.BrainTextOnly")
                }
                if let error = viewModel.brainError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                HStack {
                    Spacer()
                    Button("Cancel") { viewModel.showAddBrain = false }
                    Button("Save Brain") {
                        viewModel.newBrainName = brainDraftName
                        viewModel.newBrainURL = brainDraftURL
                        viewModel.newBrainModel = brainDraftModel
                        viewModel.newBrainAPIKey = brainDraftAPIKey
                        viewModel.newBrainTextOnly = brainDraftTextOnly
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
            }
        }
    }
}

private extension CUAViewModel {
    func toggle() async {
        if phase.isBusy {
            await cancel()
        } else {
            await start()
        }
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
                        Text(outcome)
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
}
