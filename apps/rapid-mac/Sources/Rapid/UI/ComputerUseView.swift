import SwiftUI

struct ComputerUseView: View {
    let languageRuntime: DraftPostLanguageRuntime?
    let visualRuntime: DraftPostVisualRuntime?
    @State private var showingDraftPost = false
    @State private var showingFreeUpSpace = false

    init(
        languageRuntime: DraftPostLanguageRuntime? = nil,
        visualRuntime: DraftPostVisualRuntime? = nil
    ) {
        self.languageRuntime = languageRuntime
        self.visualRuntime = visualRuntime
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                HStack(alignment: .top) {
                    SectionHeader(
                        "Computer Use",
                        subtitle: "Let Rapid handle useful work across apps on this Mac. Everything runs locally.",
                        emphasis: .page
                    )
                    Spacer()
                    Text("EXPERIMENTAL")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(.orange.opacity(0.1), in: Capsule())
                }

                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    Text("Start with a flow").font(.headline)
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 260, maximum: 320), spacing: 12)],
                        spacing: 12
                    ) {
                        ForEach(ComputerUseStarter.catalog) { starter in
                            starterCard(starter)
                        }
                    }
                    // Three cards at their maximum width plus two gaps. The
                    // adaptive grid can still collapse to two or one column.
                    .frame(maxWidth: 984, alignment: .leading)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("CREATE YOUR OWN")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 14) {
                        Image(systemName: "record.circle")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Teach Rapid a new task").font(.headline)
                            Text("Show Rapid how you work when no starter fits. Planned for the next Computer Use preview.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Coming next") {}
                            .buttonStyle(.rapidSecondaryCompact)
                            .disabled(true)
                            .accessibilityIdentifier("ComputerUse.Teach.ComingNext")
                    }
                    .padding(16)
                    .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.secondary.opacity(0.2)))
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RapidTheme.surfaceCanvas)
        .accessibilityIdentifier("ComputerUse.Panel")
        .sheet(isPresented: $showingDraftPost) {
            DraftPostFlowSheet(
                languageRuntime: languageRuntime,
                visualRuntime: visualRuntime
            )
        }
        .sheet(isPresented: $showingFreeUpSpace) {
            FreeUpSpaceFlowSheet()
        }
    }

    private func starterCard(_ starter: ComputerUseStarter) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: starter.systemImage)
                    .font(.title2)
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                Spacer()
                Text(availabilityLabel(starter.availability))
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
            Text(starter.title).font(.headline)
            Text(starter.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(starter.applications)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            Label(starter.approvalNote, systemImage: "checkmark.shield")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if starter.availability == .available {
                Button("Start flow") {
                    start(starter.kind)
                }
                .buttonStyle(.rapidPrimaryCompact)
                .accessibilityIdentifier(startIdentifier(starter.kind))
            }
        }
        .frame(maxWidth: .infinity, minHeight: 118, alignment: .topLeading)
        .padding(16)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(
                    .secondary.opacity(starter.availability == .reserved ? 0.28 : 0.16),
                    style: StrokeStyle(
                        lineWidth: 1,
                        dash: starter.availability == .reserved ? [5] : []
                    )
                )
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ComputerUse.Starter.\(starter.kind.rawValue)")
        .accessibilityLabel(starter.title)
    }

    private func start(_ kind: ComputerUseStarter.Kind) {
        switch kind {
        case .freeUpSpace:
            showingFreeUpSpace = true
        case .draftAndPost:
            showingDraftPost = true
        case .tidyInbox, .prospectCustomers, .createDemoVideo, .reserved:
            break
        }
    }

    private func startIdentifier(_ kind: ComputerUseStarter.Kind) -> String {
        switch kind {
        case .freeUpSpace:
            "ComputerUse.Starter.FreeUpSpace.Start"
        case .draftAndPost:
            "ComputerUse.Starter.DraftAndPost.Start"
        case .tidyInbox, .prospectCustomers, .createDemoVideo, .reserved:
            "ComputerUse.Starter.Unavailable.Start"
        }
    }

    private func availabilityLabel(
        _ availability: ComputerUseStarter.Availability
    ) -> String {
        switch availability {
        case .available: "PREVIEW"
        case .comingSoon: "COMING NEXT"
        case .reserved: "RESERVED"
        }
    }
}

private struct DraftPostFlowSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var viewModel: DraftPostInstructionFlowViewModel
    private let languageRuntime: DraftPostLanguageRuntime?
    private let visualRecoveryAvailable: Bool

    init(
        languageRuntime: DraftPostLanguageRuntime?,
        visualRuntime: DraftPostVisualRuntime?
    ) {
        self.languageRuntime = languageRuntime
        self.visualRecoveryAvailable = visualRuntime != nil
        _viewModel = State(initialValue: DraftPostInstructionFlowViewModel(
            planner: languageRuntime?.makePlanner(),
            driver: MacOSDraftPostFlowDriver(
                visualRecovery: visualRuntime?.makeRecovery()
            )
        ))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Draft and post an update")
                        .font(.title2.weight(.semibold))
                    Text("Describe what you want to say. Rapid drafts it locally, fills the selected browser, and stops before publishing.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Close") {
                    dismiss()
                }
                    .disabled(viewModel.isActive)
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.DraftPost.Close")
            }

            Divider()

            switch viewModel.phase {
            case .loading:
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Finding available browser windows…")
                }

            case .ready:
                requestForm

            case .analyzing:
                VStack(alignment: .leading, spacing: 10) {
                    ProgressView()
                    Text("Understanding your request and drafting a plan…")
                        .font(.headline)
                    Text("This runs on the selected local model. Rapid has not touched the browser.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Stop") { viewModel.stop() }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ComputerUse.DraftPost.StopAnalysis")
                }

            case .reviewing:
                planReview

            case .running:
                VStack(alignment: .leading, spacing: 10) {
                    ProgressView()
                    Text("Filling and verifying the draft…")
                        .font(.headline)
                    Text("Rapid may bring each selected window forward. It will retry a safe step at most twice.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Stop") { viewModel.stop() }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ComputerUse.DraftPost.Stop")
                }

            case .stopping:
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Stopping at a safe boundary…")
                }

            case .readyForReview(let metrics):
                result(
                    title: "Ready for your review",
                    message: "The browser composer matches the draft you reviewed. Check it in the browser and publish it yourself when ready.",
                    symbol: "checkmark.shield.fill",
                    color: .green,
                    metrics: metrics
                )

            case .planningFailed(let error):
                result(
                    title: "Rapid needs a clearer request",
                    message: error.userMessage,
                    symbol: "text.bubble.fill",
                    color: .orange,
                    metrics: nil
                )
                Button("Edit request") { viewModel.editRequest() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.DraftPost.EditAfterPlanningFailure")
                if error == .permissionMissing {
                    HStack {
                        Button("Allow Screen Recording") {
                            _ = MacAutomationPermissions.request(.screenRecording)
                            Task { await viewModel.load() }
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier(
                            "ComputerUse.DraftPost.AllowScreenRecordingForPlanning"
                        )
                        Button("Allow Accessibility") {
                            _ = MacAutomationPermissions.request(.accessibility)
                            Task { await viewModel.load() }
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier(
                            "ComputerUse.DraftPost.AllowAccessibilityForPlanning"
                        )
                    }
                }

            case .executionFailed(let failure, let metrics):
                result(
                    title: "Rapid paused safely",
                    message: failure.userMessage,
                    symbol: "pause.circle.fill",
                    color: .orange,
                    metrics: metrics
                )
                HStack {
                    if viewModel.plan != nil, failure.permitsReviewedRetry {
                        Button("Back to plan") { viewModel.returnToPlan() }
                            .buttonStyle(.rapidSecondaryCompact)
                            .accessibilityIdentifier("ComputerUse.DraftPost.BackToPlan")
                    } else {
                        Button(viewModel.plan == nil ? "Refresh windows" : "Start over") {
                            Task { await viewModel.load() }
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier(
                            viewModel.plan == nil
                                ? "ComputerUse.DraftPost.RefreshAfterFailure"
                                : "ComputerUse.DraftPost.StartOverAfterFailure"
                        )
                    }
                    if failure == .permissionMissing {
                        Button("Allow Screen Recording") {
                            _ = MacAutomationPermissions.request(.screenRecording)
                            Task { await viewModel.load() }
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ComputerUse.DraftPost.AllowScreenRecording")
                        Button("Allow Accessibility") {
                            _ = MacAutomationPermissions.request(.accessibility)
                            Task { await viewModel.load() }
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ComputerUse.DraftPost.AllowAccessibility")
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(24)
        .frame(width: 700, height: 650)
        .task { await viewModel.load() }
        .onChange(of: languageRuntime?.viewIdentity) { _, _ in
            // Keep the in-flight/reviewed flow intact. An analysis captures
            // its planner before it starts; replacing this reference affects
            // only the next analysis request and never browser execution.
            viewModel.updatePlanner(languageRuntime?.makePlanner())
        }
        .onDisappear { viewModel.cancelTask() }
        .interactiveDismissDisabled(viewModel.isActive)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Draft and post flow")
    }

    private var requestForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text("What would you like Rapid to draft?")
                    .font(.headline)
                TextEditor(text: $viewModel.instruction)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(10)
                    .frame(minHeight: 150)
                    .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(.secondary.opacity(0.25))
                    )
                    .accessibilityIdentifier("ComputerUse.DraftPost.Instruction")
                Text("Include the purpose, audience, key points, tone, and destination site. Rapid will ask one question if something essential is missing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let question = viewModel.clarificationQuestion {
                Label(question, systemImage: "questionmark.bubble")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.orange)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                    .accessibilityIdentifier("ComputerUse.DraftPost.Clarification")
            }

            HStack(spacing: 8) {
                hint("Purpose")
                hint("Audience")
                hint("Key points")
                hint("Tone")
                hint("Destination")
            }

            Label(
                visualRecoveryAvailable
                    ? "Visual recovery is ready with the current local model."
                    : "Accessibility mode is ready. A compatible local Computer Use model adds visual recovery for unlabeled composers.",
                systemImage: visualRecoveryAvailable ? "eye.circle.fill" : "eye.slash"
            )
            .font(.caption)
            .foregroundStyle(.secondary)

            picker(
                title: "Signed-in browser destination",
                prompt: "Choose a browser window",
                options: viewModel.destinationOptions,
                selection: $viewModel.destinationID,
                identifier: "ComputerUse.DraftPost.Destination"
            )

            if viewModel.destinationOptions.isEmpty {
                Text("Open an empty English-language composer in Safari or Google Chrome, then refresh. Leave the selected window unchanged until Rapid stops.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            if !viewModel.hasPlanner {
                Text("Use a running local chat model with per-start authentication before asking Rapid to analyze the request.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            HStack {
                Button("Refresh windows") {
                    Task { await viewModel.load() }
                }
                .buttonStyle(.rapidSecondaryCompact)
                .accessibilityIdentifier("ComputerUse.DraftPost.Refresh")
                Spacer()
                Button("Analyze request") { viewModel.analyze() }
                    .buttonStyle(.rapidPrimaryCompact)
                    .disabled(!viewModel.canAnalyze)
                    .accessibilityIdentifier("ComputerUse.DraftPost.Analyze")
            }
        }
    }

    private var planReview: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let plan = viewModel.plan {
                HStack(alignment: .top, spacing: 14) {
                    planField("Purpose", plan.purpose)
                    planField("Audience", plan.audience)
                    planField("Tone", plan.tone)
                }

                VStack(alignment: .leading, spacing: 5) {
                    Text("Key points").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    ForEach(Array(plan.talkingPoints.enumerated()), id: \.offset) { _, point in
                        Label(point, systemImage: "checkmark.circle")
                            .font(.callout)
                    }
                }

                VStack(alignment: .leading, spacing: 5) {
                    Label("Intended destination: \(plan.destination)", systemImage: "safari")
                    if let window = viewModel.plannedDestinationDisplayName {
                        Label("Browser window: \(window)", systemImage: "macwindow")
                    }
                    Label(DraftPostPlan.stopCondition, systemImage: "checkmark.shield")
                        .foregroundStyle(.secondary)
                }
                .font(.caption)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Draft").font(.headline)
                    TextEditor(text: $viewModel.editableDraft)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .padding(10)
                        .frame(minHeight: 170)
                        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(.secondary.opacity(0.25))
                        )
                        .accessibilityIdentifier("ComputerUse.DraftPost.Draft")
                    Text("You can edit this before Rapid touches the browser.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                HStack {
                    Button("Edit request") { viewModel.editRequest() }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ComputerUse.DraftPost.EditRequest")
                    Spacer()
                    Button("Fill in browser") { viewModel.execute() }
                        .buttonStyle(.rapidPrimaryCompact)
                        .disabled(!viewModel.canExecute)
                        .accessibilityIdentifier("ComputerUse.DraftPost.Execute")
                }
            }
        }
    }

    private func hint(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.secondary.opacity(0.08), in: Capsule())
    }

    private func planField(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text(value).font(.callout).lineLimit(3)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func picker(
        title: String,
        prompt: String,
        options: [ComputerUseWindowOption],
        selection: Binding<String?>,
        identifier: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            Picker(prompt, selection: selection) {
                Text(prompt).tag(String?.none)
                ForEach(options) { option in
                    Text(option.displayName).tag(Optional(option.id))
                }
            }
            .labelsHidden()
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityIdentifier(identifier)
        }
    }

    private func result(
        title: String,
        message: String,
        symbol: String,
        color: Color,
        metrics: DraftPostFlowMetrics?
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: symbol)
                .font(.headline)
                .foregroundStyle(color)
            Text(message).font(.callout)
            if let metrics {
                HStack(spacing: 20) {
                    metric("Attempts", metrics.attempts)
                    metric("Auto-recoveries", metrics.automaticRecoveries)
                    metric("Verified steps", metrics.completedSteps)
                }
                .padding(12)
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func metric(_ title: String, _ value: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(value)").font(.headline.monospacedDigit())
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }
}
