import SwiftUI

struct ComputerUseView: View {
    @State private var showingDraftPost = false

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
            DraftPostFlowSheet()
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
                    showingDraftPost = true
                }
                .buttonStyle(.rapidPrimaryCompact)
                .accessibilityIdentifier("ComputerUse.Starter.DraftAndPost.Start")
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
    @State private var viewModel = DraftPostFlowViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Draft and post an update")
                        .font(.title2.weight(.semibold))
                    Text("Rapid copies a TextEdit draft into an empty browser composer, verifies it, and stops before publishing.")
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
                    Text("Finding available TextEdit and browser windows…")
                }

            case .ready:
                setup

            case .running:
                VStack(alignment: .leading, spacing: 10) {
                    ProgressView()
                    Text("Reading, transferring, and verifying the draft…")
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
                    message: "The browser composer matches the TextEdit draft. Review it in the browser and publish it yourself when ready.",
                    symbol: "checkmark.shield.fill",
                    color: .green,
                    metrics: metrics
                )

            case .failed(let failure, let metrics):
                result(
                    title: "Rapid paused safely",
                    message: failure.userMessage,
                    symbol: "pause.circle.fill",
                    color: .orange,
                    metrics: metrics
                )
                HStack {
                    Button("Refresh windows") {
                        Task { await viewModel.load() }
                    }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.DraftPost.RefreshAfterFailure")
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
        .frame(width: 620, height: 450)
        .task { await viewModel.load() }
        .onDisappear { viewModel.stop() }
        .interactiveDismissDisabled(viewModel.isActive)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Draft and post setup")
    }

    private var setup: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("No screenshots or draft text are stored.", systemImage: "lock.shield")
                .font(.callout)
                .foregroundStyle(.secondary)

            picker(
                title: "1. TextEdit draft",
                prompt: "Choose a TextEdit window",
                options: viewModel.sourceOptions,
                selection: $viewModel.sourceID,
                identifier: "ComputerUse.DraftPost.Source"
            )
            picker(
                title: "2. Signed-in browser composer",
                prompt: "Choose a browser window",
                options: viewModel.destinationOptions,
                selection: $viewModel.destinationID,
                identifier: "ComputerUse.DraftPost.Destination"
            )

            if viewModel.sourceOptions.isEmpty || viewModel.destinationOptions.isEmpty {
                Text("Open the draft in TextEdit and an empty English-language post composer in Safari, then refresh. Leave both selected windows unchanged until Rapid stops.")
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
                Button("Run locally") { viewModel.run() }
                    .buttonStyle(.rapidPrimaryCompact)
                    .disabled(!viewModel.canRun)
                    .accessibilityIdentifier("ComputerUse.DraftPost.Run")
            }
        }
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
