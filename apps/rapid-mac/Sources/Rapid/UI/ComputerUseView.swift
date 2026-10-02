import SwiftUI

/// Hosts the general goal-driven Computer Use task surface and the lifecycle
/// of its app-owned local sidecar. Task planning and target selection live in
/// ``CUASection``; this view owns only page framing and server recovery.
struct ComputerUseView: View {
    @Bindable var cuaServer: CUAServerManager
    let cuaViewModel: CUAViewModel?

    init(
        cuaServer: CUAServerManager,
        cuaViewModel: CUAViewModel? = nil
    ) {
        self.cuaServer = cuaServer
        self.cuaViewModel = cuaViewModel
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                HStack(alignment: .top) {
                    SectionHeader(
                        "Computer Use",
                        subtitle: "Actions run on this Mac using the local or cloud model you choose.",
                        emphasis: .page
                    )
                    Spacer()
                }

                cuaRuntimeSection
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RapidTheme.surfaceCanvas)
        .accessibilityIdentifier("ComputerUse.Panel")
        .onAppear { cuaServer.startIfNeeded() }
    }

    @ViewBuilder
    private var cuaRuntimeSection: some View {
        switch cuaServer.state {
        case .ready:
            if let cuaViewModel {
                CUASection(viewModel: cuaViewModel)
            } else {
                ProgressView().controlSize(.small)
            }
        case .idle, .starting:
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Preparing Computer Use on this Mac…")
                        .foregroundStyle(.secondary)
                }
                .accessibilityIdentifier("ComputerUse.Server.Starting")
                if let cuaViewModel, showsDetachedContext(cuaViewModel) {
                    CUASection(viewModel: cuaViewModel)
                }
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 10) {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Button("Try Again") {
                    Task { await cuaServer.retry() }
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("ComputerUse.Server.Retry")
                if let cuaViewModel, showsDetachedContext(cuaViewModel) {
                    CUASection(viewModel: cuaViewModel)
                }
            }
            .accessibilityIdentifier("ComputerUse.Server.Error")
        }
    }

    private func showsDetachedContext(_ viewModel: CUAViewModel) -> Bool {
        guard viewModel.isSessionDetached else { return false }
        if viewModel.runContext != nil { return true }
        return switch viewModel.phase {
        case .finished, .failed: true
        case .idle, .starting, .running, .awaitingApproval: false
        }
    }
}
