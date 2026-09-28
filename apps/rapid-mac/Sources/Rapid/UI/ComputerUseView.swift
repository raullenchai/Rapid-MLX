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
                        subtitle: "Actions run on this Mac. You choose the brain endpoint used for planning.",
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
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Preparing Computer Use on this Mac…")
                    .foregroundStyle(.secondary)
            }
            .accessibilityIdentifier("ComputerUse.Server.Starting")
        case .failed(let message):
            VStack(alignment: .leading, spacing: 10) {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Button("Try Again") {
                    Task { await cuaServer.retry() }
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("ComputerUse.Server.Retry")
            }
            .accessibilityIdentifier("ComputerUse.Server.Error")
        }
    }
}
