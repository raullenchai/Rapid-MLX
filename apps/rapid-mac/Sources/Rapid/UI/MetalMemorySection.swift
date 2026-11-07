import SwiftUI

/// Poll only while the Performance panel is mounted. A sample belongs to one
/// launch and credential, so restarts and rotation cannot display stale data.
struct MetalMemorySection: View {
    @Environment(ServerManager.self) private var server
    private struct Runtime: Equatable {
        let sessionID: UUID?
        let port: Int
        let bearer: String?
    }
    @State private var sampledRuntime: Runtime?
    @State private var snapshot: MetalMemorySnapshot?

    private var runtime: Runtime {
        Runtime(sessionID: server.activeServerSessionID, port: server.activePort, bearer: server.activeBearer)
    }

    var body: some View {
        let current = runtime
        let visible = sampledRuntime == current ? snapshot : nil
        SettingsSection(String(localized: "Engine Metal memory")) {
            LabeledContent("Active allocations", value: MetalMemorySnapshot.label(visible?.active))
            LabeledContent("Allocator cache", value: MetalMemorySnapshot.label(visible?.cache))
            LabeledContent("Peak allocations", value: MetalMemorySnapshot.label(visible?.peak))
            Text("Live counters for the inference engine, in decimal GB. These are not separate totals for model weights or the prefix cache, and should not be added to system memory use. A dash means the engine has no current measurement.")
                .font(RapidFont.caption)
                .foregroundStyle(.secondary)
        }
        .accessibilityIdentifier("Settings.Performance.MetalMemory")
        .task(id: current) {
            snapshot = nil
            sampledRuntime = current
            guard current.sessionID != nil, let bearer = current.bearer else { return }
            let client = MetalMemoryClient()
            while !Task.isCancelled {
                let next = await client.fetch(port: current.port, bearer: bearer)
                guard !Task.isCancelled, runtime == current else { return }
                snapshot = next
                do { try await Task.sleep(for: .seconds(3)) }
                catch { return }
            }
        }
    }
}
