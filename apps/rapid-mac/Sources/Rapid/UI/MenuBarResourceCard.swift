import SwiftUI

/// Endpoint-authored memory accounting; the physical RAM denominator is not
/// the engine's admission budget. No allocator/cache counters are added here.
@MainActor
struct MenuBarResourceModel {
    struct Row: Identifiable {
        let id: String
        let name: String
        let memoryLabel: String
        let canUnload: Bool
        let kept: Bool
    }

    let residency: ModelResidencySnapshot?
    let totalBytes: UInt64?
    let status: MetalMemorySnapshot?

    var rows: [Row] {
        (residency?.models ?? []).filter(\.isHoldingMemory).map { model in
            Row(id: model.id, name: ModelDisplayName.name(for: model.displayName()),
                memoryLabel: model.state == "loading" ? String(localized: "Reloading…") : MemoryOverviewFormat.bytes(model.displayBytes),
                canUnload: model.state == "resident" && !model.pinned && !model.primary && model.activeRequests == 0,
                kept: model.pinned || model.primary)
        }
    }

    var audioRows: [Row] {
        (residency?.audioLanes ?? []).filter { $0.state == "resident" }.map { lane in
            Row(id: lane.lane, name: SidebarView.audioLaneDisplayName(model: lane.model, lane: lane.lane, catalog: []),
                memoryLabel: "—", canUnload: false, kept: true)
        }
    }

    var memoryLabel: String {
        "\(MemoryOverviewFormat.bytes(residency?.memoryUsedBytes)) / \(MemoryOverviewFormat.bytes(totalBytes))"
    }

    var unloadAllLabel: String {
        let freed = MemoryOverviewFormat.bytes(residency?.memoryUsedBytes ?? 0)
        return rows.count + audioRows.count == 1
            ? String(localized: "Unload model and free \(freed)")
            : String(localized: "Unload all models and free \(freed)")
    }

    var fraction: Double? {
        guard let residency, let totalBytes, totalBytes > 0 else { return nil }
        return min(1, Double(residency.memoryUsedBytes) / Double(totalBytes))
    }

    var safety: ModelSizing.MemorySafety? {
        guard let residency, let totalBytes, totalBytes > 0 else { return nil }
        return ModelSizing.memorySafety(footprintGB: 0, usedBytes: residency.memoryUsedBytes, totalBytes: totalBytes)
    }

    var isActive: Bool {
        residency?.hasActiveRequests == true || (status?.numRunning ?? 0) > 0 || (status?.numWaiting ?? 0) > 0
    }

    var activityLabel: String {
        guard isActive else {
            return residency == nil && status?.numRunning == nil ? "—" : String(localized: "Idle")
        }
        guard let rate = MemoryOverviewFormat.valid(status?.generationTPS), (status?.numRunning ?? 0) > 0 else {
            return String(localized: "Generating")
        }
        let speed = String(localized: "\(String(format: "%.1f", rate)) tok/s")
        return String(localized: "Generating") + " · " + speed
    }

    var canUnloadAll: Bool {
        guard let residency else { return false }
        return (!rows.isEmpty || !audioRows.isEmpty) && !isActive
            && !residency.models.contains(where: \.isTransitioning)
            && !residency.audioLanes.contains { $0.state == "resident" && $0.activeRequests == nil }
    }
}

/// Lives only for one open menu. Capture the runtime before suspension and
/// discard results if it changes, including a restart on the same port.
@MainActor @Observable
final class MenuBarResourceMonitor {
    private(set) var residency: ModelResidencySnapshot?
    private(set) var status: MetalMemorySnapshot?
    private(set) var totalBytes: UInt64?
    private(set) var sampledSession: MemoryOverviewSession?
    var notice: String?
    private(set) var unloading = false
    private let residencyClient: ServerResidencyClient
    private let statusClient: MetalMemoryClient

    init(residencyClient: ServerResidencyClient? = nil, statusClient: MetalMemoryClient = MetalMemoryClient()) {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 3
        config.timeoutIntervalForResource = 3
        self.residencyClient = residencyClient ?? ServerResidencyClient(session: URLSession(configuration: config))
        self.statusClient = statusClient
    }

    static func session(for server: ServerManager) -> MemoryOverviewSession {
        MemoryOverviewSession(sessionID: server.activeServerSessionID, port: server.activePort,
                              bearer: server.activeBearer, model: server.servingAlias)
    }

    func model(for server: ServerManager) -> MenuBarResourceModel {
        if server.state == .idle || server.state == .stopped {
            return MenuBarResourceModel(residency: .empty, totalBytes: totalBytes, status: nil)
        }
        guard sampledSession == Self.session(for: server), case .ready = server.state else {
            return MenuBarResourceModel(residency: nil, totalBytes: totalBytes, status: nil)
        }
        return MenuBarResourceModel(residency: residency, totalBytes: totalBytes, status: status)
    }

    func refresh(server: ServerManager) async {
        let current = Self.session(for: server)
        if sampledSession != current {
            residency = nil
            status = nil
            notice = nil
        }
        totalBytes = ProcessInfo.processInfo.physicalMemory
        guard case .ready = server.state, let bearer = current.bearer else {
            residency = nil
            status = nil
            sampledSession = current
            return
        }
        async let nextResidency = residencyClient.fetch(port: current.port, bearer: bearer)
        async let nextStatus = statusClient.fetch(port: current.port, bearer: bearer)
        let values = await (nextResidency, nextStatus)
        guard !Task.isCancelled, current == Self.session(for: server), case .ready = server.state else { return }
        sampledSession = current
        residency = values.0
        status = values.1
    }

    func unload(id: String?, server: ServerManager) async {
        guard !unloading, !server.isOperating, server.residentLoadsInFlight.isEmpty else { return }
        unloading = true
        defer { unloading = false }
        notice = nil
        await refresh(server: server)
        guard !Task.isCancelled, !server.isOperating, server.residentLoadsInFlight.isEmpty else { return }
        let current = Self.session(for: server)
        let model = model(for: server)
        if let id {
            guard model.rows.contains(where: { $0.id == id && $0.canUnload }) else { return }
            let success = await residencyClient.unload(id: id, port: current.port, bearer: current.bearer)
            guard !Task.isCancelled, current == Self.session(for: server) else { return }
            if !success {
                notice = String(localized: "Rapid couldn't confirm that the models were idle. Wait a moment, then try again.")
            }
            await server.refreshResidency()
        } else {
            guard model.canUnloadAll else { return }
            notice = SidebarView.residentUnloadMessage(for: await server.unloadResidentModelsIfIdle())
        }
        await refresh(server: server)
    }
}

struct MenuBarResourceCard: View {
    let server: ServerManager
    let monitor: MenuBarResourceMonitor

    private var color: Color {
        switch monitor.model(for: server).safety {
        case .safe: .green
        case .tight: .orange
        case .unsafe: .red
        case nil: .secondary
        }
    }

    var body: some View {
        let model = monitor.model(for: server)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Resident").font(.headline)
                Spacer()
                Text(model.activityLabel).font(.caption).foregroundStyle(model.isActive ? .primary : .secondary)
                    .accessibilityIdentifier("MenuBar.Resources.Activity")
            }
            HStack(spacing: 4) {
                Text(model.memoryLabel)
                Text("·")
                Text("Total RAM")
            }
            .font(.caption.monospacedDigit())
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("MenuBar.Resources.Memory")
            ProgressView(value: model.fraction ?? 0).tint(color)
                .accessibilityLabel(String(localized: "Memory used"))
                .accessibilityValue(model.memoryLabel)
                .accessibilityIdentifier("MenuBar.Resources.Meter")
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    if model.rows.isEmpty && model.audioRows.isEmpty {
                        Text(model.residency == nil ? "—" : String(localized: "Unloaded"))
                            .foregroundStyle(.secondary)
                    }
                    ForEach(model.rows) { row in
                        HStack {
                            Text(row.name).lineLimit(1).truncationMode(.middle)
                            Spacer(minLength: 4)
                            Text(row.memoryLabel).monospacedDigit().foregroundStyle(.secondary)
                            Button {
                                Task { await monitor.unload(id: row.id, server: server) }
                            } label: {
                                Image(systemName: "eject")
                            }
                            .buttonStyle(.borderless)
                            .disabled(!row.canUnload || monitor.unloading || server.isOperating || !server.residentLoadsInFlight.isEmpty)
                            .help(row.kept ? String(localized: "Kept in memory") : String(localized: "Unload"))
                            .accessibilityLabel(String(localized: "Unload") + " " + row.name)
                            .accessibilityIdentifier("MenuBar.Resources.Unload.\(row.id)")
                        }
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("MenuBar.Resources.Model.\(row.id)")
                    }
                    ForEach(model.audioRows) { row in
                        HStack {
                            Text(row.name).lineLimit(1)
                            Spacer()
                            Text(row.memoryLabel).foregroundStyle(.secondary)
                        }
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("MenuBar.Resources.Audio.\(row.id)")
                    }
                }
                .font(.caption)
            }
            .frame(height: 72)
            Button(model.unloadAllLabel) {
                Task { await monitor.unload(id: nil, server: server) }
            }
            .disabled(!model.canUnloadAll || monitor.unloading || server.isOperating || !server.residentLoadsInFlight.isEmpty)
            .accessibilityIdentifier("MenuBar.Resources.UnloadAll")
            if let notice = monitor.notice {
                Text(notice).font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("MenuBar.Resources.Notice")
            }
        }
        .padding(12)
        .frame(width: 320, height: 236, alignment: .topLeading)
    }
}
