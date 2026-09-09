import SwiftUI

struct ShareComputeView: View {
    @Bindable var manager: ShareComputeManager
    @Bindable var downloads: DownloadManager
    let catalog: [ModelEntry]
    let catalogLoaded: Bool

    @AppStorage("Rapid.shareCompute.selectedModel") private var selectedID = "qwen3.8-27b"
    @AppStorage("Rapid.shareCompute.worker") private var worker = Host.current().localizedName ?? "Mac"
    @State private var providerKey = ""
    @State private var showingConnection = false

    private var selected: ShareComputeModel {
        ShareComputeModel.supported.first { $0.catalogID == selectedID }
            ?? ShareComputeModel.supported[0]
    }

    private var selectedEntry: ModelEntry? {
        catalog.first { $0.alias.caseInsensitiveCompare(selected.alias) == .orderedSame }
    }

    private var selectedIsCached: Bool { selectedEntry?.cached == true }

    private var requiresRegistration: Bool {
        if !manager.hasRegistration(for: selected, worker: worker) { return true }
        if case .failed(let message) = manager.state {
            return message.localizedCaseInsensitiveContains("register again")
                || message.localizedCaseInsensitiveContains("re-register")
        }
        return false
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                header
                providerCard
                if manager.state.isActive || manager.activeModel != nil {
                    activeCard
                } else {
                    setupCard
                }
                safetyNotice
            }
            .frame(maxWidth: RapidTheme.Layout.contentMaxWidth)
            .padding(RapidTheme.Space.xl)
        }
        .background(RapidTheme.surfaceCanvas)
        .sheet(isPresented: $showingConnection) { connectionSheet }
        .onChange(of: manager.state) { _, state in
            if case .failed(let message) = state,
               message.localizedCaseInsensitiveContains("provider key") {
                showingConnection = true
            }
        }
    }

    private var header: some View {
        SectionHeader(
            "Share Compute",
            subtitle: "Put your Mac's idle MLX capacity to work and earn through a compute provider.",
            emphasis: .page
        )
        .accessibilityIdentifier("ShareCompute.Header")
    }

    private var providerCard: some View {
        SettingsSection("Compute provider") {
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                HStack(alignment: .top, spacing: RapidTheme.Space.md) {
                    Image(systemName: "bolt.horizontal.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(RapidTheme.brandAmber)
                    VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                        Text("QuickSilver")
                            .font(RapidFont.sectionTitle)
                        Text("One OpenAI-compatible API spanning 48 frontier and open models. Rapid Macs add high-quality open-model capacity at the edge, with public pricing and live status.")
                            .font(RapidFont.secondary)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: RapidTheme.Space.md)
                    Link("Learn more", destination: URL(string: "https://quicksilverpro.io/")!)
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("ShareCompute.ProviderLearnMore")
                }
                HStack(spacing: RapidTheme.Space.lg) {
                    proof("48", "models")
                    proof("1", "compatible API")
                    proof("Live", "status")
                }
            }
        }
    }

    private func proof(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(RapidFont.body).fontWeight(.semibold)
            Text(label).font(RapidFont.caption).foregroundStyle(.secondary)
        }
    }

    private var setupCard: some View {
        SettingsSection("This Mac") {
            VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                if case .failed(let message) = manager.state {
                    InlineNotice(message: message, tone: .warning)
                }
                Picker("Pool model", selection: $selectedID) {
                    ForEach(ShareComputeModel.supported) { model in
                        Text(model.title).tag(model.catalogID)
                    }
                }
                .accessibilityIdentifier("ShareCompute.ModelPicker")

                HStack {
                    VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                        Text(selected.detail).font(RapidFont.body)
                        Text(!catalogLoaded
                             ? "Checking this Mac…"
                             : selectedIsCached
                                ? "Ready on this Mac"
                                : "Download required before sharing")
                            .font(RapidFont.secondary)
                            .foregroundStyle(selectedIsCached ? .green : .secondary)
                    }
                    Spacer()
                    if !catalogLoaded {
                        Button("Checking…") {}
                            .buttonStyle(.rapidPrimaryCompact)
                            .disabled(true)
                            .accessibilityIdentifier("ShareCompute.CatalogChecking")
                    } else if selectedIsCached {
                        Button(requiresRegistration ? "Connect & Share" : "Start Sharing") {
                            if !requiresRegistration {
                                Task { await manager.join(model: selected, worker: worker) }
                            } else {
                                showingConnection = true
                            }
                        }
                        .buttonStyle(.rapidPrimaryCompact)
                        .accessibilityIdentifier("ShareCompute.Start")
                    } else {
                        Button(downloads.isDownloading(selected.alias) ? "Downloading…" : "Download Model") {
                            _ = downloads.startDownload(
                                alias: selected.alias,
                                hfPath: selectedEntry?.hfRepo
                            )
                        }
                        .buttonStyle(.rapidPrimaryCompact)
                        .disabled(downloads.isDownloading(selected.alias))
                        .accessibilityIdentifier("ShareCompute.Download")
                    }
                }
            }
        }
    }

    private var activeCard: some View {
        SettingsSection("This Mac") {
            VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                HStack(spacing: RapidTheme.Space.md) {
                    if manager.state == .online {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                        Text(statusTitle).font(RapidFont.body).fontWeight(.semibold)
                        Text(manager.activeModel?.title ?? selected.title)
                            .font(RapidFont.secondary).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Stop Sharing") { manager.leave() }
                        .buttonStyle(.rapidSecondaryCompact)
                        .disabled(manager.state == .stopping)
                        .accessibilityIdentifier("ShareCompute.Stop")
                }
                if let node = manager.snapshot?.nodeID {
                    Divider()
                    HStack {
                        Text("Node").foregroundStyle(.secondary)
                        Spacer()
                        Text(node).font(.system(.body, design: .monospaced))
                    }
                }
                if manager.state == .online {
                    Divider()
                    HStack {
                        Label("Online", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                        Spacer()
                        Text("\(manager.snapshot?.inflight ?? 0) active requests")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var statusTitle: String {
        switch manager.state {
        case .idle: "Ready"
        case .preparing: "Pausing the current model…"
        case .registering: "Registering with QuickSilver…"
        case .starting: "Starting the model…"
        case .warming: "Warming up…"
        case .connecting: "Connecting to the pool…"
        case .online: "Sharing compute"
        case .reconnecting: "Reconnecting…"
        case .stopping: "Stopping safely…"
        case .failed(let message): message
        }
    }

    private var safetyNotice: some View {
        InlineNotice(
            message: "While sharing, requests from QuickSilver customers run locally on this Mac. Rapid pauses your current model so two large models never compete for unified memory. Stop Sharing disconnects the pool and restores your previous model.",
            tone: .info
        )
    }

    private var connectionSheet: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            SectionHeader(
                "Connect QuickSilver",
                subtitle: "Paste the one-time provider key from your QuickSilver account. Rapid sends it directly to the local provider process; it is never put in command arguments or saved by Rapid.",
                emphasis: .page
            )
            SecureField("qsppk-…", text: $providerKey)
                .textFieldStyle(.roundedBorder)
                .accessibilityIdentifier("ShareCompute.ProviderKey")
            TextField("Worker name", text: $worker)
                .textFieldStyle(.roundedBorder)
                .accessibilityIdentifier("ShareCompute.Worker")
            HStack {
                Link("Open QuickSilver", destination: URL(string: "https://quicksilverpro.io/")!)
                    .accessibilityIdentifier("ShareCompute.OpenQuickSilver")
                Spacer()
                Button("Cancel") {
                    providerKey = ""
                    showingConnection = false
                }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("ShareCompute.Cancel")
                Button("Connect & Share") {
                    let key = providerKey
                    providerKey = ""
                    showingConnection = false
                    Task { await manager.join(model: selected, worker: worker, providerKey: key) }
                }
                .buttonStyle(.rapidPrimary)
                .disabled(providerKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .accessibilityIdentifier("ShareCompute.Connect")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 520)
    }
}
