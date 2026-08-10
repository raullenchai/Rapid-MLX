import SwiftUI

/// Settings → Connectors. The whole user-facing surface for MCP (issue #1716).
///
/// Before this, MCP was engine-complete and app-invisible: the only way to use
/// it from the desktop was to hand-author `~/.config/rapid-mlx/mcp.json` and
/// hope. This panel does four things that file cannot:
///
///   * add / edit / remove servers, and show whether each one actually
///     connected — including the reason when it didn't;
///   * list the tools each server exposes, with a per-tool off switch;
///   * show and revoke the per-tool consent record;
///   * apply an edit without restarting the model, or say plainly when it
///     can't.
struct SettingsConnectorsPanel: View {
    @Environment(MCPConfigStore.self) private var config
    @Environment(MCPCatalog.self) private var catalog
    @Environment(MCPToolApprovalStore.self) private var approval
    @Environment(MCPToolRegistry.self) private var registry
    @Environment(ServerManager.self) private var server

    /// The server being added or edited, when the sheet is up.
    @State private var editing: EditorTarget?
    /// Non-nil when a save / remove / reload failed, shown inline.
    @State private var actionError: String?
    @State private var confirmingRemoval: MCPServerConfig?
    /// True while the banner's Restart button is cycling the child.
    @State private var isRestarting: Bool = false

    /// Whether the running model has to be restarted before connectors can
    /// work — **derived**, never stored.
    ///
    /// This was `@State` set from the reload result, and that was wrong in a
    /// way a user hit immediately: `@State` dies with the view, so switching
    /// Settings tabs or closing the window reset it to false while the
    /// condition it described was still true. The banner vanished and the user
    /// was left with only the engine's raw complaint. The condition is fully
    /// determined by durable state, so read it from there every time:
    /// connectors are on, a child is running, and that child reports it has no
    /// MCP config — which is exactly what happens when it was spawned before
    /// connectors were switched on, since `--mcp-config` is read once at spawn.
    private var needsRestart: Bool {
        // Requires at least one ENABLED server: with none, `launchConfigPath`
        // intentionally stays nil (nothing to start the subsystem for), so the
        // child is correctly unconfigured and a restart could never change
        // that — showing a restart banner it can't clear.
        config.isEnabled
            && config.servers.contains(where: { $0.enabled })
            && server.launchedChildAlias != nil
            && !catalog.isConfigured
    }

    struct EditorTarget: Identifiable {
        /// nil when adding.
        let original: MCPServerConfig?
        var id: String { original?.name ?? "" }
    }

    var body: some View {
        @Bindable var config = config
        return VStack(alignment: .leading, spacing: 20) {
            masterSection
            if config.isEnabled {
                serversSection
                if !catalog.tools.isEmpty {
                    toolsSection
                }
                approvalSection
            }
        }
        .task(id: config.isEnabled) {
            // Reflect reality on open: the panel is the one place a user comes
            // to ask "did it work?", and a stale list is worse than a blank
            // one. Cheap — two loopback GETs.
            guard config.isEnabled else { return }
            await catalog.refresh()
        }
        .sheet(item: $editing) { target in
            MCPServerEditorSheet(
                original: target.original,
                onSave: { updated in save(updated, replacing: target.original?.name) },
                onCancel: { editing = nil }
            )
        }
        .confirmationDialog(
            "Remove “\(confirmingRemoval?.name ?? "")”?",
            isPresented: Binding(
                get: { confirmingRemoval != nil },
                set: { if !$0 { confirmingRemoval = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button("Remove", role: .destructive) {
                if let target = confirmingRemoval { remove(target) }
                confirmingRemoval = nil
            }
            .accessibilityIdentifier("Settings.Connectors.ConfirmRemove")
            Button("Cancel", role: .cancel) { confirmingRemoval = nil }
                .accessibilityIdentifier("Settings.Connectors.CancelRemove")
        } message: {
            Text("Its tools stop being offered to the model. The program itself isn't uninstalled.")
        }
    }

    // MARK: - Master switch

    private var masterSection: some View {
        @Bindable var config = config
        return VStack(alignment: .leading, spacing: 8) {
            header(
                "Connectors",
                "Connect the model to MCP servers — programs on this Mac that expose tools like file access, databases or search. Off by default: a connector is a program that runs on your machine and that the model can invoke."
            )
            card {
                Toggle(isOn: $config.isEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Enable connectors")
                            .font(.system(size: 12, weight: .medium))
                        Text("The local server only loads connectors when this is on.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityIdentifier("Settings.Connectors.MasterToggle")
                // Turning the master switch on or off changes whether the child
                // gets --mcp-config at all, which a hot reload cannot express —
                // the flag is read once at spawn. ``needsRestart`` derives that
                // from live state, so nothing is recorded here.
                .onChange(of: config.isEnabled) { _, isOn in
                    if !isOn {
                        // Don't make the user wait for that restart to stop
                        // offering connector tools. The child may still have
                        // them loaded, but "connectors are off" has to mean
                        // the model is not handed them on the very next turn —
                        // dropping the catalog is what enforces that, since
                        // ``MCPToolRegistry/definitions`` reads from it.
                        catalog.clear()
                    }
                }
            }
        }
    }

    // MARK: - Servers

    private var serversSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                header(
                    "Servers",
                    "Each server runs as its own program and exposes a set of tools."
                )
                Spacer()
                Button("Add…") { editing = EditorTarget(original: nil) }
                    .accessibilityIdentifier("Settings.Connectors.AddButton")
            }

            if let why = config.loadError {
                banner(why, systemImage: "exclamationmark.triangle.fill", tone: .orange)
            }
            // The restart case owns its own banner. Suppressing the engine's
            // string here is deliberate: when the child has no config path the
            // engine says "start the server with --mcp-config", which is
            // operator language for a situation the desktop user reaches
            // without ever seeing a command line. Telling them to pass a flag
            // they have no way to pass is worse than saying nothing.
            if needsRestart {
                restartBanner
            } else if let why = catalog.subsystemError {
                banner(
                    "Connectors couldn't start: \(why)",
                    systemImage: "exclamationmark.triangle.fill",
                    tone: .orange
                )
                .accessibilityIdentifier("Settings.Connectors.SubsystemError")
            }
            if let why = actionError {
                banner(why, systemImage: "exclamationmark.triangle.fill", tone: .red)
            }

            card {
                if config.servers.isEmpty {
                    Text("No connectors yet. Add one to give the model tools beyond the built-ins.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(config.servers.enumerated()), id: \.element.name) { idx, entry in
                            if idx > 0 { Divider().padding(.vertical, 10) }
                            serverRow(entry)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func serverRow(_ entry: MCPServerConfig) -> some View {
        let status = catalog.servers.first { $0.name == entry.name }
        HStack(alignment: .top, spacing: 10) {
            statusDot(for: entry, status: status)
                .padding(.top, 4)
            VStack(alignment: .leading, spacing: 3) {
                Text(entry.name)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                Text(entry.summaryLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(statusLine(for: entry, status: status))
                    .font(.caption)
                    .foregroundStyle(status?.error != nil ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("Settings.Connectors.Row.Status.\(entry.name)")
            }
            Spacer(minLength: 8)
            Toggle("", isOn: Binding(
                get: { entry.enabled },
                set: { setEnabled(entry, $0) }
            ))
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
            .accessibilityIdentifier("Settings.Connectors.Row.Toggle.\(entry.name)")
            Menu {
                Button("Edit…") { editing = EditorTarget(original: entry) }
                    .accessibilityIdentifier("Settings.Connectors.Row.Edit.\(entry.name)")
                Button("Remove", role: .destructive) { confirmingRemoval = entry }
                    .accessibilityIdentifier("Settings.Connectors.Row.Remove.\(entry.name)")
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .accessibilityIdentifier("Settings.Connectors.Row.Menu.\(entry.name)")
        }
    }

    @ViewBuilder
    private func statusDot(for entry: MCPServerConfig, status: MCPCatalog.ServerStatus?) -> some View {
        let color: Color = {
            if !entry.enabled { return .secondary }
            guard let status else { return .secondary }
            if status.error != nil || status.state == "error" { return .orange }
            return status.isConnected ? .green : .secondary
        }()
        Circle().fill(color).frame(width: 8, height: 8)
    }

    /// One line saying what this server is doing right now — the question the
    /// panel exists to answer.
    private func statusLine(for entry: MCPServerConfig, status: MCPCatalog.ServerStatus?) -> String {
        if !entry.enabled { return "Turned off" }
        // The engine's error string can carry a connector's own stderr, so
        // scrub it the same way the tool rows and approval sheet scrub server
        // text — a bidi/zero-width scalar must not spoof this status line.
        if let error = status?.error { return BrowseApprovalStore.displaySafe(error) }
        if let status {
            if status.isConnected {
                let n = status.toolsCount
                return n == 1 ? "Connected · 1 tool" : "Connected · \(n) tools"
            }
            return status.state.capitalized
        }
        // No row from the engine at all.
        if server.launchedChildAlias == nil {
            return "Start a model to connect"
        }
        if catalog.fetchError != nil {
            return "Couldn't check — the local server didn't answer"
        }
        return needsRestart ? "Not applied yet" : "Not connected"
    }

    /// Shown when the running model predates the connectors being switched on.
    ///
    /// Carries a real button rather than an instruction. Telling a user to go
    /// find the model picker and cycle it themselves is asking them to do the
    /// app's job — and the earlier version of this banner did exactly that,
    /// alongside an engine message about a command-line flag.
    private var restartBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 6) {
                Text("Restart the model to finish turning connectors on.")
                    .font(.callout)
                Text("The running model started before connectors were enabled, so it isn't loading them yet. Restarting takes a moment and keeps your conversation.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button(isRestarting ? "Restarting…" : "Restart") { restartModel() }
                .disabled(isRestarting || server.isOperating)
                .accessibilityIdentifier("Settings.Connectors.RestartButton")
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.12)))
    }

    /// Stop-then-start the current alias so the child is respawned WITH
    /// ``--mcp-config``. Mirrors the model-switch path in ``ContentView``.
    private func restartModel() {
        guard let alias = server.launchedChildAlias else { return }
        isRestarting = true
        Task {
            await server.stop()
            await server.start(alias: alias)
            // The fresh child publishes its connector state on /healthz; the
            // ready transition in ContentView refreshes the catalog, but this
            // panel may be the only thing on screen — refresh here too so the
            // rows update without the user poking anything.
            await catalog.refresh()
            isRestarting = false
        }
    }

    // MARK: - Tools

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            header(
                "Tools",
                "What the connected servers expose. Turn one off and it is never offered to the model — and never runs, even if the model asks for it by name."
            )
            card {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(registry.allKnownTools, id: \.function.name) { def in
                        toolRow(def)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func toolRow(_ def: ToolDefinition) -> some View {
        let name = def.function.name
        Toggle(isOn: Binding(
            get: { registry.isToolEnabled(name) },
            set: { registry.setToolEnabled(name, $0) }
        )) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "wrench.and.screwdriver")
                    .foregroundStyle(RapidTheme.brand)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        // Server-supplied text (tool name, owning server,
                        // description) is scrubbed the same way the approval
                        // sheet scrubs it — a bidi or zero-width scalar in a
                        // server's tool metadata must not visually spoof a row.
                        Text(BrowseApprovalStore.displaySafe(MCPToolApprovalStore.shortToolName(name)))
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                        if let source = catalog.serverForTool[name] {
                            Text(BrowseApprovalStore.displaySafe(source))
                                .font(.system(size: 10, weight: .medium))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(RapidTheme.brand.opacity(0.15)))
                        }
                        if approval.grantedTools.contains(name) {
                            Text("always allowed")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                    }
                    Text(BrowseApprovalStore.displaySafe(def.function.description))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .toggleStyle(TrailingSettingsToggleStyle())
        .accessibilityIdentifier("Settings.Connectors.Tool.Toggle.\(name)")
    }

    // MARK: - Approvals

    private var approvalSection: some View {
        @Bindable var approval = approval
        return VStack(alignment: .leading, spacing: 8) {
            header(
                "Approvals",
                "The first time the model calls a connector tool, Rapid asks. Your answer is remembered per tool."
            )
            card {
                VStack(alignment: .leading, spacing: 14) {
                    Toggle(isOn: Binding(
                        get: { approval.mode == .autoApproveAll },
                        set: { approval.mode = $0 ? .autoApproveAll : .ask }
                    )) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Auto-approve all tool calls")
                                .font(.system(size: 12, weight: .medium))
                            Text("Skips every prompt, including for connectors added later. For unattended use only.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .toggleStyle(TrailingSettingsToggleStyle())
                    .accessibilityIdentifier("Settings.Connectors.AutoApproveToggle")

                    Divider()

                    HStack(alignment: .firstTextBaseline) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(approval.grantedTools.isEmpty
                                ? "No tools are permanently allowed."
                                : "\(approval.grantedTools.count) tool\(approval.grantedTools.count == 1 ? "" : "s") permanently allowed.")
                                .font(.callout)
                            Text("Resetting makes Rapid ask again the next time each one is called.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Reset") { approval.resetGrants() }
                            .disabled(approval.grantedTools.isEmpty)
                            .accessibilityIdentifier("Settings.Connectors.ResetApprovals")
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func save(_ updated: MCPServerConfig, replacing originalName: String?) {
        do {
            try config.upsert(updated, replacing: originalName)
            editing = nil
            actionError = nil
            applyChange()
        } catch {
            actionError = error.localizedDescription
        }
    }

    private func remove(_ entry: MCPServerConfig) {
        do {
            try config.remove(named: entry.name)
            actionError = nil
            applyChange()
        } catch {
            actionError = error.localizedDescription
        }
    }

    private func setEnabled(_ entry: MCPServerConfig, _ enabled: Bool) {
        do {
            try config.setServerEnabled(entry.name, enabled)
            actionError = nil
            applyChange()
        } catch {
            actionError = error.localizedDescription
        }
    }

    /// Push a config edit through to the running engine.
    ///
    /// Issue #1716 acceptance item 4: apply without a restart, or say so.
    /// ``MCPCatalog/reload()`` hits the engine's reload route so an edit takes
    /// effect immediately. When it can't — engine not running, no config path
    /// (child predates the master switch), or an older build with no reload
    /// route — the reload leaves `catalog.isConfigured` false and the derived
    /// ``needsRestart`` raises the banner. Nothing is recorded here, so the
    /// banner survives a tab switch.
    private func applyChange() {
        // Nothing running means the next spawn picks the file up anyway.
        guard server.launchedChildAlias != nil else { return }
        Task { await catalog.reload() }
    }

    // MARK: - Chrome

    private func header(_ title: String, _ subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.title3.weight(.semibold))
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func banner(_ text: String, systemImage: String, tone: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: systemImage).foregroundStyle(tone)
            Text(text)
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(tone.opacity(0.12)))
    }

    @ViewBuilder
    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .fill(RapidTheme.card)
            )
            .clipShape(RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
                    .stroke(RapidTheme.hairline, lineWidth: 1)
            )
    }
}

extension MCPServerConfig {
    /// One-line "what is this" for the server row — the command that will run,
    /// or the URL that will be contacted.
    var summaryLine: String {
        switch transport {
        case .stdio:
            let parts = ([command ?? ""] + args).filter { !$0.isEmpty }
            return parts.joined(separator: " ")
        case .sse:
            return url ?? ""
        }
    }
}
