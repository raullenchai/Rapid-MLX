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
        return VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Connectors",
                subtitle: "Connect the model to MCP servers — programs on this Mac that expose tools like file access, databases or search. Off by default: a connector is a program that runs on your machine and that the model can invoke.",
                emphasis: .page
            )
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
        return SettingsSection {
                Toggle(isOn: $config.isEnabled) {
                    SettingsRowLabel(
                        title: "Enable connectors",
                        description: "The local server only loads connectors when this is on."
                    )
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

    // MARK: - Servers

    private var serversSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            if let why = config.loadError {
                InlineNotice(message: why, tone: .warning)
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
                InlineNotice(
                    message: "Connectors couldn't start: \(why)",
                    tone: .warning
                )
                .accessibilityIdentifier("Settings.Connectors.SubsystemError")
            }
            if let why = actionError {
                InlineNotice(message: why, tone: .error)
            }

            SettingsSection("Servers", subtitle: "Each server runs as its own program and exposes a set of tools.") {
                Button("Add…") { editing = EditorTarget(original: nil) }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("Settings.Connectors.AddButton")
            } content: {
                if config.servers.isEmpty {
                    Text("No connectors yet. Add one to give the model tools beyond the built-ins.")
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    ForEach(Array(config.servers.enumerated()), id: \.element.name) { idx, entry in
                        if idx > 0 { SettingsRowDivider() }
                        serverRow(entry)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func serverRow(_ entry: MCPServerConfig) -> some View {
        let status = catalog.servers.first { $0.name == entry.name }
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            statusDot(for: entry, status: status)
                .padding(.top, RapidTheme.Space.xs)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(entry.name)
                    .font(RapidFont.code)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(entry.summaryLine)
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(statusLine(for: entry, status: status))
                    .font(RapidFont.caption)
                    .foregroundStyle(
                        status?.error != nil
                            ? RapidTheme.statusWarning
                            : RapidTheme.textSecondary
                    )
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("Settings.Connectors.Row.Status.\(entry.name)")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
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
            .foregroundStyle(RapidTheme.utilityActionLabel)
            .accessibilityLabel("Connector actions")
            .accessibilityIdentifier("Settings.Connectors.Row.Menu.\(entry.name)")
        }
    }

    /// The row's state, as one dot. Every colour is a status token —
    /// these were `.orange` / `.green` / `.secondary` literals, which put
    /// a second, slightly different amber and a second green on a window
    /// that already had one of each.
    @ViewBuilder
    private func statusDot(for entry: MCPServerConfig, status: MCPCatalog.ServerStatus?) -> some View {
        let color: Color = {
            if !entry.enabled { return RapidTheme.statusIdle }
            guard let status else { return RapidTheme.statusIdle }
            if status.error != nil || status.state == "error" { return RapidTheme.statusWarning }
            return status.isConnected ? RapidTheme.statusReady : RapidTheme.statusIdle
        }()
        Circle()
            .fill(color)
            .frame(width: 8, height: 8)
            .accessibilityHidden(true)
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
        // Same shape and copy; the container is now the shared notice
        // rather than a local `Color.orange.opacity(0.12)` rectangle, and
        // the button carries a real tier. It keeps a two-line body (the
        // shared notice takes one message), so it composes the notice's
        // tokens rather than the notice itself.
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(RapidTheme.statusWarning)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                Text("Restart the model to finish turning connectors on.")
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("The running model started before connectors were enabled, so it isn't loading them yet. Restarting takes a moment and keeps your conversation.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button(isRestarting ? "Restarting…" : "Restart") { restartModel() }
                .buttonStyle(.rapidSecondaryCompact)
                .fixedSize()
                .disabled(isRestarting || server.isOperating)
                .accessibilityIdentifier("Settings.Connectors.RestartButton")
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.sm)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .fill(RapidTheme.statusWarningTint)
        )
    }

    /// Stop-then-start the current alias so the child is respawned WITH
    /// ``--mcp-config``. Mirrors the model-switch path in ``ContentView``.
    private func restartModel() {
        // Audio is intentionally non-resident and can own the sidecar after a
        // trip through Speech. Restarting that launch alias cannot initialize
        // MCP, even when a ready text model is resident beside it. Prefer the
        // resident text engine; retain the launch alias for legacy engines
        // that do not publish residency data.
        guard let alias = server.residency.preferredTextAlias(
            fallback: server.launchedChildAlias
        ) else { return }
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
        SettingsSection(
            "Tools",
            subtitle: "What the connected servers expose. Turn one off and it is never offered to the model — and never runs, even if the model asks for it by name."
        ) {
            let tools = registry.allKnownTools
            ForEach(Array(tools.enumerated()), id: \.element.function.name) { index, def in
                if index > 0 { SettingsRowDivider() }
                toolRow(def)
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
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Image(systemName: "wrench.and.screwdriver")
                    .foregroundStyle(RapidTheme.utilityActionLabel)
                    .frame(width: RapidTheme.Layout.iconSlot)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                    HStack(spacing: RapidTheme.Space.xs) {
                        // Server-supplied text (tool name, owning server,
                        // description) is scrubbed the same way the approval
                        // sheet scrubs it — a bidi or zero-width scalar in a
                        // server's tool metadata must not visually spoof a row.
                        Text(BrowseApprovalStore.displaySafe(MCPToolApprovalStore.shortToolName(name)))
                            .font(RapidFont.code)
                            .foregroundStyle(RapidTheme.textPrimary)
                        if let source = catalog.serverForTool[name] {
                            Text(BrowseApprovalStore.displaySafe(source))
                                .font(RapidFont.caption)
                                .foregroundStyle(RapidTheme.textSecondary)
                                .padding(.horizontal, RapidTheme.Space.xs)
                                .padding(.vertical, RapidTheme.Space.xxs)
                                .background(Capsule().fill(RapidTheme.hoverFill))
                        }
                        if approval.grantedTools.contains(name) {
                            Text("always allowed")
                                .font(RapidFont.caption)
                                .foregroundStyle(RapidTheme.textTertiary)
                        }
                    }
                    Text(BrowseApprovalStore.displaySafe(def.function.description))
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .toggleStyle(TrailingSettingsToggleStyle())
        .accessibilityIdentifier("Settings.Connectors.Tool.Toggle.\(name)")
    }

    // MARK: - Approvals

    private var approvalSection: some View {
        @Bindable var approval = approval
        return SettingsSection(
            "Approvals",
            subtitle: "The first time the model calls a connector tool, Rapid asks. Your answer is remembered per tool."
        ) {
            Toggle(isOn: Binding(
                get: { approval.mode == .autoApproveAll },
                set: { approval.mode = $0 ? .autoApproveAll : .ask }
            )) {
                SettingsRowLabel(
                    title: "Auto-approve all tool calls",
                    description: "Skips every prompt, including for connectors added later. For unattended use only."
                )
            }
            .toggleStyle(TrailingSettingsToggleStyle())
            .accessibilityIdentifier("Settings.Connectors.AutoApproveToggle")

            SettingsRowDivider()

            SettingsRow(
                title: approval.grantedTools.isEmpty
                    ? "No tools are permanently allowed."
                    : "\(approval.grantedTools.count) tool\(approval.grantedTools.count == 1 ? "" : "s") permanently allowed.",
                description: "Resetting makes Rapid ask again the next time each one is called."
            ) {
                Button("Reset") { approval.resetGrants() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .disabled(approval.grantedTools.isEmpty)
                    .accessibilityIdentifier("Settings.Connectors.ResetApprovals")
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

    // NOTE: private ``header(_:_:)``, ``banner(_:systemImage:tone:)`` and
    // ``card(_:)`` helpers lived here — the third copy of the card/header
    // pair, plus a local banner that took a raw ``Color`` and washed it
    // to 12%. All three are now ``SectionHeader`` / ``SettingsSection`` /
    // ``InlineNotice``, which is what moved this panel's four banner call
    // sites off `Color.orange` and `Color.red` onto the status tokens.
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
