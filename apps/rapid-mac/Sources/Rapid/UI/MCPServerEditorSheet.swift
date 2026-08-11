import SwiftUI

/// Add or edit one MCP server (issue #1716).
///
/// This is the form that replaces hand-authoring `mcp.json`. It deliberately
/// validates the name locally — the engine namespaces every tool as
/// `server__tool`, so a name with a space in it produces tool names the model
/// can never call, and finding that out through "the model just ignores that
/// server" is a bad afternoon.
///
/// The engine still gets the final say on the command itself
/// (`vllm_mlx/mcp/security.py` allowlists what may be spawned). We don't
/// duplicate that list here: it moves independently of the app, and a
/// client-side copy that drifts would either block something valid or promise
/// something that then fails at connect. The rejection reason comes back on
/// the server row instead.
struct MCPServerEditorSheet: View {
    /// nil when adding.
    let original: MCPServerConfig?
    let onSave: (MCPServerConfig) -> Void
    let onCancel: () -> Void

    @State private var name: String
    @State private var transport: MCPServerConfig.Transport
    @State private var command: String
    @State private var url: String
    @State private var enabled: Bool
    /// Args and env are edited as text — one per line, `KEY=value` for env.
    /// A table of add/remove rows is more clicks for the same result, and this
    /// shape pastes straight out of any MCP README.
    @State private var argsText: String
    @State private var envText: String

    init(
        original: MCPServerConfig?,
        onSave: @escaping (MCPServerConfig) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.original = original
        self.onSave = onSave
        self.onCancel = onCancel
        _name = State(initialValue: original?.name ?? "")
        _transport = State(initialValue: original?.transport ?? .stdio)
        _command = State(initialValue: original?.command ?? "")
        _url = State(initialValue: original?.url ?? "")
        _enabled = State(initialValue: original?.enabled ?? true)
        _argsText = State(initialValue: (original?.args ?? []).joined(separator: "\n"))
        _envText = State(initialValue: (original?.env ?? [:])
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n"))
    }

    /// The value that would be saved. Built fresh each render so the inline
    /// validation message and the Save button can never disagree about it.
    private var draft: MCPServerConfig {
        MCPServerConfig(
            name: name.trimmingCharacters(in: .whitespaces),
            transport: transport,
            command: transport == .stdio
                ? command.trimmingCharacters(in: .whitespaces)
                : nil,
            args: transport == .stdio ? Self.parseLines(argsText) : [],
            env: transport == .stdio ? Self.parseEnv(envText) : [:],
            url: transport == .sse ? url.trimmingCharacters(in: .whitespaces) : nil,
            enabled: enabled,
            timeout: original?.timeout ?? 30
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            SectionHeader(
                original == nil ? "Add connector" : "Edit “\(original?.name ?? "")”",
                emphasis: .section
            )

            // The form itself stays native ``.formStyle(.grouped)``:
            // macOS owns field/picker/toggle layout here and does it
            // better than a hand-rolled grid would. Only the typography
            // of the help lines and the footer's button tiers move onto
            // the shared system.
            Form {
                TextField("Name", text: $name)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Name")
                Text("Letters, numbers, dashes and underscores. Becomes the prefix on every tool this connector exposes.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)

                Picker("Type", selection: $transport) {
                    ForEach(MCPServerConfig.Transport.allCases, id: \.self) { t in
                        Text(t.displayName).tag(t)
                    }
                }
                .accessibilityIdentifier("Settings.Connectors.Editor.Transport")

                switch transport {
                case .stdio:
                    TextField("Command", text: $command)
                        .accessibilityIdentifier("Settings.Connectors.Editor.Command")
                    Text("For example `uvx` or `npx`. Rapid's engine only runs commands on its allowlist.")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)

                    VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                        Text("Arguments — one per line")
                            .font(RapidFont.caption)
                            .foregroundStyle(RapidTheme.textSecondary)
                        codeEditor(
                            text: $argsText, height: 64,
                            axIdentifier: "Settings.Connectors.Editor.AddArgument"
                        )
                    }

                    VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                        Text("Environment — one KEY=value per line")
                            .font(RapidFont.caption)
                            .foregroundStyle(RapidTheme.textSecondary)
                        codeEditor(
                            text: $envText, height: 56,
                            axIdentifier: "Settings.Connectors.Editor.AddEnv"
                        )
                    }

                case .sse:
                    TextField("URL", text: $url)
                        .accessibilityIdentifier("Settings.Connectors.Editor.URL")
                    Text("An http:// or https:// endpoint speaking MCP over SSE.")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textSecondary)
                }

                Toggle("Enabled", isOn: $enabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Enabled")
            }
            .formStyle(.grouped)

            if let why = draft.validationError, !name.isEmpty || !command.isEmpty || !url.isEmpty {
                // Held back until the user has typed something — an empty form
                // that scolds you before you start is noise, not guidance.
                InlineNotice(message: why, tone: .warning)
            }

            // Cancel and Save are now the SAME height (both regular, 32):
            // `.rapidPrimary` used to default to 36 and the pair stepped.
            // A disabled Save keeps the amber fill at ``disabledOpacity``
            // rather than going grey, so it stays findable and its label
            // stays readable while it explains nothing can be saved yet.
            HStack(spacing: RapidTheme.Space.sm) {
                Spacer()
                Button("Cancel", action: onCancel)
                    .buttonStyle(.rapidSecondary)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Cancel")
                Button("Save") { onSave(draft) }
                    .buttonStyle(.rapidPrimary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(draft.validationError != nil)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Allow")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 520)
        .background(RapidTheme.surfaceOverlay)
    }

    /// A multi-line code field with a ground of its own.
    ///
    /// A bare ``TextEditor`` draws no background, so in Dark Mode the
    /// Arguments and Environment boxes were an outline around the sheet
    /// colour — they read as empty space, not as fields you can type in.
    /// ``surfaceCode`` is the same recessed ground inline code uses
    /// everywhere else in the app.
    @ViewBuilder
    private func codeEditor(
        text: Binding<String>, height: CGFloat, axIdentifier: String
    ) -> some View {
        // The identifier is applied to the ``TextEditor`` itself (not at the
        // call site) so the AX-identifier gate sees the control carry it and
        // the AX driver lands on the editable field, not a wrapper.
        TextEditor(text: text)
            .accessibilityIdentifier(axIdentifier)
            .font(RapidFont.code)
            .scrollContentBackground(.hidden)
            .padding(RapidTheme.Space.xs)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.code, style: .continuous)
                    .fill(RapidTheme.surfaceCode)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.code, style: .continuous)
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            )
    }

    /// Non-empty, whitespace-trimmed lines. `static` so the parsing can be
    /// pinned by tests without standing up the view.
    static func parseLines(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// `KEY=value` per line. Splits on the FIRST `=` so a value containing one
    /// survives; a line with no `=` is skipped rather than becoming an empty
    /// key the engine would then have to reject.
    static func parseEnv(_ text: String) -> [String: String] {
        var out: [String: String] = [:]
        for line in parseLines(text) {
            guard let idx = line.firstIndex(of: "=") else { continue }
            let key = String(line[line.startIndex..<idx]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: idx)...])
            if !key.isEmpty { out[key] = value }
        }
        return out
    }
}
